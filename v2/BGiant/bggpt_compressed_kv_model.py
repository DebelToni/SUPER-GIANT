
# bggpt_compressed_kv_model.py
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma2.modeling_gemma2 import (
    repeat_kv,
)


# -------------------------
# Small helper for RoPE
# -------------------------

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Helper used in RoPE: splits last dim into [x1, x2] and returns [-x2, x1].

    This matches Gemma2's rotate_half implementation. 
    """
    d2 = x.shape[-1] // 2
    x1 = x[..., :d2]
    x2 = x[..., d2:]
    return torch.cat((-x2, x1), dim=-1)


# -------------------------
# KV compressor
# -------------------------

class KVCompressor(nn.Module):
    """
    Simple linear compressor/decompressor for KV cache.

    Operates on (..., d_model) and maps:
      - compress:   d_model -> d_latent
      - decompress: d_latent -> d_model
    """

    def __init__(self, d_model: int, d_latent: int):
        super().__init__()
        assert d_latent > 0 and d_latent <= d_model
        self.d_model = d_model
        self.d_latent = d_latent

        self.compress = nn.Linear(d_model, d_latent, bias=False)
        self.decompress = nn.Linear(d_latent, d_model, bias=False)

        # Optional: near-identity-ish init in the shared subspace
        with torch.no_grad():
            nn.init.zeros_(self.compress.weight)
            nn.init.zeros_(self.decompress.weight)
            k = min(d_model, d_latent)
            self.compress.weight[:k, :k] = torch.eye(k)
            self.decompress.weight[:k, :k] = torch.eye(k)

    def forward_compress(self, x: torch.Tensor) -> torch.Tensor:
        return self.compress(x)

    def forward_decompress(self, z: torch.Tensor) -> torch.Tensor:
        return self.decompress(z)


# -------------------------
# Compressed KV Gemma2 attention with manual RoPE scaling
# -------------------------

class CompressedKVGemma2Attention(nn.Module):
    """
    Gemma2 self-attention that:
      - reuses Gemma2's learned Q/K/V/O projections
      - stores KV **compressed** in the cache
      - decompresses KV on each forward for attention
      - does its *own* RoPE scaling to extend context (e.g. 8k -> 64k)

    Cache format per layer:
      past_key_value = (k_comp, v_comp)
        k_comp: (b, num_kv_heads, s_kv, d_latent)
        v_comp: (b, num_kv_heads, s_kv, d_latent)
    """

    def __init__(
        self,
        base_attn: nn.Module,
        config,
        is_sliding: bool,
        kv_compression_ratio: float = 1.0,
        rope_factor: float = 1.0,
    ):
        super().__init__()

        # Pull stable attributes from Gemma2Config instead of base_attn.*
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.sliding_window = config.sliding_window if is_sliding else None

        self.attention_dropout = config.attention_dropout

        # Gemma2-specific bits (match Gemma2Attention) 
        self.attn_logit_softcapping = getattr(config, "attn_logit_softcapping", None)
        query_scalar = getattr(config, "query_pre_attn_scalar", None)
        if query_scalar is not None:
            self.scaling = query_scalar ** -0.5
        else:
            self.scaling = 1.0 / math.sqrt(self.head_dim)

        # Reuse learned projections
        self.q_proj = base_attn.q_proj
        self.k_proj = base_attn.k_proj
        self.v_proj = base_attn.v_proj
        self.o_proj = base_attn.o_proj

        # Manual RoPE setup
        self.rope_theta = getattr(config, "rope_theta", 10000.0)
        self.rope_factor = rope_factor  # e.g. 8 -> 8x context

        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Compressed dim per KV head
        latent_dim = max(1, int(self.head_dim * kv_compression_ratio))
        self.k_compressor = KVCompressor(self.head_dim, latent_dim)
        self.v_compressor = KVCompressor(self.head_dim, latent_dim)

    def _scaled_rope(self, s_kv: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute RoPE cos/sin with linear position scaling.

        Returns cos, sin: (1, s_kv, head_dim)
        (batch dim will be broadcast).
        """
        positions = torch.arange(0, s_kv, device=device, dtype=torch.float32)
        scaled_pos = positions / self.rope_factor  # (s_kv,)

        inv_freq = self.inv_freq[None, :, None].float()  # (1, dim/2, 1)
        pos_expanded = scaled_pos[None, None, :]         # (1, 1, s_kv)

        freqs = (inv_freq @ pos_expanded).transpose(1, 2)  # (1, s_kv, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)            # (1, s_kv, dim)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def _apply_rope_qk(
        self,
        q: torch.Tensor,          # (b, num_heads, q_len, head_dim)
        k: torch.Tensor,          # (b, num_kv_heads, s_kv, head_dim)
        cos_full: torch.Tensor,   # (1, s_kv, head_dim)
        sin_full: torch.Tensor,   # (1, s_kv, head_dim)
        q_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to queries and keys with *different* sequence lengths:

          - keys: all s_kv positions
          - queries: only last q_len positions in the same coordinate frame

        cos_full/sin_full index range: [0 .. s_kv-1]
        """
        bsz = q.size(0)
        s_kv = k.size(2)
        device = q.device

        # Broadcast cos/sin to batch
        cos_full = cos_full.to(device=device).expand(bsz, -1, -1)  # (b, s_kv, d)
        sin_full = sin_full.to(device=device).expand(bsz, -1, -1)  # (b, s_kv, d)

        # Keys: all positions
        cos_k = cos_full    # (b, s_kv, d)
        sin_k = sin_full    # (b, s_kv, d)

        # Queries: last q_len positions
        cos_q = cos_full[:, -q_len:, :]  # (b, q_len, d)
        sin_q = sin_full[:, -q_len:, :]  # (b, q_len, d)

        # Unsqueeze for broadcasting over heads
        cos_k = cos_k.unsqueeze(1)  # (b, 1, s_kv, d)
        sin_k = sin_k.unsqueeze(1)  # (b, 1, s_kv, d)
        cos_q = cos_q.unsqueeze(1)  # (b, 1, q_len, d)
        sin_q = sin_q.unsqueeze(1)  # (b, 1, q_len, d)

        # Apply RoPE: (x * cos) + (rotate_half(x) * sin)
        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)

        return q_embed, k_embed

    def forward(
        self,
        hidden_states: torch.Tensor,          # (b, q_len, hidden_size)
        position_ids: torch.LongTensor,       # (b, q_len) -- unused, we track local positions
        attention_mask: Optional[torch.Tensor] = None,  # (b, 1, q_len, s_kv) or None
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        device = hidden_states.device

        # 1) Q, K, V projections
        q = self.q_proj(hidden_states)  # (b, q_len, num_heads * head_dim)
        k = self.k_proj(hidden_states)  # (b, q_len, num_kv_heads * head_dim)
        v = self.v_proj(hidden_states)

        # 2) Reshape to heads
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # shapes:
        #   q: (b, num_heads,     q_len, head_dim)
        #   k: (b, num_kv_heads,  q_len, head_dim)
        #   v: (b, num_kv_heads,  q_len, head_dim)

        # 3) Compress new KV
        k_comp_new = self.k_compressor.forward_compress(k)  # (b, num_kv_heads, q_len, d_latent)
        v_comp_new = self.v_compressor.forward_compress(v)  # (b, num_kv_heads, q_len, d_latent)

        if past_key_value is not None:
            past_k_comp, past_v_comp = past_key_value
            k_comp = torch.cat([past_k_comp, k_comp_new], dim=2)
            v_comp = torch.cat([past_v_comp, v_comp_new], dim=2)
        else:
            k_comp, v_comp = k_comp_new, v_comp_new

        # 4) Apply sliding window at compressed level (for local layers)
        if self.sliding_window is not None and k_comp.size(2) > self.sliding_window:
            k_comp = k_comp[:, :, -self.sliding_window :, :]
            v_comp = v_comp[:, :, -self.sliding_window :, :]

        # 5) Decompress KV for attention
        #    k_full: (b, num_kv_heads, s_kv, head_dim)
        k_full = self.k_compressor.forward_decompress(k_comp)
        v_full = self.v_compressor.forward_decompress(v_comp)
        s_kv = k_full.size(2)

        # 6) Apply RoPE with scaled positions:
        #    - keys: all s_kv tokens
        #    - queries: last q_len tokens in that frame
        cos_full, sin_full = self._scaled_rope(
            s_kv=s_kv,
            device=device,
            dtype=k_full.dtype,
        )
        q, k_full = self._apply_rope_qk(q, k_full, cos_full, sin_full, q_len=q_len)

        # 7) Expand KV heads to match num_heads via GQA
        #    shapes -> (b, num_heads, s_kv, head_dim)
        k_full = repeat_kv(k_full, self.num_key_value_groups)
        v_full = repeat_kv(v_full, self.num_key_value_groups)

        # 8) Scaled dot-product attention (match eager_attention_forward) 
        attn_scores = torch.matmul(q, k_full.transpose(-1, -2)) * self.scaling

        # Match Gemma2's logit soft-capping if present
        if self.attn_logit_softcapping is not None:
            softcap = self.attn_logit_softcapping
            attn_scores = attn_scores / softcap
            attn_scores = torch.tanh(attn_scores)
            attn_scores = attn_scores * softcap

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask.to(
                device=attn_scores.device, dtype=attn_scores.dtype
            )

        attn_probs = F.softmax(attn_scores, dim=-1)
        if self.attention_dropout > 0 and self.training:
            attn_probs = F.dropout(attn_probs, p=self.attention_dropout, training=True)

        value_states = v_full  # (b, num_heads, s_kv, head_dim)
        attn_output = torch.matmul(attn_probs, value_states)  # (b, num_heads, q_len, head_dim)

        # 9) Merge heads + output projection
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(bsz, q_len, -1)    # let PyTorch infer num_heads*head_dim
        )
        attn_output = self.o_proj(attn_output)

        present_kv = (k_comp, v_comp) if use_cache else None
        return attn_output, present_kv


# -------------------------
# Compressed Gemma2 layer
# -------------------------

class CompressedGemma2Layer(nn.Module):
    """
    A Gemma2 decoder layer that:
      - reuses all learned submodules from the original layer
      - swaps self_attn with the compressed-KV version
    """

    def __init__(
        self,
        base_layer: nn.Module,
        kv_compression_ratio: float = 1.0,
        rope_factor: float = 1.0,
    ):
        super().__init__()
        # Reuse the existing norms and MLP
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm
        self.pre_feedforward_layernorm = base_layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = base_layer.post_feedforward_layernorm
        self.mlp = base_layer.mlp

        # Is this a sliding-window layer?
        is_sliding = getattr(base_layer, "attention_type", None) == "sliding_attention"
        config = base_layer.config

        # New attention wrapper that reuses Q/K/V/O weights
        self.self_attn = CompressedKVGemma2Attention(
            base_attn=base_layer.self_attn,
            config=config,
            is_sliding=is_sliding,
            kv_compression_ratio=kv_compression_ratio,
            rope_factor=rope_factor,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,           # (b, t, d_model)
        position_ids: torch.LongTensor,        # (b, t)
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_kv = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + self.post_attention_layernorm(attn_output)

        # Feed-forward block
        residual = hidden_states
        ff_out = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        hidden_states = residual + self.post_feedforward_layernorm(ff_out)

        return hidden_states, present_kv


# -------------------------
# Top-level compressed BgGPT LM
# -------------------------

class CompressedBgGPTForCausalLM(nn.Module):
    """
    Lightweight wrapper around a Gemma2/BgGPT model that:

      - reuses embeddings, MLPs, norms and lm_head from the base model
      - swaps each decoder layer for a CompressedGemma2Layer
      - uses its own simple cache format: list of (k_comp, v_comp)
      - implements its own RoPE scaling to ~64k via CompressedKVGemma2Attention
    """

    def __init__(
        self,
        base_model: nn.Module,
        kv_compression_ratio: float = 1.0,
        rope_factor: float = 1.0,
    ):
        super().__init__()
        self.config = base_model.config

        # Reuse embedding, layers, final norm, lm_head
        self.embed_tokens = base_model.model.embed_tokens
        self.layers = nn.ModuleList(
            [
                CompressedGemma2Layer(
                    layer,
                    kv_compression_ratio=kv_compression_ratio,
                    rope_factor=rope_factor,
                )
                for layer in base_model.model.layers
            ]
        )
        self.norm = base_model.model.norm
        self.lm_head = base_model.lm_head

        # For final logit softcapping
        self.final_logit_softcapping = getattr(
            self.config, "final_logit_softcapping", None
        )

    def _build_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Standard lower-triangular causal mask: shape (1, 1, seq_len, seq_len)
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        input_ids: (b, t)
        past_key_values: list of length n_layers with (k_comp, v_comp) per layer, or None.
        """
        bsz, seq_len = input_ids.size()
        device = input_ids.device

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            past_len = 0
        else:
            first = past_key_values[0]
            past_len = 0 if first is None else first[0].size(2)

        # positions: [past_len, ..., past_len+seq_len-1]
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=device, dtype=torch.long
        ).unsqueeze(0).expand(bsz, -1)  # (b, t)

        # token embeddings
        hidden_states = self.embed_tokens(input_ids)

        # *** CRITICAL: match Gemma2 embed scaling *** 
        normalizer = torch.tensor(
            self.config.hidden_size ** 0.5,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states = hidden_states * normalizer

        # For the initial prompt with no cache, full causal mask.
        if past_len == 0 and seq_len > 1:
            attention_mask = self._build_causal_mask(
                seq_len, device, hidden_states.dtype
            )
        else:
            attention_mask = None

        new_past = [] if use_cache else None

        for layer, layer_past in zip(self.layers, past_key_values):
            hidden_states, present_kv = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                new_past.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # *** Match Gemma2 final logit softcapping ***
        if self.final_logit_softcapping is not None:
            softcap = self.final_logit_softcapping
            logits = logits / softcap
            logits = torch.tanh(logits)
            logits = logits * softcap

        return logits, new_past

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: Optional[int] = 40,
    ) -> torch.LongTensor:
        """
        Simple autoregressive sampling using the compressed KV cache.
        """
        self.eval()
        generated = input_ids
        past_kv = None

        for _ in range(max_new_tokens):
            if past_kv is None:
                logits, past_kv = self(generated, past_key_values=None, use_cache=True)
            else:
                logits, past_kv = self(
                    generated[:, -1:], past_key_values=past_kv, use_cache=True
                )

            next_logits = logits[:, -1, :]  # (b, vocab_size)

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_k is not None and top_k > 0:
                top_values, top_indices = torch.topk(next_logits, top_k, dim=-1)
                filtered = torch.full_like(next_logits, float("-inf"))
                filtered.scatter_(dim=-1, index=top_indices, src=top_values)
                next_logits = filtered

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (b, 1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


# -------------------------
# Convenience loader
# -------------------------

def load_bggpt_compressed(
    model_name: str = "INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0",
    kv_compression_ratio: float = 1.0,
    rope_factor: float = 1.0,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        attn_implementation="eager",
        device_map=None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_default_system_prompt=False,
    )

    compressed_model = CompressedBgGPTForCausalLM(
        base_model=base_model,
        kv_compression_ratio=kv_compression_ratio,
        rope_factor=rope_factor,
    )

    # 🔴 This is the important part: force *all* params/buffers to same dtype + device
    compressed_model.to(device=device, dtype=dtype)

    return compressed_model, tokenizer

from transformers import BitsAndBytesConfig

def load_bggpt_compressed_int8(
    model_name: str = "INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0",
    kv_compression_ratio: float = 1.0,   # you can later set 0.25 etc.
    rope_factor: float = 1.0,            # later: 8.0 for ~64k
    device: str = "cuda",
):
    """
    Load BgGPT with bitsandbytes 8-bit weight quantization, then wrap with
    our compressed-KV + RoPE-scaling model.

    - Weights: int8 under the hood (LLM.int8)
    - Activations + KV + compressors: fp16
    """

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,           # turn on 8-bit weights
        llm_int8_threshold=6.0,      # default threshold
        llm_int8_has_fp16_weight=False,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,  # HF v4.46+ quantization API 
        device_map=None,                 # single GPU (no sharding, simpler for our wrapper)
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_default_system_prompt=False,
    )

    # Wrap with our compressed KV model.
    compressed_model = CompressedBgGPTForCausalLM(
        base_model=base_model,
        kv_compression_ratio=kv_compression_ratio,
        rope_factor=rope_factor,
    ).to(device)

    # Make sure KVCompressor weights are fp16 on the GPU.
    for module in compressed_model.modules():
        if isinstance(module, KVCompressor):
            module.to(device=device, dtype=torch.float16)

    return compressed_model, tokenizer


# compressed_kv_transformer.py
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Config
# -------------------------

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    d_ff: int = 2048
    max_seq_len: int = 65536          # "new" context length (64k)
    orig_max_seq_len: int = 8192      # "trained" context length (for RoPE scaling)
    kv_compression_ratio: float = 0.5 # head_dim -> head_dim * ratio in KV cache
    dropout: float = 0.0
    rope_base: float = 10000.0        # standard RoPE base


# -------------------------
# RoPE with linear scaling to 64k
# -------------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary embeddings with simple linear scaling of positions:

        scaled_pos = pos / scaling_factor

    where scaling_factor = new_max_seq_len / orig_max_seq_len.

    This mimics HF's "linear" rope_scaling idea:
    using the same RoPE angular range over a bigger context.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        orig_max_position: int = 8192,
        new_max_position: int = 65536,
    ):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        self.dim = dim
        self.base = base
        self.orig_max_position = orig_max_position
        self.new_max_position = new_max_position
        self.scaling_factor = float(new_max_position) / float(orig_max_position)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_cos_sin(
        self,
        position_ids: torch.LongTensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        position_ids: (batch, seq_len)
        returns cos, sin: (batch, 1, seq_len, dim)
        """
        # Scale positions down so longer context fits into same angular range
        scaled_pos = position_ids.float() / self.scaling_factor  # (b, s)

        # (b, s, dim/2)
        freqs = torch.einsum(
            "bs,d->bsd",
            scaled_pos.to(device),
            self.inv_freq.to(device),
        )
        # Duplicate to full dim: (..., dim)
        emb = torch.cat([freqs, freqs], dim=-1)  # (b, s, dim)

        cos = emb.cos().to(dtype).unsqueeze(1)  # (b, 1, s, dim)
        sin = emb.sin().to(dtype).unsqueeze(1)  # (b, 1, s, dim)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half: (x_even, x_odd) -> (-x_odd, x_even)
        x: (..., dim)
        """
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., ::2] = -x_odd
        out[..., 1::2] = x_even
        return out

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (b, n_heads, seq_len, head_dim)
        position_ids: (b, seq_len)
        """
        bsz, n_heads, seq_len, dim = q.shape
        device = q.device
        dtype = q.dtype

        cos, sin = self._get_cos_sin(position_ids, device, dtype)  # (b, 1, s, dim)

        # broadcast cos/sin to (b, n_heads, s, dim)
        cos = cos.expand(bsz, n_heads, seq_len, dim)
        sin = sin.expand(bsz, n_heads, seq_len, dim)

        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# -------------------------
# KV Compressor
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

        # Optional: initialize as near-identity in subspace
        with torch.no_grad():
            nn.init.zeros_(self.compress.weight)
            nn.init.zeros_(self.decompress.weight)
            k = min(d_model, d_latent)
            self.compress.weight[:k, :k] = torch.eye(k)
            self.decompress.weight[:k, :k] = torch.eye(k)

    def forward_compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d_model)
        returns: (..., d_latent)
        """
        return self.compress(x)

    def forward_decompress(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (..., d_latent)
        returns: (..., d_model)
        """
        return self.decompress(z)


# -------------------------
# Compressed-KV Multihead Attention
# -------------------------

class CompressedKVMultiheadAttention(nn.Module):
    """
    Decoder-only causal attention with compressed KV cache.

    - Inputs: hidden_states (b, t, d_model)
    - Stores compressed KV in past_key_value.
    - On each forward, decompresses KV before attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_compression_ratio: float,
        dropout: float,
        rotary_emb: RotaryEmbedding,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.rotary_emb = rotary_emb

        # Standard QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Compressed KV dimension
        latent_dim = max(1, int(self.head_dim * kv_compression_ratio))
        self.k_compressor = KVCompressor(self.head_dim, latent_dim)
        self.v_compressor = KVCompressor(self.head_dim, latent_dim)

    def _shape(self, x: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        # (b, s, d_model) -> (b, n_heads, s, head_dim)
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        hidden_states: (b, t, d_model)
        position_ids:  (b, t)
        attention_mask: (b, 1, t, s_kv) or None
        past_key_value: tuple(compressed_k, compressed_v)
            compressed_k: (b, n_heads, s_past, d_latent)
            compressed_v: (b, n_heads, s_past, d_latent)
        """
        bsz, seq_len, _ = hidden_states.size()

        # 1) Project to Q, K, V
        q = self.q_proj(hidden_states)  # (b, t, d_model)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 2) Reshape to heads
        q = self._shape(q, bsz, seq_len)  # (b, h, t, d_head)
        k = self._shape(k, bsz, seq_len)
        v = self._shape(v, bsz, seq_len)

        # 3) Compress new K, V for cache
        k_comp_new = self.k_compressor.forward_compress(k)  # (b, h, t, d_latent)
        v_comp_new = self.v_compressor.forward_compress(v)  # (b, h, t, d_latent)

        if past_key_value is not None:
            k_comp_past, v_comp_past = past_key_value
            k_comp = torch.cat([k_comp_past, k_comp_new], dim=2)  # (b, h, s_kv, d_latent)
            v_comp = torch.cat([v_comp_past, v_comp_new], dim=2)
        else:
            k_comp, v_comp = k_comp_new, v_comp_new

        # 4) Decompress for attention
        k_full = self.k_compressor.forward_decompress(k_comp)  # (b, h, s_kv, d_head)
        v_full = self.v_compressor.forward_decompress(v_comp)  # (b, h, s_kv, d_head)

        # 5) Apply RoPE to q and k_full
        q, k_full = self.rotary_emb.apply_rotary(q, k_full, position_ids)

        # 6) Scaled dot-product attention
        # q: (b, h, t, d_head), k_full: (b, h, s_kv, d_head)
        attn_scores = torch.matmul(q, k_full.transpose(-1, -2))  # (b, h, t, s_kv)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask  # add large negative on masked positions

        attn_probs = F.softmax(attn_scores, dim=-1)
        if self.dropout > 0.0 and self.training:
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=True)

        attn_output = torch.matmul(attn_probs, v_full)  # (b, h, t, d_head)

        # 7) Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)

        present_kv = (k_comp, v_comp) if use_cache else None
        return attn_output, present_kv


# -------------------------
# Feedforward / Block / LM
# -------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, rotary_emb: RotaryEmbedding):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

        self.self_attn = CompressedKVMultiheadAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            kv_compression_ratio=config.kv_compression_ratio,
            dropout=config.dropout,
            rotary_emb=rotary_emb,
        )
        self.mlp = FeedForward(config.d_model, config.d_ff, config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)

        attn_output, present_kv = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, present_kv


class CompressedKVTransformerLM(nn.Module):
    """
    Simple decoder-only LM with:
      - RoPE (extended to 64k via linear scaling)
      - Compressed KV cache
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        head_dim = config.d_model // config.n_heads
        self.rotary_emb = RotaryEmbedding(
            dim=head_dim,
            base=config.rope_base,
            orig_max_position=config.orig_max_seq_len,
            new_max_position=config.max_seq_len,
        )

        self.layers = nn.ModuleList(
            [TransformerBlock(config, self.rotary_emb) for _ in range(config.n_layers)]
        )

        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def _build_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Standard lower-triangular causal mask for full-sequence forward.
        returns shape: (1, 1, seq_len, seq_len)
        """
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)  # upper triangle gets -inf
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
        past_key_values: list of length n_layers, each (k_comp, v_comp) or None
        """
        bsz, seq_len = input_ids.size()
        device = input_ids.device
        dtype = self.embed_tokens.weight.dtype

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            past_len = 0
        else:
            # compressed_k: (b, h, s_past, d_latent)
            past_len = past_key_values[0][0].size(2)

        # positions: [past_len, ..., past_len+seq_len-1]
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=device, dtype=torch.long
        ).unsqueeze(0).expand(bsz, -1)  # (b, t)

        # embed tokens
        hidden_states = self.embed_tokens(input_ids)  # (b, t, d_model)

        # for the very first forward of the prompt we build a full causal mask
        # for subsequent decode steps (seq_len=1) we let attention_mask = None
        if past_len == 0 and seq_len > 1:
            attention_mask = self._build_causal_mask(seq_len, device, hidden_states.dtype)
        else:
            attention_mask = None

        new_past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
        new_past_key_values = [] if use_cache else None

        for layer, layer_past in zip(self.layers, past_key_values):
            hidden_states, present_kv = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(present_kv)

        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)  # (b, t, vocab_size)
        return logits, new_past_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Very simple autoregressive sampler (no batching tricks).
        """
        self.eval()
        device = input_ids.device
        generated = input_ids
        past_kv = None

        for _ in range(max_new_tokens):
            logits, past_kv = self(generated[:, -1:].to(device) if past_kv is not None else generated,  # step vs prompt
                                   past_key_values=past_kv,
                                   use_cache=True)
            next_logits = logits[:, -1, :]  # (b, vocab_size)

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_k is not None and top_k > 0:
                # top-k filtering
                values, indices = torch.topk(next_logits, top_k, dim=-1)
                probs = torch.zeros_like(next_logits).scatter_(-1, indices, values)
                next_logits = probs

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (b, 1)

            generated = torch.cat([generated, next_token], dim=1)

        return generated


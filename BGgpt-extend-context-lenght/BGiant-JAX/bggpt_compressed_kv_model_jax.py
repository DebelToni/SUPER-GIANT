"""
JAX/FLAX implementation of compressed KV BgGPT model.
Converted from PyTorch version in BGiant/bggpt_compressed_kv_model.py
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, List
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

from omegaconf import OmegaConf


def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)


PARAM_DTYPE = jnp.float32
COMPUTE_DTYPE = jnp.bfloat16


class Gemma2RMSNorm(nn.Module):
    """Gemma2 RMSNorm (x * (1 + w))."""
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        weight = self.param(
            'scale',
            nn.initializers.zeros,
            (x.shape[-1],),
            self.param_dtype,
        )
        weight = jnp.asarray(weight, self.dtype)
        
        # Calculate RMS
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        hidden_states = x * jax.lax.rsqrt(variance + self.epsilon)
        
        # Gemma2 scaling: (1 + weight)
        return hidden_states * (1.0 + weight)


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    """Helper for RoPE: splits last dim [x1, x2] and returns [-x2, x1]."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return jnp.concatenate((-x2, x1), axis=-1)


class KVCompressor(nn.Module):
    """Linear compressor/decompressor for KV cache."""
    d_model: int
    d_latent: int
    dtype: jnp.dtype = COMPUTE_DTYPE
    
    def setup(self):
        assert self.d_latent > 0 and self.d_latent <= self.d_model
        
        # Initialize with near-identity in shared subspace
        k = min(self.d_model, self.d_latent)
        
        def compress_init(key, shape, dtype=jnp.float32):
            w = jnp.zeros(shape, dtype=dtype)
            w = w.at[:k, :k].set(jnp.eye(k, dtype=dtype))
            return w
            
        def decompress_init(key, shape, dtype=jnp.float32):
            w = jnp.zeros(shape, dtype=dtype)
            w = w.at[:k, :k].set(jnp.eye(k, dtype=dtype))
            return w
        
        self.compress_proj = nn.Dense(
            self.d_latent,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            kernel_init=compress_init,
            name="compress"
        )
        self.decompress_proj = nn.Dense(
            self.d_model,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            kernel_init=decompress_init,
            name="decompress"
        )
    
    def forward_compress(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.compress_proj(x)
    
    def forward_decompress(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decompress_proj(z)


class CompressedKVGemma2Attention(nn.Module):
    """Gemma2-style attention with compressed KV cache and RoPE scaling."""
    hidden_size: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    sliding_window: Optional[int] = None
    attention_dropout: float = 0.0
    attn_logit_softcapping: Optional[float] = None
    query_pre_attn_scalar: Optional[float] = None
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    kv_compression_ratio: float = 1.0
    dtype: jnp.dtype = COMPUTE_DTYPE
    
    def setup(self):
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        
        if self.query_pre_attn_scalar is not None:
            self.scaling = self.query_pre_attn_scalar ** -0.5
        else:
            self.scaling = 1.0 / math.sqrt(self.head_dim)
        
        # Q/K/V/O projections
        self.q_proj = nn.Dense(
            self.num_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            name="q_proj"
        )
        self.k_proj = nn.Dense(
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            name="k_proj"
        )
        self.v_proj = nn.Dense(
            self.num_kv_heads * self.head_dim,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            name="v_proj"
        )
        self.o_proj = nn.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            name="o_proj"
        )
        
        # RoPE precomputation
        inv_freq = 1.0 / (
            self.rope_theta ** (jnp.arange(0, self.head_dim, 2, dtype=jnp.float32) / self.head_dim)
        )
        self.inv_freq = inv_freq
        
        # KV compressors
        latent_dim = max(1, int(self.head_dim * self.kv_compression_ratio))
        self.k_compressor = KVCompressor(self.head_dim, latent_dim, dtype=self.dtype)
        self.v_compressor = KVCompressor(self.head_dim, latent_dim, dtype=self.dtype)
    
    def _scaled_rope(self, s_kv: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute RoPE cos/sin with linear position scaling."""
        positions = jnp.arange(0, s_kv, dtype=jnp.float32)
        scaled_pos = positions / self.rope_factor
        
        # (1, head_dim/2, 1) @ (1, 1, s_kv) -> (1, head_dim/2, s_kv)
        inv_freq = self.inv_freq[None, :, None]
        pos_expanded = scaled_pos[None, None, :]
        
        freqs = (inv_freq @ pos_expanded).transpose(0, 2, 1)  # (1, s_kv, head_dim/2)
        emb = jnp.concatenate([freqs, freqs], axis=-1)  # (1, s_kv, head_dim)
        cos = jnp.cos(emb).astype(self.dtype)
        sin = jnp.sin(emb).astype(self.dtype)
        return cos, sin
    
    def _apply_rope_qk(
        self,
        q: jnp.ndarray,         # (b, num_heads, q_len, head_dim)
        k: jnp.ndarray,         # (b, num_kv_heads, s_kv, head_dim)
        cos_full: jnp.ndarray,  # (1, s_kv, head_dim)
        sin_full: jnp.ndarray,  # (1, s_kv, head_dim)
        q_len: int,
        cache_position: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply RoPE to queries and keys."""
        bsz = q.shape[0]
        s_kv = k.shape[2]
        
        # Broadcast to batch
        cos_full = jnp.broadcast_to(cos_full, (bsz, s_kv, self.head_dim))
        sin_full = jnp.broadcast_to(sin_full, (bsz, s_kv, self.head_dim))
        
        # Keys: all positions
        cos_k = cos_full[:, None, :, :]  # (b, 1, s_kv, d)
        sin_k = sin_full[:, None, :, :]
        
        # Queries
        if cache_position is not None:
            # Dynamic slice for query position
            # cache_position is scalar-like index
            # We want cos_full[:, cache_position:cache_position+q_len, :]
            # But cos_full has batch dim now.
            
            # Use dynamic_slice
            # start_indices: (0, cache_position, 0)
            # slice_sizes: (bsz, q_len, head_dim)
            
            # We need to broadcast cache_position to match batch if it's per-batch?
            # Usually cache_position is shared or we assume bsz=1 for simple inference loop
            # But here cos_full is (bsz, s_kv, d).
            
            # If cache_position is scalar (0-D array), we can use it directly in dynamic_slice
            # if we replicate it or use vmap.
            # But simpler: slice from the original (1, s_kv, d) BEFORE broadcasting
            # then broadcast.
            pass
            
        # Re-implementing to handle cache_position correctly
        
        if cache_position is not None:
            # Slice from original un-broadcasted cos/sin (1, s_kv, d)
            # Assuming cos_full passed in is (1, s_kv, d) - wait, I broadcasted it above.
            # Let's use the original input args if possible, but I overwrote them.
            # Let's undo the overwrite or use a new var.
            
            # Actually, let's slice from the broadcasted one, it's fine.
            # start: (0, cache_position, 0)
            # size: (bsz, q_len, self.head_dim)
            
            start_indices = (0, cache_position, 0)
            slice_sizes = (bsz, q_len, self.head_dim)
            
            cos_q = jax.lax.dynamic_slice(cos_full, start_indices, slice_sizes)
            sin_q = jax.lax.dynamic_slice(sin_full, start_indices, slice_sizes)
            
            cos_q = cos_q[:, None, :, :]
            sin_q = sin_q[:, None, :, :]
        else:
            # Standard behavior: last q_len positions
            cos_q = cos_full[:, -q_len:, :][:, None, :, :]  # (b, 1, q_len, d)
            sin_q = sin_full[:, -q_len:, :][:, None, :, :]
        
        # Apply RoPE
        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
        
        return q_embed, k_embed
    
    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        use_cache: bool = True,
        deterministic: bool = True,
        cache_position: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        
        bsz, q_len, _ = hidden_states.shape
        
        # Q, K, V projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to heads
        q = q.reshape(bsz, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Compress new KV
        k_comp_new = self.k_compressor.forward_compress(k)
        v_comp_new = self.v_compressor.forward_compress(v)
        
        if past_key_value is not None:
            past_k_comp, past_v_comp = past_key_value
            
            if cache_position is not None:
                # Fixed cache update
                # cache_position should be [1] or scalar, we need to slice
                # Assuming cache_position is the start index for update
                
                # Use dynamic_update_slice
                # k_comp_new shape: (bsz, num_kv_heads, q_len, latent_dim)
                # past_k_comp shape: (bsz, num_kv_heads, max_len, latent_dim)
                
                # We need to broadcast cache_position to match batch dims if needed, 
                # but dynamic_update_slice takes start_indices.
                
                # For batch processing, we might need vmap if indices differ per batch,
                # but usually in inference they are same.
                
                start_indices = (0, 0, cache_position, 0)
                
                # Cast to cache dtype if needed
                if k_comp_new.dtype != past_k_comp.dtype:
                    k_comp_new = k_comp_new.astype(past_k_comp.dtype)
                if v_comp_new.dtype != past_v_comp.dtype:
                    v_comp_new = v_comp_new.astype(past_v_comp.dtype)
                
                k_comp = jax.lax.dynamic_update_slice(past_k_comp, k_comp_new, start_indices)
                v_comp = jax.lax.dynamic_update_slice(past_v_comp, v_comp_new, start_indices)
            else:
                # Append mode (growing cache)
                k_comp = jnp.concatenate([past_k_comp, k_comp_new], axis=2)
                v_comp = jnp.concatenate([past_v_comp, v_comp_new], axis=2)
        else:
            k_comp, v_comp = k_comp_new, v_comp_new
        
        # Apply sliding window
        if self.sliding_window is not None and k_comp.shape[2] > self.sliding_window:
            # For fixed cache, we don't slice here, we assume the cache is circular or large enough.
            # But if we are just appending, we slice.
            # If using cache_position, we assume the caller handles windowing or the cache is large enough.
            if cache_position is None:
                k_comp = k_comp[:, :, -self.sliding_window:, :]
                v_comp = v_comp[:, :, -self.sliding_window:, :]
        
        # Decompress KV
        k_full = self.k_compressor.forward_decompress(k_comp)
        v_full = self.v_compressor.forward_decompress(v_comp)
        s_kv = k_full.shape[2]
        
        # Apply RoPE
        cos_full, sin_full = self._scaled_rope(s_kv)
        q, k_full = self._apply_rope_qk(q, k_full, cos_full, sin_full, q_len, cache_position)
        
        # Expand KV heads via GQA
        if self.num_key_value_groups > 1:
            k_full = jnp.repeat(k_full, self.num_key_value_groups, axis=1)
            v_full = jnp.repeat(v_full, self.num_key_value_groups, axis=1)
        
        # Scaled dot-product attention
        # Gemma2: If softcapping is used, do NOT use 1/sqrt(d) scaling
        attn_scores = jnp.matmul(q, k_full.transpose(0, 1, 3, 2)) * self.scaling
        
        if self.attn_logit_softcapping is not None:
            softcap = self.attn_logit_softcapping
            attn_scores = attn_scores / softcap
            attn_scores = jnp.tanh(attn_scores)
            attn_scores = attn_scores * softcap
        
        # Logit soft-capping
        if self.attn_logit_softcapping is not None:
            attn_scores = attn_scores / self.attn_logit_softcapping
            attn_scores = jnp.tanh(attn_scores)
            attn_scores = attn_scores * self.attn_logit_softcapping
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        
        if self.attention_dropout > 0 and not deterministic:
            attn_probs = nn.Dropout(rate=self.attention_dropout)(attn_probs, deterministic=deterministic)
        
        attn_output = jnp.matmul(attn_probs, v_full)
        
        # Merge heads
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        
        present_kv = (k_comp, v_comp) if use_cache else None
        return attn_output, present_kv


class CompressedGemma2Layer(nn.Module):
    """Gemma2 decoder layer with compressed KV attention."""
    hidden_size: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    d_ff: int
    is_sliding: bool = False
    sliding_window: Optional[int] = None
    attention_dropout: float = 0.0
    attn_logit_softcapping: Optional[float] = None
    query_pre_attn_scalar: Optional[float] = None
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    kv_compression_ratio: float = 1.0
    dtype: jnp.dtype = COMPUTE_DTYPE
    
    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        use_cache: bool = True,
        deterministic: bool = True,
        cache_position: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        
        # Self-attention block
        residual = hidden_states
        hidden_states = Gemma2RMSNorm(epsilon=1e-6, dtype=self.dtype, name="input_layernorm")(hidden_states)
        
        attn_output, present_kv = CompressedKVGemma2Attention(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            sliding_window=self.sliding_window if self.is_sliding else None,
            attention_dropout=self.attention_dropout,
            attn_logit_softcapping=self.attn_logit_softcapping,
            query_pre_attn_scalar=self.query_pre_attn_scalar,
            rope_theta=self.rope_theta,
            rope_factor=self.rope_factor,
            kv_compression_ratio=self.kv_compression_ratio,
            dtype=self.dtype,
            name="self_attn"
        )(hidden_states, attention_mask, past_key_value, use_cache, deterministic, cache_position)
        
        hidden_states = residual + Gemma2RMSNorm(epsilon=1e-6, dtype=self.dtype, name="post_attention_layernorm")(attn_output)
        
        # Feed-forward block
        residual = hidden_states
        ff_input = Gemma2RMSNorm(epsilon=1e-6, dtype=self.dtype, name="pre_feedforward_layernorm")(hidden_states)
        
        # SwiGLU MLP
        gate_proj = nn.Dense(
            self.d_ff,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            name="gate_proj"
        )(ff_input)
        up_proj = nn.Dense(
            self.d_ff,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            name="up_proj"
        )(ff_input)
        
        ff_hidden = nn.silu(gate_proj) * up_proj
        
        ff_output = nn.Dense(
            self.hidden_size,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            name="down_proj"
        )(ff_hidden)
        
        hidden_states = residual + Gemma2RMSNorm(epsilon=1e-6, dtype=self.dtype, name="post_feedforward_layernorm")(ff_output)
        
        return hidden_states, present_kv


class CompressedBgGPTForCausalLM(nn.Module):
    """JAX/FLAX compressed BgGPT model."""
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    d_ff: int
    sliding_window: Optional[int] = None
    attention_dropout: float = 0.0
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    query_pre_attn_scalar: Optional[float] = None
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    kv_compression_ratio: float = 1.0
    dtype: jnp.dtype = COMPUTE_DTYPE
    
    def setup(self):
        # Embedding
        self.embed_tokens = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            name="embed_tokens"
        )
        
        # Normalizer for embeddings (Gemma2 style)
        self.embed_normalizer = math.sqrt(self.hidden_size)
    
    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        past_key_values: Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]] = None,
        use_cache: bool = True,
        deterministic: bool = True,
        cache_position: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, Optional[List[Tuple[jnp.ndarray, jnp.ndarray]]]]:
        
        bsz, seq_len = input_ids.shape
        
        if past_key_values is None:
            past_key_values = [None] * self.num_layers
            past_len = 0
        else:
            first = past_key_values[0]
            past_len = 0 if first is None else first[0].shape[2]
        
        # Token embeddings with Gemma2 scaling
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * self.embed_normalizer
        
        # Causal mask for initial prompt
        if past_len == 0 and seq_len > 1:
            mask = jnp.triu(jnp.full((seq_len, seq_len), float("-inf")), k=1)
            attention_mask = mask[None, None, :, :]
        else:
            attention_mask = None
        
        new_past = [] if use_cache else None
        
        # Pass through layers
        for i in range(self.num_layers):
            # Alternate sliding/global attention in Gemma2 style
            is_sliding = (i % 2 == 0)  # Even layers use sliding window
            
            layer_past = past_key_values[i] if past_key_values else None
            hidden_states, present_kv = CompressedGemma2Layer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                num_kv_heads=self.num_kv_heads,
                d_ff=self.d_ff,
                is_sliding=is_sliding,
                sliding_window=self.sliding_window,
                attention_dropout=self.attention_dropout,
                attn_logit_softcapping=self.attn_logit_softcapping,
                query_pre_attn_scalar=self.query_pre_attn_scalar,
                rope_theta=self.rope_theta,
                rope_factor=self.rope_factor,
                kv_compression_ratio=self.kv_compression_ratio,
                dtype=self.dtype,
                name=f"layer_{i}"
            )(hidden_states, attention_mask, layer_past, use_cache, deterministic, cache_position)
            
            if use_cache:
                new_past.append(present_kv)
        
        # Final norm
        hidden_states = Gemma2RMSNorm(epsilon=1e-6, dtype=self.dtype, name="norm")(hidden_states)
        
        # LM head (tied with embedding)
        logits = jnp.einsum("bld,vd->blv", hidden_states.astype(jnp.float32), self.embed_tokens.embedding)
        
        # Final logit soft-capping
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = jnp.tanh(logits)
            logits = logits * self.final_logit_softcapping
        
        return logits, new_past


def create_bggpt_compressed_model(
    vocab_size: int = 256000,
    hidden_size: int = 2304,
    num_layers: int = 26,
    num_heads: int = 8,
    num_kv_heads: int = 4,
    d_ff: int = 9216,
    head_dim: int = 256,  # Explicitly set head_dim
    sliding_window: int = 4096,
    kv_compression_ratio: float = 1.0,
    rope_factor: float = 1.0,
    attn_logit_softcapping: float = 50.0,
    final_logit_softcapping: float = 30.0,
    query_pre_attn_scalar: int = 256,
) -> CompressedBgGPTForCausalLM:
    """Create a compressed BgGPT model with Gemma2 configuration."""
    # head_dim = hidden_size // num_heads  <-- This was the bug! 2304/8 = 288, but Gemma2 uses 256
    
    return CompressedBgGPTForCausalLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        d_ff=d_ff,
        sliding_window=sliding_window,
        attention_dropout=0.0,
        attn_logit_softcapping=attn_logit_softcapping,
        final_logit_softcapping=final_logit_softcapping,
        query_pre_attn_scalar=query_pre_attn_scalar,
        rope_theta=10000.0,
        rope_factor=rope_factor,
        kv_compression_ratio=kv_compression_ratio,
        dtype=COMPUTE_DTYPE,
    )

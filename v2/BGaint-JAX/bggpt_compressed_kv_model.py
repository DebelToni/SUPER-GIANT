"""
JAX/Flax port of the compressed KV Gemma2/BgGPT wrapper.

This mirrors the PyTorch version's data flow:
  - linear Q/K/V/O
  - compressed KV cache (compress + decompress)
  - scaled RoPE for extended context
  - causal attention with optional sliding window
  - simple SwiGLU MLP + RMSNorm
"""

import math
from dataclasses import dataclass, replace
from typing import List, NamedTuple, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


# -------------------------
# Config and helpers
# -------------------------


@dataclass
class ModelConfig:
    hidden_size: int = 2304
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: Optional[int] = None  # defaults to hidden_size // num_attention_heads
    vocab_size: int = 256000
    intermediate_size: int = 9216
    num_hidden_layers: int = 2
    attention_dropout: float = 0.0
    attn_logit_softcapping: Optional[float] = 50.0
    final_logit_softcapping: Optional[float] = 30.0
    query_pre_attn_scalar: Optional[float] = 256.0
    rope_theta: float = 10000.0
    rope_factor: float = 1.0
    sliding_window: Optional[int] = None
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    kv_compression_ratio: float = 1.0
    layer_sliding_windows: Optional[List[Optional[int]]] = None

    @property
    def actual_head_dim(self) -> int:
        return self.head_dim or (self.hidden_size // self.num_attention_heads)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def latent_dim(self) -> int:
        return max(1, int(self.actual_head_dim * self.kv_compression_ratio))


class KVCache(NamedTuple):
    k_comp: jnp.ndarray
    v_comp: jnp.ndarray


def rotate_half(x: jnp.ndarray) -> jnp.ndarray:
    d2 = x.shape[-1] // 2
    return jnp.concatenate([-x[..., d2:], x[..., :d2]], axis=-1)


def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    b, kv_heads, seq_len, dim = x.shape
    x = jnp.repeat(x, repeats=n_rep, axis=1)
    return x.reshape(b, kv_heads * n_rep, seq_len, dim)


def identity_subspace_init():
    def init(key, shape, dtype=jnp.float32):
        w = jnp.zeros(shape, dtype=dtype)
        k = min(shape[0], shape[1])
        return w.at[:k, :k].set(jnp.eye(k, dtype=dtype))

    return init


def scaled_rope(
    s_kv: int,
    head_dim: int,
    rope_theta: float,
    rope_factor: float,
    dtype=jnp.float32,
):
    inv_freq = 1.0 / (
        rope_theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim)
    )
    positions = jnp.arange(0, s_kv, dtype=jnp.float32)
    scaled_pos = positions / rope_factor
    freqs = jnp.einsum("d,s->sd", inv_freq, scaled_pos)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb).astype(dtype)
    sin = jnp.sin(emb).astype(dtype)
    return cos, sin


# -------------------------
# Modules
# -------------------------


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        norm = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        return x * (scale / norm)


class MLP(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic: bool):
        hidden = nn.Dense(self.config.intermediate_size, use_bias=False)(x)
        gate = nn.Dense(self.config.intermediate_size, use_bias=False)(x)
        activated = nn.silu(gate) * hidden
        activated = nn.Dropout(rate=self.config.dropout)(activated, deterministic=deterministic)
        out = nn.Dense(self.config.hidden_size, use_bias=False)(activated)
        return out


class KVCompressor(nn.Module):
    d_model: int
    d_latent: int

    @nn.compact
    def __call__(self, x, mode: str):
        if mode == "compress":
            layer = nn.Dense(
                self.d_latent,
                use_bias=False,
                kernel_init=identity_subspace_init(),
            )
        else:
            layer = nn.Dense(
                self.d_model,
                use_bias=False,
                kernel_init=identity_subspace_init(),
            )
        return layer(x)


class CompressedKVGemma2Attention(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        past_key_value: Optional[KVCache],
        use_cache: bool,
        deterministic: bool,
    ) -> Tuple[jnp.ndarray, Optional[KVCache]]:
        cfg = self.config
        head_dim = cfg.actual_head_dim
        num_heads = cfg.num_attention_heads
        num_kv_heads = cfg.num_key_value_heads
        bsz, q_len, _ = hidden_states.shape

        q_proj = nn.Dense(num_heads * head_dim, use_bias=False, name="q_proj")
        k_proj = nn.Dense(num_kv_heads * head_dim, use_bias=False, name="k_proj")
        v_proj = nn.Dense(num_kv_heads * head_dim, use_bias=False, name="v_proj")
        o_proj = nn.Dense(cfg.hidden_size, use_bias=False, name="o_proj")
        k_compressor = KVCompressor(head_dim, cfg.latent_dim, name="k_compressor")
        v_compressor = KVCompressor(head_dim, cfg.latent_dim, name="v_compressor")

        q = q_proj(hidden_states)
        k = k_proj(hidden_states)
        v = v_proj(hidden_states)

        q = q.reshape(bsz, q_len, num_heads, head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, q_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, q_len, num_kv_heads, head_dim).transpose(0, 2, 1, 3)

        k_comp_new = k_compressor(k, mode="compress")
        v_comp_new = v_compressor(v, mode="compress")

        if past_key_value is not None:
            k_comp = jnp.concatenate([past_key_value.k_comp, k_comp_new], axis=2)
            v_comp = jnp.concatenate([past_key_value.v_comp, v_comp_new], axis=2)
        else:
            k_comp, v_comp = k_comp_new, v_comp_new

        if cfg.sliding_window is not None and k_comp.shape[2] > cfg.sliding_window:
            k_comp = k_comp[:, :, -cfg.sliding_window :, :]
            v_comp = v_comp[:, :, -cfg.sliding_window :, :]

        k_full = k_compressor(k_comp, mode="decompress")
        v_full = v_compressor(v_comp, mode="decompress")
        s_kv = k_full.shape[2]

        cos_full, sin_full = scaled_rope(
            s_kv=s_kv,
            head_dim=head_dim,
            rope_theta=cfg.rope_theta,
            rope_factor=cfg.rope_factor,
            dtype=k_full.dtype,
        )
        cos_k = cos_full[None, None, :, :]
        sin_k = sin_full[None, None, :, :]
        cos_q = cos_full[None, None, -q_len:, :]
        sin_q = sin_full[None, None, -q_len:, :]

        q = (q * cos_q) + (rotate_half(q) * sin_q)
        k_full = (k_full * cos_k) + (rotate_half(k_full) * sin_k)

        k_full = repeat_kv(k_full, cfg.num_key_value_groups)
        v_full = repeat_kv(v_full, cfg.num_key_value_groups)

        scaling = cfg.query_pre_attn_scalar ** -0.5 if cfg.query_pre_attn_scalar is not None else 1.0 / math.sqrt(head_dim)
        attn_scores = jnp.einsum("bhqd,bhkd->bhqk", q, k_full) * scaling
        if cfg.attn_logit_softcapping is not None:
            softcap = cfg.attn_logit_softcapping
            attn_scores = jnp.tanh(attn_scores / softcap) * softcap

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_probs = nn.softmax(attn_scores, axis=-1)
        attn_probs = nn.Dropout(rate=cfg.attention_dropout)(attn_probs, deterministic=deterministic)

        attn_output = jnp.einsum("bhqk,bhkd->bhqd", attn_probs, v_full)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bsz, q_len, num_heads * head_dim)
        attn_output = o_proj(attn_output)

        present_kv = KVCache(k_comp, v_comp) if use_cache else None
        return attn_output, present_kv


class CompressedGemma2Layer(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        past_key_value: Optional[KVCache],
        use_cache: bool,
        deterministic: bool,
    ) -> Tuple[jnp.ndarray, Optional[KVCache]]:
        cfg = self.config

        norm1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, name="input_layernorm")
        norm2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, name="post_attention_layernorm")
        norm3 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, name="pre_feedforward_layernorm")
        norm4 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, name="post_feedforward_layernorm")

        attn = CompressedKVGemma2Attention(cfg, name="self_attn")
        mlp = MLP(cfg, name="mlp")

        residual = hidden_states
        hidden_states = norm1(hidden_states)
        attn_out, present_kv = attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            deterministic=deterministic,
        )
        hidden_states = residual + norm2(attn_out)

        residual = hidden_states
        ff_out = mlp(norm3(hidden_states), deterministic=deterministic)
        hidden_states = residual + norm4(ff_out)

        return hidden_states, present_kv


class CompressedBgGPTForCausalLM(nn.Module):
    config: ModelConfig

    def setup(self):
        cfg = self.config
        self.embed_tokens = nn.Embed(cfg.vocab_size, cfg.hidden_size, name="embed_tokens")
        layers = []
        for i in range(cfg.num_hidden_layers):
            sw = None
            if cfg.layer_sliding_windows is not None and i < len(cfg.layer_sliding_windows):
                sw = cfg.layer_sliding_windows[i]
            else:
                sw = cfg.sliding_window
            layer_cfg = replace(cfg, sliding_window=sw)
            layers.append(CompressedGemma2Layer(layer_cfg, name=f"layers_{i}"))
        self.layers = layers
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps, name="final_norm")
        self.lm_head = nn.Dense(cfg.vocab_size, use_bias=False, name="lm_head")

    def _causal_mask(self, seq_len: int, dtype) -> jnp.ndarray:
        mask = jnp.triu(jnp.full((seq_len, seq_len), -jnp.inf, dtype=dtype), k=1)
        return mask[None, None, :, :]

    def __call__(
        self,
        input_ids: jnp.ndarray,
        past_key_values: Optional[List[KVCache]] = None,
        use_cache: bool = True,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, Optional[List[KVCache]]]:
        cfg = self.config
        bsz, seq_len = input_ids.shape

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
            past_len = 0
        else:
            first = past_key_values[0]
            past_len = 0 if first is None else first.k_comp.shape[2]

        position_ids = jnp.arange(past_len, past_len + seq_len, dtype=jnp.int32)
        position_ids = jnp.broadcast_to(position_ids[None, :], (bsz, seq_len))

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * jnp.sqrt(jnp.array(cfg.hidden_size, dtype=hidden_states.dtype))

        if past_len == 0 and seq_len > 1:
            attention_mask = self._causal_mask(seq_len, hidden_states.dtype)
        else:
            attention_mask = None

        new_past = [] if use_cache else None
        for layer, layer_past in zip(self.layers, past_key_values):
            hidden_states, present_kv = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
                deterministic=deterministic,
            )
            if use_cache:
                new_past.append(present_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if cfg.final_logit_softcapping is not None:
            softcap = cfg.final_logit_softcapping
            logits = jnp.tanh(logits / softcap) * softcap

        return logits, new_past


# -------------------------
# Loader and simple helpers
# -------------------------


def load_bggpt_compressed(
    config: Optional[ModelConfig] = None,
) -> CompressedBgGPTForCausalLM:
    cfg = config or ModelConfig()
    return CompressedBgGPTForCausalLM(cfg)


def generate(
    model: CompressedBgGPTForCausalLM,
    params: dict,
    input_ids: jnp.ndarray,
    max_new_tokens: int = 8,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    rng: jax.Array = jax.random.PRNGKey(0),
) -> jnp.ndarray:
    generated = input_ids
    past_kv = None
    rng = rng
    for _ in range(max_new_tokens):
        logits, past_kv = model.apply(
            params,
            generated if past_kv is None else generated[:, -1:],
            past_key_values=past_kv,
            use_cache=True,
            deterministic=True,
        )
        next_logits = logits[:, -1, :]
        if temperature != 1.0:
            next_logits = next_logits / temperature
        if top_k is not None and top_k > 0:
            top_values, top_indices = jax.lax.top_k(next_logits, top_k)
            masked = jnp.full_like(next_logits, -jnp.inf)
            next_logits = masked.at[jnp.arange(next_logits.shape[0])[:, None], top_indices].set(top_values)
        rng, sub = jax.random.split(rng)
        probs = jax.nn.softmax(next_logits, axis=-1)
        next_token = jax.random.categorical(sub, jnp.log(probs), axis=-1)
        generated = jnp.concatenate([generated, next_token[:, None]], axis=1)
    return generated

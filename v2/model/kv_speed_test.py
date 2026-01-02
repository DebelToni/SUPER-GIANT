from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm


# ------------------------------- Config ------------------------------------ #

@dataclass(frozen=True)
class ModelConfig:
    # Model sizes
    vocab_size: int = 32000
    context_length: int = 2048
    embedding_size: int = 1280
    num_heads: int = 20
    num_kv_heads: int = 10
    num_layers: int = 28
    feed_forward_size: int = 7680
    rope_dim: int = 64
    dropout_rate: float = 0.0
    use_remat: bool = False
    # Dtypes
    param_dtype: str = "float32"
    compute_dtype: str = "bfloat16"

MODEL_CFG = ModelConfig()


def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)


PARAM_DTYPE = _to_dtype(MODEL_CFG.param_dtype)
COMPUTE_DTYPE = _to_dtype(MODEL_CFG.compute_dtype)

from jax import config as jax_config

jax_config.update("jax_default_matmul_precision", MODEL_CFG.compute_dtype)

IS_GPU = any(dev.platform == "gpu" for dev in jax.local_devices())


# ----------------------------- Rope helpers -------------------------------- #

def _rotate_every_two(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_partial_rope(x, sin, cos, rot_dim):
    """Apply RoPE to the first rot_dim scalars of x (..., H, D)."""
    x_rot, x_pass = jnp.split(x, [rot_dim], axis=-1)
    x_rot = (x_rot * cos) + (_rotate_every_two(x_rot) * sin)
    return jnp.concatenate([x_rot, x_pass], axis=-1)


def _build_rope_cache(seq_len: int, rotary_dim: int, dtype: jnp.dtype):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, rotary_dim, 2) / rotary_dim))
    positions = jnp.arange(seq_len)
    angles = jnp.einsum("i,j->ij", positions, inv_freq)
    emb = jnp.concatenate([angles, angles], axis=-1)
    sin = jnp.sin(emb)[None, :, None, :].astype(dtype)
    cos = jnp.cos(emb)[None, :, None, :].astype(dtype)
    return sin, cos


# ---------------------------- Model blocks --------------------------------- #

class NativeJaxSelfAttention(nn.Module):
    """Multi-head self-attention using jax.nn.dot_product_attention."""

    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0
    num_kv: int = 1
    dtype: jnp.dtype = COMPUTE_DTYPE
    rotary_dim: int = MODEL_CFG.rope_dim

    def setup(self):
        if self.qkv_features % self.num_heads != 0:
            raise ValueError("qkv_features must be divisible by num_heads")
        self.head_dim = self.qkv_features // self.num_heads
        if self.num_heads % self.num_kv != 0:
            raise ValueError("num_heads must be divisible by num_kv")
        if self.rotary_dim > self.head_dim:
            raise ValueError("rotary_dim must be <= head_dim")
        if self.rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be even")

        total_out = self.qkv_features + 2 * self.num_kv * self.head_dim
        self.qkv_proj = nn.Dense(
            total_out,
            use_bias=False,
            name="qkv_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.o_proj = nn.Dense(
            self.qkv_features,
            use_bias=False,
            name="o_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self._rope_sin, self._rope_cos = _build_rope_cache(
            MODEL_CFG.context_length, self.rotary_dim, self.dtype
        )

    @nn.compact
    def __call__(
        self,
        x,
        *,
        deterministic: bool,
        use_kv_cache: bool = False,
        cur_index: Optional[int] = None,
    ):
        b, l, _ = x.shape
        impl = "cudnn" if (IS_GPU and l >= 128 and l % 2 == 0) else "xla"

        head_dim = self.head_dim
        q_size = self.num_heads * head_dim
        kv_size = self.num_kv * head_dim

        qkv = self.qkv_proj(x)
        q_chunk, k_chunk, v_chunk = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)
        q = q_chunk.reshape(b, l, self.num_heads, head_dim)
        k = k_chunk.reshape(b, l, self.num_kv, head_dim)
        v = v_chunk.reshape(b, l, self.num_kv, head_dim)

        group = max(1, self.num_heads // self.num_kv)
        kv_indices = None
        if self.num_kv != self.num_heads:
            kv_indices = jnp.arange(self.num_heads) // group

        if use_kv_cache:
            sin = jax.lax.dynamic_slice(
                self._rope_sin, (0, cur_index, 0, 0), (1, 1, 1, self.rotary_dim)
            )
            cos = jax.lax.dynamic_slice(
                self._rope_cos, (0, cur_index, 0, 0), (1, 1, 1, self.rotary_dim)
            )
        else:
            sin = self._rope_sin[:, :l, :, :]
            cos = self._rope_cos[:, :l, :, :]

        q = apply_partial_rope(q, sin, cos, self.rotary_dim)
        k = apply_partial_rope(k, sin, cos, self.rotary_dim)

        if use_kv_cache:
            if cur_index is None:
                raise ValueError("cur_index is required when use_kv_cache=True")
            cached_k = self.variable(
                "cache",
                "k",
                jnp.zeros,
                (b, self.num_kv, MODEL_CFG.context_length, head_dim),
                self.dtype,
            )
            cached_v = self.variable(
                "cache",
                "v",
                jnp.zeros,
                (b, self.num_kv, MODEL_CFG.context_length, head_dim),
                self.dtype,
            )

            k_to_cache = jnp.swapaxes(k, 1, 2)
            v_to_cache = jnp.swapaxes(v, 1, 2)
            if l == 1:
                cached_k.value = cached_k.value.at[:, :, cur_index, :].set(k_to_cache[:, :, 0, :])
                cached_v.value = cached_v.value.at[:, :, cur_index, :].set(v_to_cache[:, :, 0, :])
            else:
                cached_k.value = cached_k.value.at[:, :, cur_index : cur_index + l, :].set(k_to_cache)
                cached_v.value = cached_v.value.at[:, :, cur_index : cur_index + l, :].set(v_to_cache)

            k_full = jnp.swapaxes(cached_k.value, 1, 2)
            v_full = jnp.swapaxes(cached_v.value, 1, 2)
            if kv_indices is not None:
                k_full = jnp.take(k_full, kv_indices, axis=2)
                v_full = jnp.take(v_full, kv_indices, axis=2)

            key_len = k_full.shape[1]
            cur_max = cur_index + (l - 1)
            valid = jnp.arange(key_len) <= cur_max
            attn_bias = jnp.where(valid, 0.0, -1e10).astype(self.dtype)
            attn_bias = attn_bias[None, None, None, :]
            y = jax.nn.dot_product_attention(
                q, k_full, v_full, bias=attn_bias, is_causal=False, implementation=impl
            )
            y = y.reshape(b, l, self.qkv_features)
        else:
            k_full = k if kv_indices is None else jnp.take(k, kv_indices, axis=2)
            v_full = v if kv_indices is None else jnp.take(v, kv_indices, axis=2)
            y = jax.nn.dot_product_attention(q, k_full, v_full, is_causal=True, implementation=impl)
            y = y.reshape(b, l, self.qkv_features)

        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class TinyTransformerBlock(nn.Module):
    """Decoder-style transformer block with optional KV cache."""

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    dtype: jnp.dtype = COMPUTE_DTYPE

    @nn.compact
    def __call__(
        self,
        x,
        *,
        deterministic: bool,
        use_kv_cache: bool = False,
        cur_index: Optional[int] = None,
    ):
        def _block(module: "TinyTransformerBlock", h: jnp.ndarray) -> jnp.ndarray:
            residual = h
            h_norm = RMSNorm(name="rms1", dtype=self.dtype, epsilon=1e-5)(h)
            h_attn = NativeJaxSelfAttention(
                num_heads=module.n_heads,
                num_kv=MODEL_CFG.num_kv_heads,
                qkv_features=module.d_model,
                dropout_rate=module.dropout_rate,
                dtype=module.dtype,
            )(h_norm, deterministic=deterministic, use_kv_cache=use_kv_cache, cur_index=cur_index)
            h = residual + h_attn

            residual = h
            h_norm = RMSNorm(name="rms2", dtype=self.dtype, epsilon=1e-5)(h)

            gate_dim = module.d_ff
            proj_dim = gate_dim * 2
            h_proj = nn.Dense(
                proj_dim,
                name="fc1",
                dtype=module.dtype,
                param_dtype=PARAM_DTYPE,
                use_bias=False,
            )(h_norm)
            u, v = jnp.split(h_proj, 2, axis=-1)
            h_gate = nn.silu(u)
            h_ffn = h_gate * v
            h_ffn = nn.Dense(
                module.d_model,
                name="fc2",
                dtype=module.dtype,
                param_dtype=PARAM_DTYPE,
                use_bias=False,
            )(h_ffn)
            h_ffn = nn.Dropout(rate=module.dropout_rate)(h_ffn, deterministic=deterministic)
            return residual + h_ffn

        block_fn = nn.remat(_block) if MODEL_CFG.use_remat else _block
        return block_fn(self, x)


class GiantGPT(nn.Module):
    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        tokens,
        *,
        deterministic: bool = False,
        use_kv_cache: bool = False,
        cur_index: Optional[int] = None,
    ):
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAM_DTYPE,
        )
        x = embed(tokens)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        for _ in range(self.n_layers):
            x = TinyTransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                dtype=COMPUTE_DTYPE,
            )(x, deterministic=deterministic, use_kv_cache=use_kv_cache, cur_index=cur_index)

        logits = jnp.einsum("bld,vd->blv", x.astype(jnp.float32), embed.embedding)
        return logits


# ------------------------- KV cache inference ------------------------------ #

Array = jnp.ndarray
PyTree = Dict[str, Any]


def init_inference_state(
    model: GiantGPT,
    key_params: jax.Array,
    key_dropout: jax.Array,
    batch_size: int,
    *,
    pad_token_id: int = 0,
    use_kv_cache: bool = True,
) -> Tuple[PyTree, PyTree]:
    """Initialize params + nonparam state (including cache)."""
    dummy = jnp.full((batch_size, 1), pad_token_id, dtype=jnp.int32)
    variables = model.init(
        {"params": key_params, "dropout": key_dropout},
        dummy,
        deterministic=True,
        use_kv_cache=use_kv_cache,
        cur_index=0,
    )
    params = variables["params"]
    nonparam = {k: v for k, v in variables.items() if k != "params"}
    if use_kv_cache and "cache" not in nonparam:
        raise ValueError("Model did not create a 'cache' collection during init.")
    return params, nonparam


def _apply_with_cache(
    model: GiantGPT,
    params: PyTree,
    nonparam: PyTree,
    tokens_1: Array,
    cur_idx: Array,
):
    variables = {"params": params, **nonparam}
    logits, new_vars = model.apply(
        variables,
        tokens_1,
        deterministic=True,
        use_kv_cache=True,
        cur_index=cur_idx,
        mutable=["cache"],
    )
    nonparam_out = {**nonparam, "cache": new_vars["cache"]}
    return logits, nonparam_out


def make_prefill_and_decode_fns(model: GiantGPT):
    @jax.jit
    def prefill(
        params: PyTree,
        nonparam: PyTree,
        prompt_tokens: Array,
    ):
        b, lp = prompt_tokens.shape
        t0 = jnp.array(0, jnp.int32)

        def prefill_step(carry, tok_t_2d):
            nonparam, t = carry
            _logits, nonparam = _apply_with_cache(model, params, nonparam, tok_t_2d, t)
            return (nonparam, t + 1), jnp.int32(0)

        if lp > 0:
            xs = jnp.expand_dims(jnp.swapaxes(prompt_tokens, 0, 1), -1)
            (nonparam, t), _ = jax.lax.scan(prefill_step, init=(nonparam, t0), xs=xs)
            last_tok_2d = prompt_tokens[:, -1:]
        else:
            nonparam, t = nonparam, t0
            last_tok_2d = jnp.zeros((b, 1), dtype=jnp.int32)

        last_pos = jnp.maximum(t - 1, jnp.array(0, jnp.int32))
        return nonparam, last_pos, last_tok_2d

    @partial(jax.jit, static_argnames=("steps",), donate_argnums=(1,))
    def decode(
        params: PyTree,
        nonparam: PyTree,
        last_tok_2d: Array,
        t: Array,
        *,
        steps: int,
    ):
        b = last_tok_2d.shape[0]
        out = jnp.zeros((b, steps), dtype=jnp.int32)

        def body(carry, i):
            nonparam, t, tok_prev_2d, out = carry
            logits, nonparam = _apply_with_cache(model, params, nonparam, tok_prev_2d, t)
            step_logits = logits[:, -1, :]
            next_tok = jnp.argmax(step_logits, axis=-1)
            out = jax.lax.dynamic_update_slice(out, next_tok[:, None], (0, i))
            next_tok_2d = next_tok[:, None]
            return (nonparam, t + 1, next_tok_2d, out), None

        (nonparam, t, _tok2d, out), _ = jax.lax.scan(
            body,
            init=(nonparam, t, last_tok_2d, out),
            xs=jnp.arange(steps, dtype=jnp.int32),
        )
        return out, nonparam

    return prefill, decode


def block_until_ready(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


# ------------------------------- Benchmark --------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Speed test a JAX transformer with KV cache.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--prompt_len", type=int, default=128, help="Prompt length for prefill.")
    parser.add_argument("--steps", type=int, default=128, help="Decode steps.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs.")
    parser.add_argument("--runs", type=int, default=1, help="Number of timed runs.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.prompt_len < 0:
        raise ValueError("prompt_len must be >= 0")
    if args.steps <= 0:
        raise ValueError("steps must be > 0")
    if args.prompt_len + args.steps > MODEL_CFG.context_length:
        raise ValueError(
            "prompt_len + steps must be <= context_length ({}).".format(MODEL_CFG.context_length)
        )
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    if args.runs <= 0:
        raise ValueError("runs must be > 0")

    model = GiantGPT(
        vocab_size=MODEL_CFG.vocab_size,
        context_length=MODEL_CFG.context_length,
        d_model=MODEL_CFG.embedding_size,
        n_heads=MODEL_CFG.num_heads,
        d_ff=MODEL_CFG.feed_forward_size,
        n_layers=MODEL_CFG.num_layers,
        dropout_rate=MODEL_CFG.dropout_rate,
    )

    rng = jax.random.PRNGKey(args.seed)
    key_params, key_dropout, key_data = jax.random.split(rng, 3)

    params, nonparam = init_inference_state(
        model,
        key_params,
        key_dropout,
        batch_size=args.batch_size,
        pad_token_id=0,
        use_kv_cache=True,
    )

    prompt = jax.random.randint(
        key_data,
        (args.batch_size, args.prompt_len),
        minval=0,
        maxval=MODEL_CFG.vocab_size,
        dtype=jnp.int32,
    )

    prefill_fn, decode_fn = make_prefill_and_decode_fns(model)

    compiled_prefill = prefill_fn.lower(params, nonparam, prompt).compile()
    compiled_decode = decode_fn.lower(
        params,
        nonparam,
        jnp.zeros((args.batch_size, 1), dtype=jnp.int32),
        jnp.array(0, jnp.int32),
        steps=args.steps,
    ).compile()

    # Warmup
    for _ in range(args.warmup):
        nonparam_filled, t_cur, last_tok = compiled_prefill(params, nonparam, prompt)
        out, nonparam_filled = compiled_decode(params, nonparam_filled, last_tok, t_cur)
        block_until_ready(out)

    # Timed runs
    prefill_times = []
    decode_times = []
    for _ in range(args.runs):
        prefill_start = time.perf_counter()
        nonparam_filled, t_cur, last_tok = compiled_prefill(params, nonparam, prompt)
        block_until_ready(nonparam_filled)
        prefill_times.append(time.perf_counter() - prefill_start)

        decode_start = time.perf_counter()
        out, _ = compiled_decode(params, nonparam_filled, last_tok, t_cur)
        out.block_until_ready()
        decode_times.append(time.perf_counter() - decode_start)

    prefill_time = sum(prefill_times) / len(prefill_times)
    decode_time = sum(decode_times) / len(decode_times)
    tokens_per_s = (args.steps * args.batch_size) / decode_time if decode_time > 0 else float("inf")

    print("\n[config]")
    print(f"vocab_size: {MODEL_CFG.vocab_size}")
    print(f"context_length: {MODEL_CFG.context_length}")
    print(f"d_model: {MODEL_CFG.embedding_size}")
    print(f"num_heads: {MODEL_CFG.num_heads}")
    print(f"num_kv_heads: {MODEL_CFG.num_kv_heads}")
    print(f"num_layers: {MODEL_CFG.num_layers}")
    print(f"feed_forward_size: {MODEL_CFG.feed_forward_size}")
    print(f"rope_dim: {MODEL_CFG.rope_dim}")
    print(f"param_dtype: {MODEL_CFG.param_dtype}")
    print(f"compute_dtype: {MODEL_CFG.compute_dtype}")
    print("\n[bench]")
    print(f"batch_size: {args.batch_size}")
    print(f"prompt_len: {args.prompt_len}")
    print(f"decode_steps: {args.steps}")
    print(f"prefill_time_s: {prefill_time:.6f}")
    print(f"decode_time_s: {decode_time:.6f}")
    print(f"tokens_per_second_decode: {tokens_per_s:.6f}")


if __name__ == "__main__":
    main()

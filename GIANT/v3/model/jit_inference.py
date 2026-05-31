# - Initializes params/nonparam state (incl. 'cache')
# - Prefills a prompt into KV-cache
# - Decodes with a single JIT-compiled lax.scan loop
import os
from functools import partial
from typing import Any, Dict, Optional, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import jax
import jax.numpy as jnp

# IMPORTANT: this imports your model from /mnt/data/GiantGPT.py
from GIANT.v3.model.GiantGPT import GiantGPT

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
    """
    Initialize model variables (params + nonparam collections).
    We pass a [B, 1] dummy token so the cache structure is created when use_kv_cache=True.
    """
    if use_kv_cache and not bool(getattr(model, "causal", True)):
        raise ValueError("init_inference_state(use_kv_cache=True) requires a causal decoder model.")
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
        raise ValueError(
            "Model did not create a 'cache' collection during init. "
            "Check that GiantGPT uses a Flax variable collection named 'cache' when use_kv_cache=True."
        )
    return params, nonparam


def _apply_with_cache(
    model: GiantGPT,
    params: PyTree,
    nonparam: PyTree,
    tokens_1: Array,          # [B, 1]
    cur_idx: Array,           # scalar int32
):
    """
    Single step forward with KV cache enabled (deterministic=True).
    Returns logits and updated nonparam with refreshed 'cache'.
    Note: intended to be used **inside** a jitted function.
    """
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
    """
    Returns two JIT-compiled functions:
      * prefill(params, nonparam, prompt_tokens) -> (nonparam, last_pos, last_tok_2d)
      * decode(params, nonparam, last_tok_2d, t, steps, do_sample, top_k, temperature, rng_key)
          -> (tokens_new [B, steps], nonparam)
    We capture `model` in the closure (static to the JIT).
    """

    @jax.jit
    def prefill(
        params: PyTree,
        nonparam: PyTree,
        prompt_tokens: Array,           # [B, Lp]
    ):
        B, Lp = prompt_tokens.shape
        t0 = jnp.array(0, jnp.int32)

        def prefill_step(carry, tok_t_2d):
            nonparam, t = carry
            logits, nonparam = _apply_with_cache(model, params, nonparam, tok_t_2d, t)
            return (nonparam, t + 1), logits

        if Lp > 0:
            xs = jnp.expand_dims(jnp.swapaxes(prompt_tokens, 0, 1), -1)  # [Lp, B, 1]
            (nonparam, t), _ = jax.lax.scan(prefill_step, init=(nonparam, t0), xs=xs)
            last_tok_2d = prompt_tokens[:, -1:]
        else:
            nonparam, t = nonparam, t0
            last_tok_2d = jnp.zeros((B, 1), dtype=jnp.int32)

        # Return index of the last processed token so decoding starts at that position.
        last_pos = jnp.maximum(t - 1, jnp.array(0, jnp.int32))
        return nonparam, last_pos, last_tok_2d

    def _top_k_logits(logits: Array, k: int) -> Array:
        """Mask everything below the kth largest logit."""
        if k <= 0:
            return logits
        topk_vals, _ = jax.lax.top_k(logits, k)      # [..., k]
        kth = topk_vals[..., -1, None]               # [..., 1]
        return jnp.where(logits < kth, -jnp.inf, logits)

    @partial(
        jax.jit,
        static_argnames=("steps", "do_sample", "top_k"),
        donate_argnums=(1,),  # donate nonparam
    )
    def decode(
        params: PyTree,
        nonparam: PyTree,
        last_tok_2d: Array,         # [B, 1] (the last prompt token or previous generated)
        t: Array,                   # scalar int32, index of last_tok_2d in the sequence
        *,
        steps: int,                 # number of new tokens to generate
        do_sample: bool = False,
        top_k: int = 0,
        temperature: float = 1.0,
        rng_key: Optional[jax.Array] = None,
    ):
        B = last_tok_2d.shape[0]
        # Preallocate output tokens [B, steps]
        out = jnp.zeros((B, steps), dtype=jnp.int32)

        def body(carry, i):
            nonparam, t, tok_prev_2d, rng, out = carry
            logits, nonparam = _apply_with_cache(model, params, nonparam, tok_prev_2d, t)
            step_logits = logits[:, -1, :]

            if do_sample:
                assert rng is not None, "rng_key must be provided when do_sample=True"
                scaled = step_logits / jnp.maximum(temperature, 1e-6)
                scaled = _top_k_logits(scaled, top_k) if top_k > 0 else scaled
                rng, sub = jax.random.split(rng)
                next_tok = jax.random.categorical(sub, scaled, axis=-1)
            else:
                next_tok = jnp.argmax(step_logits, axis=-1)

            # write to output
            out = jax.lax.dynamic_update_slice(out, next_tok[:, None], (0, i))
            next_tok_2d = next_tok[:, None]
            return (nonparam, t + 1, next_tok_2d, rng, out), None

        (nonparam, t, _tok2d, _rng, out), _ = jax.lax.scan(
            body,
            init=(nonparam, t, last_tok_2d, rng_key if do_sample else jnp.zeros((2,), jnp.uint32), out),
            xs=jnp.arange(steps, dtype=jnp.int32),
        )
        return out, nonparam

    return prefill, decode

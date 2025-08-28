# jit_infer.py
# Fully JIT-ed inference wrappers for GiantGPT (built from your /mnt/data/GiantGPT.py and /mnt/data/Transformer_block.py)
# - Initializes params/nonparam state (incl. 'cache')
# - Prefills a prompt into KV-cache
# - Decodes with a single JIT-compiled lax.scan loop
from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

# IMPORTANT: this imports your model from /mnt/data/GiantGPT.py
from GiantGPT import GiantGPT


Array = jnp.ndarray
PyTree = Dict[str, Any]


def init_inference_state(
    model: GiantGPT,
    key_params: jax.random.KeyArray,
    key_dropout: jax.random.KeyArray,
    batch_size: int,
    *,
    pad_token_id: int = 0,
    use_kv_cache: bool = True,
) -> Tuple[PyTree, PyTree]:
    """
    Initialize model variables (params + nonparam collections).
    We pass a [B, 1] dummy token so the cache structure is created when use_kv_cache=True.
    """
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
    # Ensure there's a 'cache' collection present when we plan to use_kv_cache
    if use_kv_cache and "cache" not in nonparam:
        raise ValueError(
            "Model did not create a 'cache' collection during init. "
            "Check that GiantGPT uses Flax variable collection named 'cache' when use_kv_cache=True."
        )
    return params, nonparam


def _apply_with_cache(
    model: GiantGPT,
    params: PyTree,
    nonparam: PyTree,
    tokens_1: Array,          # [B, 1]
    cur_idx: Array,           # scalar int32
) -> Tuple[Array, PyTree]:
    """
    Single step forward with KV cache enabled (deterministic=True).
    Returns logits and updated nonparam with refreshed 'cache'.
    Note: this function is intended to be used **inside** a jitted function.
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


def make_generate_fn(model: GiantGPT):
    """
    Returns a single JIT-compiled function that:
      * Prefills the prompt into the KV cache using scan
      * Generates `max_new_tokens` tokens (greedy or sampled) using scan
    We capture `model` in the closure (static to the JIT).
    """

    def _top_k_logits(logits: Array, k: int) -> Array:
        if k <= 0:
            return logits
        # Keep top-k, set others to -inf
        kth = jnp.sort(logits, axis=-1)[..., -k:jnp.shape(logits)[-1]-k+1:-1]
        # The threshold is the smallest of the top-k (i.e., index -k)
        # Simpler: gather topk and build a mask
        topk_vals = jax.lax.top_k(logits, k)[0][..., -1:]  # the kth largest
        mask = logits < topk_vals
        return jnp.where(mask, -jnp.inf, logits)

    @partial(
        jax.jit,
        static_argnames=("max_new_tokens", "do_sample", "top_k"),
        donate_argnums=(1,),  # donate `nonparam` (arg index 1) to reduce copies
    )
    def generate(
        params: PyTree,
        nonparam: PyTree,
        prompt_tokens: Array,              # [B, Lp] int32
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        top_k: int = 0,
        temperature: float = 1.0,
        rng_key: Optional[jax.random.KeyArray] = None,
    ) -> Tuple[Array, PyTree]:
        """
        Returns:
          tokens_new: [B, max_new_tokens] int32
          nonparam:   updated nonparam (with final cache)
        """
        B = prompt_tokens.shape[0]
        Lp = prompt_tokens.shape[1]

        # -------------------------
        # Prefill the KV cache
        # -------------------------
        def prefill_step(carry, tok_t_2d):
            nonparam, t = carry
            logits, nonparam = _apply_with_cache(model, params, nonparam, tok_t_2d, t)
            return (nonparam, t + 1), logits

        if Lp > 0:
            # xs for scan: [Lp, B, 1]
            xs = jnp.expand_dims(jnp.moveaxis(prompt_tokens, 1, 0), -1)
            (nonparam, t), prefill_logits = jax.lax.scan(
                prefill_step,
                init=(nonparam, jnp.array(0, jnp.int32)),
                xs=xs,
            )
            # last prefill token becomes the context for decoding
            last_logits = prefill_logits[-1]  # [B, 1, V]
            token_prev = jnp.argmax(last_logits[:, -1, :], axis=-1)  # [B]
        else:
            # If there's no prompt, start from zeros token (or user should feed BOS via prompt)
            t = jnp.array(0, jnp.int32)
            token_prev = jnp.zeros((B,), dtype=jnp.int32)

        token_prev_2d = token_prev[:, None]  # [B, 1]

        # -------------------------
        # Decode loop (generate)
        # -------------------------
        def decode_step(carry, _):
            nonparam, t, tok_prev_2d, rng = carry
            logits, nonparam = _apply_with_cache(model, params, nonparam, tok_prev_2d, t)
            step_logits = logits[:, -1, :]  # [B, V]

            if do_sample:
                assert rng is not None, "rng_key must be provided when do_sample=True"
                # Temperature + optional top-k
                scaled = step_logits / jnp.maximum(temperature, 1e-6)
                scaled = _top_k_logits(scaled, top_k) if top_k > 0 else scaled
                next_tok = jax.random.categorical(rng, scaled, axis=-1)
                rng, _ = jax.random.split(rng)
            else:
                next_tok = jnp.argmax(step_logits, axis=-1)

            next_tok_2d = next_tok[:, None]  # [B, 1]
            return (nonparam, t + 1, next_tok_2d, rng), next_tok

        init_rng = rng_key if do_sample else jnp.zeros((2,), dtype=jnp.uint32)
        (nonparam, _t, _tok2d, _rng), tokens_new = jax.lax.scan(
            decode_step,
            init=(nonparam, t, token_prev_2d, init_rng),
            xs=None,
            length=max_new_tokens,
        )
        # tokens_new: [T, B] -> [B, T]
        tokens_new = jnp.moveaxis(tokens_new, 0, 1)
        return tokens_new, nonparam

    return generate


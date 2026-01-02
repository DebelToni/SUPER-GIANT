#!/usr/bin/env python
from __future__ import annotations

import jax
import jax.numpy as jnp

from Transformer_block import TinyTransformerBlock, MODEL_CFG, COMPUTE_DTYPE


def main() -> None:
    batch = 2
    seq = 4
    d_model = MODEL_CFG.embedding_size

    block = TinyTransformerBlock(
        d_model=d_model,
        n_heads=MODEL_CFG.num_heads,
        d_ff=MODEL_CFG.feed_forward_size,
        dropout_rate=0.0,
        dtype=COMPUTE_DTYPE,
    )

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (batch, seq, d_model), dtype=COMPUTE_DTYPE)

    variables = block.init(
        {"params": key},
        x,
        deterministic=True,
        use_kv_cache=True,
        cur_index=0,
    )

    y = block.apply(
        variables,
        x,
        deterministic=True,
        use_kv_cache=False,
    )
    print("no_cache_out", y.shape, y.dtype)

    x1 = x[:, :1, :]
    y1, new_vars = block.apply(
        variables,
        x1,
        deterministic=True,
        use_kv_cache=True,
        cur_index=0,
        mutable=["cache"],
    )
    print("cache_out", y1.shape, y1.dtype)

    key2 = jax.random.PRNGKey(1)
    x2 = jax.random.normal(key2, (batch, 1, d_model), dtype=COMPUTE_DTYPE)
    y2, _ = block.apply(
        {**variables, "cache": new_vars["cache"]},
        x2,
        deterministic=True,
        use_kv_cache=True,
        cur_index=1,
        mutable=["cache"],
    )
    print("cache_step2", y2.shape, y2.dtype)

    print("num_heads", MODEL_CFG.num_heads, "num_kv", MODEL_CFG.num_kv_heads)


if __name__ == "__main__":
    main()

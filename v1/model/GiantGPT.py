from __future__ import annotations
from typing import Dict, Tuple

import jax.numpy as jnp
import flax.linen as nn

import Config as cfg
from Transformer_block import (
    TransformerBlock,
    sinusoid_position_encoding,            
)

class GiantGPT(nn.Module):

    vocab_size: int      = cfg.vocab_size
    context_length: int  = cfg.context_length
    d_model: int         = cfg.embedding_size
    n_heads: int         = cfg.num_heads
    d_ff: int            = cfg.feed_forward_size
    n_layers: int        = cfg.num_layers
    dropout_rate: float  = 0.0               

    def setup(self):

        self.token_emb = nn.Embed(
            self.vocab_size,
            self.d_model,
            dtype=cfg.compute_dtype,
            param_dtype=cfg.param_dtype,
        )
        self.pos_emb = self.param(
            "pos_emb",
            lambda *_: jnp.asarray(
                sinusoid_position_encoding(
                    self.context_length,
                    self.d_model,
                    cfg.compute_dtype
                )[None, ...]                 
            ),
        )

        self.blocks = [
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout_rate,
                name=f"blk{i}",
            )
            for i in range(self.n_layers)
        ]

        self.final_ln = nn.LayerNorm(dtype=cfg.compute_dtype)

        self.lm_head = nn.Dense(
            self.vocab_size, use_bias=False, dtype=jnp.float32, name="lm_head",
            param_dtype=cfg.param_dtype,  
        )

    def init_cache(
        self, *, batch_size: int, max_length: int, dtype=jnp.float16
    ) -> Dict[str, Dict]:
        """
        Create a cache with:
            cache["pos"]   – current position (int32 scalar)
            cache["layer<i>"]["k" | "v" | "idx"]
        """
        cache: Dict[str, Dict] = {"pos": jnp.array(0, jnp.int32)}
        Dh = self.d_model // self.n_heads
        for i in range(self.n_layers):
            k = jnp.zeros((batch_size, self.n_heads, max_length, Dh), dtype)
            v = jnp.zeros_like(k)
            cache[f"layer{i}"] = {
                "k": k,
                "v": v,
                "idx": jnp.array(0, jnp.int32),
            }
        return cache

    def __call__(
        self,
        x: jnp.ndarray,                             
        *,
        cache: Dict | None = None,
        deterministic: bool = True,
    ) -> Tuple[jnp.ndarray, Dict | None]:
        B, T = x.shape

        h = self.token_emb(x).astype(cfg.compute_dtype)        

        if cache is None:                                     
            h += self.pos_emb[:, :T, :]
            pos_cursor = None
        else:                                                 
            assert (
                T == 1
            ), "When a cache is supplied, forward pass expects exactly one token."
            pos_cursor = cache["pos"]
            h += self.pos_emb[:, pos_cursor : pos_cursor + 1, :]
            cache["pos"] = pos_cursor + 1

        for i, block in enumerate(self.blocks):
            layer_cache = None if cache is None else cache[f"layer{i}"]
            h, new_layer_cache = block(
                h, cache=layer_cache, deterministic=deterministic
            )
            if cache is not None:
                cache[f"layer{i}"] = new_layer_cache

        h = self.final_ln(h)
        logits = self.lm_head(h.astype(jnp.float32))           
        return logits, cache

from __future__ import annotations
from typing import Dict, Tuple

import jax.numpy as jnp
import flax.linen as nn

import Config
from Transformer_block import TransformerBlock, sinusoid_position_encoding

cfg = Config                                 

class GiantGPT(nn.Module):
    """
    Minimal-but-complete GPT architecture that supports
    * fp16 / bf16 / fp32 / int8 (set in Config)
    * KV cache initialisation for autoregressive inference
    """

    vocab_size: int           = cfg.vocab_size
    context_length: int       = cfg.context_length
    d_model: int              = cfg.embedding_size
    n_heads: int              = cfg.num_heads
    d_ff: int                 = cfg.feed_forward_size
    n_layers: int             = cfg.num_layers
    dropout_rate: float       = 0.0            

    def setup(self):
        self.token_emb = nn.Embed(
            self.vocab_size, self.d_model, dtype=cfg.compute_dtype)

        self.pos_emb = self.param(
            "pos_emb",
            lambda *_: jnp.asarray(
                jnp.expand_dims(
                    sinusoid_position_encoding(
                        self.context_length,
                        self.d_model,
                        cfg.compute_dtype),
                    axis=0)                    
            )
        )

        self.blocks = [
            TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout_rate,
                name=f"blk{i}")
            for i in range(self.n_layers)
        ]

        self.final_ln = nn.LayerNorm(dtype=cfg.compute_dtype)

        self.lm_head = nn.Dense(
            self.vocab_size,
            use_bias=False,
            dtype=jnp.float32,        
            name="lm_head")

    def init_cache(self, *, batch_size: int, max_length: int,
                   dtype=jnp.float32) -> Dict[str, jnp.ndarray]:
        """Create an **empty** KV cache on the model's device."""
        Dh = self.d_model // self.n_heads
        k = jnp.zeros((batch_size, self.n_heads, max_length, Dh), dtype)
        v = jnp.zeros_like(k)
        return {"k": k, "v": v, "idx": jnp.array(0, jnp.int32)}

    def __call__(
        self,
        x: jnp.ndarray,                       
        *,
        cache: Dict[str, jnp.ndarray],
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, Dict]:
        """
        Returns
        -------
        logits : [B, 1, vocab]  –  fp32
        cache  : updated KV cache
        """
        B, T = x.shape
        assert T == 1, "GiantGPT expects one token per forward pass."

        h = self.token_emb(x).astype(cfg.compute_dtype)         
        idx = cache["idx"]
        h = h + self.pos_emb[:, idx : idx + 1, :]

        for i in range(self.n_layers):
            h, cache_layer = self.blocks[i](h,
                                            cache=cache[f"layer{i}"],
                                            deterministic=deterministic)
            cache[f"layer{i}"] = cache_layer

        h = self.final_ln(h)
        logits = self.lm_head(h.astype(jnp.float32))            
        return logits, cache

    @staticmethod
    def empty_cache(batch_size: int,
                    max_length: int,
                    dtype=jnp.float32) -> Dict[str, Dict]:
        """Convenience: build per-layer caches in one call."""
        layer_cache = {}
        for i in range(cfg.num_layers):
            Dh = cfg.embedding_size // cfg.num_heads
            k = jnp.zeros((batch_size, cfg.num_heads, max_length, Dh), dtype)
            v = jnp.zeros_like(k)
            layer_cache[f"layer{i}"] = {"k": k, "v": v, "idx": jnp.array(0, jnp.int32)}
        return layer_cache

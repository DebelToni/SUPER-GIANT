from typing import Optional
from pathlib import Path

import jax.numpy as jnp
from flax import linen as nn
from omegaconf import OmegaConf

from TiDAR.model.Transformer_block import TinyTransformerBlock

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent

cfg = OmegaConf.merge(
    OmegaConf.load(PROJECT_ROOT / "Global_Config.yml"),
    OmegaConf.load(MODEL_DIR / "Config.yml"),
)
MODEL_CFG = cfg.model


def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)


PARAM_DTYPE = _to_dtype(MODEL_CFG.param_dtype)
COMPUTE_DTYPE = _to_dtype(MODEL_CFG.compute_dtype)

class TiDAR(nn.Module):
    vocab_size:     int
    context_length: int
    d_model:        int
    n_heads:        int
    d_ff:           int
    n_layers:       int
    dropout_rate:   float = 0.1

    @nn.compact
    def __call__(
        self,
        tokens,
        *,
        deterministic: bool = False,
        attn_bias: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        use_kv_cache: bool = False,
        cur_index: Optional[int] = None,
        write_to_cache: bool = True,
        prefix_len: Optional[int] = None,
        cache_write_len: Optional[int] = None,
        kv_cache_len: Optional[int] = None,
        return_hidden: bool = False,
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
            )(
                x,
                deterministic=deterministic,
                attn_bias=attn_bias,
                position_ids=position_ids,
                use_kv_cache=use_kv_cache,
                cur_index=cur_index,
                write_to_cache=write_to_cache,
                prefix_len=prefix_len,
                cache_write_len=cache_write_len,
                kv_cache_len=kv_cache_len,
            )


        if return_hidden:
            return x

        logits = jnp.einsum("bld,vd->blv",
                            x.astype(jnp.float32),
                            embed.embedding)
        return logits

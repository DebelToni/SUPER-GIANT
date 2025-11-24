
from typing import Optional
from pathlib import Path

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm
from omegaconf import OmegaConf

from Transformer_block import TinyTransformerBlock

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


class GiantGPT(nn.Module):
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
        use_kv_cache: bool = False,
        cur_index: Optional[int] = None,
    ):
        # Token embedding (tied to LM head)
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAM_DTYPE,
        )
        x = embed(tokens)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Decoder blocks
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
                use_kv_cache=use_kv_cache,
                cur_index=cur_index,
            )

        # SmolLM/LLaMA-style final RMSNorm
        x = RMSNorm(name="final_norm", dtype=COMPUTE_DTYPE, epsilon=1e-5)(x)

        # Tied LM head: logits = x @ W_embed^T
        logits = jnp.einsum(
            "bld,vd->blv",
            x.astype(jnp.float32),
            embed.embedding,
        )
        return logits

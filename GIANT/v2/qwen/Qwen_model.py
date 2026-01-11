from __future__ import annotations

from typing import Optional
from pathlib import Path

import jax.numpy as jnp
from flax import linen as nn
from omegaconf import OmegaConf

from Qwen_block import QwenBlock, PARAM_DTYPE, COMPUTE_DTYPE

QWEN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = QWEN_DIR.parent

cfg = OmegaConf.merge(
    OmegaConf.load(PROJECT_ROOT / "Global_Config.yml"),
    OmegaConf.load(QWEN_DIR / "Config.yml"),
)
MODEL_CFG = cfg.model


class QwenGPT(nn.Module):
    vocab_size: int
    context_length: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    n_layers: int
    dropout_rate: float = 0.0
    rope_dim: int = 64
    rope_theta: float = 1e6

    @nn.compact
    def __call__(self, tokens, *, deterministic: bool = False, use_kv_cache: bool = False, cur_index: Optional[int] = None):
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
            x = QwenBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                rope_dim=self.rope_dim,
                rope_theta=self.rope_theta,
                dtype=COMPUTE_DTYPE,
            )(x, deterministic=deterministic, use_kv_cache=use_kv_cache, cur_index=cur_index)

        x = nn.RMSNorm(epsilon=float(MODEL_CFG.rms_norm_eps), dtype=COMPUTE_DTYPE, name="norm")(x)
        logits = jnp.einsum("bld,vd->blv", x.astype(jnp.float32), embed.embedding)
        return logits

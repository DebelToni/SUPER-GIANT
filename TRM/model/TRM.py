from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm
from omegaconf import OmegaConf
from pathlib import Path

from TRM_block import TinyRecurrentNet


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


class TRM(nn.Module):
    vocab_size: int
    context_length: int
    d_model: int
    tiny_layers: int
    variant: str
    num_heads: int
    d_ff: int
    mixer_hidden: int

    dropout_rate: float = 0.0
    activation: str = str(MODEL_CFG.activation)
    add_positional_embedding: bool = bool(getattr(MODEL_CFG, "add_positional_embedding", True))

    # Recursion / refinement
    L_cycles: int = int(MODEL_CFG.recursion.L_cycles)
    H_cycles: int = int(MODEL_CFG.recursion.H_cycles)
    max_supervision_steps: int = int(MODEL_CFG.recursion.max_supervision_steps)
    enable_early_stop: bool = bool(MODEL_CFG.recursion.enable_early_stop)
    halt_threshold_logit: float = float(MODEL_CFG.recursion.halt_threshold_logit)

    # Optional augmentation embedding
    aug_enabled: bool = bool(MODEL_CFG.augmentation.enabled)
    aug_num_embeddings: int = int(MODEL_CFG.augmentation.num_embeddings)
    aug_default_id: int = int(MODEL_CFG.augmentation.default_id)

    @nn.compact
    def encode(
        self,
        tokens: jnp.ndarray,  # (B, L) int32
        *,
        deterministic: bool = True,
        aug_ids: Optional[jnp.ndarray] = None,  # (B,) int32
    ) -> jnp.ndarray:
        b, l = tokens.shape
        if l != self.context_length:
            raise ValueError(f"Expected tokens.shape[1]==context_length ({self.context_length}), got {l}")

        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAM_DTYPE,
            name="in_embed",
        )
        x = embed(tokens)

        if self.add_positional_embedding:
            pos = self.param(
                "pos_embed",
                nn.initializers.normal(stddev=0.02),
                (1, self.context_length, self.d_model),
                PARAM_DTYPE,
            ).astype(COMPUTE_DTYPE)
            x = x + pos

        if self.aug_enabled:
            if aug_ids is None:
                aug_ids = jnp.full((b,), self.aug_default_id, dtype=jnp.int32)
            aug_embed = nn.Embed(
                num_embeddings=self.aug_num_embeddings,
                features=self.d_model,
                embedding_init=nn.initializers.normal(stddev=0.02),
                dtype=COMPUTE_DTYPE,
                param_dtype=PARAM_DTYPE,
                name="aug_embed",
            )
            aug = aug_embed(aug_ids).astype(COMPUTE_DTYPE)  # (B, D)
            x = x + aug[:, None, :]

        x = nn.Dropout(rate=self.dropout_rate, name="in_dropout")(x, deterministic=deterministic)
        return x

    @nn.compact
    def initial_state(self, tokens: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        b, l = tokens.shape
        if l != self.context_length:
            raise ValueError(f"Expected tokens.shape[1]==context_length ({self.context_length}), got {l}")

        init_std = float(getattr(MODEL_CFG.init, "init_std", 0.02))
        learned_y = bool(getattr(MODEL_CFG.init, "learned_y", True))
        learned_z = bool(getattr(MODEL_CFG.init, "learned_z", True))

        if learned_y:
            y0 = self.param(
                "y_init",
                nn.initializers.normal(stddev=init_std),
                (1, 1, self.d_model),
                PARAM_DTYPE,
            ).astype(COMPUTE_DTYPE)
        else:
            y0 = jnp.zeros((1, 1, self.d_model), dtype=COMPUTE_DTYPE)

        if learned_z:
            z0 = self.param(
                "z_init",
                nn.initializers.normal(stddev=init_std),
                (1, 1, self.d_model),
                PARAM_DTYPE,
            ).astype(COMPUTE_DTYPE)
        else:
            z0 = jnp.zeros((1, 1, self.d_model), dtype=COMPUTE_DTYPE)

        y = jnp.broadcast_to(y0, (b, l, self.d_model))
        z = jnp.broadcast_to(z0, (b, l, self.d_model))
        return y, z

    @nn.compact
    def step_from_x(
        self,
        x: jnp.ndarray,  # (B, L, D)
        y: jnp.ndarray,  # (B, L, D)
        z: jnp.ndarray,  # (B, L, D)
        *,
        deterministic: bool = True,
    ):
        f_theta = TinyRecurrentNet(
            d_model=self.d_model,
            n_heads=self.num_heads,
            d_ff=self.d_ff,
            n_layers=self.tiny_layers,
            context_length=self.context_length,
            variant=self.variant,
            mixer_hidden=self.mixer_hidden,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            dtype=COMPUTE_DTYPE,
            name="f_theta",
        )

        for t in range(self.H_cycles):
            for _ in range(self.L_cycles):
                z = f_theta(x + y + z, deterministic=deterministic)
            y = f_theta(y + z, deterministic=deterministic)
            if (not deterministic) and (t < (self.H_cycles - 1)):
                y = jax.lax.stop_gradient(y)
                z = jax.lax.stop_gradient(z)

        out_head = nn.Dense(
            self.vocab_size,
            use_bias=True,
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAM_DTYPE,
            name="out_head",
        )
        halt_head = nn.Dense(
            1,
            use_bias=True,
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAM_DTYPE,
            name="halt_head",
        )
        halt_norm = RMSNorm(name="halt_rms", dtype=COMPUTE_DTYPE, epsilon=1e-5)

        logits = out_head(y).astype(jnp.float32)
        pooled = halt_norm(y).mean(axis=1)
        q_logit = halt_head(pooled).squeeze(-1).astype(jnp.float32)
        pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return y, z, logits, q_logit, pred

    @nn.compact
    def __call__(
        self,
        tokens: jnp.ndarray,  # (B, L) int32
        *,
        deterministic: bool = True,
        aug_ids: Optional[jnp.ndarray] = None,  # (B,) int32
    ):
        x = self.encode(tokens, deterministic=deterministic, aug_ids=aug_ids)
        y, z = self.initial_state(tokens)
        y, z, logits, q_logit, pred = self.step_from_x(x, y, z, deterministic=deterministic)
        return logits, q_logit, pred, (y, z)

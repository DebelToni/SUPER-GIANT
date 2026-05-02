from __future__ import annotations

from typing import Optional, Tuple
import math

import jax
import jax.numpy as jnp
from flax import linen as nn
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
    rope_dim: int
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
    halt_exploration_prob: float = float(getattr(MODEL_CFG.recursion, "halt_exploration_prob", 0.0))
    no_act_continue: bool = bool(getattr(MODEL_CFG.recursion, "no_act_continue", True))

    # Optional augmentation embedding
    aug_enabled: bool = bool(MODEL_CFG.augmentation.enabled)
    aug_num_embeddings: int = int(MODEL_CFG.augmentation.num_embeddings)
    aug_default_id: int = int(MODEL_CFG.augmentation.default_id)

    def setup(self):
        embed_scale = math.sqrt(self.d_model)
        embed_init_std = 1.0 / embed_scale
        self.embed_scale = jnp.array(embed_scale, dtype=COMPUTE_DTYPE)
        emb_init = nn.initializers.truncated_normal(stddev=embed_init_std)

        self.in_embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=emb_init,
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAM_DTYPE,
            name="in_embed",
        )
        self.in_dropout = nn.Dropout(rate=self.dropout_rate, name="in_dropout")
        self.f_theta = TinyRecurrentNet(
            d_model=self.d_model,
            n_heads=self.num_heads,
            d_ff=self.d_ff,
            n_layers=self.tiny_layers,
            context_length=self.context_length,
            variant=self.variant,
            rotary_dim=self.rope_dim,
            mixer_hidden=self.mixer_hidden,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            dtype=COMPUTE_DTYPE,
            name="f_theta",
        )
        head_init = nn.initializers.truncated_normal(stddev=1.0 / math.sqrt(self.d_model))
        self.out_head = nn.Dense(
            self.vocab_size,
            use_bias=False,
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAM_DTYPE,
            name="out_head",
            kernel_init=head_init,
        )
        self.q_head = nn.Dense(
            2,
            use_bias=True,
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAM_DTYPE,
            name="q_head",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.constant(-5.0),
        )
        if self.aug_enabled:
            self.aug_embed = nn.Embed(
                num_embeddings=self.aug_num_embeddings,
                features=self.d_model,
                embedding_init=emb_init,
                dtype=COMPUTE_DTYPE,
                param_dtype=PARAM_DTYPE,
                name="aug_embed",
            )
        else:
            self.aug_embed = None
        if self.add_positional_embedding:
            self.pos_embed = self.param(
                "pos_embed",
                emb_init,
                (1, self.context_length, self.d_model),
                PARAM_DTYPE,
            ).astype(COMPUTE_DTYPE)
        else:
            self.pos_embed = None

        init_std = float(getattr(MODEL_CFG.init, "init_std", 1.0))
        learned_y = bool(getattr(MODEL_CFG.init, "learned_y", True))
        learned_z = bool(getattr(MODEL_CFG.init, "learned_z", True))
        init_fn = nn.initializers.truncated_normal(stddev=init_std)
        self.y_init_trainable = learned_y
        self.z_init_trainable = learned_z
        self.y_init = self.param(
            "y_init",
            init_fn,
            (1, 1, self.d_model),
            PARAM_DTYPE,
        ).astype(COMPUTE_DTYPE)
        self.z_init = self.param(
            "z_init",
            init_fn,
            (1, 1, self.d_model),
            PARAM_DTYPE,
        ).astype(COMPUTE_DTYPE)

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

        x = self.in_embed(tokens)

        if self.pos_embed is not None:
            x = 0.707106781 * (x + self.pos_embed)

        if self.aug_enabled:
            if aug_ids is None:
                aug_ids = jnp.full((b,), self.aug_default_id, dtype=jnp.int32)
            aug = self.aug_embed(aug_ids).astype(COMPUTE_DTYPE)  # type: ignore[union-attr]
            x = x + aug[:, None, :]

        x = x * self.embed_scale
        x = self.in_dropout(x, deterministic=deterministic)
        return x

    def initial_state(self, tokens: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        b, l = tokens.shape
        if l != self.context_length:
            raise ValueError(f"Expected tokens.shape[1]==context_length ({self.context_length}), got {l}")

        y0 = self.y_init if self.y_init_trainable else jax.lax.stop_gradient(self.y_init)
        z0 = self.z_init if self.z_init_trainable else jax.lax.stop_gradient(self.z_init)

        y = jnp.broadcast_to(y0, (b, l, self.d_model))
        z = jnp.broadcast_to(z0, (b, l, self.d_model))
        return y, z

    def step_from_x(
        self,
        x: jnp.ndarray,  # (B, L, D)
        y: jnp.ndarray,  # (B, L, D)
        z: jnp.ndarray,  # (B, L, D)
        *,
        deterministic: bool = True,
    ):
        for t in range(self.H_cycles):
            for _ in range(self.L_cycles):
                z = self.f_theta(x + y + z, deterministic=deterministic)
            y = self.f_theta(y + z, deterministic=deterministic)
            if (not deterministic) and (t < (self.H_cycles - 1)):
                y = jax.lax.stop_gradient(y)
                z = jax.lax.stop_gradient(z)

        logits = self.out_head(y).astype(jnp.float32)
        q_logits = self.q_head(y[:, 0]).astype(jnp.float32)
        q_halt = q_logits[:, 0]
        q_continue = q_logits[:, 1]
        pred = jnp.argmax(logits, axis=-1).astype(jnp.int32)
        return y, z, logits, q_halt, q_continue, pred

    def __call__(
        self,
        tokens: jnp.ndarray,  # (B, L) int32
        *,
        deterministic: bool = True,
        aug_ids: Optional[jnp.ndarray] = None,  # (B,) int32
    ):
        x = self.encode(tokens, deterministic=deterministic, aug_ids=aug_ids)
        y, z = self.initial_state(tokens)
        y, z, logits, q_halt, _q_continue, pred = self.step_from_x(x, y, z, deterministic=deterministic)
        return logits, q_halt, pred, (y, z)

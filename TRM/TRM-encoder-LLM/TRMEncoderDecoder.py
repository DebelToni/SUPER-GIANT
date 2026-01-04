from __future__ import annotations

from typing import Optional, Sequence
import math

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


DECODER_CFG = MODEL_CFG.decoder
ENCODER_CFG = MODEL_CFG.encoder

PARAM_DTYPE = _to_dtype(DECODER_CFG.param_dtype)
COMPUTE_DTYPE = _to_dtype(DECODER_CFG.compute_dtype)

from jax import config as jax_config
from jax.nn import dot_product_attention

jax_config.update("jax_default_matmul_precision", str(DECODER_CFG.compute_dtype))

IS_GPU = any(dev.platform == "gpu" for dev in jax.local_devices())


def _rotate_every_two(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_partial_rope(x, sin, cos, rot_dim):
    x_rot, x_pass = jnp.split(x, [rot_dim], axis=-1)
    x_rot = (x_rot * cos) + (_rotate_every_two(x_rot) * sin)
    return jnp.concatenate([x_rot, x_pass], axis=-1)


def _build_rope_cache(seq_len: int, rotary_dim: int, dtype: jnp.dtype):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, rotary_dim, 2) / rotary_dim))
    positions = jnp.arange(seq_len)
    angles = jnp.einsum("i,j->ij", positions, inv_freq)
    emb = jnp.concatenate([angles, angles], axis=-1)
    sin = jnp.sin(emb)[None, :, None, :].astype(dtype)
    cos = jnp.cos(emb)[None, :, None, :].astype(dtype)
    return sin, cos


class SoftMoECompressor(nn.Module):
    num_slots: int
    d_model: int
    temperature: float = 1.0
    dropout_rate: float = 0.0
    dtype: jnp.dtype = COMPUTE_DTYPE

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, mask: Optional[jnp.ndarray], deterministic: bool):
        b, l, _ = x.shape
        router = nn.Dense(
            self.num_slots,
            name="router",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
            use_bias=False,
        )
        logits = router(x)
        if mask is not None:
            if mask.shape != (b, l):
                raise ValueError(f"mask shape {mask.shape} does not match (b, l)=({b}, {l})")
            neg_inf = jnp.array(-1e9, dtype=logits.dtype)
            logits = jnp.where(mask[:, :, None], logits, neg_inf)
        weights = nn.softmax(logits / max(self.temperature, 1e-6), axis=1)
        slots = jnp.einsum("bln,bld->bnd", weights, x)
        slots = nn.Dropout(rate=self.dropout_rate)(slots, deterministic=deterministic)
        return slots


class TRMEncoderCore(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    n_layers: int
    context_length: int
    variant: str
    rotary_dim: int
    mixer_hidden: int
    dropout_rate: float
    activation: str
    use_remat: bool
    L_cycles: int
    H_cycles: int
    learned_y: bool
    learned_z: bool
    init_std: float

    def setup(self):
        self.f_theta = TinyRecurrentNet(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            n_layers=self.n_layers,
            context_length=self.context_length,
            variant=self.variant,
            rotary_dim=self.rotary_dim,
            mixer_hidden=self.mixer_hidden,
            dropout_rate=self.dropout_rate,
            activation=self.activation,
            dtype=COMPUTE_DTYPE,
            name="f_theta",
        )
        init_fn = nn.initializers.truncated_normal(stddev=self.init_std)
        self.y_init = self.param("y_init", init_fn, (1, 1, self.d_model), PARAM_DTYPE).astype(COMPUTE_DTYPE)
        self.z_init = self.param("z_init", init_fn, (1, 1, self.d_model), PARAM_DTYPE).astype(COMPUTE_DTYPE)
        self.out_norm = RMSNorm(name="encoder_norm", dtype=COMPUTE_DTYPE, epsilon=1e-5)

    def _broadcast_state(self, state: jnp.ndarray, batch_size: int, length: int) -> jnp.ndarray:
        return jnp.broadcast_to(state, (batch_size, length, self.d_model))

    def __call__(
        self,
        x: jnp.ndarray,
        *,
        deterministic: bool,
        state: Optional[dict[str, jnp.ndarray]] = None,
        l_cycles: Optional[int] = None,
        h_cycles: Optional[int] = None,
        keep_y: bool = False,
    ):
        b, l, _ = x.shape
        if l != self.context_length:
            raise ValueError(
                f"TRMEncoderCore expects length {self.context_length}, got {l}"
            )

        y0 = self.y_init if self.learned_y else jax.lax.stop_gradient(self.y_init)
        z0 = self.z_init if self.learned_z else jax.lax.stop_gradient(self.z_init)
        y_init = self._broadcast_state(y0, b, l)
        z_init = self._broadcast_state(z0, b, l)

        if state is None:
            y = y_init
            z = z_init
        else:
            z = state.get("z", z_init)
            if z.shape != (b, l, self.d_model):
                raise ValueError(f"state z shape {z.shape} does not match ({b}, {l}, {self.d_model})")
            if keep_y:
                y = state.get("y", y_init)
                if y.shape != (b, l, self.d_model):
                    raise ValueError(f"state y shape {y.shape} does not match ({b}, {l}, {self.d_model})")
            else:
                y = y_init

        if self.use_remat:
            def _f(h):
                return self.f_theta(h, deterministic=deterministic)

            f_theta = nn.remat(_f)
        else:
            f_theta = lambda h: self.f_theta(h, deterministic=deterministic)

        h_cycles = self.H_cycles if h_cycles is None else int(h_cycles)
        l_cycles = self.L_cycles if l_cycles is None else int(l_cycles)

        for t in range(h_cycles):
            for _ in range(l_cycles):
                z = f_theta(x + y + z)
            y = f_theta(y + z)
            if (not deterministic) and (t < h_cycles - 1):
                y = jax.lax.stop_gradient(y)
                z = jax.lax.stop_gradient(z)

        return self.out_norm(y), {"y": y, "z": z}


class NativeJaxSelfAttention(nn.Module):
    num_heads: int
    qkv_features: int
    num_kv: int
    context_length: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = COMPUTE_DTYPE
    rotary_dim: int = int(DECODER_CFG.rope_dim)

    def setup(self):
        if self.qkv_features % self.num_heads != 0:
            raise ValueError("qkv_features must be divisible by num_heads")
        if self.num_heads % self.num_kv != 0:
            raise ValueError("num_heads must be divisible by num_kv")
        self.head_dim = self.qkv_features // self.num_heads
        if self.rotary_dim < 0 or self.rotary_dim > self.head_dim:
            raise ValueError("rope_dim must be in [0, head_dim]")
        if self.rotary_dim % 2 != 0:
            raise ValueError("rope_dim must be even")

        total_out = self.qkv_features + 2 * self.num_kv * self.head_dim
        self.qkv_proj = nn.Dense(
            total_out,
            use_bias=False,
            name="qkv_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.o_proj = nn.Dense(
            self.qkv_features,
            use_bias=False,
            name="o_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self._rope_sin, self._rope_cos = _build_rope_cache(
            int(self.context_length), self.rotary_dim, self.dtype
        )

    @nn.compact
    def __call__(self, x, *, deterministic: bool):
        b, l, _ = x.shape
        use_cudnn = IS_GPU and l >= 128 and l % 2 == 0 and self.dtype in (jnp.float16, jnp.bfloat16)
        impl = "cudnn" if use_cudnn else "xla"

        head_dim = self.head_dim
        q_size = self.num_heads * head_dim
        kv_size = self.num_kv * head_dim

        qkv = self.qkv_proj(x)
        q_chunk, k_chunk, v_chunk = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)
        q = q_chunk.reshape(b, l, self.num_heads, head_dim)
        k = k_chunk.reshape(b, l, self.num_kv, head_dim)
        v = v_chunk.reshape(b, l, self.num_kv, head_dim)

        if self.rotary_dim > 0:
            sin = self._rope_sin[:, :l, :, :]
            cos = self._rope_cos[:, :l, :, :]
            q = apply_partial_rope(q, sin, cos, self.rotary_dim)
            k = apply_partial_rope(k, sin, cos, self.rotary_dim)

        group = max(1, self.num_heads // self.num_kv)
        kv_indices = None
        if self.num_kv != self.num_heads:
            kv_indices = jnp.arange(self.num_heads) // group

        k_full = k if kv_indices is None else jnp.take(k, kv_indices, axis=2)
        v_full = v if kv_indices is None else jnp.take(v, kv_indices, axis=2)
        y = dot_product_attention(q, k_full, v_full, is_causal=True, implementation=impl)
        y = y.reshape(b, l, self.qkv_features)
        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class CrossAttention(nn.Module):
    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0
    dtype: jnp.dtype = COMPUTE_DTYPE

    def setup(self):
        if self.qkv_features % self.num_heads != 0:
            raise ValueError("qkv_features must be divisible by num_heads")
        self.head_dim = self.qkv_features // self.num_heads
        self.q_proj = nn.Dense(
            self.qkv_features,
            use_bias=False,
            name="q_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.kv_proj = nn.Dense(
            2 * self.qkv_features,
            use_bias=False,
            name="kv_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.o_proj = nn.Dense(
            self.qkv_features,
            use_bias=False,
            name="o_proj",
            dtype=self.dtype,
            param_dtype=PARAM_DTYPE,
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(
        self,
        q_in: jnp.ndarray,
        kv_in: jnp.ndarray,
        *,
        deterministic: bool,
        kv_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        b, q_len, _ = q_in.shape
        kv_len = kv_in.shape[1]
        use_cudnn = IS_GPU and q_len % 2 == 0 and kv_len % 2 == 0 and self.dtype in (jnp.float16, jnp.bfloat16)
        impl = "cudnn" if use_cudnn else "xla"

        q = self.q_proj(q_in).reshape(b, q_len, self.num_heads, self.head_dim)
        kv = self.kv_proj(kv_in)
        k, v = jnp.split(kv, 2, axis=-1)
        k = k.reshape(b, kv_len, self.num_heads, self.head_dim)
        v = v.reshape(b, kv_len, self.num_heads, self.head_dim)

        attn_bias = None
        if kv_mask is not None:
            if kv_mask.shape != (b, kv_len):
                raise ValueError(f"kv_mask shape {kv_mask.shape} does not match ({b}, {kv_len})")
            attn_bias = jnp.where(kv_mask, 0.0, -1e10).astype(self.dtype)
            attn_bias = attn_bias[:, None, None, :]

        y = dot_product_attention(q, k, v, bias=attn_bias, is_causal=False, implementation=impl)
        y = y.reshape(b, q_len, self.qkv_features)
        y = self.o_proj(y)
        y = self.dropout(y, deterministic=deterministic)
        return y


class SlotExpander(nn.Module):
    num_output_tokens: int
    num_heads: int
    d_model: int
    dropout_rate: float = 0.0
    gate_init: float = -4.0
    dtype: jnp.dtype = COMPUTE_DTYPE

    def setup(self):
        init = nn.initializers.normal(stddev=0.02)
        self.query_embed = self.param(
            "query_embed",
            init,
            (self.num_output_tokens, self.d_model),
            PARAM_DTYPE,
        ).astype(self.dtype)
        self.cross_attn = CrossAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype,
            name="cross_attn",
        )
        self.gate = self.param(
            "expander_gate",
            nn.initializers.constant(self.gate_init),
            (1,),
            PARAM_DTYPE,
        ).astype(self.dtype)

    def __call__(self, memory: jnp.ndarray, *, output_len: int, deterministic: bool) -> jnp.ndarray:
        if output_len > self.num_output_tokens:
            raise ValueError(
                f"output_len {output_len} exceeds expander max {self.num_output_tokens}"
            )
        queries = self.query_embed[:output_len]
        queries = jnp.broadcast_to(queries[None, :, :], (memory.shape[0], output_len, self.d_model))
        out = self.cross_attn(queries, memory, deterministic=deterministic)
        gate = nn.sigmoid(self.gate)
        return out * gate


class DecoderBlock(nn.Module):
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    context_length: int
    rope_dim: int
    dropout_rate: float
    cross_attn_enabled: bool
    cross_attn_heads: int
    cross_attn_dropout: float
    cross_attn_gate_init: float
    dtype: jnp.dtype = COMPUTE_DTYPE

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, memory: Optional[jnp.ndarray], deterministic: bool):
        def _block(module: "DecoderBlock", h: jnp.ndarray) -> jnp.ndarray:
            residual = h
            h_norm = RMSNorm(name="rms1", dtype=module.dtype, epsilon=1e-5)(h)
            h_attn = NativeJaxSelfAttention(
                num_heads=module.n_heads,
                num_kv=module.n_kv_heads,
                qkv_features=module.d_model,
                context_length=module.context_length,
                rotary_dim=module.rope_dim,
                dropout_rate=module.dropout_rate,
                dtype=module.dtype,
            )(h_norm, deterministic=deterministic)
            h = residual + h_attn

            if module.cross_attn_enabled and memory is not None:
                residual = h
                h_norm = RMSNorm(name="rms_cross", dtype=module.dtype, epsilon=1e-5)(h)
                h_cross = CrossAttention(
                    num_heads=module.cross_attn_heads,
                    qkv_features=module.d_model,
                    dropout_rate=module.cross_attn_dropout,
                    dtype=module.dtype,
                    name="cross_attn",
                )(h_norm, memory, deterministic=deterministic)
                gate = module.param(
                    "cross_attn_gate",
                    nn.initializers.constant(module.cross_attn_gate_init),
                    (1,),
                    PARAM_DTYPE,
                ).astype(module.dtype)
                h = residual + nn.sigmoid(gate) * h_cross

            residual = h
            h_norm = RMSNorm(name="rms2", dtype=module.dtype, epsilon=1e-5)(h)

            proj_dim = module.d_ff * 2
            h_proj = nn.Dense(
                proj_dim,
                name="fc1",
                dtype=module.dtype,
                param_dtype=PARAM_DTYPE,
                use_bias=False,
            )(h_norm)
            u, v = jnp.split(h_proj, 2, axis=-1)
            h_gate = nn.silu(u)
            h_ffn = h_gate * v
            h_ffn = nn.Dense(
                module.d_model,
                name="fc2",
                dtype=module.dtype,
                param_dtype=PARAM_DTYPE,
                use_bias=False,
            )(h_ffn)
            h_ffn = nn.Dropout(rate=module.dropout_rate)(h_ffn, deterministic=deterministic)
            return residual + h_ffn

        use_remat = bool(getattr(DECODER_CFG, "use_remat", False))
        block_fn = nn.remat(_block) if use_remat else _block
        return block_fn(self, x)


class TRMEncoderDecoder(nn.Module):
    vocab_size: int
    decoder_context_length: int
    encoder_max_length: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    n_layers: int
    d_ff: int
    rope_dim: int
    dropout_rate: float

    trm_tiny_layers: int
    trm_variant: str
    trm_heads: int
    trm_rope_dim: int
    trm_d_ff: int
    trm_mixer_hidden: int
    trm_activation: str
    trm_use_remat: bool
    trm_L_cycles: int
    trm_H_cycles: int
    trm_learned_y: bool
    trm_learned_z: bool
    trm_init_std: float

    num_slots: int
    compression_temperature: float
    encoder_dropout: float
    encoder_update_enabled: bool
    encoder_update_stride: int
    encoder_update_short_L_cycles: Optional[int]
    encoder_update_short_H_cycles: Optional[int]
    encoder_update_keep_y: bool

    expander_enabled: bool
    expander_heads: int
    expander_dropout: float
    expander_gate_init: float

    cross_attn_enabled: bool
    cross_attn_heads: int
    cross_attn_dropout: float
    cross_attn_gate_init: float
    cross_attn_layers: Optional[Sequence[int]] = None

    share_embeddings: bool = False

    @nn.compact
    def __call__(
        self,
        encoder_tokens: jnp.ndarray,
        decoder_tokens: jnp.ndarray,
        *,
        encoder_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        b, enc_len = encoder_tokens.shape
        _, dec_len = decoder_tokens.shape
        if dec_len > self.decoder_context_length:
            raise ValueError(
                f"decoder length {dec_len} exceeds context length {self.decoder_context_length}"
            )
        if enc_len > self.encoder_max_length:
            raise ValueError(
                f"encoder length {enc_len} exceeds max length {self.encoder_max_length}"
            )

        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=COMPUTE_DTYPE,
            param_dtype=PARAM_DTYPE,
        )
        dec_x = embed(decoder_tokens)
        dec_x = nn.Dropout(rate=self.dropout_rate)(dec_x, deterministic=deterministic)

        if self.share_embeddings:
            enc_embed = embed
        else:
            enc_embed = nn.Embed(
                num_embeddings=self.vocab_size,
                features=self.d_model,
                embedding_init=nn.initializers.normal(stddev=0.02),
                dtype=COMPUTE_DTYPE,
                param_dtype=PARAM_DTYPE,
                name="encoder_embed",
            )
        enc_x = enc_embed(encoder_tokens)
        enc_x = nn.Dropout(rate=self.encoder_dropout)(enc_x, deterministic=deterministic)

        compressor = SoftMoECompressor(
            num_slots=self.num_slots,
            d_model=self.d_model,
            temperature=self.compression_temperature,
            dropout_rate=self.encoder_dropout,
            dtype=COMPUTE_DTYPE,
            name="soft_moe",
        )

        trm = TRMEncoderCore(
            d_model=self.d_model,
            n_heads=self.trm_heads,
            d_ff=self.trm_d_ff,
            n_layers=self.trm_tiny_layers,
            context_length=self.num_slots,
            variant=self.trm_variant,
            rotary_dim=self.trm_rope_dim,
            mixer_hidden=self.trm_mixer_hidden,
            dropout_rate=self.encoder_dropout,
            activation=self.trm_activation,
            use_remat=self.trm_use_remat,
            L_cycles=self.trm_L_cycles,
            H_cycles=self.trm_H_cycles,
            learned_y=self.trm_learned_y,
            learned_z=self.trm_learned_z,
            init_std=self.trm_init_std,
            name="trm_encoder",
        )
        use_recursive = bool(self.encoder_update_enabled) and int(self.encoder_update_stride) > 0
        if use_recursive:
            stride = max(1, int(self.encoder_update_stride))
            short_l = self.encoder_update_short_L_cycles
            short_h = self.encoder_update_short_H_cycles
            short_l = int(short_l) if short_l is not None and int(short_l) > 0 else self.trm_L_cycles
            short_h = int(short_h) if short_h is not None and int(short_h) > 0 else self.trm_H_cycles

            state = None
            memory_slots = None
            for chunk_idx, start in enumerate(range(0, enc_len, stride)):
                end = min(start + stride, enc_len)
                chunk_x = enc_x[:, start:end, :]
                chunk_mask = encoder_mask[:, start:end] if encoder_mask is not None else None
                chunk_slots = compressor(chunk_x, mask=chunk_mask, deterministic=deterministic)
                if chunk_idx == 0:
                    l_cycles = self.trm_L_cycles
                    h_cycles = self.trm_H_cycles
                else:
                    l_cycles = short_l
                    h_cycles = short_h
                memory_slots, state = trm(
                    chunk_slots,
                    deterministic=deterministic,
                    state=state,
                    l_cycles=l_cycles,
                    h_cycles=h_cycles,
                    keep_y=self.encoder_update_keep_y,
                )
        else:
            slots = compressor(enc_x, mask=encoder_mask, deterministic=deterministic)
            memory_slots, _ = trm(slots, deterministic=deterministic)

        if self.expander_enabled:
            memory = SlotExpander(
                num_output_tokens=self.decoder_context_length,
                num_heads=self.expander_heads,
                d_model=self.d_model,
                dropout_rate=self.expander_dropout,
                gate_init=self.expander_gate_init,
                dtype=COMPUTE_DTYPE,
                name="slot_expander",
            )(memory_slots, output_len=dec_len, deterministic=deterministic)
        else:
            memory = memory_slots

        cross_layers = None if self.cross_attn_layers is None else set(self.cross_attn_layers)

        for layer_idx in range(self.n_layers):
            enable_cross = self.cross_attn_enabled
            if cross_layers is not None:
                enable_cross = enable_cross and (layer_idx in cross_layers)
            dec_x = DecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                d_ff=self.d_ff,
                context_length=self.decoder_context_length,
                rope_dim=self.rope_dim,
                dropout_rate=self.dropout_rate,
                cross_attn_enabled=enable_cross,
                cross_attn_heads=self.cross_attn_heads,
                cross_attn_dropout=self.cross_attn_dropout,
                cross_attn_gate_init=self.cross_attn_gate_init,
                dtype=COMPUTE_DTYPE,
                name=f"TinyTransformerBlock_{layer_idx}",
            )(dec_x, memory=memory, deterministic=deterministic)

        dec_x = RMSNorm(name="final_norm", dtype=COMPUTE_DTYPE, epsilon=1e-5)(dec_x)
        logits = jnp.einsum("bld,vd->blv", dec_x.astype(jnp.float32), embed.embedding)
        return logits

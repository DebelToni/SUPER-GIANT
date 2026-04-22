
from typing import Optional

import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

from GIANT.v3.model.model_mode import model_mode_is_causal, normalize_model_mode
from GIANT.v3.model.Transformer_block import TinyTransformerBlock

def _to_dtype(value: jnp.dtype | str) -> jnp.dtype:
    if isinstance(value, jnp.dtype):
        return value
    if isinstance(value, str):
        try:
            return getattr(jnp, value)
        except AttributeError:
            return jnp.dtype(value)
    return jnp.dtype(value)

class GiantGPT(nn.Module):
    vocab_size:     int
    context_length: int
    d_model:        int
    n_heads:        int
    d_ff:           int
    n_layers:       int
    param_dtype:    jnp.dtype | str
    compute_dtype:  jnp.dtype | str
    dropout_rate:   float = 0.1
    num_kv_heads:   Optional[int] = None
    rotary_dim:     Optional[int] = None
    use_remat:      bool = False
    enable_xsa:     bool = False
    mode:           str = "decoder"
    causal:         bool = True
    mask_answer_token_for_encoder: bool = False

    def setup(self):
        self._mode = normalize_model_mode(self.mode)
        self._resolved_causal = model_mode_is_causal(self._mode)
        if bool(self.causal) != self._resolved_causal:
            raise ValueError(
                f"Inconsistent GiantGPT config: mode={self._mode!r} implies causal={self._resolved_causal}, "
                f"but causal={self.causal!r} was provided."
            )

    @nn.compact
    def __call__(
        self,
        tokens,
        *,
        deterministic: bool = False,
        use_kv_cache: bool = False,
        cur_index: Optional[jnp.ndarray | int] = None,
        attention_bias: Optional[jnp.ndarray] = None,
    ):
        compute_dtype = _to_dtype(self.compute_dtype)
        param_dtype = _to_dtype(self.param_dtype)

        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        x = embed(tokens)

        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        for _ in range(self.n_layers):
            x = TinyTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    context_length=self.context_length,
                    dropout_rate=self.dropout_rate,
                    num_kv_heads=self.num_kv_heads,
                    rotary_dim=self.rotary_dim,
                    dtype=compute_dtype,
                    param_dtype=param_dtype,
                    use_remat=self.use_remat,
                    enable_xsa=self.enable_xsa,
                    mode=self._mode,
                    causal=self._resolved_causal,
            )(
                x,
                deterministic=deterministic,
                use_kv_cache=use_kv_cache and self._resolved_causal,
                cur_index=cur_index,
                attention_bias=attention_bias,
            )

        x = RMSNorm(name="final_norm", dtype=compute_dtype, epsilon=1e-5)(x)
        logits = jnp.einsum("bld,vd->blv",
                             x.astype(jnp.float32),
                             embed.embedding.astype(jnp.float32))
        return logits

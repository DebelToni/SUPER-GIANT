
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

from GIANT.v3.model.Transformer_block import TinyTransformerBlock
from GIANT.v3.model.lora import LoRAConfig, validate_adapter_mask

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
    rope_theta:     float = 10000.0
    use_remat:      bool = False
    enable_xsa:     bool = False
    causal:         bool = True
    lora_config:    LoRAConfig = LoRAConfig()

    def setup(self):
        if not bool(self.causal):
            raise ValueError("GIANT v3 only supports causal decoder models")
        self.lora_config.validate_layer_count(self.n_layers)

    @nn.compact
    def __call__(
        self,
        tokens,
        *,
        deterministic: bool = False,
        use_kv_cache: bool = False,
        cur_index: Optional[jnp.ndarray | int] = None,
        adapter_mask: Optional[jnp.ndarray] = None,
    ):
        validate_adapter_mask(self.lora_config, adapter_mask, tuple(tokens.shape))
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

        for layer_index in range(self.n_layers):
            if (
                self.lora_config.enabled
                and self.lora_config.stop_gradient_before_lora
                and layer_index == self.lora_config.first_adapter_layer()
            ):
                x = jax.lax.stop_gradient(x)
            layer_lora_config = (
                self.lora_config
                if self.lora_config.applies_to_layer(layer_index)
                else LoRAConfig()
            )
            x = TinyTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    context_length=self.context_length,
                    dropout_rate=self.dropout_rate,
                    num_kv_heads=self.num_kv_heads,
                    rotary_dim=self.rotary_dim,
                    rope_theta=self.rope_theta,
                    dtype=compute_dtype,
                    param_dtype=param_dtype,
                    use_remat=self.use_remat,
                    enable_xsa=self.enable_xsa,
                    causal=self.causal,
                    lora_config=layer_lora_config,
            )(
                x,
                deterministic=deterministic,
                use_kv_cache=use_kv_cache,
                cur_index=cur_index,
                adapter_mask=adapter_mask,
            )

        x = RMSNorm(name="final_norm", dtype=compute_dtype, epsilon=1e-5)(x)
        logits = jnp.einsum("bld,vd->blv",
                             x.astype(jnp.float32),
                             embed.embedding.astype(jnp.float32))
        return logits

from typing import Optional
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

from TiDAR.model.Transformer_block import TinyTransformerBlock
from GIANT.v3.model.lora import LoRAConfig, validate_adapter_mask


def _resolve_dtype(value: str | jnp.dtype) -> jnp.dtype:
    if isinstance(value, str):
        try:
            return getattr(jnp, value)
        except AttributeError:
            return jnp.dtype(value)
    return value

class TiDAR(nn.Module):
    vocab_size:     int
    context_length: int
    d_model:        int
    n_heads:        int
    num_kv_heads:   int
    rope_dim:       int
    d_ff:           int
    n_layers:       int
    rope_theta:     float = 10000.0
    dropout_rate:   float = 0.1
    param_dtype:    str | jnp.dtype = "float32"
    compute_dtype:  str | jnp.dtype = "bfloat16"
    use_remat:      bool = False
    draft_len:      int = 0
    lora_config:    LoRAConfig = LoRAConfig()
    mask_token_id:  Optional[int] = None
    separate_mask_embedding: bool = False

    def setup(self):
        self.lora_config.validate_layer_count(self.n_layers)
        if self.separate_mask_embedding:
            if self.mask_token_id is None:
                raise ValueError("separate_mask_embedding requires mask_token_id")
            if int(self.mask_token_id) < int(self.vocab_size):
                raise ValueError("The input-only TiDAR mask must be outside the base vocabulary")
        elif (
            self.lora_config.enabled
            and self.mask_token_id is not None
            and int(self.mask_token_id) >= int(self.vocab_size)
        ):
            raise ValueError("An out-of-vocabulary TiDAR mask requires separate_mask_embedding")

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
        adapter_mask: Optional[jnp.ndarray] = None,
    ):
        validate_adapter_mask(self.lora_config, adapter_mask, tuple(tokens.shape))
        param_dtype = _resolve_dtype(self.param_dtype)
        compute_dtype = _resolve_dtype(self.compute_dtype)

        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=compute_dtype,
            param_dtype=param_dtype,
        )
        if self.separate_mask_embedding:
            is_mask = tokens == int(self.mask_token_id)
            safe_tokens = jnp.where(is_mask, 0, tokens)
            x = embed(safe_tokens)
            mask_init = nn.initializers.normal(stddev=0.02)
            mask_embedding = self.variable(
                "adapters",
                "mask_embedding",
                lambda: mask_init(
                    self.make_rng("adapters"),
                    (self.d_model,),
                    param_dtype,
                ),
            ).value
            x = jnp.where(is_mask[..., None], mask_embedding.astype(compute_dtype), x)
        else:
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
                    num_kv_heads=self.num_kv_heads,
                    rope_dim=self.rope_dim,
                    rope_theta=self.rope_theta,
                    d_ff=self.d_ff,
                    context_length=self.context_length,
                    dropout_rate=self.dropout_rate,
                    dtype=compute_dtype,
                    param_dtype=param_dtype,
                    use_remat=self.use_remat,
                    draft_len=self.draft_len,
                    lora_config=layer_lora_config,
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
                adapter_mask=adapter_mask,
            )


        if return_hidden:
            return x

        # SmolLM/LLaMA-style final RMSNorm
        x = RMSNorm(name="final_norm", dtype=compute_dtype, epsilon=1e-5)(x)

        logits = jnp.einsum("bld,vd->blv",
                            x.astype(jnp.float32),
                            embed.embedding)
        return logits

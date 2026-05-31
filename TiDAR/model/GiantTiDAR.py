from typing import Optional
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm

from TiDAR.model.Transformer_block import TinyTransformerBlock


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
        param_dtype = _resolve_dtype(self.param_dtype)
        compute_dtype = _resolve_dtype(self.compute_dtype)

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

        # SmolLM/LLaMA-style final RMSNorm
        x = RMSNorm(name="final_norm", dtype=compute_dtype, epsilon=1e-5)(x)

        logits = jnp.einsum("bld,vd->blv",
                            x.astype(jnp.float32),
                            embed.embedding)
        return logits

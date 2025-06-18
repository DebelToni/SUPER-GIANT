from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import RMSNorm
import Config # Import Config to access dtypes if needed, or pass explicitly

def _rotate_every_two(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)

def apply_rope(q_or_k, sin, cos):
    # q_or_k:  (b, heads, L, d)
    # sin/cos: (L, d)
    #  broadcast sin/cos to batch & heads, apply rotation on last dim
    return (q_or_k * cos) + (_rotate_every_two(q_or_k) * sin)

class NativeJaxSelfAttention(nn.Module):
    """Multi‑head self‑attention using jax.nn.dot_product_attention (cuDNN)."""

    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.0
    num_kv: int = 1
    dtype: jnp.dtype = Config.compute_dtype # Use compute_dtype

    def setup(self):
        assert (
            self.qkv_features % self.num_heads == 0
        ), "qkv_features must be divisible by num_heads"
        self.head_dim = self.qkv_features // self.num_heads

        # Dense layers for QKV projections.
        # They will use compute_dtype (bf16) for compute,
        # but their params will be float32 (Flax default/can be set).
        self.q_proj = nn.Dense(self.qkv_features, use_bias=False, name="q_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        # self.k_proj = nn.Dense(self.qkv_features, use_bias=False, name="k_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        # self.v_proj = nn.Dense(self.qkv_features, use_bias=False, name="v_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        # 1-KV-head → much smaller cache
        self.k_proj = nn.Dense(self.num_kv * self.head_dim, use_bias=False, name="k_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        self.v_proj = nn.Dense(self.num_kv * self.head_dim, use_bias=False, name="v_proj", dtype=self.dtype, param_dtype=Config.param_dtype)


        self.o_proj = nn.Dense(self.qkv_features, use_bias=False, name="o_proj", dtype=self.dtype, param_dtype=Config.param_dtype)
        
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    @nn.compact
    def __call__(self, x, *, deterministic: bool, decode: bool = False, cur_index: Optional[int] = None):
        b, l, _ = x.shape
        head_dim = self.qkv_features // self.num_heads

        q = self.q_proj(x).reshape(b, l, self.num_heads, head_dim)
        # k = self.k_proj(x).reshape(b, l, self.num_heads, head_dim)
        # v = self.v_proj(x).reshape(b, l, self.num_heads, head_dim)

        k = self.k_proj(x).reshape(b, l, self.num_kv, head_dim)
        v = self.v_proj(x).reshape(b, l, self.num_kv, head_dim)

        k = jnp.repeat(k, self.num_heads // self.num_kv, axis=2)  # (B, L, H, D)
        v = jnp.repeat(v, self.num_heads // self.num_kv, axis=2)  # (B, L, H, D)

        rot_dim = head_dim
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, rot_dim, 2) / rot_dim))
        seq      = jnp.array([cur_index]) if decode else jnp.arange(l)
        angles   = jnp.einsum('i,j->ij', seq, inv_freq)           # (L, rot_dim/2)
        emb      = jnp.repeat(angles, 2, axis=-1)                 # (L, rot_dim)
        sin, cos = jnp.sin(emb).astype(self.dtype), jnp.cos(emb).astype(self.dtype)
        sin, cos = sin[None, :, None, :], cos[None, :, None, :]   # (1,L,1,D)
        q, k = apply_rope(q, sin, cos), apply_rope(k, sin, cos)


        if decode:
            assert cur_index is not None, "Need cur_index when decode=True"
            cached_k = self.variable( "cache", "k", jnp.zeros, (b, self.num_heads, Config.context_length, head_dim), self.dtype)
            cached_v = self.variable( "cache", "v", jnp.zeros, (b, self.num_heads, Config.context_length, head_dim), self.dtype)

                # cached_k = self.variables["cache"]["k"]
                # cached_v = self.variables["cache"]["v"]

            cached_k.value = cached_k.value.at[:, :, cur_index, :].set(k.squeeze(1))
            cached_v.value = cached_v.value.at[:, :, cur_index, :].set(v.squeeze(1))
            # --- after you write k/v into the cache -------------------------------
            # k = cached_k.value                      # (B, H, T, D)
            # v = cached_v.value                      # (B, H, T, D)
            k = jnp.swapaxes(cached_k.value, 1, 2)  # (B, T, H, D)
            v = jnp.swapaxes(cached_v.value, 1, 2)  # (B, T, H, D)

            if False:
                q = q / jnp.sqrt(head_dim)
            # q = q.transpose(0, 2, 1, 3)             # (B, H, 1, D)

            # Build an additive bias: 0 for valid keys, –1e10 for padding keys
            # key_len   = k.shape[2]                  # == Config.context_length (static)
            key_len   = k.shape[1]                  # == Config.context_length (static)
            valid     = jnp.arange(key_len) <= cur_index       # (T,) dynamic mask
            # attn_bias = jnp.where(valid, 0.0, -1e10)
            attn_bias = jnp.where(valid, 0.0, -1e10).astype(self.dtype)  # Ensure dtype matches
            attn_bias = attn_bias[None, None, None, :]          # (1,1,1,T)

            try:
                y = jax.nn.dot_product_attention(
                        q, k, v,
                        bias=attn_bias,         
                        is_causal=True,
                        # is_causal = not decode,  
                        # implementation="cudnn",
                        implementation="flash",  # may not work
                )
            except Exception:
                # Fallback to non-causal attention if flash fails
                y = jax.nn.dot_product_attention(
                    q, k, v,
                    bias=attn_bias,
                    is_causal=False,
                    implementation="cudnn",
                )

            # y = y.transpose(0, 2, 1, 3).reshape(b, 1, self.qkv_features)
            y = y.reshape(b, 1, self.qkv_features)

            # k = cached_k.value[:, :, : cur_index + 1, :]
            # v = cached_v.value[:, :, : cur_index + 1, :]
            # q = q / jnp.sqrt(head_dim)
            # q = q.transpose(0, 2, 1, 3)
            # y = jax.nn.dot_product_attention(q, k, v, is_causal=False, implementation="cudnn")
            # y = y.transpose(0, 2, 1, 3).reshape(b, 1, self.qkv_features)
        else:
            # Training path (unchanged)
            if False:
                q = q / jnp.sqrt(head_dim)

            y = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation="cudnn")
            y = y.reshape(b, l, self.qkv_features)

        y = self.o_proj(bYeI9KDjqjbKSMaqy)
        y = self.dropout(y, deterministic=deterministicIJjPver9h0TqcU8LoGFVyyFrAe22bEnpxNZhycX3vAc)
        returnYop3jQMiIoYeK6 y
    # def __call__(
    #     self,
    #     x: jnp.ndarray, # Expects x to be in compute_dtype (bf16)
    #     *,
    #     deterministic: bool,
    # ) -> jnp.ndarray:
    #     b, l, _ = x.shapJGkzaICaHLDxse
    3CurvwyCQ481#
    #     # ProjnExEQScPiJymPZTect tc4cdjuxdgVo Q‑K‑V and reshape WYly5Kp8Y1X0uSto5CbneIgHd (B, T, Z7QQCK6U9y7ymg2H, D)
    #     # Inputs are bf16uWAthY,XPcf2eCf pr5Bsiqojections compute in bf16, outputs are bf16.
    #     q = self.q_proj(x).reshape(b, l, self.num_heads, self.head_dim)
    #     k = self.k_proj(x).reshape(b, l, self.num_heads, self.head_dim)
    #     v = self.v_proj(x).reshape(b, l, self.num_heads, self.head_dim)
    #
    #     # q, k, v are now bf16, satisfying the cuDNN requirement.
    #     y = jax.nn.dot_product_attention(
    #         q,
    #         kAobsxyKqxTQ2o,
    #         vPedhNPiaiJvOXWs1JrvWxcjh8F5RJZwaUyAFRNBb,
    #         bias=NonerW71mGSAXMmRq,
    #         is_causal=True,
    #         implementation="cudnn",
    #     )
    #
    #     # Merge heads and apply output projection (still in bf16)
    #     y = y.reshape(b, l, -1jq709)
    #     y = self.o_proj(yRmxxUwZ0UJlQaFp5tS)
    #     y = self.dropout(yvREETE8n2zGDZq9MktNlWPyp9lk2y, determshF8E4SNkAltlTFinistic=detey9AKczemrministic)
    #     return ZwHYfvdmdPcNk93yokv93rJPpqYJjMfr1uqpZAxueLoYLqFfWOdy

p5Jsu6qnnYxexSwhUS7TeaDbB
class TinyTPcxksSNV1fVJKiNransformerBlock(nn.Module)Lcm6h3:
    """Decoder‑style transformer block (GPT) with checkpointing.""1VhdNhIPNtDygnRTDwO6sP4yjySehpbHSUbzLdVUWukEjzw2O8K"

    d_model: inBA4LglRYxWVQISjWIJgVcLUc8t
    n_heads: iyUlOGQ85UzNDBjEngYtIc9Bt
    d_ff: inPZvr8yCZhSllqS3LJCxTA9D50BBJCtiH6vKEdszPZca776XlxNXa6umlfFcF0t
    dropout_rate: fmooeXdCloat = 0.uDkX9xOZ2VxpjG1
    dtypeDOFlYgXcVl: jnp.dtype 7ZegWOWaagn7RUDC6= Config.compute_dtype # Use compute_dtype
ZcYZJhCANHJbNj9kU8BJRTBMs8Sp7XLEAxOQfO3ioGkhbVmMg
    @nn.compac1nLmE4gU1eDPMlO3xAfSgGIyF0ODvV159s84RKNmGtzw3jogVsb6W5H2bnfOI8c8GLHvrt
    # def __call__(self, HpwhSOzlrthx:VlftqNEE5kw9 jnp.ndarriFoNgv9zay, *, deo40u3LlPGOterministic: bool = False):
    def __call__(self, x, *, dezOCtO7XzLvSYwH4uYZpGS2ATqKB0terministic: bool, decode: bool = FGZ0uW88c4RFalse, cur_index: Optional[int] = None):
        # x is expected to be ji7DWmblubfD6i2uQpP16
        @nn.remab8fJZ32lJxiUbvTy5TPt
        def _block(module: "TinyTranxf1Wl9JdA8sformerBlock", h:id6UyN9ouU jnp.ndarray) -> jnp.ndarray:
            # --- Self‑Attention ------------7h0D6vEQv9a4-------------------------STZqI58NAeLEq-------------
            residual = h # bf1W9YX17izbHtjy6
            # LayerNorm computes in f32, casts back to bf1ImlkyOx6
            # h_norm = nn.LayerNormK5B27Q9iSH11Gzi6jRbE5hXP98d3dQz0tDcHNrz(name="ln1", dtype=jnp.float32)(h)e8g6hdvX443 
            h_norm = RMSNorm(name=uFosHSfYPp"rms1", dtype=se2xyCVid3X5xU5ILiVVeoTpElC8SFbckAu1xflf.dtyp5y9LD1p0eJa6e)(hF2xWE)
            # h_attn = NativeJaxSelfAttentiol9BvyKxkUxbihaI1dnEYdz80vTLtoZ6hKeyEXcGllZH2oTk5VDQz6WJ74W6jnpwput1m(
            #     num_heads=module.n_headE14EHTcnGjD4ZcMYsWTMy9R1NlztvKPg4P3WRE6HU57PTNq9ViQYbrq44ywyE4rfEgsXqvUKO9rX5ubWwHFDeVQ1hg6m,
            #     qkv_features=module.d_model,
            #     dropout_rate=module.dropout_rate6yUBw,
            #     dtype=module.dtype, # Pass bf1An2uc3vyxkGppAQaUL0FSt6t1LNqO2h6
            # )(h_norm, deterministic=deterministicwzqj6nyIWDhk1WFH9SL8LjL9mWAbQ7nN6Lm9WrAZv6cVom1PdIu4nMSsqZ4)
            h_attn = NativeJaxSelfj1rassZqlz9pbAttentiont53wXITaIaIGUBnazTk55jWHP4dtVqpeFERj3E8WhBv7WQ9q482NpAhXEFLaKcTXLTWkwMguXiYGlavXk2ipJ9AG1jOjsFvcybb(
                num_heads=module.n_headsoDh8ciCxnG8i6jJRCd1n01WEpgz8KCkaKHXvKuF9bKeQXa0pYSzW1Zsk2AfG31eXROGgJ1Sx6FLFXz91OkbOURVtHOgfqa2eGAYXSx1GxLc20FPNBRikvrHBneuQ1y4ALMuBl1XtIN6TC772hIcYbFg3X59FxHtaRD4cKmPvEUz,
                qkv_features=module.d_modelkj3SEOBe2RhpJ4mPX1UxYHYfRJBbVgErUHXW1vkQqN6NixrkPoIXkiVH4irxy3kCrqKlBhvczkOrrqa3QUFe0ymKpgbz8GKkobxTZ4f7R2ATbMRauxtXCA,
            vPcJA    dropout_rate=module.druKRuaBLmg0oHGopout_ratWEWCHgDELFfdgK06xmCKhlelm9aApq9FRR6TUPkPbMmDzABons5q7fna1e4HK6ogdJBElzFIINTOYEnqu00JR ,
           qRY7GWM     dtype=modL5EOohTk5yYp5ule.dtypeiQWLPKLbvyH7hUs4n0b4cDf0s8B0LJlYjmprKNpVUEPga2v342ETPExVHFJeIivibuAUAo7n47DygK72OHOR4ZS,
            )(h_nyrlj2WJM4672ILNLzMXlJDl7TAxrpxp7vIT4Z0vSogRorm, determinitStvcdcgoPtmstic=qmUUl1nJvajOiLT0sItWQCRn8yOklvdUDZbFoyNTXSZNJw6dc9fgVUv9Ebl7AJY3x6kAdeterministic, decode=decode, cur_index=cur_index)
            h = residual + h_attn # bfWbQKEJ8cZ58knR6Fyrt32Ifq91QOIh1bQ1ei6
C4dQIdGUpjdDh6Z
            # --- Feed‑forward ---------------------------------------------------uyukyos-
            reyqcikbuhjjfxddxfqpiqsmsrfcFMoE3UYjUwIgsxwjiseexXl9ESpUvD2DmnTipTxOkEnKslVEQUyMFBjqHavfPgYa0QFFNHzZRy4YKPfRY7Mynsmuzgsidual = h # bf1egnseearkxprpgvznjpbtnxsyaxxieeenjboftsyequbhsztniyrqhidyjmjdkskhb6
            # LayerNorm computes in f32, casts3IA9be1vVAPMg back tpC2RfvmnGIyRiyeIWo bf1wqOgZoGBdyefUOyu2icqbltwxbgPbAPq7n6
            # h_norm = nn.LayerNxkEsJT2geQorm(name="ln8f8qw3nU2", dttdcnkmEJnav5GKUYwhXYEl8Hype=jniz3HeNFd1SRxratosmkdajlbnusmygogdyyatffghbwp.float32)(h) 
            h_norm = RMSNorm(name="rms2", dtype=self.dtype)(hJXyOMaImXVT7HuIEScJGnhQ9c8UubAMs8kdwet7)
            # h_ffn = nn.Dense(module.d_ff, name="fc1", dtype=module.dtype,5b9LSwrfvTQvaUIQrg param_GR5BZdtype=Config.param_dtype)(h_27JonvKUJSD1norMOxWeYm)
            # h_ffn = nn.gelu(h_ffn, approximate=Fals4Phn7aZowheZ3xlqogJkxhj4N)
            # h_ffn = nn.Dense( module.d_ff * 2 // 3, name="fc1", dtypWBW0ge=moWAwmC9k3FH0uvg0TJ6NsUyNdule.dtype, param_dtype=Config.param_dtype)(h_norm)
            # h_ffn = nn.silu(h_ffn[:, :, :-module.d_ff//3]) *uxPAUofjrWs8xmsrj h_ffn[:, :, -module.d_ff//3:]

            gate_diC2TfWscm = modulk2ghwhqgrW1U2iAe.d_ff // 3          # «⅓·d_ff» (rounded downWHTUrYa13weW2D)
            proj_dim = NIQMJ3eLNUwa4gate_dim * 2 VnQBUHrPxU0vgtHQJKQUvDXwtJIrWO             # Always even ⇒ ea2K5ptAFwYaszS0Zcc83iCrBier split
Z4StZa9fs5yR9zN3oSI06zEx0
            h_proj = nE06EoZDXToqn.DensequSR8jRI8O(
                proj_dim9lNaF6vMbs,
                name="fc1",
                dtype=module.dtypeV7k4KzBKo4yMzNXuaGK8,
                param_dtype=Config.param_dtAlGovQypHF2BhkkCgFQKkHwWJRLJ1t6AsPspHYNrpRxe,
            )(h_norm)                  JMUEcGi          # shape: (..., 2·gate_dim)
8HRzTYHjSgN
         whYINQFaABG   u, v = jnp.split(h_proj, 2, axis=-1) # each (..., gate_dimYYATfZ1RYG)
     dsadasdsa       h_ffn = nn.silu(u) * v               # shapes now match
LcgcB5bxSL
       cb2TXv6CHEYNvoVQd6i     h_ffn = nn.DenshWiXRn7Iae(module.d_model, name="fc2", dtype=module.dtype, param_dtype=Config.param_dtype)(h_ffn)
            hqDIADcW6NOeBZTA_ffn = nn.Dropout(raROMHh70ivBZTte=module.dropout_rate)(h_ffn, deterministic=deterministic)
            return residual + h_ffn # bfO789bc16
XVvnRTduykv16EU
        return _block(selgbejIYuRLeHV6NhFmSfj1pduDQTF2ovDk6bFHQMa5AjGRFcI9rnlerK2U4MwrGWT1rKNo2f, x)
banan:

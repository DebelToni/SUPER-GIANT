from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
Config = OmegaConf.load("Config.yml")

# -----------------------------------------------------------------------------
# Utilities: RMSNorm and RoPE (partial rotary on the first rotary_dim dims)
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Flax-compatible).

    Important: cast the scale to the activation dtype to avoid upcasting to fp32
    during autocast (prevents q/k/v dtype mismatches).
    """

    def __init__(self, dim: int, eps: float = 1e-6, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.eps = eps
        # Keep parameter in higher precision by default (often float32)
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # Cast scale to activation dtype so output keeps x.dtype
        scale = self.weight.to(dtype=x.dtype)
        return norm_x * scale


def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.reshape(x.shape)


def _build_sin_cos(
    positions: torch.Tensor, rotary_dim: int, *, device: torch.device, dtype: torch.dtype
):
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim)
    )
    freqs = torch.outer(positions.to(dtype), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    sin = emb.sin()[None, None, :, :]
    cos = emb.cos()[None, None, :, :]
    return sin, cos


def _apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    x_rot_2 = _rotate_every_two(x_rot)
    x_rotated = x_rot * cos + x_rot_2 * sin
    return torch.cat([x_rotated, x_pass], dim=-1)


# -----------------------------------------------------------------------------
# Multi-Query Self-Attention with optional KV cache (decode-time)
# -----------------------------------------------------------------------------

class NativeTorchSelfAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        qkv_features: int,
        dropout_rate: float = 0.0,
        num_kv: int = 1,
        dtype: Optional[torch.dtype] = None,
        rotary_dim: Optional[int] = None,
    ):
        super().__init__()
        assert qkv_features % num_heads == 0, "qkv_features must be divisible by num_heads"
        self.num_heads = num_heads
        self.qkv_features = qkv_features
        self.dropout_rate = dropout_rate
        self.num_kv = num_kv
        self.head_dim = qkv_features // num_heads
        self.dtype = dtype
        self.rotary_dim = int(rotary_dim if rotary_dim is not None else int(Config.rope_dim))
        assert self.rotary_dim <= self.head_dim and self.rotary_dim % 2 == 0

        self.q_proj = nn.Linear(qkv_features, qkv_features, bias=False)
        self.k_proj = nn.Linear(qkv_features, self.num_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(qkv_features, self.num_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(qkv_features, qkv_features, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv == self.num_heads:
            return x
        assert self.num_heads % self.num_kv == 0
        repeat = self.num_heads // self.num_kv
        return x.repeat_interleave(repeat, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_kv_cache: bool = False,
        cur_index: Optional[int] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        B, L, D = x.shape
        device = x.device

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,Dh)
        k = self.k_proj(x).view(B, L, self.num_kv, self.head_dim).transpose(1, 2)  # (B,K,L,Dh)
        v = self.v_proj(x).view(B, L, self.num_kv, self.head_dim).transpose(1, 2)  # (B,K,L,Dh)
        k = self._repeat_kv(k)  # (B,H,L,Dh)
        v = self._repeat_kv(v)  # (B,H,L,Dh)

        # Ensure dtype consistency (avoid float vs half mismatches)
        if use_kv_cache and kv_cache is not None:
            target_dtype = kv_cache['k'].dtype
        else:
            target_dtype = q.dtype
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)

        if use_kv_cache:
            assert kv_cache is not None and cur_index is not None
            pos = torch.arange(cur_index, cur_index + L, device=device)
            sin, cos = _build_sin_cos(pos, self.rotary_dim, device=device, dtype=target_dtype)
            q = _apply_rope(q, sin, cos, self.rotary_dim)
            k = _apply_rope(k, sin, cos, self.rotary_dim)

            kv_cache['k'][:, :, cur_index : cur_index + L, :] = k
            kv_cache['v'][:, :, cur_index : cur_index + L, :] = v

            key_len = cur_index + L
            k_used = kv_cache['k'][:, :, :key_len, :]
            v_used = kv_cache['v'][:, :, :key_len, :]

            p = 0.0 if (deterministic or not self.training) else self.dropout_rate
            y = F.scaled_dot_product_attention(q, k_used, v_used, dropout_p=p, is_causal=False)
        else:
            pos = torch.arange(L, device=device)
            sin, cos = _build_sin_cos(pos, self.rotary_dim, device=device, dtype=target_dtype)
            q = _apply_rope(q, sin, cos, self.rotary_dim)
            k = _apply_rope(k, sin, cos, self.rotary_dim)

            p = 0.0 if (deterministic or not self.training) else self.dropout_rate
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, L, D)
        # Ensure projection input matches layer weight dtype to avoid sync/cast stalls
        y = y.to(x.dtype)
        y = self.o_proj(y)
        return y


# -----------------------------------------------------------------------------
# Transformer Block (RMSNorm -> MHA -> residual -> RMSNorm -> Gated FFN -> residual)
# -----------------------------------------------------------------------------

class TinyTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout_rate: float = 0.0,
        num_kv: int = 1,
        dtype: Optional[torch.dtype] = None,
        rotary_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.dtype = dtype

        self.rms1 = RMSNorm(d_model, dtype=dtype)
        self.attn = NativeTorchSelfAttention(
            num_heads=n_heads,
            qkv_features=d_model,
            dropout_rate=dropout_rate,
            num_kv=num_kv,
            dtype=dtype,
            rotary_dim=rotary_dim if rotary_dim is not None else int(Config.rope_dim),
        )
        self.dropout = nn.Dropout(dropout_rate)

        self.rms2 = RMSNorm(d_model, dtype=dtype)
        gate_dim = d_ff // 3
        proj_dim = gate_dim * 2
        self.fc1 = nn.Linear(d_model, proj_dim, bias=True)
        self.fc2 = nn.Linear(gate_dim, d_model, bias=True)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        *,
        deterministic: bool = False,
        use_kv_cache: bool = False,
        cur_index: Optional[int] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.rms1(x)
        h = self.attn(
            x_norm,
            kv_cache=kv_cache,
            use_kv_cache=use_kv_cache,
            cur_index=cur_index,
            deterministic=deterministic,
        )
        h = self.dropout(h) if (self.training and not deterministic) else h
        h = residual + h

        residual = h
        h_norm = self.rms2(h)
        h_proj = self.fc1(h_norm)
        u, v = torch.chunk(h_proj, 2, dim=-1)
        h_ffn = self.act(u) * v
        h_ffn = self.fc2(h_ffn)
        h_ffn = self.dropout(h_ffn) if (self.training and not deterministic) else h_ffn
        return residual + h_ffn

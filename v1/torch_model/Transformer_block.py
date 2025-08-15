from __future__ import annotations

from typing import Optional, Dict

import math
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

    Matches Flax's RMSNorm defaults:
      - no bias
      - learnable scale (weight)
      - epsilon = 1e-6
      - normalization over the last dimension only
    """

    def __init__(self, dim: int, eps: float = 1e-6, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight


def _rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    """Helper for RoPE rotation: [-x2, x1] over last dim grouped by pairs."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    # Interleave [-x2, x1]
    out_even = -x2
    out_odd = x1
    out = torch.stack((out_even, out_odd), dim=-1)
    out = out.reshape(x.shape)
    return out


def _build_sin_cos(
    positions: torch.Tensor, rotary_dim: int, *, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sin/cos tables for given positions (shape: [L]) and rotary_dim.

    Returns sin, cos with shape (1, 1, L, rotary_dim) for broadcasting over (B, H, L, Dh).
    """
    assert rotary_dim % 2 == 0, "rotary_dim must be even"
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim)
    )  # (rotary_dim/2)
    # positions: (L,)
    freqs = torch.outer(positions.to(dtype), inv_freq)  # (L, rotary_dim/2)
    # Repeat to full rotary_dim by interleaving even/odd
    emb = torch.cat([freqs, freqs], dim=-1)  # (L, rotary_dim)
    sin = emb.sin()[None, None, :, :]  # (1, 1, L, rotary_dim)
    cos = emb.cos()[None, None, :, :]  # (1, 1, L, rotary_dim)
    return sin, cos


def _apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """Apply rotary embeddings to the first rotary_dim dims of x.

    x:   (B, H, L, Dh)
    sin: (1, 1, L, rotary_dim)
    cos: (1, 1, L, rotary_dim)
    """
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    # rotate pairs
    x_rot_2 = _rotate_every_two(x_rot)
    x_rotated = x_rot * cos + x_rot_2 * sin
    return torch.cat([x_rotated, x_pass], dim=-1)


# -----------------------------------------------------------------------------
# Multi-Query Self-Attention with optional KV cache (decode-time)
# -----------------------------------------------------------------------------

class NativeTorchSelfAttention(nn.Module):
    """Multi-head self-attention using torch.nn.functional.scaled_dot_product_attention.

    Mirrors the structure of the original Flax module:
      - q_proj, k_proj, v_proj, o_proj (no bias for projections)
      - Optional multi-query: num_kv <= num_heads (K/V shared then repeated)
      - Partial RoPE on the first `rotary_dim` dims
      - Optional KV cache for fast autoregressive decoding
    """

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
        assert self.rotary_dim <= self.head_dim, "rotary_dim must be <= head_dim"
        assert self.rotary_dim % 2 == 0, "rotary_dim must be even"

        # Projections (bias=False to match typical transformer attention and your JAX code)
        self.q_proj = nn.Linear(qkv_features, qkv_features, bias=False)
        self.k_proj = nn.Linear(qkv_features, self.num_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(qkv_features, self.num_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(qkv_features, qkv_features, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat K/V heads from num_kv to num_heads.

        Input x shape: (B, num_kv, L, Dh)
        Output shape:  (B, num_heads, L, Dh)
        """
        if self.num_kv == self.num_heads:
            return x
        assert self.num_heads % self.num_kv == 0, "num_heads must be a multiple of num_kv"
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
        """x: (B, L, D). Returns: (B, L, D).

        If `use_kv_cache` is True, `kv_cache` must be a dict with keys 'k' and 'v'
        of shapes (B, num_heads, T, Dh). This function *updates it in-place*.
        """
        B, L, D = x.shape
        device = x.device
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,L,Dh)

        # K/V with possibly fewer heads (multi-query)
        k = self.k_proj(x).view(B, L, self.num_kv, self.head_dim).transpose(1, 2)  # (B,K,L,Dh)
        v = self.v_proj(x).view(B, L, self.num_kv, self.head_dim).transpose(1, 2)  # (B,K,L,Dh)
        k = self._repeat_kv(k)  # -> (B,H,L,Dh)
        v = self._repeat_kv(v)  # -> (B,H,L,Dh)

        # RoPE on first rotary_dim dims
        if use_kv_cache:
            assert kv_cache is not None and cur_index is not None, "cache and cur_index required"
            # Positions for the *new* tokens only (usually L==1)
            pos = torch.arange(cur_index, cur_index + L, device=device)
            sin, cos = _build_sin_cos(pos, self.rotary_dim, device=device, dtype=q.dtype)
            q = _apply_rope(q, sin, cos, self.rotary_dim)
            k = _apply_rope(k, sin, cos, self.rotary_dim)

            # Write into cache at [cur_index : cur_index + L)
            # Expecting cache shapes: (B, H, T, Dh)
            assert 'k' in kv_cache and 'v' in kv_cache, "kv_cache must have 'k' and 'v'"
            kv_cache['k'][:, :, cur_index : cur_index + L, :] = k
            kv_cache['v'][:, :, cur_index : cur_index + L, :] = v

            # Attend to all keys up to the current position (inclusive)
            key_len = cur_index + L
            k_used = kv_cache['k'][:, :, :key_len, :]
            v_used = kv_cache['v'][:, :, :key_len, :]

            # SDPA: not causal because we already truncated keys to past-only
            p = 0.0 if (deterministic or not self.training) else self.dropout_rate
            y = F.scaled_dot_product_attention(q, k_used, v_used, dropout_p=p, is_causal=False)
        else:
            # Full prompt prefill: positions 0..L-1
            pos = torch.arange(L, device=device)
            sin, cos = _build_sin_cos(pos, self.rotary_dim, device=device, dtype=q.dtype)
            q = _apply_rope(q, sin, cos, self.rotary_dim)
            k = _apply_rope(k, sin, cos, self.rotary_dim)

            p = 0.0 if (deterministic or not self.training) else self.dropout_rate
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)

        # Merge heads and project out
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        y = self.o_proj(y)
        y = self.dropout(y) if (self.training and not deterministic) else y
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
        # Flax Dense defaults to use_bias=True unless explicitly disabled in your attention.
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
        # Attention sub-layer
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

        # FFN sub-layer (gated)
        residual = h
        h_norm = self.rms2(h)
        h_proj = self.fc1(h_norm)
        u, v = torch.chunk(h_proj, 2, dim=-1)
        h_ffn = self.act(u) * v
        h_ffn = self.fc2(h_ffn)
        h_ffn = self.dropout(h_ffn) if (self.training and not deterministic) else h_ffn
        return residual + h_ffn


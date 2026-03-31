# pyright: reportMissingImports=false
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNormNoWeight(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).to(x.dtype)


class AttnResOperator(nn.Module):
    """Depth-wise attention residual operator.

    sources: [N_src, B, T, D]
    output:  [B, T, D]
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))
        self.key_norm = RMSNormNoWeight(eps=eps)

    def forward(self, sources: Tensor) -> Tensor:
        keys = self.key_norm(sources)
        logits = torch.einsum("d,n b t d->n b t", self.pseudo_query, keys)
        weights = F.softmax(logits, dim=0)
        out = torch.einsum("n b t,n b t d->b t d", weights, sources)
        return out


def two_phase_block_attnres_inference(
    pseudo_queries: Tensor,
    block_reps: Tensor,
    key_norm: RMSNormNoWeight,
) -> tuple[Tensor, Tensor, Tensor]:
    """Batched inter-block pass that returns softmax stats for online merge."""
    keys = key_norm(block_reps)  # [N, B, T, D]
    logits = torch.einsum("s d,n b t d->s n b t", pseudo_queries, keys)

    maxes = logits.max(dim=1).values
    shifted = logits - maxes.unsqueeze(1)
    exp_shifted = shifted.exp()
    lse = exp_shifted.sum(dim=1)
    outputs = torch.einsum("s n b t,n b t d->s b t d", exp_shifted, block_reps)
    return outputs, maxes, lse


def online_softmax_merge(
    o1: Tensor,
    m1: Tensor,
    l1: Tensor,
    o2: Tensor,
    m2: Tensor,
    l2: Tensor,
) -> Tensor:
    m = torch.maximum(m1, m2)
    exp1 = (m1 - m).exp().unsqueeze(-1)
    exp2 = (m2 - m).exp().unsqueeze(-1)
    l1_adj = (m1 - m).exp() * l1
    l2_adj = (m2 - m).exp() * l2
    denom = (l1_adj + l2_adj).unsqueeze(-1)
    return (exp1 * o1 + exp2 * o2) / denom

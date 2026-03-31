# pyright: reportMissingImports=false
from __future__ import annotations

import torch
from torch import Tensor


def forward_block_attnres(
    layers,
    token_embedding: Tensor,
    block_size: int,
    rope_freqs: Tensor,
    causal_mask: Tensor,
) -> Tensor:
    """Minimal block AttnRes flow extracted from the reference implementation.

    layers: iterable of modules with signature layer(sources, rope_freqs, mask)
    token_embedding: [B, T, D]
    """
    blocks: list[Tensor] = [token_embedding]
    partial_block: Tensor | None = None

    for i, layer in enumerate(layers):
        source_list = blocks + ([partial_block] if partial_block is not None else [])
        sources = torch.stack(source_list, dim=0)
        output = layer(sources, rope_freqs, causal_mask)

        # Boundary handling from the reference flow.
        if i > 0 and i % block_size == 0 and partial_block is not None:
            blocks.append(partial_block)
            partial_block = None

        partial_block = output if partial_block is None else partial_block + output

    final_sources = torch.stack(blocks + ([partial_block] if partial_block is not None else []), dim=0)
    return final_sources.sum(dim=0)

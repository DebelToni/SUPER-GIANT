from __future__ import annotations

from typing import Optional, List, Dict

import torch
import torch.nn as nn

from omegaconf import OmegaConf
from Transformer_block import TinyTransformerBlock

# -----------------------------------------------------------------------------
# Config & dtype helpers
# -----------------------------------------------------------------------------

Config = OmegaConf.load("Config.yml")

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def _get_dtype(name: Optional[str], default: torch.dtype) -> torch.dtype:
    if name is None:
        return default
    return _DTYPE_MAP.get(str(name).lower(), default)


# -----------------------------------------------------------------------------
# GiantGPT model wrapper (PyTorch)
# -----------------------------------------------------------------------------

class GiantGPT(nn.Module):
    """Transformer LM with tied output projection to the token embedding.

    This is a 1:1 PyTorch port of your Flax model:
      - token embedding
      - N repeated TinyTransformerBlock blocks
      - logits computed via einsum with the *same* embedding weights (weight tying)
      - optional KV cache per layer for fast decoding
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int = int(Config.d_model),
        n_layers: int = int(Config.n_layers),
        n_heads: int = int(Config.n_heads),
        d_ff: int = int(Config.d_ff),
        dropout: float = float(Config.dropout),
        num_kv: int = int(getattr(Config, "num_kv", 1)),
        rotary_dim: int = int(getattr(Config, "rope_dim", max(2, int(int(Config.d_model) // int(Config.n_heads))))),
        param_dtype: Optional[str] = getattr(Config, "param_dtype", "float32"),
        compute_dtype: Optional[str] = getattr(Config, "compute_dtype", "bfloat16"),
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.d_ff = int(d_ff)
        self.dropout_rate = float(dropout)
        self.num_kv = int(num_kv)
        self.rotary_dim = int(rotary_dim)

        # dtype policy (params vs compute). Parameters typically live in float32, compute in bf16.
        self.param_dtype = _get_dtype(param_dtype, torch.float32)
        self.compute_dtype = _get_dtype(compute_dtype, torch.bfloat16)

        # Token embedding (tied to output projection)
        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.drop = nn.Dropout(self.dropout_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TinyTransformerBlock(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_ff=self.d_ff,
                    dropout_rate=self.dropout_rate,
                    num_kv=self.num_kv,
                    dtype=self.param_dtype,
                    rotary_dim=self.rotary_dim,
                )
                for _ in range(self.n_layers)
            ]
        )

        # Parameter init stays default; you'll load real weights via a converter.

    # --------------------------- convenience utilities ---------------------------
    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def init_kv_cache(
        self,
        *,
        batch_size: int,
        max_seq_len: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Allocate per-layer KV caches matching the attention expectations.

        Shapes per layer: (B, n_heads, T, head_dim). Uses compute_dtype by default.
        Returns a list of dicts: [{"k": ..., "v": ...}, ...] with length n_layers.
        """
        if device is None:
            device = self.embed.weight.device
        if dtype is None:
            dtype = self.compute_dtype
        B, H, T, Dh = int(batch_size), self.n_heads, int(max_seq_len), self.head_dim
        caches: List[Dict[str, torch.Tensor]] = []
        for _ in range(self.n_layers):
            caches.append(
                {
                    "k": torch.zeros(B, H, T, Dh, device=device, dtype=dtype),
                    "v": torch.zeros(B, H, T, Dh, device=device, dtype=dtype),
                }
            )
        return caches

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # --------------------------------- forward ----------------------------------
    def forward(
        self,
        tokens: torch.Tensor,  # (B, L) int64
        *,
        deterministic: bool = False,
        use_kv_cache: bool = False,
        cur_index: Optional[int] = None,
        kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Return logits over the vocabulary with tied output weights.

        Args:
          tokens: (B, L) token IDs (torch.long)
          deterministic: disables dropout when True (in addition to .eval())
          use_kv_cache: enable fast decode path; requires kv_caches (list per layer)
          cur_index: index in the cache to write the current tokens (decode)
          kv_caches: list of dicts with 'k' and 'v' tensors per layer
        Returns:
          logits: (B, L, vocab_size)
        """
        if tokens.dtype != torch.long:
            tokens = tokens.long()

        x = self.embed(tokens)  # (B, L, D)
        x = self.drop(x) if (self.training and not deterministic) else x

        for i, block in enumerate(self.blocks):
            cache_i = None
            if use_kv_cache:
                if kv_caches is None:
                    raise ValueError("use_kv_cache=True requires kv_caches list")
                cache_i = kv_caches[i]
            x = block(
                x,
                deterministic=deterministic,
                use_kv_cache=use_kv_cache,
                cur_index=cur_index,
                kv_cache=cache_i,
            )

        # Weight-tying to embedding matrix; logits computed in float32 for parity
        logits = torch.einsum("bld,vd->blv", x.float(), self.embed.weight.float())
        return logits


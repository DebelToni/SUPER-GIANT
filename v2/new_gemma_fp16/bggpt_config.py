# bggpt_config.py

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class BgGPTConfig:
    """
    Minimal config for BgGPT-Gemma-2-2.6B-IT (Gemma 2 2B architecture).

    Numbers are taken from publicly visible Gemma 2 2B configs:
      - hidden_size: 2304
      - intermediate_size: 9216
      - num_hidden_layers: 26
      - num_attention_heads: 8
      - num_key_value_heads: 4
      - head_dim: 256
      - max_position_embeddings: 8192
      - sliding_window: 4096
      - rope_theta: 10000.0
      - rms_norm_eps: 1e-6
      - query_pre_attn_scalar: 256.0
    """

    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256

    max_position_embeddings: int = 8192
    sliding_window: int = 4096  # not used in current implementation
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    query_pre_attn_scalar: float = 256.0  # currently unused

    vocab_size: Optional[int] = None

    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1

    def with_vocab(self, vocab_size: int) -> "BgGPTConfig":
        return replace(self, vocab_size=int(vocab_size))


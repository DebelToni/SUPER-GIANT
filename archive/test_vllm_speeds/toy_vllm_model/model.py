from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Set

import torch
import torch.nn as nn

from vllm.attention.layer import Attention
from vllm.config import VllmConfig


@dataclass(frozen=True)
class ToyHParams:
    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    layer_norm_eps: float = 1e-5


class ToyDecoderLayer(nn.Module):
    def __init__(self, hp: ToyHParams, *, vllm_config: VllmConfig, layer_idx: int, prefix: str):
        super().__init__()
        assert hp.hidden_size % hp.num_attention_heads == 0
        head_dim = hp.hidden_size // hp.num_attention_heads
        scale = head_dim ** -0.5

        self.ln_1 = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.qkv = nn.Linear(hp.hidden_size, 3 * hp.hidden_size, bias=False)

        # vLLM attention owns KV cache and uses forward_context metadata.
        self.attn = Attention(
            num_heads=hp.num_attention_heads,
            head_size=head_dim,
            scale=scale,
            num_kv_heads=hp.num_attention_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.layers.{layer_idx}.attn",
        )
        self.out_proj = nn.Linear(hp.hidden_size, hp.hidden_size, bias=False)

        self.ln_2 = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)
        self.fc1 = nn.Linear(hp.hidden_size, hp.intermediate_size, bias=False)
        self.fc2 = nn.Linear(hp.intermediate_size, hp.hidden_size, bias=False)

        # Swap this to whatever you want (SiLU/GELU/ReLU/etc.).
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        h = self.ln_1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        a = self.attn(q, k, v)
        x = x + self.out_proj(a)

        # MLP block
        h = self.ln_2(x)
        h = self.fc2(self.act(self.fc1(h)))
        x = x + h
        return x


class ToyDecoderModel(nn.Module):
    def __init__(self, hp: ToyHParams, *, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        self.hp = hp
        self.tok_emb = nn.Embedding(hp.vocab_size, hp.hidden_size)
        self.pos_emb = nn.Embedding(hp.max_position_embeddings, hp.hidden_size)

        self.layers = nn.ModuleList(
            [
                ToyDecoderLayer(hp, vllm_config=vllm_config, layer_idx=i, prefix=prefix)
                for i in range(hp.num_hidden_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(hp.hidden_size, eps=hp.layer_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(input_ids)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # vLLM commonly flattens tokens: [num_tokens] (no batch dim).
        x = self.embed_input_ids(input_ids) + self.pos_emb(positions)
        for layer in self.layers:
            x = layer(x)
        return self.ln_f(x)


class ToyDecoderForCausalLM(nn.Module):
    """
    vLLM "generative model" style:
      - embed_input_ids(input_ids) -> embeddings
      - forward(input_ids, positions) -> final hidden states
      - compute_logits(hidden_states) -> logits
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "", **kwargs):
        super().__init__()
        hf_cfg = vllm_config.model_config.hf_config

        # We deliberately store our checkpoint in a GPT-2-shaped config.json
        # so transformers can parse it without a custom AutoConfig class.
        hp = ToyHParams(
            vocab_size=int(hf_cfg.vocab_size),
            max_position_embeddings=int(
                getattr(hf_cfg, "n_positions", getattr(hf_cfg, "max_position_embeddings", 2048))
            ),
            hidden_size=int(getattr(hf_cfg, "n_embd", hf_cfg.hidden_size)),
            num_hidden_layers=int(getattr(hf_cfg, "n_layer", hf_cfg.num_hidden_layers)),
            num_attention_heads=int(getattr(hf_cfg, "n_head", hf_cfg.num_attention_heads)),
            intermediate_size=int(getattr(hf_cfg, "n_inner", 4 * getattr(hf_cfg, "n_embd", hf_cfg.hidden_size))),
            layer_norm_eps=float(getattr(hf_cfg, "layer_norm_epsilon", 1e-5)),
        )

        self.model = ToyDecoderModel(hp, vllm_config=vllm_config, prefix=prefix)

        # Tie LM head to token embedding like GPT-2.
        self.lm_head = nn.Linear(hp.hidden_size, hp.vocab_size, bias=False)
        self.lm_head.weight = self.model.tok_emb.weight

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        return self.model(input_ids, positions, **kwargs)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """
        vLLM expects models to implement a weight-loading hook for checkpoints.
        This simplistic loader matches checkpoint tensor names to parameter names.
        """
        params = dict(self.named_parameters())
        loaded: Set[str] = set()

        for name, tensor in weights:
            if name not in params:
                continue
            p = params[name]
            if p.shape != tensor.shape:
                raise ValueError(f"Shape mismatch for {name}: {p.shape} vs {tensor.shape}")
            p.data.copy_(tensor)
            loaded.add(name)

        return loaded

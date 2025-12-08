# bggpt_gguf_loader.py

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from huggingface_hub import hf_hub_download
from gguf.gguf_reader import GGUFReader, ReaderField

from bggpt_config import BgGPTConfig


HF_GGUF_REPO = "INSAIT-Institute/BgGPT-Gemma-2-2.6B-IT-v1.0-GGUF"
GGUF_FILENAME = "BgGPT-Gemma-2-2B-IT-v1.0.F16.gguf"


PyTree = Dict[str, Any]


def _reader_field_to_str(field: ReaderField) -> str:
    """Decode a scalar string metadata field."""
    arr = field.parts[field.data[0]]
    return bytes(arr).decode("utf-8")


def _reader_field_to_int(field: ReaderField) -> int:
    arr = field.parts[field.data[0]]
    return int(arr[0])


def download_gguf_model(local_dir: str = "models") -> str:
    """
    Download the FP16 GGUF file for BgGPT from Hugging Face if not present.

    Returns:
        Local filesystem path to the GGUF file.
    """
    os.makedirs(local_dir, exist_ok=True)

    # huggingface_hub will handle caching under its own cache dir; we just
    # ensure we know where the file is.
    path = hf_hub_download(
        repo_id=HF_GGUF_REPO,
        filename=GGUF_FILENAME,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    return path


def _build_name_to_tensor(reader: GGUFReader) -> Dict[str, Any]:
    name_to_tensor = {}
    for t in reader.tensors:
        # t.data is a numpy array with already-correct shape & dtype
        name_to_tensor[t.name] = t
    return name_to_tensor


def _infer_vocab_and_hidden(name_to_tensor: Dict[str, Any]) -> Tuple[int, int]:
    """
    Use the token embedding weight to infer vocab size and hidden size.

    Gemma / Gemma2 GGUF uses:
        token_embd.weight: shape (vocab_size, hidden_size)
    """
    token_emb = name_to_tensor.get("token_embd.weight")
    if token_emb is None:
        raise KeyError("GGUF is missing 'token_embd.weight' tensor")

    data = token_emb.data
    if data.ndim != 2:
        raise ValueError(
            f"token_embd.weight should be 2D, got shape={data.shape}"
        )
    vocab_size, hidden_size = data.shape
    return int(vocab_size), int(hidden_size)


def load_bggpt_params(
    gguf_path: str,
    config: BgGPTConfig | None = None,
    dtype=jnp.float16,
    device: jax.Device | None = None,
) -> Tuple[BgGPTConfig, PyTree]:
    """
    Load BgGPT Gemma-2 2.6B weights from GGUF and return (config, params).

    Args:
        gguf_path: path to .gguf file (FP16).
        config: if given, used as base; otherwise BgGPTConfig() with inferred vocab.
        dtype: JAX dtype to use for weights (default: float16).
        device: optional JAX device to place params on.

    Returns:
        (config, params) where params is a JAX PyTree:
            {
              "tok_embeddings": (vocab, hidden),
              "layers": [
                  {
                    "attn_norm": (hidden,),
                    "ffn_norm": (hidden,),
                    "wq": (q_out, hidden),
                    "wk": (kv_out, hidden),
                    "wv": (kv_out, hidden),
                    "wo": (hidden, q_out),
                    "w_gate": (ff_dim, hidden),
                    "w_up": (ff_dim, hidden),
                    "w_down": (hidden, ff_dim),
                  }, ...
              ],
              "final_norm": (hidden,),
              "lm_head": (vocab, hidden) or (hidden, vocab) depending on tie
            }
    """
    reader = GGUFReader(gguf_path)
    name_to_tensor = _build_name_to_tensor(reader)

    vocab_size, hidden_size = _infer_vocab_and_hidden(name_to_tensor)

    if config is None:
        config = BgGPTConfig().with_vocab(vocab_size)
    else:
        if config.vocab_size is None:
            config = config.with_vocab(vocab_size)
        elif config.vocab_size != vocab_size:
            raise ValueError(
                f"Config vocab_size={config.vocab_size} does not match "
                f"GGUF embedding vocab_size={vocab_size}"
            )

    if hidden_size != config.hidden_size:
        # This would indicate a mismatch against Gemma-2 2B assumptions.
        raise ValueError(
            f"Hidden size from GGUF ({hidden_size}) != config.hidden_size "
            f"({config.hidden_size}). Update BgGPTConfig if needed."
        )

    # Build params tree
    params: PyTree = {}

    # Token embedding
    tok_emb_np = name_to_tensor["token_embd.weight"].data.astype(
        np.float16 if dtype == jnp.float16 else np.float32
    )
    tok_embeddings = jnp.array(tok_emb_np, dtype=dtype)
    if device is not None:
        tok_embeddings = jax.device_put(tok_embeddings, device=device)
    params["tok_embeddings"] = tok_embeddings

    # Transformer blocks
    layers: List[Dict[str, jnp.ndarray]] = []
    L = config.num_hidden_layers

    def to_jax(name: str) -> jnp.ndarray:
        t = name_to_tensor.get(name)
        if t is None:
            raise KeyError(f"Missing tensor '{name}' in GGUF file")
        arr = t.data.astype(
            np.float16 if dtype == jnp.float16 else np.float32
        )
        x = jnp.array(arr, dtype=dtype)
        if device is not None:
            x = jax.device_put(x, device=device)
        return x

    for layer_idx in range(L):
        prefix = f"blk.{layer_idx}"
        layer: Dict[str, jnp.ndarray] = {}

        layer["attn_norm"] = to_jax(f"{prefix}.attn_norm.weight")
        layer["ffn_norm"] = to_jax(f"{prefix}.ffn_norm.weight")

        layer["wq"] = to_jax(f"{prefix}.attn_q.weight")
        layer["wk"] = to_jax(f"{prefix}.attn_k.weight")
        layer["wv"] = to_jax(f"{prefix}.attn_v.weight")
        layer["wo"] = to_jax(f"{prefix}.attn_output.weight")

        layer["w_gate"] = to_jax(f"{prefix}.ffn_gate.weight")
        layer["w_up"] = to_jax(f"{prefix}.ffn_up.weight")
        layer["w_down"] = to_jax(f"{prefix}.ffn_down.weight")

        layers.append(layer)

    params["layers"] = tuple(layers)

    # Final norm
    params["final_norm"] = to_jax("output_norm.weight")

    # LM head: sometimes present as "output.weight"; if missing, tie to embeddings
    lm_head_tensor = name_to_tensor.get("output.weight")
    if lm_head_tensor is not None:
        lm_np = lm_head_tensor.data.astype(
            np.float16 if dtype == jnp.float16 else np.float32
        )
        lm_head = jnp.array(lm_np, dtype=dtype)
        if device is not None:
            lm_head = jax.device_put(lm_head, device=device)
    else:
        # Tie weights (standard for Gemma): use embedding matrix as LM head.
        lm_head = params["tok_embeddings"]

    params["lm_head"] = lm_head

    return config, params


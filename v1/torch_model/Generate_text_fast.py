from __future__ import annotations

import argparse
import time
from typing import Optional, List

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from GiantGPT import GiantGPT
from checkpoint_torch_io import load_any_checkpoint

# -----------------------------------------------------------------------------
# Config & dtype helpers
# -----------------------------------------------------------------------------

DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def get_dtype(name: Optional[str], default: torch.dtype) -> torch.dtype:
    if name is None:
        return default
    return DTYPE_MAP.get(str(name).lower(), default)


# -----------------------------------------------------------------------------
# Torch runtime setup
# -----------------------------------------------------------------------------

def setup_torch_runtime(cfg) -> torch.device:
    # Device
    device_str = str(getattr(cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)

    # Performance flags (TF32 like JAX default on Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    return device


# -----------------------------------------------------------------------------
# Tokenizer / Model builders
# -----------------------------------------------------------------------------

def build_tokenizer(cfg):
    tok_name = getattr(cfg, "tokenizer_name", None) or getattr(cfg, "tokenizer_path", None) or "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    if tokenizer.pad_token_id is None:
        # Ensure we have a pad token for easier batching; use eos as pad if missing
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(cfg, tokenizer, device: torch.device) -> GiantGPT:
    vocab_size = int(getattr(cfg, "vocab_size", tokenizer.vocab_size))

    model = GiantGPT(
        vocab_size=vocab_size,
        d_model=int(cfg.d_model),
        n_layers=int(cfg.n_layers),
        n_heads=int(cfg.n_heads),
        d_ff=int(cfg.d_ff),
        dropout=float(cfg.dropout),
        num_kv=int(getattr(cfg, "num_kv", 1)),
        rotary_dim=int(getattr(cfg, "rope_dim", max(2, int(cfg.d_model) // int(cfg.n_heads)))),
        param_dtype=str(getattr(cfg, "param_dtype", "float32")),
        compute_dtype=str(getattr(cfg, "compute_dtype", "bfloat16")),
    )

    # Move to device; keep params in param_dtype
    param_dtype = get_dtype(getattr(cfg, "param_dtype", "float32"), torch.float32)
    model = model.to(device=device, dtype=param_dtype)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Sampling utils
# -----------------------------------------------------------------------------

def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k)
    min_values = values[..., -1, None]
    return torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

def generate(
    model: GiantGPT,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    eos_token_id: Optional[int],
    device: torch.device,
    compute_dtype: torch.dtype,
):
    model.eval()

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    B, L = input_ids.shape

    # Allocate caches for fast decoding
    max_len = L + max_new_tokens
    caches = model.init_kv_cache(batch_size=B, max_seq_len=max_len, device=device, dtype=compute_dtype)

    # Prime the cache with all tokens except the last
    with torch.no_grad():
        if L > 1:
            for t in range(L - 1):
                cur = input_ids[:, t : t + 1]
                _ = model(
                    cur,
                    deterministic=True,
                    use_kv_cache=True,
                    cur_index=t,
                    kv_caches=caches,
                )

        # decoding loop
        generated: List[int] = []
        cur_token = input_ids[:, L - 1 : L]
        cur_index = L - 1

        # Timing just the decode portion
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        for _ in range(max_new_tokens):
            cur_index += 1
            # Autocast for compute dtype on CUDA
            ctx = (
                torch.cuda.amp.autocast(dtype=compute_dtype)
                if device.type == "cuda"
                else torch.autocast("cpu", dtype=compute_dtype, enabled=False)
            )
            with ctx:
                logits = model(
                    cur_token,
                    deterministic=True,
                    use_kv_cache=True,
                    cur_index=cur_index,
                    kv_caches=caches,
                )
                next_logits = logits[:, -1, :]
                if temperature == 0.0:
                    next_token = torch.argmax(next_logits, dim=-1)
                else:
                    next_logits = next_logits / temperature
                    if top_k is not None and top_k > 0:
                        next_logits = top_k_logits(next_logits, top_k)
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            generated.append(next_token.item())
            cur_token = next_token[:, None]

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

    total_new = len(generated)
    tok_per_s = total_new / max(t1 - t0, 1e-8)

    out_ids = torch.cat([input_ids[0], torch.tensor(generated, device=device, dtype=torch.long)])
    text = tokenizer.decode(out_ids.tolist(), skip_special_tokens=True)
    return text, total_new, tok_per_s


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GiantGPT PyTorch Inference")
    parser.add_argument("--config", type=str, default="Config.yml", help="Path to Config.yml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt/.pth or .npz)")
    parser.add_argument("--prompt", type=str, default="Hello", help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0, help="0 means disabled")
    parser.add_argument("--greedy", action="store_true", help="Shortcut for temperature=0.0")
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g., cpu or cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)
    if args.device is not None:
        cfg.device = args.device

    # Greedy shortcut
    if args.greedy:
        args.temperature = 0.0

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device & precision
    device = setup_torch_runtime(cfg)
    if args.device is not None:
        device = torch.device(args.device)

    compute_dtype = get_dtype(getattr(cfg, "compute_dtype", "bfloat16"), torch.bfloat16)

    # Tokenizer & model
    tokenizer = build_tokenizer(cfg)
    model = build_model(cfg, tokenizer, device)

    # Load weights (supports .pt/.pth and .npz via translator)
    load_any_checkpoint(model, args.checkpoint, device=device)

    # EOS behavior
    eos_id = getattr(tokenizer, "eos_token_id", None)

    # Generate
    text, n_new, tok_per_s = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_k=int(args.top_k) if int(args.top_k) > 0 else None,
        eos_token_id=eos_id,
        device=device,
        compute_dtype=compute_dtype,
    )

    # Output
    print("==== OUTPUT ====")
    print(text)
    print()
    print(f"Generated tokens: {n_new} | Speed: {tok_per_s:.2f} tok/s | Device: {device}")


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import time
from typing import Optional, List

import numpy as np
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from torch.amp import autocast

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
    
    # Prefer fast SDPA kernels when available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
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
    # IMPORTANT: use len(tokenizer) to include any added special tokens
    vocab_size = int(getattr(cfg, "vocab_size", len(tokenizer)))
    
    model = GiantGPT(
        vocab_size=vocab_size,
    )
    model.to(device)
    return model


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Apply top-k filtering to logits."""
    if k <= 0:
        return logits
    
    # Get the top k values and indices
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # Create a mask for the top k values
    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(-1, top_k_indices, top_k_values)
    
    return mask


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------

def generate(
    model: GiantGPT,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    device: torch.device = torch.device("cpu"),
    compute_dtype: torch.dtype = torch.bfloat16,
):
    model.eval()

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    B, L = input_ids.shape

    # Allocate caches for fast decoding
    max_len = L + max_new_tokens
    caches = model.init_kv_cache(batch_size=B, max_seq_len=max_len, device=device, dtype=compute_dtype)

    # Prime + decode under a single autocast & inference context
    with torch.inference_mode():
        with (autocast("cuda", dtype=compute_dtype) if device.type == "cuda"
              else torch.autocast("cpu", dtype=compute_dtype, enabled=False)):
            # Prefill cache with prompt (except last token)
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

            # decoding loop (keep tokens on device; no .item() inside the loop)
            gen = torch.empty(max_new_tokens, device=device, dtype=torch.long)
            cur_token = input_ids[:, L - 1 : L]
            cur_index = L - 1

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            for i in range(max_new_tokens):
                cur_index += 1
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

                # B==1: write to buffer; keep on device
                gen[i] = next_token
                cur_token = next_token[:, None]

                # Optional early stop: check EOS every 32 steps to amortize sync
                if eos_token_id is not None and ((i & 31) == 31):
                    if int(next_token.item()) == int(eos_token_id):
                        gen = gen[: i + 1]
                        break

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

    total_new = int(gen.shape[0])
    tok_per_s = total_new / max(t1 - t0, 1e-8)

    out_ids = torch.cat([input_ids[0], gen])
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

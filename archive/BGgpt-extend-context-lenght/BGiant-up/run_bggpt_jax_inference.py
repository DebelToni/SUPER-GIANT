import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from transformers import (
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForCausalLM,
    FlaxGemmaForCausalLM,
)

MODEL_DIR = Path("/Volumes/SSD/r2/BGiant/BgGPT-Gemma-2-2.6B-IT-v1.0")

def load_tokenizer():
    if not MODEL_DIR.exists():
        raise SystemExit(
            f"Model directory {MODEL_DIR} not found. "
            "Run download_bggpt_jax.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_default_system_prompt=False,
    )
    # Gemma uses left-padding for batched generation
    tokenizer.padding_side = "left"
    return tokenizer


def load_flax_model(dtype=jnp.bfloat16):
    """
    Load the BgGPT Gemma2 2.6B checkpoint as a Flax/JAX model.

    We use FlaxGemmaForCausalLM (or FlaxAutoModelForCausalLM) with from_pt=True
    to convert the PyTorch safetensors to JAX on the fly.
    """
    config = AutoConfig.from_pretrained(MODEL_DIR)

    # First try the explicit class (cleaner); if transformers version is old or
    # doesn't wire Gemma correctly, fall back to FlaxAutoModelForCausalLM.
    try:
        model, params = FlaxGemmaForCausalLM.from_pretrained(
            MODEL_DIR,
            config=config,
            from_pt=True,
            dtype=dtype,
            _do_init=False,  # do not re-init params, we load all from PT
        )
    except Exception as e:
        print(
            f"FlaxGemmaForCausalLM.from_pretrained failed ({e}). "
            "Falling back to FlaxAutoModelForCausalLM."
        )
        model, params = FlaxAutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            config=config,
            from_pt=True,
            dtype=dtype,
            _do_init=False,
        )

    return model, params


def build_inputs(tokenizer, prompt: str):
    """
    Build Gemma 2 chat-style inputs using the HF chat template.
    Returns a dict of NumPy arrays suitable for Flax (JAX).
    """
    messages = [
        {"role": "user", "content": prompt},
    ]

    # Gemma 2 chat template – returns np.int32 array
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="np",
    )  # shape: [1, L]

    # Build an attention mask (1 for real tokens, 0 for padding)
    pad_id = tokenizer.pad_token_id or 0
    attention_mask = (input_ids != pad_id).astype("i4")

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def generate_once(model, params, inputs, max_new_tokens: int):
    """
    Run one JAX generation and return (sequences, num_new_tokens, elapsed_sec).
    """
    # Recommended generation params from the BgGPT card :contentReference[oaicite:2]{index=2}
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        top_k=25,
        top_p=1.0,
        repetition_penalty=1.1,
        eos_token_id=[1, 107],  # Gemma2 eos & end-of-turn
        do_sample=True,
    )

    # Because JAX is async, we *must* block on the result before stopping timer.
    start = time.perf_counter()
    outputs = model.generate(
        **inputs,
        params=params,
        **gen_kwargs,
    )
    sequences = jax.device_get(jax.block_until_ready(outputs.sequences))
    end = time.perf_counter()

    elapsed = end - start
    num_new = sequences.shape[1] - inputs["input_ids"].shape[1]

    return sequences, int(num_new), float(elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        type=str,
        default="Кога е основан Софийският университет?",
        help="User prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per run.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of timed runs to average over.",
    )
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    print("Loading JAX model (this may take a bit)...")
    model, params = load_flax_model(dtype=jnp.bfloat16)

    inputs = build_inputs(tokenizer, args.prompt)

    # Warmup (not timed) to trigger any compilation
    print("Warmup run (not timed)...")
    _ = generate_once(model, params, inputs, max_new_tokens=16)

    total_tokens = 0
    total_time = 0.0
    last_sequences = None

    print(f"\nRunning {args.runs} timed runs...")
    for i in range(args.runs):
        seq, n_tok, dt = generate_once(
            model, params, inputs, max_new_tokens=args.max_new_tokens
        )
        last_sequences = seq
        total_tokens += n_tok
        total_time += dt
        tps = n_tok / dt if dt > 0 else float("nan")
        print(f"Run {i+1}: {n_tok} tokens in {dt:.3f}s  ({tps:.2f} tok/s)")

    avg_tokens = total_tokens / max(args.runs, 1)
    avg_time = total_time / max(args.runs, 1)
    avg_tps = avg_tokens / avg_time if avg_time > 0 else float("nan")

    print("\n=== Throughput summary ===")
    print(f"Average generated tokens: {avg_tokens:.1f}")
    print(f"Average time per run:    {avg_time:.3f} s")
    print(f"Average throughput:      {avg_tps:.2f} tokens/s")

    # Decode the last run
    print("\n=== Sample output (last run) ===")
    text = tokenizer.decode(
        np.asarray(last_sequences[0]),
        skip_special_tokens=True,
    )
    print(text)


if __name__ == "__main__":
    main()


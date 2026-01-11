from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# If you hit custom-model issues in v1 mode, some users report needing VLLM_USE_V1=0.
# Set before importing vllm:
#   VLLM_USE_V1=0 python scripts/bench_vllm.py --model toy-100m
from vllm import LLM, SamplingParams


def make_prompts(tokenizer, num_prompts: int, input_len: int):
    # Make prompts with approximately input_len tokens by decoding random token IDs.
    vocab = tokenizer.vocab_size
    prompts = []
    for _ in range(num_prompts):
        ids = [random.randrange(0, vocab - 1) for _ in range(input_len)]
        prompts.append(tokenizer.decode(ids))
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="toy-100m")
    ap.add_argument("--tokenizer", type=str, default="gpt2")
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--input-len", type=int, default=128)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    prompts = make_prompts(tok, args.num_prompts, args.input_len)

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_len,
        ignore_eos=True,
    )

    model_path = Path(args.model)
    model_arg = str(model_path.resolve()) if model_path.exists() else args.model

    llm = LLM(
        model=model_arg,
        tokenizer=args.tokenizer,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=True,
    )

    # Warmup
    _ = llm.generate(prompts[:8], sp)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    t1 = time.perf_counter()

    dt = t1 - t0

    prefill_tokens = 0
    decode_tokens = 0
    for o in outs:
        prefill_tokens += len(o.prompt_token_ids)
        decode_tokens += len(o.outputs[0].token_ids)

    total_tokens = prefill_tokens + decode_tokens

    print("==== vLLM throughput ====")
    print(f"wall_time_s:      {dt:.4f}")
    print(f"num_prompts:      {args.num_prompts}")
    print(f"input_len:        {args.input_len} (approx)")
    print(f"output_len:       {args.output_len} (max)")
    print(f"prefill_tokens:   {prefill_tokens}")
    print(f"decode_tokens:    {decode_tokens}")
    print(f"total_tokens:     {total_tokens}")
    print("")
    print(f"prefill tok/s:    {prefill_tokens / dt:,.2f}")
    print(f"decode tok/s:     {decode_tokens / dt:,.2f}")
    print(f"total tok/s:      {total_tokens / dt:,.2f}")


if __name__ == "__main__":
    main()

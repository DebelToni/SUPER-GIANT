import argparse
import json
import time
from pathlib import Path
from typing import List, Optional

import flax
import jax
import jax.numpy as jnp

from bggpt_compressed_kv_model import (
    ModelConfig,
    generate,
    load_bggpt_compressed,
)


def tokenize(prompt: str, vocab_size: int) -> List[int]:
    # simple hash-based tokenizer for offline smoke tests
    tokens = [1]  # BOS
    for word in prompt.split():
        tokens.append((hash(word) % (vocab_size - 3)) + 3)
    tokens.append(2)  # EOS
    return tokens


def main():
    parser = argparse.ArgumentParser(description="JAX/Flax compressed KV BgGPT smoke test")
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--rope-factor", type=float, default=1.0)
    parser.add_argument("--kv-compression-ratio", type=float, default=1.0)
    parser.add_argument("--prompt", default="How are you?")
    parser.add_argument("--max-new-tokens", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--params", type=Path, default=None, help="Optional msgpack params from convert_bggpt_weights.py")
    parser.add_argument("--config-json", type=Path, default=None, help="Optional ModelConfig JSON (defaults to CLI args)")
    args = parser.parse_args()

    if args.config_json is not None:
        cfg_data = json.loads(args.config_json.read_text())
        cfg = ModelConfig(**cfg_data)
    else:
        cfg = ModelConfig(
            hidden_size=args.hidden_size,
            num_attention_heads=args.heads,
            num_key_value_heads=args.kv_heads,
            intermediate_size=args.intermediate_size,
            num_hidden_layers=args.layers,
            vocab_size=args.vocab_size,
            rope_factor=args.rope_factor,
            kv_compression_ratio=args.kv_compression_ratio,
        )

    model = load_bggpt_compressed(cfg)
    tokens = tokenize(args.prompt, args.vocab_size)
    input_ids = jnp.array(tokens, dtype=jnp.int32)[None, :]

    rng = jax.random.PRNGKey(args.seed)
    if args.params is not None and args.params.exists():
        raw = args.params.read_bytes()
        loaded = flax.serialization.from_bytes(None, raw)
        variables = {"params": loaded["params"] if isinstance(loaded, dict) and "params" in loaded else loaded}
        print(f"Loaded params from {args.params}")
    else:
        t0 = time.time()
        variables = model.init(rng, input_ids, past_key_values=None, use_cache=True, deterministic=True)
        init_dt = time.time() - t0
        print(f"init params done in {init_dt:.3f}s")

    t0 = time.time()
    logits, past = model.apply(variables, input_ids, past_key_values=None, use_cache=True, deterministic=True)
    fwd_dt = time.time() - t0
    print(f"forward logits shape {logits.shape}, cache layers={len(past)} in {fwd_dt:.3f}s")
    if past:
        k_comp, v_comp = past[0]
        print(f"layer0 compressed shapes k={k_comp.shape}, v={v_comp.shape}")

    t0 = time.time()
    generated = generate(
        model=model,
        params=variables,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rng=rng,
    )
    gen_dt = time.time() - t0
    print(f"generated ids: {generated.shape} in {gen_dt:.3f}s")
    print("ids:", generated[0].tolist())


if __name__ == "__main__":
    main()

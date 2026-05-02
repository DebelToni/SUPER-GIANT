from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from Experiments.Ivancho_Brain.train import make_model


DEFAULT_PROMPTS = (
    "Once upon a time there was a little girl named Lily.",
    "Tom and Jen went to the park and saw",
    "The small dog was very happy because",
    "Max tried to keep going, but he was too tired.",
)


def load_params(model, cfg, checkpoint_dir: Path, dummy: jnp.ndarray):
    params = model.init(jax.random.PRNGKey(int(cfg.seed)), dummy, deterministic=True)["params"]
    return serialization.from_bytes(params, (checkpoint_dir / "params.msgpack").read_bytes())


def sample_from_topk(token_ids: np.ndarray, logits: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    scaled = logits.astype(np.float64) / max(float(temperature), 1e-6)
    scaled = scaled - scaled.max()
    probs = np.exp(scaled)
    probs = probs / probs.sum()
    return int(rng.choice(token_ids.astype(np.int64), p=probs))


def choose_emitted_token(
    pred: np.ndarray,
    hard_emit: np.ndarray,
    top_tokens: np.ndarray,
    top_logits: np.ndarray,
    *,
    sample: bool,
    temperature: float,
    rng: np.random.Generator,
) -> tuple[int, bool, int]:
    last_pred = pred[0, -1]
    last_emit = hard_emit[0, -1]
    if bool(last_emit.any()):
        idx = int(last_emit.argmax())
        if sample:
            return sample_from_topk(top_tokens[0, -1, idx], top_logits[0, -1, idx], temperature, rng), True, idx + 1
        return int(last_pred[idx]), True, idx + 1
    if sample:
        return sample_from_topk(top_tokens[0, -1, -1], top_logits[0, -1, -1], temperature, rng), False, int(last_pred.shape[0])
    return int(last_pred[-1]), False, int(last_pred.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("Config.yml")))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=48)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.tokenizer_path, use_fast=True)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    model = make_model(cfg)
    dummy = jnp.zeros((1, args.context_tokens), dtype=jnp.int32)
    params = load_params(model, cfg, Path(args.checkpoint), dummy)
    rng = np.random.default_rng(args.seed)

    for i, prompt in enumerate(DEFAULT_PROMPTS):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        emitted = 0
        emit_steps: list[int] = []
        for _ in range(args.max_new_tokens):
            ctx = np.full((1, args.context_tokens), int(pad_id), dtype=np.int32)
            window = np.asarray(ids[-args.context_tokens :], dtype=np.int32)
            ctx[0, -window.shape[0] :] = window
            out = model.apply({"params": params}, jnp.asarray(ctx), deterministic=True)
            token, did_emit, step = choose_emitted_token(
                np.asarray(jax.device_get(out["pred"])),
                np.asarray(jax.device_get(out["hard_emit"])),
                np.asarray(jax.device_get(out["top_tokens"])),
                np.asarray(jax.device_get(out["top_logits"])),
                sample=args.sample,
                temperature=args.temperature,
                rng=rng,
            )
            ids.append(token)
            emitted += int(did_emit)
            emit_steps.append(step)
        text = tokenizer.decode(ids, skip_special_tokens=True)
        continuation = tokenizer.decode(ids[-args.max_new_tokens :], skip_special_tokens=True)
        print(f"SAMPLE {i}")
        print(f"mode={'topk_sample' if args.sample else 'greedy'} temperature={args.temperature:.3f}")
        print(f"emit_rate={emitted / args.max_new_tokens:.6f} avg_emit_step={float(np.mean(emit_steps)):.4f}")
        print("PROMPT:", repr(prompt))
        print("CONTINUATION:", repr(continuation[:800]))
        print("FULL:", repr(text[:1000]))


if __name__ == "__main__":
    main()

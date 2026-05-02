from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from Experiments.Ivancho_Brain.data import TokenBatcher, load_train_val_tokens
from Experiments.Ivancho_Brain.graph_rnn_llm import estimate_param_count
from Experiments.Ivancho_Brain.train import make_model


def load_params(model, cfg, checkpoint_dir: Path, dummy: jnp.ndarray):
    params = model.init(jax.random.PRNGKey(int(cfg.seed)), dummy, deterministic=True)["params"]
    blob = (checkpoint_dir / "params.msgpack").read_bytes()
    return serialization.from_bytes(params, blob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("Config.yml")))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    _, val_tokens = load_train_val_tokens(args.config)
    loader = TokenBatcher(val_tokens, seq_len=int(cfg.data.seq_len), batch_size=args.batch_size, seed=int(cfg.seed) + 123)
    batch = jnp.asarray(loader.next_batch())
    inp = batch[:, :-1]
    target = batch[:, 1:]
    model = make_model(cfg)
    params = load_params(model, cfg, Path(args.checkpoint), inp)
    out = model.apply({"params": params}, inp, target, deterministic=True)

    hard_emit = np.asarray(jax.device_get(out["hard_emit"]))
    pred = np.asarray(jax.device_get(out["pred"]))
    target_np = np.asarray(jax.device_get(target))
    emit_any = hard_emit.any(axis=-1)
    first_idx = hard_emit.argmax(axis=-1)
    row_idx = np.arange(pred.shape[0])[:, None]
    col_idx = np.arange(pred.shape[1])[None, :]
    emitted_pred = pred[row_idx, col_idx, first_idx]
    final_pred = pred[:, :, -1]
    chosen = np.where(emit_any, emitted_pred, final_pred)

    tokenizer = AutoTokenizer.from_pretrained(cfg.paths.tokenizer_path, use_fast=True)
    emit_rate = float(hard_emit.mean())
    forced_rate = float(1.0 - emit_any.mean())
    avg_emit_step = float(((first_idx + 1) * emit_any).sum() / max(emit_any.sum(), 1))
    acc = float((chosen == target_np).mean())
    print(
        f"emit_rate={emit_rate:.6f} forced_rate={forced_rate:.6f} "
        f"avg_emit_step={avg_emit_step:.4f} chosen_acc={acc:.6f} params={estimate_param_count(params)}"
    )
    for i in range(min(args.samples, chosen.shape[0])):
        prompt = tokenizer.decode(np.asarray(inp[i, :24]), skip_special_tokens=True)
        predicted = tokenizer.decode(chosen[i], skip_special_tokens=True)
        gold = tokenizer.decode(target_np[i], skip_special_tokens=True)
        print(f"SAMPLE {i}")
        print("PROMPT:", repr(prompt[:300]))
        print("EMITTED:", repr(predicted[:500]))
        print("GOLD:", repr(gold[:500]))


if __name__ == "__main__":
    main()

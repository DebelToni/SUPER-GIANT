from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import serialization
from flax.training import train_state
import yaml

from GIANT.v3.data_curation.quality_filter.common import (
    load_yaml,
    merge_dicts,
    read_jsonl,
    tokenize_words,
    hash_token,
)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Bulgarian text-quality classifier.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def resolve_output_root(base_cfg: dict, cfg: dict) -> Path:
    return Path(base_cfg["paths"]["data_root"]) / cfg["output"]["run_dir"]


def encode_texts(rows: list[dict[str, Any]], *, vocab_size: int, max_tokens: int) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.zeros((len(rows), max_tokens), dtype=np.int32)
    mask = np.zeros((len(rows), max_tokens), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        tokens = tokenize_words(str(row["text"]))[:max_tokens]
        ids = [hash_token(token, vocab_size) for token in tokens]
        if not ids:
            ids = [1]
        token_ids[row_idx, : len(ids)] = np.asarray(ids, dtype=np.int32)
        mask[row_idx, : len(ids)] = 1.0
    return token_ids, mask


class FastTextClassifier(nn.Module):
    vocab_size: int
    embed_dim: int
    num_classes: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, token_ids, mask, *, train: bool):
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)(token_ids)
        mask_f = mask[..., None]
        x = (x * mask_f).sum(axis=1) / jnp.maximum(mask_f.sum(axis=1), 1.0)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return nn.Dense(self.num_classes)(x)


def create_model(cfg: dict, num_classes: int):
    model_type = cfg["model"]["type"]
    vocab_size = int(cfg["features"]["vocab_size"])
    max_tokens = int(cfg["features"]["max_tokens"])
    if model_type == "fasttext":
        return FastTextClassifier(
            vocab_size=vocab_size,
            embed_dim=int(cfg["models"]["fasttext"]["embed_dim"]),
            num_classes=num_classes,
            dropout_rate=float(cfg["models"]["fasttext"].get("dropout_rate", 0.0)),
        )
    raise ValueError(f"Unsupported model.type={model_type!r}; expected 'fasttext'")


def iter_batches(token_ids: np.ndarray, mask: np.ndarray, labels: np.ndarray, batch_size: int, *, shuffle: bool, seed: int):
    indices = np.arange(len(labels))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield {
            "token_ids": token_ids[batch_idx],
            "mask": mask[batch_idx],
            "labels": labels[batch_idx],
        }


def softmax_metrics(logits: np.ndarray, labels: np.ndarray, num_classes: int) -> dict[str, float]:
    preds = logits.argmax(axis=-1)
    accuracy = float((preds == labels).mean()) if len(labels) else 0.0
    f1_scores = []
    for cls in range(num_classes):
        tp = np.sum((preds == cls) & (labels == cls))
        fp = np.sum((preds == cls) & (labels != cls))
        fn = np.sum((preds != cls) & (labels == cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        f1_scores.append(float(f1))
    return {"accuracy": accuracy, "macro_f1": float(sum(f1_scores) / max(len(f1_scores), 1))}


def main() -> None:
    args = parse_args()
    base_cfg = load_yaml(Path(__file__).resolve().parent / "Config.yml")
    cfg = merge_dicts(base_cfg, load_yaml(args.config))

    benchmark_root = Path(base_cfg["paths"]["data_root"]) / cfg["input"]["benchmark_dir"]
    train_rows = read_jsonl(benchmark_root / "train.jsonl")
    val_rows = read_jsonl(benchmark_root / "val.jsonl")
    test_rows = read_jsonl(benchmark_root / "test.jsonl")

    num_classes = len(cfg["labels"])
    max_tokens = int(cfg["features"]["max_tokens"])
    vocab_size = int(cfg["features"]["vocab_size"])
    batch_size = int(cfg["training"]["batch_size"])
    seed = int(cfg["training"]["seed"])
    num_epochs = int(cfg["training"]["num_epochs"])

    x_train, m_train = encode_texts(train_rows, vocab_size=vocab_size, max_tokens=max_tokens)
    x_val, m_val = encode_texts(val_rows, vocab_size=vocab_size, max_tokens=max_tokens)
    x_test, m_test = encode_texts(test_rows, vocab_size=vocab_size, max_tokens=max_tokens)
    y_train = np.asarray([row["label_id"] for row in train_rows], dtype=np.int32)
    y_val = np.asarray([row["label_id"] for row in val_rows], dtype=np.int32)
    y_test = np.asarray([row["label_id"] for row in test_rows], dtype=np.int32)

    model = create_model(cfg, num_classes=num_classes)
    rng = jax.random.PRNGKey(seed)
    init_batch = {
        "token_ids": jnp.asarray(x_train[: min(len(x_train), batch_size)]),
        "mask": jnp.asarray(m_train[: min(len(m_train), batch_size)]),
    }
    params = model.init(rng, init_batch["token_ids"], init_batch["mask"], train=False)["params"]
    optimizer = optax.adamw(
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    @jax.jit
    def train_step(state, token_ids, mask, labels, dropout_key):
        def loss_fn(params):
            logits = model.apply({"params": params}, token_ids, mask, train=True, rngs={"dropout": dropout_key})
            labels_onehot = jax.nn.one_hot(labels, num_classes)
            loss = optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, logits

    @jax.jit
    def predict_step(params, token_ids, mask):
        return model.apply({"params": params}, token_ids, mask, train=False)

    best_val = -1.0
    best_params = state.params
    history = []
    train_start = time.perf_counter()

    for epoch in range(num_epochs):
        train_losses = []
        for batch_idx, batch in enumerate(iter_batches(x_train, m_train, y_train, batch_size, shuffle=True, seed=seed + epoch)):
            rng, dropout_key = jax.random.split(rng)
            state, loss, _ = train_step(
                state,
                jnp.asarray(batch["token_ids"]),
                jnp.asarray(batch["mask"]),
                jnp.asarray(batch["labels"]),
                dropout_key,
            )
            train_losses.append(float(loss))

        val_logits = []
        for batch in iter_batches(x_val, m_val, y_val, batch_size, shuffle=False, seed=seed):
            logits = predict_step(state.params, jnp.asarray(batch["token_ids"]), jnp.asarray(batch["mask"]))
            val_logits.append(np.asarray(logits))
        val_logits_np = np.concatenate(val_logits, axis=0) if val_logits else np.zeros((0, num_classes), dtype=np.float32)
        val_metrics = softmax_metrics(val_logits_np, y_val, num_classes)
        if val_metrics["macro_f1"] >= best_val:
            best_val = val_metrics["macro_f1"]
            best_params = state.params
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(sum(train_losses) / max(len(train_losses), 1)),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(epoch_record)
        print(f"[train] epoch={epoch_record['epoch']} loss={epoch_record['train_loss']:.4f} val_acc={epoch_record['val_accuracy']:.4f} val_f1={epoch_record['val_macro_f1']:.4f}")

    train_elapsed = time.perf_counter() - train_start

    test_logits = []
    start = time.perf_counter()
    for batch in iter_batches(x_test, m_test, y_test, batch_size, shuffle=False, seed=seed):
        logits = predict_step(best_params, jnp.asarray(batch["token_ids"]), jnp.asarray(batch["mask"]))
        test_logits.append(np.asarray(logits))
    elapsed = time.perf_counter() - start
    test_logits_np = np.concatenate(test_logits, axis=0) if test_logits else np.zeros((0, num_classes), dtype=np.float32)
    test_metrics = softmax_metrics(test_logits_np, y_test, num_classes)
    docs_per_second = len(y_test) / max(elapsed, 1e-6)

    output_root = resolve_output_root(base_cfg, cfg)
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "model_type": cfg["model"]["type"],
        "backend": jax.default_backend(),
        "num_train": int(len(y_train)),
        "num_val": int(len(y_val)),
        "num_test": int(len(y_test)),
        "train_wall_time_s": train_elapsed,
        "history": history,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_docs_per_second": docs_per_second,
    }
    (output_root / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata = {
        "model_type": cfg["model"]["type"],
        "labels": list(cfg["labels"]),
        "features": cfg["features"],
        "models": cfg["models"],
        "training": cfg["training"],
    }
    (output_root / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_root / "resolved_config.yml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (output_root / "model_params.msgpack").write_bytes(serialization.to_bytes(best_params))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

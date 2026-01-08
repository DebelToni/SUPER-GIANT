from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import core as flax_core
from jax import config as jax_config
from omegaconf import OmegaConf
from datasets import load_dataset

from config_utils import load_config
from tokenizer_utils import build_custom_tokenizer, load_tokenizer


PROJECT_DIR = Path(__file__).resolve().parent
V2_ROOT = PROJECT_DIR.parent.parent / "v2"
SMOL_DIR = V2_ROOT / "smol"
MODEL_DIR = V2_ROOT / "model"
sys.path.insert(0, str(SMOL_DIR))
sys.path.insert(1, str(MODEL_DIR))

from GiantGPT import GiantGPT  # noqa: E402
from checkpoint_io import load_npz  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Evaluate general loss: base vs fine-tuned SmolLM.")
    ap.add_argument("--stage", default="simple_wiki", help="Stage name from v2/data_pipeline/Config.yml.")
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_batches", type=int, default=20)
    ap.add_argument("--max_rows", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base_ckpt", default=None)
    ap.add_argument("--finetuned_ckpt", default=None)
    return ap.parse_args()


def _to_dtype(name: str) -> jnp.dtype:
    try:
        return getattr(jnp, name)
    except AttributeError:
        return jnp.dtype(name)


def _apply_compute_dtype_override(cfg) -> None:
    desired = None
    if "model" in cfg and "compute_dtype" in cfg.model:
        desired = cfg.model.compute_dtype
    if not desired:
        return
    dtype = _to_dtype(str(desired))
    try:
        import GiantGPT as smol_gpt_module
        import Transformer_block as smol_block_module
    except Exception as exc:
        print(f"[dtype] Failed to import smol modules for override: {exc}")
        return

    smol_gpt_module.COMPUTE_DTYPE = dtype
    smol_block_module.COMPUTE_DTYPE = dtype
    try:
        smol_gpt_module.MODEL_CFG.compute_dtype = str(desired)
        smol_block_module.MODEL_CFG.compute_dtype = str(desired)
    except Exception:
        pass
    jax_config.update("jax_default_matmul_precision", str(desired))
    print(f"[dtype] Overriding SmolLM compute dtype to {desired}")


def _resize_embedding(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    new = np.array(target, copy=True)
    src = np.asarray(source)
    rows = min(new.shape[0], src.shape[0])
    new[:rows] = src[:rows]
    return new


def _merge_params(init_params, loaded_params) -> flax_core.FrozenDict:
    merged = flax_core.unfreeze(init_params)
    replaced = 0
    resized = 0

    def assign(dst, src, path=()):
        nonlocal replaced, resized
        for key, val in src.items():
            if key not in dst:
                continue
            if isinstance(val, dict):
                if isinstance(dst[key], dict):
                    assign(dst[key], val, path + (key,))
                continue
            dst_val = dst[key]
            if hasattr(dst_val, "shape") and hasattr(val, "shape"):
                if dst_val.shape == val.shape:
                    dst[key] = val
                    replaced += 1
                elif path + (key,) == ("Embed_0", "embedding"):
                    dst[key] = _resize_embedding(dst_val, val)
                    resized += 1

    assign(merged, loaded_params)
    merged = flax_core.freeze(merged)
    merged = jax.tree_util.tree_map(lambda x: jnp.asarray(x), merged)
    if resized:
        print(f"[init] resized {resized} embedding tensors for new vocab")
    print(f"[init] loaded {replaced} tensors from checkpoint")
    return merged


def _load_params(model, batch_size: int, seq_len: int, ckpt_path: str) -> flax_core.FrozenDict:
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.zeros((batch_size, seq_len), dtype=jnp.int32))["params"]
    if isinstance(params, dict):
        params = flax_core.freeze(params)
    loaded = load_npz(ckpt_path)
    return _merge_params(params, loaded)


def _build_model(cfg, vocab_size: int, seq_len: int) -> GiantGPT:
    model_cfg = cfg.model
    return GiantGPT(
        vocab_size=vocab_size,
        context_length=seq_len,
        d_model=model_cfg.embedding_size,
        n_heads=model_cfg.num_heads,
        d_ff=model_cfg.feed_forward_size,
        n_layers=model_cfg.num_layers,
        dropout_rate=0.0,
    )


def _make_loss_fn(model):
    @jax.jit
    def loss_fn(params, batch):
        logits = model.apply(
            {"params": params},
            batch["input"],
            deterministic=True,
            use_kv_cache=False,
            cur_index=None,
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["target"])
        loss = (loss * batch["mask"]).sum() / batch["mask"].sum()
        return loss

    return loss_fn


def _load_stage_source(stage_name: str) -> Tuple[str, str, Dict]:
    cfg_path = V2_ROOT / "data_pipeline" / "Config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Data pipeline config not found: {cfg_path}")
    dp_cfg = OmegaConf.load(cfg_path)
    if stage_name not in dp_cfg.stages:
        raise KeyError(f"Stage '{stage_name}' not found in {cfg_path}")
    stage = dp_cfg.stages[stage_name]
    sources = stage.get("sources", [])
    if not sources:
        raise ValueError(f"Stage '{stage_name}' has no sources.")
    source = sources[0]
    dataset_name = source.get("dataset_name")
    split = source.get("split", "train")
    return dataset_name, split, source


def _iter_text_rows(ds, source: Dict, max_rows: int):
    text_field = source.get("text_field")
    join_fields = source.get("join_fields")
    join_separator = source.get("join_separator", " ")

    count = 0
    for row in ds:
        if text_field:
            text = row.get(text_field, "")
        elif join_fields:
            parts = [str(row.get(field, "")).strip() for field in join_fields]
            text = join_separator.join([p for p in parts if p])
        else:
            text = ""
        text = str(text).strip()
        if not text:
            continue
        yield text
        count += 1
        if max_rows and count >= max_rows:
            break


def _build_token_windows(
    tokenizer,
    texts,
    *,
    seq_len: int,
    max_samples: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    buffer: List[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        buffer.extend(ids)
        while len(buffer) >= seq_len + 1 and len(samples) < max_samples:
            window = buffer[: seq_len + 1]
            buffer = buffer[seq_len:]
            inputs = np.asarray(window[:-1], dtype=np.int32)
            targets = np.asarray(window[1:], dtype=np.int32)
            mask = np.ones(seq_len, dtype=np.float32)
            samples.append((inputs, targets, mask))
        if len(samples) >= max_samples:
            break
    return samples


def main() -> None:
    args = parse_args()
    cfg = load_config()

    build_custom_tokenizer(force=False)
    tokenizer = load_tokenizer()
    _apply_compute_dtype_override(cfg)

    dataset_name, split, source = _load_stage_source(args.stage)
    if not dataset_name:
        raise ValueError(f"Stage '{args.stage}' does not define dataset_name.")

    ds = load_dataset(dataset_name, split=split, streaming=False)
    texts = _iter_text_rows(ds, source, args.max_rows)
    max_samples = args.num_batches * args.batch_size
    samples = _build_token_windows(
        tokenizer,
        texts,
        seq_len=args.seq_len,
        max_samples=max_samples,
    )
    if len(samples) < max_samples:
        print(f"Warning: only collected {len(samples)} samples (requested {max_samples}).")

    model = _build_model(cfg, len(tokenizer), args.seq_len)
    loss_fn = _make_loss_fn(model)

    base_ckpt = args.base_ckpt or str(Path(cfg.paths.checkpoint_root) / "smollm-135m.npz")
    finetuned_ckpt = args.finetuned_ckpt or str(Path(cfg.paths.checkpoint_root) / "step_0005000.npz")

    base_params = _load_params(model, args.batch_size, args.seq_len, base_ckpt)
    finetuned_params = _load_params(model, args.batch_size, args.seq_len, finetuned_ckpt)

    base_losses: List[float] = []
    ft_losses: List[float] = []

    for batch_idx in range(args.num_batches):
        start = batch_idx * args.batch_size
        end = start + args.batch_size
        batch_samples = samples[start:end]
        if len(batch_samples) < args.batch_size:
            break
        batch = {
            "input": np.stack([s[0] for s in batch_samples], axis=0),
            "target": np.stack([s[1] for s in batch_samples], axis=0),
            "mask": np.stack([s[2] for s in batch_samples], axis=0),
        }
        batch_jax = {
            "input": jnp.asarray(batch["input"]),
            "target": jnp.asarray(batch["target"]),
            "mask": jnp.asarray(batch["mask"]),
        }
        base_loss = float(loss_fn(base_params, batch_jax))
        ft_loss = float(loss_fn(finetuned_params, batch_jax))
        base_losses.append(base_loss)
        ft_losses.append(ft_loss)

    base_mean = float(np.mean(base_losses))
    ft_mean = float(np.mean(ft_losses))
    base_std = float(np.std(base_losses))
    ft_std = float(np.std(ft_losses))

    print("=== General Loss Eval ===")
    print(f"stage         : {args.stage}")
    print(f"dataset_name  : {dataset_name}")
    print(f"split         : {split}")
    print(f"seq_len       : {args.seq_len}")
    print(f"batch_size    : {args.batch_size}")
    print(f"num_batches   : {args.num_batches}")
    print(f"base_ckpt     : {base_ckpt}")
    print(f"finetuned_ckpt: {finetuned_ckpt}")
    print()
    print(f"base loss     : {base_mean:.4f} ± {base_std:.4f}")
    print(f"finetune loss : {ft_mean:.4f} ± {ft_std:.4f}")
    print(f"delta (ft-base): {ft_mean - base_mean:+.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import math
from pathlib import Path
from functools import partial
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from v2.model.GiantGPT import GiantGPT
from v2.model.arrow_data_loader import ShardedArrowDataset, StageDataLoader
from v2.model.checkpoint_manager import latest as latest_ckpt
from v2.model.checkpoint_manager import load as load_ckpt


def load_configs() -> OmegaConf:
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    global_cfg = OmegaConf.load(project_root / "Global_Config.yml")
    local_cfg = OmegaConf.load(model_dir / "Config.yml")
    cfg = OmegaConf.merge(global_cfg, local_cfg)

    base_prefix_str = cfg.paths.get("data_root", "") if "paths" in cfg else ""
    base_prefix = Path(base_prefix_str) if base_prefix_str else None

    def resolve_path(value: str | None) -> str | None:
        if value is None:
            return None
        path = Path(str(value))
        if path.is_absolute() or base_prefix is None:
            return str(path)
        return str(base_prefix / path)

    if base_prefix is not None:
        cfg.paths.data_root = str(base_prefix)
    else:
        cfg.paths.data_root = str(project_root)

    for key in ("processed_data_root", "dataloader_state_root", "logs_root"):
        if key in cfg.paths and cfg.paths[key] is not None:
            resolved = resolve_path(cfg.paths[key])
            if resolved is not None:
                cfg.paths[key] = resolved

    if "checkpoint_dir" in cfg.qa_finetune and cfg.qa_finetune.checkpoint_dir is not None:
        resolved = resolve_path(cfg.qa_finetune.checkpoint_dir)
        if resolved is not None:
            cfg.qa_finetune.checkpoint_dir = resolved

    if "checkpoint_dir" in cfg.evaluate and cfg.evaluate.checkpoint_dir is not None:
        resolved = resolve_path(cfg.evaluate.checkpoint_dir)
        if resolved is not None:
            cfg.evaluate.checkpoint_dir = resolved

    return cfg


def load_tokenizer(cfg: OmegaConf):
    tok_cfg = cfg.tokenizer
    if tok_cfg.use_custom:
        tokenizer = AutoTokenizer.from_pretrained(tok_cfg.custom_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_cfg.name,
            use_fast=True,
            cache_dir=tok_cfg.cache_dir,
        )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def parse_stage_configs(cfg: OmegaConf) -> List[Dict]:
    return OmegaConf.to_container(cfg.stages, resolve=True)


def resolve_checkpoint(path_arg: str | None, cfg, base_root: Path) -> Tuple[str, int]:
    if path_arg and path_arg != "latest":
        p = Path(path_arg)
        if not p.is_absolute():
            p = (base_root / p).resolve()
        return str(p), -1
    ckpt_dir = Path(cfg.evaluate.checkpoint_dir or "checkpoints")
    if not ckpt_dir.is_absolute():
        ckpt_dir = (base_root / ckpt_dir).resolve()
    latest = latest_ckpt(str(ckpt_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    return latest, -1


@partial(jax.jit, static_argnames=("model",))
def eval_step(params, batch, *, model):
    logits = model.apply({"params": params}, batch["input"], deterministic=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["target"])
    loss = jnp.sum(loss * batch["mask"]) / jnp.sum(batch["mask"])
    return loss


def evaluate_on_stage(
    params,
    model,
    loader: StageDataLoader,
    max_batches: int,
) -> Tuple[float, int]:
    losses = []
    batches_seen = 0
    for batch in loader:
        loss = float(eval_step(params, batch, model=model))
        losses.append(loss)
        batches_seen += 1
        if batches_seen >= max_batches:
            break
    if not losses:
        return float("nan"), 0
    return float(np.mean(losses)), batches_seen


def main():
    parser = argparse.ArgumentParser(description="Evaluate SUPER-GIANT on a random curriculum sample.")
    parser.add_argument("--checkpoint", default="latest", help="Path to checkpoint (.npz) or 'latest'.")
    parser.add_argument("--checkpoint_dir", default=None, help="Override checkpoint directory for --checkpoint latest.")
    args = parser.parse_args()

    cfg = load_configs()
    if args.checkpoint_dir is not None:
        cfg.evaluate.checkpoint_dir = args.checkpoint_dir

    tokenizer = load_tokenizer(cfg)
    base_root = Path(cfg.paths.data_root)

    stage_cfgs = parse_stage_configs(cfg)
    dataset_root = Path(cfg.paths.processed_data_root)
    if not dataset_root.is_absolute():
        dataset_root = (base_root / dataset_root).resolve()

    max_seq_len = max(int(stage["seq_len"]) for stage in stage_cfgs)
    model = GiantGPT(
        vocab_size=len(tokenizer),
        context_length=max_seq_len,
        d_model=cfg.model.embedding_size,
        n_heads=cfg.model.num_heads,
        d_ff=cfg.model.feed_forward_size,
        n_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
    )

    ckpt_path, _ = resolve_checkpoint(args.checkpoint, cfg, base_root)
    params, ckpt_step = load_ckpt(ckpt_path)
    print(f"Loaded checkpoint: {ckpt_path} (step {ckpt_step})")

    eval_cfg = cfg.evaluate
    total_samples = int(eval_cfg.total_samples)
    samples_per_stage = int(eval_cfg.samples_per_stage)
    eval_batch_size = int(eval_cfg.batch_size)
    eval_seed = int(eval_cfg.seed)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    )

    rng = np.random.default_rng(eval_seed)
    remaining = total_samples

    results = []
    for stage in stage_cfgs:
        if remaining <= 0:
            break
        target_rows = min(samples_per_stage, remaining)
        data_path = dataset_root / stage["dataset"]
        dataset = ShardedArrowDataset(data_path)
        target_rows = min(target_rows, dataset.total_rows)
        if target_rows < eval_batch_size:
            continue

        loader = StageDataLoader(
            dataset,
            batch_size=eval_batch_size,
            seq_len=int(stage["seq_len"]),
            shuffle=True,
            seed=eval_seed ^ int(stage["seq_len"]),
            pad_token_id=pad_token_id,
            max_rows=target_rows,
        )
        max_batches = math.ceil(target_rows / eval_batch_size)
        loss, batches = evaluate_on_stage(params, model, loader, max_batches)
        if not math.isnan(loss):
            results.append((stage["name"], loss, batches, int(stage["seq_len"])))
        remaining -= target_rows

    if not results:
        print("No evaluation batches produced; check dataset paths and sample sizes.")
        return

    stage_losses = []
    for name, loss, batches, seq_len in results:
        ppl = math.exp(loss) if loss < 50 else float("inf")
        stage_losses.append(loss)
        print(f"{name:20s} | seq={seq_len:<4d} batches={batches:<4d} loss={loss:.4f} ppl={ppl:.2f}")

    mean_loss = float(np.mean(stage_losses))
    overall_ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")
    print(f"\nOverall mean loss={mean_loss:.4f} ppl={overall_ppl:.2f}")


if __name__ == "__main__":
    main()

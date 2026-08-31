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

from GIANT.v3.model.GiantGPT import GiantGPT
from GIANT.v3.model.arrow_data_loader import ShardedArrowDataset, StageDataLoader
from GIANT.v3.model.checkpoint_manager import latest as latest_ckpt
from GIANT.v3.model.checkpoint_manager import load as load_ckpt
from GIANT.v3.model.checkpoint_manager import set_npz_metadata
from GIANT.v3.model.lora import (
    assert_tree_compatible,
    count_parameters,
    lora_config_from_mapping,
    validate_adapter_checkpoint_manifest,
)


def load_configs(
    model_config_path: str | None = None,
    global_config_path: str | None = None,
) -> OmegaConf:
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    global_path = Path(global_config_path) if global_config_path else project_root / "Global_Config.yml"
    model_path = Path(model_config_path) if model_config_path else model_dir / "Config.yml"
    global_cfg = OmegaConf.load(global_path)
    local_cfg = OmegaConf.load(model_path)
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


def resolve_params_dir(root: Path) -> Path:
    return root if root.name == "params" else root / "params"


def resolve_adapter_checkpoint(path_arg: str | None, cfg, base_root: Path) -> Path | None:
    lora_config = lora_config_from_mapping(cfg.get("lora"))
    if not lora_config.enabled:
        if path_arg:
            raise ValueError("--adapter requires lora.enabled=true")
        return None
    request = path_arg or cfg.lora.get("adapter_checkpoint")
    if not request:
        raise ValueError("LoRA evaluation requires --adapter or lora.adapter_checkpoint")
    path = Path(str(request)).expanduser()
    if not path.is_absolute():
        path = (base_root / path).resolve()
    if path.is_file():
        return path
    adapters_dir = path if path.name == "adapters" else path / "adapters"
    latest = latest_ckpt(str(adapters_dir))
    if latest is None:
        raise FileNotFoundError(f"No adapter checkpoints found under {adapters_dir}")
    return Path(latest)


def resolve_checkpoint(path_arg: str | None, cfg, base_root: Path) -> Tuple[str, int]:
    if path_arg and path_arg != "latest":
        p = Path(path_arg)
        if not p.is_absolute():
            p = (base_root / p).resolve()
        if p.is_dir():
            params_dir = resolve_params_dir(p)
            latest = latest_ckpt(str(params_dir))
            if latest is None:
                raise FileNotFoundError(f"No checkpoints found under {params_dir}")
            return latest, -1
        return str(p), -1
    ckpt_dir = Path(cfg.evaluate.checkpoint_dir or "checkpoints")
    if not ckpt_dir.is_absolute():
        ckpt_dir = (base_root / ckpt_dir).resolve()
    params_dir = resolve_params_dir(ckpt_dir)
    latest = latest_ckpt(str(params_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found under {params_dir}")
    return latest, -1


@partial(jax.jit, static_argnames=("model",))
def eval_step(params, adapter_params, batch, *, model):
    variables = {"params": params}
    if adapter_params is not None:
        variables["adapters"] = adapter_params
    logits = model.apply(variables, batch["input"], deterministic=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["target"])
    loss_numer = jnp.sum(loss * batch["mask"])
    loss_denom = jnp.maximum(jnp.sum(batch["mask"]), 1.0)
    return loss_numer / loss_denom


def evaluate_on_stage(
    params,
    model,
    loader: StageDataLoader,
    max_batches: int,
    adapter_params=None,
) -> Tuple[float, int]:
    losses = []
    batches_seen = 0
    for batch in loader:
        loss = float(eval_step(params, adapter_params, batch, model=model))
        losses.append(loss)
        batches_seen += 1
        if batches_seen >= max_batches:
            break
    if not losses:
        return float("nan"), 0
    return float(np.mean(losses)), batches_seen


def main():
    parser = argparse.ArgumentParser(description="Evaluate SUPER-GIANT on a random curriculum sample.")
    parser.add_argument("--config", default=None, help="Model/training config YAML.")
    parser.add_argument("--global_config", default=None, help="Global config YAML.")
    parser.add_argument("--checkpoint", default="latest", help="Path to checkpoint (.npz) or 'latest'.")
    parser.add_argument("--checkpoint_dir", default=None, help="Override checkpoint directory for --checkpoint latest.")
    parser.add_argument("--adapter", default=None, help="Adapter NPZ or run directory.")
    parser.add_argument("--no_write_metadata", action="store_true", help="Skip writing validation metrics to checkpoint metadata.")
    args = parser.parse_args()

    cfg = load_configs(args.config, args.global_config)
    if args.checkpoint_dir is not None:
        cfg.evaluate.checkpoint_dir = args.checkpoint_dir

    tokenizer = load_tokenizer(cfg)
    base_root = Path(cfg.paths.data_root)

    stage_cfgs = parse_stage_configs(cfg)
    dataset_root = Path(cfg.paths.processed_data_root)
    if not dataset_root.is_absolute():
        dataset_root = (base_root / dataset_root).resolve()

    max_seq_len = max(int(stage["seq_len"]) for stage in stage_cfgs)
    if str(cfg.model.get("mode", "decoder")).lower() != "decoder" or bool(cfg.model.get("causal", True)) is False:
        raise ValueError("GIANT v3 only supports causal decoder models")
    lora_config = lora_config_from_mapping(cfg.get("lora"))
    model = GiantGPT(
        vocab_size=len(tokenizer),
        context_length=max_seq_len,
        d_model=cfg.model.embedding_size,
        n_heads=cfg.model.num_heads,
        d_ff=cfg.model.feed_forward_size,
        n_layers=cfg.model.num_layers,
        dropout_rate=cfg.model.dropout_rate,
        num_kv_heads=int(cfg.model.num_kv_heads),
        rotary_dim=int(cfg.model.rope_dim),
        rope_theta=float(cfg.model.get("rope_theta", 10000.0)),
        param_dtype=cfg.model.param_dtype,
        compute_dtype=cfg.model.compute_dtype,
        use_remat=bool(cfg.model.use_remat),
        enable_xsa=bool(cfg.model.enable_xsa),
        causal=True,
        lora_config=lora_config,
    )

    checkpoint_request = args.checkpoint
    if lora_config.enabled and checkpoint_request == "latest" and cfg.lora.get("base_checkpoint"):
        checkpoint_request = str(cfg.lora.base_checkpoint)
    ckpt_path, _ = resolve_checkpoint(checkpoint_request, cfg, base_root)
    params, ckpt_step = load_ckpt(ckpt_path)
    print(f"Loaded checkpoint: {ckpt_path} (step {ckpt_step})")

    key_params = jax.random.PRNGKey(int(cfg.evaluate.seed))
    key_adapters = jax.random.fold_in(key_params, 1)
    dummy = jnp.zeros((1, 1), dtype=jnp.int32)
    params_template = jax.eval_shape(
        lambda params_key, adapters_key: model.init(
            {"params": params_key, "adapters": adapters_key},
            dummy,
            deterministic=True,
        )["params"],
        key_params,
        key_adapters,
    )
    assert_tree_compatible(params_template, params, label="base checkpoint")

    adapter_params = None
    adapter_path = resolve_adapter_checkpoint(args.adapter, cfg, base_root)
    if adapter_path is not None:
        _, initialized = model.apply(
            {"params": params},
            dummy,
            deterministic=True,
            rngs={"adapters": key_adapters},
            mutable=["adapters"],
        )
        adapter_params, _ = load_ckpt(str(adapter_path))
        assert_tree_compatible(
            initialized["adapters"], adapter_params, label="LoRA adapter checkpoint"
        )
        validate_adapter_checkpoint_manifest(
            adapter_path,
            base_checkpoint=ckpt_path,
            config=lora_config,
            base_parameter_count=count_parameters(params),
            adapter_parameter_count=count_parameters(adapter_params),
        )
        print(f"Loaded adapter: {adapter_path}")

    params = jax.device_put(params)
    if adapter_params is not None:
        adapter_params = jax.device_put(adapter_params)

    eval_cfg = cfg.evaluate
    total_samples = int(eval_cfg.total_samples)
    samples_per_stage = int(eval_cfg.samples_per_stage)
    eval_batch_size = int(eval_cfg.batch_size)
    eval_seed = int(eval_cfg.seed)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
        tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    )

    rng = np.random.default_rng(eval_seed)
    
    # Weight sampling towards later stages (quadratic weighting)
    num_stages = len(stage_cfgs)
    stage_weights = np.array([(i + 1) ** 2 for i in range(num_stages)])
    stage_weights = stage_weights / stage_weights.sum()
    
    # Allocate samples per stage based on weights
    stage_samples = (stage_weights * total_samples).astype(int)
    # Ensure we use exactly total_samples by adding remainder to last stage
    remainder = total_samples - stage_samples.sum()
    stage_samples[-1] += remainder

    results = []
    for stage_idx, stage in enumerate(stage_cfgs):
        target_rows = int(stage_samples[stage_idx])
        if target_rows == 0:
            continue
        
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
        loss, batches = evaluate_on_stage(
            params, model, loader, max_batches, adapter_params=adapter_params
        )
        if not math.isnan(loss):
            results.append((stage["name"], loss, batches, int(stage["seq_len"]), target_rows))

    if not results:
        print("No evaluation batches produced; check dataset paths and sample sizes.")
        return

    stage_losses = []
    total_samples_evaluated = 0
    for name, loss, batches, seq_len, samples in results:
        ppl = math.exp(loss) if loss < 50 else float("inf")
        stage_losses.append(loss)
        total_samples_evaluated += samples
        print(f"{name:20s} | seq={seq_len:<4d} samples={samples:<6d} batches={batches:<4d} loss={loss:.4f} ppl={ppl:.2f}")

    mean_loss = float(np.mean(stage_losses))
    overall_ppl = math.exp(mean_loss) if mean_loss < 50 else float("inf")
    print(f"\nOverall: {total_samples_evaluated} samples | mean loss={mean_loss:.4f} ppl={overall_ppl:.2f}")

    if not args.no_write_metadata:
        print(f"\n[metadata] Writing validation metrics to checkpoint...")
        metadata_path = str(adapter_path) if adapter_path is not None else ckpt_path
        set_npz_metadata(metadata_path, "val_loss", f"{mean_loss:.6f}")
        set_npz_metadata(metadata_path, "val_ppl", f"{overall_ppl:.4f}")
        print(
            f"[metadata] Wrote val_loss={mean_loss:.6f}, val_ppl={overall_ppl:.4f} "
            f"to checkpoint {metadata_path}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax import traverse_util
from flax.core import unfreeze
from omegaconf import OmegaConf
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from TRMEncoderDecoder import TRMEncoderDecoder
from Training_step import eval_step, train_step
from common.checkpoint_manager import latest as latest_ckpt
from common.checkpoint_manager import load as load_ckpt
from common.checkpoint_manager import load_opt_state, save as save_ckpt, save_opt_state
from common.checkpoint_io import load_npz
from common.optimizer_utils import create_weight_decay_mask
from dataset import load_packed_dataset


def load_configs(config_path: Optional[str]) -> OmegaConf:
    global_cfg = OmegaConf.load(PROJECT_ROOT / "Global_Config.yml")
    local_cfg = OmegaConf.load(Path(__file__).resolve().parent / "Config.yml")
    cfg = OmegaConf.merge(global_cfg, local_cfg)

    base_root = cfg.paths.get("data_root") if "paths" in cfg else None
    base_root = Path(base_root) if base_root else PROJECT_ROOT
    cfg.paths.data_root = str(base_root)

    def resolve_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(str(value))
        if path.is_absolute():
            return str(path)
        return str(base_root / path)

    for key in ("dataset_dir", "checkpoint_dir", "logs_dir", "hf_cache_dir", "tokenizer_cache_dir"):
        if key in cfg.paths and cfg.paths[key] is not None:
            cfg.paths[key] = resolve_path(cfg.paths[key])

    if "dataset_out" in cfg.data:
        cfg.data.dataset_out = resolve_path(cfg.data.dataset_out)

    if "checkpoint_path" in cfg.pretrained:
        cfg.pretrained.checkpoint_path = resolve_path(cfg.pretrained.checkpoint_path)

    if config_path:
        override_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg


def load_tokenizer(cfg: OmegaConf):
    from transformers import AutoTokenizer

    tok_cfg = cfg.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tok_cfg.name,
        use_fast=bool(tok_cfg.get("use_fast", True)),
        cache_dir=tok_cfg.get("cache_dir"),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def parse_cross_layers(spec) -> Optional[Sequence[int]]:
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = spec.strip()
        if spec.lower() == "all" or spec == "":
            return None
        parts = [p for p in spec.replace(",", " ").split() if p]
        return tuple(int(p) for p in parts)
    if isinstance(spec, Iterable):
        return tuple(int(p) for p in spec)
    return None


def build_model(
    cfg: OmegaConf,
    vocab_size: int,
    *,
    disable_cross: bool = False,
    disable_expander: bool = False,
) -> TRMEncoderDecoder:
    dec = cfg.model.decoder
    enc = cfg.model.encoder
    trm = enc.trm
    comp = cfg.model.compression
    exp = cfg.model.expander
    cross = cfg.model.cross_attention
    rec = enc.get("recursive_updates", {})

    cross_layers = parse_cross_layers(cross.get("layers"))
    cross_enabled = bool(cross.enabled) and not disable_cross
    exp_enabled = bool(exp.enabled) and not disable_expander
    rec_enabled = bool(rec.get("enabled", False))
    rec_stride = rec.get("stride", 0)
    rec_short_l = rec.get("short_L_cycles")
    rec_short_h = rec.get("short_H_cycles")
    rec_keep_y = bool(rec.get("keep_y", False))

    return TRMEncoderDecoder(
        vocab_size=vocab_size,
        decoder_context_length=int(dec.context_length),
        encoder_max_length=int(enc.max_length),
        d_model=int(dec.embedding_size),
        n_heads=int(dec.num_heads),
        n_kv_heads=int(dec.num_kv_heads),
        n_layers=int(dec.num_layers),
        d_ff=int(dec.feed_forward_size),
        rope_dim=int(dec.rope_dim),
        dropout_rate=float(dec.dropout_rate),
        trm_tiny_layers=int(trm.tiny_layers),
        trm_variant=str(trm.variant),
        trm_heads=int(trm.num_heads),
        trm_rope_dim=int(trm.rope_dim),
        trm_d_ff=int(trm.feed_forward_size),
        trm_mixer_hidden=int(trm.mixer_hidden),
        trm_activation=str(trm.activation),
        trm_use_remat=bool(trm.get("use_remat", False)),
        trm_L_cycles=int(trm.recursion.L_cycles),
        trm_H_cycles=int(trm.recursion.H_cycles),
        trm_learned_y=bool(trm.init.learned_y),
        trm_learned_z=bool(trm.init.learned_z),
        trm_init_std=float(trm.init.init_std),
        num_slots=int(enc.num_slots),
        compression_temperature=float(comp.temperature),
        encoder_dropout=float(enc.dropout_rate),
        encoder_update_enabled=rec_enabled,
        encoder_update_stride=int(rec_stride) if rec_stride is not None else 0,
        encoder_update_short_L_cycles=int(rec_short_l) if rec_short_l is not None else None,
        encoder_update_short_H_cycles=int(rec_short_h) if rec_short_h is not None else None,
        encoder_update_keep_y=rec_keep_y,
        expander_enabled=exp_enabled,
        expander_heads=int(exp.num_heads),
        expander_dropout=float(exp.dropout_rate),
        expander_gate_init=float(exp.gate_init),
        cross_attn_enabled=cross_enabled,
        cross_attn_heads=int(cross.num_heads),
        cross_attn_dropout=float(cross.dropout_rate),
        cross_attn_gate_init=float(cross.gate_init),
        cross_attn_layers=cross_layers,
        share_embeddings=bool(cfg.model.get("share_embeddings", False)),
    )


def build_trainable_masks(params, cfg: OmegaConf):
    train_decoder = bool(cfg.training.train_decoder)
    if train_decoder:
        return None, None

    allow = []
    if bool(cfg.training.train_encoder):
        allow += ["soft_moe", "trm_encoder", "encoder_embed", "encoder_norm"]
    if bool(cfg.training.train_cross_attention):
        allow += ["cross_attn", "rms_cross", "slot_expander"]
    if bool(cfg.training.train_embeddings):
        allow += ["Embed_0"]

    if not allow:
        allow = ["__never_match__"]

    flat_params = traverse_util.flatten_dict(params)
    flat_trainable = {}
    for path in flat_params:
        path_str = "/".join(path)
        trainable = any(token in path_str for token in allow)
        flat_trainable[path] = bool(trainable)
    trainable_tree = traverse_util.unflatten_dict(flat_trainable)
    grad_mask = jax.tree_util.tree_map(
        lambda m, p: jnp.ones_like(p, dtype=jnp.float32) if m else jnp.zeros_like(p, dtype=jnp.float32),
        trainable_tree,
        params,
    )
    return trainable_tree, grad_mask


def build_optimizer(cfg: OmegaConf, total_steps: int, params, trainable_tree=None):
    warmup_steps = int(cfg.training.warmup_steps)
    lr = float(cfg.training.learning_rate)
    if warmup_steps > 0 and total_steps > warmup_steps:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=lr * 0.1,
        )
    else:
        schedule = lr

    exclusions = ["bias", "scale", "embedding"]
    mask = unfreeze(create_weight_decay_mask(params, exclusions))
    if trainable_tree is not None:
        mask = jax.tree_util.tree_map(lambda m, t: bool(m) and bool(t), mask, trainable_tree)

    optimizer = optax.chain(
        optax.clip_by_global_norm(float(cfg.training.gradient_clip_norm)),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=float(cfg.training.weight_decay),
            mask=mask,
        ),
    )
    return optimizer


def _gate_stats(params):
    flat = traverse_util.flatten_dict(params)
    cross = []
    expander = []
    for path, value in flat.items():
        name = path[-1]
        if name == "cross_attn_gate":
            cross.append(np.asarray(value))
        elif name == "expander_gate":
            expander.append(np.asarray(value))

    def _summary(vals):
        if not vals:
            return None, None
        raw = float(np.mean([v.mean() for v in vals]))
        gate = float(1.0 / (1.0 + np.exp(-raw)))
        return raw, gate

    cross_raw, cross_gate = _summary(cross)
    exp_raw, exp_gate = _summary(expander)
    return {
        "cross_raw": cross_raw,
        "cross_gate": cross_gate,
        "exp_raw": exp_raw,
        "exp_gate": exp_gate,
    }


def _refresh_training_state(cfg: OmegaConf, params, max_steps: int):
    trainable_tree, grad_mask = build_trainable_masks(params, cfg)
    if grad_mask is not None:
        grad_mask = jax.device_put(grad_mask)
    optimizer = build_optimizer(cfg, max_steps, params, trainable_tree=trainable_tree)
    opt_state = optimizer.init(params)
    return trainable_tree, grad_mask, optimizer, opt_state


def merge_pretrained(params, pretrained_path: Path):
    if not pretrained_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
    pretrained = load_npz(pretrained_path)
    flat_pre = traverse_util.flatten_dict(pretrained)
    flat_params = traverse_util.flatten_dict(params)

    updated = dict(flat_params)
    matched = 0
    for key, value in flat_pre.items():
        if key in flat_params and flat_params[key].shape == value.shape:
            updated[key] = value
            matched += 1
    merged = traverse_util.unflatten_dict(updated)
    return merged, matched, len(flat_pre)


def make_batch(rng: np.random.Generator, ds, batch_size: int):
    idx = rng.integers(0, ds.encoder_tokens.shape[0], size=(batch_size,))
    return {
        "encoder_tokens": jnp.asarray(ds.encoder_tokens[idx], dtype=jnp.int32),
        "encoder_mask": jnp.asarray(ds.encoder_mask[idx], dtype=jnp.bool_),
        "decoder_input": jnp.asarray(ds.decoder_input[idx], dtype=jnp.int32),
        "decoder_target": jnp.asarray(ds.decoder_target[idx], dtype=jnp.int32),
        "decoder_mask": jnp.asarray(ds.decoder_mask[idx], dtype=jnp.float32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("TRM encoder + decoder cross-attn training")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.config)

    if args.data_root is not None:
        cfg.paths.data_root = str(Path(args.data_root).resolve())

    dataset_path = args.dataset or cfg.data.dataset_out
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = Path(cfg.paths.data_root) / dataset_path

    checkpoint_dir = args.checkpoint_dir or cfg.paths.checkpoint_dir
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path(cfg.paths.data_root) / checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(args.batch_size or cfg.training.batch_size)
    max_steps = int(args.max_steps or cfg.training.max_steps)

    seed = int(cfg.training.seed)
    rng_np = np.random.default_rng(seed)
    rng = jax.random.PRNGKey(seed)

    tokenizer = load_tokenizer(cfg)
    vocab_size = len(tokenizer)

    data = load_packed_dataset(dataset_path)
    model = build_model(cfg, vocab_size)
    eval_no_cross = bool(cfg.training.get("eval_no_cross", False))
    no_cross_model = None
    if eval_no_cross:
        no_cross_model = build_model(cfg, vocab_size, disable_cross=True, disable_expander=True)

    dummy_enc = jnp.zeros((1, data.train.encoder_tokens.shape[1]), dtype=jnp.int32)
    dummy_dec = jnp.zeros((1, data.train.decoder_input.shape[1]), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, data.train.encoder_tokens.shape[1]), dtype=jnp.float32)

    key_params, key_dropout = jax.random.split(rng)
    variables = model.init(
        {"params": key_params, "dropout": key_dropout},
        dummy_enc,
        dummy_dec,
        encoder_mask=dummy_mask,
        deterministic=True,
    )
    params = unfreeze(variables["params"])

    if bool(cfg.pretrained.load):
        ckpt_path = Path(cfg.pretrained.checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(cfg.paths.data_root) / ckpt_path
        params, matched, total = merge_pretrained(params, ckpt_path)
        print(f"[pretrained] loaded {matched}/{total} tensors from {ckpt_path}")

    trainable_tree, grad_mask, optimizer, opt_state = _refresh_training_state(cfg, params, max_steps)

    analytics_enabled = bool(cfg.training.get("analytics", False))
    trainable_fraction = None
    if analytics_enabled and trainable_tree is not None:
        flat_trainable = traverse_util.flatten_dict(trainable_tree)
        if flat_trainable:
            trainable_fraction = sum(1 for v in flat_trainable.values() if v) / len(flat_trainable)

    start_step = 0
    if args.resume:
        ckpt_path = latest_ckpt(str(checkpoint_dir)) if args.resume == "latest" else args.resume
        if ckpt_path is None:
            raise FileNotFoundError("No checkpoint found to resume.")
        loaded, step = load_ckpt(ckpt_path)
        params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), loaded)
        start_step = step + 1
        opt_bytes = load_opt_state(step, str(checkpoint_dir))
        if opt_bytes is not None:
            opt_state = serialization.from_bytes(opt_state, opt_bytes)
        print(f"[resume] step {step} from {ckpt_path}")

    params = jax.device_put(params)
    opt_state = jax.device_put(opt_state)

    pbar = tqdm(range(start_step, max_steps), desc="train")
    last_log = time.time()

    unfreeze_after = cfg.training.get("train_decoder_after", None)
    if unfreeze_after is not None:
        unfreeze_after = int(unfreeze_after)

    for step in pbar:
        rng, step_key = jax.random.split(rng)
        batch = make_batch(rng_np, data.train, batch_size)
        params, opt_state, loss, grad_norm = train_step(
            params,
            opt_state,
            batch,
            model=model,
            optimizer=optimizer,
            dropout_rng=step_key,
            grad_mask=grad_mask,
        )

        if (step + 1) % int(cfg.training.log_every) == 0:
            postfix = {"loss": float(loss)}
            if analytics_enabled:
                postfix["grad_norm"] = float(grad_norm)
                gates = _gate_stats(params)
                if gates["cross_gate"] is not None:
                    postfix["cross_gate"] = gates["cross_gate"]
                if gates["exp_gate"] is not None:
                    postfix["exp_gate"] = gates["exp_gate"]
                if trainable_fraction is not None:
                    postfix["train_frac"] = trainable_fraction
            pbar.set_postfix(postfix)
            now = time.time()
            if now - last_log > 0.0:
                last_log = now

        if (step + 1) % int(cfg.training.eval_every) == 0:
            val_batch = make_batch(rng_np, data.val, min(batch_size, data.val.encoder_tokens.shape[0]))
            val_loss = eval_step(params, val_batch, model=model)
            if no_cross_model is not None:
                val_loss_nc = eval_step(params, val_batch, model=no_cross_model)
                print(
                    f"[eval] step={step + 1} loss={float(val_loss):.4f} no_cross={float(val_loss_nc):.4f}"
                )
            else:
                print(f"[eval] step={step + 1} loss={float(val_loss):.4f}")

        if (step + 1) % int(cfg.training.checkpoint_every) == 0:
            ckpt_path = save_ckpt(params, step + 1, str(checkpoint_dir))
            save_opt_state(opt_state, step + 1, str(checkpoint_dir))
            print(f"[ckpt] saved {ckpt_path}")

        if unfreeze_after is not None and (step + 1) == unfreeze_after:
            cfg.training.train_decoder = True
            trainable_tree, grad_mask, optimizer, opt_state = _refresh_training_state(cfg, params, max_steps)
            if analytics_enabled:
                trainable_fraction = None
                if trainable_tree is not None:
                    flat_trainable = traverse_util.flatten_dict(trainable_tree)
                    if flat_trainable:
                        trainable_fraction = sum(1 for v in flat_trainable.values() if v) / len(flat_trainable)
            print(f"[train] unfreezing decoder at step {step + 1}")


if __name__ == "__main__":
    main()

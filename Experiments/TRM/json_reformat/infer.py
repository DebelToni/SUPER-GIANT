from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import core as flax_core
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from TRM import TRM
from common.checkpoint_manager import latest as latest_ckpt
from common.checkpoint_manager import load as load_ckpt
from dataset import decode, encode, load_vocab
from jit_inference import make_jitted_inference


def load_cfg(config_path: str | None):
    cfg = OmegaConf.merge(
        OmegaConf.load(PROJECT_ROOT / "Global_Config.yml"),
        OmegaConf.load(PROJECT_ROOT / "model" / "Config.yml"),
        OmegaConf.load(config_path or (Path(__file__).resolve().parent / "Config.yml")),
    )
    return cfg


def build_model(cfg: OmegaConf, *, vocab_size: int) -> TRM:
    m = cfg.model
    return TRM(
        vocab_size=int(vocab_size),
        context_length=int(m.context_length),
        d_model=int(m.embedding_size),
        tiny_layers=int(m.tiny_layers),
        variant=str(m.variant),
        num_heads=int(m.num_heads),
        rope_dim=int(m.rope_dim),
        d_ff=int(m.feed_forward_size),
        mixer_hidden=int(m.mixer_hidden),
        dropout_rate=float(m.dropout_rate),
        activation=str(m.activation),
        add_positional_embedding=bool(getattr(m, "add_positional_embedding", True)),
        L_cycles=int(m.recursion.L_cycles),
        H_cycles=int(m.recursion.H_cycles),
        max_supervision_steps=int(m.recursion.max_supervision_steps),
        enable_early_stop=bool(m.recursion.enable_early_stop),
        halt_threshold_logit=float(m.recursion.halt_threshold_logit),
        halt_exploration_prob=float(getattr(m.recursion, "halt_exploration_prob", 0.0)),
        no_act_continue=bool(getattr(m.recursion, "no_act_continue", True)),
        aug_enabled=bool(m.augmentation.enabled),
        aug_num_embeddings=int(m.augmentation.num_embeddings),
        aug_default_id=int(m.augmentation.default_id),
    )


def parse_args():
    p = argparse.ArgumentParser("TRM JSON reformat inference")
    p.add_argument("--config", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--checkpoint_dir", default="checkpoints/trm_json_reformat")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--input", required=True, help="Raw JSON-ish string")
    return p.parse_args()


def load_params(ckpt_path: str):
    params_np, step = load_ckpt(ckpt_path)
    params = jax.tree_util.tree_map(lambda x: jnp.asarray(x), params_np)
    if isinstance(params, dict):
        params = flax_core.freeze(params)
    return params, step


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    if args.data_root is not None:
        base_root = Path(str(args.data_root)).resolve()
        if "paths" not in cfg:
            cfg.paths = OmegaConf.create({})
        cfg.paths.data_root = str(base_root)
    else:
        base_root = Path(cfg.paths.data_root) if "paths" in cfg and cfg.paths.get("data_root") else Path.cwd()

    json_cfg = cfg.json_reformat
    vocab_path = Path(json_cfg.vocab_path)
    if not vocab_path.is_absolute():
        vocab_path = (base_root / vocab_path).resolve()
    vocab = load_vocab(vocab_path)

    model = build_model(cfg, vocab_size=len(vocab.chars))

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_dir = Path(args.checkpoint_dir)
        if not ckpt_dir.is_absolute():
            ckpt_dir = (base_root / ckpt_dir).resolve()
        ckpt_path = latest_ckpt(str(ckpt_dir))
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    else:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_absolute():
            ckpt_path = (base_root / ckpt_path).resolve()
        ckpt_path = str(ckpt_path)

    params, step = load_params(ckpt_path)
    print(f"[ckpt] {ckpt_path} (step {step})")

    input_ids, _mask = encode(args.input, vocab, max_len=int(cfg.model.context_length))
    tokens = jnp.asarray(input_ids[None, :], dtype=jnp.int32)

    infer = make_jitted_inference(model)
    logits, q_logit, pred, _state, steps, halted = infer(params, tokens)
    pred_np = jnp.asarray(pred[0]).astype(jnp.int32)
    output = decode(pred_np, vocab)

    print("[raw]")
    print(args.input)
    print("\n[fixed]")
    print(output)
    print(f"\n[q_logit] {float(q_logit[0]):+.3f}  steps={int(steps)} halted={bool(halted[0])}")


if __name__ == "__main__":
    main()

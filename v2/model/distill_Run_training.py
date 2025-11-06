# distill_Run_training.py
import os
import sys
import math
import time
from pathlib import Path
from typing import Optional

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"

try:
    import jax
except ImportError:
    print("JAX not found – installing …")
    os.system("pip install jax[cuda12] transformers datasets flax")
    import jax

import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf

from GiantGPT import GiantGPT
from optimizer_utils import create_weight_decay_mask
from distill_Training_step import train_step
from distill_prepare_dataset import get_data, data_loader
from checkpoint_manager import save as save_ckpt, load as load_ckpt, latest as latest_ckpt

from jax import config as jax_config

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent

CFG = OmegaConf.merge(
    OmegaConf.load(PROJECT_ROOT / "Global_Config.yml"),
    OmegaConf.load(MODEL_DIR / "Config.yml"),
)
MODEL_CFG = CFG.model
QA_CFG = CFG.qa_finetune
TOKENIZER_CFG = CFG.tokenizer

BASE_PREFIX_STR = CFG.paths.get("data_root", "") if "paths" in CFG else ""
BASE_PREFIX = Path(BASE_PREFIX_STR) if BASE_PREFIX_STR else None


def _resolve_path(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute() or BASE_PREFIX is None:
        return str(path)
    return str(BASE_PREFIX / path)


if BASE_PREFIX is not None:
    CFG.paths.data_root = str(BASE_PREFIX)
else:
    CFG.paths.data_root = str(PROJECT_ROOT)

for key in ("processed_data_root", "dataloader_state_root", "logs_root"):
    if key in CFG.paths and CFG.paths[key] is not None:
        resolved = _resolve_path(CFG.paths[key])
        if resolved is not None:
            CFG.paths[key] = resolved

answers_path_resolved = _resolve_path(QA_CFG.get("answers_arrow"))
if answers_path_resolved is not None:
    QA_CFG.answers_arrow = answers_path_resolved

checkpoint_dir_resolved = _resolve_path(QA_CFG.get("checkpoint_dir"))
if checkpoint_dir_resolved is not None:
    QA_CFG.checkpoint_dir = checkpoint_dir_resolved

DEFAULT_QA_CKPT_DIR = QA_CFG.get("checkpoint_dir", "checkpoints_qa")

jax_config.update("jax_default_matmul_precision", "tensorfloat32")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser("SUPER-GIANT QA distillation finetune")
    parser.add_argument(
        "--checkpoint_dir",
        default=str(DEFAULT_QA_CKPT_DIR),
        help="Directory for QA finetune checkpoints",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        help="Override checkpoint interval (steps)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from latest or specific checkpoint",
    )
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        help="Path to base-model checkpoint to initialise from before QA finetune",
    )
    return parser.parse_args()


def _maybe_load_initial_params(path: Optional[str], params):
    if not path:
        return params, 0
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {ckpt_path}")
    loaded_params, step = load_ckpt(str(ckpt_path))
    print(f"▶ Loaded initial parameters from {ckpt_path} (step {step})")
    return loaded_params, step


def main() -> None:
    args = parse_args()

    base_root = Path(CFG.paths.data_root)

    checkpoint_dir_path = Path(args.checkpoint_dir)
    if not checkpoint_dir_path.is_absolute():
        checkpoint_dir_path = (base_root / checkpoint_dir_path).resolve()
    checkpoint_dir = checkpoint_dir_path
    checkpoint_every = int(args.checkpoint_every or QA_CFG.checkpoint_every)
    resume_request = args.resume

    print("» QA finetune configuration:")
    print(OmegaConf.to_yaml(QA_CFG))
    print(f" checkpoint_dir   = {checkpoint_dir}")
    print(f" checkpoint_every = {checkpoint_every}")
    print(f" resume_request   = {resume_request}")
    if args.init_checkpoint:
        print(f" init_checkpoint  = {args.init_checkpoint}")

    context_length = int(QA_CFG.context_length)
    batch_size = int(QA_CFG.batch_size)
    dataset_fraction = float(QA_CFG.get("dataset_fraction", 1.0))

    print("Preparing QA dataset …")
    train_factory, val_factory, tokenizer = get_data(
        subset_pct=dataset_fraction * 100.0,
        context_length=context_length,
        batch_size=batch_size,
    )

    def make_train_loader():
        return data_loader(train_factory())

    def make_val_loader():
        return data_loader(val_factory())

    train_batches = sum(1 for _ in make_train_loader())
    val_batches = sum(1 for _ in make_val_loader())
    print(f"train batches: {train_batches}   val batches: {val_batches}")
    if train_batches == 0:
        raise RuntimeError("No QA training batches available – check answers.arrow and context length")

    model = GiantGPT(
        vocab_size=len(tokenizer),
        context_length=context_length,
        d_model=MODEL_CFG.embedding_size,
        n_heads=MODEL_CFG.num_heads,
        d_ff=MODEL_CFG.feed_forward_size,
        n_layers=MODEL_CFG.num_layers,
        dropout_rate=MODEL_CFG.dropout_rate,
    )

    seed = int(QA_CFG.get("seed", 0))
    rng = jax.random.PRNGKey(seed)
    dummy = jnp.zeros((batch_size, context_length), dtype=jnp.int32)
    params = model.init(rng, dummy, deterministic=True)["params"]

    init_checkpoint_arg = args.init_checkpoint
    if init_checkpoint_arg and resume_request != "latest":
        init_checkpoint_path = Path(init_checkpoint_arg)
        if not init_checkpoint_path.is_absolute():
            init_checkpoint_path = (base_root / init_checkpoint_path).resolve()
        init_checkpoint_arg = str(init_checkpoint_path)

    if init_checkpoint_arg and not resume_request:
        params, _ = _maybe_load_initial_params(init_checkpoint_arg, params)

    steps_per_epoch = train_batches
    total_steps = max(1, steps_per_epoch * int(QA_CFG.num_epochs))
    warmup_ratio = float(QA_CFG.get("warmup_ratio", 0.1))
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    decay_steps = max(1, total_steps - warmup_steps)

    learning_rate = float(QA_CFG.learning_rate)
    end_lr = learning_rate * 0.1

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_lr,
    )
    exclusions = QA_CFG.get("weight_decay_exclusions", [])
    mask = create_weight_decay_mask(params, exclusions) if exclusions else None

    optimizer = optax.chain(
        optax.clip_by_global_norm(QA_CFG.get("gradient_clip_norm", 1.0)),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=float(QA_CFG.get("weight_decay", 0.01)),
            mask=mask,
        ),
    )
    opt_state = optimizer.init(params)
    global_step = 0

    if resume_request and resume_request != "latest":
        resume_path = Path(resume_request)
        if not resume_path.is_absolute():
            resume_request = str((base_root / resume_path).resolve())

    if resume_request:
        if resume_request == "latest":
            ckpt_path = latest_ckpt(str(checkpoint_dir))
            if ckpt_path is None:
                raise FileNotFoundError(f"No checkpoints found in '{checkpoint_dir}' to resume from.")
        else:
            ckpt_path = resume_request
        params, global_step = load_ckpt(ckpt_path)
        opt_state = optimizer.init(params)
        print(f"▶ Resumed QA finetune from {ckpt_path} (step {global_step})")

    log_every = int(QA_CFG.get("log_every", 100))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting QA finetune for {QA_CFG.num_epochs} epochs | "
        f"batch_size={batch_size} | total_steps≈{total_steps}"
    )

    base_rng = jax.random.PRNGKey(seed)

    for epoch in range(int(QA_CFG.num_epochs)):
        t0 = time.time()
        train_iter = make_train_loader()
        for batch in train_iter:
            base_rng, dropout_rng = jax.random.split(base_rng)
            params, opt_state, loss = train_step(
                params,
                opt_state,
                batch,
                model=model,
                optimizer=optimizer,
                dropout_rng=dropout_rng,
            )
            global_step += 1

            if global_step % log_every == 0:
                ppl = float(np.exp(loss)) if loss < 20 else float("inf")
                print(
                    f"step {global_step:>7}/{total_steps:<7} | epoch {epoch+1:<3} | "
                    f"loss {loss:.4f} ppl {ppl:.2f}"
                )

            if global_step % checkpoint_every == 0:
                ckpt_file = save_ckpt(params, global_step, str(checkpoint_dir))
                print(f"💾 checkpoint → {ckpt_file}")

        dt = time.time() - t0
        print(f"epoch {epoch + 1} finished in {dt:.1f}s")

    final_ckpt = save_ckpt(params, global_step, str(checkpoint_dir))
    print(f"✔ QA finetune complete. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    print("Starting QA distillation finetune …")
    main()

from __future__ import annotations

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.92")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from Experiments.Ivancho_Brain.data import TokenBatcher, load_train_val_tokens
from Experiments.Ivancho_Brain.graph_rnn_llm import IvanchoBrain, build_graph_spec, estimate_param_count


def _to_dtype(value: str):
    return getattr(jnp, value) if isinstance(value, str) else value


def make_model(cfg) -> IvanchoBrain:
    graph = build_graph_spec(
        num_heads=int(cfg.model.num_heads),
        avg_out_degree=int(cfg.model.avg_out_degree),
        max_out_degree=int(cfg.model.max_out_degree),
        input_head_count=int(cfg.model.input_head_count),
        output_head_index=int(cfg.model.output_head_index),
        seed=int(cfg.seed),
    )
    return IvanchoBrain(
        vocab_size=int(cfg.model.vocab_size),
        graph=graph,
        num_heads=int(cfg.model.num_heads),
        state_dim=int(cfg.model.state_dim),
        inner_steps=int(cfg.model.inner_steps),
        memory_slots=int(cfg.model.memory_slots),
        memory_attn_heads=int(cfg.model.memory_attn_heads),
        memory_kv_heads=int(cfg.model.memory_kv_heads),
        residual_init=float(cfg.model.residual_init),
        residual_decay=float(cfg.model.residual_decay),
        state_norm_cap=float(cfg.model.state_norm_cap),
        threshold_hi=float(cfg.model.threshold_hi),
        threshold_lo=float(cfg.model.threshold_lo),
        threshold_alpha=float(cfg.model.threshold_alpha),
        emit_beta=float(cfg.model.emit_beta),
        step_penalty=float(cfg.model.step_penalty),
        readiness_delta=float(cfg.model.readiness_delta),
        readiness_tau=float(cfg.model.readiness_tau),
        readiness_loss_weight=float(cfg.model.readiness_loss_weight),
        collapse_token_ids=tuple(int(x) for x in cfg.model.collapse_token_ids),
        collapse_penalty_weight=float(cfg.model.collapse_penalty_weight),
        halt_bias_init=float(cfg.model.halt_bias_init),
        min_emit_step=int(cfg.model.min_emit_step),
        dropout_rate=float(cfg.model.dropout_rate),
        param_dtype=_to_dtype(str(cfg.model.param_dtype)),
        compute_dtype=_to_dtype(str(cfg.model.compute_dtype)),
    )


def create_optimizer(cfg, total_steps: int):
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(cfg.optimizer.learning_rate),
        warmup_steps=int(cfg.optimizer.warmup_steps),
        decay_steps=max(total_steps, int(cfg.optimizer.warmup_steps) + 1),
        end_value=float(cfg.optimizer.min_learning_rate),
    )
    return optax.chain(
        optax.clip_by_global_norm(float(cfg.optimizer.gradient_clip_norm)),
        optax.apply_if_finite(
            optax.adamw(schedule, weight_decay=float(cfg.optimizer.weight_decay)),
            max_consecutive_errors=8,
        ),
    )


def loss_fn(params, batch, readiness_scale, *, model: IvanchoBrain):
    inp = batch[:, :-1]
    target = batch[:, 1:]
    out = model.apply({"params": params}, inp, target, deterministic=False)
    ce = out["ce"]
    emit = out["emit_weight"].astype(jnp.float32)
    halt_logit = out["halt_logit"].astype(jnp.float32)
    halt_prob = out["halt_prob"].astype(jnp.float32)
    collapse_prob = out["collapse_prob"].astype(jnp.float32)
    hard_emit = out["hard_emit"]
    steps = jnp.arange(1, model.inner_steps + 1, dtype=jnp.float32)
    forced_final = jax.nn.one_hot(model.inner_steps - 1, model.inner_steps, dtype=jnp.float32)
    weights = emit + 0.03 + 0.35 * forced_final[None, None, :]
    weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-6)
    emit_weighted_loss = jnp.sum(ce * weights, axis=-1)
    final_loss = ce[:, :, -1]
    token_loss = 0.45 * emit_weighted_loss + 0.55 * final_loss
    expected_step = jnp.sum(weights * steps[None, None, :], axis=-1) / float(model.inner_steps)

    future_best = jnp.flip(
        jax.lax.associative_scan(jnp.minimum, jnp.flip(ce, axis=-1), axis=-1),
        axis=-1,
    )
    future_best_later = jnp.concatenate([future_best[:, :, 1:], ce[:, :, -1:]], axis=-1)
    future_gain = ce - jax.lax.stop_gradient(future_best_later)
    ready_target = jax.nn.sigmoid((model.readiness_delta - jax.lax.stop_gradient(future_gain)) / model.readiness_tau)
    ready_target = ready_target.at[:, :, -1].set(1.0)
    if model.min_emit_step > 1:
        early_mask = (jnp.arange(model.inner_steps) < (model.min_emit_step - 1))[None, None, :]
        ready_target = jnp.where(early_mask, 0.0, ready_target)
    readiness_loss = optax.sigmoid_binary_cross_entropy(halt_logit, ready_target)
    collapse_ids = jnp.asarray(model.collapse_token_ids, dtype=jnp.int32)
    target_is_collapse = jnp.any(target[:, :, None] == collapse_ids[None, None, :], axis=-1).astype(jnp.float32)
    collapse_loss = jnp.sum(collapse_prob * weights, axis=-1) * (1.0 - target_is_collapse)
    loss = (
        jnp.mean(token_loss + model.step_penalty * expected_step)
        + readiness_scale * model.readiness_loss_weight * jnp.mean(readiness_loss)
        + model.collapse_penalty_weight * jnp.mean(collapse_loss)
    )

    any_emit = jnp.any(hard_emit, axis=-1)
    first_emit_idx = jnp.argmax(hard_emit, axis=-1).astype(jnp.float32) + 1.0
    avg_emit_step = jnp.sum(jnp.where(any_emit, first_emit_idx, 0.0)) / jnp.maximum(jnp.sum(any_emit.astype(jnp.float32)), 1.0)
    pred = out["pred"]
    final_pred = pred[:, :, -1]
    first_emit_idx_i = jnp.argmax(hard_emit, axis=-1)
    chosen_pred = jnp.take_along_axis(pred, first_emit_idx_i[:, :, None], axis=-1).squeeze(-1)
    chosen_pred = jnp.where(any_emit, chosen_pred, final_pred)
    acc = jnp.mean((final_pred == target).astype(jnp.float32))
    chosen_acc = jnp.mean((chosen_pred == target).astype(jnp.float32))
    metrics = {
        "loss": loss,
        "ce": jnp.mean(token_loss),
        "final_ce": jnp.mean(final_loss),
        "emit_ce": jnp.mean(emit_weighted_loss),
        "ready_loss": jnp.mean(readiness_loss),
        "ready_scale": readiness_scale,
        "ready_target": jnp.mean(ready_target),
        "collapse_loss": jnp.mean(collapse_loss),
        "collapse_prob": jnp.mean(collapse_prob),
        "acc_final": acc,
        "acc_chosen": chosen_acc,
        "conf": jnp.mean(out["confidence"]),
        "halt_prob": jnp.mean(halt_prob),
        "threshold": jnp.mean(out["threshold"]),
        "emit": jnp.mean(emit),
        "hard_emit_rate": jnp.mean(hard_emit.astype(jnp.float32)),
        "forced_rate": 1.0 - jnp.mean(any_emit.astype(jnp.float32)),
        "avg_emit_step": avg_emit_step,
        "active": jnp.mean(out["active_count"]),
        "expected_step": jnp.mean(expected_step) * float(model.inner_steps),
    }
    return loss, metrics


@partial(jax.jit, static_argnames=("model", "optimizer"))
def train_step(params, opt_state, batch, readiness_scale, *, model: IvanchoBrain, optimizer):
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch, readiness_scale, model=model)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, metrics


@partial(jax.jit, static_argnames=("model",))
def eval_step(params, batch, *, model: IvanchoBrain):
    _, metrics = loss_fn(params, batch, jnp.asarray(1.0, dtype=jnp.float32), model=model)
    return metrics


def readiness_scale_for_step(cfg, step: int) -> float:
    start = int(cfg.training.readiness_warmup_start)
    span = max(int(cfg.training.readiness_warmup_steps), 1)
    return float(np.clip((step - start) / span, 0.0, 1.0))


def save_checkpoint(path: Path, params, opt_state, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "params.msgpack").write_bytes(serialization.to_bytes(params))
    (path / "opt_state.msgpack").write_bytes(serialization.to_bytes(opt_state))
    (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def format_metrics(metrics: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("Config.yml")))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.smoke:
        cfg.training.batch_size = 2
        cfg.training.train_steps = 2
        cfg.data.seq_len = 16
        cfg.model.inner_steps = 3
        cfg.model.num_heads = 16
        cfg.model.input_head_count = 4
        cfg.model.state_dim = 32
        cfg.model.memory_attn_heads = 4
        cfg.model.avg_out_degree = 3
        cfg.model.max_out_degree = 5
    if args.steps is not None:
        cfg.training.train_steps = args.steps

    if args.smoke and not Path(cfg.paths.tokenized_path).exists():
        rng_np = np.random.default_rng(int(cfg.seed))
        synthetic = rng_np.integers(0, int(cfg.model.vocab_size), size=8192, dtype=np.uint16)
        train_tokens, val_tokens = synthetic[:-1024], synthetic[-1024:]
    else:
        train_tokens, val_tokens = load_train_val_tokens(args.config)
    train_loader = TokenBatcher(train_tokens, seq_len=int(cfg.data.seq_len), batch_size=int(cfg.training.batch_size), seed=int(cfg.seed))
    val_loader = TokenBatcher(val_tokens, seq_len=int(cfg.data.seq_len), batch_size=int(cfg.training.batch_size), seed=int(cfg.seed) + 1)
    model = make_model(cfg)
    rng = jax.random.PRNGKey(int(cfg.seed))
    dummy = jnp.asarray(train_loader.next_batch()[:, :-1])
    params = model.init(rng, dummy, deterministic=True)["params"]
    optimizer = create_optimizer(cfg, int(cfg.training.train_steps))
    opt_state = optimizer.init(params)
    param_count = estimate_param_count(params)
    print(f"device={jax.default_backend()} params={param_count:,}")

    ckpt_root = Path(cfg.paths.checkpoint_dir)
    log_dir = Path(cfg.paths.logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    last = start
    for step in tqdm(range(1, int(cfg.training.train_steps) + 1), desc="training"):
        batch = jnp.asarray(train_loader.next_batch())
        readiness_scale = jnp.asarray(readiness_scale_for_step(cfg, step), dtype=jnp.float32)
        params, opt_state, metrics = train_step(params, opt_state, batch, readiness_scale, model=model, optimizer=optimizer)
        if step % int(cfg.training.log_every) == 0 or step == 1:
            host_metrics = {k: float(v) for k, v in jax.device_get(metrics).items()}
            now = time.time()
            tok_s = int(cfg.training.batch_size) * int(cfg.data.seq_len) * (step if step == 1 else int(cfg.training.log_every)) / max(now - last, 1e-6)
            last = now
            print(f"step={step} tok/s={tok_s:.0f} {format_metrics(host_metrics)}", flush=True)
        if step % int(cfg.training.eval_every) == 0:
            eval_metrics = []
            for _ in range(8):
                eval_metrics.append(jax.device_get(eval_step(params, jnp.asarray(val_loader.next_batch()), model=model)))
            mean_metrics = {k: float(np.mean([m[k] for m in eval_metrics])) for k in eval_metrics[0]}
            print(f"eval step={step} {format_metrics(mean_metrics)}", flush=True)
        if step % int(cfg.training.checkpoint_every) == 0 or step == int(cfg.training.train_steps):
            save_checkpoint(
                ckpt_root / f"step_{step:06d}",
                params,
                opt_state,
                {"step": step, "params": param_count, "elapsed_sec": time.time() - start},
            )


if __name__ == "__main__":
    main()

# Training_step.py
import jax, jax.numpy as jnp, optax
from functools import partial
from omegaconf import OmegaConf

Config = OmegaConf.load("Config.yml")

def _kd_loss_topk(student_logits, topk_ids, topk_logprobs, mask):
    """
    student_logits: (B, T, V)
    topk_ids:      (B, T, K) int32  (-1 where N/A)
    topk_logprobs: (B, T, K) float32 (-inf where N/A)  -- teacher logprobs (per-token)
    mask:          (B, T) float32  -- 1.0 where answer tokens

    Numerically safe: if a position has *no* valid teacher candidates,
    its KD contribution is forced to zero (no NaNs).
    """
    temp = float(getattr(Config, "distill_temperature", 2.0))
    # Guard: gather logits for teacher indices; invalid ids set to a large negative
    safe_ids = jnp.maximum(topk_ids, 0)              # (B, T, K)
    student_k = jnp.take_along_axis(student_logits, safe_ids, axis=-1)  # (B, T, K)
    valid_k   = (topk_ids >= 0)                      # (B, T, K) bool
    # Replace invalid entries by a large negative (not -inf to avoid all-(-inf))
    student_k = jnp.where(valid_k, student_k, -1e9)

    # Temperature-softmax student over K
    s_log_probs = jax.nn.log_softmax(student_k / temp, axis=-1)        # (B, T, K)

    # Teacher: normalize only over valid entries; if none are valid, make KL=0 later
    t_logp_raw = topk_logprobs / temp                                  # (B, T, K)
    t_logp_raw = jnp.where(valid_k, t_logp_raw, -jnp.inf)              # mask invalid
    # logsumexp returns -inf if all inputs are -inf; handle that with a safe fallback
    t_logZ = jax.scipy.special.logsumexp(t_logp_raw, axis=-1, keepdims=True)  # (B, T, 1)
    # Positions with no valid candidates → t_logZ = -inf. Create a boolean mask.
    has_any = jnp.any(valid_k, axis=-1)                                 # (B, T) bool
    # Safe normalized teacher log-probs: where no valid, copy student's log-probs so KL=0
    t_log_probs = t_logp_raw - t_logZ
    t_log_probs = jnp.where(has_any[..., None], t_log_probs, s_log_probs)
    t_probs     = jnp.exp(t_log_probs)

    # KL(teacher || student) over K
    kl = jnp.sum(t_probs * (t_log_probs - s_log_probs), axis=-1)       # (B, T)

    # Only answer tokens contribute, and only where teacher had any candidates
    eff_mask = mask * has_any.astype(jnp.float32)                       # (B, T)
    kd = jnp.sum(kl * eff_mask) / (jnp.sum(eff_mask) + 1e-9)
    return (temp * temp) * kd

@partial(jax.jit, static_argnames=['model','optimizer'])
def train_step(params, opt_state, batch, *, model, optimizer, dropout_rng):
    """
    batch: dict with
      - input: (B, T) int32
      - target: (B, T) int32
      - mask: (B, T) float32
      - topk_ids: (B, T, K) int32
      - topk_logprobs: (B, T, K) float32
    """
    def loss_fn(p):
        logits = model.apply(
            {"params": p},
            batch["input"],
            rngs={"dropout": dropout_rng},
            deterministic=False,
        )  # (B, T, V)

        # Teacher-free CE on ground truth
        ce = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch["target"]
        )  # (B, T)
        ce = (ce * batch["mask"]).sum() / (batch["mask"].sum() + 1e-9)

        if getattr(Config, "use_distillation", True):
            kd = _kd_loss_topk(
                logits, batch["topk_ids"], batch["topk_logprobs"], batch["mask"]
            )
            w = float(getattr(Config, "distill_weight", 0.5))
            loss = (1.0 - w) * ce + w * kd
        else:
            loss = ce
        return loss

    (loss, grads) = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss


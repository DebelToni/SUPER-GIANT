# Training_step.py
import jax, jax.numpy as jnp, optax
from functools import partial
from omegaconf import OmegaConf

Config = OmegaConf.load("Config.yml")

def _kd_loss_topk(student_logits, topk_ids, topk_logprobs, mask):
    """
    student_logits: (B, T, V)
    topk_ids:      (B, T, K) int32 (-1 where N/A)
    topk_logprobs: (B, T, K) float32 (-inf where N/A)  -- teacher logprobs (per-token)
    mask:          (B, T) float32  -- 1.0 where answer tokens
    """
    T = float(getattr(Config, "distill_temperature", 2.0))
    # gather student logits on teacher's K ids
    # replace -1 by 0 to avoid OOB, and then mask them out
    safe_ids = jnp.maximum(topk_ids, 0)
    B, L, K = safe_ids.shape
    V = student_logits.shape[-1]

    # gather: (B, T, K)
    idx = jnp.expand_dims(safe_ids, axis=-1)  # (B, T, K, 1)
    # one-hot gather without allocating full one-hot: use take_along_axis
    student_k = jnp.take_along_axis(student_logits, safe_ids, axis=-1)  # (B, T, K)

    # mask out invalid entries (-1)
    valid_k = (topk_ids >= 0)
    student_k = jnp.where(valid_k, student_k, -jnp.inf)

    # temperature softmax on both sides
    s_log_probs = jax.nn.log_softmax(student_k / T, axis=-1)
    # teacher probs from provided logprobs
    t_log_probs = (topk_logprobs / T)
    # renormalize teacher over provided K only
    t_logZ = jax.scipy.special.logsumexp(t_log_probs, axis=-1, keepdims=True)
    t_log_probs = t_log_probs - t_logZ
    t_probs = jnp.exp(t_log_probs)

    # KL(teacher || student) over K
    kl = jnp.sum(t_probs * (t_log_probs - s_log_probs), axis=-1)  # (B, T)

    # mask: only answer tokens where KD exists (at least one valid_k)
    has_any = jnp.any(valid_k, axis=-1)  # (B, T) bool
    eff_mask = mask * has_any.astype(jnp.float32)

    kd = jnp.sum(kl * eff_mask) / (jnp.sum(eff_mask) + 1e-9)
    # Hinton scaling T^2
    kd = (T * T) * kd
    return kd

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


import optax
import jax
from functools import partial

from Data_loader import data_loader


@partial(jax.jit, static_argnames="model")
def _loss_on_batch(params, batch, *, model):
    logits = model.apply({"params": params}, batch["input"], deterministic=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["target"])
    loss = (loss * batch["mask"]).sum() / batch["mask"].sum()
    return loss

def evaluate(params, model, dataset_tokens, batch_size=32, *, use_scan=False):
    batches = list(data_loader(dataset_tokens, batch_size, shuffle=False))
    if not batches:
        return 0.0

    if use_scan:
        full_batches = [b for b in batches if b["input"].shape[0] == batch_size]
        if len(full_batches) != len(batches):
            print("Warning: dropping last incomplete batch for scan")
        stacked = jax.tree_util.tree_map(lambda *xs: jax.numpy.stack(xs), *full_batches)

        def step(total, batch):
            loss = _loss_on_batch(params, batch, model=model)
            return total + loss, None

        total, _ = jax.lax.scan(step, 0.0, stacked)
        return float(total / len(full_batches))

    total, n = 0.0, 0
    for batch in batches:
        loss = _loss_on_batch(params, batch, model=model)
        total += float(loss);  n += 1
    return total / n


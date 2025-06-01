import optax
from Data_loader import data_loader

def evaluate(params, model, dataset_tokens, batch_size=32):
    total, n = 0.0, 0
    for batch in data_loader(dataset_tokens, batch_size, shuffle=False):
        logits = model.apply({"params": params}, batch["input"], deterministic=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["target"])
        loss = (loss * batch["mask"]).sum() / batch["mask"].sum()
        total += float(loss);  n += 1
    return total / n


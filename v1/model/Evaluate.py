# def evaluate(params, dataset_tokens):
#     total_loss = 0.0
#     count = 0
#     for batch in data_loader(dataset_tokens, batch_size=32):  # smaller batch if needed
#         logits = model.apply({'params': params}, batch['input'], deterministic=True)
#         batch_loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['target'])
#         mask = batch['mask']
#         batch_loss = (batch_loss * mask).sum() / mask.sum()
#         total_loss += float(batch_loss)
#         count += 1
#     return total_loss / count
#
# val_loss = evaluate(params, val_data_tokens)
# print(f"Validation loss: {val_loss}, Perplexity: {np.exp(val_loss)}")

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


from Data_loader import data_loader
from Training_step import train_step
from Evaluate import evaluate
from Save_params import save_params

num_epochs = 1  # you might do multiple passes, but even 1 full pass over 2M examples might be enough for convergence on TinyStories
batch_size = 64

for epoch in range(num_epochs):
    for batch in data_loader(train_data_tokens, batch_size):
        params, opt_state, loss = train_step(params, opt_state, batch)
        if i % 1000 == 0:
            print(f"Batch {i}, Loss: {loss}")
    # (Optionally, evaluate on val set here)
    print(f"Epoch {epoch} done.")


val_loss = evaluate(params, val_data_tokens)
print(f"Validation loss: {val_loss}, Perplexity: {np.exp(val_loss)} (lower is better)") 

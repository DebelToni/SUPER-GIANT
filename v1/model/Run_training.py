import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # allocate on demand
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"   # or 0.9, any < 1.0
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"  # use platform allocator

# run before importing JAX
# os.environ["JAX_DEFAULT_DTYPE_BITS"] = "16"
os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"   


# run_training.py
import jax, jax.numpy as jnp, optax, Config
# from Transformer_block import TinyTransformerBlock
from GiantGPT import GiantGPT
from Training_step    import train_step
from Evaluate         import evaluate
from Data_loader      import data_loader
from Save_params      import save_params
from prepare_dataset  import get_data
import numpy as np, time, pathlib, pickle, functools

def main():
    for k, v in Config.__dict__.items():
        if not k.startswith("__") and not callable(v):
            print(f"{k:>20} = {v}")


    print("Setting up JAX...")
    # train_tokens, val_tokens, tokenizer = get_data()
    train_tokens, val_tokens, tokenizer = get_data(
        subset_pct = Config.dataset_percent,
        chunk_pct  = Config.chunk_percent,
        context_length = Config.context_length)
    print(f"train batches: {len(train_tokens)}  val batches: {len(val_tokens)}")
    # print train_tokens' shape:
    print(f"train_tokens shape: {train_tokens.shape}  val_tokens shape: {val_tokens.shape}")
    print(Config.num_epochs * len(train_tokens) // Config.batch_size, "total steps")

    # model = TinyTransformerBlock(
    model = GiantGPT(
        vocab_size = Config.vocab_size,
        context_length    = Config.context_length,
        d_model    = Config.embedding_size,
        n_heads    = Config.num_heads,
        d_ff       = Config.feed_forward_size,
        n_layers   = Config.num_layers,
        dropout_rate = Config.dropout_rate,
        # dtype = getattr(jnp, Config.dtype), 
    )

    # ----- initialise params & optimiser -----
    print("Initialising model parameters and optimizer...")
    rng    = jax.random.PRNGKey(0)
    dummy  = jnp.zeros((1, Config.context_length), dtype=jnp.int32)
    print("dummy:", dummy.shape, "d_model:", model.d_model)
    params = model.init(rng, dummy)["params"]
    save_params(params, "initial_params.pkl")

    optimizer = optax.adamw(Config.learning_rate, weight_decay=Config.weight_decay)
    opt_state = optimizer.init(params)

    # ----- training loop -----
    global_step = 0
    print(f"Training for {Config.num_epochs} epochs with {Config.batch_size} batch size")
    rng = jax.random.PRNGKey(0)
    for epoch in range(Config.num_epochs):
        for batch in data_loader(train_tokens, Config.batch_size):
            rng, dropout_rng = jax.random.split(rng)
            params, opt_state, loss = train_step(
                params, opt_state, batch,
                model=model, optimizer=optimizer, dropout_rng=dropout_rng
            )

            global_step += 1
            if global_step % 200 == 0:
                # print(f"step {global_step:>7} | loss {loss:.4f}")
                print(f"step {global_step:>7} out of {Config.num_epochs * len(train_tokens) // Config.batch_size:>7} | loss {loss:.4f}  ppl {np.exp(loss):.2f}")

        # --- evaluate ---
        val_loss = evaluate(params, model, val_tokens)
        print(f"✓ Epoch {epoch+1} done – val loss {val_loss:.4f}  ppl {np.exp(val_loss):.2f}")

    # save everything
    save_params(params)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("✔ parameters & tokenizer saved")

if __name__ == "__main__":
    print("Starting training...")
    main()


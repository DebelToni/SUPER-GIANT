import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # allocate on demand
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"   # or 0.9, any < 1.0

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

if __name__ == "__main__":
    print("Starting training...")
    main()


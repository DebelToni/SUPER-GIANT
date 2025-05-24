import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # allocate on demand
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"   # or 0.9, any < 1.0

# run_training.py
import jax, jax.numpy as jnp, optax, Config
from Transformer_block import TinyTransformerLM
from Training_step    import train_step
from Evaluate         import evaluate
from Data_loader      import data_loader
from Save_params      import save_params
from prepare_dataset  import get_data
import numpy as np, time, pathlib, pickle, functools

def main():
    print("Setting up JAX...")
    train_tokens, val_tokens, tokenizer = get_data()
    print(f"train batches: {len(train_tokens)}  val batches: {len(val_tokens)}")

    model = TinyTransformerLM(
        vocab_size = Config.vocab_size,
        max_len    = Config.context_length,
        d_model    = Config.embedding_size,
        n_heads    = Config.num_heads,
        d_ff       = Config.feed_forward_size,
        n_layers   = Config.num_layers,
        # dtype = getattr(jnp, Config.dtype), 
    )

    # ----- initialise params & optimiser -----
    print("Initialising model parameters and optimizer...")
    rng    = jax.random.PRNGKey(0)
    dummy  = jnp.zeros((1, Config.context_length), dtype=jnp.int32)
    params = model.init(rng, dummy)["params"]

    optimizer = optax.adamw(Config.learning_rate, weight_decay=Config.weight_decay)
    opt_state = optimizer.init(params)

    # ----- training loop -----
    global_step = 0
    print(f"Training for {Config.num_epochs} epochs with {Config.batch_size} batch size")
    rng = jax.random.PRNGKey(0)
    for epoch in range(Config.num_epochs):
        # half the batch size and iterate over to decreese memory usage
        # grad_acc = None; step_in_acc = 0
        for batch in data_loader(train_tokens, Config.batch_size):
            rng, dropout_rng = jax.random.split(rng)
            params, opt_state, loss = train_step(
                params, opt_state, batch,
                model=model, optimizer=optimizer, dropout_rng=dropout_rng
            )

            # grads = jax.grad(loss_fn)(params)
            # grad_acc = jax.tree_util.tree_add(grad_acc, grads) if grad_acc else grads
            # step_in_acc += 1
            # if step_in_acc == Config.acc_steps:
            #     updates, opt_state = optimizer.update(
            #         jax.tree_util.tree_map(lambda g: g / Config.acc_steps, grad_acc),
            #         opt_state, params)
            #     params = optax.apply_updates(params, updates)
            #     grad_acc = None; step_in_acc = 0

    # for epoch in range(Config.num_epochs):
    #     for batch in data_loader(train_tokens, Config.batch_size):
            # params, opt_state, loss = train_step(
            #     params, opt_state, batch, model=model, optimizer=optimizer)
            global_step += 1
            if global_step % 200 == 0:
                print(f"step {global_step:>7} | loss {loss:.4f}")

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


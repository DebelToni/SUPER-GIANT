import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.80"  
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# os.environ["JAX_DEFAULT_DTYPE_BITS"] = "32"   

try:
    import jax
except ImportError:
    print("JAX is not installed. Installing JAX...")
    os.system("pip install jax[cuda12] transformers datasets flax")
    import jax

import optax
# import Config
from omegaconf import OmegaConf
Config = OmegaConf.load("Config.yml")

import jax.numpy as jnp
from transformers import AutoTokenizer

from GiantGPT import GiantGPT
from Training_step    import train_step
from Evaluate         import evaluate
from Data_loader      import data_loader
from Save_params      import save_params
from prepare_dataset  import get_data
import numpy as np, time, pathlib, pickle, functools, math


def main():
    for k, v in Config.__dict__.items():
        if not k.startswith("__") and not callable(v):
            print(f"{k:>20} = {v}")


    print("Setting up JAX...")
    train_tokens, val_tokens, tokenizer = get_data(
        subset_pct = Config.dataset_percent,
        # chunk_pct  = Config.chunk_percent,
        context_length = Config.context_length)


    print(f"train batches: {len(train_tokens)}  val batches: {len(val_tokens)}")
    print(f"train_tokens shape: {train_tokens.shape}  val_tokens shape: {val_tokens.shape}")
    print(math.ceil(len(train_tokens) / Config.batch_size) * Config.num_epochs,
          "total steps")

    model = GiantGPT(
        # vocab_size = Config.vocab_size,
        vocab_size = AutoTokenizer.from_pretrained(Config.tokenizer_name).vocab_size,
        context_length    = Config.context_length,
        d_model    = Config.embedding_size,
        n_heads    = Config.num_heads,
        d_ff       = Config.feed_forward_size,
        n_layers   = Config.num_layers,
        dropout_rate = Config.dropout_rate,
    )

    print("Initialising model parameters and optimizer...")
    rng    = jax.random.PRNGKey(0)
    dummy  = jnp.zeros((1, Config.context_length), dtype=jnp.int32)
    print("dummy:", dummy.shape, "d_model:", model.d_model)
    # cpu = jax.devices("cpu")[0]
    # with jax.default_device(cpu):
        # params = model.init(rng, dummy)["params"]
    params = model.init(rng, dummy)["params"]
    save_params(params, "initial_params.pkl")

    ### optimizer = optax.adamw(Config.learning_rate, weight_decay=Config.weight_decay)

    # tokens_per_epoch = train_tokens.shape[0] // Config.batch_size
    # total_steps      = Config.num_epochs * tokens_per_epoch

    # schedule = optax.warmup_cosine_decay_schedule(
    #     init_value=0.0,          
    #     peak_value=Config.learning_rate,
    #     warmup_steps=500,
    #     decay_steps=total_steps - 500,
    # )

    steps_per_epoch = math.ceil(len(train_tokens) / Config.batch_size)
    total_steps     = steps_per_epoch * Config.num_epochs

    assert total_steps > 0, "total_steps must be positive"
    assert total_steps > 500, (
        f"warmup ({500}) >= total_steps ({total_steps}); "
        "shorten warmup_steps or train longer."
    )

    # ----------------------------------------------------------
    # 2. Build the schedule
    # ----------------------------------------------------------
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=Config.learning_rate,
        warmup_steps=500,
        decay_steps=total_steps - 500,
        end_value=Config.learning_rate * 0.1,   # optional but recommended
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),        
        optax.adamw(
            learning_rate=schedule,
            b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1,
        ),
    )
    opt_state = optimizer.init(params)

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
                print(f"step {global_step:>7} out of {Config.num_epochs * len(train_tokens) // Config.batch_size:>7} | loss {loss:.4f}  ppl {np.exp(loss):.2f}")

        val_loss = evaluate(params, model, val_tokens)
        print(f"✓ Epoch {epoch+1} done – val loss {val_loss:.4f}  ppl {np.exp(val_loss):.2f}")

    save_params(params)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("✔ parameters & tokenizer saved")

if __name__ == "__main__":
    print("Starting training...")
    main()

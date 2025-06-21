# GiantGPT Architecture Overview

This document summarises the structure of the decoder-only language model implemented in this repository and how training and inference are currently handled.

## Configuration

Model hyperparameters are defined in `Config.py`:

```
dtype = jnp.bfloat16
compute_dtype = jnp.bfloat16
param_dtype = jnp.float32
embedding_size = 256
context_length = 257
num_heads = 2
num_layers = 2
feed_forward_size = 4 * embedding_size
```

These values set the default architecture when building the model.

## Model architecture

`GiantGPT` is a Flax module defined in `GiantGPT.py`. The network consists of an embedding table, a stack of decoder transformer blocks, and a tied output projection via `jnp.einsum` with the embedding weights. Positional information is encoded with rotary embeddings inside each attention block and no standalone positional table is used.

The main forward method:

```
@nn.compact
def __call__(self, tokens, *, deterministic=False, decode=False, cur_index=None):
    embed = nn.Embed(
        num_embeddings=self.vocab_size,
        features=self.d_model,
        embedding_init=nn.initializers.normal(stddev=0.02),
        dtype=Config.compute_dtype,
        param_dtype=Config.param_dtype,
    )
    x = embed(tokens)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    for _ in range(self.n_layers):
        x = TinyTransformerBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout_rate=self.dropout_rate,
            dtype=Config.compute_dtype,
        )(x, deterministic=deterministic, decode=decode, cur_index=cur_index)
    logits = jnp.einsum("bld,vd->blv", x.astype(jnp.float32), embed.embedding)
    return logits
```

### Transformer blocks

`TinyTransformerBlock` in `Transformer_block.py` uses RMSNorm, a rotary‑aware self‑attention layer and a gated feed‑forward network. Attention parameters are projected in bfloat16 and stored in float32. During decoding, key and value tensors are stored in a cache of full length (`context_length`) for fast autoregressive generation.

The feed‑forward portion implements a SiLU‑gated module:

```
 gate_dim = d_ff // 3
 proj_dim = gate_dim * 2
 h_proj = nn.Dense(proj_dim, name="fc1", dtype=module.dtype, param_dtype=Config.param_dtype)(h_norm)
 u, v = jnp.split(h_proj, 2, axis=-1)
 h_ffn = nn.silu(u) * v
 h_ffn = nn.Dense(module.d_model, name="fc2", dtype=module.dtype, param_dtype=Config.param_dtype)(h_ffn)
```

Self‑attention is implemented via `NativeJaxSelfAttention` which uses `jax.nn.dot_product_attention` with Flash‑attention when available. It supports multi‑query keys/values by repeating a reduced set of KV heads across all attention heads. Rotary positional embeddings are applied to queries and keys before attention.

## Training procedure

`Run_training.py` orchestrates dataset preparation, model initialisation and the optimization loop. Data is packed using `prepare_dataset.get_data`, which streams **WikiText‑2** and packages it into fixed‑length windows with 50% overlap. Batches are produced by `Data_loader.data_loader` which yields dictionaries of `input`, `target` and `mask` arrays.

The optimiser is `optax.adamw` with a warmup‑cosine schedule:

```
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=Config.learning_rate,
    warmup_steps=500,
    decay_steps=total_steps - 500,
    end_value=Config.learning_rate * 0.1,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.1),
)
```

One training step is defined in `Training_step.py` and jitted for speed:

```
@partial(jax.jit, static_argnames=['model', 'optimizer'])
def train_step(params, opt_state, batch, *, model, optimizer, dropout_rng):
    def loss_fn(p):
        logits = model.apply({"params": p}, batch["input"], rngs={"dropout": dropout_rng}, deterministic=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch["target"])
        loss = (loss * batch["mask"]).sum() / batch["mask"].sum()
        return loss
    (loss, grads) = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss
```

## Inference and generation

Two helper scripts provide generation utilities:

* `Generate_text.py` – a simple loop that feeds tokens to the model one step at a time.
* `Generate_text_fast.py` – an optimised version that preallocates caches and runs a JIT‑compiled step function inside `jax.lax.scan`.

During inference, the model is called with `decode=True` so that each transformer block updates its key/value cache. The cache is a collection named `"cache"` holding tensors of shape `(batch, num_heads, context_length, head_dim)`.

Sampling uses top‑k and temperature scaling as shown in the step function:

```
logits, new_vars = model.apply({"params": params, "cache": cache}, prev_token,
                               deterministic=True, decode=True,
                               cur_index=cur_index, rngs={"dropout": rng},
                               mutable=["cache"])
cache = new_vars["cache"]
logits = logits[:, 0]
if temperature == 0.0:
    next_token = jnp.argmax(logits, axis=-1)
else:
    logits = logits / temperature
    if top_k and top_k > 0:
        values, _ = jax.lax.top_k(logits, top_k)
        kth = values[:, -1][:, None]
        logits = jnp.where(logits < kth, -jnp.inf, logits)
    next_token = jax.random.categorical(rng, logits, axis=-1)
```

## Evaluation

`Evaluate.py` computes average cross‑entropy over a validation set. It optionally supports `jax.lax.scan` for efficiency.

## Summary

The current implementation defines a lightweight GPT‑style model in Flax/JAX with rotary attention, multi‑query keys/values, SiLU‑gated feed‑forwards, RMSNorm and a tied embedding projection. Training uses AdamW with a warmup‑cosine schedule and mixed precision (`bfloat16` compute with `float32` parameters). Inference relies on a cache of full‑length key/value tensors and can stream tokens efficiently via a JIT‑compiled step function.

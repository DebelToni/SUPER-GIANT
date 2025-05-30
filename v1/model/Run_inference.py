# Run_inference.py         NEW -------------------------------------------
import functools, argparse
from pathlib import Path
import jax, jax.numpy as jnp
from transformers import AutoTokenizer

import Config, GiantGPT

def load_params(path: Path, dtype):
    import pickle, gzip
    with path.open("rb") as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: x.astype(dtype), params)

# -----------------------------------------------------------------------
@functools.partial(
    jax.jit,
    static_argnums=(4, 5, 6),              # steps / temperature / top_k
    donate_argnums=(1,)                    # donate the cache
)
def generate_scan(params, cache, rng, prompt_ids, steps, temperature, top_k):
    """Fused lax.scan autoregressive loop – no Python in the hot path."""

    def one_step(state, _):
        cache, rng, last_id = state
        rng, sub = jax.random.split(rng)

        logits, cache = model.apply({"params": params},
                                    last_id[None, None],    # shape [1, 1]
                                    cache=cache,
                                    deterministic=True)

        logits = logits.squeeze(0).squeeze(0)               # [V]
        if top_k:
            kth, top_idx = jax.lax.top_k(logits, top_k)
            logits = logits.at[:].set(-jnp.inf)
            logits = logits.at[top_idx].set(kth)

        next_id = jax.random.categorical(sub, logits / temperature, axis=-1)
        return (cache, rng, next_id), next_id

    # seed ----
    last_id = prompt_ids[-1]
    (cache, rng, _), generated = jax.lax.scan(
        one_step,
        (cache, rng, jnp.array(last_id, jnp.int32)),
        xs=None,
        length=steps,
        unroll=4                                     # good speed on Ampere+
    )
    return generated

# -----------------------------------------------------------------------

def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--prompt", type=str, default="The quick brown fox")
    cli.add_argument("--params", type=Path, required=True)
    cli.add_argument("--steps", type=int, default=50)
    cli.add_argument("--temperature", type=float, default=0.8)
    cli.add_argument("--top_k", type=int, default=40)
    cli.add_argument("--device", type=str, default=Config.default_device)
    args = cli.parse_args()

    # ----- JAX platform --------------------------------------------------
    jax.config.update("jax_platform_name", args.device)

    # ----- tokenizer -----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M",
                                              pad_token="<|pad|>",
                                              eos_token="</s>")

    prompt_ids = tokenizer(args.prompt, add_special_tokens=False)["input_ids"]
    prompt_ids = jnp.array(prompt_ids, jnp.int32)

    # ----- model + params + cache ---------------------------------------
    global model                                            # used inside jit
    model = GiantGPT.GiantGPT()
    params = load_params(args.params, Config.dtype)

    cache = model.init_cache(batch_size=1,
                             max_length=Config.context_length,
                             dtype=Config.compute_dtype)

    # preload the prompt --------------------------------------------------
    # Feed every prompt token to build the initial cache
    for tok in prompt_ids[:-1]:
        _, cache = model.apply({"params": params},
                               jnp.array([[tok]], jnp.int32),
                               cache=cache,
                               deterministic=True)

    # ----- fused autoregressive loop ------------------------------------
    rng = jax.random.PRNGKey(0)
    new_tokens = generate_scan(params, cache, rng,
                               prompt_ids,
                               args.steps,
                               args.temperature,
                               None if args.top_k <= 0 else args.top_k)

    full = jnp.concatenate([prompt_ids, new_tokens], axis=0)
    print(tokenizer.decode(full.tolist()))

if __name__ == "__main__":
    main()


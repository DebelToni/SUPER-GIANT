from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from GIANT.v3.model.GiantGPT import GiantGPT


def load_model_cfg() -> OmegaConf:
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    cfg = OmegaConf.merge(
        OmegaConf.load(project_root / "Global_Config.yml"),
        OmegaConf.load(model_dir / "Config.yml"),
    )
    return cfg.model


def build_model(cfg: OmegaConf, vocab_size: int) -> GiantGPT:
    return GiantGPT(
        vocab_size=vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.embedding_size,
        n_heads=cfg.num_heads,
        d_ff=cfg.feed_forward_size,
        n_layers=cfg.num_layers,
        dropout_rate=0.0,
    )


def make_prefill_fn(model: GiantGPT):
    @jax.jit
    def prefill_with_offsets(params, cache_state, prompt_tokens, start_offsets):
        batch, seq_len = prompt_tokens.shape

        def step(carry, tok_t):
            nonparam, t = carry
            logits, updated = model.apply(
                {"params": params, **nonparam},
                tok_t,
                deterministic=True,
                use_kv_cache=True,
                cur_index=t,
                mutable=["cache"],
            )
            nonparam = {**nonparam, "cache": updated["cache"]}
            return (nonparam, t + 1), logits

        tokens_scan = jnp.expand_dims(jnp.swapaxes(prompt_tokens, 0, 1), -1)
        (nonparam, t_out), _ = jax.lax.scan(step, (cache_state, start_offsets), tokens_scan)
        return nonparam, t_out - 1, prompt_tokens[:, -1:]

    return prefill_with_offsets


def prefill_single(model, params, cache_state, prompt_tokens, start_offset):
    seq_len = prompt_tokens.shape[0]

    def step(carry, tok_t):
        nonparam, t = carry
        logits, updated = model.apply(
            {"params": params, **nonparam},
            tok_t,
            deterministic=True,
            use_kv_cache=True,
            cur_index=t,
            mutable=["cache"],
        )
        nonparam = {**nonparam, "cache": updated["cache"]}
        return (nonparam, t + 1), logits

    tokens_scan = prompt_tokens[:, None, None]
    (nonparam, t_out), _ = jax.lax.scan(step, (cache_state, start_offset), tokens_scan)
    return nonparam, t_out - 1, prompt_tokens[-1:]


def main() -> None:
    cfg = load_model_cfg()
    vocab_size = 128
    batch = 3
    seq_len = 4

    model = build_model(cfg, vocab_size=vocab_size)

    rng = jax.random.PRNGKey(0)
    key_params, key_dropout, key_tokens = jax.random.split(rng, 3)

    prompt_tokens = jax.random.randint(key_tokens, (batch, seq_len), 0, vocab_size)
    start_offsets = jnp.array([0, 2, 5], dtype=jnp.int32)

    variables = model.init(
        {"params": key_params, "dropout": key_dropout},
        prompt_tokens[:, :1],
        deterministic=True,
        use_kv_cache=True,
        cur_index=0,
    )
    params = variables["params"]
    cache_state = {k: v for k, v in variables.items() if k != "params"}

    prefill_fn = make_prefill_fn(model)
    prefill_fn.lower(params, cache_state, prompt_tokens, start_offsets).compile()

    start = time.perf_counter()
    cache_state, last_pos, last_tok = prefill_fn(
        params,
        cache_state,
        prompt_tokens,
        start_offsets,
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), cache_state)
    elapsed = time.perf_counter() - start

    next_logits, _ = model.apply(
        {"params": params, **cache_state},
        last_tok,
        deterministic=True,
        use_kv_cache=True,
        cur_index=last_pos,
        mutable=["cache"],
    )

    per_logits = []
    per_last_pos = []
    for i in range(batch):
        single_prompt = prompt_tokens[i]
        single_cache = jax.tree_util.tree_map(lambda x: x[i : i + 1], cache_state)
        single_offset = start_offsets[i : i + 1]
        single_cache, single_pos, single_last_tok = prefill_single(
            model,
            params,
            single_cache,
            single_prompt,
            single_offset,
        )
        single_logits, _ = model.apply(
            {"params": params, **single_cache},
            single_last_tok[None, :],
            deterministic=True,
            use_kv_cache=True,
            cur_index=single_pos[0],
            mutable=["cache"],
        )
        per_logits.append(single_logits)
        per_last_pos.append(single_pos)

    per_logits = jnp.concatenate(per_logits, axis=0)
    per_last_pos = jnp.concatenate(per_last_pos, axis=0)

    logits_match = jnp.allclose(next_logits, per_logits, atol=1e-5)
    pos_match = jnp.array_equal(last_pos, per_last_pos)

    print("batched logits", next_logits.shape, next_logits.dtype)
    print("last_pos", last_pos)
    print("logits_match", logits_match)
    print("pos_match", pos_match)
    print(f"prefill_time_s {elapsed:.6f}")


if __name__ == "__main__":
    main()

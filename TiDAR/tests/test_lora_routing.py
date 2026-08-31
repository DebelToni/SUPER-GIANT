from __future__ import annotations

import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict

from GIANT.v3.model.lora import LoRAConfig
from TiDAR.model.GiantTiDAR import TiDAR
from TiDAR.model.Training_step import adapter_loss_and_grad
from TiDAR.model.tidar_core import (
    build_decode_bias_template,
    build_decode_position_template,
    init_kv_cache,
    prefill_prompt_with_draft,
)
from TiDAR.model.inference import make_anchor_tidar_generate_fn
from TiDAR.model.tidar_utils import build_train_batch


VOCAB = 24
MASK_ID = VOCAB
SEQ = 4
DRAFT = 2


def _model(
    target_modules=("o_proj",),
    compute_dtype=jnp.float32,
    *,
    n_layers=2,
    layer_indices=None,
    stop_gradient_before_lora=False,
) -> TiDAR:
    return TiDAR(
        vocab_size=VOCAB,
        context_length=16,
        d_model=8,
        n_heads=2,
        num_kv_heads=1,
        rope_dim=4,
        d_ff=16,
        n_layers=n_layers,
        dropout_rate=0.0,
        param_dtype=jnp.float32,
        compute_dtype=compute_dtype,
        draft_len=DRAFT,
        lora_config=LoRAConfig(
            enabled=True,
            rank=2,
            alpha=4.0,
            target_modules=tuple(target_modules),
            routing="token",
            layer_indices=layer_indices,
            stop_gradient_before_lora=stop_gradient_before_lora,
        ),
        mask_token_id=MASK_ID,
        separate_mask_embedding=True,
    )


def _batch():
    tokens = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    return build_train_batch(tokens, jnp.array([SEQ]), mask_id=MASK_ID, block_len=DRAFT)


def _variables():
    batch = _batch()
    return _model().init(
        {"params": jax.random.PRNGKey(0), "adapters": jax.random.PRNGKey(1)},
        batch["input_ids"],
        deterministic=True,
        attn_bias=batch["attn_bias"],
        position_ids=batch["position_ids"],
        adapter_mask=batch["adapter_mask"],
    )


def _nonzero_b(adapters):
    flat = flatten_dict(unfreeze(adapters))
    for path, value in list(flat.items()):
        if path[-1] == "b":
            flat[path] = jnp.full_like(value, 0.1)
    return freeze(unflatten_dict(flat))


def test_out_of_vocab_mask_requires_separate_embedding():
    import pytest

    model = TiDAR(
        vocab_size=VOCAB,
        context_length=8,
        d_model=8,
        n_heads=2,
        num_kv_heads=1,
        rope_dim=4,
        d_ff=16,
        n_layers=1,
        dropout_rate=0.0,
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        draft_len=DRAFT,
        lora_config=LoRAConfig(
            enabled=True,
            rank=2,
            alpha=4.0,
            target_modules=("o_proj",),
            routing="token",
        ),
        mask_token_id=MASK_ID,
        separate_mask_embedding=False,
    )
    with pytest.raises(ValueError, match="requires separate_mask_embedding"):
        model.init(
            {"params": jax.random.PRNGKey(0), "adapters": jax.random.PRNGKey(1)},
            jnp.array([[1, MASK_ID]], dtype=jnp.int32),
            deterministic=True,
            adapter_mask=jnp.array([[False, True]]),
        )


def test_training_adapter_mask_routes_clean_and_diffusion_halves():
    mask = _batch()["adapter_mask"]
    assert mask.dtype == jnp.bool_
    assert not bool(jnp.any(mask[:, :SEQ]))
    assert bool(jnp.all(mask[:, SEQ:]))


def test_separate_mask_embedding_keeps_original_output_vocabulary():
    variables = _variables()
    logits = _model().apply(
        variables,
        _batch()["input_ids"],
        deterministic=True,
        attn_bias=_batch()["attn_bias"],
        position_ids=_batch()["position_ids"],
        adapter_mask=_batch()["adapter_mask"],
    )
    assert logits.shape[-1] == VOCAB
    assert variables["params"]["Embed_0"]["embedding"].shape[0] == VOCAB
    assert variables["adapters"]["mask_embedding"].shape == (8,)


def test_mixed_decode_preserves_verifier_logits_and_cache():
    model = _model()
    variables = _variables()
    adapters = _nonzero_b(variables["adapters"])
    cache_len = 10
    cache = init_kv_cache(model, batch_size=1, pad_token_id=0)

    verify = jnp.array([5, 6], dtype=jnp.int32)
    masks = jnp.full((DRAFT * DRAFT,), MASK_ID, dtype=jnp.int32)
    step_tokens = jnp.concatenate([verify, masks])[None, :]
    positions = build_decode_position_template(DRAFT)[None, :]
    bias = build_decode_bias_template(cache_len, DRAFT)
    mixed_route = jnp.concatenate(
        [
            jnp.zeros((1, DRAFT), dtype=jnp.bool_),
            jnp.ones((1, DRAFT * DRAFT), dtype=jnp.bool_),
        ],
        axis=1,
    )
    all_off = jnp.zeros_like(mixed_route)

    def apply(route):
        logits, mutated = model.apply(
            {"params": variables["params"], "adapters": adapters, "cache": cache},
            step_tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=0,
            cache_write_len=DRAFT,
            attn_bias=bias,
            position_ids=positions,
            kv_cache_len=cache_len,
            adapter_mask=route,
            mutable=["cache"],
        )
        return logits, mutated["cache"]

    off_logits, off_cache = apply(all_off)
    mixed_logits, mixed_cache = apply(mixed_route)
    assert jnp.array_equal(off_logits[:, :DRAFT], mixed_logits[:, :DRAFT])
    assert all(bool(jnp.array_equal(x, y)) for x, y in zip(
        jax.tree_util.tree_leaves(off_cache),
        jax.tree_util.tree_leaves(mixed_cache),
    ))
    assert bool(jnp.any(off_logits[:, DRAFT:] != mixed_logits[:, DRAFT:]))


def test_qkv_routing_preserves_verifier_kvs_with_prefetched_prefix():
    model = _model(("qkv_proj",), compute_dtype=jnp.bfloat16)
    batch = _batch()
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "adapters": jax.random.PRNGKey(1)},
        batch["input_ids"],
        deterministic=True,
        attn_bias=batch["attn_bias"],
        position_ids=batch["position_ids"],
        adapter_mask=batch["adapter_mask"],
    )
    adapters = _nonzero_b(variables["adapters"])
    cache_len = 10
    cache = init_kv_cache(
        model,
        batch_size=1,
        pad_token_id=0,
        params=variables["params"],
        adapter_params=adapters,
    )
    prompt = jnp.array([[1, 2]], dtype=jnp.int32)
    _, prefetched = model.apply(
        {"params": variables["params"], "adapters": adapters, "cache": cache},
        prompt,
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=True,
        cur_index=0,
        position_ids=jnp.array([[0, 1]], dtype=jnp.int32),
        kv_cache_len=cache_len,
        adapter_mask=jnp.zeros(prompt.shape, dtype=jnp.bool_),
        mutable=["cache"],
    )
    prefix_cache = prefetched["cache"]

    verify = jnp.array([5, 6], dtype=jnp.int32)
    masks = jnp.full((DRAFT * DRAFT,), MASK_ID, dtype=jnp.int32)
    step_tokens = jnp.concatenate([verify, masks])[None, :]
    positions = (2 + build_decode_position_template(DRAFT))[None, :]
    bias = build_decode_bias_template(cache_len, DRAFT)
    mixed_route = jnp.concatenate(
        [
            jnp.zeros((1, DRAFT), dtype=jnp.bool_),
            jnp.ones((1, DRAFT * DRAFT), dtype=jnp.bool_),
        ],
        axis=1,
    )

    def apply(route):
        logits, mutated = model.apply(
            {"params": variables["params"], "adapters": adapters, "cache": prefix_cache},
            step_tokens,
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=2,
            cache_write_len=DRAFT,
            attn_bias=bias,
            position_ids=positions,
            kv_cache_len=cache_len,
            adapter_mask=route,
            mutable=["cache"],
        )
        return logits, mutated["cache"]

    off_logits, off_cache = apply(jnp.zeros_like(mixed_route))
    mixed_logits, mixed_cache = apply(mixed_route)
    assert jnp.array_equal(off_logits[:, :DRAFT], mixed_logits[:, :DRAFT])
    assert all(bool(jnp.array_equal(x, y)) for x, y in zip(
        jax.tree_util.tree_leaves(off_cache),
        jax.tree_util.tree_leaves(mixed_cache),
    ))
    assert bool(jnp.any(off_logits[:, DRAFT:] != mixed_logits[:, DRAFT:]))


def test_tidar_lora_cached_generation_smoke():
    model = _model()
    variables = _variables()
    adapters = _nonzero_b(variables["adapters"])
    cache_len = 10
    cache = init_kv_cache(model, batch_size=1, pad_token_id=0)
    cache, prefix_len, last_logit, initial_draft_logits = prefill_prompt_with_draft(
        model,
        variables["params"],
        cache,
        jnp.array([1, 2], dtype=jnp.int32),
        draft_len=DRAFT,
        mask_id=MASK_ID,
        kv_cache_len=cache_len,
        adapter_params=adapters,
    )
    output = jnp.zeros((cache_len,), dtype=jnp.int32).at[:2].set(jnp.array([1, 2]))
    generate = make_anchor_tidar_generate_fn(
        model,
        cache_len=cache_len,
        draft_len=DRAFT,
        mask_id=MASK_ID,
        pad_token_id=0,
        eos_id=-1,
        stop_on_eos=False,
        temperature=0.0,
        top_k=0,
        bias_value=-1.0e10,
    )
    output, final_len, generated, stats = generate(
        variables["params"],
        cache,
        output,
        jnp.asarray(prefix_len, dtype=jnp.int32),
        jnp.asarray(2, dtype=jnp.int32),
        last_logit,
        jax.random.PRNGKey(4),
        initial_draft_logits,
        adapters,
    )
    assert int(final_len) == 4
    assert int(generated) == 2
    assert output.shape == (cache_len,)
    assert int(stats["n_iterations"]) >= 1


def test_suffix_lora_freezes_mask_embedding_and_omits_prefix_adapters():
    model = _model(
        n_layers=4,
        layer_indices=(2, 3),
        stop_gradient_before_lora=True,
    )
    batch = _batch()
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "adapters": jax.random.PRNGKey(1)},
        batch["input_ids"],
        deterministic=True,
        attn_bias=batch["attn_bias"],
        position_ids=batch["position_ids"],
        adapter_mask=batch["adapter_mask"],
    )
    flat_adapters = flatten_dict(variables["adapters"])
    adapter_roots = {path[0] for path in flat_adapters if path != ("mask_embedding",)}
    assert adapter_roots == {"TinyTransformerBlock_2", "TinyTransformerBlock_3"}

    (loss, _metrics), grads = adapter_loss_and_grad(
        variables["adapters"],
        variables["params"],
        batch,
        model=model,
        dropout_rng=jax.random.PRNGKey(3),
        alpha=0.0,
        beta=1.0,
        rho=0.0,
        chi=0.0,
        delta=0.0,
        eta=0.0,
        eta_T=1.0,
    )
    assert jnp.isfinite(loss)
    assert jnp.array_equal(grads["mask_embedding"], jnp.zeros_like(grads["mask_embedding"]))


def test_tidar_adapter_gradients_exclude_base_parameters():
    model = _model()
    variables = _variables()
    batch = _batch()
    (loss, _metrics), grads = adapter_loss_and_grad(
        variables["adapters"],
        variables["params"],
        batch,
        model=model,
        dropout_rng=jax.random.PRNGKey(3),
        alpha=0.0,
        beta=1.0,
        rho=0.0,
        chi=0.0,
        delta=0.0,
        eta=0.0,
        eta_T=1.0,
    )
    assert jnp.isfinite(loss)
    assert set(flatten_dict(grads)) == set(flatten_dict(variables["adapters"]))

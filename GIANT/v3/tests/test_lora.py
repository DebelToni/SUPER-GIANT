from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax.core import freeze, unfreeze
from flax.traverse_util import flatten_dict

from GIANT.v3.model.GiantGPT import GiantGPT
from GIANT.v3.model.Training_step import adapter_loss_and_grad
from GIANT.v3.model.checkpoint_manager import load_npz, save_npz
from GIANT.v3.model.jit_inference import init_inference_state, make_prefill_and_decode_fns
from GIANT.v3.model.lora import (
    LoRAConfig,
    assert_tree_compatible,
    count_parameters,
    validate_adapter_checkpoint_manifest,
    write_adapter_manifest,
)


def _model(*, lora: LoRAConfig = LoRAConfig(), remat: bool = False) -> GiantGPT:
    return GiantGPT(
        vocab_size=32,
        context_length=8,
        d_model=16,
        n_heads=2,
        d_ff=32,
        n_layers=2,
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        num_kv_heads=1,
        rotary_dim=8,
        dropout_rate=0.0,
        use_remat=remat,
        lora_config=lora,
    )


def _tokens() -> jax.Array:
    return jnp.arange(8, dtype=jnp.int32)[None, :]


def _init(model: GiantGPT, *, adapter_mask=None):
    rngs = {"params": jax.random.PRNGKey(0)}
    if model.lora_config.enabled:
        rngs["adapters"] = jax.random.PRNGKey(1)
    return model.init(rngs, _tokens(), deterministic=True, adapter_mask=adapter_mask)


def _nonzero_b(adapter_params):
    mutable = unfreeze(adapter_params)
    flat = flatten_dict(mutable)
    for path, value in list(flat.items()):
        if path[-1] == "b":
            flat[path] = jnp.full_like(value, 0.05)
    from flax.traverse_util import unflatten_dict

    return freeze(unflatten_dict(flat))


def test_lora_preserves_base_parameter_tree_and_zero_init_logits():
    raw_model = _model()
    raw_vars = _init(raw_model)
    config = LoRAConfig(
        enabled=True,
        rank=4,
        alpha=8.0,
        target_modules=("qkv_proj", "o_proj", "fc1", "fc2"),
        routing="global",
    )
    lora_model = _model(lora=config)
    lora_vars = _init(lora_model)

    assert set(flatten_dict(raw_vars["params"])) == set(flatten_dict(lora_vars["params"]))
    assert_tree_compatible(raw_vars["params"], lora_vars["params"], label="base")

    raw_logits = raw_model.apply(raw_vars, _tokens(), deterministic=True)
    lora_logits = lora_model.apply(lora_vars, _tokens(), deterministic=True)
    assert jnp.array_equal(raw_logits, lora_logits)


def test_layer_indices_create_only_selected_adapters():
    config = LoRAConfig(
        enabled=True,
        rank=4,
        alpha=8.0,
        target_modules=("o_proj",),
        routing="global",
        layer_indices=(1,),
        stop_gradient_before_lora=True,
    )
    variables = _init(_model(lora=config))
    adapter_roots = {path[0] for path in flatten_dict(variables["adapters"])}
    assert adapter_roots == {"TinyTransformerBlock_1"}


def test_layer_indices_reject_out_of_range_model_layer():
    import pytest

    config = LoRAConfig(
        enabled=True,
        rank=4,
        alpha=8.0,
        target_modules=("o_proj",),
        routing="global",
        layer_indices=(2,),
    )
    with pytest.raises(ValueError, match="exceed model layer range"):
        _init(_model(lora=config))


def test_token_routing_selects_bit_identical_base_rows():
    config = LoRAConfig(
        enabled=True,
        rank=4,
        alpha=8.0,
        target_modules=("o_proj",),
        routing="token",
    )
    model = _model(lora=config)
    all_off = jnp.zeros(_tokens().shape, dtype=jnp.bool_)
    variables = _init(model, adapter_mask=all_off)
    adapters = _nonzero_b(variables["adapters"])

    raw_model = _model()
    raw_logits = raw_model.apply({"params": variables["params"]}, _tokens(), deterministic=True)
    off_logits = model.apply(
        {"params": variables["params"], "adapters": adapters},
        _tokens(),
        deterministic=True,
        adapter_mask=all_off,
    )
    assert jnp.array_equal(raw_logits, off_logits)

    all_on = jnp.ones(_tokens().shape, dtype=jnp.bool_)
    on_logits = model.apply(
        {"params": variables["params"], "adapters": adapters},
        _tokens(),
        deterministic=True,
        adapter_mask=all_on,
    )
    assert bool(jnp.any(on_logits != raw_logits))


def test_adapter_training_updates_only_adapter_tree():
    config = LoRAConfig(
        enabled=True,
        rank=4,
        alpha=8.0,
        target_modules=("o_proj",),
        routing="global",
    )
    model = _model(lora=config)
    variables = _init(model)
    base_before = jax.tree_util.tree_map(jnp.copy, variables["params"])
    tokens = _tokens()
    batch = {
        "input": tokens,
        "target": jnp.roll(tokens, -1, axis=1),
        "mask": jnp.ones(tokens.shape, dtype=jnp.float32),
    }

    loss, grads = adapter_loss_and_grad(
        variables["adapters"],
        variables["params"],
        batch,
        model=model,
        dropout_rng=jax.random.PRNGKey(2),
    )
    optimizer = optax.adam(1e-2)
    state = optimizer.init(variables["adapters"])
    updates, _ = optimizer.update(grads, state, variables["adapters"])
    adapters_after = optax.apply_updates(variables["adapters"], updates)

    assert jnp.isfinite(loss)
    assert any(bool(jnp.any(x != y)) for x, y in zip(
        jax.tree_util.tree_leaves(variables["adapters"]),
        jax.tree_util.tree_leaves(adapters_after),
    ))
    assert all(bool(jnp.array_equal(x, y)) for x, y in zip(
        jax.tree_util.tree_leaves(base_before),
        jax.tree_util.tree_leaves(variables["params"]),
    ))
    assert count_parameters(grads) == count_parameters(variables["adapters"])


def test_adapter_checkpoint_round_trip(tmp_path: Path):
    config = LoRAConfig(
        enabled=True,
        rank=2,
        alpha=4.0,
        target_modules=("o_proj",),
        routing="global",
    )
    variables = _init(_model(lora=config))
    path = tmp_path / "adapter.npz"
    save_npz(variables["adapters"], path)
    loaded = load_npz(path, print_name=False)
    assert_tree_compatible(variables["adapters"], loaded, label="adapter")
    assert all(bool(jnp.array_equal(x, y)) for x, y in zip(
        jax.tree_util.tree_leaves(variables["adapters"]),
        jax.tree_util.tree_leaves(loaded),
    ))


def test_adapter_manifest_rejects_changed_base_or_config(tmp_path: Path):
    config = LoRAConfig(
        enabled=True,
        rank=2,
        alpha=4.0,
        target_modules=("o_proj",),
        routing="global",
    )
    variables = _init(_model(lora=config))
    base_path = tmp_path / "base.npz"
    adapter_dir = tmp_path / "run" / "adapters"
    adapter_dir.mkdir(parents=True)
    adapter_path = adapter_dir / "step_0000001.npz"
    save_npz(variables["params"], base_path)
    save_npz(variables["adapters"], adapter_path)
    write_adapter_manifest(
        tmp_path / "run" / "adapter_config.json",
        base_checkpoint=base_path,
        config=config,
        base_parameter_count=count_parameters(variables["params"]),
        adapter_parameter_count=count_parameters(variables["adapters"]),
    )
    validate_adapter_checkpoint_manifest(
        adapter_path,
        base_checkpoint=base_path,
        config=config,
        base_parameter_count=count_parameters(variables["params"]),
        adapter_parameter_count=count_parameters(variables["adapters"]),
    )

    wrong_base_path = tmp_path / "wrong_base.npz"
    wrong_base = jax.tree_util.tree_map(lambda value: value + 1, variables["params"])
    save_npz(wrong_base, wrong_base_path)

    import pytest

    with pytest.raises(ValueError, match="base_checkpoint.sha256"):
        validate_adapter_checkpoint_manifest(
            adapter_path,
            base_checkpoint=wrong_base_path,
            config=config,
            base_parameter_count=count_parameters(wrong_base),
            adapter_parameter_count=count_parameters(variables["adapters"]),
        )

    changed = LoRAConfig(
        enabled=True,
        rank=2,
        alpha=8.0,
        target_modules=("o_proj",),
        routing="global",
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_adapter_checkpoint_manifest(
            adapter_path,
            base_checkpoint=base_path,
            config=changed,
            base_parameter_count=count_parameters(variables["params"]),
            adapter_parameter_count=count_parameters(variables["adapters"]),
        )


def test_nonzero_lora_works_through_jitted_cached_inference():
    config = LoRAConfig(
        enabled=True,
        rank=2,
        alpha=4.0,
        target_modules=("qkv_proj", "o_proj"),
        routing="global",
    )
    model = _model(lora=config)
    variables = _init(model)
    adapters = _nonzero_b(variables["adapters"])
    _, nonparam = init_inference_state(
        model,
        jax.random.PRNGKey(10),
        jax.random.PRNGKey(11),
        batch_size=1,
        pad_token_id=0,
        use_kv_cache=True,
        params=variables["params"],
        adapter_params=adapters,
    )
    prefill, decode = make_prefill_and_decode_fns(model)
    prompt = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    nonparam, position, last_token = prefill(
        variables["params"], nonparam, prompt, adapters
    )
    generated, _ = decode(
        variables["params"],
        nonparam,
        last_token,
        position,
        steps=2,
        do_sample=False,
        adapter_params=adapters,
    )
    assert generated.shape == (1, 2)
    assert bool(jnp.all(jnp.isfinite(generated)))


def test_lora_works_with_rematerialized_blocks():
    config = LoRAConfig(
        enabled=True,
        rank=2,
        alpha=4.0,
        target_modules=("o_proj", "fc2"),
        routing="global",
    )
    normal = _model(lora=config, remat=False)
    remat = _model(lora=config, remat=True)
    normal_vars = _init(normal)
    remat_vars = _init(remat)
    assert_tree_compatible(normal_vars["params"], remat_vars["params"], label="params")
    assert_tree_compatible(normal_vars["adapters"], remat_vars["adapters"], label="adapters")
    normal_logits = normal.apply(normal_vars, _tokens(), deterministic=True)
    remat_logits = remat.apply(remat_vars, _tokens(), deterministic=True)
    assert jnp.array_equal(normal_logits, remat_logits)

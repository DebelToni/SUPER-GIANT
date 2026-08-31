from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from GIANT.v3.model.GiantGPT import GiantGPT
from GIANT.v3.model.Training_step import adapter_loss_and_grad
from GIANT.v3.model.lora import LoRAConfig


def _assert_trees_close(left, right, atol=1e-6):
    for x, y in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        assert jnp.allclose(x, y, atol=atol, rtol=atol)


def test_lora_gradients_match_single_device_under_pmap_and_shard_map():
    devices = jax.local_devices()
    if len(devices) < 2:
        pytest.skip("run with XLA_FLAGS=--xla_force_host_platform_device_count=2")
    devices = devices[:2]

    model = GiantGPT(
        vocab_size=16,
        context_length=8,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_layers=1,
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        num_kv_heads=1,
        rotary_dim=4,
        dropout_rate=0.0,
        lora_config=LoRAConfig(
            enabled=True,
            rank=2,
            alpha=4.0,
            target_modules=("qkv_proj", "o_proj", "fc1", "fc2"),
            routing="global",
        ),
    )
    tokens = jnp.arange(32, dtype=jnp.int32).reshape(4, 8) % 16
    batch = {
        "input": tokens,
        "target": jnp.roll(tokens, -1, axis=1),
        "mask": jnp.ones(tokens.shape, dtype=jnp.float32),
    }
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "adapters": jax.random.PRNGKey(1)},
        tokens,
        deterministic=True,
    )
    rng = jax.random.PRNGKey(2)
    expected_loss, expected_grads = adapter_loss_and_grad(
        variables["adapters"],
        variables["params"],
        batch,
        model=model,
        dropout_rng=rng,
    )

    mesh = Mesh(np.asarray(devices), ("data",))
    replicated_base = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (len(devices), *x.shape)), variables["params"]
    )
    replicated_adapters = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (len(devices), *x.shape)), variables["adapters"]
    )
    sharded_batch = jax.tree_util.tree_map(lambda x: x.reshape(2, 2, *x.shape[1:]), batch)

    @partial(jax.pmap, axis_name="data", devices=devices)
    def pmap_grad(base, adapters, local_batch):
        return adapter_loss_and_grad(
            adapters,
            base,
            local_batch,
            model=model,
            dropout_rng=rng,
            axis_name="data",
        )

    pmap_loss, pmap_grads = pmap_grad(replicated_base, replicated_adapters, sharded_batch)
    assert jnp.allclose(pmap_loss[0], expected_loss, atol=1e-6, rtol=1e-6)
    pmap_first = jax.tree_util.tree_map(lambda x: x[0], pmap_grads)
    _assert_trees_close(pmap_first, expected_grads)

    batch_spec = {
        "input": P("data", None),
        "target": P("data", None),
        "mask": P("data", None),
    }
    tree_spec = jax.tree_util.tree_map(lambda _: P(), variables["adapters"])

    def shard_grad(base, adapters, global_batch):
        return adapter_loss_and_grad(
            adapters,
            base,
            global_batch,
            model=model,
            dropout_rng=rng,
            axis_name="data",
        )

    shard_grad = jax.jit(
        jax.shard_map(
            shard_grad,
            mesh=mesh,
            in_specs=(
                jax.tree_util.tree_map(lambda _: P(), variables["params"]),
                tree_spec,
                batch_spec,
            ),
            out_specs=(P(), tree_spec),
            axis_names={"data"},
            check_vma=False,
        )
    )
    shard_loss, shard_grads = shard_grad(variables["params"], variables["adapters"], batch)
    assert jnp.allclose(shard_loss, expected_loss, atol=1e-6, rtol=1e-6)
    _assert_trees_close(shard_grads, expected_grads)

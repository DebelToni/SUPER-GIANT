"""
jax_error_fuzzer_pro.py

Deliberately gnarly JAX error fuzzer with transformer-flavored names.
Each run picks ONE complex, deeply-nested failure path (18 total) to
produce a long, unreadable stack trace.

Usage:
  python jax_error_fuzzer_pro.py
  python jax_error_fuzzer_pro.py --seed 123
  python jax_error_fuzzer_pro.py --case 7

Notes:
- This script *intentionally* misuses JAX. Do not copy patterns into real code.
- Most errors are wrapped under jit/vmap/grad/scan/cond/while_loop/custom_vjp/jvp
  to enlarge traces.
"""

import argparse
import os
import random as pyrandom
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from jax import lax


def _announce(name: str, idx: int):
    print(f"\n>>> JAX error fuzzer (pro): case[{idx}] = {name}\n", flush=True)


def _randseed(seed: int | None):
    if seed is not None:
        pyrandom.seed(seed)
    else:
        pyrandom.seed(int.from_bytes(os.urandom(8), "little"))



def case_encoder_stack_scan_carry_mutation():
    batch_size, seq_len, model_dim = 2, 5, 16

    def _mlp_block(hidden_state, token_vec):
        return hidden_state[:, None] + jnp.tanh(token_vec).mean()

    @jax.jit
    def _encoder_layer_scanned(sequence_embeddings):
        def _scan_step(carry_hidden, token_embedding):
            mutated_hidden = _mlp_block(carry_hidden, token_embedding)
            return mutated_hidden, token_embedding
        init_hidden = jnp.linspace(0.0, 1.0, model_dim)
        ys = sequence_embeddings
        return lax.scan(_scan_step, init_hidden, ys)

    batched_tokens = jnp.ones((batch_size, seq_len, model_dim))
    jax.vmap(_encoder_layer_scanned)(batched_tokens)


def case_branching_attention_mismatch():
    b, h, s, d = 1, 2, 8, 4
    attn_logits = jnp.ones((b, h, s, s))
    padding_mask = jnp.tril(jnp.ones((s, s)))

    @jax.jit
    def masked_softmax_pathology(logits, mask):
        def true_branch(x):
            return jnp.sum(x, axis=-1)

        def false_branch(x):
            return jnp.any(x > 0.0)

        return lax.cond(jnp.mean(mask) > 0.5, true_branch, false_branch, logits)

    masked_softmax_pathology(attn_logits, padding_mask)


def case_custom_vjp_backward_arity_violation():
    @jax.custom_vjp
    def rotary_embed_pathological(q_proj, inv_freq):
        return q_proj * jnp.cos(inv_freq) + q_proj * 0.0

    def _fwd(q_proj, inv_freq):
        y = rotary_embed_pathological.__wrapped__(q_proj, inv_freq)
        residual = (q_proj, inv_freq, jnp.array(1.0))
        return y, residual

    def _bwd(residual, g):
        q_proj, inv_freq, _ = residual
        return g, g, g

    rotary_embed_pathological.defvjp(_fwd, _bwd)

    def loss_fn(q_proj, inv_freq):
        y = rotary_embed_pathological(q_proj, inv_freq)
        return jnp.sum(y**2)

    q_proj = jnp.ones((2, 4))
    inv_freq = jnp.linspace(0.0, 1.0, 4)
    jax.grad(loss_fn)(q_proj, inv_freq)


def case_custom_jvp_bad_tangent():
    @jax.custom_jvp
    def block_sparse_attention_scores(q_proj, k_proj):
        return jnp.einsum("bhdm,bhdn->bhmn", q_proj, k_proj)

    @block_sparse_attention_scores.defjvp
    def _bs_attn_jvp(primals, tangents):
        q, k = primals
        tq, tk = tangents
        out = block_sparse_attention_scores(q, k)
        bogus_tangent = jnp.array(0.0)
        return out, bogus_tangent

    b, h, m, n, d = 1, 2, 8, 7, 4
    q = jnp.ones((b, h, m, d))
    k = jnp.ones((b, h, n, d)) * 2.0

    def loss(q, k):
        return jnp.sum(block_sparse_attention_scores(q, k))

    jax.grad(loss)(q, k)


def case_concretization_from_data_shape():
    @jax.jit
    def allocate_cache_from_token_ids(token_ids):
        dynamic_slots = int(jnp.sum(token_ids))
        return jnp.zeros((dynamic_slots,))

    token_ids = jnp.array([1, 2, 3, 4])
    allocate_cache_from_token_ids(token_ids)


def case_scaled_dot_product_attention_head_mismatch():
    bsz, seq, d_model, n_heads = 2, 6, 12, 3
    dk_q, dk_k = 5, 7

    @jax.jit
    def attention_scores(q, k):
        return jnp.einsum("bqhd,bkhd->bhqk", q, k)

    def build_qkv_and_loss(x_tokens):
        q_proj = jnp.reshape(x_tokens, (bsz, seq, n_heads, dk_q))
        k_proj = jnp.reshape(x_tokens, (bsz, seq, n_heads, dk_k))
        logits = attention_scores(q_proj, k_proj)
        return jnp.sum(logits)

    x = jnp.arange(bsz * seq * d_model, dtype=jnp.float32).reshape(bsz, seq, d_model)
    jax.value_and_grad(build_qkv_and_loss)(x)


def case_while_loop_carry_shape_mutation():
    @jax.jit
    def training_loop_with_shape_mutation(initial_hidden):
        def cond(state):
            return state["step"] < 3

        def body(state):
            hidden = state["hidden"]
            new_hidden = jnp.stack([hidden, hidden]) if state["step"] == 1 else (hidden + 1.0)
            return {"hidden": new_hidden, "step": state["step"] + 1}

        init_state = {"hidden": initial_hidden, "step": 0}
        return lax.while_loop(cond, body, init_state)

    init = jnp.ones((8,))
    training_loop_with_shape_mutation(init)


def case_conv_dimension_numbers_incoherence():
    x = jnp.ones((1, 8, 8, 3))
    w = jnp.ones((3, 3, 3, 4))

    @jax.jit
    def bogus_conv(x, w):
        return lax.conv_general_dilated(
            x, w,
            window_strides=(1, 1, 1),
            padding="SAME",
            dimension_numbers=("NCHW", "OIHW", "NCHW")
        )

    bogus_conv(x, w)


def case_scan_dynamic_length_concretization():
    @jax.jit
    def scan_with_data_dependent_length(xs):
        def step(c, x):
            return (c + x, x)
        init = 0.0
        dynamic_len = int(jnp.sum(xs))
        return lax.scan(step, init, xs, length=dynamic_len)

    xs = jnp.array([1.0, 1.0, 1.0])
    scan_with_data_dependent_length(xs)


def case_gather_dimension_numbers_carnage():
    operand = jnp.arange(2 * 3 * 4).reshape(2, 3, 4)
    indices = jnp.array([[0, 0], [1, 2], [0, 1]])

    dnums = lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(1, 2, 3),
        start_index_map=(0, 1, 2)
    )
    lax.gather(operand, indices, dimension_numbers=dnums, slice_sizes=(1, 1))


def case_dynamic_update_slice_buried_mismatch():
    operand = jnp.zeros((3, 3))
    update = jnp.ones((6, 6))

    @jax.jit
    def cond_wrapped_update(op, up):
        def t_branch(args):
            op, up = args
            return lax.dynamic_update_slice(op, up, (0, 0))

        def f_branch(args):
            op, up = args
            return op * 0.0 + up.sum()

        return lax.cond(jnp.array(True), t_branch, f_branch, (op, up))

    cond_wrapped_update(operand, update)


def case_value_and_grad_aux_shape_violation():
    def loss_with_aux(params, token_batch):
        logits = params @ token_batch.T
        attn_weights_debug_dump = {"layer_0.qk": logits}
        return logits, attn_weights_debug_dump

    d, n = 8, 4
    params = jnp.ones((d, d))
    tokens = jnp.ones((n, d))
    jax.jit(jax.value_and_grad(loss_with_aux, has_aux=True))(params, tokens)


def case_treedef_mismatch_under_grad():
    def pipeline(params_pytree, x):
        q = params_pytree["q"] @ x
        k = params_pytree["k"] @ x
        v = params_pytree["v"] @ x
        return jnp.sum(q + k + v)

    def loss(params_pytree, x):
        broken = {"q": params_pytree["q"], "k": params_pytree["k"]}
        merged = jax.tree_util.tree_map(lambda a, b: a + b, params_pytree, broken)
        return pipeline(merged, x)

    params = {"q": jnp.ones((4, 4)), "k": jnp.ones((4, 4)), "v": jnp.ones((4, 4))}
    x = jnp.ones((4, 1))
    jax.jit(jax.grad(loss))(params, x)


def case_masked_softmax_broadcast_catastrophe():
    b, h, s, t = 2, 4, 8, 8
    logits = jnp.ones((b, h, s, t))
    non_broadcastable_mask = jnp.arange(s) % 2 == 0
    scale = 1.0 / jnp.sqrt(64.0)

    @jax.jit
    def masked_softmax(logits, mask):
        masked = jnp.where(mask, logits * scale, -1e9)
        return jax.nn.softmax(masked, axis=-1).sum()

    jax.grad(masked_softmax)(logits, non_broadcastable_mask)


def case_custom_jvp_float0_tangent_explosion():
    @jax.custom_jvp
    def weird_token_indexer(token_ids, scale):
        return jnp.sum(token_ids.astype(jnp.int32)) * scale

    @weird_token_indexer.defjvp
    def _weird_token_indexer_jvp(primals, tangents):
        token_ids, scale = primals
        t_token_ids, t_scale = tangents
        out = weird_token_indexer(token_ids, scale)
        bogus_tangent = t_token_ids + t_scale
        return out, bogus_tangent

    token_ids = jnp.array([1, 2, 3], dtype=jnp.int32)
    scale = 2.0

    def loss(token_ids, scale):
        return weird_token_indexer(token_ids, scale) ** 2

    jax.grad(loss)(token_ids, scale)


def case_sort_key_value_rank_disaster():
    @jax.jit
    def sort_with_misaligned_payload(keys, values):
        return lax.sort(keys, values=values, dimension=-1, is_stable=True)

    b, n = 3, 10
    keys = jnp.arange(b * n).reshape(b, n)
    payload = jnp.arange(b * n * 5).reshape(b, n, 5)
    jax.vmap(sort_with_misaligned_payload)(keys, payload)


def case_jit_static_unhashable_deep_vmap():
    static_cfg = {"dropout_p": 0.1, "activation": "gelu"}

    @jax.jit(static_argnums=0)
    def feedforward(static_config, x_embed):
        return x_embed + (1.0 if static_config.get("activation") == "gelu" else 0.0)

    batched = jnp.ones((4, 16))
    jax.vmap(lambda row: feedforward(static_cfg, row))(batched)


def case_multi_stage_update_shape_failure():
    def inner_update(activation_grid):
        big_update = jnp.ones((activation_grid.shape[0] + 10, activation_grid.shape[1] + 10))
        return lax.dynamic_update_slice(activation_grid, big_update, (0, 0))

    @jax.jit
    def outer_loss(activation_grid, flag):
        def t_branch(x):
            return inner_update(x).sum()

        def f_branch(x):
            return jnp.array(True)

        val = lax.cond(flag, t_branch, f_branch, activation_grid)
        return val

    grid = jnp.zeros((8, 8))
    jax.grad(outer_loss)(grid, True)


ERROR_CASES: List[Tuple[str, Callable[[], None]]] = [
    ("vmap->jit->scan: encoder carry shape mutation", case_encoder_stack_scan_carry_mutation),
    ("jit cond: branch shape/dtype mismatch in attention mask path", case_branching_attention_mismatch),
    ("custom_vjp: backward returns wrong number of cotangents", case_custom_vjp_backward_arity_violation),
    ("custom_jvp: tangent shape/dtype mismatch", case_custom_jvp_bad_tangent),
    ("jit: shape from data (ConcretizationTypeError)", case_concretization_from_data_shape),
    ("jit->value_and_grad->einsum: head-dim mismatch in attention", case_scaled_dot_product_attention_head_mismatch),
    ("while_loop: carry shape mutation mid-iteration", case_while_loop_carry_shape_mutation),
    ("conv: incoherent dimension_numbers/strides versus NHWC/HWIO", case_conv_dimension_numbers_incoherence),
    ("scan: dynamic length derived from data (ConcretizationTypeError)", case_scan_dynamic_length_concretization),
    ("gather: invalid GatherDimensionNumbers (verbose)", case_gather_dimension_numbers_carnage),
    ("cond->dynamic_update_slice: oversized update (buried)", case_dynamic_update_slice_buried_mismatch),
    ("value_and_grad(has_aux): non-scalar value + structure mismatch", case_value_and_grad_aux_shape_violation),
    ("treedef mismatch under jit->grad", case_treedef_mismatch_under_grad),
    ("jit->grad: where/softmax broadcast catastrophe", case_masked_softmax_broadcast_catastrophe),
    ("custom_jvp: bogus integer/float0 tangent propagation", case_custom_jvp_float0_tangent_explosion),
    ("vmap->sort: key/payload rank disagreement", case_sort_key_value_rank_disaster),
    ("vmap over jit: static_argnums receives unhashable dict", case_jit_static_unhashable_deep_vmap),
    ("multi-stage jit->grad->cond: dynamic_update_slice failure", case_multi_stage_update_shape_failure),
]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--case", type=int, default=None, help="force a particular error index")
    args = parser.parse_args()
    _randseed(args.seed)

    idx = (args.case % len(ERROR_CASES)) if args.case is not None else pyrandom.randrange(len(ERROR_CASES))
    name, fn = ERROR_CASES[idx]
    _announce(name, idx)
    fn()

if __name__ == "__main__":
    main()


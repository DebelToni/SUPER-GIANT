from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn


def _to_dtype(value: jnp.dtype | str) -> jnp.dtype:
    if isinstance(value, jnp.dtype):
        return value
    if isinstance(value, str):
        try:
            return getattr(jnp, value)
        except AttributeError:
            return jnp.dtype(value)
    return jnp.dtype(value)


@dataclass(frozen=True)
class GraphSpec:
    senders: tuple[int, ...]
    receivers: tuple[int, ...]
    input_heads: tuple[int, ...]
    output_head: int


def build_graph_spec(
    *,
    num_heads: int,
    avg_out_degree: int,
    max_out_degree: int,
    input_head_count: int,
    output_head_index: int,
    seed: int,
) -> GraphSpec:
    import numpy as np

    rng = np.random.default_rng(seed)
    senders: list[int] = []
    receivers: list[int] = []
    for src in range(num_heads):
        degree = int(np.clip(rng.poisson(avg_out_degree), 1, max_out_degree))
        choices = np.arange(num_heads)
        dsts = rng.choice(choices, size=degree, replace=False)
        for dst in dsts:
            senders.append(src)
            receivers.append(int(dst))

    incoming = {i: 0 for i in range(num_heads)}
    for dst in receivers:
        incoming[dst] += 1
    for dst, count in incoming.items():
        if count == 0:
            src = int(rng.integers(0, num_heads))
            senders.append(src)
            receivers.append(dst)

    if output_head_index not in receivers:
        senders.append(int(rng.integers(0, num_heads)))
        receivers.append(output_head_index)
    if output_head_index not in senders:
        dst = int(rng.integers(0, num_heads))
        senders.append(output_head_index)
        receivers.append(dst)

    input_heads = tuple(int(x) for x in rng.choice(num_heads, size=input_head_count, replace=False))
    return GraphSpec(tuple(senders), tuple(receivers), input_heads, int(output_head_index))


def _safe_lerp_or_slerp(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
    eps = jnp.asarray(1e-6, dtype=jnp.float32)
    af = a.astype(jnp.float32)
    bf = b.astype(jnp.float32)
    an = jnp.linalg.norm(af, axis=-1, keepdims=True)
    bn = jnp.linalg.norm(bf, axis=-1, keepdims=True)
    dot = jnp.sum(af * bf, axis=-1, keepdims=True) / jnp.maximum(an * bn, eps)
    dot = jnp.clip(dot, -1.0 + 1e-5, 1.0 - 1e-5)
    theta = jnp.arccos(dot)
    sin_theta = jnp.sin(theta)
    s0 = jnp.sin((1.0 - c) * theta) / jnp.maximum(sin_theta, eps)
    s1 = jnp.sin(c * theta) / jnp.maximum(sin_theta, eps)
    s0 = jax.lax.stop_gradient(jnp.nan_to_num(s0, nan=1.0, posinf=1.0, neginf=1.0))
    s1 = jax.lax.stop_gradient(jnp.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0))
    slerp = s0 * af + s1 * bf
    lerp = (1.0 - c) * af + c * bf
    use_lerp = (an < eps) | (bn < eps) | (jnp.abs(dot) > 0.999)
    return jnp.where(use_lerp, lerp, slerp).astype(a.dtype)


def spherical_segment_merge(messages: jnp.ndarray, receivers: jnp.ndarray, num_heads: int) -> jnp.ndarray:
    _, dim = messages.shape
    order = jnp.argsort(receivers)
    msgs = messages[order]
    dsts = receivers[order]
    init_acc = jnp.zeros((num_heads, dim), dtype=messages.dtype)
    init_counts = jnp.zeros((num_heads,), dtype=jnp.int32)

    def body(carry: tuple[jnp.ndarray, jnp.ndarray], item: tuple[jnp.ndarray, jnp.ndarray]):
        acc, counts = carry
        msg, dst = item
        seen = counts[dst]
        c = 1.0 / (seen.astype(jnp.float32) + 1.0)
        merged = jnp.where(seen == 0, msg, _safe_lerp_or_slerp(acc[dst], msg, c))
        acc = acc.at[dst].set(merged)
        counts = counts.at[dst].add(1)
        return (acc, counts), None

    (out, _), _ = jax.lax.scan(body, (init_acc, init_counts), (msgs, dsts))
    return out


class IvanchoBrain(nn.Module):
    vocab_size: int
    graph: GraphSpec
    num_heads: int
    state_dim: int
    inner_steps: int
    memory_slots: int
    memory_attn_heads: int
    memory_kv_heads: int
    residual_init: float
    residual_decay: float
    state_norm_cap: float
    threshold_hi: float
    threshold_lo: float
    threshold_alpha: float
    emit_beta: float
    step_penalty: float
    readiness_delta: float
    readiness_tau: float
    readiness_loss_weight: float
    collapse_token_ids: tuple[int, ...]
    collapse_penalty_weight: float
    halt_bias_init: float
    min_emit_step: int
    dropout_rate: float
    param_dtype: jnp.dtype | str
    compute_dtype: jnp.dtype | str

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, targets: jnp.ndarray | None = None, *, deterministic: bool = True) -> dict[str, jnp.ndarray]:
        compute_dtype = _to_dtype(self.compute_dtype)
        param_dtype = _to_dtype(self.param_dtype)
        batch, seq_len = tokens.shape
        senders = jnp.asarray(self.graph.senders, dtype=jnp.int32)
        receivers = jnp.asarray(self.graph.receivers, dtype=jnp.int32)
        input_heads = jnp.asarray(self.graph.input_heads, dtype=jnp.int32)
        edge_count = len(self.graph.senders)
        attn_heads = self.memory_attn_heads
        kv_heads = self.memory_kv_heads
        if self.state_dim % attn_heads != 0:
            raise ValueError("state_dim must divide memory_attn_heads")
        if attn_heads % kv_heads != 0:
            raise ValueError("memory_attn_heads must divide memory_kv_heads")
        attn_head_dim = self.state_dim // attn_heads

        token_embedding = self.param(
            "token_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.state_dim),
            param_dtype,
        )
        input_scale = self.param("input_scale", nn.initializers.ones, (len(self.graph.input_heads), self.state_dim), param_dtype)
        edge_scale = self.param("edge_scale", nn.initializers.normal(stddev=0.02), (edge_count, self.state_dim), param_dtype)
        edge_bias = self.param("edge_bias", nn.initializers.zeros, (edge_count, self.state_dim), param_dtype)
        init = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
        msg_w1 = self.param("msg_w1", init, (self.state_dim, self.state_dim), param_dtype)
        msg_w2 = self.param("msg_w2", init, (self.state_dim, self.state_dim), param_dtype)
        upd_w = self.param("upd_w", init, (self.state_dim, self.state_dim), param_dtype)
        gate_w = self.param("gate_w", init, (self.state_dim, self.state_dim), param_dtype)
        gate_b = self.param("gate_b", nn.initializers.zeros, (self.state_dim,), param_dtype)
        halt_w = self.param("halt_w", nn.initializers.normal(stddev=0.02), (self.state_dim, 1), param_dtype)
        halt_b = self.param("halt_b", nn.initializers.constant(self.halt_bias_init), (1,), param_dtype)
        q_w = self.param("mem_q_w", init, (self.state_dim, self.state_dim), param_dtype)
        k_w = self.param("mem_k_w", init, (self.state_dim, kv_heads * attn_head_dim), param_dtype)
        v_w = self.param("mem_v_w", init, (self.state_dim, kv_heads * attn_head_dim), param_dtype)
        o_w = self.param("mem_o_w", init, (self.state_dim, self.state_dim), param_dtype)
        msg_norm_scale = self.param("msg_norm_scale", nn.initializers.ones, (self.state_dim,), param_dtype)
        mem_norm_scale = self.param("mem_norm_scale", nn.initializers.ones, (self.state_dim,), param_dtype)
        upd_norm_scale = self.param("upd_norm_scale", nn.initializers.ones, (self.state_dim,), param_dtype)
        out_norm_scale = self.param("out_norm_scale", nn.initializers.ones, (self.state_dim,), param_dtype)

        def dense(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray | None = None) -> jnp.ndarray:
            y = jnp.einsum("...d,df->...f", x.astype(compute_dtype), w.astype(compute_dtype))
            if b is not None:
                y = y + b.astype(compute_dtype)
            return y

        def rms_norm(x: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
            xf = x.astype(jnp.float32)
            normed = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + 1e-5)
            return (normed * scale.astype(jnp.float32)).astype(compute_dtype)

        def memory_attention(state_normed: jnp.ndarray, memory: jnp.ndarray) -> jnp.ndarray:
            b, graph_heads, slots, dim = memory.shape
            q = dense(state_normed, q_w).reshape(b, graph_heads, attn_heads, attn_head_dim)
            k = dense(memory, k_w).reshape(b, graph_heads, slots, kv_heads, attn_head_dim)
            v = dense(memory, v_w).reshape(b, graph_heads, slots, kv_heads, attn_head_dim)
            repeat = attn_heads // kv_heads
            k = jnp.repeat(k, repeat, axis=3)
            v = jnp.repeat(v, repeat, axis=3)
            scores = jnp.einsum("bnhd,bnshd->bnhs", q.astype(jnp.float32), k.astype(jnp.float32)) / jnp.sqrt(attn_head_dim)
            weights = jax.nn.softmax(scores, axis=-1).astype(compute_dtype)
            out = jnp.einsum("bnhs,bnshd->bnhd", weights, v).reshape(b, graph_heads, dim)
            return dense(out, o_w)

        def make_graph_step(target_token: jnp.ndarray):
            def graph_step(carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], step_idx: jnp.ndarray):
                state, memory, mem_pos = carry
                src_state = state[:, senders, :]
                msg = dense(nn.silu(dense(rms_norm(src_state, msg_norm_scale), msg_w1)), msg_w2)
                msg = msg * (1.0 + edge_scale.astype(compute_dtype)[None, :, :]) + edge_bias.astype(compute_dtype)[None, :, :]
                incoming = jax.vmap(lambda m: spherical_segment_merge(m, receivers, self.num_heads))(msg)
                mem_out = memory_attention(rms_norm(state, mem_norm_scale), memory)
                mixed = rms_norm(state + incoming + mem_out, upd_norm_scale)
                upd = dense(nn.silu(mixed), upd_w)
                gate = jax.nn.sigmoid(dense(mixed, gate_w, gate_b))
                residual_scale = self.residual_init / (1.0 + self.residual_decay * step_idx.astype(jnp.float32))
                active = (jnp.linalg.norm(state, axis=-1, keepdims=True) > 1e-6) | (jnp.linalg.norm(incoming, axis=-1, keepdims=True) > 1e-6)
                new_state = jnp.where(active, state + residual_scale.astype(compute_dtype) * gate * upd, state)
                new_state = jnp.nan_to_num(new_state, nan=0.0, posinf=self.state_norm_cap, neginf=-self.state_norm_cap)
                state_norm = jnp.linalg.norm(new_state.astype(jnp.float32), axis=-1, keepdims=True)
                norm_scale = jnp.minimum(1.0, self.state_norm_cap / jnp.maximum(state_norm, 1e-6))
                new_state = (new_state.astype(jnp.float32) * norm_scale).astype(compute_dtype)
                memory = memory.at[:, :, mem_pos, :].set(new_state)
                mem_pos = (mem_pos + 1) % self.memory_slots
                out_state = rms_norm(new_state[:, self.graph.output_head, :], out_norm_scale)
                logits = jnp.einsum("bd,vd->bv", out_state.astype(jnp.float32), token_embedding.astype(jnp.float32))
                halt_logit = dense(out_state, halt_w, halt_b).astype(jnp.float32).squeeze(-1)
                halt_prob = jax.nn.sigmoid(halt_logit)
                probs = jax.nn.softmax(logits, axis=-1)
                conf = jnp.max(probs, axis=-1)
                pred = jnp.argmax(logits, axis=-1)
                top_logits, top_tokens = jax.lax.top_k(logits, 16)
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                if target_token is None:
                    ce = jnp.zeros_like(conf)
                else:
                    ce = -log_probs[jnp.arange(batch), target_token]
                collapse_ids = jnp.asarray(self.collapse_token_ids, dtype=jnp.int32)
                collapse_prob = jnp.sum(jnp.take(jnp.exp(log_probs), collapse_ids, axis=-1), axis=-1)
                active_count = jnp.maximum(jnp.sum(active.squeeze(-1), axis=-1).astype(jnp.float32), 1.0)
                threshold = self.threshold_lo + (self.threshold_hi - self.threshold_lo) * jnp.exp(
                    -self.threshold_alpha * (step_idx.astype(jnp.float32) + 1.0) / jnp.sqrt(active_count + 1.0)
                )
                can_emit = step_idx >= jnp.asarray(self.min_emit_step - 1, dtype=jnp.int32)
                emit_weight = jnp.where(can_emit, jax.nn.sigmoid(self.emit_beta * (halt_prob - threshold)), 0.0)
                hard_emit = jnp.where(can_emit, halt_prob >= threshold, False)
                return (new_state, memory, mem_pos), (
                    ce,
                    pred,
                    conf,
                    threshold,
                    emit_weight,
                    active_count,
                    halt_logit,
                    halt_prob,
                    hard_emit,
                    collapse_prob,
                    top_logits,
                    top_tokens,
                )

            return jax.checkpoint(graph_step, prevent_cse=False)

        def old_graph_step_removed(carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], step_idx: jnp.ndarray):
            state, memory, mem_pos = carry
            src_state = state[:, senders, :]
            msg = dense(nn.silu(dense(rms_norm(src_state, msg_norm_scale), msg_w1)), msg_w2)
            msg = msg * (1.0 + edge_scale.astype(compute_dtype)[None, :, :]) + edge_bias.astype(compute_dtype)[None, :, :]
            incoming = jax.vmap(lambda m: spherical_segment_merge(m, receivers, self.num_heads))(msg)
            mem_out = memory_attention(rms_norm(state, mem_norm_scale), memory)
            mixed = rms_norm(state + incoming + mem_out, upd_norm_scale)
            upd = dense(nn.silu(mixed), upd_w)
            gate = jax.nn.sigmoid(dense(mixed, gate_w, gate_b))
            residual_scale = self.residual_init / (1.0 + self.residual_decay * step_idx.astype(jnp.float32))
            active = (jnp.linalg.norm(state, axis=-1, keepdims=True) > 1e-6) | (jnp.linalg.norm(incoming, axis=-1, keepdims=True) > 1e-6)
            new_state = jnp.where(active, state + residual_scale.astype(compute_dtype) * gate * upd, state)
            new_state = jnp.nan_to_num(new_state, nan=0.0, posinf=self.state_norm_cap, neginf=-self.state_norm_cap)
            state_norm = jnp.linalg.norm(new_state.astype(jnp.float32), axis=-1, keepdims=True)
            norm_scale = jnp.minimum(1.0, self.state_norm_cap / jnp.maximum(state_norm, 1e-6))
            new_state = (new_state.astype(jnp.float32) * norm_scale).astype(compute_dtype)
            memory = memory.at[:, :, mem_pos, :].set(new_state)
            mem_pos = (mem_pos + 1) % self.memory_slots
            out_state = rms_norm(new_state[:, self.graph.output_head, :], out_norm_scale)
            logits = jnp.einsum("bd,vd->bv", out_state.astype(jnp.float32), token_embedding.astype(jnp.float32))
            probs = jax.nn.softmax(logits, axis=-1)
            conf = jnp.max(probs, axis=-1)
            active_count = jnp.maximum(jnp.sum(active.squeeze(-1), axis=-1).astype(jnp.float32), 1.0)
            threshold = self.threshold_lo + (self.threshold_hi - self.threshold_lo) * jnp.exp(
                -self.threshold_alpha * (step_idx.astype(jnp.float32) + 1.0) / jnp.sqrt(active_count + 1.0)
            )
            emit_weight = jax.nn.sigmoid(self.emit_beta * (conf - threshold))
            return (new_state, memory, mem_pos), (logits, conf, threshold, emit_weight, active_count)

        def token_step(carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], item: tuple[jnp.ndarray, jnp.ndarray]):
            token, target_token = item
            state, memory, mem_pos = carry
            tok = jnp.take(token_embedding.astype(compute_dtype), token, axis=0)
            injection = tok[:, None, :] * input_scale.astype(compute_dtype)[None, :, :]
            state = state.at[:, input_heads, :].add(injection)
            memory = memory.at[:, input_heads, mem_pos, :].set(state[:, input_heads, :])
            mem_pos = (mem_pos + 1) % self.memory_slots
            (state, memory, mem_pos), outputs = jax.lax.scan(
                make_graph_step(target_token),
                (state, memory, mem_pos),
                jnp.arange(self.inner_steps, dtype=jnp.int32),
            )
            return (state, memory, mem_pos), outputs

        init_state = jnp.zeros((batch, self.num_heads, self.state_dim), dtype=compute_dtype)
        init_memory = jnp.zeros((batch, self.num_heads, self.memory_slots, self.state_dim), dtype=compute_dtype)
        init_pos = jnp.asarray(0, dtype=jnp.int32)
        if targets is None:
            target_stream = jnp.zeros_like(tokens)
        else:
            target_stream = targets
        _, scanned = jax.lax.scan(token_step, (init_state, init_memory, init_pos), (tokens.T, target_stream.T))
        ce, pred, conf, threshold, emit_weight, active_count, halt_logit, halt_prob, hard_emit, collapse_prob, top_logits, top_tokens = scanned
        return {
            "ce": jnp.transpose(ce, (2, 0, 1)),
            "pred": jnp.transpose(pred, (2, 0, 1)),
            "confidence": jnp.transpose(conf, (2, 0, 1)),
            "threshold": jnp.transpose(threshold, (2, 0, 1)),
            "emit_weight": jnp.transpose(emit_weight, (2, 0, 1)),
            "active_count": jnp.transpose(active_count, (2, 0, 1)),
            "halt_logit": jnp.transpose(halt_logit, (2, 0, 1)),
            "halt_prob": jnp.transpose(halt_prob, (2, 0, 1)),
            "hard_emit": jnp.transpose(hard_emit, (2, 0, 1)),
            "collapse_prob": jnp.transpose(collapse_prob, (2, 0, 1)),
            "top_logits": jnp.transpose(top_logits, (2, 0, 1, 3)),
            "top_tokens": jnp.transpose(top_tokens, (2, 0, 1, 3)),
        }


def estimate_param_count(params: Any) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(x.size for x in leaves))

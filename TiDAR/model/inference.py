"""
Anchor-TiDAR Inference Script

Delayed-anchor implementation with optimistic KV writes + pointer rollback.
Each iteration starts with an anchor token, but it is committed only if accepted.

Algorithm (K=draft_len):
1. Prefill: prompt → sample first K draft tokens from mask positions
2. Each decode step:
   a. Build input: [anchor | draft[1:K]] + [K groups of K masks] 
   b. Forward pass → verify_logits + predraft_logits
   c. Optimistically write K draft KVs in the same forward
   d. Rejection sample: verify draft[1:K] against model, accept/reject
   e. Select proposal based on accept count, substitute new anchor
   f. Advance prefix_len by accepted prefix length (rollback by pointer only)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import jax
import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TiDAR.model.GiantTiDAR import TiDAR
from TiDAR.model.Prepare_mask_token import ensure_tidar_mask_token, resize_embedding_params
from TiDAR.model.config_schema import TiDARConfig, load_typed_config
from TiDAR.model.tidar_core import (
    build_decode_bias_template,
    build_decode_position_template,
    init_kv_cache,
    prefill_prompt_with_draft,
    sample_tokens,
    anchor_rejection_sample_meta,
)
from GIANT.v3.model.checkpoint_manager import load_npz, latest as latest_ckpt

_VALID_DECODE_PREDRAFT_SAMPLING_MODES = {"staged", "single_pass"}


# =============================================================================
# Config / IO
# =============================================================================

def load_configs(
    model_config_path: str | None = None,
    global_config_path: str | None = None,
) -> TiDARConfig:
    return load_typed_config(model_config_path, global_config_path)


def resolve_decode_predraft_sampling_mode(cfg: TiDARConfig) -> str:
    mode = str(getattr(cfg.tidar, "decode_predraft_sampling_mode", "staged")).strip().lower()
    if mode not in _VALID_DECODE_PREDRAFT_SAMPLING_MODES:
        raise ValueError(
            f"Invalid tidar.decode_predraft_sampling_mode='{mode}'. "
            f"Expected one of: {sorted(_VALID_DECODE_PREDRAFT_SAMPLING_MODES)}"
        )
    return mode


def resolve_params_dir(root: Path) -> Path:
    return root if root.name == "params" else root / "params"


def resolve_checkpoint_path(cfg: TiDARConfig, checkpoint: Optional[str], checkpoint_dir: Optional[str]) -> Path:
    base_root = Path(str(cfg.paths.data_root))
    ckpt_dir = Path(checkpoint_dir or cfg.paths.checkpoints_root)
    if not ckpt_dir.is_absolute():
        ckpt_dir = (base_root / ckpt_dir).resolve()
    
    if checkpoint and checkpoint.lower() != "latest":
        path = Path(checkpoint)
        if not path.is_absolute():
            path = (base_root / path).resolve()
        if path.is_dir():
            params_dir = resolve_params_dir(path)
            latest = latest_ckpt(str(params_dir))
            if latest is None:
                raise FileNotFoundError(f"No checkpoints found under {params_dir}")
            return Path(latest)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
        return path
    
    params_dir = resolve_params_dir(ckpt_dir)
    latest = latest_ckpt(str(params_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found under {params_dir}")
    return Path(latest)


def load_tokenizer(cfg: TiDARConfig):
    tok_cfg = cfg.tokenizer
    if tok_cfg.use_custom:
        tokenizer = AutoTokenizer.from_pretrained(tok_cfg.custom_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_cfg.name,
            use_fast=True,
            cache_dir=tok_cfg.cache_dir,
        )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def build_model(cfg: TiDARConfig, vocab_size: int, context_length: int, draft_len: int) -> TiDAR:
    model_cfg = cfg.model
    return TiDAR(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_cfg.embedding_size,
        n_heads=model_cfg.num_heads,
        num_kv_heads=model_cfg.num_kv_heads,
        rope_dim=model_cfg.rope_dim,
        rope_theta=float(getattr(model_cfg, "rope_theta", 10000.0)),
        d_ff=model_cfg.feed_forward_size,
        n_layers=model_cfg.num_layers,
        dropout_rate=0.0,
        param_dtype=model_cfg.param_dtype,
        compute_dtype=model_cfg.compute_dtype,
        use_remat=model_cfg.use_remat,
        draft_len=int(draft_len),
    )


def tokenize_prompt(tokenizer, prompt: str, max_len: int, *, strip_eos: bool) -> np.ndarray:
    if strip_eos:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if ids and tokenizer.eos_token_id is not None and ids[-1] == tokenizer.eos_token_id:
            ids = ids[:-1]
    else:
        ids = tokenizer(prompt, return_tensors="np").input_ids[0].tolist()
    
    if len(ids) >= max_len:
        ids = ids[-max_len:]
    return np.asarray(ids, dtype=np.int32)


def load_params(path: Path):
    params = load_npz(path)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), params)


def parse_bool_flag(value: Optional[str], *, default: bool) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(
        f"Invalid boolean value '{value}'. Use one of: true/false, yes/no, 1/0."
    )


# =============================================================================
# Anchor-TiDAR Generation
# =============================================================================

def make_anchor_tidar_generate_fn(
    model: TiDAR,
    *,
    cache_len: int,
    draft_len: int,
    mask_id: int,
    pad_token_id: int,
    eos_id: int,
    stop_on_eos: bool,
    temperature: float,
    top_k: int,
    bias_value: float,
    decode_predraft_sampling_mode: str = "staged",
    verbose_stats: bool = False,
):
    """
    Build JIT-compiled Anchor-TiDAR generation function.
    
    Returns function with signature:
        generate(params, cache_vars, out_ids, prefix_len, max_steps, prev_logit, rng_key, initial_draft_logits)
        -> (out_ids, final_prefix_len, generated_count, stats_dict)
    """
    # Pre-build templates
    decode_bias = jax.device_put(build_decode_bias_template(cache_len, draft_len, bias_value))
    position_template = jax.device_put(build_decode_position_template(draft_len))
    
    # Pre-build mask token arrays
    predraft_masks = jnp.full((draft_len * draft_len,), mask_id, dtype=jnp.int32)
    predraft_masks = jax.device_put(predraft_masks)
    
    idx_k = jax.device_put(jnp.arange(draft_len, dtype=jnp.int32))
    
    def decode_apply(params, cache_vars, step_tokens, pos_ids, prefix_len):
        """
        Forward pass for decode step with optimistic KV write for current draft.

        The first K step tokens (current_draft) are written to cache at `prefix_len`.
        They are not visible to the current attention prefix because prefix validity
        is still masked by the same `prefix_len`. Future steps commit/rollback by
        moving the prefix pointer only.
        """
        logits, mutated = model.apply(
            {"params": params, "cache": cache_vars},
            step_tokens[None, :],  # [1, q_len]
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            cache_write_len=draft_len,
            attn_bias=decode_bias,
            position_ids=pos_ids[None, :],
            kv_cache_len=cache_len,
            mutable=["cache"],
        )
        return logits[0], mutated["cache"]  # [q_len, V], cache
    
    use_single_pass_predraft_sampling = decode_predraft_sampling_mode == "single_pass"

    @jax.jit
    def generate(
        params,
        cache_vars,
        out_ids: jnp.ndarray,          # [buffer_len] pre-filled with prompt
        prefix_len: jnp.ndarray,       # scalar int32
        max_steps: jnp.ndarray,        # scalar int32
        prev_logit: jnp.ndarray,       # [V] logit for first anchor
        rng_key: jax.Array,
        initial_draft_logits: jnp.ndarray,  # [K, V] logits for initial draft
    ):
        """
        Main Anchor-TiDAR generation loop.
        
        Returns: (out_ids, final_prefix_len, generated_count, stats)
        """
        prefix_len = prefix_len.astype(jnp.int32)
        generated = jnp.array(0, dtype=jnp.int32)
        done = jnp.array(False)
        
        # Stats tracking
        total_accepts = jnp.array(0, dtype=jnp.int32)
        n_iterations = jnp.array(0, dtype=jnp.int32)
        max_accepts = jnp.array(0, dtype=jnp.int32)
        
        # === Step 1: Sample first anchor from prev_logit ===
        rng_key, anchor = sample_tokens(rng_key, prev_logit, temperature, top_k)
        anchor = anchor.astype(jnp.int32)  # scalar

        # === Step 2: Get initial draft via K mask tokens ===
        init_logits = initial_draft_logits  # [K, V]
        rng_key, init_draft = sample_tokens(rng_key, init_logits, temperature, top_k)
        init_draft = init_draft.astype(jnp.int32)  # [K]

        # Set up current_draft with sampled anchor at position 0.
        current_draft = init_draft.at[0].set(anchor)  # [K]
        current_draft_logits = init_logits  # [K, V]
        
        def cond_fn(state):
            _, _, _, _, _, generated, _, done, _, _, _ = state
            return (generated < max_steps) & (~done)
        
        def body_fn(state):
            (rng, cache, out, prefix_len, current_draft, generated, 
             current_draft_logits, done, total_accepts, n_iters, max_acc) = state
            
            # === Build decode input ===
            # Layout: [current_draft (K)] + [predraft_masks (K*K)]
            step_tokens = jnp.concatenate([current_draft, predraft_masks])  # [K + K*K]
            step_pos_ids = (prefix_len + position_template).astype(jnp.int32)
            
            # === Forward pass + optimistic KV write ===
            # Optimistically writes current_draft KVs at [prefix_len, ..., prefix_len+K-1].
            logits, cache = decode_apply(params, cache, step_tokens, step_pos_ids, prefix_len)
            
            # Extract verify logits: positions 0..K-1 predict tokens at 1..K
            verify_logits = logits[:draft_len]  # [K, V]
            
            # Extract predraft logits: reshape to [K, K, V]
            predraft_logits = logits[draft_len:].reshape(draft_len, draft_len, -1)
            
            # === Rejection sampling ===
            anchor_tok = current_draft[0]
            draft_toks = current_draft[1:]  # [K-1] tokens to verify

            if use_single_pass_predraft_sampling:
                # Legacy path: sample all K predraft rows in one pass, then choose row after rejection.
                rng, predraft_flat_tokens = sample_tokens(
                    rng,
                    predraft_logits.reshape(-1, predraft_logits.shape[-1]),
                    temperature,
                    top_k,
                )
                predraft_tokens = predraft_flat_tokens.reshape(draft_len, draft_len).astype(jnp.int32)

                rng, accept_count, _, proposal_idx, next_anchor = anchor_rejection_sample_meta(
                    rng,
                    anchor_token=anchor_tok,
                    draft_tokens=draft_toks,
                    verify_logits=verify_logits,
                    draft_logits=current_draft_logits,
                    temperature=temperature,
                    top_k=top_k,
                )
                proposal_logits = predraft_logits[proposal_idx]  # [K, V]
                next_draft = predraft_tokens[proposal_idx].at[0].set(next_anchor)
            else:
                # Staged path (default): do rejection first, then sample only selected row.
                rng, accept_count, _, proposal_idx, next_anchor = anchor_rejection_sample_meta(
                    rng,
                    anchor_token=anchor_tok,
                    draft_tokens=draft_toks,
                    verify_logits=verify_logits,
                    draft_logits=current_draft_logits,
                    temperature=temperature,
                    top_k=top_k,
                )
                proposal_logits = predraft_logits[proposal_idx]  # [K, V]
                rng, next_draft = sample_tokens(rng, proposal_logits, temperature, top_k)
                next_draft = next_draft.astype(jnp.int32).at[0].set(next_anchor)

            # accept_count: committed prefix length from current_draft.
            #               Minimum 1 (anchor), maximum K (full draft accepted)
            
            # === Clamp accept_count to remaining budget ===
            remaining = (max_steps - generated).astype(jnp.int32)
            eff_accept = jnp.minimum(accept_count, remaining)
            
            # === Check for EOS in accepted prefix of current_draft ===
            has_eos = jnp.array(False)
            if stop_on_eos and eos_id >= 0:
                eos_mask = (current_draft == eos_id) & (idx_k < eff_accept)
                first_eos = jnp.where(
                    jnp.any(eos_mask),
                    jnp.argmax(eos_mask.astype(jnp.int32)),
                    draft_len
                ).astype(jnp.int32)
                has_eos = first_eos < eff_accept
                eff_accept = jnp.where(has_eos, first_eos + 1, eff_accept)
            
            # === Commit by pointer only ===
            # Cache already contains optimistic KVs for current_draft at prefix_len.
            # We commit only the accepted prefix by advancing prefix_len.
            commit_padded = jnp.where(
                idx_k < eff_accept,
                current_draft,
                pad_token_id
            ).astype(jnp.int32)
            
            # === Update output buffer ===
            out = jax.lax.dynamic_update_slice(out, commit_padded[:draft_len], (prefix_len,))
            
            # === Advance state ===
            prefix_len2 = prefix_len + eff_accept
            generated2 = generated + eff_accept
            done2 = done | (generated2 >= max_steps) | has_eos
            
            # Update draft logits for next iteration
            next_draft_logits = proposal_logits
            
            # Stats
            total_accepts2 = total_accepts + eff_accept
            n_iters2 = n_iters + 1
            max_acc2 = jnp.maximum(max_acc, eff_accept)
            
            return (
                rng, cache, out, prefix_len2, next_draft, generated2,
                next_draft_logits, done2, total_accepts2, n_iters2, max_acc2
            )
        
        state0 = (
            rng_key, cache_vars, out_ids, prefix_len, current_draft, generated,
            current_draft_logits, done, total_accepts, n_iterations, max_accepts
        )
        
        state_final = jax.lax.while_loop(cond_fn, body_fn, state0)
        
        (_, _, out_final, prefix_len_final, _, generated_final,
         _, _, total_accepts_final, n_iters_final, max_accepts_final) = state_final
        
        # Build stats dict
        stats = {
            "total_accepts": total_accepts_final,
            "n_iterations": n_iters_final,
            "avg_accept_per_iter": total_accepts_final.astype(jnp.float32) / jnp.maximum(n_iters_final, 1),
            "max_accept_per_iter": max_accepts_final,
        }
        
        return out_final, prefix_len_final, generated_final, stats
    
    return generate


def make_anchor_tidar_component_profile_fns(
    model: TiDAR,
    *,
    cache_len: int,
    draft_len: int,
    mask_id: int,
    temperature: float,
    top_k: int,
    bias_value: float,
    decode_predraft_sampling_mode: str = "staged",
):
    """
    Build JIT-compiled component fns for decode breakdown profiling.

    Returns:
      decode_forward(params, cache_vars, current_draft, prefix_len) -> (logits, cache)
      postprocess(rng, current_draft, current_draft_logits, logits) -> (rng, accept_count, next_draft, next_draft_logits)
    """
    decode_bias = jax.device_put(build_decode_bias_template(cache_len, draft_len, bias_value))
    position_template = jax.device_put(build_decode_position_template(draft_len))
    predraft_masks = jax.device_put(jnp.full((draft_len * draft_len,), mask_id, dtype=jnp.int32))
    use_single_pass_predraft_sampling = decode_predraft_sampling_mode == "single_pass"

    @jax.jit
    def decode_forward(
        params,
        cache_vars,
        current_draft: jnp.ndarray,   # [K]
        prefix_len: jnp.ndarray,      # scalar
    ):
        step_tokens = jnp.concatenate([current_draft, predraft_masks])  # [K + K*K]
        step_pos_ids = (prefix_len + position_template).astype(jnp.int32)
        logits, mutated = model.apply(
            {"params": params, "cache": cache_vars},
            step_tokens[None, :],
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            cache_write_len=draft_len,
            attn_bias=decode_bias,
            position_ids=step_pos_ids[None, :],
            kv_cache_len=cache_len,
            mutable=["cache"],
        )
        return logits[0], mutated["cache"]

    @jax.jit
    def postprocess(
        rng: jax.Array,
        current_draft: jnp.ndarray,         # [K]
        current_draft_logits: jnp.ndarray,  # [K, V]
        logits: jnp.ndarray,                # [K + K*K, V]
    ):
        verify_logits = logits[:draft_len]
        predraft_logits = logits[draft_len:].reshape(draft_len, draft_len, -1)
        rng, accept_count, _, proposal_idx, next_anchor = anchor_rejection_sample_meta(
            rng,
            anchor_token=current_draft[0],
            draft_tokens=current_draft[1:],
            verify_logits=verify_logits,
            draft_logits=current_draft_logits,
            temperature=temperature,
            top_k=top_k,
        )

        proposal_logits = predraft_logits[proposal_idx]
        if use_single_pass_predraft_sampling:
            rng, predraft_flat_tokens = sample_tokens(
                rng,
                predraft_logits.reshape(-1, predraft_logits.shape[-1]),
                temperature,
                top_k,
            )
            predraft_tokens = predraft_flat_tokens.reshape(draft_len, draft_len).astype(jnp.int32)
            next_draft = predraft_tokens[proposal_idx].at[0].set(next_anchor)
        else:
            rng, next_draft = sample_tokens(rng, proposal_logits, temperature, top_k)
            next_draft = next_draft.astype(jnp.int32).at[0].set(next_anchor)

        return rng, accept_count, next_draft, proposal_logits

    return decode_forward, postprocess


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Anchor-TiDAR inference")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--global_config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="latest")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--draft_len", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--stop_on_eos", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strip_eos", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--bench_runs", type=int, default=3)
    parser.add_argument("--report_compile_time", type=str, default="true")
    parser.add_argument("--report_run_only_time", type=str, default="true")
    parser.add_argument("--component_breakdown_iters", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_configs(args.config, args.global_config)
    
    jax.config.update("jax_default_matmul_precision", cfg.model.compute_dtype)
    
    # Get settings from args or config
    temperature = args.temperature if args.temperature is not None else float(cfg.inference.temperature)
    top_k = args.top_k if args.top_k is not None else int(cfg.inference.top_k)
    draft_len = args.draft_len if args.draft_len is not None else int(cfg.tidar.draft_length)
    max_steps = args.steps if args.steps is not None else int(cfg.inference.max_decode_steps)
    bias_value = float(cfg.tidar.attn_bias_value)
    decode_predraft_sampling_mode = resolve_decode_predraft_sampling_mode(cfg)
    report_compile_time = parse_bool_flag(args.report_compile_time, default=True)
    report_run_only_time = parse_bool_flag(args.report_run_only_time, default=True)
    
    if args.stop_on_eos is not None:
        stop_on_eos = args.stop_on_eos.lower() in ("true", "1", "yes")
    else:
        stop_on_eos = bool(cfg.inference.stop_on_eos)
    
    model_context_length = int(cfg.model.context_length)
    context_length = args.context_length if args.context_length is not None else model_context_length
    
    # Validation
    if max_steps <= 0:
        raise ValueError("steps must be > 0")
    if top_k < 0:
        raise ValueError("top_k must be >= 0")
    if draft_len <= 1:
        raise ValueError("draft_len must be > 1")
    if context_length > model_context_length:
        raise ValueError(f"context_length {context_length} exceeds model max {model_context_length}")
    if args.warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if args.bench_runs < 0:
        raise ValueError("bench_runs must be >= 0")
    if args.component_breakdown_iters < 0:
        raise ValueError("component_breakdown_iters must be >= 0")
    
    # Load checkpoint
    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint, args.checkpoint_dir)
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(cfg)
    prompt_ids = tokenize_prompt(tokenizer, args.prompt, context_length, strip_eos=args.strip_eos)
    if prompt_ids.size == 0:
        raise ValueError("Prompt produced zero tokens")
    
    prompt_len = int(prompt_ids.shape[0])
    print(f"Prompt tokens: {prompt_len}")
    
    # Ensure we have room: prompt + max_steps + draft_len+1 (max commit per iter)
    required_len = prompt_len + max_steps + draft_len + 1
    if required_len > context_length:
        raise ValueError(f"Required length {required_len} exceeds context_length {context_length}")
    
    # Keep decode KV window tight to the actual request budget.
    cache_len = required_len
    
    # Setup mask token
    base_token = getattr(cfg.tokenizer, "mask_token_override", None) or "[MASK]"
    mask_token, mask_id, added_tokens = ensure_tidar_mask_token(tokenizer, base_token=base_token)
    if added_tokens:
        print(f"Added mask token '{mask_token}' (id={mask_id})")
    else:
        print(f"Using mask token '{mask_token}' (id={mask_id})")
    
    # Build model and load params
    model = build_model(cfg, len(tokenizer), context_length, draft_len)
    params = load_params(checkpoint_path)
    
    rng = jax.random.PRNGKey(args.seed)
    rng, resize_key = jax.random.split(rng)
    params, added_rows = resize_embedding_params(params, len(tokenizer), key=resize_key)
    if added_rows:
        print(f"Expanded embeddings by {added_rows} rows")
    
    params = jax.device_put(params)
    
    # Get pad/eos tokens
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    
    eos_id = tokenizer.eos_token_id
    eos_id_for_jit = int(eos_id) if eos_id is not None else -1
    
    # Initialize KV cache
    cache_vars = init_kv_cache(model, batch_size=1, pad_token_id=pad_token_id)
    cache_vars = jax.device_put(cache_vars)
    
    # Prefill prompt + initial draft (single pass)
    print("Prefilling prompt + initial draft...")
    cache_vars, prefix_len, prev_logit, initial_draft_logits = prefill_prompt_with_draft(
        model,
        params,
        cache_vars,
        jnp.asarray(prompt_ids),
        draft_len=draft_len,
        mask_id=int(mask_id),
        kv_cache_len=cache_len,
        bias_value=bias_value,
    )
    
    # Prepare output buffer
    buffer_len = required_len
    out_host = np.full((buffer_len,), pad_token_id, dtype=np.int32)
    out_host[:prompt_len] = prompt_ids
    out_ids = jax.device_put(jnp.asarray(out_host))
    
    # Build generator
    generate_fn = make_anchor_tidar_generate_fn(
        model,
        cache_len=cache_len,
        draft_len=draft_len,
        mask_id=int(mask_id),
        pad_token_id=int(pad_token_id),
        eos_id=eos_id_for_jit,
        stop_on_eos=stop_on_eos,
        temperature=float(temperature),
        top_k=int(top_k),
        bias_value=bias_value,
        decode_predraft_sampling_mode=decode_predraft_sampling_mode,
        verbose_stats=args.verbose,
    )
    
    # Run generation
    print(
        "Generating "
        f"{max_steps} tokens with draft_len={draft_len}, temp={temperature}, top_k={top_k}, "
        f"decode_predraft_sampling_mode={decode_predraft_sampling_mode}..."
    )
    assert prefix_len > 0, "prefix_len must be > 0 before starting the decode loop"
    generate_call_args = (
        params,
        cache_vars,
        out_ids,
        jnp.asarray(prefix_len, dtype=jnp.int32),
        jnp.asarray(max_steps, dtype=jnp.int32),
        prev_logit,
        rng,
        initial_draft_logits,
    )

    def run_generate_once():
        out = generate_fn(*generate_call_args)
        out[0].block_until_ready()
        return out

    compile_start = time.perf_counter()
    out_first = run_generate_once()
    compile_and_first_run_s = time.perf_counter() - compile_start

    for _ in range(args.warmup_runs):
        _ = run_generate_once()

    steady_run_times: list[float] = []
    out_last = out_first
    if report_run_only_time and args.bench_runs > 0:
        for _ in range(args.bench_runs):
            t0 = time.perf_counter()
            out_last = run_generate_once()
            steady_run_times.append(time.perf_counter() - t0)

    out_ids_final, final_len, generated, stats = out_last
    decode_time = compile_and_first_run_s
    
    # Extract output
    final_len = int(np.asarray(final_len))
    out_tokens = np.asarray(out_ids_final[:final_len], dtype=np.int32)
    
    # Decode text
    text = tokenizer.decode(out_tokens, skip_special_tokens=True)
    if stop_on_eos and eos_id is not None:
        eos_hits = np.where(out_tokens == eos_id)[0]
        if eos_hits.size > 0:
            cut = int(eos_hits[0])
            text = tokenizer.decode(out_tokens[:cut], skip_special_tokens=True) + "<EOS>"
    
    print("\n==================== RESULT ====================")
    print(text)
    print("================================================")
    
    if args.verbose:
        gen_count = int(np.asarray(generated))
        n_iters = int(np.asarray(stats["n_iterations"]))
        avg_accept = float(np.asarray(stats["avg_accept_per_iter"]))
        max_accept = int(np.asarray(stats["max_accept_per_iter"]))
        toks_per_s = gen_count / decode_time if decode_time > 0 else float("inf")
        steady_decode_time_mean = float(np.mean(steady_run_times)) if steady_run_times else None
        steady_decode_time_std = float(np.std(steady_run_times)) if steady_run_times else None
        steady_toks_per_s = (
            gen_count / steady_decode_time_mean
            if steady_decode_time_mean is not None and steady_decode_time_mean > 0
            else None
        )
        steady_toks_per_s_std = (
            (gen_count * steady_decode_time_std) / (steady_decode_time_mean ** 2)
            if (
                steady_decode_time_mean is not None
                and steady_decode_time_std is not None
                and steady_decode_time_mean > 0
            )
            else None
        )
        
        print("\n[stats]")
        print(f"  prompt_tokens:     {prompt_len}")
        print(f"  generated_tokens:  {gen_count}")
        print(f"  n_iterations:      {n_iters}")
        print(f"  avg_accept/iter:   {avg_accept:.2f}")
        print(f"  max_accept/iter:   {max_accept}")
        print(f"  decode_time_s:     {decode_time:.4f}")
        print(f"  tokens_per_second: {toks_per_s:.2f}")
        print("\n[benchmark]")
        print(f"  warmup_runs:       {args.warmup_runs}")
        print(f"  bench_runs:        {args.bench_runs}")
        print(f"  report_compile:    {report_compile_time}")
        print(f"  report_run_only:   {report_run_only_time}")
        if report_compile_time:
            print(f"  compile_and_first_run_s:            {compile_and_first_run_s:.6f}")
            print(f"  compile_and_first_run_tokens_per_s: {toks_per_s:.6f}")
        if report_run_only_time and steady_decode_time_mean is not None:
            print(f"  steady_decode_time_s_mean:          {steady_decode_time_mean:.6f}")
            print(f"  steady_decode_time_s_std:           {steady_decode_time_std:.6f}")
            print(f"  steady_tokens_per_second_mean:      {steady_toks_per_s:.6f}")
            print(f"  steady_tokens_per_second_std:       {steady_toks_per_s_std:.6f}")

    if args.component_breakdown_iters > 0:
        decode_forward_fn, postprocess_fn = make_anchor_tidar_component_profile_fns(
            model,
            cache_len=cache_len,
            draft_len=draft_len,
            mask_id=int(mask_id),
            temperature=float(temperature),
            top_k=int(top_k),
            bias_value=bias_value,
            decode_predraft_sampling_mode=decode_predraft_sampling_mode,
        )

        profile_rng = jax.random.PRNGKey(args.seed + 17)
        profile_rng, profile_anchor = sample_tokens(profile_rng, prev_logit, float(temperature), int(top_k))
        profile_rng, profile_init_draft = sample_tokens(
            profile_rng, initial_draft_logits, float(temperature), int(top_k)
        )
        profile_current_draft = profile_init_draft.astype(jnp.int32).at[0].set(
            profile_anchor.astype(jnp.int32)
        )
        profile_current_draft_logits = initial_draft_logits
        profile_prefix_len = jnp.asarray(prefix_len, dtype=jnp.int32)
        profile_cache = cache_vars

        # Warmup compile and stabilize.
        for _ in range(2):
            warm_logits, profile_cache = decode_forward_fn(
                params, profile_cache, profile_current_draft, profile_prefix_len
            )
            (
                profile_rng,
                _,
                profile_current_draft,
                profile_current_draft_logits,
            ) = postprocess_fn(
                profile_rng,
                profile_current_draft,
                profile_current_draft_logits,
                warm_logits,
            )

        comp_iters = args.component_breakdown_iters

        t0 = time.perf_counter()
        for _ in range(comp_iters):
            warm_logits, profile_cache = decode_forward_fn(
                params, profile_cache, profile_current_draft, profile_prefix_len
            )
        warm_logits.block_until_ready()
        decode_forward_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(comp_iters):
            (
                profile_rng,
                _,
                profile_current_draft,
                profile_current_draft_logits,
            ) = postprocess_fn(
                profile_rng,
                profile_current_draft,
                profile_current_draft_logits,
                warm_logits,
            )
        profile_current_draft_logits.block_until_ready()
        postprocess_s = time.perf_counter() - t0

        decode_ms = (decode_forward_s / comp_iters) * 1000.0
        post_ms = (postprocess_s / comp_iters) * 1000.0
        total = decode_forward_s + postprocess_s
        post_share = (postprocess_s / total) * 100.0 if total > 0 else 0.0

        print("\n[component_breakdown]")
        print(f"  component_breakdown_iters:           {comp_iters}")
        print(f"  decode_forward_ms_per_iter:          {decode_ms:.6f}")
        print(f"  reject_sample_postprocess_ms_per_iter:{post_ms:.6f}")
        print(f"  postprocess_share_pct:               {post_share:.4f}")


if __name__ == "__main__":
    main()

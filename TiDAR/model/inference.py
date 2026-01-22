"""
Anchor-TiDAR Inference Script

Clean implementation of speculative decoding with guaranteed +1 progress per step.
Each iteration starts with an "anchor" token that's already AR-sampled and committed.

Algorithm (K=draft_len):
1. Prefill: prompt → sample first K draft tokens from mask positions
2. Each decode step:
   a. Build input: [anchor | draft[1:K]] + [K groups of K masks] 
   b. Forward pass → verify_logits + predraft_logits
   c. Rejection sample: verify draft[1:K] against model, accept/reject
   d. Select proposal based on accept count, substitute new anchor
   e. Write accepted tokens to cache, advance prefix_len
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TiDAR.model.GiantTiDAR import TiDAR
from TiDAR.model.Prepare_mask_token import ensure_tidar_mask_token, resize_embedding_params
from TiDAR.model.tidar_core import (
    build_decode_bias_template,
    build_decode_position_template,
    build_prefill_draft_bias_template,
    init_kv_cache,
    prefill_prompt,
    sample_tokens,
    anchor_rejection_sample,
)
from GIANT.v2.model.checkpoint_manager import load_npz, latest as latest_ckpt


# =============================================================================
# Config / IO
# =============================================================================

def _resolve_config_path(value: str | None, default: Path) -> Path:
    if value is None:
        return default
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate


def load_configs(
    model_config_path: str | None = None,
    global_config_path: str | None = None,
) -> OmegaConf:
    model_dir = Path(__file__).resolve().parent
    project_root = model_dir.parent
    resolved_model_cfg = _resolve_config_path(model_config_path, model_dir / "Config.yml")
    resolved_global_cfg = _resolve_config_path(global_config_path, project_root / "Global_Config.yml")
    cfg = OmegaConf.merge(
        OmegaConf.load(resolved_global_cfg),
        OmegaConf.load(resolved_model_cfg),
    )
    
    base_prefix_str = cfg.paths.get("data_root", "") if "paths" in cfg else ""
    base_prefix = Path(base_prefix_str) if base_prefix_str else None
    
    def resolve_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(str(value))
        if path.is_absolute() or base_prefix is None:
            return str(path)
        return str((base_prefix / path).resolve())
    
    if base_prefix is not None:
        cfg.paths.data_root = str(base_prefix)
    else:
        cfg.paths.data_root = str(project_root)
    
    for key in ("checkpoints_root", "hf_cache_root", "logs_root"):
        if key in cfg.paths and cfg.paths[key] is not None:
            resolved = resolve_path(cfg.paths[key])
            if resolved is not None:
                cfg.paths[key] = resolved
    
    if "tokenizer" in cfg:
        cache_dir = cfg.tokenizer.get("cache_dir")
        if cache_dir:
            cache_path = Path(str(cache_dir))
            if not cache_path.is_absolute():
                cfg.tokenizer.cache_dir = str(Path(cfg.paths.data_root) / cache_path)
        custom_path = cfg.tokenizer.get("custom_path")
        if custom_path:
            custom_path = Path(str(custom_path))
            if not custom_path.is_absolute():
                cfg.tokenizer.custom_path = str(Path(cfg.paths.data_root) / custom_path)
    
    return cfg


def resolve_checkpoint_path(cfg, checkpoint: Optional[str], checkpoint_dir: Optional[str]) -> Path:
    base_root = Path(cfg.paths.data_root)
    ckpt_dir = Path(checkpoint_dir or cfg.paths.checkpoints_root)
    if not ckpt_dir.is_absolute():
        ckpt_dir = (base_root / ckpt_dir).resolve()
    
    if checkpoint and checkpoint.lower() != "latest":
        path = Path(checkpoint)
        if not path.is_absolute():
            path = (base_root / path).resolve()
        if path.is_dir():
            latest = latest_ckpt(str(path))
            if latest is None:
                raise FileNotFoundError(f"No checkpoints found under {path}")
            return Path(latest)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")
        return path
    
    latest = latest_ckpt(str(ckpt_dir))
    if latest is None:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")
    return Path(latest)


def load_tokenizer(cfg):
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


def build_model(cfg, vocab_size: int, context_length: int) -> TiDAR:
    model_cfg = cfg.model
    return TiDAR(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=model_cfg.embedding_size,
        n_heads=model_cfg.num_heads,
        d_ff=model_cfg.feed_forward_size,
        n_layers=model_cfg.num_layers,
        dropout_rate=0.0,
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
    verbose_stats: bool = False,
):
    """
    Build JIT-compiled Anchor-TiDAR generation function.
    
    Returns function with signature:
        generate(params, cache_vars, out_ids, prefix_len, max_steps, prev_logit, rng_key)
        -> (out_ids, final_prefix_len, generated_count, stats_dict)
    """
    q_len = draft_len + draft_len * draft_len
    
    # Pre-build templates
    decode_bias = jax.device_put(build_decode_bias_template(cache_len, draft_len, bias_value))
    prefill_bias = jax.device_put(build_prefill_draft_bias_template(cache_len, draft_len))
    position_template = jax.device_put(build_decode_position_template(draft_len))
    
    # Pre-build mask token arrays
    predraft_masks = jnp.full((draft_len * draft_len,), mask_id, dtype=jnp.int32)
    predraft_masks = jax.device_put(predraft_masks)
    
    initial_draft_masks = jnp.full((draft_len,), mask_id, dtype=jnp.int32)
    initial_draft_masks = jax.device_put(initial_draft_masks)
    
    idx_k = jax.device_put(jnp.arange(draft_len, dtype=jnp.int32))
    idx_kp1 = jax.device_put(jnp.arange(draft_len + 1, dtype=jnp.int32))
    
    def initial_draft_apply(params, cache_vars, mask_tokens, pos_ids, prefix_len):
        """Forward pass for initial K mask tokens to get first draft."""
        logits = model.apply(
            {"params": params, "cache": cache_vars},
            mask_tokens[None, :],  # [1, K]
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            attn_bias=prefill_bias,
            position_ids=pos_ids[None, :],
            kv_cache_len=cache_len,
        )
        return logits[0]  # [K, V]
    
    def decode_apply(params, cache_vars, step_tokens, pos_ids, prefix_len):
        """Forward pass for decode step."""
        logits = model.apply(
            {"params": params, "cache": cache_vars},
            step_tokens[None, :],  # [1, q_len]
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=False,
            prefix_len=prefix_len,
            attn_bias=decode_bias,
            position_ids=pos_ids[None, :],
            kv_cache_len=cache_len,
        )
        return logits[0]  # [q_len, V]
    
    def cache_write_apply(params, cache_vars, tokens, pos_ids, cur_index):
        """Write tokens to KV cache and return logits."""
        logits, mutated = model.apply(
            {"params": params, "cache": cache_vars},
            tokens[None, :],  # [1, N]
            deterministic=True,
            use_kv_cache=True,
            write_to_cache=True,
            cur_index=cur_index,
            position_ids=pos_ids[None, :],
            kv_cache_len=cache_len,
            mutable=["cache"],
        )
        return logits[0], mutated["cache"]  # [N, V], cache
    
    @jax.jit
    def generate(
        params,
        cache_vars,
        out_ids: jnp.ndarray,          # [buffer_len] pre-filled with prompt
        prefix_len: jnp.ndarray,       # scalar int32
        max_steps: jnp.ndarray,        # scalar int32
        prev_logit: jnp.ndarray,       # [V] logit for first anchor
        rng_key: jax.Array,
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
        
        # === Step 1: Sample first anchor from prev_logit ===
        rng_key, anchor = sample_tokens(rng_key, prev_logit, temperature, top_k)
        anchor = anchor.astype(jnp.int32)  # scalar
        
        # === Step 2: COMMIT first anchor immediately ===
        # Write first anchor to cache and output
        anchor_pos = jnp.array([prefix_len], dtype=jnp.int32)
        _, cache_vars = cache_write_apply(
            params, cache_vars, anchor[None], anchor_pos, prefix_len
        )
        out_ids = out_ids.at[prefix_len].set(anchor)
        prefix_len = prefix_len + 1
        generated = generated + 1
        
        # Check if we're already done
        done = generated >= max_steps
        if stop_on_eos and eos_id >= 0:
            done = done | (anchor == eos_id)
        
        # === Step 3: Get initial draft via K mask tokens ===
        init_pos_ids = prefix_len + jnp.arange(draft_len, dtype=jnp.int32)
        init_logits = initial_draft_apply(
            params, cache_vars, initial_draft_masks, init_pos_ids, prefix_len
        )  # [K, V]
        
        rng_key, init_draft = sample_tokens(rng_key, init_logits, temperature, top_k)
        init_draft = init_draft.astype(jnp.int32)  # [K]
        
        # Set up current_draft with anchor at position 0
        # The "anchor" for subsequent iterations comes from next_draft[0]
        # For now, use the last committed token (our first anchor) as position 0
        current_draft = init_draft.at[0].set(anchor)  # [K]
        current_draft_logits = init_logits  # [K, V]
        
        def cond_fn(state):
            _, _, _, _, _, generated, _, done, _, _ = state
            return (generated < max_steps) & (~done)
        
        def body_fn(state):
            (rng, cache, out, prefix_len, current_draft, generated, 
             current_draft_logits, done, total_accepts, n_iters) = state
            
            # The anchor at current_draft[0] is ALREADY committed (in previous iteration)
            # We only need to verify and commit positions 1..K-1 and possibly a bonus
            
            # === Build decode input ===
            # Layout: [current_draft (K)] + [predraft_masks (K*K)]
            step_tokens = jnp.concatenate([current_draft, predraft_masks])  # [K + K*K]
            step_pos_ids = (prefix_len - 1 + position_template).astype(jnp.int32)
            # Note: prefix_len - 1 because current_draft[0] (anchor) is at position prefix_len - 1
            
            # === Forward pass ===
            # Anchor was already committed; exclude it from the prefix cache used in attention.
            decode_prefix_len = prefix_len - 1
            logits = decode_apply(params, cache, step_tokens, step_pos_ids, decode_prefix_len)
            
            # Extract verify logits: positions 0..K-1 predict tokens at 1..K
            verify_logits = logits[:draft_len]  # [K, V]
            
            # Extract predraft logits: reshape to [K, K, V]
            predraft_logits = logits[draft_len:].reshape(draft_len, draft_len, -1)
            
            # === Sample from predraft groups ===
            rng, predraft_flat_tokens = sample_tokens(
                rng, 
                predraft_logits.reshape(-1, predraft_logits.shape[-1]),
                temperature, 
                top_k
            )
            predraft_tokens = predraft_flat_tokens.reshape(draft_len, draft_len).astype(jnp.int32)
            
            # === Rejection sampling ===
            anchor_tok = current_draft[0]  # Already committed!
            draft_toks = current_draft[1:]  # [K-1] tokens to verify
            
            rng, accept_count, committed, next_draft = anchor_rejection_sample(
                rng,
                anchor_token=anchor_tok,
                draft_tokens=draft_toks,
                verify_logits=verify_logits,
                draft_logits=current_draft_logits,
                predraft_tokens=predraft_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            # accept_count: number of NEW tokens to commit (not including anchor!)
            #               Minimum 1 (just resampled), maximum K (all drafts + bonus)
            # committed: [K] tokens to commit after the anchor
            # next_draft: [K] next iteration's draft (with new anchor at [0])
            
            # === Clamp accept_count to remaining budget ===
            remaining = (max_steps - generated).astype(jnp.int32)
            eff_accept = jnp.minimum(accept_count, remaining)
            
            # === Check for EOS in committed tokens ===
            has_eos = jnp.array(False)
            if stop_on_eos and eos_id >= 0:
                eos_mask = (committed == eos_id) & (idx_k < eff_accept)
                first_eos = jnp.where(
                    jnp.any(eos_mask),
                    jnp.argmax(eos_mask.astype(jnp.int32)),
                    draft_len
                ).astype(jnp.int32)
                has_eos = first_eos < eff_accept
                eff_accept = jnp.where(has_eos, first_eos + 1, eff_accept)
            
            # === Write committed tokens to cache ===
            # Pad committed to fixed size K for static shape
            commit_padded = jnp.where(
                idx_k < eff_accept,
                committed,
                pad_token_id
            ).astype(jnp.int32)
            
            # Write at positions starting at prefix_len (after the anchor)
            commit_pos_ids = (prefix_len + idx_k).astype(jnp.int32)
            
            # Write K tokens (only first eff_accept are meaningful)
            write_logits, cache = cache_write_apply(
                params, cache, commit_padded, commit_pos_ids, prefix_len
            )
            
            # === Update output buffer ===
            out = jax.lax.dynamic_update_slice(out, commit_padded[:draft_len], (prefix_len,))
            
            # === Advance state ===
            prefix_len2 = prefix_len + eff_accept
            generated2 = generated + eff_accept
            done2 = done | (generated2 >= max_steps) | has_eos
            
            # Update draft logits for next iteration
            # Compute proposal_idx same way as in anchor_rejection_sample
            k_minus_1 = draft_len - 1
            n_verified = accept_count - 1  # Number of tokens committed (= n_accepted + 1)
            stopped = accept_count < draft_len  # True if rejection occurred
            proposal_idx = jnp.where(stopped, n_verified - 1, k_minus_1)
            proposal_idx = jnp.clip(proposal_idx, 0, k_minus_1)
            
            next_draft_logits = predraft_logits[proposal_idx]  # [K, V]
            
            # Stats
            total_accepts2 = total_accepts + eff_accept
            n_iters2 = n_iters + 1
            
            return (
                rng, cache, out, prefix_len2, next_draft, generated2,
                next_draft_logits, done2, total_accepts2, n_iters2
            )
        
        state0 = (
            rng_key, cache_vars, out_ids, prefix_len, current_draft, generated,
            current_draft_logits, done, total_accepts, n_iterations
        )
        
        state_final = jax.lax.while_loop(cond_fn, body_fn, state0)
        
        (_, _, out_final, prefix_len_final, _, generated_final,
         _, _, total_accepts_final, n_iters_final) = state_final
        
        # Build stats dict
        stats = {
            "total_accepts": total_accepts_final,
            "n_iterations": n_iters_final,
            "avg_accept_per_iter": total_accepts_final.astype(jnp.float32) / jnp.maximum(n_iters_final, 1),
        }
        
        return out_final, prefix_len_final, generated_final, stats
    
    return generate


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
    
    # Use context_length as cache_len for simplicity
    cache_len = context_length
    
    # Setup mask token
    base_token = getattr(cfg.tokenizer, "mask_token_override", None) or "[MASK]"
    mask_token, mask_id, added_tokens = ensure_tidar_mask_token(tokenizer, base_token=base_token)
    if added_tokens:
        print(f"Added mask token '{mask_token}' (id={mask_id})")
    else:
        print(f"Using mask token '{mask_token}' (id={mask_id})")
    
    # Build model and load params
    model = build_model(cfg, len(tokenizer), context_length)
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
    
    # Prefill prompt
    print("Prefilling prompt...")
    cache_vars, prefix_len, prev_logit = prefill_prompt(
        model, params, cache_vars, jnp.asarray(prompt_ids), kv_cache_len=cache_len
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
        verbose_stats=args.verbose,
    )
    
    # Run generation
    print(f"Generating {max_steps} tokens with draft_len={draft_len}, temp={temperature}, top_k={top_k}...")
    assert prefix_len > 0, "prefix_len must be > 0 before starting the decode loop"
    start_time = time.perf_counter()
    
    out_ids_final, final_len, generated, stats = generate_fn(
        params,
        cache_vars,
        out_ids,
        jnp.asarray(prefix_len, dtype=jnp.int32),
        jnp.asarray(max_steps, dtype=jnp.int32),
        prev_logit,
        rng,
    )
    out_ids_final.block_until_ready()
    
    decode_time = time.perf_counter() - start_time
    
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
        toks_per_s = gen_count / decode_time if decode_time > 0 else float("inf")
        
        print("\n[stats]")
        print(f"  prompt_tokens:     {prompt_len}")
        print(f"  generated_tokens:  {gen_count}")
        print(f"  n_iterations:      {n_iters}")
        print(f"  avg_accept/iter:   {avg_accept:.2f}")
        print(f"  decode_time_s:     {decode_time:.4f}")
        print(f"  tokens_per_second: {toks_per_s:.2f}")


if __name__ == "__main__":
    main()

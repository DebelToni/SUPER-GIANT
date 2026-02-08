"""
Anchor-TiDAR: Core utilities for speculative decoding with guaranteed progress.

Key insight: Each decode step starts with an anchor token at draft position 0.
The anchor is NEVER verified; verification starts from position 1.

Layout for K draft tokens:
  Verify block:   [ANCHOR | DRAFT_1 | DRAFT_2 | ... | DRAFT_{K-1}]  (K tokens)
  Predraft block: [K groups of K mask tokens]                       (K*K tokens)
  
Position IDs (relative to prefix_len L):
  Verify:   [L, L+1, L+2, ..., L+K-1]
  Predraft: For group r in [0..K-1]:
              positions [L+r+1, L+r+2, ..., L+r+K]
              
 Attention pattern:
   - Verify tokens: causal among themselves + see all prefix
   - Predraft group r: sees prefix + verify[0:r+1] + bidirectional within group
   - Predraft groups do not attend to each other

Rejection sampling:
  - Verify starts at position 1 (not 0): check if sampled_from_logit[i] == draft[i]
  - If all K-1 drafts accepted: use proposal from last predraft group, substitute
    anchor from logit[K-1]
  - On rejection at position i: use proposal from predraft group i-1, substitute
    the resampled token as new anchor
"""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import jax
import jax.numpy as jnp


# =============================================================================
# Position ID Templates
# =============================================================================

def build_decode_position_template(draft_len: int) -> jnp.ndarray:
    """
    Build position offsets template for Anchor-TiDAR decode step.
    
    Returns offsets array of shape [K + K*K] where:
      - First K entries: 0, 1, 2, ..., K-1 (verify block)
      - Next K*K entries: predraft groups with offsets r+1+t for r in [0,K), t in [0,K)
      
    To get actual positions, add prefix_len to this template.
    """
    # Verify block: positions 0 to K-1
    pos_verify = jnp.arange(draft_len, dtype=jnp.int32)
    
    # Predraft block: for group r, positions are r+1+t for t in [0, K)
    # Group 0: [1, 2, ..., K]
    # Group 1: [2, 3, ..., K+1]
    # ...
    # Group K-1: [K, K+1, ..., 2K-1]
    r = jnp.arange(draft_len, dtype=jnp.int32)[:, None]  # [K, 1]
    t = jnp.arange(draft_len, dtype=jnp.int32)[None, :]  # [1, K]
    offsets = (r + 1 + t).reshape(-1)  # [K*K]
    
    return jnp.concatenate([pos_verify, offsets], axis=0)


def build_prefill_draft_position_template(draft_len: int) -> jnp.ndarray:
    """
    Build position offsets for initial draft prefill (K mask tokens).
    Returns: [0, 1, 2, ..., K-1]
    """
    return jnp.arange(draft_len, dtype=jnp.int32)


# =============================================================================
# Attention Bias Templates
# =============================================================================

@lru_cache(maxsize=32)
def build_decode_bias_template(cache_len: int, draft_len: int, bias_value: float = -1e10) -> jnp.ndarray:
    """
    Build attention bias template for Anchor-TiDAR decode step.
    
    Query layout: [VERIFY(K) | PREDRAFT(K*K)]
    Key layout:   [PREFIX_CACHE(cache_len) | STEP_TOKENS(K + K*K)]
    
    Attention rules:
      - Verify queries: see all prefix + causal within verify (q_idx <= k_idx for step keys)
      - Predraft group r: sees prefix + verify[0:r+1] + causal within own group
      
    Returns bias of shape [1, 1, K+K*K, cache_len + K + K*K].
    Note: prefix validity masking is handled separately in attention layer.
    """
    q_len = draft_len + draft_len * draft_len
    key_len = cache_len + q_len
    
    q_idx = jnp.arange(q_len)[:, None]      # [q_len, 1]
    k_idx = jnp.arange(key_len)[None, :]    # [1, key_len]
    
    # Classify query positions
    is_verify_q = q_idx < draft_len
    is_predraft_q = q_idx >= draft_len
    
    # Classify key positions  
    is_prefix_k = k_idx < cache_len
    step_k_idx = k_idx - cache_len  # Index within step tokens (negative if prefix)
    is_step_k = k_idx >= cache_len
    is_verify_k = is_step_k & (step_k_idx < draft_len)
    is_predraft_k = is_step_k & (step_k_idx >= draft_len)
    
    # === Verify queries ===
    # Can see all prefix
    allow_verify_prefix = is_verify_q & is_prefix_k
    # Can see verify keys causally (k <= q within verify block)
    allow_verify_verify = is_verify_q & is_verify_k & (step_k_idx <= q_idx)
    
    # === Predraft queries ===
    # Which predraft group does the query belong to?
    predraft_q_offset = q_idx - draft_len  # Offset within predraft block
    predraft_q_group = predraft_q_offset // draft_len  # Group index r in [0, K-1]
    predraft_q_within = predraft_q_offset % draft_len  # Position within group
    
    # Which predraft group does the key belong to?
    predraft_k_offset = step_k_idx - draft_len  # Offset within predraft block
    predraft_k_group = predraft_k_offset // draft_len
    predraft_k_within = predraft_k_offset % draft_len
    
    # Can see all prefix
    allow_predraft_prefix = is_predraft_q & is_prefix_k
    
    # Can see verify[0:r+1] where r = predraft_q_group
    # i.e., step_k_idx < r+1, equivalently step_k_idx <= r
    allow_predraft_verify = is_predraft_q & is_verify_k & (step_k_idx <= predraft_q_group)
    
    # Can see own predraft group bidirectionally
    same_group = predraft_q_group == predraft_k_group
    allow_predraft_predraft = is_predraft_q & is_predraft_k & same_group
    
    # Combine all allowances
    allow = (
        allow_verify_prefix |
        allow_verify_verify |
        allow_predraft_prefix |
        allow_predraft_verify |
        allow_predraft_predraft
    )
    
    bias = jnp.where(allow, 0.0, bias_value)
    return bias[None, None, :, :].astype(jnp.float32)


@lru_cache(maxsize=32)
def build_prefill_draft_bias_template(cache_len: int, draft_len: int) -> jnp.ndarray:
    """
    Build attention bias for initial draft prefill.
    
    All K mask tokens can see the entire prefix and attend bidirectionally to each other.
    Shape: [1, 1, K, cache_len + K]
    """
    q_len = draft_len
    key_len = cache_len + draft_len
    
    # Prefix validity masking is handled in the attention layer; zero bias leaves
    # the K mask tokens fully bidirectional within the draft block.
    return jnp.zeros((1, 1, q_len, key_len), dtype=jnp.float32)


def build_prefill_prompt_draft_bias_template(
    cache_len: int,
    prompt_len: int,
    draft_len: int,
    bias_value: float = -1e10,
) -> jnp.ndarray:
    """
    Build attention bias for a single-pass prefill + initial draft.

    Layout: [prompt | mask*draft_len] as both queries and step keys, with
    prefix cache keys disabled (handled by prefix validity in attention).

    Rules:
    - Prompt queries: causal within prompt, no access to mask tokens.
    - Mask queries: full access to prompt + bidirectional within mask block.

    Shape: [1, 1, prompt_len + draft_len, cache_len + prompt_len + draft_len]
    """
    q_len = prompt_len + draft_len
    key_len = cache_len + q_len
    q_idx = jnp.arange(q_len)[:, None]
    k_idx = jnp.arange(key_len)[None, :]

    is_prefix_k = k_idx < cache_len
    step_k_idx = k_idx - cache_len
    is_step_k = k_idx >= cache_len

    is_prompt_q = q_idx < prompt_len
    is_mask_q = q_idx >= prompt_len
    is_prompt_k = is_step_k & (step_k_idx < prompt_len)
    is_mask_k = is_step_k & (step_k_idx >= prompt_len)

    allow_prompt_prompt = is_prompt_q & is_prompt_k & (step_k_idx <= q_idx)
    allow_mask_prompt = is_mask_q & is_prompt_k
    allow_mask_mask = is_mask_q & is_mask_k

    allow = allow_prompt_prompt | allow_mask_prompt | allow_mask_mask
    allow = allow & (~is_prefix_k)

    bias = jnp.where(allow, 0.0, bias_value)
    return bias[None, None, :, :].astype(jnp.float32)


# =============================================================================
# Sampling Utilities
# =============================================================================

def mask_top_k(logits: jnp.ndarray, top_k: int) -> jnp.ndarray:
    """
    Mask logits to keep only top-k values.
    logits: [..., V]
    Returns logits with everything outside top-k set to -inf.
    """
    if top_k <= 0:
        return logits
    
    # For batched logits, we need to handle arbitrary leading dimensions
    original_shape = logits.shape
    vocab_size = original_shape[-1]
    
    if top_k >= vocab_size:
        return logits
    
    # Flatten to 2D for top_k operation
    flat = logits.reshape(-1, vocab_size)
    n_batch = flat.shape[0]
    
    top_vals, top_idx = jax.lax.top_k(flat, top_k)
    
    # Create masked array
    masked = jnp.full_like(flat, -jnp.inf)
    rows = jnp.arange(n_batch, dtype=jnp.int32)[:, None]
    masked = masked.at[rows, top_idx].set(top_vals)
    
    return masked.reshape(original_shape)


def prepare_logits(logits: jnp.ndarray, temperature: float, top_k: int) -> jnp.ndarray:
    """Apply temperature scaling and top-k masking."""
    if temperature <= 0:
        # Will use argmax anyway, just return as-is
        return logits
    scaled = logits / jnp.maximum(temperature, 1e-8)
    return mask_top_k(scaled, top_k)


def sample_tokens(
    key: jax.Array,
    logits: jnp.ndarray,
    temperature: float,
    top_k: int,
) -> Tuple[jax.Array, jnp.ndarray]:
    """
    Sample tokens from logits with temperature and top-k.
    
    logits: [..., V]
    Returns: (new_key, tokens) where tokens has shape [...]
    """
    vocab_size = logits.shape[-1]
    flat = logits.reshape(-1, vocab_size)
    
    def do_sample(k):
        prepared = prepare_logits(flat, temperature, top_k)
        return jax.random.categorical(k, prepared, axis=-1).astype(jnp.int32)
    
    def do_argmax(_k):
        return jnp.argmax(flat, axis=-1).astype(jnp.int32)
    
    key, subkey = jax.random.split(key)
    toks_flat = jax.lax.cond(
        jnp.asarray(temperature) > 0.0,
        do_sample,
        do_argmax,
        subkey,
    )
    
    return key, toks_flat.reshape(logits.shape[:-1])


# =============================================================================
# Anchor-TiDAR Rejection Sampling
# =============================================================================

def anchor_rejection_sample(
    key: jax.Array,
    *,
    anchor_token: jnp.ndarray,           # [] scalar - the anchor token at draft position 0
    draft_tokens: jnp.ndarray,           # [K-1] - tokens at positions 1..K-1 to verify
    verify_logits: jnp.ndarray,          # [K, V] - logits from model (shifted: logits[i] predicts position i+1)
    draft_logits: jnp.ndarray,           # [K, V] - logits used to sample draft (for rejection ratio)
    predraft_tokens: jnp.ndarray,        # [K, K] - sampled proposals from predraft groups
    temperature: float,
    top_k: int,
) -> Tuple[jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Anchor-TiDAR rejection sampling.
    
    This function verifies positions 1..K-1 and returns:
    - accepted_count: accepted prefix length from current_draft (min 1, max K)
    - committed_tokens: legacy helper tensor used by immediate-commit variants
    - selected_proposal: next-draft block with substituted anchor at [0]
    
    Returns:
        key: Updated RNG key
        accepted_count: int32 scalar, accepted prefix length from current draft
                       Minimum 1, maximum K
        committed_tokens: [K] helper tensor for immediate-commit variants
        selected_proposal: [K] next draft proposal (for next iteration)
    """
    k = draft_tokens.shape[0] + 1  # Total verify block size including anchor
    k_minus_1 = draft_tokens.shape[0]
    
    # Prepare probabilities for rejection ratio
    # We verify draft_tokens which are at positions 1..K-1
    # verify_logits[i] predicts position i+1, so verify_logits[0:K-1] predict positions 1..K-1
    # draft_logits[i] predicted position i when sampling, so draft_logits[1:K] predicted positions 1..K-1
    verify_lgts_for_ratio = verify_logits[:-1]  # [K-1, V] - predicts positions 1..K-1
    draft_lgts_for_ratio = draft_logits[1:]     # [K-1, V] - predicted positions 1..K-1 when draft was sampled
    
    p_logits = prepare_logits(verify_lgts_for_ratio, temperature, top_k)
    q_logits = prepare_logits(draft_lgts_for_ratio, temperature, top_k)
    
    p_log = jax.nn.log_softmax(p_logits, axis=-1)
    q_log = jax.nn.log_softmax(q_logits, axis=-1)
    
    # Get log probs for the draft tokens
    idx = jnp.arange(k_minus_1, dtype=jnp.int32)
    p_log_tok = p_log[idx, draft_tokens]
    q_log_tok = q_log[idx, draft_tokens]
    
    # Acceptance probability: min(1, p/q)
    # For greedy decoding (temperature=0), we use exact match instead
    ratio = jnp.exp(p_log_tok - q_log_tok)
    accept_prob_sampling = jnp.minimum(1.0, ratio)
    
    # For greedy mode: accept only if draft_token == argmax(verify_logits)
    greedy_argmax = jnp.argmax(verify_lgts_for_ratio, axis=-1)  # [K-1]
    greedy_match = (draft_tokens == greedy_argmax).astype(jnp.float32)  # 1.0 if match, 0.0 otherwise
    
    # Choose acceptance probability based on temperature
    is_greedy = jnp.asarray(temperature) <= 0.0
    accept_prob = jnp.where(is_greedy, greedy_match, accept_prob_sampling)
    
    # Sample uniform for rejection decisions
    key, key_u, key_resample = jax.random.split(key, 3)
    u = jax.random.uniform(key_u, (k_minus_1,), dtype=jnp.float32)
    
    # For greedy mode, set u to 0.5 so that accept_prob >= 1.0 accepts, accept_prob = 0.0 rejects
    u = jnp.where(is_greedy, 0.5, u)
    
    # Pre-sample replacement tokens from p for each position
    # Use argmax for greedy (temperature=0), categorical otherwise
    def do_categorical(k):
        return jax.random.categorical(k, p_logits, axis=-1).astype(jnp.int32)
    
    def do_argmax(_k):
        return jnp.argmax(p_logits, axis=-1).astype(jnp.int32)
    
    resampled = jax.lax.cond(
        jnp.asarray(temperature) > 0.0,
        do_categorical,
        do_argmax,
        key_resample,
    )
    
    # Sequential rejection sampling
    def step_fn(carry, inputs):
        stopped, accept_count = carry
        draft_tok, resamp_tok, u_i, ap_i = inputs
        
        accept = u_i < ap_i
        do_accept = (~stopped) & accept
        do_reject = (~stopped) & (~accept)
        
        # Output token: draft if accepted, resampled if rejected (first rejection only)
        out_tok = jnp.where(stopped, draft_tok, jnp.where(accept, draft_tok, resamp_tok))
        
        stopped2 = stopped | do_reject
        accept_count2 = accept_count + do_accept.astype(jnp.int32)
        
        return (stopped2, accept_count2), out_tok
    
    init = (jnp.array(False), jnp.array(0, dtype=jnp.int32))
    (stopped_final, n_accepted), verified_tokens = jax.lax.scan(
        step_fn,
        init,
        (draft_tokens, resampled, u, accept_prob),
    )
    
    # Sample bonus token from verify_logits[K-1]
    key, key_bonus = jax.random.split(key)
    bonus_logits = verify_logits[-1:]  # [1, V]
    _, bonus_token = sample_tokens(key_bonus, bonus_logits, temperature, top_k)
    bonus_token = bonus_token[0]  # scalar
    
    # Build committed tokens array: [verified[0], ..., verified[K-2], bonus]
    # Length K (without the anchor!)
    committed = jnp.concatenate([
        verified_tokens,          # [K-1]
        bonus_token[None],        # [1]
    ])  # [K]
    
    # accepted_count (NEW tokens to commit, not including anchor):
    # - If rejected at position j (1-indexed in verify block):
    #   n_accepted = j-1 (drafts accepted before rejection)
    #   We commit: verified[0:j-1] (accepted) + verified[j-1] (resampled) = j tokens
    #   So count = n_accepted + 1
    # - If all accepted: count = K-1 + 1 (bonus) = K
    
    accepted_count = jnp.where(
        stopped_final,
        n_accepted + 1,  # accepted drafts + resampled
        k,               # all K-1 drafts + bonus
    )
    
    # Select next draft proposal
    # If all accepted: use predraft[K-1], bonus becomes next anchor
    # If rejected at position j: use predraft[j-1], resampled becomes next anchor
    #   where j = n_accepted + 1 (1-indexed position of rejection)
    #   so proposal_idx = n_accepted
    
    proposal_idx = jnp.where(stopped_final, n_accepted, k_minus_1)
    proposal_idx = jnp.clip(proposal_idx, 0, k_minus_1)
    selected_proposal = predraft_tokens[proposal_idx]  # [K]
    
    # Next anchor = resampled token at rejection point, or bonus if all accepted
    resamp_idx = jnp.clip(n_accepted, 0, k_minus_1 - 1)
    next_anchor = jnp.where(
        stopped_final,
        resampled[resamp_idx],  # The resampled token
        bonus_token,            # Bonus token if all accepted
    )
    
    # Put next_anchor as position 0 of selected_proposal
    selected_proposal = selected_proposal.at[0].set(next_anchor)
    
    return key, accepted_count, committed, selected_proposal


# =============================================================================
# KV Cache Utilities  
# =============================================================================

def init_kv_cache(model, *, batch_size: int, pad_token_id: int):
    """Initialize empty KV cache by running model with dummy input."""
    dummy = jnp.full((batch_size, 1), pad_token_id, dtype=jnp.int32)
    variables = model.init(
        {"params": jax.random.PRNGKey(0)},
        dummy,
        deterministic=True,
        use_kv_cache=True,
        cur_index=0,
        write_to_cache=True,
    )
    return variables["cache"]


def prefill_prompt(
    model,
    params,
    cache_vars,
    prompt_ids: jnp.ndarray,
    *,
    kv_cache_len: int,
):
    """
    Prefill prompt into KV cache.
    
    Returns: (updated_cache, prefix_len, last_logit)
    - last_logit: [V] logit predicting the next token after prompt
    """
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None, :]
    
    batch_size, prompt_len = prompt_ids.shape
    
    if prompt_len == 0:
        dummy_logit = jnp.zeros((model.vocab_size,), dtype=jnp.float32)
        return cache_vars, 0, dummy_logit
    
    position_ids = jnp.arange(prompt_len, dtype=jnp.int32)[None, :]
    position_ids = jnp.broadcast_to(position_ids, (batch_size, prompt_len))
    
    logits, mutated = model.apply(
        {"params": params, "cache": cache_vars},
        prompt_ids,
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=True,
        cur_index=0,
        position_ids=position_ids,
        kv_cache_len=kv_cache_len,
        mutable=["cache"],
    )
    
    last_logit = logits[0, -1]  # [V]
    return mutated["cache"], prompt_len, last_logit


def prefill_prompt_with_draft(
    model,
    params,
    cache_vars,
    prompt_ids: jnp.ndarray,
    *,
    draft_len: int,
    mask_id: int,
    kv_cache_len: int,
    bias_value: float = -1.0e10,
):
    """
    Prefill prompt and compute initial draft in a single forward pass.

    Returns: (updated_cache, prefix_len, last_logit, draft_logits)
    - last_logit: [V] logit predicting the next token after prompt
    - draft_logits: [K, V] logits for the initial draft tokens
    """
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None, :]

    batch_size, prompt_len = prompt_ids.shape
    if prompt_len == 0:
        dummy_logit = jnp.zeros((model.vocab_size,), dtype=jnp.float32)
        draft_logits = jnp.zeros((draft_len, model.vocab_size), dtype=jnp.float32)
        return cache_vars, 0, dummy_logit, draft_logits

    mask_tokens = jnp.full((batch_size, draft_len), mask_id, dtype=jnp.int32)
    step_tokens = jnp.concatenate([prompt_ids, mask_tokens], axis=1)
    step_len = prompt_len + draft_len

    position_ids = jnp.arange(step_len, dtype=jnp.int32)[None, :]
    position_ids = jnp.broadcast_to(position_ids, (batch_size, step_len))

    attn_bias = build_prefill_prompt_draft_bias_template(
        kv_cache_len,
        prompt_len,
        draft_len,
        bias_value=bias_value,
    )

    logits, mutated = model.apply(
        {"params": params, "cache": cache_vars},
        step_tokens,
        deterministic=True,
        use_kv_cache=True,
        write_to_cache=False,
        prefix_len=0,
        cache_write_len=prompt_len,
        attn_bias=attn_bias,
        position_ids=position_ids,
        kv_cache_len=kv_cache_len,
        mutable=["cache"],
    )

    last_logit = logits[0, prompt_len - 1]
    draft_logits = logits[0, prompt_len:]
    return mutated["cache"], prompt_len, last_logit, draft_logits

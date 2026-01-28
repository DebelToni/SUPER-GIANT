# Anchor-TiDAR Implementation Notes

This file documents the **current code paths** for Anchor-TiDAR inference and
validation. It intentionally focuses on how the code is wired (functions,
shapes, masking, cache semantics) rather than re‑explaining TiDAR theory.

## Key files

- `TiDAR/model/tidar_core.py`
  - Mask/position templates.
  - Sampling utilities (top‑k, temperature, rejection sampling).
  - KV cache helpers.
- `TiDAR/model/inference.py`
  - Main Anchor-TiDAR decode loop and CLI.
- `TiDAR/model/Training_step.py`
  - Shared training loss and acceptance metric logic.
- `TiDAR/model/distributional_invariance_test.py`
  - Non‑greedy distributional invariance test vs pure AR baseline.

## `tidar_core.py` highlights

### Position templates

- `build_decode_position_template(draft_len)`
  - Returns offsets for the per‑step tokens:
    - Verify block: `0..K-1`
    - Predraft block: group `r` has positions `[r+1 .. r+K]`
  - Actual positions are `prefix_len - 1 + offsets` in inference (anchor is at
    `prefix_len - 1`).

### Attention bias templates

- `build_decode_bias_template(cache_len, draft_len, bias_value)`
  - Produces a `[1,1,q_len,cache_len+q_len]` bias.
  - Query layout: `[VERIFY(K) | PREDRAFT(K*K)]`.
  - Key layout: `[PREFIX_CACHE | STEP_TOKENS]`.
  - Rules:
    - Verify queries see all prefix + causal verify tokens.
- Predraft queries see prefix + verify[0..r] + bidirectional within their group.

- `build_prefill_draft_bias_template(cache_len, draft_len)`
  - Returns **all‑zero** bias: draft masks are **bidirectional** within the
    draft block and can see the prefix. Prefix validity is handled in the
    attention layer via `prefix_len` masking.

### Sampling utilities

- `prepare_logits` applies temperature scaling + top‑k masking.
- `sample_tokens`
  - `temperature > 0`: `jax.random.categorical`
  - `temperature <= 0`: argmax

### Rejection sampling

`anchor_rejection_sample` verifies tokens at positions `1..K-1` and returns:
- `accepted_count` = number of **new** tokens to commit (min 1, max K)
- `committed` = K tokens to write (last token is the bonus)
- `selected_proposal` = next draft block (with new anchor at `[0]`)

Important behavior:
- **Greedy mode** (`temperature <= 0`): draft token is accepted **only if** it
  equals `argmax(verify_logits)` for that position.
- **Sampling mode** (`temperature > 0`): uses standard speculative decoding
  acceptance ratio `min(1, p/q)` with `p` and `q` derived from temperature/top‑k
  logits.

## `inference.py` (Anchor-TiDAR decode loop)

### High‑level flow

1. **Prefill + initial draft** in one forward (`prefill_prompt_with_draft`).
2. **Sample first anchor** from `prev_logit` and **commit** it immediately.
3. **Initial draft** comes from the same forward pass (K mask tokens with
   bidirectional mask block).
4. **Iterative loop**:
   - Build `step_tokens = [current_draft | predraft_masks]`.
   - Compute `step_pos_ids = prefix_len - 1 + position_template`.
   - Run forward pass with `decode_prefix_len = prefix_len - 1` to **avoid
     double‑conditioning the anchor** (anchor is already in `step_tokens[0]`).
    - Extract `verify_logits` (first K tokens) and `predraft_logits`.
   - Sample predraft tokens, run rejection sampling.
   - Commit accepted tokens to cache and output buffer.
   - Choose next draft from the selected predraft group.

### Prefix/caching semantics

- `prefix_len` always counts **committed tokens**.
- The anchor is already committed; therefore, during decode we **exclude it from
  the prefix cache** by passing `decode_prefix_len = prefix_len - 1`.
- Cache writes are fixed-shape (always K tokens, padded), while `prefix_len`
  advances by the actual accepted count.

### Prefill mask (single pass)
```
--- P0 P1 P2 P3 P4 P5 P6 P7 P8 | M0 M1 M2
------------------------------------------
P0   .  #  #  #  #  #  #  #  # |  #  #  #
P1   .  .  #  #  #  #  #  #  # |  #  #  #
P2   .  .  .  #  #  #  #  #  # |  #  #  #
P3   .  .  .  .  #  #  #  #  # |  #  #  #
P4   .  .  .  .  .  #  #  #  # |  #  #  #
P5   .  .  .  .  .  .  #  #  # |  #  #  #
P6   .  .  .  .  .  .  .  #  # |  #  #  #
P7   .  .  .  .  .  .  .  .  # |  #  #  #
P8   .  .  .  .  .  .  .  .  . |  #  #  #
M0   .  .  .  .  .  .  .  .  . |  .  .  .
M1   .  .  .  .  .  .  .  .  . |  .  .  .
M2   .  .  .  .  .  .  .  .  . |  .  .  .
```

### EOS handling

- When `stop_on_eos` is enabled, acceptance is truncated at the first EOS
  within the committed window.

## `distributional_invariance_test.py`

This script checks that **non‑greedy sampling** from Anchor‑TiDAR matches a
pure AR baseline for the same prompt/temperature/top‑k.

What it does:
- Samples `N` runs from AR and from each `draft_len`.
- Builds histograms for the first `num_tokens` positions.
- Reports L1/KL vs AR, plus per‑token deltas and a summary block.
- Optionally writes a JSON file to `cfg.paths.data_root`.

Example (large N):

```
/opt/venv/bin/python TiDAR/model/distributional_invariance_test.py \
  --checkpoint /proj/giant-data/TiDAR/smol/smollm-135m.npz \
  --prompt "Hello" \
  --temperature 0.7 \
  --top_k 50 \
  --draft_lens 2,8,20 \
  --num_samples 2000 \
  --num_tokens 1 \
  --output_json /proj/giant-data/TiDAR/distributional_invariance/hello_t0p7_k50_n2000.json
```

## Acceptance stats (greedy)

When `--verbose` is enabled in `TiDAR/model/inference.py`, the stats include:

- `avg_accept_per_iter`: average accepted tokens per decode iteration.

Interpretation for draft length K (anchor is always accepted):

```
expected_accept_tokens = 1 + (K - 1) * acc_prob
acc_prob ~= (avg_accept_per_iter - 1) / (K - 1)
```

This is a greedy decode measurement and is not the same as the training-time
acceptance estimate, which is top-k truncated and computed on a single batch row.

## Quick CLI sanity checks

- Greedy equivalence:
```
/Users/antonhristov/v/SG/bin/python TiDAR/model/inference.py \
  --checkpoint /proj/giant-data/TiDAR/smol/smollm-135m.npz \
  --prompt "Hello" \
  --steps 10 \
  --draft_len 3 \
  --temperature 0.0
```

- Non-greedy invariance (small N):
```
/Users/antonhristov/v/SG/bin/python TiDAR/model/distributional_invariance_test.py \
  --checkpoint /proj/giant-data/TiDAR/smol/smollm-135m.npz \
  --prompt "Hello" \
  --temperature 0.7 \
  --top_k 50 \
  --draft_lens 2,8,20 \
  --num_samples 30 \
  --num_tokens 1
```

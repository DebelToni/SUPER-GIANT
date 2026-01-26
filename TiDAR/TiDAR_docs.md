# TiDAR - Paper Baseline vs Repo Additions (Anchor + Agreement Loss)
This repo implements Anchor-TiDAR, a TiDAR variant that guarantees +1 token
progress per decode step by pre-sampling an anchor token from the previous
AR logit and never verifying that anchor. The rest of the draft is verified
speculatively, and the same forward pass also produces K candidate predrafts.

This doc separates:
- By-the-book TiDAR (paper baseline).
- Repo additions (Anchor + proposed agreement loss).

Sections 2+ mirror the current implementation under `TiDAR/model/`.

---

## 0) By-the-book TiDAR (paper baseline)
TiDAR keeps one decoder-only transformer trained under two attention regimes:

- Talk (AR): standard causal next-token prediction on the clean half.
- Think (Diffusion): masked tokens with blockwise bidirectional attention and
  inter-block causality.

Training uses a doubled sequence (clean + masked) with aligned position ids
and a block diffusion mask. The baseline joint loss is:

```
L_tidar(theta) = (alpha * L_AR_CE + L_Diff_CE) / (1 + alpha)
```

- alpha controls how much training focuses on AR vs Diff.

Inference (paper-style):
- Verify K drafted tokens in AR fashion.
- Predraft K candidate blocks for each possible accept count.
- If all drafts are rejected, there is no guaranteed progress and another
  forward pass is needed to get new proposals.

---

## 1) Repo additions (Anchor + agreement loss)

### 1.1 Anchor-TiDAR inference change (implemented)
- Sample an anchor token from the previous AR logit and commit it immediately.
- Verify only positions 1..K-1; the anchor is never verified.
- Guarantees at least +1 token progress per step.

### 1.2 Agreement loss to boost acceptance (implemented)
Goal: increase acceptance rate by nudging Diff to match AR distributions at the
same positions (even if this hurts quality).

```
L_tidar(theta) = (alpha * L_AR_CE + L_Diff_CE) / (1 + alpha)
L_total(theta) = L_tidar(theta) + lambda * KL(stopgrad(p_AR) || p_Diff)
```

- `alpha` controls how much training focuses on AR vs Diff (config: `training.loss.alpha`).
- `lambda` controls how hard Diff is pushed to agree with AR (config: `training.loss.agreement_lambda`).
- `stopgrad`/`detach` makes AR a fixed teacher so gradients update Diff only.
- When `agreement_lambda == 0`, the agreement term is skipped entirely (fast path).
 - Agreement is computed on aligned positions: AR positions 0..S-2 vs Diff positions S+1..2S-1
   (both predict token t+1).

Config example (in `Config.yml` or model config):
```yaml
training:
  loss:
    alpha: 1.0              # equal weight AR/Diff
    agreement_lambda: 0.0   # disabled by default; try 0.05-0.4 for speed focus
```

Stage-level overrides (optional):
```yaml
stages:
  - name: tinystories_300m_512
    dataset: tinystories_300m_512
    seq_len: 512
    epochs: 1
    end_ratio: 1.0
    loss:
      alpha: 0.5
      agreement_lambda: 0.2
```

When a stage defines its own `loss` subsection, those `alpha` and
`agreement_lambda` values replace the global defaults just for that stage.
If a field is omitted, it falls back to `training.loss.*`.


Suggested hyperparameters (acceptance/speed focused):
- Mild nudge: alpha=1.0, lambda=0.05-0.10
- Speed-biased: alpha=0.3-0.5, lambda=0.2-0.4
- Aggressive: alpha=0.2-0.3, lambda=0.5-1.0
- One concrete go-fast start: alpha=0.3, lambda=0.2

---

## 2) Current repo summary
Anchor-TiDAR keeps one decoder-only transformer and trains it with two
attention regimes in a single forward pass:

- Talk (AR): standard causal next-token prediction on the clean half.
- Think (Diffusion): masked tokens with blockwise bidirectional attention and
  inter-block causality.

At inference time, each decode iteration runs one forward pass that:

- verifies the current draft (AR style), skipping the already committed anchor,
- produces K predraft candidates for the next iteration.

Anchor-TiDAR differs from the paper baseline by guaranteeing a committed
anchor every step (sampled from the previous AR logit). This removes the
worst-case no-progress scenario without extra passes.

---

## 3) Key symbols
- S: training sequence length.
- K: draft length (`cfg.tidar.draft_length`).
- L: prefix length (committed tokens in cache).
- MASK_ID: tokenizer id for the TiDAR mask token.
- q_len: per-step tokens in decode = K + K*K.

---

## 4) Model changes (actual code)
File: `TiDAR/model/Transformer_block.py`

- `NativeJaxSelfAttention` accepts:
  - `attn_bias` (broadcastable to [1, 1, q_len, kv_len])
  - `position_ids` (per-token RoPE positions)
  - KV cache control (`use_kv_cache`, `write_to_cache`, `prefix_len`,
    `cache_write_len`, `kv_cache_len`).
- RoPE is gathered by `position_ids` via `_rope_from_position_ids`.
- RoPE cache length = `context_length + 2 * draft_len`.
- When `attn_bias` is provided, attention runs with `is_causal=False` and the
  bias handles structure.
- Cache path has two modes:
  - Standard append (`write_to_cache=True`) for AR/prefill.
  - TiDAR decode (`write_to_cache=False`) which concatenates cached prefix +
    step tokens and applies a structured bias plus prefix-validity bias.

The `TiDAR` model wrapper in `TiDAR/model/GiantTiDAR.py` passes `attn_bias`
and `position_ids` through every layer.

---

## 5) Training pipeline (implemented)
Files: `TiDAR/model/tidar_utils.py`, `TiDAR/model/tidar_masks.py`,
`TiDAR/model/Run_training.py`

### 5.1 Input construction
For a batch of tokens [B, S]:

- `clean = tokens`
- `diff = [MASK_ID] * S`
- `input_ids = [clean | diff]` -> shape [B, 2S]

### 5.2 Position ids
Aligned positions are used for diffusion:

- `position_ids = [0..S-1 | 0..S-1]` -> shape [B, 2S]

### 5.3 Labels + loss masks
- NTP on clean half (shifted): predict `x[t+1]` at `t`, ignore last token.
- Diffusion on masked half (aligned): predict `x[t]` at `S+t`.
- Loss uses two masks (`loss_mask_ntp`, `loss_mask_diff`) and sums the means:
  `loss = mean(ntp) + mean(diff)`.

### 5.4 Training attention bias (block diffusion)
Function: `build_tidar_train_bias` in `TiDAR/model/tidar_masks.py`

Token types: 0 = clean, 1 = diff.

Rules implemented:
- Clean queries -> clean keys only, causal: `pos_k <= pos_q`.
- Diff queries -> diff keys with block causality: `block_k <= block_q`.
- Diff queries -> clean keys only before current block: `pos_k < block_start`.
- Clean queries -> diff keys are disallowed.

`key_padding_mask` (from sequence lengths and token mask) further masks
invalid keys.

---

## 6) Anchor-TiDAR inference (implemented)
Files: `TiDAR/model/inference.py`, `TiDAR/model/tidar_core.py`

### 6.1 Prefill + first anchor
1) Prefill prompt into KV cache (`prefill_prompt`).
2) Sample first anchor from `prev_logit` (AR distribution) and commit it
   immediately.
3) Initial draft: run K mask tokens with bidirectional bias to sample the
   first draft block. The anchor is inserted at position 0 of the draft.

### 6.2 Decode step layout
K = draft_len, q_len = K + K*K.

- `step_tokens = [current_draft (K)] + [predraft_masks (K*K)]`.
- `position_ids = prefix_len - 1 + decode_position_template`.
  - Position template in `build_decode_position_template`:
    - Verify block: 0..K-1
    - Predraft group r: positions [r+1 .. r+K]

### 6.3 Decode bias template
Function: `build_decode_bias_template` in `TiDAR/model/tidar_core.py`

Query layout: [VERIFY(K) | PREDRAFT(K*K)].
Key layout: [PREFIX_CACHE(cache_len) | STEP_TOKENS].

Rules:
- Verify queries see all prefix + causal verify tokens.
- Predraft group r sees:
  - all prefix,
  - verify tokens up to index r (inclusive),
  - causal within its own group.
- Predraft groups do not attend to each other.

Prefix validity (prefix_len) is enforced inside attention via a separate
key-validity bias.

### 6.4 Rejection sampling (anchor aware)
Function: `anchor_rejection_sample` in `TiDAR/model/tidar_core.py`

- The anchor (draft[0]) is already committed and never verified.
- Draft positions 1..K-1 are verified using standard speculative acceptance:
  - greedy: accept iff `draft == argmax(verify_logits)`
  - sampling: accept with `min(1, p/q)` ratio
- On the first rejection, the resampled token becomes the new anchor.
- A bonus token is sampled from `verify_logits[K-1]`.
- `accepted_count` counts new tokens committed after the anchor, min 1 max K.
- Next draft is selected from the predraft group corresponding to the accept
  count, and its slot [0] is overwritten with the new anchor.

### 6.5 Cache writes
Committed tokens are written into KV cache with fixed-shape padding:

- `cache_write_len = K` (always write K tokens; unused slots padded)
- `prefix_len` advances by the actual accepted count.

---

## 7) Mask token handling
Files: `TiDAR/model/Prepare_mask_token.py`, `TiDAR/model/tokenizer_utils.py`

- `ensure_tidar_mask_token` guarantees a [MASK]-style token exists.
- `Prepare_mask_token.py` can expand embeddings and optionally save a tokenizer.
- Training initializes the mask embedding row if needed.

---

## 8) Entry points and helpers
- `TiDAR/model/Run_training.py`: full training loop (doubled sequence inputs).
- `TiDAR/model/inference.py`: Anchor-TiDAR CLI for generation.
- `TiDAR/model/distributional_invariance_test.py`: checks non-greedy sampling
  matches AR baselines (distributional invariance).
- `TiDAR/model/tidar_core.py`: position/bias templates, sampling utilities,
  rejection sampling, KV cache helpers.

---

## 9) Practical notes for future work
- Training uses block diffusion (non-overlapping blocks) as implemented in
  `build_tidar_train_bias`.
- Inference assumes fixed `draft_len` (templates cached by `lru_cache`).
- `NativeJaxSelfAttention` supports both standard AR cache updates and
  structured TiDAR decode bias with prefix masking.

If you need to change the decoding strategy, start in
`TiDAR/model/tidar_core.py` and `TiDAR/model/inference.py`.

---

## 10) NaN Loss Fix (Jan 25, 2026)

### 10.1 Issue
Training encountered NaN loss on UltraChat chat data (stage `ultrachat_rehearsal_1m`). 
Root cause: Chat sequences with ONLY user messages (no assistant responses) resulted 
in all-zero loss masks, which caused NaN propagation through attention.

**Mechanism**:
- UltraChat data masks only assistant responses (`chat_assistant_roles: ["assistant"]`)
- Sequences with no assistant tokens → `loss_mask = [0.0, 0.0, ...]` (all zeros)
- All-zero mask → all keys masked in attention → softmax(-inf, -inf, ...) → NaN
- NaN in attention → NaN gradients → NaN parameters

**Evidence**: Row 455 in `ultrachat_rehearsal_1m-000000.arrow` had zero mask sum (0.12% of data).

### 10.2 Fix: 3-Layer Defense

**Layer 1: Data Pipeline Prevention** ✅
- File: `GIANT/v2/data_pipeline/build_corpus.py:570-584`
- Added validation in `_emit_sequence()` to reject sequences with all-zero mask
- Discarded sequences tracked in `self.stats.discarded`
- **Impact**: Future dataset generation rejects corrupted rows at source

**Layer 2: Batch-Time Validation** ✅
- File: `TiDAR/model/tidar_utils.py:48-51`
- Added safety check in `build_train_batch()` after computing `valid` mask
- If a batch row has zero valid tokens, converts mask to all-ones as fallback
- **Impact**: Existing corrupted data won't cause NaN during training (graceful degradation)

**Layer 3: Runtime NaN Detection** ✅
- Files: `TiDAR/model/Run_training.py:640-652`, `GIANT/v2/model/Run_training.py:507-516`
- Added NaN/Inf detection after gradient computation
- Skips gradient accumulation if NaN detected (training continues with warning)
- **Impact**: Last resort safety net prevents NaN propagation to parameters

### 10.3 Data Regeneration
- Regenerated all 4 UltraChat shards under `TiDAR/Sweep/Data/*/ultrachat_rehearsal_1m/`
- **Before**: 858 rows, 1 with zero mask (row 455)
- **After**: 859 rows, 0 with zero mask, min mask sum = 44.0
- Layer 1 successfully discarded the corrupted sequence during regeneration
- Backup of original corrupted shard: `sweep_1_lr1e4_a1_l0/ultrachat_rehearsal_1m/ultrachat_rehearsal_1m-000000.arrow.corrupted_backup`

### 10.4 Testing
- Test script: `TiDAR/test_nan_theory.py`
- Verified Layer 2 converts zero-masks to all-ones fallback
- Confirmed new shards have no zero-mask rows
- All attention computations remain NaN-free

**Production Ready**: All 3 layers implemented, tested, and data regenerated. Training can resume safely.

### 10.5 Follow-up Stability Hardening (Jan 26, 2026)
- Added masking of CE and agreement KL on inactive positions to avoid NaNs from masked logits:
  - File: `TiDAR/model/Training_step.py`
- Added gradient-finite check during accumulation to avoid NaN grads corrupting accumulators:
  - File: `TiDAR/model/Run_training.py`
- Added non-finite loss logging/skip to prevent NaN metadata in checkpoints:
  - File: `TiDAR/model/Run_training.py`
- Added shard inspection utility:
  - File: `TiDAR/inspect_ultrachat_shard.py`

Checkpoint sanity:
- `step_0012100.npz` is finite (safe resume point)
- `step_0012200.npz` and later contained NaNs (do not resume)

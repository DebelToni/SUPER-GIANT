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
- Think (Diffusion): masked tokens with blockwise bidirectional attention
  (blocks do not attend to each other).

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

### 1.2 Five-term loss with configurable agreement (implemented)
Goal: flexible control over AR/Diff training balance and acceptance rate optimization.
Each term can be independently enabled/disabled by setting its coefficient to 0.

```
Loss = alpha * L_AR + beta * L_Diff + rho * KL_fwd + chi * KL_rev + delta * L_hard
```

Where:
- `L_AR`: AR next-token prediction CE loss (clean half, shifted: positions 0..S-2 predict tokens 1..S-1)
- `L_Diff`: Diffusion denoising CE loss (diff half, aligned: positions S..2S-1 predict tokens 0..S-1)
- `KL_fwd`: Forward KL `KL(stopgrad(P_AR) || Q_Diff)` - punishes Diff for missing AR probability mass
- `KL_rev`: Reverse KL `KL(Q_Diff || stopgrad(P_AR))` - punishes Diff for extra probability mass
- `L_hard`: Hard agreement `CE(onehot(argmax stopgrad(P_AR)), logits_diff)` - greedy agreement loss

Key properties:
- All agreement terms use `stopgrad` on AR logits, so gradients only update Diff parameters
- Terms with coefficient == 0 are skipped entirely (no compute, different JIT traces)
- The function returns 7 values: `(total_loss, ar_loss, diff_loss, kl_fwd, kl_rev, hard_agree, accept_rate)`

**Coefficient meanings:**
- `alpha`: Weight on AR language modeling. Higher = stronger AR capability.
- `beta`: Weight on Diff denoising. Higher = better diffusion quality.
- `rho`: Forward KL weight. Punishes Diff when AR assigns probability but Diff doesn't (mode-covering).
- `chi`: Reverse KL weight. Punishes Diff when Diff assigns probability but AR doesn't (mode-seeking).
- `delta`: Hard agreement weight. Forces Diff argmax to match AR argmax (greedy alignment).

Config example (in `Config.yml` or model config):
```yaml
training:
  loss:
    # Loss = alpha*L_AR + beta*L_Diff + rho*KL_fwd + chi*KL_rev + delta*L_hard
    # Each term with coefficient 0 is skipped entirely (no compute)
    alpha: 1.0    # AR next-token prediction CE loss
    beta: 1.0     # Diffusion denoising CE loss
    rho: 0.0      # Forward KL: KL(P_AR || Q_Diff) - punishes Diff for missing AR mass
    chi: 0.0      # Reverse KL: KL(Q_Diff || P_AR) - punishes Diff for extra mass
    delta: 0.0    # Hard agreement: CE(onehot(argmax P_AR), logits_diff) - greedy agreement
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
      beta: 1.0
      rho: 0.1
      chi: 0.0
      delta: 0.0
```

When a stage defines its own `loss` subsection, those values replace the global
defaults for that stage. If a field is omitted, it falls back to `training.loss.*`.

Alignment note:
- Agreement losses (KL_fwd, KL_rev, L_hard) compare AR positions 0..S-2 to Diff positions S+1..2S-1
  so both sides predict the same token (t+1).

**Suggested hyperparameters:**
- Baseline (no agreement): alpha=1.0, beta=1.0, rho=0, chi=0, delta=0
- Mild agreement nudge: alpha=1.0, beta=1.0, rho=0.05, chi=0, delta=0
- Forward KL focused: alpha=1.0, beta=1.0, rho=0.1-0.2, chi=0, delta=0
- Hard greedy agreement: alpha=1.0, beta=1.0, rho=0, chi=0, delta=0.1-0.3
- Speed-biased: alpha=0.5, beta=1.0, rho=0.2, chi=0, delta=0.1

---

## 2) Current repo summary
Anchor-TiDAR keeps one decoder-only transformer and trains it with two
attention regimes in a single forward pass:

- Talk (AR): standard causal next-token prediction on the clean half.
- Think (Diffusion): masked tokens with blockwise bidirectional attention
  (blocks do not attend to each other).

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

### 5.5 Training acceptance metric
- The training loop logs a theoretical acceptance probability computed from AR/Diff logits.
- It is a probability in [0, 1], computed on the last batch row and top-k truncated.
- For draft length K, expected accepted tokens per decode iter (greedy) is:
  `1 + (K - 1) * acc_prob`.

### 5.4 Training attention bias (block diffusion)
Function: `build_tidar_train_bias` in `TiDAR/model/tidar_masks.py`

Token types: 0 = clean, 1 = diff.

Rules implemented:
- Clean queries -> clean keys only, causal: `pos_k <= pos_q`.
- Diff queries -> diff keys within the same block only: `block_k == block_q`.
- Diff queries -> clean keys only before current block: `pos_k < block_start`.
- Clean queries -> diff keys are disallowed.

`key_padding_mask` (from sequence lengths and token mask) further masks
invalid keys.

Training mask example (S=9, K=3):
```
---- C0 C1 C2 C3 C4 C5 C6 C7 C8 | D0 D1 D2 D3 D4 D5 D6 D7 D8
------------------------------------------------------------
 C0   .  #  #  #  #  #  #  #  # |  #  #  #  #  #  #  #  #  #
 C1   .  .  #  #  #  #  #  #  # |  #  #  #  #  #  #  #  #  #
 C2   .  .  .  #  #  #  #  #  # |  #  #  #  #  #  #  #  #  #
 C3   .  .  .  .  #  #  #  #  # |  #  #  #  #  #  #  #  #  #
 C4   .  .  .  .  .  #  #  #  # |  #  #  #  #  #  #  #  #  #
 C5   .  .  .  .  .  .  #  #  # |  #  #  #  #  #  #  #  #  #
 C6   .  .  .  .  .  .  .  #  # |  #  #  #  #  #  #  #  #  #
 C7   .  .  .  .  .  .  .  .  # |  #  #  #  #  #  #  #  #  #
 C8   .  .  .  .  .  .  .  .  . |  #  #  #  #  #  #  #  #  #
 D0   #  #  #  #  #  #  #  #  # |  .  .  .  #  #  #  #  #  #
 D1   #  #  #  #  #  #  #  #  # |  .  .  .  #  #  #  #  #  #
 D2   #  #  #  #  #  #  #  #  # |  .  .  .  #  #  #  #  #  #
 D3   .  .  .  #  #  #  #  #  # |  #  #  #  .  .  .  #  #  #
 D4   .  .  .  #  #  #  #  #  # |  #  #  #  .  .  .  #  #  #
 D5   .  .  .  #  #  #  #  #  # |  #  #  #  .  .  .  #  #  #
 D6   .  .  .  .  .  .  #  #  # |  #  #  #  #  #  #  .  .  .
 D7   .  .  .  .  .  .  #  #  # |  #  #  #  #  #  #  .  .  .
 D8   .  .  .  .  .  .  #  #  # |  #  #  #  #  #  #  .  .  .
```

---

## 6) Anchor-TiDAR inference (implemented)
Files: `TiDAR/model/inference.py`, `TiDAR/model/tidar_core.py`

### 6.1 Prefill + first anchor
1) Prefill prompt and initial draft in one forward (`prefill_prompt_with_draft`).
2) Sample first anchor from `prev_logit` (AR distribution) and commit it
   immediately.
3) Initial draft comes from the same forward pass (K mask tokens with
   bidirectional mask block). The anchor is inserted at position 0 of the draft.

Prefill mask example (prompt_len=9, K=3):
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

Anchor sampling note:
- Sample the first anchor from the last prompt logit (P8 in the example) to get
  a true AR token before verification begins.

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
  - bidirectional within its own group.
- Predraft groups do not attend to each other.

Prefix validity (prefix_len) is enforced inside attention via a separate
key-validity bias.

Decode mask example (cache_len=9, K=3):
```
---- P0 P1 P2 P3 P4 P5 P6 P7 P8 | V0 V1 V2 G00 G01 G02 G10 G11 G12 G20 G21 G22
------------------------------------------------------------------------------
 V0   .  .  .  .  .  .  .  .  . |  .  #  #  #  #  #  #  #  #  #  #  #
 V1   .  .  .  .  .  .  .  .  . |  .  .  #  #  #  #  #  #  #  #  #  #
 V2   .  .  .  .  .  .  .  .  . |  .  .  .  #  #  #  #  #  #  #  #  #
G00   .  .  .  .  .  .  .  .  . |  .  #  #  .  .  .  #  #  #  #  #  #
G01   .  .  .  .  .  .  .  .  . |  .  #  #  .  .  .  #  #  #  #  #  #
G02   .  .  .  .  .  .  .  .  . |  .  #  #  .  .  .  #  #  #  #  #  #
G10   .  .  .  .  .  .  .  .  . |  .  .  #  #  #  #  .  .  .  #  #  #
G11   .  .  .  .  .  .  .  .  . |  .  .  #  #  #  #  .  .  .  #  #  #
G12   .  .  .  .  .  .  .  .  . |  .  .  #  #  #  #  .  .  .  #  #  #
G20   .  .  .  .  .  .  .  .  . |  .  .  .  #  #  #  #  #  #  .  .  .
G21   .  .  .  .  .  .  .  .  . |  .  .  .  #  #  #  #  #  #  .  .  .
G22   .  .  .  .  .  .  .  .  . |  .  .  .  #  #  #  #  #  #  .  .  .
```

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
- `TiDAR/model/config_schema.py`: structured OmegaConf dataclasses +
  `load_typed_config` for typed config access and path resolution.

---

## 9) Practical notes for future work
- Training uses block diffusion (non-overlapping blocks) as implemented in
  `build_tidar_train_bias`.
- Training loss/metrics are centralized in `TiDAR/model/Training_step.py` and
  used by `TiDAR/model/Run_training.py`.
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

---

## 11) Sweep data relocation (Jan 28, 2026)
- Moved sweep 1-4 datasets under `TiDAR/Sweep/Data/sweeps_1-4/`.
- Flattened `TiDAR/Sweep/Data/sweep_5_300m_ctxmix/` by removing the nested duplicate folder level.
- Updated sweeps 1-4 configs to use `training.dataset_dir: "Sweep/Data/sweeps_1-4"`.

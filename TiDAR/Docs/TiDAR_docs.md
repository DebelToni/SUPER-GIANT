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
- Sample an anchor token from the previous AR logit and place it at `draft[0]`.
- Verify only positions 1..K-1; the anchor is never verified.
- Optimistically write K draft KVs during the decode forward and commit via
  prefix-pointer advance/rollback (no separate full commit pass).
- Guarantees at least +1 token progress per step.

#### How Anchor TiDAR does a forward pass:
  
```
Example of how my TiDAR variant would work in decode

Prefill Input -> Output after sample
ABC MMM  -> BCD* DEF
Current KV cache: A B C

Decode step 1 input:
D*EF MMM MMM MMM

Decode step 1 output after sampling:
E*F'G' EFG FGH GHI
Current KV cache: A B C D* E F

Now we check if E* is E from the draft on the input (assume success). Then we check F' to F of the input (assume success). That means we accept the last proposal GHI.

Decode step 2 input:
(note here we will take G' that we sampled form F on last step and replace it in the GHI block)
G'HI MMM MMM MMM

Decode step 2 output after sampling:
H*I'J' HIJ IJK JKL
Current KV cache: A B C D* E F G' H I

Now we check that I' matches the output from I in the input but for example I'!=I at the input. So we select proposal 2 which is IJK

! Here we did not accept the full draft so we need ot move the pointer that says up to where we have KV cache. Right now it says we have 9 KV caches written but because we did not accept I!=I' we need to bring back the pointer 1 step back. So its current value will be 8 and the cache would contain A B C D* E F G' H

Decode step 3 input:
(Here we take I' from the sampled from last step instead of the I that is in the IJK block).
I*JK
```

### 1.2 Seven-term loss with configurable agreement + distillation (implemented)
Goal: flexible control over AR/Diff training balance and acceptance rate optimization.
Each term can be independently enabled/disabled by setting its coefficient to 0.

```
Loss = alpha * L_AR + beta * L_Diff + rho * KL_fwd + chi * KL_rev + delta * L_hard + eta * L_distill + gamma * L_topk
```

Where:
- `L_AR`: AR next-token prediction CE loss (clean half, shifted: positions 0..S-2 predict tokens 1..S-1)
- `L_Diff`: Diffusion denoising CE loss (diff half, aligned: positions S..2S-1 predict tokens 0..S-1)
- `KL_fwd`: Forward KL `KL(stopgrad(P_AR) || Q_Diff)` - punishes Diff for missing AR probability mass
- `KL_rev`: Reverse KL `KL(Q_Diff || stopgrad(P_AR))` - punishes Diff for extra probability mass
- `L_hard`: Hard agreement `CE(onehot(argmax stopgrad(P_AR)), logits_diff)` - greedy agreement loss
- `L_distill`: Soft distillation `KL(softmax(AR/T) || softmax(Diff/T))` on drafted positions (AR stopgrad)
- `L_topk`: Top-K set distillation `-log(sum(q_diff[ar_topk]))` on drafted positions (AR stopgrad)

Key properties:
- All agreement terms use `stopgrad` on AR logits, so gradients only update Diff parameters
- Terms with coefficient == 0 are skipped entirely (no compute, different JIT traces)
- The function returns `total_loss` plus 9 metrics: `(ar_loss, diff_loss, kl_fwd, kl_rev, hard_agree, distill, topk_loss, accept_rate, greedy_accept_rate)`

**Coefficient meanings:**
- `alpha`: Weight on AR language modeling. Higher = stronger AR capability.
- `beta`: Weight on Diff denoising. Higher = better diffusion quality.
- `rho`: Forward KL weight. Punishes Diff when AR assigns probability but Diff doesn't (mode-covering).
- `chi`: Reverse KL weight. Punishes Diff when Diff assigns probability but AR doesn't (mode-seeking).
- `delta`: Hard agreement weight. Forces Diff argmax to match AR argmax (greedy alignment).
- `eta`: Distillation weight. Matches softened AR distribution at drafted positions.
- `eta_T`: Distillation temperature. Higher = softer targets, scaled by T^2.
- `gamma`: Top-K set distillation weight. Pushes Diff mass onto AR's top-K set.
- `gamma_topk`: K value for the Top-K set distillation term.

Config example (in `Config.yml` or model config):
```yaml
training:
  loss:
    # Loss = alpha*L_AR + beta*L_Diff + rho*KL_fwd + chi*KL_rev + delta*L_hard + eta*L_distill + gamma*L_topk
    # Each term with coefficient 0 is skipped entirely (no compute)
    alpha: 1.0    # AR next-token prediction CE loss
    beta: 1.0     # Diffusion denoising CE loss
    rho: 0.0      # Forward KL: KL(P_AR || Q_Diff) - punishes Diff for missing AR mass
    chi: 0.0      # Reverse KL: KL(Q_Diff || P_AR) - punishes Diff for extra mass
    delta: 0.0    # Hard agreement: CE(onehot(argmax P_AR), logits_diff) - greedy agreement
    eta: 0.0      # Soft distillation KL (AR->Diff)
    eta_T: 1.0    # Distillation temperature
    gamma: 0.0    # Top-K set distillation
    gamma_topk: 8 # Top-K size for set distillation
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
      eta: 0.0
      eta_T: 1.0
      gamma: 0.0
      gamma_topk: 8
```

When a stage defines its own `loss` subsection, those values replace the global
defaults for that stage. If a field is omitted, it falls back to `training.loss.*`.

Alignment note:
- Agreement, distillation, and top-k set losses compare AR positions 0..S-2 to Diff positions S+1..2S-1
  so both sides predict the same token (t+1).

**Suggested hyperparameters:**
- Baseline (no agreement): alpha=1.0, beta=1.0, rho=0, chi=0, delta=0
- Mild agreement nudge: alpha=1.0, beta=1.0, rho=0.05, chi=0, delta=0
- Forward KL focused: alpha=1.0, beta=1.0, rho=0.1-0.2, chi=0, delta=0
- Hard greedy agreement: alpha=1.0, beta=1.0, rho=0, chi=0, delta=0.1-0.3
- Speed-biased: alpha=0.5, beta=1.0, rho=0.2, chi=0, delta=0.1
- Distill-only nudge: alpha=1.0, beta=1.0, eta=0.02-0.05, eta_T=2.0
- Top-K set nudge: gamma=0.01, gamma_topk=8

---

## 2) Current repo summary
Anchor-TiDAR keeps one decoder-only transformer and trains it with two
attention regimes in a single forward pass:

- Talk (AR): standard causal next-token prediction on the clean half.
- Think (Diffusion): masked tokens with blockwise bidirectional attention
  (blocks do not attend to each other).

At inference time, each decode iteration runs one forward pass that:

- verifies the current draft (AR style), skipping anchor verification,
- optimistically writes K draft KVs,
- produces K predraft candidates for the next iteration.

Anchor-TiDAR differs from the paper baseline by guaranteeing at least +1 token
progress every step (under normal budget), while avoiding a separate K-token
commit forward by using pointer-based cache commit/rollback.

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
2) Sample first anchor from `prev_logit` (AR distribution) and place it at
   `current_draft[0]` for the first decode step.
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
- `position_ids = prefix_len + decode_position_template`.
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
A - anchor (basically D0 but sampled from AR)
Di - draft i (positions 1..K-1)
Mij - mask in predraft i, index j

```
---- P0 P1 P2 P3 P4 P5 P6 P7 P8 | A0 D1 D2 M00 M01 M02 M10 M11 M12 M20 M21 M22
------------------------------------------------------------------------------
 V0   .  .  .  .  .  .  .  .  . |  .  #  #  #  #  #  #  #  #  #  #  #
 D1   .  .  .  .  .  .  .  .  . |  .  .  #  #  #  #  #  #  #  #  #  #
 D2   .  .  .  .  .  .  .  .  . |  .  .  .  #  #  #  #  #  #  #  #  #
M00   .  .  .  .  .  .  .  .  . |  .  #  #  .  .  .  #  #  #  #  #  #
M01   .  .  .  .  .  .  .  .  . |  .  #  #  .  .  .  #  #  #  #  #  #
M02   .  .  .  .  .  .  .  .  . |  .  #  #  .  .  .  #  #  #  #  #  #
M10   .  .  .  .  .  .  .  .  . |  .  .  #  #  #  #  .  .  .  #  #  #
M11   .  .  .  .  .  .  .  .  . |  .  .  #  #  #  #  .  .  .  #  #  #
M12   .  .  .  .  .  .  .  .  . |  .  .  #  #  #  #  .  .  .  #  #  #
M20   .  .  .  .  .  .  .  .  . |  .  .  .  #  #  #  #  #  #  .  .  .
M21   .  .  .  .  .  .  .  .  . |  .  .  .  #  #  #  #  #  #  .  .  .
M22   .  .  .  .  .  .  .  .  . |  .  .  .  #  #  #  #  #  #  .  .  .
```

Additional Verbose explenation of how decode pass works:

At inference draft token Mij from block i sees the prefix tokens + also the anchor token + any currently verified token up to token i (including it) from the draft that is being validated from last step

So if we verify this:
A - anchor
D - drafts but starting from 1, becasue 0 is the anchor
Mij - Mask in theoretical draft i, at index j

We input this at K=5:
A D1 D2 D3 D4 M00 M01 M02 M03 M04 M10 M11 M12 M13 M14 M20 M21 M22 M23 M24 M30 M31 M32 M33 M34 M40 M41 M42 M43 M44
(we also have in KV cache any prefix)

More visual way to present it would be:
```
A   D1  D2  D3  D4  
    M00 M01 M02 M03 M04 
        M10 M11 M12 M13 M14 
            M20 M21 M22 M23 M24 
                M30 M31 M32 M33 M34 
                    M40 M41 M42 M43 M44
```
^ Here Mij sees the prefix + from the current pass the tokens from the first row to the left of the Mi0 token ^

Lets take M21 for example.
M21 will attend causally to:
* prefix
* A D1 D2 
bidiretionally:
* M20 M21 M22 M23 M24
(all tokens in the same block)


### 6.4 Rejection sampling (anchor aware)
Function: `anchor_rejection_sample` in `TiDAR/model/tidar_core.py`

- The anchor (draft[0]) is not pre-committed and is never verified.
- Draft positions 1..K-1 are verified using standard speculative acceptance:
  - greedy: accept iff `draft == argmax(verify_logits)`
  - sampling: accept with `min(1, p/q)` ratio
- On the first rejection, the resampled token becomes the new anchor.
- A bonus token is sampled from `verify_logits[K-1]`.
- `accepted_count` counts committed prefix length from current_draft, min 1 max K.
- Next draft is selected from the predraft group corresponding to the accept
  count, and its slot [0] is overwritten with the new anchor.

### 6.5 Cache writes
Decode uses optimistic cache writes plus pointer commit:

- In the decode forward (`write_to_cache=False`), `cache_write_len = K` writes
  current_draft KVs at `[prefix_len .. prefix_len+K-1]`.
- `prefix_len` advances only by the accepted prefix length.
- Rejections are handled by pointer rollback (unaccepted optimistic KVs remain
  physically present but are masked out by prefix validity).

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
- **Config handling (Jan 30, 2026)**: `GiantTiDAR` and `Transformer_block` no longer read
  `TiDAR/model/Config.yml` at import time. All model-specific settings
  (`num_kv_heads`, `rope_dim`, `param_dtype`, `compute_dtype`, `use_remat`,
  `draft_length`) are passed from the runtime config (`--config`) into the model
  constructors. This enables switching between SmolLM-135M and SmolLM2-360M
  without changing `Config.yml`.
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

# Anchor-TiDAR Implementation Notes

This file documents the **current code paths** for Anchor-TiDAR inference and
validation. It intentionally focuses on how the code is wired (functions,
shapes, masking, cache semantics) rather than re‑explaining TiDAR theory.

## Key files

TiDAR training/data plumbing uses active `GIANT/v3` utilities:
- `GIANT/v3/data_pipeline/build_corpus.py`
- `GIANT/v3/model/arrow_data_loader.py`
- `GIANT/v3/model/checkpoint_manager.py`
- `GIANT/v3/model/optimizer_utils.py`

TiDAR passes raw token masks from the loader and performs AR/diffusion alignment in `build_train_batch()`.

- `TiDAR/model/tidar_core.py`
  - Mask/position templates.
  - Sampling utilities (top‑k, temperature, rejection sampling).
  - KV cache helpers.
- `TiDAR/model/inference.py`
  - Main Anchor-TiDAR decode loop and CLI.
- `TiDAR/model/Training_step.py`
  - Shared training loss and acceptance metric logic.
  - Includes Top-K set distillation loss (`gamma`, `gamma_topk`) to push Diff mass onto AR top-K.
- `TiDAR/model/distributional_invariance_test.py`
  - Non‑greedy distributional invariance test vs pure AR baseline.

## `tidar_core.py` highlights

### Position templates

- `build_decode_position_template(draft_len)`
  - Returns offsets for the per‑step tokens:
    - Verify block: `0..K-1`
    - Predraft block: group `r` has positions `[r+1 .. r+K]`
  - Actual positions are `prefix_len + offsets` in inference.

### Attention bias templates

- `build_decode_bias_template(cache_len, draft_len, bias_value)`
  - Produces a `[1,1,q_len,cache_len+q_len]` bias.
  - Query layout: `[VERIFY(K) | PREDRAFT(K*K)]`.
  - Key layout: `[PREFIX_CACHE | STEP_TOKENS]`.
  - Rules:
    - Verify queries see all prefix + causal verify tokens.
    - Predraft queries see prefix + verify[0..r] + bidirectional within their group.
    - Predraft groups do not attend to each other.

- `build_prefill_prompt_draft_bias_template(cache_len, prompt_len, draft_len)`
  - Single-pass prefill + initial draft bias (prompt causal, masks bidirectional).

### Sampling utilities

- `prepare_logits` applies temperature scaling + top‑k masking.
- `sample_tokens`
  - `temperature > 0`: `jax.random.categorical`
  - `temperature <= 0`: argmax

### Rejection sampling

`anchor_rejection_sample` verifies tokens at positions `1..K-1` and returns:
- `accepted_count` = committed prefix length from `current_draft` (min 1, max K)
- `committed` = helper tensor for legacy commit path (current inference does not
  use it directly)
- `selected_proposal` = next draft block (with new anchor at `[0]`)

Important behavior:
- **Greedy mode** (`temperature <= 0`): draft token is accepted **only if** it
  equals `argmax(verify_logits)` for that position.
- **Sampling mode** (`temperature > 0`): uses standard speculative decoding
  acceptance ratio `min(1, p/q)` with `p` and `q` derived from temperature/top‑k
  logits. The first rejection samples from normalized `max(p-q, 0)`, not from
  `p`; `test_speculative_sampling.py` verifies recovery of the target distribution.

## `inference.py` (Anchor-TiDAR decode loop)

### High‑level flow

1. **Prefill + initial draft** in one forward (`prefill_prompt_with_draft`).
2. **Sample first anchor** from `prev_logit` and place it at `current_draft[0]`.
3. **Initial draft** comes from the same forward pass (K mask tokens with
   bidirectional mask block).
4. **Iterative loop**:
   - Build `step_tokens = [current_draft | predraft_masks]`.
   - Compute `step_pos_ids = prefix_len + position_template`.
   - Run one decode forward that also optimistically writes current_draft KVs
     via `cache_write_len = K`.
    - Extract `verify_logits` (first K tokens) and `predraft_logits`.
    - Predraft groups are bidirectional within-group and do not attend to each other.
   - Sample predraft tokens, run rejection sampling.
   - Commit by pointer only: advance `prefix_len` by accepted prefix length.
   - Choose next draft from the selected predraft group.

### Prefix/caching semantics

- `prefix_len` always counts **committed tokens**.
- Current decode pass writes K optimistic KVs at the current pointer.
- Unaccepted suffix KVs are ignored by keeping `prefix_len` at the accepted
  boundary (pointer rollback semantics).

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

## Frozen-base, token-routed LoRA

The reusable implementation lives in `GIANT/v3/model/lora.py`; TiDAR wires it
into its own transformer because TiDAR does not import GIANT's causal-only
`GiantGPT` implementation.

Runtime routes are explicit boolean arrays:

```text
training [clean | diffusion]:          [off * S | on * S]
prefill [prompt | initial masks]:      [off * P | on * K]
decode [verify | predraft]:            [off * K | on * K²]
```

`GiantTiDAR.__call__` requires `adapter_mask` whenever `lora.routing=token`.
The same route reaches every configured projection in the layers selected by
`lora.layer_indices` (`null` selects all layers). Optional
`stop_gradient_before_lora` detaches the prefix before the first selected layer;
TiDAR training also fixes the separate mask embedding in that mode. Verifier queries cannot attend predraft positions, and only the first K
adapter-off KVs are optimistically written by a decode step.

Frozen weights stay under `params`; LoRA and the mask input embedding stay under
`adapters`. The mask ID is one position beyond the original tokenizer/model
vocabulary, but the tied embedding/LM-head matrix retains its original row
count. Mask inputs are replaced with the separate adapter-owned vector before
the transformer, so verifier softmax normalization still covers exactly the
base vocabulary.

LoRA TiDAR checkpoints are adapter-only NPZs under `adapters/`. Inference loads
the original base with `--checkpoint` and the adapter with `--adapter`:

```bash
/opt/venv/bin/python TiDAR/model/inference.py \
  --config /path/to/token_routed_lora.yml \
  --checkpoint /path/to/base.npz \
  --adapter /path/to/adapter.npz \
  --prompt "Once upon"
```

See `GIANT/v3/model/LORA.md` for shared configuration, checkpoint, and test
details. The RTX 5090 last-quarter benchmark and its quality caveat are recorded
in `Docs/LoRA_Last_Quarter_Speed_Experiment.md`.

## Validated frozen-base TiDAR LoRA pod run

`model/training_configs/Ablations/tidar_smollm135_frozen_lora_o_proj_smoke.yml`
was validated on one RTX A6000 using real FineWeb-Edu rows at clean context 256
(the model input is the doubled 512-token clean/diffusion layout). The run used
rank-16, alpha-32 `o_proj` LoRA, an adapter-owned mask input, `alpha=0`, and
`beta=1`.

```text
frozen base parameters:      134,515,008
trainable adapter values:        553,536 (0.412%)
timed training:                    600 s
steps at timed stop:                 5,450
steps after resume check:             5,475
clean tokens after resume:         1,401,600
observed peak GPU memory:              2,378 MiB
first logged loss:                      8.9651
last timed loss:                         4.7394
first/last 100-log mean loss:     6.0819 / 5.7068
```

The batch size was deliberately one for a correctness run, so the 18.7% mean
sampled GPU utilization is not a throughput target. Frozen LoRA removes base
gradients and optimizer state; it does not imply a proportional reduction in
backbone forward/backward FLOPs.

At `K=8`, the production 30-layer BF16 mixed pass had bit-exact verifier logits
and committed KVs relative to an all-off route (both maximum absolute
differences were zero), while the 64 predraft rows changed. Greedy cached
inference generated 48 tokens in 32 iterations with deployed acceptance 1.50
tokens/iteration (maximum 3) and 480.5 tokens/s after compilation. These are
deployed measurements; training `accept` and `greedy_acc` remain overlap
proxies.

The 100,000-sample analytic residual-sampling test passed. In an adapter-backed
256-sample-per-arm distribution check, position 0 was pathwise identical and
position 1 had L1 0.1797. A 100,000-draw pooled-null simulation expected mean
L1 0.1958, with p=0.626 for a difference at least as large, so this run found no
statistical evidence of a sampling-distribution mismatch.

Artifacts, exact config, source diff, logs, GPU samples, adapter checkpoints,
and optimizer state are stored at:

```text
s3://giant-data/TiDAR/ablations/tidar_smollm135_frozen_lora_o_proj_smoke/
```

## `distributional_invariance_test.py`

This script checks that **non‑greedy sampling** from Anchor‑TiDAR matches a
pure AR baseline for the same prompt/temperature/top‑k.

What it does:
- Samples `N` runs from AR and from each `draft_len`.
- Builds histograms for the first `num_tokens` positions.
- Reports L1/KL vs AR, plus per‑token deltas and a summary block.
- Optionally writes a JSON file to `cfg.paths.data_root`.
- Accepts `--config`, `--global_config`, and `--adapter`; in LoRA mode the AR
  baseline runs the loaded adapter collection with an all-off route.

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

## Batch Size Finder (`TiDAR/tests/find_batch_size.py`)

Utility script to find the maximum batch size for a given TiDAR training config
that fits in GPU memory without OOM.

### How it works

1. Parses the provided training config YAML
2. Identifies the stage with the **longest `seq_len`** (worst-case memory usage)
3. Creates a temporary test config with only that stage (modified for quick testing)
4. Uses **binary search** O(log n) to find the maximum working batch size
5. Spawns training runs as **separate processes** to ensure clean GPU memory state
6. Monitors the `logs.txt` file in the checkpoint directory to detect successful training
7. Detects OOM by looking for JAX/XLA error patterns in stderr

### Important: Must run ON the GPU box

The script spawns local subprocesses for training. It must be executed **on the
remote GPU machine**, not locally. Running locally will fail because there is no
GPU available.

### Usage

First sync your local changes to the GPU box:
```bash
git add -N . && git diff --binary | ssh root@gpu-box-3 'set -e; cd /proj/SUPER-GIANT; git reset --hard; git clean -fd; git apply --index'
```

Then run the script on the remote:
```bash
ssh root@gpu-box-3 'cd /proj/SUPER-GIANT && /opt/venv/bin/python TiDAR/tests/find_batch_size.py \
  --config TiDAR/model/model_configs/your_config.yml \
  --global_config TiDAR/Global_Config.yml \
  --min_batch_size 1 \
  --max_batch_size 32 \
  --wait_for_logs 5 \
  --timeout 300'
```

### CLI arguments

- `--config` (required): Path to the training config YAML
- `--global_config`: Path to Global_Config.yml (optional, uses TiDAR/Global_Config.yml by default)
- `--min_batch_size`: Lower bound for binary search (default: 1)
- `--max_batch_size`: Upper bound for binary search (default: 64)
- `--wait_for_logs`: Number of log entries to wait for before declaring success (default: 5)
- `--timeout`: Timeout per test run in seconds (default: 600)
- `--init_checkpoint`: Optional checkpoint to initialize from
- `--python_path`: Path to Python executable (default: system Python)
- `--verbose`: Print detailed output including stderr on failures

### Output

The script prints results like:
```
================================================================================
                          BATCH SIZE FINDER RESULTS
================================================================================
Config: TiDAR/model/model_configs/batch_size_test_30L_512ctx.yml
Max context length tested: 512

Maximum working batch size: 6

Test history:
  batch_size=1   SUCCESS  Success: 5 log entries after 45.2s
  batch_size=32  FAILED   OOM detected in output
  batch_size=16  FAILED   OOM detected in output
  batch_size=8   FAILED   OOM detected in output
  batch_size=4   SUCCESS  Success: 5 log entries after 38.1s
  batch_size=6   SUCCESS  Success: 5 log entries after 40.3s
  batch_size=7   FAILED   OOM detected in output
================================================================================
```

### Notes

- The script adds CLI arguments `--batch_size`, `--gradient_accumulation`, and
  `--log_every` to `Run_training.py` that override config values
- Always uses `--gradient_accumulation 1` during batch size testing
- Creates a temporary checkpoint directory that is cleaned up after testing
- Result labels: `SUCCESS` = training ran, `FAILED` = OOM or other error, `[CFG]` = config error (e.g., dataset too small)

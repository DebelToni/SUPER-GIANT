# TiDAR SmolLM-135M: frozen LoRA versus full-training speed on RTX 5090

## Status

**Experiment complete: measurement validated, artifacts published, shared startup restored, and all pods removed.**

Last updated: 2026-07-21 02:30 CEST.

Checklist:

- [x] Identify the historical SmolLM-135M full-model TiDAR run.
- [x] Recover its committed config, dataset, S3 logs, hardware, actual batch geometry, and sustained timing.
- [x] Verify that the original 500M-token dataset still exists locally.
- [x] Define a matched current-code full-versus-LoRA experiment.
- [x] Add the two benchmark configs and validate them locally.
- [x] Upload the historical dataset to S3; 11 objects and 4,000,993,356 bytes verified by listing.
- [x] Install task-specific `sync_dirs.txt` and a no-op startup entrypoint, after backing up the shared objects.
- [x] Rent one exact `NVIDIA GeForce RTX 5090` Secure Cloud pod.
- [x] Run preflights, sustained benchmark arms, and parse results.
- [x] Upload the compact result bundle, restore shared startup objects, and remove the pod.
- [x] Add measured results and conclusions to this document.

## Question

Measure how much faster the new frozen-base, token-routed TiDAR LoRA training path is than full-model TiDAR training for SmolLM-135M. Quality is deliberately outside this experiment. The result must distinguish:

1. **Current matched speedup:** current full training versus current LoRA on the same RTX 5090, code, data, objective, effective batch, and measurement method. This is the primary result.
2. **Historical speedup:** current LoRA versus the January 2026 RTX 5090 full-training log. This preserves continuity with the old experiment but is secondary because the code, JAX stack, container, and provider host may have changed.
3. **Optional practical LoRA throughput:** LoRA with a larger microbatch while holding the effective batch fixed. This measures whether reduced trainable state can produce additional realized throughput.

A historical-only comparison is not clean enough to isolate LoRA. A same-pod current full arm is therefore mandatory even though an old full-training log exists.

## Measured result

Run identity:

```text
pod: 7pvbrydfkdad6v
GPU: NVIDIA GeForce RTX 5090, 32,607 MiB
JAX/JAXlib: 0.10.1/0.10.1
artifact run: 20260720T2336Z_7pvbrydf
S3: s3://giant-data/TiDAR/benchmarks/lora_vs_full_smollm135_5090_ctx2048/20260720T2336Z_7pvbrydf/
source diff SHA256: 89c857f41c99156df3c71af87e34daef8fa9fbfcff26c8c3caa9df7c6c36ab6a
```

Post-compile results after discarding each arm's first warm-up interval:

| Measurement | Retained time | Seconds/microbatch | Clean tokens/s | Projected 500,000,768-token time |
|---|---:|---:|---:|---:|
| Historical full, secondary | 19,390.5 s | 0.482363 | 16,983.1 | 8.178 h |
| Current full, two-run aggregate | 874.6 s | 0.448513 | 18,264.8 | 7.604 h |
| Current LoRA batch 4, two-run aggregate | 510.2 s | 0.261641 | 31,310.1 | 4.436 h |

Primary matched result:

```text
LoRA/current-full throughput ratio: 1.71423x
Throughput increase:                 71.42%
Projected 500M time reduction:       41.66%
Projected 500M time saved:           3.168 hours
```

Reproducibility gates:

- Full runs: 18,348.7 and 18,181.7 clean tokens/s; relative delta 0.92%.
- LoRA runs: 31,273.3 and 31,346.9 clean tokens/s; relative delta 0.24%.
- Full interval CVs: 2.28% and 2.15%.
- LoRA interval CVs: 0.76% and 0.80%.
- All four retained windows passed the aggregate duration requirement; each full run independently exceeded five minutes, and the two equivalent LoRA windows totalled 8.50 minutes.
- Parsing was performed twice with numerically identical JSON after excluding the creation timestamp.

State and resource result:

| Measurement | Current full | Frozen LoRA |
|---|---:|---:|
| Trainable values | 134,515,584 | 553,536 |
| Final checkpoint | about 499.15 MB | about 2.068 MB |
| Optimizer state | 1,076,141,751 bytes | 4,435,129 bytes |
| Retained peak GPU allocation | 27,201 MiB | 27,702 MiB |
| Mean sampled power | 422.2 W | 548.1 W |
| Maximum temperature | 65 C | 71 C |

The optimizer state is 242.6 times smaller and the checkpoint is about 241.4 times smaller. Sampled power was higher while LoRA was active, but estimated energy per clean token fell about 24.3% because throughput rose more than power.

Long-context activation/buffer allocation dominated memory. LoRA used 501 MiB more retained peak allocation in this compiled batch-4 path, and the optional batch-8/accumulation-2 arm failed during `jit__run_chunk` with a 26.11 GiB allocation request. This experiment therefore establishes a speed and trainable-state benefit, not an activation-memory or larger-batch benefit. Batch sizes 5–7 remain untested.

The secondary cross-version comparison is +84.36% throughput versus the January log, while current full itself is 7.55% faster than that log. Only the +71.42% same-pod result isolates the current full-versus-LoRA training path. No quality conclusion is supported.

## Historical baseline recovered

### Provenance

The closest old experiment is:

- Config introduced by commit `5745df2da93a8f20a9570d31391e362fcb8f73c5` on 2026-01-31.
- Results committed in `82c67eb5b2a78c8e912a1fd3f632d6364c551976` on 2026-02-01.
- Current config path: `TiDAR/model/training_configs/Greedy_exp_135m.yml`.
- Data config: `TiDAR/data_pipeline/data_configs/Greedy_exp_500m.yml`.
- Committed copy of the log: `TiDAR/Docs/Training_logs/greedy_135m_not_full.txt`.
- S3 log: `s3://giant-data/TiDAR/Params/Greedy_exp_135m/params/logs.txt`.
- Log SHA256: `aed9da1a550117644e4ee24606b94140e3f4b7fb6b8af88968cdb7d7d576b139` for both copies.
- Historical final checkpoint: `s3://giant-data/TiDAR/Params/Greedy_exp_135m/params/step_0040000.npz`.
- Historical dataloader state: `s3://giant-data/TiDAR/Params/Greedy_exp_135m/training_states/dataloader_state/state_0040000.json`.
- `TiDAR/Docs/Results_Greedy_runs_135_360.typ` explicitly identifies the 135M run as an RTX 5090 with 32 GB VRAM and a paid cost of USD 5.23.

The May 2026 `ablations/timing/probe_*` logs are not this baseline. Their allocator attempted approximately 94.98 GiB and they came from a 96 GB-class GPU, so they must not be presented as RTX 5090 measurements.

### Historical model and objective

- Base: Hugging Face SmolLM-135M converted to the TiDAR/GIANT parameter tree.
- Base checkpoint now available at `/proj/giant-data/TiDAR/smol/smollm-135m.npz`.
- Frozen-base file size: 538,117,944 bytes.
- Base SHA256: `52662c0a53688cecf14c95c965d2ef0d49994a95a322efc36d85d3d5da042748`.
- Base parameters: 134,515,008 before the TiDAR mask input is added.
- Architecture: width 576, 30 layers, 9 query heads, 3 KV heads, FFN width 1536, RoPE width 64.
- Clean context: 2048 tokens.
- TiDAR model input: 4096 positions, consisting of clean and diffusion halves.
- Draft length: 8.
- Parameter dtype: FP32.
- Compute dtype: BF16.
- Rematerialization: disabled.
- Loss coefficients: `alpha=1`, `beta=1`, `delta=3`; all other historical terms disabled.
- Optimizer: AdamW, learning rate `5e-5` to `5e-6`, 1,000 warmup steps, weight decay 0.01, clip norm 0.5.

### Historical dataset

Local source of truth:

```text
/proj/giant-data/TiDAR/dataset_artifacts/Greedy_exp_500m/
```

Facts:

- FineWeb-Edu packed with the Cosmo2 tokenizer.
- 244,141 records.
- 2,048 tokens per record.
- 500,000,768 clean tokens.
- 8 Arrow shards.
- 4,000,991,112 Arrow bytes, approximately 3.726 GiB.
- 11 files including manifests/statistics.

The dataset is currently absent from S3 and must be uploaded to:

```text
s3://giant-data/TiDAR/dataset_artifacts/Greedy_exp_500m/
```

Shard SHA256 values:

```text
3f215f1bdb7910e88984c8ffa2d09ff32af997a4e2550e8e36fdf5becfc897b3  fw_edu_500m-000000.arrow
9d122159ecb033afcb0191a991f3be0837ff5acce5e246a2653ba1e7b2e57430  fw_edu_500m-000001.arrow
a4590c4a48b7314b63a84ce11ecd467cd229d5e45bd3a26d4aeac911c7fe9fba  fw_edu_500m-000002.arrow
94f42e24a034829327073a752d2742fc215e88f33a47572b6f506cd2d25446a5  fw_edu_500m-000003.arrow
e41dbbf5d187f7b741047c030527dc7e0b3de6379fd7ba94c0da7c3824f71a6e  fw_edu_500m-000004.arrow
e14cbeb4ab4c25eb1314b19e04099d4ec05ee3183483f60e5b4c667993a654dd  fw_edu_500m-000005.arrow
b54373540afd9f5f458c2b4df1d6622e7d3854fcda6cda0291566e961e50fef2  fw_edu_500m-000006.arrow
20a2a9a5b1a623c15f3ec8317e9dd5f002b781649cc64230f0a89a3ee5c7343f  fw_edu_500m-000007.arrow
```

### Actual historical batch and step semantics

The committed config says `batch_size: 8`, but that was not the runtime batch:

- 244,141 rows divided into the logged 61,035 steps implies batch 4.
- The saved state at global step 40,000 reports `rows_consumed: 160004` and loader `step_in_epoch: 40001`, which proves four rows per logged step.
- The second log segment uses `log_every=300`, another CLI override from the committed `100`.
- Historical `Run_training.py` increments `global_step` for every dataloader microbatch and updates parameters only after gradient accumulation.
- The documentation calls these optimizer steps, but the logged step is a **microbatch step**.
- The committed accumulation setting is 4. No surviving command line contradicts it, so the benchmark will use batch 4 with accumulation 4: effective batch 16.

At step 40,200 the second run had therefore processed:

```text
40,200 microbatches * 4 rows * 2,048 clean tokens = 329,318,400 clean tokens
```

With accumulation 4, it had emitted approximately 10,050 optimizer updates.

### Historical sustained timing

`logs.txt` contains an abandoned first segment through step 1,600 and then a fresh second segment through step 40,200. Only the second segment is used.

- Compile plus first logged microbatch: 35.4 seconds.
- Post-first interval coverage: 40,199 microbatches.
- Sum of reported post-first intervals: 19,390.5 seconds, or 5.386 hours.
- Weighted mean: 0.482363 seconds per microbatch.
- Median interval rate: 0.480333 seconds per microbatch.
- Clean throughput: 16,983.1 tokens/second.
- Doubled TiDAR input throughput: 33,966.1 positions/second.
- Optimizer updates: 0.5183 updates/second at accumulation 4.
- Logged clean-token coverage: 65.86% of the 500M-token dataset.
- Projected full-dataset time at the sustained rate: 8.178 hours.
- The documented USD 5.23 over the recorded run corresponds to about USD 0.97/hour and is consistent with the approximately 5.4-hour log.

Checkpoint writes are included in some historical 300-step intervals. Report both the weighted mean and robust median rather than pretending the old number is a perfect kernel-only measurement.

## Current matched experiment

### Shared controls

Both current-code arms will use:

- One exact `NVIDIA GeForce RTX 5090` Secure Cloud pod from template `GIANT-container` (`bg2jwnb3zk`).
- No Spot instance and no fallback GPU SKU.
- The same pod, container, driver, JAX/JAXlib, Flax, Optax, clocks, and data files.
- The current local working-tree diff applied to the remote clone; no commit is required.
- Base checkpoint SHA256 `52662c...2748`.
- Pinned local tokenizer `/proj/giant-data/TiDAR/tokenizers/cosmo2_tidar_lora_smoke`.
- Tokenizer base vocabulary 49,152, input-only mask ID 49,152, tokenizer size 49,153.
- Tokenizer JSON SHA256 `bf346d64f6f0fbcefb4c1b6928a98241467dff36c6fbae5fe1785c4ff90667f4`.
- The historical FineWeb-Edu records, shuffle seed 7777, clean context 2048, `K=8`.
- `param_dtype=float32`, `compute_dtype=bfloat16`, dropout 0, remat disabled.
- `alpha=1`, `beta=1`, `delta=3`, all other loss coefficients zero.
- AdamW and clipping settings matching the historical config. The short 1,024-step benchmark sets `warmup_steps=1024`, selecting the existing constant-schedule branch; this avoids the scheduler's invalid near-warmup boundary while keeping optimizer work identical across arms.
- Effective batch 16 and exactly 8,388,608 clean tokens per sustained arm.
- `scan_chunk=1`, no mini checkpoints, no periodic full checkpoints, and identical logging cadence.
- Dynamic JAX allocation only if needed; the same allocator environment must be used by every arm.

Each arm uses exactly 4,096 dataset records. With dataset size 244,141, set stage fraction to `0.016778`, which resolves to 4,096 rows. This naturally bounds the run without relying on `--max_steps`; that CLI option is currently parsed but unused.

### Arm F: current full training

Planned config:

```text
TiDAR/model/training_configs/Benchmarks/tidar_smollm135_finewebedu_ctx2048_full_speed.yml
```

- `training.finetune_method: full`.
- LoRA disabled.
- Initialize from the same 134,515,008-parameter base checkpoint.
- The tied embedding/LM-head receives one trainable mask row, matching ordinary full TiDAR behavior.
- Microbatch 4, accumulation 4, effective batch 16.
- 1,024 microbatch steps and 256 optimizer updates.
- 8,192 clean tokens per microbatch and 32,768 clean tokens per optimizer update.

### Arm L4: matched frozen LoRA

Planned config:

```text
TiDAR/model/training_configs/Benchmarks/tidar_smollm135_finewebedu_ctx2048_frozen_lora_o_proj_speed.yml
```

- `training.finetune_method: lora`.
- Frozen 134,515,008-parameter base.
- Rank 16, alpha 32, dropout 0.
- `o_proj` only, token routing.
- Separate input-only mask embedding.
- 553,536 trainable values: 552,960 LoRA values plus 576 mask-embedding values.
- Microbatch 4, accumulation 4, effective batch 16.
- 1,024 microbatch steps and 256 optimizer updates.

This is the primary matched comparison. The full and LoRA arms compute the same losses. AR rows remain adapter-off, so AR loss does not train the adapters, but it is still computed in both arms to keep timing comparable to the old full objective.

### Arm L8: optional practical LoRA throughput

Run only after F and L4 are valid:

- Same LoRA config and same 4,096 records.
- Microbatch 8, accumulation 2, effective batch 16.
- 512 microbatch steps and 256 optimizer updates.
- This keeps model inputs, clean tokens, optimizer-update count, and effective batch fixed while testing whether LoRA can exploit its smaller trainable state.

If batch 8 OOMs, record the OOM and omit L8. Do not alter F or L4 after seeing results.

### Run order

Use a bracketed baseline to detect thermal, clock, or host drift:

1. `F-A`: current full, batch 4.
2. `L4`: matched LoRA, batch 4.
3. `F-B`: repeat current full, batch 4.
4. `L8`: optional LoRA batch 8.

The primary full baseline is the aggregate of F-A and F-B. If their clean throughput differs by more than 5%, inspect clocks, utilization, temperature, host contention, and interval variance. Repeat or extend before claiming a clean speedup.

## Measurement method

### Compile exclusion

The training log reports elapsed time at step 1 that includes initialization/JIT compilation. Preserve it as a separate metric but exclude it from sustained throughput.

For each arm:

1. Record process start time.
2. Record compile-plus-first-step time from the step-1 log line.
3. Treat every interval after step 1 as post-compile.
4. Drop the first post-compile interval from the headline sustained calculation as a conservative warm-up interval.
5. Require at least five minutes of retained post-compile intervals. The 4,096-record design should produce roughly 5–10 minutes per arm around the historical throughput; if any valid arm finishes sooner, extend that arm with the same config and aggregate only equivalent post-compile intervals until the requirement is met.
6. Compute weighted throughput from total retained steps divided by total retained seconds, plus median, p10, p90, mean, standard deviation, and coefficient of variation across interval rates.
7. Exclude final checkpoint serialization from training throughput. It occurs after the last training log line.

### Metrics

For each arm report:

```text
seconds per microbatch
microbatches per second
clean tokens per second
TiDAR model positions per second
optimizer updates per second
seconds per optimizer update
projected hours per 500,000,768 clean tokens
projected hours per 1B clean tokens
compile plus first step seconds
peak GPU memory MiB
mean and p95 GPU utilization
mean power draw
mean SM clock
maximum temperature
parameter count
trainable parameter count
optimizer-state bytes on disk
```

Conversions:

```text
clean_tokens_per_microbatch = microbatch_size * 2048
TiDAR_positions_per_microbatch = microbatch_size * 4096
clean_tokens_per_optimizer_update = 16 * 2048 = 32,768
matched_batch_speedup = LoRA_clean_tokens_per_second / full_clean_tokens_per_second
                      = full_seconds_per_microbatch / LoRA_seconds_per_microbatch
practical_speedup = optimized_LoRA_clean_tokens_per_second / full_clean_tokens_per_second
projected_hours = target_clean_tokens / clean_tokens_per_second / 3600
```

Headline deltas:

- `L4 / current-full`: isolates the frozen-LoRA training method at matched microbatch.
- `L8 / current-full`: practical throughput gain at matched effective batch, clearly labelled optional.
- `L4 / historical-full`: continuity estimate across six months, clearly labelled non-isolated.

Do not infer FLOP reduction from adapter parameter percentage, and do not report a quality or acceptance conclusion from this timing experiment.

### GPU telemetry

Run one-second `nvidia-smi` sampling per arm with at least:

```text
timestamp
memory.used
utilization.gpu
power.draw
clocks.sm
temperature.gpu
```

Capture before and after markers so final checkpoint I/O and inter-arm idle time can be excluded. Also save:

```text
nvidia-smi -L
nvidia-smi full query
CUDA/driver version
jax, jaxlib, flax, optax versions
jax.devices()
Python version
pod ID and RunPod GPU name
current git commit
SHA256 of the complete local source diff
```

## Validation gates

Before accepting a comparison:

1. The pod reports exactly one RTX 5090 with 32 GB-class memory.
2. Base, tokenizer, manifests, and all eight Arrow shards match the recorded SHA256 values.
3. Current TiDAR tests pass locally before syncing.
4. Both F and L4 complete a preflight update without OOM, NaN, or non-finite gradients.
5. F and L4 use the same 4,096 rows, objective, seed, microbatch, accumulation, and effective batch.
6. The full arm contains trainable base parameters; the LoRA arm differentiates only 553,536 adapter values and contains no base leaves in its adapter checkpoint.
7. At least five retained post-compile minutes exist for F-A, L4, and F-B.
8. No periodic checkpoint falls inside the measured interval.
9. F-A and F-B agree within 5%, or the discrepancy is investigated and disclosed.
10. GPU telemetry shows no thermal throttling, OOM recovery, competing process, or sustained idle periods.
11. The result parser is run twice or covered by a small synthetic parser test.
12. Raw logs, configs, telemetry, formulas, and parsed JSON are uploaded before pod removal.

If full batch 4 OOMs under the current stack, retry both matched arms at microbatch 2 and accumulation 8. Report that as a current-code matched comparison, while retaining the historical batch-4 result only as a secondary reference. Never compare full batch 2 against LoRA batch 4 as the isolated headline.

## S3 and pod preparation

### Upload missing data

Upload and verify:

```bash
s5cmd --endpoint-url "$S3_ENDPOINT_URL" sync --size-only \
  "/proj/giant-data/TiDAR/dataset_artifacts/Greedy_exp_500m/*" \
  "s3://giant-data/TiDAR/dataset_artifacts/Greedy_exp_500m/"
```

Verify 11 objects, sizes, manifests, and shard hashes after the upload.

### Startup sync list

Back up the shared startup objects before replacing them:

```text
s3://giant-data/sync_dirs.txt
s3://giant-data/entrypoint.sh
```

Use these lines in the task-specific `sync_dirs.txt`:

```text
TiDAR/dataset_artifacts/Greedy_exp_500m/
TiDAR/smol/smollm-135m.npz
TiDAR/tokenizers/cosmo2_tidar_lora_smoke/
```

Use a no-op entrypoint that only records readiness. Do not start training automatically because the uncommitted LoRA source diff must be applied over SSH first.

### Pod command

```bash
.opencode/skill/deploy-gpu/scripts/runpod-gpu.sh create \
  --gpu "NVIDIA GeForce RTX 5090" \
  --name tidar-lora-vs-full-speed \
  --wait
```

Use Secure Cloud and template `bg2jwnb3zk`; do not silently fall back to another GPU.

After the pod is reachable:

1. Confirm repo branch/base commit.
2. Apply the complete local binary diff, including untracked files via `git add -N .`.
3. Verify data and environment hashes.
4. Run preflights.
5. Execute the sustained arms under `tmux` while polling GPU telemetry.
6. Never finish the conversation while an arm or monitor remains active.

## Result artifact layout

Remote/local global-data root:

```text
/proj/giant-data/TiDAR/benchmarks/lora_vs_full_smollm135_5090_ctx2048/<run_id>/
```

S3 root:

```text
s3://giant-data/TiDAR/benchmarks/lora_vs_full_smollm135_5090_ctx2048/<run_id>/
```

Upload a compact bundle, not the disposable full checkpoints:

```text
README.md
experiment_manifest.json
historical_baseline.json
results.json
environment.json
dataset_sha256.txt
source_diff.patch
configs/
full_a/train_console.log
full_a/gpu_stats.csv
lora_b4/train_console.log
lora_b4/gpu_stats.csv
full_b/train_console.log
full_b/gpu_stats.csv
lora_b8/train_console.log       # if run
lora_b8/gpu_stats.csv           # if run
parser_output.txt
```

`results.json` must include raw numerator/denominator values, excluded intervals, retained duration, and formulas so later changes can recompute every headline number.

## Cleanup

After local and S3 verification:

1. Copy the compact result directory from the pod if direct pod S3 credentials are unavailable.
2. Verify `results.json`, manifest, raw logs, and source patch in S3.
3. Restore the exact pre-experiment `sync_dirs.txt` and `entrypoint.sh` objects.
4. Remove the pod, not merely stop it.
5. Confirm the RunPod list contains no experiment pod.
6. Update this document with measured values, limitations, and the exact S3 root.
7. Notify the user only on final success, final failure, a blocker, or required attention.

## Expected interpretation

Frozen LoRA certainly reduces trainable values, optimizer state, parameter-gradient storage, checkpoint size, and trainable-state communication. It still performs the frozen backbone forward pass and must backpropagate through frozen operations to reach earlier adapters. The experiment may therefore show a modest speedup, no speedup, or even a slowdown from LoRA branch overhead at matched batch 4. Any outcome is valid.

The optional batch-8 arm addresses a different question: whether the memory savings permit a more efficient microbatch and therefore higher practical tokens/second. Keep that result separate from the matched-batch method delta.

## Investigation journal

### 2026-07-20

- Located the January 2026 greedy experiment and its exact S3 lineage.
- Confirmed that S3 and committed logs are byte-identical by SHA256.
- Rejected the May 2026 timing probes as a 5090 baseline because their logs expose a 96 GB allocator environment.
- Reconstructed historical sustained throughput from the clean second log segment.
- Corrected the old documentation's optimizer-step interpretation: logged global steps are microbatch steps.
- Proved runtime batch 4 from total steps and saved `rows_consumed`, despite committed batch 8.
- Confirmed the full 500M-token Arrow dataset exists locally but not in S3.
- Recorded dataset, base, tokenizer, and shard hashes.
- Chose current full-versus-LoRA on the same pod as the primary comparison and historical full versus current LoRA as secondary.
- Chose 4,096 records per arm to target approximately 5–10 sustained post-compile minutes without modifying training-loop stop behavior.

### 2026-07-21 execution

- Added matched full and frozen-LoRA benchmark configs; both resolve to 4,096 rows, 1,024 batch-4 microsteps, effective batch 16, and identical model/optimizer/loss controls.
- Passed the complete local TiDAR suite: 30 tests.
- Uploaded all 11 recovered dataset objects to S3; listed total size is 4,000,993,356 bytes including manifests/statistics.
- Backed up the shared Code-JEPA startup objects, then installed and hash-verified the experiment-specific sync list and no-op entrypoint.
- The first sustained full launch stopped before training because 1,024 total steps with 1,000 warmup steps exposed an existing invalid Optax cosine span (`decay_steps=-976`). Preserved that failed log and set both benchmark configs to `warmup_steps=1024`, which selects the constant-schedule branch without changing production code or the matched optimizer workload.
- A replacement pod `7pvbrydfkdad6v` passed all input hashes and exposed one 32,607 MiB RTX 5090; the first allocation `tenny6dph6iln2` never entered container uptime and was removed before measurement.
- Full preflight saved 134,515,584 finite values; LoRA preflight saved exactly 553,536 finite adapter values. Final LoRA checkpoints contain only 60 `o_proj` LoRA leaves plus one mask embedding, and all 30 B matrices are nonzero.
- Completed and validated two full and two LoRA sustained runs. Matched LoRA delivered 1.71423x throughput with 0.24% repeat disagreement; full disagreement was 0.92%.
- The optional LoRA batch-8 arm OOMed as declared in the plan and was not used in any headline comparison.
- Published a 54-file, 2,004,799-byte compact bundle to S3 and verified key objects by independent download and SHA256.
- Restored the exact shared Code-JEPA startup objects and removed pod `7pvbrydfkdad6v`; the final RunPod list is empty.

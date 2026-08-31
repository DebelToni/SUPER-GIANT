# TiDAR LoRA last-quarter speed experiment

## Status

**Experiment complete: validated, published to S3, startup restored, and all pods removed.**

Last updated: 2026-07-21.

## Question

Measure whether restricting rank-16 `o_proj` LoRA to the final quarter of the
SmolLM-135M TiDAR transformer makes training faster than LoRA in all layers.
Quality is outside this short speed experiment.

For 30 transformer blocks, the suffix starts at:

```text
floor(0.75 * 30) = 22
selected layers = 22, 23, 24, 25, 26, 27, 28, 29
```

This is 8/30 layers (26.7%); an exact 7.5-layer quarter is impossible.

## Design

Both arms use one exact RTX 5090 and identical:

- frozen SmolLM-135M base;
- FineWeb-Edu/Cosmo2 records and shuffle seed 7777;
- clean context 2048 and doubled TiDAR context 4096;
- batch 4, gradient accumulation 4, effective batch 16;
- rank 16, alpha 32, token-routed `o_proj` LoRA;
- loss `alpha=1`, `beta=1`, `delta=3`;
- BF16 compute, FP32 parameters, no rematerialization;
- 8,192 clean rows, 2,048 microbatches, and 16,777,216 clean tokens.

Arms:

1. **All layers:** LoRA in layers 0–29, with a gradient boundary before layer 0.
2. **Last quarter:** LoRA in layers 22–29, with a gradient boundary before layer 22.

The boundary freezes the separate input-only mask embedding. Its leaf remains in
the adapter collection for inference compatibility, receives zero gradient, and
is excluded from AdamW weight decay. In the suffix arm, reverse-mode AD therefore
needs only layers 22–29; layers 0–21 execute forward-only.

Primary comparison:

```text
speedup = last-quarter clean tokens/s / all-layer clean tokens/s
```

The previous matched full-training aggregate of 18,264.8 clean tokens/s is a
secondary reference only because it was measured on a different RTX 5090 pod.

## Gates

- [x] Selected adapter checkpoint contains only layers 22–29 plus mask embedding.
- [x] Mask embedding gradient is exactly zero with the boundary enabled.
- [x] Local TiDAR and GIANT LoRA tests pass.
- [x] Both configs resolve to 8,192 rows and 2,048 microbatches.
- [x] Input hashes match the previous experiment.
- [x] Each method has at least five aggregate retained post-compile minutes.
- [x] Interval CV is reported and no thermal/OOM event contaminates the accepted result.
- [x] Raw logs, one-second GPU telemetry, configs, parser, source patch, and
      results are uploaded to S3 before pod removal.
- [x] Shared startup objects are restored and the final pod list is empty.

## Artifact root

```text
s3://giant-data/TiDAR/benchmarks/lora_last_quarter_smollm135_5090_ctx2048/20260721T1203Z_9g163qv9/
```

## Execution log

- Added manifest-backed `layer_indices` and `stop_gradient_before_lora` support.
- Preserved old adapter-manifest compatibility by omitting both new fields when
  they have default values.
- Local results: TiDAR 31 passed; GIANT v3 20 passed/1 skipped; forced
  two-device LoRA parity passed; focused Ruff and compile checks passed.
- Both benchmark configs resolve to 8,192 rows, 2,048 microbatches, effective
  batch 16, and the existing constant learning-rate branch.
- The first assigned 5090 became provider-contended at 99–100% utilization and
  575 W with no owning container process. Its early control window was mirrored
  but excluded. A concurrently allocated idle EUR-NO-1 5090 ran both accepted
  arms naturally to completion.

## Results

Accepted hardware and software:

```text
pod: 9g163qv91aupyg
GPU: NVIDIA GeForce RTX 5090, 32,607 MiB, 575 W limit
region: EUR-NO-1
JAX/JAXlib: 0.10.1/0.10.1
source diff SHA256: 67a70691ceb3973358db7fb23e3822dba30d4973571162e7ff5aa5619fdbf929
```

Post-compile results exclude each run's first 24-microbatch warm-up interval:

| Measurement | All 30 layers | Layers 22–29 |
|---|---:|---:|
| Retained microbatches | 2,000 | 2,225 across two runs |
| Retained seconds | 541.5 | 310.7 |
| Seconds/microbatch | 0.270750 | 0.139640 |
| Clean tokens/s | 30,256.7 | 58,665.0 |
| Projected 500,000,768-token time | 4.590 h | 2.367 h |
| Interval CV, main run | 1.23% | 2.62% |
| Peak GPU allocation | 27,717 MiB | 27,194 MiB |
| Mean sampled power | 454.8 W | 464.1 W weighted |
| Stored adapter values | 553,536 | 148,032 |
| Optimizer state | 4,435,129 bytes | 1,186,273 bytes |

Primary result:

```text
last-quarter/all-layer throughput ratio: 1.93891x
throughput increase:                      93.89%
projected time reduction:                 48.42%
projected 500M time saved:                2.223 hours
```

The main and short suffix runs measured 58,577.0 and 59,458.1 clean tokens/s,
a 1.49% relative difference. The accepted all-layer control is 3.36% below the
previous pod's 31,310.1 tokens/s, satisfying the cross-pod 5% sanity gate.

The selected adapter has 147,456 gradient-enabled matrix values; its extra 576
stored values are the fixed mask embedding. All final mask embeddings are
pathwise identical, all 8 selected B matrices are nonzero, and no adapter leaf
exists before layer 22. Matrix count is exactly 3.75 times smaller than the
30-layer control. Peak allocation falls by only 523 MiB (1.89%): the speedup is
primarily from removing reverse-mode work through layers 0–21, not from a large
activation-memory reduction.

Weighted sampled power rises about 2.0%, while estimated energy per clean token
falls about 47.4% because throughput nearly doubles.

The previous matched full-training result (18,264.8 clean tokens/s) implies a
secondary cross-pod ratio of 3.212x for the suffix method. It is not a primary
same-pod comparison.

Quality was not evaluated. At the final logged training batch, the suffix arm
had weaker loss/overlap proxies than the all-layer arm, which is expected from
its smaller capacity but is not a held-out quality measurement. A longer
quality-matched experiment is required before choosing the suffix design for a
production run.

The published bundle contains 45 objects and 5,475,678 bytes. Shared Code-JEPA
startup objects were restored byte-for-byte, all temporary RTX 5090 pods were
deleted, and the final RunPod list was empty.

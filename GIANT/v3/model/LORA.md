# LoRA in GIANT v3 and TiDAR

## Design

GIANT keeps immutable pretrained weights and trainable adapter state in different Flax collections:

```text
params/       original model weights
adapters/     LoRA A/B matrices and adapter-owned inputs
cache/        inference KV cache
```

Every adapted projection evaluates the original dense operation directly:

```text
base = W x
delta = (alpha / rank) * B(A(dropout(x)))
output = where(adapter_mask, base + delta, base)
```

The implementation does not replace `nn.Dense`, so old parameter paths and NPZ checkpoints remain compatible. It does not use merged-weight subtraction. A false route selects the already-computed base result.

LoRA A uses fan-in variance scaling and B starts at zero. The initial adapter therefore reproduces the base model exactly. The first update changes B; later updates train both factors.

Supported projection targets match GIANT's fused modules:

```text
qkv_proj
o_proj
fc1          # fused SwiGLU gate/up projection
fc2          # SwiGLU down projection
```

The tied input embedding / LM head is not adapted.

## Configuration

General fine-tuning uses global routing:

```yaml
lora:
  enabled: true
  rank: 16
  alpha: 32.0
  dropout: 0.0
  target_modules: [qkv_proj, o_proj, fc1, fc2]
  routing: global
  base_checkpoint: /path/to/base-run-or-params.npz

training:
  finetune_method: lora
```

`base_checkpoint` may name an NPZ, a `params/` directory, or a run directory containing `params/`. `--base_checkpoint` overrides the YAML value. For compatibility, `--init_checkpoint` is also accepted as the frozen base in LoRA mode.

TiDAR uses token routing:

```yaml
lora:
  enabled: true
  rank: 16
  alpha: 32.0
  dropout: 0.0
  target_modules: [o_proj]
  routing: token
  layer_indices: null              # null means every transformer layer
  stop_gradient_before_lora: false
  base_checkpoint: /path/to/ar-base.npz
  separate_mask_embedding: true

training:
  finetune_method: lora
```

Token mode rejects model calls that omit a boolean `adapter_mask` with the same `[batch, sequence]` shape as the token IDs. `layer_indices` can restrict adapters to explicit zero-based transformer layers. When `stop_gradient_before_lora=true`, the hidden state immediately before the first selected layer is detached; this is exact only when no earlier trainable component is intended to receive gradients.

## Training

```bash
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/model/Run_training.py \
  --config GIANT/v3/Configs/Training/lora_tinystories_30m_smoke.yml
```

Only the `adapters` tree is differentiated, accumulated, optimized, and checkpointed. An Optax parameter mask is deliberately insufficient because differentiating the full `params` tree would still allocate base gradients.

LoRA runs write:

```text
run_root/
  adapter_config.json
  adapters/step_XXXXXXX.npz
  training_states/
  run_manifest.json
```

`adapter_config.json` records the resolved base checkpoint path, size, SHA256, LoRA configuration, and parameter counts. Resume validates this immutable provenance before loading adapter state, then validates adapter keys and shapes before compilation:

```bash
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/model/Run_training.py \
  --config CONFIG.yml --resume latest
```

Set adapter weight decay explicitly through `optimizer.weight_decay`; the TinyStories smoke config uses zero.

### Layer-restricted training

`layer_indices` uses zero-based transformer block indices. With
`stop_gradient_before_lora=true`, reverse-mode differentiation stops at the
input to the first selected block. Do not enable this when an embedding or an
earlier adapter is intended to train.

A SmolLM-135M TiDAR benchmark selected layers 22–29 of 30 and fixed the separate
mask embedding. On one clean RTX 5090 at clean context 2048 and batch 4, it ran
at 58,665.0 clean tokens/s versus 30,256.7 for all-layer LoRA: 1.939x throughput
and 48.42% less projected wall time. Peak allocation changed by only 523 MiB,
so this is chiefly a backward-compute optimization. Quality was not evaluated.
See `TiDAR/Docs/LoRA_Last_Quarter_Speed_Experiment.md` for controls, telemetry,
and limitations.

## Multi-GPU

LoRA follows GIANT v3's existing replicated data-parallel paths. Base weights are replicated and read-only; adapters, adapter gradients, and adapter optimizer state are replicated and synchronized.

```bash
# pmap
GIANT_MULTI_GPU_BACKEND=pmap PYTHONPATH=. /opt/venv/bin/python \
  GIANT/v3/model/Run_training.py --config CONFIG.yml

# shard_map
GIANT_MULTI_GPU_BACKEND=shard_map PYTHONPATH=. /opt/venv/bin/python \
  GIANT/v3/model/Run_training.py --config CONFIG.yml
```

`training.batch_size` is per device unless `training.global_batch_size` is set. A configured global batch must be divisible by the local device count.

Current data parallelism replicates the base and adapter trees; it does not tensor-shard either tree.

## Validated TinyStories smoke

`Configs/Training/lora_tinystories_30m_smoke.yml` completed 10,240 updates on one RTX A5000 in 526.4 seconds. Its smoke-only base is TinyStories TiDAR `step_0090000` with only the final TiDAR mask-vocabulary row removed to restore the original 8,192-token output vocabulary. The frozen model had 31,471,488 parameters; the `o_proj` rank-16 adapter had 221,184 parameters (0.703%) and its final NPZ was 813 KiB.

```text
first logged loss (step 10): 1.365
final loss (step 10,240):    1.186
first 10-step interval:      26.1 s including compilation
median later 10-step interval: 0.4 s
peak GPU memory:             22,167 MiB / 24,564 MiB
mean active GPU utilization: 83.7% (95% peak)
```

The exact adapter-only run, optimizer state, logs, and GPU samples are stored under:

```text
s3://giant-data/TiDAR/checkpoints/TinyStories_exp/giant_lora_smoke/
```

`--resume latest` selected the newer full adapter checkpoint at step 10,240 over the retained mini checkpoint at step 10,200. Adapter-backed cached generation produced:

```text
Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a
```

This smoke used one physical GPU. Replicated `pmap` and `shard_map` equivalence is covered separately by the forced two-device test below.

## Inference

Raw generation and chat accept a separate adapter checkpoint:

```bash
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/model/Generate_faster.py \
  --config CONFIG.yml \
  --checkpoint /path/to/base.npz \
  --adapter /path/to/adapter.npz \
  --prompt "Once upon"
```

```bash
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/model/Generate_chat.py \
  --config CONFIG.yml \
  --checkpoint /path/to/base.npz \
  --adapter /path/to/adapter.npz \
  --prompt "Tell me a story"
```

`--adapter` can name an adapter NPZ or a run directory containing `adapters/`. It defaults to `lora.adapter_checkpoint` when configured. Global adapters remain active during prefill and cached decoding; changing adapters invalidates existing KVs.

Adapter-backed validation uses the same provenance checks:

```bash
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/model/Evaluate.py \
  --config CONFIG.yml \
  --checkpoint /path/to/base.npz \
  --adapter /path/to/adapter-run
```

Validation metadata is written to the adapter checkpoint rather than the frozen base.

No runtime weight merge is performed. An eventual offline merge is valid for globally routed adapters, but token-routed TiDAR adapters cannot be merged into one base matrix.

## TiDAR routing

TiDAR creates routes beside its structured inputs:

```text
training:             clean S = off, diffusion S = on
prompt + draft:       prompt P = off, initial masks K = on
K + K² decode:        verifier K = off, predraft K² = on
```

The verifier cannot attend predraft keys. Only adapter-off verifier KVs are written into the committed cache. Frozen-base TiDAR uses a separate trainable mask input embedding in `adapters`; the tied original embedding/LM-head matrix keeps its original vocabulary size.

## CPU checks

```bash
/Volumes/SSD/v/SG/bin/python -m pytest -q \
  GIANT/v3/tests/test_lora.py \
  TiDAR/tests/test_lora_routing.py \
  TiDAR/tests/test_speculative_sampling.py

XLA_FLAGS=--xla_force_host_platform_device_count=2 \
JAX_PLATFORMS=cpu \
/Volumes/SSD/v/SG/bin/python -m pytest -q \
  GIANT/v3/tests/test_lora_multigpu.py
```

The tests cover legacy parameter paths, zero initialization, hard off-routing, adapter-only gradients, rematerialization, NPZ round trips, pmap/shard-map gradient equivalence, TiDAR verifier logit/KV invariance, input-only mask embeddings, and cached TiDAR generation.

A four-update synthetic run produced byte-identical `pmap` and `shard_map` adapter NPZs (`f70ea20c…`, maximum leaf difference 0). LoRA-disabled training was also compared with pre-LoRA commit `a1add11`; both final NPZs were byte-identical (`212bce45…`, maximum difference 0 across 14 leaves).

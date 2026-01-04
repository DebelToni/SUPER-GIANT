# TRM encoder + SmolLM experiments

## Summary of what ran
- Decoder init: SmolLM-135M pretrained weights loaded (182/182 tensors).
- Encoder/cross/expander init: random (new modules), with cross-attn + expander gate_init set to -2.0 for the large run (sigmoid ~0.12).
- Encoder size for 30-layer runs: max_length=512, num_slots=128 (kept small vs decoder).
- Analytics enabled: grad_norm, cross_gate, exp_gate, trainable fraction.
- No-cross evaluation enabled for comparison.
- Decoder unfreeze plan for larger run: step 500 (~25% of 2000 steps) to let encoder/cross learn useful signals before updating the decoder.

## Key results
- 30-layer small run (200 steps): cross-attn gives a small benefit in later evals; early steps show no-cross slightly better.
- 30-layer larger run (2000 steps): cross vs no-cross stays very close with no clear consistent advantage; suggests decoder mostly ignores cross-attn at this scale, with only small or noisy gains.

## Commands and stdout (remote)

### Convert SmolLM-135M to NPZ
```bash
root@gpu-box:/workspace/app/SUPER-GIANT/v2/smol# /opt/venv/bin/python download_and_convert_smollm_135m.py --out /workspace/app/giant-data/TRM/smol/smollm-135m.npz
[HF] Loading model: HuggingFaceTB/SmolLM-135M
[HF] Config:
[JAX] Building GiantGPT skeleton...
[MAP] Embeddings...
[MAP] Final RMSNorm (model.norm.weight -> final_norm/scale)...
[MAP] Transformer blocks...
  Layer 0 -> TinyTransformerBlock_0
  Layer 1 -> TinyTransformerBlock_1
  Layer 2 -> TinyTransformerBlock_2
  Layer 3 -> TinyTransformerBlock_3
  Layer 4 -> TinyTransformerBlock_4
  Layer 5 -> TinyTransformerBlock_5
  Layer 6 -> TinyTransformerBlock_6
  Layer 7 -> TinyTransformerBlock_7
  Layer 8 -> TinyTransformerBlock_8
  Layer 9 -> TinyTransformerBlock_9
  Layer 10 -> TinyTransformerBlock_10
  Layer 11 -> TinyTransformerBlock_11
  Layer 12 -> TinyTransformerBlock_12
  Layer 13 -> TinyTransformerBlock_13
  Layer 14 -> TinyTransformerBlock_14
  Layer 15 -> TinyTransformerBlock_15
  Layer 16 -> TinyTransformerBlock_16
  Layer 17 -> TinyTransformerBlock_17
  Layer 18 -> TinyTransformerBlock_18
  Layer 19 -> TinyTransformerBlock_19
  Layer 20 -> TinyTransformerBlock_20
  Layer 21 -> TinyTransformerBlock_21
  Layer 22 -> TinyTransformerBlock_22
  Layer 23 -> TinyTransformerBlock_23
  Layer 24 -> TinyTransformerBlock_24
  Layer 25 -> TinyTransformerBlock_25
  Layer 26 -> TinyTransformerBlock_26
  Layer 27 -> TinyTransformerBlock_27
  Layer 28 -> TinyTransformerBlock_28
  Layer 29 -> TinyTransformerBlock_29
[SAVE] Writing NPZ to /workspace/app/giant-data/TRM/smol/smollm-135m.npz ...
[DONE] Conversion complete.
```
Note: stdout lines that print the HF config values were omitted from capture in this session.

### Prepare datasets
```bash
root@gpu-box:/workspace/app/SUPER-GIANT/TRM/TRM-encoder-LLM# /opt/venv/bin/python prepare_dataset.py --max_train_samples 50 --max_val_samples 10 --encoder_max_len 256 --decoder_max_len 128 --dataset_out trm_encoder_llm/datasets/smoke_ultrachat_sft.npz
[dataset] loading HuggingFaceH4/ultrachat_200k split=train_sft/test_sft
[dataset] saved /workspace/app/giant-data/TRM/trm_encoder_llm/datasets/smoke_ultrachat_sft.npz train (50, 256) val (10, 256)
```

```bash
root@gpu-box:/workspace/app/SUPER-GIANT/TRM/TRM-encoder-LLM# /opt/venv/bin/python prepare_dataset.py --max_train_samples 2000 --max_val_samples 200 --encoder_max_len 1024 --decoder_max_len 256 --dataset_out trm_encoder_llm/datasets/ultrachat_sft_2k.npz
[dataset] loading HuggingFaceH4/ultrachat_200k split=train_sft/test_sft
[dataset] saved /workspace/app/giant-data/TRM/trm_encoder_llm/datasets/ultrachat_sft_2k.npz train (2000, 1024) val (200, 1024)
```

```bash
root@gpu-box:/workspace/app/SUPER-GIANT/TRM/TRM-encoder-LLM# /opt/venv/bin/python prepare_dataset.py --max_train_samples 2000 --max_val_samples 200 --encoder_max_len 512 --decoder_max_len 256 --dataset_out trm_encoder_llm/datasets/ultrachat_sft_2k_e512.npz
[dataset] loading HuggingFaceH4/ultrachat_200k split=train_sft/test_sft
[dataset] saved /workspace/app/giant-data/TRM/trm_encoder_llm/datasets/ultrachat_sft_2k_e512.npz train (2000, 512) val (200, 512)
```

```bash
root@gpu-box:/workspace/app/SUPER-GIANT/TRM/TRM-encoder-LLM# /opt/venv/bin/python prepare_dataset.py --max_train_samples 10000 --max_val_samples 1000 --encoder_max_len 512 --decoder_max_len 256 --dataset_out trm_encoder_llm/datasets/ultrachat_sft_10k_e512.npz
[dataset] loading HuggingFaceH4/ultrachat_200k split=train_sft/test_sft
[dataset] saved /workspace/app/giant-data/TRM/trm_encoder_llm/datasets/ultrachat_sft_10k_e512.npz train (10000, 512) val (1000, 512)
```

### 30-layer run (small encoder, short training)
```bash
root@gpu-box:/workspace/app/SUPER-GIANT/TRM/TRM-encoder-LLM# /opt/venv/bin/python Run_training.py --config Config_full30.yml
[pretrained] loaded 182/182 tensors from /workspace/app/giant-data/TRM/smol/smollm-135m.npz
[eval] step=50 loss=1.8221 no_cross=1.8156
[eval] step=100 loss=1.9193 no_cross=1.9388
[ckpt] saved /workspace/app/giant-data/TRM/trm_encoder_llm/checkpoints/step_0000100
[eval] step=150 loss=1.4897 no_cross=1.5367
[eval] step=200 loss=1.5510 no_cross=1.6283
[ckpt] saved /workspace/app/giant-data/TRM/trm_encoder_llm/checkpoints/step_0000200
```
Note: tqdm progress-bar stdout (loss/grad_norm/gates per step) omitted for brevity.

### ETA check for larger run (timeout 120s)
```bash
root@gpu-box:/workspace/app/SUPER-GIANT/TRM/TRM-encoder-LLM# timeout 120 /opt/venv/bin/python Run_training.py --config Config_full30_large.yml
[pretrained] loaded 182/182 tensors from /workspace/app/giant-data/TRM/smol/smollm-135m.npz
train:   0%|##########| 7/2000 [00:04<20:11,  1.64it/s, loss=2.63, grad_norm=0.928, cross_gate=0.119, exp_gate=0.119, train_frac=0.329]
```
Note: tqdm bar characters normalized to ASCII.

### 30-layer larger run (2000 steps, analytics on, no-cross eval)
```bash
root@gpu-box:/workspace/app/SUPER-GIANT/TRM/TRM-encoder-LLM# /opt/venv/bin/python Run_training.py --config Config_full30_large.yml
[pretrained] loaded 182/182 tensors from /workspace/app/giant-data/TRM/smol/smollm-135m.npz
[eval] step=200 loss=2.2036 no_cross=2.2415
[eval] step=400 loss=2.4623 no_cross=2.4785
[ckpt] saved /workspace/app/giant-data/TRM/trm_encoder_llm/checkpoints/step_0000500
[train] unfreezing decoder at step 500
[eval] step=600 loss=1.7670 no_cross=1.7717
[eval] step=800 loss=1.8464 no_cross=1.8467
[ckpt] saved /workspace/app/giant-data/TRM/trm_encoder_llm/checkpoints/step_0001000
[eval] step=1000 loss=2.0721 no_cross=2.0736
[eval] step=1200 loss=2.1530 no_cross=2.1517
[eval] step=1400 loss=2.2641 no_cross=2.2610
[ckpt] saved /workspace/app/giant-data/TRM/trm_encoder_llm/checkpoints/step_0001500
[eval] step=1600 loss=2.2352 no_cross=2.2352
[eval] step=1800 loss=2.1117 no_cross=2.1104
[eval] step=2000 loss=2.3770 no_cross=2.3774
[ckpt] saved /workspace/app/giant-data/TRM/trm_encoder_llm/checkpoints/step_0002000
```
Note: tqdm progress-bar stdout (loss/grad_norm/gates per step) omitted for brevity.

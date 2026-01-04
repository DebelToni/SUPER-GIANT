# TRM encoder recursive updates + cross-attn ablation (prep)

This folder now supports a streaming-style TRM encoder update that reuses the previous Z state when processing prompt chunks, plus a config to apply cross-attention only in the first decoder layers with dropout.

## Large dataset prep (x10)
Uses the same source dataset but scales to 20k/2k samples and a 512-token encoder window.

```bash
/opt/venv/bin/python prepare_dataset.py --config Config_dataset_large.yml
```

Output path (from config):
- `trm_encoder_llm/datasets/ultrachat_sft_20k_e512.npz`

If you want x20 instead, edit `Config_dataset_large.yml` to set:
- `max_train_samples: 40000`
- `max_val_samples: 4000`

## Transfer dataset to GPU box
Data root comes from `TRM/Global_Config.yml`:
- `/workspace/app/giant-data/TRM`

Example copy (local -> GPU box):
```bash
scp /workspace/app/giant-data/TRM/trm_encoder_llm/datasets/ultrachat_sft_20k_e512.npz \
  root@gpu-box:/workspace/app/giant-data/TRM/trm_encoder_llm/datasets/
```
For datasets larger than ~10GB, use the S3 sync workflow instead of scp.

## Run the recursive + cross-attn-first-layers experiment
Config file:
- `Config_recursive_experiment.yml`

Command:
```bash
/opt/venv/bin/python Run_training.py --config Config_recursive_experiment.yml
```

Key settings in that config:
- `model.encoder.recursive_updates.enabled: true`
- `model.encoder.recursive_updates.stride: 64` (chunk size)
- `model.encoder.recursive_updates.short_L_cycles: 1`
- `model.encoder.recursive_updates.short_H_cycles: 1`
- `model.encoder.recursive_updates.keep_y: false` (only Z carries)
- `model.cross_attention.dropout_rate: 0.1`
- `model.cross_attention.layers: "0,1,2,3,4,5"` (cross-attn on first layers only)

Adjust these if you want a different stride, shorter update depth, or to keep Y as well.

## Notes
- No commands have been run yet for this setup.
- You can override the dataset path at runtime with `--dataset` if needed.

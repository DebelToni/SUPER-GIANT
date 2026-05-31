# GIANT v3 model

This folder contains the v3 model definition, dataloader, training loop, checkpoint handling, and generation scripts.

## Main scripts

- [Run_training.py](Run_training.py) - main training entrypoint
- [Generate_chat.py](Generate_chat.py) - chat-style generation and browser/window demo path
- [Generate_faster.py](Generate_faster.py) - raw text generation path
- [Evaluate.py](Evaluate.py) - random curriculum/sample evaluation
- [checkpoint_manager.py](checkpoint_manager.py) - checkpoint save/load helpers
- [arrow_data_loader.py](arrow_data_loader.py) - Arrow shard dataloader used by training

## Model files

- [GiantGPT.py](GiantGPT.py) - top-level Flax module
- [Transformer_block.py](Transformer_block.py) - causal attention, MLP, and XSA
- [model_mode.py](model_mode.py) - compatibility helpers that now accept decoder mode only
- [jit_inference.py](jit_inference.py) - KV-cache inference state and JIT generation functions
- [optimizer_utils.py](optimizer_utils.py) - LR schedule and optimizer construction

## Training configs to look at first

- English 100M pretrain: [../Configs/Training/1_pretraining_100m_ctx256_4b.yml](../Configs/Training/1_pretraining_100m_ctx256_4b.yml)
- English 100M booster: [../Configs/Training/2_curated_booster_100m_ctx256.yml](../Configs/Training/2_curated_booster_100m_ctx256.yml)
- English 100M SFT: [../Configs/Training/3_sft_100m_ctx256_4x.yml](../Configs/Training/3_sft_100m_ctx256_4x.yml)
- BG+EN 100M pretrain: [../Configs/Training/1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml](../Configs/Training/1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml)
- BG+EN 100M booster: [../Configs/Training/2_curated_booster_100m_bg_en_ctx256_32k.yml](../Configs/Training/2_curated_booster_100m_bg_en_ctx256_32k.yml)
- BG+EN 100M SFT: [../Configs/Training/3_sft_100m_bg_en_ctx256_32k_smoltalk.yml](../Configs/Training/3_sft_100m_bg_en_ctx256_32k_smoltalk.yml)

## Command reminders

Train from scratch or from config init:

```bash
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/model/Run_training.py --config GIANT/v3/Configs/Training/1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml
```

Move to the next stage with a fresh optimizer:

```bash
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/model/Run_training.py --config GIANT/v3/Configs/Training/2_curated_booster_100m_bg_en_ctx256_32k.yml --init_checkpoint /proj/giant-data/GIANT/GIANT-Chat/training/1_pretraining_100m_bg_en_ctx256_32k_1p8b/params/step_0109862.npz
```

Resume only an interrupted same-stage run:

```bash
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/model/Run_training.py --config GIANT/v3/Configs/Training/1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml --resume latest
```

Run chat inference:

```bash
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/model/Generate_chat.py --config GIANT/v3/Configs/Training/3_sft_100m_ctx256_4x.yml --checkpoint /proj/giant-data/GIANT/GIANT-Chat/training/3_sft_100m_ctx256_4x/params/step_0106906.npz --prompt "Explain RAM in one sentence." --greedy --max_context 256
```

## Rules

- GIANT v3 is decoder-only and supports KV-cache generation.
- `enable_xsa: true` is tested in the current chat-stack configs.
- `num_kv_heads == num_heads` means full MHA; smaller `num_kv_heads` means GQA.
- `--init_checkpoint` loads only params and starts step/optimizer fresh.
- `--resume` loads optimizer/dataloader state and should not be used for stage transitions.
- Final generated checkpoint paths live under the config's `paths.checkpoints_root`.

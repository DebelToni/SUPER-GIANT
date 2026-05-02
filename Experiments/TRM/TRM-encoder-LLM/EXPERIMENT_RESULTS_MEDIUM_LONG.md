# Medium-long TRM encoder + SmolLM experiment (30 layers, 2000 steps)

## Setup
- Config: `Config_full30_large.yml`
- Decoder init: SmolLM-135M pretrained (182/182 tensors loaded)
- Encoder/cross/expander init: random
- Cross/expander gates: `gate_init: -2.0` (sigmoid ~0.12)
- Encoder size: `max_length=512`, `num_slots=128`
- Training: batch_size=2, max_steps=2000, eval_every=200, checkpoint_every=500
- Analytics: enabled (grad_norm, gate stats)
- No-cross eval: enabled
- Decoder unfreeze: at step 500
- Dataset: as configured in `Config.yml` (`trm_encoder_llm/datasets/ultrachat_sft.npz`)

## Results (cross vs no-cross validation loss)
- step 200: 2.2036 vs 2.2415
- step 400: 2.4623 vs 2.4785
- step 600: 1.7670 vs 1.7717
- step 800: 1.8464 vs 1.8467
- step 1000: 2.0721 vs 2.0736
- step 1200: 2.1530 vs 2.1517
- step 1400: 2.2641 vs 2.2610
- step 1600: 2.2352 vs 2.2352
- step 1800: 2.1117 vs 2.1104
- step 2000: 2.3770 vs 2.3774

## Checkpoints
- `trm_encoder_llm/checkpoints/step_0000500`
- `trm_encoder_llm/checkpoints/step_0001000`
- `trm_encoder_llm/checkpoints/step_0001500`
- `trm_encoder_llm/checkpoints/step_0002000`

## Commands and stdout (remote)

### ETA check (120s timeout)
```bash
root@gpu-box:/workspace/app/SUPER-GIANT/TRM/TRM-encoder-LLM# timeout 120 /opt/venv/bin/python Run_training.py --config Config_full30_large.yml
[pretrained] loaded 182/182 tensors from /workspace/app/giant-data/TRM/smol/smollm-135m.npz
train:   0%|##########| 7/2000 [00:04<20:11,  1.64it/s, loss=2.63, grad_norm=0.928, cross_gate=0.119, exp_gate=0.119, train_frac=0.329]
```
Note: tqdm bar characters normalized to ASCII.

### Full run
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

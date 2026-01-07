# Recursive TRM + cross-attn-first-layers (5090 run)

## Run metadata
- Host: gpu-box-3 (RTX 5090)
- Config: `Config_recursive_experiment.yml`
- Dataset: `trm_encoder_llm/datasets/ultrachat_sft_20k_e512.npz` (20k train / 2k val, enc=512, dec=256)
- Log: `/workspace/app/giant-data/TRM/trm_encoder_llm/logs/recursive_experiment_20260105_134024.txt`
- Runtime: ~6m33s

## Key settings
- Decoder: 30 layers (SmolLM-135M pretrained loaded)
- Encoder: recursive updates enabled
  - stride=64, short_L_cycles=1, short_H_cycles=1
  - Z carried between chunks, Y reinit each chunk (keep_y=false)
- Cross-attn: enabled only on first 6 layers (0-5), dropout=0.1
- Decoder unfreeze: step 500
- Analytics: on (grad_norm, gates)

## Eval results (cross vs no-cross)
- step 200: 1.3943 vs 1.4876
- step 400: 1.8237 vs 1.8965
- step 600: 2.3558 vs 2.3587
- step 800: 2.5846 vs 2.5832
- step 1000: 2.3039 vs 2.3034
- step 1200: 2.3475 vs 2.3482
- step 1400: 2.0459 vs 2.0458
- step 1600: 2.5572 vs 2.5571
- step 1800: 2.4067 vs 2.4066
- step 2000: 2.0902 vs 2.0902

## Checkpoints
- `trm_encoder_llm/checkpoints/step_0000500.npz`
- `trm_encoder_llm/checkpoints/step_0001000.npz`
- `trm_encoder_llm/checkpoints/step_0001500.npz`
- `trm_encoder_llm/checkpoints/step_0002000.npz`

## Observations
- Early evals show a small cross-attn advantage (steps 200-400), then the gap collapses to ~0.
- Cross/exp gates stayed low (cross_gate ~0.018, exp_gate ~0.017) throughout.
- Decoder unfreeze at step 500 did not create a sustained advantage for cross-attn.

## Interpretation
The decoder still learns to ignore the cross-attn stream after a short warmup, even with recursive encoder updates and cross-attn limited to early layers. The hypothesis that the decoder minimizes loss by ignoring cross-attn is supported by these results.

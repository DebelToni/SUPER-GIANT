# TRM encoder + LLM decoder (text agent notes)

This document is a quick reference for using the TRM encoder + SmolLM decoder experiments.

## Purpose
We test a hypothesis: a pretrained decoder will learn to ignore cross-attention from an alien encoder, even when the encoder is made stronger.

## Repo locations
- Code: `TRM/TRM-encoder-LLM/`
- Global config: `TRM/Global_Config.yml`
- Data root (from global config): `/workspace/app/giant-data/TRM`

## Core components (high level)
- `TRMEncoderDecoder.py`: model definition (SoftMoE compressor + TRM encoder + expander + decoder with optional cross-attn).
- `Run_training.py`: training loop, analytics, eval and checkpoints.
- `prepare_dataset.py`: pulls HF dataset and stores packed arrays in the global data root.

## Data layout (global data root)
- Datasets: `trm_encoder_llm/datasets/*.npz`
- Checkpoints: `trm_encoder_llm/checkpoints/`
- Logs: `trm_encoder_llm/logs/`
- HF cache / tokenizer: `trm_encoder_llm/hf_cache`, `trm_encoder_llm/tokenizer`
- SmolLM weights: `smol/smollm-135m.npz`

## Common configs
- `Config.yml`: base config.
- `Config_full30.yml`: 30-layer decoder, short run.
- `Config_full30_large.yml`: 30-layer decoder, longer run.
- `Config_recursive_experiment.yml`: recursive encoder updates + cross-attn only in early layers + dropout.
- `Config_dataset_large.yml`: 20k/2k dataset preset (x10 over base).

## How to run (local CPU prep)
Prepare dataset (example large preset):
```
python prepare_dataset.py --config Config_dataset_large.yml
```

## How to run (remote GPU)
Use the remote GPU skill. Always call the venv explicitly:
```
/opt/venv/bin/python Run_training.py --config Config_recursive_experiment.yml
```

## Key toggles to know
- Cross-attn layers: `model.cross_attention.layers` ("all" or a list like "0,1,2,3,4,5").
- Cross-attn dropout: `model.cross_attention.dropout_rate`.
- Recursive updates: `model.encoder.recursive_updates` (stride, short cycles, keep_y).
- Unfreeze decoder: `training.train_decoder_after`.
- Evaluation with/without cross-attn: `training.eval_no_cross`.

## Evaluation signals
- Look for `[eval] step=... loss=... no_cross=...` lines in logs.
- If logs use tqdm, normalize `\r` to `\n` before parsing.

## Hypothesis status
Across multiple runs, cross-attn provides small early gains then collapses to near-zero delta. This supports the hypothesis that the pretrained decoder learns to ignore cross-attn to minimize loss.

## Outputs
- Typst overview report: `TRM/TRM-encoder-LLM/EXPERIMENTS_OVERVIEW.typ`
- Results logs: `/workspace/app/giant-data/TRM/trm_encoder_llm/logs/*.txt`

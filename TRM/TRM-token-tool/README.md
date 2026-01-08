# TRM Token Tool (SmolLM Sudoku Calls)

Fine-tunes SmolLM-135M to emit TRM Sudoku tool calls
(`<TRM-sudoku> ... </TRM-sudoku>`) with a mixed dataset so general
capabilities do not collapse.

## Current features
- Custom tokenizer with TRM tokens.
- Mixed dataset builder (Sudoku + chat + general text) with configurable fractions.
- L2 penalty toward the base checkpoint and bfloat16 compute override.
- LLM-only inference plus Python Sudoku solver, with OOD prompt overrides.
- General-loss eval vs base SmolLM and OOD prompt sanity checks.

## Latest results (mix_v4, 10k steps, L2=1e-5)
- General loss delta on simple_wiki: +0.0364 (20 batches, seq_len 256).
- OOD prompt success: 3/3 seeds with max_new_tokens=512 (exact puzzle copy + solver match).
- Final training loss: ~0.2607 at step 10,000.

## Quick Start
0) Install the repo package once (needed for v2/TRM imports):
```
python -m pip install -e ../..
```

1) Build the custom tokenizer (adds TRM tokens):
```
python tokenizer_utils.py
```

2) Generate the mixed dataset (Sudoku + chat + text):
```
python build_sudoku_trm_dataset.py
```

3) Download and convert SmolLM-135M weights (stores in TRM data root):
```
python ../../v2/smol/download_and_convert_smollm_135m.py \
  --out /workspace/app/giant-data/TRM/trm_token_tool/checkpoints/smollm-135m.npz
```

4) Fine-tune:
```
python Run_training.py
```

5) OOD prompt check (LLM-only, solver runs in Python):
```
python Run_inference_sudoku.py \
  --checkpoint /workspace/app/giant-data/TRM/trm_token_tool/checkpoints/step_0010000.npz \
  --user_prompt "I'm working on homework; please solve this Sudoku: {puzzle}." \
  --max_new_tokens 512 --seed 0
```

6) General loss check:
```
python Run_eval_general_loss.py --stage simple_wiki --seq_len 256 --batch_size 2 --num_batches 20
```

## Docs and Config
- Edit `Config.yml` to change mix fractions, steps, and L2 coefficient.
- All outputs live under `/workspace/app/giant-data/TRM/trm_token_tool/...`.
- See `EXPERIMENTS_TRM_TOKEN_TOOL.typ` for the full experiment breakdown and Typst charts.

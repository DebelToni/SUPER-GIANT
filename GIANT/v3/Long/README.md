# LongGIANT DSL

This folder contains the synthetic long-context DSL pipeline for LongGIANT.

The DSL is intentionally small-vocabulary, whitespace-tokenized, and exact-match
scorable so we can test long-context mechanisms before spending heavily on full
natural-language pretraining.

## Levels

- `level 1`: scalar memory and aliasing
  - `DEF`, `SET`, `ALIAS`, `ASK`
- `level 2`: scalar state updates
  - level 1 plus `INC`, `DEC`, `SWAP`
- `level 3`: array/triple style memory
  - `DEFARR`, `SETAT`, `SWAPAT`, `INCAT`, `GET`

## Main scripts

- `GIANT/v3/Long/longdsl.py`
  - DSL vocabulary, tokenizer creation, example generation.
- `GIANT/v3/Long/prepare_longdsl.py`
  - Generates raw train/val/test JSONL, saves a custom tokenizer, writes a
    dataset build config, and can invoke the v3 Arrow data pipeline.
- `GIANT/v3/Long/write_training_configs.py`
  - Writes GIANT v3 training configs for the 30M LongGIANT ladder.
- `GIANT/v3/Long/eval_longdsl.py`
  - Teacher-forced exact-match evaluator on held-out LongDSL examples.

## Data layout

Generated artifacts live under `/proj/giant-data/GIANT/Long/` by default:

- `tokenizers/longdsl_wordlevel/`
- `raw/level{N}_ctx{L}/train.jsonl`
- `raw/level{N}_ctx{L}/val.jsonl`
- `raw/level{N}_ctx{L}/test.jsonl`
- `configs/longdsl_datasets.yml`

Arrow shards are written to `/proj/giant-data/GIANT/dataset_artifacts/longdsl/`.

## Baseline-first experiment plan

1. Generate tokenizer + level 1/2 data for a context ladder.
2. Train a ~30M GIANT v3 model on the plain baseline objective first.
3. Move context upward when the current level/context is solved cleanly.
4. Stop once level 2 at 64K is reached cleanly.

The later `A/B + alignment` objective can then be added on top of the same DSL.

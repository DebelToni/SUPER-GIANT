# LongGIANT Hidden-World Pipeline

This folder now holds the natural-language long-context screening pipeline for
GIANT v3.

The core idea is:

- keep an exact latent world with names, aliases, relations, and updates
- render only natural-language records to the model
- keep answers short and exact-match scorable
- scale context by adding coherent filler records in the same document genre

## Current minimal scope

- one main document genre: `admin_record`
- level 1:
  - alias resolution + relation lookup
- level 2:
  - alias resolution + latest-value update + distractors
- exact one-token answers
- deterministic template renderer only

The local-LLM paraphrase expansion stage is intentionally postponed until this
deterministic base pipeline is stable.

## Main scripts

- `GIANT/v3/Long/longdsl.py`
  - hidden world, tokenizer vocab, sample generation
- `GIANT/v3/Long/Lexicon.yml`
  - editable word banks and template banks for the LongGIANT renderer
- `GIANT/v3/Long/prepare_longdsl.py`
  - raw JSONL generation, tokenizer save, dataset manifest write, Arrow build
- `GIANT/v3/Long/write_training_configs.py`
  - 40M-class GIANT v3 training configs for the current context ladder
- `GIANT/v3/Long/eval_longdsl.py`
  - teacher-forced exact-match evaluation on held-out samples
- `GIANT/v3/Long/eval_openai_long.py`
  - few-shot held-out evaluation for OpenAI models using the same Long JSONL rows

## Data layout

Generated artifacts live under `/proj/giant-data/GIANT/Long/`:

- `tokenizers/long_wordlevel/`
- `records/raw/level{N}_ctx{L}/train.jsonl`
- `records/raw/level{N}_ctx{L}/val.jsonl`
- `records/raw/level{N}_ctx{L}/test.jsonl`
- `configs/long_records_datasets.yml`

Arrow shards are written to `/proj/giant-data/GIANT/dataset_artifacts/long_records/`.

## Notes on the previous raw DSL

The older opcode-style pipeline is now treated as legacy experiment material.
Its motivation and examples are preserved in the Typst docs under
`GIANT/v3/docs/`.

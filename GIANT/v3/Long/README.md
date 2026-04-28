# LongGIANT Hidden-World Pipeline

This folder holds the natural-language long-context screening pipeline for GIANT v3.

The core idea is:

- keep an exact latent world with names, aliases, relations, and updates
- render only natural-language records to the model
- keep answers short and exact-match scorable
- scale context by adding coherent filler records in the same document genre

## Current minimal scope

- one main document genre: `admin_record`
- level 1: alias resolution + relation lookup
- level 2: alias resolution + latest-value update + distractors
- exact one-token answers
- deterministic template renderer only

The local-LLM paraphrase expansion stage is intentionally postponed until this deterministic base pipeline is stable.

## Main scripts

- [longdsl.py](longdsl.py) - hidden world, tokenizer vocab, sample generation
- [Lexicon.yml](Lexicon.yml) - editable word banks and template banks for the renderer
- [prepare_longdsl.py](prepare_longdsl.py) - raw JSONL generation, tokenizer save, dataset manifest write, Arrow build
- [write_training_configs.py](write_training_configs.py) - writes 40M-class training configs for the current context ladder
- [eval_longdsl.py](eval_longdsl.py) - teacher-forced exact-match evaluation on held-out samples
- [eval_openai_long.py](eval_openai_long.py) - few-shot held-out evaluation for OpenAI models using the same Long JSONL rows

## Config examples

- short bootstrap examples: [../Configs/Training/Long/long_40m_l1_ctx128.yml](../Configs/Training/Long/long_40m_l1_ctx128.yml), [../Configs/Training/Long/long_40m_l1_ctx256.yml](../Configs/Training/Long/long_40m_l1_ctx256.yml), [../Configs/Training/Long/long_40m_l1_ctx512.yml](../Configs/Training/Long/long_40m_l1_ctx512.yml)
- answer-hidden encoder comparison: [../Configs/Training/Long/long_40m_l1_ctx512_ans_20k_1ep_encoder_xsa_answer_hidden.yml](../Configs/Training/Long/long_40m_l1_ctx512_ans_20k_1ep_encoder_xsa_answer_hidden.yml)
- long decoder stress run: [../Configs/Training/Long/long_100m_l1_ctx128k_chi40x_decoder_xsa_5090.yml](../Configs/Training/Long/long_100m_l1_ctx128k_chi40x_decoder_xsa_5090.yml)

## Data layout

Generated artifacts live under `/proj/giant-data/GIANT/Long/`:

- `tokenizers/long_wordlevel/`
- `records/raw/level{N}_ctx{L}/train.jsonl`
- `records/raw/level{N}_ctx{L}/val.jsonl`
- `records/raw/level{N}_ctx{L}/test.jsonl`
- `configs/long_records_datasets.yml`

Arrow shards are written to `/proj/giant-data/GIANT/dataset_artifacts/long_records/`.

## Notes on the previous raw DSL

The older opcode-style pipeline is now treated as legacy experiment material. Its motivation and examples are preserved in the Typst docs under [../docs/](../docs/).

# GIANT v3

GIANT v3 is the current working version of GIANT. It keeps the v2-style pipeline but is where I am pushing multi-GPU training, better data curation, encoder/decoder mode support, and practical chat experiments.

The main idea of this folder is simple:

- `Configs/` is the source of truth for experiments
- `data_pipeline/` turns text/chat sources into packed Arrow datasets
- `model/` trains, evaluates, and runs inference
- `data_curation/` builds the small but important data layers around the raw corpora
- `Long/` is the long-context hidden-world screening project

Most paths in serious runs should point under `/proj/giant-data/GIANT/...` or `/proj/giant-data/GIANT/GIANT-Chat/...`. Do not stream heavy datasets during training if you can prepare them once and reuse the Arrow shards.

## Common workflow

1. Pick or write configs under [Configs/](Configs/).
2. Train a tokenizer with [data_pipeline/train_tokenizer.py](data_pipeline/train_tokenizer.py) if the model is fresh.
3. Build datasets with [data_pipeline/build_corpus.py](data_pipeline/build_corpus.py).
4. Train with [model/Run_training.py](model/Run_training.py).
5. Generate with [model/Generate_chat.py](model/Generate_chat.py) for chat checkpoints or [model/Generate_faster.py](model/Generate_faster.py) for raw text generation.

The three-command version is:

```bash
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/data_pipeline/train_tokenizer.py --config GIANT/v3/Configs/Tokenizer/giant_chat_bg_en_bpe32k.yml
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/data_pipeline/build_corpus.py --config GIANT/v3/Configs/Data/giant_chat_pretraining_bg_en_900m_bpe32k.yml
PYTHONPATH=. /opt/venv/bin/python GIANT/v3/model/Run_training.py --config GIANT/v3/Configs/Training/1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml
```

On the Mac, use `/Volumes/SSD/v/SG/bin/python` instead of `/opt/venv/bin/python`.

## Important examples

Current chat-stack examples:

- English 100M tokenizer: [Configs/Tokenizer/giant_chat_bpe24k.yml](Configs/Tokenizer/giant_chat_bpe24k.yml)
- English 100M pretrain data: [Configs/Data/giant_chat_pretraining_4b.yml](Configs/Data/giant_chat_pretraining_4b.yml)
- English 100M training stages: [1_pretraining_100m_ctx256_4b.yml](Configs/Training/1_pretraining_100m_ctx256_4b.yml), [2_curated_booster_100m_ctx256.yml](Configs/Training/2_curated_booster_100m_ctx256.yml), [3_sft_100m_ctx256_4x.yml](Configs/Training/3_sft_100m_ctx256_4x.yml)

BG+EN 100M scratch examples:

- BG+EN tokenizer: [Configs/Tokenizer/giant_chat_bg_en_bpe32k.yml](Configs/Tokenizer/giant_chat_bg_en_bpe32k.yml)
- BG+EN pretrain data: [Configs/Data/giant_chat_pretraining_bg_en_900m_bpe32k.yml](Configs/Data/giant_chat_pretraining_bg_en_900m_bpe32k.yml)
- BG+EN booster data: [Configs/Data/giant_chat_curated_booster_bg_en_bpe32k.yml](Configs/Data/giant_chat_curated_booster_bg_en_bpe32k.yml)
- BG+EN SFT data: [Configs/Data/giant_chat_sft_bg_en_smoltalk_bpe32k.yml](Configs/Data/giant_chat_sft_bg_en_smoltalk_bpe32k.yml)
- BG+EN training stages: [1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml](Configs/Training/1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml), [2_curated_booster_100m_bg_en_ctx256_32k.yml](Configs/Training/2_curated_booster_100m_bg_en_ctx256_32k.yml), [3_sft_100m_bg_en_ctx256_32k_smoltalk.yml](Configs/Training/3_sft_100m_bg_en_ctx256_32k_smoltalk.yml)

Quality-filter examples:

- BG+EN quality tokenizer: [Configs/Tokenizer/giant_bg_en_bpe32k_quality.yml](Configs/Tokenizer/giant_bg_en_bpe32k_quality.yml)
- filtered BG packed corpus: [Configs/Data/giant_bg_quality_keep_bpe32k.yml](Configs/Data/giant_bg_quality_keep_bpe32k.yml)
- quality filter configs live under [data_curation/quality_filter/configs/](data_curation/quality_filter/configs/)

Long-context examples:

- Long docs: [Long/README.md](Long/README.md)
- current long configs: [Configs/Training/Long/](Configs/Training/Long/)

## Training rules that matter

- Use `--init_checkpoint` when moving from pretrain to booster or SFT. That starts a new optimizer schedule.
- Use `--resume` only for continuing the same interrupted run.
- Keep tokenizer/checkpoint vocab shapes aligned. You cannot continue a 24k-tokenizer checkpoint with a 32k tokenizer.
- For chat SFT, use assistant-only loss masks from the chat data config.
- For BG+EN chat stack runs, `context_length: 256`, full MHA, and `enable_xsa: true` are the known tested path.

## Data roots

Normal local/remote layout:

- repo: `/proj/SUPER-GIANT`
- data: `/proj/giant-data`
- HF cache: `/proj/giant-data/hf_cache`
- chat artifacts: `/proj/giant-data/GIANT/GIANT-Chat/`
- shared S3 prefix: `s3://giant-data/GIANT/GIANT-Chat/`

When using the Docker/RunPod setup, keep `/proj/giant-data/sync_dirs.txt` small and explicit. For chat-stack work the safest entry is usually just `GIANT/GIANT-Chat/`.

## Current state of v3

- Decoder mode is the main production path.
- Encoder mode exists through `model.mode: encoder` and is used by classifier/Long experiments, not by chat generation.
- Data parallel training works; more serious model sharding is still future work.
- XSA is supported in the v3 block and used by the current chat-stack configs.
- Quality filtering is useful for data triage, not a replacement for human inspection of SFT data.

## More local docs

- [Configs/README.md](Configs/README.md)
- [data_pipeline/README.md](data_pipeline/README.md)
- [data_curation/README.md](data_curation/README.md)
- [data_curation/quality_filter/README.md](data_curation/quality_filter/README.md)
- [model/README.md](model/README.md)
- [Long/README.md](Long/README.md)

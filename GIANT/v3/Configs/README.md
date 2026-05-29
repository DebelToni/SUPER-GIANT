# GIANT v3 Configs

This folder is the source of truth for v3 experiments. Most scripts take `--config` and expect one of these YAMLs.

Start with [`registry.yml`](registry.yml) if you are not already familiar with the repo. It marks the recommended, stable, smoke, and utility configs.

## Layout

- [Tokenizer/](Tokenizer/) - tokenizer training configs
- [Data/](Data/) - Arrow corpus build configs
- [Training/](Training/) - model/training schedules
- [Training/Long/](Training/Long/) - LongGIANT hidden-world experiments
- [Training/Attention_size_XSA/](Training/Attention_size_XSA/) - attention/XSA comparison experiments

## Recommended current stack

BG+EN 100M scratch stack:

- tokenizer: [Tokenizer/giant_chat_bg_en_bpe32k.yml](Tokenizer/giant_chat_bg_en_bpe32k.yml)
- data: [Data/giant_chat_pretraining_bg_en_900m_bpe32k.yml](Data/giant_chat_pretraining_bg_en_900m_bpe32k.yml), [Data/giant_chat_curated_booster_bg_en_bpe32k.yml](Data/giant_chat_curated_booster_bg_en_bpe32k.yml), [Data/giant_chat_sft_bg_en_smoltalk_bpe32k.yml](Data/giant_chat_sft_bg_en_smoltalk_bpe32k.yml)
- training: [Training/1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml](Training/1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml), [Training/2_curated_booster_100m_bg_en_ctx256_32k.yml](Training/2_curated_booster_100m_bg_en_ctx256_32k.yml), [Training/3_sft_100m_bg_en_ctx256_32k_smoltalk.yml](Training/3_sft_100m_bg_en_ctx256_32k_smoltalk.yml)

English 100M reference stack:

- tokenizer: [Tokenizer/giant_chat_bpe24k.yml](Tokenizer/giant_chat_bpe24k.yml)
- data: [Data/giant_chat_pretraining_4b.yml](Data/giant_chat_pretraining_4b.yml), [Data/giant_chat_curated_booster_strong56.yml](Data/giant_chat_curated_booster_strong56.yml), [Data/giant_chat_sft_4x.yml](Data/giant_chat_sft_4x.yml)
- training: [Training/1_pretraining_100m_ctx256_4b.yml](Training/1_pretraining_100m_ctx256_4b.yml), [Training/2_curated_booster_100m_ctx256.yml](Training/2_curated_booster_100m_ctx256.yml), [Training/3_sft_100m_ctx256_4x.yml](Training/3_sft_100m_ctx256_4x.yml)

## Rules

- `Tokenizer/*.yml` produces a tokenizer directory.
- `Data/*.yml` consumes a tokenizer and produces Arrow shards + manifests.
- `Training/*.yml` consumes a tokenizer and a processed dataset root.
- Stage transitions should use `--init_checkpoint`, not `--resume`.
- `--resume` is only for a crashed run of the same training config.
- Paths should usually live under `/proj/giant-data`, not inside the repo.

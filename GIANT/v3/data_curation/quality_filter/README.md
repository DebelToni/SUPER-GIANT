# Bulgarian quality filter

This folder builds a Bulgarian text-quality classifier and uses it to filter `lumees/bulgarian-corpus-33b`. The current practical winner is the fastText-style classifier, not the tiny encoder.

## Main scripts

- [build_quality_benchmark.py](build_quality_benchmark.py) - build labeled BG quality benchmark
- [train_quality_filter.py](train_quality_filter.py) - train fastText or tiny encoder classifier
- [filter_lumees_pretrain_parallel.py](filter_lumees_pretrain_parallel.py) - parallel filtering for the full `pretrain` split
- [filter_lumees_pretrain.py](filter_lumees_pretrain.py) - simpler streaming filter path
- [materialize_tier_views.py](materialize_tier_views.py) - split filtered JSONL into `high_quality`, `acceptable`, and `keep_all`
- [score_lumees_sft_fasttext.py](score_lumees_sft_fasttext.py) - rough fastText scoring for the `sft` split
- [common.py](common.py) - shared config/path/model helpers

## Config examples

- full benchmark with BPOS positives: [configs/benchmark_bg_full_with_bpos.yml](configs/benchmark_bg_full_with_bpos.yml)
- recommended fastText run: [configs/train_fasttext_bg_full_with_bpos.yml](configs/train_fasttext_bg_full_with_bpos.yml)
- tiny encoder smoke run: [configs/train_encoder_bg_full_with_bpos_smoke.yml](configs/train_encoder_bg_full_with_bpos_smoke.yml)
- packed filtered BG corpus config: [../../Configs/Data/giant_bg_quality_keep_bpe32k.yml](../../Configs/Data/giant_bg_quality_keep_bpe32k.yml)
- tokenizer for filtered BG corpus: [../../Configs/Tokenizer/giant_bg_en_bpe32k_quality.yml](../../Configs/Tokenizer/giant_bg_en_bpe32k_quality.yml)

## Outputs used by later stages

- full filtered pretrain output: `/proj/giant-data/GIANT/data_curation/quality_filter_bg/full_lumees_fasttext/`
- split views: `/proj/giant-data/GIANT/data_curation/quality_filter_bg/full_lumees_fasttext_split/`
- filtered BG packed corpus: `/proj/giant-data/GIANT/quality_filter_bg/data/filtered_bg_keep_bpe32k/`
- BG+EN quality tokenizer: `/proj/giant-data/GIANT/quality_filter_bg/tokenizers/giant_bg_en_bpe32k_quality/`

## What the labels mean

- `high_quality` - prose-like Bulgarian, closest to BPOS/Wikipedia-style positives
- `acceptable` - usable Bulgarian text but lower signal
- `reject` - low-quality, noisy, non-BG, malformed, or very weak text

## Important warning

The fastText classifier is a Bulgarian prose-quality filter. It is not a chat-quality judge. It is useful for filtering `pretrain` text and rough triage of SFT rows, but translated chat data still needs separate inspection.

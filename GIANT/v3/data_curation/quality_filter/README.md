# Bulgarian Quality Filter

This subproject builds labeled Bulgarian text-quality benchmarks and compares two lightweight filters:

- `fasttext`: hashed token embeddings + mean pool + linear head
- `tiny_encoder`: the same token pipeline with a single non-causal transformer block before pooling

The benchmark builder is source-driven and can stream Hugging Face datasets for large runs.

Recommended positive pools:

- `wikimedia/wikipedia` config `20231101.bg`
- `lumees/bulgarian-corpus-33b` config `pretrain`, source `finewiki`
- `lumees/bulgarian-corpus-33b` config `pretrain`, source `bpos_science`

Recommended negative pools:

- `lumees/bulgarian-corpus-33b` config `pretrain`, source `fineweb-2`
- `community-datasets/clickbait_news_bg`
- optional parliamentary or crawl-derived noisy corpora

Quick local benchmark build:

```bash
PYTHONPATH=. "/Volumes/SSD/v/SG/bin/python" GIANT/v3/data_curation/quality_filter/build_quality_benchmark.py \
  --config GIANT/v3/data_curation/quality_filter/configs/benchmark_tiny_local.yml
```

Train the fastText-style baseline:

```bash
PYTHONPATH=. "/Volumes/SSD/v/SG/bin/python" GIANT/v3/data_curation/quality_filter/train_quality_filter.py \
  --config GIANT/v3/data_curation/quality_filter/configs/train_fasttext_tiny_local.yml
```

Train the one-block encoder baseline:

```bash
PYTHONPATH=. "/Volumes/SSD/v/SG/bin/python" GIANT/v3/data_curation/quality_filter/train_quality_filter.py \
  --config GIANT/v3/data_curation/quality_filter/configs/train_encoder_tiny_local.yml
```

# GIANT v3 Data Curation

`curated_wikipedia_search.py` streams English Wikipedia once and writes target-aware JSONL candidates that can be ingested directly by `GIANT/v3/data_pipeline/build_corpus.py`.

The output JSONL contains a `text` field, so a stage source can read it directly:

```yml
sources:
  - type: json
    json_root: "/proj/giant-data/GIANT/GIANT-Chat/data_curation/wikipedia_run"
    file_glob: "candidates.jsonl"
    text_field: "text"
```

Useful commands:

```bash
PYTHONPATH=. "/Volumes/SSD/v/SG/bin/python" GIANT/v3/data_curation/curated_wikipedia_search.py \
  --write-sample-targets GIANT/v3/data_curation/sample_targets_100.jsonl
```

```bash
PYTHONPATH=. "/Volumes/SSD/v/SG/bin/python" GIANT/v3/data_curation/curated_wikipedia_search.py \
  --targets GIANT/v3/data_curation/sample_targets_100.jsonl \
  --checkpoint-dir /proj/giant-data/GIANT/GIANT-Chat/data_curation/wikipedia_run/checkpoints \
  --out /proj/giant-data/GIANT/GIANT-Chat/data_curation/wikipedia_run/candidates.jsonl \
  --coverage-report /proj/giant-data/GIANT/GIANT-Chat/data_curation/wikipedia_run/coverage.json \
  --top-k 50
```

Optional second-stage rerank uses `Qwen/Qwen3-Embedding-0.6B` through `transformers`.
For local testing here, it was run from the general utility env because the project JAX env does not ship with PyTorch:

```bash
PYTHONPATH=. "/Volumes/SSD/v/py/bin/python" GIANT/v3/data_curation/curated_wikipedia_search.py \
  --targets GIANT/v3/data_curation/sample_targets_100.jsonl \
  --checkpoint-dir /proj/giant-data/GIANT/GIANT-Chat/data_curation/wikipedia_run/checkpoints \
  --out /proj/giant-data/GIANT/GIANT-Chat/data_curation/wikipedia_run/candidates.jsonl \
  --coverage-report /proj/giant-data/GIANT/GIANT-Chat/data_curation/wikipedia_run/coverage.json \
  --top-k 50 \
  --use-embeddings true
```

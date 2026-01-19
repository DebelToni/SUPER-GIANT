# TiDAR Data Pipeline Config (Config.yml)

This document describes the dataset preparation options consumed by `GIANT/v2/data_pipeline/build_corpus.py`, which the TiDAR pipeline calls via `TiDAR/data_pipeline/Run_pipeline.py`. It focuses on the YAML fields used to build Arrow shards for training.

Paths and global defaults
- The dataset config is merged with `Global_Config.yml`. Relative paths are resolved under `paths.data_root` (from `Global_Config.yml`).
- Output shards, manifest, and stats are written under `outputs.processed_root` (resolved path).

---

## Top-level keys

### outputs
Controls shard writing and metadata filenames.
- `processed_root` (str): Output directory for all stage folders. Example: `"dataset_artifacts"`.
- `rows_per_shard` (int): Default rows per Arrow shard for stages that do not override `rows_per_shard`.
- `manifest_filename` (str): Dataset manifest filename (lists stage output paths and sequence length).
- `stats_filename` (str): Stats filename (document/sequence/token counters).

Example:
```yaml
outputs:
  processed_root: "dataset_artifacts"
  rows_per_shard: 32768
  manifest_filename: "datasets_manifest.json"
  stats_filename: "dataset_stats.json"
```

### scheduling
Controls batching and reproducibility during preprocessing.
- `seed` (int): Base RNG seed for window sampling and shuffling.
- `write_batch_size` (int): Number of documents batched before tokenization.
- `dry_run_preview_rows` (int): When `--dry-run` is set, stop after this many emitted sequences.

Example:
```yaml
scheduling:
  seed: 123
  write_batch_size: 256
  dry_run_preview_rows: 200
```

### stages
Map of stage name to stage configuration. Each stage produces a separate output folder under `outputs.processed_root`.

---

## Stage configuration
Each entry under `stages:` becomes one dataset stage.

Fields:
- `description` (str): Human-readable description for logging and manifest.
- `output_dir` (str): Folder name for the stage (defaults to stage name).
- `sequence_length` (int): Target sequence length (`S`) for emitted rows.
- `target_tokens` (int, optional): Stop once this many tokens have been emitted.
- `target_sequences` (int, optional): Stop once this many sequences have been emitted.
- `min_tokens` (int): Drop documents shorter than this many tokens.
- `pack_sequences` (bool): If true, pack multiple short docs into fixed-length rows.
- `add_eos` (bool): Append EOS when packing (if EOS exists in tokenizer).
- `emit_final_partial` (bool): If true and packing, emit a final partial chunk at end.
- `long_document_strategy` (str): `"random_window"` (default) or `"sequential"`.
- `sequential_window_stride` (int, optional): Stride for sequential windows (defaults to `sequence_length`).
- `max_windows_per_document` (int, optional): Cap windows per long document.
- `random_windows_per_document` (int): Number of random windows when `random_window` is used.
- `drop_remainder_windows` (bool): If true, drop trailing windows shorter than `sequence_length`.
- `normalization` (dict): Text normalization options (see below).
- `deduplicate` (bool): If true, de-dup by hash.
- `dedup_hash_bits` (int, optional): Hash bit width for de-dup (reduces memory).
- `dedup_max_keys` (int, optional): Max hash keys to keep (sliding window).
- `rows_per_shard` (int, optional): Override `outputs.rows_per_shard` for this stage.
- `max_documents` (int, optional): Per-stage document cap (for quick tests).
- `seed_offset` (int, optional): Offset added to `scheduling.seed` for this stage.
- `sources` (list): One or more source definitions (see below).

Normalization options (under `normalization`):
- `nfkc` (bool): Apply Unicode NFKC normalization.
- `strip_markup` (bool): Strip MediaWiki-style markup (if enabled).
- `collapse_whitespace` (bool): Collapse all whitespace to single spaces.

Example:
```yaml
stages:
  chat_sft:
    description: "Chat SFT stage"
    output_dir: "chat_sft"
    sequence_length: 2048
    target_sequences: 100000
    min_tokens: 32
    pack_sequences: false
    add_eos: false
    long_document_strategy: random_window
    random_windows_per_document: 1
    normalization:
      nfkc: true
      collapse_whitespace: true
    deduplicate: false
    sources:
      - type: huggingface
        dataset_name: "HuggingFaceH4/ultrachat_200k"
        split: "train_sft"
```

---

## Source configuration
Each stage can have multiple sources. Supported `type` values:
- `huggingface` / `hf`: HF dataset via `datasets.load_dataset`.
- `json`, `jsonl`, `json_dir`: JSON/JSONL files from disk.

### Common source fields
- `type` (str): Source type.
- `max_documents` (int, optional): Stop after N documents from this source.

### HuggingFace source fields
- `dataset_name` (str): HF repo id (e.g., `"HuggingFaceH4/ultrachat_200k"`).
- `dataset_config` (str, optional): HF config/name if needed.
- `split` (str): Split name (`train`, `train_sft`, etc.).
- `streaming` (bool): Stream from HF instead of downloading the full dataset.
- `shuffle_streaming` (bool): Shuffle the streaming dataset.
- `shuffle_buffer_size` (int): Buffer size for streaming shuffle.
- `shuffle_seed` (int, optional): RNG seed for streaming shuffle.
- `data_files` (any, optional): HF `data_files` for local or custom splits.

### JSON/JSONL source fields
- `json_root` (str): Directory containing JSON/JSONL files.
- `file_glob` (str): Glob pattern within `json_root` (default `"**/*.json*"`).

### Text extraction fields
If chat mode is not detected, text is pulled in this order:
1) `text_template` (format string using row keys)
2) `join_fields` (list of fields concatenated by `join_separator`)
3) `text_field` / `text_fields`
4) default keys: `text`, `content`, `body`, `article`, `story`, `completion`

Fields:
- `text_template` (str, optional): Python format string. Example: `"Q: {question}\nA: {answer}"`.
- `join_fields` (list[str], optional): Fields to join.
- `join_separator` (str): Separator for `join_fields` (default `" \n"`).
- `text_field` (str, optional): Preferred single field.
- `text_fields` (list[str], optional): Additional fallback fields.

Example:
```yaml
- type: json
  json_root: "my_json_dataset"
  file_glob: "*.jsonl"
  join_fields: ["title", "body"]
  join_separator: "\n\n"
```

---

## Chat masking fields (assistant-only loss)
If a row contains `chat_messages_field` and it is a list of `{role, content}` dicts, the pipeline builds a chat text string and a per-token `loss_mask` that is 1 only for assistant roles. This is stored as `loss_mask` in the Arrow shards and combined with the length mask in the loader.

Fields:
- `chat_messages_field` (str): Field containing the list of messages (default `"messages"`).
- `chat_role_field` (str): Role key in each message (default `"role"`).
- `chat_content_field` (str): Content key (default `"content"`).
- `chat_assistant_roles` (list[str]): Roles counted as assistant (default `["assistant"]`).
- `chat_role_prefix` (str): Prefix template applied per message (default `"### {role}\n"`).
- `chat_turn_suffix` (str): Suffix appended after each message (default `"\n"`).

Notes:
- Normalization applies to message content only (prefix/suffix are preserved so spans stay aligned).
- If chat fields are missing or invalid, the pipeline falls back to plain text extraction.

Example:
```yaml
- type: huggingface
  dataset_name: "HuggingFaceH4/ultrachat_200k"
  split: "train_sft"
  chat_messages_field: "messages"
  chat_role_field: "role"
  chat_content_field: "content"
  chat_assistant_roles: ["assistant"]
  chat_role_prefix: "### {role}\n"
  chat_turn_suffix: "\n"
```

---

## Tips for choosing lengths and targets
- `sequence_length`: match your training context length for the stage.
- `pack_sequences: true` for short text corpora; `false` for chat or long docs you want to keep intact.
- `long_document_strategy`:
  - `random_window`: good for web corpora; set `random_windows_per_document` > 1 for more coverage.
  - `sequential`: good for books/long docs; set `sequential_window_stride` to `sequence_length`.
- `target_tokens` vs `target_sequences`:
  - Use `target_tokens` when you care about total token budget.
  - Use `target_sequences` when you want precise row counts.

---

## Example full stage (chat SFT)
```yaml
stages:
  ultrachat_sft:
    description: "UltraChat SFT"
    output_dir: "ultrachat_sft"
    sequence_length: 2048
    target_sequences: 50000
    min_tokens: 32
    pack_sequences: false
    add_eos: false
    normalization:
      nfkc: true
      collapse_whitespace: true
    sources:
      - type: huggingface
        dataset_name: "HuggingFaceH4/ultrachat_200k"
        split: "train_sft"
        streaming: true
        shuffle_streaming: true
        shuffle_buffer_size: 10000
        chat_messages_field: "messages"
        chat_role_field: "role"
        chat_content_field: "content"
        chat_assistant_roles: ["assistant"]
        chat_role_prefix: "### {role}\n"
        chat_turn_suffix: "\n"
```

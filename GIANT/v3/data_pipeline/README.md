# v3 data pipeline

This folder turns text or chat sources into packed Arrow shards used by [../model/Run_training.py](../model/Run_training.py).

The pipeline has two main entrypoints:

- [train_tokenizer.py](train_tokenizer.py) - train and validate a custom HuggingFace tokenizer
- [build_corpus.py](build_corpus.py) - tokenize text/chat sources and write Arrow shards + manifests

## Config examples

- English chat tokenizer: [../Configs/Tokenizer/giant_chat_bpe24k.yml](../Configs/Tokenizer/giant_chat_bpe24k.yml)
- BG+EN chat tokenizer: [../Configs/Tokenizer/giant_chat_bg_en_bpe32k.yml](../Configs/Tokenizer/giant_chat_bg_en_bpe32k.yml)
- English 4B chat pretrain corpus: [../Configs/Data/giant_chat_pretraining_4b.yml](../Configs/Data/giant_chat_pretraining_4b.yml)
- BG+EN 900M chat pretrain corpus: [../Configs/Data/giant_chat_pretraining_bg_en_900m_bpe32k.yml](../Configs/Data/giant_chat_pretraining_bg_en_900m_bpe32k.yml)
- SmolTalk BG+EN SFT corpus: [../Configs/Data/giant_chat_sft_bg_en_smoltalk_bpe32k.yml](../Configs/Data/giant_chat_sft_bg_en_smoltalk_bpe32k.yml)

## What the data config controls

- `tokenizer.custom_path` - tokenizer directory to use
- `outputs.processed_root` - where Arrow shards and manifests are written
- `stages.*.sequence_length` - fixed training context length
- `stages.*.target_tokens` - stop after approximately this many packed tokens
- `stages.*.pack_sequences` - pack short docs into fixed-size rows
- `stages.*.long_document_strategy` - random window or sequential windows for long docs
- `stages.*.source_mix_mode: weighted_random` - interleave sources instead of consuming them sequentially
- `sources.*.sampling_weight` - source probability when using weighted interleaving
- chat fields like `chat_messages_field`, `chat_role_prefix`, and `chat_assistant_roles` - create assistant-only SFT masks
- `s3_upload.enabled` - upload finished stage folders with `s5cmd sync --size-only`
- `s3_upload.destination_root` - S3 prefix that receives one folder per stage
- `s3_upload.env_file` - optional secrets file to `source`; leave null to use current env vars

## Notes

- Re-running `build_corpus.py` deletes the stage output directory first.
- The Arrow rows are fixed-size lists, which is what the v3 dataloader wants.
- For heavy corpora, build once locally or on a pod, sync to S3, and train from prepared shards.
- HF streaming is useful for corpus construction, not for the training loop.
- JSON sources can point at a directory and a glob, for example the curated booster or translated SmolTalk folders.
- S3 upload runs in a background thread after each stage is written, then joins before the process exits.

## Old task note

[TASK.md](TASK.md) is an old review note about design risks. Keep it as engineering context, not as the main usage guide.

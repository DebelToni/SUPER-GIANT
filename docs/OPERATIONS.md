# SUPER-GIANT operations

This is the public version of the GPU workflow: agents can automate it, but every step is normal repo code.

## Local commands

Install editable package, then use the `sg` facade:

```bash
pip install -e .
sg configs --status recommended
sg tokenizer train --config GIANT/v3/Configs/Tokenizer/giant_chat_bg_en_bpe32k.yml
sg data build --config GIANT/v3/Configs/Data/giant_chat_pretraining_bg_en_900m_bpe32k.yml
sg train --config GIANT/v3/Configs/Training/1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml
```

Direct scripts still work; `sg` is only a cleaner front door.

## Disposable GPU workflow

1. Prepare startup files:

```bash
python3 CICD/tools/gpu_job.py prepare \
  --name bg-en-pretrain \
  --config GIANT/v3/Configs/Training/1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml \
  --sync-dir GIANT/GIANT-Chat/ \
  --upload
```

2. Start a RunPod GPU:

```bash
sg gpu create --gpu auto --name bg-en-pretrain --wait
```

3. Sync local uncommitted tool/code changes to an already running GPU:

```bash
python3 CICD/tools/sync-gpu.py --dry-run --path GIANT/v3
python3 CICD/tools/sync-gpu.py --host root@gpu-box --path GIANT/v3
```

4. Monitor through Tailscale/SSH:

```bash
ssh root@gpu-box 'tmux ls'
ssh root@gpu-box 'tail -f /proj/giant-data/GIANT/job_logs/bg-en-pretrain.log'
```

5. Stop the pod after artifacts are uploaded:

```bash
sg gpu list
sg gpu stop POD_ID
```

## S3 startup files

The Docker entrypoint reads these from `s3://giant-data/`:

- `sync_dirs.txt` - prefixes to sync into `/proj/giant-data` before running.
- `entrypoint.sh` - optional command launched inside the container.

`CICD/tools/gpu_job.py prepare --upload` writes both.

## Artifact rules

- Keep generated data/checkpoints under `/proj/giant-data`.
- Keep the repo at `/proj/SUPER-GIANT`.
- Use [`docs/ARTIFACTS.md`](ARTIFACTS.md) for the local↔S3 mapping.
- Use `--init_checkpoint` for stage transitions and `--resume` only for interrupted same-stage runs.

## Config discovery

```bash
sg configs
sg configs --kind training
sg configs --status recommended
```

The source file is [`GIANT/v3/Configs/registry.yml`](../GIANT/v3/Configs/registry.yml).

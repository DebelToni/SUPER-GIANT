# SUPER-GIANT ops tools

Small command-line helpers for the disposable GPU workflow, S3 artifact movement, and remote code sync.

## Tools

| Tool | Purpose |
| --- | --- |
| `runpod-gpu.sh` | Create/list/stop/remove RunPod GPU pods from the repo, using the existing GIANT template. |
| `gpu_job.py` | Prepare `sync_dirs.txt` + `entrypoint.sh` for the next pod startup job. |
| `sync-gpu.py` | Reset a reachable GPU repo to the local commit and apply the current unstaged diff. |
| `wait-new-gpu.sh` | Wait until a new Tailscale GPU host appears or `root@gpu-box` is reachable. |
| `s3.py` | Interactive/cold-path wrapper over `s5cmd`. |

## Typical GPU workflow

Prepare startup files locally under `/proj/giant-data` and upload them to `s3://giant-data`:

```bash
python3 CICD/tools/gpu_job.py prepare \
  --name bg-en-pretrain \
  --config GIANT/v3/Configs/Training/1_pretraining_100m_bg_en_ctx256_32k_1p8b.yml \
  --sync-dir GIANT/GIANT-Chat/ \
  --upload
```

Start a pod:

```bash
CICD/tools/runpod-gpu.sh create --gpu auto --name bg-en-pretrain --wait
```

Sync local work-in-progress code to an already reachable GPU box:

```bash
python3 CICD/tools/sync-gpu.py --dry-run --path CICD/tools
python3 CICD/tools/sync-gpu.py --host root@gpu-box --path CICD/tools
```

Pods are disposable. Durable artifacts should live under `/proj/giant-data`, which maps to `s3://giant-data/`.

## `runpod-gpu.sh`

Requirements:

- `runpodctl`
- `curl`
- `python3`
- `RUNPOD_API_KEY` or `~/.runpod/config.toml`

Commands:

```bash
CICD/tools/runpod-gpu.sh create --gpu auto --name my-job --wait
CICD/tools/runpod-gpu.sh create --gpu "NVIDIA RTX A6000" --name my-job --wait
CICD/tools/runpod-gpu.sh create --gpu "NVIDIA RTX A40" --spot --id-only
CICD/tools/runpod-gpu.sh list
CICD/tools/runpod-gpu.sh stop POD_ID
CICD/tools/runpod-gpu.sh remove POD_ID
```

`--gpu auto` tries A6000, A5000, A4500, A4000, A40, then RTX 5090.

## `gpu_job.py prepare`

Writes two files consumed by the Docker entrypoint:

- `/proj/giant-data/sync_dirs.txt`
- `/proj/giant-data/entrypoint.sh`

Dry-run locally without touching `/proj`:

```bash
python3 CICD/tools/gpu_job.py prepare \
  --name smoke \
  --config GIANT/v3/Configs/Training/dev_100m_hq.yml \
  --data-root /tmp/giant-data \
  --remote-data-root /proj/giant-data \
  --dry-run
```

Useful options:

- `--kind train|data|tokenizer` chooses the standard GIANT/v3 command built from `--config`.
- `--command "..."` writes a custom startup command instead.
- `--sync-dir PREFIX/` repeats S3/data prefixes to prefetch on startup.
- `--data-root` controls where startup files are written locally.
- `--remote-data-root` controls the path used inside the pod; default is the same as `--data-root`.
- `--upload` copies startup files to `s3://giant-data/sync_dirs.txt` and `s3://giant-data/entrypoint.sh`.
- `--env-file ~/.env-R2` sources credentials before upload.

## `sync-gpu.py`

Use this for quick tests on an already running Tailscale GPU. It treats the remote repo as ephemeral:

1. `git fetch --all --prune`
2. checkout local `HEAD`
3. `git reset --hard && git clean -fd`
4. apply the local unstaged patch, including untracked files

It intentionally does not sync staged changes. Use repeatable `--path PATH` to test or send only a subset of the local diff.

## `s3.py`

S3 shell wrapper around `s5cmd` with a friendly interactive mode, tab completion, and a cold-path CLI for one-off commands.

Requirements:

- `python3`
- `s5cmd` in your PATH
- AWS credentials available through env vars, `AWS_PROFILE`, `~/.aws`, etc.
- `S3_ENDPOINT_URL` for non-AWS S3/R2/MinIO/B2 when needed

Quick start:

```bash
python3 CICD/tools/s3.py
python3 CICD/tools/s3.py ls
python3 CICD/tools/s3.py du GIANT/
```

Alias tip:

```bash
alias s3="python3 /absolute/path/to/CICD/tools/s3.py --bucket giant-data"
```

Common commands inside the shell:

```text
ls [path]              list S3/local paths
ls -R [path]           tree view
cd [path]              change prefix/cwd
cp src dst             copy between local/S3
sync src dst           size-only sync by default
cat/head/tail path     inspect S3 objects
local                  switch to local mode
s3                     switch to S3 mode
```

Examples:

```bash
s3://giant-data/> cp ./local_dir/ GIANT/tmp/
s3://giant-data/> sync GIANT/GIANT-Chat/ ./downloads/
s3://giant-data/> cat GIANT/path/to/log.json | jq
```

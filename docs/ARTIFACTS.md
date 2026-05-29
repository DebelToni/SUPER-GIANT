# SUPER-GIANT artifact layout

The repo treats compute as disposable and artifacts as durable.

## Main contract

```text
/proj/SUPER-GIANT              git checkout inside GPU containers
/proj/giant-data               durable local data root
s3://giant-data/               durable remote artifact root
/proj/giant-data/GIANT/foo     <=> s3://giant-data/GIANT/foo
```

GIANT v3 stores this mapping in [`GIANT/v3/Global_Config.yml`](../GIANT/v3/Global_Config.yml):

```yaml
artifacts:
  local_root: "/proj/giant-data"
  s3_root: "s3://giant-data"
  env_file: "~/.env-R2"
  tool: "s5cmd"
  size_only: true
  upload:
    enabled: false
    tokenizers: false
    datasets: false
    checkpoints: false
    logs: false
```

Default upload is off so local experiments do not unexpectedly write to S3.

## What writes where

| Artifact | Local path pattern | S3 path pattern |
| --- | --- | --- |
| HF cache | `/proj/giant-data/hf_cache/` | usually not uploaded manually |
| Tokenizers | `/proj/giant-data/GIANT/.../tokenizers/name/` | `s3://giant-data/GIANT/.../tokenizers/name/` |
| Arrow datasets | `/proj/giant-data/GIANT/.../data/name/` or `dataset_artifacts/name/` | matching S3 prefix |
| Checkpoints | `/proj/giant-data/GIANT/.../training/run/` | matching S3 prefix |
| Dataloader state | checkpoint `training_states/dataloader_state/` | uploaded with checkpoint root |
| Logs | run-specific logs/checkpoint upload logs | optional/future global upload |

## Enabling automatic upload

For a RunPod/training run:

```yaml
artifacts:
  upload:
    enabled: true
    tokenizers: true
    datasets: true
    checkpoints: true
```

Per-data-config `s3_upload` still overrides the global dataset default when present.

## Manual inspection

```bash
python3 CICD/tools/s3.py ls GIANT/GIANT-Chat/
python3 CICD/tools/s3.py du GIANT/GIANT-Chat/
```

Use `s5cmd sync --size-only` for R2/S3-compatible buckets because timestamp semantics can be unreliable.

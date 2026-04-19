---
name: S3-bucket-ops
description: Use s5cmd for S3 listing, move, copy, and sync tasks.
---

# s5cmd Ops

Use this skill to interact with S3 via `s5cmd` for listing, copying, moving, and syncing artifacts.

## Core rules

* The `/proj/giant-data` global data dir should be a local copy of the `s3://giant-data/` bucket. Paths should correspond between local and S3.
* use only s5cmd for S3 operations.
* Always quote S3 URIs with double quotes.
* `s5cmd mv` and `s5cmd cp` require wildcards when moving directories. Use `"s3://giant-data/path/*"`.
* Avoid destructive deletes unless explicitly requested.
* Avoid using `s5cmd sync` to or from the S3 bucket unless necessary. `cp` commands have less side effects. If you have to for whatever reason to use `sync`, use the `--size-only` flag because the user's S3 bucket is actually R2 which behaves badly with timestamps.
* The deafult and only bucket is `"s3://giant-data/"`. 

## Common commands

Populat credentials:
```bash
source ~/.env-R2 
# NEVER CAT OR READ THIS FILE!
```

List buckets or prefixes:
```bash
s5cmd ls
s5cmd ls "s3://giant-data/prefix/"
```

Move a folder (use wildcard):
```bash
s5cmd mv "s3://giant-data/src/*" "s3://bucket/dst/"
```

Copy a single file:
```bash
s5cmd cp "s3://giant-data/path/file" "/local/path/"
```

Sync a folder to local:
```bash
s5cmd sync --size-only "s3://giant-data/prefix/" "/local/dir/"
```

## Notes

* Keep local destinations under the global data root when syncing training artifacts.
* When unsure of a path, list parent prefixes first.

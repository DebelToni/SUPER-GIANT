# s3.py

S3 shell wrapper around `s5cmd` with a friendly interactive mode, tab completion,
and a cold-path CLI for one-off commands.

<br>

If you are interested for what I use this tool so much, [check out the main README of the repo!](../README.md)

Consider leaving a star ⭐️ if you find it useful :)

<br>

## Requirements

- `python3` <- no libs!
- `s5cmd` in your PATH
- AWS credentials available via the standard AWS credential chain
  (env vars, AWS_PROFILE, ~/.aws, etc.)
- `S3_ENDPOINT_URL` when using non-AWS S3 (R2/MinIO/B2). Optional for AWS S3.

## Quick start

Launch the interactive shell:

```bash
python3 CICD/tools/s3.py
```

Run a single command (cold path):

```bash
python3 CICD/tools/s3.py ls
python3 CICD/tools/s3.py ls TiDAR/
python3 CICD/tools/s3.py du TiDAR/
```

## Alias tip

Create a handy alias:

```bash
alias s3="python3 /absolute/path/to/CICD/tools/s3.py"
```

You can also bake in your default bucket:

```bash
alias s3="python3 /absolute/path/to/CICD/tools/s3.py --bucket giant-data"
```

## Default bucket

By default, the bucket comes from `S3_BUCKET`. If it's not set, it falls back to
`s3://giant-data`. 

Change it to your default bucket name!

If you want a different default on first use, pass `--bucket`:

```bash
python3 CICD/tools/s3.py --bucket my-bucket
```

This sets the bucket for that session. If you want it permanently, put it in an alias
as shown above.

## Shell modes

The shell has two modes:

- `s3` mode (default): paths are S3-relative to the current prefix
- `local` mode: paths are local filesystem-relative to your local cwd

Commands to switch:

```bash
s3     # switch back to S3 mode
local  # switch to local filesystem mode
```

The prompt shows the current mode:

```
s3://giant-data/TiDAR>
local:/Users/me/projects>
```

## Commands

- `ls [path]` (supports `--depth N`, `-R`/`--tree`)
- `du [path]`
- `rm [-r] path`
- `mkdir [-p] path`
- `cat path`
- `head [-n N] path`
- `tail [-n N] path`
- `cp src dst`
- `mv src dst`
- `sync [--true-sync] src dst`
- `cd [path]`
- `pwd`
- `local`, `s3`
- `clear`
- `help`, `exit`, `quit`

Notes:

- `sync` defaults to `--size-only` unless you pass `--true-sync`.
- Use a trailing `/` on S3 prefixes for folder-like behavior.
- `mv` does not support S3 -> local (use `cp` then `rm`).

## Examples

Copy local folder up to S3:

```bash
s3://giant-data/> cp ./local_dir/ TiDAR/
```

Copy from S3 to local:

```bash
s3://giant-data/> cp TiDAR/ ./downloads/
```

Sync S3 to local (size-only):

```bash
s3://giant-data/> sync TiDAR/ ./sync_downloads/
```

Switch to local mode to use relative local paths:

```bash
s3://giant-data/> local
local:/Users/me/project> cp ../data/ s3://giant-data/TiDAR/
local:/Users/me/project> s3
```

Pipe the output into any cli:

```bash
~/Documents❯ s3 cat /path/to/logs.json | jq
```
```json
{
  "tinystories_300m_512": {
    "path": "/datasets/tinystories/config",
    "sequence_length": 512,
    "target_tokens": 300000000
  }
}
```

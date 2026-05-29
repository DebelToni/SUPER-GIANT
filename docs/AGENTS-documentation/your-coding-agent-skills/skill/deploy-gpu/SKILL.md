---
name: deploy-gpu 
description: Create/start/stop/delete a RunPod GPU Pod using ONLY GPU type `XXX` + the existing template named `GIANT-container`. Includes Spot (Secure Cloud) via interruptible Pods.
---

# RunPod GIANT Pod

Use the helper script instead of raw `runpodctl`/`curl` commands:

```bash
CICD/tools/runpod-gpu.sh create --gpu auto --name my-job --wait
```

The script always uses the `GIANT-container` template id `bg2jwnb3zk`, keeps
everything on Secure Cloud, handles the noisy CLI/REST fallback internally, and
prints a concise summary instead of raw RunPod payloads.

The repo also bundles `CICD/tools/wait-new-gpu.sh`.
**Read `gpu-remote-exec` after the pod is ready.**

## Core rules

* Secure Cloud only.
* Use only the existing template `bg2jwnb3zk`.
* Prefer the helper script over manual commands.
* Use exact GPU if the user asks for one, otherwise use `--gpu auto`.
* `--spot` means interruptible Secure Cloud via REST.

## Recommended commands

Create with fallback order:

```bash
CICD/tools/runpod-gpu.sh create --gpu auto --name my-job --wait
```

Create exact GPU:

```bash
CICD/tools/runpod-gpu.sh create --gpu "NVIDIA RTX A6000" --name my-job --wait
```

Create spot:

```bash
CICD/tools/runpod-gpu.sh create --gpu "NVIDIA RTX A40" --name my-job --spot --wait
```

List / stop / remove:

```bash
CICD/tools/runpod-gpu.sh list
CICD/tools/runpod-gpu.sh stop "$RUNPOD_POD_ID"
CICD/tools/runpod-gpu.sh remove "$RUNPOD_POD_ID"
```

If you only need the id:

```bash
CICD/tools/runpod-gpu.sh create --gpu auto --name my-job --id-only
```

## Auto GPU order

`--gpu auto` tries:

```text
NVIDIA RTX A6000 -> NVIDIA RTX A5000 -> NVIDIA RTX A4500 -> NVIDIA RTX A4000 -> NVIDIA A40 -> NVIDIA GeForce RTX 5090
```

This matches the user's preferred fallback order.

# How to prepare before the pod starts

1. Make sure all the needed data is in the S3 bucket
2. Create a sync_dirs.txt and entrypoint.sh which will be read on Pod startup, synced and executed. They live locally on the mac at:
```
/proj/giant-data/sync_dirs.txt
/proj/giant-data/entrypoint.sh
```
You will edit them and overwrite them for the current task. Preffer tmux for the entrypoint for the user to be easy to monitor.

3. Sync them to the S3 
```bash
s5cmd sync sync_dirs.txt "s3://giant-data/sync_dirs.txt" && s5cmd sync entrypoint.sh "s3://giant-data/entrypoint.sh"
```

Now whatever is in the S3 and stated in sync_dirs.txt will be on the pod on startup and whatever it was in entrypoint.sh will be ran so the gpu can start working right away if needed.

This workflow is for training jobs, for quick tests it is not needed to have a entrypoing.sh


# What to do when the Pod starts

After the pod is created and startup sync finishes, find its Tailscale host with:

```bash
tailscale status | rg gpu
```

You should run commands to the remote gpu only like mentioned in your `remote-gpu` skill.

# What to do when the Pod stops / you are terminating it

When you stop or terminate the pod, push the generated artifacts to S3 first:

```bash
s5cmd sync /proj/giant-data/TiDAR/checkpoints/ "s3://giant-data/TiDAR/checkpoints/"
```

Replace that path with the current job's checkpoint/output directory.

Pods are ephemeral; upload artifacts before removal.

## Notes
* The helper script reads `RUNPOD_API_KEY` from the environment or `~/.runpod/config.toml`.
* `--wait` uses the repo-level helper at `CICD/tools/wait-new-gpu.sh`.
* Use spot only if the user explicitly asked for it.

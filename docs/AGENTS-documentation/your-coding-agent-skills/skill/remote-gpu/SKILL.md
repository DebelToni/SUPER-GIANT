---
name: gpu-remote-exec
description: Run commands or JAX scripts on the remote CUDA GPU box via ssh root@gpu-box when execution needs a GPU/CUDA backend or the user asks to run on the GPU box/remote container.
---

# GPU Remote Exec

Use this skill to execute commands on the CUDA-capable GPU box at `root@gpu-box`.

## Core rules

* **Edit code only locally. Never edit files on the remote GPU box.**
* After **any** local code change (even a single line), **re-sync the local diff to the remote** and then run remotely.
* If the remote machine needs data files to train on copy them with `scp` from the local data root to the remote data root if they are really small. For larger files or datasets use your skill about S3-bucket-ops to transfer data with fast backends to my ultra fast buckets.

* Never deafult to runpod's cli options for accessing pods. Use only what is described in this skill.

## Workflow

0. Decide if the task requires GPU/CUDA or the user explicitly asks for the GPU box.
    
1. Check which remote gpu answers and use it for until its down , test with:
```bash
ssh root@gpu-box "echo ok"
ssh root@gpu-box-1 "echo ok"
ssh root@gpu-box-2 "echo ok"
...
```
Or run taislcale and see if anything responds with `-` which means its up:
```bash
tailscale status | rg gpu
```

2. Ensure the remote repo exists and is on the correct branch (run once per session, or if branch changes):

```bash
ssh root@gpu-box 'set -e; cd /proj/SUPER-GIANT; git fetch --all --prune'
ssh root@gpu-box 'set -e; cd /proj/SUPER-GIANT; git status --porcelain; git branch --show-current'
```

3. **After every local edit**, force the remote into a clean state and apply your **local** changes via a binary diff.

**A) Include new files in the diff (recommended):**

```bash
git add -N .
git diff --binary | ssh root@gpu-box 'set -e; cd /proj/SUPER-GIANT; git reset --hard; git clean -fd; git apply --index'
```

**B) If you only changed already-tracked files:**

```bash
git diff --binary | ssh root@gpu-box 'set -e; cd /proj/SUPER-GIANT; git reset --hard; git clean -fd; git apply --index'
```

4. Run the desired command or JAX script remotely:

```bash
ssh root@gpu-box 'set -e; cd /proj/SUPER-GIANT; <RUN_COMMAND_HERE>'
```
> Note: python with installed JAX, FLAX etc. is available at "/opt/venv/bin/python script.py". Use it excplicitly because the venv over ssh is not auto-activated.

5. If results require code changes:

   * Make the change locally.
   * Repeat **Step 3** (sync diff).
   * Repeat **Step 4** (run remotely).

6. Clean up the remote box (optional, but good hygiene at the end of a session):

```bash
ssh root@gpu-box 'set -e; cd /proj/SUPER-GIANT; git reset --hard; git clean -fd'
```

## Useful tips

This skill is generally used with with the S3-bucket-ops skill. Note that on the remote gpu the env variables for S3 ops are sourced

## Notes

* The remote is treated as **ephemeral**: it is always reset/cleaned before applying the local patch.
* If `git apply` fails, it usually means the remote isn’t on a compatible base revision/branch. Fix by aligning the remote branch/checkout (and re-fetch) and then re-run the sync command in Step 3.
* Always use the diff-send method above to run local work-in-progress remotely; do not try to “just edit remotely and fix it there.”
* When parsing logs that include tqdm output, normalize carriage returns (`\r`) to newlines so `[eval]` lines can be found reliably.

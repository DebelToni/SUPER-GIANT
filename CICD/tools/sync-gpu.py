#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess
import sys


def run_text(cmd, cwd=None):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def run_bytes(cmd, cwd=None, allow_exit_codes=(0,)):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    if result.returncode not in allow_exit_codes:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}")
    return result.stdout


def git_text(args, cwd):
    return run_text(["git", *args], cwd=cwd)


def git_bytes(args, cwd, allow_exit_codes=(0,)):
    return run_bytes(["git", *args], cwd=cwd, allow_exit_codes=allow_exit_codes)


def collect_unstaged_patch(repo_root):
    parts = []
    parts.append(git_bytes(["diff", "--binary"], repo_root, allow_exit_codes=(0, 1)))
    untracked = git_bytes(
        ["ls-files", "--others", "--exclude-standard", "-z"], repo_root
    )
    if untracked:
        for raw_path in untracked.split(b"\0"):
            if not raw_path:
                continue
            path = os.fsdecode(raw_path)
            diff = git_bytes(
                ["diff", "--binary", "--no-index", "--", "/dev/null", path],
                repo_root,
                allow_exit_codes=(0, 1),
            )
            parts.append(diff)
    return b"".join(parts)


def has_staged_changes(repo_root):
    staged = git_text(["diff", "--cached", "--name-only"], repo_root)
    return bool(staged)


def sync_remote(host, repo_path, commit, patch):
    prep_cmd = (
        "set -e; "
        f"cd {repo_path}; "
        "git fetch --all --prune; "
        f"git checkout {commit}; "
        "git reset --hard; "
        "git clean -fd"
    )
    subprocess.run(["ssh", host, prep_cmd], check=True)

    if not patch:
        return

    apply_cmd = f"set -e; cd {repo_path}; git apply --whitespace=nowarn"
    subprocess.run(["ssh", host, apply_cmd], input=patch, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sync local unstaged changes to the GPU box, "
            "after checking out the same commit."
        )
    )
    parser.add_argument(
        "--host",
        default="root@gpu-box",
        help="SSH host for the GPU box (default: root@gpu-box)",
    )
    parser.add_argument(
        "--repo",
        default="/proj/SUPER-GIANT",
        help="Repo path on the remote host (default: /proj/SUPER-GIANT)",
    )
    args = parser.parse_args()

    repo_root = Path(git_text(["rev-parse", "--show-toplevel"], None))
    commit = git_text(["rev-parse", "HEAD"], repo_root)
    patch = collect_unstaged_patch(repo_root)

    if has_staged_changes(repo_root):
        sys.stderr.write(
            "Warning: staged changes detected; only unstaged changes are synced.\n"
        )

    sync_remote(args.host, args.repo, commit, patch)

    if patch:
        print("Synced unstaged changes to remote.")
    else:
        print("No unstaged changes found; remote checked out to matching commit.")


if __name__ == "__main__":
    main()

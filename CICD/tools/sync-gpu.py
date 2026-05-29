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


def path_args(paths):
    return ["--", *paths] if paths else []


def collect_unstaged_patch(repo_root, paths=None):
    paths = paths or []
    parts = []
    parts.append(
        git_bytes(["diff", "--binary", *path_args(paths)], repo_root, allow_exit_codes=(0, 1))
    )
    untracked = git_bytes(
        ["ls-files", "--others", "--exclude-standard", "-z", *path_args(paths)], repo_root
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
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Limit the patch to a pathspec. Repeatable. Defaults to the whole repo diff.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be synced without connecting to the GPU box.",
    )
    args = parser.parse_args()

    repo_root = Path(git_text(["rev-parse", "--show-toplevel"], None))
    commit = git_text(["rev-parse", "HEAD"], repo_root)

    staged = has_staged_changes(repo_root)
    if staged:
        sys.stderr.write(
            "Warning: staged changes detected; only unstaged changes are synced.\n"
        )

    if args.dry_run:
        modified = git_text(["diff", "--name-only", *path_args(args.path)], repo_root).splitlines()
        untracked_raw = git_bytes(["ls-files", "--others", "--exclude-standard", "-z", *path_args(args.path)], repo_root)
        untracked = [os.fsdecode(p) for p in untracked_raw.split(b"\0") if p]
        print(f"repo={repo_root}")
        print(f"commit={commit}")
        print(f"host={args.host}")
        print(f"remote_repo={args.repo}")
        print(f"paths={','.join(args.path) if args.path else '<all>'}")
        print(f"modified_files={len(modified)}")
        print(f"untracked_files={len(untracked)}")
        print(f"staged_changes={str(staged).lower()}")
        return

    patch = collect_unstaged_patch(repo_root, args.path)
    sync_remote(args.host, args.repo, commit, patch)

    if patch:
        print("Synced unstaged changes to remote.")
    else:
        print("No unstaged changes found; remote checked out to matching commit.")


if __name__ == "__main__":
    main()

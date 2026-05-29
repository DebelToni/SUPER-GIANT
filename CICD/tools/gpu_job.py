#!/usr/bin/env python3
"""Prepare disposable GPU jobs for the SUPER-GIANT RunPod/Tailscale workflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Iterable


DEFAULT_DATA_ROOT = os.environ.get("GIANT_DATA_ROOT", "/proj/giant-data")
DEFAULT_S3_ROOT = os.environ.get("S3_ROOT", "s3://giant-data")
DEFAULT_REPO_DIR = os.environ.get("GIANT_REPO_DIR", "/proj/SUPER-GIANT")
DEFAULT_PYTHON = os.environ.get("GIANT_REMOTE_PYTHON", "/opt/venv/bin/python")


def die(message: str) -> None:
    print(f"[gpu-job] {message}", file=sys.stderr)
    raise SystemExit(1)


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return value.strip("-._") or "giant-job"


def normalize_s3_root(value: str) -> str:
    value = value.rstrip("/")
    if not value.startswith("s3://"):
        die(f"--s3-root must start with s3://, got: {value}")
    return value


def quote_args(args: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def command_from_config(args: argparse.Namespace) -> str:
    if args.command:
        return args.command
    if not args.config:
        die("prepare requires --command or --config")

    if args.kind == "train":
        cmd = [
            args.python,
            "GIANT/v3/model/Run_training.py",
            "--config",
            args.config,
        ]
        if args.upload_on_checkpoint:
            cmd.append("--upload-on-checkpoint")
    elif args.kind == "data":
        cmd = [args.python, "GIANT/v3/data_pipeline/build_corpus.py", "--config", args.config]
    elif args.kind == "tokenizer":
        cmd = [args.python, "GIANT/v3/data_pipeline/train_tokenizer.py", "--config", args.config]
    else:
        raise AssertionError(args.kind)
    return f"PYTHONPATH=. {quote_args(cmd)}"


def render_entrypoint(
    *,
    name: str,
    session: str,
    repo_dir: str,
    data_root: str,
    log_dir: str,
    command: str,
) -> str:
    safe_name = sanitize_name(name)
    runner_path = f"{data_root}/.gpu_jobs/{safe_name}.sh"
    log_path = f"{data_root}/{log_dir.strip('/')}/{safe_name}.log"
    return f'''#!/usr/bin/env bash
set -euo pipefail

REPO_DIR={shlex.quote(repo_dir)}
DATA_ROOT={shlex.quote(data_root)}
SESSION={shlex.quote(session)}
RUNNER_FILE={shlex.quote(runner_path)}
LOG_FILE={shlex.quote(log_path)}

mkdir -p "$(dirname "$RUNNER_FILE")" "$(dirname "$LOG_FILE")"
cat > "$RUNNER_FILE" <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
LOG_FILE={shlex.quote(log_path)}
exec > >(tee -a "$LOG_FILE") 2>&1
cd {shlex.quote(repo_dir)}
export PYTHONPATH=.
printf '[gpu-job] started %s on %s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$(hostname)"
printf '[gpu-job] repo=%s data_root=%s log=%s\\n' "{repo_dir}" "{data_root}" "$LOG_FILE"
{command}
printf '[gpu-job] finished %s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')"
JOB
chmod +x "$RUNNER_FILE"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux kill-session -t "$SESSION"
  fi
  tmux new-session -d -s "$SESSION" "$RUNNER_FILE"
  printf '[gpu-job] launched tmux session %s; log=%s\\n' "$SESSION" "$LOG_FILE"
else
  printf '[gpu-job] tmux missing; running in foreground; log=%s\\n' "$LOG_FILE"
  "$RUNNER_FILE"
fi
'''


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def upload_file(path: Path, dest: str, env_file: str | None) -> None:
    cmd = f"s5cmd cp {shlex.quote(str(path))} {shlex.quote(dest)}"
    if env_file:
        cmd = f"source {shlex.quote(env_file)} && {cmd}"
    subprocess.run(["bash", "-lc", cmd], check=True)


def prepare_cmd(args: argparse.Namespace) -> None:
    name = sanitize_name(args.name)
    session = sanitize_name(args.session or name)
    data_root = Path(args.data_root).expanduser()
    remote_data_root = args.remote_data_root or str(data_root)
    s3_root = normalize_s3_root(args.s3_root)
    sync_dirs = [item.strip().lstrip("/") for item in args.sync_dir if item.strip()]
    command = command_from_config(args)
    log_dir = args.log_dir.strip().strip("/")

    sync_dirs_text = "\n".join(sync_dirs) + ("\n" if sync_dirs else "")
    entrypoint_text = render_entrypoint(
        name=name,
        session=session,
        repo_dir=args.repo_dir,
        data_root=remote_data_root,
        log_dir=log_dir,
        command=command,
    )

    sync_path = data_root / "sync_dirs.txt"
    entrypoint_path = data_root / "entrypoint.sh"

    if args.dry_run:
        print(f"[gpu-job] would write {sync_path}")
        print(sync_dirs_text or "<empty sync_dirs.txt: startup sync disabled>")
        print(f"[gpu-job] would write {entrypoint_path}")
        print(entrypoint_text)
        return

    write_text(sync_path, sync_dirs_text)
    write_text(entrypoint_path, entrypoint_text, executable=True)
    print(f"[gpu-job] wrote {sync_path}")
    print(f"[gpu-job] wrote {entrypoint_path}")

    if args.upload:
        upload_file(sync_path, f"{s3_root}/sync_dirs.txt", args.env_file)
        upload_file(entrypoint_path, f"{s3_root}/entrypoint.sh", args.env_file)
        print(f"[gpu-job] uploaded startup files to {s3_root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser(
        "prepare",
        help="write / upload sync_dirs.txt and entrypoint.sh for the next GPU pod",
    )
    prep.add_argument("--name", default="giant-job", help="job name used for logs and default tmux session")
    prep.add_argument("--session", default=None, help="tmux session name; defaults to sanitized --name")
    prep.add_argument("--config", help="GIANT/v3 config path used to build a standard command")
    prep.add_argument("--kind", choices=("train", "data", "tokenizer"), default="train")
    prep.add_argument("--command", help="custom command to run instead of deriving one from --config")
    prep.add_argument("--sync-dir", action="append", default=[], help="S3/data prefix to pre-sync; repeatable, e.g. GIANT/GIANT-Chat/")
    prep.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="local root where sync_dirs.txt and entrypoint.sh are written")
    prep.add_argument("--remote-data-root", default=None, help="data root baked into entrypoint.sh; defaults to --data-root")
    prep.add_argument("--s3-root", default=DEFAULT_S3_ROOT)
    prep.add_argument("--repo-dir", default=DEFAULT_REPO_DIR)
    prep.add_argument("--python", default=DEFAULT_PYTHON)
    prep.add_argument("--log-dir", default="GIANT/job_logs", help="log directory relative to the entrypoint data root")
    prep.add_argument("--upload-on-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    prep.add_argument("--upload", action="store_true", help="upload startup files to --s3-root with s5cmd")
    prep.add_argument("--env-file", help="optional env file to source before s5cmd upload")
    prep.add_argument("--dry-run", action="store_true")
    prep.set_defaults(func=prepare_cmd)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

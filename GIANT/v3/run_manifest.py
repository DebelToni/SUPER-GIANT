from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _git(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_s3_uri(local_path: str | Path, *, local_root: str | Path | None, s3_root: str | None) -> str | None:
    if not local_root or not s3_root:
        return None
    try:
        rel = Path(local_path).expanduser().resolve().relative_to(Path(local_root).expanduser().resolve())
    except Exception:
        return None
    return f"{str(s3_root).rstrip('/')}/{rel.as_posix()}"


def build_manifest(
    *,
    kind: str,
    config_path: str | Path | None,
    global_config_path: str | Path | None = None,
    outputs: list[str | Path] | None = None,
    s3_outputs: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outputs = outputs or []
    return {
        "kind": kind,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["branch", "--show-current"]),
        "git_dirty": bool(_git(["status", "--porcelain"])),
        "config_path": str(config_path) if config_path else None,
        "config_sha256": file_sha256(config_path),
        "global_config_path": str(global_config_path) if global_config_path else None,
        "global_config_sha256": file_sha256(global_config_path),
        "outputs": [str(Path(p)) for p in outputs],
        "s3_outputs": list(s3_outputs or []),
        "extra": extra or {},
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p

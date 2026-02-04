#!/usr/bin/env python3
import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
import readline
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional, Tuple

# SET YOUR FAVORITE DEFAULT BUCKET HERE
DEFAULT_BUCKET_ENV = "S3_BUCKET"
DEFAULT_BUCKET_FALLBACK = "giant-data"
DIR_MARKER = ".s3dir"
TAB_CACHE_TTL_SECONDS = 1.0
TAB_CACHE_MAX_ENTRIES = 512
HELP_HEADER = "Amazing, fast S3 shell wrapper on top of s5cmd (no Python deps)."
HELP_BODY = """Commands:
  ls [path]            list objects (use -R/--tree or --depth)
  du [path]            show disk usage in S3
  rm [-r] path          remove object or prefix (S3 only)
  mkdir [-p] path       create prefix marker (S3 only)
  cat path              print object contents (S3 only)
  head [-n N] path       first N lines (S3 only)
  tail [-n N] path       last N lines (S3 only)
  cp src dst            copy between local/S3
  mv src dst            move between local/S3 or S3/S3
  sync [--true-sync]     sync between local/S3
  cd [path]             change directory (local or S3 mode)
  pwd                   print current directory
  local                 switch to local filesystem mode
  s3                    switch to S3 mode
  clear                 clear the screen
  help                  show this help
  exit | quit | ctrl+c    exit the shell

Notes:
  You can pass any s5cmd flags to cp/mv/sync/du as needed.
"""
LS_FILE_RE = re.compile(
    r"^(?P<date>\d{4}/\d{2}/\d{2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<size>\d+)\s+(?P<key>.+)$"
)
LS_DIR_RE = re.compile(r"^\s*DIR\s+(?P<key>.+)$")
_GLOB = re.compile(r"[\*\?\[]")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OSC = re.compile(r"\x1b\].*?\x07")
_ESC = re.compile(r"\x1b.")


def die(message: str, code: int = 1) -> None:
    sys.stderr.write(message.rstrip() + "\n")
    raise SystemExit(code)


def safe_tty(s: str) -> str:
    s = _OSC.sub("", s)
    s = _ANSI.sub("", s)
    s = _ESC.sub("", s)
    s = _CTRL.sub("?", s)
    return s


def check_env() -> str:
    endpoint = (os.environ.get("S3_ENDPOINT_URL", "") or "").strip()
    endpoint = endpoint.rstrip("/")

    cred_hints: List[str] = []
    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        cred_hints.append("AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY")
    if os.environ.get("AWS_PROFILE"):
        cred_hints.append("AWS_PROFILE")
    aws_creds = os.path.expanduser("~/.aws/credentials")
    aws_cfg = os.path.expanduser("~/.aws/config")
    if os.path.exists(aws_creds) or os.path.exists(aws_cfg):
        cred_hints.append("~/.aws/{config,credentials}")
    if os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE"):
        cred_hints.append("AWS_WEB_IDENTITY_TOKEN_FILE")
    if os.environ.get("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI") or os.environ.get(
        "AWS_CONTAINER_CREDENTIALS_FULL_URI"
    ):
        cred_hints.append("container credentials")
    if not cred_hints:
        sys.stderr.write(
            "Note: no explicit AWS credential hints found in env or ~/.aws; "
            "relying on the default AWS credential provider chain.\n"
        )
    return endpoint


def normalize_bucket(value: str) -> str:
    bucket = value.strip()
    if bucket.startswith("s3://"):
        bucket = bucket[len("s3://") :]
    bucket = bucket.strip("/")
    if not bucket:
        die("Bucket name cannot be empty")
    return bucket


def default_bucket_from_env() -> Optional[str]:
    value = os.environ.get(DEFAULT_BUCKET_ENV)
    if value:
        return value
    return DEFAULT_BUCKET_FALLBACK


def build_s3_uri(bucket: str, key: str) -> str:
    key = key.lstrip("/")
    if key:
        return f"s3://{bucket}/{key}"
    return f"s3://{bucket}"


def s5cmd_base(endpoint: str) -> List[str]:
    base = ["s5cmd"]
    if endpoint:
        base += ["--endpoint-url", endpoint]
    return base


def run_s5cmd(args: List[str], endpoint: str, allow_missing: bool = False) -> str:
    cmd = s5cmd_base(endpoint) + args
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        err = result.stderr.strip()
        if allow_missing and err:
            lowered = err.lower()
            if "no object found" in lowered or "no objects found" in lowered:
                return ""
        if err:
            die(err)
        die("s5cmd failed")
    return result.stdout


def run_s5cmd_input(args: List[str], endpoint: str, data: bytes) -> None:
    cmd = s5cmd_base(endpoint) + args
    result = subprocess.run(
        cmd,
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        if err:
            die(err)
        die("s5cmd failed")


def run_s5cmd_passthrough(args: List[str], endpoint: str) -> None:
    cmd = s5cmd_base(endpoint) + args
    result = subprocess.run(cmd)
    if result.returncode != 0:
        die("s5cmd failed")


# --- ANSI color helpers (only enabled in interactive shell mode) ---
ANSI_RESET = "\x1b[0m"
ANSI_BOLD = "\x1b[1m"
ANSI_BLUE = "\x1b[34m"
ANSI_GREEN = "\x1b[32m"
ANSI_CYAN = "\x1b[36m"
ANSI_YELLOW = "\x1b[33m"
ANSI_MAGENTA = "\x1b[35m"


def open_s5cmd_stream(args: List[str], endpoint: str) -> subprocess.Popen:
    cmd = s5cmd_base(endpoint) + args
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@dataclass
class ShellState:
    bucket: str
    endpoint: str
    cwd: PurePosixPath
    cache: "DirCache"
    interactive: bool
    local_cwd: Path
    mode: str


@dataclass
class LsEntry:
    name: str
    key: PurePosixPath
    size: Optional[int]
    mtime: Optional[datetime]
    is_dir: bool


@dataclass
class _DirCacheValue:
    at_monotonic: float
    entries: List[LsEntry]


class DirCache:
    def __init__(self, ttl_seconds: float, max_entries: int) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._data: "OrderedDict[tuple[str, str], _DirCacheValue]" = OrderedDict()

    def get(self, bucket: str, prefix: PurePosixPath) -> Optional[List[LsEntry]]:
        key = (bucket, prefix.as_posix())
        item = self._data.get(key)
        if item is None:
            return None
        if (time.monotonic() - item.at_monotonic) > self.ttl_seconds:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return item.entries

    def put(self, bucket: str, prefix: PurePosixPath, entries: List[LsEntry]) -> None:
        key = (bucket, prefix.as_posix())
        self._data[key] = _DirCacheValue(at_monotonic=time.monotonic(), entries=entries)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)

    def clear(self) -> None:
        self._data.clear()


@dataclass
class TreeNode:
    entry: LsEntry
    children: List["TreeNode"]


def ensure_s5cmd() -> None:
    if shutil.which("s5cmd") is None:
        die("s5cmd is not installed or not in PATH")


def parse_args(argv: Optional[List[str]] = None) -> Tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=HELP_HEADER,
        epilog=HELP_BODY,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    default_bucket = default_bucket_from_env()
    parser.add_argument(
        "--bucket",
        default=default_bucket,
        help=f"Bucket name or s3:// URI (default: ${DEFAULT_BUCKET_ENV} or {DEFAULT_BUCKET_FALLBACK})",
    )
    return parser.parse_known_args(argv)


def normalize_posix(path: PurePosixPath) -> PurePosixPath:
    parts: List[str] = []
    for part in path.parts:
        if part in ("", "/"):
            continue
        if part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return PurePosixPath("/" + "/".join(parts))


def resolve_prefix(state: ShellState, raw: str) -> PurePosixPath:
    if not raw or raw == ".":
        return state.cwd
    if raw.startswith("/"):
        return normalize_posix(PurePosixPath("/") / raw.lstrip("/"))
    return normalize_posix(state.cwd / raw)


def has_glob(path: str) -> bool:
    return bool(_GLOB.search(path))


def glob_base(key: str) -> str:
    match = _GLOB.search(key)
    if not match:
        return key
    prefix = key[: match.start()]
    if "/" in prefix:
        return prefix.rsplit("/", 1)[0]
    return ""


def build_s3_uri_from_raw(state: ShellState, raw: str) -> str:
    if raw.startswith("s3://"):
        return raw
    prefix = resolve_prefix(state, raw)
    key = prefix.as_posix().lstrip("/")
    return build_s3_uri(state.bucket, key)


def resolve_local_path(state: ShellState, raw: str) -> str:
    expanded = os.path.expanduser(raw)
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(state.local_cwd.as_posix(), expanded))


def parse_ls_args(args: List[str]) -> Tuple[int, str]:
    depth = 1
    path = ""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-R", "--tree"):
            depth = max(depth, 2)
            i += 1
            continue
        if arg.startswith("--depth="):
            depth = int(arg.split("=", 1)[1])
            i += 1
            continue
        if arg == "--depth":
            if i + 1 >= len(args):
                die("ls --depth expects a value")
            depth = int(args[i + 1])
            i += 2
            continue
        if arg.startswith("-"):
            die(f"Unknown ls option: {arg}")
        if path:
            die("ls accepts only one path")
        path = arg
        i += 1
    if depth < 1:
        die("ls --depth must be >= 1")
    return depth, path


def s3_uri_for_prefix(state: ShellState, prefix: PurePosixPath) -> str:
    key = prefix.as_posix().lstrip("/")
    uri = build_s3_uri(state.bucket, key)
    if not uri.endswith("/"):
        uri += "/"
    return uri


def parse_ls_output(output: str, base: PurePosixPath) -> List[LsEntry]:
    entries: List[LsEntry] = []
    for line in output.splitlines():
        line = line.rstrip()
        if not line:
            continue
        dir_match = LS_DIR_RE.match(line)
        if dir_match:
            name = dir_match.group("key").rstrip("/")
            if name == DIR_MARKER:
                continue
            entries.append(
                LsEntry(
                    name=name,
                    key=normalize_posix(base / name),
                    size=None,
                    mtime=None,
                    is_dir=True,
                )
            )
            continue
        file_match = LS_FILE_RE.match(line)
        if not file_match:
            continue
        name = file_match.group("key")
        if name == DIR_MARKER:
            continue
        mtime = datetime.strptime(
            f"{file_match.group('date')} {file_match.group('time')}",
            "%Y/%m/%d %H:%M:%S",
        ).replace(tzinfo=timezone.utc)
        entries.append(
            LsEntry(
                name=name,
                key=normalize_posix(base / name),
                size=int(file_match.group("size")),
                mtime=mtime,
                is_dir=False,
            )
        )
    return entries


def parse_ls_file_line(line: str, display_name: str, key_path: PurePosixPath) -> Optional[LsEntry]:
    file_match = LS_FILE_RE.match(line)
    if not file_match:
        return None
    mtime = datetime.strptime(
        f"{file_match.group('date')} {file_match.group('time')}",
        "%Y/%m/%d %H:%M:%S",
    ).replace(tzinfo=timezone.utc)
    return LsEntry(
        name=display_name,
        key=key_path,
        size=int(file_match.group("size")),
        mtime=mtime,
        is_dir=False,
    )


def list_dir(state: ShellState, prefix: PurePosixPath) -> List[LsEntry]:
    uri = s3_uri_for_prefix(state, prefix)
    output = run_s5cmd(["ls", uri], state.endpoint)
    return parse_ls_output(output, prefix)


def list_dir_quiet(state: ShellState, prefix: PurePosixPath) -> List[LsEntry]:
    cached = state.cache.get(state.bucket, prefix)
    if cached is not None:
        return cached
    uri = s3_uri_for_prefix(state, prefix)
    output = run_s5cmd(["ls", uri], state.endpoint, allow_missing=True)
    if not output.strip():
        entries: List[LsEntry] = []
        state.cache.put(state.bucket, prefix, entries)
        return entries
    entries = parse_ls_output(output, prefix)
    state.cache.put(state.bucket, prefix, entries)
    return entries


def list_dir_quiet_bucket(
    endpoint: str, bucket: str, prefix: PurePosixPath
) -> List[LsEntry]:
    key = prefix.as_posix().lstrip("/")
    uri = build_s3_uri(bucket, key)
    if not uri.endswith("/"):
        uri += "/"
    output = run_s5cmd(["ls", uri], endpoint, allow_missing=True)
    if not output.strip():
        return []
    return parse_ls_output(output, prefix)


def max_mtime(nodes: List[TreeNode]) -> Optional[datetime]:
    best: Optional[datetime] = None
    for node in nodes:
        if node.entry.mtime and (best is None or node.entry.mtime > best):
            best = node.entry.mtime
        child_best = max_mtime(node.children)
        if child_best and (best is None or child_best > best):
            best = child_best
    return best


def sort_entries(entries: List[LsEntry]) -> List[LsEntry]:
    return sorted(
        entries,
        key=lambda entry: (
            entry.mtime or datetime.min.replace(tzinfo=timezone.utc),
            entry.name.lower(),
        ),
        reverse=True,
    )


def sort_nodes(nodes: List[TreeNode]) -> List[TreeNode]:
    return sorted(
        nodes,
        key=lambda node: (
            node.entry.mtime or datetime.min.replace(tzinfo=timezone.utc),
            node.entry.name.lower(),
        ),
        reverse=True,
    )


def build_tree(state: ShellState, prefix: PurePosixPath, depth: int) -> List[TreeNode]:
    entries = list_dir(state, prefix)
    nodes: List[TreeNode] = []
    for entry in entries:
        children: List[TreeNode] = []
        if entry.is_dir and depth > 1:
            children = build_tree(state, entry.key, depth - 1)
            latest = max_mtime(children)
            if latest and (entry.mtime is None or latest > entry.mtime):
                entry.mtime = latest
        nodes.append(TreeNode(entry=entry, children=children))
    return sort_nodes(nodes)


def human_size(size: Optional[int]) -> str:
    if size is None:
        return "-"
    units = ["B", "Ki", "Mi", "Gi", "Ti", "Pi"]
    value = float(size)
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)}"
    return f"{value:.1f}{unit}"


def human_age(when: Optional[datetime], now: datetime) -> str:
    if when is None:
        return "-"
    delta = now - when
    if delta.total_seconds() < 0:
        delta = timedelta(seconds=0)
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds} sec"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} hrs"
    days = hours // 24
    if days < 7:
        return f"{days} days"
    weeks = days // 7
    if days < 30:
        return f"{weeks} weeks"
    months = days // 30
    if days < 365:
        return f"{months} months"
    years = days // 365
    return f"{years} years"


def flatten_nodes(nodes: List[TreeNode]) -> Iterable[LsEntry]:
    for node in nodes:
        yield node.entry
        yield from flatten_nodes(node.children)


def _supports_color(state: ShellState) -> bool:
    if not state.interactive:
        return False
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR") is not None:
        return False
    term = (os.environ.get("TERM", "") or "").lower()
    if term in ("dumb", ""):
        return False
    return True


@lru_cache(maxsize=512)
def _ext_style(ext: str) -> str:
    ext = ext.lower()
    if ext in (".py", ".sh", ".bash", ".zsh", ".fish", ".ps1"):
        return ANSI_GREEN
    if ext in (".json", ".yaml", ".yml", ".toml", ".ini", ".cfg"):
        return ANSI_YELLOW
    if ext in (".txt", ".md", ".rst", ".log"):
        return ANSI_CYAN
    if ext in (".parquet", ".arrow", ".feather", ".csv", ".tsv"):
        return ANSI_MAGENTA
    if ext in (".zip", ".tar", ".gz", ".bz2", ".xz", ".zst"):
        return ANSI_YELLOW
    return ""


def _colorize_name(entry: LsEntry, name: str, enable: bool) -> str:
    if not enable:
        return name
    if entry.is_dir:
        return f"{ANSI_BLUE}{ANSI_BOLD}{name}{ANSI_RESET}"
    ext = PurePosixPath(entry.name).suffix
    style = _ext_style(ext)
    if style:
        return f"{style}{name}{ANSI_RESET}"
    return name


def format_entries(
    entries: List[LsEntry],
    prefix_map: Optional[dict] = None,
    *,
    colorize: bool = False,
) -> List[str]:
    now = datetime.now(timezone.utc)
    size_width = max((len(human_size(entry.size)) for entry in entries), default=1)
    age_width = max((len(human_age(entry.mtime, now)) for entry in entries), default=1)
    lines: List[str] = []
    for entry in entries:
        size_str = human_size(entry.size)
        age_str = human_age(entry.mtime, now)
        safe_name = safe_tty(entry.name)
        name = safe_name + ("/" if entry.is_dir else "")
        name_disp = _colorize_name(entry, name, colorize)
        prefix = ""
        if prefix_map is not None:
            prefix = prefix_map.get(entry.key.as_posix(), "")
        lines.append(
            f"{size_str:>{size_width}} {age_str:>{age_width}} {prefix}{name_disp}".rstrip()
        )
    return lines


def render_tree(nodes: List[TreeNode], *, colorize: bool = False) -> List[str]:
    prefix_map: dict = {}

    def walk(children: List[TreeNode], prefix: str) -> None:
        for idx, node in enumerate(children):
            last = idx == len(children) - 1
            branch = "`-- " if last else "|-- "
            prefix_map[node.entry.key.as_posix()] = prefix + branch
            if node.children:
                extension = "    " if last else "|   "
                walk(node.children, prefix + extension)

    walk(nodes, "")
    entries = list(flatten_nodes(nodes))
    return format_entries(entries, prefix_map=prefix_map, colorize=colorize)


def cmd_ls(state: ShellState, args: List[str]) -> None:
    depth, raw_path = parse_ls_args(args)
    prefix = resolve_prefix(state, raw_path)
    colorize = _supports_color(state)
    if raw_path and has_glob(raw_path):
        if depth != 1:
            die("ls with wildcards does not support tree view")
        uri = build_s3_uri_from_raw(state, raw_path)
        output = run_s5cmd(["ls", uri], state.endpoint, allow_missing=True)
        if not output.strip():
            return
        if raw_path.startswith("s3://"):
            _, key = split_s3_uri(raw_path)
        else:
            key = resolve_prefix(state, raw_path).as_posix().lstrip("/")
        base_key = glob_base(key)
        base = PurePosixPath("/" + base_key) if base_key else PurePosixPath("/")
        entries = sort_entries(parse_ls_output(output, base))
        for line in format_entries(entries, colorize=colorize):
            print(line)
        return
    if raw_path and not raw_path.endswith("/"):
        uri = build_s3_uri_from_raw(state, raw_path)
        output = run_s5cmd(["ls", uri], state.endpoint, allow_missing=True)
        if output.strip():
            line = output.splitlines()[0]
            key_path = resolve_prefix(state, raw_path)
            entry = parse_ls_file_line(line, raw_path, key_path)
            if entry:
                for line_out in format_entries([entry], colorize=colorize):
                    print(line_out)
                return
    if depth == 1:
        entries = sort_entries(list_dir(state, prefix))
        for line in format_entries(entries, colorize=colorize):
            print(line)
        return
    nodes = build_tree(state, prefix, depth)
    for line in render_tree(nodes, colorize=colorize):
        print(line)


def list_local_entries(path: str) -> List[LsEntry]:
    if os.path.isdir(path):
        entries: List[LsEntry] = []
        base = Path(path)
        for entry in os.scandir(path):
            try:
                stat = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            is_dir = entry.is_dir(follow_symlinks=False)
            size = None if is_dir else stat.st_size
            key = PurePosixPath(base.as_posix()) / entry.name
            entries.append(
                LsEntry(
                    name=entry.name,
                    key=key,
                    size=size,
                    mtime=mtime,
                    is_dir=is_dir,
                )
            )
        return entries
    if os.path.exists(path):
        stat = os.stat(path)
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        base = Path(path)
        return [
            LsEntry(
                name=base.name,
                key=PurePosixPath(base.as_posix()),
                size=stat.st_size,
                mtime=mtime,
                is_dir=False,
            )
        ]
    die(f"ls: {path} not found")
    return []


def build_local_tree(path: str, depth: int) -> List[TreeNode]:
    entries = list_local_entries(path)
    nodes: List[TreeNode] = []
    for entry in entries:
        children: List[TreeNode] = []
        if entry.is_dir and depth > 1:
            child_path = os.path.join(path, entry.name)
            children = build_local_tree(child_path, depth - 1)
            latest = max_mtime(children)
            if latest and (entry.mtime is None or latest > entry.mtime):
                entry.mtime = latest
        nodes.append(TreeNode(entry=entry, children=children))
    return sort_nodes(nodes)


def cmd_ls_local(state: ShellState, args: List[str]) -> None:
    depth, raw_path = parse_ls_args(args)
    target = raw_path or "."
    abs_path = resolve_local_path(state, target)
    colorize = _supports_color(state)
    if depth == 1:
        entries = sort_entries(list_local_entries(abs_path))
        for line in format_entries(entries, colorize=colorize):
            print(line)
        return
    nodes = build_local_tree(abs_path, depth)
    for line in render_tree(nodes, colorize=colorize):
        print(line)


def parse_du_args(args: List[str]) -> Tuple[str, List[str]]:
    path = ""
    passthrough: List[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--group", "-g", "--all-versions"):
            passthrough.append(arg)
            i += 1
            continue
        if arg.startswith("-"):
            die(f"Unknown du option: {arg}")
        if path:
            die("du accepts only one path")
        path = arg
        i += 1
    return path, passthrough


def cmd_du(state: ShellState, args: List[str]) -> None:
    path, passthrough = parse_du_args(args)
    prefix = resolve_prefix(state, path)
    key = prefix.as_posix().lstrip("/")
    if key:
        uri = f"s3://{state.bucket}/{key}/*"
    else:
        uri = f"s3://{state.bucket}/*"
    output = run_s5cmd(["du", "--humanize", *passthrough, uri], state.endpoint)
    print(output.rstrip())


def parse_rm_args(args: List[str]) -> Tuple[bool, List[str]]:
    recursive = False
    targets: List[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-r", "-R", "--recursive"):
            recursive = True
            i += 1
            continue
        if arg.startswith("-"):
            die(f"Unknown rm option: {arg}")
        targets.append(arg)
        i += 1
    return recursive, targets


def parse_mkdir_args(args: List[str]) -> Tuple[bool, List[str]]:
    parents = False
    targets: List[str] = []
    for arg in args:
        if arg in ("-p", "--parents"):
            parents = True
            continue
        if arg.startswith("-"):
            die(f"Unknown mkdir option: {arg}")
        targets.append(arg)
    return parents, targets


def parse_single_path(args: List[str], name: str) -> str:
    if not args:
        die(f"{name} requires a path")
    if len(args) > 1:
        die(f"{name} accepts only one path")
    if args[0].startswith("-"):
        die(f"Unknown {name} option: {args[0]}")
    return args[0]


def parse_head_tail_args(args: List[str], name: str) -> Tuple[int, str]:
    count = 10
    path = ""
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-n", "--lines"):
            if i + 1 >= len(args):
                die(f"{name} -n expects a value")
            count = int(args[i + 1])
            i += 2
            continue
        if arg.startswith("-n") and arg != "-n":
            count = int(arg[2:])
            i += 1
            continue
        if arg.startswith("-"):
            die(f"Unknown {name} option: {arg}")
        if path:
            die(f"{name} accepts only one path")
        path = arg
        i += 1
    if not path:
        die(f"{name} requires a path")
    if count < 0:
        die(f"{name} -n must be >= 0")
    return count, path


def resolve_s3_uri(state: ShellState, raw: str) -> str:
    if raw.startswith("s3://"):
        return raw.rstrip("/")
    prefix = resolve_prefix(state, raw)
    key = prefix.as_posix().lstrip("/")
    return build_s3_uri(state.bucket, key)


def split_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("s3://"):
        die(f"Invalid S3 URI: {uri}")
    without = uri[len("s3://") :]
    if not without:
        die("Invalid S3 URI")
    if "/" in without:
        bucket, key = without.split("/", 1)
    else:
        bucket, key = without, ""
    return bucket, key.strip("/")


def resolve_s3_target(state: ShellState, raw: str) -> Tuple[str, PurePosixPath]:
    if raw.startswith("s3://"):
        bucket, key = split_s3_uri(raw)
        return bucket, PurePosixPath("/" + key)
    return state.bucket, resolve_prefix(state, raw)


def classify_path(state: ShellState, raw: str) -> Tuple[str, str]:
    if raw.startswith("s3://"):
        return "s3", raw
    if state.mode == "local":
        return "local", resolve_local_path(state, raw)
    if raw in (".", ".."):
        return "local", resolve_local_path(state, raw)
    if raw.startswith(("/", "./", "../", "~")):
        return "local", resolve_local_path(state, raw)
    local_candidate = resolve_local_path(state, raw)
    if os.path.exists(local_candidate):
        die(
            "Ambiguous path: use ./ or / for local paths, or s3:// for S3 paths"
        )
    prefix = resolve_prefix(state, raw)
    key = prefix.as_posix().lstrip("/")
    return "s3", build_s3_uri(state.bucket, key)


def resolve_s3_only(state: ShellState, raw: str, name: str) -> str:
    if raw.startswith(("/", "./", "../", "~")):
        die(f"{name} only supports S3 paths")
    if "*" in raw:
        die(f"{name} does not support wildcards")
    if raw.endswith("/"):
        die(f"{name} expects a file path, not a prefix")
    if raw.startswith("s3://"):
        return raw
    prefix = resolve_prefix(state, raw)
    key = prefix.as_posix().lstrip("/")
    if not key:
        die(f"{name} expects a file path, not a bucket root")
    return build_s3_uri(state.bucket, key)


def s3_prefix_has_children(state: ShellState, uri: str) -> bool:
    prefix_uri = uri.rstrip("/") + "/"
    output = run_s5cmd(["ls", prefix_uri], state.endpoint, allow_missing=True)
    return bool(output.strip())


def s3_entry_exists(endpoint: str, bucket: str, prefix: PurePosixPath) -> bool:
    if prefix.as_posix() == "/":
        return True
    parent = prefix.parent
    entries = list_dir_quiet_bucket(endpoint, bucket, parent)
    for entry in entries:
        if entry.name == prefix.name:
            return True
    return False


def parse_cp_args(args: List[str]) -> Tuple[List[str], str, str]:
    if len(args) < 2:
        die("cp requires a source and destination")
    options = args[:-2]
    src = args[-2]
    dst = args[-1]
    return options, src, dst


def cmd_cp(state: ShellState, args: List[str]) -> None:
    options, src_raw, dst_raw = parse_cp_args(args)
    src_type, src = classify_path(state, src_raw)
    dst_type, dst = classify_path(state, dst_raw)
    if src_type == "local" and dst_type == "local":
        die("cp requires at least one S3 path")

    if dst_type == "s3" and dst_raw.endswith("/") and not dst.endswith("/"):
        dst += "/"

    if src_type == "local":
        if os.path.isdir(src):
            src = src.rstrip(os.sep) + os.sep
            if dst_type == "s3" and not dst.endswith("/"):
                dst += "/"
    else:
        is_prefix = src_raw.endswith("/") or s3_prefix_has_children(state, src)
        if is_prefix:
            src = src.rstrip("/") + "/*"
            if dst_type == "s3" and not dst.endswith("/"):
                dst += "/"
            if dst_type == "local":
                os.makedirs(dst, exist_ok=True)

    run_s5cmd(["cp", *options, src, dst], state.endpoint)
    state.cache.clear()


def cmd_mv(state: ShellState, args: List[str]) -> None:
    options, src_raw, dst_raw = parse_cp_args(args)
    src_type, src = classify_path(state, src_raw)
    dst_type, dst = classify_path(state, dst_raw)
    if src_type == "local" and dst_type == "local":
        die("mv requires at least one S3 path")
    if src_type == "s3" and dst_type == "local":
        die("mv does not support S3 -> local; use cp then rm")
    if src_type == "s3" and dst_type == "s3":
        src_base = resolve_s3_uri(state, src_raw).rstrip("/")
        dst_base = resolve_s3_uri(state, dst_raw).rstrip("/")
        if src_base == dst_base:
            sys.stderr.write("mv: source and destination are the same; nothing to do\n")
            return

    if dst_type == "s3" and dst_raw.endswith("/") and not dst.endswith("/"):
        dst += "/"

    if src_type == "local":
        if os.path.isdir(src):
            src = src.rstrip(os.sep) + os.sep
            if dst_type == "s3" and not dst.endswith("/"):
                dst += "/"
    else:
        is_prefix = src_raw.endswith("/") or s3_prefix_has_children(state, src)
        if is_prefix:
            src = src.rstrip("/") + "/*"
            if dst_type == "s3" and not dst.endswith("/"):
                dst += "/"
            if dst_type == "local":
                os.makedirs(dst, exist_ok=True)

    run_s5cmd(["mv", *options, src, dst], state.endpoint)
    state.cache.clear()


def parse_sync_args(args: List[str]) -> Tuple[bool, List[str], str, str]:
    if len(args) < 2:
        die("sync requires a source and destination")
    options = args[:-2]
    src = args[-2]
    dst = args[-1]
    true_sync = False
    passthrough: List[str] = []
    for opt in options:
        if opt == "--true-sync":
            true_sync = True
        else:
            passthrough.append(opt)
    return true_sync, passthrough, src, dst


def cmd_sync(state: ShellState, args: List[str]) -> None:
    true_sync, options, src_raw, dst_raw = parse_sync_args(args)
    src_type, src = classify_path(state, src_raw)
    dst_type, dst = classify_path(state, dst_raw)
    if src_type == "local" and dst_type == "local":
        die("sync requires at least one S3 path")

    if src_type == "local" and os.path.isdir(src) and not src.endswith(os.sep):
        src += os.sep
    if dst_type == "local":
        if not dst.endswith(os.sep):
            dst += os.sep
        os.makedirs(dst, exist_ok=True)
    if src_type == "s3":
        is_prefix = src_raw.endswith("/") or s3_prefix_has_children(state, src)
        if is_prefix:
            src = src.rstrip("/") + "/*"
    if dst_type == "s3" and not dst.endswith("/"):
        dst += "/"

    if not true_sync and "--size-only" not in options:
        options = ["--size-only", *options]

    run_s5cmd(["sync", *options, src, dst], state.endpoint)
    state.cache.clear()


def print_help() -> None:
    print(HELP_HEADER)
    print(HELP_BODY.rstrip())


def complete_local_paths(text: str, base_cwd: Path) -> List[str]:
    expanded = os.path.expanduser(text)
    is_abs = os.path.isabs(expanded)
    abs_path = expanded if is_abs else os.path.join(base_cwd.as_posix(), expanded)
    base = abs_path
    prefix = ""
    display_base = text
    if os.path.isdir(abs_path):
        base = abs_path
        prefix = ""
        display_base = text
    else:
        base = os.path.dirname(abs_path) or base_cwd.as_posix()
        prefix = os.path.basename(expanded)
        display_base = os.path.dirname(text)
    try:
        entries = os.listdir(base)
    except OSError:
        return []
    results: List[str] = []
    for entry in entries:
        if prefix and not entry.startswith(prefix):
            continue
        full = os.path.join(base, entry)
        suffix = "/" if os.path.isdir(full) else ""
        if display_base:
            joiner = "" if display_base.endswith("/") else "/"
            display = f"{display_base}{joiner}{entry}"
        else:
            display = entry
        results.append(display + suffix)
    return sorted(results)


def complete_s3_paths(state: ShellState, text: str) -> List[str]:
    bucket = state.bucket
    raw_key = text
    use_uri_prefix = False
    if text.startswith("s3://"):
        without = text[len("s3://") :]
        if "/" in without:
            bucket, raw_key = without.split("/", 1)
        else:
            return []
        if bucket != state.bucket:
            return []
        use_uri_prefix = True
    if use_uri_prefix:
        base_path = PurePosixPath("/")
    elif raw_key.startswith("/"):
        raw_key = raw_key.lstrip("/")
        base_path = PurePosixPath("/")
    else:
        base_path = state.cwd
    if "/" in raw_key:
        parent, partial = raw_key.rsplit("/", 1)
        prefix_path = normalize_posix(base_path / parent)
    else:
        prefix_path = base_path
        partial = raw_key
        parent = ""
    entries = list_dir_quiet(state, prefix_path)
    matches = []
    for entry in entries:
        if not entry.name.startswith(partial):
            continue
        name = entry.name + ("/" if entry.is_dir else "")
        if raw_key.startswith("/"):
            if parent:
                matches.append("/" + parent + "/" + name)
            else:
                matches.append("/" + name)
        elif "/" in raw_key:
            matches.append(parent + "/" + name)
        else:
            matches.append(name)
    if use_uri_prefix:
        matches = [f"s3://{bucket}/" + match.lstrip("/") for match in matches]
    return sorted(matches)


def make_completer(state: ShellState):
    commands = [
        "ls",
        "du",
        "rm",
        "mkdir",
        "cat",
        "head",
        "tail",
        "cp",
        "mv",
        "sync",
        "cd",
        "pwd",
        "local",
        "s3",
        "clear",
        "help",
        "exit",
        "quit",
    ]

    def completer(text: str, idx: int) -> Optional[str]:
        buffer = readline.get_line_buffer()
        begidx = readline.get_begidx()
        before = buffer[:begidx]
        try:
            tokens = shlex.split(before)
        except ValueError:
            tokens = before.split()
        token_prefix = ""
        if before and not before[-1].isspace():
            if tokens:
                token_prefix = tokens[-1]
            else:
                token_prefix = before
        full_token = token_prefix + text

        def strip_prefix(matches: List[str]) -> List[str]:
            if not token_prefix:
                return matches
            stripped: List[str] = []
            for match in matches:
                if match.startswith(token_prefix):
                    stripped.append(match[len(token_prefix) :])
                else:
                    stripped.append(match)
            return stripped

        options: List[str] = []
        if begidx == 0 or not tokens:
            options = [cmd for cmd in commands if cmd.startswith(text)]
        else:
            cmd = tokens[0]
            local_hint = full_token in (".", "..") or full_token.startswith(
                ("/", "./", "../", "~")
            )
            if cmd in ("cp", "mv", "sync"):
                if full_token.startswith("s3://"):
                    options = complete_s3_paths(state, full_token)
                elif state.mode == "local" or local_hint:
                    options = complete_local_paths(full_token, state.local_cwd)
                else:
                    options = complete_s3_paths(state, full_token)
                options = strip_prefix(options)
            elif cmd in ("cd", "ls", "du", "rm", "mkdir", "cat", "head", "tail"):
                if state.mode == "local":
                    options = complete_local_paths(full_token, state.local_cwd)
                else:
                    options = complete_s3_paths(state, full_token)
                options = strip_prefix(options)
        if idx < len(options):
            return options[idx]
        return None

    return completer


def shell_loop(state: ShellState) -> None:
    setup_readline()
    readline.set_completer(make_completer(state))
    while True:
        if state.mode == "local":
            prompt = f"local:{state.local_cwd}> "
        else:
            prompt = f"s3://{state.bucket}{state.cwd.as_posix()}> "
        try:
            line = input(prompt)
        except KeyboardInterrupt:
            print()
            break
        except EOFError:
            print()
            break
        line = line.strip()
        if not line:
            continue
        try:
            tokens = shlex.split(line)
        except ValueError as exc:
            sys.stderr.write(f"Parse error: {exc}\n")
            continue
        cmd = tokens[0]
        cmd_args = tokens[1:]
        if cmd in ("exit", "quit"):
            break
        if cmd == "help":
            print_help()
            continue
        if cmd == "pwd":
            if state.mode == "local":
                print(state.local_cwd)
            else:
                print(state.cwd.as_posix())
            continue
        if cmd == "local":
            if cmd_args:
                die("local takes no arguments")
            state.mode = "local"
            continue
        if cmd == "s3":
            if cmd_args:
                die("s3 takes no arguments")
            state.mode = "s3"
            continue
        if cmd == "clear":
            cmd_clear(cmd_args)
            continue
        if cmd == "cd":
            target = cmd_args[0] if cmd_args else "/"
            if state.mode == "local":
                new_path = resolve_local_path(state, target)
                if not os.path.isdir(new_path):
                    die(f"cd: {new_path} not a directory")
                state.local_cwd = Path(new_path)
            else:
                state.cwd = resolve_prefix(state, target)
            continue
        try:
            if cmd == "ls":
                if state.mode == "local":
                    cmd_ls_local(state, cmd_args)
                else:
                    cmd_ls(state, cmd_args)
            elif cmd == "du":
                if state.mode == "local":
                    die("du only works in s3 mode; use 's3' to switch")
                cmd_du(state, cmd_args)
            elif cmd == "rm":
                if state.mode == "local":
                    die("rm only works in s3 mode; use 's3' to switch")
                cmd_rm(state, cmd_args)
            elif cmd == "mkdir":
                if state.mode == "local":
                    die("mkdir only works in s3 mode; use 's3' to switch")
                cmd_mkdir(state, cmd_args)
            elif cmd == "cat":
                if state.mode == "local":
                    die("cat only works in s3 mode; use 's3' to switch")
                cmd_cat(state, cmd_args)
            elif cmd == "head":
                if state.mode == "local":
                    die("head only works in s3 mode; use 's3' to switch")
                cmd_head(state, cmd_args)
            elif cmd == "tail":
                if state.mode == "local":
                    die("tail only works in s3 mode; use 's3' to switch")
                cmd_tail(state, cmd_args)
            elif cmd == "cp":
                cmd_cp(state, cmd_args)
            elif cmd == "mv":
                cmd_mv(state, cmd_args)
            elif cmd == "sync":
                cmd_sync(state, cmd_args)
            else:
                die(f"Unknown command: {cmd}")
        except SystemExit:
            continue


def setup_readline() -> None:
    doc = readline.__doc__ or ""
    readline.set_completer_delims(" \t\n/")
    if "libedit" in doc:
        readline.parse_and_bind("bind ^I rl_complete")
        readline.parse_and_bind('bind "\\e[Z" rl_complete')
    else:
        readline.parse_and_bind("tab: complete")
        readline.parse_and_bind('"\\e[Z": complete')


def cmd_mkdir(state: ShellState, args: List[str]) -> None:
    parents, targets = parse_mkdir_args(args)
    if not targets:
        die("mkdir requires at least one path")
    for raw in targets:
        bucket, prefix = resolve_s3_target(state, raw)
        key = prefix.as_posix().lstrip("/")
        if not key:
            die("mkdir target cannot be bucket root")
        if s3_entry_exists(state.endpoint, bucket, prefix):
            if parents:
                continue
            die(f"mkdir: {prefix.as_posix()} already exists")
        marker_key = key.rstrip("/") + "/" + DIR_MARKER
        uri = build_s3_uri(bucket, marker_key)
        run_s5cmd_input(["pipe", uri], state.endpoint, data=b"")
        state.cache.clear()


def cmd_clear(args: List[str]) -> None:
    if args:
        die("clear takes no arguments")
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()


def cmd_cat(state: ShellState, args: List[str]) -> None:
    path = parse_single_path(args, "cat")
    uri = resolve_s3_only(state, path, "cat")
    run_s5cmd_passthrough(["cat", uri], state.endpoint)


def cmd_head(state: ShellState, args: List[str]) -> None:
    count, path = parse_head_tail_args(args, "head")
    if count == 0:
        return
    uri = resolve_s3_only(state, path, "head")
    proc = open_s5cmd_stream(["cat", uri], state.endpoint)
    lines = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        lines += 1
        if lines >= count:
            break
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def cmd_tail(state: ShellState, args: List[str]) -> None:
    count, path = parse_head_tail_args(args, "tail")
    if count == 0:
        return
    uri = resolve_s3_only(state, path, "tail")
    proc = open_s5cmd_stream(["cat", uri], state.endpoint)
    assert proc.stdout is not None
    assert proc.stderr is not None
    buffer: deque[str] = deque(maxlen=count)
    for line in proc.stdout:
        buffer.append(line)
    stderr = proc.stderr.read()
    returncode = proc.wait()
    if returncode != 0:
        err = stderr.strip()
        if err:
            die(err)
        die("s5cmd failed")
    for line in buffer:
        sys.stdout.write(line)


def cmd_rm(state: ShellState, args: List[str]) -> None:
    recursive, targets = parse_rm_args(args)
    if not targets:
        die("rm requires at least one path")
    for raw in targets:
        explicit_prefix = raw.endswith("/")
        base_uri = resolve_s3_uri(state, raw).rstrip("/")

        if explicit_prefix:
            if not recursive:
                sys.stderr.write(
                    f"rm: refusing to delete prefix '{raw}' without -r (use: rm -r {raw})\n"
                )
                continue
            contents_uri = base_uri + "/*"
            run_s5cmd(["rm", contents_uri], state.endpoint, allow_missing=True)
            state.cache.clear()
            continue

        file_probe = run_s5cmd(["ls", base_uri], state.endpoint, allow_missing=True)
        file_exists = bool(file_probe.strip())

        prefix_uri = base_uri + "/"
        prefix_probe = run_s5cmd(["ls", prefix_uri], state.endpoint, allow_missing=True)
        prefix_has_children = bool(prefix_probe.strip())

        if file_exists:
            if recursive and prefix_has_children:
                sys.stderr.write(
                    f"rm: note: '{raw}' also has children as a prefix; "
                    f"to delete recursively use: rm -r {raw}/\n"
                )
            run_s5cmd(["rm", base_uri], state.endpoint)
            state.cache.clear()
            continue

        if prefix_has_children:
            if not recursive:
                sys.stderr.write(
                    f"rm: refusing to delete prefix '{raw}/' without -r (use: rm -r {raw}/)\n"
                )
                continue
            contents_uri = base_uri + "/*"
            run_s5cmd(["rm", contents_uri], state.endpoint, allow_missing=True)
            state.cache.clear()
            continue

        run_s5cmd(["rm", base_uri], state.endpoint)
        state.cache.clear()


def main() -> None:
    ensure_s5cmd()
    endpoint = check_env()
    args, rest = parse_args()
    if not args.bucket:
        die(
            "No default bucket set. Provide --bucket or set S3_BUCKET/AWS_S3_BUCKET/AWS_BUCKET."
        )
    bucket = normalize_bucket(args.bucket)
    cache = DirCache(ttl_seconds=TAB_CACHE_TTL_SECONDS, max_entries=TAB_CACHE_MAX_ENTRIES)
    state = ShellState(
        bucket=bucket,
        endpoint=endpoint,
        cwd=PurePosixPath("/"),
        cache=cache,
        interactive=False,
        local_cwd=Path.cwd(),
        mode="s3",
    )

    command = rest[0] if rest else None
    command_args = rest[1:] if len(rest) > 1 else []
    if command:
        state.interactive = False
        if command == "ls":
            cmd_ls(state, command_args)
            return
        if command == "du":
            cmd_du(state, command_args)
            return
        if command == "rm":
            cmd_rm(state, command_args)
            return
        if command == "mkdir":
            cmd_mkdir(state, command_args)
            return
        if command == "cat":
            cmd_cat(state, command_args)
            return
        if command == "head":
            cmd_head(state, command_args)
            return
        if command == "tail":
            cmd_tail(state, command_args)
            return
        if command == "clear":
            cmd_clear(command_args)
            return
        if command == "cp":
            cmd_cp(state, command_args)
            return
        if command == "mv":
            cmd_mv(state, command_args)
            return
        if command == "sync":
            cmd_sync(state, command_args)
            return
        die(f"Unknown command: {command}")

    state.interactive = True
    shell_loop(state)


if __name__ == "__main__":
    main()

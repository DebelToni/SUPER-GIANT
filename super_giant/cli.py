from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOBAL_CONFIG = "GIANT/v3/Global_Config.yml"
DEFAULT_REGISTRY = "GIANT/v3/Configs/registry.yml"


def run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=REPO_ROOT)


def python_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--global_config", default=DEFAULT_GLOBAL_CONFIG)


def cmd_tokenizer_train(args: argparse.Namespace) -> int:
    return run(python_cmd("GIANT/v3/data_pipeline/train_tokenizer.py", "--config", args.config, "--global_config", args.global_config))


def cmd_data_build(args: argparse.Namespace) -> int:
    cmd = python_cmd("GIANT/v3/data_pipeline/build_corpus.py", "--config", args.config, "--global_config", args.global_config)
    if args.stage:
        cmd += ["--stage", args.stage]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.dataset_dir:
        cmd += ["--dataset_dir", args.dataset_dir]
    return run(cmd)


def cmd_train(args: argparse.Namespace) -> int:
    cmd = python_cmd("GIANT/v3/model/Run_training.py", "--config", args.config, "--global_config", args.global_config)
    if args.resume is not None:
        cmd += ["--resume", args.resume]
    if args.init_checkpoint:
        cmd += ["--init_checkpoint", args.init_checkpoint]
    if args.checkpoint_dir:
        cmd += ["--checkpoint_dir", args.checkpoint_dir]
    if args.upload_on_checkpoint:
        cmd.append("--upload-on-checkpoint")
    return run(cmd)


def cmd_chat(args: argparse.Namespace) -> int:
    cmd = python_cmd("GIANT/v3/model/Generate_chat.py", "--config", args.config, "--global_config", args.global_config)
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if args.steps is not None:
        cmd += ["--steps", str(args.steps)]
    if args.greedy:
        cmd.append("--greedy")
    if args.max_context:
        cmd += ["--max_context", str(args.max_context)]
    return run(cmd)


def cmd_gpu(args: argparse.Namespace) -> int:
    script = str(REPO_ROOT / "CICD/tools/runpod-gpu.sh")
    if args.gpu_cmd == "create":
        cmd = [script, "create", "--gpu", args.gpu, "--name", args.name]
        if args.spot:
            cmd.append("--spot")
        if args.wait:
            cmd.append("--wait")
        if args.id_only:
            cmd.append("--id-only")
    elif args.gpu_cmd in {"list", "stop", "remove"}:
        cmd = [script, args.gpu_cmd]
        if args.pod_id:
            cmd.append(args.pod_id)
    else:
        raise AssertionError(args.gpu_cmd)
    return run(cmd)


def cmd_s3(args: argparse.Namespace) -> int:
    return run([sys.executable, "CICD/tools/s3.py", *args.s3_args])


def cmd_configs(args: argparse.Namespace) -> int:
    registry_path = REPO_ROOT / args.registry
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    rows = data.get("configs", [])
    if args.kind:
        rows = [row for row in rows if row.get("kind") == args.kind]
    if args.status:
        rows = [row for row in rows if row.get("status") == args.status]
    for row in rows:
        gpu = "gpu" if row.get("needs_gpu") else "cpu"
        print(f"{row['name']:<28} {row['kind']:<10} {row['status']:<12} {gpu:<3} {row['path']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sg", description="SUPER-GIANT command-line facade")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tok = sub.add_parser("tokenizer")
    tok_sub = tok.add_subparsers(dest="tokenizer_cmd", required=True)
    tok_train = tok_sub.add_parser("train")
    add_common_config_args(tok_train)
    tok_train.set_defaults(func=cmd_tokenizer_train)

    data = sub.add_parser("data")
    data_sub = data.add_subparsers(dest="data_cmd", required=True)
    data_build = data_sub.add_parser("build")
    add_common_config_args(data_build)
    data_build.add_argument("--stage")
    data_build.add_argument("--dry-run", action="store_true")
    data_build.add_argument("--dataset_dir")
    data_build.set_defaults(func=cmd_data_build)

    train = sub.add_parser("train")
    add_common_config_args(train)
    train.add_argument("--resume", nargs="?", const="latest", default=None)
    train.add_argument("--init_checkpoint")
    train.add_argument("--checkpoint_dir")
    train.add_argument("--upload-on-checkpoint", action="store_true")
    train.set_defaults(func=cmd_train)

    chat = sub.add_parser("chat")
    chat.add_argument("--config", required=True)
    chat.add_argument("--global_config", default=DEFAULT_GLOBAL_CONFIG)
    chat.add_argument("--checkpoint", default="latest")
    chat.add_argument("--prompt")
    chat.add_argument("--steps", type=int)
    chat.add_argument("--greedy", action="store_true")
    chat.add_argument("--max_context", type=int)
    chat.set_defaults(func=cmd_chat)

    gpu = sub.add_parser("gpu")
    gpu_sub = gpu.add_subparsers(dest="gpu_cmd", required=True)
    gpu_create = gpu_sub.add_parser("create")
    gpu_create.add_argument("--gpu", default="auto")
    gpu_create.add_argument("--name", default="giant-job")
    gpu_create.add_argument("--spot", action="store_true")
    gpu_create.add_argument("--wait", action="store_true")
    gpu_create.add_argument("--id-only", action="store_true")
    gpu_create.set_defaults(func=cmd_gpu)
    for name in ("list", "stop", "remove"):
        p = gpu_sub.add_parser(name)
        p.add_argument("pod_id", nargs="?")
        p.set_defaults(func=cmd_gpu)

    s3 = sub.add_parser("s3")
    s3.add_argument("s3_args", nargs=argparse.REMAINDER)
    s3.set_defaults(func=cmd_s3)

    cfg = sub.add_parser("configs")
    cfg.add_argument("--registry", default=DEFAULT_REGISTRY)
    cfg.add_argument("--kind")
    cfg.add_argument("--status")
    cfg.set_defaults(func=cmd_configs)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

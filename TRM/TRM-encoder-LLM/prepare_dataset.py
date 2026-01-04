from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf


ROLE_MAP = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
}


def load_configs(config_path: Optional[str]) -> OmegaConf:
    root = Path(__file__).resolve().parent.parent
    global_cfg = OmegaConf.load(root / "Global_Config.yml")
    local_cfg = OmegaConf.load(Path(__file__).resolve().parent / "Config.yml")
    cfg = OmegaConf.merge(global_cfg, local_cfg)

    base_root = cfg.paths.get("data_root") if "paths" in cfg else None
    base_root = Path(base_root) if base_root else root
    cfg.paths.data_root = str(base_root)

    def resolve_path(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        path = Path(str(value))
        if path.is_absolute():
            return str(path)
        return str(base_root / path)

    for key in ("dataset_dir", "hf_cache_dir", "tokenizer_cache_dir"):
        if key in cfg.paths and cfg.paths[key] is not None:
            cfg.paths[key] = resolve_path(cfg.paths[key])

    if "dataset_out" in cfg.data:
        cfg.data.dataset_out = resolve_path(cfg.data.dataset_out)

    if config_path:
        override_cfg = OmegaConf.load(config_path)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def load_tokenizer(cfg: OmegaConf):
    from transformers import AutoTokenizer

    tok_cfg = cfg.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tok_cfg.name,
        use_fast=bool(tok_cfg.get("use_fast", True)),
        cache_dir=tok_cfg.get("cache_dir"),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def _format_messages(messages: Iterable[dict[str, Any]]) -> Optional[str]:
    parts = []
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        label = ROLE_MAP.get(role, role.title() if role else "Speaker")
        parts.append(f"{label}: {content}")
    if not parts:
        return None
    return "\n".join(parts)


def _format_conversations(messages: Iterable[dict[str, Any]]) -> Optional[str]:
    parts = []
    for msg in messages:
        role = str(msg.get("from", msg.get("role", ""))).strip().lower()
        content = str(msg.get("value", msg.get("content", ""))).strip()
        if not content:
            continue
        label = ROLE_MAP.get(role, role.title() if role else "Speaker")
        parts.append(f"{label}: {content}")
    if not parts:
        return None
    return "\n".join(parts)


def extract_text(row: dict[str, Any]) -> Optional[str]:
    if "messages" in row and isinstance(row["messages"], list):
        return _format_messages(row["messages"])
    if "conversations" in row and isinstance(row["conversations"], list):
        return _format_conversations(row["conversations"])
    if "instruction" in row and "output" in row:
        inst = str(row["instruction"]).strip()
        out = str(row["output"]).strip()
        if inst and out:
            return f"User: {inst}\nAssistant: {out}"
    if "prompt" in row and "response" in row:
        prompt = str(row["prompt"]).strip()
        resp = str(row["response"]).strip()
        if prompt and resp:
            return f"User: {prompt}\nAssistant: {resp}"
    if "text" in row:
        text = str(row["text"]).strip()
        return text if text else None
    return None


def _pad_to_len(tokens: list[int], length: int, pad_id: int) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.full((length,), pad_id, dtype=np.int32)
    mask = np.zeros((length,), dtype=np.uint8)
    if tokens:
        use = tokens[:length]
        arr[: len(use)] = np.asarray(use, dtype=np.int32)
        mask[: len(use)] = 1
    return arr, mask


def build_arrays(
    rows: Iterable[dict[str, Any]],
    *,
    tokenizer,
    encoder_len: int,
    decoder_len: int,
    pad_id: int,
    max_samples: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    enc_tokens = []
    enc_mask = []
    dec_input = []
    dec_target = []
    dec_mask = []

    count = 0
    for row in rows:
        text = extract_text(row)
        if not text:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 2:
            continue
        enc_slice = tokens[-encoder_len:]
        dec_slice = tokens[-(decoder_len + 1) :]
        dec_in = dec_slice[:-1]
        dec_tgt = dec_slice[1:]

        enc_arr, enc_m = _pad_to_len(enc_slice, encoder_len, pad_id)
        dec_in_arr, _ = _pad_to_len(dec_in, decoder_len, pad_id)
        dec_tgt_arr, dec_m = _pad_to_len(dec_tgt, decoder_len, pad_id)

        enc_tokens.append(enc_arr)
        enc_mask.append(enc_m)
        dec_input.append(dec_in_arr)
        dec_target.append(dec_tgt_arr)
        dec_mask.append(dec_m)

        count += 1
        if max_samples is not None and count >= max_samples:
            break

    if not enc_tokens:
        raise RuntimeError("No usable samples found; adjust dataset filters.")

    return (
        np.stack(enc_tokens),
        np.stack(enc_mask),
        np.stack(dec_input),
        np.stack(dec_target),
        np.stack(dec_mask),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Prepare chat dataset for TRM encoder experiments")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--train_split", type=str, default=None)
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--encoder_max_len", type=int, default=None)
    parser.add_argument("--decoder_max_len", type=int, default=None)
    parser.add_argument("--dataset_out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_configs(args.config)

    if args.data_root is not None:
        cfg.paths.data_root = str(Path(args.data_root).resolve())

    dataset_name = args.dataset_name or cfg.data.dataset_name
    dataset_config = args.dataset_config or cfg.data.get("dataset_config")
    train_split = args.train_split or cfg.data.train_split
    val_split = args.val_split or cfg.data.val_split
    encoder_len = int(args.encoder_max_len or cfg.data.encoder_max_len)
    decoder_len = int(args.decoder_max_len or cfg.data.decoder_max_len)
    max_train = args.max_train_samples or cfg.data.get("max_train_samples")
    max_val = args.max_val_samples or cfg.data.get("max_val_samples")
    seed = int(args.seed or cfg.data.get("seed", 0))

    out_path = Path(args.dataset_out or cfg.data.dataset_out)
    if not out_path.is_absolute():
        out_path = Path(cfg.paths.data_root) / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = cfg.paths.get("hf_cache_dir")

    print(f"[dataset] loading {dataset_name} split={train_split}/{val_split}")
    from datasets import load_dataset

    ds_train = load_dataset(
        dataset_name,
        dataset_config,
        split=train_split,
        cache_dir=cache_dir,
    )
    ds_val = load_dataset(
        dataset_name,
        dataset_config,
        split=val_split,
        cache_dir=cache_dir,
    )

    if max_train is not None:
        ds_train = ds_train.shuffle(seed=seed).select(range(int(max_train)))
    if max_val is not None:
        ds_val = ds_val.shuffle(seed=seed + 1).select(range(int(max_val)))

    tokenizer = load_tokenizer(cfg)
    pad_id = int(tokenizer.pad_token_id)

    train_arrays = build_arrays(
        ds_train,
        tokenizer=tokenizer,
        encoder_len=encoder_len,
        decoder_len=decoder_len,
        pad_id=pad_id,
        max_samples=max_train,
    )
    val_arrays = build_arrays(
        ds_val,
        tokenizer=tokenizer,
        encoder_len=encoder_len,
        decoder_len=decoder_len,
        pad_id=pad_id,
        max_samples=max_val,
    )

    np.savez_compressed(
        out_path,
        pad_token_id=pad_id,
        train_encoder_tokens=train_arrays[0],
        train_encoder_mask=train_arrays[1],
        train_decoder_input=train_arrays[2],
        train_decoder_target=train_arrays[3],
        train_decoder_mask=train_arrays[4],
        val_encoder_tokens=val_arrays[0],
        val_encoder_mask=val_arrays[1],
        val_decoder_input=val_arrays[2],
        val_decoder_target=val_arrays[3],
        val_decoder_mask=val_arrays[4],
    )

    print(
        "[dataset] saved",
        out_path,
        "train",
        train_arrays[0].shape,
        "val",
        val_arrays[0].shape,
    )


if __name__ == "__main__":
    main()

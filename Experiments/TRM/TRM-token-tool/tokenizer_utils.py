from __future__ import annotations

from pathlib import Path
from typing import List

from transformers import AutoTokenizer

from config_utils import load_config


def _ensure_pad_token(tokenizer, pad_override: str | None) -> None:
    if pad_override:
        if tokenizer.pad_token != pad_override:
            tokenizer.add_special_tokens({"pad_token": pad_override})
        return
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def _collect_new_specials(tokenizer, tokens: List[str]) -> List[str]:
    vocab = tokenizer.get_vocab()
    return [tok for tok in tokens if tok not in vocab]


def build_custom_tokenizer(*, force: bool = False) -> Path:
    cfg = load_config()
    tok_cfg = cfg.tokenizer
    output_dir = Path(tok_cfg.custom_path)

    if output_dir.exists() and not force:
        return output_dir

    tokenizer = AutoTokenizer.from_pretrained(
        tok_cfg.name,
        use_fast=True,
        cache_dir=tok_cfg.cache_dir,
    )

    specials = list(tok_cfg.get("add_special_tokens", []))
    new_specials = _collect_new_specials(tokenizer, specials)
    if new_specials:
        tokenizer.add_special_tokens({"additional_special_tokens": new_specials})

    _ensure_pad_token(tokenizer, tok_cfg.get("pad_token_override"))

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def load_tokenizer():
    cfg = load_config()
    tok_cfg = cfg.tokenizer
    if tok_cfg.use_custom:
        tokenizer = AutoTokenizer.from_pretrained(tok_cfg.custom_path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_cfg.name,
            use_fast=True,
            cache_dir=tok_cfg.cache_dir,
        )
    _ensure_pad_token(tokenizer, tok_cfg.get("pad_token_override"))
    return tokenizer


def main() -> None:
    out = build_custom_tokenizer(force=False)
    print(f"Saved tokenizer to {out}")


if __name__ == "__main__":
    main()

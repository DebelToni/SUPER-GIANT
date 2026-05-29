from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Optional

from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from GIANT.v3.data_pipeline.build_corpus import (
    StageSourceCfg,
    S3UploadManager,
    _artifact_s3_upload_cfg,
    _build_chat_text_and_spans,
    _extract_text,
    _iter_hf_source,
    _iter_json_dir,
    _parse_s3_upload_cfg,
    _parse_stage_sources,
)
from GIANT.v3.data_pipeline.cleaning import normalise_text
from GIANT.v3.run_manifest import build_manifest, write_manifest


DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def _set_hf_cache(hf_cache_root: Optional[str]) -> None:
    if not hf_cache_root:
        return
    hf_cache = str(Path(hf_cache_root))
    os.environ["HF_HOME"] = hf_cache
    os.environ["HF_DATASETS_CACHE"] = str(Path(hf_cache) / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_cache) / "transformers")


def _iter_source_rows(source: StageSourceCfg) -> Iterator[Dict[str, Any]]:
    source_type = (source.type or "huggingface").lower()
    if source_type in {"hf", "huggingface"}:
        yield from _iter_hf_source(source)
        return
    if source_type in {"json", "jsonl", "json_dir"}:
        yield from _iter_json_dir(source)
        return
    raise ValueError(f"Unsupported tokenizer source.type '{source.type}'")


def _normalize_text(text: str, normalization_cfg: Optional[SimpleNamespace]) -> str:
    if normalization_cfg is None:
        return text
    return normalise_text(text, normalization_cfg)


def _iter_texts(
    sources: Iterable[StageSourceCfg],
    *,
    normalization_cfg: Optional[SimpleNamespace],
) -> Iterator[str]:
    for source in sources:
        for row in _iter_source_rows(source):
            if not isinstance(row, dict):
                continue
            chat_payload = _build_chat_text_and_spans(row, source, normalization_cfg)
            if chat_payload is not None:
                text, _ = chat_payload
            else:
                text = _extract_text(row, source)
                if not text:
                    continue
                text = _normalize_text(text.strip(), normalization_cfg)
            text = text.strip()
            if text:
                yield text


def _build_chat_template(im_start: str, im_end: str) -> str:
    return (
        "{% for message in messages %}"
        "{{ '"
        + im_start
        + "' + message['role'] + '\n' + message['content'] + '"
        + im_end
        + "\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '"
        + im_start
        + "assistant\n' }}{% endif %}"
    )


def train_tokenizer(cfg: OmegaConf) -> Path:
    output_dir = Path(str(cfg.output_dir)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = list(cfg.get("special_tokens", DEFAULT_SPECIAL_TOKENS))
    if not special_tokens:
        raise ValueError("special_tokens must not be empty")

    sources = _parse_stage_sources(cfg.get("training_sources", []))
    if not sources:
        raise ValueError("training_sources must contain at least one source")

    normalization_cfg = None
    if cfg.get("normalization"):
        normalization_cfg = SimpleNamespace(**dict(cfg.normalization))

    tokenizer = Tokenizer(BPE(unk_token=special_tokens[0]))
    tokenizer.normalizer = NormalizerSequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=int(cfg.vocab_size),
        min_frequency=int(cfg.get("min_frequency", 2)),
        show_progress=True,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        _iter_texts(sources, normalization_cfg=normalization_cfg),
        trainer=trainer,
    )

    eos_token = special_tokens[0]
    additional_special_tokens = special_tokens[1:]
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=eos_token,
        eos_token=eos_token,
        unk_token=eos_token,
        pad_token=eos_token,
        additional_special_tokens=additional_special_tokens,
    )
    if len(special_tokens) >= 3:
        hf_tokenizer.chat_template = _build_chat_template(special_tokens[1], special_tokens[2])
    hf_tokenizer.save_pretrained(str(output_dir))
    return output_dir


def validate_tokenizer(cfg: OmegaConf, tokenizer_dir: Path) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    special_tokens = list(cfg.get("special_tokens", DEFAULT_SPECIAL_TOKENS))
    special_token_ids = {tok: int(tokenizer.convert_tokens_to_ids(tok)) for tok in special_tokens}
    for tok, tok_id in special_token_ids.items():
        if tok_id < 0:
            raise ValueError(f"Tokenizer failed to register special token {tok!r}")

    validation_sources = _parse_stage_sources(cfg.get("validation_sources", []))
    normalization_cfg = None
    if cfg.get("normalization"):
        normalization_cfg = SimpleNamespace(**dict(cfg.normalization))

    documents = 0
    characters = 0
    tokens = 0
    preview: List[Dict[str, Any]] = []
    preview_limit = int(cfg.get("preview_examples", 3))
    for text in _iter_texts(validation_sources, normalization_cfg=normalization_cfg):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        documents += 1
        characters += len(text)
        tokens += len(token_ids)
        if len(preview) < preview_limit:
            preview.append(
                {
                    "text_preview": text[:200],
                    "token_count": len(token_ids),
                    "tokens_preview": token_ids[:32],
                }
            )

    chat_probe = (
        f"{special_tokens[1]}user\nHello{special_tokens[2]}\n"
        f"{special_tokens[1]}assistant\nHi there{special_tokens[2]}\n"
        if len(special_tokens) >= 3
        else "Hello world"
    )
    chat_probe_ids = tokenizer.encode(chat_probe, add_special_tokens=False)
    encoded_specials = {
        tok: chat_probe_ids.count(tok_id) for tok, tok_id in special_token_ids.items()
    }

    metrics = {
        "tokenizer_dir": str(tokenizer_dir),
        "vocab_size": len(tokenizer),
        "special_token_ids": special_token_ids,
        "documents": documents,
        "characters": characters,
        "tokens": tokens,
        "chars_per_token": (characters / tokens) if tokens else None,
        "tokens_per_document": (tokens / documents) if documents else None,
        "chat_probe_ids": chat_probe_ids[:64],
        "chat_probe_special_counts": encoded_specials,
        "preview": preview,
    }
    metrics_path = tokenizer_dir / "tokenizer_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def _load_config(config_path: str, global_config_path: Optional[str]) -> tuple[OmegaConf, OmegaConf]:
    cfg = OmegaConf.load(config_path)
    if global_config_path is None:
        default_global = Path(__file__).resolve().parents[1] / "Global_Config.yml"
        global_config_path = str(default_global)
    global_cfg = OmegaConf.load(global_config_path) if Path(global_config_path).exists() else OmegaConf.create({})
    if cfg.get("hf_cache_root") is None and global_cfg.get("paths", {}).get("hf_cache_root") is not None:
        cfg.hf_cache_root = global_cfg.paths.hf_cache_root
    return cfg, global_cfg


def _upload_tokenizer_if_configured(cfg: OmegaConf, global_cfg: OmegaConf, tokenizer_dir: Path) -> None:
    raw_s3_upload = cfg.get("s3_upload")
    if raw_s3_upload is None:
        upload_cfg = _artifact_s3_upload_cfg(global_cfg, kind="tokenizers", local_path=tokenizer_dir)
    else:
        upload_cfg = _parse_s3_upload_cfg(raw_s3_upload)
    if not bool(upload_cfg.enabled):
        return
    destination = upload_cfg.destination or upload_cfg.destination_root
    if not destination:
        raise ValueError("tokenizer s3_upload.enabled=true requires destination_root or destination")
    manager = S3UploadManager()
    manager.upload_dir(tokenizer_dir, destination, upload_cfg)
    manager.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate a custom HF BPE tokenizer.")
    parser.add_argument("--config", required=True, help="Path to tokenizer config YAML")
    parser.add_argument("--global_config", default=None, help="Path to GIANT/v3 Global_Config.yml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg, global_cfg = _load_config(args.config, args.global_config)
    _set_hf_cache(cfg.get("hf_cache_root"))
    tokenizer_dir = train_tokenizer(cfg)
    metrics = validate_tokenizer(cfg, tokenizer_dir)
    _upload_tokenizer_if_configured(cfg, global_cfg, tokenizer_dir)
    s3_outputs = []
    raw_s3_upload = cfg.get("s3_upload")
    if raw_s3_upload is not None:
        upload_cfg = _parse_s3_upload_cfg(raw_s3_upload)
        dest = upload_cfg.destination or upload_cfg.destination_root
        if bool(upload_cfg.enabled) and dest:
            s3_outputs.append(str(dest))
    else:
        upload_cfg = _artifact_s3_upload_cfg(global_cfg, kind="tokenizers", local_path=tokenizer_dir)
        if bool(upload_cfg.enabled) and upload_cfg.destination_root:
            s3_outputs.append(str(upload_cfg.destination_root))
    manifest_path = tokenizer_dir / "run_manifest.json"
    write_manifest(
        manifest_path,
        build_manifest(
            kind="tokenizer",
            config_path=args.config,
            global_config_path=args.global_config,
            outputs=[tokenizer_dir, tokenizer_dir / "tokenizer_metrics.json"],
            s3_outputs=s3_outputs,
            extra={"vocab_size": metrics.get("vocab_size")},
        ),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

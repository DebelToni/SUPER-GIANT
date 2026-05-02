from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
from openai import OpenAI
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from config_utils import load_config
from tokenizer_utils import build_custom_tokenizer, load_tokenizer


PROJECT_DIR = Path(__file__).resolve().parent
TRM_ROOT = PROJECT_DIR.parent

from TRM.sudoku.sudoku_dataset import generate_dataset


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build synthetic TRM Sudoku dataset.")
    ap.add_argument("--force-prompts", action="store_true", help="Regenerate prompt templates via DeepSeek.")
    ap.add_argument("--force-dataset", action="store_true", help="Overwrite existing dataset shards.")
    ap.add_argument("--train-samples", type=int, default=None, help="Override train sample count.")
    ap.add_argument("--val-samples", type=int, default=None, help="Override val sample count.")
    ap.add_argument("--seed", type=int, default=None, help="Override RNG seed.")
    return ap.parse_args()


def _extract_json_list(text: str) -> List[str]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON array found in model output.")
    payload = text[start : end + 1]
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("Prompt template payload is not a JSON list.")
    return [str(item) for item in data]


def _sanitize_templates(templates: Iterable[str], *, placeholder: str, max_words: int) -> List[str]:
    cleaned = []
    for raw in templates:
        text = " ".join(str(raw).strip().split())
        if not text:
            continue
        if placeholder not in text:
            text = f"{text} {placeholder}"
        if max_words and len(text.split()) > max_words:
            text = " ".join(text.split()[:max_words])
            if placeholder not in text:
                text = f"{text} {placeholder}"
        cleaned.append(text)
    return cleaned


def _load_or_create_templates(cfg, *, force: bool) -> List[str]:
    path = Path(cfg.data.prompt_template_path)
    if path.exists() and not force:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return _sanitize_templates(
            data,
            placeholder=cfg.data.puzzle_placeholder,
            max_words=int(cfg.data.max_template_words),
        )

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set. Export it before running dataset generation.")

    client = OpenAI(api_key=api_key, base_url=cfg.data.openai_base_url)
    system = (
        "You write short user prompts for Sudoku solving. "
        "Return only a JSON array of strings, no extra text."
    )
    user = (
        f"Provide {cfg.data.prompt_template_count} unique, short user prompts that ask to solve a Sudoku. "
        f"Each string must include the literal placeholder {cfg.data.puzzle_placeholder} "
        f"where the 81-digit puzzle string should be inserted. "
        f"Each prompt must be <= {cfg.data.max_template_words} words."
    )
    resp = client.chat.completions.create(
        model=cfg.data.openai_model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=float(cfg.data.openai_temperature),
        max_tokens=int(cfg.data.openai_max_tokens),
    )
    text = resp.choices[0].message.content.strip()
    templates = _extract_json_list(text)
    templates = _sanitize_templates(
        templates,
        placeholder=cfg.data.puzzle_placeholder,
        max_words=int(cfg.data.max_template_words),
    )
    if len(templates) < 3:
        raise RuntimeError("Too few prompt templates returned; retry with --force-prompts.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(templates, handle, indent=2)
    return templates


def _format_puzzle(puzzle: np.ndarray) -> str:
    return " ".join(str(int(x)) for x in puzzle.reshape(-1))


def _format_example(cfg, *, user_text: str, puzzle_str: str) -> str:
    assistant = f"<TRM-sudoku> {puzzle_str} </TRM-sudoku>"
    parts = [
        f"{cfg.data.system_prefix} {cfg.data.system_prompt}",
        f"{cfg.data.user_prefix} {user_text}",
        f"{cfg.data.assistant_prefix} {assistant}",
    ]
    return "\n".join(parts)


def _load_stage_source(stage_name: str) -> tuple[str, str, dict]:
    cfg_path = TRM_ROOT.parent / "v2" / "data_pipeline" / "Config.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Data pipeline config not found: {cfg_path}")
    dp_cfg = OmegaConf.load(cfg_path)
    if stage_name not in dp_cfg.stages:
        raise KeyError(f"Stage '{stage_name}' not found in {cfg_path}")
    stage = dp_cfg.stages[stage_name]
    sources = stage.get("sources", [])
    if not sources:
        raise ValueError(f"Stage '{stage_name}' has no sources.")
    source = sources[0]
    dataset_name = source.get("dataset_name")
    split = source.get("split", "train")
    return dataset_name, split, source


def _iter_text_rows(ds, source: dict, max_rows: int):
    text_field = source.get("text_field")
    join_fields = source.get("join_fields")
    join_separator = source.get("join_separator", " ")
    count = 0
    for row in ds:
        if text_field:
            text = row.get(text_field, "")
        elif join_fields:
            parts = [str(row.get(field, "")).strip() for field in join_fields]
            text = join_separator.join([p for p in parts if p])
        else:
            text = ""
        text = str(text).strip()
        if not text:
            continue
        yield text
        count += 1
        if max_rows and count >= max_rows:
            break


def _truncate_tokens(rng: np.random.Generator, ids: list[int], max_len: int) -> list[int]:
    if len(ids) <= max_len:
        return ids
    if max_len <= 0:
        return []
    start = int(rng.integers(0, len(ids) - max_len + 1))
    return ids[start : start + max_len]


def _make_chat_example(cfg, tokenizer, rng: np.random.Generator, text: str, max_seq_len: int) -> dict | None:
    words = text.split()
    max_words = int(cfg.data.mix.max_topic_words)
    topic = " ".join(words[:max_words]) if words else text
    template = rng.choice(cfg.data.mix.chat_templates)
    user_prompt = template.format(topic=topic)

    system = f"{cfg.data.system_prefix} {cfg.data.chat_system_prompt}"
    user_line = f"{cfg.data.user_prefix} {user_prompt}"
    assistant_line = f"{cfg.data.assistant_prefix} "
    prefix = "\n".join([system, user_line, assistant_line])

    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if len(prefix_ids) >= max_seq_len:
        return None
    response_ids = tokenizer.encode(text, add_special_tokens=False)
    available = max_seq_len - len(prefix_ids)
    response_ids = response_ids[:available]
    ids = prefix_ids + response_ids
    if not ids:
        return None
    return {"input_ids": ids, "length": len(ids)}


def _make_text_example(tokenizer, rng: np.random.Generator, text: str, max_seq_len: int) -> dict | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids = _truncate_tokens(rng, ids, max_seq_len)
    if not ids:
        return None
    return {"input_ids": ids, "length": len(ids)}


def _collect_text_examples(
    cfg,
    tokenizer,
    *,
    stage_names: list[str],
    count: int,
    max_seq_len: int,
    rng: np.random.Generator,
    mode: str,
) -> list[dict]:
    if count <= 0:
        return []
    examples: list[dict] = []
    stage_texts: list[list[str]] = []
    for stage_name in stage_names:
        dataset_name, split, source = _load_stage_source(stage_name)
        ds = load_dataset(dataset_name, split=split, streaming=False, cache_dir=cfg.data.hf_cache_dir)
        texts = list(_iter_text_rows(ds, source, int(cfg.data.mix.max_source_rows)))
        stage_texts.append(texts)

    while len(examples) < count:
        available = [idx for idx, texts in enumerate(stage_texts) if texts]
        if not available:
            break
        stage_idx = int(rng.choice(available))
        text = stage_texts[stage_idx].pop()
        if mode == "chat":
            example = _make_chat_example(cfg, tokenizer, rng, text, max_seq_len)
        else:
            example = _make_text_example(tokenizer, rng, text, max_seq_len)
        if example is not None:
            examples.append(example)
    return examples


def _tokenize_examples(cfg, tokenizer, puzzles: np.ndarray, templates: List[str]) -> List[dict]:
    rng = np.random.default_rng(int(cfg.data.seed))
    placeholder = cfg.data.puzzle_placeholder
    max_seq_len = max(stage.seq_len for stage in cfg.stages)

    examples = []
    for puzzle in tqdm(puzzles, desc="tokenizing", unit="sample"):
        puzzle_str = _format_puzzle(puzzle)
        template = templates[int(rng.integers(0, len(templates)))]
        user_text = template.replace(placeholder, puzzle_str)
        text = _format_example(cfg, user_text=user_text, puzzle_str=puzzle_str)
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > max_seq_len:
            continue
        examples.append({"input_ids": ids, "length": len(ids)})
    return examples


def _write_shards(examples: List[dict], out_dir: Path, shard_rows: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"stage": out_dir.name, "shards": []}
    shard_rows = max(1, int(shard_rows))

    for shard_idx in range(0, len(examples), shard_rows):
        shard = examples[shard_idx : shard_idx + shard_rows]
        if not shard:
            continue
        input_ids = pa.array([row["input_ids"] for row in shard], type=pa.list_(pa.int32()))
        lengths = pa.array([row["length"] for row in shard], type=pa.int32())
        table = pa.Table.from_arrays([input_ids, lengths], names=["input_ids", "length"])

        filename = f"shard_{shard_idx // shard_rows:05d}.arrow"
        path = out_dir / filename
        with pa.OSFile(str(path), "wb") as sink:
            with pa_ipc.new_file(sink, table.schema) as writer:
                writer.write(table)
        manifest["shards"].append({"filename": filename, "rows": len(shard)})

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _write_sample_rows(cfg, samples: List[dict], *, max_rows: int = 5) -> None:
    path = Path(cfg.data.samples_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in samples[:max_rows]:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    cfg = load_config()

    build_custom_tokenizer(force=False)
    tokenizer = load_tokenizer()

    templates = _load_or_create_templates(cfg, force=args.force_prompts)

    if args.seed is not None:
        cfg.data.seed = int(args.seed)
    max_seq_len = max(stage.seq_len for stage in cfg.stages)

    if getattr(cfg.data.mix, "enabled", False):
        mix_cfg = cfg.data.mix
        train_samples = int(args.train_samples or mix_cfg.train_samples)
        val_samples = int(args.val_samples or mix_cfg.val_samples)
        fractions = mix_cfg.fractions
        total_fraction = float(fractions.sudoku + fractions.chat + fractions.text)
        sudoku_frac = float(fractions.sudoku) / total_fraction
        chat_frac = float(fractions.chat) / total_fraction
        text_frac = float(fractions.text) / total_fraction

        sudoku_train = int(train_samples * sudoku_frac)
        chat_train = int(train_samples * chat_frac)
        text_train = train_samples - sudoku_train - chat_train
        sudoku_val = int(val_samples * sudoku_frac)
        chat_val = int(val_samples * chat_frac)
        text_val = val_samples - sudoku_val - chat_val

        dataset = generate_dataset(
            train_samples=sudoku_train,
            val_samples=sudoku_val,
            min_clues=int(cfg.data.min_clues),
            seed=int(cfg.data.seed),
            show_progress=True,
        )

        rng = np.random.default_rng(int(cfg.data.seed))
        train_examples = _tokenize_examples(cfg, tokenizer, dataset.train_puzzle, templates)
        val_examples = _tokenize_examples(cfg, tokenizer, dataset.val_puzzle, templates)

        train_examples += _collect_text_examples(
            cfg,
            tokenizer,
            stage_names=list(mix_cfg.general_stages),
            count=text_train,
            max_seq_len=max_seq_len,
            rng=rng,
            mode="text",
        )
        train_examples += _collect_text_examples(
            cfg,
            tokenizer,
            stage_names=list(mix_cfg.chat_stages),
            count=chat_train,
            max_seq_len=max_seq_len,
            rng=rng,
            mode="chat",
        )

        rng_val = np.random.default_rng(int(cfg.data.seed) + 1)
        val_examples += _collect_text_examples(
            cfg,
            tokenizer,
            stage_names=list(mix_cfg.general_stages),
            count=text_val,
            max_seq_len=max_seq_len,
            rng=rng_val,
            mode="text",
        )
        val_examples += _collect_text_examples(
            cfg,
            tokenizer,
            stage_names=list(mix_cfg.chat_stages),
            count=chat_val,
            max_seq_len=max_seq_len,
            rng=rng_val,
            mode="chat",
        )

        rng.shuffle(train_examples)
        rng_val.shuffle(val_examples)
        dataset_root = Path(cfg.paths.processed_data_root) / mix_cfg.dataset_name
    else:
        train_samples = int(args.train_samples or cfg.data.train_samples)
        val_samples = int(args.val_samples or cfg.data.val_samples)
        dataset = generate_dataset(
            train_samples=train_samples,
            val_samples=val_samples,
            min_clues=int(cfg.data.min_clues),
            seed=int(cfg.data.seed),
            show_progress=True,
        )
        train_examples = _tokenize_examples(cfg, tokenizer, dataset.train_puzzle, templates)
        val_examples = _tokenize_examples(cfg, tokenizer, dataset.val_puzzle, templates)
        dataset_root = Path(cfg.paths.processed_data_root) / cfg.data.dataset_name
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    if dataset_root.exists() and not args.force_dataset:
        print(f"Dataset already exists at {dataset_root}. Use --force-dataset to overwrite.")
        return

    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    if not train_examples:
        raise RuntimeError("No training examples were tokenized; increase seq_len or shorten prompts.")

    _write_shards(train_examples, train_dir, cfg.data.shard_rows)
    _write_shards(val_examples, val_dir, cfg.data.shard_rows)

    _write_sample_rows(cfg, train_examples)

    lengths = [row["length"] for row in train_examples]
    print(
        f"Train rows: {len(train_examples)} | len min/avg/max "
        f"{min(lengths)}/{np.mean(lengths):.1f}/{max(lengths)}"
    )
    print(f"Wrote dataset to {dataset_root}")


if __name__ == "__main__":
    main()

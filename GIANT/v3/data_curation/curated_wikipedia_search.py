from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from datasets import load_dataset
import yaml

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    tqdm = None


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "where", "who", "why", "with", "which", "known", "about", "tell", "me", "explain",
}


def _normalize_text(value: str) -> str:
    lowered = value.lower()
    cleaned = re.sub(rf"[{re.escape(string.punctuation)}]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _tokenize_words(value: str) -> List[str]:
    return [token for token in _normalize_text(value).split() if token]


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normalize_text(value)).strip("_")


def _set_hf_cache(hf_cache_root: Optional[str]) -> None:
    if not hf_cache_root:
        return
    hf_cache = str(Path(hf_cache_root))
    os.environ["HF_HOME"] = hf_cache
    os.environ["HF_DATASETS_CACHE"] = str(Path(hf_cache) / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(hf_cache) / "transformers")


def _bool_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass
class Target:
    id: str
    canonical_name: str
    aliases: List[str]
    prompts: List[str]
    allow_list_pages: bool = False
    aliases_normalized: List[str] = field(default_factory=list)
    canonical_normalized: str = ""
    canonical_tokens: List[str] = field(default_factory=list)
    prompt_keywords: List[str] = field(default_factory=list)
    rare_keywords: List[str] = field(default_factory=list)


@dataclass
class Candidate:
    target_id: str
    canonical_name: str
    matched_alias: str
    article_id: str
    title: str
    url: str
    score_heuristic: float
    score_embed: Optional[float]
    text_lead: str
    text_full_truncated: str
    text: str
    row_index: int
    page_kind: str


@dataclass
class TargetState:
    target: Target
    candidates: Dict[str, Candidate] = field(default_factory=dict)
    near_dup_counts: Dict[str, int] = field(default_factory=dict)
    listlike_count: int = 0

    def add(self, candidate: Candidate, *, top_k: int) -> None:
        existing = self.candidates.get(candidate.article_id)
        if existing is not None and existing.score_heuristic >= candidate.score_heuristic:
            return

        fingerprint = _near_dup_key(candidate.text_full_truncated)
        is_listlike = candidate.page_kind in {"disambiguation", "list", "index"}

        old_fingerprint = None
        if existing is not None:
            old_fingerprint = _near_dup_key(existing.text_full_truncated)
            self.near_dup_counts[old_fingerprint] = max(0, self.near_dup_counts.get(old_fingerprint, 1) - 1)
            if self.near_dup_counts[old_fingerprint] == 0:
                self.near_dup_counts.pop(old_fingerprint, None)
            if existing.page_kind in {"disambiguation", "list", "index"}:
                self.listlike_count = max(0, self.listlike_count - 1)

        if self.near_dup_counts.get(fingerprint, 0) >= 2:
            if existing is not None and old_fingerprint is not None:
                self.near_dup_counts[old_fingerprint] = self.near_dup_counts.get(old_fingerprint, 0) + 1
                if existing.page_kind in {"disambiguation", "list", "index"}:
                    self.listlike_count += 1
            return
        if is_listlike and self.listlike_count >= 1 and existing is None:
            return

        self.candidates[candidate.article_id] = candidate
        self.near_dup_counts[fingerprint] = self.near_dup_counts.get(fingerprint, 0) + 1
        if is_listlike:
            self.listlike_count += 1

        if len(self.candidates) > top_k:
            worst = min(self.candidates.values(), key=lambda item: (item.score_heuristic, item.title.lower()))
            removed = self.candidates.pop(worst.article_id)
            removed_fp = _near_dup_key(removed.text_full_truncated)
            self.near_dup_counts[removed_fp] = max(0, self.near_dup_counts.get(removed_fp, 1) - 1)
            if self.near_dup_counts[removed_fp] == 0:
                self.near_dup_counts.pop(removed_fp, None)
            if removed.page_kind in {"disambiguation", "list", "index"}:
                self.listlike_count = max(0, self.listlike_count - 1)

    def sorted_candidates(self) -> List[Candidate]:
        return sorted(
            self.candidates.values(),
            key=lambda item: (
                item.score_heuristic + 20.0 * (item.score_embed if item.score_embed is not None else 0.0),
                item.score_heuristic,
                item.score_embed if item.score_embed is not None else float("-inf"),
            ),
            reverse=True,
        )


class AliasMatcher:
    def __init__(self, alias_to_targets: Dict[str, List[str]]) -> None:
        self.alias_to_targets = alias_to_targets
        self.backend = "regex"
        self._automaton = None
        self._regex = None

        padded_aliases = [f" {alias} " for alias in alias_to_targets if alias]
        try:
            import ahocorasick  # type: ignore

            automaton = ahocorasick.Automaton()
            for alias in padded_aliases:
                automaton.add_word(alias, alias.strip())
            automaton.make_automaton()
            self._automaton = automaton
            self.backend = "ahocorasick"
            return
        except Exception:
            pass

        if padded_aliases:
            alternation = "|".join(re.escape(alias) for alias in sorted(padded_aliases, key=len, reverse=True))
            self._regex = re.compile(alternation)

    def match(self, normalized_text: str) -> Dict[str, List[str]]:
        haystack = f" {normalized_text} "
        by_target: Dict[str, List[str]] = defaultdict(list)
        if self._automaton is not None:
            for _, alias in self._automaton.iter(haystack):
                for target_id in self.alias_to_targets.get(alias, []):
                    by_target[target_id].append(alias)
            return by_target
        if self._regex is None:
            return by_target
        for match in self._regex.finditer(haystack):
            alias = match.group(0).strip()
            for target_id in self.alias_to_targets.get(alias, []):
                by_target[target_id].append(alias)
        return by_target


def _near_dup_key(text: str) -> str:
    normalized = _normalize_text(text[:400])
    return normalized[:240]


def _classify_page(title: str) -> str:
    lowered = title.lower()
    if "(disambiguation)" in lowered:
        return "disambiguation"
    if "list of" in lowered:
        return "list"
    if "index of" in lowered:
        return "index"
    return "article"


def _load_targets(path: Path) -> List[Target]:
    if not path.exists():
        raise FileNotFoundError(f"Targets file not found: {path}")
    text = path.read_text(encoding="utf-8")
    items: Any
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        items = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif suffix in {".yml", ".yaml"}:
        parsed = yaml.safe_load(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("targets"), list):
            items = parsed["targets"]
        elif isinstance(parsed, list):
            items = parsed
        else:
            raise ValueError("YAML targets file must be a sequence or mapping with 'targets'.")
    else:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("targets"), list):
            items = parsed["targets"]
        elif isinstance(parsed, list):
            items = parsed
        else:
            raise ValueError("Targets file must be a JSON array, JSON object with 'targets', or JSONL.")

    if not isinstance(items, list) or not items:
        raise ValueError("Targets file is empty or invalid.")

    targets: List[Target] = []
    keyword_frequency: Counter[str] = Counter()
    for raw in items:
        canonical_name = str(raw["canonical_name"]).strip()
        aliases = [str(alias).strip() for alias in raw.get("aliases", []) if str(alias).strip()]
        prompts = [str(prompt).strip() for prompt in raw.get("prompts", []) if str(prompt).strip()]
        canonical_normalized = _normalize_text(canonical_name)
        aliases_normalized = sorted({canonical_normalized, *(_normalize_text(alias) for alias in aliases if alias.strip())})
        canonical_tokens = [token for token in canonical_normalized.split() if token]
        prompt_keywords = []
        for prompt in prompts:
            for token in _tokenize_words(prompt):
                if len(token) < 4 or token in STOPWORDS:
                    continue
                prompt_keywords.append(token)
                keyword_frequency[token] += 1
        targets.append(
            Target(
                id=str(raw.get("id") or _safe_id(canonical_name)),
                canonical_name=canonical_name,
                aliases=aliases,
                prompts=prompts,
                allow_list_pages=bool(raw.get("allow_list_pages", False)),
                aliases_normalized=aliases_normalized,
                canonical_normalized=canonical_normalized,
                canonical_tokens=canonical_tokens,
                prompt_keywords=sorted(set(prompt_keywords)),
            )
        )

    for target in targets:
        target.rare_keywords = [
            keyword for keyword in target.prompt_keywords if keyword_frequency[keyword] <= 2
        ][:12]
    return targets


def _alias_index(targets: Sequence[Target]) -> Dict[str, List[str]]:
    alias_to_targets: Dict[str, List[str]] = defaultdict(list)
    for target in targets:
        for alias in target.aliases_normalized:
            alias_to_targets[alias].append(target.id)
    return alias_to_targets


def _best_alias(target: Target, matched_aliases: Sequence[str], title_n: str, lead_n: str) -> str:
    unique_aliases = sorted(set(matched_aliases), key=len, reverse=True)
    for alias in unique_aliases:
        if title_n == alias:
            return alias
    for alias in unique_aliases:
        if f" {alias} " in f" {title_n} ":
            return alias
    for alias in unique_aliases:
        if f" {alias} " in f" {lead_n} ":
            return alias
    return unique_aliases[0] if unique_aliases else target.canonical_normalized


def _definitional_bonus(target: Target, lead_n: str) -> int:
    first_sentence = lead_n.split(" ", 24)
    snippet = " ".join(first_sentence)
    for alias in target.aliases_normalized:
        if snippet.startswith(f"{alias} is ") or snippet.startswith(f"{alias} was "):
            return 8
        if snippet.startswith(f"{alias} are ") or snippet.startswith(f"{alias} were "):
            return 6
    return 0


def _score_target(
    *,
    target: Target,
    matched_aliases: Sequence[str],
    title: str,
    title_n: str,
    lead_n: str,
    text_len: int,
) -> tuple[float, str]:
    score = 0.0
    matched_alias = _best_alias(target, matched_aliases, title_n, lead_n)
    title_tokens = title_n.split()
    alias_tokens = matched_alias.split()
    title_is_exact = title_n == target.canonical_normalized or title_n == matched_alias
    if title_n == target.canonical_normalized or title_n == matched_alias:
        score += 100
    elif f" {matched_alias} " in f" {title_n} ":
        score += 40

    overlap = len(set(target.canonical_tokens) & set(title_n.split()))
    score += min(20, overlap * 8)
    if title_tokens:
        score += 15.0 * (overlap / len(title_tokens))

    if title_n != matched_alias and f" {matched_alias} " in f" {title_n} ":
        extra_title_tokens = max(0, len(title_tokens) - len(alias_tokens))
        score -= min(24, extra_title_tokens * 4)

    if f" {matched_alias} " in f" {lead_n} ":
        score += 10

    if text_len < 800:
        score -= 10

    page_kind = _classify_page(title)
    if page_kind != "article":
        score -= 10

    definitional_bonus = _definitional_bonus(target, lead_n)
    keyword_hits = 0
    if title_is_exact or definitional_bonus > 0:
        keyword_hits = sum(
            1 for keyword in target.rare_keywords if f" {keyword} " in f" {lead_n} " or f" {keyword} " in f" {title_n} "
        )
        score += min(20, keyword_hits * 5)
    score += definitional_bonus
    if not title_is_exact and keyword_hits == 0 and definitional_bonus == 0:
        score -= 50
    return score, matched_alias


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_state(
    *,
    state_path: Path,
    rows_processed: int,
    target_states: Dict[str, TargetState],
    matcher_backend: str,
) -> None:
    payload = {
        "version": 1,
        "rows_processed": rows_processed,
        "matcher_backend": matcher_backend,
        "targets": {
            target_id: [asdict(candidate) for candidate in target_state.sorted_candidates()]
            for target_id, target_state in target_states.items()
        },
    }
    _save_json(state_path, payload)


def _load_state(state_path: Path, targets: Sequence[Target]) -> tuple[int, Dict[str, TargetState], Optional[str]]:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    targets_by_id = {target.id: target for target in targets}
    target_states = {target.id: TargetState(target=target) for target in targets}
    for target_id, candidates in payload.get("targets", {}).items():
        target_state = target_states.get(target_id)
        if target_state is None:
            continue
        for raw_candidate in candidates:
            candidate = Candidate(**raw_candidate)
            target_state.add(candidate, top_k=max(len(candidates), 1))
    return int(payload.get("rows_processed", 0)), target_states, payload.get("matcher_backend")


def _write_candidates(out_path: Path, target_states: Dict[str, TargetState]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for target_id in sorted(target_states):
            for candidate in target_states[target_id].sorted_candidates():
                handle.write(json.dumps(asdict(candidate), ensure_ascii=False) + "\n")
                total += 1
    return total


def _coverage_report(
    *,
    targets: Sequence[Target],
    target_states: Dict[str, TargetState],
    rows_processed: int,
    matcher_backend: str,
    use_embeddings: bool,
) -> Dict[str, Any]:
    num_hits_per_target = {target.id: len(target_states[target.id].candidates) for target in targets}
    best_title_per_target = {
        target.id: (target_states[target.id].sorted_candidates()[0].title if target_states[target.id].candidates else None)
        for target in targets
    }
    found = [target.id for target in targets if num_hits_per_target[target.id] > 0]
    missing = [target.id for target in targets if num_hits_per_target[target.id] == 0]
    warnings = []
    if missing:
        warnings.append(f"{len(missing)} targets have no retained candidates yet.")
    return {
        "rows_processed": rows_processed,
        "matcher_backend": matcher_backend,
        "targets_found": found,
        "targets_missing": missing,
        "num_hits_per_target": num_hits_per_target,
        "best_title_per_target": best_title_per_target,
        "warnings": warnings,
    }


def _get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"


def _build_query_text(target: Target) -> str:
    lines = [target.canonical_name]
    if target.aliases:
        lines.append("Aliases: " + ", ".join(target.aliases[:12]))
    if target.prompts:
        lines.append("Questions: " + " | ".join(target.prompts[:8]))
    return "\n".join(lines)


def _load_embedding_stack():
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional local env
        raise RuntimeError(
            "Embedding rerank requires torch and transformers. Use a Python env with those packages installed."
        ) from exc
    return torch, F, AutoModel, AutoTokenizer


def _select_embedding_device(torch, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _select_embedding_dtype(torch, device: str, requested: str):
    if requested == "auto":
        if device == "cuda":
            return torch.float16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if requested not in mapping:
        raise ValueError(f"Unsupported embedding dtype: {requested}")
    return mapping[requested]


def _last_token_pool(last_hidden_states, attention_mask, torch):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    device = last_hidden_states.device
    return last_hidden_states[
        torch.arange(batch_size, device=device),
        sequence_lengths,
    ]


def _encode_embeddings(
    *,
    torch,
    F,
    tokenizer,
    model,
    texts: Sequence[str],
    batch_size: int,
    max_length: int,
    device: str,
):
    if not texts:
        return None
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            pooled = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"], torch)
            pooled = F.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.detach().cpu())
    return torch.cat(chunks, dim=0)


def _rerank_with_embeddings(args: argparse.Namespace, targets: Sequence[Target], target_states: Dict[str, TargetState]) -> None:
    torch, F, AutoModel, AutoTokenizer = _load_embedding_stack()
    device = _select_embedding_device(torch, args.embedding_device)
    dtype = _select_embedding_dtype(torch, device, args.embedding_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model, padding_side="left")
    model_kwargs: Dict[str, Any] = {}
    if device != "cpu":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(args.embedding_model, **model_kwargs)
    model.to(device)
    model.eval()

    available_targets = [target for target in targets if target_states[target.id].candidates]
    progress = tqdm(total=len(available_targets), desc="embed_rerank", unit="targets") if tqdm is not None else None
    for target in available_targets:
        target_state = target_states[target.id]
        candidates = target_state.sorted_candidates()
        query = _get_detailed_instruct(args.embedding_task, _build_query_text(target))
        query_embedding = _encode_embeddings(
            torch=torch,
            F=F,
            tokenizer=tokenizer,
            model=model,
            texts=[query],
            batch_size=1,
            max_length=args.embedding_max_length,
            device=device,
        )
        assert query_embedding is not None
        doc_texts = [f"{candidate.title}\n\n{candidate.text_lead[: args.embedding_doc_chars]}" for candidate in candidates]
        doc_embeddings = _encode_embeddings(
            torch=torch,
            F=F,
            tokenizer=tokenizer,
            model=model,
            texts=doc_texts,
            batch_size=args.embedding_batch_size,
            max_length=args.embedding_max_length,
            device=device,
        )
        assert doc_embeddings is not None
        scores = (query_embedding @ doc_embeddings.T).squeeze(0).tolist()
        for candidate, score in zip(candidates, scores):
            candidate.score_embed = float(score)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()


def _print_sample_hits(targets: Sequence[Target], target_states: Dict[str, TargetState], *, seed: int, sample_count: int) -> None:
    available = [target for target in targets if target_states[target.id].candidates]
    if not available:
        print("No sample hits to show yet.")
        return
    rng = random.Random(seed)
    sample = rng.sample(available, min(sample_count, len(available)))
    print("\nSample hits:")
    for target in sample:
        best = target_states[target.id].sorted_candidates()[0]
        print(f"- {target.canonical_name}: {best.title} (score={best.score_heuristic:.1f})")


def _iter_wikipedia_rows(args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    ds = load_dataset(
        "wikimedia/wikipedia",
        args.dataset_config,
        split=args.split,
        streaming=True,
    )
    if args.shuffle_streaming:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
    return iter(ds)


def _stream_search(args: argparse.Namespace) -> tuple[List[Target], Dict[str, TargetState], int, str]:
    targets = _load_targets(Path(args.targets))
    target_states: Dict[str, TargetState]
    state_path = Path(args.checkpoint_dir) / "state.json"
    if args.overwrite and state_path.exists():
        state_path.unlink()
    if state_path.exists():
        rows_processed, target_states, prior_backend = _load_state(state_path, targets)
    else:
        rows_processed = 0
        target_states = {target.id: TargetState(target=target) for target in targets}
        prior_backend = None

    alias_to_targets = _alias_index(targets)
    matcher = AliasMatcher(alias_to_targets)
    matcher_backend = prior_backend or matcher.backend

    iterator = _iter_wikipedia_rows(args)
    if rows_processed > 0:
        iterator = islice(iterator, rows_processed, None)

    progress = None
    if tqdm is not None:
        total = args.max_rows if args.max_rows and rows_processed == 0 else None
        progress = tqdm(total=total, initial=0, desc="wikipedia", unit="rows")

    processed_this_run = 0
    for row in iterator:
        if args.max_rows is not None and rows_processed + processed_this_run >= args.max_rows:
            break

        title = str(row.get("title") or "").strip()
        text = str(row.get("text") or "").strip()
        url = str(row.get("url") or "")
        article_id = str(row.get("id") or f"row-{rows_processed + processed_this_run}")
        if not title or not text:
            processed_this_run += 1
            if progress is not None:
                progress.update(1)
            continue

        page_kind = _classify_page(title)
        lead = text[: args.lead_chars]
        full_truncated = text[: args.full_chars]
        title_n = _normalize_text(title)
        lead_n = _normalize_text(lead)
        matched = matcher.match(f"{title_n} {lead_n}")
        for target_id, matched_aliases in matched.items():
            target_state = target_states[target_id]
            target = target_state.target
            if page_kind != "article" and not target.allow_list_pages:
                continue
            score, matched_alias = _score_target(
                target=target,
                matched_aliases=matched_aliases,
                title=title,
                title_n=title_n,
                lead_n=lead_n,
                text_len=len(text),
            )
            if score < args.min_score:
                continue
            candidate = Candidate(
                target_id=target.id,
                canonical_name=target.canonical_name,
                matched_alias=matched_alias,
                article_id=article_id,
                title=title,
                url=url,
                score_heuristic=score,
                score_embed=None,
                text_lead=lead,
                text_full_truncated=full_truncated,
                text=f"{title}\n\n{full_truncated}",
                row_index=rows_processed + processed_this_run,
                page_kind=page_kind,
            )
            target_state.add(candidate, top_k=args.top_k)

        processed_this_run += 1
        if progress is not None:
            progress.update(1)
        if processed_this_run % args.checkpoint_every_rows == 0:
            _save_state(
                state_path=state_path,
                rows_processed=rows_processed + processed_this_run,
                target_states=target_states,
                matcher_backend=matcher_backend,
            )

    if progress is not None:
        progress.close()

    final_rows = rows_processed + processed_this_run
    _save_state(
        state_path=state_path,
        rows_processed=final_rows,
        target_states=target_states,
        matcher_backend=matcher_backend,
    )
    return targets, target_states, final_rows, matcher_backend


def _load_search_state_only(args: argparse.Namespace) -> tuple[List[Target], Dict[str, TargetState], int, str]:
    targets = _load_targets(Path(args.targets))
    state_path = Path(args.checkpoint_dir) / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Checkpoint state not found for --skip-streaming: {state_path}")
    rows_processed, target_states, prior_backend = _load_state(state_path, targets)
    return targets, target_states, rows_processed, (prior_backend or "regex")


def _write_outputs(
    *,
    args: argparse.Namespace,
    targets: Sequence[Target],
    target_states: Dict[str, TargetState],
    rows_processed: int,
    matcher_backend: str,
) -> None:
    out_path = Path(args.out)
    coverage_path = Path(args.coverage_report) if args.coverage_report else out_path.with_name(out_path.stem + "_coverage.json")
    if args.use_embeddings:
        _rerank_with_embeddings(args, targets, target_states)
    out_count = _write_candidates(out_path, target_states)
    report = _coverage_report(
        targets=targets,
        target_states=target_states,
        rows_processed=rows_processed,
        matcher_backend=matcher_backend,
        use_embeddings=args.use_embeddings,
    )
    report["output_candidates"] = out_count
    _save_json(coverage_path, report)
    _print_sample_hits(targets, target_states, seed=args.seed, sample_count=args.sample_target_count)
    print(f"\nWrote {out_count} candidate rows to {out_path}")
    print(f"Coverage report: {coverage_path}")


def _sample_specs() -> List[tuple[str, List[str], List[str]]]:
    return [
        ("Albert Einstein", ["Einstein"], ["Who developed the theory of relativity?"]),
        ("Isaac Newton", ["Newton"], ["Who formulated the laws of motion and gravity?"]),
        ("Marie Curie", ["Curie"], ["Who pioneered research on radioactivity?"]),
        ("Ada Lovelace", ["Lovelace"], ["Who is often described as the first computer programmer?"]),
        ("Alan Turing", ["Turing"], ["Who helped lay the foundations of computer science?"]),
        ("Charles Darwin", ["Darwin"], ["Who is associated with evolution by natural selection?"]),
        ("Nikola Tesla", ["Tesla"], ["Who is known for work on alternating current?"]),
        ("Thomas Edison", ["Edison"], ["Who is famous for inventions like the phonograph and improved light bulb?"]),
        ("Galileo Galilei", ["Galileo"], ["Who improved the telescope and defended heliocentrism?"]),
        ("Leonardo da Vinci", ["da Vinci", "Leonardo"], ["Who painted the Mona Lisa and drew flying machines?"]),
        ("William Shakespeare", ["Shakespeare"], ["Who wrote Hamlet and Macbeth?"]),
        ("Ludwig van Beethoven", ["Beethoven"], ["Who composed the Fifth Symphony?"]),
        ("Wolfgang Amadeus Mozart", ["Mozart"], ["Who was the classical composer of The Magic Flute?"]),
        ("Michelangelo", ["Michelangelo Buonarroti"], ["Who painted the Sistine Chapel ceiling?"]),
        ("Vincent van Gogh", ["van Gogh"], ["Who painted Starry Night?"]),
        ("Pablo Picasso", ["Picasso"], ["Who co-founded Cubism?"]),
        ("Nelson Mandela", ["Mandela"], ["Who became South Africa's first Black president?"]),
        ("Mahatma Gandhi", ["Gandhi"], ["Who led Indian independence through nonviolent resistance?"]),
        ("Martin Luther King Jr.", ["Martin Luther King", "MLK"], ["Who delivered the I Have a Dream speech?"]),
        ("Abraham Lincoln", ["Lincoln"], ["Who was U.S. president during the American Civil War?"]),
        ("George Washington", ["Washington"], ["Who was the first president of the United States?"]),
        ("Julius Caesar", ["Caesar"], ["Who was the Roman dictator assassinated in 44 BC?"]),
        ("Cleopatra", ["Cleopatra VII"], ["Who was the last active ruler of the Ptolemaic Kingdom of Egypt?"]),
        ("Napoleon Bonaparte", ["Napoleon"], ["Who became emperor of the French in the early 19th century?"]),
        ("Joan of Arc", ["Jeanne d'Arc"], ["Who was the French heroine of the Hundred Years' War?"]),
        ("Winston Churchill", ["Churchill"], ["Who was Britain's prime minister during much of World War II?"]),
        ("Queen Elizabeth II", ["Elizabeth II"], ["Who was the longest-reigning British monarch?"]),
        ("Genghis Khan", ["Chinggis Khan"], ["Who founded the Mongol Empire?"]),
        ("Alexander the Great", ["Alexander III of Macedon"], ["Who created a vast empire across Asia and the Mediterranean?"]),
        ("Florence Nightingale", ["Nightingale"], ["Who modernized nursing during the Crimean War?"]),
        ("Jane Austen", ["Austen"], ["Who wrote Pride and Prejudice?"]),
        ("Mark Twain", ["Twain", "Samuel Clemens"], ["Who wrote Adventures of Huckleberry Finn?"]),
        ("Frida Kahlo", ["Kahlo"], ["Who was the Mexican painter known for self-portraits?"]),
        ("Amelia Earhart", ["Earhart"], ["Who was the pioneering aviator who disappeared over the Pacific?"]),
        ("Steve Jobs", ["Jobs"], ["Who co-founded Apple?"]),
        ("Bill Gates", ["Gates"], ["Who co-founded Microsoft?"]),
        ("Paris", [], ["What city is the capital of France?"]),
        ("London", [], ["What city is the capital of the United Kingdom?"]),
        ("Rome", [], ["What city is the capital of Italy?"]),
        ("Tokyo", [], ["What city is the capital of Japan?"]),
        ("Beijing", [], ["What city is the capital of China?"]),
        ("New York City", ["New York"], ["What is the most populous city in the United States?"]),
        ("Los Angeles", ["LA"], ["What city is known for Hollywood?"]),
        ("Cairo", [], ["What city is the capital of Egypt?"]),
        ("Athens", [], ["What city is the capital of Greece?"]),
        ("Jerusalem", [], ["What city is holy to Judaism, Christianity, and Islam?"]),
        ("India", [], ["Which country has New Delhi as its capital?"]),
        ("China", [], ["Which country has the world's largest population for much of modern history?"]),
        ("United States", ["USA", "United States of America"], ["Which country has Washington, D.C. as its capital?"]),
        ("France", [], ["Which country has Paris as its capital?"]),
        ("Japan", [], ["Which country has Tokyo as its capital?"]),
        ("Brazil", [], ["Which country contains most of the Amazon rainforest?"]),
        ("Egypt", [], ["Which country is home to the pyramids of Giza?"]),
        ("Australia", [], ["Which country is also a continent?"]),
        ("Canada", [], ["Which country lies north of the United States?"]),
        ("Germany", [], ["Which country has Berlin as its capital?"]),
        ("Italy", [], ["Which country is shaped like a boot?"]),
        ("Russia", [], ["Which country spans Europe and Asia and is the largest by area?"]),
        ("South Africa", [], ["Which country has Cape Town, Pretoria, and Bloemfontein as capitals?"]),
        ("Mexico", [], ["Which country lies south of the United States and has Mexico City as its capital?"]),
        ("Antarctica", [], ["Which continent surrounds the South Pole?"]),
        ("Sun", [], ["What star is at the center of the Solar System?"]),
        ("Moon", ["Earth's Moon"], ["What natural satellite orbits Earth?"]),
        ("Earth", [], ["Which planet do humans live on?"]),
        ("Mars", [], ["Which planet is known as the red planet?"]),
        ("Jupiter", [], ["Which planet is the largest in the Solar System?"]),
        ("Saturn", [], ["Which planet is famous for its rings?"]),
        ("Periodic table", ["Periodic table of elements"], ["What chart organizes the chemical elements?"]),
        ("Theory of relativity", ["Relativity"], ["What theory explains gravity and high-speed motion in modern physics?"]),
        ("Gravity", [], ["What force pulls objects toward each other?"]),
        ("Photosynthesis", [], ["What process lets plants turn sunlight into chemical energy?"]),
        ("DNA", ["Deoxyribonucleic acid"], ["What molecule carries genetic information?"]),
        ("Atom", [], ["What is the basic unit of ordinary matter?"]),
        ("Black hole", [], ["What object has gravity so strong that not even light can escape?"]),
        ("Big Bang", ["Big Bang theory"], ["What model describes the early expansion of the universe?"]),
        ("Plate tectonics", [], ["What theory explains the movement of Earth's crustal plates?"]),
        ("Evolution", [], ["What biological process explains how species change over time?"]),
        ("Internet", [], ["What global network connects computers worldwide?"]),
        ("Computer", [], ["What machine processes data using instructions?"]),
        ("Python (programming language)", ["Python programming language", "Python"], ["What programming language is known for readable syntax and indentation?"]),
        ("Artificial intelligence", ["AI"], ["What field studies machines that perform tasks associated with human intelligence?"]),
        ("French Revolution", [], ["What revolution began in 1789 and overthrew the French monarchy?"]),
        ("American Revolution", ["United States War of Independence"], ["What war led to the independence of the United States?"]),
        ("World War I", ["First World War", "WWI"], ["What global war lasted from 1914 to 1918?"]),
        ("World War II", ["Second World War", "WWII"], ["What global war lasted from 1939 to 1945?"]),
        ("Cold War", [], ["What geopolitical struggle followed World War II between the United States and the Soviet Union?"]),
        ("Renaissance", [], ["What cultural rebirth began in Italy after the Middle Ages?"]),
        ("Industrial Revolution", [], ["What era transformed manufacturing with machines and factories?"]),
        ("Apollo 11", [], ["What mission first landed humans on the Moon?"]),
        ("Titanic", ["RMS Titanic"], ["What ocean liner sank on its maiden voyage in 1912?"]),
        ("Great Wall of China", ["Great Wall"], ["What famous fortification stretches across northern China?"]),
        ("Roman Empire", [], ["What ancient empire ruled much of Europe and the Mediterranean?"]),
        ("United Nations", ["UN"], ["What international organization was founded in 1945 to promote peace?"]),
        ("European Union", ["EU"], ["What political and economic union links many European countries?"]),
        ("Nobel Prize", ["Nobel prizes"], ["What award honors achievements in science, literature, and peace?"]),
        ("Olympic Games", ["Olympics"], ["What international multi-sport event is held every four years?"]),
        ("World Wide Web", ["WWW", "Web"], ["What system of linked documents runs on the internet?"]),
        ("Mona Lisa", [], ["What portrait by Leonardo da Vinci hangs in the Louvre?"]),
        ("Hamlet", [], ["What Shakespeare play features the line To be, or not to be?"]),
        ("The Beatles", ["Beatles"], ["What British band featured John Lennon and Paul McCartney?"]),
    ]


def build_sample_targets() -> List[Dict[str, Any]]:
    sample_targets = []
    for canonical_name, aliases, prompts in _sample_specs():
        sample_targets.append(
            {
                "id": _safe_id(canonical_name),
                "canonical_name": canonical_name,
                "aliases": aliases,
                "prompts": prompts,
            }
        )
    return sample_targets


def _write_sample_targets(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_targets = build_sample_targets()
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for target in sample_targets:
                handle.write(json.dumps(target, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(sample_targets, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(sample_targets)} sample targets to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream English Wikipedia once and build target-aware curated passages.")
    parser.add_argument("--targets", type=str, help="Path to targets JSON/JSONL file")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, required=False, default="/tmp/giant_v3_curated_ckpt")
    parser.add_argument("--out", type=str, required=False, default="/tmp/giant_v3_curated_candidates.jsonl")
    parser.add_argument("--coverage-report", type=str, default=None)
    parser.add_argument("--dataset-config", type=str, default="20231101.en")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lead-chars", type=int, default=2000)
    parser.add_argument("--full-chars", type=int, default=4000)
    parser.add_argument("--checkpoint-every-rows", type=int, default=5000)
    parser.add_argument("--min-score", type=float, default=35.0)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--shuffle-streaming", type=_bool_flag, default=False)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--use-embeddings", type=_bool_flag, default=False)
    parser.add_argument("--embedding-model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--embedding-task", type=str, default="Given a target concept, retrieve the Wikipedia article most directly about that target")
    parser.add_argument("--embedding-device", type=str, default="auto")
    parser.add_argument("--embedding-dtype", type=str, default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=2048)
    parser.add_argument("--embedding-doc-chars", type=int, default=2048)
    parser.add_argument("--sample-target-count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-cache-root", type=str, default="/proj/giant-data/hf_cache")
    parser.add_argument("--skip-streaming", action="store_true", help="Load checkpoint state and skip the dataset pass.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-sample-targets", type=str, default=None)
    args = parser.parse_args()
    if args.write_sample_targets:
        return args
    if not args.targets:
        parser.error("--targets is required unless --write-sample-targets is used")
    return args


def main() -> None:
    args = parse_args()
    if args.write_sample_targets:
        _write_sample_targets(Path(args.write_sample_targets))
        return
    _set_hf_cache(args.hf_cache_root)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    if args.skip_streaming:
        targets, target_states, rows_processed, matcher_backend = _load_search_state_only(args)
    else:
        targets, target_states, rows_processed, matcher_backend = _stream_search(args)
    _write_outputs(
        args=args,
        targets=targets,
        target_states=target_states,
        rows_processed=rows_processed,
        matcher_backend=matcher_backend,
    )


if __name__ == "__main__":
    main()

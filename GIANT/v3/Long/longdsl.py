from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast


ROLE_PREFIX = "@{role}"
TURN_SUFFIX = " SEP"

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
BASE_TOKENS = [
    "@user",
    "@assistant",
    "LEVEL",
    "PROGRAM",
    "QUERY",
    "ANS",
    "SEP",
    "L1",
    "L2",
    "L3",
    "DEF",
    "SET",
    "ALIAS",
    "ASK",
    "INC",
    "DEC",
    "SWAP",
    "DEFARR",
    "SETAT",
    "SWAPAT",
    "INCAT",
    "GET",
]


@dataclass
class TokenizerSpec:
    num_entities: int = 128
    num_values: int = 64
    num_arrays: int = 32
    array_width: int = 4
    num_deltas: int = 8


@dataclass
class GeneratorConfig:
    level: int
    context_length: int
    fill_ratio: float = 0.92
    num_entities: int = 128
    num_values: int = 64
    num_arrays: int = 32
    array_width: int = 4
    num_deltas: int = 8


def entity_token(idx: int) -> str:
    return f"E{idx:03d}"


def value_token(idx: int) -> str:
    return f"V{idx:03d}"


def array_token(idx: int) -> str:
    return f"A{idx:03d}"


def index_token(idx: int) -> str:
    return f"I{idx:02d}"


def delta_token(idx: int) -> str:
    return f"N{idx:02d}"


def build_vocab(spec: TokenizerSpec) -> List[str]:
    vocab: List[str] = []
    vocab.extend(SPECIAL_TOKENS)
    vocab.extend(BASE_TOKENS)
    vocab.extend(entity_token(i) for i in range(spec.num_entities))
    vocab.extend(value_token(i) for i in range(spec.num_values))
    vocab.extend(array_token(i) for i in range(spec.num_arrays))
    vocab.extend(index_token(i) for i in range(spec.array_width))
    vocab.extend(delta_token(i) for i in range(spec.num_deltas))
    return vocab


def save_tokenizer(output_dir: Path, spec: TokenizerSpec) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = {token: idx for idx, token in enumerate(build_vocab(spec))}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    fast.model_max_length = 1_000_000
    fast.save_pretrained(output_dir)
    with (output_dir / "longdsl_tokenizer_spec.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(spec), handle, indent=2)
    return output_dir


def _stmt(*parts: str) -> List[str]:
    return [*parts, "SEP"]


class ScalarWorld:
    def __init__(self, num_entities: int, num_values: int):
        self.num_entities = num_entities
        self.num_values = num_values
        self.aliases = list(range(num_entities))
        self.values = [0] * num_entities
        self.initialized = [False] * num_entities

    def resolve(self, idx: int) -> int:
        seen = set()
        cur = idx
        while self.aliases[cur] != cur and cur not in seen:
            seen.add(cur)
            cur = self.aliases[cur]
        return cur

    def define(self, idx: int, value: int) -> None:
        self.aliases[idx] = idx
        self.values[idx] = value % self.num_values
        self.initialized[idx] = True

    def set_value(self, idx: int, value: int) -> None:
        root = self.resolve(idx)
        self.values[root] = value % self.num_values
        self.initialized[root] = True

    def alias(self, src: int, dst: int) -> None:
        root_dst = self.resolve(dst)
        if src == root_dst:
            return
        self.aliases[src] = dst
        if self.initialized[root_dst]:
            self.initialized[src] = True

    def inc(self, idx: int, delta: int) -> None:
        root = self.resolve(idx)
        self.values[root] = (self.values[root] + delta) % self.num_values
        self.initialized[root] = True

    def dec(self, idx: int, delta: int) -> None:
        root = self.resolve(idx)
        self.values[root] = (self.values[root] - delta) % self.num_values
        self.initialized[root] = True

    def swap(self, left: int, right: int) -> None:
        root_l = self.resolve(left)
        root_r = self.resolve(right)
        self.values[root_l], self.values[root_r] = self.values[root_r], self.values[root_l]
        self.initialized[root_l] = True
        self.initialized[root_r] = True

    def get_value(self, idx: int) -> int:
        return self.values[self.resolve(idx)]


class ArrayWorld:
    def __init__(self, num_arrays: int, num_values: int, array_width: int):
        self.num_arrays = num_arrays
        self.num_values = num_values
        self.array_width = array_width
        self.values = [[0 for _ in range(array_width)] for _ in range(num_arrays)]
        self.initialized = [False] * num_arrays

    def define(self, idx: int, vals: Sequence[int]) -> None:
        self.values[idx] = [int(v) % self.num_values for v in vals[: self.array_width]]
        self.initialized[idx] = True

    def set_at(self, idx: int, pos: int, value: int) -> None:
        self.values[idx][pos] = value % self.num_values
        self.initialized[idx] = True

    def swap_at(self, idx: int, left: int, right: int) -> None:
        vals = self.values[idx]
        vals[left], vals[right] = vals[right], vals[left]
        self.initialized[idx] = True

    def inc_at(self, idx: int, pos: int, delta: int) -> None:
        self.values[idx][pos] = (self.values[idx][pos] + delta) % self.num_values
        self.initialized[idx] = True

    def get_at(self, idx: int, pos: int) -> int:
        return self.values[idx][pos]


def _pick_distinct(rng, pool: Sequence[int], k: int) -> List[int]:
    return [int(x) for x in rng.choice(pool, size=k, replace=False).tolist()]


def _random_scalar_distractor(world: ScalarWorld, allowed: Sequence[int], level: int, cfg: GeneratorConfig, rng) -> List[str]:
    allowed = list(allowed)
    if not allowed:
        allowed = list(range(world.num_entities))
    op_choices = ["DEF", "SET", "ALIAS"]
    if level >= 2:
        op_choices.extend(["INC", "DEC", "SWAP"])
    for _ in range(32):
        op = str(rng.choice(op_choices))
        if op == "DEF":
            ent = int(rng.choice(allowed))
            val = int(rng.integers(0, cfg.num_values))
            world.define(ent, val)
            return _stmt("DEF", entity_token(ent), value_token(val))
        if op == "SET":
            candidates = [e for e in allowed if world.initialized[world.resolve(e)]]
            if not candidates:
                continue
            ent = int(rng.choice(candidates))
            val = int(rng.integers(0, cfg.num_values))
            world.set_value(ent, val)
            return _stmt("SET", entity_token(ent), value_token(val))
        if op == "ALIAS":
            targets = [e for e in allowed if world.initialized[world.resolve(e)]]
            sources = [e for e in allowed if e not in targets or len(allowed) == 1]
            if not targets or not sources:
                continue
            src = int(rng.choice(sources))
            dst = int(rng.choice(targets))
            if src == dst or world.resolve(dst) == src:
                continue
            world.alias(src, dst)
            return _stmt("ALIAS", entity_token(src), entity_token(dst))
        if op == "INC":
            candidates = [e for e in allowed if world.initialized[world.resolve(e)]]
            if not candidates:
                continue
            ent = int(rng.choice(candidates))
            delta = int(rng.integers(1, cfg.num_deltas + 1))
            world.inc(ent, delta)
            return _stmt("INC", entity_token(ent), delta_token(delta - 1))
        if op == "DEC":
            candidates = [e for e in allowed if world.initialized[world.resolve(e)]]
            if not candidates:
                continue
            ent = int(rng.choice(candidates))
            delta = int(rng.integers(1, cfg.num_deltas + 1))
            world.dec(ent, delta)
            return _stmt("DEC", entity_token(ent), delta_token(delta - 1))
        if op == "SWAP":
            candidates = [e for e in allowed if world.initialized[world.resolve(e)]]
            if len(candidates) < 2:
                continue
            left, right = _pick_distinct(rng, candidates, 2)
            world.swap(left, right)
            return _stmt("SWAP", entity_token(left), entity_token(right))
    ent = int(rng.choice(allowed))
    val = int(rng.integers(0, cfg.num_values))
    world.define(ent, val)
    return _stmt("DEF", entity_token(ent), value_token(val))


def _random_array_distractor(world: ArrayWorld, allowed: Sequence[int], cfg: GeneratorConfig, rng) -> List[str]:
    allowed = list(allowed)
    op_choices = ["DEFARR", "SETAT", "SWAPAT", "INCAT"]
    for _ in range(32):
        op = str(rng.choice(op_choices))
        if op == "DEFARR":
            arr = int(rng.choice(allowed))
            vals = [int(rng.integers(0, cfg.num_values)) for _ in range(cfg.array_width)]
            world.define(arr, vals)
            return _stmt("DEFARR", array_token(arr), *(value_token(v) for v in vals))
        candidates = [a for a in allowed if world.initialized[a]]
        if not candidates:
            continue
        arr = int(rng.choice(candidates))
        if op == "SETAT":
            pos = int(rng.integers(0, cfg.array_width))
            value = int(rng.integers(0, cfg.num_values))
            world.set_at(arr, pos, value)
            return _stmt("SETAT", array_token(arr), index_token(pos), value_token(value))
        if op == "SWAPAT":
            left, right = _pick_distinct(rng, list(range(cfg.array_width)), 2)
            world.swap_at(arr, left, right)
            return _stmt("SWAPAT", array_token(arr), index_token(left), index_token(right))
        if op == "INCAT":
            pos = int(rng.integers(0, cfg.array_width))
            delta = int(rng.integers(1, cfg.num_deltas + 1))
            world.inc_at(arr, pos, delta)
            return _stmt("INCAT", array_token(arr), index_token(pos), delta_token(delta - 1))
    arr = int(rng.choice(allowed))
    vals = [int(rng.integers(0, cfg.num_values)) for _ in range(cfg.array_width)]
    world.define(arr, vals)
    return _stmt("DEFARR", array_token(arr), *(value_token(v) for v in vals))


def _build_level1_relevant(world: ScalarWorld, cfg: GeneratorConfig, rng) -> Tuple[List[List[str]], List[str], str, List[int]]:
    pool = list(range(cfg.num_entities))
    root, query = _pick_distinct(rng, pool, 2)
    mids = _pick_distinct(rng, [e for e in pool if e not in {root, query}], int(rng.integers(0, 3)))
    val0 = int(rng.integers(0, cfg.num_values))
    ops: List[List[str]] = []
    world.define(root, val0)
    ops.append(_stmt("DEF", entity_token(root), value_token(val0)))
    current_value = val0
    if rng.random() < 0.9:
        new_val = int(rng.integers(0, cfg.num_values))
        world.set_value(root, new_val)
        current_value = new_val
        ops.append(_stmt("SET", entity_token(root), value_token(new_val)))
    chain = [query, *mids, root]
    for src, dst in zip(chain[:-1], chain[1:]):
        world.alias(src, dst)
        ops.append(_stmt("ALIAS", entity_token(src), entity_token(dst)))
    query_tokens = ["ASK", entity_token(query)]
    return ops, query_tokens, value_token(current_value), [root, query, *mids]


def _build_level2_relevant(world: ScalarWorld, cfg: GeneratorConfig, rng) -> Tuple[List[List[str]], List[str], str, List[int]]:
    pool = list(range(cfg.num_entities))
    root_a, root_b, query = _pick_distinct(rng, pool, 3)
    ops: List[List[str]] = []
    va = int(rng.integers(0, cfg.num_values))
    vb = int(rng.integers(0, cfg.num_values))
    world.define(root_a, va)
    world.define(root_b, vb)
    ops.append(_stmt("DEF", entity_token(root_a), value_token(va)))
    ops.append(_stmt("DEF", entity_token(root_b), value_token(vb)))
    num_updates = int(rng.integers(2, 6))
    for _ in range(num_updates):
        op = str(rng.choice(["SET", "INC", "DEC", "SWAP"]))
        if op == "SET":
            ent = int(rng.choice([root_a, root_b]))
            val = int(rng.integers(0, cfg.num_values))
            world.set_value(ent, val)
            ops.append(_stmt("SET", entity_token(ent), value_token(val)))
        elif op == "INC":
            ent = int(rng.choice([root_a, root_b]))
            delta = int(rng.integers(1, cfg.num_deltas + 1))
            world.inc(ent, delta)
            ops.append(_stmt("INC", entity_token(ent), delta_token(delta - 1)))
        elif op == "DEC":
            ent = int(rng.choice([root_a, root_b]))
            delta = int(rng.integers(1, cfg.num_deltas + 1))
            world.dec(ent, delta)
            ops.append(_stmt("DEC", entity_token(ent), delta_token(delta - 1)))
        else:
            world.swap(root_a, root_b)
            ops.append(_stmt("SWAP", entity_token(root_a), entity_token(root_b)))
    if rng.random() < 0.8:
        target = int(rng.choice([root_a, root_b]))
        world.alias(query, target)
        ops.append(_stmt("ALIAS", entity_token(query), entity_token(target)))
        ask_ent = query
    else:
        ask_ent = int(rng.choice([root_a, root_b]))
    answer = value_token(world.get_value(ask_ent))
    query_tokens = ["ASK", entity_token(ask_ent)]
    return ops, query_tokens, answer, [root_a, root_b, query]


def _build_level3_relevant(world: ArrayWorld, cfg: GeneratorConfig, rng) -> Tuple[List[List[str]], List[str], str, List[int]]:
    arr = int(rng.integers(0, cfg.num_arrays))
    vals = [int(rng.integers(0, cfg.num_values)) for _ in range(cfg.array_width)]
    ops: List[List[str]] = []
    world.define(arr, vals)
    ops.append(_stmt("DEFARR", array_token(arr), *(value_token(v) for v in vals)))
    num_updates = int(rng.integers(2, 6))
    for _ in range(num_updates):
        op = str(rng.choice(["SETAT", "SWAPAT", "INCAT"]))
        if op == "SETAT":
            idx = int(rng.integers(0, cfg.array_width))
            value = int(rng.integers(0, cfg.num_values))
            world.set_at(arr, idx, value)
            ops.append(_stmt("SETAT", array_token(arr), index_token(idx), value_token(value)))
        elif op == "SWAPAT":
            left, right = _pick_distinct(rng, list(range(cfg.array_width)), 2)
            world.swap_at(arr, left, right)
            ops.append(_stmt("SWAPAT", array_token(arr), index_token(left), index_token(right)))
        else:
            idx = int(rng.integers(0, cfg.array_width))
            delta = int(rng.integers(1, cfg.num_deltas + 1))
            world.inc_at(arr, idx, delta)
            ops.append(_stmt("INCAT", array_token(arr), index_token(idx), delta_token(delta - 1)))
    ask_idx = int(rng.integers(0, cfg.array_width))
    answer = value_token(world.get_at(arr, ask_idx))
    query_tokens = ["GET", array_token(arr), index_token(ask_idx)]
    return ops, query_tokens, answer, [arr]


def generate_example(cfg: GeneratorConfig, rng) -> Dict[str, object]:
    prompt_budget = max(32, min(cfg.context_length - 8, int(cfg.context_length * cfg.fill_ratio)))
    user_tokens: List[str] = ["LEVEL", f"L{cfg.level}", "PROGRAM"]
    if cfg.level in {1, 2}:
        world = ScalarWorld(cfg.num_entities, cfg.num_values)
        if cfg.level == 1:
            relevant_ops, query_tokens, answer, reserved = _build_level1_relevant(world, cfg, rng)
        else:
            relevant_ops, query_tokens, answer, reserved = _build_level2_relevant(world, cfg, rng)
        filler_fn = lambda: _random_scalar_distractor(
            world,
            [e for e in range(cfg.num_entities) if e not in reserved],
            cfg.level,
            cfg,
            rng,
        )
    else:
        world = ArrayWorld(cfg.num_arrays, cfg.num_values, cfg.array_width)
        relevant_ops, query_tokens, answer, reserved = _build_level3_relevant(world, cfg, rng)
        filler_fn = lambda: _random_array_distractor(
            world,
            [a for a in range(cfg.num_arrays) if a not in reserved],
            cfg,
            rng,
        )

    tail_query = ["QUERY", *query_tokens]
    base_program_tokens = len(user_tokens) + sum(len(stmt) for stmt in relevant_ops) + len(tail_query)
    target_fill = max(0, prompt_budget - base_program_tokens)
    section_weights = [1 for _ in relevant_ops] + [max(3, len(relevant_ops) * 2)]
    total_weight = sum(section_weights)
    sections: List[List[str]] = [[] for _ in section_weights]

    for section_idx, weight in enumerate(section_weights):
        budget = (target_fill * weight) // max(total_weight, 1)
        while True:
            candidate = filler_fn()
            if len(sections[section_idx]) + len(candidate) > budget:
                break
            sections[section_idx].extend(candidate)

    for idx, stmt in enumerate(relevant_ops):
        user_tokens.extend(sections[idx])
        user_tokens.extend(stmt)
    user_tokens.extend(sections[-1])
    user_tokens.extend(tail_query)

    messages = [
        {"role": "user", "content": " ".join(user_tokens)},
        {"role": "assistant", "content": f"ANS {answer}"},
    ]
    flat_tokens = [*user_tokens, "ANS", answer]
    return {
        "messages": messages,
        "text": " ".join(flat_tokens),
        "answer": answer,
        "answer_token_index": len(flat_tokens) - 1,
        "level": cfg.level,
        "context_length": cfg.context_length,
        "prompt_tokens": len(user_tokens),
    }


def render_messages(messages: Sequence[Dict[str, str]]) -> Tuple[str, List[Tuple[int, int]]]:
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    offset = 0
    for message in messages:
        prefix = ROLE_PREFIX.format(role=message["role"])
        parts.append(prefix)
        offset += len(prefix)
        parts.append(" ")
        offset += 1
        content = message["content"]
        parts.append(content)
        end = offset + len(content)
        if message["role"] == "assistant":
            spans.append((offset, end))
        parts.append(TURN_SUFFIX)
        offset = end + len(TURN_SUFFIX)
    return "".join(parts), spans


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

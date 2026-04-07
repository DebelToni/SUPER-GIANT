from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from omegaconf import OmegaConf
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
TEXT_MARKERS = ["Context", "Question", "Answer", ":"]
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]+|[^\w\s]", re.UNICODE)
DEFAULT_LEXICON_PATH = Path(__file__).with_name("Lexicon.yml")


@dataclass(frozen=True)
class RelationSpec:
    key: str
    values: tuple[str, ...]
    link_templates: tuple[str, ...]
    update_templates: tuple[str, ...]
    question_templates: tuple[str, ...]


def _tuple_strs(items) -> tuple[str, ...]:
    return tuple(str(item) for item in items)


def _unique_tuple_strs(items) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in items))


def _load_lexicon(path: Path = DEFAULT_LEXICON_PATH) -> dict:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in lexicon file {path}")
    return data


LEXICON = _load_lexicon()
PERSON_NAMES = _unique_tuple_strs(LEXICON["person_names"])
ALIAS_NAMES = _unique_tuple_strs(LEXICON["alias_names"])
TIME_MARKERS = _unique_tuple_strs(LEXICON["time_markers"])
RECORD_SOURCES = _unique_tuple_strs(LEXICON["record_sources"])
DEPARTMENTS = _unique_tuple_strs(LEXICON["departments"])
RELATIONS: tuple[RelationSpec, ...] = tuple(
    RelationSpec(
        key=str(key),
        values=_tuple_strs(spec["values"]),
        link_templates=_tuple_strs(spec["link_templates"]),
        update_templates=_tuple_strs(spec["update_templates"]),
        question_templates=_tuple_strs(spec["question_templates"]),
    )
    for key, spec in dict(LEXICON["relations"]).items()
)
ALIAS_TEMPLATES = _tuple_strs(LEXICON["alias_templates"])
FILLER_TEMPLATES = _tuple_strs(LEXICON["filler_templates"])
RICH_FILLER_TEMPLATES = _tuple_strs(LEXICON["rich_filler_templates"])


@dataclass
class TokenizerSpec:
    person_names: tuple[str, ...] = PERSON_NAMES
    alias_names: tuple[str, ...] = ALIAS_NAMES
    relation_keys: tuple[str, ...] = tuple(spec.key for spec in RELATIONS)


@dataclass
class GeneratorConfig:
    level: int
    context_length: int
    fill_ratio: float = 0.88
    min_alias_chain: int = 1
    max_alias_chain: int = 3
    min_fill_events: int = 4
    max_fill_events: int = 48
    relation_keys: tuple[str, ...] = tuple(spec.key for spec in RELATIONS)


class NaturalWorld:
    def __init__(self) -> None:
        self.next_entity_id = 0
        self.name_to_entity: dict[str, str] = {}
        self.entity_primary: dict[str, str] = {}
        self.entity_values: dict[str, dict[str, str]] = {}
        self.ops: list[list[str]] = []

    def create_entity(self, primary_name: str) -> str:
        entity_id = f"person_{self.next_entity_id:03d}"
        self.next_entity_id += 1
        self.name_to_entity[primary_name] = entity_id
        self.entity_primary[entity_id] = primary_name
        self.entity_values[entity_id] = {}
        self.ops.append(["BIND", primary_name, entity_id])
        return entity_id

    def resolve(self, name: str) -> str:
        return self.name_to_entity[name]

    def alias(self, alias_name: str, target_name: str) -> None:
        entity_id = self.resolve(target_name)
        self.name_to_entity[alias_name] = entity_id
        self.ops.append(["ALIAS", alias_name, target_name])

    def link(self, entity_id: str, relation_key: str, value: str) -> None:
        self.entity_values[entity_id][relation_key] = value
        self.ops.append(["LINK", entity_id, relation_key, value])

    def set_relation(self, entity_id: str, relation_key: str, value: str) -> None:
        self.entity_values[entity_id][relation_key] = value
        self.ops.append(["SET", entity_id, relation_key, value])

    def get(self, name: str, relation_key: str) -> str:
        entity_id = self.resolve(name)
        return self.entity_values[entity_id][relation_key]

    def primary_name(self, entity_id: str) -> str:
        return self.entity_primary[entity_id]

    def has_relation(self, entity_id: str, relation_key: str) -> bool:
        return relation_key in self.entity_values[entity_id]


def surface_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text)


def _format(template: str, **kwargs: str) -> str:
    return template.format(**kwargs)


def _relation_spec_map() -> dict[str, RelationSpec]:
    return {spec.key: spec for spec in RELATIONS}


def _all_names() -> tuple[str, ...]:
    return tuple(dict.fromkeys((*PERSON_NAMES, *ALIAS_NAMES)))


def build_vocab(spec: TokenizerSpec) -> List[str]:
    vocab = set(SPECIAL_TOKENS)
    vocab.update(TEXT_MARKERS)
    vocab.update(spec.person_names)
    vocab.update(spec.alias_names)
    relation_map = _relation_spec_map()
    for relation_key in spec.relation_keys:
        relation = relation_map[relation_key]
        vocab.update(relation.values)
        vocab.update(surface_tokens(relation.key))
        example_name = spec.person_names[0]
        example_alias = spec.alias_names[0]
        example_value = relation.values[0]
        for template in relation.link_templates + relation.update_templates:
            vocab.update(surface_tokens(_format(template, name=example_name, value=example_value)))
        for template in relation.question_templates:
            vocab.update(surface_tokens(_format(template, word=example_alias)))
    for template in ALIAS_TEMPLATES:
        vocab.update(surface_tokens(_format(template, alias=spec.alias_names[0], target=spec.person_names[0])))
    for template in FILLER_TEMPLATES:
        vocab.update(surface_tokens(_format(template, name=spec.person_names[0])))
    for template in RICH_FILLER_TEMPLATES:
        for time_marker in TIME_MARKERS:
            for source in RECORD_SOURCES:
                for department in DEPARTMENTS:
                    vocab.update(
                        surface_tokens(
                            _format(
                                template,
                                name=spec.person_names[0],
                                time_marker=time_marker,
                                source=source,
                                department=department,
                            )
                        )
                    )
    vocab.update({"BIND", "ALIAS", "LINK", "SET", "ASK"})
    return list(SPECIAL_TOKENS) + sorted(vocab.difference(SPECIAL_TOKENS))


def save_tokenizer(output_dir: Path, spec: TokenizerSpec) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab = {token: idx for idx, token in enumerate(build_vocab(spec))}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    fast.model_max_length = 1_000_000
    fast.save_pretrained(output_dir)
    with (output_dir / "long_tokenizer_spec.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(spec), handle, indent=2)
    return output_dir


def _sample_distinct(rng, pool: Sequence[str], count: int) -> list[str]:
    picks = rng.choice(pool, size=count, replace=False).tolist()
    return [str(item) for item in picks]


def _pick_relation(cfg: GeneratorConfig, rng) -> RelationSpec:
    allowed = [spec for spec in RELATIONS if spec.key in set(cfg.relation_keys)]
    return allowed[int(rng.integers(0, len(allowed)))]


def _pick_different_value(relation: RelationSpec, current_value: str, rng) -> str:
    options = [value for value in relation.values if value != current_value]
    return str(options[int(rng.integers(0, len(options)))])


def _render_link(relation: RelationSpec, name: str, value: str, rng) -> str:
    template = relation.link_templates[int(rng.integers(0, len(relation.link_templates)))]
    return _format(template, name=name, value=value)


def _render_update(relation: RelationSpec, name: str, value: str, rng) -> str:
    template = relation.update_templates[int(rng.integers(0, len(relation.update_templates)))]
    return _format(template, name=name, value=value)


def _render_alias(alias_name: str, target_name: str, rng) -> str:
    template = ALIAS_TEMPLATES[int(rng.integers(0, len(ALIAS_TEMPLATES)))]
    return _format(template, alias=alias_name, target=target_name)


def _render_question(relation: RelationSpec, word: str, rng) -> str:
    template = relation.question_templates[int(rng.integers(0, len(relation.question_templates)))]
    return _format(template, word=word)


def _render_filler_note(name: str, rng) -> str:
    template_bank = FILLER_TEMPLATES if rng.random() < 0.55 else RICH_FILLER_TEMPLATES
    template = template_bank[int(rng.integers(0, len(template_bank)))]
    if template in FILLER_TEMPLATES:
        return _format(template, name=name)
    return _format(
        template,
        name=name,
        time_marker=TIME_MARKERS[int(rng.integers(0, len(TIME_MARKERS)))],
        source=RECORD_SOURCES[int(rng.integers(0, len(RECORD_SOURCES)))],
        department=DEPARTMENTS[int(rng.integers(0, len(DEPARTMENTS)))],
    )


def _append_if_room(sentences: list[list[str]], candidate: list[str], target_tokens: int, question: str) -> bool:
    flat = [item for block in sentences for item in block]
    proposed = flat + candidate
    text = "Context: " + " ".join(proposed) + " Question: " + question + " Answer: PLACEHOLDER"
    return len(surface_tokens(text)) <= target_tokens


def _build_relevant_story(cfg: GeneratorConfig, rng) -> tuple[NaturalWorld, RelationSpec, str, list[str], list[int], str, str, set[str]]:
    world = NaturalWorld()
    primary_name, alias_one, alias_two = _sample_distinct(rng, _all_names(), 3)
    relation = _pick_relation(cfg, rng)
    entity_id = world.create_entity(primary_name)
    initial_value = str(relation.values[int(rng.integers(0, len(relation.values)))])
    world.link(entity_id, relation.key, initial_value)

    sentences = [_render_link(relation, primary_name, initial_value, rng)]
    evidence_indices = [0]
    query_name = primary_name

    max_alias_depth = max(cfg.min_alias_chain, min(cfg.max_alias_chain, 2))
    alias_depth = int(rng.integers(cfg.min_alias_chain, max_alias_depth + 1))
    alias_pool = [alias_one, alias_two]
    for idx in range(alias_depth):
        alias_name = alias_pool[idx]
        world.alias(alias_name, query_name)
        sentences.append(_render_alias(alias_name, query_name, rng))
        evidence_indices.append(len(sentences) - 1)
        query_name = alias_name

    if cfg.level >= 2:
        new_value = _pick_different_value(relation, initial_value, rng)
        world.set_relation(entity_id, relation.key, new_value)
        sentences.append(_render_update(relation, primary_name, new_value, rng))
        evidence_indices.append(len(sentences) - 1)

    question = _render_question(relation, query_name, rng)
    answer = world.get(query_name, relation.key)
    used_names = {primary_name, *alias_pool[:alias_depth]}
    return world, relation, query_name, sentences, evidence_indices, question, answer, used_names


def _build_filler_event(
    world: NaturalWorld,
    relation: RelationSpec,
    used_names: set[str],
    target_entity: str,
    cfg: GeneratorConfig,
    rng,
) -> tuple[str, set[str]]:
    available_names = [name for name in _all_names() if name not in used_names]
    distractor_entities = [entity_id for entity_id in world.entity_values if entity_id != target_entity]
    op_choices = ["link", "alias", "note"]
    if cfg.level >= 2:
        op_choices.append("set")

    for _ in range(32):
        op = str(rng.choice(op_choices))
        if op == "link" and available_names:
            name = available_names[int(rng.integers(0, len(available_names)))]
            relation_pick = relation if rng.random() < 0.7 else RELATIONS[int(rng.integers(0, len(RELATIONS)))]
            value = str(relation_pick.values[int(rng.integers(0, len(relation_pick.values)))])
            entity_id = world.create_entity(name)
            world.link(entity_id, relation_pick.key, value)
            return _render_link(relation_pick, name, value, rng), {name}
        if op == "alias" and distractor_entities and available_names:
            alias_name = available_names[int(rng.integers(0, len(available_names)))]
            entity_id = distractor_entities[int(rng.integers(0, len(distractor_entities)))]
            target_name = world.primary_name(entity_id)
            world.alias(alias_name, target_name)
            return _render_alias(alias_name, target_name, rng), {alias_name}
        if op == "set" and distractor_entities:
            entity_id = distractor_entities[int(rng.integers(0, len(distractor_entities)))]
            primary_name = world.primary_name(entity_id)
            relation_key = relation.key if world.has_relation(entity_id, relation.key) else list(world.entity_values[entity_id].keys())[0]
            relation_pick = _relation_spec_map()[relation_key]
            current_value = world.entity_values[entity_id][relation_key]
            new_value = _pick_different_value(relation_pick, current_value, rng)
            world.set_relation(entity_id, relation_key, new_value)
            return _render_update(relation_pick, primary_name, new_value, rng), set()
        if op == "note" and distractor_entities:
            entity_id = distractor_entities[int(rng.integers(0, len(distractor_entities)))]
            return _render_filler_note(world.primary_name(entity_id), rng), set()

    fallback_name = available_names[0] if available_names else world.primary_name(target_entity)
    if fallback_name not in used_names and available_names:
        entity_id = world.create_entity(fallback_name)
        value = str(relation.values[int(rng.integers(0, len(relation.values)))])
        world.link(entity_id, relation.key, value)
        return _render_link(relation, fallback_name, value, rng), {fallback_name}
    return _render_filler_note(world.primary_name(target_entity), rng), set()


def generate_example(cfg: GeneratorConfig, rng) -> Dict[str, object]:
    world, relation, query_name, relevant_sentences, evidence_indices, question, answer, used_names = _build_relevant_story(cfg, rng)
    target_entity = world.resolve(query_name)
    target_tokens = max(48, min(cfg.context_length - 1, int(cfg.context_length * cfg.fill_ratio)))

    sections: list[list[str]] = [[] for _ in range(len(relevant_sentences) + 1)]
    fill_events = 0
    attempts = 0
    while fill_events < cfg.max_fill_events and attempts < cfg.max_fill_events * 8:
        attempts += 1
        candidate, new_names = _build_filler_event(world, relation, used_names, target_entity, cfg, rng)
        section_idx = int(rng.integers(0, len(sections)))
        proposal = sections[section_idx] + [candidate]
        proposal_sections = [list(block) for block in sections]
        proposal_sections[section_idx] = proposal
        ordered: list[str] = []
        for idx, sentence in enumerate(relevant_sentences):
            ordered.extend(proposal_sections[idx])
            ordered.append(sentence)
        ordered.extend(proposal_sections[-1])
        context = " ".join(ordered)
        text = f"Context: {context} Question: {question} Answer: {answer}"
        if len(surface_tokens(text)) > target_tokens:
            if fill_events >= cfg.min_fill_events:
                break
            continue
        sections = proposal_sections
        used_names.update(new_names)
        fill_events += 1

    ordered_sentences: list[str] = []
    shifted_evidence_indices: list[int] = []
    for idx, sentence in enumerate(relevant_sentences):
        ordered_sentences.extend(sections[idx])
        ordered_sentences.append(sentence)
        shifted_evidence_indices.append(len(ordered_sentences) - 1)
    ordered_sentences.extend(sections[-1])

    context = " ".join(ordered_sentences)
    text = f"Context: {context} Question: {question} Answer: {answer}"
    tokens = surface_tokens(text)
    if answer not in set(relation.values):
        raise ValueError(f"Answer must come from relation value set, got {answer}")
    if not tokens or tokens[-1] != answer:
        raise ValueError("Expected answer token to be the final token")

    messages = [
        {"role": "user", "content": f"Context: {context} Question: {question}"},
        {"role": "assistant", "content": f"Answer: {answer}"},
    ]
    return {
        "messages": messages,
        "context": context,
        "question": question,
        "text": text,
        "answer": answer,
        "answer_token_index": len(tokens) - 1,
        "level": cfg.level,
        "context_length": cfg.context_length,
        "genre": "admin_record",
        "relation_key": relation.key,
        "query_name": query_name,
        "prompt_tokens": len(tokens) - 1,
        "evidence_sentence_indices": shifted_evidence_indices,
        "latent_world": {
            "ops": world.ops,
            "query": ["ASK", query_name, relation.key],
        },
    }


def render_messages(messages: Sequence[Dict[str, str]]) -> Tuple[str, List[Tuple[int, int]]]:
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    offset = 0
    for message in messages:
        prefix = f"@{message['role']}"
        parts.append(prefix)
        offset += len(prefix)
        parts.append(" ")
        offset += 1
        content = message["content"]
        parts.append(content)
        end = offset + len(content)
        if message["role"] == "assistant":
            spans.append((offset, end))
        parts.append(" SEP")
        offset = end + 4
    return "".join(parts), spans


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

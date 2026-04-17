from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


TARGET_LINE_RE = re.compile(r"^(?P<title>.+?)\s+—\s+(?P<description>.+)$")
HEADING_RE = re.compile(r"^(?P<heading>.+?)\s+\([0-9]+.*\)$")


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return value.strip("-")


TITLE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "Alexander Nevsky Cathedral": {
        "canonical_name": "Alexander Nevsky Cathedral, Sofia",
        "aliases": ["St. Alexander Nevsky Cathedral", "Alexander Nevsky Cathedral"],
    },
    "Cyrillic alphabet": {
        "canonical_name": "Cyrillic script",
        "aliases": ["Cyrillic alphabet"],
    },
    "Golden Age of Bulgaria": {
        "canonical_name": "Golden Age of medieval Bulgarian culture",
        "aliases": ["Golden Age of Bulgaria"],
    },
    "Vitosha": {
        "canonical_name": "Vitosha",
        "aliases": ["Vitosha Mountain"],
    },
    "Planets of the Solar System": {
        "canonical_name": "Planet",
        "aliases": ["planets of the Solar System"],
    },
    "CPU": {
        "canonical_name": "Central processing unit",
        "aliases": ["CPU"],
    },
    "GPU": {
        "canonical_name": "Graphics processing unit",
        "aliases": ["GPU"],
    },
    "RAM": {
        "canonical_name": "Random-access memory",
        "aliases": ["RAM"],
    },
    "Storage": {
        "canonical_name": "Data storage",
        "aliases": ["Storage"],
    },
    "Python": {
        "canonical_name": "Python (programming language)",
        "aliases": ["Python"],
    },
    "C programming language": {
        "canonical_name": "C (programming language)",
        "aliases": ["C programming language", "C language"],
    },
    "Transformer": {
        "canonical_name": "Transformer (deep learning architecture)",
        "aliases": ["Transformer"],
    },
    "Attention mechanism": {
        "canonical_name": "Attention (machine learning)",
        "aliases": ["Attention mechanism"],
    },
    "Embedding": {
        "canonical_name": "Word embedding",
        "aliases": ["Embedding"],
    },
    "Mean, median, and mode": {
        "canonical_name": "Central tendency",
        "aliases": ["Mean, median, and mode"],
    },
    "Area of a circle": {
        "canonical_name": "Circle",
        "id": "area-of-a-circle",
        "aliases": ["area of a circle"],
    },
    "Graph of a function": {
        "canonical_name": "Graph of a function",
        "aliases": [],
    },
    "Large language model": {
        "canonical_name": "Large language model",
        "aliases": ["LLM"],
    },
    "First World War": {
        "canonical_name": "World War I",
        "aliases": ["First World War"],
    },
    "Second World War": {
        "canonical_name": "World War II",
        "aliases": ["Second World War"],
    },
    "Rose Valley": {
        "canonical_name": "Rose Valley, Bulgaria",
        "aliases": ["Rose Valley"],
    },
    "Treaty of San Stefano": {
        "canonical_name": "Treaty of San Stefano",
        "aliases": ["San Stefano Treaty"],
    },
}


ALIAS_OVERRIDES: Dict[str, List[str]] = {
    "Bulgaria": ["Republic of Bulgaria"],
    "Sofia": ["Sofia, Bulgaria"],
    "Plovdiv": ["Plovdiv, Bulgaria"],
    "Varna": ["Varna, Bulgaria"],
    "Burgas": ["Burgas, Bulgaria"],
    "Danube River": ["Danube"],
    "Balkan Mountains": ["Stara Planina"],
    "Rila Mountains": ["Rila"],
    "Bulgars": ["Proto-Bulgarians"],
    "Khan Asparuh": ["Asparuh"],
    "Boris I": ["Boris I of Bulgaria"],
    "Cyril and Methodius": ["Saints Cyril and Methodius"],
    "Simeon I the Great": ["Simeon the Great", "Simeon I of Bulgaria"],
    "Tsar Samuel": ["Samuel of Bulgaria"],
    "Tarnovo": ["Veliko Tarnovo"],
    "Paisii Hilendarski": ["Paisiy Hilendarski"],
    "Vasil Levski": ["Apostle of Freedom"],
    "Russo-Turkish War of 1877–1878": ["Russo-Turkish War (1877-1878)"],
    "Independence of Bulgaria": ["Bulgarian independence"],
    "Unification of Bulgaria": ["Bulgarian unification"],
    "Albert Einstein": ["Einstein"],
    "Isaac Newton": ["Newton"],
    "Nikola Tesla": ["Tesla"],
    "Thomas Edison": ["Edison"],
    "Alan Turing": ["Turing"],
    "Johannes Gutenberg": ["Gutenberg"],
    "William Shakespeare": ["Shakespeare"],
    "Ludwig van Beethoven": ["Beethoven"],
    "Wolfgang Amadeus Mozart": ["Mozart"],
    "Vincent van Gogh": ["Van Gogh", "van Gogh"],
    "Alexander the Great": ["Alexander III of Macedon"],
    "Mahatma Gandhi": ["Gandhi"],
    "Martin Luther King Jr.": ["Martin Luther King", "MLK"],
    "United Nations": ["UN"],
    "European Union": ["EU"],
    "World Wide Web": ["Web", "WWW"],
    "Artificial intelligence": ["AI"],
    "Machine learning": ["ML"],
}


PERSON_HINTS = {
    "who", "ruler", "scientist", "leader", "poet", "inventor", "astronomer", "naturalist", "president",
    "explorer", "missionaries", "scholar", "composer", "playwright", "artist", "prime minister", "figure",
    "codebreaker", "missionary",
}
GROUP_HINTS = {"peoples", "people", "tribes", "missionaries", "scholars"}
EVENT_HINTS = {
    "war", "revolution", "battle", "uprising", "treaty", "congress", "liberation", "independence",
    "unification", "revival", "mission", "adoption", "flourishing", "restored",
}
PLACE_HINTS = {
    "country", "city", "cities", "sea", "river", "mountain", "range", "peak", "monastery", "cathedral", "region",
    "coast", "capital", "port", "border", "valley", "peninsula", "site",
}


def _guess_kind(title: str, description: str) -> str:
    desc_lower = description.lower()
    title_lower = title.lower()
    if any(word in desc_lower for word in PLACE_HINTS):
        return "place"
    if any(word in desc_lower for word in EVENT_HINTS) or any(word in title_lower for word in {"war", "battle", "revolution", "uprising", "treaty", "congress"}):
        return "event"
    if any(word in desc_lower for word in GROUP_HINTS):
        return "group"
    if any(word in desc_lower for word in PERSON_HINTS) or re.match(r"^[A-Z][a-z]+(?: [A-Z][a-z'.-]+)+$", title):
        return "person"
    return "concept"


def _make_prompts(title: str, description: str, kind: str) -> List[str]:
    if kind == "person":
        return [
            f"Who was {title}?",
            f"What is {title} known for?",
            f"Why is {title} important?",
        ]
    if kind == "group":
        return [
            f"Who were {title}?",
            f"What were {title} known for?",
            f"Why are {title} important?",
        ]
    if kind == "event":
        return [
            f"What was {title}?",
            f"When did {title} happen?",
            f"Why was {title} important?",
        ]
    if kind == "place":
        return [
            f"What is {title}?",
            f"Where is {title}?",
            f"Why is {title} important?",
        ]
    return [
        f"What is {title}?",
        f"How does {title} work?",
        f"Why is {title} important?",
    ]


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        cleaned = item.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _parse_markdown(path: Path) -> List[Dict[str, Any]]:
    current_group = "general"
    targets: List[Dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        heading_match = HEADING_RE.match(line)
        if heading_match and "schema" not in line.lower():
            current_group = _slugify(heading_match.group("heading"))
            continue
        item_match = TARGET_LINE_RE.match(line)
        if item_match is None:
            continue
        raw_title = item_match.group("title").strip()
        description = item_match.group("description").strip().rstrip(".")
        if raw_title.startswith("{"):
            continue

        override = TITLE_OVERRIDES.get(raw_title, {})
        canonical_name = override.get("canonical_name", raw_title)
        aliases = [raw_title]
        aliases.extend(override.get("aliases", []))
        aliases.extend(ALIAS_OVERRIDES.get(raw_title, []))
        aliases = _dedupe_keep_order([alias for alias in aliases if alias != canonical_name])

        kind = _guess_kind(raw_title, description)
        prompts = override.get("prompts") or _make_prompts(raw_title, description, kind)

        targets.append(
            {
                "id": override.get("id", _slugify(raw_title)),
                "group": current_group,
                "canonical_name": canonical_name,
                "aliases": aliases,
                "prompts": prompts,
                "allow_list_pages": False,
                "notes": description,
            }
        )
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a human-readable demo target pack from the root markdown note.")
    parser.add_argument("--input", type=str, required=True, help="Path to the markdown source list")
    parser.add_argument("--out", type=str, required=True, help="Output YAML path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    targets = _parse_markdown(input_path)
    payload = {
        "version": 1,
        "description": "Demo-oriented target pack generated from the root recommendation markdown.",
        "source_markdown": str(input_path),
        "targets": targets,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Wrote {len(targets)} targets to {out_path}")


if __name__ == "__main__":
    main()

"""Text normalisation helpers used by the dataset pipeline."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

_MARKUP_SECTION_RE = re.compile(r"==+\s*([^=]+?)\s*==+", re.MULTILINE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_LINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
_EXTERNAL_LINK_RE = re.compile(r"\[https?://[^\s]+\s([^\]]+)\]")
_REF_RE = re.compile(r"</?ref[^>]*>", re.IGNORECASE)
_CATEGORY_RE = re.compile(r"\[\[(?:Category|File|Image):[^\]]+\]\]", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_FOOTNOTE_RE = re.compile(r"\[[0-9]+\]")
_TABLE_RE = re.compile(r"\{\|[^|]*\|\}", re.DOTALL)


def strip_markup(text: str) -> str:
    """Heuristically strip MediaWiki-style markup and HTML artefacts."""
    text = _REF_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _CATEGORY_RE.sub(" ", text)
    text = _TABLE_RE.sub(" ", text)

    # Remove template blocks with a best-effort approach; repeat to catch nesting.
    previous = None
    while previous != text:
        previous = text
        text = _TEMPLATE_RE.sub(" ", text)

    # Replace wiki links with their display text.
    text = _EXTERNAL_LINK_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)

    # Drop section markers entirely.
    text = _MARKUP_SECTION_RE.sub(r"\1", text)

    # Remove residual bullets and footnote markers.
    text = text.replace("*", " ")
    text = _FOOTNOTE_RE.sub(" ", text)
    return text


def normalise_text(text: str, options: Any) -> str:
    """Apply Unicode normalisation, markup stripping, and whitespace cleanup."""
    if getattr(options, "nfkc", False):
        text = unicodedata.normalize("NFKC", text)
    if getattr(options, "strip_markup", False):
        text = strip_markup(text)
    # Remove soft hyphen and non-breaking space artefacts.
    text = text.replace("\u00ad", "").replace("\u00a0", " ")
    if getattr(options, "collapse_whitespace", False):
        text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


__all__ = ["normalise_text", "strip_markup"]

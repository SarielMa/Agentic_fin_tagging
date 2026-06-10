# Normalizes raw context text for BM25 without table heuristics or LLM rewriting.
from __future__ import annotations

import re
from typing import Any


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


def strip_html(text: str) -> str:
    text = re.sub(r"</t[dh]>", " | ", str(text), flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return normalize_space(text)


def context_text(context: str) -> str:
    """Plain text context for BM25; no table row localization or LLM rewriting."""
    return strip_html(context)

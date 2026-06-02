from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class TaxonomyConcept:
    tag: str
    entity_type: str
    text: str


def tag_terms(tag: str) -> str:
    tag = tag.split(":", 1)[-1]
    chars: list[str] = []
    for i, char in enumerate(tag):
        if i > 0 and char.isupper() and (not tag[i - 1].isupper()):
            chars.append(" ")
        chars.append(char)
    return " ".join("".join(chars).split())


def load_taxonomy(path: Path) -> list[TaxonomyConcept]:
    concepts: list[TaxonomyConcept] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            tag = f"us-gaap:{item['us_gaap_tag']}"
            concepts.append(
                TaxonomyConcept(
                    tag=tag,
                    entity_type=item["entity_type"],
                    text=item.get("text") or tag_terms(tag),
                )
            )
    return concepts


def load_split(memory_csv: Path, test_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(memory_csv), pd.read_csv(test_csv)

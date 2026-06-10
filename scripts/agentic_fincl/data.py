# Loads taxonomy JSONL records and FinCL CSV examples into small typed objects.
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TaxonomyConcept:
    tag: str
    entity_type: str
    text: str


@dataclass(frozen=True)
class Example:
    row_index: int
    context: str
    category: str
    entity: Any
    entity_type: str
    answer: str | None


def load_taxonomy(path: Path) -> list[TaxonomyConcept]:
    concepts: list[TaxonomyConcept] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            concepts.append(
                TaxonomyConcept(
                    tag=f"us-gaap:{item['us_gaap_tag']}",
                    entity_type=str(item["entity_type"]),
                    text=str(item.get("text") or item["us_gaap_tag"]),
                )
            )
    return concepts


def load_examples(path: Path, limit: int = 0) -> list[Example]:
    df = pd.read_csv(path)
    if limit > 0:
        df = df.iloc[:limit]
    examples: list[Example] = []
    for row_index, row in enumerate(df.itertuples(index=False), start=0):
        answer = getattr(row, "answer", None)
        answer_text = None if answer is None or pd.isna(answer) else str(answer)
        examples.append(
            Example(
                row_index=row_index,
                context=str(row.context),
                category=str(row.category),
                entity=row.entity,
                entity_type=str(row.entity_type),
                answer=answer_text,
            )
        )
    return examples

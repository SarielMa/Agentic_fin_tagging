#!/usr/bin/env python3
"""Build a value/type extraction dataset from the FinTagging split.

Each example maps:

    input text -> JSON list of {"numeric_entity": value, "datatype": type}

The original concept/tag is intentionally omitted for this easier subtask.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SPLIT_DIR = "FinTagging_800_200_HF"
DEFAULT_OUTPUT_DIR = "FinTagging_800_200_value_type_HF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a FinTagging subtask dataset where the input is text and "
            "the output is a JSON list of numeric entity/datatype pairs."
        )
    )
    parser.add_argument(
        "--split-dir",
        default=DEFAULT_SPLIT_DIR,
        help="Input HF-style split directory containing data/train.parquet and data/test.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output HF-style dataset directory.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help=(
            "Remove duplicate numeric_entity/datatype pairs within each sample. "
            "By default duplicates and entity order are preserved."
        ),
    )
    return parser.parse_args()


def iter_numeric_entities(value: object) -> Iterable[dict[str, Any]]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for entity in value:
            if isinstance(entity, dict):
                yield entity


def build_output_entities(raw_entities: object, dedupe: bool) -> list[dict[str, str]]:
    output_entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for entity in iter_numeric_entities(raw_entities):
        numeric_entity = entity.get("value")
        datatype = entity.get("type")
        if numeric_entity is None or datatype is None:
            continue

        pair = (str(numeric_entity), str(datatype))
        if dedupe and pair in seen:
            continue

        seen.add(pair)
        output_entities.append(
            {
                "numeric_entity": pair[0],
                "datatype": pair[1],
            }
        )

    return output_entities


def convert_split(df: pd.DataFrame, split_name: str, dedupe: bool) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row_idx, row in df.iterrows():
        output_entities = build_output_entities(row["numeric_entities"], dedupe=dedupe)
        if not output_entities:
            continue

        source_sample_idx = (
            int(row["source_sample_idx"]) if "source_sample_idx" in row else int(row_idx)
        )
        context_id = int(row["context_id"]) if "context_id" in row else source_sample_idx

        rows.append(
            {
                "source_sample_idx": source_sample_idx,
                "context_id": context_id,
                "split": split_name,
                "input": row["text"],
                "output": json.dumps(output_entities, ensure_ascii=False),
                "output_entities": output_entities,
            }
        )

    return pd.DataFrame(rows)


def summarize_split(df: pd.DataFrame) -> dict[str, Any]:
    datatype_counts: Counter[str] = Counter()
    entry_count = 0

    for output_entities in df["output_entities"]:
        for entity in output_entities:
            datatype_counts[entity["datatype"]] += 1
            entry_count += 1

    return {
        "sample_count": int(len(df)),
        "output_entry_count": int(entry_count),
        "unique_datatype_count": len(datatype_counts),
        "datatype_counts": dict(sorted(datatype_counts.items())),
    }


def write_dataset_card(output_dir: Path, report: dict[str, Any]) -> None:
    train = report["splits"]["train"]
    test = report["splits"]["test"]
    readme = f"""---
license: mit
task_categories:
- text-generation
- token-classification
language:
- en
tags:
- finance
- xbrl
- numeric-tagging
- datatype-extraction
- fintagging
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
  - split: test
    path: data/test.parquet
---

# FinTagging Value-Type Extraction Split

This dataset is derived from the sampled FinTagging 800/200 split. It defines an
easier extraction subtask where the input is the table/text context and the
output is a JSON list of numeric entity/datatype pairs. The original XBRL
concept/tag is omitted.

## Task Format

Input column:

- `input`: HTML table/text context.

Target columns:

- `output`: JSON string containing a list of objects with `numeric_entity` and
  `datatype`.
- `output_entities`: structured version of the same target list.

Example target:

```json
[
  {{"numeric_entity": "62", "datatype": "monetaryItemType"}},
  {{"numeric_entity": "171", "datatype": "monetaryItemType"}}
]
```

## Splits

| Split | Samples | Output entries | Unique datatypes |
|---|---:|---:|---:|
| train | {train["sample_count"]:,} | {train["output_entry_count"]:,} | {train["unique_datatype_count"]:,} |
| test | {test["sample_count"]:,} | {test["output_entry_count"]:,} | {test["unique_datatype_count"]:,} |

## Datatype Counts

| Datatype | Train count | Test count |
|---|---:|---:|
| `monetaryItemType` | {train["datatype_counts"].get("monetaryItemType", 0):,} | {test["datatype_counts"].get("monetaryItemType", 0):,} |
| `percentItemType` | {train["datatype_counts"].get("percentItemType", 0):,} | {test["datatype_counts"].get("percentItemType", 0):,} |
| `sharesItemType` | {train["datatype_counts"].get("sharesItemType", 0):,} | {test["datatype_counts"].get("sharesItemType", 0):,} |
| `perShareItemType` | {train["datatype_counts"].get("perShareItemType", 0):,} | {test["datatype_counts"].get("perShareItemType", 0):,} |
| `integerItemType` | {train["datatype_counts"].get("integerItemType", 0):,} | {test["datatype_counts"].get("integerItemType", 0):,} |

## Columns

- `source_sample_idx`: original source row index.
- `context_id`: original context identifier.
- `split`: split label.
- `input`: model input text.
- `output`: JSON target string.
- `output_entities`: structured target list.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_outputs(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    report: dict[str, Any],
) -> None:
    data_dir = output_dir / "data"
    metadata_dir = output_dir / "metadata"
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(data_dir / "train.parquet", index=False)
    test_df.to_parquet(data_dir / "test.parquet", index=False)

    with (metadata_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    (output_dir / ".gitattributes").write_text("*.parquet binary\n", encoding="utf-8")
    (output_dir / ".gitignore").write_text(
        "!data/\n!data/*.parquet\n!metadata/\n!metadata/*.json\n",
        encoding="utf-8",
    )
    write_dataset_card(output_dir, report)


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    output_dir = Path(args.output_dir)

    train_source = pd.read_parquet(split_dir / "data" / "train.parquet")
    test_source = pd.read_parquet(split_dir / "data" / "test.parquet")

    train_df = convert_split(train_source, split_name="train", dedupe=args.dedupe)
    test_df = convert_split(test_source, split_name="test", dedupe=args.dedupe)

    train_ids = set(train_df["source_sample_idx"])
    test_ids = set(test_df["source_sample_idx"])
    train_context_ids = set(train_df["context_id"])
    test_context_ids = set(test_df["context_id"])

    report = {
        "source_split_dir": str(split_dir),
        "output_dir": str(output_dir),
        "dedupe_within_sample": bool(args.dedupe),
        "splits": {
            "train": summarize_split(train_df),
            "test": summarize_split(test_df),
        },
        "validation": {
            "sample_idx_overlap_count": len(train_ids & test_ids),
            "context_id_overlap_count": len(train_context_ids & test_context_ids),
            "empty_output_rows_train": int((train_df["output_entities"].map(len) == 0).sum()),
            "empty_output_rows_test": int((test_df["output_entities"].map(len) == 0).sum()),
        },
    }
    report["validation"]["passed"] = all(
        value == 0 for key, value in report["validation"].items() if key != "passed"
    )

    write_outputs(train_df, test_df, output_dir, report)

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote value/type dataset to: {output_dir}")


if __name__ == "__main__":
    main()

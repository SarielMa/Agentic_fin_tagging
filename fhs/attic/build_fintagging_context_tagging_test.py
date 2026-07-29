#!/usr/bin/env python3
"""Build a combined grounding test set.

This consumes the two first-stage context extraction test sets:

    FinTagging_800_200_table_context_HF/data/test.parquet
    FinTagging_800_200_text_context_HF/data/test.parquet

and joins them back to the original FinTagging test split to recover the XBRL
concept labels. The resulting test rows are inputs for the next tagging step:

Table input:
    {"numeric_entity", "datatype", "row_context", "column_context", "original_context"}

Text input:
    {"numeric_entity", "datatype", "sentence_context", "original_context"}

Output:
    JSON list of ground-truth XBRL concepts. Identical inputs are grouped, so
    ambiguous cases keep all valid tags in one target list.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ORIGINAL_DIR = "FinTagging_800_200_HF"
DEFAULT_TABLE_CONTEXT_DIR = "FinTagging_800_200_table_context_HF"
DEFAULT_TEXT_CONTEXT_DIR = "FinTagging_800_200_text_context_HF"
DEFAULT_OUTPUT_DIR = "FinTagging_800_200_grounding_test_HF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined context-aware FinTagging concept-tagging test set."
    )
    parser.add_argument(
        "--original-dir",
        default=DEFAULT_ORIGINAL_DIR,
        help="Original HF-style FinTagging split directory.",
    )
    parser.add_argument(
        "--table-context-dir",
        default=DEFAULT_TABLE_CONTEXT_DIR,
        help="HF-style table context dataset directory.",
    )
    parser.add_argument(
        "--text-context-dir",
        default=DEFAULT_TEXT_CONTEXT_DIR,
        help="HF-style text context dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output HF-style directory containing data/test.parquet.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists.",
    )
    return parser.parse_args()


def iter_records(value: object) -> Iterable[dict[str, Any]]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for entry in value:
            if isinstance(entry, dict):
                yield entry


def ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def load_original_test(original_dir: Path) -> dict[int, dict[str, Any]]:
    original_test = pd.read_parquet(original_dir / "data" / "test.parquet")
    by_source_idx: dict[int, dict[str, Any]] = {}
    for _, row in original_test.iterrows():
        source_sample_idx = int(row["source_sample_idx"])
        by_source_idx[source_sample_idx] = {
            "context_id": int(row["context_id"]),
            "original_context": str(row["text"]),
            "numeric_entities": list(iter_records(row["numeric_entities"])),
        }
    return by_source_idx


def concept_for_entity(
    original_rows: dict[int, dict[str, Any]],
    source_sample_idx: int,
    entity_index: int,
) -> str:
    if source_sample_idx not in original_rows:
        raise KeyError(f"source_sample_idx {source_sample_idx} is not in the original test split")
    entities = original_rows[source_sample_idx]["numeric_entities"]
    if entity_index < 0 or entity_index >= len(entities):
        raise IndexError(
            f"entity_index {entity_index} is invalid for source_sample_idx {source_sample_idx}"
        )
    concept = entities[entity_index].get("concept")
    if concept is None:
        raise ValueError(
            f"Missing concept for source_sample_idx {source_sample_idx}, entity_index {entity_index}"
        )
    return str(concept)


def build_input_fields(
    input_type: str,
    output_entity: dict[str, Any],
    original_context: str,
) -> dict[str, Any]:
    if input_type == "table":
        return {
            "numeric_entity": output_entity.get("numeric_entity"),
            "datatype": output_entity.get("datatype"),
            "row_context": output_entity.get("row_context"),
            "column_context": output_entity.get("column_context"),
            "original_context": original_context,
        }
    if input_type == "text":
        return {
            "numeric_entity": output_entity.get("numeric_entity"),
            "datatype": output_entity.get("datatype"),
            "sentence_context": output_entity.get("sentence_context"),
            "original_context": original_context,
        }
    raise ValueError(f"Unsupported input_type: {input_type}")


def input_group_key(input_type: str, input_fields: dict[str, Any]) -> str:
    return json.dumps(
        {"input_type": input_type, "input_fields": input_fields},
        sort_keys=True,
        ensure_ascii=False,
    )


def add_context_rows(
    grouped: dict[str, dict[str, Any]],
    context_df: pd.DataFrame,
    input_type: str,
    original_rows: dict[int, dict[str, Any]],
) -> Counter[str]:
    stats: Counter[str] = Counter()

    for _, row in context_df.iterrows():
        source_sample_idx = int(row["source_sample_idx"])
        if source_sample_idx not in original_rows:
            raise KeyError(
                f"{input_type} context row source_sample_idx {source_sample_idx} "
                "is not in the original test split"
            )

        context_id = int(row["context_id"])
        original = original_rows[source_sample_idx]
        original_context = original["original_context"]
        output_entities = list(iter_records(row["output_entities"]))
        entity_metadata = list(iter_records(row["entity_metadata"]))
        if len(output_entities) != len(entity_metadata):
            raise ValueError(
                f"Output/entity metadata length mismatch for source_sample_idx {source_sample_idx}"
            )

        for output_entity, metadata in zip(output_entities, entity_metadata):
            entity_index = int(metadata["entity_index"])
            concept = concept_for_entity(original_rows, source_sample_idx, entity_index)
            input_fields = build_input_fields(input_type, output_entity, original_context)
            key = input_group_key(input_type, input_fields)

            if key not in grouped:
                grouped[key] = {
                    "source_sample_idx": source_sample_idx,
                    "context_id": context_id,
                    "split": "test",
                    "input_type": input_type,
                    "input": json.dumps(input_fields, ensure_ascii=False),
                    "input_fields": input_fields,
                    "ground_truth_concepts": [],
                    "source_entity_indices": [],
                    "source_match_statuses": [],
                }

            grouped_row = grouped[key]
            grouped_row["ground_truth_concepts"].append(concept)
            grouped_row["source_entity_indices"].append(entity_index)
            grouped_row["source_match_statuses"].append(metadata.get("match_status"))
            stats[f"{input_type}_source_entries"] += 1

    return stats


def finalize_rows(grouped: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for grouped_row in grouped.values():
        concepts = ordered_unique(grouped_row["ground_truth_concepts"])
        source_entity_indices = [
            int(value) for value in grouped_row["source_entity_indices"]
        ]
        source_match_statuses = [str(value) for value in grouped_row["source_match_statuses"]]
        row = {
            **grouped_row,
            "ground_truth_concepts": concepts,
            "output": json.dumps(concepts, ensure_ascii=False),
            "ground_truth_count": len(concepts),
            "source_entity_indices": source_entity_indices,
            "source_match_statuses": source_match_statuses,
            "source_occurrence_count": len(source_entity_indices),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            row["source_sample_idx"],
            row["input_type"],
            min(row["source_entity_indices"]) if row["source_entity_indices"] else -1,
            row["input"],
        )
    )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, stats: Counter[str], args: argparse.Namespace) -> dict[str, Any]:
    input_type_counts = df["input_type"].value_counts().sort_index().to_dict()
    multi_tag_rows = int((df["ground_truth_count"] > 1).sum())
    max_ground_truth_count = int(df["ground_truth_count"].max()) if not df.empty else 0
    occurrence_count = int(df["source_occurrence_count"].sum()) if not df.empty else 0
    concept_counts: Counter[str] = Counter()
    for concepts in df["ground_truth_concepts"]:
        concept_counts.update(concepts)

    return {
        "original_dir": args.original_dir,
        "table_context_dir": args.table_context_dir,
        "text_context_dir": args.text_context_dir,
        "output_dir": args.output_dir,
        "format": "HF-style parquet dataset with test split only",
        "input_format": {
            "table": (
                '{"numeric_entity", "datatype", "row_context", '
                '"column_context", "original_context"}'
            ),
            "text": '{"numeric_entity", "datatype", "sentence_context", "original_context"}',
        },
        "output_format": "JSON list of ground-truth XBRL concept tags",
        "splits": {
            "test": {
                "sample_count": int(len(df)),
                "source_occurrence_count": occurrence_count,
                "input_type_counts": {str(k): int(v) for k, v in input_type_counts.items()},
                "multi_tag_input_count": multi_tag_rows,
                "max_ground_truth_count": max_ground_truth_count,
                "unique_concept_count": len(concept_counts),
            }
        },
        "source_entry_stats": {str(k): int(v) for k, v in sorted(stats.items())},
        "validation": {
            "all_rows_are_test_split": bool((df["split"] == "test").all()) if not df.empty else True,
            "output_rows_with_no_ground_truth": int((df["ground_truth_count"] == 0).sum())
            if not df.empty
            else 0,
            "passed": (
                (bool((df["split"] == "test").all()) if not df.empty else True)
                and (
                    int((df["ground_truth_count"] == 0).sum()) if not df.empty else 0
                )
                == 0
            ),
        },
    }


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "data").mkdir(parents=True)
    (output_dir / "metadata").mkdir(parents=True)


def write_readme(output_dir: Path, report: dict[str, Any]) -> None:
    test = report["splits"]["test"]
    table_count = test["input_type_counts"].get("table", 0)
    text_count = test["input_type_counts"].get("text", 0)
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
- concept-tagging
- fintagging
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test.parquet
---

# FinTagging Context-Aware Concept Tagging Test Set

This dataset combines the table-context and text-context test splits derived
from `FinTagging_800_200_HF`. It is intended for the second tagging step: given
an extracted numeric entity, datatype, local context, and original context,
predict the valid XBRL concept tag or tags.

Identical inputs are grouped. If the same input maps to multiple valid concepts,
the target is a JSON list containing all ground-truth concepts.

## Splits

| Split | Rows | Source occurrences | Table rows | Text rows | Multi-tag inputs |
|---|---:|---:|---:|---:|---:|
| test | {test["sample_count"]:,} | {test["source_occurrence_count"]:,} | {table_count:,} | {text_count:,} | {test["multi_tag_input_count"]:,} |

## Columns

- `source_sample_idx`: original source row index.
- `context_id`: original context identifier.
- `split`: always `test`.
- `input_type`: `table` or `text`.
- `input`: JSON string containing the model input fields.
- `input_fields`: structured version of `input`.
- `output`: JSON list of ground-truth XBRL concept tags.
- `ground_truth_concepts`: structured version of `output`.
- `ground_truth_count`: number of unique valid concepts for the input.
- `source_entity_indices`: original `numeric_entities` indices grouped into this row.
- `source_match_statuses`: first-stage parser match statuses for audit only.
- `source_occurrence_count`: number of source entity occurrences grouped into this row.

## Input Format

For table inputs, `input_fields` contains:

```json
{{"numeric_entity": "...", "datatype": "...", "row_context": "...", "column_context": "...", "original_context": "..."}}
```

For text inputs, `input_fields` contains:

```json
{{"numeric_entity": "...", "datatype": "...", "sentence_context": "...", "original_context": "..."}}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    original_dir = Path(args.original_dir)
    table_context_dir = Path(args.table_context_dir)
    text_context_dir = Path(args.text_context_dir)
    output_dir = Path(args.output_dir)

    original_rows = load_original_test(original_dir)
    table_test = pd.read_parquet(table_context_dir / "data" / "test.parquet")
    text_test = pd.read_parquet(text_context_dir / "data" / "test.parquet")

    grouped: dict[str, dict[str, Any]] = {}
    stats = Counter()
    stats.update(add_context_rows(grouped, table_test, "table", original_rows))
    stats.update(add_context_rows(grouped, text_test, "text", original_rows))

    test_df = finalize_rows(grouped)
    report = summarize(test_df, stats, args)

    prepare_output_dir(output_dir, overwrite=args.overwrite)
    test_df.to_parquet(output_dir / "data" / "test.parquet", index=False)
    with (output_dir / "metadata" / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    (output_dir / ".gitattributes").write_text("*.parquet binary\n", encoding="utf-8")
    (output_dir / ".gitignore").write_text(
        "!data/\n!data/*.parquet\n!metadata/\n!metadata/*.json\n",
        encoding="utf-8",
    )
    write_readme(output_dir, report)

    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    print(f"\nWrote grounding test set to: {output_dir}")


if __name__ == "__main__":
    main()

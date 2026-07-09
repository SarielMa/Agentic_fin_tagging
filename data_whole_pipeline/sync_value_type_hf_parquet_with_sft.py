#!/usr/bin/env python3
"""Rewrite the HF parquet upload folder from the SFT Arrow dataset.

The resulting parquet files keep the same task format as the SFT Arrow data:

  query  = instruction prompt + input table/text
  answer = JSON list of {"numeric_entity": ..., "datatype": ...}

The raw table/text is kept in context, and output_entities is included as a
structured copy of answer for easier dataset inspection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_from_disk


DEFAULT_SFT_DIR = "FinTagging_800_200_value_type_sft_arrow"
DEFAULT_HF_DIR = "FinTagging_800_200_value_type_HF"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync the HF parquet upload folder with the SFT Arrow dataset format."
    )
    parser.add_argument(
        "--sft-dir",
        default=DEFAULT_SFT_DIR,
        help="SFT Arrow DatasetDict directory.",
    )
    parser.add_argument(
        "--hf-dir",
        default=DEFAULT_HF_DIR,
        help="HF-style parquet upload directory to rewrite.",
    )
    return parser.parse_args()


def answer_to_entities(answer: str) -> list[dict[str, str]]:
    parsed = json.loads(answer)
    if not isinstance(parsed, list):
        return []

    entities: list[dict[str, str]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        numeric_entity = entry.get("numeric_entity")
        datatype = entry.get("datatype")
        if numeric_entity is None or datatype is None:
            continue
        entities.append(
            {
                "numeric_entity": str(numeric_entity),
                "datatype": str(datatype),
            }
        )
    return entities


def split_to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    converted = []
    for row in rows:
        output_entities = answer_to_entities(row["answer"])
        converted.append(
            {
                "source_sample_idx": int(row["source_sample_idx"]),
                "context_id": int(row["context_id"]),
                "split": row["split"],
                "query": row["query"],
                "answer": row["answer"],
                "context": row["context"],
                "output_entities": output_entities,
            }
        )
    return pd.DataFrame(converted)


def summarize_split(df: pd.DataFrame) -> dict[str, Any]:
    datatype_counts: Counter[str] = Counter()
    output_entry_count = 0
    for entities in df["output_entities"]:
        for entity in entities:
            datatype_counts[entity["datatype"]] += 1
            output_entry_count += 1
    return {
        "sample_count": int(len(df)),
        "output_entry_count": int(output_entry_count),
        "unique_datatype_count": len(datatype_counts),
        "datatype_counts": dict(sorted(datatype_counts.items())),
    }


def write_readme(hf_dir: Path, summary: dict[str, Any]) -> None:
    train = summary["splits"]["train"]
    test = summary["splits"]["test"]
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

# FinTagging Value-Type Extraction SFT Data

This dataset is derived from the sampled FinTagging 800/200 split. It defines an
LLM instruction-following subtask where the model input is a prompt plus the
financial table/text, and the model output is a JSON list of numeric
entity/datatype pairs.

## Task Format

- `query`: full model input, including the instruction template and financial
  table/text.
- `answer`: JSON string target. It is a list of objects with `numeric_entity`
  and `datatype`.
- `context`: raw financial table/text without the instruction prompt.
- `output_entities`: structured copy of `answer` for inspection.

Example answer:

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
- `query`: prompt plus financial table/text.
- `answer`: JSON list target string.
- `context`: raw table/text.
- `output_entities`: structured target list.
"""
    (hf_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    sft_dir = Path(args.sft_dir)
    hf_dir = Path(args.hf_dir)
    data_dir = hf_dir / "data"
    metadata_dir = hf_dir / "metadata"

    dataset = load_from_disk(sft_dir)
    train_df = split_to_frame([dict(row) for row in dataset["train"]])
    test_df = split_to_frame([dict(row) for row in dataset["test"]])

    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(data_dir / "train.parquet", index=False)
    test_df.to_parquet(data_dir / "test.parquet", index=False)

    summary = {
        "source_sft_dir": str(sft_dir),
        "hf_dir": str(hf_dir),
        "format": "parquet upload copy of SFT query/answer dataset",
        "splits": {
            "train": summarize_split(train_df),
            "test": summarize_split(test_df),
        },
    }

    with (metadata_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    (hf_dir / ".gitattributes").write_text("*.parquet binary\n", encoding="utf-8")
    (hf_dir / ".gitignore").write_text(
        "!data/\n!data/*.parquet\n!metadata/\n!metadata/*.json\n",
        encoding="utf-8",
    )
    write_readme(hf_dir, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSynced HF parquet folder: {hf_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Export the context-aware tagging test parquet as readable JSON files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SOURCE_DIR = "FinTagging_800_200_context_tagging_test_HF"
DEFAULT_OUTPUT_DIR = "FinTagging_800_200_context_tagging_test_JSON"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FinTagging context-aware tagging test data to readable JSON."
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help="HF-style source directory containing data/test.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for readable JSON files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists.",
    )
    return parser.parse_args()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if pd.isna(value) if not isinstance(value, (list, dict, np.ndarray)) else False:
        return None
    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        record = {column: to_jsonable(row[column]) for column in df.columns}
        # Parquet stores mixed table/text dictionaries as one nullable struct,
        # which can add irrelevant null keys. The JSON string columns preserve
        # the exact intended schema, so use them for the readable export.
        if isinstance(record.get("input"), str):
            record["input_fields"] = json.loads(record["input"])
        if isinstance(record.get("output"), str):
            record["ground_truth_concepts"] = json.loads(record["output"])
        records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_readme(output_dir: Path, source_dir: Path, row_count: int) -> None:
    readme = f"""# FinTagging Context-Aware Tagging Test JSON Export

This folder is a readable JSON export of:

`{source_dir}/data/test.parquet`

Files:

- `data/test.json`: pretty-printed JSON array with all test records.
- `data/test.jsonl`: one compact JSON record per line.
- `metadata/summary.json`: export metadata.

Each record keeps the same fields as the parquet test set. The main fields are:

- `input_type`: `table` or `text`.
- `input_fields`: structured tagger input.
- `ground_truth_concepts`: list of valid XBRL concept tags.
- `output`: JSON-string version of `ground_truth_concepts`.

Total rows: {row_count:,}
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    data_dir = output_dir / "data"
    metadata_dir = output_dir / "metadata"
    data_dir.mkdir(parents=True)
    metadata_dir.mkdir(parents=True)

    df = pd.read_parquet(source_dir / "data" / "test.parquet")
    records = dataframe_to_records(df)

    with (data_dir / "test.json").open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_jsonl(data_dir / "test.jsonl", records)

    input_type_counts = df["input_type"].value_counts().sort_index().to_dict()
    summary = {
        "source_dir": str(source_dir),
        "source_file": str(source_dir / "data" / "test.parquet"),
        "output_dir": str(output_dir),
        "row_count": int(len(df)),
        "input_type_counts": {str(key): int(value) for key, value in input_type_counts.items()},
        "multi_tag_input_count": int((df["ground_truth_count"] > 1).sum()),
        "max_ground_truth_count": int(df["ground_truth_count"].max()) if len(df) else 0,
        "files": {
            "pretty_json": "data/test.json",
            "jsonl": "data/test.jsonl",
        },
    }
    with (metadata_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    write_readme(output_dir, source_dir, row_count=len(df))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nWrote readable JSON export to: {output_dir}")


if __name__ == "__main__":
    main()

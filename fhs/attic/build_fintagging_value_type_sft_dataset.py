#!/usr/bin/env python3
"""Build a PV-Miner-style SFT dataset for FinTagging value/type extraction.

The output is a Hugging Face DatasetDict saved to disk in Arrow format with
train/test splits and PV-style columns:

  query   = instruction template + input table/text
  answer  = JSON array of {"numeric_entity": ..., "datatype": ...}
  context = original input table/text
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict


DEFAULT_SOURCE_DIR = "FinTagging_800_200_value_type_HF"
DEFAULT_OUTPUT_DIR = "FinTagging_800_200_value_type_sft_arrow"

DATATYPE_DEFINITIONS = {
    "monetaryItemType": "Monetary amount or financial value.",
    "percentItemType": "Percentage, ratio, rate, yield, margin, or tax-rate style value.",
    "sharesItemType": "Number of shares or units of stock/equity.",
    "perShareItemType": "Per-share amount such as earnings per share or dividends per share.",
    "integerItemType": "Plain integer count that is not monetary, percent, shares, or per-share.",
}


INSTRUCTION_TEMPLATE = """<role>
You are an expert financial table tagging analyst.
</role>

<task>
Extract numeric entity and datatype pairs from the provided financial text/table.

Input: one financial text or HTML table.
Output: a JSON array. Each array item must contain exactly:
- "numeric_entity": the normalized numeric value string
- "datatype": one of the allowed datatype names

The task is multi-label: one input can contain zero, one, or many pairs.
</task>

<allowed_datatypes>
- monetaryItemType: Monetary amount or financial value.
- percentItemType: Percentage, ratio, rate, yield, margin, or tax-rate style value.
- sharesItemType: Number of shares or units of stock/equity.
- perShareItemType: Per-share amount such as earnings per share or dividends per share.
- integerItemType: Plain integer count that is not monetary, percent, shares, or per-share.
</allowed_datatypes>

<normalization_rules>
- Return only the numeric value, not surrounding units or labels.
- Remove currency symbols, thousands separators, and percent signs.
- Remove surrounding parentheses used for accounting negatives.
- Preserve decimal points.
- Preserve dash-like placeholder values when they are annotated numeric entities.
- Do not invent values that are not supported by the input.
- Preserve duplicate entries when the same value/datatype appears in multiple cells.
</normalization_rules>

<output_format>
Return JSON only. Do not include explanations.
The output must be a JSON array like:
[
  {{"numeric_entity": "62", "datatype": "monetaryItemType"}},
  {{"numeric_entity": "0.6", "datatype": "percentItemType"}}
]

If no numeric entities apply, output exactly:
[]
</output_format>

INPUT:
{context}

OUTPUT:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Arrow SFT data for FinTagging numeric value/datatype extraction."
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help="HF-style source directory from build_fintagging_value_type_dataset.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for datasets.save_to_disk Arrow data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directory if it already exists.",
    )
    return parser.parse_args()


def build_query(context: str) -> str:
    return INSTRUCTION_TEMPLATE.format(context=context)


def normalize_output_entities(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, dict)):
        return []

    results: list[dict[str, str]] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        numeric_entity = entry.get("numeric_entity")
        datatype = entry.get("datatype")
        if numeric_entity is None or datatype is None:
            continue
        results.append(
            {
                "numeric_entity": str(numeric_entity),
                "datatype": str(datatype),
            }
        )
    return results


def convert_split(df: pd.DataFrame, split_name: str) -> Dataset:
    rows: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        context = str(row["input"])
        output_entities = normalize_output_entities(row["output_entities"])
        source_sample_idx = (
            int(row["source_sample_idx"]) if "source_sample_idx" in row else int(row_idx)
        )
        context_id = int(row["context_id"]) if "context_id" in row else source_sample_idx

        rows.append(
            {
                "query": build_query(context),
                "answer": json.dumps(output_entities, ensure_ascii=False),
                "context": context,
                "source_sample_idx": source_sample_idx,
                "context_id": context_id,
                "split": split_name,
            }
        )

    return Dataset.from_list(rows)


def summarize_dataset(ds: DatasetDict) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}}
    for split_name, split_ds in ds.items():
        datatype_counts: Counter[str] = Counter()
        output_entry_count = 0
        empty_answer_count = 0

        for answer in split_ds["answer"]:
            entries = json.loads(answer)
            if not entries:
                empty_answer_count += 1
            for entry in entries:
                datatype_counts[entry["datatype"]] += 1
                output_entry_count += 1

        summary["splits"][split_name] = {
            "sample_count": len(split_ds),
            "output_entry_count": output_entry_count,
            "empty_answer_count": empty_answer_count,
            "datatype_counts": dict(sorted(datatype_counts.items())),
            "unique_datatype_count": len(datatype_counts),
        }
    return summary


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_dir} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)

    train_df = pd.read_parquet(source_dir / "data" / "train.parquet")
    test_df = pd.read_parquet(source_dir / "data" / "test.parquet")

    ds = DatasetDict(
        {
            "train": convert_split(train_df, "train"),
            "test": convert_split(test_df, "test"),
        }
    )
    ds.save_to_disk(output_dir)

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "format": "PV-style query/answer Arrow dataset",
        "answer_format": 'JSON array of {"numeric_entity": string, "datatype": string}',
        "allowed_datatypes": DATATYPE_DEFINITIONS,
        **summarize_dataset(ds),
    }
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    with (metadata_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nSaved SFT Arrow dataset to: {output_dir}")


if __name__ == "__main__":
    main()

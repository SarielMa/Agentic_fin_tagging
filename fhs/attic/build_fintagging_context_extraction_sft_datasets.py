#!/usr/bin/env python3
"""Build instruction SFT datasets for table and text context extraction.

This creates two separate tasks:

1. Table extraction:
   HTML table -> JSON array of
   {"numeric_entity", "datatype", "row_context", "column_context"}

2. Text extraction:
   Financial text -> JSON array of
   {"numeric_entity", "datatype", "sentence_context"}

Arrow outputs are used by the existing PEFT trainer. Readable JSON mirrors are
also written for inspection.
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


DEFAULT_TABLE_SOURCE_DIR = "FinTagging_800_200_table_context_HF"
DEFAULT_TEXT_SOURCE_DIR = "FinTagging_800_200_text_context_HF"
DEFAULT_TABLE_ARROW_DIR = "FinTagging_800_200_table_context_sft_arrow"
DEFAULT_TEXT_ARROW_DIR = "FinTagging_800_200_text_context_sft_arrow"
DEFAULT_TABLE_JSON_DIR = "FinTagging_800_200_table_context_sft_JSON"
DEFAULT_TEXT_JSON_DIR = "FinTagging_800_200_text_context_sft_JSON"

DATATYPE_DEFINITIONS = {
    "monetaryItemType": "Monetary amount or financial value.",
    "percentItemType": "Percentage, ratio, rate, yield, margin, or tax-rate style value.",
    "sharesItemType": "Number of shares or units of stock/equity.",
    "perShareItemType": "Per-share amount such as earnings per share or dividends per share.",
    "integerItemType": "Plain integer count that is not monetary, percent, shares, or per-share.",
}

TABLE_INSTRUCTION_TEMPLATE = """<role>
You are an expert financial table tagging analyst.
</role>

<task>
Extract annotated numeric occurrences from the provided financial HTML table.

For each relevant numeric occurrence, return its normalized numeric value, its
datatype, and the row/column context that identifies where the value appears in
the table.
</task>

<allowed_datatypes>
- monetaryItemType: Monetary amount or financial value.
- percentItemType: Percentage, ratio, rate, yield, margin, or tax-rate style value.
- sharesItemType: Number of shares or units of stock/equity.
- perShareItemType: Per-share amount such as earnings per share or dividends per share.
- integerItemType: Plain integer count that is not monetary, percent, shares, or per-share.
</allowed_datatypes>

<output_fields>
Return a JSON array. Each array item must contain exactly:
- "numeric_entity": normalized numeric value string
- "datatype": one of the allowed datatype names
- "row_context": row label or row evidence string; use null if unavailable
- "column_context": column label or column evidence string; use null if unavailable
</output_fields>

<normalization_rules>
- Return only the numeric value, not currency symbols, commas, percent signs, or units.
- Remove surrounding parentheses used for accounting negatives.
- Preserve decimal points.
- Preserve dash-like placeholder values when they are annotated numeric entities.
- Preserve duplicate entries when the same value/datatype appears in multiple cells.
- Copy row_context and column_context exactly as supported by the table context.
- Do not invent values or context that are not supported by the input.
</normalization_rules>

<output_format>
Return JSON only. Do not include explanations.

Example:
[
  {{"numeric_entity": "62", "datatype": "monetaryItemType", "row_context": "Current | U.S. Federal", "column_context": "Provision for Income Taxes | 2024"}},
  {{"numeric_entity": "0.6", "datatype": "percentItemType", "row_context": "Effective tax rate", "column_context": "2023"}}
]

If no numeric entities apply, output exactly:
[]
</output_format>

INPUT_HTML_TABLE:
{context}

OUTPUT:"""

TEXT_INSTRUCTION_TEMPLATE = """<role>
You are an expert financial text tagging analyst.
</role>

<task>
Extract annotated numeric occurrences from the provided financial text.

For each relevant numeric occurrence, return its normalized numeric value, its
datatype, and the sentence from the input text that contains the occurrence.
</task>

<allowed_datatypes>
- monetaryItemType: Monetary amount or financial value.
- percentItemType: Percentage, ratio, rate, yield, margin, or tax-rate style value.
- sharesItemType: Number of shares or units of stock/equity.
- perShareItemType: Per-share amount such as earnings per share or dividends per share.
- integerItemType: Plain integer count that is not monetary, percent, shares, or per-share.
</allowed_datatypes>

<output_fields>
Return a JSON array. Each array item must contain exactly:
- "numeric_entity": normalized numeric value string
- "datatype": one of the allowed datatype names
- "sentence_context": the sentence copied from the input text that contains the numeric occurrence
</output_fields>

<normalization_rules>
- Return only the numeric value, not currency symbols, commas, percent signs, or units.
- Remove surrounding parentheses used for accounting negatives.
- Preserve decimal points.
- Preserve dash-like placeholder values when they are annotated numeric entities.
- Preserve duplicate entries when the same value/datatype appears more than once.
- Copy sentence_context exactly from the input text.
- Do not invent values or sentences that are not supported by the input.
</normalization_rules>

<output_format>
Return JSON only. Do not include explanations.

Example:
[
  {{"numeric_entity": "250", "datatype": "monetaryItemType", "sentence_context": "On February 3, 2025, we repaid $ 250 million of the outstanding Term Loan Facility."}}
]

If no numeric entities apply, output exactly:
[]
</output_format>

INPUT_TEXT:
{context}

OUTPUT:"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build table/text context extraction instruction SFT datasets."
    )
    parser.add_argument(
        "--table-source-dir",
        default=DEFAULT_TABLE_SOURCE_DIR,
        help="HF-style table context dataset directory.",
    )
    parser.add_argument(
        "--text-source-dir",
        default=DEFAULT_TEXT_SOURCE_DIR,
        help="HF-style text context dataset directory.",
    )
    parser.add_argument(
        "--table-arrow-dir",
        default=DEFAULT_TABLE_ARROW_DIR,
        help="Output Arrow DatasetDict for table context SFT.",
    )
    parser.add_argument(
        "--text-arrow-dir",
        default=DEFAULT_TEXT_ARROW_DIR,
        help="Output Arrow DatasetDict for text context SFT.",
    )
    parser.add_argument(
        "--table-json-dir",
        default=DEFAULT_TABLE_JSON_DIR,
        help="Readable JSON mirror for table context SFT.",
    )
    parser.add_argument(
        "--text-json-dir",
        default=DEFAULT_TEXT_JSON_DIR,
        help="Readable JSON mirror for text context SFT.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directories if they already exist.",
    )
    return parser.parse_args()


def iter_records(value: object) -> Iterable[dict[str, Any]]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for entry in value:
            if isinstance(entry, dict):
                yield entry


def normalize_output_entities(value: object, task: str) -> list[dict[str, Any]]:
    entities: list[dict[str, Any]] = []
    for entry in iter_records(value):
        numeric_entity = entry.get("numeric_entity")
        datatype = entry.get("datatype")
        if numeric_entity is None or datatype is None:
            continue
        if task == "table":
            entities.append(
                {
                    "numeric_entity": str(numeric_entity),
                    "datatype": str(datatype),
                    "row_context": entry.get("row_context"),
                    "column_context": entry.get("column_context"),
                }
            )
        elif task == "text":
            sentence_context = entry.get("sentence_context")
            if sentence_context is None:
                continue
            entities.append(
                {
                    "numeric_entity": str(numeric_entity),
                    "datatype": str(datatype),
                    "sentence_context": str(sentence_context),
                }
            )
        else:
            raise ValueError(f"Unsupported task: {task}")
    return entities


def build_query(context: str, task: str) -> str:
    if task == "table":
        return TABLE_INSTRUCTION_TEMPLATE.format(context=context)
    if task == "text":
        return TEXT_INSTRUCTION_TEMPLATE.format(context=context)
    raise ValueError(f"Unsupported task: {task}")


def convert_split(df: pd.DataFrame, split_name: str, task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_idx, row in df.iterrows():
        context = str(row["input"])
        output_entities = normalize_output_entities(row["output_entities"], task=task)
        source_sample_idx = (
            int(row["source_sample_idx"]) if "source_sample_idx" in row else int(row_idx)
        )
        context_id = int(row["context_id"]) if "context_id" in row else source_sample_idx

        rows.append(
            {
                "query": build_query(context, task=task),
                "answer": json.dumps(output_entities, ensure_ascii=False),
                "context": context,
                "source_sample_idx": source_sample_idx,
                "context_id": context_id,
                "split": split_name,
                "input_type": task,
                "output_entities": output_entities,
            }
        )
    return rows


def load_context_dataset(source_dir: Path, task: str) -> DatasetDict:
    train_df = pd.read_parquet(source_dir / "data" / "train.parquet")
    test_df = pd.read_parquet(source_dir / "data" / "test.parquet")
    return DatasetDict(
        {
            "train": Dataset.from_list(convert_split(train_df, "train", task=task)),
            "test": Dataset.from_list(convert_split(test_df, "test", task=task)),
        }
    )


def summarize_dataset(dataset: DatasetDict) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}}
    for split_name, split_ds in dataset.items():
        datatype_counts: Counter[str] = Counter()
        output_entry_count = 0
        empty_answer_count = 0
        rows_with_null_context = 0

        for row in split_ds:
            entities = json.loads(row["answer"])
            if not entities:
                empty_answer_count += 1
            if any(
                entity.get("row_context") is None or entity.get("column_context") is None
                for entity in entities
                if "row_context" in entity or "column_context" in entity
            ):
                rows_with_null_context += 1
            for entity in entities:
                datatype_counts[entity["datatype"]] += 1
                output_entry_count += 1

        summary["splits"][split_name] = {
            "sample_count": len(split_ds),
            "output_entry_count": output_entry_count,
            "empty_answer_count": empty_answer_count,
            "rows_with_null_context": rows_with_null_context,
            "unique_datatype_count": len(datatype_counts),
            "datatype_counts": dict(sorted(datatype_counts.items())),
        }
    return summary


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def write_json_mirror(dataset: DatasetDict, output_dir: Path, report: dict[str, Any], overwrite: bool) -> None:
    prepare_output_dir(output_dir, overwrite=overwrite)
    data_dir = output_dir / "data"
    metadata_dir = output_dir / "metadata"
    data_dir.mkdir()
    metadata_dir.mkdir()

    for split_name, split_ds in dataset.items():
        rows = [dict(row) for row in split_ds]
        with (data_dir / f"{split_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        with (data_dir / f"{split_name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

    with (metadata_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    write_readme(output_dir, report, readable_json=True)


def write_readme(output_dir: Path, report: dict[str, Any], readable_json: bool) -> None:
    train = report["splits"]["train"]
    test = report["splits"]["test"]
    task = report["task"]
    data_format = "Readable JSON mirror" if readable_json else "Arrow DatasetDict"
    if task == "table":
        target = '`{"numeric_entity", "datatype", "row_context", "column_context"}`'
    else:
        target = '`{"numeric_entity", "datatype", "sentence_context"}`'

    readme = f"""# FinTagging {task.title()} Context Extraction SFT Data

Format: {data_format}

This instruction SFT dataset is derived from `{report["source_dir"]}`. The model
input is `query`; the target output is `answer`, a JSON array of objects with
fields {target}.

The original train/test split is preserved by `source_sample_idx` and
`context_id`.

| Split | Samples | Output entries | Empty answers |
|---|---:|---:|---:|
| train | {train["sample_count"]:,} | {train["output_entry_count"]:,} | {train["empty_answer_count"]:,} |
| test | {test["sample_count"]:,} | {test["output_entry_count"]:,} | {test["empty_answer_count"]:,} |

Main columns:

- `query`: instruction prompt plus source table/text.
- `answer`: JSON target string.
- `context`: raw table HTML or source text.
- `output_entities`: structured copy of `answer`.
- `source_sample_idx`, `context_id`, `split`, `input_type`: provenance fields.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def save_arrow_dataset(dataset: DatasetDict, output_dir: Path, report: dict[str, Any], overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    dataset.save_to_disk(output_dir)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    with (metadata_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    write_readme(output_dir, report, readable_json=False)


def build_report(task: str, source_dir: Path, output_dir: Path, dataset: DatasetDict) -> dict[str, Any]:
    return {
        "task": task,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "format": "instruction SFT query/answer dataset",
        "allowed_datatypes": DATATYPE_DEFINITIONS,
        **summarize_dataset(dataset),
    }


def main() -> None:
    args = parse_args()

    table_source_dir = Path(args.table_source_dir)
    text_source_dir = Path(args.text_source_dir)
    table_arrow_dir = Path(args.table_arrow_dir)
    text_arrow_dir = Path(args.text_arrow_dir)
    table_json_dir = Path(args.table_json_dir)
    text_json_dir = Path(args.text_json_dir)

    table_dataset = load_context_dataset(table_source_dir, task="table")
    text_dataset = load_context_dataset(text_source_dir, task="text")

    table_report = build_report("table", table_source_dir, table_arrow_dir, table_dataset)
    text_report = build_report("text", text_source_dir, text_arrow_dir, text_dataset)

    save_arrow_dataset(table_dataset, table_arrow_dir, table_report, overwrite=args.overwrite)
    save_arrow_dataset(text_dataset, text_arrow_dir, text_report, overwrite=args.overwrite)

    table_json_report = {**table_report, "output_dir": str(table_json_dir), "format": "readable JSON mirror"}
    text_json_report = {**text_report, "output_dir": str(text_json_dir), "format": "readable JSON mirror"}
    write_json_mirror(table_dataset, table_json_dir, table_json_report, overwrite=args.overwrite)
    write_json_mirror(text_dataset, text_json_dir, text_json_report, overwrite=args.overwrite)

    final_report = {
        "table": table_report,
        "table_json": table_json_report,
        "text": text_report,
        "text_json": text_json_report,
    }
    print(json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"\nSaved table Arrow SFT data to: {table_arrow_dir}")
    print(f"Saved text Arrow SFT data to: {text_arrow_dir}")
    print(f"Saved table readable JSON to: {table_json_dir}")
    print(f"Saved text readable JSON to: {text_json_dir}")


if __name__ == "__main__":
    main()

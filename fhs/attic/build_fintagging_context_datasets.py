#!/usr/bin/env python3
"""Build table-context and text-context extraction datasets from FinTagging.

The source dataset keeps train/test splits in HF-style parquet files:

    FinTagging_800_200_HF/data/train.parquet
    FinTagging_800_200_HF/data/test.parquet

This script creates two derived HF-style datasets while preserving the original
split assignment:

1. Table context extraction:
   input table -> JSON list of
   {"numeric_entity", "datatype", "row_context", "column_context"}

2. Text context extraction:
   input text -> JSON list of
   {"numeric_entity", "datatype", "sentence_context"}

The original XBRL concept/tag is intentionally omitted from targets.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup


DEFAULT_SOURCE_DIR = "FinTagging_800_200_HF"
DEFAULT_TABLE_OUTPUT_DIR = "FinTagging_800_200_table_context_HF"
DEFAULT_TEXT_OUTPUT_DIR = "FinTagging_800_200_text_context_HF"

DATATYPE_DEFINITIONS = {
    "monetaryItemType": "Monetary amount or financial value.",
    "percentItemType": "Percentage, ratio, rate, yield, margin, or tax-rate style value.",
    "sharesItemType": "Number of shares or units of stock/equity.",
    "perShareItemType": "Per-share amount such as earnings per share or dividends per share.",
    "integerItemType": "Plain integer count that is not monetary, percent, shares, or per-share.",
}

DASH_CHARS = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212-"
UNIT_CELLS = {"$", "%", "$ $", "% %"}
PERIOD_RE = re.compile(
    r"\b(?:19\d\d|20\d\d|year|years|month|months|quarter|quarters|ended|"
    r"december|january|february|march|april|may|june|july|august|"
    r"september|october|november)\b",
    flags=re.IGNORECASE,
)
UNIT_RE = re.compile(
    r"\(?\s*(?:in|dollars?|shares?|millions?|thousands?|except|per|amounts?|"
    r"unaudited|usd|\$|%)\b.*",
    flags=re.IGNORECASE,
)
NUMERIC_TOKEN_RE = re.compile(
    rf"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)|[{re.escape(DASH_CHARS)}]"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=(?:[A-Z0-9(]|[$]))")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create separate FinTagging table-context and text-context datasets "
            "from the sampled 800/200 HF split."
        )
    )
    parser.add_argument(
        "--source-dir",
        default=DEFAULT_SOURCE_DIR,
        help="Input HF-style split directory with data/train.parquet and data/test.parquet.",
    )
    parser.add_argument(
        "--table-output-dir",
        default=DEFAULT_TABLE_OUTPUT_DIR,
        help="Output HF-style directory for table context extraction data.",
    )
    parser.add_argument(
        "--text-output-dir",
        default=DEFAULT_TEXT_OUTPUT_DIR,
        help="Output HF-style directory for text sentence-context extraction data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output directories if they already exist.",
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    text = html.unescape("" if value is None else str(value)).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def collapse_repeated_tokens(text: str) -> str:
    """Collapse extraction artifacts such as '2024 2024' or '$ $'."""
    text = clean_text(text)
    tokens = text.split()
    if len(tokens) >= 2 and len(tokens) % 2 == 0:
        half = len(tokens) // 2
        if tokens[:half] == tokens[half:]:
            return " ".join(tokens[:half])
    if len(tokens) == 2 and tokens[0] == tokens[1]:
        return tokens[0]
    return text


def normalize_dash_chars(text: str) -> str:
    for char in DASH_CHARS:
        if char != "-":
            text = text.replace(char, "-")
    return text


def normalize_numeric_value(value: object) -> str:
    """Normalize a numeric string for deterministic matching only.

    The dataset target still keeps the original normalized FinTagging value.
    This function collapses currency/percent signs, commas, accounting
    parentheses, repeated-token artifacts, and dash variants.
    """
    text = normalize_dash_chars(collapse_repeated_tokens(clean_text(value)))
    candidate = text.replace("$", " ").replace("%", " ").replace(",", " ")
    candidate = collapse_repeated_tokens(candidate)
    compact = re.sub(r"\s+", "", candidate)

    if len(compact) >= 2 and compact[0] == "(" and compact[-1] == ")":
        compact = compact[1:-1]

    if compact == "-":
        return compact
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)", compact):
        return compact.lstrip("+")

    tokens = [normalize_dash_chars(token).replace(",", "") for token in NUMERIC_TOKEN_RE.findall(text)]
    if not tokens:
        return ""

    non_token_text = NUMERIC_TOKEN_RE.sub("", text)
    non_token_text = re.sub(r"[\s$%,().]+", "", non_token_text)
    if non_token_text:
        return ""

    if len(set(tokens)) == 1:
        return tokens[0].lstrip("+")

    if len(tokens) % 2 == 0:
        half = len(tokens) // 2
        if tokens[:half] == tokens[half:] and len(tokens[:half]) == 1:
            return tokens[0].lstrip("+")

    return ""


def is_numeric_like(value: object) -> bool:
    return bool(normalize_numeric_value(value))


def is_period_header(value: object) -> bool:
    return bool(PERIOD_RE.search(clean_text(value)))


def is_row_context_cell(value: object) -> bool:
    text = collapse_repeated_tokens(clean_text(value))
    if not text or text in UNIT_CELLS:
        return False
    if is_numeric_like(text):
        return False
    if UNIT_RE.fullmatch(text):
        return False
    return True


def is_column_context_cell(value: object) -> bool:
    text = collapse_repeated_tokens(clean_text(value))
    if not text or text in UNIT_CELLS:
        return False
    if is_period_header(text):
        return True
    if is_numeric_like(text):
        return False
    if UNIT_RE.fullmatch(text):
        return False
    return True


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def iter_numeric_entities(value: object) -> Iterable[dict[str, Any]]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        for entity in value:
            if isinstance(entity, dict):
                yield entity


def has_table_markup(text: str) -> bool:
    return "<table" in text.lower()


def strip_html_to_text(text: str) -> str:
    if "<" not in text or ">" not in text:
        return clean_text(text)
    soup = BeautifulSoup(text, "lxml")
    return clean_text(soup.get_text(" ", strip=True))


def parse_table_rows(text: str) -> list[list[list[str]]]:
    soup = BeautifulSoup(text, "lxml")
    tables: list[list[list[str]]] = []

    for table in soup.find_all("table"):
        rows: list[list[str]] = []
        for tr in table.find_all("tr"):
            row: list[str] = []
            for cell in tr.find_all(["td", "th"]):
                cell_text = collapse_repeated_tokens(cell.get_text(" ", strip=True))
                try:
                    colspan = max(1, int(cell.get("colspan") or 1))
                except (TypeError, ValueError):
                    colspan = 1
                row.extend([cell_text] * colspan)
            rows.append(row)
        if rows:
            tables.append(rows)

    return tables


def numeric_data_cells(row: list[str]) -> list[tuple[int, str]]:
    cells = []
    for col_idx, value in enumerate(row):
        text = clean_text(value)
        if text and text not in UNIT_CELLS and is_numeric_like(text):
            cells.append((col_idx, text))
    return cells


def row_context_cells(row: list[str]) -> list[tuple[int, str]]:
    return [
        (col_idx, collapse_repeated_tokens(value))
        for col_idx, value in enumerate(row)
        if is_row_context_cell(value)
    ]


def column_context_cells(row: list[str]) -> list[tuple[int, str]]:
    return [
        (col_idx, collapse_repeated_tokens(value))
        for col_idx, value in enumerate(row)
        if is_column_context_cell(value)
    ]


def infer_row_context(rows: list[list[str]], row_idx: int, col_idx: int) -> str | None:
    pieces: list[str] = []

    # Nearby one-cell rows often act as section labels.
    for prior_row_idx in range(max(0, row_idx - 3), row_idx):
        headers = row_context_cells(rows[prior_row_idx])
        numeric_cells = numeric_data_cells(rows[prior_row_idx])
        if len(headers) == 1 and not numeric_cells:
            pieces.append(headers[0][1])

    left_context = [
        value for candidate_col, value in row_context_cells(rows[row_idx]) if candidate_col < col_idx
    ]
    if left_context:
        pieces.append(left_context[-1])

    values = dedupe_preserving_order(pieces)
    return " | ".join(values) if values else None


def infer_column_context(rows: list[list[str]], row_idx: int, col_idx: int) -> str | None:
    current_numeric_cells = numeric_data_cells(rows[row_idx])
    ordinal = None
    for idx, (candidate_col, _) in enumerate(current_numeric_cells):
        if candidate_col == col_idx:
            ordinal = idx
            break
    if ordinal is None:
        return None

    pieces: list[str] = []
    for prior_row_idx in range(row_idx):
        headers = column_context_cells(rows[prior_row_idx])
        if not headers:
            continue

        # One period/date header applies across the row.
        if len(headers) == 1:
            if is_period_header(headers[0][1]):
                pieces.append(headers[0][1])
            continue

        header_values = [value for _, value in headers]
        offset = 0
        if len(header_values) > len(current_numeric_cells):
            first = header_values[0].lower()
            if re.search(r"\b(?:in millions|in thousands|dollars|shares|except)\b", first):
                offset = 1

        header_idx = min(ordinal + offset, len(header_values) - 1)
        pieces.append(header_values[header_idx])

    values = dedupe_preserving_order(pieces[-4:])
    return " | ".join(values) if values else None


def extract_table_cells(text: str) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for table_idx, rows in enumerate(parse_table_rows(text)):
        for row_idx, row in enumerate(rows):
            for col_idx, cell_text in numeric_data_cells(row):
                cells.append(
                    {
                        "table_index": table_idx,
                        "row_index": row_idx,
                        "column_index": col_idx,
                        "cell_text": cell_text,
                        "normalized_cell_text": normalize_numeric_value(cell_text),
                        "row_context": infer_row_context(rows, row_idx, col_idx),
                        "column_context": infer_column_context(rows, row_idx, col_idx),
                    }
                )
    return cells


def build_table_entries(raw_entities: object, cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    output_entities: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    cells_by_value: dict[str, list[int]] = defaultdict(list)
    for cell_idx, cell in enumerate(cells):
        normalized_value = cell["normalized_cell_text"]
        if normalized_value:
            cells_by_value[normalized_value].append(cell_idx)

    used_cells: set[int] = set()
    for entity_idx, entity in enumerate(iter_numeric_entities(raw_entities)):
        numeric_entity = entity.get("value")
        datatype = entity.get("type")
        if numeric_entity is None or datatype is None:
            stats["skipped_entity_missing_value_or_type"] += 1
            continue

        normalized_entity = normalize_numeric_value(numeric_entity)
        candidate_cells = cells_by_value.get(normalized_entity, [])
        unused_candidates = [cell_idx for cell_idx in candidate_cells if cell_idx not in used_cells]

        selected_cell = None
        match_status = "unmatched"
        if unused_candidates:
            selected_cell = cells[unused_candidates[0]]
            used_cells.add(unused_candidates[0])
            match_status = "ambiguous_cell" if len(candidate_cells) > 1 else "exact_cell"
        elif candidate_cells:
            selected_cell = cells[candidate_cells[0]]
            match_status = "reused_cell"

        row_context = selected_cell["row_context"] if selected_cell else None
        column_context = selected_cell["column_context"] if selected_cell else None

        if row_context is not None and column_context is not None:
            stats["entries_with_both_contexts"] += 1
        else:
            if row_context is None:
                stats["entries_missing_row_context"] += 1
            if column_context is None:
                stats["entries_missing_column_context"] += 1
        stats[f"match_status_{match_status}"] += 1

        output_entities.append(
            {
                "numeric_entity": str(numeric_entity),
                "datatype": str(datatype),
                "row_context": row_context,
                "column_context": column_context,
            }
        )
        metadata.append(
            {
                "entity_index": entity_idx,
                "match_status": match_status,
                "candidate_cell_count": len(candidate_cells),
                "table_index": selected_cell["table_index"] if selected_cell else None,
                "row_index": selected_cell["row_index"] if selected_cell else None,
                "column_index": selected_cell["column_index"] if selected_cell else None,
                "cell_text": selected_cell["cell_text"] if selected_cell else None,
            }
        )

    return output_entities, metadata, stats


def split_sentences(text: str) -> list[str]:
    text = strip_html_to_text(text)
    if not text:
        return []
    sentences = [clean_text(sentence) for sentence in SENTENCE_SPLIT_RE.split(text)]
    return [sentence for sentence in sentences if sentence]


def sentence_numeric_values(sentence: str) -> list[str]:
    values = []
    for token in NUMERIC_TOKEN_RE.findall(normalize_dash_chars(sentence)):
        normalized = normalize_numeric_value(token)
        if normalized:
            values.append(normalized)
    return values


def build_text_entries(raw_entities: object, sentences: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    output_entities: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    sentence_values: list[list[str]] = [sentence_numeric_values(sentence) for sentence in sentences]
    used_occurrences: set[tuple[int, int]] = set()

    for entity_idx, entity in enumerate(iter_numeric_entities(raw_entities)):
        numeric_entity = entity.get("value")
        datatype = entity.get("type")
        if numeric_entity is None or datatype is None:
            stats["skipped_entity_missing_value_or_type"] += 1
            continue

        normalized_entity = normalize_numeric_value(numeric_entity)
        candidates: list[tuple[int, int]] = []
        for sentence_idx, values in enumerate(sentence_values):
            for occurrence_idx, value in enumerate(values):
                if value == normalized_entity:
                    candidates.append((sentence_idx, occurrence_idx))

        unused_candidates = [candidate for candidate in candidates if candidate not in used_occurrences]
        selected = None
        match_status = "unmatched"
        if unused_candidates:
            selected = unused_candidates[0]
            used_occurrences.add(selected)
            match_status = "ambiguous_sentence" if len(candidates) > 1 else "exact_sentence"
        elif candidates:
            selected = candidates[0]
            match_status = "reused_sentence"

        if selected is None:
            stats["match_status_unmatched"] += 1
            output_entities.append(
                {
                    "numeric_entity": str(numeric_entity),
                    "datatype": str(datatype),
                    "sentence_context": None,
                }
            )
            metadata.append(
                {
                    "entity_index": entity_idx,
                    "match_status": match_status,
                    "candidate_sentence_count": 0,
                    "sentence_index": None,
                }
            )
            continue

        sentence_idx, _ = selected
        stats[f"match_status_{match_status}"] += 1
        output_entities.append(
            {
                "numeric_entity": str(numeric_entity),
                "datatype": str(datatype),
                "sentence_context": sentences[sentence_idx],
            }
        )
        metadata.append(
            {
                "entity_index": entity_idx,
                "match_status": match_status,
                "candidate_sentence_count": len({candidate[0] for candidate in candidates}),
                "sentence_index": sentence_idx,
            }
        )

    return output_entities, metadata, stats


def convert_table_split(df: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for row_idx, row in df.iterrows():
        source_sample_idx = int(row["source_sample_idx"]) if "source_sample_idx" in row else int(row_idx)
        context_id = int(row["context_id"]) if "context_id" in row else source_sample_idx
        context = str(row["text"])

        if not has_table_markup(context):
            stats["skipped_non_table_examples"] += 1
            continue

        stats["source_table_examples"] += 1
        cells = extract_table_cells(context)
        if not cells:
            stats["dropped_unparsable_table_examples"] += 1
            continue

        output_entities, entity_metadata, entity_stats = build_table_entries(
            row["numeric_entities"], cells
        )
        stats.update(entity_stats)
        if not output_entities:
            stats["dropped_empty_output_examples"] += 1
            continue

        stats["retained_examples"] += 1
        stats["output_entry_count"] += len(output_entities)
        rows.append(
            {
                "source_sample_idx": source_sample_idx,
                "context_id": context_id,
                "split": split_name,
                "input_type": "table",
                "input": context,
                "output": json.dumps(output_entities, ensure_ascii=False),
                "output_entities": output_entities,
                "entity_metadata": entity_metadata,
            }
        )

    return pd.DataFrame(rows), dict(sorted(stats.items()))


def convert_text_split(df: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()

    for row_idx, row in df.iterrows():
        source_sample_idx = int(row["source_sample_idx"]) if "source_sample_idx" in row else int(row_idx)
        context_id = int(row["context_id"]) if "context_id" in row else source_sample_idx
        context = str(row["text"])

        if has_table_markup(context):
            stats["skipped_table_examples"] += 1
            continue

        stats["source_text_examples"] += 1
        sentences = split_sentences(context)
        if not sentences:
            stats["dropped_no_sentence_examples"] += 1
            continue

        output_entities, entity_metadata, entity_stats = build_text_entries(
            row["numeric_entities"], sentences
        )
        stats.update(entity_stats)

        has_unmatched = any(
            entry["match_status"] == "unmatched" for entry in entity_metadata
        )
        if not output_entities:
            stats["dropped_empty_output_examples"] += 1
            continue
        if has_unmatched:
            stats["dropped_unmatched_entity_examples"] += 1
            continue

        stats["retained_examples"] += 1
        stats["output_entry_count"] += len(output_entities)
        rows.append(
            {
                "source_sample_idx": source_sample_idx,
                "context_id": context_id,
                "split": split_name,
                "input_type": "text",
                "input": strip_html_to_text(context),
                "output": json.dumps(output_entities, ensure_ascii=False),
                "output_entities": output_entities,
                "entity_metadata": entity_metadata,
            }
        )

    return pd.DataFrame(rows), dict(sorted(stats.items()))


def count_output_entries(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df["output_entities"].map(len).sum())


def duplicate_key_summary(df: pd.DataFrame, key_fields: list[str]) -> dict[str, int]:
    duplicate_rows = 0
    duplicate_groups = 0
    duplicate_extra_entries = 0

    for _, row in df.iterrows():
        counts: Counter[tuple[Any, ...]] = Counter()
        for entity in row["output_entities"]:
            counts[tuple(entity.get(field) for field in key_fields)] += 1
        row_duplicate_groups = [count for count in counts.values() if count > 1]
        if row_duplicate_groups:
            duplicate_rows += 1
            duplicate_groups += len(row_duplicate_groups)
            duplicate_extra_entries += sum(count - 1 for count in row_duplicate_groups)

    return {
        "rows_with_duplicate_output_keys": duplicate_rows,
        "duplicate_output_key_groups": duplicate_groups,
        "extra_duplicate_output_entries": duplicate_extra_entries,
    }


def validation_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, Any]:
    train_ids = set(train_df["source_sample_idx"]) if not train_df.empty else set()
    test_ids = set(test_df["source_sample_idx"]) if not test_df.empty else set()
    train_context_ids = set(train_df["context_id"]) if not train_df.empty else set()
    test_context_ids = set(test_df["context_id"]) if not test_df.empty else set()

    summary = {
        "sample_idx_overlap_count": len(train_ids & test_ids),
        "context_id_overlap_count": len(train_context_ids & test_context_ids),
    }
    summary["passed"] = all(value == 0 for value in summary.values())
    return summary


def dataset_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_stats: dict[str, dict[str, Any]],
    duplicate_key_fields: list[str],
) -> dict[str, Any]:
    return {
        "splits": {
            "train": {
                "sample_count": int(len(train_df)),
                "output_entry_count": count_output_entries(train_df),
                "stats": split_stats["train"],
                **duplicate_key_summary(train_df, duplicate_key_fields),
            },
            "test": {
                "sample_count": int(len(test_df)),
                "output_entry_count": count_output_entries(test_df),
                "stats": split_stats["test"],
                **duplicate_key_summary(test_df, duplicate_key_fields),
            },
        },
        "validation": validation_summary(train_df, test_df),
    }


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "data").mkdir(parents=True)
    (output_dir / "metadata").mkdir(parents=True)


def write_dataset_card(output_dir: Path, title: str, task_description: str, report: dict[str, Any]) -> None:
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

# {title}

{task_description}

The dataset is derived from `FinTagging_800_200_HF` and preserves the original
train/test assignment by `source_sample_idx` and `context_id`. The XBRL concept
tag is intentionally omitted from the target.

## Splits

| Split | Samples | Output entries | Duplicate output key groups |
|---|---:|---:|---:|
| train | {train["sample_count"]:,} | {train["output_entry_count"]:,} | {train["duplicate_output_key_groups"]:,} |
| test | {test["sample_count"]:,} | {test["output_entry_count"]:,} | {test["duplicate_output_key_groups"]:,} |

## Columns

- `source_sample_idx`: original source row index.
- `context_id`: original context identifier.
- `split`: original split label.
- `input_type`: `table` or `text`.
- `input`: raw table HTML for table data, cleaned plain text for text data.
- `output`: JSON target string.
- `output_entities`: structured version of `output`.
- `entity_metadata`: deterministic parser metadata for auditing only.

## Validation

| Check | Value |
|---|---:|
| source sample index overlap | {report["validation"]["sample_idx_overlap_count"]} |
| context ID overlap | {report["validation"]["context_id_overlap_count"]} |
| passed | {report["validation"]["passed"]} |
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def write_hf_dataset(
    output_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    report: dict[str, Any],
    title: str,
    task_description: str,
    overwrite: bool,
) -> None:
    prepare_output_dir(output_dir, overwrite=overwrite)
    train_df.to_parquet(output_dir / "data" / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "data" / "test.parquet", index=False)

    with (output_dir / "metadata" / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    (output_dir / ".gitattributes").write_text("*.parquet binary\n", encoding="utf-8")
    (output_dir / ".gitignore").write_text(
        "!data/\n!data/*.parquet\n!metadata/\n!metadata/*.json\n",
        encoding="utf-8",
    )
    write_dataset_card(output_dir, title=title, task_description=task_description, report=report)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    table_output_dir = Path(args.table_output_dir)
    text_output_dir = Path(args.text_output_dir)

    train_source = pd.read_parquet(source_dir / "data" / "train.parquet")
    test_source = pd.read_parquet(source_dir / "data" / "test.parquet")

    table_train, table_train_stats = convert_table_split(train_source, "train")
    table_test, table_test_stats = convert_table_split(test_source, "test")
    text_train, text_train_stats = convert_text_split(train_source, "train")
    text_test, text_test_stats = convert_text_split(test_source, "test")

    table_report = {
        "source_dir": str(source_dir),
        "output_dir": str(table_output_dir),
        "format": "HF-style parquet dataset",
        "target_format": (
            'JSON array of {"numeric_entity", "datatype", "row_context", '
            '"column_context"}'
        ),
        "allowed_datatypes": DATATYPE_DEFINITIONS,
        **dataset_summary(
            table_train,
            table_test,
            {"train": table_train_stats, "test": table_test_stats},
            ["numeric_entity", "datatype", "row_context", "column_context"],
        ),
    }
    text_report = {
        "source_dir": str(source_dir),
        "output_dir": str(text_output_dir),
        "format": "HF-style parquet dataset",
        "target_format": 'JSON array of {"numeric_entity", "datatype", "sentence_context"}',
        "allowed_datatypes": DATATYPE_DEFINITIONS,
        **dataset_summary(
            text_train,
            text_test,
            {"train": text_train_stats, "test": text_test_stats},
            ["numeric_entity", "datatype", "sentence_context"],
        ),
    }

    write_hf_dataset(
        table_output_dir,
        table_train,
        table_test,
        table_report,
        title="FinTagging Table Context Extraction Split",
        task_description=(
            "For HTML table inputs, the target is a JSON list of numeric "
            "entity/datatype pairs enriched with deterministic row and column "
            "context. Missing row or column context is represented as null."
        ),
        overwrite=args.overwrite,
    )
    write_hf_dataset(
        text_output_dir,
        text_train,
        text_test,
        text_report,
        title="FinTagging Text Sentence Context Extraction Split",
        task_description=(
            "For non-table text inputs, the target is a JSON list of numeric "
            "entity/datatype pairs enriched with the sentence containing the "
            "matched numeric entity."
        ),
        overwrite=args.overwrite,
    )

    final_report = {
        "table_dataset": table_report,
        "text_dataset": text_report,
    }
    print(json.dumps(final_report, indent=2, sort_keys=True, ensure_ascii=False))
    print(f"\nWrote table context dataset to: {table_output_dir}")
    print(f"Wrote text context dataset to: {text_output_dir}")


if __name__ == "__main__":
    main()

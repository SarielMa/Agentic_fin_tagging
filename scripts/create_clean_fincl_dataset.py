#!/usr/bin/env python3
"""Create a clean target-centered FinCL evaluation CSV.

This script keeps the original full ``context`` field and ignores the dataset's
serialized ``query`` field. The query field has missing local table context for
some rows, but our end-to-end pipeline will use the original context directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("materials/FinCL-eval-subset.csv")
DEFAULT_TAXONOMY = Path("materials/us_gaap_2024_BM25.jsonl")
DEFAULT_OUTPUT = Path("materials/FinCL-eval-subset-clean.csv")

VALID_ENTITY_TYPES = {
    "integerItemType",
    "monetaryItemType",
    "perShareItemType",
    "sharesItemType",
    "percentItemType",
}

# Rows reviewed manually and excluded from the clean end-to-end set:
#
# 119: entity="none", sharesItemType. This is an XBRL-style non-value marker
#      rather than a normal numeric fact, so it is ill-suited for evaluating a
#      numerical extraction pipeline.
# 260 and 324: same full context, entity="22", and type="monetaryItemType",
#      but two different gold tags. A single target instance cannot have two
#      mutually exclusive labels under exact-match evaluation.
# 393: entity="nominal", monetaryItemType. This is a qualitative amount rather
#      than a numeric fact, so it is ill-suited for strict numerical extraction.
EXCLUDED_ROWS = {
    119: "non-numeric share count marker: entity='none'",
    260: "label conflict with row 324 for same context/entity/type",
    324: "label conflict with row 260 for same context/entity/type",
    393: "qualitative monetary amount: entity='nominal'",
}


def load_taxonomy_types(path: Path) -> dict[str, str]:
    tag_to_type: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            tag_to_type[f"us-gaap:{item['us_gaap_tag']}"] = item["entity_type"]
    return tag_to_type


def validate_input(df: pd.DataFrame, taxonomy_types: dict[str, str]) -> None:
    required_columns = {"context", "category", "entity", "entity_type", "query", "answer"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing_columns)}")

    if df["context"].isna().any():
        rows = df.index[df["context"].isna()].tolist()
        raise ValueError(f"Rows with missing context: {rows}")

    invalid_types = df.index[~df["entity_type"].isin(VALID_ENTITY_TYPES)].tolist()
    if invalid_types:
        raise ValueError(f"Rows with invalid entity_type: {invalid_types}")

    missing_tags = df.index[~df["answer"].isin(taxonomy_types)].tolist()
    if missing_tags:
        raise ValueError(f"Rows whose answer tag is absent from taxonomy: {missing_tags}")

    taxonomy_type = df["answer"].map(taxonomy_types)
    mismatched_type = df.index[df["entity_type"] != taxonomy_type].tolist()
    if mismatched_type:
        raise ValueError(f"Rows whose entity_type disagrees with taxonomy type: {mismatched_type}")

    conflict_counts = df.groupby(["context", "entity", "entity_type"])["answer"].nunique()
    conflicts = conflict_counts[conflict_counts > 1]
    if len(conflicts) != 1:
        raise ValueError(
            "Expected exactly one known label-conflict group before cleaning; "
            f"found {len(conflicts)}"
        )


def create_clean_dataset(input_csv: Path, taxonomy_jsonl: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    taxonomy_types = load_taxonomy_types(taxonomy_jsonl)
    validate_input(df, taxonomy_types)

    missing_reviewed_rows = sorted(set(EXCLUDED_ROWS).difference(df.index))
    if missing_reviewed_rows:
        raise ValueError(f"Reviewed exclusion rows are absent from input: {missing_reviewed_rows}")

    clean_df = df.drop(index=sorted(EXCLUDED_ROWS)).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_csv, index=False)
    return clean_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source FinCL CSV.")
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY, help="Taxonomy JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Clean CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_df = create_clean_dataset(args.input, args.taxonomy, args.output)
    print(f"Wrote {len(clean_df)} clean rows to {args.output}")
    print("Excluded rows:")
    for row_idx, reason in sorted(EXCLUDED_ROWS.items()):
        print(f"  {row_idx}: {reason}")


if __name__ == "__main__":
    main()

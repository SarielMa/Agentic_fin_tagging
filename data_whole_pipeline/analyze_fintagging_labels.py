#!/usr/bin/env python3
"""Count datatype and tag frequencies in FinTagging_Original."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


DEFAULT_DATA_PATH = "FinTagging_Original/data/test-00000-of-00001.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze FinTagging labels. Datatype means numeric_entities[].type; "
            "tag means numeric_entities[].concept."
        )
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help="Path to the FinTagging parquet file. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        default="FinTagging_Original_stats",
        help="Directory for CSV frequency tables. Default: %(default)s",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of most frequent datatypes/tags to print. Default: %(default)s",
    )
    return parser.parse_args()


def entity_count(value: object) -> int:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return len(value)
    return 0


def build_entity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample_idx, entities in enumerate(df["numeric_entities"]):
        if not isinstance(entities, Iterable) or isinstance(entities, (str, bytes, dict)):
            continue

        context_id = df.at[sample_idx, "context_id"]
        for entity_idx, entity in enumerate(entities):
            if not isinstance(entity, dict):
                continue

            rows.append(
                {
                    "sample_idx": sample_idx,
                    "context_id": context_id,
                    "entity_idx": entity_idx,
                    "concept": entity.get("concept"),
                    "type": entity.get("type"),
                    "value": entity.get("value"),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)

    df = pd.read_parquet(data_path)
    entity_df = build_entity_table(df)

    output_dir.mkdir(parents=True, exist_ok=True)

    sample_entity_counts = df["numeric_entities"].map(entity_count)
    labeled_sample_count = int((sample_entity_counts > 0).sum())
    empty_sample_count = int((sample_entity_counts == 0).sum())

    type_counts = (
        entity_df["type"]
        .value_counts(dropna=False)
        .rename_axis("type")
        .reset_index(name="count")
    )
    tag_counts = (
        entity_df["concept"]
        .value_counts(dropna=False)
        .rename_axis("concept")
        .reset_index(name="count")
    )

    type_counts.to_csv(output_dir / "datatype_counts.csv", index=False)
    tag_counts.to_csv(output_dir / "tag_counts.csv", index=False)
    entity_df.to_csv(output_dir / "entities_flat.csv", index=False)

    print(f"Data file: {data_path}")
    print(f"Total samples: {len(df):,}")
    print(f"Samples with at least one entity: {labeled_sample_count:,}")
    print(f"Samples with no entities: {empty_sample_count:,}")
    print(f"Total entity triples: {len(entity_df):,}")
    print(f"Unique datatypes: {entity_df['type'].nunique(dropna=True):,}")
    print(f"Unique tags/concepts: {entity_df['concept'].nunique(dropna=True):,}")
    print()

    print(f"Top {args.top_k} datatypes:")
    print(type_counts.head(args.top_k).to_string(index=False))
    print()

    print(f"Top {args.top_k} tags/concepts:")
    print(tag_counts.head(args.top_k).to_string(index=False))
    print()

    print(f"Wrote: {output_dir / 'datatype_counts.csv'}")
    print(f"Wrote: {output_dir / 'tag_counts.csv'}")
    print(f"Wrote: {output_dir / 'entities_flat.csv'}")


if __name__ == "__main__":
    main()

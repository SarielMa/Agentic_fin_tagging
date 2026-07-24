#!/usr/bin/env python3
"""psi feature reduction for the sequential control arms (ags_seq_arms_spec.md section 4).

The 250-instance diagnostic ran an 11-feature psi and reached a condition number near
4,800 on A_o, which makes mu_o unstable. This script reads that run's rounds.jsonl,
measures the pairwise correlation structure of psi, and emits the reduced feature list
that ags_sequential_arms.py consumes as its default.

Reduction rules, in order:

  1. structural merges declared up front. `is_table` and `is_text` are complementary
     indicators over the two modalities, so they are a rank-deficient block against
     `bias` by construction: keep `is_table`, drop `is_text`. This is a merge, not a
     data-driven drop -- the diagnostic was tabular-only, so `is_text` is constant there
     and a variance rule would drop it for the wrong reason.
  2. constant features (zero variance) other than `bias` and the representative kept by a
     structural merge. Constancy in these logs is a property of the diagnostic sample --
     it was tabular-only -- not of the feature, and the sequential arms run on both
     modalities, so `is_table` is retained despite carrying no variance here.
  3. connected components of the |r| > 0.9 graph; the earliest surviving feature in the
     canonical order represents its component.

The condition-number trajectory is reported for the full and the retained feature set on
the same record stream, so the reduction can be judged rather than asserted. Features
retained on structural grounds are collinear with `bias` *in this log* (constant), which
would report an unbounded condition number for a set that is well conditioned on the real
population; the `measurable` trajectory drops them and is the one to read.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from run_fintagging_grounding_baseline import SCRIPT_DIR


# Canonical order of the diagnostic's psi. Earlier features represent their correlation
# component, so this order is also the retention priority.
DIAGNOSTIC_PSI_ORDER = (
    "bias",
    "is_table",
    "is_text",
    "is_monetary",
    "is_shares",
    "is_percent",
    "D_plus_count",
    "D_minus_count",
    "D_question_count",
    "structural_mismatch_g",
    "neighborhood_novelty_n",
)
ALWAYS_KEEP = ("bias",)
STRUCTURAL_MERGES = (
    {
        "block": ["is_table", "is_text"],
        "keep": "is_table",
        "reason": "complementary modality indicators; rank-deficient against bias by construction",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rounds-jsonl",
        type=Path,
        default=SCRIPT_DIR / "runs_ags_reward_diagnostic" / "qwen3_32b" / "rounds.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "runs_ags_seq" / "psi_reduction",
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.9)
    parser.add_argument("--posterior-ridge", type=float, default=1.0)
    parser.add_argument(
        "--condition-checkpoints",
        default="25,50,100,250,500,1000",
        help="Record counts at which the condition number of A is reported.",
    )
    parser.add_argument("--limit-rows", type=int, default=None)
    return parser.parse_args()


def stream_psi_rows(path: Path, limit: int | None) -> tuple[list[dict[str, float]], list[str], dict[str, int]]:
    """Pull the named psi dict out of each rounds.jsonl row without holding the file."""
    rows: list[dict[str, float]] = []
    arm_counts: dict[str, int] = {}
    names: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            psi = record.get("psi")
            if not isinstance(psi, dict) or not psi:
                continue
            if not names:
                names = [name for name in DIAGNOSTIC_PSI_ORDER if name in psi]
                names += [name for name in psi if name not in names]
            rows.append({name: float(psi.get(name, 0.0)) for name in names})
            arm = str(record.get("arm", "unknown"))
            arm_counts[arm] = arm_counts.get(arm, 0) + 1
            if limit is not None and len(rows) >= limit:
                break
    return rows, names, arm_counts


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Pearson correlation with zero-variance columns reported as 0 rather than nan."""
    std = matrix.std(axis=0)
    safe = np.where(std > 1e-12, std, 1.0)
    centered = (matrix - matrix.mean(axis=0)) / safe
    corr = (centered.T @ centered) / max(matrix.shape[0], 1)
    corr[std <= 1e-12, :] = 0.0
    corr[:, std <= 1e-12] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def condition_trajectory(
    matrix: np.ndarray,
    ridge: float,
    checkpoints: list[int],
) -> list[dict[str, Any]]:
    """cond(A) after the first n records, with A = ridge*I + sum psi psi^T."""
    dim = matrix.shape[1]
    a = np.eye(dim) * ridge
    trajectory: list[dict[str, Any]] = []
    wanted = sorted({point for point in checkpoints if point > 0})
    pointer = 0
    for index in range(matrix.shape[0]):
        psi = matrix[index]
        a += np.outer(psi, psi)
        while pointer < len(wanted) and index + 1 == wanted[pointer]:
            trajectory.append(
                {
                    "n_records": index + 1,
                    "dimension": dim,
                    "condition_number": round(float(np.linalg.cond(a)), 4),
                }
            )
            pointer += 1
    trajectory.append(
        {
            "n_records": int(matrix.shape[0]),
            "dimension": dim,
            "condition_number": round(float(np.linalg.cond(a)), 4),
        }
    )
    return trajectory


def reduce_features(
    names: list[str],
    matrix: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    corr = correlation_matrix(matrix)
    std = matrix.std(axis=0)
    index_of = {name: idx for idx, name in enumerate(names)}

    dropped: dict[str, str] = {}
    protected: set[str] = set(ALWAYS_KEEP)
    for merge in STRUCTURAL_MERGES:
        present = [name for name in merge["block"] if name in index_of]
        if len(present) < 2 or merge["keep"] not in present:
            continue
        protected.add(merge["keep"])
        for name in present:
            if name != merge["keep"]:
                dropped[name] = f"structural_merge_into:{merge['keep']} ({merge['reason']})"

    for name in names:
        if name in dropped or name in protected:
            continue
        if std[index_of[name]] <= 1e-12:
            dropped[name] = "constant_in_diagnostic_logs"

    # Connected components over the surviving |r| > threshold graph.
    survivors = [name for name in names if name not in dropped]
    component_of: dict[str, str] = {}
    for position, name in enumerate(survivors):
        if name in protected:
            continue
        for other in survivors[:position]:
            if other in dropped:
                continue
            r = float(corr[index_of[name], index_of[other]])
            if abs(r) > threshold:
                representative = component_of.get(other, other)
                dropped[name] = f"correlated_with:{representative} (r={round(r, 4)})"
                component_of[name] = representative
                break

    retained = [name for name in names if name not in dropped]
    high_pairs = [
        {
            "left": names[i],
            "right": names[j],
            "r": round(float(corr[i, j]), 6),
        }
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if abs(float(corr[i, j])) > threshold
    ]
    return {
        "retained_features": retained,
        "dropped_features": dropped,
        "protected_features": sorted(protected),
        "high_correlation_pairs": high_pairs,
        "correlation_matrix": {
            names[i]: {names[j]: round(float(corr[i, j]), 6) for j in range(len(names))}
            for i in range(len(names))
        },
        "feature_std": {name: round(float(std[index_of[name]]), 6) for name in names},
        "feature_mean": {
            name: round(float(matrix[:, index_of[name]].mean()), 6) for name in names
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    checkpoints = [int(value) for value in str(args.condition_checkpoints).split(",") if value.strip()]
    rows, names, arm_counts = stream_psi_rows(args.rounds_jsonl, args.limit_rows)
    if not rows:
        raise SystemExit(f"No psi records found in {args.rounds_jsonl}")

    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=float)
    reduction = reduce_features(names, matrix, args.correlation_threshold)
    retained = reduction["retained_features"]
    retained_matrix = matrix[:, [names.index(name) for name in retained]]

    # Features retained on structural grounds carry no variance in a tabular-only log, so
    # the retained trajectory is degenerate there; `measurable` is the honest read.
    measurable = [
        name
        for name in retained
        if name in ALWAYS_KEEP or float(matrix[:, names.index(name)].std()) > 1e-12
    ]
    measurable_matrix = matrix[:, [names.index(name) for name in measurable]]

    full_trajectory = condition_trajectory(matrix, args.posterior_ridge, checkpoints)
    retained_trajectory = condition_trajectory(retained_matrix, args.posterior_ridge, checkpoints)
    measurable_trajectory = condition_trajectory(measurable_matrix, args.posterior_ridge, checkpoints)

    metrics = {
        "measurable_features": measurable,
        "measurable_dimension": len(measurable),
        "condition_number_measurable": measurable_trajectory,
        "condition_number_measurable_final": measurable_trajectory[-1]["condition_number"],
        "degenerate_in_this_log": [name for name in retained if name not in measurable],
        "source_rounds_jsonl": str(args.rounds_jsonl),
        "records": len(rows),
        "records_by_arm": arm_counts,
        "correlation_threshold": args.correlation_threshold,
        "posterior_ridge": args.posterior_ridge,
        "full_features": names,
        "full_dimension": len(names),
        "retained_dimension": len(retained),
        "structural_merges": [dict(merge) for merge in STRUCTURAL_MERGES],
        "condition_number_full": full_trajectory,
        "condition_number_retained": retained_trajectory,
        "condition_number_full_final": full_trajectory[-1]["condition_number"],
        "condition_number_retained_final": retained_trajectory[-1]["condition_number"],
        **reduction,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "psi_reduction.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(
        args.output_dir / "condition_number_trajectory.csv",
        [{"feature_set": "full", **row} for row in full_trajectory]
        + [{"feature_set": "retained", **row} for row in retained_trajectory]
        + [{"feature_set": "measurable", **row} for row in measurable_trajectory],
    )
    write_csv(
        args.output_dir / "psi_correlations.csv",
        [
            {
                "left": left,
                "right": right,
                "r": value,
                "above_threshold": abs(value) > args.correlation_threshold,
            }
            for left, row in reduction["correlation_matrix"].items()
            for right, value in row.items()
            if left < right
        ],
    )

    print(
        json.dumps(
            {
                "records": len(rows),
                "full_dimension": len(names),
                "retained_dimension": len(retained),
                "retained_features": retained,
                "dropped_features": reduction["dropped_features"],
                "degenerate_in_this_log": metrics["degenerate_in_this_log"],
                "condition_number_full_final": metrics["condition_number_full_final"],
                "condition_number_retained_final": metrics["condition_number_retained_final"],
                "condition_number_measurable_final": metrics["condition_number_measurable_final"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

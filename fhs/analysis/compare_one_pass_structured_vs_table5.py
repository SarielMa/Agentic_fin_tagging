#!/usr/bin/env python3
"""Cross-check the "One-pass grounding (structured)" baseline against the Table 5 J=1 row.

WHAT THE SPEC ASKED FOR, AND WHY IT CANNOT BE ASSERTED AS WRITTEN
----------------------------------------------------------------
The baseline spec says the new row is "the same configuration as the Table 5 ablation row
'- ensemble (J=1, single hypothesis)'" and asks to assert the two cells are numerically equal,
"the same run reached from two directions". They are not the same configuration. Three
independent differences, each verifiable in the code:

  1. The Table 5 row is a MEAN OF TWO evaluations, not one. run_test_rows.py:222-237 evaluates
     J=1 twice -- once keeping hypothesis idx0, once idx1 -- and reports the average, as
     section 3.2's deliberate mitigation for seed noise. The baseline draws one hypothesis.

  2. The Table 5 row runs at BETA=0.6, not 0. `ensemble_beta = selected["ensemble"]
     ["selected_beta"]` (run_test_rows.py:222), and that value is 0.6 in selected_betas.json.
     With J=1 the consensus term is agree(concept, h) against the single hypothesis, which is
     NOT identically zero, so beta=0.6 genuinely reorders the pool. The baseline spec fixes
     beta=0 ("with one hypothesis there is nothing to reach consensus over").

  3. The Table 5 row REPLAYS hypotheses that AGS sampled at temperature 0.8; the baseline
     GENERATES one greedily (temperature 0). Different text, therefore different queries.

So this script does not assert that. It asserts the equality that IS true and that was almost
certainly the intent -- that the online and offline code paths agree on the same inputs:

  ASSERTED: replaying the baseline's OWN logged trace through verifier.core.evaluate
  at J=1 / beta=0 / kept_hypothesis_idx=0 reproduces the metrics the online run recorded.
  That is genuinely "the same run reached from two directions", and it is what validates that
  the new row and the Table 5 machinery implement the same method.

  REPORTED (not asserted): the Table 5 -ensemble cell alongside the baseline, with the three
  differences above quantified, including a beta=0.6 replay of the baseline's own trace so the
  beta contribution is separated from the generation difference.

Usage:
  python compare_one_pass_structured_vs_table5.py \
      --trace runs_fintagging_grounding_baseline/qwen3_32b_one_pass_structured/bm25_candidates.jsonl \
      --table5-csv runs_ags_table5_ablation/qwen3_32b_fixed/ablation.csv
"""

from __future__ import annotations
# --- resolve local packages regardless of this file's depth in the tree ---
import sys as _sys, pathlib as _pathlib
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "src" / "run_fintagging_grounding_baseline.py").exists():
        _sys.path.insert(0, str(_p / "src"))
        _sys.path.insert(0, str(_p / "analysis"))
        FHS_ROOT = _p
        break
# -------------------------------------------------------------------------

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from verifier.core import (  # noqa: E402
    AblationConfig,
    FactRecord,
    aggregate,
    evaluate,
    reset_consensus_cache,
)
from verifier.data_prep import _compact, stream_jsonl  # noqa: E402
from run_fintagging_grounding_baseline import normalize_tag  # noqa: E402

METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr", "top1_accuracy")
TABLE5_ENSEMBLE_ROW = "-ensemble (mean of idx0/idx1)"


def load_baseline_trace(path: Path, limit: int | None) -> list[dict[str, Any]]:
    """One FactRecord-shaped dict per fact, plus the metrics the ONLINE run recorded."""
    facts: list[dict[str, Any]] = []
    for record in stream_jsonl(path):
        if limit is not None and len(facts) >= limit:
            break
        rankings: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for round_record in record.get("rounds", []):
            if round_record.get("label_render_skipped"):
                continue
            key = (int(round_record["hypothesis_idx"]), round_record["rendering"])
            rankings[key] = [_compact(candidate) for candidate in round_record.get("candidates", [])]
        facts.append(
            {
                "fact_id": int(record["example_idx"]),
                "context_id": record.get("context_id"),
                "modality": record.get("input_type", ""),
                "datatype": record.get("type", ""),
                "gold_tags": [normalize_tag(tag) for tag in record.get("gold_tags", [])],
                "hypotheses": {
                    int(h["hypothesis_idx"]): {
                        "dimensions": h.get("dimensions", {}),
                        "operators": h.get("operators", []),
                        "retrieval_query": h.get("retrieval_query", ""),
                    }
                    for h in record.get("frozen_ags_hypotheses", [])
                },
                "rankings": rankings,
                "online_metrics": record.get("retrieval_metrics") or {},
                "config": record.get("frozen_ags_config") or {},
            }
        )
    return facts


def replay(facts: list[dict[str, Any]], beta: float, normalization_map: dict[str, Any]) -> list[dict[str, Any]]:
    """Offline consolidation of the baseline's own rankings at the given beta."""
    reset_consensus_cache()  # the cache is not keyed by beta or by which run produced the pool
    config = AblationConfig(
        name=f"one_pass_structured replay (beta={beta})",
        n_hypotheses=1,
        kept_hypothesis_idx=0,
        beta=beta,
    )
    rows = []
    for fact in facts:
        rows.append(
            evaluate(
                FactRecord(
                    fact_id=fact["fact_id"],
                    context_id=fact["context_id"],
                    modality=fact["modality"],
                    datatype=fact["datatype"],
                    gold_tags=fact["gold_tags"],
                    hypotheses=fact["hypotheses"],
                    rankings=fact["rankings"],
                ),
                config,
                normalization_map,
            )
        )
    return rows


def by_modality(rows: list[dict[str, Any]], modality: str) -> list[dict[str, Any]]:
    return rows if modality == "pooled" else [r for r in rows if r["modality"] == modality]


def read_table5_cell(path: Path, row_name: str) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    out: dict[tuple[str, str], float] = {}
    for record in csv.DictReader(path.open(encoding="utf-8")):
        if record.get("variant") == row_name:
            out[(record["modality"], record["metric"])] = float(record["value"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--trace",
        type=Path,
        default=_HERE / "runs_fintagging_grounding_baseline" / "qwen3_32b_one_pass_structured" / "bm25_candidates.jsonl",
    )
    parser.add_argument("--table5-csv", type=Path, default=None, help="Optional ablation.csv for the reported comparison.")
    parser.add_argument("--normalization-map", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    normalization_map = load_normalization_map(args.normalization_map or DEFAULT_NORMALIZATION_MAP)
    facts = load_baseline_trace(args.trace, args.limit)
    if not facts:
        raise SystemExit(f"No facts loaded from {args.trace}")
    print(f"Loaded {len(facts)} facts from {args.trace}", flush=True)

    stated = facts[0]["config"]
    print(f"Trace config: {json.dumps(stated, sort_keys=True)}", flush=True)
    config_ok = stated.get("hypotheses") == 1 and stated.get("rerank_beta") == 0.0
    if not config_ok:
        print("  WARNING: trace was not produced at J=1/beta=0; the assertion below is meaningless.")

    # ---- the assertion that must hold: online == offline replay, same inputs, beta=0 -------
    offline_rows = replay(facts, 0.0, normalization_map)
    mismatches = []
    for fact, row in zip(facts, offline_rows):
        online = fact["online_metrics"]
        if not online:
            continue
        for metric in METRICS:
            if metric not in online or metric not in row:
                continue
            if abs(float(online[metric]) - float(row[metric])) > args.tolerance:
                mismatches.append((fact["fact_id"], metric, float(online[metric]), float(row[metric])))

    print()
    print("=== ASSERTED: online run == offline replay of its own trace (J=1, beta=0) ===")
    if mismatches:
        print(f"  FAILED: {len(mismatches)} per-fact metric mismatches; first 5:")
        for fact_id, metric, a, b in mismatches[:5]:
            print(f"    fact {fact_id} {metric}: online={a} offline={b}")
    else:
        print(f"  PASSED: all {len(facts)} facts agree on {', '.join(METRICS)}")

    # ---- reported, not asserted -------------------------------------------------------------
    beta06_rows = replay(facts, 0.6, normalization_map)
    table5 = read_table5_cell(args.table5_csv, TABLE5_ENSEMBLE_ROW) if args.table5_csv else {}

    print()
    print("=== REPORTED: baseline vs the Table 5 J=1 row (these are NOT the same config) ===")
    print("  differences: Table 5 averages idx0/idx1, runs at beta=0.6, and replays")
    print("  temperature-0.8 samples; the baseline is one greedy hypothesis at beta=0.")
    print()
    header = f"  {'modality':8s} {'metric':14s} {'baseline':>10s} {'same@beta=.6':>13s} {'Table5 J=1':>11s}"
    print(header)
    summary: dict[str, Any] = {}
    for modality in ("table", "text", "pooled"):
        base_agg = aggregate(by_modality(offline_rows, modality))
        b06_agg = aggregate(by_modality(beta06_rows, modality))
        if not base_agg.get("n"):
            continue
        for metric in METRICS:
            cell = table5.get((modality, metric))
            summary[f"{modality}/{metric}"] = {
                "baseline_beta0": base_agg[metric],
                "baseline_beta0.6": b06_agg[metric],
                "table5_ensemble_mean_idx0_idx1": cell,
            }
            cell_text = f"{cell:11.4f}" if cell is not None else f"{'--':>11s}"
            print(f"  {modality:8s} {metric:14s} {base_agg[metric]:10.4f} {b06_agg[metric]:13.4f} {cell_text}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "n_facts": len(facts),
                    "trace": str(args.trace),
                    "trace_config": stated,
                    "online_equals_offline_replay": not mismatches,
                    "n_mismatches": len(mismatches),
                    "comparison": summary,
                    "why_not_equal_to_table5": [
                        "Table 5's row is the mean of two J=1 evaluations (idx0 and idx1); the baseline draws one hypothesis.",
                        "Table 5's row runs at beta=0.6 (selected_betas.json ensemble.selected_beta); the baseline fixes beta=0.",
                        "Table 5 replays temperature-0.8 AGS samples; the baseline generates greedily at temperature 0.",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.output_json}")

    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

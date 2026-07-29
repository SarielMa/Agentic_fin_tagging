#!/usr/bin/env python3
"""The "Rd-1 -> full" column of the sequential-methods table: change in Recall@50 from
round-one candidates alone to the full episode, per method, on the test split.

WHY THIS EXISTS SEPARATELY FROM compute_ags_seq_arm_metrics.py
--------------------------------------------------------------
That script computes the same column, but only for the three AGS arms (--ags-dir, --seq-dir,
--seq-random-dir) and only from the `round1_candidates` field, which only the ags_seq record
builder writes. The four free-form iterative methods log per-round candidates but no such
field, so the column could not be produced for them at all. This script covers every method by
picking the right round-one definition per method (below).

TWO THINGS THE CALLER MUST CARRY INTO THE CAPTION
-------------------------------------------------
1. This is a RETRIEVAL-stage quantity, while the other columns of that table are rerank-stage.
   That is inherent, not an oversight: the listwise reranker runs exactly once, over the final
   candidate pool, so a "reranked round one" does not exist and cannot be constructed after the
   fact. Sanity check: the full-episode number here reproduces metrics.json's
   bm25_retrieval.recall_at_50 for every method (asserted below with --check).

2. "Round one" is NOT the same unit across methods:
     - free-form iterative methods: rounds[0] -- ONE query, 200 candidates.
     - AGS-Seq arms: the CONSOLIDATED round-one fan (J hypotheses x both renderings), which is
       what `round1_candidates` holds.
   So the round-one LEVELS are not comparable across the two groups (the AGS arms start far
   higher, ~0.55 vs ~0.39, because their first round is already an ensemble). Only the deltas
   are comparable, and even they answer "how much does iterating add" over first rounds of
   different size. Single-pass methods have no round-one/full distinction and report "---".

Usage:
  python compute_round1_vs_full.py                      # table to stdout
  python compute_round1_vs_full.py --check              # also assert against metrics.json
  python compute_round1_vs_full.py --output-json runs_ags_seq/round1_vs_full.json
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
import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_fintagging_grounding_baseline import normalize_tag  # noqa: E402

DEFAULT_RUNS_ROOT = _HERE / "runs_fintagging_grounding_baseline"
RECALL_K = 50

# (run directory suffix, table label, round-one source)
#   "rounds0"        -> rounds[0].candidates, the method's first retrieval
#   "round1_field"   -> the record's own round1_candidates (consolidated round-one fan)
METHODS: tuple[tuple[str, str, str], ...] = (
    ("intrinsic_self_refinement", "Intrinsic self-refinement", "rounds0"),
    ("retrieval_feedback_refinement", "Retrieval-feedback refinement", "rounds0"),
    ("operator_refinement", "Operator refinement", "rounds0"),
    ("memory_guided_refinement", "Memory-guided refinement", "rounds0"),
    ("ags_seq", "AGS-Seq (learned selection)", "round1_field"),
    ("ags_seq_random", "AGS-Seq-random", "round1_field"),
)


def hit_at_k(tags: list[str], gold: list[str], k: int) -> float:
    """Recall@k in this codebase's sense: did ANY gold tag land in the top k."""
    top = {normalize_tag(tag) for tag in tags[:k]}
    return 1.0 if any(normalize_tag(g) in top for g in gold) else 0.0


def round_one_tags(record: dict[str, Any], source: str) -> list[str] | None:
    if source == "round1_field":
        tags = record.get("round1_candidates")
        return list(tags) if tags else None
    rounds = record.get("rounds") or []
    if not rounds:
        return None
    return [candidate["tag"] for candidate in rounds[0].get("candidates", [])]


def full_tags(record: dict[str, Any]) -> list[str]:
    return record.get("candidate_union_tags") or [c["tag"] for c in record.get("candidates", [])]


def measure(path: Path, source: str) -> dict[str, Any]:
    n = 0
    round1_sum = 0.0
    full_sum = 0.0
    skipped = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            gold = record.get("gold_tags") or []
            if not gold:
                skipped += 1
                continue
            r1 = round_one_tags(record, source)
            if r1 is None:
                skipped += 1
                continue
            n += 1
            round1_sum += hit_at_k(r1, gold, RECALL_K)
            full_sum += hit_at_k(full_tags(record), gold, RECALL_K)
    if not n:
        raise SystemExit(f"No usable records in {path}")
    return {
        "n": n,
        "skipped": skipped,
        "round1_recall_at_50": round(round1_sum / n, 6),
        "full_recall_at_50": round(full_sum / n, 6),
        "delta": round((full_sum - round1_sum) / n, 6),
        "round1_source": source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--prefix", default="qwen3_32b_")
    parser.add_argument("--check", action="store_true", help="Assert full R@50 == metrics.json bm25_retrieval.")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    results: dict[str, Any] = {}
    failures: list[str] = []
    print(f"{'method':32s} {'n':>5s} {'Rd-1 R@50':>10s} {'full R@50':>10s} {'delta':>9s}")
    for suffix, label, source in METHODS:
        run_dir = args.runs_root / f"{args.prefix}{suffix}"
        trace = run_dir / "bm25_candidates.jsonl"
        if not trace.exists():
            print(f"{label:32s} {'--':>5s} {'(no trace)':>10s}")
            continue
        stats = measure(trace, source)
        results[label] = stats
        print(
            f"{label:32s} {stats['n']:5d} {stats['round1_recall_at_50']:10.4f} "
            f"{stats['full_recall_at_50']:10.4f} {stats['delta']:+9.4f}"
        )
        if args.check:
            metrics_path = run_dir / "metrics.json"
            if metrics_path.exists():
                reference = json.loads(metrics_path.read_text(encoding="utf-8"))["bm25_retrieval"]["recall_at_50"]
                gap = abs(reference - stats["full_recall_at_50"])
                stats["metrics_json_recall_at_50"] = reference
                stats["matches_metrics_json"] = gap <= args.tolerance
                if gap > args.tolerance:
                    failures.append(f"{label}: computed {stats['full_recall_at_50']} vs metrics.json {reference}")

    if args.check:
        print()
        if failures:
            print("CHECK FAILED against metrics.json bm25_retrieval.recall_at_50:")
            for line in failures:
                print(f"  {line}")
        else:
            print("CHECK PASSED: full-episode R@50 reproduces metrics.json for every method with one.")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(
                {
                    "recall_k": RECALL_K,
                    "stage": "retrieval (round-one candidates are never reranked)",
                    "results": results,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.output_json}")

    if args.check and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

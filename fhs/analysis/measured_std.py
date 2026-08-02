#!/usr/bin/env python3
"""Measured run-to-run mean and standard deviation per table row.

Replaces the analytic `std` column (sqrt(p(1-p)/n + g^2)) with the spread actually observed when
one configuration is run more than once. A repeat re-runs the same configuration end to end;
there is no generation-seed knob in this pipeline, so what varies is temperature-0.8 sampling in
the ensemble arms and vLLM batching non-determinism elsewhere.

Prints, per row, n and the sample standard deviation (ddof=1) of every reported metric. A row
with n=1 has no measured std and is reported as such rather than filled in with the analytic
number -- mixing the two estimators in one column is what this script exists to stop.

    python fhs/analysis/measured_std.py [--json fhs/results/measured_std.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import sys as _sys, pathlib as _pathlib
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "src" / "run_fintagging_grounding_baseline.py").exists():
        FHS_ROOT = _p
        break

PIPELINE = FHS_ROOT.parent / "data_whole_pipeline"
BASE = PIPELINE / "runs_fintagging_grounding_baseline"
ABL = PIPELINE / "runs_ags_verifier_ablation" / "qwen3_32b"
VAR = FHS_ROOT / "runs" / "variance"

METRICS = ("R@1", "R@10", "R@50", "MRR", "Acc")
_KEYS = {"R@1": "accuracy", "R@10": "recall_at_10", "R@50": "recall_at_50", "MRR": "mrr"}

# table row -> the run directories that are repeats of ONE configuration.
# The deployed run comes first in each list.
ROWS: dict[str, list[Path]] = {
    # --- tab:main_results ---
    "Direct retr.": [BASE / "qwen3_32b_direct_retrieval_wcov1"],
    "One-pass, free-text": [BASE / "qwen3_32b_one_pass_grounding_wcov1"],
    # the deployed structured run has no _wcov1 suffix: it was already built with the term on
    "One-pass, structured": [BASE / "qwen3_32b_one_pass_structured"],
    "Parallel, stochastic": [BASE / "qwen3_32b_parallel_sampling_wcov1_j2"],
    "Decomposed": [BASE / "qwen3_32b_decomposed_retrieval_wcov1"],
    "Intrinsic refine.": [BASE / "qwen3_32b_intrinsic_self_refinement_wcov1"],
    "Feedback refine.": [BASE / "qwen3_32b_retrieval_feedback_refinement_wcov1"],
    "FHS (full)": [ABL / "rerank_arm6_full"],
    # --- tab:ablation (the rows whose configuration is repeated) ---
    "Program-driven score": [
        ABL / "rerank_no_llm",
        VAR / "rep1", VAR / "rep2", VAR / "rep3",
    ],
    "- verifier": [ABL / "rerank_no_verifier"],
}
# repeats submitted 2026-08-02 land next to the deployed run with these suffixes
for label, arm in (
    ("Direct retr.", "direct_retrieval"),
    ("One-pass, free-text", "one_pass_grounding"),
    ("One-pass, structured", "one_pass_structured"),
    ("Parallel, stochastic", "parallel_sampling"),
    ("Decomposed", "decomposed_retrieval"),
    ("Intrinsic refine.", "intrinsic_self_refinement"),
    ("Feedback refine.", "retrieval_feedback_refinement"),
):
    ROWS[label] += [BASE / f"qwen3_32b_{arm}_wcov1_rep{r}" for r in (2, 3)]
ROWS["FHS (full)"] += [VAR / f"rep{r}" / "rerank_arm6" for r in (1, 2)]


def read(run: Path) -> dict[str, float] | None:
    p = run / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    ret, fin = m.get("bm25_retrieval"), m.get("qwen_reranked")
    if not isinstance(ret, dict) or not isinstance(fin, dict):
        return None
    if int(ret.get("n", 0)) != 2509:          # a partial run must never enter a std
        return None
    out = {k: float(ret[v]) for k, v in _KEYS.items()}
    out["Acc"] = float(fin["accuracy"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    result = {}
    print(f"{'row':<24}{'n':>3}  " + "".join(f"{m+' mean':>12}{'std':>9}" for m in METRICS))
    for label, runs in ROWS.items():
        vals = [v for v in (read(r) for r in runs) if v]
        n = len(vals)
        line = f"{label:<24}{n:>3}  "
        row = {"n": n, "runs": [str(r) for r in runs if (r / 'metrics.json').exists()]}
        for m in METRICS:
            xs = [v[m] for v in vals]
            mean = statistics.fmean(xs) if xs else float("nan")
            sd = statistics.stdev(xs) if n >= 2 else None
            row[m] = {"mean": mean, "std": sd, "values": xs}
            line += f"{mean:>12.4f}" + (f"{sd:>9.4f}" if sd is not None else f"{'--':>9}")
        print(line)
        result[label] = row

    missing = [k for k, v in result.items() if v["n"] < 2]
    if missing:
        print(f"\nno measured std yet ({len(missing)} rows): " + ", ".join(missing))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()

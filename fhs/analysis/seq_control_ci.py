#!/usr/bin/env python3
"""Context-clustered bootstrap intervals for the sequential control (tab:seq_outcome).

The table reports Recall@50 after round one against the full episode for FHS-Seq, plus the
share of the top-50 pool the later rounds replace. Only the delta carried an interval, and
only in the caption; this script produces intervals for every cell from the run's own trace.

WHICH RUN. The FHS-Seq row is the seq-verifier run -- FHS-Seq keeps the candidate-level
verifier, so the two ags_seq arms in runs_ags_seq_diag/ are NOT this row (they end at R@50
0.559 / 0.560, and their round-one delta is positive). It was executed as four shards,
qwen3_32b_seq_verifier_s0..s3, which together cover the 2,509 test facts; `example_idx` is
per-shard, so facts are keyed by (shard, example_idx) and clustered by `context_id`, which is
global.

ESTIMATOR. Identical to compute_ags_seq_diagnostics.context_bootstrap: resample source
contexts with replacement, weight each context by its fact count, take the 2.5/97.5 percentiles
of 2,000 draws under seed 20260725. Resampling contexts rather than facts is the point -- this
benchmark puts ~13 facts under one table, and a fact-level bootstrap would understate the
interval. The delta is paired: the difference is formed per fact first, then clustered.

CPU only, seconds to run, no GPU:
  python seq_control_ci.py
  python seq_control_ci.py --output-json .../seq_control_ci.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

_HERE = Path(__file__).resolve().parent
DEFAULT_RUNS_ROOT = _HERE.parents[1] / "data_whole_pipeline" / "runs_fintagging_grounding_baseline"
SHARDS = ("qwen3_32b_seq_verifier_s0", "qwen3_32b_seq_verifier_s1",
          "qwen3_32b_seq_verifier_s2", "qwen3_32b_seq_verifier_s3")
DEPTH = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--shards", nargs="+", default=list(SHARDS))
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def normalize(tag: Any) -> str:
    return str(tag or "").strip().lower()


def load_facts(runs_root: Path, shards: list[str]) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for shard in shards:
        path = runs_root / shard / "bm25_candidates.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                gold = {normalize(t) for t in (record.get("gold_tags") or [])}
                final = [normalize(c.get("tag")) for c in
                         (record.get("final_candidates") or record.get("candidates") or [])][:DEPTH]
                round1 = [normalize(t) for t in (record.get("round1_candidates") or [])][:DEPTH]
                if not final or not round1:
                    raise SystemExit(
                        f"{shard} fact {record.get('example_idx')}: missing final or round-one "
                        "candidates; the churn and delta columns cannot be formed from it."
                    )
                facts.append({
                    "context_id": record["context_id"],
                    "round1_r50": float(bool(gold & set(round1))),
                    "final_r50": float(bool(gold & set(final))),
                    "membership_changed": 1.0 - len(set(final) & set(round1)) / len(final),
                })
    return facts


def context_bootstrap(facts: list[dict[str, Any]], value: Callable[[dict[str, Any]], float],
                      iterations: int, seed: int) -> dict[str, Any]:
    by_context: dict[Any, list[float]] = defaultdict(list)
    for fact in facts:
        by_context[fact["context_id"]].append(float(value(fact)))
    contexts = list(by_context)
    means = np.asarray([float(np.mean(by_context[c])) for c in contexts])
    sizes = np.asarray([len(by_context[c]) for c in contexts], dtype=float)
    observed = float(np.sum(means * sizes) / np.sum(sizes))

    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        picks = rng.integers(0, len(contexts), size=len(contexts))
        weights = sizes[picks]
        draws[index] = float(np.sum(means[picks] * weights) / np.sum(weights))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"mean": round(observed, 6), "ci_low": round(float(low), 6),
            "ci_high": round(float(high), 6), "facts": len(facts), "contexts": len(contexts),
            "ci_excludes_zero": bool(low > 0 or high < 0)}


def main() -> None:
    args = parse_args()
    facts = load_facts(args.runs_root, args.shards)
    boot = lambda fn: context_bootstrap(facts, fn, args.bootstrap_samples, args.seed)  # noqa: E731
    result = {
        "round1_r50": boot(lambda f: f["round1_r50"]),
        "full_r50": boot(lambda f: f["final_r50"]),
        "delta_full_minus_round1": boot(lambda f: f["final_r50"] - f["round1_r50"]),
        "membership_changed": boot(lambda f: f["membership_changed"]),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "shards": args.shards,
    }
    for key in ("round1_r50", "full_r50", "delta_full_minus_round1", "membership_changed"):
        row = result[key]
        print(f"{key:26s} {row['mean']:+.4f}  [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
              f"  facts={row['facts']} contexts={row['contexts']}")
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n",
                                    encoding="utf-8")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

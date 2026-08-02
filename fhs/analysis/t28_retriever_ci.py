#!/usr/bin/env python3
"""Paired context-level bootstrap for tab:retriever_robustness, on MRR and on final accuracy.

The appendix claims dense and hybrid retrieval do not improve on BM25, which is why the paper
uses BM25 throughout. That was a point difference with no test. This supplies the test: for each
(retriever, method) row, the difference against the SAME METHOD under BM25, with a 95% interval
from the paper's estimator -- source context as the resampling unit, seed 20260724, 2,000
resamples -- and the same context resample applied to both sides, since every run scores the same
2,509 facts.

Two metrics, because the table reports both and they are measured at different stages:
  MRR   retrieval stage, from the ranking file's `gold_rank` (Section: metrics convention)
  Acc.  after the shared listwise selector, from `qwen_rerank_predictions.jsonl`

SELF-CHECK: every ranking file's pooled MRR is compared against the MRR its own run published;
a mismatch above --tolerance aborts, so a wrong or half-written file cannot reach the table.

    python fhs/analysis/t28_retriever_ci.py --json fhs/results/t28_retriever_ci.json
"""
from __future__ import annotations

# --- resolve local packages regardless of this file's depth in the tree ---
import sys as _sys, pathlib as _pathlib
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "src" / "run_fintagging_grounding_baseline.py").exists():
        _sys.path.insert(0, str(_p / "src"))
        FHS_ROOT = _p
        break
# -------------------------------------------------------------------------
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_fintagging_grounding_baseline import normalize_tag

PIPELINE = FHS_ROOT.parent / "data_whole_pipeline"
BASE = PIPELINE / "runs_fintagging_grounding_baseline"
ABL = PIPELINE / "runs_ags_verifier_ablation" / "qwen3_32b"
W1 = FHS_ROOT / "runs" / "t28_wcov1"
VF = FHS_ROOT / "runs" / "t28_verifier"

# (retriever, method) -> ranking dir (gold_rank -> MRR), predictions dir (Acc), published MRR.
# published_mrr is the self-check target; None means the run publishes no metrics.json yet, in
# which case the file is trusted only for the contrast and flagged in the output.
ROWS = {
    ("BM25", "one_pass"): (BASE / "qwen3_32b_one_pass_grounding_wcov1", BASE / "qwen3_32b_one_pass_grounding_wcov1"),
    ("BM25", "FHS"): (ABL / "rerank_arm6_full", ABL / "rerank_arm6_full"),
    ("Dense", "one_pass"): (W1 / "dense_one_pass", W1 / "dense_one_pass"),
    ("Dense", "FHS"): (VF / "dense" / "rerank_arm6", VF / "dense" / "rerank_arm6"),
    ("Hybrid", "one_pass"): (W1 / "hybrid_one_pass", W1 / "hybrid_one_pass"),
    ("Hybrid", "FHS"): (VF / "hybrid" / "rerank_arm6", VF / "hybrid" / "rerank_arm6"),
}
REFERENCE = "BM25"
SEED, SAMPLES = 20260724, 2000
EXPECTED_FACTS = 2509

# The ranking files run to gigabytes and only three top-level fields are needed, so pull them
# with a regex instead of parsing every candidate list.
_CTX = re.compile(rb'"context_id":\s*(\d+)')
_GOLD = re.compile(rb'"gold_rank":\s*(null|\d+)')


def per_context_rr(path: Path) -> tuple[dict[int, float], dict[int, int]]:
    """Sum of reciprocal ranks and fact count, per source context."""
    num: dict[int, float] = defaultdict(float)
    den: dict[int, int] = defaultdict(int)
    with path.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            ctx = _CTX.search(line)
            gold = _GOLD.search(line)
            if ctx is None or gold is None:
                raise SystemExit(f"{path}: a record has no context_id/gold_rank")
            c = int(ctx.group(1))
            den[c] += 1
            if gold.group(1) != b"null":
                num[c] += 1.0 / int(gold.group(1))
    return num, den


def per_context_acc(path: Path) -> tuple[dict[int, float], dict[int, int]]:
    num: dict[int, float] = defaultdict(float)
    den: dict[int, int] = defaultdict(int)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            c = int(record["context_id"])
            den[c] += 1
            gold = {normalize_tag(t) for t in (record.get("gold_tags") or [])}
            selected = record.get("selected_tag")
            if selected and normalize_tag(selected) in gold:
                num[c] += 1.0
    return num, den


def published(run_dir: Path, key: str) -> float | None:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    block = m.get("bm25_retrieval" if key == "mrr" else "qwen_reranked")
    return None if not isinstance(block, dict) else float(block["mrr" if key == "mrr" else "accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    loaded: dict[tuple[str, str], dict[str, tuple[dict, dict]]] = {}
    checks: list[str] = []
    for (retr, method), (rank_dir, pred_dir) in ROWS.items():
        entry: dict[str, tuple[dict, dict]] = {}
        ranking = rank_dir / "bm25_candidates.jsonl"
        preds = pred_dir / "qwen_rerank_predictions.jsonl"
        if ranking.exists():
            num, den = per_context_rr(ranking)
            if sum(den.values()) < EXPECTED_FACTS:
                checks.append(f"  MRR  {retr:<6} {method:<9} SKIPPED: {sum(den.values())}/{EXPECTED_FACTS} facts (still being written?)")
                num = None  # type: ignore[assignment]
            else:
                entry["mrr"] = (num, den)
            ref = published(rank_dir, "mrr") if "mrr" in entry else None
            got = (sum(num.values()) / sum(den.values())) if num is not None else float("nan")
            if ref is not None:
                ok = abs(got - ref) <= args.tolerance
                checks.append(f"  MRR  {retr:<6} {method:<9} rebuilt={got:.6f} published={ref:.6f} {'OK' if ok else 'MISMATCH'}")
                if not ok:
                    raise SystemExit(f"{ranking}: MRR {got:.6f} != published {ref:.6f}")
            elif num is not None:
                checks.append(f"  MRR  {retr:<6} {method:<9} rebuilt={got:.6f} (no metrics.json to check against)")
        if preds.exists():
            num, den = per_context_acc(preds)
            n = sum(den.values())
            if n < EXPECTED_FACTS:
                # A rerank writes its predictions incrementally, so a running job leaves a
                # perfectly parseable prefix. Pairing against it would silently compare
                # different fact sets, so drop it and say so.
                checks.append(f"  Acc  {retr:<6} {method:<9} SKIPPED: {n}/{EXPECTED_FACTS} facts (still running?)")
            else:
                entry["acc"] = (num, den)
        if entry:
            loaded[(retr, method)] = entry

    print("self-check")
    print("\n".join(checks) or "  (nothing to check)")

    contexts = None
    for entry in loaded.values():
        for num, den in entry.values():
            if contexts is None:
                contexts = sorted(den)
                den_vec = np.array([den[c] for c in contexts], dtype=float)
            elif sorted(den) != contexts:
                raise SystemExit("fact sets differ across runs; the pairing would be invalid")
    if contexts is None:
        raise SystemExit("no inputs found")

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(contexts), size=(SAMPLES, len(contexts)))
    den_draws = den_vec[idx].sum(axis=1)

    def draws(entry_key, num):
        vec = np.array([num.get(c, 0.0) for c in contexts], dtype=float)
        return vec[idx].sum(axis=1) / den_draws, float(vec.sum() / den_vec.sum())

    rows = []
    print(f"\n{'row':<18}{'metric':<6}{'value':>8}{'delta vs BM25':>15}{'95% CI':>22}{'excl.0':>8}")
    for (retr, method), entry in loaded.items():
        for metric in ("mrr", "acc"):
            if metric not in entry:
                continue
            d, value = draws(metric, entry[metric][0])
            base = loaded.get((REFERENCE, method), {}).get(metric)
            if retr == REFERENCE or base is None:
                print(f"{retr+'/'+method:<18}{metric:<6}{value:>8.4f}{'reference' if retr==REFERENCE else 'no BM25 row':>15}")
                rows.append({"retriever": retr, "method": method, "metric": metric, "value": value})
                continue
            bd, bv = draws(metric, base[0])
            diff = d - bd
            lo, hi = (float(v) for v in np.percentile(diff, [2.5, 97.5]))
            excl = not (lo <= 0.0 <= hi)
            print(f"{retr+'/'+method:<18}{metric:<6}{value:>8.4f}{value-bv:>+15.4f}   [{lo:>+7.4f},{hi:>+7.4f}]{str(excl):>8}")
            rows.append({
                "retriever": retr, "method": method, "metric": metric, "value": value,
                "delta_vs_bm25": value - bv, "ci_low": lo, "ci_high": hi, "excludes_zero": excl,
            })

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "contexts": len(contexts), "facts": int(den_vec.sum()), "seed": SEED,
            "samples": SAMPLES, "reference_retriever": REFERENCE, "rows": rows,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()

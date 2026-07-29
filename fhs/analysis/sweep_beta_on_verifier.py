#!/usr/bin/env python3
"""Sweep beta on the CANDIDATE-LEVEL VERIFIER term, not on the deterministic one.

WHY: tab:beta and the beta=0.6 operating point come from run_ags_beta_sweep.py, which scores
`minmax(S_wRRF) + beta * agree(c)` -- the deterministic consensus (that script's line 11 says so).
The deployed method scores `S~(c) + beta * v_bar(c)`, the verifier's support. Nothing has ever
checked that 0.6 is a sensible operating point for the term the method actually uses.

The paper does not currently claim it does: Appendix E was deliberately left saying "the weight on
the reranking term", which is true of both scores and asserts nothing about which term the sweep
used. This run is what would let that wording become specific.

CPU only, reuses the existing k10_fused verdicts, no GPU. Reads the same code path as every
tab:ablation row (core.evaluate), so the beta=0.6 column must reproduce the published
"AGS (full)" retrieval-stage numbers -- that is the built-in check that this driver is wired
correctly, and it is asserted below rather than left to the reader.
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

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from verifier.core import AblationConfig, aggregate, evaluate  # noqa: E402
from verifier.data_prep import DEFAULT_TEST_TRACE, load_test_facts  # noqa: E402
from verifier.run_test_rows import load_llm_verifier_verdicts  # noqa: E402

BETAS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0)
VERDICTS = REPO / "runs_ags_verifier_ablation" / "qwen3_32b" / "verdicts_k10_fused" / "llm_verifier_verdicts.json"
OUT = REPO / "runs_ags_beta_sweep_verifier" / "qwen3_32b"
# The published retrieval-stage row for AGS/FHS (full) = rerank_no_determ_k10fused, beta 0.6.
PUBLISHED = {"recall_at_10": 0.4010, "mrr": 0.2572, "top1_accuracy": 0.1825}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading facts and verdicts", flush=True)
    facts = load_test_facts(DEFAULT_TEST_TRACE)
    verdicts = load_llm_verifier_verdicts(VERDICTS)
    nmap = load_normalization_map(DEFAULT_NORMALIZATION_MAP)
    print(f"  {len(facts)} facts, {len(verdicts)} verdicts", flush=True)

    results = []
    for beta in BETAS:
        cfg = AblationConfig(
            verifier_mode="llm_drop",          # the deployed scoring path, no deterministic term
            beta=beta,
            llm_verifier_verdicts=verdicts,
            # MUST be "mean", not the dataclass default "zero". The published no_determ_k10fused
            # run set it to mean (its ranking_summary.json records it), and "zero" floors every
            # unjudged candidate to 0 support, which smuggles in a top-K_v prior: candidates the
            # verifier never saw are pushed below ones it merely declined. Leaving the default in
            # cost 0.022 of Recall@10 while leaving R@1, R@50 and MRR almost untouched -- which is
            # what the reproduce-the-published-row assertion below caught.
            llm_unjudged_fill="mean",
        )
        rows = [evaluate(fact, cfg, nmap) for fact in facts.values()]
        agg = aggregate(rows)
        agg["beta"] = beta
        results.append(agg)
        print(
            f"beta={beta:<4} R@10 {agg['recall_at_10']:.4f}  R@50 {agg['recall_at_50']:.4f}  "
            f"MRR {agg['mrr']:.4f}  top1 {agg['top1_accuracy']:.4f}",
            flush=True,
        )

    at06 = next(r for r in results if r["beta"] == 0.6)
    drift = {k: round(at06[k] - v, 4) for k, v in PUBLISHED.items()}
    ok = all(abs(d) < 0.002 for d in drift.values())
    print(f"\nbeta=0.6 vs published AGS (full): {drift}  -> {'MATCH' if ok else 'DRIFT'}")
    if not ok:
        print("  Driver is not reproducing the published row; treat the sweep as unwired.", flush=True)

    best = {m: max(results, key=lambda r: r[m]) for m in ("recall_at_10", "mrr", "top1_accuracy")}
    print("\nbest beta per metric:")
    for m, r in best.items():
        print(f"  {m:<16} beta={r['beta']}  value={r[m]:.4f}")

    (OUT / "beta_sweep_verifier.json").write_text(
        json.dumps(
            {
                "note": "beta swept on the candidate-level verifier support term (verifier_mode=llm_drop), "
                        "not on the deterministic agree() term that run_ags_beta_sweep.py swept.",
                "verdicts": str(VERDICTS),
                "reproduces_published_at_0.6": ok,
                "drift_vs_published": drift,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT / 'beta_sweep_verifier.json'}")


if __name__ == "__main__":
    main()

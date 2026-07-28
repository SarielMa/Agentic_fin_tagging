#!/usr/bin/env python3
"""Sweep the D- dimension weight for the hybrid and deterministic arms.

DIAGNOSTIC ONLY on the test split. The weight that ships must be selected on the dev sample,
exactly as beta was; a value picked because it maximises test top-1 is not a result. This
sweep exists to answer whether the signal moves the ranking at all before a dev sweep is
worth running.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, ".")
from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map
from ags_table5_ablation.core import AblationConfig, aggregate, evaluate, reset_consensus_cache
from ags_table5_ablation.data_prep import DEFAULT_TEST_TRACE, load_test_facts
from ags_table5_ablation.run_test_rows import load_llm_verifier_verdicts

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 400
nm = load_normalization_map(DEFAULT_NORMALIZATION_MAP)
verdicts = load_llm_verifier_verdicts(Path("runs_ags_table5_ablation/qwen3_32b/llm_verifier_verdicts.json"))
facts = list(load_test_facts(DEFAULT_TEST_TRACE).values())[:LIMIT]
print(f"facts: {len(facts)}  (diagnostic subset, not a selection run)\n", flush=True)

print(f"{'w_contra':>9} | {'deterministic':>14} | {'hybrid':>8} | {'hybrid mrr':>10}")
for w in (1.0, 0.75, 0.5, 0.25, 0.0):
    row = {}
    for mode in ("deterministic", "hybrid"):
        reset_consensus_cache()
        cfg = AblationConfig(
            name=mode, beta=0.6, verifier_mode=mode, truncate_pool_to_top_k=True,
            llm_unjudged_fill="mean", contradicted_dimension_weight=w,
            llm_verifier_verdicts=None if mode == "deterministic" else verdicts,
        )
        row[mode] = aggregate([evaluate(f, cfg, nm) for f in facts])
    print(f"{w:>9} | {row['deterministic']['top1_accuracy']:>14.4f} | "
          f"{row['hybrid']['top1_accuracy']:>8.4f} | {row['hybrid']['mrr']:>10.4f}", flush=True)

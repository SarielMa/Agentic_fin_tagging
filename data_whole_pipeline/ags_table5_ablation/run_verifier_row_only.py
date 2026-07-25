#!/usr/bin/env python3
"""Row 3.9 (+ LLM verification layer) only.

The other ten offline rows are already in runs_ags_table5_ablation/qwen32b_fixed/ablation.csv
and are unaffected by the verifier verdicts landing, so they are not recomputed. AGS (full) IS
recomputed -- it is the paired-bootstrap baseline row 3.9 is measured against -- and that
doubles as a reproduction check: its aggregate must match qwen32b_fixed's AGS (full) cells
exactly, otherwise the two runs are not comparable and the new row must not be pasted in.

Same seed (20260724) and iteration count (2000) as run_test_rows.py.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path("/nfs/roberts/project/pi_sjf37/lm2445/FinAI_tagging_agentic/data_whole_pipeline")
sys.path.insert(0, str(REPO))

from tqdm import tqdm

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map
from ags_table5_ablation.core import AblationConfig, aggregate, evaluate, reset_consensus_cache
from ags_table5_ablation.data_prep import DEFAULT_TEST_TRACE, load_test_facts
from ags_table5_ablation.run_test_rows import load_llm_verifier_verdicts, rows_for_modality
from compute_ags_seq_arm_metrics import paired_bootstrap

VERDICTS = REPO / "runs_ags_table5_ablation" / "qwen3_32b" / "llm_verifier_verdicts.json"
REFERENCE_CSV = REPO / "runs_ags_table5_ablation" / "qwen32b_fixed" / "ablation.csv"
OUT_DIR = Path(__file__).resolve().parent
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260724
METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr", "top1_accuracy")


def main() -> None:
    normalization_map = load_normalization_map(DEFAULT_NORMALIZATION_MAP)
    verdicts = load_llm_verifier_verdicts(VERDICTS)
    assert verdicts, "no verdicts loaded"
    print(f"verdicts loaded: {len(verdicts)}", flush=True)

    facts = list(load_test_facts(DEFAULT_TEST_TRACE).values())
    print(f"test facts: {len(facts)}", flush=True)

    def run(name: str, config: AblationConfig) -> list[dict]:
        reset_consensus_cache()
        return [evaluate(fact, config, normalization_map) for fact in tqdm(facts, desc=name, unit="fact")]

    full_rows = run("AGS (full)", AblationConfig(name="AGS (full)", beta=0.6))
    verifier_rows = run(
        "llm_verifier",
        AblationConfig(name="+ LLM verification layer", beta=0.6, llm_verifier_verdicts=verdicts),
    )

    # ---- reproduction check against the committed run -------------------------------------
    reference: dict[tuple[str, str], float] = {}
    with REFERENCE_CSV.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] == "AGS (full)":
                reference[(row["modality"], row["metric"])] = float(row["value"])

    mismatches = []
    for modality in ("pooled", "table", "text"):
        agg = aggregate(rows_for_modality(full_rows, modality))
        for metric in METRICS:
            mine = round(float(agg[metric]), 6)
            theirs = reference.get((modality, metric))
            if theirs is None or abs(mine - theirs) > 1e-9:
                mismatches.append((modality, metric, mine, theirs))
    print("\n=== AGS (full) reproduction vs qwen32b_fixed ===", flush=True)
    print("MATCH" if not mismatches else f"MISMATCH: {mismatches}", flush=True)

    # ---- row 3.9 --------------------------------------------------------------------------
    out_rows = []
    for modality in ("pooled", "table", "text"):
        subset = rows_for_modality(verifier_rows, modality)
        full_subset = rows_for_modality(full_rows, modality)
        if not subset:
            continue
        agg = aggregate(subset)
        for metric in METRICS:
            left = {
                int(r["fact_id"]): {"context_id": r["context_id"], metric: float(r[metric])} for r in subset
            }
            right = {
                int(r["fact_id"]): {"context_id": r["context_id"], metric: float(r[metric])} for r in full_subset
            }
            ci = paired_bootstrap(left, right, metric, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED)
            out_rows.append(
                {
                    "variant": "+ LLM verification layer",
                    "modality": modality,
                    "beta_used": 0.6,
                    "n_rankings_fused": subset[0].get("n_rankings_fused"),
                    "metric": metric,
                    "value": agg[metric],
                    "delta_vs_full": ci.get("mean_difference"),
                    "ci_low": ci.get("ci_low"),
                    "ci_high": ci.get("ci_high"),
                    "ci_excludes_zero": ci.get("ci_excludes_zero"),
                    "n_facts": ci.get("facts", len(subset)),
                    "n_contexts": ci.get("contexts"),
                }
            )

    with (OUT_DIR / "llm_verifier_row.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)

    (OUT_DIR / "llm_verifier_row_check.json").write_text(
        json.dumps(
            {
                "ags_full_reproduces_qwen32b_fixed": not mismatches,
                "mismatches": mismatches,
                "n_facts": len(facts),
                "n_verdict_keys": len(verdicts),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "bootstrap_seed": BOOTSTRAP_SEED,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n=== + LLM verification layer (pooled) ===", flush=True)
    for row in out_rows:
        if row["modality"] == "pooled":
            print(
                f"  {row['metric']:<16} {row['value']:.6f}  delta={row['delta_vs_full']:+.6f}  "
                f"ci=[{row['ci_low']:+.6f},{row['ci_high']:+.6f}]  excl0={row['ci_excludes_zero']}",
                flush=True,
            )


if __name__ == "__main__":
    main()

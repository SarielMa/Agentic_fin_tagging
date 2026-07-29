#!/usr/bin/env python3
"""Dev beta sweep for Table 5 (ags_table5_ablation_spec.md section 2).

Never sweep on test. Any variant whose ranking count differs from 4 (relative to AGS full's
J=2 dual), or whose scaling differs from range-normalized, gets its own dev sweep over
{0.2, 0.4, 0.6, 0.8, 1.0, 1.5}; everything else stays at AGS full's beta=0.6 by the
protocol's own literal criterion (fusion="mean" changes neither ranking count nor scaling,
so it is deliberately NOT resweeped here).

Variants swept, and why:
  ensemble        J=1, 2 rankings (dual on one hypothesis). Swept once over the pooled
                  idx=0 + idx=1 dev replicas (not twice) -- the row's own definition reports
                  the mean over both hypothesis choices, so one shared beta is the config
                  that row is actually evaluated at.
  label_form      J=2, def-only, 2 rankings.
  definition_form J=2, lab-only, 2 rankings (facts with no lab query anywhere use whatever
                  section 3.5 policy is configured; the sweep itself uses zero-recall,
                  matching the recommended test-time policy).
  raw             scaling="raw", still 4 rankings but unnormalized -- section 3.7 needs its
                  own optimum as the fair sub-row (the beta=0.6 sub-row is intentionally left
                  untuned and is not part of this sweep).

Oracle (3.11) reuses the ensemble sweep's beta rather than running its own: "beta for this
row is the J=1 optimum, since each hypothesis contributes 2 rankings" (spec's own words).

Selection rule mirrors how frozen AGS's own beta=0.6 was chosen (Task A): argmax table-subset
recall@10, MRR tie-break, lower beta wins remaining ties.
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

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from verifier.core import AblationConfig, aggregate, evaluate, reset_consensus_cache  # noqa: E402
from verifier.data_prep import (  # noqa: E402
    DEFAULT_DEV_HYPOTHESES,
    DEFAULT_DEV_SAMPLE_FACTS,
    dev_replay_sanity_check,
    load_dev_facts,
)


BETAS = (0.2, 0.4, 0.6, 0.8, 1.0, 1.5)
FROZEN_BETA = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-facts", type=Path, default=DEFAULT_DEV_SAMPLE_FACTS)
    parser.add_argument("--hypotheses", type=Path, default=DEFAULT_DEV_HYPOTHESES)
    parser.add_argument("--output-dir", type=Path, default=_PARENT / "runs_ags_table5_ablation" / "qwen3_32b")
    parser.add_argument("--betas", default=",".join(str(b) for b in BETAS))
    parser.add_argument("--sanity-sample", type=int, default=300)
    return parser.parse_args()


def sweep_one(
    facts_by_variant_fact: dict[Any, Any],
    base_config_kwargs: dict[str, Any],
    betas: list[float],
    normalization_map: dict[str, Any],
) -> list[dict[str, Any]]:
    """facts_by_variant_fact: iterable of FactRecord (already the pooled set for this sweep,
    e.g. idx=0 and idx=1 dev replicas concatenated for `ensemble`)."""
    rows: list[dict[str, Any]] = []
    for beta in betas:
        config = AblationConfig(beta=beta, **base_config_kwargs)
        results = [evaluate(fact, config, normalization_map) for fact in facts_by_variant_fact]
        for modality in ("table", "text"):
            subset = [row for row in results if row["modality"] == modality]
            if subset:
                agg = aggregate(subset)
                rows.append({"beta": beta, "modality": modality, **agg})
        rows.append({"beta": beta, "modality": "pooled", **aggregate(results)})
    return rows


def select_beta(sweep_rows: list[dict[str, Any]], on: str = "table") -> dict[str, Any]:
    candidates = [row for row in sweep_rows if row["modality"] == on]
    if not candidates:
        candidates = [row for row in sweep_rows if row["modality"] == "pooled"]
        on = "pooled"
    best = max(candidates, key=lambda row: (row["recall_at_10"], row["mrr"], -row["beta"]))
    return {
        "selected_beta": best["beta"],
        "selected_on": f"{on}/recall_at_10, mrr tie-break, lower beta wins ties",
        "recall_at_10": best["recall_at_10"],
        "mrr": best["mrr"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    betas = [float(value) for value in args.betas.split(",") if value.strip()]
    normalization_map = load_normalization_map(DEFAULT_NORMALIZATION_MAP)

    print("Replaying dev sample (full-fidelity, see data_prep.py docstring) ...", flush=True)
    dev_facts = load_dev_facts(sample_facts_path=args.sample_facts, hypotheses_path=args.hypotheses)
    print(f"Dev facts loaded: {len(dev_facts)}", flush=True)
    sanity = dev_replay_sanity_check(dev_facts, sample=args.sanity_sample)
    print(f"Replay sanity check vs Task A's logged gold_rank: {sanity}", flush=True)
    if sanity["mismatch_rate"] and sanity["mismatch_rate"] > 0.01:
        raise AssertionError(
            f"Dev replay disagrees with Task A's own logged gold_rank at rate "
            f"{sanity['mismatch_rate']}; retriever settings have drifted from the frozen "
            f"config and the sweep below cannot be trusted. Mismatches: {sanity['mismatches'][:5]}"
        )

    facts_list = list(dev_facts.values())
    all_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}

    variants = {
        "ensemble": dict(n_hypotheses=1),  # kept_hypothesis_idx set per-idx below, pooled
        "label_form": dict(renderings=("def",)),
        "definition_form": dict(renderings=("lab",), lab_only_fallback=None),
        "raw": dict(scaling="raw"),
    }

    for variant, kwargs in variants.items():
        reset_consensus_cache()
        if variant == "ensemble":
            pooled = []
            for idx in (0, 1):
                cfg_kwargs = dict(kwargs)
                cfg_kwargs["kept_hypothesis_idx"] = idx
                for fact in facts_list:
                    if idx in fact.hypotheses:
                        pooled.append((fact, cfg_kwargs))
            # sweep needs a flat fact list + a SHARED kwargs dict (kept_hypothesis_idx is
            # per-fact here since both idx replicas are pooled); evaluate per (fact,kwargs).
            rows = []
            for beta in betas:
                results = []
                for fact, cfg_kwargs in pooled:
                    config = AblationConfig(beta=beta, **cfg_kwargs)
                    results.append(evaluate(fact, config, normalization_map))
                for modality in ("table", "text"):
                    subset = [row for row in results if row["modality"] == modality]
                    if subset:
                        rows.append({"beta": beta, "modality": modality, **aggregate(subset)})
                rows.append({"beta": beta, "modality": "pooled", **aggregate(results)})
        else:
            rows = sweep_one(facts_list, kwargs, betas, normalization_map)
        for row in rows:
            row["variant"] = variant
        all_rows.extend(rows)
        selected[variant] = select_beta(rows, on="table")
        print(f"{variant}: selected beta={selected[variant]['selected_beta']} ({selected[variant]['selected_on']})", flush=True)

    # Oracle reuses the ensemble sweep's beta -- not its own sweep (spec 3.11).
    selected["oracle_best_single"] = {
        "selected_beta": selected["ensemble"]["selected_beta"],
        "selected_on": "reused from ensemble (J=1 optimum); each hypothesis contributes 2 rankings",
    }
    # AGS full, mean fusion, llm_verifier, index ablation: untouched, protocol's own criterion
    # (ranking count == 4, scaling == range) says no resweep.
    for variant in ("ags_full", "mean_fusion", "llm_verifier", "index_ablation"):
        selected[variant] = {"selected_beta": FROZEN_BETA, "selected_on": "frozen (ranking count and scaling unchanged)"}
    selected["consensus_off"] = {"selected_beta": 0.0, "selected_on": "fixed by definition (section 3.8)"}
    selected["raw_at_0.6"] = {"selected_beta": FROZEN_BETA, "selected_on": "deliberately untuned (section 3.7 sub-row a)"}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ablation_beta_sweeps.csv", all_rows)
    (args.output_dir / "selected_betas.json").write_text(
        json.dumps(
            {
                "betas_swept": betas,
                "dev_replay_sanity_check": sanity,
                "selection_rule": "argmax table/recall_at_10, mrr tie-break, lower beta wins ties",
                "protocol_note": (
                    "mean fusion changes neither ranking count (still 4 rankings fused) nor "
                    "scaling (still range-normalized), so per the protocol's own literal "
                    "criterion it is evaluated at the frozen beta=0.6, not resweeped."
                ),
                "selected": selected,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selected, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {args.output_dir / 'ablation_beta_sweeps.csv'} and selected_betas.json", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Row 3.10: label coverage off (w_cov=0), re-retrieval only, CPU (ags_table5_ablation_spec.md).

Re-issues every logged query verbatim with the coverage term disabled. No generation: the
queries already exist in each method's test-split trace. This is an index ablation, not a
method ablation -- the coverage term is applied to every compared method (spec section 6.3),
so it must be run for AGS *and* at least direct retrieval and one-pass grounding, reported as
its own small panel (index_ablation.csv) rather than as a row inside ablation.csv.

Both w_cov=0 and w_cov=1 numbers here are computed fresh, by the same re-retrieval code
path, rather than mixing a freshly computed w_cov=0 number with an old artifact's
w_cov=1 number that might have used a different label_coverage_pool_multiplier -- the delta
this panel reports must be measured on a matched basis or it isn't interpretable.

AGS's own ranking is rebuilt through verifier.core.evaluate at the frozen
beta=0.6 (label coverage is a retrieval-index property, not a consolidation setting; per the
beta-resweep protocol in section 2, w_cov changes neither ranking count nor scaling, so no
resweep is triggered here either -- see run_beta_sweep.py's protocol_note).
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
from verifier.core import AblationConfig, FactRecord, aggregate, evaluate  # noqa: E402
from verifier.data_prep import _compact, stream_jsonl  # noqa: E402
from run_fintagging_grounding_baseline import (  # noqa: E402
    DEFAULT_TAXONOMY_JSONL,
    TaxonomyRetriever,
    load_taxonomy,
    normalize_tag,
    retrieve_candidates,
)


DEFAULT_AGS_TRACE = _PARENT / "runs_fintagging_grounding_baseline" / "qwen3_32b_frozen_ags" / "bm25_candidates.jsonl"
DEFAULT_DIRECT_TRACE = _PARENT / "runs_fintagging_grounding_baseline" / "qwen3_32b_direct_retrieval" / "bm25_candidates.jsonl"
DEFAULT_ONEPASS_TRACE = _PARENT / "runs_fintagging_grounding_baseline" / "qwen3_32b_one_pass_grounding" / "bm25_candidates.jsonl"
TOP_K = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--ags-trace", type=Path, default=DEFAULT_AGS_TRACE)
    parser.add_argument("--direct-trace", type=Path, default=DEFAULT_DIRECT_TRACE)
    parser.add_argument("--onepass-trace", type=Path, default=DEFAULT_ONEPASS_TRACE)
    parser.add_argument("--output-dir", type=Path, default=_PARENT / "runs_ags_table5_ablation" / "qwen3_32b")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit.")
    parser.add_argument("--log-every", type=int, default=250)
    return parser.parse_args()


def single_query_metrics(
    trace_path: Path,
    retriever: TaxonomyRetriever,
    limit: int | None,
    log_every: int,
    label: str,
) -> list[dict[str, Any]]:
    """direct_retrieval / one_pass_grounding: one query per fact, no fusion."""
    rows: list[dict[str, Any]] = []
    for offset, record in enumerate(stream_jsonl(trace_path), start=1):
        if limit is not None and offset > limit:
            break
        candidates = retrieve_candidates(retriever, record["query"], record["type"], TOP_K)
        tags = [candidate["tag"] for candidate in candidates]
        rank = None
        gold = {normalize_tag(tag) for tag in record.get("gold_tags", [])}
        for index, tag in enumerate(tags, start=1):
            if normalize_tag(tag) in gold:
                rank = index
                break
        row = {
            "fact_id": record["example_idx"],
            "context_id": record.get("context_id"),
            "modality": record.get("input_type"),
            "rank": rank,
            "mrr": 0.0 if rank is None else 1.0 / rank,
            "top1_accuracy": bool(rank == 1),
            "recall_at_10": bool(rank is not None and rank <= 10),
            "recall_at_50": bool(rank is not None and rank <= 50),
            "recall_at_200": bool(rank is not None and rank <= 200),
        }
        rows.append(row)
        if log_every and offset % log_every == 0:
            print(f"{label}: {offset} facts re-retrieved", flush=True)
    return rows


def ags_metrics(
    trace_path: Path,
    retriever: TaxonomyRetriever,
    normalization_map: dict[str, Any],
    limit: int | None,
    log_every: int,
) -> list[dict[str, Any]]:
    """AGS: rebuild rankings at the given w_cov, then fuse+rerank via the SAME core.evaluate
    used everywhere else in this ablation, at the frozen beta=0.6."""
    config = AblationConfig(name="AGS (index ablation)")
    rows: list[dict[str, Any]] = []
    for offset, record in enumerate(stream_jsonl(trace_path), start=1):
        if limit is not None and offset > limit:
            break
        entity_type = record.get("type", "")
        hypotheses = {
            int(hypothesis["hypothesis_idx"]): {"dimensions": hypothesis.get("dimensions", {})}
            for hypothesis in record.get("frozen_ags_hypotheses", [])
        }
        rankings: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for round_record in record.get("rounds", []):
            if round_record.get("label_render_skipped"):
                continue
            candidates = retrieve_candidates(retriever, round_record["query"], entity_type, TOP_K)
            rankings[(int(round_record["hypothesis_idx"]), round_record["rendering"])] = [
                _compact(candidate) for candidate in candidates
            ]
        fact = FactRecord(
            fact_id=int(record["example_idx"]),
            context_id=record.get("context_id"),
            modality=record.get("input_type", ""),
            datatype=entity_type,
            gold_tags=[normalize_tag(tag) for tag in record.get("gold_tags", [])],
            hypotheses=hypotheses,
            rankings=rankings,
        )
        rows.append(evaluate(fact, config, normalization_map))
        if log_every and offset % log_every == 0:
            print(f"AGS (index ablation): {offset} facts re-retrieved", flush=True)
    return rows


def summarize(rows: list[dict[str, Any]], method: str, w_cov: float) -> list[dict[str, Any]]:
    out = []
    for modality in ("all", "table", "text"):
        subset = rows if modality == "all" else [row for row in rows if row["modality"] == modality]
        if not subset and modality != "all":
            continue
        agg = aggregate(subset)
        out.append({"method": method, "w_cov": w_cov, "modality": modality, **agg})
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    normalization_map = load_normalization_map(DEFAULT_NORMALIZATION_MAP)
    taxonomy = load_taxonomy(args.taxonomy_jsonl)

    summary_rows: list[dict[str, Any]] = []
    for w_cov in (1.0, 0.0):
        retriever = TaxonomyRetriever(
            taxonomy, type_filter=True, label_coverage_weight=w_cov, label_coverage_pool_multiplier=0
        )
        print(f"=== w_cov={w_cov} ===", flush=True)

        direct_rows = single_query_metrics(args.direct_trace, retriever, args.limit, args.log_every, "direct_retrieval")
        summary_rows.extend(summarize(direct_rows, "direct_retrieval", w_cov))

        onepass_rows = single_query_metrics(args.onepass_trace, retriever, args.limit, args.log_every, "one_pass_grounding")
        summary_rows.extend(summarize(onepass_rows, "one_pass_grounding", w_cov))

        ags_rows = ags_metrics(args.ags_trace, retriever, normalization_map, args.limit, args.log_every)
        summary_rows.extend(summarize(ags_rows, "AGS", w_cov))

    # Fold in the delta (w_cov=1 - w_cov=0) per (method, modality, metric).
    by_key: dict[tuple[str, str, float], dict[str, Any]] = {
        (row["method"], row["modality"], row["w_cov"]): row for row in summary_rows
    }
    delta_rows: list[dict[str, Any]] = []
    for (method, modality, w_cov), row in by_key.items():
        if w_cov != 1.0:
            continue
        off = by_key.get((method, modality, 0.0))
        if off is None:
            continue
        for metric in ("recall_at_10", "recall_at_50", "recall_at_200", "mrr", "top1_accuracy"):
            delta_rows.append(
                {
                    "method": method,
                    "modality": modality,
                    "metric": metric,
                    "w_cov_1": row[metric],
                    "w_cov_0": off[metric],
                    "delta": round(row[metric] - off[metric], 6),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "index_ablation.csv", delta_rows)
    (args.output_dir / "index_ablation_summary.json").write_text(
        json.dumps({"raw_summary": summary_rows, "deltas": delta_rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(delta_rows, indent=2), flush=True)
    print(f"Wrote {args.output_dir / 'index_ablation.csv'}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Materialise the w_cov=0 ranking as a candidates file, for tab:ablation's label-coverage row.

Every other arm of that table can be dumped by dump_reranked_ranking.py, which reorders the
candidate objects the frozen trace already logged. This one cannot: w_cov weights the retrieval
index, so turning it off changes WHICH candidates come back, not merely their order. The pool
has to be rebuilt by re-retrieving each rendering query against a retriever configured at the
ablated weight -- exactly what run_index_ablation.py does for the retrieval-stage row. That
script keeps only aggregate metrics and discards the ranking, so this one repeats the
re-retrieval and persists it in the schema run_fintagging_grounding_baseline.py reads under
--reuse-candidates.

Candidate objects come from the retriever rather than from the trace, since a w_cov=0 pool
contains concepts the w_cov=1 trace never retrieved and therefore never logged text for.

CPU only, but it re-retrieves every rendering query, so it is far slower than the other dumps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from ags_table5_ablation.core import AblationConfig, FactRecord, evaluate, reset_consensus_cache  # noqa: E402
from ags_table5_ablation.data_prep import _compact, stream_jsonl  # noqa: E402
from ags_table5_ablation.run_index_ablation import DEFAULT_AGS_TRACE, TOP_K  # noqa: E402
from ags_table5_ablation.run_test_rows import load_llm_verifier_verdicts  # noqa: E402
from run_fintagging_grounding_baseline import (  # noqa: E402
    DEFAULT_TAXONOMY_JSONL,
    TaxonomyRetriever,
    load_taxonomy,
    normalize_tag,
    retrieve_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-trace", type=Path, default=DEFAULT_AGS_TRACE)
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--label-coverage-weight", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    # AblationConfig's own defaults are verifier_mode="auto" and llm_unjudged_fill="zero", neither
    # of which is the deployed configuration. Pinning them here keeps this dump on the same code
    # path as every other row of tab:ablation instead of silently scoring the arm differently.
    parser.add_argument("--verifier-mode", default="llm_drop")
    parser.add_argument("--llm-unjudged-fill", default="mean")
    parser.add_argument("--llm-verifier-verdicts", type=Path, default=None,
                        help="Verdicts for THIS w_cov's own window (stage_arm_windows.py + "
                             "run_llm_verifier.py --window-tags). Without it the rerank term has "
                             "no LLM support to consume and the row is not comparable to FHS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    normalization_map = load_normalization_map(args.normalization_map)

    taxonomy = load_taxonomy(args.taxonomy_jsonl)
    # Same construction as run_index_ablation.py:172-174, so this ranking and that script's
    # retrieval-stage row for the same w_cov come from an identically configured index.
    retriever = TaxonomyRetriever(
        taxonomy, type_filter=True, label_coverage_weight=args.label_coverage_weight,
        label_coverage_pool_multiplier=0,
    )
    print(f"retriever built at w_cov={args.label_coverage_weight}", flush=True)

    # Two stages, because the verifier window has to be cut from THIS w_cov's own fused ranking:
    #   stage 1 (no verdicts): emit the pre-verifier fused order and the rewritten rounds, so
    #           stage_arm_windows.py can compute this arm's window and the verifier can judge it;
    #   stage 2 (verdicts):    re-emit under the deployed rerank, which is what the row reports.
    # core.py refuses a non-deterministic mode with no verdicts attached, so stage 1 is explicit
    # rather than a silent fallback to a deterministic term.
    if args.llm_verifier_verdicts is None:
        verdicts = None
        verifier_mode, beta = "deterministic", 0.0
        print("STAGE 1: no verdicts attached -> emitting the pre-verifier fused order. "
              "This file is the input to stage_arm_windows.py, NOT a reportable row.", flush=True)
    else:
        verdicts = load_llm_verifier_verdicts(args.llm_verifier_verdicts)
        verifier_mode, beta = args.verifier_mode, args.beta
        print(f"STAGE 2: {len(verdicts)} verdicts, verifier_mode={verifier_mode} beta={beta} "
              f"llm_unjudged_fill={args.llm_unjudged_fill}", flush=True)

    config = AblationConfig(
        name=f"AGS (w_cov={args.label_coverage_weight})",
        beta=beta,
        verifier_mode=verifier_mode,
        llm_unjudged_fill=args.llm_unjudged_fill,
        llm_verifier_verdicts=verdicts,
        truncate_pool_to_top_k=True,
    )
    reset_consensus_cache()

    written = 0
    unresolved_total = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for offset, record in enumerate(stream_jsonl(args.test_trace), start=1):
            if args.limit is not None and offset > args.limit:
                break
            entity_type = record.get("type", "")
            hypotheses = {
                int(h["hypothesis_idx"]): {"dimensions": h.get("dimensions", {})}
                for h in record.get("frozen_ags_hypotheses", [])
            }
            rankings: dict[tuple[int, str], list[dict[str, Any]]] = {}
            pool: dict[str, dict[str, Any]] = {}
            # Rewritten in place, so the emitted record's `rounds` carry THIS w_cov's rankings.
            # Leaving them as the source trace's was a silent defect: data_prep.load_test_facts,
            # core.fuse and stage_arm_windows.py all read rankings out of `rounds`, so an emitted
            # file with stale rounds reproduces the w_cov=1 fusion and the w_cov=1 verifier window
            # exactly (measured: 25,090/25,090 window tags identical), while only the metrics
            # computed from final_candidates moved. Every consumer must see the same ablation.
            rebuilt_rounds: list[dict[str, Any]] = []
            for round_record in record.get("rounds", []):
                emitted_round = dict(round_record)
                if round_record.get("label_render_skipped"):
                    rebuilt_rounds.append(emitted_round)
                    continue
                candidates = retrieve_candidates(retriever, round_record["query"], entity_type, args.top_k)
                compacted = [_compact(candidate) for candidate in candidates]
                rankings[(int(round_record["hypothesis_idx"]), round_record["rendering"])] = compacted
                emitted_round["candidates"] = [dict(c) for c in candidates]
                rebuilt_rounds.append(emitted_round)
                for candidate in candidates:
                    pool.setdefault(normalize_tag(candidate.get("tag", "")), candidate)

            fact = FactRecord(
                fact_id=int(record["example_idx"]),
                context_id=record.get("context_id"),
                modality=record.get("input_type", ""),
                datatype=entity_type,
                gold_tags=[normalize_tag(tag) for tag in record.get("gold_tags", [])],
                hypotheses=hypotheses,
                rankings=rankings,
            )
            row = evaluate(fact, config, normalization_map)

            rebuilt: list[dict[str, Any]] = []
            for tag in [normalize_tag(t) for t in (row.get("candidate_tags") or [])]:
                candidate = pool.get(tag)
                if candidate is None:
                    unresolved_total += 1
                    continue
                entry = dict(candidate)
                entry["rank"] = len(rebuilt) + 1
                rebuilt.append(entry)
                if len(rebuilt) >= args.top_k:
                    break

            out = dict(record)
            out["rounds"] = rebuilt_rounds
            out["candidates"] = rebuilt
            out["final_candidates"] = rebuilt
            out["index_ablation_w_cov"] = args.label_coverage_weight
            out["index_ablation_verifier_mode"] = verifier_mode
            out["index_ablation_beta"] = beta
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
            if args.log_every and offset % args.log_every == 0:
                print(f"re-retrieved {offset} facts ({written} written)", flush=True)

    summary = {
        "output": str(args.output),
        "facts_written": written,
        "unresolved_tags_dropped": unresolved_total,
        "label_coverage_weight": args.label_coverage_weight,
        "beta": beta,
        "verifier_mode": verifier_mode,
        "llm_unjudged_fill": args.llm_unjudged_fill,
        "llm_verifier_verdicts": str(args.llm_verifier_verdicts) if args.llm_verifier_verdicts else None,
        "rounds_rewritten": True,
        "top_k": args.top_k,
    }
    summary_path = args.summary or args.output.with_name("index_ablation_ranking_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

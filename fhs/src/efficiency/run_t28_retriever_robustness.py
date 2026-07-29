#!/usr/bin/env python3
"""T28: retriever robustness (comparing_methods/ags_t7_t28_spec.md).

Does AGS's advantage survive when the retriever is not sparse? AGS's largest single effect,
the label-coverage term (Eq. 10), is a fix for a BM25 length pathology that a dense retriever
may not have at all. This script re-runs one-pass grounding and AGS over dense and hybrid
indexes to locate how much of AGS's gain is architecture and how much is the BM25-specific
lexical fix.

Retrieval and consolidation only -- NO generation. Every query string is reused verbatim from
the methods' own logged traces (AGS's per-(hypothesis, rendering) `rounds[].query`, one-pass's
flat `query`), so no hypothesis is ever regenerated. Everything downstream of retrieval
(sum-RRF fusion, `agree()`, the consensus rerank, beta) is `verifier.core.evaluate`
at the frozen config, unchanged and un-retuned.

The grid (spec step 3), 8 rows:

    BM25    one-pass   w_cov=1.0      the main-results setting
    BM25    AGS        w_cov=1.0
    Dense   one-pass   w_cov=0        coverage is a lexical term; off by default off-BM25
    Dense   AGS        w_cov=0
    Dense   AGS        w_cov=1.0      the explicitly marked "+cov" variant
    Hybrid  one-pass   w_cov=0
    Hybrid  AGS        w_cov=0
    Hybrid  AGS        w_cov=1.0      "+cov"

BM25 is recomputed fresh rather than lifted from the existing artifact, for the same reason
run_index_ablation.py does: a delta is only interpretable if both sides came through the same
code path. Pass --skip-bm25 to reuse a previous run of this script instead.

Two traps this script has to avoid, both real:

1. `core.reset_consensus_cache()` MUST be called between retriever variants. The cache
   (core.py:214-252) is keyed by (fact_id, hypothesis indices, renderings, verifier active)
   and NOT by which retriever produced the candidate pool, so without a reset the dense pass
   would silently score BM25's cached consensus values.
2. beta is not re-tuned. The rerank operates on range-normalized fused scores, so beta
   transfers across retrievers; this is an assumption, it is recorded in metrics.json, and the
   spec asks for it to be stated in the caption.

    python efficiency/run_t28_retriever_robustness.py --limit 25   # smoke test
    python efficiency/run_t28_retriever_robustness.py              # full grid
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

import numpy as np
from tqdm import tqdm

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from efficiency.dense_index import (  # noqa: E402
    DEFAULT_DENSE_MODEL,
    DEFAULT_EMBEDDING_CACHE_NAME,
    DEFAULT_RRF_KAPPA,
    build_retriever,
)
from verifier.core import (  # noqa: E402
    AblationConfig,
    FactRecord,
    aggregate,
    evaluate,
    metric_row,
    reset_consensus_cache,
)
from verifier.data_prep import _compact, stream_jsonl  # noqa: E402
from compute_ags_seq_arm_metrics import paired_bootstrap  # noqa: E402
from run_fintagging_grounding_baseline import (  # noqa: E402
    DEFAULT_TAXONOMY_JSONL,
    load_taxonomy,
    normalize_tag,
    retrieve_candidates,
)

DEFAULT_AGS_TRACE = _PARENT / "runs_fintagging_grounding_baseline" / "qwen3_32b_frozen_ags" / "bm25_candidates.jsonl"
DEFAULT_ONEPASS_TRACE = (
    _PARENT / "runs_fintagging_grounding_baseline" / "qwen3_32b_one_pass_grounding" / "bm25_candidates.jsonl"
)
TOP_K = 200
# top1_accuracy is rank-1 on each row's own retrieval ranking. T28 runs no listwise reranker
# (the only "rerank" in this module is AGS's internal consensus rerank), so this column is the
# retrieval-stage quantity -- the same basis as Table 5's Acc., NOT the end-to-end accuracy the
# deployed pipeline reports after its Qwen listwise stage. Label it as such wherever it lands.
METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr", "top1_accuracy")
MODALITIES = ("table", "text", "pooled")

# (retriever, method, w_cov). Dense/hybrid default to coverage OFF because the term is a
# lexical feature; the "+cov" AGS rows are the spec's explicit "is the pathology
# sparse-specific?" probe.
GRID: tuple[tuple[str, str, float], ...] = (
    ("bm25", "one_pass", 1.0),
    ("bm25", "AGS", 1.0),
    ("dense", "one_pass", 0.0),
    ("dense", "AGS", 0.0),
    ("dense", "AGS", 1.0),
    ("hybrid", "one_pass", 0.0),
    ("hybrid", "AGS", 0.0),
    ("hybrid", "AGS", 1.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--ags-trace", type=Path, default=DEFAULT_AGS_TRACE)
    parser.add_argument("--onepass-trace", type=Path, default=DEFAULT_ONEPASS_TRACE)
    parser.add_argument("--output-dir", type=Path, default=_PARENT / "runs_ags_t7_t28" / "qwen3_32b")
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--query-instruction", default=None, help="Override the model's query prefix.")
    parser.add_argument("--embed-batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="cuda / cpu; defaults to cuda when available.")
    parser.add_argument("--rrf-kappa", type=float, default=DEFAULT_RRF_KAPPA)
    parser.add_argument("--beta", type=float, default=0.6, help="Frozen AGS beta; not re-tuned (spec).")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--skip-bm25", action="store_true", help="Only run the dense/hybrid rows.")
    parser.add_argument("--limit", type=int, default=None, help="Facts per method, for smoke tests.")
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help=(
            "Where concept/query embeddings are cached. Built if absent, loaded if present. "
            "With a populated cache the run needs no GPU and no sentence-transformers at all."
        ),
    )
    parser.add_argument(
        "--build-cache-only",
        action="store_true",
        help="Embed the taxonomy and the replayed queries, write --embedding-cache, then exit.",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------------------
# Loading: one streaming pass per trace, keeping only what re-retrieval needs.
# The AGS trace is ~3.6GB, so the logged candidate lists are dropped on the floor -- this
# script re-retrieves them. Only the query strings and the symbolic hypotheses are kept.
# --------------------------------------------------------------------------------------


def load_ags_queries(path: Path, limit: int | None) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for record in stream_jsonl(path):
        if limit is not None and len(facts) >= limit:
            break
        rounds = [
            {
                "hypothesis_idx": int(round_record["hypothesis_idx"]),
                "rendering": round_record["rendering"],
                "query": round_record["query"],
            }
            for round_record in record.get("rounds", [])
            # A null label query was skipped at generation time and is not re-issued here.
            if not round_record.get("label_render_skipped")
        ]
        facts.append(
            {
                "fact_id": int(record["example_idx"]),
                "context_id": record.get("context_id"),
                "modality": record.get("input_type", ""),
                "datatype": record.get("type", ""),
                "gold_tags": [normalize_tag(tag) for tag in record.get("gold_tags", [])],
                "hypotheses": {
                    int(hypothesis["hypothesis_idx"]): {
                        "dimensions": hypothesis.get("dimensions", {}),
                        "operators": hypothesis.get("operators", []),
                        "retrieval_query": hypothesis.get("retrieval_query", ""),
                    }
                    for hypothesis in record.get("frozen_ags_hypotheses", [])
                },
                "rounds": rounds,
            }
        )
    return facts


def load_onepass_queries(path: Path, limit: int | None) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for record in stream_jsonl(path):
        if limit is not None and len(facts) >= limit:
            break
        facts.append(
            {
                "fact_id": int(record["example_idx"]),
                "context_id": record.get("context_id"),
                "modality": record.get("input_type", ""),
                "datatype": record.get("type", ""),
                "gold_tags": [normalize_tag(tag) for tag in record.get("gold_tags", [])],
                "query": record.get("query", ""),
            }
        )
    return facts


# --------------------------------------------------------------------------------------
# Per-variant evaluation
# --------------------------------------------------------------------------------------


def run_onepass(retriever: Any, facts: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    """One query per fact, no fusion -- the ranking IS the retrieval result."""
    rows: list[dict[str, Any]] = []
    for fact in tqdm(facts, desc=label, unit="fact"):
        candidates = retrieve_candidates(retriever, fact["query"], fact["datatype"], TOP_K)
        row = metric_row([candidate["tag"] for candidate in candidates], fact["gold_tags"])
        row.update(
            {"fact_id": fact["fact_id"], "context_id": fact["context_id"], "modality": fact["modality"]}
        )
        rows.append(row)
    return rows


def run_ags(
    retriever: Any,
    facts: list[dict[str, Any]],
    normalization_map: dict[str, Any],
    beta: float,
    label: str,
) -> list[dict[str, Any]]:
    """Re-retrieve each logged (hypothesis, rendering) query, then consolidate through the
    same `core.evaluate` every other AGS table uses, at the frozen beta."""
    # Trap 1 (module docstring): the consensus cache does not know the retriever changed.
    reset_consensus_cache()
    config = AblationConfig(name=f"AGS ({label})", beta=beta)
    rows: list[dict[str, Any]] = []
    for fact in tqdm(facts, desc=label, unit="fact"):
        rankings: dict[tuple[int, str], list[dict[str, Any]]] = {}
        for round_record in fact["rounds"]:
            candidates = retrieve_candidates(retriever, round_record["query"], fact["datatype"], TOP_K)
            rankings[(round_record["hypothesis_idx"], round_record["rendering"])] = [
                _compact(candidate) for candidate in candidates
            ]
        record = FactRecord(
            fact_id=fact["fact_id"],
            context_id=fact["context_id"],
            modality=fact["modality"],
            datatype=fact["datatype"],
            gold_tags=fact["gold_tags"],
            hypotheses=fact["hypotheses"],
            rankings=rankings,
        )
        rows.append(evaluate(record, config, normalization_map))
    return rows


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def single_arm_bootstrap(
    rows: list[dict[str, Any]], field: str, iterations: int, seed: int
) -> dict[str, Any]:
    """Context-clustered CI on one arm's own mean.

    Same resampling scheme as compute_ags_seq_arm_metrics.paired_bootstrap (resample source
    contexts with replacement, weight by cluster size) so the single-arm CIs in
    retriever_robustness.csv and the paired deltas in the companion CSV are computed on a
    consistent basis. Facts inside a source context are not independent, which is why the
    resampling unit is the context and not the fact.
    """
    if not rows:
        return {"value": None, "ci_low": None, "ci_high": None, "facts": 0, "contexts": 0}
    by_context: dict[Any, list[float]] = {}
    for row in rows:
        by_context.setdefault(row["context_id"], []).append(float(row[field]))
    contexts = list(by_context)
    means = np.asarray([float(np.mean(by_context[context])) for context in contexts])
    sizes = np.asarray([len(by_context[context]) for context in contexts], dtype=float)
    observed = float(np.sum(means * sizes) / np.sum(sizes))

    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for index in range(iterations):
        picks = rng.integers(0, len(contexts), size=len(contexts))
        weights = sizes[picks]
        draws[index] = float(np.sum(means[picks] * weights) / np.sum(weights))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "value": round(observed, 6),
        "ci_low": round(float(low), 6),
        "ci_high": round(float(high), 6),
        "facts": len(rows),
        "contexts": len(contexts),
    }


def subset(rows: list[dict[str, Any]], modality: str) -> list[dict[str, Any]]:
    if modality == "pooled":
        return rows
    return [row for row in rows if row["modality"] == modality]


def as_bootstrap_input(rows: list[dict[str, Any]], field: str) -> dict[int, dict[str, Any]]:
    return {
        int(row["fact_id"]): {"context_id": row["context_id"], field: float(row[field])}
        for row in rows
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
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    normalization_map = load_normalization_map(DEFAULT_NORMALIZATION_MAP)

    print("Loading taxonomy ...", flush=True)
    taxonomy = load_taxonomy(args.taxonomy_jsonl)
    print(f"Concepts: {len(taxonomy)}", flush=True)

    print("Loading logged queries (no regeneration) ...", flush=True)
    ags_facts = load_ags_queries(args.ags_trace, args.limit)
    onepass_facts = load_onepass_queries(args.onepass_trace, args.limit)
    print(f"AGS facts: {len(ags_facts)}   one-pass facts: {len(onepass_facts)}", flush=True)

    grid = [entry for entry in GRID if not (args.skip_bm25 and entry[0] == "bm25")]
    needs_dense = any(entry[0] in ("dense", "hybrid") for entry in grid) or args.build_cache_only

    cache_path = args.embedding_cache
    if cache_path is None and args.build_cache_only:
        cache_path = args.output_dir / DEFAULT_EMBEDDING_CACHE_NAME

    # Embedding is the only GPU step here and it is ~30 seconds of a multi-hour run; everything
    # downstream (BM25 scoring, the Eq. 10 coverage rescore, core.evaluate, the bootstrap) is
    # CPU-bound. --build-cache-only splits those apart so the long half can run GPU-free.
    dense = None
    if needs_dense:
        all_queries = [fact["query"] for fact in onepass_facts]
        all_queries += [
            round_record["query"] for fact in ags_facts for round_record in fact["rounds"]
        ]
        dense = build_retriever(
            "dense",
            taxonomy,
            model_name=args.dense_model,
            label_coverage_weight=0.0,
            batch_size=args.embed_batch_size,
            device=args.device,
            query_instruction=args.query_instruction,
            cache_path=cache_path,
        )
        # Queries are all known up front (they are reused verbatim), so encode them in one
        # batched pass instead of one at a time inside the retrieval loop.
        dense.prime_query_cache(all_queries)
        if cache_path is not None:
            dense.save_cache()

    if args.build_cache_only:
        print(f"\nEmbedding cache ready at {cache_path}; exiting before the CPU replay.", flush=True)
        return

    # Both indexes are built once and reused across every grid row that needs them: tokenizing
    # 17k BM25 documents and embedding 17k passages each cost far more than the replay itself.
    needs_bm25 = any(entry[0] in ("bm25", "hybrid") for entry in grid)
    bm25 = None
    if needs_bm25:
        print("Building BM25 index ...", flush=True)
        bm25 = build_retriever("bm25", taxonomy, label_coverage_weight=0.0)

    per_variant_rows: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []

    for retriever_kind, method, w_cov in grid:
        coverage_label = "on" if w_cov > 0 else "off"
        label = f"{retriever_kind}/{method}/cov={coverage_label}"
        print(f"\n=== {label} ===", flush=True)
        retriever = build_retriever(
            retriever_kind,
            taxonomy,
            model_name=args.dense_model,
            label_coverage_weight=w_cov,
            rrf_kappa=args.rrf_kappa,
            dense=dense,
            bm25=bm25,
            batch_size=args.embed_batch_size,
            device=args.device,
            query_instruction=args.query_instruction,
            cache_path=cache_path,
        )
        if method == "one_pass":
            rows = run_onepass(retriever, onepass_facts, label)
        else:
            rows = run_ags(retriever, ags_facts, normalization_map, args.beta, label)
        per_variant_rows[(retriever_kind, method, w_cov)] = rows

        for modality in MODALITIES:
            modality_rows = subset(rows, modality)
            if not modality_rows:
                continue
            agg = aggregate(modality_rows)
            for metric in METRICS:
                stats = single_arm_bootstrap(
                    modality_rows, metric, args.bootstrap_samples, args.bootstrap_seed
                )
                summary_rows.append(
                    {
                        "retriever": retriever_kind,
                        "method": method,
                        "coverage_term": coverage_label,
                        "modality": modality,
                        "metric": metric,
                        "value": round(float(agg[metric]), 6),
                        "ci_low": stats["ci_low"],
                        "ci_high": stats["ci_high"],
                        "n_facts": stats["facts"],
                        "n_contexts": stats["contexts"],
                    }
                )
        pooled = aggregate(rows)
        print("  " + "  ".join(f"{metric}={pooled[metric]:.4f}" for metric in METRICS), flush=True)

    # ---- AGS vs one-pass, paired, per retriever (the spec's question 2) -------------------
    delta_rows: list[dict[str, Any]] = []
    for retriever_kind in ("bm25", "dense", "hybrid"):
        ags_key = next(
            (key for key in per_variant_rows if key[0] == retriever_kind and key[1] == "AGS" and key[2] == (1.0 if retriever_kind == "bm25" else 0.0)),
            None,
        )
        onepass_key = next(
            (key for key in per_variant_rows if key[0] == retriever_kind and key[1] == "one_pass"), None
        )
        if ags_key is None or onepass_key is None:
            continue
        for modality in MODALITIES:
            left = subset(per_variant_rows[ags_key], modality)
            right = subset(per_variant_rows[onepass_key], modality)
            if not left or not right:
                continue
            for metric in METRICS:
                result = paired_bootstrap(
                    as_bootstrap_input(left, metric),
                    as_bootstrap_input(right, metric),
                    metric,
                    args.bootstrap_samples,
                    args.bootstrap_seed,
                )
                if result:
                    delta_rows.append(
                        {
                            "retriever": retriever_kind,
                            "comparison": "AGS - one_pass",
                            "modality": modality,
                            **result,
                        }
                    )

    # ---- the comparison the spec asks to be called out explicitly ------------------------
    def lookup(retriever_kind: str, method: str, w_cov: float, modality: str, metric: str) -> float | None:
        rows = per_variant_rows.get((retriever_kind, method, w_cov))
        if not rows:
            return None
        modality_rows = subset(rows, modality)
        return round(float(aggregate(modality_rows)[metric]), 6) if modality_rows else None

    sparse_specific = {
        "question": (
            "Does dense one-pass, with no lexical coverage term, already reach BM25 one-pass "
            "WITH the coverage term? If yes, the length pathology Eq. 10 fixes is "
            "sparse-specific and the coverage term is a BM25 patch rather than a general gain."
        ),
        "comparisons": [
            {
                "modality": modality,
                "metric": metric,
                "bm25_one_pass_plus_cov": lookup("bm25", "one_pass", 1.0, modality, metric),
                "dense_one_pass_no_cov": lookup("dense", "one_pass", 0.0, modality, metric),
                "dense_minus_bm25": (
                    None
                    if lookup("dense", "one_pass", 0.0, modality, metric) is None
                    or lookup("bm25", "one_pass", 1.0, modality, metric) is None
                    else round(
                        lookup("dense", "one_pass", 0.0, modality, metric)
                        - lookup("bm25", "one_pass", 1.0, modality, metric),
                        6,
                    )
                ),
            }
            for modality in MODALITIES
            for metric in METRICS
        ],
    }

    metrics = {
        "dense_model": args.dense_model,
        "dense_index_type": "exact cosine over unit-normalized embeddings (no ANN approximation)",
        "dense_query_instruction": args.query_instruction,
        "hybrid_combination_rule": "reciprocal rank fusion over the BM25 and dense ranked lists",
        "hybrid_rrf_kappa": args.rrf_kappa,
        "text_representation": (
            "Concept.retrieval_text -- '{tag}. {standard_label}. {documentation}', the identical "
            "string BM25 tokenizes. Only the retrieval model changes."
        ),
        "embedding_cache": str(cache_path) if cache_path is not None else None,
        "queries_reused_no_regeneration": True,
        "query_sources": {
            "AGS": f"{args.ags_trace} rounds[].query per (hypothesis_idx, rendering), "
            "label_render_skipped rounds not re-issued",
            "one_pass": f"{args.onepass_trace} record.query",
        },
        "consolidation": (
            "verifier.core.evaluate at the frozen AGS config (sum-RRF, "
            "range-normalized, beta unchanged); reset_consensus_cache() is called at the start "
            "of every AGS variant because the cache is not keyed by retriever."
        ),
        "beta": args.beta,
        "beta_not_retuned_note": (
            "beta is NOT re-tuned per retriever. The rerank operates on range-normalized fused "
            "scores, so beta transfers; changing the retriever changes absolute scores but not "
            "the scale the rerank sees. This is an assumption and belongs in the caption. The "
            "spec suggests spot-checking one beta on dev under the dense index if time permits."
        ),
        "coverage_term_policy": (
            "The coverage term (Eq. 10) is a lexical feature and is BM25-specific. The BM25 rows "
            "use w_cov=1.0 as in the main results; dense and hybrid rows are reported WITHOUT it "
            "unless marked '+cov' (coverage_term=on), which is run for AGS only."
        ),
        "sparse_specificity_check": sparse_specific,
        "n_ags_facts": len(ags_facts),
        "n_onepass_facts": len(onepass_facts),
        "bootstrap": {"iterations": args.bootstrap_samples, "seed": args.bootstrap_seed, "unit": "source context"},
        "limit": args.limit,
        "artifact_paths": {
            "retriever_robustness_csv": str(args.output_dir / "retriever_robustness.csv"),
            "deltas_csv": str(args.output_dir / "retriever_robustness_deltas.csv"),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "retriever_robustness.csv", summary_rows)
    write_csv(args.output_dir / "retriever_robustness_deltas.csv", delta_rows)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {args.output_dir / 'retriever_robustness.csv'}, deltas CSV and metrics.json", flush=True)


if __name__ == "__main__":
    main()

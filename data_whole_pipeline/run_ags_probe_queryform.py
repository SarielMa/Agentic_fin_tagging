#!/usr/bin/env python3
"""Table 10: retrieval-readiness diagnostics, on the FROZEN TEST SPLIT.

Replaces the development placeholders (n=566 over 30 tabular contexts) that
`run_ags_coverage_query_form_interaction.py` produced. Same six query
representations, same retriever configuration, same estimator -- different data.

The two claims this table supports
----------------------------------
  "The gap is interpretive rather than index-side."
      The probe queries the index with the gold concept's own canonical label
      and definition. If the target comes back at the top, the index can reach
      it and the difficulty is in expressing the query, not in the index.

  "The gap is primarily a precision gap."
      Raw evidence reaches the gold concept within a large pool nearly as often
      as a grounding does, but ranks it far lower. The paired structured-versus-
      raw contrast at Recall@10 is the quantity that says so.

Query representations (one hypothesis throughout; the J=2 fusion of the deployed
pipeline is deliberately held out so def/lab/dual differ only in rendering):

  probe            gold concept's canonical label + definition   (no LLM)
  raw_context      build_direct_query, the fact and its context  (no LLM)
  freetext         the one-pass grounding description            (reused)
  structured_def   primary hypothesis, definition-form rendering (reused)
  structured_lab   primary hypothesis, label-form rendering      (reused)
  structured_dual  RRF fusion of that hypothesis's def and lab retrievals

Everything is replayed from logged queries; no generation happens here. BM25 is
deterministic, so replaying a logged query against the same taxonomy and the same
retriever settings reproduces the ranking that generation time saw.

Both label-coverage settings are run (w_cov = 0.0 and 1.0) because the appendix
also claims the factorized representation is not an artifact of the coverage term.

CPU only. No GPU, no generation.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    SCRIPT_DIR,
    Example,
    TaxonomyRetriever,
    build_direct_query,
    fuse_round_candidates,
    load_examples,
    load_taxonomy,
    normalize_space,
    normalize_tag,
    retrieval_query_from_grounding,
    retrieve_candidates,
)


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs_ags_probe_queryform" / "qwen3_32b"
DEFAULT_TEST_JSONL = SCRIPT_DIR / "FinTagging_800_200_grounding_test_JSON" / "data" / "test.jsonl"
DEFAULT_FROZEN_TRACE = (
    SCRIPT_DIR / "runs_fintagging_grounding_baseline" / "qwen3_32b_frozen_ags" / "bm25_candidates.jsonl"
)
DEFAULT_ONE_PASS_QUERIES = (
    SCRIPT_DIR
    / "runs_fintagging_grounding_baseline"
    / "qwen3_32b_one_pass_grounding"
    / "query_descriptions.jsonl"
)

FORMS = ("probe", "raw_context", "freetext", "structured_def", "structured_lab", "structured_dual")
# The probe is a ceiling, not a query form under study; contrasts exclude it.
CONTRAST_FORMS = ("raw_context", "freetext", "structured_def", "structured_lab", "structured_dual")
DEPTHS = (10, 50, 200)
METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr")
PRIMARY_HYPOTHESIS_IDX = 0
WEIGHTS = {"on": 1.0, "off": 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--frozen-trace", type=Path, default=DEFAULT_FROZEN_TRACE)
    parser.add_argument("--one-pass-queries", type=Path, default=DEFAULT_ONE_PASS_QUERIES)
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument(
        "--modality",
        default="table",
        choices=("table", "text", "pooled"),
        help="The paper's Table 10 is tabular evidence.",
    )
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument("--limit", type=int, default=None, help="Smoke test: first N facts.")
    parser.add_argument("--log-every", type=int, default=200)
    return parser.parse_args()


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_structured_queries(path: Path) -> dict[int, dict[str, str]]:
    """Per fact, the primary hypothesis's def and lab retrieval queries.

    Read from the frozen AGS trace's own per-round log, where each round records
    the rendering it used and the exact query string it issued.
    """
    out: dict[int, dict[str, str]] = {}
    for record in stream_jsonl(path):
        fact_id = int(record["example_idx"])
        queries: dict[str, str] = {}
        for round_record in record.get("rounds") or []:
            if int(round_record.get("hypothesis_idx", -1)) != PRIMARY_HYPOTHESIS_IDX:
                continue
            if round_record.get("label_render_skipped"):
                continue
            rendering = round_record.get("rendering")
            query = normalize_space(round_record.get("query", ""))
            if rendering in ("def", "lab") and query:
                queries[rendering] = query
        if queries:
            out[fact_id] = queries
    return out


def load_freetext_queries(path: Path, examples_by_id: dict[int, Example]) -> dict[int, str]:
    """The one-pass grounding description, rendered into a retrieval query.

    The log stores the generated description; the query actually issued is that
    description passed through retrieval_query_from_grounding, exactly as the
    one-pass run did.
    """
    out: dict[int, str] = {}
    for record in stream_jsonl(path):
        fact_id = int(record.get("example_idx", record.get("fact_id", -1)))
        example = examples_by_id.get(fact_id)
        if example is None:
            continue
        description = normalize_space(
            record.get("query_description") or record.get("description") or record.get("query") or ""
        )
        if description:
            out[fact_id] = retrieval_query_from_grounding(example, description)
    return out


def probe_query(example: Example, concept: Any) -> str:
    """The gold concept's own canonical label and definition.

    This is the empirical analogue of g* in the gap formulation: the query that
    states the intended concept outright. It uses the gold label, so it is a
    ceiling and never a method.
    """
    parts = [normalize_space(getattr(concept, "standard_label", "")), normalize_space(getattr(concept, "documentation", ""))]
    return retrieval_query_from_grounding(example, " ".join(part for part in parts if part))


def first_gold_rank(tags: list[str], gold_tags: list[str]) -> int | None:
    gold = {normalize_tag(tag) for tag in gold_tags}
    for index, tag in enumerate(tags, start=1):
        if normalize_tag(tag) in gold:
            return index
    return None


def retrieve_form(
    retriever: TaxonomyRetriever,
    example: Example,
    form: str,
    queries: dict[str, str],
    top_k: int,
    rrf_kappa: float,
) -> int | None:
    if form == "structured_dual":
        def_candidates = retrieve_candidates(retriever, queries["structured_def"], example.entity_type, top_k)
        lab_candidates = retrieve_candidates(retriever, queries["structured_lab"], example.entity_type, top_k)
        candidates = fuse_round_candidates(
            [{"round": 1, "candidates": def_candidates}, {"round": 2, "candidates": lab_candidates}],
            top_k,
            rrf_kappa,
        )
    else:
        query = queries.get(form)
        if not query:
            return None
        candidates = retrieve_candidates(retriever, query, example.entity_type, top_k)
    return first_gold_rank([candidate["tag"] for candidate in candidates], example.gold_tags)


def metric_from_rank(rank: int | None) -> dict[str, float]:
    values = {f"recall_at_{depth}": float(rank is not None and rank <= depth) for depth in DEPTHS}
    values["mrr"] = 0.0 if rank is None else 1.0 / rank
    return values


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (1.0 - (pos - lower)) + ordered[upper] * (pos - lower)


def paired_contrast(
    per_fact: dict[str, dict[int, dict[str, float]]],
    left: str,
    right: str,
    metric: str,
    context_of: dict[int, str],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    """Paired context-level bootstrap of (left - right) on one metric.

    Every form is scored on the identical fact set, so the difference is taken
    per fact and contexts are resampled jointly; that is what makes this a paired
    contrast rather than two independent intervals compared by eye.
    """
    diffs_by_context: dict[str, list[float]] = defaultdict(list)
    for fact_id, values in per_fact[left].items():
        if fact_id not in per_fact[right]:
            continue
        diffs_by_context[context_of[fact_id]].append(values[metric] - per_fact[right][fact_id][metric])
    keys = sorted(diffs_by_context)
    if not keys:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "ci_excludes_zero": False, "contexts": 0}
    flat = [value for key in keys for value in diffs_by_context[key]]
    observed = sum(flat) / len(flat)

    rng = random.Random(seed)
    samples: list[float] = []
    size = len(keys)
    for _ in range(iterations):
        pooled: list[float] = []
        for _ in range(size):
            pooled.extend(diffs_by_context[keys[rng.randrange(size)]])
        if pooled:
            samples.append(sum(pooled) / len(pooled))
    low, high = percentile(samples, 0.025), percentile(samples, 0.975)
    return {
        "mean": round(observed, 6),
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "contexts": size,
        "facts": len(flat),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in (args.test_jsonl, args.frozen_trace, args.one_pass_queries, args.taxonomy_jsonl):
        if not Path(path).exists():
            raise SystemExit(f"Missing required input: {path}")

    print(f"Loading taxonomy from {args.taxonomy_jsonl}", flush=True)
    taxonomy = load_taxonomy(args.taxonomy_jsonl)
    concepts_by_tag = {concept.tag: concept for concept in taxonomy}

    examples = load_examples(args.test_jsonl)
    if args.modality != "pooled":
        examples = [example for example in examples if example.input_type == args.modality]
    if args.limit is not None:
        examples = examples[: args.limit]
    examples_by_id = {example.example_idx: example for example in examples}
    context_of = {example.example_idx: str(example.context_id) for example in examples}
    print(f"{len(examples)} {args.modality} facts over {len(set(context_of.values()))} contexts", flush=True)

    structured = load_structured_queries(args.frozen_trace)
    freetext = load_freetext_queries(args.one_pass_queries, examples_by_id)
    print(f"structured queries: {len(structured)} facts | freetext: {len(freetext)} facts", flush=True)

    retriever = TaxonomyRetriever(
        taxonomy, type_filter=True, label_coverage_weight=1.0, label_coverage_pool_multiplier=0
    )

    # weight -> form -> fact_id -> metrics
    per_fact: dict[str, dict[str, dict[int, dict[str, float]]]] = {
        label: {form: {} for form in FORMS} for label in WEIGHTS
    }
    missing: dict[str, int] = defaultdict(int)

    for label, weight in WEIGHTS.items():
        retriever.label_coverage_weight = weight
        retriever.label_coverage_pool_multiplier = 0
        print(f"\n--- label coverage {label} (w_cov={weight}) ---", flush=True)
        for index, example in enumerate(examples, start=1):
            fact_id = example.example_idx
            concept = concepts_by_tag.get(normalize_tag(example.gold_tags[0])) if example.gold_tags else None
            structured_q = structured.get(fact_id, {})
            queries = {
                "probe": probe_query(example, concept) if concept is not None else "",
                "raw_context": build_direct_query(example),
                "freetext": freetext.get(fact_id, ""),
                "structured_def": structured_q.get("def", ""),
                "structured_lab": structured_q.get("lab", ""),
            }
            for form in FORMS:
                if form != "structured_dual" and not queries.get(form):
                    missing[f"{label}:{form}"] += 1
                    continue
                if form == "structured_dual" and not (
                    queries["structured_def"] and queries["structured_lab"]
                ):
                    missing[f"{label}:{form}"] += 1
                    continue
                rank = retrieve_form(retriever, example, form, queries, args.top_k, args.rrf_kappa)
                per_fact[label][form][fact_id] = metric_from_rank(rank)
            if args.log_every and index % args.log_every == 0:
                print(f"  {index}/{len(examples)} facts", flush=True)

    # ------------------------------------------------------------------ tables
    rows: list[dict[str, Any]] = []
    for label in WEIGHTS:
        for form in FORMS:
            values = per_fact[label][form]
            if not values:
                continue
            row = {"label_coverage": label, "query_form": form, "n": len(values)}
            for metric in METRICS:
                row[metric] = round(sum(v[metric] for v in values.values()) / len(values), 6)
            rows.append(row)

    contrasts: list[dict[str, Any]] = []
    seed_offset = 0
    contrast_specs = [
        ("structured_def", "raw_context", "recall_at_10"),
        ("structured_lab", "raw_context", "recall_at_10"),
        ("structured_dual", "raw_context", "recall_at_10"),
        ("freetext", "raw_context", "recall_at_10"),
        ("freetext", "raw_context", "recall_at_50"),
        ("freetext", "raw_context", "recall_at_200"),
        ("structured_def", "freetext", "recall_at_10"),
        ("structured_lab", "freetext", "recall_at_10"),
    ]
    for label in WEIGHTS:
        for left, right, metric in contrast_specs:
            seed_offset += 1
            result = paired_contrast(
                per_fact[label], left, right, metric, context_of, args.bootstrap_samples,
                args.bootstrap_seed + seed_offset,
            )
            contrasts.append({"label_coverage": label, "left": left, "right": right, "metric": metric, **result})

    def write_csv(name: str, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    write_csv("query_form_metrics.csv", rows)
    write_csv("paired_contrasts.csv", contrasts)

    on_rows = {row["query_form"]: row for row in rows if row["label_coverage"] == "on"}
    table10 = [
        {
            "query_representation": form,
            "recall_at_10": on_rows[form]["recall_at_10"],
            "recall_at_50": on_rows[form]["recall_at_50"],
            "recall_at_200": on_rows[form]["recall_at_200"],
            "mrr": on_rows[form]["mrr"],
            "n": on_rows[form]["n"],
        }
        for form in FORMS
        if form in on_rows
    ]
    write_csv("table10.csv", table10)

    metrics = {
        "experiment": "ags_probe_queryform",
        "split": "test",
        "table": "Table 10 (Appendix: Retrieval-Readiness Diagnostics)",
        "modality": args.modality,
        "n_facts": len(examples),
        "n_contexts": len(set(context_of.values())),
        "table10_label_coverage_on": table10,
        "all_rows": rows,
        "paired_contrasts": contrasts,
        "missing_queries": dict(missing),
        "config": {
            "test_jsonl": str(args.test_jsonl),
            "frozen_trace": str(args.frozen_trace),
            "one_pass_queries": str(args.one_pass_queries),
            "top_k": args.top_k,
            "rrf_kappa": args.rrf_kappa,
            "primary_hypothesis_idx": PRIMARY_HYPOTHESIS_IDX,
            "bootstrap": {"iterations": args.bootstrap_samples, "seed": args.bootstrap_seed, "unit": "context"},
            "probe_note": (
                "The probe queries with the gold concept's own label and definition. It is a "
                "ceiling on what the index can reach, not a method, and is excluded from contrasts."
            ),
            "single_hypothesis_note": (
                "def/lab/dual all render the primary hypothesis (idx 0), so they differ only in "
                "rendering; the deployed J=2 fusion is held out."
            ),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\n=== Table 10 ({args.modality} evidence, label coverage on, test split) ===", flush=True)
    print(f"  {'Query representation':<32}{'R@10':>8}{'R@50':>8}{'R@200':>8}{'MRR':>8}", flush=True)
    for row in table10:
        print(
            f"  {row['query_representation']:<32}{row['recall_at_10']:>8.3f}{row['recall_at_50']:>8.3f}"
            f"{row['recall_at_200']:>8.3f}{row['mrr']:>8.3f}",
            flush=True,
        )
    print(f"\n  n={len(examples)} facts / {len(set(context_of.values()))} contexts", flush=True)
    print("\n  Key paired contrasts (label coverage on):", flush=True)
    for c in contrasts:
        if c["label_coverage"] != "on":
            continue
        mark = " *" if c["ci_excludes_zero"] else ""
        print(
            f"    {c['left']:<16} - {c['right']:<12} {c['metric']:<14} "
            f"{c['mean']:+.4f} [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]{mark}",
            flush=True,
        )
    if missing:
        print(f"\n  missing queries: {dict(missing)}", flush=True)
    print(f"\nWrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

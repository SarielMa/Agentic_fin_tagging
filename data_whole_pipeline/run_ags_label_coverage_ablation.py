#!/usr/bin/env python3
"""Label-coverage ablation on the frozen AGS 661-fact sample.

Runs direct retrieval and one-pass grounding twice each, with
label_coverage_weight off (0.0) and on (1.0), over the same frozen
context-level sample. Retrieval-only: direct retrieval needs no LLM, and the
one-pass grounding generations are reused from the coverage-pilot Arm A
records (one_pass_reference) rather than regenerated.

Reports Recall@10/50/200 and MRR per query mode and weight, the paired
(on - off) gain per fact with context-level bootstrap CIs, and the same gain
stratified by gold standard-label token count.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from run_ags_coverage_pilot import (
    bootstrap_context_ci,
    fact_context_key,
    load_jsonl,
    row_to_example,
)
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    SCRIPT_DIR,
    Example,
    TaxonomyRetriever,
    build_direct_query,
    first_gold_rank,
    load_taxonomy,
    normalize_space,
    normalize_tag,
    retrieval_query_from_grounding,
    retrieve_candidates,
    tokenize,
    write_jsonl,
)


DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "runs_ags_label_coverage_ablation" / "qwen3_32b"
DEFAULT_SAMPLE_PATH = SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "sample_facts.jsonl"
DEFAULT_ONE_PASS_PATH = SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "arm_A_hypothesis_candidates.jsonl"

QUERY_MODES = ("direct_retrieval", "one_pass_grounding")
WEIGHT_LABELS = {0.0: "off", 1.0: "on"}
MODALITIES = ("pooled", "table", "text")
DEPTHS = (10, 50, 200)
METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr")
LABEL_TOKEN_STRATA = ("1", "2", "3-4", "5+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument(
        "--one-pass-path",
        type=Path,
        default=DEFAULT_ONE_PASS_PATH,
        help="Coverage-pilot Arm A records supplying the reused one-pass generations.",
    )
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--label-coverage-weight-off", type=float, default=0.0)
    parser.add_argument("--label-coverage-weight-on", type=float, default=1.0)
    parser.add_argument(
        "--label-coverage-pool-multiplier",
        type=int,
        default=0,
        help="Pool multiplier for label-coverage rescoring; <=0 scores the full type-filtered pool.",
    )
    parser.add_argument("--type-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


@contextmanager
def temporary_retriever_weight(
    retriever: TaxonomyRetriever,
    label_coverage_weight: float,
    label_coverage_pool_multiplier: int,
) -> Iterable[None]:
    old_weight = retriever.label_coverage_weight
    old_multiplier = retriever.label_coverage_pool_multiplier
    retriever.label_coverage_weight = label_coverage_weight
    retriever.label_coverage_pool_multiplier = label_coverage_pool_multiplier
    try:
        yield
    finally:
        retriever.label_coverage_weight = old_weight
        retriever.label_coverage_pool_multiplier = old_multiplier


def label_token_count(label: str) -> int:
    return len(set(tokenize(label)))


def label_token_stratum(token_count: int) -> str:
    if token_count <= 1:
        return "1"
    if token_count == 2:
        return "2"
    if token_count <= 4:
        return "3-4"
    return "5+"


def load_one_pass_generations(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Load reusable Arm A one-pass generations and their provenance."""
    generations: dict[int, dict[str, Any]] = {}
    hypothesis_indices: Counter[int] = Counter()
    temperatures: Counter[float] = Counter()
    parse_flags: Counter[bool] = Counter()
    kinds: Counter[str] = Counter()
    models: Counter[str] = Counter()
    backends: Counter[str] = Counter()
    context_chars: Counter[int] = Counter()

    for record in load_jsonl(path):
        fact_id = int(record["fact_id"])
        hypothesis_idx = int(record.get("hypothesis_idx", 0))
        hypothesis_indices[hypothesis_idx] += 1
        llm_call = record.get("llm_call") or {}
        temperatures[float(record.get("generation_temperature", llm_call.get("generation_temperature", -1.0)))] += 1
        parse_flags[bool(llm_call.get("parse_ok"))] += 1
        kinds[str(llm_call.get("kind"))] += 1
        models[str(llm_call.get("model"))] += 1
        backends[str(llm_call.get("backend"))] += 1
        context_chars[int(llm_call.get("used_context_max_chars", -1))] += 1
        if hypothesis_idx != 0:
            raise ValueError(f"Expected a single one-pass hypothesis per fact, saw idx={hypothesis_idx}")
        if fact_id in generations:
            raise ValueError(f"Duplicate one-pass generation for fact_id={fact_id}")
        generations[fact_id] = {
            "query_text": normalize_space(record.get("query_text", "")),
            "retrieval_query": normalize_space(record.get("retrieval_query", "")),
        }

    provenance = {
        "path": str(path),
        "record_count": sum(hypothesis_indices.values()),
        "fact_count": len(generations),
        "hypotheses_per_fact": dict(hypothesis_indices),
        "generation_temperature": dict(temperatures),
        "parse_ok": {str(key): value for key, value in parse_flags.items()},
        "llm_call_kind": dict(kinds),
        "query_generation_model": dict(models),
        "query_generation_backend": dict(backends),
        "used_context_max_chars": dict(context_chars),
    }
    return generations, provenance


def check_one_pass_reuse(
    examples: list[Example],
    generations: dict[int, dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Verify the reused generations line up fact-for-fact with the frozen sample."""
    sample_ids = {example.example_idx for example in examples}
    missing = sorted(sample_ids - set(generations))
    extra = sorted(set(generations) - sample_ids)

    rebuilt_mismatches = []
    empty_generations = []
    for example in examples:
        generation = generations.get(example.example_idx)
        if generation is None:
            continue
        if not generation["query_text"]:
            empty_generations.append(example.example_idx)
        rebuilt = retrieval_query_from_grounding(example, generation["query_text"])
        if generation["retrieval_query"] and rebuilt != generation["retrieval_query"]:
            rebuilt_mismatches.append(example.example_idx)

    failures = []
    if missing:
        failures.append(f"{len(missing)} sample facts have no reusable generation")
    if extra:
        failures.append(f"{len(extra)} generations are not in the frozen sample")
    if rebuilt_mismatches:
        failures.append(f"{len(rebuilt_mismatches)} rebuilt retrieval queries differ from the stored query")
    if provenance["parse_ok"].get("False"):
        failures.append("some reused generations failed JSON parsing")
    if set(provenance["llm_call_kind"]) != {"one_pass"}:
        failures.append(f"unexpected llm_call kinds: {sorted(provenance['llm_call_kind'])}")

    return {
        "reused": True,
        "passed": not failures,
        "failures": failures,
        "missing_fact_ids": missing[:20],
        "extra_fact_ids": extra[:20],
        "rebuilt_query_mismatch_fact_ids": rebuilt_mismatches[:20],
        "empty_generation_fact_ids": empty_generations[:20],
        "retriever_config_note": (
            "One-pass generation is retriever-independent: build_query_description_messages conditions "
            "only on the fact evidence, with no candidate list in the prompt. The pilot's metrics.json "
            "does not persist a retriever-config block, so the recorded-and-matching test is satisfied "
            "on the generation config (model, backend, temperature, prompt builder, context budget) "
            "rather than on retrieval, and all retrieval in this ablation is re-run here under both "
            "label_coverage_weight settings."
        ),
        "generation_provenance": provenance,
    }


def build_queries(
    examples: list[Example],
    query_mode: str,
    generations: dict[int, dict[str, Any]],
) -> dict[int, str]:
    if query_mode == "direct_retrieval":
        return {example.example_idx: build_direct_query(example) for example in examples}
    if query_mode == "one_pass_grounding":
        return {
            example.example_idx: retrieval_query_from_grounding(
                example,
                generations[example.example_idx]["query_text"],
            )
            for example in examples
        }
    raise ValueError(f"Unsupported query_mode={query_mode}")


def run_config(
    examples: list[Example],
    retriever: TaxonomyRetriever,
    queries: dict[int, str],
    top_k: int,
    log_every: int,
    log_prefix: str,
) -> dict[int, int | None]:
    ranks: dict[int, int | None] = {}
    for position, example in enumerate(examples, start=1):
        candidates = retrieve_candidates(
            retriever,
            queries[example.example_idx],
            example.entity_type,
            top_k,
        )
        ranks[example.example_idx] = first_gold_rank(
            [candidate["tag"] for candidate in candidates],
            example.gold_tags,
        )
        if log_every and position % log_every == 0:
            print(f"{log_prefix}: {position}/{len(examples)} facts", flush=True)
    return ranks


def metric_values(ranks: dict[int, int | None]) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, float]] = {metric: {} for metric in METRICS}
    for fact_id, rank in ranks.items():
        for depth in DEPTHS:
            values[f"recall_at_{depth}"][fact_id] = float(rank is not None and rank <= depth)
        values["mrr"][fact_id] = 0.0 if rank is None else 1.0 / rank
    return values


def fact_ids_for_modality(examples: list[Example], modality: str) -> list[int]:
    return [
        example.example_idx
        for example in examples
        if modality == "pooled" or example.input_type == modality
    ]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_main_rows(
    args: argparse.Namespace,
    examples: list[Example],
    examples_by_id: dict[int, Example],
    values: dict[tuple[str, str], dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for query_mode in QUERY_MODES:
        for weight_label in ("off", "on"):
            for modality in MODALITIES:
                fact_ids = fact_ids_for_modality(examples, modality)
                for metric in METRICS:
                    seed_offset += 1
                    ci = bootstrap_context_ci(
                        values[(query_mode, weight_label)][metric],
                        examples_by_id,
                        fact_ids,
                        iterations=args.bootstrap_samples,
                        seed=args.bootstrap_seed + seed_offset,
                    )
                    rows.append(
                        {
                            "query_mode": query_mode,
                            "label_coverage": weight_label,
                            "label_coverage_weight": (
                                args.label_coverage_weight_off
                                if weight_label == "off"
                                else args.label_coverage_weight_on
                            ),
                            "modality": modality,
                            "metric": metric,
                            "n_facts": ci.get("fact_count", len(fact_ids)),
                            "n_contexts": ci.get("context_count", 0),
                            "value": ci["mean"],
                            "ci_low": ci["ci_low"],
                            "ci_high": ci["ci_high"],
                        }
                    )
    return rows


def build_gain_rows(
    args: argparse.Namespace,
    examples: list[Example],
    examples_by_id: dict[int, Example],
    values: dict[tuple[str, str], dict[str, dict[int, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_offset = 10_000
    for query_mode in QUERY_MODES:
        for modality in MODALITIES:
            fact_ids = fact_ids_for_modality(examples, modality)
            for metric in METRICS:
                seed_offset += 1
                off_values = values[(query_mode, "off")][metric]
                on_values = values[(query_mode, "on")][metric]
                paired = {fact_id: on_values[fact_id] - off_values[fact_id] for fact_id in fact_ids}
                ci = bootstrap_context_ci(
                    paired,
                    examples_by_id,
                    fact_ids,
                    iterations=args.bootstrap_samples,
                    seed=args.bootstrap_seed + seed_offset,
                )
                rows.append(
                    {
                        "query_mode": query_mode,
                        "modality": modality,
                        "metric": metric,
                        "n_facts": ci.get("fact_count", len(fact_ids)),
                        "n_contexts": ci.get("context_count", 0),
                        "off": round(mean([off_values[fact_id] for fact_id in fact_ids]), 6),
                        "on": round(mean([on_values[fact_id] for fact_id in fact_ids]), 6),
                        "gain": ci["mean"],
                        "ci_low": ci["ci_low"],
                        "ci_high": ci["ci_high"],
                        "ci_excludes_zero": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
                        "facts_improved": sum(1 for fact_id in fact_ids if paired[fact_id] > 0),
                        "facts_unchanged": sum(1 for fact_id in fact_ids if paired[fact_id] == 0),
                        "facts_degraded": sum(1 for fact_id in fact_ids if paired[fact_id] < 0),
                    }
                )
    return rows


def build_stratum_rows(
    args: argparse.Namespace,
    examples: list[Example],
    examples_by_id: dict[int, Example],
    values: dict[tuple[str, str], dict[str, dict[int, float]]],
    stratum_by_fact: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_offset = 20_000
    for query_mode in QUERY_MODES:
        for stratum in LABEL_TOKEN_STRATA:
            fact_ids = [
                example.example_idx
                for example in examples
                if stratum_by_fact[example.example_idx] == stratum
            ]
            for metric in METRICS:
                seed_offset += 1
                off_values = values[(query_mode, "off")][metric]
                on_values = values[(query_mode, "on")][metric]
                paired = {fact_id: on_values[fact_id] - off_values[fact_id] for fact_id in fact_ids}
                ci = bootstrap_context_ci(
                    paired,
                    examples_by_id,
                    fact_ids,
                    iterations=args.bootstrap_samples,
                    seed=args.bootstrap_seed + seed_offset,
                )
                rows.append(
                    {
                        "query_mode": query_mode,
                        "gold_label_tokens": stratum,
                        "metric": metric,
                        "n_facts": len(fact_ids),
                        "n_contexts": ci.get("context_count", 0),
                        "off": round(mean([off_values[fact_id] for fact_id in fact_ids]), 6),
                        "on": round(mean([on_values[fact_id] for fact_id in fact_ids]), 6),
                        "gain": ci["mean"],
                        "ci_low": ci["ci_low"],
                        "ci_high": ci["ci_high"],
                        "ci_excludes_zero": bool(ci["ci_low"] > 0.0 or ci["ci_high"] < 0.0),
                        "facts_improved": sum(1 for fact_id in fact_ids if paired[fact_id] > 0),
                        "facts_unchanged": sum(1 for fact_id in fact_ids if paired[fact_id] == 0),
                        "facts_degraded": sum(1 for fact_id in fact_ids if paired[fact_id] < 0),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_rows = load_jsonl(args.sample_path)
    examples = [row_to_example(row) for row in sample_rows]
    examples_by_id = {example.example_idx: example for example in examples}
    context_keys = {fact_context_key(example) for example in examples}

    taxonomy = load_taxonomy(args.taxonomy_jsonl)
    taxonomy_by_tag = {concept.tag: concept for concept in taxonomy}

    gold_by_fact: dict[int, str] = {}
    stratum_by_fact: dict[int, str] = {}
    token_count_by_fact: dict[int, int] = {}
    missing_gold: list[int] = []
    for example in examples:
        if len(example.gold_tags) != 1:
            raise ValueError(
                f"Expected exactly one gold concept per fact, fact_id={example.example_idx} "
                f"has {len(example.gold_tags)}"
            )
        gold_tag = normalize_tag(example.gold_tags[0])
        gold_by_fact[example.example_idx] = gold_tag
        concept = taxonomy_by_tag.get(gold_tag)
        if concept is None:
            missing_gold.append(example.example_idx)
            token_count_by_fact[example.example_idx] = 0
            stratum_by_fact[example.example_idx] = "1"
            continue
        token_count = label_token_count(concept.standard_label or concept.raw_tag)
        token_count_by_fact[example.example_idx] = token_count
        stratum_by_fact[example.example_idx] = label_token_stratum(token_count)
    if missing_gold:
        raise ValueError(f"{len(missing_gold)} gold concepts are absent from the taxonomy: {missing_gold[:10]}")

    generations, provenance = load_one_pass_generations(args.one_pass_path)
    reuse_check = check_one_pass_reuse(examples, generations, provenance)
    print(json.dumps({"one_pass_reuse_check": reuse_check}, ensure_ascii=False, indent=2, sort_keys=True))
    if not reuse_check["passed"]:
        raise RuntimeError(
            "Aborting: reused one-pass generations do not match the frozen sample. "
            f"Failures: {reuse_check['failures']}"
        )

    retriever = TaxonomyRetriever(taxonomy, type_filter=args.type_filter)
    weights = {"off": args.label_coverage_weight_off, "on": args.label_coverage_weight_on}
    ranks: dict[tuple[str, str], dict[int, int | None]] = {}
    for query_mode in QUERY_MODES:
        queries = build_queries(examples, query_mode, generations)
        for weight_label, weight in weights.items():
            with temporary_retriever_weight(retriever, weight, args.label_coverage_pool_multiplier):
                ranks[(query_mode, weight_label)] = run_config(
                    examples,
                    retriever,
                    queries,
                    args.top_k,
                    args.log_every,
                    f"{query_mode}/label_coverage={weight_label}",
                )

    values = {key: metric_values(fact_ranks) for key, fact_ranks in ranks.items()}

    per_fact_rows = [
        {
            "fact_id": example.example_idx,
            "context_key": fact_context_key(example),
            "context_id": example.context_id,
            "modality": example.input_type,
            "entity_type": example.entity_type,
            "gold_tag": gold_by_fact[example.example_idx],
            "gold_standard_label": taxonomy_by_tag[gold_by_fact[example.example_idx]].standard_label,
            "gold_label_token_count": token_count_by_fact[example.example_idx],
            "gold_label_tokens": stratum_by_fact[example.example_idx],
            **{
                f"rank_{query_mode}_{weight_label}": ranks[(query_mode, weight_label)][example.example_idx]
                for query_mode in QUERY_MODES
                for weight_label in ("off", "on")
            },
        }
        for example in examples
    ]
    write_jsonl(args.output_dir / "per_fact_ranks.jsonl", per_fact_rows)

    main_rows = build_main_rows(args, examples, examples_by_id, values)
    gain_rows = build_gain_rows(args, examples, examples_by_id, values)
    stratum_rows = build_stratum_rows(args, examples, examples_by_id, values, stratum_by_fact)

    write_csv(
        args.output_dir / "label_coverage_metrics.csv",
        main_rows,
        [
            "query_mode",
            "label_coverage",
            "label_coverage_weight",
            "modality",
            "metric",
            "n_facts",
            "n_contexts",
            "value",
            "ci_low",
            "ci_high",
        ],
    )
    write_csv(
        args.output_dir / "label_coverage_paired_gain.csv",
        gain_rows,
        [
            "query_mode",
            "modality",
            "metric",
            "n_facts",
            "n_contexts",
            "off",
            "on",
            "gain",
            "ci_low",
            "ci_high",
            "ci_excludes_zero",
            "facts_improved",
            "facts_unchanged",
            "facts_degraded",
        ],
    )
    write_csv(
        args.output_dir / "label_coverage_gain_by_label_tokens.csv",
        stratum_rows,
        [
            "query_mode",
            "gold_label_tokens",
            "metric",
            "n_facts",
            "n_contexts",
            "off",
            "on",
            "gain",
            "ci_low",
            "ci_high",
            "ci_excludes_zero",
            "facts_improved",
            "facts_unchanged",
            "facts_degraded",
        ],
    )

    stratum_counts = Counter(stratum_by_fact.values())
    stratum_modality_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        stratum_modality_counts[stratum_by_fact[example.example_idx]][example.input_type] += 1

    metrics = {
        "config": {
            "sample_path": str(args.sample_path),
            "taxonomy_jsonl": str(args.taxonomy_jsonl),
            "top_k": args.top_k,
            "type_filter": args.type_filter,
            "label_coverage_weight_off": args.label_coverage_weight_off,
            "label_coverage_weight_on": args.label_coverage_weight_on,
            "label_coverage_pool_multiplier": args.label_coverage_pool_multiplier,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_unit": "context",
            "query_modes": list(QUERY_MODES),
        },
        "sample": {
            "fact_count": len(examples),
            "context_count": len(context_keys),
            "modality_fact_counts": dict(Counter(example.input_type for example in examples)),
            "modality_context_counts": {
                modality: len({fact_context_key(example) for example in examples if example.input_type == modality})
                for modality in ("table", "text")
            },
        },
        "gold_label_token_strata": {
            "definition": "unique retriever tokens in the gold concept standard_label (same tokenizer as label_coverage)",
            "fact_counts": {stratum: stratum_counts.get(stratum, 0) for stratum in LABEL_TOKEN_STRATA},
            "modality_counts": {
                stratum: dict(stratum_modality_counts.get(stratum, Counter()))
                for stratum in LABEL_TOKEN_STRATA
            },
        },
        "one_pass_reuse_check": reuse_check,
        "metrics_rows": main_rows,
        "paired_gain_rows": gain_rows,
        "gain_by_label_tokens_rows": stratum_rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "paired_gain_rows": gain_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

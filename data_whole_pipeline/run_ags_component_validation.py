#!/usr/bin/env python3
"""Experiment A: retrieval-side component validation for AGS grounding."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ags_symbolic_agreement import (
    DEFAULT_NORMALIZATION_MAP,
    agree,
    consensus_agreement,
    load_normalization_map,
    map_version,
    mean_agreement,
    normalize_hypothesis_dimensions,
)
from run_ags_coverage_pilot import (
    bootstrap_context_ci,
    compact_candidates,
    context_count_for_facts,
    fact_context_key,
    first_row,
    load_jsonl,
    parse_depths,
    prompt_under_budget,
    row_to_example,
    selected_fact_ids,
    temporary_generation_settings,
    write_csv,
)
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    DIMENSIONS,
    SCRIPT_DIR,
    Example,
    QueryGenerator,
    TaxonomyRetriever,
    build_direct_query,
    build_operator_initial_messages,
    first_gold_rank,
    fuse_round_candidates,
    llm_call_record,
    load_taxonomy,
    messages_to_prompt,
    normalize_space,
    normalize_tag,
    parse_hypothesis,
    retrieval_query_from_grounding,
    retrieve_candidates,
    scalar_text,
    tokenize,
    write_jsonl,
)


RENDERINGS = ("def", "lab", "dual")
MODALITIES = ("pooled", "table", "text")
PRIMARY_GATE_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs_ags_component_validation" / "qwen3_32b")
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "sample_facts.jsonl",
    )
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--depths", default="10,50,200")
    parser.add_argument("--hypotheses-per-fact", type=int, default=3)
    parser.add_argument("--hypothesis-temperature", type=float, default=0.8)
    parser.add_argument("--query-top-p", type=float, default=1.0)
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--dual-rrf-kappa", type=float, default=60.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument("--gate-min-effect", type=float, default=0.03)
    parser.add_argument("--agreement-top-m", type=int, default=10)
    parser.add_argument("--agreement-debug-facts", type=int, default=50)
    parser.add_argument("--agreement-debug-seed", type=int, default=20260725)
    parser.add_argument("--random-init-draws", type=int, default=100)
    parser.add_argument("--random-init-seed", type=int, default=20260726)
    parser.add_argument("--init-selection-k", type=int, default=50)
    parser.add_argument("--consensus-betas", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--type-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--dry-run-no-llm", action="store_true")

    parser.add_argument("--query-generation-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--query-generation-backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--query-context-max-chars", type=int, default=12000)
    parser.add_argument("--query-max-input-tokens", type=int, default=16000)
    parser.add_argument("--query-max-new-tokens", type=int, default=512)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--vllm-batch-size", type=int, default=32)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=30000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--query-temperature", type=float, default=0.8)
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float")
    return values


def dimensions_lower(dimensions: dict[str, Any]) -> dict[str, Any]:
    return {
        dimension.lower(): None
        if normalize_space(dimensions.get(dimension, "")).lower() == "unresolved"
        else normalize_space(dimensions.get(dimension, ""))
        for dimension in DIMENSIONS
    }


def dimensions_upper(dimensions: dict[str, Any]) -> dict[str, Any]:
    return {dimension: dimensions.get(dimension.lower()) for dimension in DIMENSIONS}


def render_definition(hypothesis: dict[str, Any]) -> str:
    query = normalize_space(hypothesis.get("retrieval_query", ""))
    if query:
        return query
    dims = dimensions_lower(hypothesis.get("dimensions", {}))
    resolved = [str(value) for value in dims.values() if value]
    return normalize_space(" ".join(resolved))


def render_label(hypothesis: dict[str, Any]) -> str:
    dims = dimensions_lower(hypothesis.get("dimensions", {}))
    values = [dims.get(dimension) for dimension in ("family", "role", "event", "qualifier", "scope", "temporal")]
    tokens: list[str] = []
    for value in values:
        if value:
            tokens.extend(tokenize(value))
    return normalize_space(" ".join(tokens))


def render_concept_label_query(concept: Any) -> str:
    return normalize_space(getattr(concept, "raw_tag", ""))


def candidate_ids(candidates: list[dict[str, Any]]) -> list[str]:
    return [normalize_tag(candidate.get("tag", "")) for candidate in candidates]


def candidate_scores(candidates: list[dict[str, Any]]) -> list[float | None]:
    return [
        candidate.get("rrf_score")
        if candidate.get("rrf_score") is not None
        else candidate.get("bm25_score")
        for candidate in candidates
    ]


def retrieve_rendering(
    retriever: TaxonomyRetriever,
    example: Example,
    rendered_query: str,
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    query = retrieval_query_from_grounding(example, rendered_query)
    return query, compact_candidates(retrieve_candidates(retriever, query, example.entity_type, top_k))


def dual_fuse(def_candidates: list[dict[str, Any]], lab_candidates: list[dict[str, Any]], top_k: int, rrf_kappa: float) -> list[dict[str, Any]]:
    return fuse_round_candidates(
        [
            {"round": 1, "candidates": def_candidates},
            {"round": 2, "candidates": lab_candidates},
        ],
        top_k,
        rrf_kappa,
    )


def fake_structured_output(example: Example, hypothesis_idx: int) -> str:
    base = build_direct_query(example)
    temporal = "duration" if example.input_type == "text" else "instant"
    gold_label = (example.gold_tags[0].replace("us-gaap:", "") if example.gold_tags else example.entity)
    query = f"{base} structured hypothesis {hypothesis_idx + 1}"
    return json.dumps(
        {
            "dimensions": {
                "FAMILY": gold_label,
                "ROLE": example.row_context or example.column_context or example.entity,
                "EVENT": "UNRESOLVED",
                "QUALIFIER": "UNRESOLVED",
                "SCOPE": "UNRESOLVED",
                "TEMPORAL": temporal,
            },
            "operators": ["direct_label"],
            "retrieval_query": query,
        },
        ensure_ascii=False,
    )


def run_tokenization_self_check(
    args: argparse.Namespace,
    examples: list[Example],
    retriever: TaxonomyRetriever,
    taxonomy_by_tag: dict[str, Any],
) -> dict[str, Any]:
    records = []
    failures = []
    seen: set[tuple[str, str]] = set()
    for example in examples:
        for gold in example.gold_tags:
            tag = normalize_tag(gold)
            if (tag, example.entity_type) in seen:
                continue
            seen.add((tag, example.entity_type))
            concept = taxonomy_by_tag.get(tag)
            if concept is None:
                failures.append({"gold_concept_id": tag, "reason": "missing_from_taxonomy"})
                continue
            query = render_concept_label_query(concept)
            candidates = compact_candidates(retrieve_candidates(retriever, query, concept.entity_type, args.top_k))
            top_tag = normalize_tag(candidates[0]["tag"]) if candidates else None
            rank = first_gold_rank(candidate_ids(candidates), [tag])
            retrieval_text_candidates = compact_candidates(
                retrieve_candidates(retriever, concept.retrieval_text, concept.entity_type, 1)
            )
            retrieval_text_top_tag = normalize_tag(retrieval_text_candidates[0]["tag"]) if retrieval_text_candidates else None
            passed = rank == 1
            record = {
                "gold_concept_id": tag,
                "entity_type": concept.entity_type,
                "standard_label": concept.standard_label,
                "label_query": query,
                "top_tag": top_tag,
                "gold_rank": rank,
                "retrieval_text_top_tag": retrieval_text_top_tag,
                "retrieval_text_rank1_passed": retrieval_text_top_tag == tag,
                "passed": passed,
            }
            records.append(record)
            if not passed:
                failures.append(record)
    return {
        "passed": not failures,
        "check": "gold_label_query_retrieves_self_at_rank_1",
        "label_query_mode": "space_lower_tokenized",
        "fact_count": len(examples),
        "unique_gold_type_pairs": len(records),
        "failure_count": len(failures),
        "failures": failures[:100],
        "records": records,
    }


def build_structured_hypotheses(
    args: argparse.Namespace,
    examples: list[Example],
    generator: QueryGenerator | None,
    normalization_map: dict[str, Any],
) -> list[dict[str, Any]]:
    output_path = args.output_dir / "hypotheses.jsonl"
    if args.resume and output_path.exists() and not args.overwrite:
        return load_jsonl(output_path)

    prompts: list[str] = []
    meta: list[tuple[Example, int, int, int]] = []
    for example in examples:
        for hypothesis_idx in range(args.hypotheses_per_fact):
            prompt, prompt_tokens, used_context_chars = prompt_under_budget(
                generator,
                args,
                lambda ctx_chars, ex=example: build_operator_initial_messages(ex, ctx_chars),
            )
            prompts.append(prompt)
            meta.append((example, hypothesis_idx, prompt_tokens, used_context_chars))

    if args.dry_run_no_llm:
        raw_outputs = [fake_structured_output(example, hypothesis_idx) for example, hypothesis_idx, _, _ in meta]
    else:
        assert generator is not None
        with temporary_generation_settings(
            generator,
            args.hypothesis_temperature,
            args.query_top_p,
            args.query_max_new_tokens,
        ):
            raw_outputs = generator.generate_many(prompts)

    records: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for output_idx, (raw_output, (example, hypothesis_idx, prompt_tokens, used_context_chars)) in enumerate(
            zip(raw_outputs, meta, strict=True),
            start=1,
        ):
            fallback = build_direct_query(example)
            hypothesis, parse_ok = parse_hypothesis(raw_output, fallback)
            dims = dimensions_lower(hypothesis.get("dimensions", {}))
            query_def = render_definition(hypothesis)
            query_lab = render_label(hypothesis)
            llm_call = llm_call_record(
                "operator_initial_hypothesis",
                raw_output=raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=0 if generator is None else generator.count_text_tokens(raw_output),
                parse_ok=parse_ok,
                backend="dry_run" if generator is None else generator.backend,
                model_name="dry_run" if generator is None else generator.model_name,
                extra_fields={
                    "used_context_max_chars": used_context_chars,
                    "hypothesis_idx": hypothesis_idx,
                    "generation_temperature": args.hypothesis_temperature,
                },
            )
            record = {
                "fact_id": example.example_idx,
                "context_id": example.context_id,
                "source_sample_idx": example.source_sample_idx,
                "modality": example.input_type,
                "hypothesis_idx": hypothesis_idx,
                "generation_temperature": args.hypothesis_temperature,
                "dimensions": dims,
                "dimensions_normalized": normalize_hypothesis_dimensions(dims, normalization_map),
                "operators": hypothesis.get("operators", []),
                "query_def": query_def,
                "query_lab": query_lab,
                "llm_call": llm_call,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            if output_idx % max(args.log_every, 1) == 0 or output_idx == len(raw_outputs):
                print(f"Generated structured hypotheses {output_idx}/{len(raw_outputs)}")
    return records


def build_retrievals(
    args: argparse.Namespace,
    examples_by_id: dict[int, Example],
    hypotheses: list[dict[str, Any]],
    retriever: TaxonomyRetriever,
) -> list[dict[str, Any]]:
    output_path = args.output_dir / "retrievals.jsonl"
    if args.resume and output_path.exists() and not args.overwrite:
        return load_jsonl(output_path)

    records: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, hypothesis in enumerate(hypotheses, start=1):
            example = examples_by_id[int(hypothesis["fact_id"])]
            query_def, def_candidates = retrieve_rendering(retriever, example, hypothesis["query_def"], args.top_k)
            query_lab, lab_candidates = retrieve_rendering(retriever, example, hypothesis["query_lab"], args.top_k)
            rendering_payloads = [
                ("def", query_def, def_candidates),
                ("lab", query_lab, lab_candidates),
                ("dual", None, dual_fuse(def_candidates, lab_candidates, args.top_k, args.dual_rrf_kappa)),
            ]
            for rendering, retrieval_query, candidates in rendering_payloads:
                ids = candidate_ids(candidates)
                record = {
                    "fact_id": hypothesis["fact_id"],
                    "context_id": hypothesis["context_id"],
                    "source_sample_idx": hypothesis["source_sample_idx"],
                    "modality": hypothesis["modality"],
                    "hypothesis_idx": hypothesis["hypothesis_idx"],
                    "rendering": rendering,
                    "query": retrieval_query,
                    "candidate_ids": ids,
                    "candidate_scores": candidate_scores(candidates),
                    "candidates": candidates,
                    "gold_concept_ids": example.gold_tags,
                    "gold_concept_id": example.gold_tags[0] if example.gold_tags else None,
                    "gold_rank": first_gold_rank(ids, example.gold_tags),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
            if idx % max(args.log_every, 1) == 0 or idx == len(hypotheses):
                print(f"Retrieved rendering variants {idx}/{len(hypotheses)}")
    return records


def retrievals_by_render_fact(retrievals: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in retrievals:
        grouped[str(record["rendering"])][int(record["fact_id"])].append(record)
    for records_by_fact in grouped.values():
        for fact_records in records_by_fact.values():
            fact_records.sort(key=lambda item: int(item["hypothesis_idx"]))
    return grouped


def mean_pairwise_jaccard(candidate_lists: list[list[str]], depth: int) -> float | None:
    if len(candidate_lists) < 2:
        return None
    values: list[float] = []
    for left_idx in range(len(candidate_lists)):
        left = set(candidate_lists[left_idx][:depth])
        for right_idx in range(left_idx + 1, len(candidate_lists)):
            right = set(candidate_lists[right_idx][:depth])
            union = left | right
            values.append(len(left & right) / len(union) if union else 0.0)
    return sum(values) / len(values) if values else None


def build_fused_from_records(records: list[dict[str, Any]], top_k: int, rrf_kappa: float) -> list[dict[str, Any]]:
    rounds = [
        {"round": idx + 1, "candidates": record.get("candidates", [])}
        for idx, record in enumerate(records)
    ]
    return fuse_round_candidates(rounds, top_k, rrf_kappa)


def compute_rendering_fact_metrics(
    retrievals: list[dict[str, Any]],
    examples: list[Example],
    depths: list[int],
    rrf_kappa: float,
    top_k: int,
) -> dict[str, dict[int, dict[int, dict[str, Any]]]]:
    grouped = retrievals_by_render_fact(retrievals)
    example_by_id = {example.example_idx: example for example in examples}
    metrics: dict[str, dict[int, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for rendering, records_by_fact in grouped.items():
        for fact_id, fact_records in records_by_fact.items():
            example = example_by_id[fact_id]
            candidate_lists = [record["candidate_ids"] for record in fact_records]
            fused = build_fused_from_records(fact_records, top_k, rrf_kappa)
            fused_ids = candidate_ids(fused)
            jaccard_200 = mean_pairwise_jaccard(candidate_lists, top_k)
            for depth in depths:
                union_ids: set[str] = set()
                for ids in candidate_lists:
                    union_ids.update(ids[:depth])
                hyp1_ids = candidate_lists[0] if candidate_lists else []
                hyp1_rank_full = first_gold_rank(hyp1_ids, example.gold_tags)
                rrf_rank_full = first_gold_rank(fused_ids, example.gold_tags)
                metrics[rendering][fact_id][depth] = {
                    "hyp1_recall": hyp1_rank_full is not None and hyp1_rank_full <= depth,
                    "hyp1_rank": hyp1_rank_full if hyp1_rank_full is not None and hyp1_rank_full <= depth else None,
                    "hyp1_mrr": 0.0
                    if hyp1_rank_full is None or hyp1_rank_full > depth
                    else 1.0 / hyp1_rank_full,
                    "coverage": bool(set(example.gold_tags) & union_ids),
                    "rrf_recall": rrf_rank_full is not None and rrf_rank_full <= depth,
                    "rrf_rank": rrf_rank_full if rrf_rank_full is not None and rrf_rank_full <= depth else None,
                    "rrf_mrr": 0.0
                    if rrf_rank_full is None or rrf_rank_full > depth
                    else 1.0 / rrf_rank_full,
                    "mean_jaccard_at_200": jaccard_200,
                    "hypothesis_count": len(candidate_lists),
                }
    return metrics


def aggregate_rendering_metrics(
    fact_metrics: dict[str, dict[int, dict[int, dict[str, Any]]]],
    examples: list[Example],
    depths: list[int],
) -> list[dict[str, Any]]:
    examples_by_id = {example.example_idx: example for example in examples}
    rows: list[dict[str, Any]] = []
    for rendering in RENDERINGS:
        for modality in MODALITIES:
            fact_ids = [fact_id for fact_id in selected_fact_ids(examples, modality) if fact_id in fact_metrics.get(rendering, {})]
            for depth in depths:
                values = [fact_metrics[rendering][fact_id][depth] for fact_id in fact_ids]
                n = len(values)
                jaccards = [value["mean_jaccard_at_200"] for value in values if value["mean_jaccard_at_200"] is not None]
                rows.append(
                    {
                        "stage": "A1",
                        "rendering": rendering,
                        "modality": modality,
                        "k": depth,
                        "fact_count": n,
                        "context_count": context_count_for_facts(examples_by_id, fact_ids),
                        "hyp1_recall_at_k": round(sum(value["hyp1_recall"] for value in values) / n, 6) if n else 0.0,
                        "hyp1_mrr_at_k": round(sum(float(value["hyp1_mrr"]) for value in values) / n, 6) if n else 0.0,
                        "coverage_at_3": round(sum(value["coverage"] for value in values) / n, 6) if n else 0.0,
                        "rrf_recall_at_k": round(sum(value["rrf_recall"] for value in values) / n, 6) if n else 0.0,
                        "rrf_mrr_at_k": round(sum(float(value["rrf_mrr"]) for value in values) / n, 6) if n else 0.0,
                        "mean_pairwise_jaccard_at_200": round(sum(jaccards) / len(jaccards), 6) if jaccards else None,
                    }
                )
    return rows


def paired_difference_rows(
    fact_metrics: dict[str, dict[int, dict[int, dict[str, Any]]]],
    examples: list[Example],
    depths: list[int],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    examples_by_id = {example.example_idx: example for example in examples}
    rows: list[dict[str, Any]] = []
    metrics = ("hyp1_recall", "hyp1_mrr", "coverage", "rrf_recall", "rrf_mrr")
    comparisons = {
        "lab_minus_def": ("lab", "def"),
        "dual_minus_def": ("dual", "def"),
    }
    for modality in MODALITIES:
        base_fact_ids = selected_fact_ids(examples, modality)
        for depth in depths:
            available = [
                fact_id
                for fact_id in base_fact_ids
                if all(fact_id in fact_metrics.get(rendering, {}) for rendering in RENDERINGS)
                and all(depth in fact_metrics[rendering][fact_id] for rendering in RENDERINGS)
            ]
            for comparison_idx, (comparison, (left, right)) in enumerate(comparisons.items()):
                for metric_idx, metric in enumerate(metrics):
                    values = {
                        fact_id: float(fact_metrics[left][fact_id][depth][metric])
                        - float(fact_metrics[right][fact_id][depth][metric])
                        for fact_id in available
                    }
                    ci = bootstrap_context_ci(
                        values,
                        examples_by_id,
                        available,
                        iterations=bootstrap_samples,
                        seed=bootstrap_seed + depth + 100 * comparison_idx + metric_idx,
                    )
                    rows.append(
                        {
                            "stage": "A1",
                            "comparison": comparison,
                            "metric": metric,
                            "modality": modality,
                            "k": depth,
                            **ci,
                        }
                    )
    return rows


def build_rendering_gate(args: argparse.Namespace, paired_rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = first_row(
        paired_rows,
        stage="A1",
        comparison="dual_minus_def",
        metric="hyp1_recall",
        modality="table",
        k=PRIMARY_GATE_K,
    )
    secondary = first_row(
        paired_rows,
        stage="A1",
        comparison="dual_minus_def",
        metric="hyp1_recall",
        modality="table",
        k=50,
    )
    lab_primary = first_row(
        paired_rows,
        stage="A1",
        comparison="lab_minus_def",
        metric="hyp1_recall",
        modality="table",
        k=PRIMARY_GATE_K,
    )
    if primary is None:
        status = "halt_requires_ack"
        reason = "missing_primary_gate_row"
        mean = 0.0
        ci_low = ci_high = 0.0
    else:
        mean = float(primary["mean"])
        ci_low = float(primary["ci_low"])
        ci_high = float(primary["ci_high"])
        if mean < 0.0:
            status = "halt_requires_ack"
            reason = "negative_dual_minus_def"
        elif mean >= args.gate_min_effect and ci_low > 0.0:
            status = "proceed_claim"
            reason = "dual_def_effect_ge_threshold_ci_excludes_zero"
        else:
            status = "proceed_weak"
            reason = "dual_kept_without_rendering_claim"
    return {
        "status": status,
        "reason": reason,
        "adopted_rendering": "dual" if status in {"proceed_claim", "proceed_weak"} else None,
        "claim_allowed": status == "proceed_claim",
        "human_ack_required": status == "halt_requires_ack",
        "primary_gate": primary,
        "secondary_k50_read": secondary,
        "lab_notable": bool(lab_primary and float(lab_primary.get("mean", 0.0)) >= 0.0),
        "lab_primary": lab_primary,
        "rule": (
            "table K=10 hyp1 recall: dual-def >= 0.03 and CI low > 0 -> proceed_claim; "
            "nonnegative weaker evidence -> proceed_weak; negative -> halt_requires_ack"
        ),
        "observed_mean": round(mean, 6),
        "observed_ci": [round(ci_low, 6), round(ci_high, 6)],
    }


def ranking_metrics_for_ids(ids: list[str], example: Example, depths: list[int]) -> dict[int, dict[str, Any]]:
    rank = first_gold_rank(ids, example.gold_tags)
    rows = {}
    for depth in depths:
        rows[depth] = {
            "recall": bool(rank is not None and rank <= depth),
            "rank": rank if rank is not None and rank <= depth else None,
            "mrr": 0.0 if rank is None or rank > depth else 1.0 / rank,
        }
    return rows


def aggregate_config_metrics(
    per_fact: dict[str, dict[int, dict[int, dict[str, Any]]]],
    examples: list[Example],
    depths: list[int],
    stage: str,
) -> list[dict[str, Any]]:
    examples_by_id = {example.example_idx: example for example in examples}
    rows: list[dict[str, Any]] = []
    for config, fact_rows in per_fact.items():
        for modality in MODALITIES:
            fact_ids = [fact_id for fact_id in selected_fact_ids(examples, modality) if fact_id in fact_rows]
            for depth in depths:
                values = [fact_rows[fact_id][depth] for fact_id in fact_ids]
                n = len(values)
                rows.append(
                    {
                        "stage": stage,
                        "config": config,
                        "modality": modality,
                        "k": depth,
                        "fact_count": n,
                        "context_count": context_count_for_facts(examples_by_id, fact_ids),
                        "recall_at_k": round(sum(value["recall"] for value in values) / n, 6) if n else 0.0,
                        "mrr_at_k": round(sum(float(value["mrr"]) for value in values) / n, 6) if n else 0.0,
                    }
                )
    return rows


def compute_a2_initialization(
    args: argparse.Namespace,
    hypotheses: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    examples: list[Example],
    depths: list[int],
    adopted_rendering: str,
    normalization_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples_by_id = {example.example_idx: example for example in examples}
    hyp_by_fact: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for hypothesis in hypotheses:
        hyp_by_fact[int(hypothesis["fact_id"])].append(hypothesis)
    for values in hyp_by_fact.values():
        values.sort(key=lambda item: int(item["hypothesis_idx"]))

    render_records = retrievals_by_render_fact([record for record in retrievals if record["rendering"] == adopted_rendering])[adopted_rendering]
    per_fact: dict[str, dict[int, dict[int, dict[str, Any]]]] = defaultdict(dict)
    symbolic_choices: dict[int, int] = {}
    rng = random.Random(args.random_init_seed)

    for example in examples:
        fact_id = example.example_idx
        records = render_records[fact_id]
        hypotheses_for_fact = hyp_by_fact[fact_id]
        per_fact["j1_hyp0"][fact_id] = ranking_metrics_for_ids(records[0]["candidate_ids"], example, depths)

        random_depth_values = {depth: {"recall": 0.0, "mrr": 0.0} for depth in depths}
        for _ in range(args.random_init_draws):
            selected = rng.randrange(len(records))
            metrics = ranking_metrics_for_ids(records[selected]["candidate_ids"], example, depths)
            for depth in depths:
                random_depth_values[depth]["recall"] += float(metrics[depth]["recall"])
                random_depth_values[depth]["mrr"] += float(metrics[depth]["mrr"])
        per_fact["j3_random_one_avg100"][fact_id] = {
            depth: {
                "recall": random_depth_values[depth]["recall"] / args.random_init_draws,
                "rank": None,
                "mrr": random_depth_values[depth]["mrr"] / args.random_init_draws,
            }
            for depth in depths
        }

        scores = [
            mean_agreement(record["candidates"], hypotheses_for_fact[idx]["dimensions"], args.agreement_top_m, normalization_map)
            for idx, record in enumerate(records)
        ]
        symbolic_idx = max(range(len(scores)), key=lambda idx: (scores[idx], -idx))
        symbolic_choices[fact_id] = symbolic_idx
        per_fact["j3_symbolic_select"][fact_id] = ranking_metrics_for_ids(records[symbolic_idx]["candidate_ids"], example, depths)

        fused = build_fused_from_records(records, args.top_k, args.rrf_kappa)
        per_fact["j3_union_rrf"][fact_id] = ranking_metrics_for_ids(candidate_ids(fused), example, depths)

    rows = aggregate_config_metrics(per_fact, examples, depths, "A2")
    examples_by_id = {example.example_idx: example for example in examples}
    paired_values = {}
    for modality in MODALITIES:
        fact_ids = selected_fact_ids(examples, modality)
        for depth in depths:
            values = {
                fact_id: float(per_fact["j3_symbolic_select"][fact_id][depth]["recall"])
                - float(per_fact["j3_random_one_avg100"][fact_id][depth]["recall"])
                for fact_id in fact_ids
            }
            paired_values[f"symbolic_minus_random_recall_{modality}_{depth}"] = bootstrap_context_ci(
                values,
                examples_by_id,
                fact_ids,
                iterations=args.bootstrap_samples,
                seed=args.bootstrap_seed + 500 + depth,
            )

    selection_k = args.init_selection_k
    pooled_rows = [row for row in rows if row["modality"] == "pooled" and int(row["k"]) == selection_k]
    winner = max(pooled_rows, key=lambda row: (float(row["recall_at_k"]), float(row["mrr_at_k"]), row["config"]))
    recommendation = {
        "adopted_initialization": winner["config"],
        "selection_k": selection_k,
        "selection_basis": "highest pooled recall_at_k, mrr tie-break",
        "symbolic_select_vs_random": paired_values,
        "symbolic_choices": symbolic_choices,
    }
    return rows, recommendation


def minmax_scores(candidates: list[dict[str, Any]]) -> dict[str, float]:
    raw_scores = {
        normalize_tag(candidate.get("tag", "")): float(
            candidate.get("bm25_score")
            if candidate.get("bm25_score") is not None
            else candidate.get("rrf_score")
            if candidate.get("rrf_score") is not None
            else 0.0
        )
        for candidate in candidates
    }
    if not raw_scores:
        return {}
    lo = min(raw_scores.values())
    hi = max(raw_scores.values())
    if hi <= lo:
        return {tag: 1.0 for tag in raw_scores}
    return {tag: (score - lo) / (hi - lo) for tag, score in raw_scores.items()}


def combsum(records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, dict[str, Any]] = {}
    first_seen: dict[str, int] = {}
    for list_idx, record in enumerate(records):
        normalized = minmax_scores(record["candidates"])
        for candidate in record["candidates"]:
            tag = normalize_tag(candidate.get("tag", ""))
            if not tag:
                continue
            scores[tag] += normalized.get(tag, 0.0)
            best.setdefault(tag, dict(candidate))
            first_seen.setdefault(tag, list_idx)
    ranked = sorted(scores, key=lambda tag: (-scores[tag], first_seen[tag], tag))[:top_k]
    output = []
    for rank, tag in enumerate(ranked, start=1):
        candidate = dict(best[tag])
        candidate["rank"] = rank
        candidate["combsum_score"] = round(scores[tag], 8)
        output.append(candidate)
    return output


def rrf_consensus(
    records: list[dict[str, Any]],
    hypotheses: list[dict[str, Any]],
    top_k: int,
    rrf_kappa: float,
    beta: float,
    normalization_map: dict[str, Any],
) -> list[dict[str, Any]]:
    base = build_fused_from_records(records, top_k, rrf_kappa)
    rescored = []
    for candidate in base:
        consensus = consensus_agreement(candidate, hypotheses, normalization_map)
        score = float(candidate.get("rrf_score", 0.0)) + beta * consensus
        updated = dict(candidate)
        updated["consensus_agreement"] = consensus
        updated["consensus_beta"] = beta
        updated["rerank_score"] = round(score, 8)
        rescored.append(updated)
    rescored.sort(key=lambda candidate: (-float(candidate.get("rerank_score", 0.0)), int(candidate.get("rank", 10**9)), candidate.get("tag", "")))
    for rank, candidate in enumerate(rescored[:top_k], start=1):
        candidate["rank"] = rank
    return rescored[:top_k]


def coverage_ceiling(records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for record in records:
        for candidate in record["candidates"]:
            tag = normalize_tag(candidate.get("tag", ""))
            if tag and tag not in seen:
                seen.add(tag)
                updated = dict(candidate)
                updated["rank"] = len(output) + 1
                output.append(updated)
            if len(output) >= top_k:
                return output
    return output


def compute_a3_consolidation(
    args: argparse.Namespace,
    hypotheses: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    examples: list[Example],
    depths: list[int],
    adopted_rendering: str,
    normalization_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hyp_by_fact: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for hypothesis in hypotheses:
        hyp_by_fact[int(hypothesis["fact_id"])].append(hypothesis)
    for values in hyp_by_fact.values():
        values.sort(key=lambda item: int(item["hypothesis_idx"]))
    render_records = retrievals_by_render_fact([record for record in retrievals if record["rendering"] == adopted_rendering])[adopted_rendering]
    betas = parse_float_list(args.consensus_betas)
    per_fact: dict[str, dict[int, dict[int, dict[str, Any]]]] = defaultdict(dict)
    detail_rows: list[dict[str, Any]] = []
    for example in examples:
        fact_id = example.example_idx
        records = render_records[fact_id]
        hypotheses_for_fact = hyp_by_fact[fact_id]
        variants = {
            "plain_rrf": build_fused_from_records(records, args.top_k, args.rrf_kappa),
            "score_normalized_combsum": combsum(records, args.top_k),
            "coverage_ceiling": coverage_ceiling(records, args.top_k),
        }
        for beta in betas:
            variants[f"rrf_consensus_beta_{beta:g}"] = rrf_consensus(
                records,
                hypotheses_for_fact,
                args.top_k,
                args.rrf_kappa,
                beta,
                normalization_map,
            )
        for variant, ranking in variants.items():
            per_fact[variant][fact_id] = ranking_metrics_for_ids(candidate_ids(ranking), example, depths)
            detail_rows.append(
                {
                    "fact_id": fact_id,
                    "context_id": example.context_id,
                    "modality": example.input_type,
                    "variant": variant,
                    "gold_rank": first_gold_rank(candidate_ids(ranking), example.gold_tags),
                    "candidate_ids": candidate_ids(ranking),
                    "candidate_scores": candidate_scores(ranking),
                }
            )
    rows = aggregate_config_metrics(per_fact, examples, depths, "A3")
    ceiling_by_key = {
        (row["modality"], row["k"]): row
        for row in rows
        if row["config"] == "coverage_ceiling"
    }
    for row in rows:
        ceiling = ceiling_by_key.get((row["modality"], row["k"]))
        if ceiling and row["config"] != "coverage_ceiling":
            row["residual_gap_to_ceiling"] = round(float(ceiling["recall_at_k"]) - float(row["recall_at_k"]), 6)
        else:
            row["residual_gap_to_ceiling"] = 0.0
    return rows, detail_rows


def write_agreement_debug(
    args: argparse.Namespace,
    hypotheses: list[dict[str, Any]],
    retrievals: list[dict[str, Any]],
    adopted_rendering: str,
    normalization_map: dict[str, Any],
) -> None:
    output_path = args.output_dir / "agreement_debug.jsonl"
    if args.resume and output_path.exists() and not args.overwrite:
        return
    rng = random.Random(args.agreement_debug_seed)
    fact_ids = sorted({int(record["fact_id"]) for record in retrievals})
    selected = set(rng.sample(fact_ids, min(args.agreement_debug_facts, len(fact_ids))))
    hyp_by_key = {(int(h["fact_id"]), int(h["hypothesis_idx"])): h for h in hypotheses}
    rows = []
    for record in retrievals:
        if int(record["fact_id"]) not in selected or record["rendering"] != adopted_rendering:
            continue
        hypothesis = hyp_by_key[(int(record["fact_id"]), int(record["hypothesis_idx"]))]
        for candidate in record["candidates"][: args.agreement_top_m]:
            result = agree(candidate, hypothesis["dimensions"], normalization_map)
            rows.append(
                {
                    "fact_id": record["fact_id"],
                    "context_id": record["context_id"],
                    "modality": record["modality"],
                    "hypothesis_idx": record["hypothesis_idx"],
                    "rendering": adopted_rendering,
                    "candidate_rank": candidate.get("rank"),
                    "candidate_id": candidate.get("tag"),
                    "candidate_label": candidate.get("standard_label"),
                    "agreement_score": result.score,
                    "matched": result.matched,
                    "evaluated": result.evaluated,
                    "verdicts": result.verdicts,
                }
            )
    write_jsonl(output_path, rows)


def write_metrics(
    args: argparse.Namespace,
    *,
    sample_summary: dict[str, Any],
    tokenization_check: dict[str, Any],
    report_rows: list[dict[str, Any]] | None = None,
    paired_rows: list[dict[str, Any]] | None = None,
    rendering_gate: dict[str, Any] | None = None,
    initialization_rows: list[dict[str, Any]] | None = None,
    initialization_recommendation: dict[str, Any] | None = None,
    consolidation_rows: list[dict[str, Any]] | None = None,
    aborted: bool = False,
    abort_stage: str | None = None,
) -> None:
    metrics = {
        "experiment": "ags_component_validation",
        "aborted": aborted,
        "abort_stage": abort_stage,
        "artifact_paths": {
            "sample_facts": str(args.sample_path),
            "hypotheses": str(args.output_dir / "hypotheses.jsonl"),
            "retrievals": str(args.output_dir / "retrievals.jsonl"),
            "agreement_debug": str(args.output_dir / "agreement_debug.jsonl"),
            "report_table": str(args.output_dir / "report_table.csv"),
            "paired_differences": str(args.output_dir / "paired_differences.csv"),
            "initialization_table": str(args.output_dir / "initialization_table.csv"),
            "consolidation_sweep": str(args.output_dir / "consolidation_sweep.csv"),
        },
        "sample_summary": sample_summary,
        "top_k": args.top_k,
        "depths": parse_depths(args.depths, args.top_k),
        "hypotheses_per_fact": args.hypotheses_per_fact,
        "hypothesis_temperature": args.hypothesis_temperature,
        "rrf_kappa": args.rrf_kappa,
        "dual_rrf_kappa": args.dual_rrf_kappa,
        "normalization_map": {
            "path": str(args.normalization_map),
            "version": map_version(args.normalization_map),
        },
        "tokenization_check": tokenization_check,
        "rendering_gate": rendering_gate,
        "initialization_recommendation": initialization_recommendation,
        "report_table": report_rows or [],
        "paired_differences": paired_rows or [],
        "initialization_table": initialization_rows or [],
        "consolidation_sweep": consolidation_rows or [],
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sample_summary_from_rows(sample_rows: list[dict[str, Any]]) -> dict[str, Any]:
    context_keys = {(row.get("input_type"), int(row.get("source_sample_idx", -1))) for row in sample_rows}
    return {
        "fact_count": len(sample_rows),
        "context_count": len(context_keys),
        "context_type_counts": {
            input_type: sum(1 for key in context_keys if key[0] == input_type)
            for input_type in ("table", "text")
        },
        "fact_type_counts": {
            input_type: sum(1 for row in sample_rows if row.get("input_type") == input_type)
            for input_type in ("table", "text")
        },
        "source": str(sample_rows[0].get("split", "train")) if sample_rows else "unknown",
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    depths = parse_depths(args.depths, args.top_k)
    if PRIMARY_GATE_K not in depths:
        raise ValueError(f"--depths must include primary gate K={PRIMARY_GATE_K}")
    if 50 not in depths:
        raise ValueError("--depths must include K=50 for the A1 read gate")
    if args.init_selection_k not in depths:
        raise ValueError("--init-selection-k must be included in --depths")

    if not args.sample_path.exists():
        raise FileNotFoundError(
            f"Frozen coverage-pilot sample not found: {args.sample_path}. Run the coverage pilot first."
        )
    sample_rows = load_jsonl(args.sample_path)
    examples = [row_to_example(row) for row in sample_rows]
    examples_by_id = {example.example_idx: example for example in examples}
    sample_summary = sample_summary_from_rows(sample_rows)

    taxonomy = load_taxonomy(args.taxonomy_jsonl)
    taxonomy_by_tag = {concept.tag: concept for concept in taxonomy}
    retriever = TaxonomyRetriever(taxonomy, type_filter=args.type_filter)
    normalization_map = load_normalization_map(args.normalization_map)

    tokenization_check = run_tokenization_self_check(args, examples, retriever, taxonomy_by_tag)
    (args.output_dir / "tokenization_check.json").write_text(
        json.dumps(tokenization_check, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not tokenization_check["passed"]:
        write_metrics(
            args,
            sample_summary=sample_summary,
            tokenization_check=tokenization_check,
            aborted=True,
            abort_stage="tokenization_self_check",
        )
        raise SystemExit("Tokenization self-check failed; metrics.json contains diagnostics.")

    generator = None
    if not args.dry_run_no_llm:
        generator = QueryGenerator(args)
    try:
        hypotheses = build_structured_hypotheses(args, examples, generator, normalization_map)
    finally:
        if generator is not None:
            generator.close()

    retrievals = build_retrievals(args, examples_by_id, hypotheses, retriever)
    fact_metrics = compute_rendering_fact_metrics(retrievals, examples, depths, args.rrf_kappa, args.top_k)
    report_rows = aggregate_rendering_metrics(fact_metrics, examples, depths)
    paired_rows = paired_difference_rows(
        fact_metrics,
        examples,
        depths,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    rendering_gate = build_rendering_gate(args, paired_rows)

    write_csv(args.output_dir / "report_table.csv", report_rows)
    write_csv(args.output_dir / "paired_differences.csv", paired_rows)

    if rendering_gate["status"] == "halt_requires_ack":
        write_metrics(
            args,
            sample_summary=sample_summary,
            tokenization_check=tokenization_check,
            report_rows=report_rows,
            paired_rows=paired_rows,
            rendering_gate=rendering_gate,
            aborted=True,
            abort_stage="rendering_gate_negative",
        )
        raise SystemExit("A1 rendering gate is negative; explicit human acknowledgment is required before B.")

    adopted_rendering = str(rendering_gate["adopted_rendering"])
    write_agreement_debug(args, hypotheses, retrievals, adopted_rendering, normalization_map)
    initialization_rows, initialization_recommendation = compute_a2_initialization(
        args,
        hypotheses,
        retrievals,
        examples,
        depths,
        adopted_rendering,
        normalization_map,
    )
    consolidation_rows, consolidation_detail_rows = compute_a3_consolidation(
        args,
        hypotheses,
        retrievals,
        examples,
        depths,
        adopted_rendering,
        normalization_map,
    )
    write_csv(args.output_dir / "initialization_table.csv", initialization_rows)
    write_csv(args.output_dir / "consolidation_sweep.csv", consolidation_rows)
    write_jsonl(args.output_dir / "consolidation_rankings.jsonl", consolidation_detail_rows)
    write_metrics(
        args,
        sample_summary=sample_summary,
        tokenization_check=tokenization_check,
        report_rows=report_rows,
        paired_rows=paired_rows,
        rendering_gate=rendering_gate,
        initialization_rows=initialization_rows,
        initialization_recommendation=initialization_recommendation,
        consolidation_rows=consolidation_rows,
    )
    print(json.dumps({"metrics_path": str(args.output_dir / "metrics.json"), "rendering_gate": rendering_gate}, indent=2))


if __name__ == "__main__":
    main()

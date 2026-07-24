#!/usr/bin/env python3
"""Task A: specialized generators vs. stochastic sampling, on the development sample.

Does replacing J stochastic samples of one prompt with J functionally specialized
generators improve the AGS ensemble? This is a configuration decision, so it runs on the
frozen 661-fact / 70-context development sample, not on test.

One generation pass, then everything else offline on the logged retrievals:

  spec:G0..spec:G4    five specialists, deterministic decoding      3,305 calls
  stoch:0..stoch:4    five stochastic samples of G0 at T=0.8        3,305 calls

S0 (the frozen J=2 baseline) is the first two stochastic samples, so no arm needs a
generation pass of its own. Arms:

  S0  stoch:0..1                  frozen AGS baseline
  S1  spec:G0..G2                 3 generators
  S2  spec:G0..G4                 5 generators
  S3  S2 restricted to generators passing the symbolic compatibility filter
  S4  stoch:0..4                  cost-matched control for S2
  S5  per-fact oracle over the five specialists' solo rankings

S4 is the arm that matters. Without it a gain at S2 is confounded with simply having five
hypotheses instead of two, so S2 must beat S4 and not just S0.

Beta is re-swept per arm because ensemble size changes the number of fused rankings and
so the fused-score distribution; arms are compared at their respective optima. Every
consolidation step is the frozen AGS one -- sum-RRF at kappa=60 over every rendering
ranking, truncate to K, range-normalize, add beta * consensus -- reused through
ags_specialized_generators, which test_ags_specialized_generators pins to
ags_frozen_grounding.frozen_ags_rerank.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ags_specialized_generators import (
    COMPATIBILITY_CHECKS,
    GENERATOR_KEYS,
    GENERATOR_NAMES,
    build_agreement_matrix,
    build_generator_messages,
    candidate_index,
    compatibility_verdict,
    consensus_for_slots,
    fuse_and_normalize,
    rerank_order,
    rerank_share,
    resolved_dimension_count,
)
from ags_symbolic_agreement import (
    DEFAULT_NORMALIZATION_MAP,
    load_normalization_map,
    map_version,
    mean_agreement,
    normalize_hypothesis_dimensions,
)
from run_ags_component_validation import (
    dimensions_lower,
    fake_structured_output,
    render_definition,
    render_label,
    retrieve_rendering,
)
from run_ags_coverage_pilot import (
    bootstrap_context_ci,
    context_count_for_facts,
    load_jsonl,
    prompt_under_budget,
    row_to_example,
    selected_fact_ids,
    temporary_generation_settings,
    write_csv,
)
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    SCRIPT_DIR,
    Example,
    QueryGenerator,
    TaxonomyRetriever,
    build_direct_query,
    first_gold_rank,
    llm_call_record,
    load_taxonomy,
    normalize_tag,
    parse_hypothesis,
    write_jsonl,
)


MODALITIES = ("pooled", "table", "text")
METRICS = ("recall_at_10", "recall_at_50", "recall_at_200", "mrr")
PRIMARY_MODALITY = "table"
SELECTION_METRIC = "recall_at_10"

SPEC_SLOTS = tuple(f"spec:{key}" for key in GENERATOR_KEYS)
STOCH_SLOT_COUNT = 5
STOCH_SLOTS = tuple(f"stoch:{idx}" for idx in range(STOCH_SLOT_COUNT))
ALL_SLOTS = SPEC_SLOTS + STOCH_SLOTS

# Fixed arm membership. S3 and S5 are resolved per fact, so they carry no static slot list.
STATIC_ARMS = {
    "S0": STOCH_SLOTS[:2],
    "S1": SPEC_SLOTS[:3],
    "S2": SPEC_SLOTS,
    "S4": STOCH_SLOTS,
}
ARM_ORDER = ("S0", "S1", "S2", "S3", "S4", "S5")
ARM_NAMES = {
    "S0": "J=2 stochastic samples of G0 (frozen AGS baseline)",
    "S1": "G0 + G1 + G2 (3 generators)",
    "S2": "G0 + G1 + G2 + G3 + G4 (5 generators)",
    "S3": "S2 restricted by the symbolic compatibility filter",
    "S4": "J=5 stochastic samples of G0 (cost-matched control for S2)",
    "S5": "per-fact oracle over the five specialists' solo rankings",
}

# Spec section 5: the four primary reads, on the table subset.
PRIMARY_CONTRASTS = (
    ("S2_minus_S0", "S2", "S0", "does specialization beat the frozen baseline?"),
    ("S2_minus_S4", "S2", "S4", "does specialization beat cost-matched stochastic sampling? the real test"),
    ("S1_minus_S0", "S1", "S0", "does it work at 3 generators, i.e. cheaper?"),
    ("S2_minus_S5", "S2", "S5", "how much of the best-single-generator oracle does fusion capture?"),
)

# Spec section 7, recorded in metrics.json so the run carries its own prior.
PRIOR_WARNING = {
    "summary": (
        "The coverage pilot measured two interventions on hypothesis diversity and both "
        "reduced coverage: a diversity-directed prompt reached 0.664 and a per-dimension "
        "assignment 0.682, against 0.735 for plain stochastic sampling at table K=200. In "
        "each case single-hypothesis recall fell 5-8 points -- diversity was purchased by "
        "degrading each hypothesis."
    ),
    "why_this_run_may_differ": (
        "Specialization is not the same manipulation: each generator is asked for its best "
        "reading rather than a different one."
    ),
    "expected_outcome": (
        "It is the same family of intervention, so a null or negative result is the expected "
        "outcome and not a bug in the run."
    ),
    "coverage_pilot_reference": {
        "diversity_directed_prompt": 0.664,
        "per_dimension_assignment": 0.682,
        "plain_stochastic_sampling": 0.735,
        "measured_at": "table K=200 coverage",
    },
}

# Frozen AGS rendering policy (spec section 1): dual for table, definition-only for text.
FROZEN_RENDERING_POLICY = {"table": "dual", "text": "def"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs_ags_specialized_generators" / "qwen3_32b")
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=SCRIPT_DIR / "runs_ags_coverage_pilot" / "qwen3_32b" / "sample_facts.jsonl",
    )
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--agreement-top-m", type=int, default=10)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.5])
    parser.add_argument(
        "--reference-beta",
        type=float,
        default=0.0,
        help="Extra beta swept as a no-rerank reference; not eligible as an arm optimum.",
    )
    parser.add_argument("--label-coverage-weight", type=float, default=1.0)
    parser.add_argument("--label-coverage-pool-multiplier", type=int, default=0)
    parser.add_argument("--type-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260724)
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.01,
        help="|S1 - S2| below this, with a CI spanning zero, reads as 'the extra two add nothing'.",
    )
    parser.add_argument(
        "--dominance-margin",
        type=float,
        default=0.03,
        help="Solo lead of the best generator over the second best that reads as dominance.",
    )
    parser.add_argument(
        "--contribution-floor",
        type=float,
        default=0.10,
        help="Selection frequency under agree() below which a generator counts as not contributing.",
    )
    parser.add_argument("--limit-facts", type=int, default=0, help="Debug: score only the first N facts.")

    parser.add_argument("--specialist-temperature", type=float, default=0.0)
    parser.add_argument("--stochastic-temperature", type=float, default=0.8)
    parser.add_argument("--query-generation-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--query-generation-backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--query-context-max-chars", type=int, default=12000)
    parser.add_argument("--query-max-input-tokens", type=int, default=16000)
    parser.add_argument("--query-max-new-tokens", type=int, default=512)
    parser.add_argument("--query-top-p", type=float, default=1.0)
    parser.add_argument("--query-temperature", type=float, default=0.0)
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

    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-retrievals", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run-no-llm", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def slot_generator_key(slot: str) -> str:
    """Which prompt a slot uses. Every stochastic slot is a sample of G0."""
    kind, value = slot.split(":", 1)
    return value if kind == "spec" else "G0"


def slot_temperature(slot: str, args: argparse.Namespace) -> float:
    return args.specialist_temperature if slot.startswith("spec:") else args.stochastic_temperature


def renderings_for(example: Example) -> tuple[str, ...]:
    """def always; lab additionally for the dual-rendering modality (table)."""
    return ("def", "lab") if FROZEN_RENDERING_POLICY.get(example.input_type) == "dual" else ("def",)


# --------------------------------------------------------------------------------------
# Stage 1: generation (the only GPU stage)
# --------------------------------------------------------------------------------------


def build_hypotheses(
    args: argparse.Namespace,
    examples: list[Example],
    generator: QueryGenerator | None,
    normalization_map: dict[str, Any],
) -> list[dict[str, Any]]:
    output_path = args.output_dir / "hypotheses.jsonl"
    if args.resume and output_path.exists() and not args.overwrite:
        print(f"Reusing hypotheses from {output_path}")
        return load_jsonl(output_path)

    # Prompts are built once per (fact, generator key) and reused by every slot that uses
    # them, so the five stochastic samples are five draws from one identical prompt.
    prompt_cache: dict[tuple[int, str], tuple[str, int, int]] = {}
    for example in examples:
        for generator_key in GENERATOR_KEYS:
            prompt_cache[(example.example_idx, generator_key)] = prompt_under_budget(
                generator,
                args,
                lambda ctx_chars, ex=example, key=generator_key: build_generator_messages(key, ex, ctx_chars),
            )

    records: list[dict[str, Any]] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = output_path.open("w", encoding="utf-8")
    try:
        # One generation block per temperature: specialists are deterministic, the
        # stochastic control is not, and the two cannot share a sampling setting.
        for slots, temperature, sampling in (
            (SPEC_SLOTS, args.specialist_temperature, "deterministic"),
            (STOCH_SLOTS, args.stochastic_temperature, "stochastic"),
        ):
            prompts: list[str] = []
            meta: list[tuple[Example, str, int, int]] = []
            for example in examples:
                for slot in slots:
                    prompt, prompt_tokens, used_context_chars = prompt_cache[
                        (example.example_idx, slot_generator_key(slot))
                    ]
                    prompts.append(prompt)
                    meta.append((example, slot, prompt_tokens, used_context_chars))

            if args.dry_run_no_llm:
                raw_outputs = [
                    fake_structured_output(example, ALL_SLOTS.index(slot))
                    for example, slot, _, _ in meta
                ]
            else:
                assert generator is not None
                with temporary_generation_settings(
                    generator,
                    temperature,
                    args.query_top_p,
                    args.query_max_new_tokens,
                ):
                    print(f"Generating {len(prompts)} {sampling} hypotheses at temperature {temperature}")
                    raw_outputs = generator.generate_many(prompts)

            for output_idx, (raw_output, (example, slot, prompt_tokens, used_context_chars)) in enumerate(
                zip(raw_outputs, meta, strict=True),
                start=1,
            ):
                fallback = build_direct_query(example)
                hypothesis, parse_ok = parse_hypothesis(raw_output, fallback)
                dims = dimensions_lower(hypothesis.get("dimensions", {}))
                record = {
                    "fact_id": example.example_idx,
                    "context_id": example.context_id,
                    "source_sample_idx": example.source_sample_idx,
                    "modality": example.input_type,
                    "entity_type": example.entity_type,
                    "slot": slot,
                    "generator": slot_generator_key(slot),
                    "generator_name": GENERATOR_NAMES[slot_generator_key(slot)],
                    "sampling": sampling,
                    "generation_temperature": temperature,
                    "dimensions": dims,
                    "dimensions_normalized": normalize_hypothesis_dimensions(dims, normalization_map),
                    "resolved_dimension_count": resolved_dimension_count(dims),
                    "operators": hypothesis.get("operators", []),
                    "retrieval_query": hypothesis.get("retrieval_query", ""),
                    "query_def": render_definition(hypothesis),
                    "query_lab": render_label(hypothesis),
                    "llm_call": llm_call_record(
                        "specialized_generator_hypothesis",
                        raw_output=raw_output,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=0 if generator is None else generator.count_text_tokens(raw_output),
                        parse_ok=parse_ok,
                        backend="dry_run" if generator is None else generator.backend,
                        model_name="dry_run" if generator is None else generator.model_name,
                        extra_fields={
                            "slot": slot,
                            "used_context_max_chars": used_context_chars,
                            "generation_temperature": temperature,
                        },
                    ),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
                if output_idx % max(args.log_every, 1) == 0 or output_idx == len(raw_outputs):
                    print(f"Parsed {sampling} hypotheses {output_idx}/{len(raw_outputs)}")
    finally:
        handle.close()
    return records


# --------------------------------------------------------------------------------------
# Stage 2: retrieval (CPU; BM25 over the enriched US-GAAP index)
# --------------------------------------------------------------------------------------


def retrievals_are_reusable(args: argparse.Namespace) -> bool:
    """A rerun that keeps its logged retrievals never touches the BM25 index."""
    path = args.output_dir / "retrievals.jsonl"
    return bool(args.resume and path.exists() and not args.overwrite and not args.refresh_retrievals)


def build_retrievals(
    args: argparse.Namespace,
    examples_by_id: dict[int, Example],
    hypotheses: list[dict[str, Any]],
    retriever: TaxonomyRetriever | None,
) -> Path:
    output_path = args.output_dir / "retrievals.jsonl"
    if retrievals_are_reusable(args):
        print(f"Reusing retrievals from {output_path}")
        return output_path
    assert retriever is not None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, hypothesis in enumerate(hypotheses, start=1):
            example = examples_by_id[int(hypothesis["fact_id"])]
            for rendering in renderings_for(example):
                rendered = hypothesis["query_def"] if rendering == "def" else hypothesis["query_lab"]
                if rendering == "lab" and not rendered:
                    # frozen_ags_rankings drops the label ranking when render_label is empty.
                    continue
                query, candidates = retrieve_rendering(retriever, example, rendered, args.top_k)
                ids = [normalize_tag(candidate["tag"]) for candidate in candidates]
                handle.write(
                    json.dumps(
                        {
                            "fact_id": hypothesis["fact_id"],
                            "context_id": hypothesis["context_id"],
                            "source_sample_idx": hypothesis["source_sample_idx"],
                            "modality": hypothesis["modality"],
                            "slot": hypothesis["slot"],
                            "generator": hypothesis["generator"],
                            "rendering": rendering,
                            "query": query,
                            "candidate_ids": ids,
                            "candidates": candidates,
                            "gold_concept_ids": example.gold_tags,
                            "gold_rank": first_gold_rank(ids, example.gold_tags),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            if idx % max(args.log_every, 1) == 0 or idx == len(hypotheses):
                print(f"Retrieved {idx}/{len(hypotheses)} hypotheses")
    return output_path


def index_retrievals_by_fact(path: Path) -> dict[int, list[int]]:
    """Byte offsets grouped by fact, so scoring holds one fact in memory at a time."""
    offsets: dict[int, list[int]] = defaultdict(list)
    with path.open("rb") as handle:
        offset = handle.tell()
        for line in handle:
            if line.strip():
                offsets[int(json.loads(line)["fact_id"])].append(offset)
            offset += len(line)
    return dict(offsets)


# --------------------------------------------------------------------------------------
# Stage 3: scoring, offline over the logged retrievals
# --------------------------------------------------------------------------------------


def rank_metrics(rank: int | None) -> dict[str, float]:
    return {
        "recall_at_10": float(rank is not None and rank <= 10),
        "recall_at_50": float(rank is not None and rank <= 50),
        "recall_at_200": float(rank is not None and rank <= 200),
        "mrr": 0.0 if rank is None else 1.0 / rank,
    }


def score_fact(
    args: argparse.Namespace,
    example: Example,
    records: list[dict[str, Any]],
    hypotheses_by_slot: dict[str, dict[str, Any]],
    normalization_map: dict[str, Any],
    betas: tuple[float, ...],
) -> dict[str, Any]:
    """Every arm at every beta for one fact, off one parse of that fact's candidates."""
    records_by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_slot[str(record["slot"])].append(record)
    # Fusion is order-sensitive only through RRF tie-breaks; fix it to (slot, rendering).
    for slot_records in records_by_slot.values():
        slot_records.sort(key=lambda record: str(record["rendering"]))

    candidate_by_tag = candidate_index(records)
    dimensions_by_slot = {slot: hypotheses_by_slot[slot]["dimensions"] for slot in hypotheses_by_slot}
    agreement_matrix = build_agreement_matrix(candidate_by_tag, dimensions_by_slot, normalization_map)

    compatibility = {
        slot: compatibility_verdict(example.entity_type, dimensions, normalization_map)
        for slot, dimensions in dimensions_by_slot.items()
    }
    passing = [slot for slot in SPEC_SLOTS if compatibility[slot].passed]
    # A fact where every specialist is filtered out would have no ranking at all; fall
    # back to the unfiltered arm and count it rather than dropping the fact.
    s3_slots = tuple(passing) if passing else SPEC_SLOTS
    arm_slots = {**STATIC_ARMS, "S3": s3_slots}

    def rounds_for(slots: tuple[str, ...]) -> list[dict[str, Any]]:
        rounds: list[dict[str, Any]] = []
        for slot in slots:
            for record in records_by_slot.get(slot, []):
                rounds.append({"round": len(rounds) + 1, "candidates": record["candidates"]})
        return rounds

    ranks: dict[tuple[str, float], int | None] = {}
    shares: dict[tuple[str, float], float | None] = {}

    for arm, slots in arm_slots.items():
        pool = fuse_and_normalize(rounds_for(slots), args.top_k, args.rrf_kappa)
        consensus = consensus_for_slots(agreement_matrix, slots)
        for beta in betas:
            order = rerank_order(pool, consensus, beta, args.top_k)
            ranks[(arm, beta)] = first_gold_rank(order, example.gold_tags)
            shares[(arm, beta)] = rerank_share(pool, consensus, beta)

    # Solo rankings: each specialist scored alone, reused for S5 and for generator_solo.csv.
    solo_ranks: dict[tuple[str, float], int | None] = {}
    for slot in SPEC_SLOTS:
        pool = fuse_and_normalize(rounds_for((slot,)), args.top_k, args.rrf_kappa)
        consensus = consensus_for_slots(agreement_matrix, (slot,))
        for beta in betas:
            order = rerank_order(pool, consensus, beta, args.top_k)
            solo_ranks[(slot, beta)] = first_gold_rank(order, example.gold_tags)
    for beta in betas:
        observed = [solo_ranks[(slot, beta)] for slot in SPEC_SLOTS if solo_ranks[(slot, beta)] is not None]
        ranks[("S5", beta)] = min(observed) if observed else None
        shares[("S5", beta)] = None

    # Which specialist the verifier prefers, on its own retrieved neighborhood.
    selection_scores = {
        slot: mean_agreement(
            [candidate for record in records_by_slot.get(slot, []) for candidate in record["candidates"]],
            dimensions_by_slot[slot],
            args.agreement_top_m,
            normalization_map,
        )
        for slot in SPEC_SLOTS
    }
    selected_slot = max(SPEC_SLOTS, key=lambda slot: (selection_scores[slot], -SPEC_SLOTS.index(slot)))

    candidate_lists = {
        slot: [
            normalize_tag(tag)
            for record in records_by_slot.get(slot, [])
            if record["rendering"] == "def"
            for tag in record["candidate_ids"]
        ]
        for slot in SPEC_SLOTS
    }

    return {
        "fact_id": example.example_idx,
        "modality": example.input_type,
        "ranks": ranks,
        "shares": shares,
        "solo_ranks": solo_ranks,
        "selection_scores": selection_scores,
        "selected_slot": selected_slot,
        "compatibility": {slot: verdict for slot, verdict in compatibility.items()},
        "s3_slots": s3_slots,
        "s3_fell_back": not passing,
        "candidate_lists": candidate_lists,
        "resolved_counts": {
            slot: int(hypotheses_by_slot[slot]["resolved_dimension_count"]) for slot in hypotheses_by_slot
        },
    }


def jaccard(left: list[str], right: list[str], depth: int) -> float:
    left_set = set(left[:depth])
    right_set = set(right[:depth])
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


# --------------------------------------------------------------------------------------
# Stage 4: aggregation, sweep, contrasts, decision rules
# --------------------------------------------------------------------------------------


def aggregate_metric(
    fact_scores: list[dict[str, Any]],
    key: tuple[str, float],
    modality: str,
    metric: str,
    source: str = "ranks",
) -> float:
    values = [
        rank_metrics(score[source][key])[metric]
        for score in fact_scores
        if modality == "pooled" or score["modality"] == modality
    ]
    return sum(values) / len(values) if values else 0.0


def build_beta_sweep_rows(
    fact_scores: list[dict[str, Any]],
    examples_by_id: dict[int, Example],
    betas: tuple[float, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        for modality in MODALITIES:
            subset = [
                score for score in fact_scores if modality == "pooled" or score["modality"] == modality
            ]
            fact_ids = [int(score["fact_id"]) for score in subset]
            for beta in betas:
                shares = [
                    score["shares"][(arm, beta)]
                    for score in subset
                    if score["shares"][(arm, beta)] is not None
                ]
                row = {
                    "arm": arm,
                    "arm_name": ARM_NAMES[arm],
                    "modality": modality,
                    "beta": beta,
                    "fact_count": len(subset),
                    "context_count": context_count_for_facts(examples_by_id, fact_ids),
                }
                for metric in METRICS:
                    row[metric] = round(aggregate_metric(subset, (arm, beta), modality, metric), 6)
                row["rerank_share"] = round(sum(shares) / len(shares), 6) if shares else None
                row["rerank_share_defined_facts"] = len(shares)
                rows.append(row)
    return rows


def select_arm_betas(
    sweep_rows: list[dict[str, Any]],
    modality: str,
    eligible_betas: tuple[float, ...],
) -> dict[str, dict[str, Any]]:
    """Peak beta per arm on `modality`, by recall@10 with MRR as the tie-break."""
    selection: dict[str, dict[str, Any]] = {}
    for arm in ARM_ORDER:
        candidates = [
            row
            for row in sweep_rows
            if row["arm"] == arm and row["modality"] == modality and row["beta"] in eligible_betas
        ]
        winner = max(
            candidates,
            key=lambda row: (float(row[SELECTION_METRIC]), float(row["mrr"]), -float(row["beta"])),
        )
        selection[arm] = {
            "beta": float(winner["beta"]),
            "selected_on": f"{modality}/{SELECTION_METRIC}, mrr tie-break, lower beta wins ties",
            SELECTION_METRIC: winner[SELECTION_METRIC],
            "mrr": winner["mrr"],
            "rerank_share": winner["rerank_share"],
        }
    return selection


def contrast_rows(
    args: argparse.Namespace,
    fact_scores: list[dict[str, Any]],
    examples_by_id: dict[int, Example],
    arm_betas: dict[str, dict[str, Any]],
    selection_label: str,
) -> list[dict[str, Any]]:
    """Paired per-fact differences with bootstrap CIs resampled at the context level."""
    rows: list[dict[str, Any]] = []
    for contrast_idx, (name, left, right, question) in enumerate(PRIMARY_CONTRASTS):
        left_beta = arm_betas[left]["beta"]
        right_beta = arm_betas[right]["beta"]
        for modality in MODALITIES:
            subset = [
                score for score in fact_scores if modality == "pooled" or score["modality"] == modality
            ]
            fact_ids = [int(score["fact_id"]) for score in subset]
            for metric_idx, metric in enumerate(METRICS):
                values = {
                    int(score["fact_id"]): rank_metrics(score["ranks"][(left, left_beta)])[metric]
                    - rank_metrics(score["ranks"][(right, right_beta)])[metric]
                    for score in subset
                }
                ci = bootstrap_context_ci(
                    values,
                    examples_by_id,
                    fact_ids,
                    iterations=args.bootstrap_samples,
                    seed=args.bootstrap_seed + 1000 * contrast_idx + 10 * metric_idx + len(modality),
                )
                rows.append(
                    {
                        "contrast": name,
                        "question": question,
                        "beta_selection": selection_label,
                        "left_arm": left,
                        "left_beta": left_beta,
                        "right_arm": right,
                        "right_beta": right_beta,
                        "modality": modality,
                        "metric": metric,
                        "left_value": round(aggregate_metric(subset, (left, left_beta), modality, metric), 6),
                        "right_value": round(aggregate_metric(subset, (right, right_beta), modality, metric), 6),
                        **ci,
                    }
                )
    return rows


def equivalence_row(
    args: argparse.Namespace,
    fact_scores: list[dict[str, Any]],
    examples_by_id: dict[int, Example],
    arm_betas: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """S1 - S2 on the primary read, for the 'the extra two specialists add nothing' rule."""
    subset = [score for score in fact_scores if score["modality"] == PRIMARY_MODALITY]
    fact_ids = [int(score["fact_id"]) for score in subset]
    values = {
        int(score["fact_id"]): rank_metrics(score["ranks"][("S1", arm_betas["S1"]["beta"])])[SELECTION_METRIC]
        - rank_metrics(score["ranks"][("S2", arm_betas["S2"]["beta"])])[SELECTION_METRIC]
        for score in subset
    }
    ci = bootstrap_context_ci(
        values,
        examples_by_id,
        fact_ids,
        iterations=args.bootstrap_samples,
        seed=args.bootstrap_seed + 9000,
    )
    return {
        "contrast": "S1_minus_S2",
        "modality": PRIMARY_MODALITY,
        "metric": SELECTION_METRIC,
        "left_beta": arm_betas["S1"]["beta"],
        "right_beta": arm_betas["S2"]["beta"],
        **ci,
    }


def generator_solo_rows(
    fact_scores: list[dict[str, Any]],
    examples_by_id: dict[int, Example],
    solo_beta: float,
) -> list[dict[str, Any]]:
    """Per generator: solo performance, verifier preference, caution, and distinctness.

    Worth as much as the ensemble result -- if one generator is solo-strong and the rest
    never contribute, the finding is one better prompt, not specialization.
    """
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        subset = [
            score for score in fact_scores if modality == "pooled" or score["modality"] == modality
        ]
        if not subset:
            continue
        fact_ids = [int(score["fact_id"]) for score in subset]
        selection_counts = Counter(score["selected_slot"] for score in subset)
        for slot in SPEC_SLOTS:
            generator = slot_generator_key(slot)
            solo = [rank_metrics(score["solo_ranks"][(slot, solo_beta)]) for score in subset]
            others = [other for other in SPEC_SLOTS if other != slot]
            jaccard_200 = [
                jaccard(score["candidate_lists"][slot], score["candidate_lists"][other], 200)
                for score in subset
                for other in others
            ]
            jaccard_10 = [
                jaccard(score["candidate_lists"][slot], score["candidate_lists"][other], 10)
                for score in subset
                for other in others
            ]
            compat_pass = sum(1 for score in subset if score["compatibility"][slot].passed)
            row = {
                "generator": generator,
                "generator_name": GENERATOR_NAMES[generator],
                "slot": slot,
                "modality": modality,
                "beta": solo_beta,
                "fact_count": len(subset),
                "context_count": context_count_for_facts(examples_by_id, fact_ids),
            }
            for metric in METRICS:
                row[f"solo_{metric}"] = round(sum(value[metric] for value in solo) / len(solo), 6)
            row["selection_frequency_under_agree"] = round(selection_counts.get(slot, 0) / len(subset), 6)
            row["mean_selection_score"] = round(
                sum(score["selection_scores"][slot] for score in subset) / len(subset), 6
            )
            row["mean_resolved_dimensions"] = round(
                sum(score["resolved_counts"][slot] for score in subset) / len(subset), 6
            )
            row["mean_pairwise_jaccard_at_200"] = round(sum(jaccard_200) / len(jaccard_200), 6) if jaccard_200 else None
            row["mean_pairwise_jaccard_at_10"] = round(sum(jaccard_10) / len(jaccard_10), 6) if jaccard_10 else None
            row["compatibility_pass_rate"] = round(compat_pass / len(subset), 6)
            for check in COMPATIBILITY_CHECKS:
                row[f"compatibility_{check}_pass_rate"] = round(
                    sum(1 for score in subset if score["compatibility"][slot].checks[check]["passed"]) / len(subset),
                    6,
                )
            rows.append(row)

        # The stochastic control's mean resolved-dimension count is the abstention anchor.
        stoch_resolved = [
            score["resolved_counts"][slot] for score in subset for slot in STOCH_SLOTS
        ]
        rows.append(
            {
                "generator": "stochastic_G0",
                "generator_name": "stochastic samples of G0 (reference)",
                "slot": "stoch:*",
                "modality": modality,
                "beta": solo_beta,
                "fact_count": len(subset),
                "context_count": context_count_for_facts(examples_by_id, fact_ids),
                "mean_resolved_dimensions": round(sum(stoch_resolved) / len(stoch_resolved), 6),
            }
        )
    return rows


def abstention_check(solo_rows: list[dict[str, Any]], modality: str = "pooled") -> dict[str, Any]:
    """Spec section 6: specialists resolving MORE dimensions than G0 must be read first.

    They would be guessing outside their remit, and nothing else in the run means much
    until the abstention instruction is tightened and the generation pass re-run.
    """
    by_generator = {
        row["generator"]: row
        for row in solo_rows
        if row["modality"] == modality and "mean_resolved_dimensions" in row
    }
    baseline = by_generator.get("G0", {}).get("mean_resolved_dimensions")
    offenders = {
        generator: row["mean_resolved_dimensions"]
        for generator, row in by_generator.items()
        if generator in ("G1", "G2", "G3", "G4")
        and baseline is not None
        and row["mean_resolved_dimensions"] > baseline
    }
    return {
        "modality": modality,
        "g0_mean_resolved_dimensions": baseline,
        "stochastic_g0_mean_resolved_dimensions": by_generator.get("stochastic_G0", {}).get(
            "mean_resolved_dimensions"
        ),
        "specialist_mean_resolved_dimensions": {
            generator: row["mean_resolved_dimensions"]
            for generator, row in by_generator.items()
            if generator in ("G1", "G2", "G3", "G4")
        },
        "violating_generators": offenders,
        "violated": bool(offenders),
        "action_if_violated": (
            "Specialists are resolving more dimensions than G0, i.e. guessing outside their "
            "remit. Tighten the abstention instruction and re-run the generation pass before "
            "reading anything else in this run."
        ),
    }


def decision_rule_outcome(
    args: argparse.Namespace,
    contrasts: list[dict[str, Any]],
    equivalence: dict[str, Any],
    solo_rows: list[dict[str, Any]],
    abstention: dict[str, Any],
) -> dict[str, Any]:
    """Spec section 6, evaluated exactly as fixed in advance."""

    def primary(contrast: str, metric: str) -> dict[str, Any]:
        return next(
            row
            for row in contrasts
            if row["contrast"] == contrast and row["modality"] == PRIMARY_MODALITY and row["metric"] == metric
        )

    decisive = {metric: primary("S2_minus_S4", metric) for metric in ("recall_at_10", "mrr")}
    wins = {
        metric: row["mean"] > 0.0 and row["ci_low"] > 0.0
        for metric, row in decisive.items()
    }
    positive = any(row["mean"] > 0.0 for row in decisive.values())

    if any(wins.values()):
        primary_outcome = "adopt_specialization"
        reading = "specialization beats cost-matched sampling"
        action = (
            "Adopt. Section 4.2 becomes specialized generators and the multi-agent framing "
            "is earned."
        )
    elif positive:
        primary_outcome = "report_as_ablation"
        reading = "suggestive, underpowered at 30 table contexts"
        action = "Report as an ablation; keep stochastic sampling as the method."
    else:
        primary_outcome = "keep_current_method"
        reading = "specialization does not help"
        action = (
            "Keep the current section 4.2, add one ablation row; the 'On functional "
            "specialization' paragraph stands as written."
        )

    flags: list[dict[str, Any]] = []
    if (
        abs(float(equivalence["mean"])) < args.equivalence_margin
        and float(equivalence["ci_low"]) <= 0.0 <= float(equivalence["ci_high"])
    ):
        flags.append(
            {
                "flag": "three_generators_suffice",
                "observation": "S1 ~= S2",
                "reading": "the extra two specialists add nothing",
                "action": "Use 3 generators if adopting.",
                "evidence": equivalence,
            }
        )

    pooled_solo = sorted(
        (row for row in solo_rows if row["modality"] == "pooled" and row["slot"] in SPEC_SLOTS),
        key=lambda row: -float(row[f"solo_{SELECTION_METRIC}"]),
    )
    if len(pooled_solo) >= 2:
        lead = float(pooled_solo[0][f"solo_{SELECTION_METRIC}"]) - float(pooled_solo[1][f"solo_{SELECTION_METRIC}"])
        quiet = [
            row["generator"]
            for row in pooled_solo[1:]
            if float(row["selection_frequency_under_agree"]) < args.contribution_floor
        ]
        if lead >= args.dominance_margin and len(quiet) == len(pooled_solo) - 1:
            flags.append(
                {
                    "flag": "prompt_improvement_not_architecture",
                    "observation": "one generator dominates solo and the others rarely contribute",
                    "reading": "it is a prompt improvement, not an architecture",
                    "action": f"Fold {pooled_solo[0]['generator']}'s prompt into G0 and drop the rest.",
                    "evidence": {
                        "dominant_generator": pooled_solo[0]["generator"],
                        "solo_lead_on_" + SELECTION_METRIC: round(lead, 6),
                        "dominance_margin": args.dominance_margin,
                        "generators_below_contribution_floor": quiet,
                        "contribution_floor": args.contribution_floor,
                    },
                }
            )

    return {
        "primary_outcome": primary_outcome,
        "reading": reading,
        "action": action,
        "decisive_contrast": "S2_minus_S4 on the table subset, recall@10 or MRR",
        "decisive_rows": decisive,
        "ci_excludes_zero": wins,
        "flags": flags,
        "blocked_by": "abstention_violation" if abstention["violated"] else None,
        "blocked_note": abstention["action_if_violated"] if abstention["violated"] else None,
    }


# --------------------------------------------------------------------------------------


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
    }


def compatibility_summary(fact_scores: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(fact_scores)
    reasons: Counter = Counter()
    for score in fact_scores:
        for slot in SPEC_SLOTS:
            for check in score["compatibility"][slot].failed_checks():
                reasons[f"{slot}/{check}"] += 1
    # Report what passed and what fell back separately: a fallback fact also contributes
    # five generators to S3, so a single "retained" mean would read as a clean pass.
    passing_counts = [
        sum(1 for slot in SPEC_SLOTS if score["compatibility"][slot].passed) for score in fact_scores
    ]
    fallbacks = sum(1 for score in fact_scores if score["s3_fell_back"])
    return {
        "fact_count": total,
        "checks": list(COMPATIBILITY_CHECKS),
        "mean_generators_passing": round(sum(passing_counts) / total, 6) if total else 0.0,
        "generator_pass_rate": round(sum(passing_counts) / (total * len(SPEC_SLOTS)), 6) if total else 0.0,
        "facts_with_no_passing_generator": fallbacks,
        "fallback_rate": round(fallbacks / total, 6) if total else 0.0,
        "fallback_policy": "a fact where every generator fails falls back to the unfiltered S2 set",
        "mean_generators_scored_in_s3": round(
            sum(len(score["s3_slots"]) for score in fact_scores) / total, 6
        )
        if total
        else 0.0,
        "failed_check_counts": dict(sorted(reasons.items())),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    betas = tuple(sorted({float(beta) for beta in args.betas} | {float(args.reference_beta)}))
    eligible_betas = tuple(sorted({float(beta) for beta in args.betas}))

    if not args.sample_path.exists():
        raise FileNotFoundError(
            f"Frozen development sample not found: {args.sample_path}. Run the coverage pilot first."
        )
    sample_rows = load_jsonl(args.sample_path)
    examples = [row_to_example(row) for row in sample_rows]
    if args.limit_facts > 0:
        examples = examples[: args.limit_facts]
        sample_rows = sample_rows[: args.limit_facts]
    examples_by_id = {example.example_idx: example for example in examples}
    sample_summary = sample_summary_from_rows(sample_rows)
    print(json.dumps({"sample_summary": sample_summary}, indent=2))

    normalization_map = load_normalization_map(args.normalization_map)

    generator = None
    if not args.dry_run_no_llm:
        generator = QueryGenerator(args)
    try:
        hypotheses = build_hypotheses(args, examples, generator, normalization_map)
    finally:
        if generator is not None:
            generator.close()

    hypotheses = [h for h in hypotheses if int(h["fact_id"]) in examples_by_id]
    hypotheses_by_fact: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for hypothesis in hypotheses:
        hypotheses_by_fact[int(hypothesis["fact_id"])][str(hypothesis["slot"])] = hypothesis
    missing = {
        fact_id: sorted(set(ALL_SLOTS) - set(slots))
        for fact_id, slots in hypotheses_by_fact.items()
        if set(slots) != set(ALL_SLOTS)
    }
    if missing:
        raise ValueError(f"Incomplete generator coverage for {len(missing)} facts, e.g. {list(missing.items())[:3]}")

    # Building the BM25 index costs minutes, so an analysis-only rerun skips it entirely.
    retriever = None
    if not retrievals_are_reusable(args):
        print("Loading taxonomy and building the BM25 index ...", flush=True)
        retriever = TaxonomyRetriever(
            load_taxonomy(args.taxonomy_jsonl),
            type_filter=args.type_filter,
            label_coverage_weight=args.label_coverage_weight,
            label_coverage_pool_multiplier=args.label_coverage_pool_multiplier,
        )
    retrievals_path = build_retrievals(args, examples_by_id, hypotheses, retriever)

    print(f"Indexing {retrievals_path} ...", flush=True)
    offsets_by_fact = index_retrievals_by_fact(retrievals_path)

    fact_scores: list[dict[str, Any]] = []
    with retrievals_path.open("rb") as handle:
        for idx, example in enumerate(examples, start=1):
            records = []
            for offset in offsets_by_fact.get(example.example_idx, []):
                handle.seek(offset)
                records.append(json.loads(handle.readline()))
            if not records:
                raise ValueError(f"No retrievals logged for fact {example.example_idx}")
            fact_scores.append(
                score_fact(
                    args,
                    example,
                    records,
                    hypotheses_by_fact[example.example_idx],
                    normalization_map,
                    betas,
                )
            )
            if idx % max(args.log_every, 1) == 0 or idx == len(examples):
                print(f"Scored {idx}/{len(examples)} facts")

    sweep_rows = build_beta_sweep_rows(fact_scores, examples_by_id, betas)

    # Primary selection is on the table subset, which is also the primary read; the pooled
    # selection is reported alongside so the effect of that choice is visible rather than
    # assumed away.
    arm_betas_table = select_arm_betas(sweep_rows, PRIMARY_MODALITY, eligible_betas)
    arm_betas_pooled = select_arm_betas(sweep_rows, "pooled", eligible_betas)

    contrasts = contrast_rows(args, fact_scores, examples_by_id, arm_betas_table, "table_recall_at_10")
    contrasts_pooled = contrast_rows(args, fact_scores, examples_by_id, arm_betas_pooled, "pooled_recall_at_10")
    equivalence = equivalence_row(args, fact_scores, examples_by_id, arm_betas_table)

    solo_rows = generator_solo_rows(fact_scores, examples_by_id, arm_betas_table["S2"]["beta"])
    abstention = abstention_check(solo_rows)
    outcome = decision_rule_outcome(args, contrasts, equivalence, solo_rows, abstention)

    pair_jaccard = {}
    for left_idx, left in enumerate(SPEC_SLOTS):
        for right in SPEC_SLOTS[left_idx + 1 :]:
            values = [
                jaccard(score["candidate_lists"][left], score["candidate_lists"][right], 200)
                for score in fact_scores
            ]
            pair_jaccard[f"{slot_generator_key(left)}|{slot_generator_key(right)}"] = round(
                sum(values) / len(values), 6
            ) if values else None

    write_csv(args.output_dir / "beta_sweep_multigen.csv", sweep_rows)
    write_csv(args.output_dir / "specialization.csv", contrasts + contrasts_pooled)
    write_csv(args.output_dir / "generator_solo.csv", solo_rows)

    metrics = {
        "experiment": "ags_task_a_specialized_generators",
        "decides": (
            "Only whether section 4.2 describes stochastic sampling or specialized generators. "
            "It does not affect the frozen AGS test-split run, which proceeds independently."
        ),
        "prior_warning": PRIOR_WARNING,
        "abstention_check": abstention,
        "decision_rule_outcome": outcome,
        "selected_configuration": {
            "arm_betas_table_selected": arm_betas_table,
            "arm_betas_pooled_selected": arm_betas_pooled,
            "primary_modality": PRIMARY_MODALITY,
            "selection_metric": SELECTION_METRIC,
            "adopted_arm": "S2" if outcome["primary_outcome"] == "adopt_specialization" else "S0",
            "adopted_beta": (
                arm_betas_table["S2"]["beta"]
                if outcome["primary_outcome"] == "adopt_specialization"
                else arm_betas_table["S0"]["beta"]
            ),
        },
        "arms": {
            arm: {
                "name": ARM_NAMES[arm],
                # S3 and S5 are resolved per fact, so they carry no static slot list.
                "slots": list(STATIC_ARMS.get(arm, ())) or None,
            }
            for arm in ARM_ORDER
        },
        "generators": {key: GENERATOR_NAMES[key] for key in GENERATOR_KEYS},
        "sample_summary": sample_summary,
        "config": {
            "top_k": args.top_k,
            "rrf_kappa": args.rrf_kappa,
            "agreement_top_m": args.agreement_top_m,
            "label_coverage_weight": args.label_coverage_weight,
            "type_filter": bool(args.type_filter),
            "rendering_policy": FROZEN_RENDERING_POLICY,
            "betas": list(betas),
            "eligible_betas": list(eligible_betas),
            "specialist_temperature": args.specialist_temperature,
            "stochastic_temperature": args.stochastic_temperature,
            "query_generation_model": args.query_generation_model,
            "query_generation_backend": args.query_generation_backend,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "generation_calls": len(ALL_SLOTS) * len(examples),
            "normalization_map": {
                "path": str(args.normalization_map),
                "version": map_version(args.normalization_map),
            },
        },
        "compatibility_filter": compatibility_summary(fact_scores),
        "pairwise_neighborhood_jaccard_at_200": pair_jaccard,
        "primary_contrasts_table_selected": [
            row for row in contrasts if row["modality"] == PRIMARY_MODALITY
        ],
        "s1_minus_s2_equivalence": equivalence,
        "artifact_paths": {
            "sample_facts": str(args.sample_path),
            "hypotheses": str(args.output_dir / "hypotheses.jsonl"),
            "retrievals": str(retrievals_path),
            "specialization": str(args.output_dir / "specialization.csv"),
            "generator_solo": str(args.output_dir / "generator_solo.csv"),
            "beta_sweep_multigen": str(args.output_dir / "beta_sweep_multigen.csv"),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if abstention["violated"]:
        print("=" * 78)
        print("ABSTENTION VIOLATION: " + abstention["action_if_violated"])
        print(json.dumps(abstention["violating_generators"], indent=2))
        print("=" * 78)
    print(
        json.dumps(
            {
                "metrics_path": str(args.output_dir / "metrics.json"),
                "decision_rule_outcome": outcome["primary_outcome"],
                "action": outcome["action"],
                "arm_betas": {arm: arm_betas_table[arm]["beta"] for arm in ARM_ORDER},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

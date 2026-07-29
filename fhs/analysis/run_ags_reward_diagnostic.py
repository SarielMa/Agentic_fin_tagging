#!/usr/bin/env python3
"""Experiment B: online reward diagnostic for AGS controller learning."""

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
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ags_symbolic_agreement import (
    DEFAULT_NORMALIZATION_MAP,
    consensus_agreement,
    load_normalization_map,
    merge_feedback_layers,
    symbolic_feedback_from_candidates,
    utility_from_rank,
)
from run_ags_component_validation import (
    build_fused_from_records,
    candidate_ids,
    dimensions_upper,
    retrievals_by_render_fact,
)
from run_ags_coverage_pilot import (
    compact_candidates,
    load_jsonl,
    parse_depths,
    prompt_under_budget,
    row_to_example,
    write_csv,
)
from run_fintagging_grounding_baseline import (
    DEFAULT_TAXONOMY_JSONL,
    DIMENSIONS,
    SCRIPT_DIR,
    Example,
    QueryGenerator,
    TaxonomyRetriever,
    build_feedback_prompt_under_query_budget,
    build_operator_feedback_messages,
    build_operator_initial_messages,
    build_operator_revision_messages,
    first_gold_rank,
    fuse_round_candidates,
    llm_call_record,
    load_taxonomy,
    neighborhood_novelty,
    normalize_space,
    normalize_tag,
    parse_feedback,
    parse_hypothesis,
    retrieval_query_from_grounding,
    retrieve_candidates,
    write_jsonl,
)


B_OPERATORS = (
    "O_refine_family",
    "O_refine_role",
    "O_refine_event",
    "O_refine_qualifier",
    "O_refine_scope",
    "O_refine_temporal",
    "O_resample",
    "O_free",
)
DIMENSION_OPERATOR = {
    "FAMILY": "O_refine_family",
    "ROLE": "O_refine_role",
    "EVENT": "O_refine_event",
    "QUALIFIER": "O_refine_qualifier",
    "SCOPE": "O_refine_scope",
    "TEMPORAL": "O_refine_temporal",
}
OPERATOR_DIMENSION = {value: key for key, value in DIMENSION_OPERATOR.items()}
ARMS = ("bandit", "random", "resample_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=FHS_ROOT / "runs" / "runs_ags_reward_diagnostic" / "qwen3_32b")
    parser.add_argument("--component-output-dir", type=Path, default=FHS_ROOT / "runs" / "runs_ags_component_validation" / "qwen3_32b")
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=FHS_ROOT / "data" / "dev" / "sample_facts.jsonl",
    )
    parser.add_argument("--taxonomy-jsonl", type=Path, default=DEFAULT_TAXONOMY_JSONL)
    parser.add_argument("--normalization-map", type=Path, default=DEFAULT_NORMALIZATION_MAP)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--depths", default="10,50,200")
    parser.add_argument("--stream-facts", type=int, default=250)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--feedback-candidate-count", type=int, default=10)
    parser.add_argument("--candidate-doc-max-chars", type=int, default=320)
    parser.add_argument("--agreement-top-m", type=int, default=10)
    parser.add_argument("--posterior-snapshot-every", type=int, default=25)
    parser.add_argument("--posterior-alpha", type=float, default=0.75)
    parser.add_argument("--posterior-ridge", type=float, default=1.0)
    parser.add_argument("--informative-delta", type=float, default=0.01)
    parser.add_argument("--novelty-threshold", type=float, default=0.02)
    parser.add_argument("--live-consensus-beta", type=float, default=None)
    parser.add_argument("--diagnostic-consensus-betas", default="")
    parser.add_argument("--label-coverage-weight", type=float, default=1.0)
    parser.add_argument(
        "--label-coverage-pool-multiplier",
        type=int,
        default=0,
        help="Pool multiplier for label-coverage rescoring; <=0 scores the full type-filtered pool.",
    )
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--type-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--acknowledge-negative-rendering-gate", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--dry-run-no-llm", action="store_true")

    parser.add_argument("--query-generation-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--query-generation-backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--query-context-max-chars", type=int, default=12000)
    parser.add_argument("--query-max-input-tokens", type=int, default=16000)
    parser.add_argument("--query-max-new-tokens", type=int, default=512)
    parser.add_argument("--query-temperature", type=float, default=0.8)
    parser.add_argument("--query-top-p", type=float, default=1.0)
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
    return parser.parse_args()


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float")
    return values


def beta_key(beta: float) -> str:
    return f"{beta:g}"


def diagnostic_betas(args: argparse.Namespace) -> list[float]:
    resolved = getattr(args, "resolved_diagnostic_consensus_betas", None)
    if resolved is not None:
        values = list(resolved)
    else:
        values = parse_float_list(args.diagnostic_consensus_betas or "0.05,0.1,0.2,0.4")
    live_beta = resolved_live_consensus_beta(args)
    if not any(math.isclose(value, live_beta) for value in values):
        values.insert(0, live_beta)
    deduped: list[float] = []
    for value in values:
        if not any(math.isclose(value, existing) for existing in deduped):
            deduped.append(value)
    return deduped


def resolved_live_consensus_beta(args: argparse.Namespace) -> float:
    value = getattr(args, "resolved_live_consensus_beta", None)
    if value is not None:
        return float(value)
    if args.live_consensus_beta is not None:
        return float(args.live_consensus_beta)
    return 0.1


def resolve_consensus_beta_settings(args: argparse.Namespace, gate: dict[str, Any]) -> None:
    recommendation = gate.get("consensus_beta_recommendation") or {}
    recommended_beta = recommendation.get("adopted_consensus_beta")
    if args.live_consensus_beta is not None:
        live_beta = float(args.live_consensus_beta)
        source = "cli"
    elif recommended_beta is not None:
        live_beta = float(recommended_beta)
        source = "component_A3_recommendation"
    else:
        live_beta = 0.1
        source = "fallback"

    if args.diagnostic_consensus_betas:
        betas = parse_float_list(args.diagnostic_consensus_betas)
        beta_source = "cli"
    else:
        betas = [float(value) for value in recommendation.get("candidate_betas", [])]
        if not betas:
            betas = [0.05, 0.1, 0.2, 0.4]
        beta_source = "component_A3_candidates" if recommendation.get("candidate_betas") else "fallback"

    if not any(math.isclose(value, live_beta) for value in betas):
        betas.insert(0, live_beta)
    args.resolved_live_consensus_beta = live_beta
    args.resolved_live_consensus_beta_source = source
    args.resolved_diagnostic_consensus_betas = sorted(set(betas))
    args.resolved_diagnostic_consensus_betas_source = beta_source


def load_component_gate(args: argparse.Namespace) -> dict[str, Any]:
    path = args.component_output_dir / "metrics.json"
    if not path.exists():
        return {
            "passed": False,
            "reason": "missing_component_metrics",
            "metrics_path": str(path),
        }
    metrics = json.loads(path.read_text(encoding="utf-8"))
    tokenization = metrics.get("tokenization_check", {})
    gate = metrics.get("rendering_gate") or {}
    gate_status = gate.get("status")
    allowed = bool(tokenization.get("passed")) and gate_status in {"proceed_claim", "proceed_weak"}
    if gate_status == "halt_requires_ack" and args.acknowledge_negative_rendering_gate:
        allowed = bool(tokenization.get("passed"))
    rendering_by_modality = gate.get("adopted_rendering_by_modality") or {
        "table": "dual",
        "text": "def",
        "pooled": "modality_conditional",
    }
    return {
        "passed": allowed,
        "metrics_path": str(path),
        "tokenization_passed": bool(tokenization.get("passed")),
        "rendering_gate_status": gate_status,
        "rendering_claim_allowed": bool(gate.get("claim_allowed")),
        "adopted_rendering": "modality_conditional",
        "adopted_rendering_by_modality": rendering_by_modality,
        "adopted_candidate_pool_initialization": "j3_union_rrf",
        "adopted_guided_hypothesis_initialization": "j3_symbolic_select",
        "adopted_initialization": "j3_union_rrf_symbolic_select",
        "initialization_recommendation": metrics.get("initialization_recommendation") or {},
        "consensus_beta_recommendation": metrics.get("consensus_beta_recommendation") or {},
        "human_ack_used": bool(args.acknowledge_negative_rendering_gate),
        "reason": "ok" if allowed else "A metrics gate is not in an allowed proceed state",
    }


class PosteriorBank:
    def __init__(self, operators: tuple[str, ...], dim: int, ridge: float, alpha: float, seed: int) -> None:
        self.operators = operators
        self.dim = dim
        self.ridge = ridge
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.a = {operator: np.eye(dim) * ridge for operator in operators}
        self.b = {operator: np.zeros(dim) for operator in operators}
        self.n_records = Counter()
        self.n_informative = Counter()

    def posterior(self, operator: str) -> tuple[np.ndarray, np.ndarray]:
        inv = np.linalg.inv(self.a[operator])
        mu = inv @ self.b[operator]
        return mu, inv

    def sample_score(self, operator: str, psi: np.ndarray) -> float:
        mu, inv = self.posterior(operator)
        cov = (self.alpha ** 2) * inv
        theta = self.rng.multivariate_normal(mu, cov)
        return float(theta @ psi)

    def mean_score(self, operator: str, psi: np.ndarray) -> float:
        mu, _ = self.posterior(operator)
        return float(mu @ psi)

    def update(self, operator: str, psi: np.ndarray, reward: float, informative_delta: float) -> None:
        self.a[operator] += np.outer(psi, psi)
        self.b[operator] += reward * psi
        self.n_records[operator] += 1
        if abs(reward) > informative_delta:
            self.n_informative[operator] += 1

    def snapshot_rows(self, arm: str, instance_idx: int) -> list[dict[str, Any]]:
        rows = []
        for operator in self.operators:
            mu, inv = self.posterior(operator)
            rows.append(
                {
                    "arm": arm,
                    "instance_idx": instance_idx,
                    "operator": operator,
                    "mu": [round(float(value), 8) for value in mu.tolist()],
                    "sigma_diag": [round(float(value), 8) for value in np.diag(inv).tolist()],
                    "n_records": int(self.n_records[operator]),
                    "n_informative_records": int(self.n_informative[operator]),
                    "condition_number": round(float(np.linalg.cond(self.a[operator])), 8),
                }
            )
        return rows


def stream_examples(sample_rows: list[dict[str, Any]], limit: int) -> list[Example]:
    rows = sorted(
        sample_rows,
        key=lambda row: (
            row.get("input_type"),
            int(row.get("source_sample_idx", -1)),
            int(row.get("fact_id", -1)),
        ),
    )
    return [row_to_example(row) for row in rows[:limit]]


def compact_candidate_list(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": int(candidate.get("rank", idx + 1)),
            "tag": candidate.get("tag"),
            "type": candidate.get("type"),
            "standard_label": candidate.get("standard_label"),
            "bm25_score": candidate.get("bm25_score"),
            "bm25_normalized_score": candidate.get("bm25_normalized_score"),
            "label_coverage": candidate.get("label_coverage"),
            "query_label_coverage": candidate.get("query_label_coverage"),
            "retrieval_score": candidate.get("retrieval_score"),
            "rrf_score": candidate.get("rrf_score"),
            "combsum_score": candidate.get("combsum_score"),
            "consensus_agreement": candidate.get("consensus_agreement"),
            "consensus_beta": candidate.get("consensus_beta"),
            "rerank_score": candidate.get("rerank_score"),
        }
        for idx, candidate in enumerate(candidates)
    ]


def utility(candidates: list[dict[str, Any]], gold_tags: list[str]) -> tuple[float, int | None, bool]:
    ids = candidate_ids(candidates)
    rank = first_gold_rank(ids, gold_tags)
    return utility_from_rank(rank), rank, rank is not None


def consolidation_hypotheses(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hypotheses = []
    for round_record in rounds:
        hypothesis = round_record.get("hypothesis")
        if isinstance(hypothesis, dict):
            hypotheses.append(hypothesis)
    return hypotheses


def consolidate(
    rounds: list[dict[str, Any]],
    rrf_kappa: float,
    beta: float,
    normalization_map: dict[str, Any],
) -> list[dict[str, Any]]:
    return rerank_consolidated_base(consolidate_base(rounds, rrf_kappa, normalization_map), beta)


def consolidate_base(
    rounds: list[dict[str, Any]],
    rrf_kappa: float,
    normalization_map: dict[str, Any],
) -> list[dict[str, Any]]:
    base = fuse_round_candidates(
        [{"round": idx + 1, "candidates": round_record["candidates"]} for idx, round_record in enumerate(rounds)],
        None,
        rrf_kappa,
    )
    hypotheses = consolidation_hypotheses(rounds)
    rescored = []
    for candidate in base:
        consensus = consensus_agreement(candidate, hypotheses, normalization_map)
        updated = dict(candidate)
        updated["consensus_agreement"] = consensus
        rescored.append(updated)
    return rescored


def rerank_consolidated_base(candidates: list[dict[str, Any]], beta: float) -> list[dict[str, Any]]:
    rescored = []
    for candidate in candidates:
        updated = dict(candidate)
        score = float(updated.get("rrf_score", 0.0)) + beta * float(updated.get("consensus_agreement", 0.0))
        updated["consensus_beta"] = beta
        updated["rerank_score"] = round(score, 8)
        rescored.append(updated)
    rescored.sort(
        key=lambda candidate: (
            -float(candidate.get("rerank_score", 0.0)),
            int(candidate.get("rank", 10**9)),
            candidate.get("tag", ""),
        )
    )
    for rank, candidate in enumerate(rescored, start=1):
        candidate["rank"] = rank
    return rescored


def score_rounds_by_beta(
    rounds: list[dict[str, Any]],
    gold_tags: list[str],
    rrf_kappa: float,
    betas: list[float],
    normalization_map: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    scored = {}
    base = consolidate_base(rounds, rrf_kappa, normalization_map)
    for beta in betas:
        ranking = rerank_consolidated_base(base, beta)
        y, rank, in_union = utility(ranking, gold_tags)
        scored[beta_key(beta)] = {
            "utility": y,
            "rank": rank,
            "in_union": in_union,
            "ranking": ranking,
        }
    return scored


def feedback_counts(feedback: dict[str, Any], layer: str = "merged") -> dict[str, int]:
    if layer == "merged":
        source = feedback
    elif layer == "llm":
        source = feedback.get("llm_feedback", {}) if isinstance(feedback.get("llm_feedback"), dict) else {}
    elif layer in {"exact", "lexical"}:
        counts = {"D_plus": 0, "D_minus": 0, "D_question": 0}
        symbolic = feedback.get("symbolic_feedback", {}) if isinstance(feedback.get("symbolic_feedback"), dict) else {}
        for verdict in symbolic.get("dimension_verdicts", []):
            if symbolic_verdict_layer(verdict) != layer:
                continue
            verdict_name = verdict.get("verdict")
            if verdict_name == "support":
                counts["D_plus"] += 1
            elif verdict_name == "contradict":
                counts["D_minus"] += 1
            else:
                counts["D_question"] += 1
        return counts
    else:
        source = {}
    return {
        "D_plus": len(source.get("supported_dimensions", [])),
        "D_minus": len(source.get("contradicted_dimensions", [])),
        "D_question": len(source.get("unresolved_dimensions", [])),
    }


def symbolic_verdict_layer(verdict: dict[str, Any]) -> str:
    candidate_reasons = {
        str(candidate_verdict.get("reason", ""))
        for candidate_verdict in verdict.get("candidate_verdicts", [])
        if isinstance(candidate_verdict, dict)
    }
    if "token_overlap" in candidate_reasons or "no_comparable_tokens" in candidate_reasons:
        return "lexical"
    return "exact"


def feedback_counts_by_layer(feedback: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {
        layer: feedback_counts(feedback, layer)
        for layer in ("merged", "exact", "lexical", "llm")
    }


def empty_feedback() -> dict[str, Any]:
    return {
        "supported_dimensions": [],
        "contradicted_dimensions": [],
        "unresolved_dimensions": [],
        "dimension_verdicts": [],
        "structural_mismatch": {"is_mismatch": False, "reason": ""},
        "symbolic_feedback": {"dimension_verdicts": []},
        "llm_feedback": {
            "supported_dimensions": [],
            "contradicted_dimensions": [],
            "unresolved_dimensions": [],
        },
        "source_layer": "none",
    }


def psi_vector(example: Example, feedback: dict[str, Any], novelty: float) -> tuple[dict[str, Any], np.ndarray]:
    counts = feedback_counts(feedback)
    mismatch = bool((feedback.get("structural_mismatch") or {}).get("is_mismatch"))
    dtype = example.entity_type.lower()
    named = {
        "bias": 1.0,
        "is_table": 1.0 if example.input_type == "table" else 0.0,
        "is_text": 1.0 if example.input_type == "text" else 0.0,
        "is_monetary": 1.0 if "monetary" in dtype else 0.0,
        "is_shares": 1.0 if "share" in dtype else 0.0,
        "is_percent": 1.0 if "percent" in dtype or "pure" in dtype else 0.0,
        "D_plus_count": float(counts["D_plus"]),
        "D_minus_count": float(counts["D_minus"]),
        "D_question_count": float(counts["D_question"]),
        "structural_mismatch_g": 1.0 if mismatch else 0.0,
        "neighborhood_novelty_n": float(novelty),
    }
    return named, np.asarray(list(named.values()), dtype=float)


def admissible_slate(feedback: dict[str, Any]) -> list[dict[str, Any]]:
    slate = []
    contradicted = [normalize_space(dim).upper() for dim in feedback.get("contradicted_dimensions", [])]
    unresolved = [normalize_space(dim).upper() for dim in feedback.get("unresolved_dimensions", [])]
    for dim in contradicted:
        operator = DIMENSION_OPERATOR.get(dim)
        if operator:
            slate.append({"mode": "REFINE", "operator": operator, "target_dimension": dim})
    for dim in unresolved:
        operator = DIMENSION_OPERATOR.get(dim)
        if operator:
            slate.append({"mode": "BRANCH", "operator": operator, "target_dimension": dim})
    if bool((feedback.get("structural_mismatch") or {}).get("is_mismatch")):
        slate.append({"mode": "CHANGE_STRATEGY", "operator": "O_resample", "target_dimension": ""})
    if not slate:
        slate.append({"mode": "BRANCH", "operator": "O_free", "target_dimension": ""})
    seen = set()
    unique = []
    for item in slate:
        key = (item["mode"], item["operator"], item["target_dimension"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def choose_directive(
    arm: str,
    slate: list[dict[str, Any]],
    posterior: PosteriorBank,
    psi: np.ndarray,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, float]]:
    if arm == "resample_only":
        selected = {"mode": "CHANGE_STRATEGY", "operator": "O_resample", "target_dimension": ""}
        return selected, None, {"O_resample": 1.0}
    if arm == "random":
        selected = rng.choice(slate)
        scores = {item["operator"]: 1.0 / len(slate) for item in slate}
    else:
        scores = {item["operator"]: posterior.sample_score(item["operator"], psi) for item in slate}
        selected = max(slate, key=lambda item: scores[item["operator"]])
    runner_up = None
    if len(slate) > 1:
        runner_up = sorted(
            [item for item in slate if item is not selected],
            key=lambda item: scores.get(item["operator"], 0.0),
            reverse=True,
        )[0]
    return selected, runner_up, scores


def directive_payload(item: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:
    target = item.get("target_dimension", "")
    alternatives = feedback.get("alternative_values", {}) if isinstance(feedback.get("alternative_values"), dict) else {}
    patch_values = alternatives.get(target, []) if target else []
    patch = "; ".join(patch_values[:3]) if patch_values else f"revise {target or 'interpretation'}"
    return {
        "mode": item.get("mode", "BRANCH"),
        "operator": item.get("operator", "O_free"),
        "target_dimension": target,
        "semantic_patch": patch,
        "preserve": feedback.get("supported_dimensions", []),
        "rationale": "diagnostic controller directive",
    }


def fake_revision(current: dict[str, Any], directive: dict[str, Any]) -> dict[str, Any]:
    revised = json.loads(json.dumps(current))
    target = directive.get("target_dimension")
    if target:
        revised.setdefault("dimensions", {})[target] = directive.get("semantic_patch") or revised.get("dimensions", {}).get(target, "UNRESOLVED")
    revised["retrieval_query"] = normalize_space(
        f"{current.get('retrieval_query', '')} {directive.get('operator', '')} {directive.get('semantic_patch', '')}"
    )
    revised["operators"] = [directive.get("operator", "O_free")]
    return revised


def generate_feedback(
    args: argparse.Namespace,
    generator: QueryGenerator | None,
    example: Example,
    hypothesis: dict[str, Any],
    candidates: list[dict[str, Any]],
    normalization_map: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    symbolic = symbolic_feedback_from_candidates(
        hypothesis.get("dimensions", {}),
        candidates,
        top_m=args.agreement_top_m,
        normalization_map=normalization_map,
    )
    if args.dry_run_no_llm:
        llm_feedback = {
            "supported_dimensions": symbolic.get("supported_dimensions", []),
            "contradicted_dimensions": symbolic.get("contradicted_dimensions", []),
            "unresolved_dimensions": symbolic.get("unresolved_dimensions", []),
            "alternative_values": {},
            "structural_mismatch": symbolic.get("structural_mismatch", {}),
        }
        call = llm_call_record(
            "operator_feedback",
            raw_output=json.dumps(llm_feedback, ensure_ascii=False),
            prompt_tokens=0,
            completion_tokens=0,
            parse_ok=True,
            backend="dry_run",
            model_name="dry_run",
        )
    else:
        assert generator is not None
        prompt, prompt_tokens, used_context_chars, used_doc_chars = build_feedback_prompt_under_query_budget(
            generator.tokenizer,
            lambda ctx_chars, doc_chars: build_operator_feedback_messages(
                example,
                hypothesis,
                candidates,
                ctx_chars,
                doc_chars,
                args.feedback_candidate_count,
            ),
            context_max_chars=args.query_context_max_chars,
            doc_max_chars=args.candidate_doc_max_chars,
            max_input_tokens=args.query_max_input_tokens,
        )
        raw_output = generator.generate_one(prompt)
        llm_feedback, parse_ok = parse_feedback(raw_output)
        call = llm_call_record(
            "operator_feedback",
            raw_output=raw_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=generator.count_text_tokens(raw_output),
            parse_ok=parse_ok,
            backend=generator.backend,
            model_name=generator.model_name,
            extra_fields={
                "used_context_max_chars": used_context_chars,
                "used_candidate_doc_max_chars": used_doc_chars,
            },
        )
    merged = merge_feedback_layers(symbolic, llm_feedback)
    merged["symbolic_feedback"] = symbolic
    merged["llm_feedback"] = llm_feedback
    return merged, [call]


def retrieve_hypothesis(
    retriever: TaxonomyRetriever,
    example: Example,
    hypothesis: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    query = retrieval_query_from_grounding(example, hypothesis.get("retrieval_query", ""))
    return compact_candidates(retrieve_candidates(retriever, query, example.entity_type, top_k))


def generate_revision(
    args: argparse.Namespace,
    generator: QueryGenerator | None,
    retriever: TaxonomyRetriever,
    example: Example,
    current_hypothesis: dict[str, Any],
    directive: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if directive.get("operator") in {"O_resample", "O_free"}:
        if args.dry_run_no_llm:
            revised = fake_revision(current_hypothesis, directive)
            call = llm_call_record(
                "operator_resample",
                raw_output=json.dumps(revised, ensure_ascii=False),
                prompt_tokens=0,
                completion_tokens=0,
                parse_ok=True,
                backend="dry_run",
                model_name="dry_run",
            )
        else:
            assert generator is not None
            prompt, prompt_tokens, used_context_chars = prompt_under_budget(
                generator,
                args,
                lambda ctx_chars: build_operator_initial_messages(example, ctx_chars),
            )
            raw_output = generator.generate_one(prompt)
            revised, parse_ok = parse_hypothesis(raw_output, current_hypothesis.get("retrieval_query", ""))
            call = llm_call_record(
                "operator_resample",
                raw_output=raw_output,
                prompt_tokens=prompt_tokens,
                completion_tokens=generator.count_text_tokens(raw_output),
                parse_ok=parse_ok,
                backend=generator.backend,
                model_name=generator.model_name,
                extra_fields={"used_context_max_chars": used_context_chars},
            )
    elif args.dry_run_no_llm:
        revised = fake_revision(current_hypothesis, directive)
        call = llm_call_record(
            "operator_revision",
            raw_output=json.dumps(revised, ensure_ascii=False),
            prompt_tokens=0,
            completion_tokens=0,
            parse_ok=True,
            backend="dry_run",
            model_name="dry_run",
        )
    else:
        assert generator is not None
        prompt, prompt_tokens, used_context_chars = prompt_under_budget(
            generator,
            args,
            lambda ctx_chars: build_operator_revision_messages(example, current_hypothesis, directive, ctx_chars),
        )
        raw_output = generator.generate_one(prompt)
        revised, parse_ok = parse_hypothesis(raw_output, current_hypothesis.get("retrieval_query", ""))
        call = llm_call_record(
            "operator_revision",
            raw_output=raw_output,
            prompt_tokens=prompt_tokens,
            completion_tokens=generator.count_text_tokens(raw_output),
            parse_ok=parse_ok,
            backend=generator.backend,
            model_name=generator.model_name,
            extra_fields={"used_context_max_chars": used_context_chars},
        )
    return revised, retrieve_hypothesis(retriever, example, revised, args.top_k), [call]


def load_component_initial_state(
    args: argparse.Namespace,
    gate: dict[str, Any],
    examples: list[Example],
    normalization_map: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    hypotheses = load_jsonl(args.component_output_dir / "hypotheses.jsonl")
    retrievals = load_jsonl(args.component_output_dir / "retrievals.jsonl")
    rendering_by_modality = gate.get("adopted_rendering_by_modality") or {"table": "dual", "text": "def"}
    hyp_by_fact: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for hypothesis in hypotheses:
        hyp_by_fact[int(hypothesis["fact_id"])].append(hypothesis)
    for values in hyp_by_fact.values():
        values.sort(key=lambda item: int(item["hypothesis_idx"]))
    render_records_by_render = retrievals_by_render_fact(retrievals)
    symbolic_choices = gate.get("initialization_recommendation", {}).get("symbolic_choices", {})

    state = {}
    for example in examples:
        fact_id = example.example_idx
        rendering = rendering_by_modality.get(example.input_type, "dual")
        records = render_records_by_render[rendering][fact_id]
        hypotheses_for_fact = hyp_by_fact[fact_id]
        selected_idx = int(symbolic_choices.get(str(fact_id), symbolic_choices.get(fact_id, 0)))
        selected_idx = min(max(selected_idx, 0), len(hypotheses_for_fact) - 1)
        fused = build_fused_from_records(records, None, args.rrf_kappa)
        initial_rounds = [
            {
                "hypothesis": hypotheses_for_fact[selected_idx],
                "candidates": fused,
                "source_rendering": rendering,
                "source_hypothesis_count": len(records),
                "candidate_pool_initialization": "j3_union_rrf",
                "guided_hypothesis_initialization": "j3_symbolic_select",
                "selected_hypothesis_idx": selected_idx,
            }
        ]
        state[fact_id] = {
            "current_hypothesis": {
                "dimensions": dimensions_upper(hypotheses_for_fact[selected_idx]["dimensions"]),
                "operators": hypotheses_for_fact[selected_idx].get("operators", []),
                "retrieval_query": hypotheses_for_fact[selected_idx].get("query_def", ""),
            },
            "initialization": {
                "rendering": rendering,
                "candidate_pool_initialization": "j3_union_rrf",
                "guided_hypothesis_initialization": "j3_symbolic_select",
                "selected_hypothesis_idx": selected_idx,
                "candidate_pool_size": len(fused),
            },
            "rounds": [
                {
                    "round_idx": idx,
                    "hypothesis": round_record["hypothesis"],
                    "candidates": round_record["candidates"],
                    "source_rendering": round_record.get("source_rendering"),
                    "source_hypothesis_count": round_record.get("source_hypothesis_count"),
                    "candidate_pool_initialization": round_record.get("candidate_pool_initialization"),
                    "guided_hypothesis_initialization": round_record.get("guided_hypothesis_initialization"),
                    "selected_hypothesis_idx": round_record.get("selected_hypothesis_idx"),
                }
                for idx, round_record in enumerate(initial_rounds, start=1)
            ],
        }
    return state


def run_arm(
    args: argparse.Namespace,
    arm: str,
    examples: list[Example],
    initial_state: dict[int, dict[str, Any]],
    retriever: TaxonomyRetriever,
    generator: QueryGenerator | None,
    normalization_map: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_offsets = {"bandit": 0, "random": 100000, "resample_only": 200000}
    rng = random.Random(args.seed + seed_offsets.get(arm, 0))
    posterior = PosteriorBank(
        B_OPERATORS,
        dim=11,
        ridge=args.posterior_ridge,
        alpha=args.posterior_alpha,
        seed=args.seed + seed_offsets.get(arm, 0),
    )
    betas = diagnostic_betas(args)
    live_beta = resolved_live_consensus_beta(args)
    live_beta_key = beta_key(live_beta)
    rounds_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []

    for instance_idx, example in enumerate(examples, start=1):
        state = json.loads(json.dumps(initial_state[example.example_idx]))
        current_hypothesis = state["current_hypothesis"]
        episode_rounds = state["rounds"]
        stop_reason = "budget"
        for round_idx in range(2, args.rounds + 1):
            before_by_beta = score_rounds_by_beta(
                episode_rounds,
                example.gold_tags,
                args.rrf_kappa,
                betas,
                normalization_map,
            )
            live_before = before_by_beta[live_beta_key]
            y_before = float(live_before["utility"])
            rank_before = live_before["rank"]
            in_before = bool(live_before["in_union"])
            current_candidates = episode_rounds[-1]["candidates"]
            if arm == "resample_only":
                feedback = empty_feedback()
                feedback_calls: list[dict[str, Any]] = []
            else:
                feedback, feedback_calls = generate_feedback(
                    args,
                    generator,
                    example,
                    current_hypothesis,
                    current_candidates,
                    normalization_map,
                )
            novelty = neighborhood_novelty(current_candidates, episode_rounds[:-1])
            psi_named, psi = psi_vector(example, feedback, novelty)
            slate = (
                [{"mode": "CHANGE_STRATEGY", "operator": "O_resample", "target_dimension": ""}]
                if arm == "resample_only"
                else admissible_slate(feedback)
            )
            if not slate:
                stop_reason = "no_admissible_directive"
                break
            selected, runner_up, selection_scores = choose_directive(arm, slate, posterior, psi, rng)
            directive = directive_payload(selected, feedback)
            revised, selected_candidates, revision_calls = generate_revision(
                args,
                generator,
                retriever,
                example,
                current_hypothesis,
                directive,
            )
            selected_round = {
                "round_idx": round_idx,
                "hypothesis": revised,
                "candidates": selected_candidates,
                "directive": directive,
            }
            after_by_beta = score_rounds_by_beta(
                episode_rounds + [selected_round],
                example.gold_tags,
                args.rrf_kappa,
                betas,
                normalization_map,
            )
            live_after = after_by_beta[live_beta_key]
            after_consolidated = live_after["ranking"]
            rank_after = live_after["rank"]
            in_after = bool(live_after["in_union"])
            delta_y_by_beta = {
                key: round(float(after_by_beta[key]["utility"]) - float(before_by_beta[key]["utility"]), 8)
                for key in before_by_beta
            }
            delta_y = float(delta_y_by_beta[live_beta_key])

            counterfactual_candidates: list[dict[str, Any]] = []
            counterfactual_delta_y_by_beta = {key: 0.0 for key in before_by_beta}
            replay_calls: list[dict[str, Any]] = []
            if runner_up is not None:
                replay_directive = directive_payload(runner_up, feedback)
                _, counterfactual_candidates, replay_calls = generate_revision(
                    args,
                    generator,
                    retriever,
                    example,
                    current_hypothesis,
                    replay_directive,
                )
                replay_round = {
                    "round_idx": round_idx,
                    "hypothesis": current_hypothesis,
                    "candidates": counterfactual_candidates,
                    "directive": replay_directive,
                }
                replay_by_beta = score_rounds_by_beta(
                    episode_rounds + [replay_round],
                    example.gold_tags,
                    args.rrf_kappa,
                    betas,
                    normalization_map,
                )
                counterfactual_delta_y_by_beta = {
                    key: round(float(replay_by_beta[key]["utility"]) - float(before_by_beta[key]["utility"]), 8)
                    for key in before_by_beta
                }

            counterfactual_delta_y = float(counterfactual_delta_y_by_beta[live_beta_key])
            delta_replay_by_beta = {
                key: round(float(delta_y_by_beta[key]) - float(counterfactual_delta_y_by_beta[key]), 8)
                for key in before_by_beta
            }
            reward_combined_by_beta = {
                key: round(float(delta_y_by_beta[key]) + 0.5 * float(delta_replay_by_beta[key]), 8)
                for key in before_by_beta
            }
            delta_replay = float(delta_replay_by_beta[live_beta_key])
            reward_combined = float(reward_combined_by_beta[live_beta_key])
            if arm != "resample_only":
                posterior.update(selected["operator"], psi, reward_combined, args.informative_delta)
                if runner_up is not None:
                    posterior.update(runner_up["operator"], psi, counterfactual_delta_y, args.informative_delta)

            selected_novelty = neighborhood_novelty(selected_candidates, episode_rounds)
            gate_rejections = 1 if selected_novelty < args.novelty_threshold else 0
            if gate_rejections:
                stop_reason = "gate_exhaustion"
            else:
                episode_rounds.append(selected_round)
                current_hypothesis = revised

            counts_by_layer = feedback_counts_by_layer(feedback)
            counts = counts_by_layer["merged"]
            exact_counts = counts_by_layer["exact"]
            lexical_counts = counts_by_layer["lexical"]
            llm_counts = counts_by_layer["llm"]
            rounds_rows.append(
                {
                    "fact_id": example.example_idx,
                    "context_id": example.context_id,
                    "source_sample_idx": example.source_sample_idx,
                    "modality": example.input_type,
                    "arm": arm,
                    "round_idx": round_idx,
                    "initialization": state.get("initialization", {}),
                    "psi": psi_named,
                    "psi_values": [round(float(value), 8) for value in psi.tolist()],
                    "slate": slate,
                    "selected_operator": selected["operator"],
                    "selected_mode": selected["mode"],
                    "runner_up_operator": runner_up["operator"] if runner_up else None,
                    "selection_scores": selection_scores,
                    "live_consensus_beta": live_beta,
                    "diagnostic_consensus_betas": betas,
                    "gold_in_union_before": in_before,
                    "gold_in_union_after": in_after,
                    "gold_rank_before": rank_before,
                    "gold_rank_after": rank_after,
                    "gold_rank_before_by_beta": {key: before_by_beta[key]["rank"] for key in before_by_beta},
                    "gold_rank_after_by_beta": {key: after_by_beta[key]["rank"] for key in after_by_beta},
                    "gold_in_union_before_by_beta": {key: before_by_beta[key]["in_union"] for key in before_by_beta},
                    "gold_in_union_after_by_beta": {key: after_by_beta[key]["in_union"] for key in after_by_beta},
                    "delta_y": round(delta_y, 8),
                    "delta_y_by_beta": delta_y_by_beta,
                    "delta_replay": round(delta_replay, 8),
                    "delta_replay_by_beta": delta_replay_by_beta,
                    "counterfactual_delta_y": round(counterfactual_delta_y, 8),
                    "counterfactual_delta_y_by_beta": counterfactual_delta_y_by_beta,
                    "reward_combined": round(reward_combined, 8),
                    "reward_combined_by_beta": reward_combined_by_beta,
                    "gate_rejections_this_round": gate_rejections,
                    "stop_reason": stop_reason,
                    "D_plus_count": counts["D_plus"],
                    "D_minus_count": counts["D_minus"],
                    "D_question_count": counts["D_question"],
                    "D_plus_exact_count": exact_counts["D_plus"],
                    "D_minus_exact_count": exact_counts["D_minus"],
                    "D_question_exact_count": exact_counts["D_question"],
                    "D_plus_lexical_count": lexical_counts["D_plus"],
                    "D_minus_lexical_count": lexical_counts["D_minus"],
                    "D_question_lexical_count": lexical_counts["D_question"],
                    "D_plus_llm_count": llm_counts["D_plus"],
                    "D_minus_llm_count": llm_counts["D_minus"],
                    "D_question_llm_count": llm_counts["D_question"],
                    "feedback_counts_by_layer": counts_by_layer,
                    "structural_mismatch_flag_g": bool((feedback.get("structural_mismatch") or {}).get("is_mismatch")),
                    "neighborhood_novelty_n": round(float(novelty), 8),
                    "selected_neighborhood_novelty": round(float(selected_novelty), 8),
                    "dimension_verdicts": feedback.get("dimension_verdicts", []),
                    "feedback": feedback,
                    "candidates_before": compact_candidate_list(current_candidates),
                    "selected_candidates_after": compact_candidate_list(selected_candidates),
                    "counterfactual_candidates": compact_candidate_list(counterfactual_candidates),
                    "accumulated_candidates_after": compact_candidate_list(after_consolidated),
                    "llm_calls": feedback_calls + revision_calls + replay_calls,
                }
            )
            if gate_rejections:
                break

        if instance_idx % args.posterior_snapshot_every == 0 or instance_idx == len(examples):
            posterior_rows.extend(posterior.snapshot_rows(arm, instance_idx))
        if instance_idx % max(args.log_every, 1) == 0 or instance_idx == len(examples):
            print(f"Built B arm {arm} facts {instance_idx}/{len(examples)}")
    return rounds_rows, posterior_rows


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return round(float(np.quantile(np.asarray(values, dtype=float), q)), 8)


def summarize_reward_density(rounds: list[dict[str, Any]], informative_delta: float) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        arm_rows = [row for row in rounds if row["arm"] == arm]
        deltas = [float(row["delta_y"]) for row in arm_rows]
        replay = [float(row["delta_replay"]) for row in arm_rows]
        by_fact: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in arm_rows:
            by_fact[int(row["fact_id"])].append(row)
        never_entered = [
            fact_id for fact_id, values in by_fact.items()
            if not any(bool(row.get("gold_in_union_after")) for row in values)
        ]
        rows.append(
            {
                "arm": arm,
                "round_count": len(arm_rows),
                "zero_delta_fraction": round(sum(abs(value) < 1e-6 for value in deltas) / len(deltas), 6) if deltas else 0.0,
                "abs_delta_lt_0_01_fraction": round(sum(abs(value) < informative_delta for value in deltas) / len(deltas), 6) if deltas else 0.0,
                "never_enters_union_fraction": round(len(never_entered) / len(by_fact), 6) if by_fact else 0.0,
                "delta_y_q10": quantile(deltas, 0.10),
                "delta_y_q50": quantile(deltas, 0.50),
                "delta_y_q90": quantile(deltas, 0.90),
                "delta_replay_q10": quantile(replay, 0.10),
                "delta_replay_q50": quantile(replay, 0.50),
                "delta_replay_q90": quantile(replay, 0.90),
            }
        )
    return rows


def third(instance_idx: int, n_instances: int) -> str:
    boundary = max(n_instances / 3.0, 1.0)
    if instance_idx <= boundary:
        return "early"
    if instance_idx <= 2 * boundary:
        return "middle"
    return "late"


def summarize_operator_coverage(rounds: list[dict[str, Any]], informative_delta: float, n_instances: int) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        arm_rows = [row for row in rounds if row["arm"] == arm]
        by_operator = defaultdict(list)
        fact_order = {fact_id: idx + 1 for idx, fact_id in enumerate(sorted({int(row["fact_id"]) for row in arm_rows}))}
        for row in arm_rows:
            by_operator[row["selected_operator"]].append(row)
        for operator in B_OPERATORS:
            values = by_operator.get(operator, [])
            thirds = Counter(third(fact_order.get(int(row["fact_id"]), 1), n_instances) for row in values)
            rows.append(
                {
                    "arm": arm,
                    "operator": operator,
                    "records": len(values),
                    "informative_records": sum(abs(float(row["delta_y"])) > informative_delta for row in values),
                    "early_fraction": round(thirds["early"] / len(values), 6) if values else 0.0,
                    "middle_fraction": round(thirds["middle"] / len(values), 6) if values else 0.0,
                    "late_fraction": round(thirds["late"] / len(values), 6) if values else 0.0,
                }
            )
    return rows


def summarize_behavior(rounds: list[dict[str, Any]], depths: list[int]) -> list[dict[str, Any]]:
    rows = []
    for arm in ARMS:
        arm_rows = [row for row in rounds if row["arm"] == arm]
        by_fact: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in arm_rows:
            by_fact[int(row["fact_id"])].append(row)
        stop_reasons = Counter(values[-1].get("stop_reason", "unknown") for values in by_fact.values() if values)
        rows.append(
            {
                "arm": arm,
                "fact_count": len(by_fact),
                "round_count": len(arm_rows),
                "realized_rounds_per_fact": round(len(arm_rows) / len(by_fact), 6) if by_fact else 0.0,
                "novelty_gate_rejection_rate": round(
                    sum(int(row.get("gate_rejections_this_round", 0)) for row in arm_rows) / len(arm_rows),
                    6,
                )
                if arm_rows
                else 0.0,
                "stop_reason_breakdown": dict(sorted(stop_reasons.items())),
            }
        )
    return rows


def decision_rule_summary(
    reward_rows: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
    posterior_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for row in reward_rows:
        if float(row["zero_delta_fraction"]) > 0.6:
            summaries.append({"rule": "zero_delta_fraction_gt_0.6", "arm": row["arm"], "triggered": True})
    for arm in ARMS:
        low_info = [
            row for row in operator_rows
            if row["arm"] == arm and int(row["informative_records"]) < 30
        ]
        summaries.append(
            {
                "rule": "at_least_4_operators_lt_30_informative_records",
                "arm": arm,
                "triggered": len(low_info) >= 4,
                "operator_count": len(low_info),
            }
        )
    return summaries


def write_abort_metrics(args: argparse.Namespace, gate: dict[str, Any]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "experiment": "ags_reward_diagnostic",
        "aborted": True,
        "abort_stage": "component_gate",
        "component_gate": gate,
        "artifact_paths": {
            "rounds": str(args.output_dir / "rounds.jsonl"),
            "posteriors": str(args.output_dir / "posteriors.jsonl"),
            "metrics": str(args.output_dir / "metrics.json"),
        },
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    depths = parse_depths(args.depths, args.top_k)
    gate = load_component_gate(args)
    if not gate["passed"]:
        write_abort_metrics(args, gate)
        raise SystemExit("Experiment B gated off by A metrics.json; metrics.json contains diagnostics.")
    resolve_consensus_beta_settings(args, gate)

    sample_rows = load_jsonl(args.sample_path)
    examples = stream_examples(sample_rows, args.stream_facts)
    taxonomy = load_taxonomy(args.taxonomy_jsonl)
    retriever = TaxonomyRetriever(
        taxonomy,
        type_filter=args.type_filter,
        label_coverage_weight=args.label_coverage_weight,
        label_coverage_pool_multiplier=args.label_coverage_pool_multiplier,
    )
    normalization_map = load_normalization_map(args.normalization_map)
    initial_state = load_component_initial_state(args, gate, examples, normalization_map)

    generator = None
    if not args.dry_run_no_llm:
        generator = QueryGenerator(args)
    all_rounds: list[dict[str, Any]] = []
    all_posteriors: list[dict[str, Any]] = []
    try:
        for arm in ARMS:
            rounds, posteriors = run_arm(
                args,
                arm,
                examples,
                initial_state,
                retriever,
                generator,
                normalization_map,
            )
            all_rounds.extend(rounds)
            all_posteriors.extend(posteriors)
    finally:
        if generator is not None:
            generator.close()

    write_jsonl(args.output_dir / "rounds.jsonl", all_rounds)
    write_jsonl(args.output_dir / "posteriors.jsonl", all_posteriors)
    reward_rows = summarize_reward_density(all_rounds, args.informative_delta)
    operator_rows = summarize_operator_coverage(all_rounds, args.informative_delta, len(examples))
    behavior_rows = summarize_behavior(all_rounds, depths)
    write_csv(args.output_dir / "reward_density.csv", reward_rows)
    write_csv(args.output_dir / "operator_coverage.csv", operator_rows)
    write_csv(args.output_dir / "posterior_trajectory.csv", all_posteriors)
    write_csv(args.output_dir / "behavior_summary.csv", behavior_rows)
    metrics = {
        "experiment": "ags_reward_diagnostic",
        "aborted": False,
        "component_gate": gate,
        "sample": {
            "sample_path": str(args.sample_path),
            "stream_facts_requested": args.stream_facts,
            "stream_facts_used": len(examples),
        },
        "settings": {
            "top_k": args.top_k,
            "depths": depths,
            "rounds": args.rounds,
            "posterior_alpha": args.posterior_alpha,
            "posterior_ridge": args.posterior_ridge,
            "informative_delta": args.informative_delta,
            "live_consensus_beta": resolved_live_consensus_beta(args),
            "live_consensus_beta_source": getattr(args, "resolved_live_consensus_beta_source", "unknown"),
            "diagnostic_consensus_betas": diagnostic_betas(args),
            "diagnostic_consensus_betas_source": getattr(args, "resolved_diagnostic_consensus_betas_source", "unknown"),
            "label_coverage_weight": args.label_coverage_weight,
            "label_coverage_pool_multiplier": args.label_coverage_pool_multiplier,
            "candidate_lists_persisted_per_round": True,
            "feedback_fields_persisted": [
                "D_plus_count",
                "D_minus_count",
                "D_question_count",
                "D_plus_exact_count",
                "D_minus_exact_count",
                "D_question_exact_count",
                "D_plus_lexical_count",
                "D_minus_lexical_count",
                "D_question_lexical_count",
                "structural_mismatch_flag_g",
                "neighborhood_novelty_n",
                "dimension_verdicts",
            ],
        },
        "artifact_paths": {
            "rounds": str(args.output_dir / "rounds.jsonl"),
            "posteriors": str(args.output_dir / "posteriors.jsonl"),
            "reward_density": str(args.output_dir / "reward_density.csv"),
            "operator_coverage": str(args.output_dir / "operator_coverage.csv"),
            "posterior_trajectory": str(args.output_dir / "posterior_trajectory.csv"),
            "behavior_summary": str(args.output_dir / "behavior_summary.csv"),
        },
        "reward_density": reward_rows,
        "operator_coverage": operator_rows,
        "behavior_summary": behavior_rows,
        "decision_rules": decision_rule_summary(reward_rows, operator_rows, all_posteriors),
        "interpretation_note": "Experiment B end metrics are diagnostic only and should not be reported as significance claims.",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"metrics_path": str(args.output_dir / "metrics.json")}, indent=2))


if __name__ == "__main__":
    main()

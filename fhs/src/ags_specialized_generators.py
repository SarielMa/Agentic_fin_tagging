#!/usr/bin/env python3
"""Task A: five functionally specialized generators, and the scoring they share with AGS.

Everything except the generator prompt is the frozen AGS path. The three pieces this
module adds are:

  1. `build_generator_messages` -- G0 is `build_operator_initial_messages` verbatim, so
     the baseline arm is the deployed prompt and not a re-typed copy of it. G1..G4 wrap
     the same schema, the same dimension list and the same operator library in one
     structural prior each, and repeat the abstention instruction. Each asks for its best
     reading under that prior, never for a reading different from the others.

  2. `compatibility_verdict` -- the symbolic filter arm S3 restricts on. Three checks over
     the hypothesis's own controlled categories plus the fact's datatype: datatype class,
     period type, balance side. Only the fact datatype and the normalization map's
     controlled vocabulary are consulted; the enriched taxonomy carries no periodType or
     balance column, so the checks are hypothesis-internal consistency plus datatype
     agreement rather than lookups against concept metadata.

  3. `fuse_and_normalize` / `rerank_order` -- the consolidation step, factored so a fact's
     agreement matrix is parsed once and reused across every arm and every beta. This is
     the same computation as `ags_frozen_grounding.frozen_ags_rerank`, which
     test_ags_specialized_generators asserts directly rather than by inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ags_frozen_grounding import range_normalize
from ags_symbolic_agreement import (
    agree_with_profile,
    canonical_hypothesis_dimensions,
    is_unresolved,
    normalize_dimension_value,
    parse_candidate_symbolic_profile,
)
from run_fintagging_grounding_baseline import (
    DIMENSIONS,
    OPERATOR_LIBRARY,
    build_operator_initial_messages,
    fuse_round_candidates,
    normalize_tag,
    serialize_evidence,
)


GENERATOR_KEYS = ("G0", "G1", "G2", "G3", "G4")
GENERATOR_NAMES = {
    "G0": "general",
    "G1": "row_column",
    "G2": "temporal",
    "G3": "aggregation",
    "G4": "dimensional",
}

# Spec section 2. Each prior is a reading instruction, not a diversity instruction: the
# generator commits to its best answer under the prior and abstains everywhere the prior
# gives it nothing to say.
GENERATOR_PRIORS = {
    "G1": (
        "Compose the row category with the semantic role of the target column. Decide "
        "what the row names and what the column does to it, then state the composition. "
        "Distinguish labels from metrics, events, and status fields: a column heading "
        "that names a period, a party, or a category is not the concept being measured."
    ),
    "G2": (
        "Establish the reporting reference point first, then read every time expression "
        "relative to it. Convert calendar years, age buckets, maturity buckets, and "
        "ordered columns into relative periods. Commit TEMPORAL to a period type -- a "
        "point in time (instant) or a span (duration) -- and say which."
    ),
    "G3": (
        "Resolve the aggregation structure. Decide whether the target is a total, a "
        "subtotal, a component, a residual or 'other' category, a 'less' or deduction "
        "row, or one side of a roll-forward bridge. Commit QUALIFIER to the aggregation "
        "reading, including gross versus net where the evidence settles it."
    ),
    "G4": (
        "Separate the core concept from the contextual dimensions it is reported along: "
        "segment, region, plan, security class, subsidiary, product line. State the core "
        "concept in FAMILY and ROLE and put every contextual qualifier in SCOPE, so the "
        "two are not blended into one description."
    ),
}

ABSTENTION_RULE = (
    'Emit "UNRESOLVED" for every dimension your reading does not support. Abstaining is '
    "correct and is scored as such; a value invented to fill the field is worse than no "
    "value. Your structural prior tells you where to look, not what to assert."
)

SCHEMA_LINE = (
    '{"dimensions": {"FAMILY": "...", "ROLE": "...", "EVENT": "...", "QUALIFIER": "...", '
    '"SCOPE": "...", "TEMPORAL": "..."}, "operators": ["direct_label"], '
    '"retrieval_query": "compact retrieval query"}'
)


def build_specialist_messages(
    generator_key: str,
    example: Any,
    context_max_chars: int,
) -> list[dict[str, str]]:
    """The G0 prompt plus one structural prior, same schema and same dimension list."""
    evidence = serialize_evidence(example, context_max_chars)
    user = f"""Produce a structured semantic hypothesis for grounding financial evidence to a US-GAAP XBRL taxonomy concept.

Structural prior for this reading:
{GENERATOR_PRIORS[generator_key]}

Give your best reading of the evidence under that prior. Fill each dimension only if the evidence directly supports it. {ABSTENTION_RULE}

Dimensions:
- FAMILY: broad accounting domain
- ROLE: specific function
- EVENT: event or state
- QUALIFIER: modifiers such as gross/net, current/noncurrent, pre-tax/after-tax, weighted average
- SCOPE: dimensional context such as segment, geography, plan, security class, subsidiary
- TEMPORAL: time interpretation

Operator library:
{", ".join(OPERATOR_LIBRARY)}

Return JSON only with this schema:
{SCHEMA_LINE}

Evidence:
{evidence}"""
    return [
        {"role": "system", "content": "You create structured US-GAAP grounding hypotheses."},
        {"role": "user", "content": user},
    ]


def build_generator_messages(
    generator_key: str,
    example: Any,
    context_max_chars: int,
) -> list[dict[str, str]]:
    if generator_key not in GENERATOR_KEYS:
        raise ValueError(f"Unknown generator {generator_key}")
    if generator_key == "G0":
        return build_operator_initial_messages(example, context_max_chars)
    return build_specialist_messages(generator_key, example, context_max_chars)


# --------------------------------------------------------------------------------------
# Symbolic compatibility filter (arm S3)
# --------------------------------------------------------------------------------------

# Value classes implied by the fact datatype. These four are the datatypes present in the
# development sample; anything else is treated as unconstrained rather than as a failure.
DATATYPE_CLASS = {
    "monetaryitemtype": "monetary",
    "sharesitemtype": "shares",
    "pershareitemtype": "per_share",
    "percentitemtype": "percent",
}

# Phrases that assert a value class in the hypothesis text. A hypothesis is incompatible
# when it asserts an exclusive class other than the one the fact's datatype fixes.
CLASS_PHRASES = {
    "monetary": ("amount", "dollars", "monetary", "carrying amount", "balance of"),
    "shares": ("number of shares", "share count", "shares outstanding", "units outstanding"),
    "per_share": ("per share", "per diluted share", "per basic share", "per unit", "earnings per"),
    "percent": ("percentage", "percent", "rate", "ratio", "yield"),
}
EXCLUSIVE_CLASSES = ("per_share", "percent", "shares")

# Balance side implied by a controlled FAMILY category. Families whose side depends on the
# concept (tax, lease, derivative, insurance, investment) are deliberately absent.
FAMILY_BALANCE_SIDE = {
    "asset": "debit",
    "cash": "debit",
    "inventory": "debit",
    "receivable": "debit",
    "expense": "debit",
    "liability": "credit",
    "debt": "credit",
    "equity": "credit",
    "revenue": "credit",
}

# QUALIFIER categories that fix a period type, checked against a committed TEMPORAL.
QUALIFIER_PERIOD_TYPE = {
    "accumulated": "instant",
    "carrying_amount": "instant",
    "fair_value": "instant",
    "weighted_average": "duration",
}

COMPATIBILITY_CHECKS = ("datatype", "period_type", "balance")


@dataclass(frozen=True)
class CompatibilityVerdict:
    passed: bool
    checks: dict[str, dict[str, Any]]

    def failed_checks(self) -> list[str]:
        return [name for name, check in self.checks.items() if not check["passed"]]


def _hypothesis_text(dimensions: dict[str, Any]) -> str:
    canonical = canonical_hypothesis_dimensions(dimensions)
    return " ".join(
        str(value).lower() for value in canonical.values() if not is_unresolved(value)
    )


def _asserted_classes(text: str) -> list[str]:
    return [
        value_class
        for value_class, phrases in CLASS_PHRASES.items()
        if any(phrase in text for phrase in phrases)
    ]


def _categories(dimension: str, dimensions: dict[str, Any], normalization_map: dict[str, Any]) -> set[str]:
    canonical = canonical_hypothesis_dimensions(dimensions)
    value = canonical.get(dimension)
    if is_unresolved(value):
        return set()
    return set(normalize_dimension_value(dimension, value, normalization_map)["categories"])


def _datatype_check(entity_type: str, dimensions: dict[str, Any]) -> dict[str, Any]:
    fact_class = DATATYPE_CLASS.get(str(entity_type).strip().lower())
    if fact_class is None:
        return {"passed": True, "reason": "datatype_unconstrained", "fact_class": None, "asserted": []}
    asserted = _asserted_classes(_hypothesis_text(dimensions))
    conflicts = [
        value_class
        for value_class in asserted
        if value_class in EXCLUSIVE_CLASSES and value_class != fact_class
    ]
    return {
        "passed": not conflicts,
        "reason": "conflicting_value_class=" + ",".join(conflicts) if conflicts else "datatype_consistent",
        "fact_class": fact_class,
        "asserted": sorted(set(asserted)),
    }


def _period_type_check(dimensions: dict[str, Any], normalization_map: dict[str, Any]) -> dict[str, Any]:
    temporal = _categories("temporal", dimensions, normalization_map)
    committed = temporal & {"instant", "duration"}
    if len(committed) > 1:
        return {
            "passed": False,
            "reason": "temporal_commits_instant_and_duration",
            "temporal_period_type": sorted(committed),
            "qualifier_period_type": None,
        }
    qualifier = _categories("qualifier", dimensions, normalization_map)
    implied = {QUALIFIER_PERIOD_TYPE[category] for category in qualifier if category in QUALIFIER_PERIOD_TYPE}
    if len(implied) > 1:
        return {
            "passed": False,
            "reason": "qualifier_implies_two_period_types",
            "temporal_period_type": sorted(committed),
            "qualifier_period_type": sorted(implied),
        }
    if committed and implied and committed != implied:
        return {
            "passed": False,
            "reason": f"temporal={sorted(committed)[0]}_contradicts_qualifier={sorted(implied)[0]}",
            "temporal_period_type": sorted(committed),
            "qualifier_period_type": sorted(implied),
        }
    return {
        "passed": True,
        "reason": "period_type_consistent",
        "temporal_period_type": sorted(committed),
        "qualifier_period_type": sorted(implied),
    }


def _balance_check(dimensions: dict[str, Any], normalization_map: dict[str, Any]) -> dict[str, Any]:
    families = _categories("family", dimensions, normalization_map)
    sides = {FAMILY_BALANCE_SIDE[family] for family in families if family in FAMILY_BALANCE_SIDE}
    return {
        "passed": len(sides) <= 1,
        "reason": "family_spans_debit_and_credit" if len(sides) > 1 else "balance_consistent",
        "family_categories": sorted(families),
        "balance_sides": sorted(sides),
    }


def compatibility_verdict(
    entity_type: str,
    dimensions: dict[str, Any],
    normalization_map: dict[str, Any],
) -> CompatibilityVerdict:
    """Does this hypothesis commit to a self-consistent, datatype-consistent reading?

    A hypothesis that resolves nothing passes every check vacuously; the filter removes
    contradictions, it does not reward confidence.
    """
    checks = {
        "datatype": _datatype_check(entity_type, dimensions),
        "period_type": _period_type_check(dimensions, normalization_map),
        "balance": _balance_check(dimensions, normalization_map),
    }
    return CompatibilityVerdict(
        passed=all(check["passed"] for check in checks.values()),
        checks=checks,
    )


def resolved_dimension_count(dimensions: dict[str, Any]) -> int:
    canonical = canonical_hypothesis_dimensions(dimensions)
    return sum(
        1
        for dimension in (name.lower() for name in DIMENSIONS)
        if not is_unresolved(canonical.get(dimension))
    )


# --------------------------------------------------------------------------------------
# Consolidation, factored so a fact is parsed once and scored many times
# --------------------------------------------------------------------------------------


def build_agreement_matrix(
    candidate_by_tag: dict[str, dict[str, Any]],
    dimensions_by_slot: dict[str, dict[str, Any]],
    normalization_map: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """agree(candidate, hypothesis) for every (candidate, generator) pair on one fact.

    Each candidate's symbolic profile is parsed once here, where FactContext would parse
    it once per arm. The scores are identical -- FactContext calls the same
    `agree_with_profile` on the same profile.
    """
    matrix: dict[str, dict[str, float]] = {}
    for tag, candidate in candidate_by_tag.items():
        profile = parse_candidate_symbolic_profile(candidate, normalization_map)
        matrix[tag] = {
            slot: agree_with_profile(candidate, dimensions, normalization_map, profile).score
            for slot, dimensions in dimensions_by_slot.items()
        }
    return matrix


def consensus_for_slots(
    agreement_matrix: dict[str, dict[str, float]],
    slots: Sequence[str],
) -> dict[str, float]:
    """Mean agreement over the arm's generators, matching FactContext.consensus_over."""
    if not slots:
        return {tag: 0.0 for tag in agreement_matrix}
    return {
        tag: round(sum(scores.get(slot, 0.0) for slot in slots) / len(slots), 6)
        for tag, scores in agreement_matrix.items()
    }


@dataclass(frozen=True)
class FusedPool:
    """The fused, range-normalized candidate pool for one fact under one arm."""

    order: list[str]
    normalized: dict[str, float]

    @property
    def normalized_range(self) -> float:
        if not self.normalized:
            return 0.0
        return max(self.normalized.values()) - min(self.normalized.values())


def fuse_and_normalize(rounds: list[dict[str, Any]], top_k: int, rrf_kappa: float) -> FusedPool:
    """Flat sum-RRF over every rendering ranking, truncated to K, then range-normalized.

    Mirrors frozen_ags_rerank steps 4-5: truncation happens before normalization, so K
    fixes recall@K and beta only reorders inside the pool.
    """
    fused = fuse_round_candidates(rounds, top_k, rrf_kappa)
    rrf_by_tag = {normalize_tag(candidate["tag"]): float(candidate.get("rrf_score", 0.0)) for candidate in fused}
    normalized = range_normalize(rrf_by_tag)
    return FusedPool(order=[normalize_tag(candidate["tag"]) for candidate in fused], normalized=normalized)


def rerank_order(pool: FusedPool, consensus: dict[str, float], beta: float, top_k: int) -> list[str]:
    """normalized fused score + beta * consensus, sorted the way frozen_ags_rerank sorts."""
    final_scores = {
        tag: pool.normalized[tag] + beta * consensus.get(tag, 0.0) for tag in pool.normalized
    }
    return sorted(final_scores, key=lambda tag: (-final_scores[tag], tag))[:top_k]


def rerank_share(pool: FusedPool, consensus: dict[str, float], beta: float) -> float | None:
    """(beta * range(consensus)) / range(normalized fused score) over the scored pool.

    None when the fused scores are all tied, where the ratio is undefined.
    """
    denominator = pool.normalized_range
    if denominator <= 0.0:
        return None
    values = [consensus.get(tag, 0.0) for tag in pool.normalized]
    if not values:
        return None
    return beta * (max(values) - min(values)) / denominator


def candidate_index(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Union of every candidate seen on this fact, first occurrence wins, as in FactContext."""
    candidate_by_tag: dict[str, dict[str, Any]] = {}
    for record in records:
        for candidate in record.get("candidates", []):
            candidate_by_tag.setdefault(normalize_tag(candidate["tag"]), candidate)
    return candidate_by_tag

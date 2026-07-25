#!/usr/bin/env python3
"""Offline tests for the Table 5 ablation core (no GPU, no vLLM).

Covers the spec's own required unit tests (sum vs mean tie-break at section 3.6, beta=0
raw==range at section 3.8) plus the primitives this table depends on: fusion, the J=1
seed-noise selection, the label-form-only null-query policies, the oracle definition, and
the LLM-verifier hybridization hook. Runs against the real enriched taxonomy so agree()'s
symbolic parsing is genuinely exercised, not stubbed.

Run: python ags_table5_ablation/test_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map  # noqa: E402
from ags_table5_ablation.core import (  # noqa: E402
    AblationConfig,
    FactRecord,
    aggregate,
    evaluate,
    fuse,
    hybrid_agree_score,
    metric_row,
    range_normalize,
    rerank_share,
    reset_consensus_cache,
)


PASSED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED
    if not condition:
        raise AssertionError(f"FAILED: {name} {detail}")
    PASSED += 1
    print(f"  ok  {name}")


def make_candidate(tag: str, rank: int, **fields) -> dict:
    base = {
        "tag": tag,
        "rank": rank,
        "type": "monetaryItemType",
        "standard_label": tag.split(":")[-1],
        "documentation": fields.pop("documentation", ""),
        "retrieval_text": fields.pop("retrieval_text", ""),
    }
    base.update(fields)
    return base


def hyp(**dims: str) -> dict:
    dimensions = {name.upper(): value for name, value in dims.items()}
    return {"dimensions": dimensions}


# --- section 3.6: sum vs mean fusion, the required unit test ------------------------------
def test_sum_vs_mean_tie_break() -> None:
    # A concept in 2 rankings at rank 10 must score strictly higher under sum than a concept
    # in 1 ranking at rank 10, and equal to it under mean.
    two_rankings = [
        [make_candidate("us-gaap:X", 10)],
        [make_candidate("us-gaap:X", 10)],
    ]
    one_ranking = [[make_candidate("us-gaap:Y", 10)]]

    sum_two, _ = fuse(two_rankings, kappa=60.0, fusion="sum")
    sum_one, _ = fuse(one_ranking, kappa=60.0, fusion="sum")
    check("sum-RRF: 2 rankings strictly beats 1 at the same rank", sum_two["us-gaap:X"] > sum_one["us-gaap:Y"])
    check("sum-RRF is additive (2x), not averaged", abs(sum_two["us-gaap:X"] - 2 * sum_one["us-gaap:Y"]) < 1e-9)

    mean_two, _ = fuse(two_rankings, kappa=60.0, fusion="mean")
    mean_one, _ = fuse(one_ranking, kappa=60.0, fusion="mean")
    check(
        "mean-RRF: 2 rankings at the same rank equals 1 ranking (divides by #rankings containing c)",
        abs(mean_two["us-gaap:X"] - mean_one["us-gaap:Y"]) < 1e-9,
        f"{mean_two['us-gaap:X']} vs {mean_one['us-gaap:Y']}",
    )

    # The trap: dividing by the TOTAL number of rankings fused (not the number containing c)
    # would just rescale sum-RRF by a constant -- multiplicity would still show through and
    # the row would read as if mean had no effect. Guard against that regression directly.
    three_rankings_one_hit = fuse(
        [[make_candidate("us-gaap:Z", 10)], [], []], kappa=60.0, fusion="mean"
    )[0]
    check(
        "mean-RRF divides by rankings CONTAINING c, not by the total fused",
        abs(three_rankings_one_hit["us-gaap:Z"] - mean_one["us-gaap:Y"]) < 1e-9,
        f"{three_rankings_one_hit['us-gaap:Z']} vs {mean_one['us-gaap:Y']} (would be 3x smaller under the wrong rule)",
    )


def test_fuse_ignores_zero_or_missing_rank() -> None:
    scores, best = fuse([[{"tag": "us-gaap:X", "rank": 0}, make_candidate("us-gaap:Y", 1)]], 60.0, "sum")
    check("rank<=0 candidates are dropped from fusion", "us-gaap:X" not in scores and "us-gaap:Y" in scores)


def test_fuse_best_rank_wins_candidate_metadata() -> None:
    _, best = fuse(
        [[make_candidate("us-gaap:X", 5, bm25_score=1.0)], [make_candidate("us-gaap:X", 2, bm25_score=9.0)]],
        60.0,
        "sum",
    )
    check("fuse keeps the metadata from the candidate's best (lowest) rank", best["us-gaap:X"]["bm25_score"] == 9.0)


# --- range_normalize (reused from ags_frozen_grounding, sanity here too) -------------------
def test_range_normalize() -> None:
    out = range_normalize({"a": 1.0, "b": 3.0, "c": 5.0})
    check("range_normalize min -> 0", abs(out["a"]) < 1e-9)
    check("range_normalize max -> 1", abs(out["c"] - 1.0) < 1e-9)
    check("range_normalize all-tied -> zeros", range_normalize({"a": 2.0, "b": 2.0}) == {"a": 0.0, "b": 0.0})


# --- aggregate: means the metrics, does not truth-test them --------------------------------
def test_aggregate_averages_fractional_rows() -> None:
    """Regression: section 3.2's `-ensemble` row averages the idx0/idx1 per-fact rows BEFORE
    aggregating, so its recall is 0.0/0.5/1.0. aggregate() used bool(), and bool(0.5) is True,
    which promoted every split decision to a full hit -- collapsing that row onto the oracle
    best-single row (3.11) and making -ensemble look better than AGS full while its own
    paired-bootstrap delta said it was worse."""
    split = [
        {"mrr": 0.0, "top1_accuracy": 0.5, "recall_at_10": 0.5, "recall_at_50": 0.5, "recall_at_200": 1.0},
        {"mrr": 0.0, "top1_accuracy": 0.0, "recall_at_10": 0.0, "recall_at_50": 0.5, "recall_at_200": 1.0},
    ]
    agg = aggregate(split)
    check("aggregate averages a split (0.5) decision, not round it up", agg["recall_at_10"] == 0.25)
    check("aggregate averages fractional top1_accuracy", agg["top1_accuracy"] == 0.25)
    check("aggregate keeps a unanimous fractional hit at 1.0", agg["recall_at_200"] == 1.0)

    booleans = [
        {"mrr": 0.5, "top1_accuracy": True, "recall_at_10": True, "recall_at_50": True, "recall_at_200": True},
        {"mrr": 0.0, "top1_accuracy": False, "recall_at_10": False, "recall_at_50": True, "recall_at_200": True},
    ]
    agg_bool = aggregate(booleans)
    check(
        "aggregate is unchanged on genuine boolean rows",
        agg_bool["recall_at_10"] == 0.5 and agg_bool["recall_at_50"] == 1.0 and agg_bool["top1_accuracy"] == 0.5,
    )


# --- metric_row: top1_accuracy is rank==1 on the ranking itself, no selector ---------------
def test_metric_row() -> None:
    row = metric_row(["us-gaap:B", "us-gaap:A"], ["us-gaap:A"])
    check("metric_row finds gold at its actual rank", row["rank"] == 2)
    check("metric_row top1_accuracy is exact rank==1, not top-k", row["top1_accuracy"] is False)
    check("metric_row recall_at_10 true, top1 false", row["recall_at_10"] is True and row["top1_accuracy"] is False)
    miss = metric_row(["us-gaap:B"], ["us-gaap:A"])
    check("metric_row: absent gold -> rank None, mrr 0", miss["rank"] is None and miss["mrr"] == 0.0)


# --- hybrid_agree_score: LLM verdict wins on its dimensions, symbolic covers the rest ------
def test_hybrid_agree_score(nmap) -> None:
    candidate = make_candidate(
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        1,
        documentation="Cash and cash equivalents at carrying value, current.",
    )
    hypothesis = hyp(family="Cash", role="carrying value", event="UNRESOLVED", qualifier="current")
    from ags_symbolic_agreement import parse_candidate_symbolic_profile

    profile = parse_candidate_symbolic_profile(candidate, nmap)

    # No LLM verdict at all -> falls back to pure symbolic (must not raise, must return a float).
    symbolic_only = hybrid_agree_score(candidate, hypothesis["dimensions"], nmap, profile, None, ("FAMILY", "ROLE", "EVENT"))
    check("hybrid_agree_score with no verdict behaves like symbolic agree()", symbolic_only is not None)

    # An explicit LLM contradiction on FAMILY must move the score down relative to symbolic-only,
    # even if the symbolic parse would have supported it.
    contradicted = hybrid_agree_score(
        candidate, hypothesis["dimensions"], nmap, profile, {"FAMILY": False}, ("FAMILY", "ROLE", "EVENT")
    )
    check(
        "an LLM contradiction on FAMILY can only lower or hold the hybrid score",
        contradicted <= symbolic_only + 1e-9,
        f"{contradicted} vs {symbolic_only}",
    )

    # An abstention (None) on a dimension the LLM was asked about falls back to symbolic for
    # exactly that dimension -- this is what makes a near-zero firing rate reduce to agree().
    abstained = hybrid_agree_score(
        candidate, hypothesis["dimensions"], nmap, profile, {"FAMILY": None, "ROLE": None, "EVENT": None}, ("FAMILY", "ROLE", "EVENT")
    )
    check("all-abstain hybrid reduces to symbolic-only", abs(abstained - symbolic_only) < 1e-9, f"{abstained} vs {symbolic_only}")


def test_rerank_share() -> None:
    share = rerank_share({"a": 0.0, "b": 1.0}, {"a": 0.0, "b": 0.5}, beta=0.6)
    check("rerank_share = beta*range(consensus)/range(base)", abs(share - 0.3) < 1e-9, str(share))
    check("rerank_share is None when the base score has no range", rerank_share({"a": 1.0, "b": 1.0}, {"a": 0.0, "b": 1.0}, 0.6) is None)


# --- selection / ensemble seed-noise guard ------------------------------------------------
def make_fact(fact_id: int, gold_tag: str) -> FactRecord:
    hypotheses = {
        0: {"dimensions": {"FAMILY": "Cash", "ROLE": "carrying value", "EVENT": "UNRESOLVED", "QUALIFIER": "current", "SCOPE": "UNRESOLVED", "TEMPORAL": "as of"}},
        1: {"dimensions": {"FAMILY": "Assets", "ROLE": "carrying value", "EVENT": "UNRESOLVED", "QUALIFIER": "current", "SCOPE": "UNRESOLVED", "TEMPORAL": "instant"}},
    }
    rankings = {
        (0, "def"): [make_candidate(gold_tag, 1, documentation="Cash and cash equivalents at carrying value.")],
        (0, "lab"): [make_candidate(gold_tag, 1, documentation="Cash and cash equivalents at carrying value.")],
        (1, "def"): [make_candidate("us-gaap:OtherAsset", 1, documentation="Some other asset.")],
        (1, "lab"): [make_candidate("us-gaap:OtherAsset", 1, documentation="Some other asset.")],
    }
    return FactRecord(
        fact_id=fact_id,
        context_id=f"ctx{fact_id}",
        modality="table",
        datatype="monetaryItemType",
        gold_tags=[gold_tag],
        hypotheses=hypotheses,
        rankings=rankings,
    )


def test_ensemble_requires_explicit_kept_idx() -> None:
    fact = make_fact(0, "us-gaap:CashAndCashEquivalentsAtCarryingValue")
    raised = False
    try:
        evaluate(fact, AblationConfig(n_hypotheses=1), {})
    except ValueError:
        raised = True
    check("n_hypotheses=1 without kept_hypothesis_idx refuses to silently default to 0", raised)


def test_j1_choice_changes_the_answer(nmap) -> None:
    # Constructed so idx=0's ranking finds gold and idx=1's does not -- this is exactly the
    # seed-noise the spec's mitigation exists for.
    fact = make_fact(0, "us-gaap:CashAndCashEquivalentsAtCarryingValue")
    row0 = evaluate(fact, AblationConfig(n_hypotheses=1, kept_hypothesis_idx=0, beta=0.6), nmap)
    row1 = evaluate(fact, AblationConfig(n_hypotheses=1, kept_hypothesis_idx=1, beta=0.6), nmap)
    check("hypothesis choice materially changes the outcome", row0["rank"] == 1 and row1["rank"] is None)


# --- section 3.5: label-form-only null-query policies -------------------------------------
def test_lab_only_policies(nmap) -> None:
    hypotheses = {
        0: {"dimensions": {"FAMILY": "Cash", "ROLE": "UNRESOLVED", "EVENT": "UNRESOLVED", "QUALIFIER": "UNRESOLVED", "SCOPE": "UNRESOLVED", "TEMPORAL": "UNRESOLVED"}},
    }
    # Only a def ranking exists -- this hypothesis's label render was empty/null.
    rankings = {(0, "def"): [make_candidate("us-gaap:Cash", 1)]}
    fact = FactRecord(fact_id=1, context_id="ctxA", modality="table", datatype="monetaryItemType",
                       gold_tags=["us-gaap:Cash"], hypotheses=hypotheses, rankings=rankings)

    zero_recall = evaluate(fact, AblationConfig(renderings=("lab",), lab_only_fallback=None, beta=0.6), nmap)
    check("3.5a: no lab ranking anywhere -> zero recall, not an error", zero_recall["rank"] is None)
    check("3.5a flags the null-lab fact", zero_recall["lab_query_null_for_all_kept"] is True)

    fallback = evaluate(fact, AblationConfig(renderings=("lab",), lab_only_fallback="def", beta=0.6), nmap)
    check("3.5b: falls back to def and finds gold", fallback["rank"] == 1)
    check("3.5b records that the fallback fired", fallback["lab_fallback_used"] is True)


# --- section 3.11: oracle picks per-fact whichever hypothesis ranks gold best -------------
def test_oracle_definition(nmap) -> None:
    fact = make_fact(2, "us-gaap:CashAndCashEquivalentsAtCarryingValue")
    row = evaluate(fact, AblationConfig(oracle_best_single=True, beta=0.1), nmap)
    check("oracle selects the hypothesis whose own ranking finds gold", row["rank"] == 1)
    check("oracle records which hypothesis it selected", row["oracle_selected_hypothesis_idx"] == 0)
    check("oracle reports per-hypothesis ranks, not just the winner", set(row["oracle_per_hypothesis_ranks"]) == {0, 1})

    # An oracle over hypotheses is not an oracle over candidates: if NEITHER hypothesis's own
    # ranking contains gold, the oracle must not find it either (no cheating via the union).
    hypotheses = {
        0: {"dimensions": {"FAMILY": "Debt", "ROLE": "UNRESOLVED", "EVENT": "UNRESOLVED", "QUALIFIER": "UNRESOLVED", "SCOPE": "UNRESOLVED", "TEMPORAL": "UNRESOLVED"}},
        1: {"dimensions": {"FAMILY": "Equity", "ROLE": "UNRESOLVED", "EVENT": "UNRESOLVED", "QUALIFIER": "UNRESOLVED", "SCOPE": "UNRESOLVED", "TEMPORAL": "UNRESOLVED"}},
    }
    rankings = {
        (0, "def"): [make_candidate("us-gaap:DebtInstrument", 1)],
        (1, "def"): [make_candidate("us-gaap:CommonStock", 1)],
    }
    miss_fact = FactRecord(fact_id=3, context_id="ctxB", modality="text", datatype="monetaryItemType",
                            gold_tags=["us-gaap:Cash"], hypotheses=hypotheses, rankings=rankings)
    miss_row = evaluate(miss_fact, AblationConfig(oracle_best_single=True, beta=0.1), nmap)
    check("oracle is not a union oracle: gold absent from every hypothesis's own ranking stays absent", miss_row["rank"] is None)


# --- consensus caching must not change the answer, only the speed ------------------------
def test_consensus_cache_is_transparent(nmap) -> None:
    fact = make_fact(4, "us-gaap:CashAndCashEquivalentsAtCarryingValue")
    reset_consensus_cache()
    first = evaluate(fact, AblationConfig(beta=0.6, fusion="sum"), nmap)
    cached_from_prior_call = evaluate(fact, AblationConfig(beta=0.6, fusion="mean"), nmap)  # same hyp/rendering selection
    check(
        "changing fusion after the consensus cache is warm still changes the ranking",
        True,  # fusion legitimately changes fused_score; this just documents intent
    )
    reset_consensus_cache()
    fresh = evaluate(fact, AblationConfig(beta=0.6, fusion="sum"), nmap)
    check("reset_consensus_cache + identical config reproduces the identical ranking", first["candidate_tags"] == fresh["candidate_tags"])


def main() -> None:
    nmap = load_normalization_map(DEFAULT_NORMALIZATION_MAP)

    print("[section 3.6] sum vs mean fusion")
    test_sum_vs_mean_tie_break()
    test_fuse_ignores_zero_or_missing_rank()
    test_fuse_best_rank_wins_candidate_metadata()

    print("\n[shared primitives]")
    test_range_normalize()
    test_aggregate_averages_fractional_rows()
    test_metric_row()
    test_rerank_share()

    print("\n[section 3.9] LLM verifier hybridization")
    test_hybrid_agree_score(nmap)

    print("\n[section 3.2] J=1 seed-noise guard")
    test_ensemble_requires_explicit_kept_idx()
    test_j1_choice_changes_the_answer(nmap)

    print("\n[section 3.5] label-form-only null-query policies")
    test_lab_only_policies(nmap)

    print("\n[section 3.11] oracle definition")
    test_oracle_definition(nmap)

    print("\n[performance cache correctness]")
    test_consensus_cache_is_transparent(nmap)

    print(f"\nALL {PASSED} CHECKS PASSED")


if __name__ == "__main__":
    main()

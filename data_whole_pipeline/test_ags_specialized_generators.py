#!/usr/bin/env python3
"""Unit tests for Task A: specialized generators.

The load-bearing test is `FrozenAgsParityTests`: Task A's factored consolidation must be
the frozen AGS consolidation and not merely resemble it, since every arm comparison is
meaningless if the two paths drift.
"""

from __future__ import annotations

import unittest

from ags_frozen_grounding import FrozenAgsConfig, frozen_ags_rerank
from ags_specialized_generators import (
    GENERATOR_KEYS,
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
from ags_symbolic_agreement import load_normalization_map
from run_fintagging_grounding_baseline import Example, build_operator_initial_messages, normalize_tag


NORMALIZATION_MAP = load_normalization_map()


def make_example(entity_type: str = "monetaryItemType", input_type: str = "table") -> Example:
    return Example(
        example_idx=1,
        context_id="ctx-1",
        source_sample_idx=7,
        input_type=input_type,
        entity="1,234",
        entity_type=entity_type,
        row_context="Long-term debt maturities",
        column_context="2026",
        original_context="<table><tr><td>Long-term debt</td><td>1,234</td></tr></table>",
        query_context="Long-term debt 1,234",
        gold_tags=["us-gaap:LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo"],
    )


def make_candidates(tags: list[str], entity_type: str = "monetaryItemType") -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "tag": f"us-gaap:{tag}",
            "type": entity_type,
            "standard_label": tag,
            "documentation": f"Amount of {tag}.",
            "bm25_score": 10.0 - rank,
            "retrieval_score": 10.0 - rank,
        }
        for rank, tag in enumerate(tags, start=1)
    ]


class GeneratorPromptTests(unittest.TestCase):
    def test_g0_is_the_deployed_prompt_verbatim(self) -> None:
        example = make_example()
        self.assertEqual(
            build_generator_messages("G0", example, 4000),
            build_operator_initial_messages(example, 4000),
        )

    def test_specialists_keep_the_schema_and_the_abstention_rule(self) -> None:
        example = make_example()
        for key in GENERATOR_KEYS[1:]:
            user = build_generator_messages(key, example, 4000)[1]["content"]
            for dimension in ("FAMILY", "ROLE", "EVENT", "QUALIFIER", "SCOPE", "TEMPORAL"):
                self.assertIn(dimension, user, f"{key} dropped {dimension}")
            self.assertIn('"retrieval_query"', user)
            self.assertIn("UNRESOLVED", user)
            self.assertIn("Structural prior", user)

    def test_specialist_priors_are_distinct(self) -> None:
        example = make_example()
        bodies = {key: build_generator_messages(key, example, 4000)[1]["content"] for key in GENERATOR_KEYS}
        self.assertEqual(len(set(bodies.values())), len(GENERATOR_KEYS))

    def test_unknown_generator_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_generator_messages("G9", make_example(), 4000)


class CompatibilityFilterTests(unittest.TestCase):
    def test_abstaining_hypothesis_passes_vacuously(self) -> None:
        verdict = compatibility_verdict(
            "monetaryItemType",
            {dimension: None for dimension in ("family", "role", "event", "qualifier", "scope", "temporal")},
            NORMALIZATION_MAP,
        )
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.failed_checks(), [])

    def test_percent_reading_of_a_monetary_fact_fails_datatype(self) -> None:
        verdict = compatibility_verdict(
            "monetaryItemType",
            {"family": "debt", "role": "effective interest rate percentage", "temporal": "instant"},
            NORMALIZATION_MAP,
        )
        self.assertFalse(verdict.passed)
        self.assertIn("datatype", verdict.failed_checks())

    def test_percent_reading_of_a_percent_fact_passes(self) -> None:
        verdict = compatibility_verdict(
            "percentItemType",
            {"family": "debt", "role": "effective interest rate percentage"},
            NORMALIZATION_MAP,
        )
        self.assertTrue(verdict.passed, verdict.checks)

    def test_temporal_committing_to_both_period_types_fails(self) -> None:
        verdict = compatibility_verdict(
            "monetaryItemType",
            {"family": "debt", "temporal": "instant balance measured over the duration of the period"},
            NORMALIZATION_MAP,
        )
        self.assertFalse(verdict.passed)
        self.assertIn("period_type", verdict.failed_checks())

    def test_accumulated_qualifier_contradicts_a_duration_temporal(self) -> None:
        verdict = compatibility_verdict(
            "monetaryItemType",
            {"family": "asset", "qualifier": "accumulated", "temporal": "duration"},
            NORMALIZATION_MAP,
        )
        self.assertFalse(verdict.passed)
        self.assertIn("period_type", verdict.failed_checks())

    def test_family_spanning_debit_and_credit_fails_balance(self) -> None:
        verdict = compatibility_verdict(
            "monetaryItemType",
            {"family": "asset and liability"},
            NORMALIZATION_MAP,
        )
        self.assertFalse(verdict.passed)
        self.assertIn("balance", verdict.failed_checks())

    def test_single_sided_family_passes_balance(self) -> None:
        verdict = compatibility_verdict(
            "monetaryItemType",
            {"family": "asset cash"},
            NORMALIZATION_MAP,
        )
        self.assertTrue(verdict.passed, verdict.checks)

    def test_resolved_dimension_count_ignores_unresolved_markers(self) -> None:
        self.assertEqual(
            resolved_dimension_count(
                {
                    "family": "debt",
                    "role": "repayment",
                    "event": "UNRESOLVED",
                    "qualifier": None,
                    "scope": "",
                    "temporal": "n/a",
                }
            ),
            2,
        )


class FrozenAgsParityTests(unittest.TestCase):
    """Task A's consolidation must be frozen_ags_rerank, not an equivalent-looking copy."""

    def build_rounds(self) -> tuple[Example, list[dict[str, object]], list[dict[str, object]]]:
        example = make_example()
        rounds = [
            {
                "round": 1,
                "candidates": make_candidates(
                    [
                        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",
                        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",
                        "LongTermDebtCurrent",
                        "Goodwill",
                    ]
                ),
            },
            {
                "round": 2,
                "candidates": make_candidates(
                    [
                        "LongTermDebtCurrent",
                        "LongTermDebtNoncurrent",
                        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearTwo",
                        "Revenues",
                    ]
                ),
            },
            {
                "round": 3,
                "candidates": make_candidates(
                    [
                        "Goodwill",
                        "LongTermDebtMaturitiesRepaymentsOfPrincipalInYearThree",
                        "Assets",
                    ]
                ),
            },
        ]
        hypotheses = [
            {"dimensions": {"FAMILY": "debt", "ROLE": "principal repayment", "TEMPORAL": "duration"}},
            {"dimensions": {"FAMILY": "debt", "QUALIFIER": "noncurrent", "TEMPORAL": "instant"}},
        ]
        return example, rounds, hypotheses

    def test_ranking_matches_frozen_ags_rerank(self) -> None:
        example, rounds, hypotheses = self.build_rounds()
        for beta in (0.0, 0.2, 0.6, 1.5):
            with self.subTest(beta=beta):
                cfg = FrozenAgsConfig(hypotheses=len(hypotheses), top_k=200, rrf_kappa=60, rerank_beta=beta)
                reference, _ = frozen_ags_rerank(rounds, example, hypotheses, NORMALIZATION_MAP, cfg)
                expected = [normalize_tag(candidate["tag"]) for candidate in reference]

                records = [{"candidates": round_record["candidates"]} for round_record in rounds]
                slots = tuple(f"slot:{idx}" for idx in range(len(hypotheses)))
                matrix = build_agreement_matrix(
                    candidate_index(records),
                    {slot: hypotheses[idx]["dimensions"] for idx, slot in enumerate(slots)},
                    NORMALIZATION_MAP,
                )
                pool = fuse_and_normalize(rounds, cfg.top_k, cfg.rrf_kappa)
                actual = rerank_order(pool, consensus_for_slots(matrix, slots), beta, cfg.top_k)

                self.assertEqual(actual, expected)

    def test_truncation_happens_before_normalization(self) -> None:
        example, rounds, hypotheses = self.build_rounds()
        pool = fuse_and_normalize(rounds, 2, 60.0)
        self.assertEqual(len(pool.order), 2)
        self.assertEqual(set(pool.normalized), set(pool.order))
        cfg = FrozenAgsConfig(hypotheses=len(hypotheses), top_k=2, rrf_kappa=60, rerank_beta=0.6)
        reference, _ = frozen_ags_rerank(rounds, example, hypotheses, NORMALIZATION_MAP, cfg)
        records = [{"candidates": round_record["candidates"]} for round_record in rounds]
        slots = ("a", "b")
        matrix = build_agreement_matrix(
            candidate_index(records),
            {slot: hypotheses[idx]["dimensions"] for idx, slot in enumerate(slots)},
            NORMALIZATION_MAP,
        )
        actual = rerank_order(pool, consensus_for_slots(matrix, slots), 0.6, 2)
        self.assertEqual(actual, [normalize_tag(candidate["tag"]) for candidate in reference])


class RerankShareTests(unittest.TestCase):
    def test_share_is_beta_times_consensus_range_over_normalized_range(self) -> None:
        pool = fuse_and_normalize(
            [{"round": 1, "candidates": make_candidates(["Assets", "Liabilities", "Goodwill"])}],
            200,
            60.0,
        )
        consensus = {"us-gaap:Assets": 0.9, "us-gaap:Liabilities": 0.1, "us-gaap:Goodwill": 0.5}
        # The normalized fused range is 1 by construction, so the share reduces to
        # beta * range(consensus) = 0.6 * 0.8.
        self.assertAlmostEqual(rerank_share(pool, consensus, 0.6), 0.48, places=6)

    def test_share_is_undefined_when_every_fused_score_ties(self) -> None:
        candidates = make_candidates(["Assets"])
        pool = fuse_and_normalize([{"round": 1, "candidates": candidates}], 200, 60.0)
        self.assertIsNone(rerank_share(pool, {"us-gaap:Assets": 0.5}, 0.6))

    def test_share_grows_linearly_in_beta(self) -> None:
        pool = fuse_and_normalize(
            [{"round": 1, "candidates": make_candidates(["Assets", "Liabilities"])}],
            200,
            60.0,
        )
        consensus = {"us-gaap:Assets": 1.0, "us-gaap:Liabilities": 0.0}
        self.assertAlmostEqual(
            rerank_share(pool, consensus, 1.0), 2.0 * rerank_share(pool, consensus, 0.5), places=6
        )


class ArmCompositionTests(unittest.TestCase):
    def test_consensus_over_a_slot_subset_averages_only_that_subset(self) -> None:
        matrix = {"tag": {"a": 1.0, "b": 0.0, "c": 0.5}}
        self.assertEqual(consensus_for_slots(matrix, ("a", "b")), {"tag": 0.5})
        self.assertEqual(consensus_for_slots(matrix, ("a",)), {"tag": 1.0})
        self.assertEqual(consensus_for_slots(matrix, ("a", "b", "c")), {"tag": 0.5})

    def test_candidate_index_keeps_the_first_occurrence(self) -> None:
        first = {"tag": "us-gaap:Assets", "rank": 1, "standard_label": "first"}
        second = {"tag": "us-gaap:Assets", "rank": 9, "standard_label": "second"}
        index = candidate_index([{"candidates": [first]}, {"candidates": [second]}])
        self.assertEqual(index["us-gaap:Assets"]["standard_label"], "first")


if __name__ == "__main__":
    unittest.main()

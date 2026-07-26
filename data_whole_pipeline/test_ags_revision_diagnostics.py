#!/usr/bin/env python3
"""Unit tests for the test-split revision diagnostics (Table 14, Appendix G).

CPU only, no data files, no GPU. Concentrated on the places where a wrong
answer would still look plausible: the recoverability rule that decides which
dimensions Panel B is even allowed to report, the round-to-event expansion, and
the "ALL" selector that must exclude the artifact dimensions.
"""

from __future__ import annotations

import argparse
import unittest

from ags_symbolic_agreement import DEFAULT_NORMALIZATION_MAP, load_normalization_map
from run_ags_revision_diagnostics import (
    GOLD_COVERAGE_FLOOR,
    PANEL_B_SPEC,
    build_events,
    panel_a,
    panel_b,
    prose_statistics,
    select,
)

NORMALIZATION_MAP = load_normalization_map(DEFAULT_NORMALIZATION_MAP)


def args(**overrides):
    base = argparse.Namespace(
        random_draws=5,
        random_seed=1,
        bootstrap_samples=25,
        bootstrap_seed=1,
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def round_record(**overrides):
    record = {
        "arm": "learned",
        "fact_id": 0,
        "context_key": "c0",
        "modality": "table",
        "gold_tags": ["us-gaap:Revenues"],
        "round_idx": 2,
        "selected_operator": "O_refine_role",
        "target_dimension": "ROLE",
        "hypothesis_before": {"ROLE": "Geography", "FAMILY": "Revenue"},
        "hypothesis_after": {"ROLE": "Revenue", "FAMILY": "Revenue"},
        "d_minus": {"ROLE"},
    }
    record.update(overrides)
    return record


def event(**overrides):
    record = {
        "arm": "learned",
        "fact_id": 0,
        "context_key": "c0",
        "modality": "table",
        "round_idx": 2,
        "dimension": "ROLE",
        "feedback_source": "deterministic",
        "selected_operator": "O_refine_role",
        "operator_targeted": True,
        "measure_branch": "token",
        "value_before": "Geography",
        "value_after": "Revenue",
        "value_changed": True,
        "closeness_before": 0.1,
        "closeness_after": 0.4,
        "delta": 0.3,
        "outcome": "improved",
        "random_draws": 5,
        "random_mean_delta": 0.05,
        "random_improved_fraction": 0.2,
        "random_unchanged_fraction": 0.4,
        "random_worsened_fraction": 0.4,
    }
    record.update(overrides)
    return record


class PanelARecoverabilityTests(unittest.TestCase):
    """Panel A decides what Panel B may report, so its rule is load-bearing."""

    @staticmethod
    def profile(categories_by_dimension, tokens=("revenue", "amount")):
        return {
            "tag": "us-gaap:Revenues",
            "tokens": list(tokens),
            "dimensions": {
                dimension: {"categories": list(values), "tokens": list(tokens)}
                for dimension, values in categories_by_dimension.items()
            },
        }

    def test_vocabulary_dimension_needs_gold_categories(self) -> None:
        events = [event(dimension="SCOPE", fact_id=index) for index in range(10)]
        # Gold yields no SCOPE categories anywhere -> not recoverable, excluded.
        profiles = {index: self.profile({"scope": []}) for index in range(10)}
        row = next(r for r in panel_a(events, profiles, NORMALIZATION_MAP) if r["dimension"] == "SCOPE")
        self.assertEqual(row["recoverable"], 0.0)
        self.assertFalse(row["included"])

    def test_vocabulary_dimension_included_when_gold_resolves(self) -> None:
        events = [event(dimension="SCOPE", fact_id=index) for index in range(10)]
        profiles = {index: self.profile({"scope": ["segment"]}) for index in range(10)}
        row = next(r for r in panel_a(events, profiles, NORMALIZATION_MAP) if r["dimension"] == "SCOPE")
        self.assertEqual(row["recoverable"], 1.0)
        self.assertTrue(row["included"])

    def test_label_derived_dimensions_are_recoverable_by_construction(self) -> None:
        """ROLE and EVENT define no controlled vocabulary; their attribute is the
        concept's own tokens, so they can never hit the empty-gold-set failure."""
        events = [event(dimension="ROLE", fact_id=index) for index in range(10)]
        profiles = {index: self.profile({}) for index in range(10)}
        row = next(r for r in panel_a(events, profiles, NORMALIZATION_MAP) if r["dimension"] == "ROLE")
        self.assertTrue(row["label_derived"])
        self.assertEqual(row["recoverable"], 1.0)
        self.assertTrue(row["included"])

    def test_gold_with_no_tokens_makes_a_label_derived_dimension_unrecoverable(self) -> None:
        events = [event(dimension="ROLE", fact_id=index) for index in range(10)]
        profiles = {index: self.profile({}, tokens=()) for index in range(10)}
        row = next(r for r in panel_a(events, profiles, NORMALIZATION_MAP) if r["dimension"] == "ROLE")
        self.assertEqual(row["recoverable"], 0.0)
        self.assertFalse(row["included"])

    def test_floor_is_applied_at_the_boundary(self) -> None:
        # 5 of 10 recoverable == the floor exactly, which is included.
        events = [event(dimension="SCOPE", fact_id=index) for index in range(10)]
        profiles = {
            index: self.profile({"scope": ["segment"] if index < 5 else []}) for index in range(10)
        }
        row = next(r for r in panel_a(events, profiles, NORMALIZATION_MAP) if r["dimension"] == "SCOPE")
        self.assertAlmostEqual(row["recoverable"], GOLD_COVERAGE_FLOOR)
        self.assertTrue(row["included"])

    def test_dimension_with_no_events_is_not_included(self) -> None:
        row = next(r for r in panel_a([], {}, NORMALIZATION_MAP) if r["dimension"] == "SCOPE")
        self.assertEqual(row["n"], 0)
        self.assertFalse(row["included"])


class EventBuildTests(unittest.TestCase):
    @staticmethod
    def gold_profile():
        return {
            "tag": "us-gaap:Revenues",
            "tokens": ["revenue", "amount", "good", "service"],
            "dimensions": {name: {"categories": [], "tokens": []} for name in ("role", "family")},
        }

    def test_only_dimensions_where_d_minus_fired_become_events(self) -> None:
        rounds = [round_record(d_minus={"ROLE"})]
        events, _ = build_events(
            args(), rounds, {0: self.gold_profile()}, {"ROLE": ["Revenue", "Geography"]}, NORMALIZATION_MAP
        )
        self.assertEqual([e["dimension"] for e in events], ["ROLE"])

    def test_missing_after_value_is_skipped_not_scored_as_unchanged(self) -> None:
        rounds = [round_record(hypothesis_after={"FAMILY": "Revenue"})]
        events, coverage = build_events(
            args(), rounds, {0: self.gold_profile()}, {"ROLE": ["Revenue"]}, NORMALIZATION_MAP
        )
        self.assertEqual(events, [])
        self.assertEqual(coverage["skipped"].get("value_missing"), 1)

    def test_operator_targeted_follows_the_targeted_dimension(self) -> None:
        rounds = [round_record(d_minus={"ROLE", "FAMILY"}, target_dimension="ROLE")]
        events, _ = build_events(
            args(),
            rounds,
            {0: self.gold_profile()},
            {"ROLE": ["Revenue"], "FAMILY": ["Revenue"]},
            NORMALIZATION_MAP,
        )
        targeted = {e["dimension"]: e["operator_targeted"] for e in events}
        self.assertTrue(targeted["ROLE"])
        self.assertFalse(targeted["FAMILY"])

    def test_missing_gold_profile_is_counted(self) -> None:
        events, coverage = build_events(args(), [round_record()], {}, {}, NORMALIZATION_MAP)
        self.assertEqual(events, [])
        self.assertEqual(coverage["skipped"].get("gold_profile_missing"), 1)

    def test_unchanged_value_is_classified_unchanged(self) -> None:
        rounds = [round_record(hypothesis_after={"ROLE": "Geography", "FAMILY": "Revenue"})]
        events, _ = build_events(
            args(), rounds, {0: self.gold_profile()}, {"ROLE": ["Revenue"]}, NORMALIZATION_MAP
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["outcome"], "unchanged")
        self.assertFalse(events[0]["value_changed"])


class SelectTests(unittest.TestCase):
    def test_all_excludes_artifact_dimensions(self) -> None:
        """The whole point of Panel A: TEMPORAL alone was ~36% of dev events, so
        leaking it into ALL would move the aggregate on its own."""
        events = [
            event(dimension="ROLE"),
            event(dimension="EVENT"),
            event(dimension="TEMPORAL"),
            event(dimension="SCOPE"),
        ]
        included = {"ROLE", "EVENT"}
        self.assertEqual(len(select(events, "ALL", "all", included)), 2)

    def test_named_dimension_ignores_the_included_set(self) -> None:
        events = [event(dimension="TEMPORAL")]
        self.assertEqual(len(select(events, "TEMPORAL", "all", {"ROLE"})), 1)

    def test_target_groups_partition(self) -> None:
        events = [event(operator_targeted=True), event(operator_targeted=False), event(operator_targeted=False)]
        included = {"ROLE"}
        targeted = select(events, "ALL", "operator_targeted", included)
        untargeted = select(events, "ALL", "not_targeted", included)
        self.assertEqual(len(targeted), 1)
        self.assertEqual(len(untargeted), 2)
        self.assertEqual(len(targeted) + len(untargeted), len(events))


class PanelBTests(unittest.TestCase):
    def test_rows_render_in_paper_order_with_missing_groups_tolerated(self) -> None:
        grid = [
            {
                "arm": "pooled",
                "dimension": "ALL",
                "target_group": "all",
                "n": 100,
                "improved_fraction": 0.072,
                "random_improved_fraction": 0.116,
                "improved_minus_random_ci_low": -0.066,
                "improved_minus_random_ci_high": -0.008,
            }
        ]
        rows = panel_b(grid)
        self.assertEqual([row["group"] for row in rows], [label for label, _, _ in PANEL_B_SPEC])
        self.assertEqual(rows[0]["n"], 100)
        self.assertFalse(rows[0]["ci_excludes_zero_favoring_revision"])
        # Groups with no matching grid row must not crash or fabricate a number.
        self.assertEqual(rows[1]["n"], 0)
        self.assertIsNone(rows[1]["improved"])

    def test_positive_interval_is_flagged(self) -> None:
        grid = [
            {
                "arm": "pooled",
                "dimension": "ROLE",
                "target_group": "operator_targeted",
                "n": 59,
                "improved_fraction": 0.220,
                "random_improved_fraction": 0.095,
                "improved_minus_random_ci_low": 0.026,
                "improved_minus_random_ci_high": 0.283,
            }
        ]
        row = next(r for r in panel_b(grid) if r["group"] == "ROLE, targ.")
        self.assertTrue(row["ci_excludes_zero_favoring_revision"])


class ProseStatisticsTests(unittest.TestCase):
    def test_unchanged_and_targeted_change_rates(self) -> None:
        events = [
            event(value_changed=False, operator_targeted=False),
            event(value_changed=False, operator_targeted=False),
            event(value_changed=True, operator_targeted=True),
        ]
        rounds = [
            round_record(hypothesis_before={"ROLE": "A"}, hypothesis_after={"ROLE": "A"}),
            round_record(hypothesis_before={"ROLE": "A"}, hypothesis_after={"ROLE": "B"}),
        ]
        stats = prose_statistics(events, rounds, {"ROLE"})
        # Reported values are rounded to 6 places, so compare at that precision.
        self.assertAlmostEqual(stats["value_unchanged_fraction_all_events"], 2 / 3, places=6)
        self.assertAlmostEqual(stats["value_changed_fraction_targeted"], 1.0, places=6)
        self.assertAlmostEqual(stats["rounds_changing_any_value_fraction"], 0.5, places=6)

    def test_excluded_dimension_share_is_reported(self) -> None:
        events = [event(dimension="ROLE"), event(dimension="TEMPORAL"), event(dimension="TEMPORAL")]
        stats = prose_statistics(events, [], {"ROLE"})
        self.assertAlmostEqual(stats["excluded_dimension_event_share"], 2 / 3, places=6)
        self.assertEqual(stats["events_by_dimension"]["TEMPORAL"], 2)


class PanelACounterfactualTests(unittest.TestCase):
    """The compact-vs-full column separates "attribute absent" from "attribute
    not looked for", which is the difference between a real exclusion and a
    configuration artifact."""

    @staticmethod
    def profile(categories):
        return {
            "tag": "us-gaap:X",
            "tokens": ["amount"],
            "dimensions": {"family": {"categories": list(categories), "tokens": ["amount"]}},
        }

    def test_gap_between_field_settings_is_flagged(self) -> None:
        events = [event(dimension="FAMILY", fact_id=index) for index in range(10)]
        deployed = {index: self.profile([]) for index in range(10)}
        other = {index: self.profile(["revenue"]) for index in range(10)}
        row = next(
            r
            for r in panel_a(events, deployed, NORMALIZATION_MAP, other, "full")
            if r["dimension"] == "FAMILY"
        )
        self.assertEqual(row["recoverable"], 0.0)
        self.assertEqual(row["recoverable_full"], 1.0)
        self.assertTrue(row["attribute_missed_by_deployed_view"])
        self.assertFalse(row["included"])

    def test_no_gap_is_not_flagged(self) -> None:
        events = [event(dimension="FAMILY", fact_id=index) for index in range(10)]
        profiles = {index: self.profile([]) for index in range(10)}
        row = next(
            r
            for r in panel_a(events, profiles, NORMALIZATION_MAP, dict(profiles), "full")
            if r["dimension"] == "FAMILY"
        )
        self.assertFalse(row["attribute_missed_by_deployed_view"])

    def test_counterfactual_columns_are_absent_when_not_supplied(self) -> None:
        events = [event(dimension="FAMILY", fact_id=0)]
        row = next(
            r for r in panel_a(events, {0: self.profile([])}, NORMALIZATION_MAP) if r["dimension"] == "FAMILY"
        )
        self.assertNotIn("recoverable_full", row)
        self.assertNotIn("attribute_missed_by_deployed_view", row)

    def test_event_and_concept_weighting_can_diverge(self) -> None:
        """D- fires where gold is unrecoverable, so event-weighting is biased
        downward relative to concept-weighting. Both are reported."""
        events = [event(dimension="FAMILY", fact_id=0) for _ in range(9)]
        events.append(event(dimension="FAMILY", fact_id=1))
        profiles = {0: self.profile([]), 1: self.profile(["revenue"])}
        row = next(
            r for r in panel_a(events, profiles, NORMALIZATION_MAP) if r["dimension"] == "FAMILY"
        )
        self.assertAlmostEqual(row["recoverable"], 0.1, places=6)
        self.assertAlmostEqual(row["recoverable_concept_weighted"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()

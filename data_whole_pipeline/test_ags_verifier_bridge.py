#!/usr/bin/env python3
"""Unit tests for the verifier bridge diagnostic.

CPU only, no data files, no GPU. The ranking statistics (AUROC, average
precision) are checked against cases with known closed-form answers, because a
subtly wrong AUROC would still land in a plausible range and would silently
change the paper's central reconciliation claim.
"""

from __future__ import annotations

import unittest

from run_ags_verifier_bridge import (
    SWEEP_THRESHOLDS,
    auroc,
    average_precision,
    bootstrap_contexts,
    decision,
    panel_a,
    panel_b,
    percentile,
    threshold_sweep,
)


def call(**overrides):
    record = {
        "fact_id": 0,
        "hypothesis_idx": 0,
        "context_key": "c0",
        "modality": "table",
        "gold_in_window": True,
        "gold_support": 3,
        "gold_total": 3,
        "distractor_support": 9,
        "distractor_total": 27,
    }
    record.update(overrides)
    return record


def dimension(**overrides):
    record = {
        "fact_id": 0,
        "hypothesis_idx": 0,
        "context_key": "c0",
        "dimension": "FAMILY",
        "truth_disagrees": False,
        "gold_in_window": True,
        "llm_judged": True,
        "llm_support_fraction": 0.8,
        "llm_comparable": 10,
        "deterministic_support_fraction": 0.9,
    }
    record.update(overrides)
    return record


class AurocTests(unittest.TestCase):
    def test_perfect_chance_and_inverted(self) -> None:
        self.assertEqual(auroc([0, 0, 1, 1], [False, False, True, True]), 1.0)
        self.assertEqual(auroc([0, 1, 0, 1], [False, False, True, True]), 0.5)
        self.assertEqual(auroc([1, 1, 0, 0], [False, False, True, True]), 0.0)

    def test_all_ties_is_exactly_chance(self) -> None:
        """Mid-rank handling matters: without it, ties skew the statistic."""
        self.assertEqual(auroc([0.5] * 6, [True, False, True, False, True, False]), 0.5)

    def test_partial_ties(self) -> None:
        # Positives at 1.0 and 0.5, negatives at 0.5 and 0.0.
        # Pairs: (1.0>0.5) win, (1.0>0.0) win, (0.5==0.5) half, (0.5>0.0) win => 3.5/4
        self.assertAlmostEqual(auroc([1.0, 0.5, 0.5, 0.0], [True, True, False, False]), 0.875)

    def test_degenerate_labels_return_none(self) -> None:
        self.assertIsNone(auroc([0.1, 0.2], [True, True]))
        self.assertIsNone(auroc([0.1, 0.2], [False, False]))

    def test_average_precision_bounds(self) -> None:
        self.assertEqual(average_precision([1, 1, 0, 0], [True, True, False, False]), 1.0)
        self.assertIsNone(average_precision([1, 0], [False, False]))
        # Worst case: both positives ranked last -> 0.5*(1/3 + 2/4)
        self.assertAlmostEqual(
            average_precision([1, 1, 0, 0], [False, False, True, True]), (1 / 3 + 2 / 4) / 2
        )


class PercentileTests(unittest.TestCase):
    def test_interpolates(self) -> None:
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(percentile(values, 0.0), 0.0)
        self.assertAlmostEqual(percentile(values, 1.0), 4.0)
        self.assertAlmostEqual(percentile(values, 0.5), 2.0)


class PanelATests(unittest.TestCase):
    def test_support_rates_and_gap(self) -> None:
        result = panel_a([call(), call(fact_id=1)], iterations=10, seed=1)
        window = result["gold_in_window"]
        self.assertAlmostEqual(window["support_rate_gold"], 1.0)
        self.assertAlmostEqual(window["support_rate_distractor"], 1 / 3, places=6)
        self.assertAlmostEqual(window["gold_minus_distractor_gap"], 2 / 3, places=6)
        self.assertAlmostEqual(window["pct_calls_favoring_gold"], 100.0)

    def test_calls_without_gold_in_window_are_excluded_from_gold_column(self) -> None:
        rows = [call(), call(fact_id=1, gold_in_window=False, gold_support=0, gold_total=0)]
        result = panel_a(rows, iterations=10, seed=1)
        self.assertEqual(result["calls_with_gold_in_window"], 1)
        self.assertAlmostEqual(result["pct_calls_with_gold_in_window"], 50.0)
        # The gold-in-window block sees only the one usable call.
        self.assertEqual(result["gold_in_window"]["n_calls"], 1)

    def test_no_calls_is_safe(self) -> None:
        result = panel_a([], iterations=5, seed=1)
        self.assertEqual(result["all_calls"]["n_calls"], 0)
        self.assertIsNone(result["pct_calls_with_gold_in_window"])


class PanelBTests(unittest.TestCase):
    def test_calibrated_verifier_shows_negative_difference_and_high_auroc(self) -> None:
        rows = [dimension(truth_disagrees=True, llm_support_fraction=0.1, context_key=f"c{i}") for i in range(20)]
        rows += [dimension(truth_disagrees=False, llm_support_fraction=0.9, context_key=f"c{i}") for i in range(20)]
        result = panel_b(rows, iterations=20, seed=1)
        self.assertAlmostEqual(result["support_difference_wrong_minus_right"], -0.8, places=6)
        self.assertAlmostEqual(result["llm_auroc_d_minus_score"], 1.0)

    def test_uninformative_verifier_lands_at_chance(self) -> None:
        rows = []
        for index in range(40):
            rows.append(
                dimension(
                    truth_disagrees=index % 2 == 0,
                    llm_support_fraction=0.55,
                    context_key=f"c{index % 8}",
                )
            )
        result = panel_b(rows, iterations=20, seed=1)
        self.assertAlmostEqual(result["support_difference_wrong_minus_right"], 0.0, places=6)
        self.assertAlmostEqual(result["llm_auroc_d_minus_score"], 0.5)

    def test_abstentions_stay_in_the_denominator_but_out_of_the_score(self) -> None:
        rows = [
            dimension(llm_judged=True, llm_support_fraction=0.2, truth_disagrees=True),
            dimension(llm_judged=False, llm_support_fraction=None, truth_disagrees=True),
        ]
        result = panel_b(rows, iterations=5, seed=1)
        self.assertEqual(result["n_observations"], 2)
        self.assertEqual(result["n_llm_judged"], 1)
        self.assertAlmostEqual(result["llm_non_abstention_rate"], 0.5)
        # Base rate uses every observation; the judged base rate may differ.
        self.assertAlmostEqual(result["base_rate_true_disagreement"], 1.0)


class ThresholdSweepTests(unittest.TestCase):
    def test_monotone_recall_and_all_thresholds_present(self) -> None:
        rows = [
            dimension(llm_support_fraction=value / 10.0, truth_disagrees=value < 5)
            for value in range(10)
        ]
        sweep = [row for row in threshold_sweep(rows) if row["layer"] == "llm"]
        self.assertEqual([row["fires_when_support_at_or_below"] for row in sweep], list(SWEEP_THRESHOLDS))
        recalls = [row["recall"] for row in sweep]
        self.assertEqual(recalls, sorted(recalls), "recall must be non-decreasing in the threshold")

    def test_table3_operating_point_is_marked_once_per_layer(self) -> None:
        rows = [dimension(llm_support_fraction=0.2, truth_disagrees=True)]
        sweep = threshold_sweep(rows)
        marked = [row for row in sweep if row["is_table3_operating_point"]]
        self.assertEqual({row["layer"] for row in marked}, {"llm", "deterministic"})

    def test_deterministic_layer_is_swept_too(self) -> None:
        rows = [dimension(deterministic_support_fraction=0.1, truth_disagrees=True)]
        self.assertTrue(any(row["layer"] == "deterministic" for row in threshold_sweep(rows)))


class BootstrapTests(unittest.TestCase):
    def test_constant_statistic_gives_zero_width_interval(self) -> None:
        by_context = {f"c{i}": [1.0] for i in range(10)}
        result = bootstrap_contexts(by_context, lambda rows: 0.42, 50, 1)
        self.assertAlmostEqual(result["ci_low"], 0.42)
        self.assertAlmostEqual(result["ci_high"], 0.42)
        self.assertTrue(result["ci_excludes_zero"])

    def test_interval_straddling_zero_is_flagged(self) -> None:
        by_context = {f"c{i}": [float(i - 5)] for i in range(11)}
        result = bootstrap_contexts(by_context, lambda rows: sum(rows) / len(rows), 200, 3)
        self.assertLess(result["ci_low"], 0.0)
        self.assertGreater(result["ci_high"], 0.0)
        self.assertFalse(result["ci_excludes_zero"])

    def test_empty_is_safe(self) -> None:
        self.assertEqual(bootstrap_contexts({}, lambda rows: 1.0, 10, 1)["contexts"], 0)


class DecisionRuleTests(unittest.TestCase):
    @staticmethod
    def comparison(deployed_known: bool, mrr_delta: float, significant: bool):
        return {
            "retrieval_stage": [
                {"metric": "mrr", "delta": mrr_delta, "ci_excludes_zero": significant},
                {"metric": "top1_accuracy", "delta": mrr_delta, "ci_excludes_zero": significant},
            ],
            "deployed_stage": {
                "with_candidate_level_llm_reranking": {"final_tagging_accuracy": 0.25}
                if deployed_known
                else None
            },
        }

    def test_missing_end_to_end_yields_provisional_rule_two(self) -> None:
        ruling = decision(self.comparison(False, 0.065, True))
        self.assertIn("PROVISIONAL", ruling["outcome"])
        self.assertFalse(ruling["end_to_end_available"])

    def test_no_retrieval_gain_yields_rule_three(self) -> None:
        ruling = decision(self.comparison(True, 0.0, False))
        self.assertIn("rule 3", ruling["outcome"])

    def test_end_to_end_present_defers_to_measured_accuracy(self) -> None:
        ruling = decision(self.comparison(True, 0.065, True))
        self.assertTrue(ruling["end_to_end_available"])
        self.assertNotIn("PROVISIONAL", ruling["outcome"])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Unit tests for the test-split verification-quality experiment (Tables 3, 13).

CPU only, no data files, no GPU. Covers the pieces that could silently produce a
plausible-looking wrong number: the LLM fold-up, the abstention convention, the
bootstrap estimator, and the paper-table row mapping.
"""

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
import random
import unittest

from ags_symbolic_agreement import (
    VERDICT_CONTRADICT,
    VERDICT_SUPPORT,
    VERDICT_UNRESOLVED,
)
from run_ags_verification_quality import (
    bootstrap_context,
    count_rows,
    hypothesis_dimensions_from_round,
    llm_feedback_from_verdicts,
    llm_firing_by_dimension,
    paired_arm_bootstrap,
    paired_layer_bootstrap,
    rates,
    slice_specs,
    subset_for,
    table3,
    table13,
)


def verdict_for(feedback, dimension):
    return next(
        record["verdict"]
        for record in feedback["dimension_verdicts"]
        if record["dimension"] == dimension
    )


def observation(**overrides):
    row = {
        "fact_id": 0,
        "context_key": "c0",
        "hypothesis_idx": 0,
        "modality": "table",
        "datatype": "monetaryItemType",
        "dimension": "FAMILY",
        "truth_layer": "exact",
        "truth_reason": "controlled_category_intersection",
        "truth_disagrees": False,
        "gold_rank": 3,
        "gold_in_window": True,
        "assessed_window_size": 10,
        "llm_call_present": True,
    }
    for source in ("deterministic", "llm", "merged"):
        row[f"d_minus_{source}"] = False
        row[f"d_plus_{source}"] = False
    row.update(overrides)
    return row


class LlmFoldUpTests(unittest.TestCase):
    """The LLM layer must be folded up exactly as the deterministic one is."""

    HYPOTHESIS = {
        "FAMILY": "Revenues",
        "ROLE": "Revenue",
        "EVENT": "Total Revenues",
        "QUALIFIER": "UNRESOLVED",
        "SCOPE": "Americas",
        "TEMPORAL": "December 31, 2024",
    }

    def _verdicts(self, family_values):
        tags = [f"us-gaap:C{index}" for index in range(len(family_values))]
        return tags, {
            tag: {"FAMILY": value, "ROLE": None, "EVENT": None, "confidence": 0.5}
            for tag, value in zip(tags, family_values)
        }

    def test_support_threshold_is_six_tenths(self) -> None:
        tags, verdicts = self._verdicts([True] * 6 + [False] * 4)
        feedback = llm_feedback_from_verdicts(self.HYPOTHESIS, tags, verdicts)
        self.assertEqual(verdict_for(feedback, "FAMILY"), VERDICT_SUPPORT)
        self.assertIn("FAMILY", feedback["supported_dimensions"])

    def test_contradict_threshold_is_a_quarter(self) -> None:
        tags, verdicts = self._verdicts([True] * 2 + [False] * 8)
        feedback = llm_feedback_from_verdicts(self.HYPOTHESIS, tags, verdicts)
        self.assertEqual(verdict_for(feedback, "FAMILY"), VERDICT_CONTRADICT)
        self.assertIn("FAMILY", feedback["contradicted_dimensions"])

    def test_middle_band_stays_unresolved(self) -> None:
        tags, verdicts = self._verdicts([True] * 5 + [False] * 5)
        feedback = llm_feedback_from_verdicts(self.HYPOTHESIS, tags, verdicts)
        self.assertEqual(verdict_for(feedback, "FAMILY"), VERDICT_UNRESOLVED)

    def test_null_judgements_leave_the_denominator(self) -> None:
        """A `null` is an abstention, not a disagreement.

        Two contradicts out of two *comparable* candidates is a contradiction
        even when eight others abstained; counting the nulls as support would
        make it unresolved instead.
        """
        tags, verdicts = self._verdicts([False, False] + [None] * 8)
        feedback = llm_feedback_from_verdicts(self.HYPOTHESIS, tags, verdicts)
        self.assertEqual(verdict_for(feedback, "FAMILY"), VERDICT_CONTRADICT)
        record = next(r for r in feedback["dimension_verdicts"] if r["dimension"] == "FAMILY")
        self.assertEqual(record["comparable_candidates"], 2)

    def test_dimensions_the_llm_never_judged_are_unresolved(self) -> None:
        tags, verdicts = self._verdicts([False] * 10)
        feedback = llm_feedback_from_verdicts(self.HYPOTHESIS, tags, verdicts)
        for dimension in ("QUALIFIER", "SCOPE", "TEMPORAL"):
            self.assertIn(dimension, feedback["unresolved_dimensions"])
            self.assertEqual(verdict_for(feedback, dimension), VERDICT_UNRESOLVED)
        self.assertNotIn("SCOPE", feedback["contradicted_dimensions"])

    def test_missing_call_abstains_rather_than_crashing(self) -> None:
        feedback = llm_feedback_from_verdicts(self.HYPOTHESIS, [], {})
        self.assertEqual(feedback["contradicted_dimensions"], [])
        self.assertEqual(feedback["supported_dimensions"], [])


class RateTests(unittest.TestCase):
    def test_counts_and_rates(self) -> None:
        rows = [
            observation(truth_disagrees=True, d_minus_deterministic=True),   # tp
            observation(truth_disagrees=True, d_minus_deterministic=True),   # tp
            observation(truth_disagrees=False, d_minus_deterministic=True),  # fp
            observation(truth_disagrees=True, d_minus_deterministic=False),  # fn
            observation(truth_disagrees=False, d_minus_deterministic=False), # tn
        ]
        stats = rates(count_rows(rows, "deterministic"))
        self.assertEqual(stats["n"], 5)
        self.assertEqual(stats["true_positive"], 2)
        self.assertEqual(stats["false_positive"], 1)
        self.assertEqual(stats["false_negative"], 1)
        self.assertAlmostEqual(stats["precision"], 2 / 3)
        self.assertAlmostEqual(stats["recall"], 2 / 3)
        self.assertAlmostEqual(stats["base_rate"], 3 / 5)
        self.assertAlmostEqual(stats["precision_minus_base_rate"], 2 / 3 - 3 / 5)

    def test_abstaining_layer_scores_zero_recall_not_undefined(self) -> None:
        rows = [observation(truth_disagrees=True) for _ in range(4)]
        stats = rates(count_rows(rows, "llm"))
        self.assertEqual(stats["d_minus_fired"], 0)
        self.assertEqual(stats["precision"], 0.0)
        self.assertEqual(stats["recall"], 0.0)


class BootstrapTests(unittest.TestCase):
    """The fast path must equal the naive row-level resample it replaces."""

    @staticmethod
    def naive_bootstrap(rows, source, iterations, seed):
        by_context = {}
        for row in rows:
            by_context.setdefault(row["context_key"], []).append(row)
        keys = sorted(by_context)
        rng = random.Random(seed)
        samples = {"precision": [], "recall": [], "precision_minus_base_rate": []}
        for _ in range(iterations):
            resampled = []
            for key in (keys[rng.randrange(len(keys))] for _ in keys):
                resampled.extend(by_context[key])
            stats = rates(count_rows(resampled, source))
            for name in samples:
                samples[name].append(stats[name])

        def percentile(values, q):
            ordered = sorted(values)
            pos = (len(ordered) - 1) * q
            lower = int(pos)
            upper = min(lower + 1, len(ordered) - 1)
            weight = pos - lower
            return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

        return {name: (percentile(v, 0.025), percentile(v, 0.975)) for name, v in samples.items()}

    def test_matches_naive_resampling(self) -> None:
        rng = random.Random(7)
        rows = [
            observation(
                context_key=f"c{index % 9}",
                truth_disagrees=rng.random() < 0.6,
                d_minus_deterministic=rng.random() < 0.5,
            )
            for index in range(200)
        ]
        fast = bootstrap_context(rows, "deterministic", 200, 1234)
        slow = self.naive_bootstrap(rows, "deterministic", 200, 1234)
        for key in fast:
            self.assertAlmostEqual(fast[key][0], slow[key][0], places=9, msg=key)
            self.assertAlmostEqual(fast[key][1], slow[key][1], places=9, msg=key)

    def test_empty_subset_is_safe(self) -> None:
        self.assertEqual(bootstrap_context([], "llm", 10, 1)["precision"], (0.0, 0.0))


class SliceTests(unittest.TestCase):
    def test_window_slices_partition_the_observations(self) -> None:
        rows = [
            observation(gold_in_window=True),
            observation(gold_in_window=False),
            observation(gold_in_window=False),
        ]
        inside = subset_for(rows, "ALL", "all", "gold_in_window", "all")
        outside = subset_for(rows, "ALL", "all", "gold_outside_window", "all")
        self.assertEqual(len(inside), 1)
        self.assertEqual(len(outside), 2)
        self.assertEqual(len(inside) + len(outside), len(rows))

    def test_dimension_and_layer_filters(self) -> None:
        rows = [
            observation(dimension="FAMILY", truth_layer="exact"),
            observation(dimension="FAMILY", truth_layer="lexical"),
            observation(dimension="SCOPE", truth_layer="exact"),
        ]
        self.assertEqual(len(subset_for(rows, "FAMILY", "all", "all", "all")), 2)
        self.assertEqual(len(subset_for(rows, "FAMILY", "exact", "all", "all")), 1)
        self.assertEqual(len(subset_for(rows, "ALL", "exact", "all", "all")), 2)

    def test_every_paper_row_has_a_slice_backing_it(self) -> None:
        """table3/table13 look their rows up by key; a missing spec is a crash at
        the end of a multi-hour job, so assert the mapping up front."""
        specs = set(slice_specs())
        from run_ags_verification_quality import TABLE13_SPEC, TABLE3_SPEC

        for _, dimension, layer, window, modality, _ in TABLE3_SPEC:
            self.assertIn((dimension, layer, window, modality), specs)
        for _, _, dimension, layer, window, modality, _, _ in TABLE13_SPEC:
            self.assertIn((dimension, layer, window, modality), specs)


class PaperTableTests(unittest.TestCase):
    @staticmethod
    def scored_rows():
        args = argparse.Namespace(bootstrap_samples=25, bootstrap_seed=1)
        rng = random.Random(3)
        rows = []
        for index in range(400):
            dimension = ("FAMILY", "ROLE", "EVENT", "QUALIFIER", "SCOPE", "TEMPORAL")[index % 6]
            disagrees = rng.random() < 0.6
            rows.append(
                observation(
                    context_key=f"c{index % 11}",
                    dimension=dimension,
                    truth_disagrees=disagrees,
                    gold_in_window=index % 3 == 0,
                    truth_layer="exact" if index % 2 else "lexical",
                    modality="table" if index % 5 else "text",
                    d_minus_deterministic=disagrees if rng.random() < 0.8 else not disagrees,
                    d_minus_llm=dimension == "FAMILY" and rng.random() < 0.1,
                    d_minus_merged=disagrees if rng.random() < 0.8 else not disagrees,
                )
            )
        from run_ags_verification_quality import build_rows

        return build_rows(args, rows), rows

    def test_tables_render_expected_shape(self) -> None:
        rows, observations = self.scored_rows()
        t3 = table3(rows)
        t13 = table13(rows)
        self.assertEqual([row["verifier"] for row in t3], ["Deterministic", "LLM", "Merged"])
        self.assertEqual(len(t13), 7)
        # The two window rows print no delta-vs-base column, matching the paper.
        self.assertIsNone(t13[3]["precision_minus_base"])
        self.assertIsNone(t13[4]["precision_minus_base"])
        # All three Table 3 layers are scored over the same denominator.
        self.assertEqual({row["n"] for row in t3}, {len(observations)})

    def test_llm_firing_summary_counts_opportunities_per_dimension(self) -> None:
        _, observations = self.scored_rows()
        firing = llm_firing_by_dimension(observations)
        self.assertEqual(
            sum(entry["opportunities"] for entry in firing.values()), len(observations)
        )
        self.assertLessEqual(firing["FAMILY"]["llm_d_minus"], firing["FAMILY"]["opportunities"])
        self.assertEqual(firing["SCOPE"]["llm_d_minus"], 0)


class PairedArmBootstrapTests(unittest.TestCase):
    def test_identical_arms_give_a_zero_delta_interval(self) -> None:
        per_context = {f"c{i}": {"n": 20, "tp": 9, "fp": 2, "fn": 3} for i in range(30)}
        result = paired_arm_bootstrap(per_context, dict(per_context), 200, 5)
        self.assertAlmostEqual(result["mean_delta"], 0.0, places=9)
        self.assertFalse(result["ci_excludes_zero"])

    def test_a_large_separation_is_detected(self) -> None:
        a = {f"c{i}": {"n": 20, "tp": 18, "fp": 1, "fn": 1} for i in range(30)}
        b = {f"c{i}": {"n": 20, "tp": 4, "fp": 14, "fn": 2} for i in range(30)}
        result = paired_arm_bootstrap(a, b, 200, 5)
        self.assertGreater(result["mean_delta"], 0.4)
        self.assertTrue(result["ci_excludes_zero"])

    def test_contexts_missing_from_one_arm_are_treated_as_empty(self) -> None:
        a = {"c0": {"n": 10, "tp": 5, "fp": 1, "fn": 2}, "c1": {"n": 10, "tp": 5, "fp": 1, "fn": 2}}
        b = {"c0": {"n": 10, "tp": 5, "fp": 1, "fn": 2}}
        result = paired_arm_bootstrap(a, b, 50, 5)
        self.assertEqual(result["contexts"], 2)

    def test_empty_input_is_safe(self) -> None:
        result = paired_arm_bootstrap({}, {}, 10, 1)
        self.assertEqual(result["contexts"], 0)
        self.assertFalse(result["ci_excludes_zero"])


class SeqRoundRecoveryTests(unittest.TestCase):
    def test_recovers_hypothesis_values_from_candidate_verdicts(self) -> None:
        record = {
            "dimension_verdicts": [
                {
                    "dimension": "FAMILY",
                    "candidate_verdicts": [
                        {"raw_hypothesis_value": "Revenues"},
                        {"raw_hypothesis_value": "Revenues"},
                    ],
                },
                {"dimension": "QUALIFIER", "reason": "hypothesis_unresolved"},
            ]
        }
        self.assertEqual(hypothesis_dimensions_from_round(record), {"FAMILY": "Revenues"})

    def test_no_verdicts_yields_nothing(self) -> None:
        self.assertEqual(hypothesis_dimensions_from_round({}), {})


class F1CoverageTests(unittest.TestCase):
    def test_f1_is_the_harmonic_mean(self) -> None:
        rows = [
            observation(truth_disagrees=True, d_minus_deterministic=True),
            observation(truth_disagrees=True, d_minus_deterministic=True),
            observation(truth_disagrees=False, d_minus_deterministic=True),
            observation(truth_disagrees=True, d_minus_deterministic=False),
        ]
        stats = rates(count_rows(rows, "deterministic"))
        self.assertAlmostEqual(stats["precision"], 2 / 3)
        self.assertAlmostEqual(stats["recall"], 2 / 3)
        self.assertAlmostEqual(stats["f1"], 2 / 3)

    def test_f1_is_zero_when_nothing_fires(self) -> None:
        rows = [observation(truth_disagrees=True) for _ in range(3)]
        self.assertEqual(rates(count_rows(rows, "llm"))["f1"], 0.0)

    def test_coverage_counts_any_verdict_not_just_d_minus(self) -> None:
        rows = [
            observation(d_minus_llm=True),
            observation(d_plus_llm=True),
            observation(),  # abstained
        ]
        stats = rates(count_rows(rows, "llm"))
        self.assertEqual(stats["verdicts_issued"], 2)
        self.assertAlmostEqual(stats["coverage"], 2 / 3)

    def test_abstentions_stay_in_the_denominator(self) -> None:
        """An abstention is a missed detection; excluding it would flatter a
        layer that mostly stays silent."""
        rows = [observation(truth_disagrees=True) for _ in range(10)]
        rows[0]["d_minus_llm"] = True
        stats = rates(count_rows(rows, "llm"))
        self.assertEqual(stats["n"], 10)
        self.assertAlmostEqual(stats["recall"], 0.1)
        self.assertAlmostEqual(stats["coverage"], 0.1)


class PairedLayerBootstrapTests(unittest.TestCase):
    def test_identical_layers_give_a_zero_interval(self) -> None:
        rows = []
        for index in range(40):
            row = observation(context_key=f"c{index % 8}", truth_disagrees=index % 3 == 0)
            row["d_minus_deterministic"] = row["d_minus_merged"] = index % 2 == 0
            rows.append(row)
        result = paired_layer_bootstrap(rows, "merged", "deterministic", "f1", 100, 1)
        self.assertAlmostEqual(result["mean"], 0.0, places=9)
        self.assertFalse(result["ci_excludes_zero"])

    def test_a_clearly_better_layer_is_detected(self) -> None:
        rows = []
        for index in range(60):
            disagrees = index % 2 == 0
            row = observation(context_key=f"c{index % 10}", truth_disagrees=disagrees)
            row["d_minus_deterministic"] = disagrees          # perfect
            row["d_minus_llm"] = not disagrees                # inverted
            rows.append(row)
        result = paired_layer_bootstrap(rows, "deterministic", "llm", "f1", 200, 1)
        self.assertGreater(result["mean"], 0.5)
        self.assertTrue(result["ci_excludes_zero"])

    def test_empty_is_safe(self) -> None:
        self.assertEqual(paired_layer_bootstrap([], "deterministic", "llm", "f1", 10, 1)["contexts"], 0)


if __name__ == "__main__":
    unittest.main()

"""Tests for scripts/compare_inheritance_arms.py helper logic."""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.compare_inheritance_arms import (  # noqa: E402
    ALL_METRIC_KEYS,
    _build_markdown,
    _ci_excludes_zero,
    _classify_regime_verdict,
    _compute_mechanism_coverage,
    _extract_metadata_metrics,
    _paired_delta_summary,
    _paired_metric_deltas,
    _resolve_metric_value,
    _summarize_warmstart_coverage,
    _warmstart_rate_from_run,
)


class TestPairedDeltaSummary(unittest.TestCase):
    def test_empty(self):
        out = _paired_delta_summary([])
        self.assertEqual(out["n"], 0)
        # Empty summaries must use ``None`` (valid JSON ``null``), not NaN
        # (which json.dump emits as the non-standard ``NaN`` token).
        self.assertIsNone(out["mean_delta"])
        self.assertEqual(out["ci95"], [None, None])

    def test_empty_summary_is_json_serializable(self):
        out = _paired_delta_summary([])
        # Strict mode rejects NaN/Infinity tokens, so this exercises the
        # contract that empty summaries round-trip through valid JSON.
        encoded = json.dumps(out, allow_nan=False)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["n"], 0)
        self.assertIsNone(decoded["mean_delta"])

    def test_non_empty(self):
        out = _paired_delta_summary([1.0, 2.0, 3.0])
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["mean_delta"], 2.0)
        self.assertAlmostEqual(out["sign_agreement"], 1.0)


class TestCiExcludesZero(unittest.TestCase):
    def test_positive(self):
        self.assertTrue(_ci_excludes_zero([0.1, 0.5]))

    def test_crosses_zero(self):
        self.assertFalse(_ci_excludes_zero([-0.1, 0.2]))

    def test_nan(self):
        self.assertFalse(_ci_excludes_zero([float("nan"), 0.5]))


class TestVerdictClassification(unittest.TestCase):
    def _summary(self, mean, ci, sign=1.0, n=10):
        return {"mean_delta": mean, "ci95": ci, "sign_agreement": sign, "n": n}

    def _empty(self):
        return self._summary(0.0, [-1.0, 1.0], sign=0.5)

    def test_recommend_treatment_label_when_only_perf_wins(self):
        verdict = _classify_regime_verdict(
            {
                "population_mean": self._summary(5.0, [2.0, 8.0]),
                "population_final": self._empty(),
                "startup_transient.peak_death_rate": self._empty(),
                "startup_transient.oscillation_amplitude": self._empty(),
                "lineage.churn_rate": self._empty(),
                "speciation_slope": self._empty(),
            },
            treatment_label="lamarckian",
            baseline_label="baldwinian",
        )
        self.assertEqual(verdict, "net recommend lamarckian")

    def test_recommend_baseline_when_stability_loses(self):
        verdict = _classify_regime_verdict(
            {
                "population_mean": self._empty(),
                "population_final": self._empty(),
                "startup_transient.peak_death_rate": self._empty(),
                "startup_transient.oscillation_amplitude": self._summary(3.0, [1.0, 5.0]),
                "lineage.churn_rate": self._empty(),
                "speciation_slope": self._empty(),
            },
            treatment_label="lamarckian",
            baseline_label="baldwinian",
        )
        self.assertEqual(verdict, "net recommend baldwinian")

    def test_label_swap(self):
        """The label appears verbatim in the verdict string."""
        deltas = {
            "population_mean": self._summary(5.0, [2.0, 8.0]),
            "population_final": self._empty(),
            "startup_transient.peak_death_rate": self._empty(),
            "startup_transient.oscillation_amplitude": self._empty(),
            "lineage.churn_rate": self._empty(),
            "speciation_slope": self._empty(),
        }
        verdict = _classify_regime_verdict(
            deltas, treatment_label="hinton-nowlan", baseline_label="cold-start"
        )
        self.assertEqual(verdict, "net recommend hinton-nowlan")

    def test_lineage_churn_counts_as_stability(self):
        verdict = _classify_regime_verdict(
            {
                "population_mean": self._empty(),
                "population_final": self._empty(),
                "startup_transient.peak_death_rate": self._empty(),
                "startup_transient.oscillation_amplitude": self._empty(),
                "lineage.churn_rate": self._summary(0.3, [0.1, 0.5]),
                "speciation_slope": self._empty(),
            },
            treatment_label="lamarckian",
            baseline_label="baldwinian",
        )
        self.assertEqual(verdict, "net recommend baldwinian")

    def test_perf_plus_stability_loss(self):
        verdict = _classify_regime_verdict(
            {
                "population_mean": self._summary(5.0, [2.0, 8.0]),
                "population_final": self._empty(),
                "startup_transient.peak_death_rate": self._empty(),
                "startup_transient.oscillation_amplitude": self._summary(3.0, [1.0, 5.0]),
                "lineage.churn_rate": self._empty(),
                "speciation_slope": self._empty(),
            },
            treatment_label="lamarckian",
            baseline_label="baldwinian",
        )
        self.assertEqual(verdict, "performance win + stability loss")

    def test_perf_win_with_speciation_collapse_is_flagged(self):
        """A treatment that wins on population but collapses speciation should not silently recommend the treatment."""
        verdict = _classify_regime_verdict(
            {
                "population_mean": self._summary(5.0, [2.0, 8.0]),
                "population_final": self._empty(),
                "startup_transient.peak_death_rate": self._empty(),
                "startup_transient.oscillation_amplitude": self._empty(),
                "lineage.churn_rate": self._empty(),
                "speciation_slope": self._summary(-0.05, [-0.1, -0.01]),
            },
            treatment_label="lamarckian",
            baseline_label="baldwinian",
        )
        self.assertEqual(verdict, "speciation collapse risk")

    def test_no_robust_effect(self):
        verdict = _classify_regime_verdict(
            {
                "population_mean": self._empty(),
                "population_final": self._empty(),
                "startup_transient.peak_death_rate": self._empty(),
                "startup_transient.oscillation_amplitude": self._empty(),
                "lineage.churn_rate": self._empty(),
                "speciation_slope": self._empty(),
            },
            treatment_label="lamarckian",
            baseline_label="baldwinian",
        )
        self.assertEqual(verdict, "no robust effect")


class TestResolveMetricValue(unittest.TestCase):
    def test_top_level(self):
        run = {"population_mean": 100.0}
        self.assertEqual(_resolve_metric_value(run, "population_mean"), 100.0)

    def test_nested(self):
        run = {"startup_transient": {"peak_death_rate": 0.3}}
        self.assertEqual(
            _resolve_metric_value(run, "startup_transient.peak_death_rate"),
            0.3,
        )

    def test_missing_segment(self):
        run = {"startup_transient": {}}
        self.assertIsNone(_resolve_metric_value(run, "startup_transient.peak_death_rate"))

    def test_non_numeric(self):
        run = {"label": "hello"}
        self.assertIsNone(_resolve_metric_value(run, "label"))


class TestPairedMetricDeltas(unittest.TestCase):
    def _run(self, pop=100.0, spec=0.5, slope=0.02):
        return {
            "speciation_final": spec,
            "speciation_slope": slope,
            "speciation_mean": spec - 0.05,
            "population_mean": pop,
            "population_final": pop + 5.0,
            "lineage": {"mean_k": 2.0, "churn_rate": 0.1},
            "startup_transient": {
                "peak_birth_rate": 0.6,
                "peak_death_rate": 0.4,
                "oscillation_amplitude": 12.0,
            },
            "lamarckian_warmstart_rate": 0.75,
        }

    def test_seed_pairing_returns_tuple(self):
        baseline = {
            42: self._run(pop=100.0, spec=0.6),
            7: self._run(pop=90.0, spec=0.5),
        }
        treatment = {
            42: self._run(pop=110.0, spec=0.55),
            7: self._run(pop=95.0, spec=0.45),
            999: self._run(pop=150.0, spec=0.9),
        }
        deltas, paired_seeds = _paired_metric_deltas(baseline, treatment)
        self.assertEqual(paired_seeds, [7, 42])
        self.assertAlmostEqual(deltas["population_mean"]["mean_delta"], 7.5)
        self.assertAlmostEqual(deltas["speciation_final"]["mean_delta"], -0.05)
        # The sentinel key must not leak into the deltas dict any more.
        self.assertNotIn("_paired_seeds", deltas)
        # Every documented metric key gets a summary, even if empty.
        for key in ALL_METRIC_KEYS:
            self.assertIn(key, deltas)

    def test_skips_seeds_with_missing_metric(self):
        baseline = {1: {"population_mean": 100.0}}
        treatment = {1: {}}
        deltas, paired_seeds = _paired_metric_deltas(baseline, treatment)
        self.assertEqual(paired_seeds, [1])
        self.assertEqual(deltas["population_mean"]["n"], 0)


class TestExtractMetadataMetrics(unittest.TestCase):
    def test_carries_skip_reasons(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "intrinsic_evolution_metadata.json").write_text(
                json.dumps(
                    {
                        "startup_transient_metrics": {
                            "peak_birth_rate": 0.5,
                            "peak_death_rate": 0.4,
                            "oscillation_amplitude": 12,
                        },
                        "policy_inheritance_metrics": {
                            "lamarckian_warmstart_applied": 8,
                            "lamarckian_warmstart_skipped": 2,
                            "lamarckian_warmstart_skipped_reasons": {
                                "incompatible_state": 2,
                            },
                            "decide_action_failures": 4,
                            "decide_action_failure_reasons": {
                                "RuntimeError": 3,
                                "ValueError": 1,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            metrics = _extract_metadata_metrics(run_dir)
        self.assertAlmostEqual(metrics["lamarckian_warmstart_rate"], 0.8)
        self.assertEqual(metrics["lamarckian_warmstart_applied"], 8)
        self.assertEqual(metrics["lamarckian_warmstart_skipped"], 2)
        self.assertEqual(
            metrics["lamarckian_warmstart_skipped_reasons"], {"incompatible_state": 2}
        )
        self.assertEqual(metrics["decide_action_failures"], 4.0)
        self.assertEqual(
            metrics["decide_action_failure_reasons"],
            {"RuntimeError": 3, "ValueError": 1},
        )

    def test_missing_metadata_returns_empty(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(_extract_metadata_metrics(Path(tmp)), {})


class TestWarmstartCoverage(unittest.TestCase):
    def _run(self, applied: int, skipped: int, reasons: Optional[dict] = None):
        return {
            "lamarckian_warmstart_applied": applied,
            "lamarckian_warmstart_skipped": skipped,
            "lamarckian_warmstart_skipped_reasons": reasons or {},
        }

    def test_rate_from_run(self):
        self.assertAlmostEqual(_warmstart_rate_from_run(self._run(8, 2)), 0.8)
        self.assertIsNone(_warmstart_rate_from_run(self._run(0, 0)))

    def test_summarize_warmstart_coverage(self):
        runs = {
            42: self._run(8, 2, {"incompatible_state": 2}),
            7: self._run(9, 1, {"incompatible_state": 1}),
        }
        summary = _summarize_warmstart_coverage(runs)
        self.assertEqual(summary["n"], 2)
        self.assertAlmostEqual(summary["mean_rate"], 0.85)
        self.assertEqual(summary["total_applied"], 17)
        self.assertEqual(summary["total_skipped"], 3)
        self.assertEqual(summary["skip_reasons"], {"incompatible_state": 3})
        self.assertEqual(summary["per_seed"]["42"]["applied"], 8)

    def test_compute_mechanism_coverage(self):
        treatments = {
            "lamarckian": {
                "conservative": {
                    42: self._run(8, 2, {"incompatible_state": 2}),
                }
            }
        }
        coverage = _compute_mechanism_coverage(treatments, ["conservative"], ["lamarckian"])
        warmstart = coverage["conservative"]["lamarckian"]["lamarckian_warmstart"]
        self.assertAlmostEqual(warmstart["mean_rate"], 0.8)
        self.assertEqual(warmstart["total_applied"], 8)

    def test_markdown_includes_mechanism_coverage(self):
        coverage = {
            "conservative": {
                "lamarckian": {
                    "lamarckian_warmstart": {
                        "n": 1,
                        "mean_rate": 0.8,
                        "rate_ci95": [0.8, 0.8],
                        "total_applied": 8,
                        "total_skipped": 2,
                        "skip_reasons": {"incompatible_state": 2},
                        "per_seed": {},
                    }
                }
            }
        }
        md = _build_markdown({}, {}, ["lamarckian"], "baldwinian", mechanism_coverage=coverage)
        self.assertIn("## Mechanism coverage (treatment only)", md)
        self.assertIn("Lamarckian warm-start rate is an absolute treatment-arm statistic", md)
        self.assertNotIn("| lamarckian | warmstart rate delta |", md)


if __name__ == "__main__":
    unittest.main()

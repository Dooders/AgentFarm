"""Tests for scripts/compare_inheritance_arms.py helper logic."""

from __future__ import annotations

import math
import os
import sys
import unittest

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.compare_inheritance_arms import (  # noqa: E402
    _ci_excludes_zero,
    _classify_regime_verdict,
    _paired_delta_summary,
    _paired_metric_deltas,
)


class TestPairedDeltaSummary(unittest.TestCase):
    def test_empty(self):
        out = _paired_delta_summary([])
        self.assertEqual(out["n"], 0)
        self.assertTrue(math.isnan(out["mean_delta"]))

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


class TestVerdictClassification(unittest.TestCase):
    def _summary(self, mean, ci, sign=1.0):
        return {"mean_delta": mean, "ci95": ci, "sign_agreement": sign}

    def test_recommend_lamarckian(self):
        verdict = _classify_regime_verdict(
            {
                "population_mean": self._summary(5.0, [2.0, 8.0]),
                "startup_transient.oscillation_amplitude": self._summary(0.1, [-0.2, 0.3], sign=0.5),
                "speciation_slope": self._summary(0.0, [-0.1, 0.1], sign=0.5),
            }
        )
        self.assertEqual(verdict, "net recommend lamarckian")

    def test_recommend_baldwinian(self):
        verdict = _classify_regime_verdict(
            {
                "population_mean": self._summary(0.1, [-0.4, 0.6], sign=0.5),
                "startup_transient.oscillation_amplitude": self._summary(3.0, [1.0, 5.0]),
                "speciation_slope": self._summary(0.0, [-0.1, 0.1], sign=0.5),
            }
        )
        self.assertEqual(verdict, "net recommend baldwinian")


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

    def test_seed_pairing(self):
        baseline = {
            42: self._run(pop=100.0, spec=0.6),
            7: self._run(pop=90.0, spec=0.5),
        }
        treatment = {
            42: self._run(pop=110.0, spec=0.55),
            7: self._run(pop=95.0, spec=0.45),
            999: self._run(pop=150.0, spec=0.9),
        }
        out = _paired_metric_deltas(baseline, treatment)
        self.assertEqual(sorted(out["_paired_seeds"]), [7, 42])
        self.assertAlmostEqual(out["population_mean"]["mean_delta"], 7.5)
        self.assertAlmostEqual(out["speciation_final"]["mean_delta"], -0.05)


if __name__ == "__main__":
    unittest.main()

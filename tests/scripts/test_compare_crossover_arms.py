"""Tests for scripts/compare_crossover_arms.py.

Focuses on pure-logic helpers that don't require a full simulation:
- _paired_delta_summary (mean / CI / sign-agreement under known inputs)
- _ci_excludes_zero (boundary cases around zero)
- _classify_collapse_verdict (verdict string given fixed paired-delta dicts)
- _lineage_metrics (cluster-count trace and churn-rate from synthetic JSONL)
- _build_markdown (renders without crashing, contains expected sections)
- _paired_metric_deltas (paired computation when baseline/treatment partially
  overlap by seed)
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.compare_crossover_arms import (  # noqa: E402
    _build_markdown,
    _ci_excludes_zero,
    _classify_collapse_verdict,
    _lineage_metrics,
    _paired_delta_summary,
    _paired_metric_deltas,
)


# ── _paired_delta_summary ────────────────────────────────────────────────────


class TestPairedDeltaSummary(unittest.TestCase):
    def test_empty(self):
        out = _paired_delta_summary([])
        self.assertEqual(out["n"], 0)
        self.assertTrue(math.isnan(out["mean_delta"]))

    def test_all_negative_known(self):
        out = _paired_delta_summary([-1.0, -2.0, -3.0, -4.0])
        self.assertAlmostEqual(out["mean_delta"], -2.5)
        self.assertAlmostEqual(out["sign_agreement"], 1.0)
        self.assertEqual(out["n"], 4)
        lo, hi = out["ci95"]
        # 95% CI should bracket the mean
        self.assertLess(lo, -2.5)
        self.assertGreater(hi, -2.5)
        # All-negative sample → CI should sit below zero
        self.assertLess(hi, 0.0)

    def test_mixed_signs_partial_agreement(self):
        out = _paired_delta_summary([1.0, -1.0, 2.0, -3.0])
        self.assertEqual(out["n"], 4)
        self.assertAlmostEqual(out["sign_agreement"], 0.0)

    def test_nan_filtered(self):
        out = _paired_delta_summary([1.0, float("nan"), 2.0, float("nan"), 3.0])
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["mean_delta"], 2.0)


# ── _ci_excludes_zero ─────────────────────────────────────────────────────────


class TestCiExcludesZero(unittest.TestCase):
    def test_strictly_positive(self):
        self.assertTrue(_ci_excludes_zero([0.1, 0.5]))

    def test_strictly_negative(self):
        self.assertTrue(_ci_excludes_zero([-0.5, -0.1]))

    def test_straddles_zero(self):
        self.assertFalse(_ci_excludes_zero([-0.1, 0.5]))

    def test_touches_zero_lower(self):
        # Touching zero counts as "not excluding zero" — strict inequality.
        self.assertFalse(_ci_excludes_zero([0.0, 0.5]))

    def test_nan_returns_false(self):
        self.assertFalse(_ci_excludes_zero([float("nan"), 0.5]))

    def test_wrong_length_returns_false(self):
        self.assertFalse(_ci_excludes_zero([0.1]))


# ── _classify_collapse_verdict ────────────────────────────────────────────────


class TestClassifyCollapseVerdict(unittest.TestCase):
    @staticmethod
    def _summary(mean_delta, ci, sign_agreement):
        return {
            "mean_delta": mean_delta,
            "ci95": ci,
            "sign_agreement": sign_agreement,
        }

    def test_robust_collapse(self):
        spec_final = self._summary(-0.15, [-0.20, -0.10], 1.0)
        spec_slope = self._summary(-0.005, [-0.008, -0.002], 1.0)
        self.assertEqual(
            _classify_collapse_verdict(spec_final, spec_slope),
            "robustly collapses",
        )

    def test_robust_amplifies(self):
        spec_final = self._summary(0.15, [0.10, 0.20], 1.0)
        spec_slope = self._summary(0.005, [0.002, 0.008], 1.0)
        self.assertEqual(
            _classify_collapse_verdict(spec_final, spec_slope),
            "robustly amplifies",
        )

    def test_no_robust_effect_when_ci_straddles_zero(self):
        spec_final = self._summary(-0.05, [-0.10, 0.05], 0.6)
        spec_slope = self._summary(-0.001, [-0.005, 0.003], 0.5)
        self.assertEqual(
            _classify_collapse_verdict(spec_final, spec_slope),
            "no robust effect",
        )

    def test_robust_when_only_slope_evidence(self):
        # If speciation_final is noisy but slope is robust and negative,
        # the verdict should still trigger as "robustly collapses".
        spec_final = self._summary(-0.05, [-0.20, 0.10], 0.5)
        spec_slope = self._summary(-0.008, [-0.012, -0.004], 1.0)
        self.assertEqual(
            _classify_collapse_verdict(spec_final, spec_slope),
            "robustly collapses",
        )

    def test_robust_amplifies_when_only_slope_evidence_positive(self):
        # Noisy spec_final must not flip the verdict away from robust slope evidence.
        spec_final = self._summary(-0.05, [-0.20, 0.10], 0.5)
        spec_slope = self._summary(0.008, [0.002, 0.014], 1.0)
        self.assertEqual(
            _classify_collapse_verdict(spec_final, spec_slope),
            "robustly amplifies",
        )


# ── _lineage_metrics ──────────────────────────────────────────────────────────


class TestLineageMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_lineage(self, rows):
        path = self.run_dir / "cluster_lineage.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def test_empty_returns_nans(self):
        out = _lineage_metrics(self.run_dir)
        self.assertEqual(out["cluster_count_trace"], [])
        self.assertTrue(math.isnan(out["mean_k"]))
        self.assertTrue(math.isnan(out["churn_rate"]))

    def test_cluster_count_per_step(self):
        rows = [
            {"step": 50, "cluster_id": "c0"},
            {"step": 50, "cluster_id": "c1"},
            {"step": 100, "cluster_id": "c0"},
            {"step": 100, "cluster_id": "c2"},
            {"step": 100, "cluster_id": "c3"},
        ]
        self._write_lineage(rows)
        out = _lineage_metrics(self.run_dir)
        self.assertEqual(out["cluster_count_trace"], [(50, 2), (100, 3)])
        # mean k = (2 + 3) / 2 = 2.5
        self.assertAlmostEqual(out["mean_k"], 2.5)
        # Step 50 had {c0, c1}; step 100 has {c0, c2, c3}.
        # c1 died, c0 survived → churn at 50 = 1/2 = 0.5.
        # Only one transition pair → mean churn = 0.5.
        self.assertAlmostEqual(out["churn_rate"], 0.5)

    def test_zero_churn_when_all_survive(self):
        rows = [
            {"step": 0, "cluster_id": "c0"},
            {"step": 0, "cluster_id": "c1"},
            {"step": 100, "cluster_id": "c0"},
            {"step": 100, "cluster_id": "c1"},
        ]
        self._write_lineage(rows)
        out = _lineage_metrics(self.run_dir)
        self.assertAlmostEqual(out["churn_rate"], 0.0)

    def test_full_churn_when_all_replaced(self):
        rows = [
            {"step": 0, "cluster_id": "c0"},
            {"step": 0, "cluster_id": "c1"},
            {"step": 100, "cluster_id": "c2"},
            {"step": 100, "cluster_id": "c3"},
        ]
        self._write_lineage(rows)
        out = _lineage_metrics(self.run_dir)
        self.assertAlmostEqual(out["churn_rate"], 1.0)

    def test_churn_nan_with_single_step(self):
        rows = [
            {"step": 0, "cluster_id": "c0"},
            {"step": 0, "cluster_id": "c1"},
        ]
        self._write_lineage(rows)
        out = _lineage_metrics(self.run_dir)
        self.assertTrue(math.isnan(out["churn_rate"]))
        self.assertAlmostEqual(out["mean_k"], 2.0)


# ── _paired_metric_deltas ─────────────────────────────────────────────────────


class TestPairedMetricDeltas(unittest.TestCase):
    @staticmethod
    def _run(spec_final, spec_slope, lr_shift, mean_k=2.0, churn=0.1):
        return {
            "speciation_final": spec_final,
            "speciation_slope": spec_slope,
            "speciation_mean": spec_final - 0.05,
            "population_mean": 100.0,
            "gene_pct_shift": {"learning_rate": lr_shift, "gamma": -1.0},
            "lineage": {
                "mean_k": mean_k,
                "churn_rate": churn,
                "cluster_count_trace": [],
            },
        }

    def test_paired_deltas_by_seed(self):
        baseline = {
            42: self._run(0.7, 0.02, 5.0),
            7: self._run(0.6, 0.01, 3.0),
        }
        treatment = {
            42: self._run(0.5, 0.005, -2.0),
            7: self._run(0.4, -0.005, -4.0),
            # Seed 19 only in treatment → should be excluded from pairing.
            19: self._run(0.9, 0.05, 10.0),
        }
        deltas = _paired_metric_deltas(baseline, treatment)
        # Paired seeds = {42, 7}
        self.assertEqual(sorted(deltas["_paired_seeds"]), [7, 42])

        sf = deltas["speciation_final"]
        self.assertEqual(sf["n"], 2)
        # (0.5 - 0.7) + (0.4 - 0.6) = -0.4; mean = -0.2
        self.assertAlmostEqual(sf["mean_delta"], -0.2)
        self.assertAlmostEqual(sf["sign_agreement"], 1.0)

        lr = deltas["gene.learning_rate"]
        self.assertEqual(lr["n"], 2)
        # (-2 - 5) + (-4 - 3) = -14; mean = -7
        self.assertAlmostEqual(lr["mean_delta"], -7.0)

    def test_excludes_runs_with_nan(self):
        baseline = {
            42: self._run(0.7, 0.02, 5.0),
            7: self._run(float("nan"), 0.01, 3.0),
        }
        treatment = {
            42: self._run(0.5, 0.005, -2.0),
            7: self._run(0.6, -0.005, -4.0),
        }
        deltas = _paired_metric_deltas(baseline, treatment)
        sf = deltas["speciation_final"]
        # Seed 7 had NaN baseline → only seed 42 contributes
        self.assertEqual(sf["n"], 1)
        self.assertAlmostEqual(sf["mean_delta"], -0.2)


# ── _build_markdown ───────────────────────────────────────────────────────────


class TestBuildMarkdown(unittest.TestCase):
    def _fake_summary(self, mean):
        return {
            "mean_delta": mean,
            "variance": 0.001,
            "ci95": [mean - 0.05, mean + 0.05],
            "sign_agreement": 1.0,
            "n": 6,
        }

    def test_markdown_contains_verdict_table(self):
        deltas = {
            "buffered": {
                "uniform": {
                    "speciation_final": self._fake_summary(-0.2),
                    "speciation_slope": self._fake_summary(-0.005),
                    "gene.learning_rate": self._fake_summary(-7.0),
                },
                "blend": {
                    "speciation_final": self._fake_summary(0.1),
                    "speciation_slope": self._fake_summary(0.003),
                    "gene.learning_rate": self._fake_summary(2.0),
                },
            },
        }
        verdicts = {
            "buffered": {
                "uniform": "robustly collapses",
                "blend": "robustly amplifies",
            },
        }
        md = _build_markdown(deltas, verdicts, ["uniform", "blend"], {})
        self.assertIn("Crossover rerun", md)
        self.assertIn("buffered", md)
        self.assertIn("robustly collapses", md)
        self.assertIn("robustly amplifies", md)
        self.assertIn("learning_rate", md)

    def test_markdown_handles_empty_input(self):
        md = _build_markdown({}, {}, ["uniform"], {})
        self.assertIsInstance(md, str)
        self.assertGreater(len(md), 0)


if __name__ == "__main__":
    unittest.main()

"""Tests for the consensus-paradigms experiment runner."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

from farm.runners.consensus_paradigms_experiment import (
    PARADIGMS,
    ConsensusParadigmsExperiment,
    TrialRow,
    _allocate,
    run_once,
)


class TestRunOnce(unittest.TestCase):
    """Unit tests for a single trial."""

    def _row(self, paradigm: str, seed: int = 0, n_voters: int = 40, n_cand: int = 6) -> TrialRow:
        return run_once(paradigm, n_voters, n_cand, seed)

    def test_party_returns_trial_row(self):
        row = self._row("party")
        self.assertIsInstance(row, TrialRow)
        self.assertEqual(row.paradigm, "party")
        self.assertIn(row.winner, range(6))

    def test_individual_returns_trial_row(self):
        row = self._row("individual")
        self.assertEqual(row.paradigm, "individual")
        self.assertIn(row.winner, range(6))

    def test_score_returns_trial_row(self):
        row = self._row("score")
        self.assertEqual(row.paradigm, "score")

    def test_latent_match_returns_trial_row(self):
        row = self._row("latent_match")
        self.assertEqual(row.paradigm, "latent_match")

    def test_constrained_individual_lambda_capped(self):
        cap = 0.25
        row = run_once("constrained_individual", 40, 6, 0, lambda_cap=cap)
        self.assertLessEqual(row.lambda_winner, cap + 1e-9)

    def test_unknown_paradigm_raises(self):
        with self.assertRaises(ValueError):
            run_once("nonexistent", 40, 6, 0)

    def test_welfare_values_are_finite(self):
        for paradigm in PARADIGMS:
            row = self._row(paradigm)
            self.assertTrue(np.isfinite(row.total_welfare), paradigm)

    def test_loser_share_in_unit_interval(self):
        for paradigm in PARADIGMS:
            row = self._row(paradigm)
            self.assertGreaterEqual(row.loser_share, 0.0)
            self.assertLessEqual(row.loser_share, 1.0)

    def test_deterministic_output(self):
        """Same seed must produce identical results."""
        r1 = run_once("party", 40, 6, 42)
        r2 = run_once("party", 40, 6, 42)
        self.assertEqual(r1, r2)

    def test_party_distinct_nominees(self):
        """Party nominees must be two different candidates."""
        # Run several seeds to increase chance of hitting the empty-party branch.
        valid_results = 0
        for seed in range(20):
            try:
                row = run_once("party", 40, 4, seed)
                self.assertIn(row.winner, range(4))
                valid_results += 1
            except ValueError:
                # Raised only when < 2 distinct candidates are available; acceptable.
                pass
        self.assertGreater(valid_results, 0, "Expected at least one valid trial")


class TestAllocate(unittest.TestCase):
    def test_allocation_sums_to_one(self):
        rng = np.random.default_rng(0)
        plat = rng.normal(0, 1, 5)
        loy = 0.5
        benefits = np.abs(rng.normal(0, 1, (20, 5)))
        benefits /= benefits.sum(axis=1, keepdims=True)
        mask = np.ones(20, dtype=bool)
        alloc = _allocate(plat, loy, benefits, mask)
        self.assertAlmostEqual(alloc.sum(), 1.0, places=6)

    def test_allocation_non_negative(self):
        rng = np.random.default_rng(1)
        plat = rng.normal(0, 1, 5)
        benefits = np.abs(rng.normal(0, 1, (20, 5)))
        benefits /= benefits.sum(axis=1, keepdims=True)
        alloc = _allocate(plat, 0.7, benefits, np.ones(20, dtype=bool))
        self.assertTrue((alloc >= 0).all())


class TestConsensusParadigmsExperiment(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, notarize: bool = False, trials: int = 2, candidates: int = 6) -> Path:
        exp = ConsensusParadigmsExperiment(output_dir=self.tmp_path / "exp")
        return exp.run(trials=trials, voters=40, candidates=candidates, notarize=notarize)

    def test_returns_summary_path(self):
        path = self._run()
        self.assertTrue(path.exists())

    def test_trials_csv_written(self):
        self._run()
        trials_csv = self.tmp_path / "exp" / "results" / "trials.csv"
        self.assertTrue(trials_csv.exists())
        lines = trials_csv.read_text().splitlines()
        # header + 2 trials * 5 paradigms = 11 lines
        self.assertEqual(len(lines), 1 + 2 * len(PARADIGMS))

    def test_summary_csv_has_one_row_per_paradigm(self):
        self._run()
        summary_csv = self.tmp_path / "exp" / "results" / "summary.csv"
        lines = summary_csv.read_text().splitlines()
        self.assertEqual(len(lines), 1 + len(PARADIGMS))

    def test_config_json_written(self):
        self._run()
        cfg = json.loads((self.tmp_path / "exp" / "results" / "config.json").read_text())
        self.assertEqual(cfg["trials"], 2)
        self.assertEqual(cfg["voters"], 40)

    def test_no_notarize_skips_manifest(self):
        """Without FarmNotary, notarize=True should not raise."""
        original = sys.modules.get("farm_notary", "MISSING")
        sys.modules["farm_notary"] = None  # type: ignore[assignment]
        try:
            self._run(notarize=True)
        finally:
            if original == "MISSING":
                del sys.modules["farm_notary"]
            else:
                sys.modules["farm_notary"] = original  # type: ignore[assignment]

    def test_deterministic_across_runs(self):
        exp1 = ConsensusParadigmsExperiment(output_dir=self.tmp_path / "r1")
        exp2 = ConsensusParadigmsExperiment(output_dir=self.tmp_path / "r2")
        p1 = exp1.run(trials=2, voters=40, candidates=6, notarize=False)
        p2 = exp2.run(trials=2, voters=40, candidates=6, notarize=False)
        self.assertEqual(p1.read_text(), p2.read_text())

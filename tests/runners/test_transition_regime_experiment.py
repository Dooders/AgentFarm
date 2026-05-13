"""Tests for the transition-regime experiment runner."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from farm.runners.intrinsic_evolution_experiment import STABLE_SUB_PROFILES
from scripts import run_transition_regime_experiment as runner_mod


class TestResourceBufferLine(unittest.TestCase):
    def test_resource_buffer_line_matches_stable_sub_profiles(self):
        for profile, level in [
            ("conservative", 8.0),
            ("balanced", 10.0),
            ("buffered", 12.0),
        ]:
            with self.subTest(profile=profile):
                resolved = runner_mod.resolve_resource_buffer_line(level)
                expected = STABLE_SUB_PROFILES[profile]
                self.assertEqual(
                    resolved["initial_agent_resource_level"],
                    float(expected["initial_agent_resource_level"]),
                )
                self.assertEqual(
                    resolved["initial_resource_count"],
                    expected["initial_resource_count"],
                )
                self.assertAlmostEqual(
                    resolved["resource_regen_rate"],
                    expected["resource_regen_rate"],
                )
                self.assertEqual(
                    resolved["resource_regen_amount"],
                    expected["resource_regen_amount"],
                )

    def test_fractional_level_preserved(self):
        resolved = runner_mod.resolve_resource_buffer_line(9.5)
        self.assertAlmostEqual(resolved["initial_agent_resource_level"], 9.5)
        self.assertEqual(resolved["initial_resource_count"], 34)
        self.assertAlmostEqual(resolved["resource_regen_rate"], 0.1475)


class TestTransitionRunnerFlow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_does_not_create_output_dir(self):
        out_dir = Path(self.tmp.name) / "dry"
        argv = [
            "run_transition_regime_experiment.py",
            "--resource-levels",
            "10",
            "--seeds",
            "42",
            "--output-dir",
            str(out_dir),
            "--dry-run",
        ]
        with patch.object(sys, "argv", argv):
            rc = runner_mod.main()
        self.assertEqual(rc, 0)
        self.assertFalse(out_dir.exists())

    def test_planned_runs_include_intervention_factors(self):
        args = runner_mod._build_parser().parse_args(
            [
                "--resource-levels",
                "9",
                "10",
                "--seeds",
                "1",
                "--interventions",
                "baseline",
                "crossover_on",
            ]
        )
        planned = runner_mod._planned_runs(args, Path("out"))
        self.assertEqual(len(planned), 4)
        interventions = {factor["intervention"] for factor in planned}
        self.assertEqual(interventions, {"baseline", "crossover_on"})
        self.assertTrue(any(factor["crossover_enabled"] for factor in planned))

    def test_execute_matrix_aborts_on_fail_fast(self):
        args = runner_mod._build_parser().parse_args(
            [
                "--resource-levels",
                "9",
                "10",
                "--seeds",
                "1",
                "2",
                "--fail-fast",
            ]
        )
        calls = []

        def fake_run_one(factor, _args, _logger):
            calls.append(factor)
            success = len(calls) == 1
            record = dict(factor)
            record["status"] = "ok" if success else "error"
            return success, record

        with patch.object(runner_mod, "_run_one", side_effect=fake_run_one):
            records, n_ok, n_fail = runner_mod.execute_matrix(args, Path("out"), _NullLogger())

        self.assertEqual(len(calls), 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(n_ok, 1)
        self.assertEqual(n_fail, 1)


class _NullLogger:
    def info(self, *_args, **_kwargs):
        pass

    def error(self, *_args, **_kwargs):
        pass


if __name__ == "__main__":
    unittest.main()

"""Smoke tests for the run_evolution_experiment CLI helpers."""

import importlib
import os
import sys
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPT_PATH not in sys.path:
    sys.path.insert(0, _SCRIPT_PATH)

run_evolution_experiment = importlib.import_module("run_evolution_experiment")


class TestParsePerGeneMultipliers(unittest.TestCase):
    def test_returns_empty_for_none(self):
        self.assertEqual(
            run_evolution_experiment._parse_per_gene_multipliers(None, label="x"),
            {},
        )

    def test_returns_empty_for_blank(self):
        self.assertEqual(
            run_evolution_experiment._parse_per_gene_multipliers("", label="x"),
            {},
        )

    def test_parses_single_pair(self):
        self.assertEqual(
            run_evolution_experiment._parse_per_gene_multipliers(
                "learning_rate=0.5", label="x"
            ),
            {"learning_rate": 0.5},
        )

    def test_parses_multiple_pairs_with_whitespace(self):
        result = run_evolution_experiment._parse_per_gene_multipliers(
            " learning_rate=0.5 , gamma=2.0 ", label="x"
        )
        self.assertEqual(result, {"learning_rate": 0.5, "gamma": 2.0})

    def test_rejects_missing_equals(self):
        with self.assertRaises(ValueError):
            run_evolution_experiment._parse_per_gene_multipliers(
                "learning_rate0.5", label="x"
            )

    def test_rejects_non_numeric_value(self):
        with self.assertRaises(ValueError):
            run_evolution_experiment._parse_per_gene_multipliers(
                "learning_rate=abc", label="x"
            )

    def test_rejects_empty_gene_name(self):
        with self.assertRaises(ValueError):
            run_evolution_experiment._parse_per_gene_multipliers(
                "=0.5", label="x"
            )


class TestAdaptiveCliFlags(unittest.TestCase):
    def test_adaptive_flags_round_trip_into_namespace(self):
        argv = sys.argv[:]
        try:
            sys.argv = [
                "run_evolution_experiment.py",
                "--generations", "1",
                "--population-size", "2",
                "--steps-per-candidate", "1",
                "--adaptive-mutation",
                "--adaptive-improve-threshold", "0.01",
                "--adaptive-stall-multiplier", "1.7",
                "--adaptive-improve-multiplier", "0.7",
                "--adaptive-diversity-threshold", "0.04",
                "--adaptive-diversity-multiplier", "1.6",
                "--adaptive-per-gene-rate", "learning_rate=0.5,gamma=2.0",
                "--adaptive-per-gene-scale", "learning_rate=0.25",
            ]
            args = run_evolution_experiment._parse_args()
        finally:
            sys.argv = argv
        self.assertTrue(args.adaptive_mutation)
        self.assertAlmostEqual(args.adaptive_improve_threshold, 0.01)
        self.assertAlmostEqual(args.adaptive_stall_multiplier, 1.7)
        self.assertAlmostEqual(args.adaptive_improve_multiplier, 0.7)
        self.assertEqual(args.adaptive_per_gene_rate, "learning_rate=0.5,gamma=2.0")
        self.assertEqual(args.adaptive_per_gene_scale, "learning_rate=0.25")


if __name__ == "__main__":
    unittest.main()

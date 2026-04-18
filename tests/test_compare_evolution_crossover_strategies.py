"""Unit tests for crossover strategy comparison report helpers."""

import importlib
import os
import sys
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts")
if _SCRIPT_PATH not in sys.path:
    sys.path.insert(0, _SCRIPT_PATH)

compare_evolution_crossover_strategies = importlib.import_module(
    "compare_evolution_crossover_strategies"
)


class TestCompareEvolutionCrossoverStrategies(unittest.TestCase):
    def test_parse_modes_returns_enum_values(self):
        modes = compare_evolution_crossover_strategies._parse_modes("uniform,blend,multi_point")
        self.assertEqual([mode.value for mode in modes], ["uniform", "blend", "multi_point"])

    def test_parse_modes_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            compare_evolution_crossover_strategies._parse_modes("uniform,not_a_mode")

    def test_summarize_mode_runs_includes_diversity_counts(self):
        mode = compare_evolution_crossover_strategies.CrossoverMode.UNIFORM
        runs = [
            {
                "mode": "uniform",
                "seed": 1,
                "best_candidate_fitness": 10.0,
                "final_best_fitness": 8.0,
                "final_mean_fitness": 5.0,
                "final_diversity": None,
            },
            {
                "mode": "uniform",
                "seed": 2,
                "best_candidate_fitness": 12.0,
                "final_best_fitness": 9.0,
                "final_mean_fitness": 6.0,
                "final_diversity": 0.2,
            },
        ]
        summary = compare_evolution_crossover_strategies._summarize_mode_runs(mode, runs)
        self.assertEqual(summary["mode"], "uniform")
        self.assertEqual(summary["num_runs"], 2)
        self.assertEqual(summary["num_runs_with_diversity"], 1)
        self.assertIsNotNone(summary["final_best_fitness"])
        self.assertIsNotNone(summary["final_mean_fitness"])
        self.assertIsNotNone(summary["final_diversity"])


if __name__ == "__main__":
    unittest.main()

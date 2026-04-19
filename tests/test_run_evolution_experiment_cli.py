"""Smoke tests for the run_evolution_experiment CLI helpers."""

import importlib
import json
import os
import sys
import tempfile
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

    def test_crossover_flags_round_trip_into_namespace(self):
        argv = sys.argv[:]
        try:
            sys.argv = [
                "run_evolution_experiment.py",
                "--generations", "1",
                "--population-size", "2",
                "--steps-per-candidate", "1",
                "--crossover-mode", "multi_point",
                "--blend-alpha", "0.9",
                "--num-crossover-points", "4",
            ]
            args = run_evolution_experiment._parse_args()
        finally:
            sys.argv = argv
        self.assertEqual(args.crossover_mode, "multi_point")
        self.assertAlmostEqual(args.blend_alpha, 0.9)
        self.assertEqual(args.num_crossover_points, 4)


class TestPresets(unittest.TestCase):
    def test_stable_hyper_evo_preset_exists(self):
        self.assertIn("stable_hyper_evo", run_evolution_experiment.PRESETS)

    def test_stable_hyper_evo_preset_has_required_keys(self):
        preset = run_evolution_experiment.PRESETS["stable_hyper_evo"]
        self.assertIn("selection_method", preset)
        self.assertIn("boundary_mode", preset)
        self.assertIn("mutation_rate", preset)
        self.assertIn("mutation_scale", preset)
        self.assertIn("adaptive_mutation", preset)

    def test_stable_hyper_evo_preset_selection_is_tournament(self):
        preset = run_evolution_experiment.PRESETS["stable_hyper_evo"]
        self.assertEqual(preset["selection_method"], "tournament")

    def test_stable_hyper_evo_preset_boundary_mode_is_reflect(self):
        preset = run_evolution_experiment.PRESETS["stable_hyper_evo"]
        self.assertEqual(preset["boundary_mode"], "reflect")

    def test_stable_hyper_evo_preset_enables_adaptive_mutation(self):
        preset = run_evolution_experiment.PRESETS["stable_hyper_evo"]
        self.assertTrue(preset["adaptive_mutation"])

    def test_preset_values_applied_when_flag_given(self):
        argv = sys.argv[:]
        try:
            sys.argv = [
                "run_evolution_experiment.py",
                "--preset", "stable_hyper_evo",
                "--generations", "1",
                "--population-size", "2",
                "--steps-per-candidate", "1",
            ]
            args = run_evolution_experiment._parse_args()
        finally:
            sys.argv = argv
        self.assertEqual(args.preset, "stable_hyper_evo")
        self.assertEqual(args.selection_method, "tournament")
        self.assertEqual(args.boundary_mode, "reflect")
        self.assertAlmostEqual(args.mutation_rate, 0.20)
        self.assertAlmostEqual(args.mutation_scale, 0.15)
        self.assertTrue(args.adaptive_mutation)

    def test_explicit_flag_overrides_preset_value(self):
        argv = sys.argv[:]
        try:
            sys.argv = [
                "run_evolution_experiment.py",
                "--preset", "stable_hyper_evo",
                "--generations", "1",
                "--population-size", "2",
                "--steps-per-candidate", "1",
                "--mutation-rate", "0.5",
                "--boundary-mode", "clamp",
            ]
            args = run_evolution_experiment._parse_args()
        finally:
            sys.argv = argv
        # Preset defaults: mutation_rate=0.20, boundary_mode=reflect
        # Explicit flags should win.
        self.assertAlmostEqual(args.mutation_rate, 0.5)
        self.assertEqual(args.boundary_mode, "clamp")
        # Other preset values still applied.
        self.assertEqual(args.selection_method, "tournament")
        self.assertTrue(args.adaptive_mutation)

    def test_no_preset_uses_original_defaults(self):
        argv = sys.argv[:]
        try:
            sys.argv = [
                "run_evolution_experiment.py",
                "--generations", "1",
                "--population-size", "2",
                "--steps-per-candidate", "1",
            ]
            args = run_evolution_experiment._parse_args()
        finally:
            sys.argv = argv
        self.assertIsNone(args.preset)
        self.assertAlmostEqual(args.mutation_rate, 0.25)
        self.assertEqual(args.boundary_mode, "clamp")
        self.assertFalse(args.adaptive_mutation)


class TestRunManifest(unittest.TestCase):
    def test_manifest_written_to_output_dir(self):
        """main() should persist run_manifest.json in output_dir before the run."""
        from unittest.mock import patch, MagicMock

        with tempfile.TemporaryDirectory() as output_dir:
            argv = sys.argv[:]
            try:
                sys.argv = [
                    "run_evolution_experiment.py",
                    "--preset", "stable_hyper_evo",
                    "--generations", "1",
                    "--population-size", "2",
                    "--steps-per-candidate", "1",
                    "--output-dir", output_dir,
                ]
                fake_result = MagicMock()
                fake_result.generation_summaries = []
                fake_result.evaluations = []
                fake_result.best_candidate.candidate_id = "test-id"
                fake_result.best_candidate.fitness = 1.0
                fake_result.best_candidate.learning_rate = 0.001
                fake_result.best_candidate.parent_ids = []

                with patch.object(
                    run_evolution_experiment.EvolutionExperiment,
                    "run",
                    return_value=fake_result,
                ):
                    run_evolution_experiment.main()
            finally:
                sys.argv = argv

            manifest_path = os.path.join(output_dir, "run_manifest.json")
            self.assertTrue(os.path.isfile(manifest_path))
            with open(manifest_path) as f:
                manifest = json.load(f)

            self.assertEqual(manifest["preset"], "stable_hyper_evo")
            self.assertEqual(manifest["selection_method"], "tournament")
            self.assertEqual(manifest["boundary_mode"], "reflect")
            self.assertAlmostEqual(manifest["mutation_rate"], 0.20)
            self.assertAlmostEqual(manifest["mutation_scale"], 0.15)
            self.assertTrue(manifest["adaptive_mutation"])
            self.assertEqual(manifest["script"], "scripts/run_evolution_experiment.py")

    def test_manifest_preset_is_null_when_no_preset_given(self):
        """run_manifest.json should include preset=null when no --preset flag is used."""
        from unittest.mock import patch, MagicMock

        with tempfile.TemporaryDirectory() as output_dir:
            argv = sys.argv[:]
            try:
                sys.argv = [
                    "run_evolution_experiment.py",
                    "--generations", "1",
                    "--population-size", "2",
                    "--steps-per-candidate", "1",
                    "--output-dir", output_dir,
                ]
                fake_result = MagicMock()
                fake_result.generation_summaries = []
                fake_result.evaluations = []
                fake_result.best_candidate.candidate_id = "test-id"
                fake_result.best_candidate.fitness = 1.0
                fake_result.best_candidate.learning_rate = 0.001
                fake_result.best_candidate.parent_ids = []

                with patch.object(
                    run_evolution_experiment.EvolutionExperiment,
                    "run",
                    return_value=fake_result,
                ):
                    run_evolution_experiment.main()
            finally:
                sys.argv = argv

            manifest_path = os.path.join(output_dir, "run_manifest.json")
            with open(manifest_path) as f:
                manifest = json.load(f)
            self.assertIsNone(manifest["preset"])


if __name__ == "__main__":
    unittest.main()

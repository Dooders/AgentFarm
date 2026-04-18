"""Tests for multi-generation hyperparameter evolution runner."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import BoundaryMode, BoundaryPenaltyConfig
from farm.runners.evolution_experiment import (
    EvolutionExperiment,
    EvolutionExperimentConfig,
    EvolutionFitnessMetric,
)


class TestEvolutionExperimentConfig(unittest.TestCase):
    def test_rejects_invalid_population_size(self):
        with self.assertRaises(ValueError):
            EvolutionExperimentConfig(population_size=1)

    def test_rejects_invalid_generation_count(self):
        with self.assertRaises(ValueError):
            EvolutionExperimentConfig(num_generations=0)

    def test_default_boundary_mode_is_clamp(self):
        config = EvolutionExperimentConfig()
        self.assertEqual(config.boundary_mode, BoundaryMode.CLAMP)

    def test_accepts_reflect_boundary_mode(self):
        config = EvolutionExperimentConfig(boundary_mode=BoundaryMode.REFLECT)
        self.assertEqual(config.boundary_mode, BoundaryMode.REFLECT)

    def test_default_boundary_penalty_config_is_none(self):
        config = EvolutionExperimentConfig()
        self.assertIsNone(config.boundary_penalty_config)

    def test_accepts_boundary_penalty_config(self):
        penalty_cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.05)
        config = EvolutionExperimentConfig(boundary_penalty_config=penalty_cfg)
        self.assertTrue(config.boundary_penalty_config.enabled)


class TestEvolutionExperiment(unittest.TestCase):
    def test_runs_two_generations_with_population_four(self):
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=4,
            num_steps_per_candidate=1,
            seed=7,
        )
        experiment = EvolutionExperiment(base_config, config)
        evaluation_calls = []

        def evaluator(candidate, candidate_config, generation, member_index):
            evaluation_calls.append((candidate.candidate_id, generation, member_index))
            fitness = candidate.chromosome.get_value("learning_rate") * 1000.0
            return fitness, {"member_index": member_index}

        result = experiment.run(fitness_evaluator=evaluator)

        self.assertEqual(len(result.generation_summaries), 2)
        self.assertEqual(len(result.evaluations), 8)
        self.assertEqual(len(evaluation_calls), 8)
        self.assertTrue(result.best_candidate.learning_rate > 0.0)
        self.assertEqual(result.evaluations[0].generation, 0)
        self.assertEqual(result.evaluations[-1].generation, 1)
        self.assertEqual(result.evaluations[4].parent_ids[0][:2], "g0")

    def test_custom_fitness_receives_learning_rate_applied_config(self):
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=4,
            num_steps_per_candidate=1,
            seed=11,
        )
        experiment = EvolutionExperiment(base_config, config)
        seen_learning_rates = []

        def evaluator(candidate, candidate_config, generation, member_index):
            seen_learning_rates.append(candidate_config.learning.learning_rate)
            fitness = candidate_config.learning.learning_rate
            return fitness, {"generation": generation, "member": member_index}

        result = experiment.run(fitness_evaluator=evaluator)
        self.assertEqual(len(seen_learning_rates), 8)
        self.assertTrue(any(rate != base_config.learning.learning_rate for rate in seen_learning_rates[1:]))
        self.assertGreater(result.best_candidate.fitness, 0.0)

    @patch("farm.runners.evolution_experiment.run_simulation")
    def test_default_fitness_metric_uses_environment_summary(self, run_simulation_mock):
        base_config = SimulationConfig()
        run_simulation_mock.return_value = SimpleNamespace(
            agents=["a", "b", "c"],
            cached_total_resources=42.0,
            metrics_tracker=SimpleNamespace(
                cumulative_metrics=SimpleNamespace(total_births=5),
            ),
        )
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=4,
            num_steps_per_candidate=1,
            fitness_metric=EvolutionFitnessMetric.FINAL_POPULATION,
        )
        result = EvolutionExperiment(base_config, config).run()
        self.assertEqual(len(result.evaluations), 8)
        self.assertEqual(result.best_candidate.fitness, 3.0)
        self.assertEqual(run_simulation_mock.call_count, 8)

    def test_persists_generation_and_lineage_json(self):
        base_config = SimulationConfig()
        with tempfile.TemporaryDirectory() as output_dir:
            config = EvolutionExperimentConfig(
                num_generations=2,
                population_size=4,
                num_steps_per_candidate=1,
                output_dir=output_dir,
            )
            experiment = EvolutionExperiment(base_config, config)
            experiment.run(
                fitness_evaluator=lambda candidate, cfg, generation, member: (
                    float(member + generation),
                    {"member": member},
                )
            )

            with open(
                f"{output_dir}/evolution_generation_summaries.json",
                encoding="utf-8",
            ) as summaries_file:
                summaries = json.load(summaries_file)

            with open(
                f"{output_dir}/evolution_lineage.json",
                encoding="utf-8",
            ) as lineage_file:
                lineage = json.load(lineage_file)

            self.assertEqual(len(summaries), 2)
            self.assertEqual(len(lineage), 8)
            self.assertEqual(lineage[0]["generation"], 0)
            self.assertEqual(lineage[-1]["generation"], 1)
            self.assertIn("gene_statistics", summaries[0])
            self.assertIn("learning_rate", summaries[0]["gene_statistics"])
            self.assertIn("mean", summaries[0]["gene_statistics"]["learning_rate"])
            self.assertIn("median", summaries[0]["gene_statistics"]["learning_rate"])
            self.assertIn("std", summaries[0]["gene_statistics"]["learning_rate"])
            self.assertIn("best_chromosome", summaries[0])
            self.assertIn("learning_rate", summaries[0]["best_chromosome"])

    def test_repeated_run_on_same_instance_is_deterministic_with_seed(self):
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=4,
            num_steps_per_candidate=1,
            seed=17,
        )
        experiment = EvolutionExperiment(base_config, config)

        def evaluator(candidate, candidate_config, generation, member_index):
            fitness = candidate_config.learning.learning_rate * 1000.0
            return fitness, {"generation": generation, "member": member_index}

        result_a = experiment.run(fitness_evaluator=evaluator)
        result_b = experiment.run(fitness_evaluator=evaluator)

        lineage_a = [(ev.candidate_id, ev.parent_ids, ev.fitness) for ev in result_a.evaluations]
        lineage_b = [(ev.candidate_id, ev.parent_ids, ev.fitness) for ev in result_b.evaluations]
        self.assertEqual(lineage_a, lineage_b)
        self.assertEqual(result_a.best_candidate.candidate_id, result_b.best_candidate.candidate_id)

    def test_boundary_penalty_reduces_fitness_when_enabled(self):
        base_config = SimulationConfig()
        penalty_cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=1000.0, near_boundary_threshold=0.5)
        config_with_penalty = EvolutionExperimentConfig(
            num_generations=1,
            population_size=2,
            seed=42,
            boundary_penalty_config=penalty_cfg,
        )
        config_without_penalty = EvolutionExperimentConfig(
            num_generations=1,
            population_size=2,
            seed=42,
        )

        raw_fitness = 10.0

        def evaluator(candidate, candidate_config, generation, member_index):
            return raw_fitness, {}

        result_with = EvolutionExperiment(base_config, config_with_penalty).run(fitness_evaluator=evaluator)
        result_without = EvolutionExperiment(base_config, config_without_penalty).run(fitness_evaluator=evaluator)

        for eval_with, eval_without in zip(result_with.evaluations, result_without.evaluations):
            self.assertLessEqual(eval_with.fitness, eval_without.fitness)

    def test_boundary_penalty_metadata_recorded_when_nonzero(self):
        base_config = SimulationConfig()
        penalty_cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=1000.0, near_boundary_threshold=0.5)
        config = EvolutionExperimentConfig(
            num_generations=1,
            population_size=2,
            seed=5,
            boundary_penalty_config=penalty_cfg,
        )
        raw_fitness = 50.0

        def evaluator(candidate, candidate_config, generation, member_index):
            return raw_fitness, {}

        result = EvolutionExperiment(base_config, config).run(fitness_evaluator=evaluator)
        penalized = [ev for ev in result.evaluations if "boundary_penalty" in ev.metadata]
        self.assertTrue(len(penalized) > 0)
        for ev in penalized:
            penalty = ev.metadata["boundary_penalty"]
            self.assertGreater(penalty, 0.0)
            self.assertAlmostEqual(ev.fitness, raw_fitness - penalty)


if __name__ == "__main__":
    unittest.main()

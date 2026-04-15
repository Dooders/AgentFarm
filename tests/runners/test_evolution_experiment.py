"""Tests for multi-generation hyperparameter evolution runner."""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from farm.config import SimulationConfig
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


if __name__ == "__main__":
    unittest.main()

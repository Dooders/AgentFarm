"""Tests for multi-generation hyperparameter evolution runner."""

import json
import tempfile
import unittest
from dataclasses import fields
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
    def test_config_fields_define_boundary_settings_once(self):
        field_names = [item.name for item in fields(EvolutionExperimentConfig)]
        self.assertEqual(field_names.count("boundary_mode"), 1)
        self.assertEqual(field_names.count("boundary_penalty"), 1)

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

    def test_rejects_invalid_boundary_mode(self):
        with self.assertRaises(ValueError):
            EvolutionExperimentConfig(boundary_mode="not-a-mode")

    def test_default_boundary_penalty_config_uses_defaults(self):
        config = EvolutionExperimentConfig()
        self.assertFalse(config.boundary_penalty.enabled)
        self.assertEqual(config.boundary_penalty.penalty_strength, 0.01)

    def test_accepts_boundary_penalty_config(self):
        penalty_cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.05)
        config = EvolutionExperimentConfig(boundary_penalty=penalty_cfg)
        self.assertTrue(config.boundary_penalty.enabled)


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

    def test_boundary_penalty_adjusts_candidate_fitness(self):
        base_config = SimulationConfig()
        base_config.learning.learning_rate = 1e-6
        config = EvolutionExperimentConfig(
            num_generations=1,
            population_size=3,
            num_steps_per_candidate=1,
            mutation_scale=0.0,
            boundary_penalty=BoundaryPenaltyConfig(
                enabled=True,
                penalty_strength=0.2,
                near_boundary_threshold=0.05,
            ),
            seed=23,
        )
        experiment = EvolutionExperiment(base_config, config)

        def evaluator(candidate, candidate_config, generation, member_index):
            return 1.0, {"member": member_index}

        result = experiment.run(fitness_evaluator=evaluator)
        self.assertEqual(len(result.evaluations), 3)
        for evaluation in result.evaluations:
            self.assertAlmostEqual(evaluation.metadata["raw_fitness"], 1.0)
            self.assertAlmostEqual(evaluation.metadata["boundary_penalty"], 0.2)
            self.assertAlmostEqual(evaluation.fitness, 0.8)

    @patch("farm.runners.evolution_experiment.mutate_chromosome")
    def test_boundary_mode_is_forwarded_to_mutation_calls(self, mutate_mock):
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=4,
            num_steps_per_candidate=1,
            boundary_mode=BoundaryMode.REFLECT,
            seed=31,
        )
        mutate_mock.side_effect = lambda chromosome, **kwargs: chromosome
        experiment = EvolutionExperiment(base_config, config)
        experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                float(member + generation),
                {"member": member},
            )
        )

        self.assertTrue(mutate_mock.called)
        self.assertTrue(
            all(call.kwargs.get("boundary_mode") == BoundaryMode.REFLECT for call in mutate_mock.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()

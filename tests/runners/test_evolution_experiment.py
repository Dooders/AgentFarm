"""Tests for multi-generation hyperparameter evolution runner."""

import json
import tempfile
import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import patch

from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import BoundaryMode, BoundaryPenaltyConfig, CrossoverMode
from farm.runners.adaptive_mutation import AdaptiveMutationConfig
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

    def test_blend_mode_rejects_negative_blend_alpha(self):
        with self.assertRaises(ValueError):
            EvolutionExperimentConfig(
                crossover_mode=CrossoverMode.BLEND,
                blend_alpha=-0.1,
            )

    def test_non_blend_mode_rejects_negative_blend_alpha(self):
        with self.assertRaises(ValueError):
            EvolutionExperimentConfig(
                crossover_mode=CrossoverMode.UNIFORM,
                blend_alpha=-0.1,
            )

    def test_multi_point_mode_rejects_zero_crossover_points(self):
        with self.assertRaises(ValueError):
            EvolutionExperimentConfig(
                crossover_mode=CrossoverMode.MULTI_POINT,
                num_crossover_points=0,
            )

    def test_non_multi_point_mode_rejects_zero_crossover_points(self):
        with self.assertRaises(ValueError):
            EvolutionExperimentConfig(
                crossover_mode=CrossoverMode.UNIFORM,
                num_crossover_points=0,
            )


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
        # Move all other evolvable genes well inside bounds so only learning_rate
        # (at its min boundary) incurs a penalty, keeping the expected values simple.
        base_config.learning.gamma = 0.5
        base_config.learning.epsilon_decay = 0.5
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

    @patch("farm.runners.evolution_experiment.mutate_chromosome")
    @patch("farm.runners.evolution_experiment.crossover_chromosomes")
    def test_crossover_mode_parameters_are_forwarded(self, crossover_mock, mutate_mock):
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=4,
            num_steps_per_candidate=1,
            crossover_mode=CrossoverMode.MULTI_POINT,
            blend_alpha=0.7,
            num_crossover_points=3,
            seed=37,
        )
        crossover_mock.side_effect = lambda parent_a, parent_b, **kwargs: parent_a
        mutate_mock.side_effect = lambda chromosome, **kwargs: chromosome
        experiment = EvolutionExperiment(base_config, config)
        experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                float(member + generation),
                {"member": member},
            )
        )

        self.assertTrue(crossover_mock.called)
        for call in crossover_mock.call_args_list:
            self.assertEqual(call.kwargs.get("mode"), CrossoverMode.MULTI_POINT)
            self.assertAlmostEqual(call.kwargs.get("blend_alpha"), 0.7)
            self.assertEqual(call.kwargs.get("num_crossover_points"), 3)


class TestEvolutionExperimentAdaptiveMutation(unittest.TestCase):
    def test_summary_telemetry_describes_what_produced_each_generation(self):
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=4,
            num_steps_per_candidate=1,
            mutation_rate=0.25,
            mutation_scale=0.2,
            seed=5,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                float(member + generation),
                {"member": member},
            )
        )
        # Generation 0 is seeded by `_initialize_population`, not by the
        # adaptive controller, so its mutation telemetry is None and the
        # event is the special `initial_seeding` tag.
        gen0 = result.generation_summaries[0]
        self.assertIsNone(gen0.mutation_rate_used)
        self.assertIsNone(gen0.mutation_scale_used)
        self.assertIsNone(gen0.mutation_rate_multiplier)
        self.assertIsNone(gen0.mutation_scale_multiplier)
        self.assertEqual(gen0.adaptive_event, "initial_seeding")
        # Diversity is always measured on the current generation.
        self.assertIsNotNone(gen0.diversity)
        self.assertGreaterEqual(gen0.diversity, 0.0)

        # Subsequent generations were produced by the controller (in this
        # case "disabled" because adaptive_mutation defaults to off).
        gen1 = result.generation_summaries[1]
        self.assertAlmostEqual(gen1.mutation_rate_used, 0.25)
        self.assertAlmostEqual(gen1.mutation_scale_used, 0.2)
        self.assertAlmostEqual(gen1.mutation_rate_multiplier, 1.0)
        self.assertAlmostEqual(gen1.mutation_scale_multiplier, 1.0)
        self.assertEqual(gen1.adaptive_event, "disabled")

    @patch("farm.runners.evolution_experiment.mutate_chromosome")
    def test_adaptive_stall_boosts_effective_mutation_for_next_generation(self, mutate_mock):
        mutate_mock.side_effect = lambda chromosome, **kwargs: chromosome
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=3,
            population_size=3,
            num_steps_per_candidate=1,
            mutation_rate=0.2,
            mutation_scale=0.1,
            adaptive_mutation=AdaptiveMutationConfig(
                enabled=True,
                use_fitness_adaptation=True,
                use_diversity_adaptation=False,
                stall_window=1,
                stall_multiplier=2.0,
                improvement_threshold=1e-6,
                max_rate_multiplier=4.0,
                max_scale_multiplier=4.0,
            ),
            seed=42,
        )
        experiment = EvolutionExperiment(base_config, config)
        # Flat fitness => every generation after the first is a stall.
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (1.0, {"member": member})
        )
        gen0, gen1, gen2 = result.generation_summaries
        # Gen 0 is seeded; gen 1 was produced before any stall observation
        # had taken effect, so its multiplier is still 1.0.  Gen 2 reflects
        # the post-gen-1 stall.
        self.assertIsNone(gen0.mutation_rate_multiplier)
        self.assertAlmostEqual(gen1.mutation_rate_multiplier, 1.0)
        self.assertGreater(gen2.mutation_rate_multiplier, gen1.mutation_rate_multiplier)
        self.assertAlmostEqual(gen2.mutation_rate_used, 0.2 * gen2.mutation_rate_multiplier)
        self.assertIn("stalled", gen2.adaptive_event)

    @patch("farm.runners.evolution_experiment.mutate_chromosome")
    def test_adaptive_improving_tightens_effective_mutation(self, mutate_mock):
        mutate_mock.side_effect = lambda chromosome, **kwargs: chromosome
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=3,
            population_size=3,
            num_steps_per_candidate=1,
            mutation_rate=0.4,
            mutation_scale=0.2,
            adaptive_mutation=AdaptiveMutationConfig(
                enabled=True,
                use_fitness_adaptation=True,
                use_diversity_adaptation=False,
                stall_window=1,
                improve_multiplier=0.5,
                min_rate_multiplier=0.01,
                min_scale_multiplier=0.01,
            ),
            seed=99,
        )
        experiment = EvolutionExperiment(base_config, config)
        fitness_by_generation = {0: 1.0, 1: 2.0, 2: 3.0}
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                fitness_by_generation[generation],
                {"member": member},
            )
        )
        gen0, gen1, gen2 = result.generation_summaries
        self.assertIsNone(gen0.mutation_rate_multiplier)
        # Gen 1 was produced before any observe() had been called, so the
        # multiplier is still 1.0.  Gen 2 reflects the gen-1 improvement.
        self.assertAlmostEqual(gen1.mutation_rate_multiplier, 1.0)
        self.assertLess(gen2.mutation_rate_multiplier, gen1.mutation_rate_multiplier)
        self.assertIn("improving", gen2.adaptive_event)

    @patch("farm.runners.evolution_experiment.mutate_chromosome")
    def test_per_gene_multipliers_forwarded_to_mutate(self, mutate_mock):
        mutate_mock.side_effect = lambda chromosome, **kwargs: chromosome
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=3,
            num_steps_per_candidate=1,
            adaptive_mutation=AdaptiveMutationConfig(
                enabled=True,
                use_fitness_adaptation=False,
                use_diversity_adaptation=False,
                per_gene_rate_multipliers={"learning_rate": 2.0},
                per_gene_scale_multipliers={"learning_rate": 0.5},
            ),
            seed=8,
        )
        experiment = EvolutionExperiment(base_config, config)
        experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                float(member),
                {"member": member},
            )
        )
        self.assertTrue(mutate_mock.called)
        # Child generations call mutate_chromosome with per-gene multiplier dicts.
        # The initial population seeding uses mutation_rate=1.0 with no
        # adaptive per-gene multipliers, so filter those out.
        child_calls = [
            call for call in mutate_mock.call_args_list
            if call.kwargs.get("per_gene_rate_multipliers") is not None
        ]
        self.assertGreater(len(child_calls), 0)
        for call in child_calls:
            self.assertEqual(call.kwargs.get("per_gene_rate_multipliers"), {"learning_rate": 2.0})
            self.assertEqual(call.kwargs.get("per_gene_scale_multipliers"), {"learning_rate": 0.5})

    def test_adaptive_telemetry_persisted_to_generation_summaries(self):
        base_config = SimulationConfig()
        with tempfile.TemporaryDirectory() as output_dir:
            config = EvolutionExperimentConfig(
                num_generations=2,
                population_size=3,
                num_steps_per_candidate=1,
                mutation_rate=0.2,
                mutation_scale=0.1,
                adaptive_mutation=AdaptiveMutationConfig(
                    enabled=True,
                    use_fitness_adaptation=True,
                    use_diversity_adaptation=False,
                    stall_window=1,
                    stall_multiplier=2.0,
                ),
                output_dir=output_dir,
                seed=1,
            )
            experiment = EvolutionExperiment(base_config, config)
            experiment.run(
                fitness_evaluator=lambda candidate, cfg, generation, member: (
                    1.0,
                    {"member": member},
                )
            )
            with open(
                f"{output_dir}/evolution_generation_summaries.json",
                encoding="utf-8",
            ) as summaries_file:
                summaries = json.load(summaries_file)
            self.assertEqual(len(summaries), 2)
            for summary in summaries:
                self.assertIn("mutation_rate_used", summary)
                self.assertIn("mutation_scale_used", summary)
                self.assertIn("mutation_rate_multiplier", summary)
                self.assertIn("mutation_scale_multiplier", summary)
                self.assertIn("diversity", summary)
                self.assertIn("adaptive_event", summary)
            # Generation 0 was seeded, so its mutation telemetry is null.
            self.assertIsNone(summaries[0]["mutation_rate_used"])
            self.assertEqual(summaries[0]["adaptive_event"], "initial_seeding")
            # Generation 1 was produced before any controller observation
            # took effect, so multiplier is unity but event reflects the
            # very first observation made on gen 0.
            self.assertEqual(summaries[1]["mutation_rate_multiplier"], 1.0)

    def test_none_diversity_persists_as_json_null_and_skips_diversity_rule(self):
        base_config = SimulationConfig()
        with tempfile.TemporaryDirectory() as output_dir:
            config = EvolutionExperimentConfig(
                num_generations=2,
                population_size=3,
                num_steps_per_candidate=1,
                adaptive_mutation=AdaptiveMutationConfig(
                    enabled=True,
                    use_fitness_adaptation=False,
                    use_diversity_adaptation=True,
                    diversity_threshold=0.99,  # would always fire if diversity were measured
                ),
                output_dir=output_dir,
                seed=1,
            )
            experiment = EvolutionExperiment(base_config, config)
            with patch.object(EvolutionExperiment, "_compute_diversity", return_value=None):
                experiment.run(
                    fitness_evaluator=lambda candidate, cfg, generation, member: (
                        1.0,
                        {"member": member},
                    )
                )
            with open(
                f"{output_dir}/evolution_generation_summaries.json",
                encoding="utf-8",
            ) as summaries_file:
                summaries = json.load(summaries_file)
            for summary in summaries:
                self.assertIsNone(summary["diversity"])
            # No diversity_collapse fired because diversity was None.
            self.assertNotIn("diversity_collapse", summaries[1]["adaptive_event"])


if __name__ == "__main__":
    unittest.main()

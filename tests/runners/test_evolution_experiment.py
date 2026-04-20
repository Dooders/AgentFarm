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
            self.assertIn("chromosome", lineage[0])
            self.assertIn("learning_rate", lineage[0]["chromosome"])
            self.assertIn("gamma", lineage[0]["chromosome"])
            self.assertIn("epsilon_decay", lineage[0]["chromosome"])
            self.assertIn("gene_statistics", summaries[0])
            self.assertIn("learning_rate", summaries[0]["gene_statistics"])
            self.assertIn("gamma", summaries[0]["gene_statistics"])
            self.assertIn("epsilon_decay", summaries[0]["gene_statistics"])
            self.assertIn("mean", summaries[0]["gene_statistics"]["learning_rate"])
            self.assertIn("median", summaries[0]["gene_statistics"]["learning_rate"])
            self.assertIn("std", summaries[0]["gene_statistics"]["learning_rate"])
            self.assertIn("best_chromosome", summaries[0])
            self.assertIn("learning_rate", summaries[0]["best_chromosome"])
            self.assertIn("gamma", summaries[0]["best_chromosome"])
            self.assertIn("epsilon_decay", summaries[0]["best_chromosome"])

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

    def test_adaptive_warmup_uses_partial_history_before_full_stall_window(self):
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=3,
            population_size=3,
            num_steps_per_candidate=1,
            mutation_rate=0.2,
            adaptive_mutation=AdaptiveMutationConfig(
                enabled=True,
                use_fitness_adaptation=True,
                use_diversity_adaptation=False,
                stall_window=5,
                stall_multiplier=2.0,
                improvement_threshold=1e-6,
            ),
            seed=123,
        )
        experiment = EvolutionExperiment(base_config, config)
        # Flat fitness should trigger stall adaptation as soon as enough history
        # exists to compare latest-vs-prior best (without waiting for a full
        # stall_window+1 observations).
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (1.0, {"member": member})
        )
        _, gen1, gen2 = result.generation_summaries
        self.assertAlmostEqual(gen1.mutation_rate_multiplier, 1.0)
        self.assertGreater(gen2.mutation_rate_multiplier, 1.0)
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


class TestConvergenceCriteria(unittest.TestCase):
    def test_defaults_are_disabled(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        criteria = ConvergenceCriteria()
        self.assertFalse(criteria.enabled)

    def test_rejects_zero_fitness_window(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        with self.assertRaises(ValueError):
            ConvergenceCriteria(fitness_window=0)

    def test_rejects_negative_fitness_threshold(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        with self.assertRaises(ValueError):
            ConvergenceCriteria(fitness_threshold=-0.1)

    def test_rejects_zero_diversity_window(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        with self.assertRaises(ValueError):
            ConvergenceCriteria(diversity_window=0)

    def test_rejects_negative_diversity_threshold(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        with self.assertRaises(ValueError):
            ConvergenceCriteria(diversity_threshold=-0.1)

    def test_rejects_negative_min_generations(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        with self.assertRaises(ValueError):
            ConvergenceCriteria(min_generations=-1)

    def test_accepts_valid_config(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        criteria = ConvergenceCriteria(
            enabled=True,
            fitness_window=3,
            fitness_threshold=0.01,
            diversity_window=2,
            diversity_threshold=0.05,
            min_generations=2,
            early_stop=False,
        )
        self.assertTrue(criteria.enabled)
        self.assertEqual(criteria.fitness_window, 3)
        self.assertEqual(criteria.min_generations, 2)
        self.assertFalse(criteria.early_stop)

    def test_zero_threshold_is_accepted(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        # threshold=0 means any improvement (strictly > 0) avoids plateau.
        criteria = ConvergenceCriteria(fitness_threshold=0.0, diversity_threshold=0.0)
        self.assertEqual(criteria.fitness_threshold, 0.0)


class TestConvergenceDisabledRegressionMode(unittest.TestCase):
    """Regression: with convergence disabled all generations always run."""

    def test_disabled_convergence_runs_all_generations(self):
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=4,
            population_size=3,
            num_steps_per_candidate=1,
            seed=99,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (1.0, {"member": member})
        )
        self.assertEqual(len(result.generation_summaries), 4)
        self.assertFalse(result.converged)
        self.assertIsNone(result.convergence_reason)
        self.assertIsNone(result.generation_of_convergence)

    def test_disabled_convergence_persists_no_metadata_fields(self):
        base_config = SimulationConfig()
        with tempfile.TemporaryDirectory() as output_dir:
            config = EvolutionExperimentConfig(
                num_generations=2,
                population_size=3,
                num_steps_per_candidate=1,
                output_dir=output_dir,
                seed=77,
            )
            experiment = EvolutionExperiment(base_config, config)
            experiment.run(
                fitness_evaluator=lambda candidate, cfg, gen, member: (1.0, {"member": member})
            )
            import os as _os
            metadata_path = _os.path.join(output_dir, "evolution_metadata.json")
            with open(metadata_path, encoding="utf-8") as mf:
                metadata = json.load(mf)
            self.assertFalse(metadata["converged"])
            self.assertIsNone(metadata["convergence_reason"])
            self.assertIsNone(metadata["generation_of_convergence"])
            self.assertEqual(metadata["num_generations_completed"], 2)
            # Regression: existing summaries file is still an array.
            summaries_path = _os.path.join(output_dir, "evolution_generation_summaries.json")
            with open(summaries_path, encoding="utf-8") as sf:
                summaries = json.load(sf)
            self.assertIsInstance(summaries, list)


class TestConvergenceFitnessPlateau(unittest.TestCase):
    def test_plateau_triggers_convergence_when_no_improvement(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=10,
            population_size=3,
            num_steps_per_candidate=1,
            convergence_criteria=ConvergenceCriteria(
                enabled=True,
                fitness_window=2,
                fitness_threshold=0.0,
                min_generations=0,
                early_stop=True,
            ),
            seed=1,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (5.0, {"member": member})
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.convergence_reason, "fitness_plateau")
        # With window=2 and min_generations=0, plateau fires when we have >=3 entries
        # with no improvement: generation 2 (0-indexed) at the earliest.
        self.assertIsNotNone(result.generation_of_convergence)
        self.assertLess(result.generation_of_convergence, 10)
        # Early stop: fewer than all 10 generations should have run.
        self.assertLess(len(result.generation_summaries), 10)

    def test_plateau_not_triggered_while_fitness_improves(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=5,
            population_size=3,
            num_steps_per_candidate=1,
            convergence_criteria=ConvergenceCriteria(
                enabled=True,
                fitness_window=2,
                fitness_threshold=0.5,
                min_generations=0,
                early_stop=True,
            ),
            seed=2,
        )
        experiment = EvolutionExperiment(base_config, config)
        # Fitness strictly increases by 2 each generation: well above threshold=0.5.
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (
                float(gen * 2 + 1),
                {"member": member},
            )
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.convergence_reason, "budget_exhausted")
        self.assertEqual(len(result.generation_summaries), 5)

    def test_plateau_respects_min_generations(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=10,
            population_size=3,
            num_steps_per_candidate=1,
            convergence_criteria=ConvergenceCriteria(
                enabled=True,
                fitness_window=1,
                fitness_threshold=0.0,
                min_generations=5,
                early_stop=True,
            ),
            seed=3,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (1.0, {"member": member})
        )
        self.assertTrue(result.converged)
        # Plateau cannot fire until five generations are recorded (indices 0–4).
        self.assertGreaterEqual(result.generation_of_convergence, 4)


class TestConvergenceDiversityCollapse(unittest.TestCase):
    def test_diversity_collapse_triggers_convergence(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=10,
            population_size=3,
            num_steps_per_candidate=1,
            convergence_criteria=ConvergenceCriteria(
                enabled=True,
                # Set an unreachably high fitness threshold so plateau never fires.
                fitness_window=100,
                fitness_threshold=1e9,
                diversity_window=2,
                diversity_threshold=1.0,  # always satisfied
                min_generations=0,
                early_stop=True,
            ),
            seed=4,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (
                float(gen),
                {"member": member},
            )
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.convergence_reason, "diversity_collapse")
        self.assertLess(len(result.generation_summaries), 10)

    def test_diversity_collapse_skipped_when_diversity_is_none(self):
        """Diversity collapse must not fire when _compute_diversity returns None."""
        from farm.runners.evolution_experiment import ConvergenceCriteria
        from unittest.mock import patch

        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=5,
            population_size=3,
            num_steps_per_candidate=1,
            convergence_criteria=ConvergenceCriteria(
                enabled=True,
                fitness_window=100,
                fitness_threshold=1e9,
                diversity_window=2,
                diversity_threshold=1.0,  # would always fire if diversity were not None
                min_generations=0,
                early_stop=True,
            ),
            seed=5,
        )
        experiment = EvolutionExperiment(base_config, config)
        with patch.object(EvolutionExperiment, "_compute_diversity", return_value=None):
            result = experiment.run(
                fitness_evaluator=lambda candidate, cfg, gen, member: (1.0, {"member": member})
            )
        # Diversity is always None so collapse never triggers; budget exhausted instead.
        self.assertFalse(result.converged)
        self.assertEqual(result.convergence_reason, "budget_exhausted")


class TestConvergenceEarlyStop(unittest.TestCase):
    def test_early_stop_true_halts_run(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=20,
            population_size=3,
            num_steps_per_candidate=1,
            convergence_criteria=ConvergenceCriteria(
                enabled=True,
                fitness_window=1,
                fitness_threshold=0.0,
                min_generations=0,
                early_stop=True,
            ),
            seed=6,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (1.0, {"member": member})
        )
        self.assertTrue(result.converged)
        # Run must have stopped before all 20 generations completed.
        self.assertLess(len(result.generation_summaries), 20)

    def test_early_stop_false_annotates_without_halting(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=6,
            population_size=3,
            num_steps_per_candidate=1,
            convergence_criteria=ConvergenceCriteria(
                enabled=True,
                fitness_window=1,
                fitness_threshold=0.0,
                min_generations=0,
                early_stop=False,  # annotate only, don't stop
            ),
            seed=7,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (1.0, {"member": member})
        )
        # Converged is True (criterion met) but all 6 generations ran.
        self.assertTrue(result.converged)
        self.assertEqual(result.convergence_reason, "fitness_plateau")
        self.assertEqual(len(result.generation_summaries), 6)

    def test_early_stop_records_first_convergence_generation(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=10,
            population_size=3,
            num_steps_per_candidate=1,
            convergence_criteria=ConvergenceCriteria(
                enabled=True,
                fitness_window=2,
                fitness_threshold=0.0,
                min_generations=0,
                early_stop=False,
            ),
            seed=8,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (1.0, {"member": member})
        )
        first_detection = result.generation_of_convergence
        self.assertIsNotNone(first_detection)
        # The detection generation must be within the completed window.
        self.assertLess(first_detection, len(result.generation_summaries))


class TestConvergenceBudgetExhausted(unittest.TestCase):
    def test_budget_exhausted_annotated_when_no_criterion_met(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria

        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=3,
            population_size=3,
            num_steps_per_candidate=1,
            convergence_criteria=ConvergenceCriteria(
                enabled=True,
                # Extremely high threshold so plateau never fires.
                fitness_window=100,
                fitness_threshold=1e9,
                # Extremely low threshold so diversity collapse never fires.
                diversity_window=100,
                diversity_threshold=0.0,
                min_generations=0,
            ),
            seed=9,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (
                float(gen * 100),
                {"member": member},
            )
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.convergence_reason, "budget_exhausted")
        self.assertEqual(result.generation_of_convergence, 2)  # last generation index
        self.assertEqual(len(result.generation_summaries), 3)


class TestConvergenceMetadataPersisted(unittest.TestCase):
    def test_convergence_metadata_written_to_file(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria
        import os as _os

        base_config = SimulationConfig()
        with tempfile.TemporaryDirectory() as output_dir:
            config = EvolutionExperimentConfig(
                num_generations=10,
                population_size=3,
                num_steps_per_candidate=1,
                convergence_criteria=ConvergenceCriteria(
                    enabled=True,
                    fitness_window=1,
                    fitness_threshold=0.0,
                    min_generations=0,
                    early_stop=True,
                ),
                output_dir=output_dir,
                seed=10,
            )
            experiment = EvolutionExperiment(base_config, config)
            result = experiment.run(
                fitness_evaluator=lambda candidate, cfg, gen, member: (1.0, {"member": member})
            )
            metadata_path = _os.path.join(output_dir, "evolution_metadata.json")
            self.assertTrue(_os.path.exists(metadata_path))
            with open(metadata_path, encoding="utf-8") as mf:
                metadata = json.load(mf)
            self.assertIn("converged", metadata)
            self.assertIn("convergence_reason", metadata)
            self.assertIn("generation_of_convergence", metadata)
            self.assertIn("num_generations_completed", metadata)
            self.assertTrue(metadata["converged"])
            self.assertEqual(metadata["convergence_reason"], result.convergence_reason)
            self.assertEqual(metadata["generation_of_convergence"], result.generation_of_convergence)
            self.assertEqual(metadata["num_generations_completed"], len(result.generation_summaries))

    def test_convergence_reason_enum_values_are_strings_in_json(self):
        from farm.runners.evolution_experiment import ConvergenceCriteria
        import os as _os

        base_config = SimulationConfig()
        with tempfile.TemporaryDirectory() as output_dir:
            config = EvolutionExperimentConfig(
                num_generations=5,
                population_size=3,
                num_steps_per_candidate=1,
                convergence_criteria=ConvergenceCriteria(
                    enabled=True,
                    fitness_window=100,
                    fitness_threshold=1e9,
                    diversity_window=100,
                    diversity_threshold=0.0,
                    min_generations=0,
                ),
                output_dir=output_dir,
                seed=11,
            )
            experiment = EvolutionExperiment(base_config, config)
            experiment.run(
                fitness_evaluator=lambda candidate, cfg, gen, member: (
                    float(gen),
                    {"member": member},
                )
            )
            metadata_path = _os.path.join(output_dir, "evolution_metadata.json")
            with open(metadata_path, encoding="utf-8") as mf:
                metadata = json.load(mf)
            # convergence_reason must be a plain string, not a dict or enum repr.
            self.assertIsInstance(metadata["convergence_reason"], str)
            self.assertEqual(metadata["convergence_reason"], "budget_exhausted")


class TestAdaptiveMutationHardenedBehavior(unittest.TestCase):
    """Tests for the hardened adaptive mutation update rules (issue: Evolution v2)."""

    # ------------------------------------------------------------------ #
    # Damping / bounded change per generation                             #
    # ------------------------------------------------------------------ #

    @patch("farm.runners.evolution_experiment.mutate_chromosome")
    def test_max_step_multiplier_limits_single_generation_change(self, mutate_mock):
        """max_step_multiplier prevents a large stall_multiplier from fully applying."""
        mutate_mock.side_effect = lambda chromosome, **kwargs: chromosome
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=3,
            population_size=3,
            num_steps_per_candidate=1,
            mutation_rate=0.3,
            mutation_scale=0.1,
            adaptive_mutation=AdaptiveMutationConfig(
                enabled=True,
                use_fitness_adaptation=True,
                use_diversity_adaptation=False,
                stall_window=1,
                stall_multiplier=5.0,
                max_step_multiplier=1.4,
                max_rate_multiplier=100.0,
            ),
            seed=77,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (1.0, {"member": member})
        )
        gen0, gen1, gen2 = result.generation_summaries
        # Gen 1 was produced before any observe() had effect: multiplier still 1.0.
        self.assertAlmostEqual(gen1.mutation_rate_multiplier, 1.0)
        # Gen 2 was produced after one stall observation.
        # Without damping: multiplier would be 5.0.
        # With max_step=1.4: multiplier should be exactly 1.4.
        self.assertIsNotNone(gen2.mutation_rate_multiplier)
        self.assertAlmostEqual(gen2.mutation_rate_multiplier, 1.4, places=5)
        self.assertIn("stalled", gen2.adaptive_event)

    @patch("farm.runners.evolution_experiment.mutate_chromosome")
    def test_stable_adaptation_over_8_generations(self, mutate_mock):
        """Integration test: with damping, multiplier stays within bounds over >= 8 gens."""
        mutate_mock.side_effect = lambda chromosome, **kwargs: chromosome
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=9,
            population_size=3,
            num_steps_per_candidate=1,
            mutation_rate=0.25,
            mutation_scale=0.15,
            adaptive_mutation=AdaptiveMutationConfig(
                enabled=True,
                use_fitness_adaptation=True,
                use_diversity_adaptation=False,
                stall_window=2,
                stall_multiplier=2.0,
                improve_multiplier=0.7,
                max_step_multiplier=1.5,
                min_rate_multiplier=0.05,
                max_rate_multiplier=10.0,
            ),
            seed=42,
        )
        experiment = EvolutionExperiment(base_config, config)
        # Alternating fitness: improves on even generations, stalls on odd.
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                float(generation if generation % 2 == 0 else generation - 1),
                {"member": member},
            )
        )
        self.assertEqual(len(result.generation_summaries), 9)

        # All non-seed generations must have bounded multipliers.
        for summary in result.generation_summaries[1:]:
            self.assertIsNotNone(summary.mutation_rate_multiplier)
            self.assertGreaterEqual(summary.mutation_rate_multiplier, 0.05)
            self.assertLessEqual(summary.mutation_rate_multiplier, 10.0)

        # The effective mutation rate must always stay in [0, 1].
        for summary in result.generation_summaries[1:]:
            self.assertIsNotNone(summary.mutation_rate_used)
            self.assertGreaterEqual(summary.mutation_rate_used, 0.0)
            self.assertLessEqual(summary.mutation_rate_used, 1.0)

    # ------------------------------------------------------------------ #
    # Per-gene defaults without manual CLI flags                          #
    # ------------------------------------------------------------------ #

    @patch("farm.runners.evolution_experiment.mutate_chromosome")
    def test_use_default_per_gene_multipliers_forwarded_to_mutate(self, mutate_mock):
        """use_default_per_gene_multipliers=True passes built-in defaults to mutate_chromosome."""
        mutate_mock.side_effect = lambda chromosome, **kwargs: chromosome
        from farm.runners.adaptive_mutation import DEFAULT_PER_GENE_SCALE_MULTIPLIERS

        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=3,
            num_steps_per_candidate=1,
            adaptive_mutation=AdaptiveMutationConfig(
                enabled=True,
                use_fitness_adaptation=False,
                use_diversity_adaptation=False,
                use_default_per_gene_multipliers=True,
            ),
            seed=11,
        )
        experiment = EvolutionExperiment(base_config, config)
        experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                float(member),
                {"member": member},
            )
        )
        self.assertTrue(mutate_mock.called)
        child_calls = [
            call for call in mutate_mock.call_args_list
            if call.kwargs.get("per_gene_scale_multipliers") is not None
        ]
        self.assertGreater(len(child_calls), 0)
        for call in child_calls:
            scale_mults = call.kwargs.get("per_gene_scale_multipliers", {})
            self.assertAlmostEqual(
                scale_mults.get("learning_rate"),
                DEFAULT_PER_GENE_SCALE_MULTIPLIERS["learning_rate"],
            )

    @patch("farm.runners.evolution_experiment.mutate_chromosome")
    def test_user_per_gene_overrides_default_per_gene(self, mutate_mock):
        """User per-gene values override built-in defaults when both are active."""
        mutate_mock.side_effect = lambda chromosome, **kwargs: chromosome
        base_config = SimulationConfig()
        custom_lr_scale = 0.1  # different from the built-in 0.5
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=3,
            num_steps_per_candidate=1,
            adaptive_mutation=AdaptiveMutationConfig(
                enabled=True,
                use_fitness_adaptation=False,
                use_diversity_adaptation=False,
                use_default_per_gene_multipliers=True,
                per_gene_scale_multipliers={"learning_rate": custom_lr_scale},
            ),
            seed=22,
        )
        experiment = EvolutionExperiment(base_config, config)
        experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                float(member),
                {"member": member},
            )
        )
        child_calls = [
            call for call in mutate_mock.call_args_list
            if call.kwargs.get("per_gene_scale_multipliers") is not None
        ]
        self.assertGreater(len(child_calls), 0)
        for call in child_calls:
            scale_mults = call.kwargs.get("per_gene_scale_multipliers", {})
            # User value wins over built-in default.
            self.assertAlmostEqual(scale_mults.get("learning_rate"), custom_lr_scale)

    # ------------------------------------------------------------------ #
    # best_fitness_delta telemetry in EvolutionGenerationSummary          #
    # ------------------------------------------------------------------ #

    def test_best_fitness_delta_is_none_for_initial_seeding(self):
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=1,
            population_size=3,
            num_steps_per_candidate=1,
            seed=1,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (1.0, {"member": member})
        )
        self.assertIsNone(result.generation_summaries[0].best_fitness_delta)

    def test_best_fitness_delta_persisted_to_json(self):
        base_config = SimulationConfig()
        with tempfile.TemporaryDirectory() as output_dir:
            config = EvolutionExperimentConfig(
                num_generations=3,
                population_size=3,
                num_steps_per_candidate=1,
                mutation_rate=0.2,
                adaptive_mutation=AdaptiveMutationConfig(
                    enabled=True,
                    use_fitness_adaptation=True,
                    use_diversity_adaptation=False,
                    stall_window=1,
                    stall_multiplier=2.0,
                ),
                output_dir=output_dir,
                seed=5,
            )
            experiment = EvolutionExperiment(base_config, config)
            experiment.run(
                fitness_evaluator=lambda candidate, cfg, generation, member: (
                    float(generation),
                    {"member": member},
                )
            )
            import os as _os
            with open(
                _os.path.join(output_dir, "evolution_generation_summaries.json"),
                encoding="utf-8",
            ) as fp:
                summaries = json.load(fp)
            # Field must be present in all summary records.
            for summary in summaries:
                self.assertIn("best_fitness_delta", summary)
            # Gen 0 is seeded: no delta.
            self.assertIsNone(summaries[0]["best_fitness_delta"])
            # Gen 1 was produced before any controller observation, so delta is None.
            self.assertIsNone(summaries[1]["best_fitness_delta"])
            # Gen 2 reflects the improvement observed from gen 1.
            self.assertIsNotNone(summaries[2]["best_fitness_delta"])

    def test_best_fitness_delta_correct_magnitude_on_improvement(self):
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=3,
            population_size=3,
            num_steps_per_candidate=1,
            adaptive_mutation=AdaptiveMutationConfig(
                enabled=True,
                use_fitness_adaptation=True,
                use_diversity_adaptation=False,
                stall_window=1,
            ),
            seed=9,
        )
        experiment = EvolutionExperiment(base_config, config)
        fitness_by_gen = {0: 1.0, 1: 5.0, 2: 8.0}
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                fitness_by_gen[generation],
                {"member": member},
            )
        )
        gen0, gen1, gen2 = result.generation_summaries
        # Gen 0 and gen 1 have no delta (initial seeding / first observation).
        self.assertIsNone(gen0.best_fitness_delta)
        self.assertIsNone(gen1.best_fitness_delta)
        # Gen 2 reflects the gen-1 observation: 5.0 - 1.0 = 4.0.
        self.assertIsNotNone(gen2.best_fitness_delta)
        self.assertAlmostEqual(gen2.best_fitness_delta, 4.0)


class TestBoundaryOccupancyMetrics(unittest.TestCase):
    """Tests for per-generation boundary occupancy reporting."""

    def test_boundary_occupancy_present_in_generation_summaries(self):
        """boundary_occupancy is populated for every generation."""
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=4,
            num_steps_per_candidate=1,
            seed=42,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                float(member + generation),
                {"member": member},
            )
        )
        for summary in result.generation_summaries:
            self.assertIsInstance(summary.boundary_occupancy, dict)
            # All evolvable genes should appear.
            self.assertIn("learning_rate", summary.boundary_occupancy)
            self.assertIn("gamma", summary.boundary_occupancy)
            self.assertIn("epsilon_decay", summary.boundary_occupancy)
            # Values must be fractions in [0, 1].
            for gene_name, frac in summary.boundary_occupancy.items():
                self.assertGreaterEqual(frac, 0.0, msg=f"{gene_name} fraction < 0")
                self.assertLessEqual(frac, 1.0, msg=f"{gene_name} fraction > 1")

    def test_gene_statistics_include_boundary_counts(self):
        """gene_statistics dict contains at_min_count, at_max_count, boundary_fraction."""
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=1,
            population_size=4,
            num_steps_per_candidate=1,
            seed=7,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                float(member),
                {"member": member},
            )
        )
        gen_stats = result.generation_summaries[0].gene_statistics
        for gene_name, stats in gen_stats.items():
            self.assertIn("at_min_count", stats, msg=f"at_min_count missing for {gene_name}")
            self.assertIn("at_max_count", stats, msg=f"at_max_count missing for {gene_name}")
            self.assertIn("boundary_fraction", stats, msg=f"boundary_fraction missing for {gene_name}")
            self.assertGreaterEqual(stats["at_min_count"], 0.0)
            self.assertGreaterEqual(stats["at_max_count"], 0.0)
            self.assertGreaterEqual(stats["boundary_fraction"], 0.0)
            self.assertLessEqual(stats["boundary_fraction"], 1.0)

    def test_boundary_occupancy_persisted_to_summaries_json(self):
        """boundary_occupancy appears in the serialized generation summaries JSON."""
        base_config = SimulationConfig()
        with tempfile.TemporaryDirectory() as output_dir:
            config = EvolutionExperimentConfig(
                num_generations=2,
                population_size=3,
                num_steps_per_candidate=1,
                output_dir=output_dir,
                seed=13,
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

            for summary in summaries:
                self.assertIn("boundary_occupancy", summary)
                self.assertIsInstance(summary["boundary_occupancy"], dict)
                self.assertIn("learning_rate", summary["boundary_occupancy"])

    def test_all_at_boundary_gives_occupancy_one(self):
        """When every candidate is pinned to min_value, boundary_fraction == 1.0."""
        base_config = SimulationConfig()
        # Force learning_rate to minimum across all candidates (mutation_scale=0
        # with initial learning_rate at min).
        base_config.learning.learning_rate = 1e-6
        config = EvolutionExperimentConfig(
            num_generations=1,
            population_size=3,
            num_steps_per_candidate=1,
            mutation_scale=0.0,
            seed=17,
        )
        experiment = EvolutionExperiment(base_config, config)
        result = experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (1.0, {"member": member})
        )
        gen_stats = result.generation_summaries[0].gene_statistics
        # learning_rate is at its minimum for all three candidates.
        lr_stats = gen_stats["learning_rate"]
        self.assertAlmostEqual(lr_stats["at_min_count"], 3.0)
        self.assertAlmostEqual(lr_stats["boundary_fraction"], 1.0)
        # boundary_occupancy should reflect this.
        self.assertAlmostEqual(
            result.generation_summaries[0].boundary_occupancy["learning_rate"], 1.0
        )


class TestInteriorBiasedModeIntegration(unittest.TestCase):
    """Integration tests for INTERIOR_BIASED boundary mode in the evolution runner."""

    def test_config_accepts_interior_biased_boundary_mode(self):
        config = EvolutionExperimentConfig(
            boundary_mode=BoundaryMode.INTERIOR_BIASED,
            interior_bias_fraction=1e-3,
        )
        self.assertEqual(config.boundary_mode, BoundaryMode.INTERIOR_BIASED)
        self.assertAlmostEqual(config.interior_bias_fraction, 1e-3)

    def test_config_rejects_negative_interior_bias_fraction(self):
        with self.assertRaises(ValueError):
            EvolutionExperimentConfig(
                boundary_mode=BoundaryMode.INTERIOR_BIASED,
                interior_bias_fraction=-0.01,
            )

    @patch("farm.runners.evolution_experiment.mutate_chromosome")
    def test_interior_bias_fraction_forwarded_to_mutation_calls(self, mutate_mock):
        mutate_mock.side_effect = lambda chromosome, **kwargs: chromosome
        base_config = SimulationConfig()
        config = EvolutionExperimentConfig(
            num_generations=2,
            population_size=4,
            num_steps_per_candidate=1,
            boundary_mode=BoundaryMode.INTERIOR_BIASED,
            interior_bias_fraction=5e-3,
            seed=55,
        )
        experiment = EvolutionExperiment(base_config, config)
        experiment.run(
            fitness_evaluator=lambda candidate, cfg, generation, member: (
                float(member + generation),
                {"member": member},
            )
        )
        self.assertTrue(mutate_mock.called)
        for call in mutate_mock.call_args_list:
            self.assertEqual(call.kwargs.get("boundary_mode"), BoundaryMode.INTERIOR_BIASED)
            self.assertAlmostEqual(call.kwargs.get("interior_bias_fraction"), 5e-3)

    def test_interior_biased_reduces_boundary_occupancy_vs_clamp(self):
        """Population-level: INTERIOR_BIASED should produce lower boundary occupancy
        than CLAMP when starting at the minimum boundary with heavy mutation."""
        base_config_clamp = SimulationConfig()
        base_config_clamp.learning.learning_rate = 1e-6

        base_config_ib = SimulationConfig()
        base_config_ib.learning.learning_rate = 1e-6

        def make_config(mode):
            return EvolutionExperimentConfig(
                num_generations=3,
                population_size=6,
                num_steps_per_candidate=1,
                mutation_rate=1.0,
                mutation_scale=0.5,
                boundary_mode=mode,
                interior_bias_fraction=0.1,
                seed=100,
            )

        exp_clamp = EvolutionExperiment(base_config_clamp, make_config(BoundaryMode.CLAMP))
        result_clamp = exp_clamp.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (1.0, {"member": member})
        )

        exp_ib = EvolutionExperiment(base_config_ib, make_config(BoundaryMode.INTERIOR_BIASED))
        result_ib = exp_ib.run(
            fitness_evaluator=lambda candidate, cfg, gen, member: (1.0, {"member": member})
        )

        # Gene values must remain within their declared per-gene bounds.
        # The chromosome schema validation already enforces this, but we verify
        # here that no out-of-range value slips through.
        from farm.core.hyperparameter_chromosome import default_hyperparameter_chromosome
        gene_bounds = {gene.name: (gene.min_value, gene.max_value) for gene in default_hyperparameter_chromosome().genes}
        for summary in result_ib.generation_summaries:
            for gene_name, stats in summary.gene_statistics.items():
                if gene_name in gene_bounds:
                    lo, hi = gene_bounds[gene_name]
                    self.assertGreaterEqual(stats["min"], lo, msg=f"{gene_name} min below bound")
                    self.assertLessEqual(stats["max"], hi, msg=f"{gene_name} max above bound")

        # At least one generation should have lower lr boundary occupancy with
        # INTERIOR_BIASED than with CLAMP (since INTERIOR_BIASED nudges away from walls).
        clamp_occ = [s.boundary_occupancy.get("learning_rate", 0.0) for s in result_clamp.generation_summaries]
        ib_occ = [s.boundary_occupancy.get("learning_rate", 0.0) for s in result_ib.generation_summaries]
        # Mean boundary fraction should be ≤ clamp's across the run.
        import statistics as _stats
        mean_clamp = _stats.mean(clamp_occ)
        mean_ib = _stats.mean(ib_occ)
        self.assertLessEqual(
            mean_ib,
            mean_clamp,
            msg=f"INTERIOR_BIASED mean occupancy {mean_ib:.3f} should be ≤ CLAMP {mean_clamp:.3f}",
        )


if __name__ == "__main__":
    unittest.main()

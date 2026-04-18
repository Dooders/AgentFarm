"""Tests for adaptive mutation rate/scale controller and config."""

import unittest

from farm.runners.adaptive_mutation import (
    AdaptiveMutationConfig,
    AdaptiveMutationController,
    compute_normalized_diversity,
)


class TestAdaptiveMutationConfig(unittest.TestCase):
    def test_defaults_are_disabled(self):
        config = AdaptiveMutationConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.per_gene_rate_multipliers, {})
        self.assertEqual(config.per_gene_scale_multipliers, {})

    def test_rejects_invalid_stall_window(self):
        with self.assertRaises(ValueError):
            AdaptiveMutationConfig(stall_window=0)

    def test_rejects_negative_improvement_threshold(self):
        with self.assertRaises(ValueError):
            AdaptiveMutationConfig(improvement_threshold=-0.1)

    def test_rejects_inverted_rate_bounds(self):
        with self.assertRaises(ValueError):
            AdaptiveMutationConfig(min_rate_multiplier=2.0, max_rate_multiplier=1.0)

    def test_rejects_diversity_threshold_out_of_range(self):
        with self.assertRaises(ValueError):
            AdaptiveMutationConfig(diversity_threshold=1.5)

    def test_rejects_negative_per_gene_multiplier(self):
        with self.assertRaises(ValueError):
            AdaptiveMutationConfig(per_gene_rate_multipliers={"learning_rate": -0.1})


class TestComputeNormalizedDiversity(unittest.TestCase):
    def test_returns_zero_when_no_genes(self):
        self.assertEqual(compute_normalized_diversity({}, [], {}), 0.0)

    def test_normalizes_std_by_span(self):
        stats = {
            "learning_rate": {"std": 0.01},
            "gamma": {"std": 0.1},
        }
        bounds = {
            "learning_rate": (0.0, 0.1),
            "gamma": (0.5, 1.0),
        }
        diversity = compute_normalized_diversity(stats, ["learning_rate", "gamma"], bounds)
        # (0.01 / 0.1 + 0.1 / 0.5) / 2 == (0.1 + 0.2) / 2 == 0.15
        self.assertAlmostEqual(diversity, 0.15)

    def test_skips_zero_span_genes(self):
        stats = {
            "learning_rate": {"std": 0.01},
            "fixed_gene": {"std": 0.0},
        }
        bounds = {
            "learning_rate": (0.0, 0.1),
            "fixed_gene": (0.5, 0.5),
        }
        diversity = compute_normalized_diversity(stats, ["learning_rate", "fixed_gene"], bounds)
        self.assertAlmostEqual(diversity, 0.1)


class TestAdaptiveMutationController(unittest.TestCase):
    def test_disabled_controller_keeps_multipliers_at_unity(self):
        controller = AdaptiveMutationController(AdaptiveMutationConfig(enabled=False))
        controller.observe(best_fitness=1.0, diversity=0.2)
        controller.observe(best_fitness=1.0, diversity=0.2)
        self.assertEqual(controller.rate_multiplier, 1.0)
        self.assertEqual(controller.scale_multiplier, 1.0)
        self.assertEqual(controller.effective_rate(0.25), 0.25)
        self.assertEqual(controller.effective_scale(0.2), 0.2)
        self.assertEqual(controller.last_event, "disabled")

    def test_stalled_fitness_increases_multipliers(self):
        config = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=True,
            use_diversity_adaptation=False,
            stall_window=2,
            stall_multiplier=2.0,
            improvement_threshold=1e-6,
        )
        controller = AdaptiveMutationController(config)
        # No improvement across three generations => two stall events.
        controller.observe(best_fitness=1.0, diversity=None)
        controller.observe(best_fitness=1.0, diversity=None)
        controller.observe(best_fitness=1.0, diversity=None)
        self.assertGreater(controller.rate_multiplier, 1.0)
        self.assertAlmostEqual(controller.rate_multiplier, 4.0)
        self.assertIn("stalled", controller.last_event)

    def test_improving_fitness_tightens_multipliers(self):
        config = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=True,
            use_diversity_adaptation=False,
            stall_window=1,
            improve_multiplier=0.5,
            min_rate_multiplier=0.01,
            min_scale_multiplier=0.01,
        )
        controller = AdaptiveMutationController(config)
        controller.observe(best_fitness=1.0, diversity=None)
        controller.observe(best_fitness=2.0, diversity=None)
        self.assertAlmostEqual(controller.rate_multiplier, 0.5)
        self.assertAlmostEqual(controller.scale_multiplier, 0.5)
        self.assertEqual(controller.last_event, "improving")

    def test_diversity_collapse_boosts_multipliers(self):
        config = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=False,
            use_diversity_adaptation=True,
            diversity_threshold=0.1,
            diversity_multiplier=3.0,
        )
        controller = AdaptiveMutationController(config)
        controller.observe(best_fitness=1.0, diversity=0.05)
        self.assertAlmostEqual(controller.rate_multiplier, 3.0)
        self.assertAlmostEqual(controller.scale_multiplier, 3.0)
        self.assertEqual(controller.last_event, "diversity_collapse")

    def test_high_diversity_does_not_boost(self):
        config = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=False,
            use_diversity_adaptation=True,
            diversity_threshold=0.1,
            diversity_multiplier=3.0,
        )
        controller = AdaptiveMutationController(config)
        controller.observe(best_fitness=1.0, diversity=0.2)
        self.assertEqual(controller.rate_multiplier, 1.0)
        self.assertEqual(controller.last_event, "baseline")

    def test_multipliers_are_clamped(self):
        config = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=True,
            use_diversity_adaptation=False,
            stall_window=1,
            stall_multiplier=10.0,
            max_rate_multiplier=4.0,
            max_scale_multiplier=4.0,
        )
        controller = AdaptiveMutationController(config)
        for _ in range(5):
            controller.observe(best_fitness=0.0, diversity=None)
        self.assertLessEqual(controller.rate_multiplier, 4.0)
        self.assertLessEqual(controller.scale_multiplier, 4.0)

    def test_effective_rate_is_clamped_to_unit_interval(self):
        config = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=True,
            use_diversity_adaptation=False,
            stall_window=1,
            stall_multiplier=5.0,
            max_rate_multiplier=5.0,
            max_scale_multiplier=5.0,
        )
        controller = AdaptiveMutationController(config)
        controller.observe(best_fitness=0.0, diversity=None)
        controller.observe(best_fitness=0.0, diversity=None)
        self.assertLessEqual(controller.effective_rate(0.5), 1.0)
        self.assertGreaterEqual(controller.effective_scale(0.1), 0.0)

    def test_per_gene_multipliers_only_active_when_enabled(self):
        config_disabled = AdaptiveMutationConfig(
            enabled=False,
            per_gene_rate_multipliers={"learning_rate": 2.0},
        )
        controller = AdaptiveMutationController(config_disabled)
        self.assertEqual(controller.per_gene_rate_multipliers(), {})

        config_enabled = AdaptiveMutationConfig(
            enabled=True,
            per_gene_rate_multipliers={"learning_rate": 2.0},
            per_gene_scale_multipliers={"learning_rate": 0.5},
        )
        controller_enabled = AdaptiveMutationController(config_enabled)
        self.assertEqual(
            controller_enabled.per_gene_rate_multipliers()["learning_rate"], 2.0
        )
        self.assertEqual(
            controller_enabled.per_gene_scale_multipliers()["learning_rate"], 0.5
        )


if __name__ == "__main__":
    unittest.main()

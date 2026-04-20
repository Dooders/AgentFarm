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

    def test_combined_stall_and_diversity_collapse_event_tag(self):
        config = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=True,
            use_diversity_adaptation=True,
            stall_window=1,
            stall_multiplier=1.5,
            improvement_threshold=1e-6,
            diversity_threshold=0.1,
            diversity_multiplier=1.5,
        )
        controller = AdaptiveMutationController(config)
        controller.observe(best_fitness=1.0, diversity=0.05)
        # First observation only records baseline (fitness adaptation needs >=2 points).
        # Diversity collapse can still fire on the first observation.
        self.assertIn("diversity_collapse", controller.last_event)

        controller.observe(best_fitness=1.0, diversity=0.05)
        # Now both stalled and diversity_collapse should fire together.
        self.assertIn("stalled", controller.last_event)
        self.assertIn("diversity_collapse", controller.last_event)
        # And multipliers compounded: 1.5 (initial diversity) * 1.5 (stall) * 1.5 (diversity).
        self.assertAlmostEqual(controller.rate_multiplier, 1.5 * 1.5 * 1.5)

    def test_clamp_emits_saturation_event_tag(self):
        config = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=True,
            use_diversity_adaptation=False,
            stall_window=1,
            stall_multiplier=10.0,
            max_rate_multiplier=2.0,
            max_scale_multiplier=2.0,
            # Set max_step_multiplier large so the per-step bound does not
            # absorb the full stall_multiplier before the global clamp fires.
            max_step_multiplier=100.0,
        )
        controller = AdaptiveMutationController(config)
        controller.observe(best_fitness=0.0, diversity=None)
        controller.observe(best_fitness=0.0, diversity=None)
        self.assertEqual(controller.rate_multiplier, 2.0)
        self.assertEqual(controller.scale_multiplier, 2.0)
        self.assertIn("rate_clamped", controller.last_event)
        self.assertIn("scale_clamped", controller.last_event)

    def test_per_gene_mapping_is_immutable_after_construction(self):
        original = {"learning_rate": 2.0}
        config = AdaptiveMutationConfig(
            enabled=True,
            per_gene_rate_multipliers=original,
        )
        # Frozen via MappingProxyType: write attempts raise TypeError.
        with self.assertRaises(TypeError):
            config.per_gene_rate_multipliers["learning_rate"] = 9.9  # type: ignore[index]
        # Mutating the original dict does not bleed through into the config.
        original["learning_rate"] = 9.9
        self.assertEqual(config.per_gene_rate_multipliers["learning_rate"], 2.0)

    def test_rejects_nan_per_gene_multiplier(self):
        with self.assertRaises(ValueError):
            AdaptiveMutationConfig(per_gene_scale_multipliers={"learning_rate": float("nan")})

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

    # ------------------------------------------------------------------ #
    # New tests: max_step_multiplier damping                              #
    # ------------------------------------------------------------------ #

    def test_rejects_max_step_multiplier_below_one(self):
        with self.assertRaises(ValueError):
            AdaptiveMutationConfig(max_step_multiplier=0.5)

    def test_max_step_multiplier_bounds_stall_increase(self):
        """A large stall_multiplier is capped per-step by max_step_multiplier."""
        config = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=True,
            use_diversity_adaptation=False,
            stall_window=1,
            stall_multiplier=10.0,
            max_step_multiplier=1.5,
            max_rate_multiplier=100.0,
        )
        controller = AdaptiveMutationController(config)
        controller.observe(best_fitness=0.0, diversity=None)
        controller.observe(best_fitness=0.0, diversity=None)
        # Without damping: 1.0 * 10.0 = 10.0.  With max_step=1.5: 1.0 * 1.5 = 1.5.
        self.assertAlmostEqual(controller.rate_multiplier, 1.5)
        self.assertIn("stalled", controller.last_event)

    def test_max_step_multiplier_bounds_improve_reduction(self):
        """A very aggressive improve_multiplier is capped per-step by max_step_multiplier."""
        config = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=True,
            use_diversity_adaptation=False,
            stall_window=1,
            improve_multiplier=0.1,
            max_step_multiplier=1.5,
            min_rate_multiplier=0.001,
        )
        controller = AdaptiveMutationController(config)
        controller.observe(best_fitness=1.0, diversity=None)
        controller.observe(best_fitness=2.0, diversity=None)
        # Without damping: 1.0 * 0.1 = 0.1.  With max_step=1.5: 1.0 / 1.5 ≈ 0.667.
        self.assertAlmostEqual(controller.rate_multiplier, 1.0 / 1.5)
        self.assertIn("improving", controller.last_event)

    def test_default_max_step_multiplier_does_not_affect_typical_configs(self):
        """With typical stall=1.5 / improve=0.8 the default max_step=2.0 has no effect."""
        config = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=True,
            use_diversity_adaptation=False,
            stall_window=1,
            stall_multiplier=1.5,
            improve_multiplier=0.8,
            # default max_step_multiplier=2.0
        )
        controller = AdaptiveMutationController(config)
        controller.observe(best_fitness=1.0, diversity=None)
        controller.observe(best_fitness=1.0, diversity=None)  # stall
        self.assertAlmostEqual(controller.rate_multiplier, 1.5)
        controller.observe(best_fitness=2.0, diversity=None)  # improve
        self.assertAlmostEqual(controller.rate_multiplier, 1.5 * 0.8)

    def test_damping_reduces_oscillation_over_alternating_stall_improve(self):
        """Alternating stall/improve with damping keeps multiplier within tighter bounds."""
        tight_step = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=True,
            use_diversity_adaptation=False,
            stall_window=1,
            stall_multiplier=3.0,
            improve_multiplier=0.3,
            max_step_multiplier=1.3,
            min_rate_multiplier=0.01,
            max_rate_multiplier=100.0,
        )
        controller_tight = AdaptiveMutationController(tight_step)

        no_step = AdaptiveMutationConfig(
            enabled=True,
            use_fitness_adaptation=True,
            use_diversity_adaptation=False,
            stall_window=1,
            stall_multiplier=3.0,
            improve_multiplier=0.3,
            max_step_multiplier=1000.0,
            min_rate_multiplier=0.01,
            max_rate_multiplier=1000.0,
        )
        controller_nodamp = AdaptiveMutationController(no_step)

        # Alternate: first gen sets history, then stall/improve alternate.
        fitnesses = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
        for fit in fitnesses:
            controller_tight.observe(best_fitness=fit, diversity=None)
            controller_nodamp.observe(best_fitness=fit, diversity=None)

        # The damped controller should have a smaller multiplier range.
        self.assertLess(controller_tight.rate_multiplier, controller_nodamp.rate_multiplier)

    # ------------------------------------------------------------------ #
    # New tests: last_fitness_delta telemetry                             #
    # ------------------------------------------------------------------ #

    def test_last_fitness_delta_is_none_before_any_observation(self):
        controller = AdaptiveMutationController(AdaptiveMutationConfig(enabled=True))
        self.assertIsNone(controller.last_fitness_delta)

    def test_last_fitness_delta_is_none_after_first_observation(self):
        controller = AdaptiveMutationController(
            AdaptiveMutationConfig(enabled=True, use_fitness_adaptation=True)
        )
        controller.observe(best_fitness=1.0, diversity=None)
        # Only one history point: no comparison possible.
        self.assertIsNone(controller.last_fitness_delta)

    def test_last_fitness_delta_positive_on_improvement(self):
        controller = AdaptiveMutationController(
            AdaptiveMutationConfig(enabled=True, stall_window=1)
        )
        controller.observe(best_fitness=1.0, diversity=None)
        controller.observe(best_fitness=3.0, diversity=None)
        self.assertIsNotNone(controller.last_fitness_delta)
        self.assertAlmostEqual(controller.last_fitness_delta, 2.0)

    def test_last_fitness_delta_zero_on_stall(self):
        controller = AdaptiveMutationController(
            AdaptiveMutationConfig(enabled=True, stall_window=1)
        )
        controller.observe(best_fitness=5.0, diversity=None)
        controller.observe(best_fitness=5.0, diversity=None)
        self.assertIsNotNone(controller.last_fitness_delta)
        self.assertAlmostEqual(controller.last_fitness_delta, 0.0)

    def test_last_fitness_delta_is_none_when_fitness_adaptation_disabled(self):
        controller = AdaptiveMutationController(
            AdaptiveMutationConfig(
                enabled=True,
                use_fitness_adaptation=False,
                use_diversity_adaptation=False,
            )
        )
        controller.observe(best_fitness=1.0, diversity=None)
        controller.observe(best_fitness=2.0, diversity=None)
        # No fitness adaptation → no delta computed.
        self.assertIsNone(controller.last_fitness_delta)

    # ------------------------------------------------------------------ #
    # New tests: use_default_per_gene_multipliers                         #
    # ------------------------------------------------------------------ #

    def test_use_default_per_gene_multipliers_false_returns_empty_when_none_configured(self):
        config = AdaptiveMutationConfig(enabled=True, use_default_per_gene_multipliers=False)
        controller = AdaptiveMutationController(config)
        self.assertEqual(controller.per_gene_scale_multipliers(), {})
        self.assertEqual(controller.per_gene_rate_multipliers(), {})

    def test_use_default_per_gene_multipliers_true_applies_learning_rate_default(self):
        from farm.runners.adaptive_mutation import DEFAULT_PER_GENE_SCALE_MULTIPLIERS

        config = AdaptiveMutationConfig(enabled=True, use_default_per_gene_multipliers=True)
        controller = AdaptiveMutationController(config)
        scale_mults = controller.per_gene_scale_multipliers()
        self.assertIn("learning_rate", scale_mults)
        self.assertAlmostEqual(
            scale_mults["learning_rate"],
            DEFAULT_PER_GENE_SCALE_MULTIPLIERS["learning_rate"],
        )

    def test_use_default_per_gene_multipliers_true_applies_gamma_and_epsilon_decay(self):
        from farm.runners.adaptive_mutation import DEFAULT_PER_GENE_SCALE_MULTIPLIERS

        config = AdaptiveMutationConfig(enabled=True, use_default_per_gene_multipliers=True)
        controller = AdaptiveMutationController(config)
        scale_mults = controller.per_gene_scale_multipliers()
        for gene in ("gamma", "epsilon_decay"):
            self.assertIn(gene, scale_mults)
            self.assertAlmostEqual(
                scale_mults[gene],
                DEFAULT_PER_GENE_SCALE_MULTIPLIERS[gene],
            )

    def test_user_per_gene_multipliers_override_defaults(self):
        """User-supplied per-gene values take precedence over built-in defaults."""
        config = AdaptiveMutationConfig(
            enabled=True,
            use_default_per_gene_multipliers=True,
            per_gene_scale_multipliers={"learning_rate": 0.1},
        )
        controller = AdaptiveMutationController(config)
        # User override wins.
        self.assertAlmostEqual(controller.per_gene_scale_multipliers()["learning_rate"], 0.1)
        # Other genes still come from defaults.
        self.assertIn("gamma", controller.per_gene_scale_multipliers())

    def test_defaults_not_returned_when_controller_disabled(self):
        """When enabled=False, no per-gene multipliers (including defaults) are returned."""
        config = AdaptiveMutationConfig(
            enabled=False, use_default_per_gene_multipliers=True
        )
        controller = AdaptiveMutationController(config)
        self.assertEqual(controller.per_gene_scale_multipliers(), {})
        self.assertEqual(controller.per_gene_rate_multipliers(), {})


if __name__ == "__main__":
    unittest.main()

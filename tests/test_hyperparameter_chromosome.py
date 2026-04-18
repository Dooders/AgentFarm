"""Tests for typed hyperparameter chromosome schema."""

import unittest
import json
import random
from unittest.mock import patch

from farm.core.hyperparameter_chromosome import (
    BoundaryMode,
    BoundaryPenaltyConfig,
    compute_boundary_penalty,
    CrossoverMode,
    MutationMode,
    apply_chromosome_to_learning_config,
    crossover_chromosomes,
    decode_chromosome,
    decode_chromosome_vector,
    encode_chromosome,
    encode_chromosome_vector,
    GeneValueType,
    HyperparameterChromosome,
    HyperparameterGene,
    chromosome_from_learning_config,
    chromosome_from_values,
    default_hyperparameter_registry,
    default_hyperparameter_chromosome,
    hyperparameter_evolution_registry,
    mutate_chromosome,
)
from farm.core.decision.config import DecisionConfig


class _LearningConfigStub:
    learning_rate = 0.02
    epsilon_decay = 0.97
    memory_size = 4096


class _LowEpsilonLearningConfigStub:
    learning_rate = 0.02
    epsilon_decay = 0.1
    memory_size = 4096


class TestHyperparameterGene(unittest.TestCase):
    def test_rejects_out_of_range_value(self):
        with self.assertRaises(ValueError):
            HyperparameterGene(
                name="learning_rate",
                value_type=GeneValueType.REAL,
                value=1.5,
                min_value=1e-6,
                max_value=1.0,
                default=0.001,
            )

    def test_rejects_out_of_range_default(self):
        with self.assertRaises(ValueError):
            HyperparameterGene(
                name="learning_rate",
                value_type=GeneValueType.REAL,
                value=0.01,
                min_value=1e-6,
                max_value=1.0,
                default=2.0,
            )


class TestHyperparameterChromosome(unittest.TestCase):
    def test_default_chromosome_contains_learning_rate(self):
        chromosome = default_hyperparameter_chromosome()
        self.assertEqual(chromosome.get_value("learning_rate"), 0.001)

    def test_default_registry_marks_evolvable_vs_fixed(self):
        registry = hyperparameter_evolution_registry()
        self.assertTrue(registry["learning_rate"])
        self.assertFalse(registry["epsilon_decay"])
        self.assertFalse(registry["memory_size"])

    def test_short_registry_alias_matches_evolution_registry(self):
        self.assertEqual(default_hyperparameter_registry(), hyperparameter_evolution_registry())

    def test_rejects_unknown_override_name(self):
        chromosome = default_hyperparameter_chromosome()
        with self.assertRaises(KeyError):
            chromosome.with_overrides({"not_a_gene": 0.1})

    def test_rejects_out_of_range_override(self):
        chromosome = default_hyperparameter_chromosome()
        with self.assertRaises(ValueError):
            chromosome.with_overrides({"learning_rate": 2.0})

    def test_serialization_round_trip(self):
        original = chromosome_from_values({"learning_rate": 0.05, "epsilon_decay": 0.97})
        serialized = original.to_dict()
        restored = HyperparameterChromosome.from_dict(serialized)

        self.assertEqual(
            [gene.to_dict() for gene in original.genes],
            [gene.to_dict() for gene in restored.genes],
        )
        self.assertEqual(restored.get_value("learning_rate"), 0.05)
        self.assertEqual(restored.get_value("epsilon_decay"), 0.97)

    def test_serialization_preserves_gene_order(self):
        chromosome = chromosome_from_values({"learning_rate": 0.03, "epsilon_decay": 0.92})
        payload = chromosome.to_dict()
        restored = HyperparameterChromosome.from_dict(payload)
        self.assertEqual(
            [gene.name for gene in chromosome.genes],
            [gene.name for gene in restored.genes],
        )

    def test_serialization_round_trip_through_json_keeps_float_precision(self):
        original = chromosome_from_values({"learning_rate": 0.123456789012345, "epsilon_decay": 0.987654321012345})
        as_json = json.dumps(original.to_dict(), sort_keys=True)
        restored = HyperparameterChromosome.from_dict(json.loads(as_json))
        self.assertAlmostEqual(restored.get_value("learning_rate"), 0.123456789012345, places=15)
        self.assertAlmostEqual(restored.get_value("epsilon_decay"), 0.987654321012345, places=15)

    def test_builds_from_learning_config_like_object(self):
        chromosome = chromosome_from_learning_config(_LearningConfigStub())
        self.assertEqual(chromosome.get_value("learning_rate"), 0.02)
        self.assertEqual(chromosome.get_value("epsilon_decay"), 0.97)
        self.assertEqual(chromosome.get_value("memory_size"), 4096.0)

    def test_accepts_low_valid_epsilon_decay_from_learning_config(self):
        chromosome = chromosome_from_learning_config(_LowEpsilonLearningConfigStub())
        self.assertEqual(chromosome.get_value("epsilon_decay"), 0.1)

    def test_accepts_tiny_valid_epsilon_decay_matching_decision_config(self):
        """Values in (0, 1] accepted by DecisionConfig must encode into the chromosome."""
        decision = DecisionConfig(learning_rate=0.02, epsilon_decay=5e-7, memory_size=1000)
        chromosome = chromosome_from_learning_config(decision)
        self.assertEqual(chromosome.get_value("epsilon_decay"), 5e-7)

    def test_accepts_large_memory_size_override(self):
        chromosome = chromosome_from_values({"memory_size": 250_000.0})
        self.assertEqual(chromosome.get_value("memory_size"), 250_000.0)

    def test_mutation_only_changes_evolvable_genes(self):
        chromosome = default_hyperparameter_chromosome()
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=0.05):
            mutated = mutate_chromosome(chromosome, mutation_rate=1.0, mutation_scale=0.1)
        self.assertNotEqual(mutated.get_value("learning_rate"), chromosome.get_value("learning_rate"))
        self.assertEqual(mutated.get_value("epsilon_decay"), chromosome.get_value("epsilon_decay"))
        self.assertEqual(mutated.get_value("memory_size"), chromosome.get_value("memory_size"))

    def test_gaussian_mutation_clamps_to_gene_bounds(self):
        chromosome = chromosome_from_values({"learning_rate": 0.999})
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=5.0):
            mutated = mutate_chromosome(chromosome, mutation_rate=1.0, mutation_scale=1.0)
        self.assertEqual(mutated.get_value("learning_rate"), 1.0)

    def test_mutation_supports_legacy_multiplicative_mode(self):
        chromosome = chromosome_from_values({"learning_rate": 0.01})
        with patch("farm.core.hyperparameter_chromosome.random.uniform", return_value=0.2):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=0.2,
                mutation_mode=MutationMode.MULTIPLICATIVE,
            )
        self.assertAlmostEqual(mutated.get_value("learning_rate"), 0.012)

    def test_mutation_uses_per_gene_controls_when_globals_not_provided(self):
        chromosome = HyperparameterChromosome(
            genes=(
                HyperparameterGene(
                    name="learning_rate",
                    value_type=GeneValueType.REAL,
                    value=0.01,
                    min_value=1e-6,
                    max_value=1.0,
                    default=0.001,
                    evolvable=True,
                    mutation_scale=0.5,
                    mutation_probability=1.0,
                    mutation_strategy=MutationMode.MULTIPLICATIVE,
                ),
            )
        )
        with patch("farm.core.hyperparameter_chromosome.random.uniform", return_value=0.5):
            mutated = mutate_chromosome(chromosome)
        self.assertAlmostEqual(mutated.get_value("learning_rate"), 0.015)

    def test_mutation_globals_override_per_gene_controls(self):
        chromosome = HyperparameterChromosome(
            genes=(
                HyperparameterGene(
                    name="learning_rate",
                    value_type=GeneValueType.REAL,
                    value=0.01,
                    min_value=1e-6,
                    max_value=1.0,
                    default=0.001,
                    evolvable=True,
                    mutation_scale=0.0,
                    mutation_probability=0.0,
                    mutation_strategy=MutationMode.MULTIPLICATIVE,
                ),
            )
        )
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=0.001):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=0.1,
                mutation_mode=MutationMode.GAUSSIAN,
            )
        self.assertAlmostEqual(mutated.get_value("learning_rate"), 0.011)

    def test_fixed_genes_remain_unchanged_even_with_full_global_mutation(self):
        chromosome = default_hyperparameter_chromosome()
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=1000.0):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=1.0,
                mutation_mode=MutationMode.GAUSSIAN,
            )
        self.assertEqual(mutated.get_value("epsilon_decay"), chromosome.get_value("epsilon_decay"))
        self.assertEqual(mutated.get_value("memory_size"), chromosome.get_value("memory_size"))

    def test_apply_chromosome_to_learning_config(self):
        decision = DecisionConfig(learning_rate=0.02, epsilon_decay=0.98, memory_size=5000)
        chromosome = chromosome_from_values({
            "learning_rate": 0.03,
            "epsilon_decay": 0.97,
            "memory_size": 4096.0,
        })
        updated = apply_chromosome_to_learning_config(decision, chromosome)

        self.assertEqual(updated.learning_rate, 0.03)
        self.assertEqual(updated.epsilon_decay, 0.97)
        self.assertEqual(updated.memory_size, 4096)

    def test_apply_chromosome_to_learning_config_plain_object_does_not_mutate_original(self):
        stub = _LearningConfigStub()
        chromosome = chromosome_from_values({"learning_rate": 0.05})
        updated = apply_chromosome_to_learning_config(stub, chromosome)

        self.assertIsNot(updated, stub)
        self.assertEqual(updated.learning_rate, 0.05)
        self.assertEqual(stub.learning_rate, 0.02)


class TestHyperparameterEncoding(unittest.TestCase):
    def test_learning_rate_quantized_encoding_uses_8bit_bounds(self):
        chromosome = default_hyperparameter_chromosome()
        learning_rate_gene = chromosome.get_gene("learning_rate")
        assert learning_rate_gene is not None

        self.assertEqual(learning_rate_gene.encode(learning_rate_gene.min_value), 0)
        self.assertEqual(learning_rate_gene.encode(learning_rate_gene.max_value), 255)

    def test_learning_rate_encoding_is_monotonic(self):
        chromosome = default_hyperparameter_chromosome()
        learning_rate_gene = chromosome.get_gene("learning_rate")
        assert learning_rate_gene is not None

        values = [1e-6, 1e-4, 1e-2, 0.1, 1.0]
        encoded = [learning_rate_gene.encode(value) for value in values]
        self.assertEqual(encoded, sorted(encoded))

    def test_learning_rate_encode_decode_round_trip_with_quantization_tolerance(self):
        chromosome = default_hyperparameter_chromosome()
        learning_rate_gene = chromosome.get_gene("learning_rate")
        assert learning_rate_gene is not None

        original = 0.0123
        encoded = learning_rate_gene.encode(original)
        decoded = learning_rate_gene.decode(encoded)
        self.assertAlmostEqual(decoded, original, delta=original * 0.06)

    def test_chromosome_dict_encode_decode_round_trip(self):
        chromosome = chromosome_from_values({"learning_rate": 0.02})
        encoded = encode_chromosome(chromosome)
        decoded = decode_chromosome(encoded, template=chromosome)
        self.assertAlmostEqual(
            decoded.get_value("learning_rate"),
            chromosome.get_value("learning_rate"),
            delta=chromosome.get_value("learning_rate") * 0.06,
        )

    def test_chromosome_vector_encode_decode_round_trip(self):
        chromosome = chromosome_from_values({"learning_rate": 0.02})
        encoded = encode_chromosome_vector(chromosome)
        decoded = decode_chromosome_vector(encoded, template=chromosome)
        self.assertAlmostEqual(
            decoded.get_value("learning_rate"),
            chromosome.get_value("learning_rate"),
            delta=chromosome.get_value("learning_rate") * 0.06,
        )

    def test_decode_rejects_out_of_range_quantized_value(self):
        chromosome = default_hyperparameter_chromosome()
        learning_rate_gene = chromosome.get_gene("learning_rate")
        assert learning_rate_gene is not None

        with self.assertRaises(ValueError):
            learning_rate_gene.decode(256)

    def test_decode_chromosome_rejects_unknown_gene_name(self):
        with self.assertRaises(KeyError):
            decode_chromosome({"unknown_gene": 12})


class TestHyperparameterCrossover(unittest.TestCase):
    def test_single_point_crossover_combines_parent_vectors(self):
        parent_a = chromosome_from_values({"learning_rate": 0.01, "epsilon_decay": 0.8})
        parent_b = chromosome_from_values({"learning_rate": 0.5, "epsilon_decay": 0.9})

        child = crossover_chromosomes(
            parent_a,
            parent_b,
            mode=CrossoverMode.SINGLE_POINT,
            include_fixed=True,
            rng=random.Random(7),
        )

        # With include_fixed=True and deterministic RNG, suffix should come from parent_b.
        self.assertIn(child.get_value("learning_rate"), (0.01, 0.5))
        self.assertIn(child.get_value("epsilon_decay"), (0.8, 0.9))
        self.assertIn(child.get_value("memory_size"), (10000.0,))

    def test_uniform_crossover_uses_probability_for_parent_b(self):
        parent_a = chromosome_from_values({"learning_rate": 0.01})
        parent_b = chromosome_from_values({"learning_rate": 0.5})

        child = crossover_chromosomes(
            parent_a,
            parent_b,
            mode=CrossoverMode.UNIFORM,
            uniform_parent_b_probability=1.0,
        )
        self.assertEqual(child.get_value("learning_rate"), 0.5)

    def test_uniform_crossover_keeps_values_in_gene_ranges(self):
        parent_a = chromosome_from_values({"learning_rate": 1e-6})
        parent_b = chromosome_from_values({"learning_rate": 1.0})

        child = crossover_chromosomes(
            parent_a,
            parent_b,
            mode=CrossoverMode.UNIFORM,
            uniform_parent_b_probability=0.5,
            rng=random.Random(3),
        )
        learning_rate = child.get_value("learning_rate")
        self.assertGreaterEqual(learning_rate, 1e-6)
        self.assertLessEqual(learning_rate, 1.0)

    def test_crossover_then_mutation_keeps_offspring_in_range(self):
        parent_a = chromosome_from_values({"learning_rate": 1e-6})
        parent_b = chromosome_from_values({"learning_rate": 1.0})
        child = crossover_chromosomes(
            parent_a,
            parent_b,
            mode=CrossoverMode.UNIFORM,
            uniform_parent_b_probability=0.5,
            rng=random.Random(11),
        )
        mutated = mutate_chromosome(
            child,
            mutation_rate=1.0,
            mutation_scale=1.0,
            mutation_mode=MutationMode.GAUSSIAN,
            rng=random.Random(19),
        )
        learning_rate = mutated.get_value("learning_rate")
        self.assertGreaterEqual(learning_rate, 1e-6)
        self.assertLessEqual(learning_rate, 1.0)


class TestBoundaryMode(unittest.TestCase):
    """Tests for reflective (bounce) mutation boundary handling."""

    def test_reflect_simple_overshoot_above_max(self):
        # Gene [0, 1], value 0.9, mutation pushes to 1.2 → reflects to 0.8
        chromosome = chromosome_from_values({"learning_rate": 0.9})
        # Gaussian perturbation: sigma = (1.0 - 1e-6) * 1.0 ≈ 1.0
        # gauss returns 0.3 → raw = 0.9 + 0.3 = 1.2; span ≈ 1.0 → reflect to 0.8
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=0.3):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=1.0,
                boundary_mode=BoundaryMode.REFLECT,
            )
        lr = mutated.get_value("learning_rate")
        self.assertGreaterEqual(lr, 1e-6)
        self.assertLessEqual(lr, 1.0)
        # Should NOT be at the max boundary (was reflected back in)
        self.assertLess(lr, 1.0)

    def test_reflect_simple_overshoot_below_min(self):
        # Gene [1e-6, 1], value at min + small amount, mutation pushes below min
        chromosome = chromosome_from_values({"learning_rate": 0.001})
        # gauss returns large negative: raw = 0.001 - 5.0 << min
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=-5.0):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=1.0,
                boundary_mode=BoundaryMode.REFLECT,
            )
        lr = mutated.get_value("learning_rate")
        self.assertGreaterEqual(lr, 1e-6)
        self.assertLessEqual(lr, 1.0)

    def test_clamp_stays_at_boundary_on_large_overshoot(self):
        chromosome = chromosome_from_values({"learning_rate": 0.999})
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=5.0):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=1.0,
                boundary_mode=BoundaryMode.CLAMP,
            )
        self.assertEqual(mutated.get_value("learning_rate"), 1.0)

    def test_reflect_does_not_stick_at_boundary_on_overshoot(self):
        chromosome = chromosome_from_values({"learning_rate": 0.999})
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=0.5):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=1.0,
                boundary_mode=BoundaryMode.REFLECT,
            )
        # With reflect, value should not be exactly at max (1.0)
        self.assertLess(mutated.get_value("learning_rate"), 1.0)

    def test_reflect_value_stays_in_bounds_for_many_mutations(self):
        rng = random.Random(42)
        chromosome = chromosome_from_values({"learning_rate": 0.5})
        for _ in range(200):
            chromosome = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=2.0,
                boundary_mode=BoundaryMode.REFLECT,
                rng=rng,
            )
            lr = chromosome.get_value("learning_rate")
            self.assertGreaterEqual(lr, 1e-6, msg=f"Fell below min: {lr}")
            self.assertLessEqual(lr, 1.0, msg=f"Exceeded max: {lr}")

    def test_clamp_default_backward_compatible(self):
        """mutate_chromosome without boundary_mode still clamps (backward compat)."""
        chromosome = chromosome_from_values({"learning_rate": 0.999})
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=5.0):
            mutated = mutate_chromosome(chromosome, mutation_rate=1.0, mutation_scale=1.0)
        self.assertEqual(mutated.get_value("learning_rate"), 1.0)

    def test_reflect_string_alias_accepted(self):
        chromosome = chromosome_from_values({"learning_rate": 0.5})
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=0.0):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=0.1,
                boundary_mode="reflect",
            )
        lr = mutated.get_value("learning_rate")
        self.assertGreaterEqual(lr, 1e-6)
        self.assertLessEqual(lr, 1.0)

    def test_clamp_string_alias_accepted(self):
        chromosome = chromosome_from_values({"learning_rate": 0.5})
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=0.0):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=0.1,
                boundary_mode="clamp",
            )
        lr = mutated.get_value("learning_rate")
        self.assertGreaterEqual(lr, 1e-6)
        self.assertLessEqual(lr, 1.0)


class TestBoundaryPenaltyConfig(unittest.TestCase):
    """Tests for BoundaryPenaltyConfig validation."""

    def test_default_config_is_disabled(self):
        cfg = BoundaryPenaltyConfig()
        self.assertFalse(cfg.enabled)

    def test_rejects_negative_penalty_strength(self):
        with self.assertRaises(ValueError):
            BoundaryPenaltyConfig(enabled=True, penalty_strength=-0.01)

    def test_rejects_zero_near_boundary_threshold(self):
        with self.assertRaises(ValueError):
            BoundaryPenaltyConfig(enabled=True, near_boundary_threshold=0.0)

    def test_rejects_threshold_above_half(self):
        with self.assertRaises(ValueError):
            BoundaryPenaltyConfig(enabled=True, near_boundary_threshold=0.6)

    def test_accepts_threshold_at_half(self):
        cfg = BoundaryPenaltyConfig(enabled=True, near_boundary_threshold=0.5)
        self.assertEqual(cfg.near_boundary_threshold, 0.5)


class TestComputeBoundaryPenalty(unittest.TestCase):
    """Tests for compute_boundary_penalty()."""

    def test_returns_zero_when_disabled(self):
        chromosome = chromosome_from_values({"learning_rate": 1.0})
        self.assertEqual(compute_boundary_penalty(chromosome), 0.0)

    def test_returns_zero_when_config_disabled_explicitly(self):
        chromosome = chromosome_from_values({"learning_rate": 1.0})
        cfg = BoundaryPenaltyConfig(enabled=False)
        self.assertEqual(compute_boundary_penalty(chromosome, cfg), 0.0)

    def test_full_penalty_at_max_boundary(self):
        chromosome = chromosome_from_values({"learning_rate": 1.0})
        cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.05, near_boundary_threshold=0.1)
        penalty = compute_boundary_penalty(chromosome, cfg)
        self.assertAlmostEqual(penalty, 0.05)

    def test_full_penalty_at_min_boundary(self):
        chromosome = chromosome_from_values({"learning_rate": 1e-6})
        cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.05, near_boundary_threshold=0.1)
        penalty = compute_boundary_penalty(chromosome, cfg)
        self.assertAlmostEqual(penalty, 0.05)

    def test_zero_penalty_well_inside_bounds(self):
        chromosome = chromosome_from_values({"learning_rate": 0.5})
        cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.1, near_boundary_threshold=0.05)
        penalty = compute_boundary_penalty(chromosome, cfg)
        self.assertEqual(penalty, 0.0)

    def test_penalty_ramps_linearly(self):
        """Penalty at half the threshold distance should be ~50% of strength."""
        gene = HyperparameterGene(
            name="learning_rate",
            value_type=GeneValueType.REAL,
            value=0.05,  # normalized = 0.05 in [0, 1] range
            min_value=0.0,
            max_value=1.0,
            default=0.5,
        )
        chromosome = HyperparameterChromosome(genes=(gene,))
        cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.10, near_boundary_threshold=0.10)
        penalty = compute_boundary_penalty(chromosome, cfg)
        # distance = 0.05, threshold = 0.10 → fraction = 1 - 0.05/0.10 = 0.5
        self.assertAlmostEqual(penalty, 0.05)

    def test_no_penalty_for_fixed_genes(self):
        chromosome = default_hyperparameter_chromosome()
        # epsilon_decay and memory_size are fixed (evolvable=False)
        # Only learning_rate is evolvable; at its default of 0.001 it is very
        # close to min (1e-6) on a linear scale → may receive a penalty.
        # Override learning_rate to midpoint to guarantee zero penalty.
        chromosome = chromosome.with_overrides({"learning_rate": 0.5})
        cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.1, near_boundary_threshold=0.05)
        # Even though epsilon_decay is near its min, it is fixed → no penalty
        penalty = compute_boundary_penalty(chromosome, cfg)
        self.assertEqual(penalty, 0.0)

    def test_penalty_sums_over_multiple_evolvable_genes(self):
        gene_a = HyperparameterGene(
            name="a",
            value_type=GeneValueType.REAL,
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            default=0.5,
            evolvable=True,
        )
        gene_b = HyperparameterGene(
            name="b",
            value_type=GeneValueType.REAL,
            value=1.0,
            min_value=0.0,
            max_value=1.0,
            default=0.5,
            evolvable=True,
        )
        chromosome = HyperparameterChromosome(genes=(gene_a, gene_b))
        cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.10, near_boundary_threshold=0.05)
        penalty = compute_boundary_penalty(chromosome, cfg)
        # Both genes exactly on a boundary → 2 × 0.10 = 0.20
        self.assertAlmostEqual(penalty, 0.20)


if __name__ == "__main__":
    unittest.main()

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
        self.assertTrue(registry["gamma"])
        self.assertTrue(registry["epsilon_decay"])
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

    def test_per_gene_rate_multiplier_can_suppress_mutation(self):
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
                    mutation_scale=0.1,
                    mutation_probability=0.5,
                    mutation_strategy=MutationMode.GAUSSIAN,
                ),
            )
        )
        # Rate multiplier of 0.0 suppresses mutation for this gene even when
        # the global rate override is 1.0.
        mutated = mutate_chromosome(
            chromosome,
            mutation_rate=1.0,
            mutation_scale=0.1,
            per_gene_rate_multipliers={"learning_rate": 0.0},
        )
        self.assertEqual(mutated.get_value("learning_rate"), 0.01)

    def test_per_gene_scale_multiplier_scales_perturbation(self):
        chromosome = HyperparameterChromosome(
            genes=(
                HyperparameterGene(
                    name="learning_rate",
                    value_type=GeneValueType.REAL,
                    value=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    default=0.5,
                    evolvable=True,
                    mutation_scale=0.1,
                    mutation_probability=1.0,
                    mutation_strategy=MutationMode.GAUSSIAN,
                ),
            )
        )
        # Force a deterministic Gaussian perturbation of sigma-magnitude by
        # returning the sigma value from random.gauss.
        with patch(
            "farm.core.hyperparameter_chromosome.random.gauss",
            side_effect=lambda mu, sigma: sigma,
        ):
            mutated_base = mutate_chromosome(chromosome, mutation_rate=1.0)
            mutated_scaled = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                per_gene_scale_multipliers={"learning_rate": 2.0},
            )
        delta_base = mutated_base.get_value("learning_rate") - 0.5
        delta_scaled = mutated_scaled.get_value("learning_rate") - 0.5
        self.assertAlmostEqual(delta_scaled, delta_base * 2.0)

    def test_per_gene_multiplier_rejects_negative_values(self):
        chromosome = default_hyperparameter_chromosome()
        with self.assertRaises(ValueError):
            mutate_chromosome(
                chromosome,
                per_gene_rate_multipliers={"learning_rate": -0.1},
            )
        with self.assertRaises(ValueError):
            mutate_chromosome(
                chromosome,
                per_gene_scale_multipliers={"learning_rate": -0.1},
            )

    def test_fixed_genes_remain_unchanged_even_with_full_global_mutation(self):
        chromosome = default_hyperparameter_chromosome()
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=1000.0):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=1.0,
                mutation_mode=MutationMode.GAUSSIAN,
            )
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


class TestAdditionalContinuousGenes(unittest.TestCase):
    """Tests for the two additional evolvable continuous genes: gamma and epsilon_decay."""

    # --- gamma gene: bounds, defaults, serialization ---

    def test_default_chromosome_contains_gamma(self):
        chromosome = default_hyperparameter_chromosome()
        self.assertAlmostEqual(chromosome.get_value("gamma"), 0.99)

    def test_gamma_gene_bounds_valid(self):
        chromosome = chromosome_from_values({"gamma": 0.0})
        self.assertEqual(chromosome.get_value("gamma"), 0.0)
        chromosome = chromosome_from_values({"gamma": 1.0})
        self.assertEqual(chromosome.get_value("gamma"), 1.0)

    def test_gamma_gene_rejects_out_of_range_value(self):
        with self.assertRaises(ValueError):
            chromosome_from_values({"gamma": 1.5})

    def test_gamma_gene_rejects_below_min(self):
        with self.assertRaises(ValueError):
            chromosome_from_values({"gamma": -0.1})

    # --- epsilon_decay gene: now evolvable ---

    def test_epsilon_decay_is_evolvable(self):
        chromosome = default_hyperparameter_chromosome()
        self.assertIn("epsilon_decay", chromosome.evolvable_gene_names())

    def test_gamma_is_evolvable(self):
        chromosome = default_hyperparameter_chromosome()
        self.assertIn("gamma", chromosome.evolvable_gene_names())

    def test_memory_size_is_still_fixed(self):
        chromosome = default_hyperparameter_chromosome()
        self.assertIn("memory_size", chromosome.fixed_gene_names())
        self.assertNotIn("memory_size", chromosome.evolvable_gene_names())

    # --- serialization round-trips for new genes ---

    def test_gamma_serialization_round_trip(self):
        original = chromosome_from_values({"gamma": 0.75})
        restored = HyperparameterChromosome.from_dict(original.to_dict())
        self.assertAlmostEqual(restored.get_value("gamma"), 0.75, places=15)

    def test_all_genes_serialization_round_trip(self):
        original = chromosome_from_values({
            "learning_rate": 0.005,
            "gamma": 0.95,
            "epsilon_decay": 0.99,
            "memory_size": 5000.0,
        })
        restored = HyperparameterChromosome.from_dict(original.to_dict())
        self.assertAlmostEqual(restored.get_value("learning_rate"), 0.005, places=15)
        self.assertAlmostEqual(restored.get_value("gamma"), 0.95, places=15)
        self.assertAlmostEqual(restored.get_value("epsilon_decay"), 0.99, places=15)
        self.assertAlmostEqual(restored.get_value("memory_size"), 5000.0, places=15)

    def test_all_genes_serialization_through_json(self):
        original = chromosome_from_values({"learning_rate": 0.003, "gamma": 0.97, "epsilon_decay": 0.998})
        as_json = json.dumps(original.to_dict(), sort_keys=True)
        restored = HyperparameterChromosome.from_dict(json.loads(as_json))
        self.assertAlmostEqual(restored.get_value("learning_rate"), 0.003, places=15)
        self.assertAlmostEqual(restored.get_value("gamma"), 0.97, places=15)
        self.assertAlmostEqual(restored.get_value("epsilon_decay"), 0.998, places=15)

    # --- encode/decode for new genes ---

    def test_gamma_encode_decode_round_trip(self):
        chromosome = default_hyperparameter_chromosome()
        gamma_gene = chromosome.get_gene("gamma")
        assert gamma_gene is not None
        for value in (0.0, 0.5, 0.9, 0.99, 1.0):
            encoded = gamma_gene.encode(value)
            decoded = gamma_gene.decode(encoded)
            self.assertAlmostEqual(decoded, value, delta=0.01)

    def test_gamma_encoding_uses_8bit_bounds(self):
        chromosome = default_hyperparameter_chromosome()
        gamma_gene = chromosome.get_gene("gamma")
        assert gamma_gene is not None
        self.assertEqual(gamma_gene.encode(gamma_gene.min_value), 0)
        self.assertEqual(gamma_gene.encode(gamma_gene.max_value), 255)

    def test_epsilon_decay_encode_decode_round_trip(self):
        chromosome = default_hyperparameter_chromosome()
        ed_gene = chromosome.get_gene("epsilon_decay")
        assert ed_gene is not None
        for value in (0.9, 0.95, 0.995, 1.0):
            encoded = ed_gene.encode(value)
            decoded = ed_gene.decode(encoded)
            # Linear 8-bit encoding has quantization error ~1/255 of the range (~0.004)
            self.assertAlmostEqual(decoded, value, delta=0.005)

    def test_epsilon_decay_encoding_uses_8bit_bounds(self):
        chromosome = default_hyperparameter_chromosome()
        ed_gene = chromosome.get_gene("epsilon_decay")
        assert ed_gene is not None
        self.assertEqual(ed_gene.encode(ed_gene.min_value), 0)
        self.assertEqual(ed_gene.encode(ed_gene.max_value), 255)

    def test_multi_gene_chromosome_encode_decode_round_trip(self):
        chromosome = chromosome_from_values({"learning_rate": 0.005, "gamma": 0.95, "epsilon_decay": 0.99})
        encoded = encode_chromosome(chromosome)
        decoded = decode_chromosome(encoded, template=chromosome)
        self.assertAlmostEqual(decoded.get_value("learning_rate"), 0.005, delta=0.005 * 0.06)
        self.assertAlmostEqual(decoded.get_value("gamma"), 0.95, delta=0.01)
        self.assertAlmostEqual(decoded.get_value("epsilon_decay"), 0.99, delta=0.005)

    def test_multi_gene_chromosome_vector_encode_decode_round_trip(self):
        chromosome = chromosome_from_values({"learning_rate": 0.005, "gamma": 0.95, "epsilon_decay": 0.99})
        vec = encode_chromosome_vector(chromosome)
        # 3 evolvable genes → 3-element vector
        self.assertEqual(len(vec), 3)
        decoded = decode_chromosome_vector(vec, template=chromosome)
        self.assertAlmostEqual(decoded.get_value("learning_rate"), 0.005, delta=0.005 * 0.06)
        self.assertAlmostEqual(decoded.get_value("gamma"), 0.95, delta=0.01)
        self.assertAlmostEqual(decoded.get_value("epsilon_decay"), 0.99, delta=0.005)

    # --- mutation: all evolvable genes change, fixed unchanged ---

    def test_multi_gene_mutation_changes_all_evolvable_genes(self):
        chromosome = chromosome_from_values({
            "learning_rate": 0.5,
            "gamma": 0.5,
            "epsilon_decay": 0.5,
        })
        rng = random.Random(42)
        mutated = mutate_chromosome(chromosome, mutation_rate=1.0, mutation_scale=0.1, rng=rng)
        # All three evolvable genes should move; memory_size stays fixed.
        self.assertNotEqual(mutated.get_value("learning_rate"), 0.5)
        self.assertNotEqual(mutated.get_value("gamma"), 0.5)
        self.assertNotEqual(mutated.get_value("epsilon_decay"), 0.5)
        self.assertEqual(mutated.get_value("memory_size"), chromosome.get_value("memory_size"))

    def test_gamma_mutation_stays_in_bounds(self):
        chromosome = chromosome_from_values({"gamma": 0.99})
        rng = random.Random(7)
        for _ in range(50):
            chromosome = mutate_chromosome(
                chromosome, mutation_rate=1.0, mutation_scale=1.0, rng=rng
            )
            g = chromosome.get_value("gamma")
            self.assertGreaterEqual(g, 0.0)
            self.assertLessEqual(g, 1.0)

    def test_epsilon_decay_mutation_stays_in_bounds(self):
        chromosome = chromosome_from_values({"epsilon_decay": 0.5})
        ed_min = chromosome.get_gene("epsilon_decay").min_value
        rng = random.Random(13)
        for _ in range(50):
            chromosome = mutate_chromosome(
                chromosome, mutation_rate=1.0, mutation_scale=1.0, rng=rng
            )
            ed = chromosome.get_value("epsilon_decay")
            self.assertGreaterEqual(ed, ed_min)
            self.assertLessEqual(ed, 1.0)

    # --- crossover: all evolvable genes participate ---

    def test_crossover_mixes_all_evolvable_genes(self):
        parent_a = chromosome_from_values({"learning_rate": 0.01, "gamma": 0.8, "epsilon_decay": 0.9})
        parent_b = chromosome_from_values({"learning_rate": 0.5, "gamma": 0.99, "epsilon_decay": 0.999})
        child = crossover_chromosomes(
            parent_a, parent_b, mode=CrossoverMode.UNIFORM, uniform_parent_b_probability=0.5,
            rng=random.Random(3),
        )
        # Each evolvable gene should come from one of the two parents.
        self.assertIn(child.get_value("learning_rate"), (0.01, 0.5))
        self.assertIn(child.get_value("gamma"), (0.8, 0.99))
        self.assertIn(child.get_value("epsilon_decay"), (0.9, 0.999))

    def test_crossover_then_mutation_keeps_all_evolvable_in_bounds(self):
        parent_a = chromosome_from_values({"learning_rate": 1e-6, "gamma": 0.0, "epsilon_decay": 0.9})
        parent_b = chromosome_from_values({"learning_rate": 1.0, "gamma": 1.0, "epsilon_decay": 1.0})
        ed_min = parent_a.get_gene("epsilon_decay").min_value
        child = crossover_chromosomes(
            parent_a, parent_b, mode=CrossoverMode.BLEND, blend_alpha=0.5, rng=random.Random(9)
        )
        mutated = mutate_chromosome(child, mutation_rate=1.0, mutation_scale=1.0, rng=random.Random(17))
        self.assertGreaterEqual(mutated.get_value("learning_rate"), 1e-6)
        self.assertLessEqual(mutated.get_value("learning_rate"), 1.0)
        self.assertGreaterEqual(mutated.get_value("gamma"), 0.0)
        self.assertLessEqual(mutated.get_value("gamma"), 1.0)
        self.assertGreaterEqual(mutated.get_value("epsilon_decay"), ed_min)
        self.assertLessEqual(mutated.get_value("epsilon_decay"), 1.0)

    # --- config projection ---

    def test_apply_chromosome_projects_gamma_to_decision_config(self):
        decision = DecisionConfig(learning_rate=0.001, gamma=0.99)
        chromosome = chromosome_from_values({"learning_rate": 0.005, "gamma": 0.95})
        updated = apply_chromosome_to_learning_config(decision, chromosome)
        self.assertAlmostEqual(updated.learning_rate, 0.005)
        self.assertAlmostEqual(updated.gamma, 0.95)

    def test_apply_chromosome_projects_epsilon_decay_to_decision_config(self):
        decision = DecisionConfig(epsilon_decay=0.995)
        chromosome = chromosome_from_values({"epsilon_decay": 0.98})
        updated = apply_chromosome_to_learning_config(decision, chromosome)
        self.assertAlmostEqual(updated.epsilon_decay, 0.98)

    def test_chromosome_from_learning_config_picks_up_gamma(self):
        decision = DecisionConfig(gamma=0.95)
        chromosome = chromosome_from_learning_config(decision)
        self.assertAlmostEqual(chromosome.get_value("gamma"), 0.95)

    def test_chromosome_from_learning_config_picks_up_all_new_evolvable_genes(self):
        decision = DecisionConfig(learning_rate=0.005, gamma=0.97, epsilon_decay=0.992, memory_size=2000)
        chromosome = chromosome_from_learning_config(decision)
        self.assertAlmostEqual(chromosome.get_value("learning_rate"), 0.005)
        self.assertAlmostEqual(chromosome.get_value("gamma"), 0.97)
        self.assertAlmostEqual(chromosome.get_value("epsilon_decay"), 0.992)
        self.assertAlmostEqual(chromosome.get_value("memory_size"), 2000.0)


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

    # --- Blend (BLX-α) crossover ---

    def test_blend_crossover_value_within_gene_bounds(self):
        parent_a = chromosome_from_values({"learning_rate": 0.01})
        parent_b = chromosome_from_values({"learning_rate": 0.5})
        for seed in range(20):
            child = crossover_chromosomes(
                parent_a,
                parent_b,
                mode=CrossoverMode.BLEND,
                blend_alpha=0.5,
                rng=random.Random(seed),
            )
            lr = child.get_value("learning_rate")
            self.assertGreaterEqual(lr, 1e-6)
            self.assertLessEqual(lr, 1.0)

    def test_blend_crossover_zero_alpha_stays_between_parents(self):
        parent_a = chromosome_from_values({"learning_rate": 0.1})
        parent_b = chromosome_from_values({"learning_rate": 0.9})
        for seed in range(20):
            child = crossover_chromosomes(
                parent_a,
                parent_b,
                mode=CrossoverMode.BLEND,
                blend_alpha=0.0,
                rng=random.Random(seed),
            )
            lr = child.get_value("learning_rate")
            self.assertGreaterEqual(lr, 0.1)
            self.assertLessEqual(lr, 0.9)

    def test_blend_crossover_deterministic_with_seed(self):
        parent_a = chromosome_from_values({"learning_rate": 0.01})
        parent_b = chromosome_from_values({"learning_rate": 0.5})
        child1 = crossover_chromosomes(
            parent_a, parent_b, mode=CrossoverMode.BLEND, rng=random.Random(42)
        )
        child2 = crossover_chromosomes(
            parent_a, parent_b, mode=CrossoverMode.BLEND, rng=random.Random(42)
        )
        self.assertEqual(child1.get_value("learning_rate"), child2.get_value("learning_rate"))

    def test_blend_crossover_rejects_negative_alpha(self):
        parent_a = chromosome_from_values({"learning_rate": 0.01})
        parent_b = chromosome_from_values({"learning_rate": 0.5})
        with self.assertRaises(ValueError):
            crossover_chromosomes(
                parent_a, parent_b, mode=CrossoverMode.BLEND, blend_alpha=-0.1
            )

    def test_blend_crossover_string_mode(self):
        parent_a = chromosome_from_values({"learning_rate": 0.1})
        parent_b = chromosome_from_values({"learning_rate": 0.5})
        child = crossover_chromosomes(
            parent_a, parent_b, mode="blend", rng=random.Random(7)
        )
        lr = child.get_value("learning_rate")
        self.assertGreaterEqual(lr, 1e-6)
        self.assertLessEqual(lr, 1.0)

    def test_blend_crossover_equal_parents_returns_same_value(self):
        parent_a = chromosome_from_values({"learning_rate": 0.3})
        parent_b = chromosome_from_values({"learning_rate": 0.3})
        child = crossover_chromosomes(
            parent_a, parent_b, mode=CrossoverMode.BLEND, blend_alpha=0.5, rng=random.Random(1)
        )
        # span is 0, so uniform(0, 0) → exact value
        self.assertAlmostEqual(child.get_value("learning_rate"), 0.3)

    # --- Multi-point crossover ---

    def test_multi_point_crossover_values_from_parents(self):
        parent_a = chromosome_from_values({"learning_rate": 0.01, "epsilon_decay": 0.8})
        parent_b = chromosome_from_values({"learning_rate": 0.5, "epsilon_decay": 0.9})
        child = crossover_chromosomes(
            parent_a,
            parent_b,
            mode=CrossoverMode.MULTI_POINT,
            include_fixed=True,
            num_crossover_points=2,
            rng=random.Random(5),
        )
        self.assertIn(child.get_value("learning_rate"), (0.01, 0.5))
        self.assertIn(child.get_value("epsilon_decay"), (0.8, 0.9))

    def test_multi_point_crossover_deterministic_with_seed(self):
        parent_a = chromosome_from_values({"learning_rate": 0.01, "epsilon_decay": 0.8})
        parent_b = chromosome_from_values({"learning_rate": 0.5, "epsilon_decay": 0.9})
        child1 = crossover_chromosomes(
            parent_a, parent_b, mode=CrossoverMode.MULTI_POINT,
            include_fixed=True, num_crossover_points=1, rng=random.Random(99)
        )
        child2 = crossover_chromosomes(
            parent_a, parent_b, mode=CrossoverMode.MULTI_POINT,
            include_fixed=True, num_crossover_points=1, rng=random.Random(99)
        )
        self.assertEqual(
            [g.value for g in child1.genes],
            [g.value for g in child2.genes],
        )

    def test_multi_point_crossover_alternates_segments_across_many_genes(self):
        parent_a = default_hyperparameter_chromosome()
        parent_b = HyperparameterChromosome(
            genes=tuple(
                gene.with_value(gene.max_value if gene.value != gene.max_value else gene.min_value)
                for gene in parent_a.genes
            )
        )
        seed = 123
        num_points = 3
        child = crossover_chromosomes(
            parent_a,
            parent_b,
            mode=CrossoverMode.MULTI_POINT,
            include_fixed=True,
            num_crossover_points=num_points,
            rng=random.Random(seed),
        )

        selected_indices = list(range(len(parent_a.genes)))
        n = len(selected_indices)
        effective_points = min(num_points, n - 1) if n > 1 else 0
        expected_genes = list(parent_a.genes)
        if effective_points == 0:
            selected_parent = parent_b if random.Random(seed).random() < 0.5 else parent_a
            expected_genes[selected_indices[0]] = selected_parent.genes[selected_indices[0]]
        else:
            pivot_positions = sorted(random.Random(seed).sample(range(1, n), effective_points))
            segment = 0
            for selected_position, gene_idx in enumerate(selected_indices):
                if segment < len(pivot_positions) and selected_position >= pivot_positions[segment]:
                    segment += 1
                source = parent_a if segment % 2 == 0 else parent_b
                expected_genes[gene_idx] = source.genes[gene_idx]

        self.assertEqual(
            [gene.value for gene in child.genes],
            [gene.value for gene in expected_genes],
        )

    def test_multi_point_crossover_single_gene_still_works(self):
        parent_a = chromosome_from_values({"learning_rate": 0.01})
        parent_b = chromosome_from_values({"learning_rate": 0.5})
        child = crossover_chromosomes(
            parent_a,
            parent_b,
            mode=CrossoverMode.MULTI_POINT,
            num_crossover_points=5,
            rng=random.Random(0),
        )
        self.assertIn(child.get_value("learning_rate"), (0.01, 0.5))

    def test_multi_point_crossover_rejects_zero_points(self):
        parent_a = chromosome_from_values({"learning_rate": 0.01})
        parent_b = chromosome_from_values({"learning_rate": 0.5})
        with self.assertRaises(ValueError):
            crossover_chromosomes(
                parent_a, parent_b, mode=CrossoverMode.MULTI_POINT, num_crossover_points=0
            )

    def test_multi_point_crossover_string_mode(self):
        parent_a = chromosome_from_values({"learning_rate": 0.01, "epsilon_decay": 0.8})
        parent_b = chromosome_from_values({"learning_rate": 0.5, "epsilon_decay": 0.9})
        child = crossover_chromosomes(
            parent_a, parent_b, mode="multi_point",
            include_fixed=True, rng=random.Random(3)
        )
        self.assertIn(child.get_value("learning_rate"), (0.01, 0.5))

    def test_blend_then_mutation_keeps_offspring_in_range(self):
        parent_a = chromosome_from_values({"learning_rate": 1e-6})
        parent_b = chromosome_from_values({"learning_rate": 1.0})
        child = crossover_chromosomes(
            parent_a, parent_b, mode=CrossoverMode.BLEND,
            blend_alpha=0.5, rng=random.Random(7)
        )
        mutated = mutate_chromosome(
            child, mutation_rate=1.0, mutation_scale=1.0,
            mutation_mode=MutationMode.GAUSSIAN, rng=random.Random(13)
        )
        lr = mutated.get_value("learning_rate")
        self.assertGreaterEqual(lr, 1e-6)
        self.assertLessEqual(lr, 1.0)


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
        # Set all other evolvable genes to midpoints so only learning_rate (at max) incurs a penalty.
        chromosome = chromosome_from_values({"learning_rate": 1.0, "gamma": 0.5, "epsilon_decay": 0.5})
        cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.05, near_boundary_threshold=0.1)
        penalty = compute_boundary_penalty(chromosome, cfg)
        self.assertAlmostEqual(penalty, 0.05)

    def test_full_penalty_at_min_boundary(self):
        # Set all other evolvable genes to midpoints so only learning_rate (at min) incurs a penalty.
        chromosome = chromosome_from_values({"learning_rate": 1e-6, "gamma": 0.5, "epsilon_decay": 0.5})
        cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.05, near_boundary_threshold=0.1)
        penalty = compute_boundary_penalty(chromosome, cfg)
        self.assertAlmostEqual(penalty, 0.05)

    def test_zero_penalty_well_inside_bounds(self):
        chromosome = chromosome_from_values({"learning_rate": 0.5, "gamma": 0.5, "epsilon_decay": 0.5})
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
        # memory_size is the only fixed gene (evolvable=False).
        # All evolvable genes (learning_rate, gamma, epsilon_decay) must be set
        # well inside bounds to guarantee zero penalty.
        chromosome = chromosome.with_overrides({
            "learning_rate": 0.5,
            "gamma": 0.5,
            "epsilon_decay": 0.5,
        })
        cfg = BoundaryPenaltyConfig(enabled=True, penalty_strength=0.1, near_boundary_threshold=0.05)
        # memory_size is fixed → no penalty even though its value is arbitrary
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


class TestInteriorBiasedBoundaryMode(unittest.TestCase):
    """Tests for the INTERIOR_BIASED boundary mode."""

    def test_interior_biased_string_alias_accepted(self):
        chromosome = chromosome_from_values({"learning_rate": 0.5})
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=0.0):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=0.1,
                boundary_mode="interior_biased",
            )
        lr = mutated.get_value("learning_rate")
        self.assertGreaterEqual(lr, 1e-6)
        self.assertLessEqual(lr, 1.0)

    def test_interior_biased_does_not_produce_exact_max_on_large_overshoot(self):
        """When clamped value would be at max, INTERIOR_BIASED nudges it inward."""
        chromosome = chromosome_from_values({"learning_rate": 0.999})
        rng = random.Random(1)
        # Large gauss pushes value above max → clamp to 1.0 → nudge inward
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=5.0):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=1.0,
                boundary_mode=BoundaryMode.INTERIOR_BIASED,
                interior_bias_fraction=1e-3,
                rng=rng,
            )
        lr = mutated.get_value("learning_rate")
        self.assertGreaterEqual(lr, 1e-6)
        self.assertLess(lr, 1.0, msg="INTERIOR_BIASED should nudge value below max boundary")

    def test_interior_biased_does_not_produce_exact_min_on_large_undershoot(self):
        """When clamped value would be at min, INTERIOR_BIASED nudges it inward."""
        chromosome = chromosome_from_values({"learning_rate": 0.001})
        rng = random.Random(2)
        # Large negative gauss pushes value below min → clamp to 1e-6 → nudge inward
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=-5.0):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=1.0,
                boundary_mode=BoundaryMode.INTERIOR_BIASED,
                interior_bias_fraction=1e-3,
                rng=rng,
            )
        lr = mutated.get_value("learning_rate")
        self.assertGreater(lr, 1e-6, msg="INTERIOR_BIASED should nudge value above min boundary")
        self.assertLessEqual(lr, 1.0)

    def test_interior_biased_strictly_interior_value_is_unchanged(self):
        """When the clamped value is strictly interior, no nudge is applied."""
        chromosome = chromosome_from_values({"learning_rate": 0.5})
        # gauss returns 0 → raw = 0.5 (no overshoot) → stays at 0.5
        # Note: do not pass rng so the module-level random.gauss patch applies.
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=0.0):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=0.1,
                boundary_mode=BoundaryMode.INTERIOR_BIASED,
                interior_bias_fraction=1e-3,
            )
        self.assertAlmostEqual(mutated.get_value("learning_rate"), 0.5)

    def test_interior_biased_stays_in_bounds_for_many_mutations(self):
        """No gene value escapes [min, max] under INTERIOR_BIASED over many steps."""
        rng = random.Random(99)
        chromosome = chromosome_from_values({"learning_rate": 1e-6})
        for _ in range(300):
            chromosome = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=3.0,
                boundary_mode=BoundaryMode.INTERIOR_BIASED,
                interior_bias_fraction=1e-3,
                rng=rng,
            )
            lr = chromosome.get_value("learning_rate")
            self.assertGreaterEqual(lr, 1e-6, msg=f"Fell below min: {lr}")
            self.assertLessEqual(lr, 1.0, msg=f"Exceeded max: {lr}")

    def test_interior_biased_zero_fraction_behaves_like_clamp(self):
        """interior_bias_fraction=0.0 means no nudge; value at exact boundary is kept."""
        chromosome = chromosome_from_values({"learning_rate": 0.999})
        rng = random.Random(7)
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=5.0):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=1.0,
                boundary_mode=BoundaryMode.INTERIOR_BIASED,
                interior_bias_fraction=0.0,
                rng=rng,
            )
        self.assertEqual(mutated.get_value("learning_rate"), 1.0)

    def test_interior_biased_rejects_negative_fraction(self):
        chromosome = chromosome_from_values({"learning_rate": 0.5})
        with self.assertRaises(ValueError):
            mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=0.1,
                boundary_mode=BoundaryMode.INTERIOR_BIASED,
                interior_bias_fraction=-0.01,
            )

    def test_interior_biased_nudge_within_bias_fraction_span(self):
        """The nudged value must be within interior_bias_fraction * span of the boundary."""
        chromosome = chromosome_from_values({"learning_rate": 0.999})
        bias_fraction = 0.01
        # Patch gauss to force overshoot → clamped to max (1.0).
        # Patch uniform to return a deterministic nudge of 0.005.
        with patch("farm.core.hyperparameter_chromosome.random.gauss", return_value=5.0), \
             patch("farm.core.hyperparameter_chromosome.random.uniform", return_value=0.005):
            mutated = mutate_chromosome(
                chromosome,
                mutation_rate=1.0,
                mutation_scale=1.0,
                boundary_mode=BoundaryMode.INTERIOR_BIASED,
                interior_bias_fraction=bias_fraction,
            )
        lr = mutated.get_value("learning_rate")
        # nudge = 0.005 → result = 1.0 - 0.005 = 0.995
        self.assertAlmostEqual(lr, 1.0 - 0.005)
        self.assertLessEqual(lr, 1.0)
        span = 1.0 - 1e-6  # gene span
        # Value should be within bias_fraction * span of max
        self.assertGreaterEqual(lr, 1.0 - bias_fraction * span)


if __name__ == "__main__":
    unittest.main()

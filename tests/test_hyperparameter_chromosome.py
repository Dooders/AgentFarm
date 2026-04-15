"""Tests for typed hyperparameter chromosome schema."""

import unittest
from unittest.mock import patch

from farm.core.hyperparameter_chromosome import (
    apply_chromosome_to_learning_config,
    GeneValueType,
    HyperparameterChromosome,
    HyperparameterGene,
    chromosome_from_learning_config,
    chromosome_from_values,
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

    def test_builds_from_learning_config_like_object(self):
        chromosome = chromosome_from_learning_config(_LearningConfigStub())
        self.assertEqual(chromosome.get_value("learning_rate"), 0.02)
        self.assertEqual(chromosome.get_value("epsilon_decay"), 0.97)
        self.assertEqual(chromosome.get_value("memory_size"), 4096.0)

    def test_accepts_low_valid_epsilon_decay_from_learning_config(self):
        chromosome = chromosome_from_learning_config(_LowEpsilonLearningConfigStub())
        self.assertEqual(chromosome.get_value("epsilon_decay"), 0.1)

    def test_accepts_large_memory_size_override(self):
        chromosome = chromosome_from_values({"memory_size": 250_000.0})
        self.assertEqual(chromosome.get_value("memory_size"), 250_000.0)

    def test_mutation_only_changes_evolvable_genes(self):
        chromosome = default_hyperparameter_chromosome()
        with patch("farm.core.hyperparameter_chromosome.random.uniform", return_value=0.1):
            mutated = mutate_chromosome(chromosome, mutation_rate=1.0, mutation_scale=0.1)
        self.assertNotEqual(mutated.get_value("learning_rate"), chromosome.get_value("learning_rate"))
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


if __name__ == "__main__":
    unittest.main()

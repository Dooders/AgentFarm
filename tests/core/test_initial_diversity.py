"""Tests for the platform-wide initial genotype diversity seeding feature.

Mirrors the structure of ``tests/runners/test_intrinsic_evolution_experiment.py``
but exercises the new shared module rather than the now-deleted
``seed_population_diversity`` helper.
"""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from farm.core.hyperparameter_chromosome import (
    BoundaryMode,
    MutationMode,
    chromosome_from_learning_config,
)
from farm.core.initial_diversity import (
    ChromosomeDiversitySource,
    InitialDiversityConfig,
    InitialDiversityMetrics,
    SeedingMode,
    apply_initial_diversity,
    persist_initial_diversity_metrics,
)


def _make_fake_agent(learning_rate: float = 0.01, agent_id: str = "a"):
    """Lightweight stand-in carrying the attributes the seeder reads/writes."""
    config = SimpleNamespace(decision=SimpleNamespace(learning_rate=learning_rate))
    chromosome = chromosome_from_learning_config(config.decision)
    return SimpleNamespace(
        agent_id=agent_id,
        agent_type="system",
        alive=True,
        config=config,
        hyperparameter_chromosome=chromosome,
    )


class _FakeEnvironment:
    """Minimal environment compatible with the seeder iteration contract."""

    def __init__(self, agents):
        self._agents = list(agents)

    @property
    def agent_objects(self):
        return list(self._agents)

    @property
    def alive_agent_objects(self):
        return [a for a in self._agents if a.alive]


class TestInitialDiversityConfig(unittest.TestCase):
    def test_defaults_are_off(self):
        cfg = InitialDiversityConfig()
        self.assertEqual(cfg.mode, SeedingMode.NONE)
        self.assertEqual(cfg.mutation_mode, MutationMode.GAUSSIAN)
        self.assertEqual(cfg.boundary_mode, BoundaryMode.CLAMP)

    def test_rejects_invalid_mutation_rate(self):
        with self.assertRaises(ValueError):
            InitialDiversityConfig(mutation_rate=1.5)

    def test_rejects_negative_mutation_scale(self):
        with self.assertRaises(ValueError):
            InitialDiversityConfig(mutation_scale=-0.1)

    def test_rejects_zero_max_retries(self):
        with self.assertRaises(ValueError):
            InitialDiversityConfig(max_retries_per_agent=0)

    def test_rejects_negative_min_distance(self):
        with self.assertRaises(ValueError):
            InitialDiversityConfig(min_distance=-0.01)

    def test_string_enums_coerced(self):
        cfg = InitialDiversityConfig(
            mode="independent_mutation",  # type: ignore[arg-type]
            mutation_mode="multiplicative",  # type: ignore[arg-type]
            boundary_mode="reflect",  # type: ignore[arg-type]
        )
        self.assertIs(cfg.mode, SeedingMode.INDEPENDENT_MUTATION)
        self.assertIs(cfg.mutation_mode, MutationMode.MULTIPLICATIVE)
        self.assertIs(cfg.boundary_mode, BoundaryMode.REFLECT)

    def test_invalid_string_enum_raises(self):
        with self.assertRaises(ValueError):
            InitialDiversityConfig(mode="bogus")  # type: ignore[arg-type]

    def test_to_dict_serializes_enums_as_strings(self):
        cfg = InitialDiversityConfig(mode=SeedingMode.UNIQUE)
        raw = cfg.to_dict()
        self.assertEqual(raw["mode"], "unique")
        self.assertEqual(raw["mutation_mode"], "gaussian")
        self.assertEqual(raw["boundary_mode"], "clamp")


class TestApplyInitialDiversityNoneMode(unittest.TestCase):
    def test_none_mode_does_not_touch_agents_or_call_source(self):
        agents = [_make_fake_agent(0.01, agent_id=f"a{i}") for i in range(3)]
        env = _FakeEnvironment(agents)
        original_lrs = [
            a.hyperparameter_chromosome.get_value("learning_rate") for a in agents
        ]
        sentinel_source = MagicMock()
        metrics = apply_initial_diversity(
            env, InitialDiversityConfig(mode=SeedingMode.NONE), random.Random(0),
            source=sentinel_source,
        )
        new_lrs = [
            a.hyperparameter_chromosome.get_value("learning_rate") for a in agents
        ]
        self.assertEqual(original_lrs, new_lrs)
        self.assertEqual(metrics.mode, SeedingMode.NONE)
        self.assertEqual(metrics.agents_processed, 0)
        sentinel_source.seed.assert_not_called()


class TestChromosomeDiversitySourceIndependentMutation(unittest.TestCase):
    def test_mutates_each_agent(self):
        agents = [_make_fake_agent(0.01, agent_id=f"a{i}") for i in range(5)]
        env = _FakeEnvironment(agents)
        cfg = InitialDiversityConfig(
            mode=SeedingMode.INDEPENDENT_MUTATION,
            mutation_rate=1.0,
            mutation_scale=0.3,
        )
        metrics = apply_initial_diversity(env, cfg, random.Random(0))
        new_lrs = [
            a.hyperparameter_chromosome.get_value("learning_rate") for a in agents
        ]
        self.assertTrue(any(lr != 0.01 for lr in new_lrs))
        self.assertEqual(metrics.agents_processed, 5)
        self.assertEqual(metrics.retries_used, 0)
        self.assertEqual(metrics.fallbacks, 0)
        # Decision config tracks the chromosome in lock-step.
        for agent, lr in zip(agents, new_lrs):
            self.assertEqual(agent.config.decision.learning_rate, lr)

    def test_decision_module_reinitialized_when_present(self):
        dm = MagicMock()
        behavior = SimpleNamespace(decision_module=dm)
        config = SimpleNamespace(decision=SimpleNamespace(learning_rate=0.01))
        chromosome = chromosome_from_learning_config(config.decision)
        agent = SimpleNamespace(
            agent_id="agent_with_module",
            agent_type="system",
            alive=True,
            config=config,
            hyperparameter_chromosome=chromosome,
            behavior=behavior,
        )
        env = _FakeEnvironment([agent])
        cfg = InitialDiversityConfig(
            mode=SeedingMode.INDEPENDENT_MUTATION,
            mutation_rate=1.0,
            mutation_scale=0.5,
        )
        apply_initial_diversity(env, cfg, random.Random(42))
        dm.reinitialize_algorithm.assert_called_once_with(agent.config.decision)

    def test_skips_agents_without_chromosome(self):
        agent_with = _make_fake_agent(0.02, agent_id="ok")
        agent_without = SimpleNamespace(
            agent_id="x",
            agent_type="system",
            alive=True,
            config=SimpleNamespace(decision=SimpleNamespace(learning_rate=0.02)),
            hyperparameter_chromosome=None,
        )
        env = _FakeEnvironment([agent_with, agent_without])
        cfg = InitialDiversityConfig(
            mode=SeedingMode.INDEPENDENT_MUTATION,
            mutation_rate=1.0,
            mutation_scale=0.2,
        )
        metrics = apply_initial_diversity(env, cfg, random.Random(0))
        self.assertEqual(metrics.agents_processed, 1)


class TestChromosomeDiversitySourceUnique(unittest.TestCase):
    def test_unique_mode_produces_distinct_signatures(self):
        agents = [_make_fake_agent(0.01, agent_id=f"a{i}") for i in range(8)]
        env = _FakeEnvironment(agents)
        cfg = InitialDiversityConfig(
            mode=SeedingMode.UNIQUE,
            mutation_rate=1.0,
            mutation_scale=0.4,
            max_retries_per_agent=64,
        )
        metrics = apply_initial_diversity(env, cfg, random.Random(0))
        # All accepted chromosomes should be distinct.
        self.assertEqual(metrics.unique_count, 8)
        # No fallbacks expected for this generous configuration.
        self.assertEqual(metrics.fallbacks, 0)
        self.assertEqual(metrics.collision_count, 0)

    def test_unique_mode_falls_back_under_tight_budget(self):
        # mutation_scale=0 + UNIQUE => every draw is identical, triggering fallback.
        agents = [_make_fake_agent(0.01, agent_id=f"a{i}") for i in range(4)]
        env = _FakeEnvironment(agents)
        cfg = InitialDiversityConfig(
            mode=SeedingMode.UNIQUE,
            mutation_rate=1.0,
            mutation_scale=0.0,
            max_retries_per_agent=2,
        )
        metrics = apply_initial_diversity(env, cfg, random.Random(0))
        # Agent 0 always succeeds, agents 1-3 must fall back.
        self.assertEqual(metrics.fallbacks, 3)
        self.assertEqual(metrics.agents_processed, 4)
        # retries_used = (max_retries - 1) per fallback.
        self.assertEqual(metrics.retries_used, 3 * (2 - 1))


class TestChromosomeDiversitySourceMinDistance(unittest.TestCase):
    def test_min_distance_satisfied_for_loose_constraint(self):
        agents = [_make_fake_agent(0.01, agent_id=f"a{i}") for i in range(6)]
        env = _FakeEnvironment(agents)
        cfg = InitialDiversityConfig(
            mode=SeedingMode.MIN_DISTANCE,
            mutation_rate=1.0,
            mutation_scale=0.4,
            min_distance=0.01,
            max_retries_per_agent=64,
        )
        metrics = apply_initial_diversity(env, cfg, random.Random(0))
        self.assertEqual(metrics.fallbacks, 0)
        self.assertIsNotNone(metrics.min_pairwise_distance)
        assert metrics.min_pairwise_distance is not None  # for type narrowing
        self.assertGreaterEqual(metrics.min_pairwise_distance, 0.01)

    def test_min_distance_falls_back_for_unsatisfiable_constraint(self):
        agents = [_make_fake_agent(0.01, agent_id=f"a{i}") for i in range(5)]
        env = _FakeEnvironment(agents)
        # min_distance >> achievable in unit hypercube => guaranteed fallback after 1st.
        cfg = InitialDiversityConfig(
            mode=SeedingMode.MIN_DISTANCE,
            mutation_rate=1.0,
            mutation_scale=0.1,
            min_distance=10.0,
            max_retries_per_agent=3,
        )
        metrics = apply_initial_diversity(env, cfg, random.Random(0))
        self.assertEqual(metrics.fallbacks, 4)
        self.assertEqual(metrics.agents_processed, 5)


class TestDeterminism(unittest.TestCase):
    def _seed_run(self, mode: SeedingMode, seed: int):
        agents = [_make_fake_agent(0.01, agent_id=f"a{i}") for i in range(6)]
        env = _FakeEnvironment(agents)
        cfg = InitialDiversityConfig(
            mode=mode,
            mutation_rate=1.0,
            mutation_scale=0.3,
            min_distance=0.02,
            max_retries_per_agent=32,
        )
        metrics = apply_initial_diversity(env, cfg, random.Random(seed))
        chromosome_values = [
            tuple((g.name, g.value) for g in a.hyperparameter_chromosome.genes)
            for a in agents
        ]
        return chromosome_values, metrics

    def test_independent_mutation_is_deterministic(self):
        a, m_a = self._seed_run(SeedingMode.INDEPENDENT_MUTATION, seed=1234)
        b, m_b = self._seed_run(SeedingMode.INDEPENDENT_MUTATION, seed=1234)
        self.assertEqual(a, b)
        self.assertEqual(m_a.to_dict(), m_b.to_dict())

    def test_unique_mode_is_deterministic(self):
        a, m_a = self._seed_run(SeedingMode.UNIQUE, seed=1234)
        b, m_b = self._seed_run(SeedingMode.UNIQUE, seed=1234)
        self.assertEqual(a, b)
        self.assertEqual(m_a.to_dict(), m_b.to_dict())

    def test_min_distance_is_deterministic(self):
        a, m_a = self._seed_run(SeedingMode.MIN_DISTANCE, seed=1234)
        b, m_b = self._seed_run(SeedingMode.MIN_DISTANCE, seed=1234)
        self.assertEqual(a, b)
        self.assertEqual(m_a.to_dict(), m_b.to_dict())


class TestInitialDiversityMetricsSerialization(unittest.TestCase):
    def test_to_dict_round_trips_via_persistence_helper(self):
        import json
        import os
        import tempfile

        metrics = InitialDiversityMetrics(
            mode=SeedingMode.UNIQUE,
            agents_processed=4,
            unique_count=4,
            collision_count=0,
            retries_used=7,
            fallbacks=0,
            min_pairwise_distance=0.12,
            mean_pairwise_distance=0.34,
            notes=["ok"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = persist_initial_diversity_metrics(tmp, metrics)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        self.assertEqual(payload["mode"], "unique")
        self.assertEqual(payload["agents_processed"], 4)
        self.assertEqual(payload["unique_count"], 4)
        self.assertEqual(payload["retries_used"], 7)
        self.assertEqual(payload["min_pairwise_distance"], 0.12)


class TestChromosomeDiversitySourceCustomEncodings(unittest.TestCase):
    def test_source_accepts_custom_encoding_specs(self):
        # Smoke test: providing encoding specs should not raise and should
        # still produce metrics with the expected mode.  Behaviour parity with
        # the default encoding is asserted indirectly through other tests.
        from farm.core.hyperparameter_chromosome import (
            GeneEncodingScale,
            GeneEncodingSpec,
        )

        agents = [_make_fake_agent(0.01, agent_id=f"a{i}") for i in range(3)]
        env = _FakeEnvironment(agents)
        source = ChromosomeDiversitySource(
            encoding_specs={
                "learning_rate": GeneEncodingSpec(
                    scale=GeneEncodingScale.LOG, bit_width=6
                )
            }
        )
        cfg = InitialDiversityConfig(
            mode=SeedingMode.UNIQUE,
            mutation_rate=1.0,
            mutation_scale=0.3,
        )
        metrics = source.seed(env, cfg, random.Random(0))
        self.assertEqual(metrics.mode, SeedingMode.UNIQUE)
        self.assertEqual(metrics.agents_processed, 3)


if __name__ == "__main__":
    unittest.main()

"""Tests for the intrinsic-goals experiment runner."""

from __future__ import annotations

import json
import os
import random
from types import SimpleNamespace

import pytest

from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import (
    INTRINSIC_REWARD_GENE_NAMES,
    default_hyperparameter_chromosome,
)
from farm.runners.intrinsic_goals_experiment import (
    IntrinsicGoalsExperiment,
    IntrinsicGoalsExperimentConfig,
    assign_unique_goals,
    sample_goal_chromosome,
)


def test_sample_goal_chromosome_randomizes_only_goal_genes():
    base = default_hyperparameter_chromosome()
    rng = random.Random(0)
    sampled = sample_goal_chromosome(base, rng)

    # Non-goal genes untouched.
    assert sampled.get_value("learning_rate") == base.get_value("learning_rate")
    assert sampled.get_value("move_weight") == base.get_value("move_weight")

    # At least one goal gene changed and all stay within bounds.
    changed = False
    for name in INTRINSIC_REWARD_GENE_NAMES:
        gene = sampled.get_gene(name)
        assert gene.min_value <= gene.value <= gene.max_value
        if gene.value != base.get_value(name):
            changed = True
    assert changed


def test_sample_goal_chromosome_is_deterministic_for_seed():
    base = default_hyperparameter_chromosome()
    a = sample_goal_chromosome(base, random.Random(123))
    b = sample_goal_chromosome(base, random.Random(123))
    for name in INTRINSIC_REWARD_GENE_NAMES:
        assert a.get_value(name) == b.get_value(name)


def test_assign_unique_goals_gives_each_agent_a_distinct_goal():
    agents = [
        SimpleNamespace(
            agent_id=f"a{i}",
            hyperparameter_chromosome=default_hyperparameter_chromosome(),
        )
        for i in range(8)
    ]
    env = SimpleNamespace(alive_agent_objects=agents)

    n = assign_unique_goals(env, random.Random(7))
    assert n == 8

    # Collect each agent's share-bonus gene; with random sampling across [0,2]
    # they should not all be identical.
    share_values = {
        a.hyperparameter_chromosome.get_value("reward_share_bonus") for a in agents
    }
    assert len(share_values) > 1


@pytest.mark.integration
def test_short_experiment_runs_and_writes_artifacts(tmp_path):
    base_config = SimulationConfig.from_centralized_config(environment="development")
    # Keep the smoke run tiny but non-trivial.
    base_config.population.system_agents = 4
    base_config.population.independent_agents = 4
    base_config.population.control_agents = 0

    config = IntrinsicGoalsExperimentConfig(
        num_steps=12,
        seed=5,
        output_dir=str(tmp_path / "goals"),
        selection_pressure="none",
    )
    result = IntrinsicGoalsExperiment(base_config, config).run()

    assert os.path.exists(result.summary_path)
    with open(result.summary_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "uniform" in payload and "unique" in payload
    assert "comparison" in payload

    # The unique arm should start with strictly more goal diversity than the
    # uniform arm (which starts as a goal monoculture).
    start_div_unique = sum(
        result.unique.goal_diversity_start.get(g, 0.0)
        for g in INTRINSIC_REWARD_GENE_NAMES
    )
    start_div_uniform = sum(
        result.uniform.goal_diversity_start.get(g, 0.0)
        for g in INTRINSIC_REWARD_GENE_NAMES
    )
    assert start_div_uniform == pytest.approx(0.0)
    assert start_div_unique > 0.0

    # Per-step telemetry was recorded.
    assert len(result.uniform.population) > 0
    assert len(result.unique.population) > 0

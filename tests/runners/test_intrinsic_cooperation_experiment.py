"""Orchestration tests for the intrinsic cooperation runner (``run_simulation`` patched)."""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("structlog")
pytest.importorskip("torch")

from farm.config import SimulationConfig
from farm.core.cooperation import CooperationConfig
from farm.core.cooperation.genes import extend_chromosome
from farm.core.hyperparameter_chromosome import default_hyperparameter_chromosome
from farm.runners.intrinsic_cooperation_experiment import (
    ARM_NAMES,
    HYPOTHESES,
    MANIPULATION_CHECKS,
    IntrinsicCooperationExperiment,
    IntrinsicCooperationExperimentConfig,
    holm,
)

pytestmark = pytest.mark.unit


class _FakeEnv:
    def __init__(self, config, seed, arm_dir):
        self.config = config
        self.seed = seed
        self.time = 0
        coop = config.cooperation
        self.cooperation = None
        self._agents = []
        for i in range(6):
            chromosome = default_hyperparameter_chromosome()
            if coop.enabled:
                chromosome = extend_chromosome(chromosome, {"coop_exit_cost": 0.1 * i})
            self._agents.append(
                SimpleNamespace(
                    agent_id=f"a{i}", alive=True, hyperparameter_chromosome=chromosome, last_action_name="gather"
                )
            )
        if coop.enabled:
            from farm.core.cooperation import CooperationService

            self.cooperation = CooperationService(coop, environment=self, seed=seed)

    @property
    def alive_agent_objects(self):
        return [a for a in self._agents if a.alive]


def _fake_run_simulation(
    num_steps, config, path, save_config, seed, disable_console_logging, on_environment_ready, on_step_end, **_
):
    env = _FakeEnv(config, seed, path)
    on_environment_ready(env)
    for step in range(num_steps):
        env.time = step
        if step == 2:
            env._agents[0].alive = False
        env._agents[1].last_action_name = "bond" if env.cooperation is not None else "move"
        if env.cooperation is not None:
            env.cooperation.settle()
        on_step_end(env, step)
    return env


def test_config_validation():
    with pytest.raises(ValueError):
        IntrinsicCooperationExperimentConfig(arms=("sideline",))
    with pytest.raises(ValueError):
        IntrinsicCooperationExperimentConfig(cooperation=CooperationConfig(enabled=False))
    cfg = IntrinsicCooperationExperimentConfig(mutation_mode="gaussian", boundary_mode="reflect")
    assert cfg.arms == ARM_NAMES


def test_arms_get_the_right_cooperation_config():
    cfg = IntrinsicCooperationExperimentConfig(cooperation=CooperationConfig(enabled=True, bond_range=3.0))
    exp = IntrinsicCooperationExperiment(SimulationConfig.from_centralized_config(environment="testing"), cfg)
    base = exp._build_run_config("baseline", 5, "/tmp/x")
    placebo = exp._build_run_config("placebo", 5, "/tmp/x")
    coop = exp._build_run_config("cooperation", 5, "/tmp/x")
    assert not base.cooperation.enabled
    assert placebo.cooperation.enabled and placebo.cooperation.mode == "placebo"
    assert coop.cooperation.active and coop.cooperation.bond_range == 3.0
    no_motive = exp._build_run_config("cooperation_no_motive", 5, "/tmp/x")
    assert no_motive.cooperation.active and no_motive.cooperation.membership_reward_scale == 0.0
    assert coop.cooperation.membership_reward_scale == 1.0
    assert coop.cooperation.seed == 5 and coop.cooperation.output_dir == "/tmp/x"
    assert exp._make_policy(5).crossover_enabled


def test_run_records_arms_and_writes_summary():
    base = SimulationConfig.from_centralized_config(environment="testing")
    with tempfile.TemporaryDirectory() as out:
        cfg = IntrinsicCooperationExperimentConfig(num_steps=5, num_replicates=2, output_dir=out, seed=3)
        with patch("farm.runners.intrinsic_cooperation_experiment.run_simulation", _fake_run_simulation):
            result = IntrinsicCooperationExperiment(base, cfg).run()
        assert set(result.arms) == set(ARM_NAMES)
        coop = result.arms["cooperation"]
        assert coop.population == [6, 6, 5, 5, 5]
        assert coop.deaths[2] == 1
        assert coop.action_mix["bond"][-1] == pytest.approx(1 / 5)
        assert result.arms["baseline"].action_mix["bond"] == [0.0] * 5
        assert coop.lifespans == [2]
        assert np.isnan(coop.policy_bond_prob[0])  # fake agents carry no decision module
        assert not np.isnan(coop.expected_exit_payment[0])
        assert np.isnan(result.arms["baseline"].expected_exit_payment[0])
        no_motive = result.arms["cooperation_no_motive"].cooperation_summary["config"]
        assert no_motive["membership_reward_scale"] == 0.0
        assert not np.isnan(coop.gene_means["coop_exit_cost"][0])
        assert np.isnan(result.arms["baseline"].gene_means["coop_exit_cost"][0])
        with open(result.summary_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert len(payload["replicates"]) == 2
        assert payload["aggregate"]["num_replicates"] == 2
        keys = {h["key"] for h in payload["hypotheses"]}
        assert keys == set(HYPOTHESES) | set(MANIPULATION_CHECKS)
        family = [h for h in payload["hypotheses"] if not h.get("manipulation_check")]
        assert {h["key"] for h in family} == set(HYPOTHESES) and all(h["p_holm"] is not None for h in family)
        assert {h["key"] for h in payload["hypotheses"] if h.get("manipulation_check")} == set(MANIPULATION_CHECKS)
        assert os.path.exists(os.path.join(out, "rep_00", "cooperation", "cooperation_summary.json"))


def test_holm_step_down():
    results = [{"p_value": p} for p in (0.001, 0.04, 0.03)]
    holm(results, 0.05)
    assert [r["significant"] for r in results] == [True, False, False]
    assert results[0]["p_holm"] == pytest.approx(0.003)

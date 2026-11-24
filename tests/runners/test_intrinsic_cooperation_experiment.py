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
from farm.core.cooperation.genes import COOPERATION_GENE_NAMES, extend_chromosome
from farm.core.hyperparameter_chromosome import (
    ACTION_PRIOR_GENE_NAMES,
    default_hyperparameter_chromosome,
)
from farm.core.initial_diversity import SeedingMode
from farm.runners.intrinsic_cooperation_experiment import (
    ARM_NAMES,
    ARM_RESULT_FILE,
    DEFAULT_ARMS,
    FROZEN_PLATFORM_GENES,
    HYPOTHESES,
    LEARNING_CHECKS,
    REPLICATE_SUMMARY_FILE,
    TRACKED_ACTIONS,
    ArmResult,
    IntrinsicCooperationExperiment,
    IntrinsicCooperationExperimentConfig,
    expected_test_keys,
    holm,
    hypothesis_kind,
)

pytestmark = pytest.mark.unit

_ACTION_NAMES = ("move", "gather", "share", "attack", "defend", "reproduce", "pass")


def _fake_agent(i: int, chromosome, with_coop_actions: bool) -> SimpleNamespace:
    names = _ACTION_NAMES + (("bond", "leave") if with_coop_actions else ())
    actions = [SimpleNamespace(name=n, weight=1.0) for n in names]
    # Policy: a mild preference for gather; bond gets the second-highest mass.
    policy = np.full(len(names), 1.0)
    policy[names.index("gather")] = 3.0
    if with_coop_actions:
        policy[names.index("bond")] = 2.0
    policy /= policy.sum()
    module = SimpleNamespace(last_policy_probs=policy, last_action_probs=policy.copy())
    return SimpleNamespace(
        agent_id=f"a{i}",
        alive=True,
        hyperparameter_chromosome=chromosome,
        last_action_name="gather",
        actions=actions,
        behavior=SimpleNamespace(decision_module=module),
        death_cause=None,
        generation=0,
    )


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
            self._agents.append(_fake_agent(i, chromosome, with_coop_actions=coop.enabled))
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
            env._agents[0].death_cause = "starvation"
        if step == 3:
            env._agents[5].alive = False  # no cause recorded -> "unknown"
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
    with pytest.raises(ValueError):
        IntrinsicCooperationExperimentConfig(policy_temperature=0.0)
    with pytest.raises(ValueError):
        IntrinsicCooperationExperimentConfig(initial_diversity_scale=-0.1)
    cfg = IntrinsicCooperationExperimentConfig(mutation_mode="gaussian", boundary_mode="reflect")
    assert cfg.arms == DEFAULT_ARMS
    assert "cooperation_zero_synergy" in ARM_NAMES and "cooperation_zero_synergy" not in DEFAULT_ARMS
    assert cfg.cooperation.exit_cost_base == "received"
    assert cfg.selection_pressure == "medium"
    assert cfg.initial_diversity_scale is None and cfg.freeze_platform_genes and cfg.resume
    assert "communicate" in TRACKED_ACTIONS
    assert cfg.to_dict()["policy_temperature"] == pytest.approx(0.1)
    assert cfg.jobs == 1
    with pytest.raises(ValueError):
        IntrinsicCooperationExperimentConfig(jobs=0)


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
    zero_synergy = exp._build_run_config("cooperation_zero_synergy", 5, "/tmp/x")
    assert zero_synergy.cooperation.active and zero_synergy.cooperation.synergy == 0.0
    assert coop.cooperation.synergy == pytest.approx(0.2)
    assert zero_synergy.cooperation.membership_reward_scale == 1.0
    assert coop.cooperation.seed == 5 and coop.cooperation.output_dir == "/tmp/x"
    assert exp._make_policy(5).crossover_enabled


def test_learning_regime_is_applied_to_every_arm():
    base = SimulationConfig.from_centralized_config(environment="testing")
    exp = IntrinsicCooperationExperiment(base, IntrinsicCooperationExperimentConfig(policy_temperature=0.25))
    for arm in ARM_NAMES:
        run_config = exp._build_run_config(arm, 5, "/tmp/x")
        assert run_config.initial_diversity.mode == SeedingMode.NONE
        assert run_config.learning.policy_temperature == pytest.approx(0.25)

    diverse = IntrinsicCooperationExperiment(base, IntrinsicCooperationExperimentConfig(initial_diversity_scale=0.2))
    run_config = diverse._build_run_config("cooperation", 5, "/tmp/x")
    assert run_config.initial_diversity.mode == SeedingMode.INDEPENDENT_MUTATION
    assert run_config.initial_diversity.mutation_scale == pytest.approx(0.2)
    assert run_config.initial_diversity.seed == 5


def test_reproduction_economics_reach_every_arm():
    base = SimulationConfig.from_centralized_config(environment="testing")
    untouched = IntrinsicCooperationExperiment(base, IntrinsicCooperationExperimentConfig())
    run_config = untouched._build_run_config("baseline", 5, "/tmp/x")
    assert run_config.agent_behavior.offspring_initial_resources == base.agent_behavior.offspring_initial_resources
    assert run_config.agent_behavior.min_reproduction_resources == base.agent_behavior.min_reproduction_resources

    cfg = IntrinsicCooperationExperimentConfig(offspring_initial_resources=8.0, min_reproduction_resources=5.0)
    exp = IntrinsicCooperationExperiment(base, cfg)
    for arm in ARM_NAMES:
        run_config = exp._build_run_config(arm, 5, "/tmp/x")
        assert run_config.agent_behavior.offspring_initial_resources == 8.0
        assert run_config.agent_behavior.min_reproduction_resources == 5.0
    assert cfg.to_dict()["offspring_initial_resources"] == 8.0


def test_freeze_platform_genes_pins_chromosomes_a_and_c_only():
    chromosome = extend_chromosome(default_hyperparameter_chromosome())
    agent = SimpleNamespace(hyperparameter_chromosome=chromosome)
    IntrinsicCooperationExperiment._freeze_platform_genes(agent)
    frozen = agent.hyperparameter_chromosome
    assert frozen is not chromosome
    for gene in frozen.genes:
        if gene.name in FROZEN_PLATFORM_GENES:
            assert not gene.evolvable, gene.name
        else:
            assert gene.evolvable, gene.name
    evolving = {g.name for g in frozen.genes if g.evolvable}
    assert evolving == set(ACTION_PRIOR_GENE_NAMES) | set(COOPERATION_GENE_NAMES)
    assert {"dqn_hidden_size", "memory_size", "learning_rate", "reward_resource_weight"} <= FROZEN_PLATFORM_GENES
    # Agents without a chromosome are left alone.
    IntrinsicCooperationExperiment._freeze_platform_genes(SimpleNamespace())


def test_run_records_arms_and_writes_summary():
    base = SimulationConfig.from_centralized_config(environment="testing")
    with tempfile.TemporaryDirectory() as out:
        cfg = IntrinsicCooperationExperimentConfig(num_steps=5, num_replicates=2, output_dir=out, seed=3)
        with patch("farm.runners.intrinsic_cooperation_experiment.run_simulation", _fake_run_simulation):
            result = IntrinsicCooperationExperiment(base, cfg).run()
        assert set(result.arms) == set(DEFAULT_ARMS)
        coop = result.arms["cooperation"]
        assert coop.population == [6, 6, 5, 4, 4]
        assert coop.deaths[2] == 1 and coop.deaths[3] == 1
        assert coop.action_mix["bond"][-1] == pytest.approx(1 / 4)
        assert result.arms["baseline"].action_mix["bond"] == [0.0] * 5
        assert coop.lifespans == [2, 3]
        assert coop.death_causes == {"starvation": 1, "unknown": 1}
        assert not np.isnan(coop.expected_exit_payment[0])
        assert np.isnan(result.arms["baseline"].expected_exit_payment[0])
        no_motive = result.arms["cooperation_no_motive"].cooperation_summary["config"]
        assert no_motive["membership_reward_scale"] == 0.0
        assert not np.isnan(coop.gene_means["coop_exit_cost"][0])
        assert np.isnan(result.arms["baseline"].gene_means["coop_exit_cost"][0])

        # Policy telemetry from the fake decision modules.
        baseline = result.arms["baseline"]
        assert baseline.num_actions == 7 and coop.num_actions == 9
        assert np.isnan(baseline.policy_bond_prob[0]) and np.isnan(baseline.acting_bond_prob[0])
        assert coop.policy_bond_prob[0] == pytest.approx(2 / 12)
        assert coop.acting_bond_prob[0] == pytest.approx(2 / 12)
        assert coop.bond_rank[0] == pytest.approx(2.0)
        assert baseline.policy_gather_prob[0] == pytest.approx(3 / 9)
        assert baseline.prior_gather_share[0] == pytest.approx(1 / 7)
        assert baseline.policy_peak[0] == pytest.approx(3 / 9)
        summary = baseline.summary()
        assert summary["policy_gather_lift_late"] == pytest.approx(3 / 9 - 1 / 7)
        assert summary["gather_share_over_prior_late"] == pytest.approx(3 / 4 - 1 / 7)  # agent 1 moves
        assert summary["starvation_deaths"] == 1 and summary["combat_deaths"] == 0

        with open(result.summary_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert len(payload["replicates"]) == 2
        assert payload["aggregate"]["num_replicates"] == 2
        assert "git_commit" in payload["config"] and payload["config"]["arms"] == list(DEFAULT_ARMS)
        assert payload["config"]["jobs"] == 1
        keys = {h["key"] for h in payload["hypotheses"]}
        assert keys == expected_test_keys(DEFAULT_ARMS)
        family = [h for h in payload["hypotheses"] if hypothesis_kind(h) == "family"]
        assert {h["key"] for h in family} == set(HYPOTHESES) and all(h["p_holm"] is not None for h in family)
        learning = {h["key"]: h for h in payload["hypotheses"] if hypothesis_kind(h) == "learning_check"}
        assert set(learning) == set(LEARNING_CHECKS) and learning["L1"]["contrast"] == "baseline vs 0"
        assert all(h["p_holm"] is None for h in payload["hypotheses"] if hypothesis_kind(h) != "family")
        assert os.path.exists(os.path.join(out, "rep_00", "cooperation", "cooperation_summary.json"))
        assert os.path.exists(os.path.join(out, "rep_00", "cooperation", ARM_RESULT_FILE))
        with open(os.path.join(out, "rep_01", REPLICATE_SUMMARY_FILE), encoding="utf-8") as handle:
            rep = json.load(handle)
        assert rep["seed"] == 4 and rep["cooperation"] == payload["replicates"][1]["cooperation"]


def test_zero_synergy_arm_adds_its_exploratory_contrasts():
    base = SimulationConfig.from_centralized_config(environment="testing")
    arms = ("placebo", "cooperation", "cooperation_zero_synergy")
    with tempfile.TemporaryDirectory() as out:
        cfg = IntrinsicCooperationExperimentConfig(num_steps=4, num_replicates=2, output_dir=out, arms=arms)
        with patch("farm.runners.intrinsic_cooperation_experiment.run_simulation", _fake_run_simulation):
            result = IntrinsicCooperationExperiment(base, cfg).run()
        keys = {h["key"] for h in result.hypotheses}
        assert keys == expected_test_keys(arms)
        assert {"X_zero_synergy_population", "X_synergy_share"} <= keys
        assert "M2" not in keys and "L1" not in keys  # baseline not run
        assert result.arms["cooperation_zero_synergy"].cooperation_summary["config"]["synergy"] == 0.0


def test_arm_result_round_trips_through_json():
    result = ArmResult(arm="cooperation", steps=[0, 1], population=[5, 4], death_causes={"starvation": 1})
    result.action_mix = {"gather": [0.5, 0.25]}
    result.gene_means = {"coop_scope": [0.5, 0.6]}
    result.policy_bond_prob = [0.1, float("nan")]
    result.lifespans = [3]
    result.num_actions = 9
    restored = ArmResult.from_dict(json.loads(json.dumps(result.to_dict(), default=str)))
    assert restored.steps == [0, 1] and restored.population == [5, 4]
    assert restored.action_mix == result.action_mix and restored.gene_means == result.gene_means
    assert restored.policy_bond_prob[0] == pytest.approx(0.1) and np.isnan(restored.policy_bond_prob[1])
    assert restored.death_causes == {"starvation": 1} and restored.num_actions == 9
    original, again = result.summary(), restored.summary()
    assert again["mean_population"] == original["mean_population"]
    assert again["mean_action_share"] == original["mean_action_share"]
    assert again["gene_mean_change"] == original["gene_mean_change"]
    assert again["mean_lifespan_at_death"] == original["mean_lifespan_at_death"]


def test_resume_loads_finished_arms_instead_of_rerunning():
    base = SimulationConfig.from_centralized_config(environment="testing")
    with tempfile.TemporaryDirectory() as out:
        cfg = IntrinsicCooperationExperimentConfig(num_steps=4, num_replicates=1, output_dir=out, arms=("baseline",))
        with patch("farm.runners.intrinsic_cooperation_experiment.run_simulation", _fake_run_simulation):
            IntrinsicCooperationExperiment(base, cfg).run()
        path = os.path.join(out, "baseline", ARM_RESULT_FILE)
        assert os.path.exists(path)

        calls: list[str] = []

        def _counting(*args, **kwargs):
            calls.append("run")
            return _fake_run_simulation(*args, **kwargs)

        with patch("farm.runners.intrinsic_cooperation_experiment.run_simulation", _counting):
            resumed = IntrinsicCooperationExperiment(base, cfg).run()
        assert calls == []
        assert resumed.arms["baseline"].population == [6, 6, 5, 4]

        fresh_cfg = IntrinsicCooperationExperimentConfig(
            num_steps=4, num_replicates=1, output_dir=out, arms=("baseline",), resume=False
        )
        with patch("farm.runners.intrinsic_cooperation_experiment.run_simulation", _counting):
            IntrinsicCooperationExperiment(base, fresh_cfg).run()
        assert calls == ["run"]


def test_resume_reruns_arms_made_under_another_configuration():
    base = SimulationConfig.from_centralized_config(environment="testing")
    calls: list[str] = []

    def _counting(*args, **kwargs):
        calls.append("run")
        return _fake_run_simulation(*args, **kwargs)

    with tempfile.TemporaryDirectory() as out:

        def cfg(**overrides):
            return IntrinsicCooperationExperimentConfig(
                num_steps=4, num_replicates=1, output_dir=out, arms=("baseline",), **overrides
            )

        with patch("farm.runners.intrinsic_cooperation_experiment.run_simulation", _counting):
            IntrinsicCooperationExperiment(base, cfg(selection_pressure="low")).run()
            IntrinsicCooperationExperiment(base, cfg(selection_pressure="medium")).run()
            assert calls == ["run", "run"]

            # Concurrency and replicate count do not change an arm, so they do not invalidate it.
            IntrinsicCooperationExperiment(base, cfg(selection_pressure="medium", jobs=3)).run()
            assert calls == ["run", "run"]

            other_base = base.copy()
            other_base.resources.resource_regen_rate += 0.05
            IntrinsicCooperationExperiment(other_base, cfg(selection_pressure="medium")).run()
            assert calls == ["run", "run", "run"]

            # A file written before provenance was recorded is never trusted.
            path = os.path.join(out, "baseline", ARM_RESULT_FILE)
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            assert data["provenance"]["arm"] == "baseline" and data["provenance"]["config"]["selection_pressure"]
            data.pop("provenance")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            IntrinsicCooperationExperiment(other_base, cfg(selection_pressure="medium")).run()
            assert calls == ["run", "run", "run", "run"]


def test_summary_records_run_provenance():
    base = SimulationConfig.from_centralized_config(environment="testing")
    with tempfile.TemporaryDirectory() as out:
        cfg = IntrinsicCooperationExperimentConfig(num_steps=4, num_replicates=1, output_dir=out, arms=("baseline",))
        with patch("farm.runners.intrinsic_cooperation_experiment.run_simulation", _fake_run_simulation):
            result = IntrinsicCooperationExperiment(base, cfg).run()
        with open(result.summary_path, encoding="utf-8") as handle:
            config = json.load(handle)["config"]
        assert {"git_commit", "git_dirty", "python_hash_seed"} <= set(config)
        assert config["python_hash_seed"] == os.environ.get("PYTHONHASHSEED")


def test_lifespan_at_death_is_nan_when_nobody_died():
    assert np.isnan(ArmResult(arm="baseline", population=[5, 5]).summary()["mean_lifespan_at_death"])


def test_holm_step_down():
    results = [{"p_value": p} for p in (0.001, 0.04, 0.03)]
    holm(results, 0.05)
    assert [r["significant"] for r in results] == [True, False, False]
    assert results[0]["p_holm"] == pytest.approx(0.003)

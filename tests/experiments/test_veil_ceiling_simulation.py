"""Tests for the Veil Ceiling learner, inheritance surface and simulation loop."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from farm.core.policy_inheritance import apply_lamarckian_policy_warmstart
from farm.experiments.veil_ceiling.agents import LEDGER_STATS, N_FEATURES, VeilAgent
from farm.experiments.veil_ceiling.config import CONDITIONS, LearnerConfig, RunConfig, WorldConfig
from farm.experiments.veil_ceiling.learner import MLPQLearner
from farm.experiments.veil_ceiling.simulation import VeilSimulation, run_simulation

pytestmark = pytest.mark.unit

SHORT = {"train_ticks": 100, "eval_ticks": 50, "window_ticks": 50}


def _learner(seed: int = 0, **overrides: object) -> MLPQLearner:
    return MLPQLearner(N_FEATURES, 4, LearnerConfig(**overrides), np.random.default_rng(seed))


def _agent(agent_id: int, learner: MLPQLearner) -> VeilAgent:
    return VeilAgent(agent_id=agent_id, x=0, y=0, energy=10.0, learner=learner, generation=0, born_tick=0)


# ── learner ───────────────────────────────────────────────────────────────
def test_learner_learns_to_prefer_rewarded_action() -> None:
    learner = _learner(1, epsilon_start=0.0, epsilon_min=0.0)
    s = np.zeros(N_FEATURES)
    for _ in range(400):
        learner.observe(s, 2, 1.0, s, False)
        learner.observe(s, 0, -1.0, s, False)
    q = learner.q_values(s)
    assert int(np.argmax(q)) == 2
    assert learner.select_action(s) == 2


def test_learner_state_round_trip_and_frozen_learning() -> None:
    source = _learner(2)
    s = np.ones(N_FEATURES)
    for _ in range(50):
        source.observe(s, 1, 0.5, s, False)
    state = source.get_model_state()
    target = _learner(3)
    assert not np.allclose(target.q_values(s), source.q_values(s))
    target.load_model_state(state)
    assert np.allclose(target.q_values(s), source.q_values(s))

    target.learning_enabled = False
    before = target.policy.state_dict()
    for _ in range(50):
        target.observe(s, 0, -5.0, s, False)
    after = target.policy.state_dict()
    assert all(np.array_equal(before[k], after[k]) for k in before)


def test_lamarckian_warmstart_copies_parent_policy_through_core_helper() -> None:
    parent = _agent(0, _learner(4))
    s = np.linspace(0.0, 1.0, N_FEATURES)
    for _ in range(50):
        parent.learner.observe(s, 3, 1.0, s, False)
    child = _agent(1, _learner(5))
    assert not np.allclose(child.learner.q_values(s), parent.learner.q_values(s))
    reason = apply_lamarckian_policy_warmstart(parent, child)
    assert reason is None
    assert np.allclose(child.learner.q_values(s), parent.learner.q_values(s))


# ── simulation ────────────────────────────────────────────────────────────
def _run(condition: str, seed: int = 1, mode: str = "baldwinian", **kwargs: object):
    cfg = RunConfig(condition=CONDITIONS[condition], seed=seed, inheritance_mode=mode, **{**SHORT, **kwargs})
    return run_simulation(cfg)


def test_run_produces_consistent_outputs() -> None:
    result = _run("C2")
    summary = result.summary
    assert summary["run_id"] == "C2__baldwinian__s1"
    assert summary["ticks_completed"] == 150
    assert summary["extinct"] == 0
    assert summary["total_agents"] == len(result.agents)
    assert summary["births"] == (result.agents["parent_id"] >= 0).sum()
    assert summary["deaths"] == (result.agents["death_tick"] >= 0).sum()
    assert list(result.windows["phase"]) == ["train", "train", "eval"]
    assert list(result.windows["window_end_tick"]) == [50, 100, 150]
    assert result.windows["population"].iloc[-1] == summary["final_population"]
    for stat in LEDGER_STATS:
        assert any(col.startswith(f"{stat}__train__") for col in result.agents.columns)


def test_window_counts_match_agent_ledgers() -> None:
    result = _run("C2", seed=2)
    agents, windows = result.agents, result.windows
    for stat in ("opportunities", "defections", "penalties"):
        for region in ("train", "heldout"):
            for m in (0, 1):
                for c in (0, 1):
                    ledger_total = sum(
                        agents[f"{stat}__{phase}__{region}__m{m}__c{c}"].sum() for phase in ("train", "eval")
                    )
                    window_total = windows[f"{stat}__{region}__m{m}__c{c}"].sum()
                    assert ledger_total == pytest.approx(window_total), (stat, region, m, c)


def test_same_seed_is_bit_for_bit_reproducible() -> None:
    a = _run("C2", seed=3, mode="lamarckian")
    b = _run("C2", seed=3, mode="lamarckian")
    assert a.summary == b.summary
    pd.testing.assert_frame_equal(a.agents, b.agents)
    pd.testing.assert_frame_equal(a.windows, b.windows)


def test_seed_matching_shares_world_and_monitor_map_across_conditions() -> None:
    sims = {
        name: VeilSimulation(RunConfig(condition=CONDITIONS[name], seed=4, **SHORT))
        for name in ("C0", "C1", "C2", "C4")
    }
    base = sims["C0"]
    for name, sim in sims.items():
        assert np.array_equal(sim.field.xs, base.field.xs) and np.array_equal(sim.field.ys, base.field.ys), name
        assert [(a.x, a.y) for a in sim.agents] == [(a.x, a.y) for a in base.agents], name
    for sim in sims.values():
        sim.monitoring.maybe_resample(0, include_heldout=False)
    assert np.array_equal(sims["C1"].monitoring.true_mask, sims["C2"].monitoring.true_mask)
    assert np.array_equal(sims["C2"].monitoring.true_mask, sims["C4"].monitoring.true_mask)


def test_c0_has_no_monitoring_or_penalties() -> None:
    result = _run("C0", seed=5)
    assert result.summary["expected_penalty_per_defection"] == 0.0
    assert result.summary["realised_enforcement"] == 0.0
    monitored_cols = [c for c in result.agents.columns if "__m1__" in c]
    assert result.agents[monitored_cols].to_numpy().sum() == 0.0
    cue1_cols = [c for c in result.agents.columns if c.endswith("__c1")]
    assert result.agents[cue1_cols].to_numpy().sum() == 0.0
    assert result.summary["fidelity"] == -1.0


def test_penalties_only_occur_at_monitored_defections() -> None:
    result = _run("C2", seed=6, train_ticks=200, eval_ticks=0, window_ticks=50)
    agents = result.agents
    unmonitored_penalties = sum(
        agents[f"penalties__train__{r}__m0__c{c}"].sum() for r in ("train", "heldout") for c in (0, 1)
    )
    assert unmonitored_penalties == 0.0
    monitored_penalties = sum(
        agents[f"penalties__train__{r}__m1__c{c}"].sum() for r in ("train", "heldout") for c in (0, 1)
    )
    monitored_defections = sum(
        agents[f"defections__train__{r}__m1__c{c}"].sum() for r in ("train", "heldout") for c in (0, 1)
    )
    assert monitored_penalties == monitored_defections
    assert monitored_defections > 0
    assert result.summary["realised_enforcement"] == pytest.approx(
        agents[[f"penalties__train__train__m{m}__c{c}" for m in (0, 1) for c in (0, 1)]].to_numpy().sum()
        / agents[[f"defections__train__train__m{m}__c{c}" for m in (0, 1) for c in (0, 1)]].to_numpy().sum()
    )


def test_heldout_band_is_never_monitored_during_training() -> None:
    result = _run("C2", seed=7)
    agents = result.agents
    heldout_monitored_ticks = sum(agents[f"ticks__train__heldout__m1__c{c}"].sum() for c in (0, 1))
    assert heldout_monitored_ticks == 0.0
    eval_heldout_monitored = sum(agents[f"ticks__eval__heldout__m1__c{c}"].sum() for c in (0, 1))
    assert eval_heldout_monitored > 0.0


def test_perfect_cue_matches_monitor_state_in_ledgers() -> None:
    agents = _run("C2", seed=8).agents
    mismatched = sum(
        agents[f"ticks__{phase}__{region}__m{m}__c{1 - m}"].sum()
        for phase in ("train", "eval")
        for region in ("train", "heldout")
        for m in (0, 1)
    )
    assert mismatched == 0.0


def test_lamarckian_mode_applies_warmstart_and_baldwinian_does_not() -> None:
    lam = _run("C0", seed=9, mode="lamarckian", train_ticks=200, eval_ticks=0)
    bald = _run("C0", seed=9, mode="baldwinian", train_ticks=200, eval_ticks=0)
    assert lam.summary["births"] > 0
    assert lam.summary["warmstart_applied"] == lam.summary["births"]
    assert lam.summary["warmstart_skipped"] == 0
    assert bald.summary["warmstart_applied"] == 0


def test_learning_is_frozen_during_evaluation() -> None:
    sim = VeilSimulation(RunConfig(condition=CONDITIONS["C1"], seed=10, **SHORT))
    while sim.tick < sim.cfg.train_ticks:
        sim.step()
    sim.freeze_learning()
    alive = [a for a in sim.agents if a.alive]
    assert alive
    before = {a.agent_id: a.learner.policy.state_dict() for a in alive}
    while sim.tick < sim.cfg.total_ticks:
        sim.step()
    for agent in alive:
        after = agent.learner.policy.state_dict()
        assert all(np.array_equal(before[agent.agent_id][k], after[k]) for k in after)
    for agent in sim.agents:
        if agent.born_tick >= sim.cfg.train_ticks:
            assert not agent.learner.learning_enabled


def test_extinction_closes_final_window_and_flags_run() -> None:
    cfg = RunConfig(
        condition=CONDITIONS["C0"],
        seed=11,
        train_ticks=100,
        eval_ticks=0,
        window_ticks=50,
        world=replace(WorldConfig(), base_consumption=5.0),
    )
    result = run_simulation(cfg)
    assert result.summary["extinct"] == 1
    assert result.summary["ticks_completed"] < 100
    assert len(result.windows) >= 1
    assert result.windows["population"].iloc[-1] == 0

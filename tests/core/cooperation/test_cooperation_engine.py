"""Cooperation running inside the real spatial engine (needs torch).

These exercise the hooks that the service tests cannot reach: the
environment's action space and settlement, the agents' action lists,
rewards and observations, chromosome extension at founding, and the
telemetry written at cleanup.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import patch

import pytest

pytest.importorskip("torch")

from farm.config import SimulationConfig  # noqa: E402
from farm.core.cooperation import CooperationConfig, CooperationService  # noqa: E402
from farm.core.cooperation.genes import COOPERATION_GENE_NAMES, has_cooperation_genes  # noqa: E402
from farm.core.cooperation.service import SUMMARY_FILE, TIMESERIES_FILE, UNITS_FILE  # noqa: E402
from farm.core.simulation import run_simulation  # noqa: E402

pytestmark = pytest.mark.unit

STEPS = 25


def _config(cooperation: CooperationConfig) -> SimulationConfig:
    cfg = SimulationConfig.from_centralized_config(environment="testing")
    cfg.simulation_steps = STEPS
    cfg.max_steps = STEPS
    cfg.population.system_agents = 4
    cfg.population.independent_agents = 2
    cfg.population.control_agents = 0
    cfg.population.order_agents = 0
    cfg.population.chaos_agents = 0
    cfg.environment.width = 15
    cfg.environment.height = 15
    cfg.database.use_in_memory_db = True
    cfg.database.persist_db_on_completion = False
    cfg.cooperation = cooperation
    return cfg


def _active(out: str) -> CooperationConfig:
    # Everyone in range, a strong innate pull to bond, cheap exits: units
    # should form within a few steps whatever the untrained policies do.
    return CooperationConfig(
        enabled=True,
        bond_range=50.0,
        randomize_founder_genes=False,
        founder_gene_values={"bond_weight": 0.3, "leave_weight": 0.03, "coop_scope": 0.5},
        output_dir=out,
    )


def test_disabled_cooperation_leaves_the_world_unchanged():
    with tempfile.TemporaryDirectory() as out:
        env = run_simulation(num_steps=3, config=_config(CooperationConfig()), path=out, save_config=False, seed=3)
        assert env.cooperation is None
        assert env.action_space().n == 8
        for agent in env.agent_objects:
            assert not {"bond", "leave"} & {a.name for a in agent.actions}
            assert not has_cooperation_genes(agent.hyperparameter_chromosome)
        assert not os.path.exists(os.path.join(out, SUMMARY_FILE))


def test_active_cooperation_runs_end_to_end():
    with tempfile.TemporaryDirectory() as out:
        env = run_simulation(num_steps=STEPS, config=_config(_active(out)), path=out, save_config=True, seed=5)
        service = env.cooperation
        assert isinstance(service, CooperationService)
        assert env.action_space().n == 10

        founders = [a for a in env.agent_objects if a.generation == 0]
        assert founders
        for agent in founders:
            names = [a.name for a in agent.actions]
            assert names[-2:] == ["bond", "leave"]
            assert len(names) == env.action_space().n
            assert has_cooperation_genes(agent.hyperparameter_chromosome)
            assert agent.hyperparameter_chromosome.get_value("bond_weight") == pytest.approx(0.3, abs=0.15)
            assert sum(a.weight for a in agent.actions) == pytest.approx(1.0)
            assert agent.behavior.decision_module.num_actions == 10

        counters = service.counters
        assert counters["bond_attempts"] > 0
        assert counters["units_formed"] >= 1
        assert len(service.timeseries) >= STEPS - 1
        assert all(0.0 <= row["coverage"] <= 1.0 for row in service.timeseries)

        # Written by Environment.cleanup at the end of the run.
        with open(os.path.join(out, SUMMARY_FILE), encoding="utf-8") as handle:
            summary = json.load(handle)
        assert summary["units_formed"] >= 1
        assert summary["action_indices"] == {"bond": 8, "leave": 9}
        for name in (TIMESERIES_FILE, UNITS_FILE):
            assert os.path.getsize(os.path.join(out, name)) > 0

        # Every unit member is a live agent the environment knows about.
        alive = {a.agent_id for a in env.alive_agent_objects}
        for unit in service.registry.active_units():
            assert set(unit.members) <= alive


def test_partners_are_allies_in_observations():
    with tempfile.TemporaryDirectory() as out:
        env = run_simulation(num_steps=STEPS, config=_config(_active(out)), path=out, save_config=False, seed=5)
        service = env.cooperation
        members = [a for a in env.alive_agent_objects if service.is_member(a.agent_id)]
        if not members:
            pytest.skip("no live unit members at the end of this run")
        agent = members[0]
        obs = env.agent_observations[agent.agent_id]
        allies, enemies = obs._compute_entities_from_spatial_index(
            env.spatial_index, agent, (int(agent.position[1]), int(agent.position[0]))
        )
        # The membership marker sits on the agent's own cell.
        assert (int(agent.position[1]), int(agent.position[0]), 1.0) in allies
        # Partners are allies whatever their type; same-type agents stay allies too.
        partner_ids = service.partner_ids(agent.agent_id)
        others = [a for a in env.alive_agent_objects if a is not agent]
        ally_cells = {(y, x) for y, x, _ in allies}
        for other in others:
            if other.agent_id in partner_ids or other.agent_type == agent.agent_type:
                cell = (int(other.position[1]), int(other.position[0]))
                if cell in ally_cells:
                    continue
                enemy_cells = {(y, x) for y, x, _ in enemies}
                assert cell not in enemy_cells, "an ally was listed as an enemy"


def test_cooperation_config_survives_simulation_config_round_trip():
    cfg = _config(CooperationConfig(enabled=True, mode="placebo", fixed_settings={"scope": 0.3}))
    payload = json.loads(json.dumps(cfg.to_dict()["cooperation"]))
    again = SimulationConfig.from_dict({"cooperation": payload})
    assert again.cooperation == cfg.cooperation
    assert SimulationConfig.from_dict({}).cooperation == CooperationConfig()
    assert len(COOPERATION_GENE_NAMES) == 13


def _reward_for(env, agent, credit_payouts: bool) -> tuple[float, float]:
    """Reward for a ``pass`` after a payout of 5 lands between actions."""
    from dataclasses import replace

    from farm.core.action import action_registry

    service = env.cooperation
    service.config = replace(service.config, credit_payouts=credit_payouts)
    service.consume_pending(agent.agent_id)  # start from a clean slate
    agent._update_state_snapshot()
    pre = agent.state.snapshot(env.time)
    service._credit({agent.agent_id: 5.0}, "payout")  # settlement between actions
    service._pending_reward[agent.agent_id] = 0.25  # membership reward this step
    agent._update_state_snapshot()
    post = agent.state.snapshot(env.time)
    reward = agent._calculate_reward(pre, post, action_registry.get("pass"))
    weights = agent._reward_weights()
    base = weights["reward_survival_weight"]
    return reward - base, weights["reward_resource_weight"]


def test_pointless_cooperation_actions_are_masked_for_real_agents():
    with tempfile.TemporaryDirectory() as out:
        env = run_simulation(num_steps=2, config=_config(_active(out)), path=out, save_config=False, seed=5)
        service = env.cooperation
        for agent in env.alive_agent_objects:
            enabled = agent._get_enabled_actions()
            names = {a.name for a in agent.actions} if enabled is None else {a.name for a in enabled}
            expected_hidden = service.unavailable_actions(agent)
            assert env.unavailable_actions(agent) == expected_hidden
            assert not (expected_hidden & names)
            if service.is_member(agent.agent_id):
                assert "leave" in names
            else:
                assert "leave" not in names

    placebo = CooperationConfig(enabled=True, mode="placebo", bond_range=50.0, output_dir="")
    with tempfile.TemporaryDirectory() as out:
        env = run_simulation(num_steps=2, config=_config(placebo), path=out, save_config=False, seed=5)
        for agent in env.alive_agent_objects:
            names = {a.name for a in agent._get_enabled_actions()}
            assert "leave" not in names and "bond" in names  # everyone is in range, nobody is ever a member


def test_members_never_attack_their_partners():
    from farm.core.action import attack_action

    with tempfile.TemporaryDirectory() as out:
        env = run_simulation(num_steps=STEPS, config=_config(_active(out)), path=out, save_config=False, seed=5)
        service = env.cooperation
        members = [a for a in env.alive_agent_objects if service.is_member(a.agent_id)]
        if not members:
            pytest.skip("no live unit members at the end of this run")
        agent = members[0]
        partners = service.partner_ids(agent.agent_id)
        assert partners
        partner = next(a for a in env.alive_agent_objects if a.agent_id in partners)
        stranger = next((a for a in env.alive_agent_objects if a is not agent and a.agent_id not in partners), None)
        # Only the partner is in range: the attack finds no valid target.
        nearby = {"agents": [agent, partner]}
        with patch.object(agent.spatial_service, "get_nearby", lambda *a, **k: nearby):
            before = partner.current_health
            result = attack_action(agent)
        assert result["success"] is False and "No valid targets" in result["error"]
        assert partner.current_health == pytest.approx(before)
        if stranger is not None:
            # With a stranger also in range, the stranger is the one hit.
            nearby = {"agents": [agent, partner, stranger]}
            with patch.object(agent.spatial_service, "get_nearby", lambda *a, **k: nearby):
                result = attack_action(agent)
            assert result["details"]["target_id"] == stranger.agent_id


def test_death_causes_are_recorded_on_the_agent():
    with tempfile.TemporaryDirectory() as out:
        env = run_simulation(num_steps=1, config=_config(CooperationConfig()), path=out, save_config=False, seed=5)
        starving, fighter, other = env.alive_agent_objects[:3]
        assert starving.death_cause is None
        resource = starving.get_component("resource")
        starving.resource_level = 0.0
        resource.starvation_counter = resource.config.starvation_threshold - 1
        resource._check_starvation()
        assert not starving.alive and starving.death_cause == "starvation"
        fighter.get_component("combat").take_damage(fighter.current_health * 10)
        assert not fighter.alive and fighter.death_cause == "combat"
        other.terminate()
        assert not other.alive and other.death_cause is None


def test_payouts_count_once_in_the_reward():
    with tempfile.TemporaryDirectory() as out:
        env = run_simulation(num_steps=2, config=_config(_active(out)), path=out, save_config=False, seed=5)
        agent = next(a for a in env.alive_agent_objects)
        extra, w = _reward_for(env, agent, credit_payouts=True)
        assert extra == pytest.approx(5.0 * w + 0.25)  # the delta carries the payout; no second credit
        extra, w = _reward_for(env, agent, credit_payouts=False)
        assert extra == pytest.approx(0.25)  # payout hidden from the reward

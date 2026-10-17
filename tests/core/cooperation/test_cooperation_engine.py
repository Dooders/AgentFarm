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
        founder_gene_values={"bond_weight": 1.0, "leave_weight": 0.05, "coop_scope": 0.5},
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
            assert agent.hyperparameter_chromosome.get_value("bond_weight") == pytest.approx(1.0, abs=0.5)
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
        partner_ids = service.partner_ids(agent.agent_id)
        others = [a for a in env.alive_agent_objects if a is not agent]
        assert len(allies) - 1 <= len([a for a in others if a.agent_id in partner_ids])
        del enemies


def test_cooperation_config_survives_simulation_config_round_trip():
    cfg = _config(CooperationConfig(enabled=True, mode="placebo", fixed_settings={"scope": 0.3}))
    payload = json.loads(json.dumps(cfg.to_dict()["cooperation"]))
    again = SimulationConfig.from_dict({"cooperation": payload})
    assert again.cooperation == cfg.cooperation
    assert SimulationConfig.from_dict({}).cooperation == CooperationConfig()
    assert len(COOPERATION_GENE_NAMES) == 13

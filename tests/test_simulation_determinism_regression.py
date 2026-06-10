"""Regression check that simulation outcomes remain deterministic."""

import json

import pytest

from farm.config import SimulationConfig
from farm.config.config import (
    DatabaseConfig,
    EnvironmentConfig,
    PopulationConfig,
    ResourceConfig,
)
from farm.core.simulation import run_simulation


def _capture_final_state(environment) -> dict:
    """Capture a stable, serializable view of simulation end state."""
    agents = []
    for agent in sorted(environment._agent_objects.values(), key=lambda item: item.agent_id):
        agents.append(
            {
                "id": agent.agent_id,
                "type": agent.__class__.__name__,
                "position": list(agent.position),
                "resource_level": agent.resource_level,
                "alive": agent.alive,
                "generation": agent.generation,
            }
        )

    resources = []
    for resource in sorted(environment.resources, key=lambda item: item.resource_id):
        resources.append(
            {
                "id": resource.resource_id,
                "position": list(resource.position),
                "amount": resource.amount,
            }
        )

    return {
        "time": environment.time,
        "agent_count": len(agents),
        "resource_count": len(resources),
        "agents": agents,
        "resources": resources,
    }


def _run_deterministic_trial(tmp_path, seed: int, trial_name: str) -> dict:
    """Run one short simulation trial with fixed config + seed."""
    config = SimulationConfig(
        environment=EnvironmentConfig(width=40, height=40),
        resources=ResourceConfig(initial_resources=40),
        population=PopulationConfig(
            system_agents=3,
            independent_agents=3,
            control_agents=0,
        ),
        database=DatabaseConfig(
            use_in_memory_db=True,
            persist_db_on_completion=False,
        ),
    )
    config.seed = seed

    output_dir = tmp_path / trial_name
    output_dir.mkdir()

    # No external RNG seeding here: reproducibility must come from run_simulation(seed=...).
    environment = run_simulation(
        num_steps=8,
        config=config,
        path=str(output_dir),
        simulation_id="determinism_regression",
        seed=seed,
    )

    try:
        return _capture_final_state(environment)
    finally:
        environment.cleanup()


@pytest.mark.unit
@pytest.mark.determinism_regression
def test_simulation_repeatability_fixed_seed(tmp_path):
    """The same seed and config should produce the same final state."""
    seed = 20260505
    snapshots = [
        _run_deterministic_trial(tmp_path, seed=seed, trial_name="trial_1"),
        _run_deterministic_trial(tmp_path, seed=seed, trial_name="trial_2"),
        _run_deterministic_trial(tmp_path, seed=seed, trial_name="trial_3"),
    ]

    baseline = json.dumps(snapshots[0], sort_keys=True)
    for snapshot in snapshots[1:]:
        assert json.dumps(snapshot, sort_keys=True) == baseline

"""Tests for simulation-level deferred RL training batching."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from farm.config import SimulationConfig
from farm.core.simulation import _run_deferred_learning_updates, run_simulation


class _StubAgent:
    """Minimal agent stub for exercising deferred training scheduling."""

    def __init__(self, agent_id: str, train_results: list[bool]) -> None:
        self.agent_id = agent_id
        self._train_results = list(train_results)
        self.train_call_count = 0

    def act(self) -> None:
        """No-op action execution for simulation loop compatibility."""
        return None

    def train_learning_if_ready(self) -> bool:
        """Return scripted training readiness outcomes."""
        self.train_call_count += 1
        if self._train_results:
            return bool(self._train_results.pop(0))
        return False


@pytest.mark.unit
def test_run_simulation_limits_deferred_learning_updates_per_step(tmp_path):
    """Simulation loop should cap deferred RL updates per step."""
    config = SimulationConfig.from_centralized_config(environment="testing")
    config.population.system_agents = 0
    config.population.independent_agents = 0
    config.population.control_agents = 0
    config.population.order_agents = 0
    config.population.chaos_agents = 0
    config.max_steps = 10
    config.database.use_in_memory_db = True
    config.database.persist_db_on_completion = False
    config.performance.defer_learning_training = True
    config.performance.max_learning_updates_per_step = 2

    # 2 simulation steps, each agent always ready to train.
    steps = 2
    scripted_results = [True] * (steps * config.performance.max_learning_updates_per_step)
    agent_a = _StubAgent("stub-a", train_results=scripted_results.copy())
    agent_b = _StubAgent("stub-b", train_results=scripted_results.copy())
    stub_agents = [agent_a, agent_b]
    step_training_call_totals: list[int] = []

    def _inject_agents(environment) -> None:
        environment.defer_learning_training = True
        environment._agent_objects = {agent.agent_id: agent for agent in stub_agents}
        environment._alive_agents = {agent.agent_id for agent in stub_agents}
        environment.agents = [agent.agent_id for agent in stub_agents]
        environment.agent_selection = environment.agents[0]

    def _on_step_end(_environment, _step_index: int) -> None:
        total_calls = agent_a.train_call_count + agent_b.train_call_count
        step_training_call_totals.append(total_calls)

    with patch("farm.core.simulation.create_initial_agents", return_value=[]):
        run_simulation(
            num_steps=steps,
            config=config,
            path=str(tmp_path),
            save_config=False,
            on_environment_ready=_inject_agents,
            on_step_end=_on_step_end,
            disable_console_logging=True,
        )

    expected_max_calls = steps * config.performance.max_learning_updates_per_step
    assert agent_a.train_call_count + agent_b.train_call_count <= expected_max_calls
    assert agent_a.train_call_count >= 1
    assert agent_b.train_call_count >= 1
    assert len(step_training_call_totals) == steps

    previous_total = 0
    for step_total in step_training_call_totals:
        calls_this_step = step_total - previous_total
        assert calls_this_step <= config.performance.max_learning_updates_per_step
        previous_total = step_total


@pytest.mark.unit
def test_run_deferred_learning_updates_autoscale_zero_means_alive_count():
    """``max_updates == 0`` should auto-scale to len(alive_agents).

    Previously ``max_updates <= 0`` short-circuited and ran zero training
    updates, which combined with the old default of 4 to throttle learning
    to a tiny fraction of what agents could absorb in a typical run.
    """
    agents = [_StubAgent(f"a{i}", train_results=[True]) for i in range(7)]
    env = SimpleNamespace(alive_agent_objects=agents)

    updates = _run_deferred_learning_updates(env, max_updates=0, rr_cursor=0)

    assert updates == len(agents)
    for agent in agents:
        assert agent.train_call_count == 1


@pytest.mark.unit
def test_run_deferred_learning_updates_negative_disables_training():
    """A negative cap is treated as 'no training this step' (no auto-scale)."""
    agents = [_StubAgent("a", train_results=[True])]
    env = SimpleNamespace(alive_agent_objects=agents)

    updates = _run_deferred_learning_updates(env, max_updates=-1, rr_cursor=0)

    assert updates == 0
    assert agents[0].train_call_count == 0


@pytest.mark.unit
def test_run_simulation_negative_max_updates_disables_deferred_training(tmp_path):
    """Negative cap should disable deferred updates instead of auto-scaling."""
    config = SimulationConfig.from_centralized_config(environment="testing")
    config.population.system_agents = 0
    config.population.independent_agents = 0
    config.population.control_agents = 0
    config.population.order_agents = 0
    config.population.chaos_agents = 0
    config.max_steps = 10
    config.database.use_in_memory_db = True
    config.database.persist_db_on_completion = False
    config.performance.defer_learning_training = True
    config.performance.max_learning_updates_per_step = -1

    steps = 3
    agent_a = _StubAgent("stub-a", train_results=[True] * 10)
    agent_b = _StubAgent("stub-b", train_results=[True] * 10)
    stub_agents = [agent_a, agent_b]

    def _inject_agents(environment) -> None:
        environment.defer_learning_training = True
        environment._agent_objects = {agent.agent_id: agent for agent in stub_agents}
        environment._alive_agents = {agent.agent_id for agent in stub_agents}
        environment.agents = [agent.agent_id for agent in stub_agents]
        environment.agent_selection = environment.agents[0]

    with patch("farm.core.simulation.create_initial_agents", return_value=[]):
        run_simulation(
            num_steps=steps,
            config=config,
            path=str(tmp_path),
            save_config=False,
            on_environment_ready=_inject_agents,
            disable_console_logging=True,
        )

    assert agent_a.train_call_count == 0
    assert agent_b.train_call_count == 0

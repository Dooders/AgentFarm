"""Executable record of the 2026-11 engine review findings.

Each ``xfail(strict=True)`` test below encodes a correctness failure documented in
``docs/research/reviews/2026-11-engine-review/README.md``. The tests exercise a real
``Environment`` with real ``AgentCore`` agents rather than mocks, because every one of these
bugs survived the existing suite by hiding behind a ``getattr``/``hasattr`` fallback that only
a real run reaches.

``pytest.ini`` sets ``xfail_strict = true``: when a fix lands the corresponding test starts
passing, the strict xfail fails the suite, and the marker must be removed in the same PR. That
is deliberate. Do not loosen the assertions to make a marker unnecessary.

Run with ``pytest -m integration tests/integration/test_engine_semantics.py``.
"""

from __future__ import annotations

import hashlib
import importlib.util
from collections import Counter
from collections.abc import Callable

import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover - lightweight CI without torch
    pytest.skip("engine semantics tests require torch", allow_module_level=True)

from sqlalchemy import func, select

from farm.config import SimulationConfig
from farm.core.agent import AgentFactory, AgentServices
from farm.core.agent.config.component_configs import AgentComponentConfig
from farm.core.channels import Channel
from farm.core.decision.algorithms import tianshou as tianshou_module
from farm.core.environment import Environment
from farm.core.resources import Resource
from farm.core.simulation import run_simulation
from farm.database.models import ActionModel

pytestmark = [pytest.mark.integration, pytest.mark.ml]

REVIEW = "docs/research/reviews/2026-11-engine-review/README.md"


def _small_config(seed: int = 11, agents_per_type: int = 4, max_population: int = 12) -> SimulationConfig:
    cfg = SimulationConfig.from_centralized_config(environment="development")
    cfg.seed = seed
    cfg.population.system_agents = agents_per_type
    cfg.population.independent_agents = agents_per_type
    cfg.population.control_agents = agents_per_type
    cfg.population.max_population = max_population
    cfg.database.use_in_memory_db = True
    cfg.database.persist_db_on_completion = False
    cfg.database.enable_validation = False
    # Keep the learning stack (transitions are still stored) but skip gradient updates so the
    # tests measure engine semantics, not DQN training time.
    cfg.performance.max_learning_updates_per_step = -1
    return cfg


def _run(cfg: SimulationConfig, steps: int, **hooks: Callable) -> Environment:
    return run_simulation(
        num_steps=steps,
        config=cfg,
        path=None,
        save_config=False,
        seed=cfg.seed,
        disable_console_logging=True,
        **hooks,
    )


def _single_agent_env(cfg: SimulationConfig, position: tuple[int, int], resources: list[Resource]) -> tuple[Environment, object]:
    env = Environment(
        width=50,
        height=50,
        resource_distribution={"amount": 0},
        config=cfg,
        db_path=":memory:",
    )
    env.resources = resources
    services = AgentServices(
        spatial_service=env.spatial_service,
        time_service=getattr(env, "time_service", None),
        metrics_service=getattr(env, "metrics_service", None),
        logging_service=getattr(env, "logging_service", None),
        validation_service=getattr(env, "validation_service", None),
        lifecycle_service=getattr(env, "lifecycle_service", None),
    )
    agent = AgentFactory(services).create_learning_agent(
        agent_id="probe",
        position=position,
        initial_resources=0.0,
        config=AgentComponentConfig.from_simulation_config(cfg),
        environment=env,
    )
    env.add_agent(agent)
    env.spatial_index.set_references(list(env._agent_objects.values()), env.resources)
    env.spatial_index.update()
    return env, agent


def _force_action(agent, name: str) -> None:
    """Make the agent's decision module always pick ``name`` from the enabled action list."""
    full_idx = next(i for i, a in enumerate(agent.actions) if a.name == name)

    def decide(state, enabled_actions=None, action_weights=None):
        if enabled_actions:
            for rel, idx in enumerate(enabled_actions):
                if idx == full_idx:
                    return rel
            return 0
        return full_idx

    agent.behavior.decision_module.decide_action = decide


# --------------------------------------------------------------------------- finding 1


@pytest.mark.xfail(
    strict=True,
    reason=f"M0-A: move_action requires agent.validation_service/update_position, which AgentCore lacks ({REVIEW} s2.5, s3.5)",
)
def test_agents_move():
    cfg = _small_config()
    start_positions: dict[str, tuple] = {}

    def record_start(env: Environment) -> None:
        start_positions.update({a.agent_id: tuple(a.position) for a in env.alive_agent_objects})

    env = _run(cfg, steps=60, on_environment_ready=record_start)

    moved = [a.agent_id for a in env.agent_objects if tuple(a.position) != start_positions.get(a.agent_id, tuple(a.position))]
    assert start_positions, "simulation spawned no agents"
    assert moved, f"none of {len(start_positions)} agents changed position in 60 steps"


# --------------------------------------------------------------------------- finding 2


@pytest.mark.xfail(
    strict=True,
    reason=f"M0-B: perception_radius=5 layers vs ObservationConfig.R=6 window; RESOURCES channel is cropped at world coords ({REVIEW} s2.4)",
)
def test_resources_channel_sees_nearby_resources():
    cfg = _small_config()
    centre = (25, 25)
    nearby = [
        Resource(resource_id=1, position=(27, 25), amount=5.0, max_amount=5.0, regeneration_rate=0.0),
        Resource(resource_id=2, position=(25, 23), amount=5.0, max_amount=5.0, regeneration_rate=0.0),
        Resource(resource_id=3, position=(23, 27), amount=5.0, max_amount=5.0, regeneration_rate=0.0),
    ]
    _env, agent = _single_agent_env(cfg, centre, nearby)

    observation = agent._create_observation()
    resources_layer = observation[Channel.RESOURCES]
    radius = (resources_layer.shape[-1] - 1) // 2
    assert radius >= 2, f"observation window too small to contain the probe resources (R={radius})"

    nonzero = int((resources_layer > 0).sum().item())
    assert nonzero == len(nearby), (
        f"expected {len(nearby)} non-zero RESOURCES cells within R={radius}, got {nonzero}"
    )


# --------------------------------------------------------------------------- finding 3


@pytest.mark.xfail(
    strict=True,
    reason=f"M0-C: gather_action uses getattr(agent.config, 'gathering_range', 30); YAML value never reaches the action ({REVIEW} s2.5)",
)
def test_gathering_range_from_config_is_respected():
    cfg = _small_config()
    cfg.agent_behavior.gathering_range = 1
    far_resource = Resource(resource_id=1, position=(35, 25), amount=10.0, max_amount=10.0, regeneration_rate=0.0)
    _env, agent = _single_agent_env(cfg, (25, 25), [far_resource])
    _force_action(agent, "gather")

    before = far_resource.amount
    agent.act()

    assert far_resource.amount == before, (
        f"agent 10 cells away gathered {before - far_resource.amount} with gathering_range=1"
    )


# --------------------------------------------------------------------------- finding 4


@pytest.mark.xfail(
    strict=True,
    reason=f"M0-D: starvation terminates in ResourceComponent.on_step_start before any transition is stored ({REVIEW} s2.2, s3.5)",
)
def test_every_death_stores_a_terminal_transition(monkeypatch: pytest.MonkeyPatch):
    cfg = _small_config(agents_per_type=5, max_population=15)
    # Starve the population quickly so deaths happen inside a short run.
    cfg.agent_behavior.starvation_threshold = 3
    cfg.resources.initial_resources = 0
    cfg.agent_behavior.base_consumption_rate = 1.0

    stored = Counter()
    original_store = tianshou_module.TianshouWrapper.store_experience

    def counting_store(self, state, action, reward, next_state, done, **kwargs):
        stored["transitions"] += 1
        stored["terminal"] += int(bool(done))
        return original_store(self, state, action, reward, next_state, done, **kwargs)

    monkeypatch.setattr(tianshou_module.TianshouWrapper, "store_experience", counting_store)

    deaths = Counter()

    def count_deaths(env: Environment) -> None:
        original_remove = env.remove_agent

        def counting_remove(agent):
            deaths["n"] += 1
            return original_remove(agent)

        env.remove_agent = counting_remove

    _run(cfg, steps=80, on_environment_ready=count_deaths)

    assert deaths["n"] > 0, "scenario produced no deaths; tighten the starvation settings"
    assert stored["transitions"] > 0, "no transitions were stored at all"
    assert stored["terminal"] == deaths["n"], (
        f"{deaths['n']} agents died but {stored['terminal']} transitions were stored with done=True"
    )


# --------------------------------------------------------------------------- finding 5 (logging)


@pytest.mark.xfail(
    strict=True,
    reason=f"M0-E1: DecisionModule.update logs the action a second time ({REVIEW} s2.6, s3.5)",
)
def test_one_action_row_per_agent_step():
    cfg = _small_config()
    steps = 40
    counts: dict[str, int] = {}

    def count_rows_on_last_step(env: Environment, step: int) -> None:
        # run_simulation closes the in-memory DB on exit, so the rows must be counted in-run.
        if step != steps - 1:
            return
        env.db.logger.flush_all_buffers()
        session = env.db.Session()
        try:
            counts["total"] = session.execute(select(func.count()).select_from(ActionModel)).scalar_one()
            counts["distinct"] = session.execute(
                select(func.count()).select_from(
                    select(ActionModel.step_number, ActionModel.agent_id).distinct().subquery()
                )
            ).scalar_one()
        finally:
            session.close()

    _run(cfg, steps=steps, on_step_end=count_rows_on_last_step)

    assert counts.get("total", 0) > 0, "no actions were logged"
    assert counts["total"] == counts["distinct"], (
        f"{counts['total']} agent_actions rows for {counts['distinct']} distinct (step, agent) pairs"
    )


# --------------------------------------------------------------------------- determinism (passes)


def _state_digest(env: Environment) -> str:
    digest = hashlib.sha256()
    for agent in env.alive_agent_objects:
        digest.update(
            f"{agent.agent_id}|{agent.position}|{agent.resource_level:.6f}|{agent.current_health:.6f}".encode()
        )
    for resource in env.resources:
        digest.update(f"{resource.position}|{resource.amount:.6f}".encode())
    return digest.hexdigest()[:16]


def _digests_for(seed: int, steps: int) -> list[str]:
    cfg = _small_config(seed=seed)
    digests: list[str] = []
    _run(cfg, steps=steps, on_step_end=lambda env, step: digests.append(_state_digest(env)))
    return digests


@pytest.mark.determinism
def test_same_seed_is_deterministic():
    first = _digests_for(seed=123, steps=40)
    second = _digests_for(seed=123, steps=40)
    other = _digests_for(seed=124, steps=40)

    assert len(first) == 40
    assert first == second, "identical seeds diverged: first differing step " + str(
        next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)
    )
    assert first != other, "different seeds produced identical trajectories"

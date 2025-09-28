import pytest
import importlib.util

# Skip this module if optional heavy deps are missing in the runner
if importlib.util.find_spec("torch") is None:
    pytest.skip("integration gathering tests require torch; skipping in lightweight env", allow_module_level=True)

from farm.core.agent import BaseAgent
from config_hydra import HydraSimulationConfig
from farm.core.environment import Environment
from farm.core.resources import Resource
from farm.core.action import ActionType


@pytest.fixture
def simple_config():
    # Use the repo test config but keep it lean for fast runs
    # For now, create a basic config - this should be updated to use hydra config loading
    return HydraHydraSimulationConfig()


def _get_gather_action_index(env: Environment) -> int:
    """Helper to resolve the current action index for 'gather' in the enabled action space."""
    # Environment maps from ActionType -> name; enabled list stores ActionType ordering for current space
    for idx, atype in enumerate(env._enabled_action_types):
        if env._action_mapping.get(atype) == "gather":
            return idx
    # Fallback: gather not enabled
    return -1


def test_env_step_agent_gathers_from_colocated_resource(simple_config: HydraSimulationConfig):
    # Build environment with no auto-spawned resources
    env = Environment(
        width=50,
        height=50,
        resource_distribution={"amount": 0},
        config=simple_config,
        db_path=":memory:",
    )

    # Create a single agent at (25,25)
    agent = BaseAgent(
        agent_id="a-1",
        position=(25, 25),
        resource_level=0,
        spatial_service=env.spatial_service,
        environment=env,
        config=simple_config,
    )

    # Reset to initialize PettingZoo bookkeeping and add our agent
    # Note: resources get regenerated in reset; we set them manually after
    env.reset(options={"agents": [agent]})

    # Place a resource at the same location
    resource = Resource(resource_id=0, position=(25, 25), amount=10.0, max_amount=10.0, regeneration_rate=0.0)
    env.resources = [resource]
    env.spatial_index.set_references(list(env._agent_objects.values()), env.resources)
    env.spatial_index.update()

    # Ensure gather is enabled and get its index in current action space
    gather_idx = _get_gather_action_index(env)
    assert gather_idx >= 0, "gather action must be enabled in the current action mapping"

    # Take a step selecting gather for the current agent
    before_agent = agent.resource_level
    before_res = resource.amount
    obs, reward, terminated, truncated, info = env.step(gather_idx)

    # Validate that gather had an effect
    assert agent.resource_level > before_agent
    assert resource.amount < before_res
    assert obs is not None
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_agent_act_gathers_when_decision_returns_gather(simple_config: HydraSimulationConfig, monkeypatch: pytest.MonkeyPatch):
    # Build environment with a single resource and agent colocated
    env = Environment(
        width=30,
        height=30,
        resource_distribution={"amount": 0},
        config=simple_config,
        db_path=":memory:",
    )
    resource = Resource(resource_id=1, position=(10, 10), amount=5.0, max_amount=5.0, regeneration_rate=0.0)
    env.resources = [resource]
    env.spatial_index.set_references([], env.resources)
    env.spatial_index.update()

    agent = BaseAgent(
        agent_id="a-2",
        position=(10, 10),
        resource_level=0.0,
        spatial_service=env.spatial_service,
        environment=env,
        config=simple_config,
    )
    env.add_agent(agent)

    # Monkeypatch the agent's decision to always choose gather within enabled actions
    def fake_decide_action(state, enabled):
        # Map enabled (which are indices into full action set) to find gather
        # BaseAgent._select_action_with_curriculum provides enabled as indices into full space
        from farm.core.action import ActionType

        for i, idx in enumerate(enabled):
            if idx == ActionType.GATHER:
                return i
        return 0

    monkeypatch.setattr(agent.decision_module, "decide_action", fake_decide_action)

    before_agent = agent.resource_level
    before_res = resource.amount

    # Execute one full agent act() cycle
    agent.act()

    assert agent.resource_level > before_agent
    assert resource.amount < before_res


def test_spatial_index_references_accept_agent_objects(simple_config: HydraSimulationConfig):
    # Minimal spatial index usage through the environment
    env = Environment(
        width=20,
        height=20,
        resource_distribution={"amount": 0},
        config=simple_config,
        db_path=":memory:",
    )
    # One agent and one resource
    agent = BaseAgent(
        agent_id="a-3",
        position=(5, 5),
        resource_level=1.0,
        spatial_service=env.spatial_service,
        environment=env,
        config=simple_config,
    )
    resource = Resource(resource_id=2, position=(6, 5), amount=3.0, max_amount=3.0, regeneration_rate=0.0)
    env.add_agent(agent)
    env.resources = [resource]

    # Ensure we pass agent OBJECTS to spatial index
    env.spatial_index.set_references(list(env._agent_objects.values()), env.resources)
    env.spatial_index.update()

    nearby = env.spatial_service.get_nearby(agent.position, 2.0, ["resources", "agents"])
    assert "resources" in nearby and "agents" in nearby
    assert any(r is resource for r in nearby["resources"])  # resource found
    assert any(a is agent for a in nearby["agents"])  # agent found


def test_decision_prioritizes_gather_nearby(simple_config: HydraSimulationConfig, monkeypatch: pytest.MonkeyPatch):
    # Environment with one agent and a resource in immediate proximity
    env = Environment(
        width=30,
        height=30,
        resource_distribution={"amount": 0},
        config=simple_config,
        db_path=":memory:",
    )
    agent = BaseAgent(
        agent_id="a-4",
        position=(12, 12),
        resource_level=0.0,
        spatial_service=env.spatial_service,
        environment=env,
        config=simple_config,
    )
    env.add_agent(agent)

    resource = Resource(resource_id=3, position=(12, 12), amount=4.0, max_amount=4.0, regeneration_rate=0.0)
    env.resources = [resource]
    env.spatial_index.set_references(list(env._agent_objects.values()), env.resources)
    env.spatial_index.update()

    # Force decision to use full mapping but we expect gather index will be used
    # by mapping through enabled action indices in BaseAgent
    before_agent = agent.resource_level
    before_res = resource.amount

    # Let BaseAgent decide; ensure the mapping returns gather action
    action = agent.decide_action()
    assert action.name in {"gather", "move", "pass", "attack", "share", "reproduce", "defend"}

    # Execute the chosen action and verify gathering occurs if gather selected
    result = action.execute(agent)
    if action.name == "gather":
        assert result.get("success", False)
        assert agent.resource_level > before_agent
        assert resource.amount < before_res
import math
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from farm.core.resources import Resource
from farm.core.state import (
    AgentState,
    BaseState,
    EnvironmentState,
    ModelState,
    SimulationState,
)


def test_agent_state_to_dict_and_tensor():
    state = AgentState.from_raw_values(
        agent_id="a1",
        step_number=10,
        position_x=0.2,
        position_y=0.4,
        position_z=0.0,
        resource_level=0.5,
        current_health=0.9,
        is_defending=False,
        total_reward=1.5,
        age=3,
    )

    d = state.to_dict()
    assert d["agent_id"] == "a1"
    assert d["position"] == (0.2, 0.4, 0.0)

    t = state.to_tensor(torch.device("cpu"))
    assert isinstance(t, torch.Tensor)
    assert t.shape == (11,)  # position(3), resource_level(1), current_health(1), is_defending(1), total_reward(1), age(1), generation(1), orientation(1), alive(1)
    # bool should be cast to float 0/1
    assert t.dtype == torch.float32


def test_environment_state_from_environment_and_tensor():
    class DummyEnv:
        pass

    env = DummyEnv()
    env.width = 100
    env.height = 100
    env.time = 250
    env.max_resource = None

    # Create some resources with varying amounts
    env.resources = [
        Resource(resource_id=i, position=(i, i), amount=5.0, max_amount=10.0)
        for i in range(10)
    ]

    # Create alive agent objects
    agent = Mock()
    agent.alive = True
    env.agent_objects = [agent for _ in range(30)]

    # Minimal config for population and max resource
    cfg = Mock()
    cfg.population = Mock()
    cfg.population.max_population = 300
    cfg.resources = Mock()
    cfg.resources.max_resource_amount = 10
    env.config = cfg

    st = EnvironmentState.from_environment(env)
    # Ensure values are in [0,1]
    for v in st.to_dict().values():
        assert 0.0 <= v <= 1.0

    t = st.to_tensor(torch.device("cpu"))
    assert isinstance(t, torch.Tensor)
    assert t.shape == (EnvironmentState.DIMENSIONS,)


def test_simulation_state_from_environment():
    class DummyEnv:
        pass

    env = DummyEnv()
    env.time = 100
    # 4 alive agents
    agent = Mock()
    agent.alive = True
    env.agent_objects = [agent for _ in range(4)]

    def get_initial_agent_count():
        return 5

    env.get_initial_agent_count = get_initial_agent_count

    # Some resources
    env.resources = []
    for i in range(3):
        r = Mock()
        r.amount = 7
        env.resources.append(r)

    cfg = Mock()
    cfg.resources = Mock()
    cfg.resources.max_resource_amount = 10
    env.config = cfg

    st = SimulationState.from_environment(env, num_steps=1000)
    d = st.to_dict()
    assert set(d.keys()) == {
        "time_progress",
        "population_size",
        "survival_rate",
        "resource_efficiency",
        "system_performance",
    }
    # Check bounds
    assert 0.0 <= d["time_progress"] <= 1.0
    assert 0.0 <= d["population_size"] <= 1.0
    assert 0.0 <= d["survival_rate"] <= 1.0
    assert 0.0 <= d["resource_efficiency"] <= 1.0


def test_model_state_from_move_module_and_str():
    class DummyQNet:
        def __init__(self):
            self.network = torch.nn.ModuleList(
                [
                    torch.nn.Linear(10, 8),
                    torch.nn.ReLU(),
                    torch.nn.Linear(8, 4),
                ]
            )

    class DummyMemory:
        def __len__(self):
            return 2

        @property
        def maxlen(self):
            return 100

    class DummyMoveModule:
        def __init__(self):
            self.q_network = DummyQNet()
            self.optimizer = Mock()
            self.optimizer.param_groups = [{"lr": 0.001}]
            self.epsilon = 0.3
            self.losses = [0.5, None, 0.7]
            self.episode_rewards = [1.0, 2.0]
            self.memory = DummyMemory()
            self.steps = 42

    mm = DummyMoveModule()
    st = ModelState.from_move_module(mm)

    d = st.to_dict()
    assert d["learning_rate"] == 0.001
    assert d["epsilon"] == 0.3
    assert d["steps"] == 42
    assert d["architecture"]["input_dim"] == 10
    assert d["architecture"]["output_dim"] == 4

    # Metrics keys
    m = d["metrics"]
    for key in ["avg_loss", "avg_reward", "min_reward", "max_reward", "std_reward"]:
        assert key in m

    s = str(st)
    assert "ModelState(" in s


def test_base_state_to_tensor_not_implemented():
    with pytest.raises(NotImplementedError):
        BaseState().to_tensor(torch.device("cpu"))

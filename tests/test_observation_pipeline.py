"""Tests for the observation-to-policy pipeline integration.

Covers the complete flow from:
1. Agent observation generation
2. DecisionModule processing (fallback algorithm)
3. Action selection with curriculum support (action masking)
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from farm.core.channels import get_channel_registry
from farm.core.decision.config import DecisionConfig
from farm.core.decision.decision import DecisionModule
from farm.core.observations import AgentObservation, ObservationConfig

NUM_ACTIONS = 7
OBSERVATION_RADIUS = 6


@pytest.fixture()
def decision_module():
    """DecisionModule wired to the deterministic fallback algorithm."""
    return DecisionModule(
        agent=SimpleNamespace(agent_id="test_agent"),
        action_space=SimpleNamespace(n=NUM_ACTIONS),
        observation_space=SimpleNamespace(shape=(13, 13, 13)),
        config=DecisionConfig(algorithm_type="fallback"),
    )


def test_channel_registry_initialized_with_core_channels():
    registry = get_channel_registry()
    assert registry.num_channels == 13


def test_observation_tensor_has_expected_shape():
    config = ObservationConfig(R=OBSERVATION_RADIUS)
    registry = get_channel_registry()

    obs = AgentObservation(config)
    tensor = obs.tensor()

    side = 2 * OBSERVATION_RADIUS + 1
    assert tensor.shape == (registry.num_channels, side, side)


def test_decision_module_initialization_with_fallback(decision_module):
    assert decision_module.algorithm is not None
    assert decision_module.num_actions == NUM_ACTIONS
    assert decision_module.state_dim > 0
    assert decision_module.observation_shape == (13, 13, 13)


def test_observation_to_action_flow(decision_module):
    config = ObservationConfig(R=OBSERVATION_RADIUS)
    obs = AgentObservation(config)

    # Populate a few channels so the observation is non-trivial.
    obs._store_sparse_point(0, 6, 6, 1.0)  # Agent health at center
    obs._store_sparse_point(1, 5, 6, 0.8)  # Ally nearby
    obs._store_sparse_point(3, 4, 4, 0.5)  # Resource nearby

    observation_tensor = obs.tensor()
    assert torch.count_nonzero(observation_tensor) > 0

    action = decision_module.decide_action(observation_tensor.numpy())

    assert isinstance(action, int)
    assert 0 <= action < NUM_ACTIONS


def test_curriculum_action_masking_restricts_selection(decision_module):
    enabled_actions = [0, 2, 4]
    test_obs = np.random.rand(1, 13, 13, 13).astype(np.float32)

    # decide_action returns an index into enabled_actions when a curriculum
    # mask is active, so it must stay within [0, len(enabled_actions)).
    for _ in range(20):
        action = decision_module.decide_action(test_obs, enabled_actions=enabled_actions)
        assert 0 <= action < len(enabled_actions)

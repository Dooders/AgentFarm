"""
Unit tests for IAgentBehavior interface.

Tests verify:
- Interface can be implemented correctly
- Behavior can execute turns
- State management works
"""

import pytest
from unittest.mock import Mock
from farm.core.agent.behaviors.base_behavior import IAgentBehavior


class ConcreteBehavior(IAgentBehavior):
    """Concrete implementation for testing."""

    def __init__(self):
        self.execute_turn_called = False
        self.last_agent = None

    def execute_turn(self, agent) -> None:
        self.execute_turn_called = True
        self.last_agent = agent


class TestIAgentBehavior:
    """Tests for IAgentBehavior interface."""

    def test_execute_turn_is_called(self):
        """Test that execute_turn can be called with agent."""
        behavior = ConcreteBehavior()
        mock_agent = Mock()

        behavior.execute_turn(mock_agent)

        assert behavior.execute_turn_called
        assert behavior.last_agent == mock_agent

    def test_reset_default_implementation(self):
        """Test default reset does nothing."""
        behavior = ConcreteBehavior()
        # Should not raise
        behavior.reset()

    def test_get_state_returns_empty_dict(self):
        """Test default get_state returns empty dict."""
        behavior = ConcreteBehavior()
        assert behavior.get_state() == {}

    def test_load_state_accepts_dict(self):
        """Test default load_state accepts dict without error."""
        behavior = ConcreteBehavior()
        # Should not raise
        behavior.load_state({"key": "value"})


class StatefulBehavior(IAgentBehavior):
    """Behavior with state for testing serialization."""

    def __init__(self):
        self._episode_count = 0
        self._total_reward = 0.0

    def execute_turn(self, agent) -> None:
        pass

    def reset(self) -> None:
        self._episode_count += 1
        self._total_reward = 0.0

    def get_state(self) -> dict:
        return {
            "episode_count": self._episode_count,
            "total_reward": self._total_reward,
        }

    def load_state(self, state: dict) -> None:
        self._episode_count = state.get("episode_count", 0)
        self._total_reward = state.get("total_reward", 0.0)


class TestStatefulBehavior:
    """Tests for behavior with state."""

    def test_reset_updates_state(self):
        """Test reset updates internal state."""
        behavior = StatefulBehavior()
        behavior._total_reward = 100.0

        behavior.reset()

        assert behavior._episode_count == 1
        assert behavior._total_reward == 0.0

    def test_get_state_returns_custom_data(self):
        """Test get_state returns behavior data."""
        behavior = StatefulBehavior()
        behavior._episode_count = 5
        behavior._total_reward = 123.45

        state = behavior.get_state()

        assert state == {"episode_count": 5, "total_reward": 123.45}

    def test_load_state_restores_data(self):
        """Test load_state restores behavior data."""
        behavior = StatefulBehavior()
        behavior.load_state({"episode_count": 10, "total_reward": 500.0})

        assert behavior._episode_count == 10
        assert behavior._total_reward == 500.0

    def test_round_trip_serialization(self):
        """Test save/load preserves state."""
        behavior1 = StatefulBehavior()
        behavior1._episode_count = 7
        behavior1._total_reward = 999.99

        # Save state
        state = behavior1.get_state()

        # Load into new behavior
        behavior2 = StatefulBehavior()
        behavior2.load_state(state)

        assert behavior2._episode_count == 7
        assert behavior2._total_reward == 999.99
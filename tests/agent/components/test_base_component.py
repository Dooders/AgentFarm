"""
Unit tests for IAgentComponent interface.

Tests verify:
- Interface can be implemented correctly
- Default implementations work as expected
- Lifecycle methods are called at appropriate times
"""

import pytest
from unittest.mock import Mock
from farm.core.agent.components.base import IAgentComponent


class ConcreteComponent(IAgentComponent):
    """Concrete implementation for testing."""

    def __init__(self):
        self.step_start_called = False
        self.step_end_called = False
        self.terminate_called = False
        self._agent = None

    @property
    def name(self) -> str:
        return "test_component"


class TestIAgentComponent:
    """Tests for IAgentComponent interface."""

    def test_component_has_name(self):
        """Test that component has a name property."""
        component = ConcreteComponent()
        assert component.name == "test_component"

    def test_attach_stores_agent_reference(self):
        """Test that attach() stores agent reference."""
        component = ConcreteComponent()
        mock_agent = Mock()

        component.attach(mock_agent)
        assert component._agent == mock_agent

    def test_on_step_start_default_implementation(self):
        """Test default on_step_start does nothing."""
        component = ConcreteComponent()
        # Should not raise
        component.on_step_start()

    def test_on_step_end_default_implementation(self):
        """Test default on_step_end does nothing."""
        component = ConcreteComponent()
        # Should not raise
        component.on_step_end()

    def test_on_terminate_default_implementation(self):
        """Test default on_terminate does nothing."""
        component = ConcreteComponent()
        # Should not raise
        component.on_terminate()

    def test_get_state_returns_empty_dict(self):
        """Test default get_state returns empty dict."""
        component = ConcreteComponent()
        assert component.get_state() == {}

    def test_load_state_accepts_dict(self):
        """Test default load_state accepts dict without error."""
        component = ConcreteComponent()
        # Should not raise
        component.load_state({"key": "value"})


class TestComponentWithState(IAgentComponent):
    """Test component that overrides state methods."""

    def __init__(self):
        self._counter = 0

    @property
    def name(self) -> str:
        return "stateful_component"

    def get_state(self) -> dict:
        return {"counter": self._counter}

    def load_state(self, state: dict) -> None:
        self._counter = state.get("counter", 0)


class TestStatefulComponent:
    """Tests for component with state serialization."""

    def test_get_state_returns_custom_data(self):
        """Test custom get_state returns component data."""
        component = TestComponentWithState()
        component._counter = 42

        state = component.get_state()
        assert state == {"counter": 42}

    def test_load_state_restores_data(self):
        """Test load_state restores component data."""
        component = TestComponentWithState()
        component.load_state({"counter": 99})

        assert component._counter == 99

    def test_round_trip_serialization(self):
        """Test save/load preserves state."""
        component1 = TestComponentWithState()
        component1._counter = 123

        # Save state
        state = component1.get_state()

        # Load into new component
        component2 = TestComponentWithState()
        component2.load_state(state)

        assert component2._counter == 123
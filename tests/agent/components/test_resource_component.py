"""
Unit tests for ResourceComponent.

Tests verify:
- Resource tracking
- Addition and consumption
- Starvation mechanics
- Death triggering
"""

import pytest
from unittest.mock import Mock
from farm.core.agent.config.agent_config import ResourceConfig
from farm.core.agent.components.resource import ResourceComponent


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = Mock()
    agent.agent_id = "test_agent"
    agent.alive = True
    agent.terminate = Mock()
    return agent


@pytest.fixture
def resource_component(mock_agent):
    """Create a ResourceComponent attached to mock agent."""
    config = ResourceConfig(
        base_consumption_rate=1,
        starvation_threshold=3
    )
    component = ResourceComponent(initial_resources=100, config=config)
    component.attach(mock_agent)
    return component


class TestResourceComponent:
    """Tests for ResourceComponent."""
    
    def test_component_name(self, resource_component):
        """Test component has correct name."""
        assert resource_component.name == "resource"
    
    def test_initial_resources(self, resource_component):
        """Test initial resource level."""
        assert resource_component.level == 100
    
    def test_add_resources(self, resource_component):
        """Test adding resources."""
        resource_component.add(50)
        assert resource_component.level == 150
    
    def test_add_negative_resources(self, resource_component):
        """Test adding negative resources (consumption)."""
        resource_component.add(-20)
        assert resource_component.level == 80
    
    def test_consume_sufficient_resources(self, resource_component):
        """Test consuming resources when sufficient."""
        result = resource_component.consume(30)
        
        assert result is True
        assert resource_component.level == 70
    
    def test_consume_insufficient_resources(self, resource_component):
        """Test consuming resources when insufficient."""
        result = resource_component.consume(200)
        
        assert result is False
        assert resource_component.level == 100  # Unchanged
    
    def test_has_resources_true(self, resource_component):
        """Test has_resources when sufficient."""
        assert resource_component.has_resources(50) is True
        assert resource_component.has_resources(100) is True
    
    def test_has_resources_false(self, resource_component):
        """Test has_resources when insufficient."""
        assert resource_component.has_resources(150) is False
    
    def test_set_level(self, resource_component):
        """Test setting resource level directly."""
        resource_component.set_level(75)
        assert resource_component.level == 75
    
    def test_is_starving_false(self, resource_component):
        """Test is_starving when agent has resources."""
        assert resource_component.is_starving is False
    
    def test_is_starving_true(self, resource_component):
        """Test is_starving when resources depleted."""
        resource_component.set_level(0)
        assert resource_component.is_starving is True
    
    def test_starvation_steps(self, resource_component):
        """Test starvation step counter."""
        assert resource_component.starvation_steps == 0
    
    def test_on_step_end_normal(self, resource_component, mock_agent):
        """Test step end with sufficient resources."""
        resource_component.on_step_end()
        
        # Should consume base rate (1)
        assert resource_component.level == 99
        assert resource_component.starvation_steps == 0
        mock_agent.terminate.assert_not_called()
    
    def test_on_step_end_starvation(self, resource_component, mock_agent):
        """Test step end triggers starvation counter."""
        resource_component.set_level(0)
        
        resource_component.on_step_end()
        
        assert resource_component.level == -1  # Consumed 1
        assert resource_component.starvation_steps == 1
        mock_agent.terminate.assert_not_called()  # Not dead yet
    
    def test_on_step_end_death(self, resource_component, mock_agent):
        """Test step end triggers death after starvation threshold."""
        resource_component.set_level(0)
        
        # Starve for threshold steps
        for _ in range(3):
            resource_component.on_step_end()
        
        assert resource_component.starvation_steps == 3
        mock_agent.terminate.assert_called_once()
    
    def test_starvation_counter_reset(self, resource_component):
        """Test starvation counter resets when resources gained."""
        resource_component.set_level(0)
        resource_component.on_step_end()
        assert resource_component.starvation_steps == 1
        
        # Add resources
        resource_component.add(10)
        
        # Counter should reset
        assert resource_component.starvation_steps == 0
    
    def test_get_state(self, resource_component):
        """Test state serialization."""
        resource_component.set_level(75)
        resource_component.on_step_end()  # Increment starvation counter
        
        state = resource_component.get_state()
        
        assert state["resources"] == 74  # 75 - 1
        assert state["starvation_counter"] == 0  # Has resources, so 0
    
    def test_load_state(self, resource_component):
        """Test state deserialization."""
        state = {
            "resources": 50,
            "starvation_counter": 2
        }
        
        resource_component.load_state(state)
        
        assert resource_component.level == 50
        assert resource_component.starvation_steps == 2
    
    def test_round_trip_serialization(self, resource_component):
        """Test save/load preserves state."""
        resource_component.set_level(42)
        resource_component._starvation_counter = 1
        
        state = resource_component.get_state()
        
        new_component = ResourceComponent(
            initial_resources=0,
            config=ResourceConfig()
        )
        new_component.load_state(state)
        
        assert new_component.level == 42
        assert new_component.starvation_steps == 1
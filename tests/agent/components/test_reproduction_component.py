"""
Unit tests for ReproductionComponent.

Tests verify:
- Reproduction requirements checking
- Offspring creation
- Resource costs
- Generation tracking
"""

import pytest
from unittest.mock import Mock
from farm.core.agent.config.agent_config import ReproductionConfig, ResourceConfig
from farm.core.agent.components.reproduction import ReproductionComponent
from farm.core.agent.components.resource import ResourceComponent


@pytest.fixture
def mock_lifecycle_service():
    """Create a mock lifecycle service."""
    service = Mock()
    service.get_next_agent_id = Mock(return_value="offspring_001")
    service.add_agent = Mock()
    return service


@pytest.fixture
def mock_agent(mock_lifecycle_service):
    """Create a mock agent with resource component."""
    agent = Mock()
    agent.agent_id = "parent_agent"
    agent.alive = True
    agent._lifecycle_service = mock_lifecycle_service
    
    # State manager
    state_manager = Mock()
    state_manager.position = (50.0, 50.0)
    state_manager.generation = 1
    agent.state_manager = state_manager
    
    # Resource component
    resource = ResourceComponent(
        initial_resources=100,
        config=ResourceConfig()
    )
    resource.attach(agent)
    
    def get_component(name):
        if name == "resource":
            return resource
        return None
    
    agent.get_component = get_component
    
    return agent


@pytest.fixture
def reproduction_component(mock_agent, mock_lifecycle_service):
    """Create a ReproductionComponent attached to mock agent."""
    config = ReproductionConfig(
        offspring_cost=10,
        offspring_initial_resources=20,
        reproduction_threshold=50
    )
    component = ReproductionComponent(
        config=config,
        lifecycle_service=mock_lifecycle_service
    )
    component.attach(mock_agent)
    return component


class TestReproductionComponent:
    """Tests for ReproductionComponent."""
    
    def test_component_name(self, reproduction_component):
        """Test component has correct name."""
        assert reproduction_component.name == "reproduction"
    
    def test_offspring_cost(self, reproduction_component):
        """Test offspring cost property."""
        assert reproduction_component.offspring_cost == 10
    
    def test_reproduction_count_initial(self, reproduction_component):
        """Test initial reproduction count is zero."""
        assert reproduction_component.reproduction_count == 0
    
    def test_can_reproduce_sufficient_resources(self, reproduction_component):
        """Test can_reproduce when agent has enough resources."""
        assert reproduction_component.can_reproduce() is True
    
    def test_can_reproduce_insufficient_resources(self, reproduction_component, mock_agent):
        """Test can_reproduce when agent lacks resources."""
        resource = mock_agent.get_component("resource")
        resource.set_level(30)  # Below threshold of 50
        
        assert reproduction_component.can_reproduce() is False
    
    def test_can_reproduce_dead_agent(self, reproduction_component, mock_agent):
        """Test can_reproduce returns False for dead agent."""
        mock_agent.alive = False
        
        assert reproduction_component.can_reproduce() is False
    
    def test_can_reproduce_no_resource_component(self, reproduction_component, mock_agent):
        """Test can_reproduce when agent has no resource component."""
        mock_agent.get_component = Mock(return_value=None)
        
        assert reproduction_component.can_reproduce() is False
    
    def test_reproduce_success(self, reproduction_component, mock_agent, mock_lifecycle_service):
        """Test successful reproduction."""
        result = reproduction_component.reproduce()
        
        assert result["success"] is True
        assert result["offspring_id"] == "offspring_001"
        assert result["cost"] == 10
        assert result["offspring_resources"] == 20
        
        # Check resource consumption
        resource = mock_agent.get_component("resource")
        assert resource.level == 90  # 100 - 10
        
        # Check reproduction count
        assert reproduction_component.reproduction_count == 1
        
        # Lifecycle service should be called
        mock_lifecycle_service.get_next_agent_id.assert_called_once()
    
    def test_reproduce_insufficient_resources(self, reproduction_component, mock_agent):
        """Test reproduction fails with insufficient resources."""
        resource = mock_agent.get_component("resource")
        resource.set_level(30)  # Below threshold
        
        result = reproduction_component.reproduce()
        
        assert result["success"] is False
        assert "Insufficient resources" in result["error"]
        assert reproduction_component.reproduction_count == 0
    
    def test_reproduce_dead_agent(self, reproduction_component, mock_agent):
        """Test reproduction fails for dead agent."""
        mock_agent.alive = False
        
        result = reproduction_component.reproduce()
        
        assert result["success"] is False
        assert "Dead agents" in result["error"]
    
    def test_reproduce_no_resource_component(self, reproduction_component, mock_agent):
        """Test reproduction fails without resource component."""
        mock_agent.get_component = Mock(return_value=None)
        
        result = reproduction_component.reproduce()
        
        assert result["success"] is False
        assert "no resource component" in result["error"]
    
    def test_reproduce_multiple_times(self, reproduction_component):
        """Test reproducing multiple times increments counter."""
        reproduction_component.reproduce()
        reproduction_component.reproduce()
        reproduction_component.reproduce()
        
        assert reproduction_component.reproduction_count == 3
    
    def test_get_reproduction_info(self, reproduction_component, mock_agent):
        """Test getting reproduction info."""
        info = reproduction_component.get_reproduction_info()
        
        assert info["can_reproduce"] is True
        assert info["required_resources"] == 50
        assert info["current_resources"] == 100
        assert info["offspring_cost"] == 10
        assert info["offspring_initial_resources"] == 20
        assert info["reproduction_count"] == 0
    
    def test_get_reproduction_info_after_reproduction(self, reproduction_component, mock_agent):
        """Test reproduction info updates after reproduction."""
        reproduction_component.reproduce()
        
        info = reproduction_component.get_reproduction_info()
        
        assert info["current_resources"] == 90  # After cost
        assert info["reproduction_count"] == 1
    
    def test_get_state(self, reproduction_component):
        """Test state serialization."""
        reproduction_component.reproduce()
        reproduction_component.reproduce()
        
        state = reproduction_component.get_state()
        
        assert state["reproduction_count"] == 2
    
    def test_load_state(self, reproduction_component):
        """Test state deserialization."""
        state = {"reproduction_count": 5}
        
        reproduction_component.load_state(state)
        
        assert reproduction_component.reproduction_count == 5
    
    def test_round_trip_serialization(self, reproduction_component):
        """Test save/load preserves state."""
        reproduction_component._reproduction_count = 7
        
        state = reproduction_component.get_state()
        
        new_component = ReproductionComponent(
            config=ReproductionConfig()
        )
        new_component.load_state(state)
        
        assert new_component.reproduction_count == 7
    
    def test_component_without_agent(self, mock_lifecycle_service):
        """Test component methods without attached agent."""
        config = ReproductionConfig()
        component = ReproductionComponent(
            config=config,
            lifecycle_service=mock_lifecycle_service
        )
        
        # Methods should fail gracefully
        assert component.can_reproduce() is False
        
        result = component.reproduce()
        assert result["success"] is False
        assert "Component not attached to agent" in result["error"]
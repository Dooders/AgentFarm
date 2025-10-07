"""
Unit tests for PerceptionComponent.

Tests verify:
- Nearby entity queries
- Perception grid creation
- Visibility checks
- Entity counting
"""

import pytest
import numpy as np
from unittest.mock import Mock
from farm.core.agent.config.agent_config import PerceptionConfig
from farm.core.agent.components.perception import PerceptionComponent


@pytest.fixture
def mock_spatial_service():
    """Create a mock spatial service."""
    service = Mock()
    service.get_nearby = Mock(return_value={})
    service.get_nearest = Mock(return_value={})
    return service


@pytest.fixture
def mock_agent():
    """Create a mock agent with state manager."""
    agent = Mock()
    agent.agent_id = "test_agent"
    
    state_manager = Mock()
    state_manager.position = (50.0, 50.0)
    agent.state_manager = state_manager
    
    return agent


@pytest.fixture
def perception_component(mock_spatial_service, mock_agent):
    """Create a PerceptionComponent attached to mock agent."""
    config = PerceptionConfig(perception_radius=5)
    component = PerceptionComponent(mock_spatial_service, config)
    component.attach(mock_agent)
    return component


class TestPerceptionComponent:
    """Tests for PerceptionComponent."""
    
    def test_component_name(self, perception_component):
        """Test component has correct name."""
        assert perception_component.name == "perception"
    
    def test_radius(self, perception_component):
        """Test perception radius property."""
        assert perception_component.radius == 5
    
    def test_grid_size(self, perception_component):
        """Test grid size calculation."""
        assert perception_component.grid_size == 11  # 2*5 + 1
    
    def test_get_nearby_entities_all_types(self, perception_component, mock_spatial_service):
        """Test getting nearby entities of all types."""
        mock_spatial_service.get_nearby.return_value = {
            "agents": [Mock(), Mock()],
            "resources": [Mock()]
        }
        
        result = perception_component.get_nearby_entities()
        
        mock_spatial_service.get_nearby.assert_called_once_with(
            position=(50.0, 50.0),
            radius=5,
            index_names=None
        )
        assert len(result["agents"]) == 2
        assert len(result["resources"]) == 1
    
    def test_get_nearby_entities_specific_types(self, perception_component, mock_spatial_service):
        """Test getting nearby entities of specific types."""
        mock_spatial_service.get_nearby.return_value = {
            "resources": [Mock(), Mock(), Mock()]
        }
        
        result = perception_component.get_nearby_entities(["resources"])
        
        mock_spatial_service.get_nearby.assert_called_once_with(
            position=(50.0, 50.0),
            radius=5,
            index_names=["resources"]
        )
        assert len(result["resources"]) == 3
    
    def test_get_nearby_entities_custom_radius(self, perception_component, mock_spatial_service):
        """Test getting nearby entities with custom radius."""
        perception_component.get_nearby_entities(radius=10.0)
        
        mock_spatial_service.get_nearby.assert_called_once_with(
            position=(50.0, 50.0),
            radius=10.0,
            index_names=None
        )
    
    def test_get_nearest_entity(self, perception_component, mock_spatial_service):
        """Test getting nearest entity."""
        nearest_resource = Mock()
        mock_spatial_service.get_nearest.return_value = {
            "resources": nearest_resource
        }
        
        result = perception_component.get_nearest_entity(["resources"])
        
        mock_spatial_service.get_nearest.assert_called_once_with(
            position=(50.0, 50.0),
            index_names=["resources"]
        )
        assert result["resources"] == nearest_resource
    
    def test_create_perception_grid_empty(self, perception_component, mock_spatial_service):
        """Test creating perception grid with no entities."""
        mock_spatial_service.get_nearby.return_value = {
            "resources": [],
            "agents": []
        }
        
        grid = perception_component.create_perception_grid()
        
        assert grid.shape == (11, 11)
        assert grid.dtype == np.int8
        assert np.all(grid == 0)  # All empty
    
    def test_create_perception_grid_with_resources(self, perception_component, mock_spatial_service):
        """Test creating perception grid with resources."""
        # Create mock resource at position (51, 52) - relative to agent at (50, 50)
        resource = Mock()
        resource.position = (51.0, 52.0)
        
        mock_spatial_service.get_nearby.return_value = {
            "resources": [resource],
            "agents": []
        }
        
        grid = perception_component.create_perception_grid()
        
        # Resource should be at grid position (6, 7) - center is (5, 5)
        # grid[y, x] so grid[7, 6]
        assert grid[7, 6] == 1  # Resource marker
    
    def test_create_perception_grid_with_agents(self, perception_component, mock_spatial_service, mock_agent):
        """Test creating perception grid with other agents."""
        # Create mock agent at position (48, 49)
        other_agent = Mock()
        other_agent.agent_id = "other_agent"
        other_agent.position = (48.0, 49.0)
        
        mock_spatial_service.get_nearby.return_value = {
            "resources": [],
            "agents": [mock_agent, other_agent]  # Include self and other
        }
        
        grid = perception_component.create_perception_grid()
        
        # Other agent should be at grid position (3, 4) - center is (5, 5)
        assert grid[4, 3] == 2  # Agent marker
        # Self should not be marked
        assert grid[5, 5] == 0
    
    def test_can_see_within_radius(self, perception_component):
        """Test visibility check for position within radius."""
        # Position 3 units away
        assert perception_component.can_see((53.0, 50.0)) is True
    
    def test_can_see_beyond_radius(self, perception_component):
        """Test visibility check for position beyond radius."""
        # Position 10 units away
        assert perception_component.can_see((60.0, 50.0)) is False
    
    def test_count_nearby(self, perception_component, mock_spatial_service):
        """Test counting nearby entities."""
        mock_spatial_service.get_nearby.return_value = {
            "resources": [Mock(), Mock(), Mock()]
        }
        
        count = perception_component.count_nearby("resources")
        
        assert count == 3
    
    def test_count_nearby_none(self, perception_component, mock_spatial_service):
        """Test counting when no entities found."""
        mock_spatial_service.get_nearby.return_value = {
            "resources": []
        }
        
        count = perception_component.count_nearby("resources")
        
        assert count == 0
    
    def test_get_state(self, perception_component):
        """Test state serialization."""
        state = perception_component.get_state()
        assert state["perception_radius"] == 5
    
    def test_component_without_agent(self, mock_spatial_service):
        """Test component methods without attached agent."""
        config = PerceptionConfig(perception_radius=5)
        component = PerceptionComponent(mock_spatial_service, config)
        
        # Methods should return safe defaults
        assert component.get_nearby_entities() == {}
        assert component.get_nearest_entity() == {}
        assert component.can_see((10.0, 10.0)) is False
        
        grid = component.create_perception_grid()
        assert grid.shape == (11, 11)
        assert np.all(grid == 0)
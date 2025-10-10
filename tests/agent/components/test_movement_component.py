"""
Unit tests for MovementComponent.

Tests verify:
- Movement within max_movement constraint
- Various movement methods (move_to, move_by, random_move)
- Distance calculations
- Reachability checks
"""

import pytest
import math
from unittest.mock import Mock
from farm.core.agent.config.agent_config import MovementConfig
from farm.core.agent.components.movement import MovementComponent


@pytest.fixture
def mock_agent():
    """Create a mock agent with state manager."""
    agent = Mock()
    agent.agent_id = "test_agent"
    
    # Mock state manager
    state_manager = Mock()
    state_manager.position = (0.0, 0.0)
    
    def set_position(pos):
        state_manager.position = pos
    
    state_manager.set_position = set_position
    agent.state_manager = state_manager
    
    return agent


@pytest.fixture
def movement_component(mock_agent):
    """Create a MovementComponent attached to mock agent."""
    config = MovementConfig(max_movement=10.0)
    component = MovementComponent(config)
    component.attach(mock_agent)
    return component


class TestMovementComponent:
    """Tests for MovementComponent."""
    
    def test_component_name(self, movement_component):
        """Test component has correct name."""
        assert movement_component.name == "movement"
    
    def test_max_movement(self, movement_component):
        """Test max_movement property."""
        assert movement_component.max_movement == 10.0
    
    def test_move_to_within_range(self, movement_component, mock_agent):
        """Test moving to position within max_movement."""
        result = movement_component.move_to((5.0, 5.0))
        
        assert result is True
        assert mock_agent.state_manager.position == (5.0, 5.0)
    
    def test_move_to_beyond_range(self, movement_component, mock_agent):
        """Test moving to position beyond max_movement."""
        # Try to move to (20, 20) which is 28.28 units away
        result = movement_component.move_to((20.0, 20.0))
        
        assert result is True
        # Should move as far as possible (10 units) in that direction
        pos = mock_agent.state_manager.position
        distance = math.sqrt(pos[0]**2 + pos[1]**2)
        assert abs(distance - 10.0) < 0.01  # Within floating point tolerance
    
    def test_move_to_same_position(self, movement_component, mock_agent):
        """Test moving to current position."""
        result = movement_component.move_to((0.0, 0.0))
        
        assert result is True
        assert mock_agent.state_manager.position == (0.0, 0.0)
    
    def test_move_by_within_range(self, movement_component, mock_agent):
        """Test relative movement within range."""
        result = movement_component.move_by(3.0, 4.0)  # 5 units total
        
        assert result is True
        assert mock_agent.state_manager.position == (3.0, 4.0)
    
    def test_move_by_beyond_range(self, movement_component, mock_agent):
        """Test relative movement beyond range."""
        result = movement_component.move_by(8.0, 8.0)  # 11.3 units, exceeds 10
        
        assert result is True
        pos = mock_agent.state_manager.position
        distance = math.sqrt(pos[0]**2 + pos[1]**2)
        assert abs(distance - 10.0) < 0.01
    
    def test_random_move(self, movement_component, mock_agent):
        """Test random movement."""
        result = movement_component.random_move()
        
        assert result is True
        pos = mock_agent.state_manager.position
        distance = math.sqrt(pos[0]**2 + pos[1]**2)
        assert distance <= 10.0  # Within max_movement
    
    def test_random_move_with_distance(self, movement_component, mock_agent):
        """Test random movement with specified distance."""
        result = movement_component.random_move(distance=5.0)
        
        assert result is True
        pos = mock_agent.state_manager.position
        distance = math.sqrt(pos[0]**2 + pos[1]**2)
        assert abs(distance - 5.0) < 0.01
    
    def test_move_toward_entity(self, movement_component, mock_agent):
        """Test moving toward an entity."""
        target_pos = (30.0, 40.0)  # 50 units away
        result = movement_component.move_toward_entity(target_pos)
        
        assert result is True
        pos = mock_agent.state_manager.position
        # Should move 10 units toward target
        distance_traveled = math.sqrt(pos[0]**2 + pos[1]**2)
        assert abs(distance_traveled - 10.0) < 0.01
    
    def test_move_toward_entity_with_stop_distance(self, movement_component, mock_agent):
        """Test moving toward entity with stop distance."""
        target_pos = (15.0, 0.0)  # 15 units away
        result = movement_component.move_toward_entity(target_pos, stop_distance=5.0)
        
        assert result is True
        pos = mock_agent.state_manager.position
        # Should move to 10 units from start (15 - 5 = 10, but limited by max_movement)
        assert abs(pos[0] - 10.0) < 0.01
    
    def test_move_toward_entity_already_close(self, movement_component, mock_agent):
        """Test moving toward entity when already within stop distance."""
        target_pos = (3.0, 0.0)  # 3 units away
        result = movement_component.move_toward_entity(target_pos, stop_distance=5.0)
        
        assert result is True
        # Should not move (already within stop distance)
        assert mock_agent.state_manager.position == (0.0, 0.0)
    
    def test_can_reach_within_range(self, movement_component):
        """Test can_reach for position within range."""
        assert movement_component.can_reach((5.0, 5.0)) is True
    
    def test_can_reach_beyond_range(self, movement_component):
        """Test can_reach for position beyond range."""
        assert movement_component.can_reach((20.0, 20.0)) is False
    
    def test_distance_to(self, movement_component):
        """Test distance calculation."""
        distance = movement_component.distance_to((3.0, 4.0))
        assert abs(distance - 5.0) < 0.01  # 3-4-5 triangle
    
    def test_get_state(self, movement_component):
        """Test state serialization."""
        state = movement_component.get_state()
        assert state["max_movement"] == 10.0
    
    def test_component_without_agent(self):
        """Test component methods without attached agent."""
        config = MovementConfig(max_movement=10.0)
        component = MovementComponent(config)
        
        # Methods should return False or safe defaults
        assert component.move_to((5.0, 5.0)) is False
        assert component.move_by(1.0, 1.0) is False
        assert component.random_move() is False
        assert component.can_reach((5.0, 5.0)) is False
        assert component.distance_to((5.0, 5.0)) == float('inf')
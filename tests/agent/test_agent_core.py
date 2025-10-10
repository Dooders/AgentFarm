"""
Integration tests for AgentCore.

Tests verify:
- Agent creation and initialization
- Component management
- Behavior execution
- Lifecycle management
- State serialization
"""

import pytest
from unittest.mock import Mock
from farm.core.agent.core import AgentCore
from farm.core.agent.components import (
    MovementComponent,
    ResourceComponent,
    CombatComponent,
)
from farm.core.agent.behaviors import DefaultAgentBehavior
from farm.core.agent.config.agent_config import (
    AgentConfig,
    MovementConfig,
    ResourceConfig,
    CombatConfig,
)


@pytest.fixture
def mock_services():
    """Create mock services."""
    spatial_service = Mock()
    spatial_service.get_nearby = Mock(return_value={})
    spatial_service.get_nearest = Mock(return_value={})
    spatial_service.mark_positions_dirty = Mock()
    
    time_service = Mock()
    time_service.current_time = Mock(return_value=100)
    
    lifecycle_service = Mock()
    
    return {
        "spatial": spatial_service,
        "time": time_service,
        "lifecycle": lifecycle_service,
    }


@pytest.fixture
def basic_agent(mock_services):
    """Create a basic agent with default components."""
    config = AgentConfig()
    
    components = [
        MovementComponent(config.movement),
        ResourceComponent(100, config.resource),
        CombatComponent(config.combat),
    ]
    
    behavior = DefaultAgentBehavior()
    
    agent = AgentCore(
        agent_id="test_agent_001",
        position=(50.0, 50.0),
        spatial_service=mock_services["spatial"],
        behavior=behavior,
        components=components,
        time_service=mock_services["time"],
        lifecycle_service=mock_services["lifecycle"],
    )
    
    return agent


class TestAgentCoreInitialization:
    """Tests for agent initialization."""
    
    def test_agent_has_identity(self, basic_agent):
        """Test agent has correct identity."""
        assert basic_agent.agent_id == "test_agent_001"
        assert basic_agent.alive is True
    
    def test_agent_has_position(self, basic_agent):
        """Test agent has correct position."""
        assert basic_agent.position == (50.0, 50.0)
        assert basic_agent.position_3d == (50.0, 50.0, 0.0)
    
    def test_agent_has_state_manager(self, basic_agent):
        """Test agent has state manager."""
        assert basic_agent.state_manager is not None
        assert basic_agent.state_manager.position == (50.0, 50.0)
    
    def test_agent_has_components(self, basic_agent):
        """Test agent has all expected components."""
        assert basic_agent.has_component("movement")
        assert basic_agent.has_component("resource")
        assert basic_agent.has_component("combat")
    
    def test_birth_time_set(self, basic_agent, mock_services):
        """Test birth time is set from time service."""
        assert basic_agent.state_manager.birth_time == 100


class TestComponentManagement:
    """Tests for component management."""
    
    def test_get_component(self, basic_agent):
        """Test getting component by name."""
        movement = basic_agent.get_component("movement")
        assert movement is not None
        assert movement.name == "movement"
    
    def test_get_nonexistent_component(self, basic_agent):
        """Test getting component that doesn't exist."""
        component = basic_agent.get_component("nonexistent")
        assert component is None
    
    def test_has_component(self, basic_agent):
        """Test checking component existence."""
        assert basic_agent.has_component("movement") is True
        assert basic_agent.has_component("nonexistent") is False
    
    def test_add_component(self, basic_agent, mock_services):
        """Test adding component after initialization."""
        from farm.core.agent.components import PerceptionComponent
        from farm.core.agent.config.agent_config import PerceptionConfig
        
        perception = PerceptionComponent(
            mock_services["spatial"],
            PerceptionConfig()
        )
        
        basic_agent.add_component(perception)
        
        assert basic_agent.has_component("perception")
        assert basic_agent.get_component("perception") == perception
    
    def test_remove_component(self, basic_agent):
        """Test removing component."""
        movement = basic_agent.remove_component("movement")
        
        assert movement is not None
        assert movement.name == "movement"
        assert basic_agent.has_component("movement") is False
    
    def test_component_attached_to_agent(self, basic_agent):
        """Test that components are attached to agent."""
        movement = basic_agent.get_component("movement")
        assert movement._agent == basic_agent


class TestBehaviorExecution:
    """Tests for behavior execution."""
    
    def test_act_executes_behavior(self, basic_agent):
        """Test that act() executes behavior."""
        # Mock behavior to track execution
        behavior = Mock()
        behavior.execute_turn = Mock()
        basic_agent._behavior = behavior
        
        basic_agent.act()
        
        behavior.execute_turn.assert_called_once_with(basic_agent)
    
    def test_act_calls_component_lifecycle(self, basic_agent):
        """Test that act() calls component lifecycle methods."""
        # Mock components to track calls
        movement = basic_agent.get_component("movement")
        movement.on_step_start = Mock()
        movement.on_step_end = Mock()
        
        basic_agent.act()
        
        movement.on_step_start.assert_called_once()
        movement.on_step_end.assert_called_once()
    
    def test_act_does_nothing_when_dead(self, basic_agent):
        """Test that dead agents don't act."""
        basic_agent.alive = False
        
        behavior = Mock()
        basic_agent._behavior = behavior
        
        basic_agent.act()
        
        behavior.execute_turn.assert_not_called()
    
    def test_act_handles_behavior_errors(self, basic_agent):
        """Test that act() handles behavior execution errors gracefully."""
        # Mock behavior that raises error
        behavior = Mock()
        behavior.execute_turn = Mock(side_effect=Exception("Behavior error"))
        basic_agent._behavior = behavior
        
        # Should not raise
        basic_agent.act()
        
        # Components should still be notified
        movement = basic_agent.get_component("movement")
        movement.on_step_end = Mock()
        basic_agent.act()
        movement.on_step_end.assert_called()


class TestLifecycleManagement:
    """Tests for agent lifecycle."""
    
    def test_terminate_marks_not_alive(self, basic_agent):
        """Test terminate marks agent as not alive."""
        basic_agent.terminate()
        
        assert basic_agent.alive is False
    
    def test_terminate_sets_death_time(self, basic_agent, mock_services):
        """Test terminate sets death time."""
        mock_services["time"].current_time.return_value = 200
        
        basic_agent.terminate()
        
        assert basic_agent.state_manager.death_time == 200
    
    def test_terminate_notifies_components(self, basic_agent):
        """Test terminate notifies all components."""
        movement = basic_agent.get_component("movement")
        movement.on_terminate = Mock()
        
        basic_agent.terminate()
        
        movement.on_terminate.assert_called_once()
    
    def test_terminate_removes_from_lifecycle_service(self, basic_agent, mock_services):
        """Test terminate removes agent from lifecycle service."""
        basic_agent.terminate()
        
        mock_services["lifecycle"].remove_agent.assert_called_once_with(basic_agent)
    
    def test_terminate_idempotent(self, basic_agent, mock_services):
        """Test terminate can be called multiple times safely."""
        basic_agent.terminate()
        basic_agent.terminate()
        
        # Should only call lifecycle service once
        assert mock_services["lifecycle"].remove_agent.call_count == 1


class TestStateSerialization:
    """Tests for state serialization."""
    
    def test_get_state_dict(self, basic_agent):
        """Test getting complete state."""
        state = basic_agent.get_state_dict()
        
        assert "agent_id" in state
        assert "alive" in state
        assert "state_manager" in state
        assert "components" in state
        assert "behavior" in state
        
        assert state["agent_id"] == "test_agent_001"
        assert state["alive"] is True
    
    def test_get_state_includes_components(self, basic_agent):
        """Test state includes all component states."""
        state = basic_agent.get_state_dict()
        
        assert "movement" in state["components"]
        assert "resource" in state["components"]
        assert "combat" in state["components"]
    
    def test_load_state_dict(self, basic_agent):
        """Test loading state."""
        # Create state
        state = {
            "agent_id": "different_agent",
            "alive": False,
            "state_manager": {
                "position": (100.0, 200.0, 0.0),
                "generation": 5,
            },
            "components": {
                "resource": {
                    "resources": 50,
                    "starvation_counter": 2,
                }
            },
            "behavior": {},
        }
        
        basic_agent.load_state_dict(state)
        
        assert basic_agent.agent_id == "different_agent"
        assert basic_agent.alive is False
        assert basic_agent.position == (100.0, 200.0)
        assert basic_agent.state_manager.generation == 5
        
        resource = basic_agent.get_component("resource")
        assert resource.level == 50
    
    def test_round_trip_serialization(self, basic_agent):
        """Test save/load preserves state."""
        # Modify state
        basic_agent.state_manager.set_generation(3)
        resource = basic_agent.get_component("resource")
        resource.set_level(75)
        
        # Save
        state = basic_agent.get_state_dict()
        
        # Create new agent and load
        from farm.core.agent.behaviors import DefaultAgentBehavior
        new_agent = AgentCore(
            agent_id="temp",
            position=(0, 0),
            spatial_service=Mock(),
            behavior=DefaultAgentBehavior(),
            components=[
                MovementComponent(MovementConfig()),
                ResourceComponent(0, ResourceConfig()),
                CombatComponent(CombatConfig()),
            ]
        )
        new_agent.load_state_dict(state)
        
        # Verify
        assert new_agent.state_manager.generation == 3
        assert new_agent.get_component("resource").level == 75


class TestAgentRepr:
    """Tests for string representation."""
    
    def test_repr(self, basic_agent):
        """Test agent string representation."""
        repr_str = repr(basic_agent)
        
        assert "AgentCore" in repr_str
        assert "test_agent_001" in repr_str
        assert "alive=True" in repr_str
        assert "position=(50.0, 50.0)" in repr_str
        assert "movement" in repr_str
        assert "resource" in repr_str
        assert "combat" in repr_str
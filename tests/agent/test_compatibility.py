"""
Tests for compatibility layer (BaseAgentAdapter).

Verifies that the adapter provides full backward compatibility with BaseAgent.
"""

import pytest
from unittest.mock import Mock
from farm.core.agent.compat import (
    BaseAgentAdapter,
    migrate_to_core,
    is_new_agent,
    get_core,
)
from farm.core.agent import AgentCore, AgentFactory, AgentConfig


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
    lifecycle_service.get_next_agent_id = Mock(return_value="agent_002")
    
    return {
        "spatial": spatial_service,
        "time": time_service,
        "lifecycle": lifecycle_service,
    }


@pytest.fixture
def adapter(mock_services):
    """Create adapter for testing."""
    return BaseAgentAdapter.from_old_style(
        agent_id="test_agent",
        position=(50.0, 50.0),
        resource_level=100,
        spatial_service=mock_services["spatial"],
        time_service=mock_services["time"],
        lifecycle_service=mock_services["lifecycle"],
    )


class TestAdapterCreation:
    """Tests for creating adapters."""
    
    def test_from_old_style(self, mock_services):
        """Test creating adapter from old-style parameters."""
        adapter = BaseAgentAdapter.from_old_style(
            agent_id="test_001",
            position=(10.0, 20.0),
            resource_level=75,
            spatial_service=mock_services["spatial"]
        )
        
        assert isinstance(adapter, BaseAgentAdapter)
        assert adapter.agent_id == "test_001"
        assert adapter.position == (10.0, 20.0)
        assert adapter.resource_level == 75
    
    def test_deprecation_warning(self, mock_services):
        """Test that adapter issues deprecation warning."""
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            adapter = BaseAgentAdapter.from_old_style(
                agent_id="test",
                position=(0, 0),
                resource_level=100,
                spatial_service=mock_services["spatial"]
            )
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


class TestAdapterIdentityProperties:
    """Tests for identity properties compatibility."""
    
    def test_agent_id(self, adapter):
        """Test agent_id property."""
        assert adapter.agent_id == "test_agent"
    
    def test_alive(self, adapter):
        """Test alive property."""
        assert adapter.alive is True
        
        adapter.terminate()
        assert adapter.alive is False
    
    def test_agent_type(self, adapter):
        """Test agent_type property."""
        assert isinstance(adapter.agent_type, str)


class TestAdapterPositionProperties:
    """Tests for position properties compatibility."""
    
    def test_get_position(self, adapter):
        """Test reading position."""
        assert adapter.position == (50.0, 50.0)
    
    def test_set_position(self, adapter):
        """Test setting position (backward compatibility)."""
        adapter.position = (60.0, 70.0)
        assert adapter.position == (60.0, 70.0)
    
    def test_get_orientation(self, adapter):
        """Test reading orientation."""
        assert adapter.orientation == 0.0
    
    def test_set_orientation(self, adapter):
        """Test setting orientation."""
        adapter.orientation = 45.0
        assert adapter.orientation == 45.0


class TestAdapterResourceProperties:
    """Tests for resource properties compatibility."""
    
    def test_get_resource_level(self, adapter):
        """Test reading resource level."""
        assert adapter.resource_level == 100
    
    def test_set_resource_level(self, adapter):
        """Test setting resource level."""
        adapter.resource_level = 75
        assert adapter.resource_level == 75
    
    def test_resource_level_without_component(self):
        """Test resource_level when component missing."""
        # Create minimal agent without resource component
        factory = AgentFactory(spatial_service=Mock())
        core = factory.create_minimal_agent(
            agent_id="minimal",
            position=(0, 0),
            components=[]
        )
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adapter = BaseAgentAdapter(core)
        
        assert adapter.resource_level == 0


class TestAdapterHealthProperties:
    """Tests for health properties compatibility."""
    
    def test_get_current_health(self, adapter):
        """Test reading current health."""
        assert adapter.current_health == 100.0
    
    def test_set_current_health(self, adapter):
        """Test setting current health."""
        adapter.current_health = 75.0
        assert adapter.current_health == 75.0
    
    def test_get_starting_health(self, adapter):
        """Test reading max health."""
        assert adapter.starting_health == 100.0
    
    def test_is_defending(self, adapter):
        """Test is_defending property."""
        assert adapter.is_defending is False
        
        # Start defense
        combat = adapter.core.get_component("combat")
        combat.start_defense()
        
        assert adapter.is_defending is True


class TestAdapterLifecycleProperties:
    """Tests for lifecycle properties compatibility."""
    
    def test_birth_time(self, adapter):
        """Test birth_time property."""
        assert adapter.birth_time == 100
    
    def test_generation(self, adapter):
        """Test generation property."""
        assert adapter.generation == 0
    
    def test_genome_id(self, adapter):
        """Test genome_id property."""
        assert isinstance(adapter.genome_id, str)


class TestAdapterMethods:
    """Tests for method compatibility."""
    
    def test_act(self, adapter):
        """Test act() method."""
        # Should execute without error
        adapter.act()
        assert adapter.alive is True
    
    def test_terminate(self, adapter, mock_services):
        """Test terminate() method."""
        adapter.terminate()
        
        assert adapter.alive is False
        mock_services["lifecycle"].remove_agent.assert_called()
    
    def test_update_position(self, adapter):
        """Test update_position() legacy method."""
        adapter.update_position((75.0, 85.0))
        assert adapter.position == (75.0, 85.0)
    
    def test_handle_combat(self, adapter):
        """Test handle_combat() legacy method."""
        attacker = Mock()
        damage = adapter.handle_combat(attacker, 25.0)
        
        assert damage == 25.0
        assert adapter.current_health == 75.0
    
    def test_take_damage(self, adapter):
        """Test take_damage() legacy method."""
        result = adapter.take_damage(30.0)
        
        assert result is True
        assert adapter.current_health == 70.0
    
    def test_attack_strength(self, adapter):
        """Test attack_strength property."""
        strength = adapter.attack_strength
        assert isinstance(strength, float)
        assert strength > 0
    
    def test_defense_strength(self, adapter):
        """Test defense_strength property."""
        # Not defending
        assert adapter.defense_strength == 0.0
        
        # Start defending
        combat = adapter.core.get_component("combat")
        combat.start_defense()
        assert adapter.defense_strength > 0.0


class TestAdapterCoreAccess:
    """Tests for accessing underlying AgentCore."""
    
    def test_core_property(self, adapter):
        """Test accessing underlying core."""
        core = adapter.core
        
        assert isinstance(core, AgentCore)
        assert core.agent_id == "test_agent"
    
    def test_can_use_new_api(self, adapter):
        """Test that new API is accessible."""
        movement = adapter.core.get_component("movement")
        
        assert movement is not None
        movement.move_to((100.0, 100.0))
        
        # Position should have moved
        assert adapter.position != (50.0, 50.0)


class TestMigrationUtilities:
    """Tests for migration utility functions."""
    
    def test_migrate_to_core(self, adapter):
        """Test extracting core from adapter."""
        core = migrate_to_core(adapter)
        
        assert isinstance(core, AgentCore)
        assert core.agent_id == adapter.agent_id
    
    def test_is_new_agent_with_adapter(self, adapter):
        """Test is_new_agent with adapter."""
        assert is_new_agent(adapter) is True
    
    def test_is_new_agent_with_core(self):
        """Test is_new_agent with AgentCore."""
        factory = AgentFactory(spatial_service=Mock())
        core = factory.create_default_agent(
            agent_id="test",
            position=(0, 0)
        )
        
        assert is_new_agent(core) is True
    
    def test_is_new_agent_with_old_style(self):
        """Test is_new_agent with non-AgentCore object."""
        old_agent = Mock()
        assert is_new_agent(old_agent) is False
    
    def test_get_core_from_adapter(self, adapter):
        """Test get_core with adapter."""
        core = get_core(adapter)
        
        assert isinstance(core, AgentCore)
    
    def test_get_core_from_core(self):
        """Test get_core with AgentCore."""
        factory = AgentFactory(spatial_service=Mock())
        core = factory.create_default_agent(
            agent_id="test",
            position=(0, 0)
        )
        
        result = get_core(core)
        assert result is core
    
    def test_get_core_from_other(self):
        """Test get_core with non-agent object."""
        other = Mock()
        assert get_core(other) is None


class TestAdapterStateManagement:
    """Tests for state management."""
    
    def test_get_state(self, adapter):
        """Test getting state from adapter."""
        state = adapter.get_state()
        
        assert isinstance(state, dict)
        assert "agent_id" in state
        assert state["agent_id"] == "test_agent"


class TestAdapterRepr:
    """Tests for string representation."""
    
    def test_repr(self, adapter):
        """Test adapter string representation."""
        repr_str = repr(adapter)
        
        assert "BaseAgentAdapter" in repr_str
        assert "AgentCore" in repr_str
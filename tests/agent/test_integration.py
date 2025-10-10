"""
End-to-end integration tests for complete agent system.

Tests verify that all components work together correctly.
"""

import pytest
from unittest.mock import Mock
from farm.core.agent import (
    AgentFactory,
    AgentConfig,
    MovementConfig,
    ResourceConfig,
    CombatConfig,
)


@pytest.fixture
def mock_services():
    """Create mock services for testing."""
    spatial_service = Mock()
    spatial_service.get_nearby = Mock(return_value={"agents": [], "resources": []})
    spatial_service.get_nearest = Mock(return_value={"agents": None, "resources": None})
    spatial_service.mark_positions_dirty = Mock()
    
    time_service = Mock()
    time_service.current_time = Mock(return_value=0)
    
    lifecycle_service = Mock()
    lifecycle_service.get_next_agent_id = Mock(side_effect=lambda: f"agent_{lifecycle_service.get_next_agent_id.call_count}")
    lifecycle_service.add_agent = Mock()
    lifecycle_service.remove_agent = Mock()
    
    return {
        "spatial": spatial_service,
        "time": time_service,
        "lifecycle": lifecycle_service,
    }


@pytest.fixture
def factory(mock_services):
    """Create agent factory."""
    return AgentFactory(
        spatial_service=mock_services["spatial"],
        time_service=mock_services["time"],
        lifecycle_service=mock_services["lifecycle"],
    )


class TestAgentLifecycle:
    """Tests for complete agent lifecycle."""
    
    def test_agent_creation_and_initial_state(self, factory):
        """Test agent starts with correct initial state."""
        agent = factory.create_default_agent(
            agent_id="test_001",
            position=(50.0, 50.0),
            initial_resources=100
        )
        
        assert agent.alive is True
        assert agent.position == (50.0, 50.0)
        
        # Check components
        movement = agent.get_component("movement")
        resource = agent.get_component("resource")
        combat = agent.get_component("combat")
        
        assert movement is not None
        assert resource.level == 100
        assert combat.health == combat.max_health
    
    def test_agent_executes_turns(self, factory, mock_services):
        """Test agent can execute multiple turns."""
        agent = factory.create_default_agent(
            agent_id="test_001",
            position=(50.0, 50.0),
            initial_resources=100
        )
        
        # Execute several turns
        for i in range(10):
            mock_services["time"].current_time.return_value = i
            agent.act()
        
        # Agent should still be alive
        assert agent.alive is True
        
        # Resources should have decreased (base consumption per turn)
        resource = agent.get_component("resource")
        assert resource.level < 100
    
    def test_agent_starvation_death(self, factory, mock_services):
        """Test agent dies from starvation."""
        config = AgentConfig(
            resource=ResourceConfig(
                base_consumption_rate=10,
                starvation_threshold=3
            )
        )
        
        agent = factory.create_default_agent(
            agent_id="test_001",
            position=(50.0, 50.0),
            initial_resources=20,
            config=config
        )
        
        # Execute turns until death
        for i in range(10):
            mock_services["time"].current_time.return_value = i
            agent.act()
            if not agent.alive:
                break
        
        # Agent should be dead
        assert agent.alive is False
        
        # Lifecycle service should be notified
        mock_services["lifecycle"].remove_agent.assert_called()
    
    def test_agent_combat_death(self, factory, mock_services):
        """Test agent dies from combat."""
        agent = factory.create_default_agent(
            agent_id="attacker",
            position=(0.0, 0.0),
            initial_resources=100
        )
        
        victim = factory.create_default_agent(
            agent_id="victim",
            position=(1.0, 1.0),
            initial_resources=100
        )
        
        # Attack until death
        attacker_combat = agent.get_component("combat")
        victim_combat = victim.get_component("combat")
        
        while victim.alive:
            attacker_combat.attack(victim)
        
        assert victim.alive is False
        assert victim_combat.health == 0.0


class TestComponentInteractions:
    """Tests for component interactions."""
    
    def test_movement_updates_position(self, factory):
        """Test movement component updates agent position."""
        agent = factory.create_default_agent(
            agent_id="test_001",
            position=(0.0, 0.0)
        )
        
        movement = agent.get_component("movement")
        movement.move_to((10.0, 10.0))
        
        # Agent position should update
        assert agent.position != (0.0, 0.0)
    
    def test_resource_consumption_each_turn(self, factory, mock_services):
        """Test resources consumed each turn."""
        agent = factory.create_default_agent(
            agent_id="test_001",
            position=(0.0, 0.0),
            initial_resources=100
        )
        
        resource = agent.get_component("resource")
        initial_resources = resource.level
        
        # Execute one turn
        agent.act()
        
        # Resources should decrease
        assert resource.level < initial_resources
    
    def test_combat_affects_health(self, factory):
        """Test combat reduces victim health."""
        agent1 = factory.create_default_agent(
            agent_id="agent1",
            position=(0.0, 0.0)
        )
        
        agent2 = factory.create_default_agent(
            agent_id="agent2",
            position=(1.0, 1.0)
        )
        
        combat1 = agent1.get_component("combat")
        combat2 = agent2.get_component("combat")
        
        initial_health = combat2.health
        
        # Agent1 attacks agent2
        combat1.attack(agent2)
        
        # Agent2 health should decrease
        assert combat2.health < initial_health
    
    def test_defense_reduces_damage(self, factory):
        """Test defensive stance reduces damage taken."""
        agent1 = factory.create_default_agent(
            agent_id="attacker",
            position=(0.0, 0.0)
        )
        
        agent2 = factory.create_default_agent(
            agent_id="defender",
            position=(1.0, 1.0)
        )
        
        combat1 = agent1.get_component("combat")
        combat2 = agent2.get_component("combat")
        
        # Attack without defense
        combat2.set_health(100.0)
        result1 = combat1.attack(agent2)
        damage_normal = result1["damage_dealt"]
        
        # Reset and attack with defense
        combat2.set_health(100.0)
        combat2.start_defense()
        result2 = combat1.attack(agent2)
        damage_defended = result2["damage_dealt"]
        
        # Defended damage should be less
        assert damage_defended < damage_normal
    
    def test_perception_finds_nearby_agents(self, factory, mock_services):
        """Test perception component finds nearby agents."""
        agent1 = factory.create_default_agent(
            agent_id="agent1",
            position=(50.0, 50.0)
        )
        
        agent2 = factory.create_default_agent(
            agent_id="agent2",
            position=(52.0, 52.0)
        )
        
        # Mock spatial service to return agent2
        mock_services["spatial"].get_nearby.return_value = {
            "agents": [agent2],
            "resources": []
        }
        
        perception = agent1.get_component("perception")
        nearby = perception.get_nearby_entities(["agents"])
        
        assert len(nearby["agents"]) == 1
        assert nearby["agents"][0] == agent2


class TestStatePersistence:
    """Tests for state serialization and loading."""
    
    def test_save_and_load_agent_state(self, factory):
        """Test agent state can be saved and loaded."""
        # Create agent and modify state
        agent1 = factory.create_default_agent(
            agent_id="test_001",
            position=(50.0, 50.0),
            initial_resources=100
        )
        
        # Modify some state
        agent1.state_manager.set_generation(3)
        movement = agent1.get_component("movement")
        movement.move_to((60.0, 60.0))
        resource = agent1.get_component("resource")
        resource.consume(25)
        
        # Save state
        state = agent1.get_state_dict()
        
        # Create new agent and load state
        agent2 = factory.create_default_agent(
            agent_id="temp",
            position=(0.0, 0.0),
            initial_resources=0
        )
        agent2.load_state_dict(state)
        
        # Verify state matches
        assert agent2.agent_id == "test_001"
        assert agent2.position[0] >= 50.0  # Moved toward (60, 60)
        assert agent2.state_manager.generation == 3
        assert agent2.get_component("resource").level == 75  # 100 - 25


class TestAgentBehavior:
    """Tests for agent behavior execution."""
    
    def test_default_behavior_executes(self, factory, mock_services):
        """Test default behavior executes without errors."""
        agent = factory.create_default_agent(
            agent_id="test_001",
            position=(50.0, 50.0),
            initial_resources=100
        )
        
        # Execute several turns
        for i in range(5):
            mock_services["time"].current_time.return_value = i
            agent.act()
        
        # Should complete without errors
        assert agent.alive is True
    
    def test_learning_behavior_executes(self, factory, mock_services):
        """Test learning behavior executes without errors."""
        agent = factory.create_learning_agent(
            agent_id="learner_001",
            position=(50.0, 50.0),
            initial_resources=100
        )
        
        # Execute several turns
        for i in range(5):
            mock_services["time"].current_time.return_value = i
            agent.act()
        
        # Should complete without errors
        assert agent.alive is True


class TestReproduction:
    """Tests for agent reproduction."""
    
    def test_reproduction_creates_offspring(self, factory, mock_services):
        """Test agent can reproduce."""
        config = AgentConfig(
            resource=ResourceConfig(initial_resources=100),
        )
        
        agent = factory.create_default_agent(
            agent_id="parent_001",
            position=(50.0, 50.0),
            initial_resources=100,
            config=config
        )
        
        reproduction = agent.get_component("reproduction")
        resource = agent.get_component("resource")
        
        # Ensure has enough resources
        assert reproduction.can_reproduce()
        
        # Reproduce
        result = reproduction.reproduce()
        
        # Should succeed
        assert result["success"] is True
        assert "offspring_id" in result
        
        # Parent resources should decrease
        assert resource.level < 100
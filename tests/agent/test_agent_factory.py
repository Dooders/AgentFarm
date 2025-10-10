"""
Integration tests for AgentFactory.

Tests verify:
- Agent creation with factory
- Different agent types
- Component injection
- Configuration handling
"""

import pytest
from unittest.mock import Mock
from farm.core.agent import (
    AgentFactory,
    AgentCore,
    AgentConfig,
    DefaultAgentBehavior,
    LearningAgentBehavior,
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
    lifecycle_service.get_next_agent_id = Mock(return_value="agent_002")
    lifecycle_service.add_agent = Mock()
    
    return {
        "spatial": spatial_service,
        "time": time_service,
        "lifecycle": lifecycle_service,
    }


@pytest.fixture
def factory(mock_services):
    """Create agent factory with mock services."""
    return AgentFactory(
        spatial_service=mock_services["spatial"],
        default_config=AgentConfig(),
        time_service=mock_services["time"],
        lifecycle_service=mock_services["lifecycle"],
    )


class TestFactoryInitialization:
    """Tests for factory initialization."""
    
    def test_factory_has_services(self, factory, mock_services):
        """Test factory stores services."""
        assert factory._spatial_service == mock_services["spatial"]
        assert factory._time_service == mock_services["time"]
        assert factory._lifecycle_service == mock_services["lifecycle"]
    
    def test_factory_has_default_config(self, factory):
        """Test factory has default configuration."""
        assert factory._default_config is not None
        assert isinstance(factory._default_config, AgentConfig)


class TestCreateDefaultAgent:
    """Tests for creating default agents."""
    
    def test_create_default_agent(self, factory):
        """Test creating default agent."""
        agent = factory.create_default_agent(
            agent_id="agent_001",
            position=(10.0, 20.0),
            initial_resources=100
        )
        
        assert isinstance(agent, AgentCore)
        assert agent.agent_id == "agent_001"
        assert agent.position == (10.0, 20.0)
        assert agent.alive is True
    
    def test_default_agent_has_components(self, factory):
        """Test default agent has all standard components."""
        agent = factory.create_default_agent(
            agent_id="agent_001",
            position=(0.0, 0.0)
        )
        
        assert agent.has_component("movement")
        assert agent.has_component("resource")
        assert agent.has_component("combat")
        assert agent.has_component("perception")
        assert agent.has_component("reproduction")
    
    def test_default_agent_has_resources(self, factory):
        """Test default agent has specified resources."""
        agent = factory.create_default_agent(
            agent_id="agent_001",
            position=(0.0, 0.0),
            initial_resources=75
        )
        
        resource = agent.get_component("resource")
        assert resource.level == 75
    
    def test_default_agent_has_default_behavior(self, factory):
        """Test default agent uses DefaultAgentBehavior."""
        agent = factory.create_default_agent(
            agent_id="agent_001",
            position=(0.0, 0.0)
        )
        
        assert isinstance(agent._behavior, DefaultAgentBehavior)
    
    def test_default_agent_with_custom_config(self, factory):
        """Test creating default agent with custom configuration."""
        from farm.core.agent.config.agent_config import (
            AgentConfig,
            MovementConfig,
            CombatConfig,
        )
        
        custom_config = AgentConfig(
            movement=MovementConfig(max_movement=15.0),
            combat=CombatConfig(starting_health=150.0),
        )
        
        agent = factory.create_default_agent(
            agent_id="agent_001",
            position=(0.0, 0.0),
            config=custom_config
        )
        
        movement = agent.get_component("movement")
        combat = agent.get_component("combat")
        
        assert movement.max_movement == 15.0
        assert combat.max_health == 150.0


class TestCreateLearningAgent:
    """Tests for creating learning agents."""
    
    def test_create_learning_agent(self, factory):
        """Test creating learning agent."""
        agent = factory.create_learning_agent(
            agent_id="learner_001",
            position=(50.0, 50.0),
            initial_resources=100
        )
        
        assert isinstance(agent, AgentCore)
        assert agent.agent_id == "learner_001"
        assert agent.position == (50.0, 50.0)
    
    def test_learning_agent_has_components(self, factory):
        """Test learning agent has all standard components."""
        agent = factory.create_learning_agent(
            agent_id="learner_001",
            position=(0.0, 0.0)
        )
        
        assert agent.has_component("movement")
        assert agent.has_component("resource")
        assert agent.has_component("combat")
        assert agent.has_component("perception")
        assert agent.has_component("reproduction")
    
    def test_learning_agent_has_learning_behavior(self, factory):
        """Test learning agent uses LearningAgentBehavior."""
        agent = factory.create_learning_agent(
            agent_id="learner_001",
            position=(0.0, 0.0)
        )
        
        assert isinstance(agent._behavior, LearningAgentBehavior)
    
    def test_learning_agent_with_decision_module(self, factory):
        """Test creating learning agent with decision module."""
        decision_module = Mock()
        
        agent = factory.create_learning_agent(
            agent_id="learner_001",
            position=(0.0, 0.0),
            decision_module=decision_module
        )
        
        assert agent._behavior._decision_module == decision_module


class TestCreateMinimalAgent:
    """Tests for creating minimal agents."""
    
    def test_create_minimal_agent_no_components(self, factory):
        """Test creating minimal agent with no components."""
        agent = factory.create_minimal_agent(
            agent_id="minimal_001",
            position=(0.0, 0.0),
            components=[]
        )
        
        assert isinstance(agent, AgentCore)
        assert agent.agent_id == "minimal_001"
        assert len(agent._components) == 0
    
    def test_create_minimal_agent_with_components(self, factory):
        """Test creating minimal agent with specific components."""
        from farm.core.agent.components import MovementComponent
        from farm.core.agent.config.agent_config import MovementConfig
        
        components = [MovementComponent(MovementConfig())]
        
        agent = factory.create_minimal_agent(
            agent_id="minimal_001",
            position=(0.0, 0.0),
            components=components
        )
        
        assert agent.has_component("movement")
        assert not agent.has_component("resource")
        assert not agent.has_component("combat")


class TestCreateCustomAgent:
    """Tests for creating custom agents."""
    
    def test_create_custom_agent(self, factory):
        """Test creating agent with custom behavior and components."""
        from farm.core.agent.components import MovementComponent, ResourceComponent
        from farm.core.agent.config.agent_config import MovementConfig, ResourceConfig
        
        behavior = DefaultAgentBehavior()
        components = [
            MovementComponent(MovementConfig()),
            ResourceComponent(50, ResourceConfig()),
        ]
        
        agent = factory.create_agent(
            agent_id="custom_001",
            position=(25.0, 75.0),
            behavior=behavior,
            components=components
        )
        
        assert isinstance(agent, AgentCore)
        assert agent.agent_id == "custom_001"
        assert agent.position == (25.0, 75.0)
        assert agent.has_component("movement")
        assert agent.has_component("resource")
        assert not agent.has_component("combat")
    
    def test_create_custom_agent_uses_defaults(self, factory):
        """Test creating custom agent with default components."""
        behavior = DefaultAgentBehavior()
        
        agent = factory.create_agent(
            agent_id="custom_001",
            position=(0.0, 0.0),
            behavior=behavior,
            components=None  # Use defaults
        )
        
        # Should have all default components
        assert agent.has_component("movement")
        assert agent.has_component("resource")
        assert agent.has_component("combat")
        assert agent.has_component("perception")
        assert agent.has_component("reproduction")


class TestOffspringFactory:
    """Tests for offspring creation."""
    
    def test_offspring_factory_creates_agent(self, factory, mock_services):
        """Test offspring factory creates new agent."""
        # Get the offspring factory
        config = AgentConfig()
        offspring_factory_func = factory._create_offspring_factory(config)
        
        # Create offspring
        offspring = offspring_factory_func(
            agent_id="offspring_001",
            position=(10.0, 10.0),
            initial_resources=20,
            parent_ids=["parent_001"],
            generation=2
        )
        
        assert isinstance(offspring, AgentCore)
        assert offspring.agent_id == "offspring_001"
        assert offspring.position == (10.0, 10.0)
        assert offspring.state_manager.generation == 2
        assert offspring.state_manager.parent_ids == ["parent_001"]
    
    def test_reproduction_component_uses_factory(self, factory):
        """Test reproduction component gets offspring factory."""
        agent = factory.create_default_agent(
            agent_id="parent_001",
            position=(0.0, 0.0),
            initial_resources=100
        )
        
        reproduction = agent.get_component("reproduction")
        
        # Should have offspring factory set
        assert reproduction._offspring_factory is not None
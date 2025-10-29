"""Integration tests for the new component-based AgentCore system.

Tests verify that all components work together correctly, the factory creates
agents properly, and the system integrates with existing components like actions.
"""

import pytest
from unittest.mock import Mock, MagicMock

from farm.core.agent import (
    AgentCore,
    AgentFactory,
    AgentServices,
    AgentComponentConfig,
    DefaultAgentBehavior,
    LearningAgentBehavior,
)
from farm.core.agent.components import (
    MovementComponent,
    ResourceComponent,
    CombatComponent,
    PerceptionComponent,
    ReproductionComponent,
)
from farm.core.agent.config import ResourceConfig


class TestAgentCoreBasics:
    """Test basic AgentCore functionality."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services for testing."""
        return AgentServices(
            spatial_service=Mock(),
            time_service=Mock(current_time=Mock(return_value=0)),
            metrics_service=Mock(),
            logging_service=Mock(),
            validation_service=Mock(is_valid_position=Mock(return_value=True)),
            lifecycle_service=Mock(),
        )
    
    def test_agent_creation(self, mock_services):
        """Test basic agent creation."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent(
            agent_id="test_001",
            position=(50.0, 50.0),
            initial_resources=100.0,
        )
        
        assert agent.agent_id == "test_001"
        assert agent.position == (50.0, 50.0)
        assert agent.resource_level == 100.0
        assert agent.alive is True
    
    def test_agent_has_all_components(self, mock_services):
        """Test that created agents have all expected components."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent("test_002", (0.0, 0.0))
        
        assert agent.get_component("movement") is not None
        assert agent.get_component("resource") is not None
        assert agent.get_component("combat") is not None
        assert agent.get_component("perception") is not None
        assert agent.get_component("reproduction") is not None
    
    def test_minimal_agent_has_fewer_components(self, mock_services):
        """Test that minimal agents don't have combat/perception."""
        factory = AgentFactory(mock_services)
        agent = factory.create_minimal_agent("test_003", (0.0, 0.0))
        
        assert agent.get_component("movement") is not None
        assert agent.get_component("resource") is not None
        # Minimal agents should not have these
        assert agent.get_component("combat") is None
        assert agent.get_component("perception") is None


class TestComponentInteraction:
    """Test that components interact correctly."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        return AgentServices(
            spatial_service=Mock(),
            time_service=Mock(current_time=Mock(return_value=0)),
            metrics_service=Mock(),
            logging_service=Mock(),
            validation_service=Mock(is_valid_position=Mock(return_value=True)),
            lifecycle_service=Mock(),
        )
    
    def test_resource_consumption(self, mock_services):
        """Test that resources are consumed each step."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent(
            "test_004",
            (0.0, 0.0),
            initial_resources=100.0,
        )
        
        resource_comp = agent.get_component("resource")
        assert resource_comp.level == 100.0
        
        # Simulate consumption
        resource_comp.on_step_start()
        assert resource_comp.level < 100.0
    
    def test_starvation_triggers_death(self, mock_services):
        """Test that prolonged starvation kills agent."""
        config = AgentComponentConfig(
            resource=ResourceConfig(starvation_threshold=3)
        )
        
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent(
            "test_005",
            (0.0, 0.0),
            initial_resources=0.0,
            config=config,
        )
        
        resource_comp = agent.get_component("resource")
        
        # Agent should survive first few steps without resources
        assert agent.alive is True
        
        # After threshold steps without resources, should die
        for _ in range(3):
            resource_comp.on_step_start()
        
        # Starvation counter should trigger termination
        assert resource_comp.starvation_counter >= config.resource.starvation_threshold
    
    def test_combat_damage(self, mock_services):
        """Test that damage is applied correctly."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent("test_006", (0.0, 0.0))
        
        combat_comp = agent.get_component("combat")
        initial_health = combat_comp.health
        
        # Apply damage
        actual_damage = combat_comp.take_damage(10.0)
        
        assert actual_damage == 10.0
        assert combat_comp.health == initial_health - 10.0
    
    def test_defense_reduces_damage(self, mock_services):
        """Test that defense reduces damage taken."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent("test_007", (0.0, 0.0))
        
        combat_comp = agent.get_component("combat")
        initial_health = combat_comp.health
        
        # Start defense
        combat_comp.start_defense()
        assert combat_comp.is_defending is True
        
        # Apply damage (should be reduced)
        damage_reduction_factor = combat_comp.config.defense_damage_reduction
        actual_damage = combat_comp.take_damage(10.0)
        
        assert actual_damage == 10.0 * damage_reduction_factor
        assert combat_comp.health == initial_health - (10.0 * damage_reduction_factor)


class TestBehaviorSystem:
    """Test behavior strategies."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        return AgentServices(
            spatial_service=Mock(),
            time_service=Mock(current_time=Mock(return_value=0)),
            metrics_service=Mock(),
            logging_service=Mock(),
            validation_service=Mock(is_valid_position=Mock(return_value=True)),
            lifecycle_service=Mock(),
        )
    
    def test_default_behavior_selects_action(self, mock_services):
        """Test that default behavior can select actions."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent("test_008", (0.0, 0.0))
        
        # Get mock state
        import torch
        state = torch.zeros((1, 11, 11))
        
        # Behavior should select an action
        action = agent.behavior.decide_action(agent, state)
        assert action is not None
        assert hasattr(action, 'name')
    
    def test_learning_behavior_integrates_with_decision_module(self, mock_services):
        """Test that learning behavior works with decision module."""
        factory = AgentFactory(mock_services)
        agent = factory.create_learning_agent("test_009", (0.0, 0.0))
        
        assert isinstance(agent.behavior, LearningAgentBehavior)
        assert agent.behavior.decision_module is not None


class TestAgentConfiguration:
    """Test agent configuration system."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        return AgentServices(
            spatial_service=Mock(),
            time_service=Mock(current_time=Mock(return_value=0)),
            metrics_service=Mock(),
            logging_service=Mock(),
            validation_service=Mock(is_valid_position=Mock(return_value=True)),
            lifecycle_service=Mock(),
        )
    
    def test_aggressive_configuration(self, mock_services):
        """Test aggressive agent has high combat stats."""
        factory = AgentFactory(mock_services)
        agent = factory.create_aggressive_agent("test_010", (0.0, 0.0))
        
        combat_comp = agent.get_component("combat")
        config = AgentComponentConfig.aggressive()
        
        assert combat_comp.config.starting_health == config.combat.starting_health
        assert combat_comp.config.base_attack_strength == config.combat.base_attack_strength
    
    def test_defensive_configuration(self, mock_services):
        """Test defensive agent has high health and defense."""
        factory = AgentFactory(mock_services)
        agent = factory.create_defensive_agent("test_011", (0.0, 0.0))
        
        combat_comp = agent.get_component("combat")
        config = AgentComponentConfig.defensive()
        
        assert combat_comp.config.starting_health == config.combat.starting_health
        assert combat_comp.config.base_defense_strength == config.combat.base_defense_strength
    
    def test_efficient_configuration(self, mock_services):
        """Test efficient agent has low consumption."""
        factory = AgentFactory(mock_services)
        agent = factory.create_efficient_agent("test_012", (0.0, 0.0))
        
        resource_comp = agent.get_component("resource")
        config = AgentComponentConfig.efficient()
        
        assert resource_comp.config.base_consumption_rate == config.resource.base_consumption_rate


class TestAgentProperties:
    """Test that agent properties delegate to components correctly."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        return AgentServices(
            spatial_service=Mock(),
            time_service=Mock(current_time=Mock(return_value=0)),
            metrics_service=Mock(),
            logging_service=Mock(),
            validation_service=Mock(is_valid_position=Mock(return_value=True)),
            lifecycle_service=Mock(),
        )
    
    def test_position_property(self, mock_services):
        """Test that position property delegates to movement component."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent("test_013", (50.0, 60.0))
        
        # Position should match what we set
        assert agent.position == (50.0, 60.0)
        
        # Setting position should update component
        agent.position = (70.0, 80.0)
        movement = agent.get_component("movement")
        assert movement.position == (70.0, 80.0)
    
    def test_resource_level_property(self, mock_services):
        """Test that resource_level property delegates to resource component."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent("test_014", (0.0, 0.0), initial_resources=100.0)
        
        assert agent.resource_level == 100.0
        
        # Setting should update component
        agent.resource_level = 50.0
        assert agent.resource_level == 50.0
    
    def test_health_property(self, mock_services):
        """Test that current_health property delegates to combat component."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent("test_015", (0.0, 0.0))
        
        combat = agent.get_component("combat")
        assert agent.current_health == combat.health


class TestActMethod:
    """Test backward compatibility with act() method."""
    
    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        return AgentServices(
            spatial_service=Mock(),
            time_service=Mock(current_time=Mock(return_value=0)),
            metrics_service=Mock(),
            logging_service=Mock(),
            validation_service=Mock(is_valid_position=Mock(return_value=True)),
            lifecycle_service=Mock(),
        )
    
    def test_act_method_exists(self, mock_services):
        """Test that act() method exists for backward compatibility."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent("test_016", (0.0, 0.0))
        
        # act() should exist
        assert hasattr(agent, 'act')
        assert callable(agent.act)
    
    def test_act_executes_step(self, mock_services):
        """Test that calling act() executes a step."""
        factory = AgentFactory(mock_services)
        agent = factory.create_default_agent("test_017", (0.0, 0.0), initial_resources=100.0)
        
        initial_resources = agent.resource_level
        
        # act() should consume resources (via step)
        agent.act()
        
        # Resources should be consumed
        assert agent.resource_level < initial_resources


class TestConsolidatedObservationSystem:
    """Test the consolidated observation system in AgentCore."""
    
    @pytest.fixture
    def mock_services_with_environment(self):
        """Create mock services with environment for observation testing."""
        from farm.core.observations import ObservationConfig
        
        # Create environment mock
        env = Mock()
        env.observation_config = ObservationConfig(R=3)
        env.height = 100
        env.width = 100
        env.config = Mock()
        env.config.environment = Mock()
        env.config.environment.position_discretization_method = "floor"
        env.config.environment.use_bilinear_interpolation = True
        env.spatial_index = Mock()
        env.spatial_index.get_nearby.return_value = {"resources": [], "agents": []}
        env.max_resource = 10.0
        
        services = AgentServices(
            spatial_service=Mock(),
            time_service=Mock(current_time=Mock(return_value=0)),
            metrics_service=Mock(),
            logging_service=Mock(),
            validation_service=Mock(is_valid_position=Mock(return_value=True)),
            lifecycle_service=Mock(),
        )
        
        return services, env
    
    def test_single_observation_path(self, mock_services_with_environment):
        """Test that AgentCore uses only the perception component for observations."""
        import torch
        from unittest.mock import patch
        
        services, env = mock_services_with_environment
        
        factory = AgentFactory(services)
        agent = factory.create_default_agent(
            agent_id="test_obs_001",
            position=(50.0, 50.0),
            initial_resources=100.0,
        )
        
        # Set up environment
        agent.environment = env
        
        # Mock the perception component
        perception_comp = agent.get_component("perception")
        assert perception_comp is not None
        
        # Mock the observation tensor generation
        expected_tensor = torch.zeros((1, 7, 7), dtype=torch.float32)
        perception_comp.get_observation_tensor = Mock(return_value=expected_tensor)
        
        # Test that _create_observation uses perception component
        observation = agent._create_observation()
        
        # Should call perception component
        perception_comp.get_observation_tensor.assert_called_once_with(agent.device)
        
        # Should return the expected tensor
        assert torch.equal(observation, expected_tensor)
    
    def test_observation_fallback_when_no_perception_component(self, mock_services_with_environment):
        """Test fallback behavior when no perception component is available."""
        import torch
        
        services, env = mock_services_with_environment
        
        factory = AgentFactory(services)
        agent = factory.create_default_agent(
            agent_id="test_obs_002",
            position=(50.0, 50.0),
            initial_resources=100.0,
        )
        
        # Remove perception component
        agent._components = {k: v for k, v in agent._components.items() if k != "perception"}
        
        # Test observation generation
        observation = agent._create_observation()
        
        # Should return zero tensor as fallback
        # Shape is computed from perception radius: (1, 2*R+1, 2*R+1) where R=5
        perception_radius = 5  # Default perception radius
        expected_shape = (1, 2 * perception_radius + 1, 2 * perception_radius + 1)
        assert observation.shape == expected_shape
        assert observation.dtype == torch.float32
        assert torch.all(observation == 0)
    
    def test_observation_error_handling(self, mock_services_with_environment):
        """Test error handling in observation generation."""
        import torch
        from unittest.mock import patch
        
        services, env = mock_services_with_environment
        
        factory = AgentFactory(services)
        agent = factory.create_default_agent(
            agent_id="test_obs_003",
            position=(50.0, 50.0),
            initial_resources=100.0,
        )
        
        # Mock perception component to raise exception
        perception_comp = agent.get_component("perception")
        perception_comp.get_observation_tensor = Mock(side_effect=Exception("Perception error"))
        
        # Test observation generation with error
        with patch('farm.core.agent.core.logger') as mock_logger:
            observation = agent._create_observation()
            
            # Should log error
            mock_logger.error.assert_called()
            
            # Should return zero tensor as fallback
            # Shape is computed from perception radius: (1, 2*R+1, 2*R+1) where R=5
            perception_radius = 5  # Default perception radius
            expected_shape = (1, 2 * perception_radius + 1, 2 * perception_radius + 1)
            assert observation.shape == expected_shape
            assert observation.dtype == torch.float32
            assert torch.all(observation == 0)
    
    def test_agent_step_uses_observation(self, mock_services_with_environment):
        """Test that agent step process uses the observation system."""
        import torch
        
        services, env = mock_services_with_environment
        
        factory = AgentFactory(services)
        agent = factory.create_default_agent(
            agent_id="test_obs_004",
            position=(50.0, 50.0),
            initial_resources=100.0,
        )
        
        # Set up environment
        agent.environment = env
        
        # Mock the behavior to track calls
        behavior = agent.behavior
        behavior.decide_action = Mock(return_value=agent.actions[0])
        
        # Mock perception component
        perception_comp = agent.get_component("perception")
        expected_tensor = torch.zeros((1, 7, 7), dtype=torch.float32)
        perception_comp.get_observation_tensor = Mock(return_value=expected_tensor)
        
        # Execute agent step
        agent.step()
        
        # Should have called perception component (may be called multiple times due to action execution)
        assert perception_comp.get_observation_tensor.call_count >= 1
        
        # Should have called behavior with observation
        behavior.decide_action.assert_called_once()
        call_args = behavior.decide_action.call_args
        assert torch.equal(call_args[0][1], expected_tensor)  # state_tensor argument
    
    def test_observation_consistency_across_steps(self, mock_services_with_environment):
        """Test that observation generation is consistent across multiple steps."""
        import torch
        
        services, env = mock_services_with_environment
        
        factory = AgentFactory(services)
        agent = factory.create_default_agent(
            agent_id="test_obs_005",
            position=(50.0, 50.0),
            initial_resources=100.0,
        )
        
        # Set up environment
        agent.environment = env
        
        # Mock perception component
        perception_comp = agent.get_component("perception")
        expected_tensor = torch.zeros((1, 7, 7), dtype=torch.float32)
        perception_comp.get_observation_tensor = Mock(return_value=expected_tensor)
        
        # Execute multiple steps
        for _ in range(5):
            agent.step()
        
        # Should have called perception component for each step (may be called multiple times per step)
        assert perception_comp.get_observation_tensor.call_count >= 5
        
        # All calls should be with the same device
        for call in perception_comp.get_observation_tensor.call_args_list:
            assert call[0][0] == agent.device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

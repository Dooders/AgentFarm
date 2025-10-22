"""
Tests for the reward component.
"""

import pytest
from unittest.mock import Mock, MagicMock

from farm.core.agent.components.reward import RewardComponent
from farm.core.agent.config.component_configs import RewardConfig
from farm.core.agent.services import AgentServices


class TestRewardComponent:
    """Test cases for RewardComponent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.services = Mock(spec=AgentServices)
        self.services.get_current_time.return_value = 0
        self.config = RewardConfig()
        self.component = RewardComponent(self.services, self.config)
        
        # Mock agent core with proper numeric values
        self.core = Mock()
        self.core.agent_id = "test_agent"
        self.core.resource_level = 50.0
        self.core.current_health = 80.0
        self.core.alive = True
        self.core.position = (10.0, 20.0)
        self.core.age = 5
        self.core.total_reward = 0.0
        
        # Mock state_manager for proper state handling
        self.core.state_manager = Mock()
        self.core.state_manager.get_state.return_value = Mock(
            resource_level=50.0,
            health=80.0,
            alive=True,
            position_x=10.0,
            position_y=20.0,
            age=5
        )
        self.core.state_manager.total_reward = 0.0
        self.core.state_manager.add_reward = Mock()
        
        self.component.attach(self.core)
    
    def test_initialization(self):
        """Test component initialization."""
        assert self.component.cumulative_reward == 0.0
        assert self.component.step_reward == 0.0
        assert self.component.reward_history == []
        assert self.component.name == "RewardComponent"
        assert self.component.config is not None
    
    def test_attach(self):
        """Test component attachment to core."""
        assert self.component.core == self.core
    
    def test_step_lifecycle(self):
        """Test step start and end lifecycle."""
        # Step start should capture pre-action state
        self.component.on_step_start()
        assert self.component.pre_action_state is not None
        assert "resource_level" in self.component.pre_action_state
        
        # Step end should calculate and apply reward
        self.component.on_step_end()
        assert self.component.step_reward != 0.0
        assert self.component.cumulative_reward != 0.0
        assert len(self.component.reward_history) == 1
    
    def test_delta_reward_calculation(self):
        """Test delta-based reward calculation."""
        # Set up pre-action state
        self.component.pre_action_state = {
            "resource_level": 40.0,
            "health": 70.0,
            "alive": True,
            "position": (10.0, 20.0),
            "age": 4,
        }
        
        # Current state has better resources and health
        self.core.resource_level = 50.0
        self.core.current_health = 80.0
        self.core.age = 5
        
        reward = self.component._calculate_reward()
        
        # Should get positive reward for resource and health increases
        assert reward > 0.0
        assert reward == self.component.last_action_reward
    
    def test_state_reward_calculation(self):
        """Test state-based reward calculation (fallback)."""
        self.component.pre_action_state = None
        
        reward = self.component._calculate_reward()
        
        # Should get positive reward for being alive with resources
        assert reward > 0.0
        assert reward == self.component.last_action_reward
    
    def test_death_penalty(self):
        """Test death penalty application."""
        self.core.alive = False
        self.component.on_terminate()
        
        # Should apply death penalty
        assert self.component.cumulative_reward < 0.0
        assert self.component.cumulative_reward == self.config.death_penalty
    
    def test_manual_reward_addition(self):
        """Test manual reward addition."""
        initial_reward = self.component.cumulative_reward
        self.component.add_reward(5.0, "test bonus")
        
        assert self.component.cumulative_reward == initial_reward + 5.0
        assert len(self.component.reward_history) == 1
        assert self.component.reward_history[0] == 5.0
    
    def test_reward_stats(self):
        """Test reward statistics calculation."""
        # Add some rewards
        self.component.add_reward(1.0)
        self.component.add_reward(2.0)
        self.component.add_reward(3.0)
        
        stats = self.component.get_reward_stats()
        
        assert stats["cumulative"] == 6.0
        assert stats["average"] == 2.0
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["recent_average"] == 2.0
    
    def test_reward_reset(self):
        """Test reward tracking reset."""
        # Add some rewards
        self.component.add_reward(5.0)
        self.component.add_reward(3.0)
        
        # Reset
        self.component.reset_rewards()
        
        assert self.component.cumulative_reward == 0.0
        assert self.component.step_reward == 0.0
        assert self.component.reward_history == []
        assert self.component.last_action_reward == 0.0
        assert self.component.pre_action_state is None
    
    def test_reward_history_limits(self):
        """Test reward history length limits."""
        # Add more rewards than max_history_length
        for i in range(self.config.max_history_length + 10):
            self.component.add_reward(1.0)
        
        assert len(self.component.reward_history) == self.config.max_history_length
    
    def test_properties(self):
        """Test component properties."""
        self.component.step_reward = 2.5
        self.component.cumulative_reward = 10.0
        
        assert self.component.current_reward == 2.5
        assert self.component.total_reward == 10.0
    
    def test_configuration_integration(self):
        """Test that configuration values are used correctly."""
        custom_config = RewardConfig(
            resource_reward_scale=2.0,
            health_reward_scale=1.0,
            survival_bonus=0.5,
            death_penalty=-20.0,
        )
        
        component = RewardComponent(self.services, custom_config)
        component.attach(self.core)
        
        # Test that custom values are used
        assert component.config.resource_reward_scale == 2.0
        assert component.config.health_reward_scale == 1.0
        assert component.config.survival_bonus == 0.5
        assert component.config.death_penalty == -20.0


class TestRewardConfig:
    """Test cases for RewardConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RewardConfig()
        
        assert config.resource_reward_scale == 1.0
        assert config.health_reward_scale == 0.5
        assert config.survival_bonus == 0.1
        assert config.death_penalty == -10.0
        assert config.age_bonus == 0.01
        assert config.max_history_length == 1000
        assert config.recent_window == 100
        assert config.use_delta_rewards is True
        assert config.normalize_rewards is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RewardConfig(
            resource_reward_scale=2.0,
            survival_bonus=0.5,
            death_penalty=-15.0,
            max_history_length=500,
        )
        
        assert config.resource_reward_scale == 2.0
        assert config.survival_bonus == 0.5
        assert config.death_penalty == -15.0
        assert config.max_history_length == 500


class TestRewardComponentAsDefaultSystem:
    """Test that reward component is the default reward system."""
    
    def test_reward_component_is_default_system(self):
        """Test that reward component is the default reward system for agents."""
        # This test verifies that the reward component is automatically included
        # in all agents created through the factory, making it the default reward system
        
        # Create agent through factory
        from farm.core.agent.factory import AgentFactory
        from farm.core.agent.services import AgentServices
        from unittest.mock import Mock
        
        services = Mock(spec=AgentServices)
        services.get_current_time.return_value = 0
        services.logging_service = Mock()
        services.metrics_service = Mock()
        services.validation_service = Mock()
        services.time_service = Mock()
        services.spatial_service = Mock()
        services.lifecycle_service = Mock()
        
        factory = AgentFactory(services)
        agent = factory.create_default_agent(
            agent_id="default_agent",
            position=(0.0, 0.0),
        )
        
        # Verify reward component is present using public API
        reward_component = agent.get_component("reward")
        assert reward_component is not None
        assert reward_component.name == "RewardComponent"
        
        # Verify it's the default configuration
        assert reward_component.config.resource_reward_scale == 1.0
        assert reward_component.config.survival_bonus == 0.1
        assert reward_component.config.death_penalty == -10.0
    
    def test_learning_agents_also_have_reward_component(self):
        """Test that learning agents also have reward component by default."""
        from farm.core.agent.factory import AgentFactory
        from farm.core.agent.services import AgentServices
        from unittest.mock import Mock
        
        services = Mock(spec=AgentServices)
        services.get_current_time.return_value = 0
        services.logging_service = Mock()
        services.metrics_service = Mock()
        services.validation_service = Mock()
        services.time_service = Mock()
        services.spatial_service = Mock()
        services.lifecycle_service = Mock()
        
        factory = AgentFactory(services)
        agent = factory.create_learning_agent(
            agent_id="learning_agent",
            position=(0.0, 0.0),
        )
        
        # Verify reward component is present using public API
        reward_component = agent.get_component("reward")
        assert reward_component is not None
        assert reward_component.name == "RewardComponent"
"""
Tests to verify that the reward component is the default reward system.
"""

import pytest
from unittest.mock import Mock, MagicMock

from farm.core.agent.factory import AgentFactory
from farm.core.agent.config.component_configs import AgentComponentConfig, RewardConfig
from farm.core.agent.services import AgentServices


class TestRewardComponentAsDefaultSystem:
    """Test that reward component is the default reward system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock services
        self.services = Mock(spec=AgentServices)
        self.services.get_current_time.return_value = 0
        self.services.logging_service = Mock()
        self.services.metrics_service = Mock()
        self.services.validation_service = Mock()
        self.services.time_service = Mock()
        self.services.spatial_service = Mock()
        self.services.lifecycle_service = Mock()
        
        # Create factory
        self.factory = AgentFactory(self.services)
    
    def test_default_agents_have_reward_component(self):
        """Test that default agents automatically include reward component."""
        agent = self.factory.create_default_agent(
            agent_id="default_agent",
            position=(0.0, 0.0),
        )
        
        # Verify reward component is present
        # Get reward component using public API
        reward_component = agent.get_component("reward")
        assert reward_component is not None
        assert reward_component.name == "RewardComponent"
        assert reward_component.config is not None
    
    def test_learning_agents_have_reward_component(self):
        """Test that learning agents automatically include reward component."""
        agent = self.factory.create_learning_agent(
            agent_id="learning_agent",
            position=(0.0, 0.0),
        )
        
        # Verify reward component is present
        # Get reward component using public API
        reward_component = agent.get_component("reward")
        assert reward_component is not None
        assert reward_component.name == "RewardComponent"
        assert reward_component.config is not None
    
    def test_reward_component_uses_default_config(self):
        """Test that reward component uses default configuration by default."""
        agent = self.factory.create_default_agent(
            agent_id="config_agent",
            position=(0.0, 0.0),
        )
        
        # Get reward component using public API
        reward_component = agent.get_component("reward")
        assert reward_component is not None
        
        # Verify default configuration
        config = reward_component.config
        assert config.resource_reward_scale == 1.0
        assert config.health_reward_scale == 0.5
        assert config.survival_bonus == 0.1
        assert config.death_penalty == -10.0
        assert config.age_bonus == 0.01
        assert config.combat_success_bonus == 2.0
        assert config.reproduction_bonus == 5.0
        assert config.cooperation_bonus == 1.0
        assert config.max_history_length == 1000
        assert config.recent_window == 100
        assert config.use_delta_rewards is True
        assert config.normalize_rewards is False
    
    def test_custom_reward_config_is_used(self):
        """Test that custom reward configuration is properly applied."""
        custom_config = RewardConfig(
            resource_reward_scale=2.0,
            survival_bonus=0.5,
            death_penalty=-15.0,
            combat_success_bonus=10.0,
        )
        
        agent_config = AgentComponentConfig(reward=custom_config)
        
        agent = self.factory.create_default_agent(
            agent_id="custom_agent",
            position=(0.0, 0.0),
            config=agent_config,
        )
        
        # Get reward component using public API
        reward_component = agent.get_component("reward")
        assert reward_component is not None
        
        # Verify custom configuration is used
        config = reward_component.config
        assert config.resource_reward_scale == 2.0
        assert config.survival_bonus == 0.5
        assert config.death_penalty == -15.0
        assert config.combat_success_bonus == 10.0
    
    def test_reward_component_participates_in_lifecycle(self):
        """Test that reward component participates in agent lifecycle."""
        agent = self.factory.create_default_agent(
            agent_id="lifecycle_agent",
            position=(0.0, 0.0),
        )
        
        # Get reward component using public API
        reward_component = agent.get_component("reward")
        assert reward_component is not None
        
        # Test initial state
        assert reward_component.cumulative_reward == 0.0
        assert reward_component.step_reward == 0.0
        assert reward_component.reward_history == []
        
        # Simulate agent state through components
        agent.resource_level = 50.0
        # Health is managed by combat component, so we'll work with the reward component directly
        agent.alive = True
        agent.age = 5
        
        # Test step start
        reward_component.on_step_start()
        assert reward_component.pre_action_state is not None
        assert "resource_level" in reward_component.pre_action_state
        
        # Test step end
        reward_component.on_step_end()
        assert len(reward_component.reward_history) == 1
        assert reward_component.cumulative_reward != 0.0
        assert reward_component.step_reward != 0.0
    
    def test_reward_component_calculates_rewards_correctly(self):
        """Test that reward component calculates rewards correctly."""
        agent = self.factory.create_default_agent(
            agent_id="calculation_agent",
            position=(0.0, 0.0),
        )
        
        # Get reward component using public API
        reward_component = agent.get_component("reward")
        assert reward_component is not None
        
        # Set up agent state through components
        agent.resource_level = 50.0
        # Health is managed by combat component, so we'll work with the reward component directly
        agent.alive = True
        agent.age = 5
        
        # Test delta-based reward calculation
        reward_component.on_step_start()
        
        # Change state through components
        agent.resource_level += 10.0  # Resource gain
        # Health is managed by combat component, so we'll work with the reward component directly
        
        reward_component.on_step_end()
        
        # Should have positive reward for improvements
        assert reward_component.step_reward > 0.0
        assert reward_component.cumulative_reward > 0.0
        
        # Test state-based reward calculation (fallback)
        reward_component.pre_action_state = None
        reward_component.on_step_start()
        reward_component.on_step_end()
        
        # Should still calculate reward
        assert reward_component.step_reward != 0.0
    
    def test_reward_component_handles_death_correctly(self):
        """Test that reward component handles agent death correctly."""
        agent = self.factory.create_default_agent(
            agent_id="death_agent",
            position=(0.0, 0.0),
        )
        
        # Get reward component using public API
        reward_component = agent.get_component("reward")
        assert reward_component is not None
        
        # Set up agent state through components
        agent.resource_level = 50.0
        # Health is managed by combat component, so we'll work with the reward component directly
        agent.alive = True
        
        # Simulate step
        reward_component.on_step_start()
        reward_component.on_step_end()
        
        initial_reward = reward_component.cumulative_reward
        
        # Simulate death
        agent.alive = False
        reward_component.on_terminate()
        
        # Should apply death penalty
        assert reward_component.cumulative_reward < initial_reward
        assert reward_component.cumulative_reward == initial_reward + reward_component.config.death_penalty
    
    def test_reward_component_provides_statistics(self):
        """Test that reward component provides useful statistics."""
        agent = self.factory.create_default_agent(
            agent_id="stats_agent",
            position=(0.0, 0.0),
        )
        
        # Get reward component using public API
        reward_component = agent.get_component("reward")
        assert reward_component is not None
        
        # Add some rewards
        reward_component.add_reward(1.0)
        reward_component.add_reward(2.0)
        reward_component.add_reward(3.0)
        
        # Get statistics
        stats = reward_component.get_reward_stats()
        
        # Verify statistics
        assert stats["cumulative"] == 6.0
        assert stats["average"] == 2.0
        assert stats["min"] == 1.0
        assert stats["max"] == 3.0
        assert stats["recent_average"] == 2.0
        assert "last_action" in stats
    
    def test_reward_component_can_be_reset(self):
        """Test that reward component can be reset for new episodes."""
        agent = self.factory.create_default_agent(
            agent_id="reset_agent",
            position=(0.0, 0.0),
        )
        
        # Get reward component using public API
        reward_component = agent.get_component("reward")
        assert reward_component is not None
        
        # Add some rewards
        reward_component.add_reward(5.0)
        reward_component.add_reward(3.0)
        
        # Reset
        reward_component.reset_rewards()
        
        # Should be reset
        assert reward_component.cumulative_reward == 0.0
        assert reward_component.step_reward == 0.0
        assert reward_component.reward_history == []
        assert reward_component.last_action_reward == 0.0
        assert reward_component.pre_action_state is None
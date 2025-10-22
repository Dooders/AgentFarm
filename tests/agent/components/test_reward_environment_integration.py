"""
Integration tests for reward component with the environment system.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from farm.core.agent.factory import AgentFactory
from farm.core.agent.config.component_configs import AgentComponentConfig, RewardConfig
from farm.core.agent.services import AgentServices
from farm.core.environment import Environment


class TestRewardEnvironmentIntegration:
    """Integration tests for RewardComponent with Environment."""
    
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
    
    def test_environment_uses_reward_component(self):
        """Test that environment uses reward component for reward calculation."""
        # Create agent with reward component
        agent = self.factory.create_default_agent(
            agent_id="test_agent",
            position=(0.0, 0.0),
        )
        
        # Find reward component
        reward_component = None
        for comp in agent.components:
            if hasattr(comp, 'cumulative_reward'):
                reward_component = comp
                break
        
        assert reward_component is not None
        
        # Mock environment
        env = Mock(spec=Environment)
        env._agent_objects = {"test_agent": agent}
        env._get_agent_reward = Environment._get_agent_reward.__get__(env, Environment)
        env._calculate_reward = Mock(return_value=5.0)  # Fallback reward
        
        # Test that environment gets reward from component
        # Simulate agent state
        agent.resource_level = 50.0
        agent.current_health = 80.0
        agent.alive = True
        
        # Simulate step lifecycle
        reward_component.on_step_start()
        reward_component.on_step_end()
        
        # Get reward through environment method
        reward = env._get_agent_reward("test_agent", None)
        
        # Should use reward component's calculation
        assert reward == reward_component.step_reward
        assert reward != 0.0
    
    def test_environment_fallback_to_original_calculation(self):
        """Test that environment falls back to original calculation when no reward component."""
        # Create agent without reward component (simulate old agent)
        agent = Mock()
        agent.agent_id = "old_agent"
        agent.components = []  # No components
        agent.resource_level = 50.0
        agent.current_health = 80.0
        agent.alive = True
        
        # Mock environment
        env = Mock(spec=Environment)
        env._agent_objects = {"old_agent": agent}
        env._get_agent_reward = Environment._get_agent_reward.__get__(env, Environment)
        env._calculate_reward = Mock(return_value=3.0)  # Fallback reward
        
        # Test fallback
        reward = env._get_agent_reward("old_agent", None)
        
        # Should use fallback calculation
        assert reward == 3.0
        env._calculate_reward.assert_called_once_with("old_agent", None)
    
    def test_environment_handles_missing_agent(self):
        """Test that environment handles missing agents correctly."""
        # Mock environment
        env = Mock(spec=Environment)
        env._agent_objects = {}  # No agents
        env._get_agent_reward = Environment._get_agent_reward.__get__(env, Environment)
        
        # Test missing agent
        reward = env._get_agent_reward("missing_agent", None)
        
        # Should return death penalty
        assert reward == -10.0
    
    def test_reward_component_configuration_affects_environment_rewards(self):
        """Test that reward component configuration affects environment rewards."""
        # Create custom reward configuration
        custom_config = RewardConfig(
            resource_reward_scale=2.0,
            survival_bonus=0.5,
            death_penalty=-15.0,
        )
        
        agent_config = AgentComponentConfig(reward=custom_config)
        
        # Create agent with custom config
        agent = self.factory.create_default_agent(
            agent_id="custom_agent",
            position=(0.0, 0.0),
            config=agent_config,
        )
        
        # Find reward component
        reward_component = None
        for comp in agent.components:
            if hasattr(comp, 'cumulative_reward'):
                reward_component = comp
                break
        
        assert reward_component is not None
        
        # Mock environment
        env = Mock(spec=Environment)
        env._agent_objects = {"custom_agent": agent}
        env._get_agent_reward = Environment._get_agent_reward.__get__(env, Environment)
        
        # Simulate agent state
        agent.resource_level = 50.0
        agent.current_health = 80.0
        agent.alive = True
        
        # Simulate step lifecycle
        reward_component.on_step_start()
        reward_component.on_step_end()
        
        # Get reward through environment
        reward = env._get_agent_reward("custom_agent", None)
        
        # Should use custom configuration
        assert reward == reward_component.step_reward
        assert reward != 0.0
        
        # Verify custom config is being used
        assert reward_component.config.resource_reward_scale == 2.0
        assert reward_component.config.survival_bonus == 0.5
        assert reward_component.config.death_penalty == -15.0
    
    def test_reward_component_step_lifecycle_in_environment(self):
        """Test that reward component step lifecycle works correctly in environment context."""
        agent = self.factory.create_default_agent(
            agent_id="lifecycle_agent",
            position=(0.0, 0.0),
        )
        
        # Find reward component
        reward_component = None
        for comp in agent.components:
            if hasattr(comp, 'cumulative_reward'):
                reward_component = comp
                break
        
        assert reward_component is not None
        
        # Mock environment
        env = Mock(spec=Environment)
        env._agent_objects = {"lifecycle_agent": agent}
        env._get_agent_reward = Environment._get_agent_reward.__get__(env, Environment)
        
        # Simulate multiple steps
        for step in range(3):
            # Simulate agent state changes
            agent.resource_level = 50.0 + step * 5.0
            agent.current_health = 80.0 + step * 2.0
            agent.alive = True
            
            # Simulate step lifecycle
            reward_component.on_step_start()
            reward_component.on_step_end()
            
            # Get reward through environment
            reward = env._get_agent_reward("lifecycle_agent", None)
            
            # Should have calculated reward
            assert reward == reward_component.step_reward
            assert reward != 0.0
        
        # Should have accumulated rewards
        assert len(reward_component.reward_history) == 3
        assert reward_component.cumulative_reward != 0.0
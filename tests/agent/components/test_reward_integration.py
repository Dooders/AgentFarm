"""
Integration tests for the reward component with the agent system.
"""

import pytest
from unittest.mock import Mock, MagicMock

from farm.core.agent.factory import AgentFactory
from farm.core.agent.config.component_configs import AgentComponentConfig, RewardConfig
from farm.core.agent.services import AgentServices


class TestRewardComponentIntegration:
    """Integration tests for RewardComponent with AgentFactory."""
    
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
    
    def test_factory_includes_reward_component(self):
        """Test that factory includes reward component by default."""
        agent = self.factory.create_default_agent(
            agent_id="test_agent",
            position=(0.0, 0.0),
        )
        
        # Check that reward component is included
        reward_components = [
            comp for comp in agent.components 
            if hasattr(comp, 'cumulative_reward')
        ]
        
        assert len(reward_components) == 1
        assert reward_components[0].name == "RewardComponent"
    
    def test_custom_reward_config(self):
        """Test that custom reward configuration is used."""
        custom_reward_config = RewardConfig(
            resource_reward_scale=2.0,
            survival_bonus=0.5,
            death_penalty=-15.0,
        )
        
        custom_agent_config = AgentComponentConfig(
            reward=custom_reward_config
        )
        
        agent = self.factory.create_default_agent(
            agent_id="custom_agent",
            position=(0.0, 0.0),
            config=custom_agent_config,
        )
        
        # Find reward component
        reward_component = None
        for comp in agent.components:
            if hasattr(comp, 'cumulative_reward'):
                reward_component = comp
                break
        
        assert reward_component is not None
        assert reward_component.config.resource_reward_scale == 2.0
        assert reward_component.config.survival_bonus == 0.5
        assert reward_component.config.death_penalty == -15.0
    
    def test_reward_component_lifecycle(self):
        """Test that reward component participates in agent lifecycle."""
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
        
        # Test initial state
        assert reward_component.cumulative_reward == 0.0
        assert reward_component.step_reward == 0.0
        assert reward_component.reward_history == []
        
        # Simulate step start
        reward_component.on_step_start()
        assert reward_component.pre_action_state is not None
        
        # Simulate step end
        reward_component.on_step_end()
        assert len(reward_component.reward_history) == 1
        assert reward_component.cumulative_reward != 0.0
    
    def test_learning_agent_includes_reward_component(self):
        """Test that learning agents also include reward component."""
        agent = self.factory.create_learning_agent(
            agent_id="learning_agent",
            position=(0.0, 0.0),
        )
        
        # Check that reward component is included
        reward_components = [
            comp for comp in agent.components 
            if hasattr(comp, 'cumulative_reward')
        ]
        
        assert len(reward_components) == 1
        assert reward_components[0].name == "RewardComponent"
    
    def test_reward_component_services_access(self):
        """Test that reward component can access agent services."""
        agent = self.factory.create_default_agent(
            agent_id="services_agent",
            position=(0.0, 0.0),
        )
        
        # Find reward component
        reward_component = None
        for comp in agent.components:
            if hasattr(comp, 'cumulative_reward'):
                reward_component = comp
                break
        
        assert reward_component is not None
        
        # Test service access
        assert reward_component.logging_service is not None
        assert reward_component.metrics_service is not None
        assert reward_component.validation_service is not None
        assert reward_component.time_service is not None
        assert reward_component.spatial_service is not None
        assert reward_component.lifecycle_service is not None
    
    def test_reward_component_with_agent_state_changes(self):
        """Test reward component with actual agent state changes."""
        agent = self.factory.create_default_agent(
            agent_id="state_agent",
            position=(0.0, 0.0),
        )
        
        # Find reward component
        reward_component = None
        for comp in agent.components:
            if hasattr(comp, 'cumulative_reward'):
                reward_component = comp
                break
        
        assert reward_component is not None
        
        # Simulate agent state changes
        if hasattr(agent, 'resource_level'):
            agent.resource_level = 50.0
        if hasattr(agent, 'current_health'):
            agent.current_health = 80.0
        if hasattr(agent, 'alive'):
            agent.alive = True
        if hasattr(agent, 'age'):
            agent.age = 5
        
        # Test step lifecycle
        reward_component.on_step_start()
        
        # Change state
        if hasattr(agent, 'resource_level'):
            agent.resource_level += 10.0
        if hasattr(agent, 'current_health'):
            agent.current_health += 5.0
        
        reward_component.on_step_end()
        
        # Should have calculated rewards
        assert len(reward_component.reward_history) == 1
        assert reward_component.cumulative_reward != 0.0
        assert reward_component.step_reward != 0.0
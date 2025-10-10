"""
Tests for perception system migration to multi-channel observations.

This test suite verifies that the migration from the old 4-value grid system
to the new multi-channel AgentObservation system is working correctly.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from farm.core.agent import AgentFactory
from farm.core.environment import Environment
from farm.core.agent.behaviors.learning_behavior import LearningAgentBehavior


class TestPerceptionMigration:
    """Test suite for perception system migration."""

    def test_multi_channel_observation_integration(self):
        """Test that PerceptionComponent provides multi-channel observations."""
        # Create environment and agent
        env = Environment(width=100, height=100, resource_distribution={'amount': 10})
        spatial_service = Mock()
        spatial_service.get_nearby = Mock(return_value={'agents': [], 'resources': []})
        spatial_service.get_nearest = Mock(return_value={'agents': None, 'resources': None})

        factory = AgentFactory(spatial_service=spatial_service)
        agent = factory.create_default_agent('test_agent', position=(50.0, 50.0))
        env.add_agent(agent)

        # Test PerceptionComponent multi-channel observation
        perception = agent.get_component('perception')
        assert perception is not None
        
        observation = perception.get_observation()
        assert observation is not None
        
        tensor = observation.tensor()
        assert tensor.shape == (13, 13, 13)  # 13 channels, 13x13 grid
        assert str(tensor.dtype) == 'torch.float32'

    def test_environment_uses_perception_component_observations(self):
        """Test that Environment uses PerceptionComponent observations."""
        # Create environment and agent
        env = Environment(width=100, height=100, resource_distribution={'amount': 10})
        spatial_service = Mock()
        spatial_service.get_nearby = Mock(return_value={'agents': [], 'resources': []})
        spatial_service.get_nearest = Mock(return_value={'agents': None, 'resources': None})

        factory = AgentFactory(spatial_service=spatial_service)
        agent = factory.create_default_agent('test_agent', position=(50.0, 50.0))
        env.add_agent(agent)

        # Test Environment observation
        env_obs = env._get_observation('test_agent')
        assert env_obs.shape == (13, 13, 13)
        assert env_obs.dtype == np.float32

        # Test that Environment and PerceptionComponent use same observation
        perception = agent.get_component('perception')
        perception_obs = perception.get_observation().tensor().cpu().numpy()
        assert env_obs.shape == perception_obs.shape

    def test_learning_behavior_uses_multi_channel_observations(self):
        """Test that LearningAgentBehavior uses multi-channel observations."""
        # Create environment and agent
        env = Environment(width=100, height=100, resource_distribution={'amount': 10})
        spatial_service = Mock()
        spatial_service.get_nearby = Mock(return_value={'agents': [], 'resources': []})
        spatial_service.get_nearest = Mock(return_value={'agents': None, 'resources': None})

        factory = AgentFactory(spatial_service=spatial_service)
        agent = factory.create_default_agent('test_agent', position=(50.0, 50.0))
        env.add_agent(agent)

        # Test LearningAgentBehavior observation
        behavior = LearningAgentBehavior()
        behavior_obs = behavior._create_state_observation(agent)
        
        # Should be flattened multi-channel observation
        assert behavior_obs.shape == (2197,)  # 13 * 13 * 13 = 2197
        assert str(behavior_obs.dtype) == 'torch.float32'

    def test_observation_shape_consistency(self):
        """Test that all observation systems provide consistent shapes."""
        # Create environment and agent
        env = Environment(width=100, height=100, resource_distribution={'amount': 10})
        spatial_service = Mock()
        spatial_service.get_nearby = Mock(return_value={'agents': [], 'resources': []})
        spatial_service.get_nearest = Mock(return_value={'agents': None, 'resources': None})

        factory = AgentFactory(spatial_service=spatial_service)
        agent = factory.create_default_agent('test_agent', position=(50.0, 50.0))
        env.add_agent(agent)

        # Get observations from all systems
        perception = agent.get_component('perception')
        perception_obs = perception.get_observation().tensor()
        
        env_obs = env._get_observation('test_agent')
        
        behavior = LearningAgentBehavior()
        behavior_obs = behavior._create_state_observation(agent)

        # Test shape consistency
        assert perception_obs.shape == (13, 13, 13)
        assert env_obs.shape == (13, 13, 13)
        assert behavior_obs.shape == (2197,)  # Flattened version
        
        # Test that flattened behavior observation matches perception observation
        perception_flattened = perception_obs.flatten()
        assert behavior_obs.shape == perception_flattened.shape

    def test_backward_compatibility_deprecated_method(self):
        """Test that deprecated create_perception_grid still works with warning."""
        # Create environment and agent
        env = Environment(width=100, height=100, resource_distribution={'amount': 10})
        spatial_service = Mock()
        spatial_service.get_nearby = Mock(return_value={'agents': [], 'resources': []})
        spatial_service.get_nearest = Mock(return_value={'agents': None, 'resources': None})

        factory = AgentFactory(spatial_service=spatial_service)
        agent = factory.create_default_agent('test_agent', position=(50.0, 50.0))
        env.add_agent(agent)

        # Test deprecated method with warning
        perception = agent.get_component('perception')
        
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            grid = perception.create_perception_grid()
            
            # Should show deprecation warning
            assert len(w) > 0
            assert "deprecated" in str(w[0].message).lower()
            
            # Should still return valid grid
            assert grid.shape == (11, 11)  # Old grid size
            assert grid.dtype == np.int8

    def test_memory_efficiency_no_duplicate_observations(self):
        """Test that there are no duplicate observation systems."""
        # Create environment and agent
        env = Environment(width=100, height=100, resource_distribution={'amount': 10})
        spatial_service = Mock()
        spatial_service.get_nearby = Mock(return_value={'agents': [], 'resources': []})
        spatial_service.get_nearest = Mock(return_value={'agents': None, 'resources': None})

        factory = AgentFactory(spatial_service=spatial_service)
        agent = factory.create_default_agent('test_agent', position=(50.0, 50.0))
        env.add_agent(agent)

        # Test that Environment doesn't have separate agent_observations
        assert not hasattr(env, 'agent_observations') or len(env.agent_observations) == 0
        
        # Test that PerceptionComponent provides the observation
        perception = agent.get_component('perception')
        assert perception.get_observation() is not None

    def test_configuration_synchronization(self):
        """Test that PerceptionComponent syncs with Environment observation config."""
        # Create environment and agent
        env = Environment(width=100, height=100, resource_distribution={'amount': 10})
        spatial_service = Mock()
        spatial_service.get_nearby = Mock(return_value={'agents': [], 'resources': []})
        spatial_service.get_nearest = Mock(return_value={'agents': None, 'resources': None})

        factory = AgentFactory(spatial_service=spatial_service)
        agent = factory.create_default_agent('test_agent', position=(50.0, 50.0))
        
        # Before adding to environment, should have default radius
        perception = agent.get_component('perception')
        initial_obs = perception.get_observation()
        initial_shape = initial_obs.tensor().shape if initial_obs else None
        
        # Add to environment (should trigger sync)
        env.add_agent(agent)
        
        # After adding to environment, should sync with environment config
        final_obs = perception.get_observation()
        final_shape = final_obs.tensor().shape
        
        # Should have synced to environment's observation config (R=6 -> 13x13)
        assert final_shape == (13, 13, 13)
        assert final_shape != initial_shape  # Should have changed from default

    def test_migration_completeness(self):
        """Test that migration is complete and no old PerceptionData is used."""
        # Test that old PerceptionData class is not available
        try:
            from farm.core.perception import PerceptionData  # type: ignore
            pytest.fail("PerceptionData should not be available - file should be removed")
        except ImportError:
            # This is expected - the file should be removed
            pass
        
        # Test that new system works without old dependencies
        env = Environment(width=100, height=100, resource_distribution={'amount': 10})
        spatial_service = Mock()
        spatial_service.get_nearby = Mock(return_value={'agents': [], 'resources': []})
        spatial_service.get_nearest = Mock(return_value={'agents': None, 'resources': None})

        factory = AgentFactory(spatial_service=spatial_service)
        agent = factory.create_default_agent('test_agent', position=(50.0, 50.0))
        env.add_agent(agent)

        # Should work without any old perception dependencies
        perception = agent.get_component('perception')
        observation = perception.get_observation()
        assert observation is not None
        assert observation.tensor().shape == (13, 13, 13)

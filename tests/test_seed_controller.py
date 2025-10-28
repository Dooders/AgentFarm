"""Unit tests for the SeedController class.

Tests verify deterministic RNG generation, per-agent isolation,
and proper integration with the simulation system.
"""

import pytest
import random
import numpy as np
import torch
from unittest.mock import Mock

from farm.core.seed_controller import SeedController


class TestSeedController:
    """Test cases for SeedController class."""

    def test_initialization(self):
        """Test SeedController initialization."""
        seed = 42
        controller = SeedController(seed)
        
        assert controller.global_seed == seed

    def test_get_agent_rng_deterministic(self):
        """Test that get_agent_rng returns deterministic results."""
        controller = SeedController(42)
        agent_id = "test_agent_001"
        
        # Get RNG instances twice
        py_rng1, np_rng1, torch_gen1 = controller.get_agent_rng(agent_id)
        py_rng2, np_rng2, torch_gen2 = controller.get_agent_rng(agent_id)
        
        # Should be different instances but produce same sequences
        assert py_rng1 is not py_rng2
        assert np_rng1 is not np_rng2
        assert torch_gen1 is not torch_gen2
        
        # Test deterministic behavior
        values1 = [py_rng1.random() for _ in range(10)]
        values2 = [py_rng2.random() for _ in range(10)]
        assert values1 == values2
        
        # Test numpy RNG
        np_values1 = [np_rng1.random() for _ in range(10)]
        np_values2 = [np_rng2.random() for _ in range(10)]
        assert np_values1 == np_values2
        
        # Test torch generator
        torch_values1 = [torch.rand(1, generator=torch_gen1).item() for _ in range(10)]
        torch_values2 = [torch.rand(1, generator=torch_gen2).item() for _ in range(10)]
        assert torch_values1 == torch_values2

    def test_different_agents_get_different_seeds(self):
        """Test that different agents get different RNG seeds."""
        controller = SeedController(42)
        
        agent1_rng = controller.get_agent_rng("agent_001")
        agent2_rng = controller.get_agent_rng("agent_002")
        
        # Different agents should produce different random sequences
        py_values1 = [agent1_rng[0].random() for _ in range(10)]
        py_values2 = [agent2_rng[0].random() for _ in range(10)]
        assert py_values1 != py_values2
        
        np_values1 = [agent1_rng[1].random() for _ in range(10)]
        np_values2 = [agent2_rng[1].random() for _ in range(10)]
        assert np_values1 != np_values2

    def test_same_agent_same_seed_across_controllers(self):
        """Test that same agent gets same seed across different controllers."""
        controller1 = SeedController(42)
        controller2 = SeedController(42)
        
        agent_id = "consistent_agent"
        
        py_rng1, np_rng1, torch_gen1 = controller1.get_agent_rng(agent_id)
        py_rng2, np_rng2, torch_gen2 = controller2.get_agent_rng(agent_id)
        
        # Should produce identical sequences
        py_values1 = [py_rng1.random() for _ in range(10)]
        py_values2 = [py_rng2.random() for _ in range(10)]
        assert py_values1 == py_values2
        
        np_values1 = [np_rng1.random() for _ in range(10)]
        np_values2 = [np_rng2.random() for _ in range(10)]
        assert np_values1 == np_values2

    def test_different_global_seeds_produce_different_results(self):
        """Test that different global seeds produce different agent seeds."""
        controller1 = SeedController(42)
        controller2 = SeedController(123)
        
        agent_id = "test_agent"
        
        py_rng1, np_rng1, torch_gen1 = controller1.get_agent_rng(agent_id)
        py_rng2, np_rng2, torch_gen2 = controller2.get_agent_rng(agent_id)
        
        # Different global seeds should produce different sequences
        py_values1 = [py_rng1.random() for _ in range(10)]
        py_values2 = [py_rng2.random() for _ in range(10)]
        assert py_values1 != py_values2
        
        np_values1 = [np_rng1.random() for _ in range(10)]
        np_values2 = [np_rng2.random() for _ in range(10)]
        assert np_values1 != np_values2

    def test_get_component_rng(self):
        """Test component-specific RNG generation."""
        controller = SeedController(42)
        agent_id = "test_agent"
        component_name = "movement"
        
        py_rng, np_rng, torch_gen = controller.get_component_rng(agent_id, component_name)
        
        # Should return valid RNG instances
        assert isinstance(py_rng, random.Random)
        assert isinstance(np_rng, np.random.Generator)
        assert isinstance(torch_gen, torch.Generator)
        
        # Should be deterministic
        py_rng2, np_rng2, torch_gen2 = controller.get_component_rng(agent_id, component_name)
        
        py_values1 = [py_rng.random() for _ in range(5)]
        py_values2 = [py_rng2.random() for _ in range(5)]
        assert py_values1 == py_values2

    def test_different_components_different_seeds(self):
        """Test that different components get different seeds."""
        controller = SeedController(42)
        agent_id = "test_agent"
        
        movement_rng = controller.get_component_rng(agent_id, "movement")
        perception_rng = controller.get_component_rng(agent_id, "perception")
        
        # Different components should produce different sequences
        movement_values = [movement_rng[0].random() for _ in range(10)]
        perception_values = [perception_rng[0].random() for _ in range(10)]
        assert movement_values != perception_values

    def test_seed_value_range(self):
        """Test that derived seeds are within valid range."""
        controller = SeedController(42)
        
        # Test with various agent IDs
        agent_ids = ["agent_001", "agent_002", "very_long_agent_id_12345", "short"]
        
        for agent_id in agent_ids:
            py_rng, np_rng, torch_gen = controller.get_agent_rng(agent_id)
            
            # All RNGs should work without errors
            py_value = py_rng.random()
            np_value = np_rng.random()
            torch_value = torch.rand(1, generator=torch_gen).item()
            
            assert 0 <= py_value <= 1
            assert 0 <= np_value <= 1
            assert 0 <= torch_value <= 1

    def test_hash_determinism(self):
        """Test that hash-based seed derivation is deterministic."""
        controller = SeedController(42)
        agent_id = "hash_test_agent"
        
        # Get RNG multiple times
        rngs1 = [controller.get_agent_rng(agent_id) for _ in range(5)]
        rngs2 = [controller.get_agent_rng(agent_id) for _ in range(5)]
        
        # All should produce identical sequences
        for i in range(5):
            py_values1 = [rngs1[i][0].random() for _ in range(10)]
            py_values2 = [rngs2[i][0].random() for _ in range(10)]
            assert py_values1 == py_values2

    def test_edge_case_agent_ids(self):
        """Test edge cases for agent IDs."""
        controller = SeedController(42)
        
        edge_cases = [
            "",  # Empty string
            " ",  # Space
            "agent_with_special_chars!@#$%^&*()",
            "agent\nwith\nnewlines",
            "agent\twith\ttabs",
            "agent_with_unicode_🚀",
            "very_long_agent_id_" + "x" * 1000,  # Very long ID
        ]
        
        for agent_id in edge_cases:
            # Should not raise exceptions
            py_rng, np_rng, torch_gen = controller.get_agent_rng(agent_id)
            
            # Should produce valid random values
            assert 0 <= py_rng.random() <= 1
            assert 0 <= np_rng.random() <= 1
            assert 0 <= torch.rand(1, generator=torch_gen).item() <= 1

    def test_edge_case_seeds(self):
        """Test edge cases for global seeds."""
        edge_seeds = [0, 1, -1, 2**31 - 1, -2**31]
        
        for seed in edge_seeds:
            controller = SeedController(seed)
            py_rng, np_rng, torch_gen = controller.get_agent_rng("test_agent")
            
            # Should produce valid random values
            assert 0 <= py_rng.random() <= 1
            assert 0 <= np_rng.random() <= 1
            assert 0 <= torch.rand(1, generator=torch_gen).item() <= 1


class TestSeedControllerIntegration:
    """Integration tests for SeedController with other components."""

    def test_with_agent_factory(self):
        """Test SeedController integration with AgentFactory."""
        from farm.core.agent.factory import AgentFactory
        from farm.core.agent.services import AgentServices
        
        # Create services with seed controller
        seed_controller = SeedController(42)
        services = AgentServices(
            spatial_service=Mock(),
            seed_controller=seed_controller,
        )
        
        factory = AgentFactory(services)
        
        # Create agents
        agent1 = factory.create_default_agent("agent_001", (0.0, 0.0))
        agent2 = factory.create_default_agent("agent_002", (0.0, 0.0))
        
        # Should have per-agent RNGs injected
        assert hasattr(agent1, '_py_rng')
        assert hasattr(agent1, '_np_rng')
        assert hasattr(agent1, '_torch_gen')
        
        assert hasattr(agent2, '_py_rng')
        assert hasattr(agent2, '_np_rng')
        assert hasattr(agent2, '_torch_gen')
        
        # Different agents should have different RNGs
        assert agent1._py_rng is not agent2._py_rng
        assert agent1._np_rng is not agent2._np_rng
        assert agent1._torch_gen is not agent2._torch_gen

    def test_without_seed_controller(self):
        """Test AgentFactory behavior when no SeedController is provided."""
        from farm.core.agent.factory import AgentFactory
        from farm.core.agent.services import AgentServices
        
        # Create services without seed controller
        services = AgentServices(
            spatial_service=Mock(),
            seed_controller=None,
        )
        
        factory = AgentFactory(services)
        agent = factory.create_default_agent("agent_001", (0.0, 0.0))
        
        # Should not have per-agent RNGs
        assert not hasattr(agent, '_py_rng')
        assert not hasattr(agent, '_np_rng')
        assert not hasattr(agent, '_torch_gen')

    def test_behavior_uses_per_agent_rng(self):
        """Test that DefaultAgentBehavior uses per-agent RNG when available."""
        from farm.core.agent.factory import AgentFactory
        from farm.core.agent.services import AgentServices
        from farm.core.agent.behaviors.default import DefaultAgentBehavior
        import torch
        
        # Create services with seed controller
        seed_controller = SeedController(42)
        services = AgentServices(
            spatial_service=Mock(),
            seed_controller=seed_controller,
        )
        
        factory = AgentFactory(services)
        agent = factory.create_default_agent("agent_001", (0.0, 0.0))
        
        # Mock actions
        agent.actions = [Mock(name="action1"), Mock(name="action2")]
        
        # Create behavior
        behavior = DefaultAgentBehavior()
        
        # Test action selection
        state = torch.zeros((1, 11, 11))
        action = behavior.decide_action(agent, state)
        
        # Should select a valid action
        assert action in agent.actions
        
        # Test multiple calls produce different results (due to randomness)
        actions = [behavior.decide_action(agent, state) for _ in range(10)]
        # Should have some variation (not all the same action)
        assert len(set(action.name for action in actions)) > 1

    def test_decision_module_uses_per_agent_rng(self):
        """Test that DecisionModule uses per-agent RNG when available."""
        from farm.core.decision.decision import DecisionModule
        from farm.core.decision.config import DecisionConfig
        from gymnasium import spaces
        import torch
        
        # Create mock agent with per-agent RNG
        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        
        # Create per-agent RNG
        seed_controller = SeedController(42)
        py_rng, np_rng, torch_gen = seed_controller.get_agent_rng("test_agent")
        mock_agent._np_rng = np_rng
        
        # Create decision module with fallback algorithm
        action_space = spaces.Discrete(7)
        observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        config = DecisionConfig(algorithm_type="fallback")  # Force fallback algorithm
        
        decision_module = DecisionModule(
            mock_agent, action_space, observation_space, config
        )
        
        # Test action decision
        state = torch.zeros((1, 8))
        action = decision_module.decide_action(state)
        
        # Should return valid action
        assert 0 <= action < 7
        
        # Test that per-agent RNG is being used by checking fallback algorithm
        # The fallback algorithm should use the per-agent RNG
        fallback_algo = decision_module.algorithm
        assert hasattr(fallback_algo, 'agent')
        assert fallback_algo.agent._np_rng is np_rng


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

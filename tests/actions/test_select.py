"""Unit tests for the select module.

This module tests the action selection functionality including configuration,
neural network architecture, action selection, and state creation.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch

from farm.core.decision.decision import (
    DecisionConfig,
    DecisionModule,
    DecisionQNetwork,
    create_decision_state,
)


class TestSelectConfig(unittest.TestCase):
    """Test cases for SelectConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DecisionConfig()

        # Base action weights
        self.assertEqual(config.move_weight, 0.3)
        self.assertEqual(config.gather_weight, 0.3)
        self.assertEqual(config.share_weight, 0.15)
        self.assertEqual(config.attack_weight, 0.1)
        self.assertEqual(config.reproduce_weight, 0.15)

        # State-based multipliers
        self.assertEqual(config.move_mult_no_resources, 1.5)
        self.assertEqual(config.gather_mult_low_resources, 1.5)
        self.assertEqual(config.share_mult_wealthy, 1.3)
        self.assertEqual(config.share_mult_poor, 0.5)
        self.assertEqual(config.attack_mult_desperate, 1.4)
        self.assertEqual(config.attack_mult_stable, 0.6)
        self.assertEqual(config.reproduce_mult_wealthy, 1.4)
        self.assertEqual(config.reproduce_mult_poor, 0.3)

        # Thresholds
        self.assertEqual(config.attack_starvation_threshold, 0.5)
        self.assertEqual(config.attack_defense_threshold, 0.3)
        self.assertEqual(config.reproduce_resource_threshold, 0.7)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DecisionConfig()
        config.move_weight = 0.4
        config.gather_weight = 0.2
        config.share_weight = 0.2
        config.attack_weight = 0.1
        config.reproduce_weight = 0.1

        self.assertEqual(config.move_weight, 0.4)
        self.assertEqual(config.gather_weight, 0.2)
        self.assertEqual(config.share_weight, 0.2)
        self.assertEqual(config.attack_weight, 0.1)
        self.assertEqual(config.reproduce_weight, 0.1)

    def test_inheritance_from_base_config(self):
        """Test that SelectConfig inherits from BaseDQNConfig."""
        config = DecisionConfig()

        # Check that base DQN config attributes are available
        self.assertEqual(config.memory_size, 10000)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.gamma, 0.99)


class TestSelectQNetwork(unittest.TestCase):
    """Test cases for SelectQNetwork class."""

    def test_initialization(self):
        """Test SelectQNetwork initialization."""
        network = DecisionQNetwork(input_dim=8, num_actions=5, hidden_size=64)

        self.assertEqual(network.network[-1].out_features, 5)  # 5 actions

    def test_forward_pass(self):
        """Test SelectQNetwork forward pass."""
        network = DecisionQNetwork(input_dim=8, num_actions=5, hidden_size=64)
        x = torch.randn(2, 8)

        output = network(x)

        self.assertEqual(output.shape, (2, 5))  # Batch size 2, 5 actions

    def test_forward_pass_single_sample(self):
        """Test SelectQNetwork forward pass with single sample."""
        network = DecisionQNetwork(input_dim=8, num_actions=5, hidden_size=64)
        x = torch.randn(8)

        # Move input to the same device as the network
        device = next(network.parameters()).device
        x = x.to(device)

        output = network(x)

        self.assertEqual(output.shape, (5,))  # 5 actions

    def test_initialization_with_shared_encoder(self):
        """Test SelectQNetwork initialization with shared encoder."""
        from farm.core.decision.base_dqn import SharedEncoder

        shared_encoder = SharedEncoder(input_dim=8, hidden_size=64)
        network = DecisionQNetwork(
            input_dim=8, num_actions=5, hidden_size=64, shared_encoder=shared_encoder
        )

        self.assertIs(network.shared_encoder, shared_encoder)


class TestSelectModule(unittest.TestCase):
    """Test cases for SelectModule class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DecisionConfig()
        self.config.memory_size = 100
        self.config.batch_size = 4
        self.config.epsilon_start = 1.0
        self.config.epsilon_min = 0.01
        self.config.epsilon_decay = 0.9

    def test_initialization(self):
        """Test SelectModule initialization."""
        module = DecisionModule(num_actions=5, config=self.config)

        self.assertEqual(module.output_dim, 5)  # 5 actions
        self.assertEqual(module.config, self.config)

    def test_select_action_exploration(self):
        """Test action selection during exploration."""
        module = DecisionModule(num_actions=5, config=self.config)
        module.epsilon = 1.0  # Always explore

        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        # Set up required attributes for the mock agent
        mock_agent.resource_level = 50.0
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 80.0
        mock_agent.starvation_threshold = 10.0
        mock_agent.max_starvation = 20.0

        # Use a simple object instead of Mock for config to avoid Mock return values
        class MockConfig:
            gathering_range = 30
            social_range = 30
            min_reproduction_resources = 8
            max_population = 300

        mock_agent.config = MockConfig()
        mock_agent.environment = Mock()
        mock_agent.environment.get_nearby_resources.return_value = []
        mock_agent.environment.get_nearby_agents.return_value = [
            Mock(),
            Mock(),
        ]  # Add nearby agents
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        # Create proper Action objects
        from farm.core.action import Action

        mock_actions = [
            Action("move", 0.3, Mock()),
            Action("gather", 0.3, Mock()),
            Action("share", 0.15, Mock()),
            Action("attack", 0.1, Mock()),
            Action("reproduce", 0.15, Mock()),
        ]
        state = torch.randn(8).to(module.device)

        # Test multiple selections to ensure randomness
        actions = set()
        for _ in range(10):
            action = module.decide_action(mock_agent, mock_actions, state)
            actions.add(action)

        # Should have some variety in actions
        self.assertGreater(len(actions), 1)

    def test_select_action_exploitation(self):
        """Test action selection during exploitation."""
        module = DecisionModule(num_actions=5, config=self.config)
        module.epsilon = 0.0  # Always exploit

        mock_agent = Mock()
        mock_agent.agent_id = "test_agent"
        # Set up required attributes for the mock agent
        mock_agent.resource_level = 50.0
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 80.0
        mock_agent.starvation_threshold = 10.0
        mock_agent.max_starvation = 20.0

        # Use a simple object instead of Mock for config to avoid Mock return values
        class MockConfig:
            gathering_range = 30
            social_range = 30
            min_reproduction_resources = 8
            max_population = 300

        mock_agent.config = MockConfig()
        mock_agent.environment = Mock()
        mock_agent.environment.get_nearby_resources.return_value = []
        mock_agent.environment.get_nearby_agents.return_value = [
            Mock(),
            Mock(),
        ]  # Add nearby agents
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        # Create proper Action objects
        from farm.core.action import Action

        mock_actions = [
            Action("move", 0.3, Mock()),
            Action("gather", 0.3, Mock()),
            Action("share", 0.15, Mock()),
            Action("attack", 0.1, Mock()),
            Action("reproduce", 0.15, Mock()),
        ]
        state = torch.randn(8).to(module.device)

        # Should return actions from the same probability distribution
        # (Note: Even with epsilon=0, there's still some randomness in the final selection)
        # Test that the method doesn't crash and returns valid actions
        action1 = module.decide_action(mock_agent, mock_actions, state)
        action2 = module.decide_action(mock_agent, mock_actions, state)

        # Both should be valid actions from the available actions list
        self.assertIn(action1, mock_actions)
        self.assertIn(action2, mock_actions)

    def test_fast_adjust_probabilities(self):
        """Test fast probability adjustment based on agent state."""
        module = DecisionModule(num_actions=5, config=self.config)

        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.resource_level = 10.0
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 30.0  # Low health
        mock_agent.starvation_threshold = 10.0
        mock_agent.max_starvation = 20.0

        # Use a simple object instead of Mock for config to avoid Mock return values
        class MockConfig:
            gathering_range = 30
            social_range = 30
            min_reproduction_resources = 8
            max_population = 300

        mock_agent.config = MockConfig()
        mock_agent.environment = Mock()
        mock_agent.environment.get_nearby_resources.return_value = []
        mock_agent.environment.get_nearby_agents.return_value = [
            Mock(),
            Mock(),
        ]  # Add nearby agents
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        base_probs = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal probabilities
        action_indices = {
            "move": 0,
            "gather": 1,
            "share": 2,
            "attack": 3,
            "reproduce": 4,
        }

        adjusted_probs = module._fast_adjust_probabilities(
            mock_agent, base_probs, action_indices
        )

        # Should have different probabilities after adjustment
        self.assertNotEqual(adjusted_probs, base_probs)

        # Should still sum to approximately 1.0
        self.assertAlmostEqual(sum(adjusted_probs), 1.0, places=5)

    def test_fast_adjust_probabilities_wealthy_agent(self):
        """Test probability adjustment for wealthy agent."""
        module = DecisionModule(num_actions=5, config=self.config)

        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.resource_level = 80.0  # High resources
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 90.0  # High health
        mock_agent.starvation_threshold = 10.0
        mock_agent.max_starvation = 20.0

        # Use a simple object instead of Mock for config to avoid Mock return values
        class MockConfig:
            gathering_range = 30
            social_range = 30
            min_reproduction_resources = 8
            max_population = 300

        mock_agent.config = MockConfig()
        mock_agent.environment = Mock()
        mock_agent.environment.get_nearby_resources.return_value = []
        mock_agent.environment.get_nearby_agents.return_value = [
            Mock(),
            Mock(),
        ]  # Add nearby agents
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        base_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        action_indices = {
            "move": 0,
            "gather": 1,
            "share": 2,
            "attack": 3,
            "reproduce": 4,
        }

        adjusted_probs = module._fast_adjust_probabilities(
            mock_agent, base_probs, action_indices
        )

        # Share and reproduce should be boosted for wealthy agent
        self.assertGreater(adjusted_probs[2], base_probs[2])  # share
        self.assertGreater(adjusted_probs[4], base_probs[4])  # reproduce

    def test_fast_adjust_probabilities_desperate_agent(self):
        """Test probability adjustment for desperate agent."""
        module = DecisionModule(num_actions=5, config=self.config)

        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.resource_level = 5.0  # Low resources
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 35.0  # Low health but above defense threshold
        mock_agent.starvation_threshold = 11.0  # Make starvation_risk > 0.5
        mock_agent.max_starvation = 20.0

        # Use a simple object instead of Mock for config to avoid Mock return values
        class MockConfig:
            gathering_range = 30
            social_range = 30
            min_reproduction_resources = 8
            max_population = 300

        mock_agent.config = MockConfig()
        mock_agent.environment = Mock()
        mock_agent.environment.get_nearby_resources.return_value = []
        mock_agent.environment.get_nearby_agents.return_value = [
            Mock(),
            Mock(),
        ]  # Add nearby agents
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        base_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
        action_indices = {
            "move": 0,
            "gather": 1,
            "share": 2,
            "attack": 3,
            "reproduce": 4,
        }

        adjusted_probs = module._fast_adjust_probabilities(
            mock_agent, base_probs, action_indices
        )

        # Attack should be boosted for desperate agent
        self.assertGreater(adjusted_probs[3], base_probs[3])  # attack

    def test_combine_probs_and_qvalues(self):
        """Test combining probabilities and Q-values."""
        module = DecisionModule(num_actions=5, config=self.config)

        probs = [0.2, 0.3, 0.1, 0.2, 0.2]
        q_values = np.array([0.5, 0.8, 0.3, 0.6, 0.4])

        combined_probs = module._combine_probs_and_qvalues(probs, q_values)

        # Should have different probabilities after combination
        # Use numpy array comparison instead of direct comparison
        self.assertFalse(np.array_equal(combined_probs, probs))

        # Should still sum to approximately 1.0
        self.assertAlmostEqual(sum(combined_probs), 1.0, places=5)

    def test_network_initialization(self):
        """Test that Q-networks are properly initialized."""
        module = DecisionModule(num_actions=5, config=self.config)

        self.assertIsInstance(module.q_network, DecisionQNetwork)
        self.assertIsInstance(module.target_network, DecisionQNetwork)

        # Check that networks have correct output dimensions
        test_input = torch.randn(8)
        # Move input to the same device as the network
        device = next(module.q_network.parameters()).device
        test_input = test_input.to(device)

        q_output = module.q_network(test_input)
        target_output = module.target_network(test_input)

        self.assertEqual(q_output.shape, (5,))  # 5 actions
        self.assertEqual(target_output.shape, (5,))  # 5 actions


class TestCreateSelectionState(unittest.TestCase):
    """Test cases for create_selection_state function."""

    def test_create_selection_state_basic(self):
        """Test basic selection state creation."""
        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.resource_level = 50.0
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 80.0
        mock_agent.starvation_threshold = 10.0
        mock_agent.max_starvation = 20.0
        mock_agent.is_defending = False
        mock_agent.alive = True
        mock_agent.device = torch.device("cpu")
        mock_agent.config = Mock()
        mock_agent.config.min_reproduction_resources = 8
        mock_agent.config.gathering_range = 30
        mock_agent.config.social_range = 30
        mock_agent.environment = Mock()
        mock_agent.environment.width = 20.0
        mock_agent.environment.height = 20.0
        mock_agent.environment.time = 10
        mock_agent.environment.resources = [Mock(), Mock()]  # 2 resources
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        # Mock environment to return some resources
        mock_resource = Mock()
        mock_resource.position = (12.0, 10.0)
        mock_agent.environment.get_nearby_resources.return_value = [mock_resource]

        # Mock environment to return some agents
        mock_other_agent = Mock()
        mock_other_agent.position = (11.0, 10.0)
        mock_other_agent.resource_level = 30.0
        mock_agent.environment.get_nearby_agents.return_value = [mock_other_agent]

        state = create_decision_state(mock_agent)

        self.assertEqual(state.shape, (8,))
        self.assertIsInstance(state, torch.Tensor)

    def test_create_selection_state_no_resources(self):
        """Test selection state creation when no resources are nearby."""
        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.resource_level = 50.0
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 80.0
        mock_agent.starvation_threshold = 10.0
        mock_agent.max_starvation = 20.0
        mock_agent.is_defending = False
        mock_agent.alive = True
        mock_agent.device = torch.device("cpu")
        mock_agent.config = Mock()
        mock_agent.config.min_reproduction_resources = 8
        mock_agent.config.gathering_range = 30
        mock_agent.config.social_range = 30
        mock_agent.environment = Mock()
        mock_agent.environment.width = 20.0
        mock_agent.environment.height = 20.0
        mock_agent.environment.time = 10
        mock_agent.environment.resources = [Mock(), Mock()]  # 2 resources
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        # Mock environment to return no resources
        mock_agent.environment.get_nearby_resources.return_value = []
        mock_agent.environment.get_nearby_agents.return_value = [
            Mock(),
            Mock(),
        ]  # Add nearby agents

        state = create_decision_state(mock_agent)

        self.assertEqual(state.shape, (8,))
        self.assertIsInstance(state, torch.Tensor)

    def test_create_selection_state_edge_positions(self):
        """Test selection state creation at edge positions."""
        mock_agent = Mock()
        mock_agent.position = (0.0, 0.0)  # At edge
        mock_agent.resource_level = 50.0
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 80.0
        mock_agent.starvation_threshold = 10.0
        mock_agent.max_starvation = 20.0
        mock_agent.is_defending = False
        mock_agent.alive = True
        mock_agent.device = torch.device("cpu")
        mock_agent.config = Mock()
        mock_agent.config.min_reproduction_resources = 8
        mock_agent.config.gathering_range = 30
        mock_agent.config.social_range = 30
        mock_agent.environment = Mock()
        mock_agent.environment.width = 20.0
        mock_agent.environment.height = 20.0
        mock_agent.environment.time = 10
        mock_agent.environment.resources = [Mock(), Mock()]  # 2 resources
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        mock_agent.environment.get_nearby_resources.return_value = []
        mock_agent.environment.get_nearby_agents.return_value = [
            Mock(),
            Mock(),
        ]  # Add nearby agents

        state = create_decision_state(mock_agent)

        self.assertEqual(state.shape, (8,))
        self.assertIsInstance(state, torch.Tensor)

    def test_create_selection_state_high_resources(self):
        """Test selection state creation when agent has high resources."""
        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.resource_level = 90.0  # High resources
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 95.0  # High health
        mock_agent.starvation_threshold = 10.0
        mock_agent.max_starvation = 20.0
        mock_agent.is_defending = False
        mock_agent.alive = True
        mock_agent.device = torch.device("cpu")
        mock_agent.config = Mock()
        mock_agent.config.min_reproduction_resources = 8
        mock_agent.config.gathering_range = 30
        mock_agent.config.social_range = 30
        mock_agent.environment = Mock()
        mock_agent.environment.width = 20.0
        mock_agent.environment.height = 20.0
        mock_agent.environment.time = 10
        mock_agent.environment.resources = [Mock(), Mock()]  # 2 resources
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        mock_agent.environment.get_nearby_resources.return_value = []
        mock_agent.environment.get_nearby_agents.return_value = [
            Mock(),
            Mock(),
        ]  # Add nearby agents

        state = create_decision_state(mock_agent)

        self.assertEqual(state.shape, (8,))
        self.assertIsInstance(state, torch.Tensor)

    def test_create_selection_state_low_resources(self):
        """Test selection state creation when agent has low resources."""
        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.resource_level = 5.0  # Low resources
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 20.0  # Low health
        mock_agent.starvation_threshold = 10.0
        mock_agent.max_starvation = 20.0
        mock_agent.is_defending = False
        mock_agent.alive = True
        mock_agent.device = torch.device("cpu")
        mock_agent.config = Mock()
        mock_agent.config.min_reproduction_resources = 8
        mock_agent.config.gathering_range = 30
        mock_agent.config.social_range = 30
        mock_agent.environment = Mock()
        mock_agent.environment.width = 20.0
        mock_agent.environment.height = 20.0
        mock_agent.environment.time = 10
        mock_agent.environment.resources = [Mock(), Mock()]  # 2 resources
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        mock_agent.environment.get_nearby_resources.return_value = []
        mock_agent.environment.get_nearby_agents.return_value = [
            Mock(),
            Mock(),
        ]  # Add nearby agents

        state = create_decision_state(mock_agent)

        self.assertEqual(state.shape, (8,))
        self.assertIsInstance(state, torch.Tensor)

    def test_create_selection_state_with_many_resources(self):
        """Test selection state creation with many nearby resources."""
        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.resource_level = 50.0
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 80.0
        mock_agent.starvation_threshold = 10.0
        mock_agent.max_starvation = 20.0
        mock_agent.is_defending = False
        mock_agent.alive = True
        mock_agent.device = torch.device("cpu")
        mock_agent.config = Mock()
        mock_agent.config.min_reproduction_resources = 8
        mock_agent.config.gathering_range = 30
        mock_agent.config.social_range = 30
        mock_agent.environment = Mock()
        mock_agent.environment.width = 20.0
        mock_agent.environment.height = 20.0
        mock_agent.environment.time = 10
        mock_agent.environment.resources = [Mock(), Mock()]  # 2 resources
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        # Mock environment to return multiple resources
        mock_resources = []
        for i in range(5):
            mock_resource = Mock()
            mock_resource.position = (10.0 + i, 10.0)
            mock_resources.append(mock_resource)

        mock_agent.environment.get_nearby_resources.return_value = mock_resources
        mock_agent.environment.get_nearby_agents.return_value = [
            Mock(),
            Mock(),
        ]  # Add nearby agents

        state = create_decision_state(mock_agent)

        self.assertEqual(state.shape, (8,))
        self.assertIsInstance(state, torch.Tensor)

    def test_create_selection_state_with_many_agents(self):
        """Test selection state creation with many nearby agents."""
        mock_agent = Mock()
        mock_agent.position = (10.0, 10.0)
        mock_agent.resource_level = 50.0
        mock_agent.starting_health = 100.0
        mock_agent.current_health = 80.0
        mock_agent.starvation_threshold = 10.0
        mock_agent.max_starvation = 20.0
        mock_agent.is_defending = False
        mock_agent.alive = True
        mock_agent.device = torch.device("cpu")
        mock_agent.config = Mock()
        mock_agent.config.min_reproduction_resources = 8
        mock_agent.config.gathering_range = 30
        mock_agent.config.social_range = 30
        mock_agent.environment = Mock()
        mock_agent.environment.width = 20.0
        mock_agent.environment.height = 20.0
        mock_agent.environment.time = 10
        mock_agent.environment.resources = [Mock(), Mock()]  # 2 resources
        mock_agent.environment.agents = [Mock(), Mock(), Mock()]  # 3 agents

        # Mock environment to return multiple agents
        mock_agents = []
        for i in range(3):
            mock_other_agent = Mock()
            mock_other_agent.position = (10.0 + i, 10.0)
            mock_other_agent.resource_level = 30.0 + i * 10
            mock_agents.append(mock_other_agent)

        mock_agent.environment.get_nearby_resources.return_value = []
        mock_agent.environment.get_nearby_agents.return_value = mock_agents

        state = create_decision_state(mock_agent)

        self.assertEqual(state.shape, (8,))
        self.assertIsInstance(state, torch.Tensor)


if __name__ == "__main__":
    unittest.main()

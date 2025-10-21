"""
Unit tests for agent configuration classes.

This module tests the agent configuration system including:
- Individual component configs (Movement, Resource, Combat, etc.)
- AgentComponentConfig integration
- Preset configurations
- Configuration serialization/deserialization
"""

import unittest
from unittest.mock import MagicMock, patch
from dataclasses import FrozenInstanceError

# Mock torch to avoid dependency issues
import sys
sys.modules['torch'] = MagicMock()

from farm.core.agent.config import (
    AgentComponentConfig,
    MovementConfig,
    ResourceConfig,
    CombatConfig,
    PerceptionConfig,
    ReproductionConfig,
)
from farm.core.decision.config import DecisionConfig


class TestMovementConfig(unittest.TestCase):
    """Test cases for MovementConfig class."""

    def test_default_initialization(self):
        """Test that MovementConfig initializes with correct defaults."""
        config = MovementConfig()
        
        self.assertEqual(config.max_movement, 8.0)
        self.assertEqual(config.perception_radius, 5)

    def test_custom_initialization(self):
        """Test that MovementConfig can be initialized with custom values."""
        config = MovementConfig(
            max_movement=10.0,
            perception_radius=8
        )
        
        self.assertEqual(config.max_movement, 10.0)
        self.assertEqual(config.perception_radius, 8)

    def test_immutability(self):
        """Test that MovementConfig is immutable (frozen dataclass)."""
        config = MovementConfig()
        
        with self.assertRaises(FrozenInstanceError):
            config.max_movement = 15.0


class TestResourceConfig(unittest.TestCase):
    """Test cases for ResourceConfig class."""

    def test_default_initialization(self):
        """Test that ResourceConfig initializes with correct defaults."""
        config = ResourceConfig()
        
        self.assertEqual(config.base_consumption_rate, 1.0)
        self.assertEqual(config.starvation_threshold, 100)
        self.assertEqual(config.offspring_initial_resources, 10.0)
        self.assertEqual(config.offspring_cost, 5.0)

    def test_custom_initialization(self):
        """Test that ResourceConfig can be initialized with custom values."""
        config = ResourceConfig(
            base_consumption_rate=2.0,
            starvation_threshold=150,
            offspring_initial_resources=15.0,
            offspring_cost=8.0
        )
        
        self.assertEqual(config.base_consumption_rate, 2.0)
        self.assertEqual(config.starvation_threshold, 150)
        self.assertEqual(config.offspring_initial_resources, 15.0)
        self.assertEqual(config.offspring_cost, 8.0)

    def test_immutability(self):
        """Test that ResourceConfig is immutable (frozen dataclass)."""
        config = ResourceConfig()
        
        with self.assertRaises(FrozenInstanceError):
            config.base_consumption_rate = 3.0


class TestCombatConfig(unittest.TestCase):
    """Test cases for CombatConfig class."""

    def test_default_initialization(self):
        """Test that CombatConfig initializes with correct defaults."""
        config = CombatConfig()
        
        self.assertEqual(config.starting_health, 100.0)
        self.assertEqual(config.base_attack_strength, 10.0)
        self.assertEqual(config.base_defense_strength, 5.0)
        self.assertEqual(config.defense_damage_reduction, 0.5)
        self.assertEqual(config.defense_timer_duration, 3)

    def test_custom_initialization(self):
        """Test that CombatConfig can be initialized with custom values."""
        config = CombatConfig(
            starting_health=150.0,
            base_attack_strength=20.0,
            base_defense_strength=8.0,
            defense_damage_reduction=0.7,
            defense_timer_duration=5
        )
        
        self.assertEqual(config.starting_health, 150.0)
        self.assertEqual(config.base_attack_strength, 20.0)
        self.assertEqual(config.base_defense_strength, 8.0)
        self.assertEqual(config.defense_damage_reduction, 0.7)
        self.assertEqual(config.defense_timer_duration, 5)

    def test_immutability(self):
        """Test that CombatConfig is immutable (frozen dataclass)."""
        config = CombatConfig()
        
        with self.assertRaises(FrozenInstanceError):
            config.starting_health = 200.0


class TestPerceptionConfig(unittest.TestCase):
    """Test cases for PerceptionConfig class."""

    def test_default_initialization(self):
        """Test that PerceptionConfig initializes with correct defaults."""
        config = PerceptionConfig()
        
        self.assertEqual(config.perception_radius, 5)
        self.assertEqual(config.position_discretization_method, "floor")

    def test_custom_initialization(self):
        """Test that PerceptionConfig can be initialized with custom values."""
        config = PerceptionConfig(
            perception_radius=8,
            position_discretization_method="round"
        )
        
        self.assertEqual(config.perception_radius, 8)
        self.assertEqual(config.position_discretization_method, "round")

    def test_immutability(self):
        """Test that PerceptionConfig is immutable (frozen dataclass)."""
        config = PerceptionConfig()
        
        with self.assertRaises(FrozenInstanceError):
            config.perception_radius = 10


class TestReproductionConfig(unittest.TestCase):
    """Test cases for ReproductionConfig class."""

    def test_default_initialization(self):
        """Test that ReproductionConfig initializes with correct defaults."""
        config = ReproductionConfig()
        
        self.assertEqual(config.offspring_initial_resources, 10.0)
        self.assertEqual(config.offspring_cost, 5.0)

    def test_custom_initialization(self):
        """Test that ReproductionConfig can be initialized with custom values."""
        config = ReproductionConfig(
            offspring_initial_resources=15.0,
            offspring_cost=8.0
        )
        
        self.assertEqual(config.offspring_initial_resources, 15.0)
        self.assertEqual(config.offspring_cost, 8.0)

    def test_immutability(self):
        """Test that ReproductionConfig is immutable (frozen dataclass)."""
        config = ReproductionConfig()
        
        with self.assertRaises(FrozenInstanceError):
            config.offspring_initial_resources = 20.0


class TestAgentComponentConfig(unittest.TestCase):
    """Test cases for AgentComponentConfig class."""

    def test_default_initialization(self):
        """Test that AgentComponentConfig initializes with correct defaults."""
        config = AgentComponentConfig()
        
        # Check that all components are present
        self.assertIsInstance(config.movement, MovementConfig)
        self.assertIsInstance(config.resource, ResourceConfig)
        self.assertIsInstance(config.combat, CombatConfig)
        self.assertIsInstance(config.perception, PerceptionConfig)
        self.assertIsInstance(config.reproduction, ReproductionConfig)
        self.assertIsInstance(config.decision, DecisionConfig)

    def test_custom_initialization(self):
        """Test that AgentComponentConfig can be initialized with custom components."""
        custom_movement = MovementConfig(max_movement=10.0)
        custom_resource = ResourceConfig(base_consumption_rate=2.0)
        
        config = AgentComponentConfig(
            movement=custom_movement,
            resource=custom_resource
        )
        
        self.assertEqual(config.movement.max_movement, 10.0)
        self.assertEqual(config.resource.base_consumption_rate, 2.0)
        # Other components should use defaults
        self.assertEqual(config.combat.starting_health, 100.0)

    def test_aggressive_preset(self):
        """Test that aggressive preset creates correct configuration."""
        config = AgentComponentConfig.aggressive()
        
        # Check combat stats are higher
        self.assertGreater(config.combat.starting_health, 100.0)
        self.assertGreater(config.combat.base_attack_strength, 10.0)
        self.assertGreater(config.combat.base_defense_strength, 5.0)
        
        # Check resource consumption is higher
        self.assertGreater(config.resource.base_consumption_rate, 1.0)
        self.assertGreater(config.resource.offspring_cost, 5.0)

    def test_defensive_preset(self):
        """Test that defensive preset creates correct configuration."""
        config = AgentComponentConfig.defensive()
        
        # Check health is higher
        self.assertGreater(config.combat.starting_health, 100.0)
        
        # Check defense is stronger
        self.assertGreater(config.combat.base_defense_strength, 5.0)
        self.assertGreater(config.combat.defense_damage_reduction, 0.5)
        
        # Check attack is weaker
        self.assertLess(config.combat.base_attack_strength, 10.0)

    def test_efficient_preset(self):
        """Test that efficient preset creates correct configuration."""
        config = AgentComponentConfig.efficient()
        
        # Check resource consumption is lower
        self.assertLess(config.resource.base_consumption_rate, 1.0)
        self.assertLess(config.resource.offspring_cost, 5.0)
        
        # Check health is lower (trade-off for efficiency)
        self.assertLess(config.combat.starting_health, 100.0)

    @patch('farm.core.agent.config.DecisionConfig')
    def test_from_simulation_config_with_agent_behavior(self, mock_decision_config):
        """Test from_simulation_config with agent_behavior config."""
        # Mock simulation config with agent_behavior
        mock_sim_config = MagicMock()
        mock_agent_behavior = MagicMock()
        mock_agent_behavior.base_consumption_rate = 2.0
        mock_agent_behavior.starvation_threshold = 150
        mock_agent_behavior.offspring_cost = 8.0
        mock_agent_behavior.offspring_initial_resources = 15.0
        mock_agent_behavior.starting_health = 200.0
        mock_agent_behavior.base_attack_strength = 20.0
        mock_agent_behavior.base_defense_strength = 10.0
        
        mock_sim_config.agent_behavior = mock_agent_behavior
        
        config = AgentComponentConfig.from_simulation_config(mock_sim_config)
        
        self.assertEqual(config.resource.base_consumption_rate, 2.0)
        self.assertEqual(config.resource.starvation_threshold, 150)
        self.assertEqual(config.resource.offspring_cost, 8.0)
        self.assertEqual(config.resource.offspring_initial_resources, 15.0)
        self.assertEqual(config.combat.starting_health, 200.0)
        self.assertEqual(config.combat.base_attack_strength, 20.0)
        self.assertEqual(config.combat.base_defense_strength, 10.0)

    @patch('farm.core.agent.config.DecisionConfig')
    def test_from_simulation_config_without_agent_behavior(self, mock_decision_config):
        """Test from_simulation_config without agent_behavior config."""
        # Mock simulation config without agent_behavior
        mock_sim_config = MagicMock()
        mock_sim_config.agent_behavior = None
        
        config = AgentComponentConfig.from_simulation_config(mock_sim_config)
        
        # Should use default values
        self.assertEqual(config.resource.base_consumption_rate, 1.0)
        self.assertEqual(config.resource.starvation_threshold, 10)
        self.assertEqual(config.resource.offspring_cost, 5.0)
        self.assertEqual(config.resource.offspring_initial_resources, 10.0)
        self.assertEqual(config.combat.starting_health, 100.0)
        self.assertEqual(config.combat.base_attack_strength, 10.0)
        self.assertEqual(config.combat.base_defense_strength, 5.0)

    def test_immutability(self):
        """Test that AgentComponentConfig is mutable (not frozen)."""
        config = AgentComponentConfig()
        
        # Should be able to modify (not frozen)
        config.movement = MovementConfig(max_movement=15.0)
        self.assertEqual(config.movement.max_movement, 15.0)


class TestAgentConfigSerialization(unittest.TestCase):
    """Test cases for agent configuration serialization."""

    def test_agent_config_to_dict(self):
        """Test that AgentComponentConfig can be converted to dict."""
        config = AgentComponentConfig()
        config_dict = config.__dict__
        
        # Check that all components are present
        self.assertIn('movement', config_dict)
        self.assertIn('resource', config_dict)
        self.assertIn('combat', config_dict)
        self.assertIn('perception', config_dict)
        self.assertIn('reproduction', config_dict)
        self.assertIn('decision', config_dict)
        
        # Check that components are the right type
        self.assertIsInstance(config_dict['movement'], MovementConfig)
        self.assertIsInstance(config_dict['resource'], ResourceConfig)
        self.assertIsInstance(config_dict['combat'], CombatConfig)
        self.assertIsInstance(config_dict['perception'], PerceptionConfig)
        self.assertIsInstance(config_dict['reproduction'], ReproductionConfig)

    def test_agent_config_from_dict(self):
        """Test that AgentComponentConfig can be created from dict."""
        config_dict = {
            'movement': MovementConfig(max_movement=10.0),
            'resource': ResourceConfig(base_consumption_rate=2.0),
            'combat': CombatConfig(starting_health=150.0),
            'perception': PerceptionConfig(perception_radius=8),
            'reproduction': ReproductionConfig(offspring_cost=8.0),
            'decision': DecisionConfig()
        }
        
        config = AgentComponentConfig(**config_dict)
        
        self.assertEqual(config.movement.max_movement, 10.0)
        self.assertEqual(config.resource.base_consumption_rate, 2.0)
        self.assertEqual(config.combat.starting_health, 150.0)
        self.assertEqual(config.perception.perception_radius, 8)
        self.assertEqual(config.reproduction.offspring_cost, 8.0)


if __name__ == '__main__':
    unittest.main()
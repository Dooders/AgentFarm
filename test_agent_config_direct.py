#!/usr/bin/env python3
"""
Direct tests for agent configuration classes.

This module tests the agent configuration classes directly without importing
the full agent module to avoid dependency issues.
"""

import unittest
import sys
import os
from dataclasses import dataclass, field
from typing import Optional

# Add the farm directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'farm'))

# Mock heavy dependencies
from unittest.mock import MagicMock
sys.modules['torch'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['structlog'] = MagicMock()
sys.modules['farm.utils.logging'] = MagicMock()
sys.modules['farm.core.action'] = MagicMock()
sys.modules['farm.core.agent.behaviors'] = MagicMock()
sys.modules['farm.core.agent.behaviors.base'] = MagicMock()
sys.modules['farm.core.agent.components'] = MagicMock()
sys.modules['farm.core.agent.components.perception'] = MagicMock()
sys.modules['farm.core.agent.components.movement'] = MagicMock()
sys.modules['farm.core.agent.components.resource'] = MagicMock()
sys.modules['farm.core.agent.components.combat'] = MagicMock()
sys.modules['farm.core.agent.components.reproduction'] = MagicMock()

# Mock the DecisionConfig to avoid torch dependency
sys.modules['farm.core.decision.config'] = MagicMock()
sys.modules['farm.core.decision'] = MagicMock()

# Now we can import the config classes directly
from farm.core.agent.config.component_configs import (
    MovementConfig,
    ResourceConfig,
    CombatConfig,
    PerceptionConfig,
    ReproductionConfig,
    AgentComponentConfig,
)


class TestAgentConfigDirect(unittest.TestCase):
    """Direct test cases for agent configuration classes."""

    def test_movement_config_defaults(self):
        """Test MovementConfig default values."""
        config = MovementConfig()
        self.assertEqual(config.max_movement, 8.0)
        self.assertEqual(config.perception_radius, 5)

    def test_movement_config_custom(self):
        """Test MovementConfig with custom values."""
        config = MovementConfig(max_movement=10.0, perception_radius=8)
        self.assertEqual(config.max_movement, 10.0)
        self.assertEqual(config.perception_radius, 8)

    def test_movement_config_immutability(self):
        """Test that MovementConfig is immutable."""
        config = MovementConfig()
        with self.assertRaises(Exception):  # FrozenInstanceError
            config.max_movement = 15.0

    def test_resource_config_defaults(self):
        """Test ResourceConfig default values."""
        config = ResourceConfig()
        self.assertEqual(config.base_consumption_rate, 1.0)
        self.assertEqual(config.starvation_threshold, 100)
        self.assertEqual(config.offspring_initial_resources, 10.0)
        self.assertEqual(config.offspring_cost, 5.0)

    def test_resource_config_custom(self):
        """Test ResourceConfig with custom values."""
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

    def test_combat_config_defaults(self):
        """Test CombatConfig default values."""
        config = CombatConfig()
        self.assertEqual(config.starting_health, 100.0)
        self.assertEqual(config.base_attack_strength, 10.0)
        self.assertEqual(config.base_defense_strength, 5.0)
        self.assertEqual(config.defense_damage_reduction, 0.5)
        self.assertEqual(config.defense_timer_duration, 3)

    def test_combat_config_custom(self):
        """Test CombatConfig with custom values."""
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

    def test_perception_config_defaults(self):
        """Test PerceptionConfig default values."""
        config = PerceptionConfig()
        self.assertEqual(config.perception_radius, 5)
        self.assertEqual(config.position_discretization_method, "floor")

    def test_perception_config_custom(self):
        """Test PerceptionConfig with custom values."""
        config = PerceptionConfig(
            perception_radius=8,
            position_discretization_method="round"
        )
        self.assertEqual(config.perception_radius, 8)
        self.assertEqual(config.position_discretization_method, "round")

    def test_reproduction_config_defaults(self):
        """Test ReproductionConfig default values."""
        config = ReproductionConfig()
        self.assertEqual(config.offspring_initial_resources, 10.0)
        self.assertEqual(config.offspring_cost, 5.0)

    def test_reproduction_config_custom(self):
        """Test ReproductionConfig with custom values."""
        config = ReproductionConfig(
            offspring_initial_resources=15.0,
            offspring_cost=8.0
        )
        self.assertEqual(config.offspring_initial_resources, 15.0)
        self.assertEqual(config.offspring_cost, 8.0)

    def test_agent_component_config_defaults(self):
        """Test AgentComponentConfig default values."""
        config = AgentComponentConfig()
        
        # Check that all components are present
        self.assertIsInstance(config.movement, MovementConfig)
        self.assertIsInstance(config.resource, ResourceConfig)
        self.assertIsInstance(config.combat, CombatConfig)
        self.assertIsInstance(config.perception, PerceptionConfig)
        self.assertIsInstance(config.reproduction, ReproductionConfig)

    def test_agent_component_config_custom(self):
        """Test AgentComponentConfig with custom components."""
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

    def test_agent_component_config_aggressive_preset(self):
        """Test AgentComponentConfig aggressive preset."""
        config = AgentComponentConfig.aggressive()
        
        # Check combat stats are higher
        self.assertGreater(config.combat.starting_health, 100.0)
        self.assertGreater(config.combat.base_attack_strength, 10.0)
        self.assertGreater(config.combat.base_defense_strength, 5.0)
        
        # Check resource consumption is higher
        self.assertGreater(config.resource.base_consumption_rate, 1.0)
        self.assertGreater(config.resource.offspring_cost, 5.0)

    def test_agent_component_config_defensive_preset(self):
        """Test AgentComponentConfig defensive preset."""
        config = AgentComponentConfig.defensive()
        
        # Check health is higher
        self.assertGreater(config.combat.starting_health, 100.0)
        
        # Check defense is stronger
        self.assertGreater(config.combat.base_defense_strength, 5.0)
        self.assertGreater(config.combat.defense_damage_reduction, 0.5)
        
        # Check attack is weaker
        self.assertLess(config.combat.base_attack_strength, 10.0)

    def test_agent_component_config_efficient_preset(self):
        """Test AgentComponentConfig efficient preset."""
        config = AgentComponentConfig.efficient()
        
        # Check resource consumption is lower
        self.assertLess(config.resource.base_consumption_rate, 1.0)
        self.assertLess(config.resource.offspring_cost, 5.0)
        
        # Check health is lower (trade-off for efficiency)
        self.assertLess(config.combat.starting_health, 100.0)

    def test_agent_component_config_from_simulation_config(self):
        """Test AgentComponentConfig.from_simulation_config method."""
        # Mock simulation config
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
        
        # Check that values were extracted correctly
        self.assertEqual(config.resource.base_consumption_rate, 2.0)
        self.assertEqual(config.resource.starvation_threshold, 150)
        self.assertEqual(config.resource.offspring_cost, 8.0)
        self.assertEqual(config.resource.offspring_initial_resources, 15.0)
        self.assertEqual(config.combat.starting_health, 200.0)
        self.assertEqual(config.combat.base_attack_strength, 20.0)
        self.assertEqual(config.combat.base_defense_strength, 10.0)

    def test_agent_component_config_from_simulation_config_no_agent_behavior(self):
        """Test AgentComponentConfig.from_simulation_config without agent_behavior."""
        # Mock simulation config without agent_behavior
        mock_sim_config = MagicMock()
        mock_sim_config.agent_behavior = None
        
        config = AgentComponentConfig.from_simulation_config(mock_sim_config)
        
        # Should use default values
        self.assertEqual(config.resource.base_consumption_rate, 1.0)
        self.assertEqual(config.resource.starvation_threshold, 100)
        self.assertEqual(config.resource.offspring_cost, 5.0)
        self.assertEqual(config.resource.offspring_initial_resources, 10.0)
        self.assertEqual(config.combat.starting_health, 100.0)
        self.assertEqual(config.combat.base_attack_strength, 10.0)
        self.assertEqual(config.combat.base_defense_strength, 5.0)

    def test_agent_component_config_serialization(self):
        """Test AgentComponentConfig serialization."""
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

    def test_agent_component_config_deserialization(self):
        """Test AgentComponentConfig deserialization."""
        config_dict = {
            'movement': MovementConfig(max_movement=10.0),
            'resource': ResourceConfig(base_consumption_rate=2.0),
            'combat': CombatConfig(starting_health=150.0),
            'perception': PerceptionConfig(perception_radius=8),
            'reproduction': ReproductionConfig(offspring_cost=8.0),
            'decision': MagicMock()  # Mock DecisionConfig
        }
        
        config = AgentComponentConfig(**config_dict)
        
        self.assertEqual(config.movement.max_movement, 10.0)
        self.assertEqual(config.resource.base_consumption_rate, 2.0)
        self.assertEqual(config.combat.starting_health, 150.0)
        self.assertEqual(config.perception.perception_radius, 8)
        self.assertEqual(config.reproduction.offspring_cost, 8.0)


if __name__ == '__main__':
    unittest.main()
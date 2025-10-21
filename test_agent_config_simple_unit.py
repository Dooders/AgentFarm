#!/usr/bin/env python3
"""
Simple unit tests for agent configuration integration.

This module tests the agent configuration system without heavy dependencies.
"""

import unittest
import sys
import os

# Add the farm directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'farm'))

# Mock heavy dependencies
from unittest.mock import MagicMock, patch
import sys
sys.modules['torch'] = MagicMock()
sys.modules['structlog'] = MagicMock()
sys.modules['farm.utils.logging'] = MagicMock()
sys.modules['farm.core.action'] = MagicMock()
sys.modules['farm.core.agent.behaviors'] = MagicMock()
sys.modules['farm.core.agent.behaviors.base'] = MagicMock()

# Mock the DecisionConfig to avoid torch dependency
with patch.dict('sys.modules', {'farm.core.decision.config': MagicMock()}):
    from farm.core.agent.config import (
        AgentComponentConfig,
        MovementConfig,
        ResourceConfig,
        CombatConfig,
        PerceptionConfig,
        ReproductionConfig,
    )


class TestAgentConfigSimple(unittest.TestCase):
    """Simple test cases for agent configuration."""

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

    def test_resource_config_defaults(self):
        """Test ResourceConfig default values."""
        config = ResourceConfig()
        self.assertEqual(config.base_consumption_rate, 1.0)
        self.assertEqual(config.starvation_threshold, 100)
        self.assertEqual(config.offspring_initial_resources, 10.0)
        self.assertEqual(config.offspring_cost, 5.0)

    def test_combat_config_defaults(self):
        """Test CombatConfig default values."""
        config = CombatConfig()
        self.assertEqual(config.starting_health, 100.0)
        self.assertEqual(config.base_attack_strength, 10.0)
        self.assertEqual(config.base_defense_strength, 5.0)
        self.assertEqual(config.defense_damage_reduction, 0.5)
        self.assertEqual(config.defense_timer_duration, 3)

    def test_perception_config_defaults(self):
        """Test PerceptionConfig default values."""
        config = PerceptionConfig()
        self.assertEqual(config.perception_radius, 5)
        self.assertEqual(config.position_discretization_method, "floor")

    def test_reproduction_config_defaults(self):
        """Test ReproductionConfig default values."""
        config = ReproductionConfig()
        self.assertEqual(config.offspring_initial_resources, 10.0)
        self.assertEqual(config.offspring_cost, 5.0)

    def test_agent_component_config_defaults(self):
        """Test AgentComponentConfig default values."""
        config = AgentComponentConfig()
        
        # Check that all components are present
        self.assertIsInstance(config.movement, MovementConfig)
        self.assertIsInstance(config.resource, ResourceConfig)
        self.assertIsInstance(config.combat, CombatConfig)
        self.assertIsInstance(config.perception, PerceptionConfig)
        self.assertIsInstance(config.reproduction, ReproductionConfig)

    def test_agent_component_config_aggressive_preset(self):
        """Test AgentComponentConfig aggressive preset."""
        config = AgentComponentConfig.aggressive()
        
        # Check that aggressive preset has higher values
        self.assertGreater(config.combat.starting_health, 100.0)
        self.assertGreater(config.combat.base_attack_strength, 10.0)
        self.assertGreater(config.combat.base_defense_strength, 5.0)
        self.assertGreater(config.resource.base_consumption_rate, 1.0)
        self.assertGreater(config.resource.offspring_cost, 5.0)

    def test_agent_component_config_defensive_preset(self):
        """Test AgentComponentConfig defensive preset."""
        config = AgentComponentConfig.defensive()
        
        # Check that defensive preset has higher health and defense
        self.assertGreater(config.combat.starting_health, 100.0)
        self.assertGreater(config.combat.base_defense_strength, 5.0)
        self.assertGreater(config.combat.defense_damage_reduction, 0.5)
        self.assertLess(config.combat.base_attack_strength, 10.0)

    def test_agent_component_config_efficient_preset(self):
        """Test AgentComponentConfig efficient preset."""
        config = AgentComponentConfig.efficient()
        
        # Check that efficient preset has lower consumption
        self.assertLess(config.resource.base_consumption_rate, 1.0)
        self.assertLess(config.resource.offspring_cost, 5.0)
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
        self.assertEqual(config.resource.starvation_threshold, 10)
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
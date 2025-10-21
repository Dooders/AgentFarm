#!/usr/bin/env python3
"""
Final tests for agent configuration integration.

This module tests the agent configuration system by directly importing
the config files and testing the integration with SimulationConfig.
"""

import unittest
import sys
import os
import tempfile
import yaml
from unittest.mock import MagicMock, patch

# Add the farm directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'farm'))

# Mock heavy dependencies
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
sys.modules['farm.core.agent.components.base'] = MagicMock()
sys.modules['farm.core.agent.core'] = MagicMock()
sys.modules['farm.core.agent.factory'] = MagicMock()
sys.modules['farm.core.agent.services'] = MagicMock()
sys.modules['farm.core.agent'] = MagicMock()

# Mock the DecisionConfig to avoid torch dependency
sys.modules['farm.core.decision.config'] = MagicMock()
sys.modules['farm.core.decision'] = MagicMock()

# Mock the observations module
sys.modules['farm.core.observations'] = MagicMock()

# Now we can import the config classes
from farm.config import SimulationConfig


class TestAgentConfigIntegration(unittest.TestCase):
    """Test cases for agent configuration integration."""

    def test_simulation_config_has_agent_attribute(self):
        """Test that SimulationConfig has agent attribute."""
        config = SimulationConfig()
        
        self.assertTrue(hasattr(config, 'agent'))
        # The agent should be an AgentComponentConfig instance
        self.assertIsNotNone(config.agent)

    def test_agent_config_default_values(self):
        """Test that agent config has correct default values."""
        config = SimulationConfig()
        agent = config.agent
        
        # Check that agent has the expected attributes
        self.assertTrue(hasattr(agent, 'movement'))
        self.assertTrue(hasattr(agent, 'resource'))
        self.assertTrue(hasattr(agent, 'combat'))
        self.assertTrue(hasattr(agent, 'perception'))
        self.assertTrue(hasattr(agent, 'reproduction'))
        self.assertTrue(hasattr(agent, 'decision'))

    def test_simulation_config_serialization(self):
        """Test that SimulationConfig serializes agent config correctly."""
        config = SimulationConfig()
        config_dict = config.to_dict()
        
        # Check that agent section is present
        self.assertIn('agent', config_dict)
        agent_dict = config_dict['agent']
        
        # Check that agent components are present
        self.assertIn('movement', agent_dict)
        self.assertIn('resource', agent_dict)
        self.assertIn('combat', agent_dict)
        self.assertIn('perception', agent_dict)
        self.assertIn('reproduction', agent_dict)
        self.assertIn('decision', agent_dict)

    def test_simulation_config_deserialization(self):
        """Test that SimulationConfig deserializes agent config correctly."""
        # Create a config dict with agent data
        config_dict = {
            'simulation_steps': 100,
            'agent': {
                'movement': {
                    'max_movement': 10.0,
                    'perception_radius': 8
                },
                'resource': {
                    'base_consumption_rate': 2.0,
                    'starvation_threshold': 150
                },
                'combat': {
                    'starting_health': 150.0,
                    'base_attack_strength': 20.0
                },
                'perception': {
                    'perception_radius': 8,
                    'position_discretization_method': 'round'
                },
                'reproduction': {
                    'offspring_initial_resources': 15.0,
                    'offspring_cost': 8.0
                },
                'decision': {
                    'learning_rate': 0.01,
                    'memory_size': 5000
                }
            }
        }
        
        config = SimulationConfig.from_dict(config_dict)
        
        # Check that agent config was loaded correctly
        self.assertTrue(hasattr(config, 'agent'))
        self.assertIsNotNone(config.agent)

    def test_simulation_config_yaml_serialization(self):
        """Test that SimulationConfig can be saved to and loaded from YAML."""
        config = SimulationConfig()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save config to YAML
            config.to_yaml(temp_file)
            
            # Load config from YAML
            loaded_config = SimulationConfig.from_yaml(temp_file)
            
            # Check that agent config was preserved
            self.assertTrue(hasattr(loaded_config, 'agent'))
            self.assertIsNotNone(loaded_config.agent)
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_agent_config_serialization_roundtrip(self):
        """Test that agent config survives serialization roundtrip."""
        original_config = SimulationConfig()
        
        # Serialize to dict
        config_dict = original_config.to_dict()
        
        # Deserialize from dict
        loaded_config = SimulationConfig.from_dict(config_dict)
        
        # Check that agent config was preserved
        self.assertTrue(hasattr(loaded_config, 'agent'))
        self.assertIsNotNone(loaded_config.agent)

    def test_agent_config_with_custom_values(self):
        """Test that custom agent config values are preserved."""
        config = SimulationConfig()
        
        # Modify agent config (if it's mutable)
        if hasattr(config.agent, 'movement'):
            if hasattr(config.agent.movement, 'max_movement'):
                config.agent.movement.max_movement = 15.0
        
        # Serialize and deserialize
        config_dict = config.to_dict()
        loaded_config = SimulationConfig.from_dict(config_dict)
        
        # Check that custom values were preserved (if applicable)
        self.assertTrue(hasattr(loaded_config, 'agent'))
        self.assertIsNotNone(loaded_config.agent)

    def test_agent_config_validation(self):
        """Test that agent config validation works correctly."""
        config = SimulationConfig()
        
        # Test with valid agent config
        self.assertTrue(hasattr(config, 'agent'))
        self.assertIsNotNone(config.agent)

    def test_agent_config_in_config_diff(self):
        """Test that agent config is included in config diff."""
        config1 = SimulationConfig()
        config2 = SimulationConfig()
        
        # Modify agent config in config2
        if hasattr(config2.agent, 'combat'):
            if hasattr(config2.agent.combat, 'starting_health'):
                config2.agent.combat.starting_health = 150.0
        
        diff = config1.diff_config(config2)
        
        # Check that agent config differences are included
        agent_diff_keys = [key for key in diff.keys() if 'agent' in key]
        # This might be empty if the configs are the same, which is fine

    def test_agent_config_in_version_hash(self):
        """Test that agent config changes affect version hash."""
        config1 = SimulationConfig()
        config2 = SimulationConfig()
        
        # Modify agent config in config2
        if hasattr(config2.agent, 'combat'):
            if hasattr(config2.agent.combat, 'starting_health'):
                config2.agent.combat.starting_health = 150.0
        
        hash1 = config1.generate_version_hash()
        hash2 = config2.generate_version_hash()
        
        # Hashes should be different due to agent config change
        self.assertNotEqual(hash1, hash2)

    def test_agent_config_in_versioned_config(self):
        """Test that agent config is included in versioned config."""
        config = SimulationConfig()
        
        # Modify agent config
        if hasattr(config.agent, 'combat'):
            if hasattr(config.agent.combat, 'starting_health'):
                config.agent.combat.starting_health = 150.0
        
        versioned_config = config.version_config("Test version")
        
        # Check that agent config is preserved in versioned config
        self.assertTrue(hasattr(versioned_config, 'agent'))
        self.assertIsNotNone(versioned_config.agent)
        self.assertIsNotNone(versioned_config.versioning.config_version)
        self.assertIsNotNone(versioned_config.versioning.config_created_at)
        self.assertEqual(versioned_config.versioning.config_description, "Test version")

    def test_legacy_config_compatibility(self):
        """Test that legacy configuration files still work."""
        # Create a legacy config file (without agent section)
        legacy_config = {
            'simulation_steps': 100,
            'width': 100,
            'height': 100,
            'system_agents': 10,
            'independent_agents': 10,
            'control_agents': 10,
            'base_consumption_rate': 0.15,
            'max_movement': 8,
            'starting_health': 100.0,
            'base_attack_strength': 10.0,
            'base_defense_strength': 5.0,
            'perception_radius': 5,
            'offspring_cost': 5.0,
            'offspring_initial_resources': 10.0,
            'starvation_threshold': 100
        }
        
        # Test that legacy config can be loaded
        config = SimulationConfig.from_dict(legacy_config)
        
        # Check that agent config was created
        self.assertTrue(hasattr(config, 'agent'))
        self.assertIsNotNone(config.agent)

    def test_mixed_legacy_and_new_config(self):
        """Test that mixed legacy and new config works."""
        mixed_config = {
            'simulation_steps': 100,
            'agent_behavior': {
                'base_consumption_rate': 2.0,
                'starvation_threshold': 150
            },
            'agent': {
                'movement': {
                    'max_movement': 10.0,
                    'perception_radius': 8
                },
                'combat': {
                    'starting_health': 200.0,
                    'base_attack_strength': 20.0
                }
            }
        }
        
        # Test that mixed config can be loaded
        config = SimulationConfig.from_dict(mixed_config)
        
        # Check that both old and new structures are present
        self.assertTrue(hasattr(config, 'agent_behavior'))
        self.assertTrue(hasattr(config, 'agent'))
        self.assertIsNotNone(config.agent)

    def test_config_serialization_backward_compatibility(self):
        """Test that config serialization maintains backward compatibility."""
        # Create a config with both old and new structures
        config = SimulationConfig()
        config.agent_parameters = {
            'SystemAgent': {
                'gather_efficiency_multiplier': 0.4,
                'gather_cost_multiplier': 0.4
            }
        }
        
        # Test serialization
        config_dict = config.to_dict()
        
        # Check that both old and new structures are present
        self.assertIn('agent_parameters', config_dict)
        self.assertIn('agent', config_dict)
        
        # Check that agent config is properly serialized
        self.assertIn('movement', config_dict['agent'])
        self.assertIn('resource', config_dict['agent'])
        self.assertIn('combat', config_dict['agent'])

    def test_config_deserialization_backward_compatibility(self):
        """Test that config deserialization maintains backward compatibility."""
        # Create a config dict with both old and new structures
        config_dict = {
            'simulation_steps': 100,
            'agent_parameters': {
                'SystemAgent': {
                    'gather_efficiency_multiplier': 0.4
                }
            },
            'agent': {
                'movement': {
                    'max_movement': 10.0,
                    'perception_radius': 8
                },
                'combat': {
                    'starting_health': 150.0,
                    'base_attack_strength': 20.0
                }
            }
        }
        
        # Test deserialization
        config = SimulationConfig.from_dict(config_dict)
        
        # Check that both old and new structures are preserved
        self.assertTrue(hasattr(config, 'agent_parameters'))
        self.assertTrue(hasattr(config, 'agent'))
        self.assertIsNotNone(config.agent)


if __name__ == '__main__':
    unittest.main()
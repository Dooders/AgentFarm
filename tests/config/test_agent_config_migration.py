"""
Migration tests for agent configuration integration.

This module tests the migration from the old agent configuration system
to the new centralized configuration system, ensuring backward compatibility.
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import yaml

# Mock torch to avoid dependency issues
import sys
sys.modules['torch'] = MagicMock()

from farm.config import SimulationConfig
from farm.core.agent.config import AgentComponentConfig


class TestAgentConfigMigration(unittest.TestCase):
    """Test cases for agent configuration migration and backward compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

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
        
        # Check that agent config was created with default values
        self.assertIsInstance(config.agent, AgentComponentConfig)
        self.assertEqual(config.agent.movement.max_movement, 8.0)
        self.assertEqual(config.agent.resource.base_consumption_rate, 1.0)  # Default value
        self.assertEqual(config.agent.combat.starting_health, 100.0)
        self.assertEqual(config.agent.combat.base_attack_strength, 10.0)
        self.assertEqual(config.agent.combat.base_defense_strength, 5.0)
        self.assertEqual(config.agent.perception.perception_radius, 5)
        self.assertEqual(config.agent.reproduction.offspring_cost, 5.0)
        self.assertEqual(config.agent.reproduction.offspring_initial_resources, 10.0)

    def test_legacy_config_with_agent_behavior(self):
        """Test that legacy config with agent_behavior section works."""
        legacy_config = {
            'simulation_steps': 100,
            'agent_behavior': {
                'base_consumption_rate': 2.0,
                'starvation_threshold': 150,
                'offspring_cost': 8.0,
                'offspring_initial_resources': 15.0,
                'starting_health': 200.0,
                'base_attack_strength': 20.0,
                'base_defense_strength': 10.0
            }
        }
        
        # Test that legacy config with agent_behavior can be loaded
        config = SimulationConfig.from_dict(legacy_config)
        
        # Check that agent config was created from agent_behavior
        self.assertIsInstance(config.agent, AgentComponentConfig)
        self.assertEqual(config.agent.resource.base_consumption_rate, 2.0)
        self.assertEqual(config.agent.resource.starvation_threshold, 150)
        self.assertEqual(config.agent.reproduction.offspring_cost, 8.0)
        self.assertEqual(config.agent.reproduction.offspring_initial_resources, 15.0)
        self.assertEqual(config.agent.combat.starting_health, 200.0)
        self.assertEqual(config.agent.combat.base_attack_strength, 20.0)
        self.assertEqual(config.agent.combat.base_defense_strength, 10.0)

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
        
        # Check that new agent config takes precedence
        self.assertEqual(config.agent.movement.max_movement, 10.0)
        self.assertEqual(config.agent.movement.perception_radius, 8)
        self.assertEqual(config.agent.combat.starting_health, 200.0)
        self.assertEqual(config.agent.combat.base_attack_strength, 20.0)
        
        # Check that agent_behavior values are used for resource config
        self.assertEqual(config.agent.resource.base_consumption_rate, 2.0)
        self.assertEqual(config.agent.resource.starvation_threshold, 150)

    def test_agent_parameters_legacy_compatibility(self):
        """Test that legacy agent_parameters section still works."""
        legacy_config = {
            'simulation_steps': 100,
            'agent_parameters': {
                'SystemAgent': {
                    'gather_efficiency_multiplier': 0.4,
                    'gather_cost_multiplier': 0.4,
                    'min_resource_threshold': 0.2,
                    'share_weight': 0.3,
                    'attack_weight': 0.05
                },
                'IndependentAgent': {
                    'gather_efficiency_multiplier': 0.7,
                    'gather_cost_multiplier': 0.2,
                    'min_resource_threshold': 0.05,
                    'share_weight': 0.05,
                    'attack_weight': 0.25
                }
            }
        }
        
        # Test that legacy agent_parameters can be loaded
        config = SimulationConfig.from_dict(legacy_config)
        
        # Check that agent_parameters are preserved
        self.assertIn('agent_parameters', config.__dict__)
        self.assertIn('SystemAgent', config.agent_parameters)
        self.assertIn('IndependentAgent', config.agent_parameters)
        
        # Check that agent config was created with defaults
        self.assertIsInstance(config.agent, AgentComponentConfig)

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
        config.agent.combat.starting_health = 150.0
        
        # Test serialization
        config_dict = config.to_dict()
        
        # Check that both old and new structures are present
        self.assertIn('agent_parameters', config_dict)
        self.assertIn('agent', config_dict)
        
        # Check that agent config is properly serialized
        self.assertIn('movement', config_dict['agent'])
        self.assertIn('resource', config_dict['agent'])
        self.assertIn('combat', config_dict['agent'])
        self.assertEqual(config_dict['agent']['combat']['starting_health'], 150.0)

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
        self.assertIn('agent_parameters', config.__dict__)
        self.assertIsInstance(config.agent, AgentComponentConfig)
        
        # Check that values are correct
        self.assertEqual(config.agent.movement.max_movement, 10.0)
        self.assertEqual(config.agent.movement.perception_radius, 8)
        self.assertEqual(config.agent.combat.starting_health, 150.0)
        self.assertEqual(config.agent.combat.base_attack_strength, 20.0)

    def test_yaml_loading_backward_compatibility(self):
        """Test that YAML loading maintains backward compatibility."""
        # Create a legacy YAML file
        legacy_yaml = """
simulation_steps: 100
width: 100
height: 100
system_agents: 10
independent_agents: 10
control_agents: 10
base_consumption_rate: 0.15
max_movement: 8
starting_health: 100.0
base_attack_strength: 10.0
base_defense_strength: 5.0
perception_radius: 5
offspring_cost: 5.0
offspring_initial_resources: 10.0
starvation_threshold: 100
"""
        
        yaml_file = os.path.join(self.temp_dir, 'legacy_config.yaml')
        with open(yaml_file, 'w') as f:
            f.write(legacy_yaml)
        
        # Test loading legacy YAML
        config = SimulationConfig.from_yaml(yaml_file)
        
        # Check that agent config was created with default values
        self.assertIsInstance(config.agent, AgentComponentConfig)
        self.assertEqual(config.agent.movement.max_movement, 8.0)
        self.assertEqual(config.agent.combat.starting_health, 100.0)

    def test_yaml_saving_backward_compatibility(self):
        """Test that YAML saving maintains backward compatibility."""
        # Create a config with both old and new structures
        config = SimulationConfig()
        config.agent_parameters = {
            'SystemAgent': {
                'gather_efficiency_multiplier': 0.4
            }
        }
        config.agent.combat.starting_health = 150.0
        
        # Test saving to YAML
        yaml_file = os.path.join(self.temp_dir, 'mixed_config.yaml')
        config.to_yaml(yaml_file)
        
        # Test loading the saved YAML
        loaded_config = SimulationConfig.from_yaml(yaml_file)
        
        # Check that both structures are preserved
        self.assertIn('agent_parameters', loaded_config.__dict__)
        self.assertIsInstance(loaded_config.agent, AgentComponentConfig)
        self.assertEqual(loaded_config.agent.combat.starting_health, 150.0)

    def test_agent_config_preset_migration(self):
        """Test that agent config presets work with migration."""
        # Test that presets can be used in migrated configs
        config = SimulationConfig()
        
        # Test aggressive preset
        config.agent = AgentComponentConfig.aggressive()
        self.assertGreater(config.agent.combat.starting_health, 100.0)
        self.assertGreater(config.agent.combat.base_attack_strength, 10.0)
        
        # Test defensive preset
        config.agent = AgentComponentConfig.defensive()
        self.assertGreater(config.agent.combat.starting_health, 100.0)
        self.assertGreater(config.agent.combat.base_defense_strength, 5.0)
        
        # Test efficient preset
        config.agent = AgentComponentConfig.efficient()
        self.assertLess(config.agent.resource.base_consumption_rate, 1.0)
        self.assertLess(config.agent.combat.starting_health, 100.0)

    def test_from_simulation_config_migration(self):
        """Test that from_simulation_config works with migration."""
        # Create a simulation config with agent_behavior
        sim_config = SimulationConfig()
        sim_config.agent_behavior.base_consumption_rate = 2.0
        sim_config.agent_behavior.starvation_threshold = 150
        sim_config.agent_behavior.starting_health = 200.0
        
        # Test creating agent config from simulation config
        agent_config = AgentComponentConfig.from_simulation_config(sim_config)
        
        # Check that values were extracted correctly
        self.assertEqual(agent_config.resource.base_consumption_rate, 2.0)
        self.assertEqual(agent_config.resource.starvation_threshold, 150)
        self.assertEqual(agent_config.combat.starting_health, 200.0)

    def test_config_validation_migration(self):
        """Test that config validation works with migration."""
        # Test with valid migrated config
        config = SimulationConfig()
        config.agent = AgentComponentConfig.aggressive()
        
        # Test that config is valid
        self.assertIsInstance(config.agent, AgentComponentConfig)
        self.assertGreater(config.agent.combat.starting_health, 100.0)
        
        # Test that config can be serialized and deserialized
        config_dict = config.to_dict()
        loaded_config = SimulationConfig.from_dict(config_dict)
        
        self.assertIsInstance(loaded_config.agent, AgentComponentConfig)
        self.assertEqual(loaded_config.agent.combat.starting_health, 
                        config.agent.combat.starting_health)

    def test_environment_override_migration(self):
        """Test that environment overrides work with migration."""
        # Test that environment-specific agent configs work
        config = SimulationConfig()
        
        # Simulate development environment override
        config.agent.movement.max_movement = 5.0
        config.agent.resource.base_consumption_rate = 0.5
        config.agent.combat.starting_health = 50.0
        
        # Test that overrides are preserved
        self.assertEqual(config.agent.movement.max_movement, 5.0)
        self.assertEqual(config.agent.resource.base_consumption_rate, 0.5)
        self.assertEqual(config.agent.combat.starting_health, 50.0)
        
        # Test serialization and deserialization
        config_dict = config.to_dict()
        loaded_config = SimulationConfig.from_dict(config_dict)
        
        self.assertEqual(loaded_config.agent.movement.max_movement, 5.0)
        self.assertEqual(loaded_config.agent.resource.base_consumption_rate, 0.5)
        self.assertEqual(loaded_config.agent.combat.starting_health, 50.0)


if __name__ == '__main__':
    unittest.main()
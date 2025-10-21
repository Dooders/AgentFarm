"""
Integration tests for agent configuration in SimulationConfig.

This module tests the integration of AgentComponentConfig into the main
SimulationConfig system including serialization, deserialization, and
centralized config loading.
"""

import unittest
from unittest.mock import MagicMock, patch
import tempfile
import os
import yaml

# Mock torch to avoid dependency issues
import sys
sys.modules['torch'] = MagicMock()

from farm.config import SimulationConfig, load_config
from farm.core.agent.config import AgentComponentConfig


class TestAgentConfigInSimulationConfig(unittest.TestCase):
    """Test cases for agent configuration integration in SimulationConfig."""

    def test_simulation_config_has_agent_attribute(self):
        """Test that SimulationConfig has agent attribute."""
        config = SimulationConfig()
        
        self.assertTrue(hasattr(config, 'agent'))
        self.assertIsInstance(config.agent, AgentComponentConfig)

    def test_agent_config_default_values(self):
        """Test that agent config has correct default values."""
        config = SimulationConfig()
        agent = config.agent
        
        # Check movement defaults
        self.assertEqual(agent.movement.max_movement, 8.0)
        self.assertEqual(agent.movement.perception_radius, 5)
        
        # Check resource defaults
        self.assertEqual(agent.resource.base_consumption_rate, 1.0)
        self.assertEqual(agent.resource.starvation_threshold, 100)
        
        # Check combat defaults
        self.assertEqual(agent.combat.starting_health, 100.0)
        self.assertEqual(agent.combat.base_attack_strength, 10.0)

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
        
        # Check that values are correct
        self.assertEqual(agent_dict['movement']['max_movement'], 8.0)
        self.assertEqual(agent_dict['resource']['base_consumption_rate'], 1.0)
        self.assertEqual(agent_dict['combat']['starting_health'], 100.0)

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
        self.assertEqual(config.agent.movement.max_movement, 10.0)
        self.assertEqual(config.agent.movement.perception_radius, 8)
        self.assertEqual(config.agent.resource.base_consumption_rate, 2.0)
        self.assertEqual(config.agent.resource.starvation_threshold, 150)
        self.assertEqual(config.agent.combat.starting_health, 150.0)
        self.assertEqual(config.agent.combat.base_attack_strength, 20.0)
        self.assertEqual(config.agent.perception.perception_radius, 8)
        self.assertEqual(config.agent.perception.position_discretization_method, 'round')
        self.assertEqual(config.agent.reproduction.offspring_initial_resources, 15.0)
        self.assertEqual(config.agent.reproduction.offspring_cost, 8.0)

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
            self.assertIsInstance(loaded_config.agent, AgentComponentConfig)
            self.assertEqual(loaded_config.agent.movement.max_movement, 8.0)
            self.assertEqual(loaded_config.agent.resource.base_consumption_rate, 1.0)
            self.assertEqual(loaded_config.agent.combat.starting_health, 100.0)
            
        finally:
            # Clean up
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_agent_config_presets_in_simulation_config(self):
        """Test that agent config presets work in SimulationConfig."""
        # Test aggressive preset
        config = SimulationConfig()
        config.agent = AgentComponentConfig.aggressive()
        
        self.assertGreater(config.agent.combat.starting_health, 100.0)
        self.assertGreater(config.agent.combat.base_attack_strength, 10.0)
        self.assertGreater(config.agent.resource.base_consumption_rate, 1.0)
        
        # Test defensive preset
        config.agent = AgentComponentConfig.defensive()
        
        self.assertGreater(config.agent.combat.starting_health, 100.0)
        self.assertGreater(config.agent.combat.base_defense_strength, 5.0)
        self.assertLess(config.agent.combat.base_attack_strength, 10.0)
        
        # Test efficient preset
        config.agent = AgentComponentConfig.efficient()
        
        self.assertLess(config.agent.resource.base_consumption_rate, 1.0)
        self.assertLess(config.agent.resource.offspring_cost, 5.0)
        self.assertLess(config.agent.combat.starting_health, 100.0)

    def test_agent_config_serialization_roundtrip(self):
        """Test that agent config survives serialization roundtrip."""
        original_config = SimulationConfig()
        original_config.agent = AgentComponentConfig.aggressive()
        
        # Serialize to dict
        config_dict = original_config.to_dict()
        
        # Deserialize from dict
        loaded_config = SimulationConfig.from_dict(config_dict)
        
        # Check that agent config was preserved
        self.assertEqual(loaded_config.agent.combat.starting_health, 
                        original_config.agent.combat.starting_health)
        self.assertEqual(loaded_config.agent.combat.base_attack_strength, 
                        original_config.agent.combat.base_attack_strength)
        self.assertEqual(loaded_config.agent.resource.base_consumption_rate, 
                        original_config.agent.resource.base_consumption_rate)

    def test_agent_config_with_custom_values(self):
        """Test that custom agent config values are preserved."""
        config = SimulationConfig()
        
        # Modify agent config
        config.agent.movement.max_movement = 15.0
        config.agent.resource.base_consumption_rate = 3.0
        config.agent.combat.starting_health = 200.0
        
        # Serialize and deserialize
        config_dict = config.to_dict()
        loaded_config = SimulationConfig.from_dict(config_dict)
        
        # Check that custom values were preserved
        self.assertEqual(loaded_config.agent.movement.max_movement, 15.0)
        self.assertEqual(loaded_config.agent.resource.base_consumption_rate, 3.0)
        self.assertEqual(loaded_config.agent.combat.starting_health, 200.0)

    def test_agent_config_validation(self):
        """Test that agent config validation works correctly."""
        config = SimulationConfig()
        
        # Test with valid agent config
        config.agent = AgentComponentConfig()
        self.assertIsInstance(config.agent, AgentComponentConfig)
        
        # Test that agent config can be replaced
        config.agent = AgentComponentConfig.aggressive()
        self.assertIsInstance(config.agent, AgentComponentConfig)
        self.assertGreater(config.agent.combat.starting_health, 100.0)

    def test_agent_config_in_config_diff(self):
        """Test that agent config is included in config diff."""
        config1 = SimulationConfig()
        config2 = SimulationConfig()
        config2.agent.combat.starting_health = 150.0
        
        diff = config1.diff_config(config2)
        
        # Check that agent config differences are included
        agent_diff_keys = [key for key in diff.keys() if 'agent' in key]
        self.assertGreater(len(agent_diff_keys), 0)

    def test_agent_config_in_version_hash(self):
        """Test that agent config changes affect version hash."""
        config1 = SimulationConfig()
        config2 = SimulationConfig()
        config2.agent.combat.starting_health = 150.0
        
        hash1 = config1.generate_version_hash()
        hash2 = config2.generate_version_hash()
        
        # Hashes should be different due to agent config change
        self.assertNotEqual(hash1, hash2)

    def test_agent_config_in_versioned_config(self):
        """Test that agent config is included in versioned config."""
        config = SimulationConfig()
        config.agent.combat.starting_health = 150.0
        
        versioned_config = config.version_config("Test version")
        
        # Check that agent config is preserved in versioned config
        self.assertEqual(versioned_config.agent.combat.starting_health, 150.0)
        self.assertIsNotNone(versioned_config.versioning.config_version)
        self.assertIsNotNone(versioned_config.versioning.config_created_at)
        self.assertEqual(versioned_config.versioning.config_description, "Test version")


class TestAgentConfigCentralizedLoading(unittest.TestCase):
    """Test cases for agent configuration in centralized config loading."""

    @patch('farm.config.orchestrator.ConfigurationOrchestrator')
    def test_load_config_includes_agent_config(self, mock_orchestrator):
        """Test that load_config includes agent configuration."""
        # Mock the orchestrator to return a config with agent
        mock_config = SimulationConfig()
        mock_config.agent = AgentComponentConfig.aggressive()
        mock_orchestrator.return_value.load_config.return_value = mock_config
        
        # Mock the global orchestrator
        with patch('farm.config.orchestrator.get_global_orchestrator', return_value=mock_orchestrator.return_value):
            config = load_config("development")
            
            self.assertIsInstance(config.agent, AgentComponentConfig)
            self.assertGreater(config.agent.combat.starting_health, 100.0)

    def test_agent_config_environment_overrides(self):
        """Test that agent config can be overridden by environment."""
        # This test would require actual config files, so we'll test the structure
        # by checking that the config loading logic handles agent config
        
        config = SimulationConfig()
        
        # Test that agent config can be modified
        config.agent.movement.max_movement = 20.0
        config.agent.resource.base_consumption_rate = 5.0
        
        # Test that changes are preserved
        self.assertEqual(config.agent.movement.max_movement, 20.0)
        self.assertEqual(config.agent.resource.base_consumption_rate, 5.0)

    def test_agent_config_profile_overrides(self):
        """Test that agent config can be overridden by profile."""
        config = SimulationConfig()
        
        # Test different agent configs
        config.agent = AgentComponentConfig.aggressive()
        self.assertGreater(config.agent.combat.starting_health, 100.0)
        
        config.agent = AgentComponentConfig.defensive()
        self.assertGreater(config.agent.combat.base_defense_strength, 5.0)
        
        config.agent = AgentComponentConfig.efficient()
        self.assertLess(config.agent.resource.base_consumption_rate, 1.0)


if __name__ == '__main__':
    unittest.main()
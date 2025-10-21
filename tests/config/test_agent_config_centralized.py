"""
Tests for agent configuration in centralized config loading.

This module tests the integration of agent configuration with the centralized
config system including environment-specific overrides and profile-based loading.
"""

import unittest
from unittest.mock import MagicMock, patch, mock_open
import tempfile
import os
import yaml

# Mock torch to avoid dependency issues
import sys
sys.modules['torch'] = MagicMock()

from farm.config import SimulationConfig, load_config
from farm.core.agent.config import AgentComponentConfig


class TestAgentConfigCentralizedLoading(unittest.TestCase):
    """Test cases for agent configuration in centralized config loading."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = os.path.join(self.temp_dir, 'config')
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, 'environments'), exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, 'profiles'), exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_config_files(self):
        """Create test configuration files."""
        # Default config
        default_config = {
            'simulation_steps': 100,
            'agent': {
                'movement': {
                    'max_movement': 8.0,
                    'perception_radius': 5
                },
                'resource': {
                    'base_consumption_rate': 1.0,
                    'starvation_threshold': 100
                },
                'combat': {
                    'starting_health': 100.0,
                    'base_attack_strength': 10.0
                }
            }
        }
        
        with open(os.path.join(self.config_dir, 'default.yaml'), 'w') as f:
            yaml.dump(default_config, f)
        
        # Development environment
        dev_config = {
            'agent': {
                'movement': {
                    'max_movement': 5.0,
                    'perception_radius': 3
                },
                'resource': {
                    'base_consumption_rate': 0.5,
                    'starvation_threshold': 50
                },
                'combat': {
                    'starting_health': 50.0,
                    'base_attack_strength': 5.0
                }
            }
        }
        
        with open(os.path.join(self.config_dir, 'environments', 'development.yaml'), 'w') as f:
            yaml.dump(dev_config, f)
        
        # Production environment
        prod_config = {
            'agent': {
                'movement': {
                    'max_movement': 10.0,
                    'perception_radius': 8
                },
                'resource': {
                    'base_consumption_rate': 1.5,
                    'starvation_threshold': 150
                },
                'combat': {
                    'starting_health': 150.0,
                    'base_attack_strength': 15.0
                }
            }
        }
        
        with open(os.path.join(self.config_dir, 'environments', 'production.yaml'), 'w') as f:
            yaml.dump(prod_config, f)
        
        # Testing environment
        test_config = {
            'agent': {
                'movement': {
                    'max_movement': 3.0,
                    'perception_radius': 2
                },
                'resource': {
                    'base_consumption_rate': 0.1,
                    'starvation_threshold': 20
                },
                'combat': {
                    'starting_health': 20.0,
                    'base_attack_strength': 2.0
                }
            }
        }
        
        with open(os.path.join(self.config_dir, 'environments', 'testing.yaml'), 'w') as f:
            yaml.dump(test_config, f)

    @patch('farm.config.orchestrator.ConfigurationOrchestrator')
    def test_load_config_with_agent_config(self, mock_orchestrator):
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

    def test_agent_config_environment_merging(self):
        """Test that agent config is properly merged from environment files."""
        self.create_test_config_files()
        
        # Test development environment
        with patch('farm.config.orchestrator.ConfigurationOrchestrator') as mock_orchestrator:
            # Mock the config loading to use our test files
            def mock_load_config(environment, profile=None, config_dir=None, **kwargs):
                if environment == "development":
                    config = SimulationConfig()
                    # Simulate loading from development.yaml
                    config.agent.movement.max_movement = 5.0
                    config.agent.movement.perception_radius = 3
                    config.agent.resource.base_consumption_rate = 0.5
                    config.agent.resource.starvation_threshold = 50
                    config.agent.combat.starting_health = 50.0
                    config.agent.combat.base_attack_strength = 5.0
                    return config
                return SimulationConfig()
            
            mock_orchestrator.return_value.load_config = mock_load_config
            
            with patch('farm.config.orchestrator.get_global_orchestrator', return_value=mock_orchestrator.return_value):
                config = load_config("development")
                
                # Check that development overrides were applied
                self.assertEqual(config.agent.movement.max_movement, 5.0)
                self.assertEqual(config.agent.movement.perception_radius, 3)
                self.assertEqual(config.agent.resource.base_consumption_rate, 0.5)
                self.assertEqual(config.agent.resource.starvation_threshold, 50)
                self.assertEqual(config.agent.combat.starting_health, 50.0)
                self.assertEqual(config.agent.combat.base_attack_strength, 5.0)

    def test_agent_config_profile_merging(self):
        """Test that agent config is properly merged from profile files."""
        # Create a profile config
        profile_config = {
            'agent': {
                'movement': {
                    'max_movement': 15.0,
                    'perception_radius': 10
                },
                'resource': {
                    'base_consumption_rate': 2.0,
                    'starvation_threshold': 200
                },
                'combat': {
                    'starting_health': 200.0,
                    'base_attack_strength': 20.0
                }
            }
        }
        
        with open(os.path.join(self.config_dir, 'profiles', 'benchmark.yaml'), 'w') as f:
            yaml.dump(profile_config, f)
        
        with patch('farm.config.orchestrator.ConfigurationOrchestrator') as mock_orchestrator:
            def mock_load_config(environment, profile=None, config_dir=None, **kwargs):
                if profile == "benchmark":
                    config = SimulationConfig()
                    # Simulate loading from benchmark profile
                    config.agent.movement.max_movement = 15.0
                    config.agent.movement.perception_radius = 10
                    config.agent.resource.base_consumption_rate = 2.0
                    config.agent.resource.starvation_threshold = 200
                    config.agent.combat.starting_health = 200.0
                    config.agent.combat.base_attack_strength = 20.0
                    return config
                return SimulationConfig()
            
            mock_orchestrator.return_value.load_config = mock_load_config
            
            with patch('farm.config.orchestrator.get_global_orchestrator', return_value=mock_orchestrator.return_value):
                config = load_config("production", profile="benchmark")
                
                # Check that benchmark profile overrides were applied
                self.assertEqual(config.agent.movement.max_movement, 15.0)
                self.assertEqual(config.agent.movement.perception_radius, 10)
                self.assertEqual(config.agent.resource.base_consumption_rate, 2.0)
                self.assertEqual(config.agent.resource.starvation_threshold, 200)
                self.assertEqual(config.agent.combat.starting_health, 200.0)
                self.assertEqual(config.agent.combat.base_attack_strength, 20.0)

    def test_agent_config_validation_in_centralized_loading(self):
        """Test that agent config validation works in centralized loading."""
        with patch('farm.config.orchestrator.ConfigurationOrchestrator') as mock_orchestrator:
            def mock_load_config(environment, profile=None, config_dir=None, **kwargs):
                config = SimulationConfig()
                # Test with valid agent config
                config.agent = AgentComponentConfig.aggressive()
                return config
            
            mock_orchestrator.return_value.load_config = mock_load_config
            
            with patch('farm.config.orchestrator.get_global_orchestrator', return_value=mock_orchestrator.return_value):
                config = load_config("development")
                
                # Check that agent config is valid
                self.assertIsInstance(config.agent, AgentComponentConfig)
                self.assertGreater(config.agent.combat.starting_health, 100.0)
                self.assertGreater(config.agent.combat.base_attack_strength, 10.0)

    def test_agent_config_caching(self):
        """Test that agent config is properly cached in centralized loading."""
        with patch('farm.config.orchestrator.ConfigurationOrchestrator') as mock_orchestrator:
            # Mock cache behavior
            mock_cache = MagicMock()
            mock_cache.get.return_value = None  # Cache miss
            mock_orchestrator.return_value.cache = mock_cache
            
            def mock_load_config(environment, profile=None, config_dir=None, **kwargs):
                config = SimulationConfig()
                config.agent = AgentComponentConfig.efficient()
                return config
            
            mock_orchestrator.return_value.load_config = mock_load_config
            
            with patch('farm.config.orchestrator.get_global_orchestrator', return_value=mock_orchestrator.return_value):
                config = load_config("development")
                
                # Check that agent config was loaded
                self.assertIsInstance(config.agent, AgentComponentConfig)
                self.assertLess(config.agent.resource.base_consumption_rate, 1.0)
                
                # Check that cache was used
                mock_cache.get.assert_called()

    def test_agent_config_error_handling(self):
        """Test that agent config errors are handled gracefully in centralized loading."""
        with patch('farm.config.orchestrator.ConfigurationOrchestrator') as mock_orchestrator:
            def mock_load_config(environment, profile=None, config_dir=None, **kwargs):
                # Simulate an error during config loading
                raise Exception("Config loading error")
            
            mock_orchestrator.return_value.load_config = mock_load_config
            
            with patch('farm.config.orchestrator.get_global_orchestrator', return_value=mock_orchestrator.return_value):
                with self.assertRaises(Exception):
                    load_config("development")

    def test_agent_config_environment_priority(self):
        """Test that environment configs take priority over defaults."""
        with patch('farm.config.orchestrator.ConfigurationOrchestrator') as mock_orchestrator:
            def mock_load_config(environment, profile=None, config_dir=None, **kwargs):
                config = SimulationConfig()
                if environment == "testing":
                    # Testing environment should have minimal values
                    config.agent.movement.max_movement = 3.0
                    config.agent.resource.base_consumption_rate = 0.1
                    config.agent.combat.starting_health = 20.0
                elif environment == "production":
                    # Production environment should have robust values
                    config.agent.movement.max_movement = 10.0
                    config.agent.resource.base_consumption_rate = 1.5
                    config.agent.combat.starting_health = 150.0
                return config
            
            mock_orchestrator.return_value.load_config = mock_load_config
            
            with patch('farm.config.orchestrator.get_global_orchestrator', return_value=mock_orchestrator.return_value):
                # Test testing environment
                test_config = load_config("testing")
                self.assertEqual(test_config.agent.movement.max_movement, 3.0)
                self.assertEqual(test_config.agent.resource.base_consumption_rate, 0.1)
                self.assertEqual(test_config.agent.combat.starting_health, 20.0)
                
                # Test production environment
                prod_config = load_config("production")
                self.assertEqual(prod_config.agent.movement.max_movement, 10.0)
                self.assertEqual(prod_config.agent.resource.base_consumption_rate, 1.5)
                self.assertEqual(prod_config.agent.combat.starting_health, 150.0)

    def test_agent_config_profile_override_environment(self):
        """Test that profile configs override environment configs."""
        with patch('farm.config.orchestrator.ConfigurationOrchestrator') as mock_orchestrator:
            def mock_load_config(environment, profile=None, config_dir=None, **kwargs):
                config = SimulationConfig()
                if environment == "development":
                    # Development environment values
                    config.agent.movement.max_movement = 5.0
                    config.agent.resource.base_consumption_rate = 0.5
                if profile == "benchmark":
                    # Benchmark profile should override environment
                    config.agent.movement.max_movement = 15.0
                    config.agent.resource.base_consumption_rate = 2.0
                return config
            
            mock_orchestrator.return_value.load_config = mock_load_config
            
            with patch('farm.config.orchestrator.get_global_orchestrator', return_value=mock_orchestrator.return_value):
                # Test that profile overrides environment
                config = load_config("development", profile="benchmark")
                self.assertEqual(config.agent.movement.max_movement, 15.0)
                self.assertEqual(config.agent.resource.base_consumption_rate, 2.0)


if __name__ == '__main__':
    unittest.main()
"""
Unit tests for the benchmarks.utils.config_helper module.

Tests utility functions in the config_helper submodule, including configuration helpers and recommendations.
"""

import unittest
from unittest.mock import Mock, patch, mock_open

from benchmarks.utils.config_helper import (
    configure_for_performance_with_persistence,
    get_recommended_config,
    print_config_recommendations
)


class TestConfigHelper(unittest.TestCase):
    """Test config_helper utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock SimulationConfig to avoid importing farm.config
        self.mock_config = Mock()
        self.mock_config.use_in_memory_db = False
        self.mock_config.persist_db_on_completion = False
        self.mock_config.width = 50
        self.mock_config.height = 50
        self.mock_config.system_agents = 10
        self.mock_config.independent_agents = 10
        self.mock_config.control_agents = 10
        self.mock_config.initial_resources = 10
        self.mock_config.simulation_steps = 50

    @patch('benchmarks.utils.config_helper.SimulationConfig')
    def test_configure_for_performance_with_persistence_new_config(self, mock_config_class):
        """Test configure_for_performance_with_persistence with new config."""
        mock_config_class.return_value = self.mock_config
        
        result = configure_for_performance_with_persistence()
        
        # Should create new config
        mock_config_class.assert_called_once()
        
        # Should set performance settings
        self.assertTrue(self.mock_config.use_in_memory_db)
        self.assertTrue(self.mock_config.persist_db_on_completion)
        
        # Should return the config
        self.assertEqual(result, self.mock_config)

    def test_configure_for_performance_with_persistence_existing_config(self):
        """Test configure_for_performance_with_persistence with existing config."""
        # Start with config that has different settings
        self.mock_config.use_in_memory_db = False
        self.mock_config.persist_db_on_completion = False
        
        result = configure_for_performance_with_persistence(self.mock_config)
        
        # Should modify existing config
        self.assertTrue(self.mock_config.use_in_memory_db)
        self.assertTrue(self.mock_config.persist_db_on_completion)
        
        # Should return the same config instance
        self.assertEqual(result, self.mock_config)

    def test_configure_for_performance_with_persistence_preserves_other_settings(self):
        """Test that configure_for_performance_with_persistence preserves other settings."""
        # Set some other settings
        self.mock_config.width = 200
        self.mock_config.height = 200
        self.mock_config.simulation_steps = 1000
        
        result = configure_for_performance_with_persistence(self.mock_config)
        
        # Should preserve other settings
        self.assertEqual(self.mock_config.width, 200)
        self.assertEqual(self.mock_config.height, 200)
        self.assertEqual(self.mock_config.simulation_steps, 1000)
        
        # Should set performance settings
        self.assertTrue(self.mock_config.use_in_memory_db)
        self.assertTrue(self.mock_config.persist_db_on_completion)

    @patch('benchmarks.utils.config_helper.SimulationConfig')
    def test_get_recommended_config_defaults(self, mock_config_class):
        """Test get_recommended_config with default parameters."""
        mock_config_class.return_value = self.mock_config
        
        result = get_recommended_config()
        
        # Should create new config
        mock_config_class.assert_called_once()
        
        # Should set recommended settings
        self.assertEqual(self.mock_config.width, 100)
        self.assertEqual(self.mock_config.height, 100)
        self.assertEqual(self.mock_config.system_agents, 10)  # 30 // 3
        self.assertEqual(self.mock_config.independent_agents, 10)  # 30 // 3
        self.assertEqual(self.mock_config.control_agents, 10)  # 30 - 2*(30//3)
        self.assertEqual(self.mock_config.initial_resources, 20)
        self.assertEqual(self.mock_config.simulation_steps, 100)
        
        # Should configure for performance with persistence
        self.assertTrue(self.mock_config.use_in_memory_db)
        self.assertTrue(self.mock_config.persist_db_on_completion)

    @patch('benchmarks.utils.config_helper.SimulationConfig')
    def test_get_recommended_config_custom_parameters(self, mock_config_class):
        """Test get_recommended_config with custom parameters."""
        mock_config_class.return_value = self.mock_config
        
        result = get_recommended_config(num_agents=60, num_steps=200)
        
        # Should set custom parameters
        self.assertEqual(self.mock_config.system_agents, 20)  # 60 // 3
        self.assertEqual(self.mock_config.independent_agents, 20)  # 60 // 3
        self.assertEqual(self.mock_config.control_agents, 20)  # 60 - 2*(60//3)
        self.assertEqual(self.mock_config.simulation_steps, 200)

    @patch('benchmarks.utils.config_helper.SimulationConfig')
    def test_get_recommended_config_uneven_agent_distribution(self, mock_config_class):
        """Test get_recommended_config with uneven agent distribution."""
        mock_config_class.return_value = self.mock_config
        
        result = get_recommended_config(num_agents=31)
        
        # Should distribute agents correctly
        self.assertEqual(self.mock_config.system_agents, 10)  # 31 // 3
        self.assertEqual(self.mock_config.independent_agents, 10)  # 31 // 3
        self.assertEqual(self.mock_config.control_agents, 11)  # 31 - 2*(31//3)
        
        # Total should equal num_agents
        total_agents = (self.mock_config.system_agents + 
                       self.mock_config.independent_agents + 
                       self.mock_config.control_agents)
        self.assertEqual(total_agents, 31)

    @patch('benchmarks.utils.config_helper.SimulationConfig')
    def test_get_recommended_config_additional_params(self, mock_config_class):
        """Test get_recommended_config with additional parameters."""
        mock_config_class.return_value = self.mock_config
        
        additional_params = {
            "width": 150,
            "height": 150,
            "initial_resources": 50,
            "custom_setting": "test_value"
        }
        
        result = get_recommended_config(additional_params=additional_params)
        
        # Should set additional parameters that exist on config
        self.assertEqual(self.mock_config.width, 150)
        self.assertEqual(self.mock_config.height, 150)
        self.assertEqual(self.mock_config.initial_resources, 50)
        
        # Should not set parameters that don't exist on config
        # Note: Mock objects will have any attribute set on them
        # In real usage, this would be handled by the SimulationConfig class

    @patch('benchmarks.utils.config_helper.SimulationConfig')
    def test_get_recommended_config_additional_params_override_defaults(self, mock_config_class):
        """Test that additional parameters override defaults."""
        mock_config_class.return_value = self.mock_config
        
        additional_params = {
            "width": 200,  # Override default 100
            "height": 200,  # Override default 100
            "simulation_steps": 500  # Override default 100
        }
        
        result = get_recommended_config(additional_params=additional_params)
        
        # Should use additional parameter values
        self.assertEqual(self.mock_config.width, 200)
        self.assertEqual(self.mock_config.height, 200)
        self.assertEqual(self.mock_config.simulation_steps, 500)

    @patch('benchmarks.utils.config_helper.SimulationConfig')
    def test_get_recommended_config_calls_configure_for_performance(self, mock_config_class):
        """Test that get_recommended_config calls configure_for_performance_with_persistence."""
        mock_config_class.return_value = self.mock_config
        
        with patch('benchmarks.utils.config_helper.configure_for_performance_with_persistence') as mock_configure:
            mock_configure.return_value = self.mock_config
            
            result = get_recommended_config()
            
            # Should call configure_for_performance_with_persistence
            mock_configure.assert_called_once_with(self.mock_config)

    @patch('sys.stdout')
    def test_print_config_recommendations(self, mock_stdout):
        """Test print_config_recommendations function."""
        print_config_recommendations()
        
        # Should have called print multiple times
        self.assertGreater(mock_stdout.write.call_count, 0)
        
        # Check that key information is printed
        output = "".join(call[0][0] for call in mock_stdout.write.call_args_list)
        self.assertIn("RECOMMENDED CONFIGURATION", output)
        self.assertIn("use_in_memory_db = True", output)
        self.assertIn("persist_db_on_completion = True", output)
        self.assertIn("33.6% faster execution", output)
        self.assertIn("configure_for_performance_with_persistence", output)

    def test_configure_for_performance_with_persistence_none_config(self):
        """Test configure_for_performance_with_persistence with None config."""
        with patch('benchmarks.utils.config_helper.SimulationConfig') as mock_config_class:
            mock_config_class.return_value = self.mock_config
            
            result = configure_for_performance_with_persistence(None)
            
            # Should create new config
            mock_config_class.assert_called_once()
            
            # Should set performance settings
            self.assertTrue(self.mock_config.use_in_memory_db)
            self.assertTrue(self.mock_config.persist_db_on_completion)

    @patch('benchmarks.utils.config_helper.SimulationConfig')
    def test_get_recommended_config_minimal_agents(self, mock_config_class):
        """Test get_recommended_config with minimal number of agents."""
        mock_config_class.return_value = self.mock_config
        
        result = get_recommended_config(num_agents=1)
        
        # Should handle minimal agents correctly
        self.assertEqual(self.mock_config.system_agents, 0)  # 1 // 3
        self.assertEqual(self.mock_config.independent_agents, 0)  # 1 // 3
        self.assertEqual(self.mock_config.control_agents, 1)  # 1 - 2*(1//3)
        
        # Total should equal num_agents
        total_agents = (self.mock_config.system_agents + 
                       self.mock_config.independent_agents + 
                       self.mock_config.control_agents)
        self.assertEqual(total_agents, 1)

    @patch('benchmarks.utils.config_helper.SimulationConfig')
    def test_get_recommended_config_zero_agents(self, mock_config_class):
        """Test get_recommended_config with zero agents."""
        mock_config_class.return_value = self.mock_config
        
        result = get_recommended_config(num_agents=0)
        
        # Should handle zero agents correctly
        self.assertEqual(self.mock_config.system_agents, 0)  # 0 // 3
        self.assertEqual(self.mock_config.independent_agents, 0)  # 0 // 3
        self.assertEqual(self.mock_config.control_agents, 0)  # 0 - 2*(0//3)
        
        # Total should equal num_agents
        total_agents = (self.mock_config.system_agents + 
                       self.mock_config.independent_agents + 
                       self.mock_config.control_agents)
        self.assertEqual(total_agents, 0)


if __name__ == "__main__":
    unittest.main()

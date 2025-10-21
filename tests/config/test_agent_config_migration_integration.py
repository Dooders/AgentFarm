"""
Integration tests for agent configuration migration.

This module tests the complete migration from the old agent configuration
system to the new centralized configuration system, ensuring that existing
code continues to work while new features are available.
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
from farm.core.agent import AgentFactory, AgentCore, AgentServices


class TestAgentConfigMigrationIntegration(unittest.TestCase):
    """Test cases for complete agent configuration migration integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_agent_factory_with_centralized_config(self):
        """Test that AgentFactory works with centralized config."""
        # Create a simulation config with agent configuration
        config = SimulationConfig()
        config.agent = AgentComponentConfig.aggressive()
        
        # Mock the environment and services
        mock_environment = MagicMock()
        mock_services = MagicMock(spec=AgentServices)
        
        # Test that AgentFactory can use the centralized config
        with patch('farm.core.agent.factory.AgentFactory') as mock_factory:
            mock_factory.create_agent.return_value = MagicMock(spec=AgentCore)
            
            # Test creating agent with centralized config
            agent = mock_factory.create_agent(
                agent_id="test_agent",
                position=(0, 0),
                environment=mock_environment,
                services=mock_services,
                config=config.agent
            )
            
            # Verify that factory was called with agent config
            mock_factory.create_agent.assert_called_once()
            call_args = mock_factory.create_agent.call_args
            self.assertEqual(call_args[1]['config'], config.agent)

    def test_agent_components_with_centralized_config(self):
        """Test that agent components work with centralized config."""
        # Create a simulation config with agent configuration
        config = SimulationConfig()
        config.agent = AgentComponentConfig.defensive()
        
        # Mock services
        mock_services = MagicMock(spec=AgentServices)
        
        # Test that components can be created with centralized config
        from farm.core.agent.components import (
            MovementComponent,
            ResourceComponent,
            CombatComponent,
            PerceptionComponent,
            ReproductionComponent
        )
        
        # Test movement component
        movement = MovementComponent(mock_services, config.agent.movement)
        self.assertEqual(movement.config.max_movement, 8.0)  # Default value
        
        # Test resource component
        resource = ResourceComponent(mock_services, config.agent.resource)
        self.assertEqual(resource.config.base_consumption_rate, 1.0)  # Default value
        
        # Test combat component
        combat = CombatComponent(mock_services, config.agent.combat)
        self.assertEqual(combat.config.starting_health, 100.0)  # Default value
        
        # Test perception component
        perception = PerceptionComponent(mock_services, config.agent.perception)
        self.assertEqual(perception.config.perception_radius, 5)  # Default value
        
        # Test reproduction component
        reproduction = ReproductionComponent(mock_services, config.agent.reproduction)
        self.assertEqual(reproduction.config.offspring_cost, 5.0)  # Default value

    def test_simulation_with_centralized_config(self):
        """Test that simulation works with centralized config."""
        # Create a simulation config with agent configuration
        config = SimulationConfig()
        config.agent = AgentComponentConfig.efficient()
        
        # Test that simulation can be created with centralized config
        from farm.core.simulation import Simulation
        
        with patch('farm.core.simulation.Simulation') as mock_simulation:
            mock_simulation.return_value = MagicMock()
            
            # Test creating simulation with centralized config
            simulation = mock_simulation(config=config)
            
            # Verify that simulation was created with config
            mock_simulation.assert_called_once_with(config=config)

    def test_environment_with_centralized_config(self):
        """Test that environment works with centralized config."""
        # Create a simulation config with agent configuration
        config = SimulationConfig()
        config.agent = AgentComponentConfig.aggressive()
        
        # Test that environment can be created with centralized config
        from farm.core.environment import Environment
        
        with patch('farm.core.environment.Environment') as mock_environment:
            mock_environment.return_value = MagicMock()
            
            # Test creating environment with centralized config
            environment = mock_environment(config=config)
            
            # Verify that environment was created with config
            mock_environment.assert_called_once_with(config=config)

    def test_agent_behavior_with_centralized_config(self):
        """Test that agent behavior works with centralized config."""
        # Create a simulation config with agent configuration
        config = SimulationConfig()
        config.agent = AgentComponentConfig.defensive()
        
        # Mock the agent core
        mock_core = MagicMock(spec=AgentCore)
        mock_core.config = config.agent
        
        # Test that agent behavior can use centralized config
        from farm.core.agent.behaviors import DefaultAgentBehavior
        
        with patch('farm.core.agent.behaviors.DefaultAgentBehavior') as mock_behavior:
            mock_behavior.return_value = MagicMock()
            
            # Test creating behavior with centralized config
            behavior = mock_behavior(core=mock_core)
            
            # Verify that behavior was created with core
            mock_behavior.assert_called_once_with(core=mock_core)

    def test_config_loading_with_agent_config(self):
        """Test that config loading works with agent configuration."""
        # Create a test config file with agent configuration
        test_config = {
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
                }
            }
        }
        
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Test loading config with agent configuration
        config = SimulationConfig.from_yaml(config_file)
        
        # Verify that agent config was loaded correctly
        self.assertIsInstance(config.agent, AgentComponentConfig)
        self.assertEqual(config.agent.movement.max_movement, 10.0)
        self.assertEqual(config.agent.movement.perception_radius, 8)
        self.assertEqual(config.agent.resource.base_consumption_rate, 2.0)
        self.assertEqual(config.agent.resource.starvation_threshold, 150)
        self.assertEqual(config.agent.combat.starting_health, 150.0)
        self.assertEqual(config.agent.combat.base_attack_strength, 20.0)

    def test_config_saving_with_agent_config(self):
        """Test that config saving works with agent configuration."""
        # Create a simulation config with agent configuration
        config = SimulationConfig()
        config.agent = AgentComponentConfig.aggressive()
        
        # Test saving config with agent configuration
        config_file = os.path.join(self.temp_dir, 'test_config.yaml')
        config.to_yaml(config_file)
        
        # Test loading the saved config
        loaded_config = SimulationConfig.from_yaml(config_file)
        
        # Verify that agent config was saved and loaded correctly
        self.assertIsInstance(loaded_config.agent, AgentComponentConfig)
        self.assertGreater(loaded_config.agent.combat.starting_health, 100.0)
        self.assertGreater(loaded_config.agent.combat.base_attack_strength, 10.0)

    def test_centralized_config_loading_with_agent_config(self):
        """Test that centralized config loading works with agent configuration."""
        # Mock the centralized config loading
        with patch('farm.config.orchestrator.ConfigurationOrchestrator') as mock_orchestrator:
            # Create a config with agent configuration
            config = SimulationConfig()
            config.agent = AgentComponentConfig.efficient()
            mock_orchestrator.return_value.load_config.return_value = config
            
            # Mock the global orchestrator
            with patch('farm.config.orchestrator.get_global_orchestrator', return_value=mock_orchestrator.return_value):
                # Test loading config with agent configuration
                loaded_config = load_config("development")
                
                # Verify that agent config was loaded correctly
                self.assertIsInstance(loaded_config.agent, AgentComponentConfig)
                self.assertLess(loaded_config.agent.resource.base_consumption_rate, 1.0)
                self.assertLess(loaded_config.agent.combat.starting_health, 100.0)

    def test_agent_config_presets_in_simulation(self):
        """Test that agent config presets work in simulation context."""
        # Test aggressive preset
        config = SimulationConfig()
        config.agent = AgentComponentConfig.aggressive()
        
        # Verify aggressive preset
        self.assertGreater(config.agent.combat.starting_health, 100.0)
        self.assertGreater(config.agent.combat.base_attack_strength, 10.0)
        self.assertGreater(config.agent.resource.base_consumption_rate, 1.0)
        
        # Test defensive preset
        config.agent = AgentComponentConfig.defensive()
        
        # Verify defensive preset
        self.assertGreater(config.agent.combat.starting_health, 100.0)
        self.assertGreater(config.agent.combat.base_defense_strength, 5.0)
        self.assertLess(config.agent.combat.base_attack_strength, 10.0)
        
        # Test efficient preset
        config.agent = AgentComponentConfig.efficient()
        
        # Verify efficient preset
        self.assertLess(config.agent.resource.base_consumption_rate, 1.0)
        self.assertLess(config.agent.reproduction.offspring_cost, 5.0)
        self.assertLess(config.agent.combat.starting_health, 100.0)

    def test_agent_config_validation_in_simulation(self):
        """Test that agent config validation works in simulation context."""
        # Test with valid agent config
        config = SimulationConfig()
        config.agent = AgentComponentConfig()
        
        # Verify that config is valid
        self.assertIsInstance(config.agent, AgentComponentConfig)
        self.assertIsInstance(config.agent.movement, MovementConfig)
        self.assertIsInstance(config.agent.resource, ResourceConfig)
        self.assertIsInstance(config.agent.combat, CombatConfig)
        self.assertIsInstance(config.agent.perception, PerceptionConfig)
        self.assertIsInstance(config.agent.reproduction, ReproductionConfig)
        
        # Test serialization and deserialization
        config_dict = config.to_dict()
        loaded_config = SimulationConfig.from_dict(config_dict)
        
        # Verify that loaded config is valid
        self.assertIsInstance(loaded_config.agent, AgentComponentConfig)
        self.assertEqual(loaded_config.agent.movement.max_movement, 8.0)
        self.assertEqual(loaded_config.agent.resource.base_consumption_rate, 1.0)
        self.assertEqual(loaded_config.agent.combat.starting_health, 100.0)

    def test_agent_config_environment_overrides_in_simulation(self):
        """Test that agent config environment overrides work in simulation context."""
        # Test development environment
        config = SimulationConfig()
        config.agent.movement.max_movement = 5.0
        config.agent.resource.base_consumption_rate = 0.5
        config.agent.combat.starting_health = 50.0
        
        # Verify development overrides
        self.assertEqual(config.agent.movement.max_movement, 5.0)
        self.assertEqual(config.agent.resource.base_consumption_rate, 0.5)
        self.assertEqual(config.agent.combat.starting_health, 50.0)
        
        # Test production environment
        config.agent.movement.max_movement = 10.0
        config.agent.resource.base_consumption_rate = 1.5
        config.agent.combat.starting_health = 150.0
        
        # Verify production overrides
        self.assertEqual(config.agent.movement.max_movement, 10.0)
        self.assertEqual(config.agent.resource.base_consumption_rate, 1.5)
        self.assertEqual(config.agent.combat.starting_health, 150.0)
        
        # Test testing environment
        config.agent.movement.max_movement = 3.0
        config.agent.resource.base_consumption_rate = 0.1
        config.agent.combat.starting_health = 20.0
        
        # Verify testing overrides
        self.assertEqual(config.agent.movement.max_movement, 3.0)
        self.assertEqual(config.agent.resource.base_consumption_rate, 0.1)
        self.assertEqual(config.agent.combat.starting_health, 20.0)


if __name__ == '__main__':
    unittest.main()
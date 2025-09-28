"""
Integration tests for the environment-specific configuration system.

This module tests the EnvironmentConfigManager class and its integration
with the hierarchical configuration system.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open

from farm.core.config import (
    EnvironmentConfigManager,
    HierarchicalConfig,
    ConfigurationValidator,
    ConfigurationLoadError,
    ConfigurationError
)


class TestEnvironmentConfigManager:
    """Test cases for EnvironmentConfigManager class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory with test configuration files."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        
        # Create base config
        base_config = {
            'simulation_id': 'test-simulation',
            'max_steps': 1000,
            'debug': False,
            'learning_rate': 0.001,
            'database': {
                'host': 'localhost',
                'port': 5439
            }
        }
        
        base_file = config_dir / "base.yaml"
        import yaml
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create environments directory
        env_dir = config_dir / "environments"
        env_dir.mkdir()
        
        # Create development environment config
        dev_config = {
            'debug': True,
            'max_steps': 100,
            'database': {
                'host': 'dev-db'
            }
        }
        
        dev_file = env_dir / "development.yaml"
        with open(dev_file, 'w') as f:
            yaml.dump(dev_config, f)
        
        # Create production environment config
        prod_config = {
            'debug': False,
            'max_steps': 2000,
            'learning_rate': 0.0005,
            'database': {
                'host': 'prod-db',
                'port': 5432
            }
        }
        
        prod_file = env_dir / "production.yaml"
        with open(prod_file, 'w') as f:
            yaml.dump(prod_config, f)
        
        # Create agents directory
        agents_dir = config_dir / "agents"
        agents_dir.mkdir()
        
        # Create system agent config
        system_agent_config = {
            'learning_rate': 0.0008,
            'agent_parameters': {
                'SystemAgent': {
                    'share_weight': 0.4
                }
            }
        }
        
        system_agent_file = agents_dir / "system_agent.yaml"
        with open(system_agent_file, 'w') as f:
            yaml.dump(system_agent_config, f)
        
        yield config_dir
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_initialization_with_explicit_environment(self, temp_config_dir):
        """Test EnvironmentConfigManager initialization with explicit environment."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(
            base_config_path=str(base_config_path),
            environment="development"
        )
        
        assert manager.environment == "development"
        assert manager.base_config_path == base_config_path
        assert manager.config_dir == temp_config_dir
    
    def test_initialization_with_auto_detection(self, temp_config_dir):
        """Test EnvironmentConfigManager initialization with auto-detection."""
        base_config_path = temp_config_dir / "base.yaml"
        
        with patch.dict(os.environ, {'FARM_ENVIRONMENT': 'production'}):
            manager = EnvironmentConfigManager(base_config_path=str(base_config_path))
            assert manager.environment == "production"
    
    def test_initialization_with_fallback_environment(self, temp_config_dir):
        """Test EnvironmentConfigManager initialization with fallback to default."""
        base_config_path = temp_config_dir / "base.yaml"
        
        with patch.dict(os.environ, {}, clear=True):
            manager = EnvironmentConfigManager(base_config_path=str(base_config_path))
            assert manager.environment == "development"  # Default environment
    
    def test_load_config_hierarchy(self, temp_config_dir):
        """Test loading configuration hierarchy from files."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(
            base_config_path=str(base_config_path),
            environment="development"
        )
        
        config_hierarchy = manager.get_config_hierarchy()
        
        assert isinstance(config_hierarchy, HierarchicalConfig)
        
        # Check global config (base)
        assert config_hierarchy.global_config['simulation_id'] == 'test-simulation'
        assert config_hierarchy.global_config['debug'] == False
        assert config_hierarchy.global_config['database']['host'] == 'localhost'
        
        # Check environment config (development)
        assert config_hierarchy.environment_config['debug'] == True
        assert config_hierarchy.environment_config['max_steps'] == 100
        assert config_hierarchy.environment_config['database']['host'] == 'dev-db'
        
        # Check agent config (system_agent)
        assert config_hierarchy.agent_config['learning_rate'] == 0.0008
        assert config_hierarchy.agent_config['agent_parameters']['SystemAgent']['share_weight'] == 0.4
    
    def test_hierarchical_lookup(self, temp_config_dir):
        """Test hierarchical configuration lookup."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(
            base_config_path=str(base_config_path),
            environment="development"
        )
        
        # Agent config should override environment and global
        assert manager.get('learning_rate') == 0.0008  # From agent config
        
        # Environment config should override global
        assert manager.get('debug') == True  # From environment config
        assert manager.get('max_steps') == 100  # From environment config
        
        # Global config should be used when not overridden
        assert manager.get('simulation_id') == 'test-simulation'  # From global config
        
        # Nested lookup
        assert manager.get_nested('database.host') == 'dev-db'  # From environment config
        assert manager.get_nested('database.port') == 5439  # From global config (not overridden)
    
    def test_environment_switching(self, temp_config_dir):
        """Test switching between environments."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(
            base_config_path=str(base_config_path),
            environment="development"
        )
        
        # Check development environment values
        assert manager.get('debug') == True
        assert manager.get('max_steps') == 100
        
        # Switch to production environment
        manager.set_environment("production")
        
        # Check production environment values
        assert manager.get('debug') == False
        assert manager.get('max_steps') == 2000
        assert manager.get('learning_rate') == 0.0005
        assert manager.get_nested('database.host') == 'prod-db'
        assert manager.get_nested('database.port') == 5432
    
    def test_get_effective_config(self, temp_config_dir):
        """Test getting effective configuration with all overrides applied."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(
            base_config_path=str(base_config_path),
            environment="development"
        )
        
        effective_config = manager.get_effective_config()
        
        # Should have all keys with highest priority values
        assert effective_config['simulation_id'] == 'test-simulation'  # From global
        assert effective_config['debug'] == True  # From environment
        assert effective_config['max_steps'] == 100  # From environment
        assert effective_config['learning_rate'] == 0.0008  # From agent
        assert effective_config['database']['host'] == 'dev-db'  # From environment
        assert effective_config['database']['port'] == 5439  # From global
    
    def test_available_environments(self, temp_config_dir):
        """Test getting list of available environments."""
        base_config_path = temp_config_path / "base.yaml"
        
        manager = EnvironmentConfigManager(base_config_path=str(base_config_path))
        
        available_envs = manager.get_available_environments()
        assert 'development' in available_envs
        assert 'production' in available_envs
        assert len(available_envs) == 2
    
    def test_available_agent_types(self, temp_config_dir):
        """Test getting list of available agent types."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(base_config_path=str(base_config_path))
        
        available_agents = manager.get_available_agent_types()
        assert 'system_agent' in available_agents
        assert len(available_agents) == 1
    
    def test_configuration_file_validation(self, temp_config_dir):
        """Test validation of configuration files."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(base_config_path=str(base_config_path))
        
        validation_results = manager.validate_configuration_files()
        
        # All files should be valid
        for file_path, errors in validation_results.items():
            assert len(errors) == 0, f"File {file_path} has validation errors: {errors}"
    
    def test_configuration_summary(self, temp_config_dir):
        """Test getting configuration summary."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(
            base_config_path=str(base_config_path),
            environment="development"
        )
        
        summary = manager.get_configuration_summary()
        
        assert summary['environment'] == 'development'
        assert summary['base_config_path'] == str(base_config_path)
        assert 'development' in summary['available_environments']
        assert 'system_agent' in summary['available_agent_types']
        assert summary['config_layers']['global_keys'] > 0
        assert summary['config_layers']['environment_keys'] > 0
        assert summary['config_layers']['agent_keys'] > 0
    
    def test_create_environment_config(self, temp_config_dir):
        """Test creating new environment configuration."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(base_config_path=str(base_config_path))
        
        test_config = {
            'debug': True,
            'max_steps': 500,
            'custom_setting': 'test_value'
        }
        
        manager.create_environment_config('test_env', test_config)
        
        # Verify file was created
        test_env_file = temp_config_dir / "environments" / "test_env.yaml"
        assert test_env_file.exists()
        
        # Verify content
        import yaml
        with open(test_env_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config == test_config
    
    def test_create_agent_config(self, temp_config_dir):
        """Test creating new agent configuration."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(base_config_path=str(base_config_path))
        
        test_config = {
            'learning_rate': 0.002,
            'custom_parameter': 'test_value'
        }
        
        manager.create_agent_config('test_agent', test_config)
        
        # Verify file was created
        test_agent_file = temp_config_dir / "agents" / "test_agent.yaml"
        assert test_agent_file.exists()
        
        # Verify content
        import yaml
        with open(test_agent_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config == test_config
    
    def test_missing_base_config_file(self, temp_config_dir):
        """Test handling of missing base configuration file."""
        missing_config_path = temp_config_dir / "missing.yaml"
        
        with pytest.raises(ConfigurationError, match="Failed to load base configuration"):
            EnvironmentConfigManager(base_config_path=str(missing_config_path))
    
    def test_invalid_yaml_file(self, temp_config_dir):
        """Test handling of invalid YAML files."""
        base_config_path = temp_config_dir / "base.yaml"
        
        # Create invalid YAML file
        invalid_file = temp_config_dir / "environments" / "invalid.yaml"
        with open(invalid_file, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        manager = EnvironmentConfigManager(base_config_path=str(base_config_path))
        
        # Should not raise exception during initialization
        # But validation should show errors
        validation_results = manager.validate_configuration_files()
        assert len(validation_results[str(invalid_file)]) > 0
    
    def test_force_reload(self, temp_config_dir):
        """Test force reloading configuration."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(
            base_config_path=str(base_config_path),
            environment="development"
        )
        
        # Get initial config
        initial_debug = manager.get('debug')
        assert initial_debug == True
        
        # Modify environment file
        env_file = temp_config_dir / "environments" / "development.yaml"
        import yaml
        with open(env_file, 'r') as f:
            config = yaml.safe_load(f)
        
        config['debug'] = False
        
        with open(env_file, 'w') as f:
            yaml.dump(config, f)
        
        # Get config without reload (should be cached)
        cached_debug = manager.get('debug')
        assert cached_debug == True  # Should still be cached value
        
        # Force reload
        reloaded_debug = manager.get('debug', force_reload=True)
        assert reloaded_debug == False  # Should be updated value
    
    def test_json_config_support(self, temp_config_dir):
        """Test loading JSON configuration files."""
        base_config_path = temp_config_dir / "base.yaml"
        
        # Create JSON environment config
        json_config = {
            'debug': True,
            'json_setting': 'test_value'
        }
        
        json_file = temp_config_dir / "environments" / "json_test.json"
        import json
        with open(json_file, 'w') as f:
            json.dump(json_config, f)
        
        manager = EnvironmentConfigManager(base_config_path=str(base_config_path))
        
        # Switch to json_test environment (would need to create base JSON file)
        # For now, just test that JSON files are recognized
        validation_results = manager.validate_configuration_files()
        assert str(json_file) in validation_results
        assert len(validation_results[str(json_file)]) == 0  # Should be valid
    
    def test_repr(self, temp_config_dir):
        """Test string representation of EnvironmentConfigManager."""
        base_config_path = temp_config_dir / "base.yaml"
        
        manager = EnvironmentConfigManager(
            base_config_path=str(base_config_path),
            environment="development"
        )
        
        repr_str = repr(manager)
        assert 'EnvironmentConfigManager' in repr_str
        assert 'development' in repr_str
        assert 'base.yaml' in repr_str


class TestEnvironmentConfigIntegration:
    """Integration tests for environment configuration with validation."""
    
    @pytest.fixture
    def temp_config_dir_with_validation(self):
        """Create temporary directory with configuration files for validation testing."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        
        # Create base config with validation-compatible structure
        base_config = {
            'simulation_id': 'integration-test',
            'max_steps': 1000,
            'environment': 'test',
            'width': 100,
            'height': 100,
            'learning_rate': 0.001,
            'gamma': 0.95,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'batch_size': 32,
            'memory_size': 10000
        }
        
        base_file = config_dir / "base.yaml"
        import yaml
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create test environment config
        test_env_config = {
            'learning_rate': 0.002,
            'max_steps': 500
        }
        
        env_dir = config_dir / "environments"
        env_dir.mkdir()
        
        test_env_file = env_dir / "test.yaml"
        with open(test_env_file, 'w') as f:
            yaml.dump(test_env_config, f)
        
        yield config_dir
        
        shutil.rmtree(temp_dir)
    
    def test_environment_config_with_validation(self, temp_config_dir_with_validation):
        """Test environment configuration with validation system."""
        from farm.core.config import DEFAULT_SIMULATION_SCHEMA
        
        base_config_path = temp_config_dir_with_validation / "base.yaml"
        
        manager = EnvironmentConfigManager(
            base_config_path=str(base_config_path),
            environment="test"
        )
        
        validator = ConfigurationValidator(DEFAULT_SIMULATION_SCHEMA)
        
        # Get configuration hierarchy
        config_hierarchy = manager.get_config_hierarchy()
        
        # Validate configuration
        result = validator.validate_config(config_hierarchy)
        
        assert result.is_valid == True
        assert len(result.errors) == 0
        
        # Test that environment overrides are applied
        assert config_hierarchy.get('learning_rate') == 0.002  # From environment
        assert config_hierarchy.get('max_steps') == 500  # From environment
        assert config_hierarchy.get('simulation_id') == 'integration-test'  # From global
    
    def test_invalid_environment_config_with_validation(self, temp_config_dir_with_validation):
        """Test validation of invalid environment configuration."""
        from farm.core.config import DEFAULT_SIMULATION_SCHEMA
        
        base_config_path = temp_config_dir_with_validation / "base.yaml"
        
        # Create invalid environment config
        invalid_env_config = {
            'learning_rate': 2.0,  # Above maximum
            'max_steps': -1,       # Below minimum
            'width': 5             # Below minimum
        }
        
        env_file = temp_config_dir_with_validation / "environments" / "invalid.yaml"
        import yaml
        with open(env_file, 'w') as f:
            yaml.dump(invalid_env_config, f)
        
        manager = EnvironmentConfigManager(
            base_config_path=str(base_config_path),
            environment="invalid"
        )
        
        validator = ConfigurationValidator(DEFAULT_SIMULATION_SCHEMA)
        
        # Get configuration hierarchy
        config_hierarchy = manager.get_config_hierarchy()
        
        # Validate configuration
        result = validator.validate_config(config_hierarchy)
        
        assert result.is_valid == False
        assert len(result.errors) > 0
        
        # Check that errors are related to invalid values
        error_messages = ' '.join(result.errors)
        assert 'learning_rate' in error_messages
        assert 'max_steps' in error_messages
        assert 'width' in error_messages


if __name__ == '__main__':
    pytest.main([__file__])
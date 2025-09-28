"""
Unit tests for the hierarchical configuration system.

This module tests the core functionality of the HierarchicalConfig class,
including hierarchical lookup, validation, and configuration management.
"""

import pytest
from unittest.mock import patch
import logging

from farm.core.config import (
    HierarchicalConfig,
    ConfigurationValidator,
    ValidationResult,
    ValidationException,
    ConfigurationError,
    DEFAULT_SIMULATION_SCHEMA
)


class TestHierarchicalConfig:
    """Test cases for HierarchicalConfig class."""
    
    def test_initialization(self):
        """Test HierarchicalConfig initialization."""
        config = HierarchicalConfig()
        
        assert config.global_config == {}
        assert config.environment_config == {}
        assert config.agent_config == {}
    
    def test_initialization_with_data(self):
        """Test HierarchicalConfig initialization with data."""
        global_config = {'debug': False, 'timeout': 30}
        environment_config = {'debug': True}
        agent_config = {'timeout': 60}
        
        config = HierarchicalConfig(
            global_config=global_config,
            environment_config=environment_config,
            agent_config=agent_config
        )
        
        assert config.global_config == global_config
        assert config.environment_config == environment_config
        assert config.agent_config == agent_config
    
    def test_hierarchical_lookup_priority(self):
        """Test that hierarchical lookup follows correct priority order."""
        config = HierarchicalConfig(
            global_config={'debug': False, 'timeout': 30, 'host': 'localhost'},
            environment_config={'debug': True, 'timeout': 45},
            agent_config={'timeout': 60}
        )
        
        # Agent config has highest priority
        assert config.get('timeout') == 60
        
        # Environment config has medium priority
        assert config.get('debug') == True
        
        # Global config has lowest priority
        assert config.get('host') == 'localhost'
        
        # Default value when key not found
        assert config.get('missing_key', 'default') == 'default'
        assert config.get('missing_key') is None
    
    def test_nested_lookup(self):
        """Test nested configuration lookup using dot notation."""
        config = HierarchicalConfig(
            global_config={
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'ssl': False
                }
            },
            environment_config={
                'database': {
                    'host': 'prod-db',
                    'ssl': True
                }
            }
        )
        
        # Environment config overrides global for nested values
        assert config.get_nested('database.host') == 'prod-db'
        
        # Global config value when not overridden
        assert config.get_nested('database.port') == 5432
        
        # Environment config overrides global
        assert config.get_nested('database.ssl') == True
        
        # Default value when key not found
        assert config.get_nested('database.missing', 'default') == 'default'
        assert config.get_nested('database.missing') is None
    
    def test_nested_lookup_with_custom_separator(self):
        """Test nested lookup with custom separator."""
        config = HierarchicalConfig(
            global_config={
                'redis': {
                    'connection': {
                        'host': 'localhost',
                        'port': 6379
                    }
                }
            }
        )
        
        assert config.get_nested('redis:connection:host', separator=':') == 'localhost'
        assert config.get_nested('redis:connection:port', separator=':') == 6379
    
    def test_set_configuration(self):
        """Test setting configuration values in different layers."""
        config = HierarchicalConfig()
        
        # Set in global layer
        config.set('debug', False, 'global')
        assert config.global_config['debug'] == False
        
        # Set in environment layer
        config.set('debug', True, 'environment')
        assert config.environment_config['debug'] == True
        
        # Set in agent layer
        config.set('timeout', 60, 'agent')
        assert config.agent_config['timeout'] == 60
        
        # Test invalid layer
        with pytest.raises(ValueError, match="Invalid layer 'invalid'"):
            config.set('key', 'value', 'invalid')
    
    def test_set_nested_configuration(self):
        """Test setting nested configuration values."""
        config = HierarchicalConfig()
        
        # Set nested value
        config.set_nested('database.host', 'localhost', 'environment')
        assert config.environment_config['database']['host'] == 'localhost'
        
        # Set another nested value in same structure
        config.set_nested('database.port', 5432, 'environment')
        assert config.environment_config['database']['port'] == 5432
        
        # Test with custom separator
        config.set_nested('redis:host', 'redis-server', 'agent', separator=':')
        assert config.agent_config['redis']['host'] == 'redis-server'
    
    def test_has_key(self):
        """Test checking if configuration key exists."""
        config = HierarchicalConfig(
            global_config={'global_key': 'value'},
            environment_config={'env_key': 'value'},
            agent_config={'agent_key': 'value'}
        )
        
        assert config.has('global_key') == True
        assert config.has('env_key') == True
        assert config.has('agent_key') == True
        assert config.has('missing_key') == False
    
    def test_has_nested_key(self):
        """Test checking if nested configuration key exists."""
        config = HierarchicalConfig(
            global_config={
                'database': {
                    'host': 'localhost'
                }
            }
        )
        
        assert config.has_nested('database.host') == True
        assert config.has_nested('database.port') == False
        assert config.has_nested('missing.nested') == False
    
    def test_get_all_keys(self):
        """Test getting all configuration keys."""
        config = HierarchicalConfig(
            global_config={'key1': 'value1', 'key2': 'value2'},
            environment_config={'key2': 'value2_override', 'key3': 'value3'},
            agent_config={'key3': 'value3_override', 'key4': 'value4'}
        )
        
        all_keys = config.get_all_keys()
        expected_keys = ['key1', 'key2', 'key3', 'key4']
        
        assert sorted(all_keys) == expected_keys
    
    def test_get_effective_config(self):
        """Test getting effective configuration with all overrides applied."""
        config = HierarchicalConfig(
            global_config={'debug': False, 'timeout': 30, 'host': 'localhost'},
            environment_config={'debug': True, 'timeout': 45},
            agent_config={'timeout': 60}
        )
        
        effective = config.get_effective_config()
        
        # Should have all keys with highest priority values
        assert effective['debug'] == True  # From environment
        assert effective['timeout'] == 60  # From agent
        assert effective['host'] == 'localhost'  # From global
    
    def test_validation_with_required_keys(self):
        """Test configuration validation with required keys."""
        config = HierarchicalConfig(
            global_config={'simulation_id': 'test-123', 'max_steps': 1000},
            environment_config={'environment': 'test'}
        )
        
        # Should not raise exception
        config.validate(['simulation_id', 'max_steps', 'environment'])
        
        # Should raise exception for missing required key
        with pytest.raises(ValidationException) as exc_info:
            config.validate(['simulation_id', 'max_steps', 'environment', 'missing_key'])
        
        assert 'missing_key' in str(exc_info.value)
    
    def test_validation_with_default_required_keys(self):
        """Test configuration validation with default required keys."""
        config = HierarchicalConfig(
            global_config={'simulation_id': 'test-123', 'max_steps': 1000},
            environment_config={'environment': 'test'}
        )
        
        # Should not raise exception
        config.validate()
        
        # Should raise exception for missing required key
        with pytest.raises(ValidationException) as exc_info:
            config = HierarchicalConfig()
            config.validate()
        
        assert 'simulation_id' in str(exc_info.value)
    
    def test_merge_configurations(self):
        """Test merging two HierarchicalConfig instances."""
        config1 = HierarchicalConfig(
            global_config={'key1': 'value1', 'key2': 'value2'},
            environment_config={'key2': 'value2_override'}
        )
        
        config2 = HierarchicalConfig(
            global_config={'key3': 'value3'},
            environment_config={'key2': 'value2_override2'},
            agent_config={'key4': 'value4'}
        )
        
        merged = config1.merge(config2)
        
        # Should have all keys from both configs
        assert merged.global_config['key1'] == 'value1'
        assert merged.global_config['key2'] == 'value2'
        assert merged.global_config['key3'] == 'value3'
        
        # Environment config should have overrides from both
        assert merged.environment_config['key2'] == 'value2_override2'
        
        # Agent config should have values from config2
        assert merged.agent_config['key4'] == 'value4'
    
    def test_copy_configuration(self):
        """Test creating a deep copy of configuration."""
        config = HierarchicalConfig(
            global_config={'key1': 'value1'},
            environment_config={'key2': 'value2'},
            agent_config={'key3': 'value3'}
        )
        
        copied = config.copy()
        
        # Should have same values
        assert copied.global_config == config.global_config
        assert copied.environment_config == config.environment_config
        assert copied.agent_config == config.agent_config
        
        # Should be independent objects
        copied.global_config['key1'] = 'modified'
        assert config.global_config['key1'] == 'value1'  # Original unchanged
    
    def test_clear_layer(self):
        """Test clearing configuration layers."""
        config = HierarchicalConfig(
            global_config={'key1': 'value1'},
            environment_config={'key2': 'value2'},
            agent_config={'key3': 'value3'}
        )
        
        # Clear environment layer
        config.clear_layer('environment')
        assert config.environment_config == {}
        assert config.global_config == {'key1': 'value1'}
        assert config.agent_config == {'key3': 'value3'}
        
        # Test invalid layer
        with pytest.raises(ValueError, match="Invalid layer 'invalid'"):
            config.clear_layer('invalid')
    
    def test_to_dict_and_from_dict(self):
        """Test converting configuration to/from dictionary."""
        original_config = HierarchicalConfig(
            global_config={'key1': 'value1'},
            environment_config={'key2': 'value2'},
            agent_config={'key3': 'value3'}
        )
        
        # Convert to dict
        config_dict = original_config.to_dict()
        expected_dict = {
            'global': {'key1': 'value1'},
            'environment': {'key2': 'value2'},
            'agent': {'key3': 'value3'}
        }
        assert config_dict == expected_dict
        
        # Convert back from dict
        restored_config = HierarchicalConfig.from_dict(config_dict)
        assert restored_config.global_config == original_config.global_config
        assert restored_config.environment_config == original_config.environment_config
        assert restored_config.agent_config == original_config.agent_config
    
    def test_repr_and_len(self):
        """Test string representation and length methods."""
        config = HierarchicalConfig(
            global_config={'key1': 'value1', 'key2': 'value2'},
            environment_config={'key3': 'value3'},
            agent_config={'key4': 'value4'}
        )
        
        # Test __repr__
        repr_str = repr(config)
        assert 'HierarchicalConfig' in repr_str
        assert 'global_keys=2' in repr_str
        assert 'environment_keys=1' in repr_str
        assert 'agent_keys=1' in repr_str
        
        # Test __len__
        assert len(config) == 4  # Total unique keys
    
    def test_contains_operator(self):
        """Test 'in' operator for configuration keys."""
        config = HierarchicalConfig(
            global_config={'global_key': 'value'},
            environment_config={'env_key': 'value'},
            agent_config={'agent_key': 'value'}
        )
        
        assert 'global_key' in config
        assert 'env_key' in config
        assert 'agent_key' in config
        assert 'missing_key' not in config


class TestConfigurationValidator:
    """Test cases for ConfigurationValidator class."""
    
    def test_initialization(self):
        """Test ConfigurationValidator initialization."""
        validator = ConfigurationValidator()
        assert validator.schema == {}
        assert validator.field_validators == {}
    
    def test_initialization_with_schema(self):
        """Test ConfigurationValidator initialization with schema."""
        schema = {
            'required': ['field1'],
            'fields': {
                'field1': {'type': str, 'required': True},
                'field2': {'type': int, 'min': 0, 'max': 100}
            }
        }
        
        validator = ConfigurationValidator(schema)
        assert validator.schema == schema
        assert 'field1' in validator.field_validators
        assert 'field2' in validator.field_validators
    
    def test_validate_required_fields(self):
        """Test validation of required fields."""
        schema = {
            'required': ['field1', 'field2']
        }
        
        validator = ConfigurationValidator(schema)
        
        # Valid configuration
        config = HierarchicalConfig(
            global_config={'field1': 'value1', 'field2': 'value2'}
        )
        result = validator.validate_config(config)
        assert result.is_valid == True
        assert len(result.errors) == 0
        
        # Missing required field
        config = HierarchicalConfig(
            global_config={'field1': 'value1'}
        )
        result = validator.validate_config(config)
        assert result.is_valid == False
        assert len(result.errors) == 1
        assert 'field2' in result.errors[0]
    
    def test_validate_field_types(self):
        """Test validation of field types."""
        schema = {
            'fields': {
                'string_field': {'type': str},
                'int_field': {'type': int},
                'float_field': {'type': float}
            }
        }
        
        validator = ConfigurationValidator(schema)
        
        # Valid types
        config = HierarchicalConfig(
            global_config={
                'string_field': 'hello',
                'int_field': 42,
                'float_field': 3.14
            }
        )
        result = validator.validate_config(config)
        assert result.is_valid == True
        
        # Invalid types
        config = HierarchicalConfig(
            global_config={
                'string_field': 123,  # Should be string
                'int_field': 'hello',  # Should be int
                'float_field': 'world'  # Should be float
            }
        )
        result = validator.validate_config(config)
        assert result.is_valid == False
        assert len(result.errors) == 3
    
    def test_validate_numeric_constraints(self):
        """Test validation of numeric constraints."""
        schema = {
            'fields': {
                'positive_int': {'type': int, 'min': 0},
                'bounded_float': {'type': float, 'min': 0.0, 'max': 1.0}
            }
        }
        
        validator = ConfigurationValidator(schema)
        
        # Valid constraints
        config = HierarchicalConfig(
            global_config={
                'positive_int': 5,
                'bounded_float': 0.5
            }
        )
        result = validator.validate_config(config)
        assert result.is_valid == True
        
        # Invalid constraints
        config = HierarchicalConfig(
            global_config={
                'positive_int': -1,  # Below minimum
                'bounded_float': 1.5  # Above maximum
            }
        )
        result = validator.validate_config(config)
        assert result.is_valid == False
        assert len(result.errors) == 2
    
    def test_validate_string_constraints(self):
        """Test validation of string constraints."""
        schema = {
            'fields': {
                'short_string': {'type': str, 'min_length': 3, 'max_length': 10},
                'pattern_string': {'type': str, 'pattern': r'^[a-z]+$'}
            }
        }
        
        validator = ConfigurationValidator(schema)
        
        # Valid constraints
        config = HierarchicalConfig(
            global_config={
                'short_string': 'hello',
                'pattern_string': 'abc'
            }
        )
        result = validator.validate_config(config)
        assert result.is_valid == True
        
        # Invalid constraints
        config = HierarchicalConfig(
            global_config={
                'short_string': 'hi',  # Too short
                'pattern_string': 'ABC123'  # Doesn't match pattern
            }
        )
        result = validator.validate_config(config)
        assert result.is_valid == False
        assert len(result.errors) == 2
    
    def test_validate_choices(self):
        """Test validation of choice constraints."""
        schema = {
            'fields': {
                'choice_field': {'type': str, 'choices': ['option1', 'option2', 'option3']}
            }
        }
        
        validator = ConfigurationValidator(schema)
        
        # Valid choice
        config = HierarchicalConfig(
            global_config={'choice_field': 'option2'}
        )
        result = validator.validate_config(config)
        assert result.is_valid == True
        
        # Invalid choice
        config = HierarchicalConfig(
            global_config={'choice_field': 'invalid_option'}
        )
        result = validator.validate_config(config)
        assert result.is_valid == False
        assert len(result.errors) == 1
    
    def test_validate_dependencies(self):
        """Test validation of field dependencies."""
        schema = {
            'dependencies': {
                'field1': {
                    'field2': {'equals': 'required_value'}
                }
            }
        }
        
        validator = ConfigurationValidator(schema)
        
        # Valid dependency
        config = HierarchicalConfig(
            global_config={
                'field1': 'value1',
                'field2': 'required_value'
            }
        )
        result = validator.validate_config(config)
        assert result.is_valid == True
        
        # Invalid dependency
        config = HierarchicalConfig(
            global_config={
                'field1': 'value1',
                'field2': 'wrong_value'
            }
        )
        result = validator.validate_config(config)
        assert result.is_valid == False
        assert len(result.errors) == 1
    
    def test_add_and_remove_field_validators(self):
        """Test adding and removing field validators."""
        validator = ConfigurationValidator()
        
        # Add field validator
        validator.add_field_validator('test_field', {'type': str, 'required': True})
        assert 'test_field' in validator.field_validators
        
        # Remove field validator
        validator.remove_field_validator('test_field')
        assert 'test_field' not in validator.field_validators
    
    def test_default_simulation_schema(self):
        """Test the default simulation schema."""
        validator = ConfigurationValidator(DEFAULT_SIMULATION_SCHEMA)
        
        # Valid simulation configuration
        config = HierarchicalConfig(
            global_config={
                'simulation_id': 'test-123',
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
        )
        
        result = validator.validate_config(config)
        assert result.is_valid == True
        
        # Invalid configuration
        config = HierarchicalConfig(
            global_config={
                'simulation_id': '',  # Too short
                'max_steps': -1,  # Below minimum
                'width': 5,  # Below minimum
                'learning_rate': 2.0,  # Above maximum
                'epsilon_min': 1.5  # Above epsilon_start
            }
        )
        
        result = validator.validate_config(config)
        assert result.is_valid == False
        assert len(result.errors) > 0


class TestValidationResult:
    """Test cases for ValidationResult class."""
    
    def test_initialization(self):
        """Test ValidationResult initialization."""
        result = ValidationResult()
        
        assert result.errors == []
        assert result.warnings == []
        assert result.is_valid == True
    
    def test_initialization_with_data(self):
        """Test ValidationResult initialization with data."""
        result = ValidationResult(
            errors=['error1', 'error2'],
            warnings=['warning1'],
            is_valid=False
        )
        
        assert result.errors == ['error1', 'error2']
        assert result.warnings == ['warning1']
        assert result.is_valid == False
    
    def test_add_error(self):
        """Test adding validation errors."""
        result = ValidationResult()
        
        result.add_error('test error')
        assert result.errors == ['test error']
        assert result.is_valid == False
    
    def test_add_warning(self):
        """Test adding validation warnings."""
        result = ValidationResult()
        
        result.add_warning('test warning')
        assert result.warnings == ['test warning']
        assert result.is_valid == True  # Warnings don't affect validity
    
    def test_merge_results(self):
        """Test merging validation results."""
        result1 = ValidationResult(
            errors=['error1'],
            warnings=['warning1']
        )
        
        result2 = ValidationResult(
            errors=['error2'],
            warnings=['warning2']
        )
        
        merged = result1.merge(result2)
        
        assert merged.errors == ['error1', 'error2']
        assert merged.warnings == ['warning1', 'warning2']
        assert merged.is_valid == False  # Has errors


if __name__ == '__main__':
    pytest.main([__file__])
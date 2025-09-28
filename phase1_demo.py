#!/usr/bin/env python3
"""
Phase 1 Demonstration: Hierarchical Configuration Framework

This script demonstrates the core functionality of the hierarchical configuration
system implemented in Phase 1, including:
- Hierarchical configuration with inheritance
- Runtime configuration validation
- Configuration management operations
"""

import sys
import os
sys.path.append('/workspace')

from farm.core.config import (
    HierarchicalConfig, 
    ConfigurationValidator, 
    ValidationResult,
    DEFAULT_SIMULATION_SCHEMA
)


def demo_hierarchical_lookup():
    """Demonstrate hierarchical configuration lookup."""
    print("=" * 60)
    print("DEMO: Hierarchical Configuration Lookup")
    print("=" * 60)
    
    # Create a hierarchical configuration
    config = HierarchicalConfig(
        global_config={
            'debug': False,
            'timeout': 30,
            'host': 'localhost',
            'database': {
                'host': 'db-local',
                'port': 5432
            }
        },
        environment_config={
            'debug': True,
            'timeout': 45,
            'database': {
                'host': 'db-staging'
            }
        },
        agent_config={
            'timeout': 60,
            'database': {
                'port': 3306
            }
        }
    )
    
    print("Configuration layers:")
    print(f"  Global: {config.global_config}")
    print(f"  Environment: {config.environment_config}")
    print(f"  Agent: {config.agent_config}")
    print()
    
    print("Hierarchical lookup results:")
    print(f"  debug: {config.get('debug')} (from environment - highest priority)")
    print(f"  timeout: {config.get('timeout')} (from agent - highest priority)")
    print(f"  host: {config.get('host')} (from global - lowest priority)")
    print(f"  database.host: {config.get_nested('database.host')} (from environment)")
    print(f"  database.port: {config.get_nested('database.port')} (from agent)")
    print(f"  missing_key: {config.get('missing_key', 'default_value')} (default)")
    print()


def demo_configuration_management():
    """Demonstrate configuration management operations."""
    print("=" * 60)
    print("DEMO: Configuration Management Operations")
    print("=" * 60)
    
    config = HierarchicalConfig()
    
    # Set values in different layers
    config.set('global_setting', 'global_value', 'global')
    config.set('env_setting', 'env_value', 'environment')
    config.set('agent_setting', 'agent_value', 'agent')
    
    # Set nested values
    config.set_nested('database.host', 'localhost', 'environment')
    config.set_nested('database.port', 5432, 'environment')
    config.set_nested('redis.host', 'redis-server', 'agent')
    
    print("After setting values:")
    print(f"  global_setting: {config.get('global_setting')}")
    print(f"  env_setting: {config.get('env_setting')}")
    print(f"  agent_setting: {config.get('agent_setting')}")
    print(f"  database.host: {config.get_nested('database.host')}")
    print(f"  database.port: {config.get_nested('database.port')}")
    print(f"  redis.host: {config.get_nested('redis.host')}")
    print()
    
    # Check if keys exist
    print("Key existence checks:")
    print(f"  'global_setting' exists: {config.has('global_setting')}")
    print(f"  'missing_key' exists: {config.has('missing_key')}")
    print(f"  'database.host' exists: {config.has_nested('database.host')}")
    print()
    
    # Get all keys
    print(f"All configuration keys: {config.get_all_keys()}")
    print()
    
    # Get effective configuration
    effective = config.get_effective_config()
    print("Effective configuration (all overrides applied):")
    for key, value in effective.items():
        print(f"  {key}: {value}")
    print()


def demo_validation_system():
    """Demonstrate configuration validation."""
    print("=" * 60)
    print("DEMO: Configuration Validation System")
    print("=" * 60)
    
    # Create validator with default simulation schema
    validator = ConfigurationValidator(DEFAULT_SIMULATION_SCHEMA)
    
    # Valid configuration
    print("Testing VALID configuration:")
    valid_config = HierarchicalConfig(
        global_config={
            'simulation_id': 'demo-simulation-123',
            'max_steps': 1000,
            'environment': 'demo',
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
    
    result = validator.validate_config(valid_config)
    print(f"  Validation result: valid={result.is_valid}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    print()
    
    # Invalid configuration
    print("Testing INVALID configuration:")
    invalid_config = HierarchicalConfig(
        global_config={
            'simulation_id': '',  # Too short
            'max_steps': -1,  # Below minimum
            'width': 5,  # Below minimum
            'learning_rate': 2.0,  # Above maximum
            'epsilon_min': 1.5,  # Above epsilon_start
            'batch_size': 'invalid'  # Wrong type
        }
    )
    
    result = validator.validate_config(invalid_config)
    print(f"  Validation result: valid={result.is_valid}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Warnings: {len(result.warnings)}")
    print()
    
    print("Sample validation errors:")
    for i, error in enumerate(result.errors[:5], 1):
        print(f"  {i}. {error}")
    if len(result.errors) > 5:
        print(f"  ... and {len(result.errors) - 5} more errors")
    print()


def demo_configuration_operations():
    """Demonstrate advanced configuration operations."""
    print("=" * 60)
    print("DEMO: Advanced Configuration Operations")
    print("=" * 60)
    
    # Create two configurations
    config1 = HierarchicalConfig(
        global_config={'setting1': 'value1', 'setting2': 'value2'},
        environment_config={'setting2': 'value2_override'}
    )
    
    config2 = HierarchicalConfig(
        global_config={'setting3': 'value3'},
        environment_config={'setting2': 'value2_override2'},
        agent_config={'setting4': 'value4'}
    )
    
    print("Configuration 1:")
    print(f"  Global: {config1.global_config}")
    print(f"  Environment: {config1.environment_config}")
    print()
    
    print("Configuration 2:")
    print(f"  Global: {config2.global_config}")
    print(f"  Environment: {config2.environment_config}")
    print(f"  Agent: {config2.agent_config}")
    print()
    
    # Merge configurations
    merged = config1.merge(config2)
    print("Merged configuration:")
    print(f"  Global: {merged.global_config}")
    print(f"  Environment: {merged.environment_config}")
    print(f"  Agent: {merged.agent_config}")
    print()
    
    # Copy configuration
    copied = merged.copy()
    copied.set('new_setting', 'new_value', 'agent')
    
    print("After modifying copy:")
    print(f"  Original agent keys: {list(merged.agent_config.keys())}")
    print(f"  Copy agent keys: {list(copied.agent_config.keys())}")
    print()
    
    # Clear layer
    print("Clearing environment layer:")
    merged.clear_layer('environment')
    print(f"  Environment config after clear: {merged.environment_config}")
    print()


def demo_custom_validation():
    """Demonstrate custom validation rules."""
    print("=" * 60)
    print("DEMO: Custom Validation Rules")
    print("=" * 60)
    
    # Create custom schema
    custom_schema = {
        'required': ['name', 'age'],
        'fields': {
            'name': {
                'type': str,
                'required': True,
                'min_length': 2,
                'max_length': 50
            },
            'age': {
                'type': int,
                'required': True,
                'min': 0,
                'max': 150
            },
            'email': {
                'type': str,
                'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            },
            'scores': {
                'type': list,
                'min_items': 1,
                'max_items': 10,
                'item_type': int
            }
        },
        'cross_constraints': [
            {
                'fields': ['name', 'age'],
                'condition': 'all_present',
                'message': 'Both name and age must be specified'
            }
        ]
    }
    
    validator = ConfigurationValidator(custom_schema)
    
    # Test valid configuration
    print("Testing valid custom configuration:")
    valid_config = HierarchicalConfig(
        global_config={
            'name': 'John Doe',
            'age': 30,
            'email': 'john.doe@example.com',
            'scores': [85, 92, 78, 96]
        }
    )
    
    result = validator.validate_config(valid_config)
    print(f"  Valid: {result.is_valid}, Errors: {len(result.errors)}")
    print()
    
    # Test invalid configuration
    print("Testing invalid custom configuration:")
    invalid_config = HierarchicalConfig(
        global_config={
            'name': 'A',  # Too short
            'age': -5,  # Below minimum
            'email': 'invalid-email',  # Invalid format
            'scores': ['not', 'numbers']  # Wrong item types
        }
    )
    
    result = validator.validate_config(invalid_config)
    print(f"  Valid: {result.is_valid}, Errors: {len(result.errors)}")
    print("  Sample errors:")
    for error in result.errors[:3]:
        print(f"    - {error}")
    print()


def main():
    """Run all demonstrations."""
    print("PHASE 1: HIERARCHICAL CONFIGURATION FRAMEWORK")
    print("=" * 60)
    print("This demonstration shows the core functionality implemented")
    print("in Phase 1 of the hierarchical configuration system.")
    print()
    
    demo_hierarchical_lookup()
    demo_configuration_management()
    demo_validation_system()
    demo_configuration_operations()
    demo_custom_validation()
    
    print("=" * 60)
    print("PHASE 1 IMPLEMENTATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Key Features Implemented:")
    print("✓ HierarchicalConfig class with inheritance-based lookup")
    print("✓ Configuration validation system with schema support")
    print("✓ Comprehensive exception handling")
    print("✓ Configuration management operations (set, get, merge, copy)")
    print("✓ Nested configuration support with dot notation")
    print("✓ Custom validation rules and cross-field constraints")
    print("✓ Default simulation schema for common use cases")
    print()
    print("Next Steps (Phase 2):")
    print("- Environment-specific configuration overrides")
    print("- Configuration file loading and management")
    print("- EnvironmentConfigManager implementation")


if __name__ == '__main__':
    main()
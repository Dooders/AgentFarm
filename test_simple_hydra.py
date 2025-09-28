#!/usr/bin/env python3
"""
Test script for simplified Hydra configuration system.
This script tests the simplified Hydra-based configuration management.
"""

import sys
import os
sys.path.append('/workspace')

from farm.core.config_hydra_simple import create_simple_hydra_config_manager

def test_basic_config_loading():
    """Test basic configuration loading."""
    print("Testing basic configuration loading...")
    
    # Create config manager
    config_manager = create_simple_hydra_config_manager(
        config_dir="/workspace/config_hydra/conf",
        environment="development",
        agent="system_agent"
    )
    
    # Test getting configuration
    config = config_manager.get_config()
    print(f"Config type: {type(config)}")
    print(f"Max steps: {config_manager.get('max_steps')}")
    print(f"Environment: {config_manager.get('environment')}")
    print(f"Debug mode: {config_manager.get('debug')}")
    
    return True

def test_environment_switching():
    """Test switching between environments."""
    print("\nTesting environment switching...")
    
    config_manager = create_simple_hydra_config_manager(
        config_dir="/workspace/config_hydra/conf",
        environment="development"
    )
    
    # Test development environment
    print(f"Development max_steps: {config_manager.get('max_steps')}")
    print(f"Development debug: {config_manager.get('debug')}")
    
    # Switch to production
    config_manager.update_environment("production")
    print(f"Production max_steps: {config_manager.get('max_steps')}")
    print(f"Production debug: {config_manager.get('debug')}")
    
    # Switch to staging
    config_manager.update_environment("staging")
    print(f"Staging max_steps: {config_manager.get('max_steps')}")
    print(f"Staging debug: {config_manager.get('debug')}")
    
    return True

def test_agent_switching():
    """Test switching between agent types."""
    print("\nTesting agent switching...")
    
    config_manager = create_simple_hydra_config_manager(
        config_dir="/workspace/config_hydra/conf",
        environment="development",
        agent="system_agent"
    )
    
    # Test system agent
    print(f"System agent share_weight: {config_manager.get('agent_parameters.SystemAgent.share_weight')}")
    print(f"System agent attack_weight: {config_manager.get('agent_parameters.SystemAgent.attack_weight')}")
    
    # Switch to independent agent
    config_manager.update_agent("independent_agent")
    print(f"Independent agent share_weight: {config_manager.get('agent_parameters.IndependentAgent.share_weight')}")
    print(f"Independent agent attack_weight: {config_manager.get('agent_parameters.IndependentAgent.attack_weight')}")
    
    # Switch to control agent
    config_manager.update_agent("control_agent")
    print(f"Control agent share_weight: {config_manager.get('agent_parameters.ControlAgent.share_weight')}")
    print(f"Control agent attack_weight: {config_manager.get('agent_parameters.ControlAgent.attack_weight')}")
    
    return True

def test_overrides():
    """Test configuration overrides."""
    print("\nTesting configuration overrides...")
    
    config_manager = create_simple_hydra_config_manager(
        config_dir="/workspace/config_hydra/conf",
        environment="development",
        overrides=["max_steps=200", "debug=false"]
    )
    
    print(f"Max steps with override: {config_manager.get('max_steps')}")
    print(f"Debug with override: {config_manager.get('debug')}")
    
    # Add more overrides
    config_manager.add_override("max_population=100")
    print(f"Max population after adding override: {config_manager.get('max_population')}")
    
    return True

def test_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    config_manager = create_simple_hydra_config_manager(
        config_dir="/workspace/config_hydra/conf",
        environment="development"
    )
    
    errors = config_manager.validate_configuration()
    print(f"Validation errors: {errors}")
    
    if not errors:
        print("Configuration validation passed!")
    else:
        print("Configuration validation failed!")
    
    return len(errors) == 0

def test_summary():
    """Test configuration summary."""
    print("\nTesting configuration summary...")
    
    config_manager = create_simple_hydra_config_manager(
        config_dir="/workspace/config_hydra/conf",
        environment="development",
        agent="system_agent"
    )
    
    summary = config_manager.get_configuration_summary()
    print(f"Environment: {summary['environment']}")
    print(f"Agent: {summary['agent']}")
    print(f"Available environments: {summary['available_environments']}")
    print(f"Available agents: {summary['available_agents']}")
    print(f"Overrides: {summary['overrides']}")
    print(f"Config keys count: {len(summary['config_keys'])}")
    
    return True

def test_to_dict():
    """Test converting configuration to dictionary."""
    print("\nTesting configuration to dictionary conversion...")
    
    config_manager = create_simple_hydra_config_manager(
        config_dir="/workspace/config_hydra/conf",
        environment="development"
    )
    
    config_dict = config_manager.to_dict()
    print(f"Config dict type: {type(config_dict)}")
    print(f"Config dict keys: {list(config_dict.keys())[:5]}...")  # Show first 5 keys
    
    return True

def main():
    """Run all tests."""
    print("Testing Simplified Hydra Configuration System")
    print("=" * 50)
    
    tests = [
        test_basic_config_loading,
        test_environment_switching,
        test_agent_switching,
        test_overrides,
        test_validation,
        test_summary,
        test_to_dict
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✓ {test.__name__} passed")
            else:
                failed += 1
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    
    if failed == 0:
        print("All tests passed! 🎉")
        return 0
    else:
        print("Some tests failed! ❌")
        return 1

if __name__ == "__main__":
    sys.exit(main())
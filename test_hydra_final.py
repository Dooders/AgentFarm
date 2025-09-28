#!/usr/bin/env python3
"""
Final test script for Hydra configuration system.
This script tests the complete Hydra functionality.
"""

import sys
import os
sys.path.append('/workspace')

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

def test_basic_hydra():
    """Test basic Hydra functionality."""
    print("Testing basic Hydra functionality...")
    
    try:
        # Initialize Hydra
        with initialize_config_dir(
            config_dir="/workspace/config_hydra/conf",
            version_base=None
        ):
            # Compose configuration
            config = compose(
                config_name="config",
                overrides=["environment=development"]
            )
            
            print(f"Config type: {type(config)}")
            print(f"Max steps: {config.max_steps}")
            print(f"Environment: {config.environment}")
            print(f"Debug mode: {config.debug}")
            print(f"System agent share_weight: {config.agent_parameters.SystemAgent.share_weight}")
            
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_environment_switching():
    """Test switching between environments."""
    print("\nTesting environment switching...")
    
    try:
        with initialize_config_dir(
            config_dir="/workspace/config_hydra/conf",
            version_base=None
        ):
            # Test development
            config = compose(config_name="config", overrides=["environment=development"])
            print(f"Development max_steps: {config.max_steps}")
            print(f"Development debug: {config.debug}")
            
            # Test production
            config = compose(config_name="config", overrides=["environment=production"])
            print(f"Production max_steps: {config.max_steps}")
            print(f"Production debug: {config.debug}")
            
            # Test staging
            config = compose(config_name="config", overrides=["environment=staging"])
            print(f"Staging max_steps: {config.max_steps}")
            print(f"Staging debug: {config.debug}")
            
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_agent_switching():
    """Test switching between agent types."""
    print("\nTesting agent switching...")
    
    try:
        with initialize_config_dir(
            config_dir="/workspace/config_hydra/conf",
            version_base=None
        ):
            # Test system agent (default)
            config = compose(config_name="config", overrides=["environment=development"])
            print(f"System agent share_weight: {config.agent_parameters.SystemAgent.share_weight}")
            
            # Test independent agent by creating a new config with different defaults
            config = compose(config_name="config", overrides=["environment=development", "agents=independent_agent"])
            print(f"Independent agent share_weight: {config.agent_parameters.IndependentAgent.share_weight}")
            
            # Test control agent
            config = compose(config_name="config", overrides=["environment=development", "agents=control_agent"])
            print(f"Control agent share_weight: {config.agent_parameters.ControlAgent.share_weight}")
            
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_overrides():
    """Test configuration overrides."""
    print("\nTesting configuration overrides...")
    
    try:
        with initialize_config_dir(
            config_dir="/workspace/config_hydra/conf",
            version_base=None
        ):
            config = compose(
                config_name="config",
                overrides=["environment=development", "max_steps=200", "debug=false"]
            )
            
            print(f"Max steps with override: {config.max_steps}")
            print(f"Debug with override: {config.debug}")
            
            # Test nested overrides
            config = compose(
                config_name="config",
                overrides=["environment=development", "agent_parameters.SystemAgent.share_weight=0.5"]
            )
            
            print(f"System agent share_weight with override: {config.agent_parameters.SystemAgent.share_weight}")
            
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_configuration_structure():
    """Test the configuration structure."""
    print("\nTesting configuration structure...")
    
    try:
        with initialize_config_dir(
            config_dir="/workspace/config_hydra/conf",
            version_base=None
        ):
            config = compose(config_name="config", overrides=["environment=development"])
            
            # Test that all expected keys are present
            expected_keys = [
                'max_steps', 'environment', 'debug', 'width', 'height',
                'system_agents', 'independent_agents', 'control_agents',
                'agent_parameters', 'visualization', 'redis'
            ]
            
            missing_keys = []
            for key in expected_keys:
                if not OmegaConf.select(config, key):
                    missing_keys.append(key)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
                return False
            else:
                print("All expected configuration keys are present")
                return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Hydra Configuration System (Final)")
    print("=" * 50)
    
    tests = [
        test_basic_hydra,
        test_environment_switching,
        test_agent_switching,
        test_overrides,
        test_configuration_structure
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
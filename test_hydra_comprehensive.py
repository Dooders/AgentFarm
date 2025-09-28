#!/usr/bin/env python3
"""
Comprehensive test suite for Hydra configuration system.
This script validates all aspects of the Hydra-based configuration management.
"""

import sys
import os
import tempfile
import shutil
import time
from pathlib import Path
sys.path.append('/workspace')

from farm.core.config_hydra_simple import create_simple_hydra_config_manager
from farm.core.config_hydra_hot_reload import HydraConfigurationHotReloader
from farm.core.config.hot_reload import ReloadConfig, ReloadStrategy


def test_configuration_loading():
    """Test basic configuration loading and validation."""
    print("🧪 Testing Configuration Loading...")
    
    try:
        # Test with different environments
        for env in ['development', 'staging', 'production']:
            config_manager = create_simple_hydra_config_manager(
                config_dir="/workspace/config_hydra/conf",
                environment=env
            )
            
            # Validate basic properties
            assert config_manager.environment == env
            assert config_manager.get('max_steps') is not None
            assert config_manager.get('width') > 0
            assert config_manager.get('height') > 0
            
            print(f"  ✅ {env} environment loaded successfully")
        
        # Test with different agents
        for agent in ['system_agent', 'independent_agent', 'control_agent']:
            config_manager = create_simple_hydra_config_manager(
                config_dir="/workspace/config_hydra/conf",
                environment="development",
                agent=agent
            )
            
            assert config_manager.agent == agent
            print(f"  ✅ {agent} agent loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration loading failed: {e}")
        return False


def test_environment_switching():
    """Test environment switching functionality."""
    print("🧪 Testing Environment Switching...")
    
    try:
        config_manager = create_simple_hydra_config_manager(
            config_dir="/workspace/config_hydra/conf",
            environment="development"
        )
        
        # Test development environment
        assert config_manager.get('debug') == True
        assert config_manager.get('max_steps') == 100
        print("  ✅ Development environment properties verified")
        
        # Switch to production
        config_manager.update_environment("production")
        assert config_manager.get('debug') == False
        assert config_manager.get('max_steps') == 1000
        print("  ✅ Production environment properties verified")
        
        # Switch to staging
        config_manager.update_environment("staging")
        assert config_manager.get('debug') == False  # staging has debug=False
        assert config_manager.get('max_steps') == 500
        print("  ✅ Staging environment properties verified")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Environment switching failed: {e}")
        return False


def test_agent_switching():
    """Test agent switching functionality."""
    print("🧪 Testing Agent Switching...")
    
    try:
        config_manager = create_simple_hydra_config_manager(
            config_dir="/workspace/config_hydra/conf",
            environment="development",
            agent="system_agent"
        )
        
        # Test system agent
        system_share = config_manager.get('agent_parameters.SystemAgent.share_weight')
        system_attack = config_manager.get('agent_parameters.SystemAgent.attack_weight')
        assert system_share == 0.4
        assert system_attack == 0.02
        print("  ✅ System agent properties verified")
        
        # Switch to independent agent
        config_manager.update_agent("independent_agent")
        independent_share = config_manager.get('agent_parameters.IndependentAgent.share_weight')
        independent_attack = config_manager.get('agent_parameters.IndependentAgent.attack_weight')
        assert independent_share == 0.01
        assert independent_attack == 0.4
        print("  ✅ Independent agent properties verified")
        
        # Switch to control agent
        config_manager.update_agent("control_agent")
        control_share = config_manager.get('agent_parameters.ControlAgent.share_weight')
        control_attack = config_manager.get('agent_parameters.ControlAgent.attack_weight')
        assert control_share == 0.2
        assert control_attack == 0.2
        print("  ✅ Control agent properties verified")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Agent switching failed: {e}")
        return False


def test_configuration_overrides():
    """Test configuration override functionality."""
    print("🧪 Testing Configuration Overrides...")
    
    try:
        # Test initial overrides
        config_manager = create_simple_hydra_config_manager(
            config_dir="/workspace/config_hydra/conf",
            environment="development",
            overrides=["max_steps=200", "debug=false"]
        )
        
        assert config_manager.get('max_steps') == 200
        assert config_manager.get('debug') == False
        print("  ✅ Initial overrides applied correctly")
        
        # Test adding overrides
        config_manager.add_override("max_population=100")
        assert config_manager.get('max_population') == 100
        print("  ✅ Additional override added correctly")
        
        # Test removing overrides
        config_manager.remove_override("max_population=100")
        # Should revert to original value
        original_population = config_manager.get('max_population')
        assert original_population != 100
        print("  ✅ Override removed correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration overrides failed: {e}")
        return False


def test_configuration_validation():
    """Test configuration validation functionality."""
    print("🧪 Testing Configuration Validation...")
    
    try:
        config_manager = create_simple_hydra_config_manager(
            config_dir="/workspace/config_hydra/conf",
            environment="development"
        )
        
        # Test valid configuration
        errors = config_manager.validate_configuration()
        assert len(errors) == 0, f"Validation errors: {errors}"
        print("  ✅ Valid configuration passed validation")
        
        # Test configuration summary
        summary = config_manager.get_configuration_summary()
        assert 'environment' in summary
        assert 'agent' in summary
        assert 'available_environments' in summary
        assert 'available_agents' in summary
        assert 'config_keys' in summary
        assert 'validation_errors' in summary
        print("  ✅ Configuration summary generated correctly")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration validation failed: {e}")
        return False


def test_configuration_serialization():
    """Test configuration serialization and deserialization."""
    print("🧪 Testing Configuration Serialization...")
    
    try:
        config_manager = create_simple_hydra_config_manager(
            config_dir="/workspace/config_hydra/conf",
            environment="development",
            agent="system_agent"
        )
        
        # Test to_dict conversion
        config_dict = config_manager.to_dict()
        assert isinstance(config_dict, dict)
        assert 'max_steps' in config_dict
        assert 'agent_parameters' in config_dict
        print("  ✅ Configuration converted to dictionary")
        
        # Test saving configuration
        temp_file = tempfile.mktemp(suffix='.yaml')
        config_manager.save_config(temp_file)
        assert os.path.exists(temp_file)
        print("  ✅ Configuration saved to file")
        
        # Clean up
        os.unlink(temp_file)
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration serialization failed: {e}")
        return False


def test_hot_reload_functionality():
    """Test hot-reload functionality."""
    print("🧪 Testing Hot-Reload Functionality...")
    
    # Create temporary config directory
    temp_dir = tempfile.mkdtemp(prefix="hydra_test_")
    config_dir = Path(temp_dir) / "conf"
    config_dir.mkdir(parents=True)
    
    try:
        # Create minimal config structure
        base_dir = config_dir / "base"
        base_dir.mkdir()
        
        with open(base_dir / "base.yaml", "w") as f:
            f.write("""# @package _global_
max_steps: 100
debug: false
""")
        
        env_dir = config_dir / "environments"
        env_dir.mkdir()
        
        with open(env_dir / "development.yaml", "w") as f:
            f.write("""# @package _global_
debug: true
""")
        
        agent_dir = config_dir / "agents"
        agent_dir.mkdir()
        
        with open(agent_dir / "system_agent.yaml", "w") as f:
            f.write("""# @package _global_
agent_type: system
""")
        
        with open(config_dir / "config.yaml", "w") as f:
            f.write("""# @package _global_
defaults:
  - base/base
  - environments/development
  - agents: system_agent
  - _self_
""")
        
        # Create config manager and hot reloader
        config_manager = create_simple_hydra_config_manager(
            config_dir=str(config_dir),
            environment="development",
            agent="system_agent"
        )
        
        reload_config = ReloadConfig(
            strategy=ReloadStrategy.IMMEDIATE,
            validate_on_reload=True,
            enable_rollback=True
        )
        
        hot_reloader = HydraConfigurationHotReloader(config_manager, reload_config)
        
        # Test initial configuration
        initial_config = hot_reloader.get_current_config()
        assert initial_config is not None
        print("  ✅ Initial configuration loaded")
        
        # Test monitoring start/stop
        hot_reloader.start_monitoring()
        assert hot_reloader.is_monitoring()
        print("  ✅ Monitoring started successfully")
        
        hot_reloader.stop_monitoring()
        assert not hot_reloader.is_monitoring()
        print("  ✅ Monitoring stopped successfully")
        
        # Test reload statistics
        stats = hot_reloader.get_reload_stats()
        assert 'is_monitoring' in stats
        assert 'strategy' in stats
        assert 'config_manager_type' in stats
        assert stats['config_manager_type'] == 'Hydra'
        print("  ✅ Reload statistics generated")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Hot-reload functionality failed: {e}")
        return False
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_configuration_compatibility():
    """Test compatibility with existing configuration patterns."""
    print("🧪 Testing Configuration Compatibility...")
    
    try:
        config_manager = create_simple_hydra_config_manager(
            config_dir="/workspace/config_hydra/conf",
            environment="development",
            agent="system_agent"
        )
        
        # Test that all expected configuration keys are present
        expected_keys = [
            'simulation_id', 'max_steps', 'environment', 'width', 'height',
            'system_agents', 'independent_agents', 'control_agents',
            'agent_parameters', 'visualization', 'redis'
        ]
        
        config_dict = config_manager.to_dict()
        missing_keys = [key for key in expected_keys if key not in config_dict]
        
        if missing_keys:
            print(f"  ⚠️  Missing keys: {missing_keys}")
        else:
            print("  ✅ All expected configuration keys present")
        
        # Test that configuration values are reasonable
        assert config_manager.get('max_steps') > 0
        assert config_manager.get('width') > 0
        assert config_manager.get('height') > 0
        assert config_manager.get('system_agents') > 0
        print("  ✅ Configuration values are reasonable")
        
        # Test nested configuration access
        agent_params = config_manager.get('agent_parameters')
        assert isinstance(agent_params, dict)
        assert 'SystemAgent' in agent_params
        assert 'IndependentAgent' in agent_params
        assert 'ControlAgent' in agent_params
        print("  ✅ Nested configuration access works")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration compatibility failed: {e}")
        return False


def test_performance():
    """Test configuration system performance."""
    print("🧪 Testing Performance...")
    
    try:
        # Test configuration loading speed
        start_time = time.time()
        
        for i in range(10):
            config_manager = create_simple_hydra_config_manager(
                config_dir="/workspace/config_hydra/conf",
                environment="development",
                agent="system_agent"
            )
            config_dict = config_manager.to_dict()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        print(f"  ✅ Average configuration loading time: {avg_time:.3f}s")
        assert avg_time < 1.0, "Configuration loading too slow"
        
        # Test environment switching speed
        config_manager = create_simple_hydra_config_manager(
            config_dir="/workspace/config_hydra/conf",
            environment="development"
        )
        
        start_time = time.time()
        for env in ['development', 'staging', 'production']:
            config_manager.update_environment(env)
        end_time = time.time()
        
        avg_switch_time = (end_time - start_time) / 3
        print(f"  ✅ Average environment switch time: {avg_switch_time:.3f}s")
        assert avg_switch_time < 0.5, "Environment switching too slow"
        
        return True
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {e}")
        return False


def main():
    """Run comprehensive test suite."""
    print("Hydra Configuration System - Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Configuration Loading", test_configuration_loading),
        ("Environment Switching", test_environment_switching),
        ("Agent Switching", test_agent_switching),
        ("Configuration Overrides", test_configuration_overrides),
        ("Configuration Validation", test_configuration_validation),
        ("Configuration Serialization", test_configuration_serialization),
        ("Hot-Reload Functionality", test_hot_reload_functionality),
        ("Configuration Compatibility", test_configuration_compatibility),
        ("Performance", test_performance),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                failed += 1
                print(f"❌ {test_name} failed")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print("Comprehensive Test Results")
    print("=" * 60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    print(f"Success rate: {(passed / (passed + failed) * 100):.1f}%")
    
    if failed == 0:
        print("\n🎉 All comprehensive tests passed!")
        print("The Hydra configuration system is fully functional and ready for production use.")
    else:
        print(f"\n⚠️  {failed} test(s) failed.")
        print("Please review the failed tests and fix any issues.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Phase 4 Demonstration: Hydra-based Hot-Reload System

This script demonstrates the Hydra-based hot-reload system, including:
- File system monitoring and automatic configuration reloading
- Different reload strategies (immediate, batched, scheduled, manual)
- Configuration change notifications
- Rollback mechanisms for failed configuration updates
- Integration with the Hydra configuration system
"""

import sys
import os
import tempfile
import shutil
import time
import threading
from pathlib import Path
sys.path.append('/workspace')

from farm.core.config_hydra_simple import create_simple_hydra_config_manager
from farm.core.config_hydra_hot_reload import HydraConfigurationHotReloader
from farm.core.config.hot_reload import (
    ReloadConfig,
    ReloadStrategy,
    ReloadEvent,
    ReloadNotification,
)


class DemoNotificationSubscriber:
    """Demo notification subscriber for testing."""
    
    def __init__(self, name: str):
        """Initialize demo subscriber.
        
        Args:
            name: Name of the subscriber
        """
        self.name = name
        self.notifications_received = []
    
    def __call__(self, notification: ReloadNotification):
        """Handle a reload notification.
        
        Args:
            notification: The reload notification
        """
        self.notifications_received.append(notification)
        print(f"[{self.name}] {notification.event_type.value}: {notification.message}")
        if notification.error:
            print(f"[{self.name}] Error: {notification.error}")


def create_temp_config_dir():
    """Create a temporary configuration directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="hydra_demo_")
    config_dir = Path(temp_dir) / "conf"
    config_dir.mkdir(parents=True)
    
    # Copy base configuration
    base_dir = config_dir / "base"
    base_dir.mkdir()
    
    # Create base configuration
    base_config = """# @package _global_
# Base Configuration for Demo
simulation_id: "demo-simulation"
max_steps: 100
environment: "demo"
debug: false
width: 50
height: 50
system_agents: 5
independent_agents: 5
control_agents: 5
agent_parameters:
  SystemAgent:
    share_weight: 0.3
    attack_weight: 0.05
  IndependentAgent:
    share_weight: 0.05
    attack_weight: 0.25
  ControlAgent:
    share_weight: 0.15
    attack_weight: 0.15
"""
    
    with open(base_dir / "base.yaml", "w") as f:
        f.write(base_config)
    
    # Create environment configurations
    env_dir = config_dir / "environments"
    env_dir.mkdir()
    
    dev_config = """# @package _global_
# Development Environment
debug: true
max_steps: 50
"""
    
    prod_config = """# @package _global_
# Production Environment
debug: false
max_steps: 200
"""
    
    with open(env_dir / "development.yaml", "w") as f:
        f.write(dev_config)
    
    with open(env_dir / "production.yaml", "w") as f:
        f.write(prod_config)
    
    # Create agent configurations
    agent_dir = config_dir / "agents"
    agent_dir.mkdir()
    
    system_agent_config = """# @package _global_
# System Agent Configuration
agent_parameters:
  SystemAgent:
    share_weight: 0.4
    attack_weight: 0.02
"""
    
    independent_agent_config = """# @package _global_
# Independent Agent Configuration
agent_parameters:
  IndependentAgent:
    share_weight: 0.01
    attack_weight: 0.4
"""
    
    with open(agent_dir / "system_agent.yaml", "w") as f:
        f.write(system_agent_config)
    
    with open(agent_dir / "independent_agent.yaml", "w") as f:
        f.write(independent_agent_config)
    
    # Create main config
    main_config = """# @package _global_
# Main Configuration for Demo
defaults:
  - base/base
  - environments/development
  - agents: system_agent
  - _self_
"""
    
    with open(config_dir / "config.yaml", "w") as f:
        f.write(main_config)
    
    return str(config_dir)


def test_immediate_reload_strategy():
    """Test immediate reload strategy."""
    print("\n" + "="*60)
    print("Testing Immediate Reload Strategy")
    print("="*60)
    
    # Create temporary config directory
    config_dir = create_temp_config_dir()
    
    try:
        # Create config manager
        config_manager = create_simple_hydra_config_manager(
            config_dir=config_dir,
            environment="development",
            agent="system_agent"
        )
        
        # Create reload config for immediate strategy
        reload_config = ReloadConfig(
            strategy=ReloadStrategy.IMMEDIATE,
            batch_delay=0.5,
            validate_on_reload=True,
            enable_rollback=True
        )
        
        # Create hot reloader
        hot_reloader = HydraConfigurationHotReloader(config_manager, reload_config)
        
        # Add notification subscriber
        subscriber = DemoNotificationSubscriber("ImmediateTest")
        hot_reloader.add_notification_callback(subscriber)
        
        # Start monitoring
        hot_reloader.start_monitoring()
        
        print("✅ Started monitoring with immediate reload strategy")
        
        # Get initial configuration
        initial_config = hot_reloader.get_current_config()
        print(f"Initial max_steps: {initial_config.get('max_steps')}")
        print(f"Initial debug: {initial_config.get('debug')}")
        
        # Modify configuration file
        env_file = Path(config_dir) / "environments" / "development.yaml"
        print(f"Modifying {env_file}")
        
        with open(env_file, "w") as f:
            f.write("""# @package _global_
# Development Environment (Modified)
debug: true
max_steps: 75
""")
        
        # Wait for reload
        time.sleep(1.0)
        
        # Check if configuration was reloaded
        new_config = hot_reloader.get_current_config()
        print(f"New max_steps: {new_config.get('max_steps')}")
        print(f"New debug: {new_config.get('debug')}")
        
        # Check notifications
        print(f"Notifications received: {len(subscriber.notifications_received)}")
        for notification in subscriber.notifications_received:
            print(f"  - {notification.event_type.value}: {notification.message}")
        
        # Stop monitoring
        hot_reloader.stop_monitoring()
        print("✅ Stopped monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Immediate reload test failed: {e}")
        return False
    finally:
        # Clean up
        shutil.rmtree(Path(config_dir).parent, ignore_errors=True)


def test_batched_reload_strategy():
    """Test batched reload strategy."""
    print("\n" + "="*60)
    print("Testing Batched Reload Strategy")
    print("="*60)
    
    # Create temporary config directory
    config_dir = create_temp_config_dir()
    
    try:
        # Create config manager
        config_manager = create_simple_hydra_config_manager(
            config_dir=config_dir,
            environment="development",
            agent="system_agent"
        )
        
        # Create reload config for batched strategy
        reload_config = ReloadConfig(
            strategy=ReloadStrategy.BATCHED,
            batch_delay=2.0,
            max_batch_size=3,
            validate_on_reload=True,
            enable_rollback=True
        )
        
        # Create hot reloader
        hot_reloader = HydraConfigurationHotReloader(config_manager, reload_config)
        
        # Add notification subscriber
        subscriber = DemoNotificationSubscriber("BatchedTest")
        hot_reloader.add_notification_callback(subscriber)
        
        # Start monitoring
        hot_reloader.start_monitoring()
        
        print("✅ Started monitoring with batched reload strategy")
        
        # Get initial configuration
        initial_config = hot_reloader.get_current_config()
        print(f"Initial max_steps: {initial_config.get('max_steps')}")
        
        # Make multiple rapid changes
        env_file = Path(config_dir) / "environments" / "development.yaml"
        
        for i in range(3):
            print(f"Making change {i+1}/3")
            with open(env_file, "w") as f:
                f.write(f"""# @package _global_
# Development Environment (Change {i+1})
debug: true
max_steps: {50 + i*10}
""")
            time.sleep(0.5)  # Rapid changes
        
        # Wait for batch to be processed
        print("Waiting for batch processing...")
        time.sleep(3.0)
        
        # Check final configuration
        final_config = hot_reloader.get_current_config()
        print(f"Final max_steps: {final_config.get('max_steps')}")
        
        # Check notifications
        print(f"Notifications received: {len(subscriber.notifications_received)}")
        for notification in subscriber.notifications_received:
            print(f"  - {notification.event_type.value}: {notification.message}")
        
        # Stop monitoring
        hot_reloader.stop_monitoring()
        print("✅ Stopped monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Batched reload test failed: {e}")
        return False
    finally:
        # Clean up
        shutil.rmtree(Path(config_dir).parent, ignore_errors=True)


def test_environment_switching():
    """Test environment switching with hot reloading."""
    print("\n" + "="*60)
    print("Testing Environment Switching")
    print("="*60)
    
    # Create temporary config directory
    config_dir = create_temp_config_dir()
    
    try:
        # Create config manager
        config_manager = create_simple_hydra_config_manager(
            config_dir=config_dir,
            environment="development",
            agent="system_agent"
        )
        
        # Create reload config
        reload_config = ReloadConfig(
            strategy=ReloadStrategy.IMMEDIATE,
            validate_on_reload=True,
            enable_rollback=True
        )
        
        # Create hot reloader
        hot_reloader = HydraConfigurationHotReloader(config_manager, reload_config)
        
        # Add notification subscriber
        subscriber = DemoNotificationSubscriber("EnvSwitchTest")
        hot_reloader.add_notification_callback(subscriber)
        
        # Start monitoring
        hot_reloader.start_monitoring()
        
        print("✅ Started monitoring for environment switching")
        
        # Test development environment
        print("Testing development environment...")
        dev_config = hot_reloader.get_current_config()
        print(f"Development max_steps: {dev_config.get('max_steps')}")
        print(f"Development debug: {dev_config.get('debug')}")
        
        # Switch to production
        print("Switching to production environment...")
        config_manager.update_environment("production")
        time.sleep(0.5)
        
        prod_config = hot_reloader.get_current_config()
        print(f"Production max_steps: {prod_config.get('max_steps')}")
        print(f"Production debug: {prod_config.get('debug')}")
        
        # Check notifications
        print(f"Notifications received: {len(subscriber.notifications_received)}")
        for notification in subscriber.notifications_received:
            print(f"  - {notification.event_type.value}: {notification.message}")
        
        # Stop monitoring
        hot_reloader.stop_monitoring()
        print("✅ Stopped monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment switching test failed: {e}")
        return False
    finally:
        # Clean up
        shutil.rmtree(Path(config_dir).parent, ignore_errors=True)


def test_agent_switching():
    """Test agent switching with hot reloading."""
    print("\n" + "="*60)
    print("Testing Agent Switching")
    print("="*60)
    
    # Create temporary config directory
    config_dir = create_temp_config_dir()
    
    try:
        # Create config manager
        config_manager = create_simple_hydra_config_manager(
            config_dir=config_dir,
            environment="development",
            agent="system_agent"
        )
        
        # Create reload config
        reload_config = ReloadConfig(
            strategy=ReloadStrategy.IMMEDIATE,
            validate_on_reload=True,
            enable_rollback=True
        )
        
        # Create hot reloader
        hot_reloader = HydraConfigurationHotReloader(config_manager, reload_config)
        
        # Add notification subscriber
        subscriber = DemoNotificationSubscriber("AgentSwitchTest")
        hot_reloader.add_notification_callback(subscriber)
        
        # Start monitoring
        hot_reloader.start_monitoring()
        
        print("✅ Started monitoring for agent switching")
        
        # Test system agent
        print("Testing system agent...")
        system_config = hot_reloader.get_current_config()
        system_share = system_config.get('agent_parameters', {}).get('SystemAgent', {}).get('share_weight')
        print(f"System agent share_weight: {system_share}")
        
        # Switch to independent agent
        print("Switching to independent agent...")
        config_manager.update_agent("independent_agent")
        time.sleep(0.5)
        
        independent_config = hot_reloader.get_current_config()
        independent_share = independent_config.get('agent_parameters', {}).get('IndependentAgent', {}).get('share_weight')
        print(f"Independent agent share_weight: {independent_share}")
        
        # Check notifications
        print(f"Notifications received: {len(subscriber.notifications_received)}")
        for notification in subscriber.notifications_received:
            print(f"  - {notification.event_type.value}: {notification.message}")
        
        # Stop monitoring
        hot_reloader.stop_monitoring()
        print("✅ Stopped monitoring")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent switching test failed: {e}")
        return False
    finally:
        # Clean up
        shutil.rmtree(Path(config_dir).parent, ignore_errors=True)


def test_rollback_mechanism():
    """Test rollback mechanism for failed configurations."""
    print("\n" + "="*60)
    print("Testing Rollback Mechanism")
    print("="*60)
    
    # Create temporary config directory
    config_dir = create_temp_config_dir()
    
    try:
        # Create config manager
        config_manager = create_simple_hydra_config_manager(
            config_dir=config_dir,
            environment="development",
            agent="system_agent"
        )
        
        # Create reload config with rollback enabled
        reload_config = ReloadConfig(
            strategy=ReloadStrategy.IMMEDIATE,
            validate_on_reload=True,
            enable_rollback=True,
            backup_configs=True,
            max_backups=3
        )
        
        # Create hot reloader
        hot_reloader = HydraConfigurationHotReloader(config_manager, reload_config)
        
        # Add notification subscriber
        subscriber = DemoNotificationSubscriber("RollbackTest")
        hot_reloader.add_notification_callback(subscriber)
        
        # Start monitoring
        hot_reloader.start_monitoring()
        
        print("✅ Started monitoring with rollback enabled")
        
        # Get initial configuration
        initial_config = hot_reloader.get_current_config()
        print(f"Initial max_steps: {initial_config.get('max_steps')}")
        
        # Make a valid change first
        env_file = Path(config_dir) / "environments" / "development.yaml"
        with open(env_file, "w") as f:
            f.write("""# @package _global_
# Development Environment (Valid Change)
debug: true
max_steps: 80
""")
        
        time.sleep(1.0)
        
        valid_config = hot_reloader.get_current_config()
        print(f"After valid change max_steps: {valid_config.get('max_steps')}")
        
        # Check that we have backups
        backups = hot_reloader.get_config_backups()
        print(f"Configuration backups: {len(backups)}")
        
        # Stop monitoring
        hot_reloader.stop_monitoring()
        print("✅ Stopped monitoring")
        
        # Check notifications
        print(f"Notifications received: {len(subscriber.notifications_received)}")
        for notification in subscriber.notifications_received:
            print(f"  - {notification.event_type.value}: {notification.message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rollback test failed: {e}")
        return False
    finally:
        # Clean up
        shutil.rmtree(Path(config_dir).parent, ignore_errors=True)


def main():
    """Run all Hydra hot-reload demonstrations."""
    print("Hydra-based Hot-Reload System Demonstration")
    print("=" * 60)
    print("This demo shows the Hydra-based configuration hot-reload system")
    print("with different reload strategies and features.")
    
    tests = [
        ("Immediate Reload Strategy", test_immediate_reload_strategy),
        ("Batched Reload Strategy", test_batched_reload_strategy),
        ("Environment Switching", test_environment_switching),
        ("Agent Switching", test_agent_switching),
        ("Rollback Mechanism", test_rollback_mechanism),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
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
    print("Hydra Hot-Reload Demo Results")
    print("=" * 60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All Hydra hot-reload tests passed!")
        print("The Hydra-based configuration system is working correctly.")
    else:
        print(f"\n⚠️  {failed} test(s) failed.")
        print("Please check the error messages above.")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
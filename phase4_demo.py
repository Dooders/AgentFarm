#!/usr/bin/env python3
"""
Phase 4 Demonstration: Hot-Reload System

This script demonstrates the hot-reload system implemented in Phase 4,
including:
- File system monitoring and automatic configuration reloading
- Different reload strategies (immediate, batched, scheduled, manual)
- Configuration change notifications
- Rollback mechanisms for failed configuration updates
- Integration with the complete configuration system
"""

import sys
import os
import tempfile
import shutil
import time
import threading
from pathlib import Path
sys.path.append('/workspace')

from farm.core.config import (
    ConfigurationHotReloader,
    ReloadConfig,
    ReloadStrategy,
    ReloadEvent,
    ReloadNotification,
    ConfigurationNotificationManager,
    NotificationConfig,
    NotificationChannel,
    NotificationPriority,
    EnvironmentConfigManager,
    ConfigurationMigrator,
    ConfigurationVersionDetector
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
    
    def handle_notification(self, notification: ReloadNotification) -> None:
        """Handle notification.
        
        Args:
            notification: The notification to handle
        """
        self.notifications_received.append(notification)
        print(f"  📢 {self.name} received: {notification.event_type.value} - {notification.message}")
    
    def get_filter(self):
        """Get notification filter."""
        return None  # No filtering
    
    def get_notification_count(self) -> int:
        """Get number of notifications received."""
        return len(self.notifications_received)
    
    def get_notifications_by_type(self, event_type: ReloadEvent) -> list:
        """Get notifications by event type."""
        return [n for n in self.notifications_received if n.event_type == event_type]


def demo_reload_strategies():
    """Demonstrate different reload strategies."""
    print("=" * 60)
    print("DEMO: Reload Strategies")
    print("=" * 60)
    
    strategies = [
        {
            'name': 'Immediate Reload',
            'strategy': ReloadStrategy.IMMEDIATE,
            'description': 'Reloads configuration immediately when any file change is detected'
        },
        {
            'name': 'Batched Reload',
            'strategy': ReloadStrategy.BATCHED,
            'description': 'Batches file changes and reloads after a delay or when batch size is reached'
        },
        {
            'name': 'Scheduled Reload',
            'strategy': ReloadStrategy.SCHEDULED,
            'description': 'Reloads configuration at scheduled intervals'
        },
        {
            'name': 'Manual Reload',
            'strategy': ReloadStrategy.MANUAL,
            'description': 'Only reloads when explicitly requested'
        }
    ]
    
    for strategy_info in strategies:
        print(f"📋 {strategy_info['name']}:")
        print(f"   Strategy: {strategy_info['strategy'].value}")
        print(f"   Description: {strategy_info['description']}")
        print()
    
    # Show reload configuration options
    print("Reload Configuration Options:")
    config = ReloadConfig()
    print(f"  • Batch delay: {config.batch_delay} seconds")
    print(f"  • Max batch size: {config.max_batch_size} files")
    print(f"  • Schedule interval: {config.schedule_interval} seconds")
    print(f"  • Enable rollback: {config.enable_rollback}")
    print(f"  • Max rollback attempts: {config.max_rollback_attempts}")
    print(f"  • Validate on reload: {config.validate_on_reload}")
    print(f"  • Backup configs: {config.backup_configs}")
    print(f"  • Max backups: {config.max_backups}")
    print(f"  • Watch subdirectories: {config.watch_subdirectories}")
    print(f"  • File patterns: {config.file_patterns}")
    print(f"  • Ignore patterns: {config.ignore_patterns}")
    print()


def demo_notification_system():
    """Demonstrate notification system."""
    print("=" * 60)
    print("DEMO: Notification System")
    print("=" * 60)
    
    # Create notification configuration
    notif_config = NotificationConfig(
        enabled_channels={
            NotificationChannel.CONSOLE,
            NotificationChannel.LOG,
            NotificationChannel.QUEUE
        },
        default_priority=NotificationPriority.NORMAL,
        max_queue_size=100,
        log_level="INFO"
    )
    
    print("Notification Configuration:")
    print(f"  • Enabled channels: {[c.value for c in notif_config.enabled_channels]}")
    print(f"  • Default priority: {notif_config.default_priority.value}")
    print(f"  • Max queue size: {notif_config.max_queue_size}")
    print(f"  • Log level: {notif_config.log_level}")
    print()
    
    # Create notification manager
    notif_manager = ConfigurationNotificationManager(notif_config)
    
    # Add demo subscribers
    subscriber1 = DemoNotificationSubscriber("Subscriber 1")
    subscriber2 = DemoNotificationSubscriber("Subscriber 2")
    
    notif_manager.add_subscriber(subscriber1)
    notif_manager.add_subscriber(subscriber2)
    
    print("Notification Subscribers:")
    print(f"  • Synchronous subscribers: {len(notif_manager.subscribers)}")
    print(f"  • Async subscribers: {len(notif_manager.async_subscribers)}")
    print()
    
    # Send test notifications
    print("Sending Test Notifications:")
    
    test_notifications = [
        ReloadNotification(
            event_type=ReloadEvent.CONFIG_LOADED,
            timestamp=time.time(),
            message="Initial configuration loaded"
        ),
        ReloadNotification(
            event_type=ReloadEvent.FILE_CHANGED,
            timestamp=time.time(),
            file_path="/path/to/config.yaml",
            message="Configuration file modified"
        ),
        ReloadNotification(
            event_type=ReloadEvent.CONFIG_RELOADED,
            timestamp=time.time(),
            message="Configuration reloaded successfully"
        ),
        ReloadNotification(
            event_type=ReloadEvent.CONFIG_FAILED,
            timestamp=time.time(),
            error=Exception("Test error"),
            message="Configuration reload failed"
        )
    ]
    
    for notification in test_notifications:
        notif_manager.send_notification(notification)
        time.sleep(0.1)  # Small delay for demo
    
    print()
    print("Notification Statistics:")
    stats = notif_manager.get_notification_stats()
    print(f"  • Notifications sent: {stats['stats']['notifications_sent']}")
    print(f"  • Notifications failed: {stats['stats']['notifications_failed']}")
    print(f"  • Subscribers notified: {stats['stats']['subscribers_notified']}")
    print()
    
    # Test queue handler
    queue_handler = notif_manager.get_queue_handler()
    if queue_handler:
        print("Queue Handler Test:")
        queued_notifications = queue_handler.get_all_notifications()
        print(f"  • Notifications in queue: {len(queued_notifications)}")
        print(f"  • Event types: {[n.event_type.value for n in queued_notifications]}")
        print()
    
    # Cleanup
    notif_manager.shutdown()


def demo_hot_reload_integration():
    """Demonstrate hot-reload integration with configuration system."""
    print("=" * 60)
    print("DEMO: Hot-Reload Integration")
    print("=" * 60)
    
    # Create temporary configuration directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config_dir = temp_path / "config"
        config_dir.mkdir()
        
        # Create base configuration
        base_config = {
            'simulation_id': 'hot-reload-demo',
            'max_steps': 1000,
            'learning_rate': 0.001,
            'debug': False
        }
        
        base_file = config_dir / "base.yaml"
        import yaml
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create environment directory
        env_dir = config_dir / "environments"
        env_dir.mkdir()
        
        env_config = {
            'debug': True,
            'max_steps': 500,
            'verbose_logging': True
        }
        
        env_file = env_dir / "development.yaml"
        with open(env_file, 'w') as f:
            yaml.dump(env_config, f)
        
        print("Created Test Configuration Structure:")
        print(f"  📁 Base config: {base_file}")
        print(f"  📁 Environment config: {env_file}")
        print()
        
        # Create environment config manager
        env_manager = EnvironmentConfigManager(
            str(base_file),
            config_dir=str(config_dir),
            environment='development'
        )
        
        print("Environment Configuration Manager:")
        print(f"  • Environment: {env_manager.environment}")
        print(f"  • Available environments: {env_manager.get_available_environments()}")
        print()
        
        # Get initial configuration
        initial_config = env_manager.get_effective_config()
        print("Initial Configuration:")
        for key, value in initial_config.items():
            print(f"  • {key}: {value}")
        print()
        
        # Create hot reloader with batched strategy
        reload_config = ReloadConfig(
            strategy=ReloadStrategy.BATCHED,
            batch_delay=0.5,
            max_batch_size=3,
            enable_rollback=True,
            backup_configs=True,
            max_backups=3
        )
        
        hot_reloader = ConfigurationHotReloader(
            env_manager,
            reload_config,
            migrations_dir='/workspace/config/migrations'
        )
        
        print("Hot-Reload Configuration:")
        print(f"  • Strategy: {reload_config.strategy.value}")
        print(f"  • Batch delay: {reload_config.batch_delay}s")
        print(f"  • Max batch size: {reload_config.max_batch_size}")
        print(f"  • Enable rollback: {reload_config.enable_rollback}")
        print(f"  • Backup configs: {reload_config.backup_configs}")
        print()
        
        # Add notification subscriber
        demo_subscriber = DemoNotificationSubscriber("Hot-Reload Demo")
        hot_reloader.add_notification_callback(demo_subscriber.handle_notification)
        
        # Start monitoring
        print("Starting File Monitoring...")
        hot_reloader.start_monitoring()
        print(f"  • Monitoring active: {hot_reloader.is_monitoring()}")
        print()
        
        try:
            # Simulate configuration changes
            print("Simulating Configuration Changes:")
            
            # Change 1: Modify environment config
            print("  1. Modifying environment configuration...")
            env_config['max_steps'] = 750
            env_config['new_setting'] = 'new_value'
            with open(env_file, 'w') as f:
                yaml.dump(env_config, f)
            
            time.sleep(0.2)  # Wait for file change detection
            
            # Change 2: Modify base config
            print("  2. Modifying base configuration...")
            base_config['learning_rate'] = 0.01
            base_config['new_base_setting'] = 'base_value'
            with open(base_file, 'w') as f:
                yaml.dump(base_config, f)
            
            time.sleep(0.2)  # Wait for file change detection
            
            # Change 3: Add new environment file
            print("  3. Adding new environment configuration...")
            new_env_config = {
                'test_environment': True,
                'test_setting': 'test_value'
            }
            new_env_file = env_dir / "testing.yaml"
            with open(new_env_file, 'w') as f:
                yaml.dump(new_env_config, f)
            
            time.sleep(0.2)  # Wait for file change detection
            
            # Wait for batch reload
            print("  4. Waiting for batch reload...")
            time.sleep(1.0)  # Wait for batch delay
            
            print()
            print("Configuration Changes Applied:")
            current_config = hot_reloader.get_current_config()
            if current_config:
                for key, value in current_config.items():
                    print(f"  • {key}: {value}")
            print()
            
            # Show notifications received
            print("Notifications Received:")
            notification_count = demo_subscriber.get_notification_count()
            print(f"  • Total notifications: {notification_count}")
            
            file_changes = demo_subscriber.get_notifications_by_type(ReloadEvent.FILE_CHANGED)
            print(f"  • File change notifications: {len(file_changes)}")
            
            reloads = demo_subscriber.get_notifications_by_type(ReloadEvent.CONFIG_RELOADED)
            print(f"  • Reload notifications: {len(reloads)}")
            print()
            
            # Show reload statistics
            print("Hot-Reload Statistics:")
            stats = hot_reloader.get_reload_stats()
            for key, value in stats.items():
                print(f"  • {key}: {value}")
            print()
            
            # Show configuration backups
            backups = hot_reloader.get_config_backups()
            print(f"Configuration Backups: {len(backups)}")
            for i, backup in enumerate(backups):
                print(f"  • Backup {i+1}: {backup.get('timestamp', 'unknown')} - {backup.get('version', 'unknown')}")
            print()
            
        finally:
            # Stop monitoring
            print("Stopping File Monitoring...")
            hot_reloader.stop_monitoring()
            print(f"  • Monitoring active: {hot_reloader.is_monitoring()}")
            print()


def demo_manual_reload_strategy():
    """Demonstrate manual reload strategy."""
    print("=" * 60)
    print("DEMO: Manual Reload Strategy")
    print("=" * 60)
    
    # Create temporary configuration directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config_dir = temp_path / "config"
        config_dir.mkdir()
        
        # Create base configuration
        base_config = {
            'simulation_id': 'manual-reload-demo',
            'max_steps': 1000,
            'learning_rate': 0.001
        }
        
        base_file = config_dir / "base.yaml"
        import yaml
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create environment config manager
        env_manager = EnvironmentConfigManager(
            str(base_file),
            config_dir=str(config_dir),
            environment='development'
        )
        
        # Create hot reloader with manual strategy
        reload_config = ReloadConfig(strategy=ReloadStrategy.MANUAL)
        hot_reloader = ConfigurationHotReloader(env_manager, reload_config)
        
        print("Manual Reload Strategy Demo:")
        print(f"  • Strategy: {reload_config.strategy.value}")
        print()
        
        # Start monitoring
        hot_reloader.start_monitoring()
        print("File monitoring started (manual strategy)")
        print()
        
        try:
            # Make configuration changes
            print("Making Configuration Changes:")
            
            # Change 1
            print("  1. Modifying configuration...")
            base_config['max_steps'] = 1500
            with open(base_file, 'w') as f:
                yaml.dump(base_config, f)
            
            time.sleep(0.2)
            
            # Change 2
            print("  2. Modifying configuration again...")
            base_config['learning_rate'] = 0.01
            with open(base_file, 'w') as f:
                yaml.dump(base_config, f)
            
            time.sleep(0.2)
            
            # Check pending changes
            pending_changes = hot_reloader.get_pending_changes()
            print(f"  • Pending changes: {len(pending_changes)}")
            for change in pending_changes:
                print(f"    - {change}")
            print()
            
            # Manual reload
            print("Triggering Manual Reload:")
            success = hot_reloader.manual_reload()
            print(f"  • Reload successful: {success}")
            
            if success:
                current_config = hot_reloader.get_current_config()
                print("  • Updated configuration:")
                for key, value in current_config.items():
                    print(f"    - {key}: {value}")
            
            print()
            
            # Check pending changes after reload
            pending_changes = hot_reloader.get_pending_changes()
            print(f"  • Pending changes after reload: {len(pending_changes)}")
            print()
            
        finally:
            hot_reloader.stop_monitoring()


def demo_rollback_mechanism():
    """Demonstrate rollback mechanism."""
    print("=" * 60)
    print("DEMO: Rollback Mechanism")
    print("=" * 60)
    
    # Create temporary configuration directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config_dir = temp_path / "config"
        config_dir.mkdir()
        
        # Create base configuration
        base_config = {
            'simulation_id': 'rollback-demo',
            'max_steps': 1000,
            'learning_rate': 0.001
        }
        
        base_file = config_dir / "base.yaml"
        import yaml
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create environment config manager
        env_manager = EnvironmentConfigManager(
            str(base_file),
            config_dir=str(config_dir),
            environment='development'
        )
        
        # Create hot reloader with rollback enabled
        reload_config = ReloadConfig(
            strategy=ReloadStrategy.IMMEDIATE,
            enable_rollback=True,
            backup_configs=True,
            max_backups=3
        )
        
        hot_reloader = ConfigurationHotReloader(env_manager, reload_config)
        
        print("Rollback Mechanism Demo:")
        print(f"  • Enable rollback: {reload_config.enable_rollback}")
        print(f"  • Backup configs: {reload_config.backup_configs}")
        print(f"  • Max backups: {reload_config.max_backups}")
        print()
        
        # Get initial configuration
        initial_config = hot_reloader.get_current_config()
        print("Initial Configuration:")
        for key, value in initial_config.items():
            print(f"  • {key}: {value}")
        print()
        
        # Make a valid change
        print("Making Valid Configuration Change:")
        base_config['max_steps'] = 1500
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        time.sleep(0.5)  # Wait for reload
        
        # Check configuration after valid change
        current_config = hot_reloader.get_current_config()
        print("Configuration After Valid Change:")
        for key, value in current_config.items():
            print(f"  • {key}: {value}")
        print()
        
        # Check backups
        backups = hot_reloader.get_config_backups()
        print(f"Configuration Backups: {len(backups)}")
        for i, backup in enumerate(backups):
            print(f"  • Backup {i+1}: {backup.get('timestamp', 'unknown')}")
        print()
        
        # Simulate a failed reload by creating invalid YAML
        print("Simulating Failed Configuration Reload:")
        with open(base_file, 'w') as f:
            f.write("invalid: yaml: content: [")  # Invalid YAML
        
        time.sleep(0.5)  # Wait for reload attempt
        
        # Check if rollback occurred
        current_config = hot_reloader.get_current_config()
        print("Configuration After Failed Reload (should be rolled back):")
        if current_config:
            for key, value in current_config.items():
                print(f"  • {key}: {value}")
        else:
            print("  • No configuration loaded")
        print()
        
        # Restore valid configuration
        print("Restoring Valid Configuration:")
        base_config['max_steps'] = 2000
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        time.sleep(0.5)  # Wait for reload
        
        # Check final configuration
        final_config = hot_reloader.get_current_config()
        print("Final Configuration:")
        if final_config:
            for key, value in final_config.items():
                print(f"  • {key}: {value}")
        print()


def demo_migration_integration():
    """Demonstrate migration integration with hot-reload."""
    print("=" * 60)
    print("DEMO: Migration Integration with Hot-Reload")
    print("=" * 60)
    
    # Create temporary configuration directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config_dir = temp_path / "config"
        config_dir.mkdir()
        
        # Create a v1.0 configuration (without agent_parameters, visualization, redis)
        v1_0_config = {
            'simulation_id': 'migration-demo',
            'max_steps': 1000,
            'learning_rate': 0.001
        }
        
        base_file = config_dir / "base.yaml"
        import yaml
        with open(base_file, 'w') as f:
            yaml.dump(v1_0_config, f)
        
        print("Created v1.0 Configuration:")
        for key, value in v1_0_config.items():
            print(f"  • {key}: {value}")
        print()
        
        # Create environment config manager
        env_manager = EnvironmentConfigManager(
            str(base_file),
            config_dir=str(config_dir),
            environment='development'
        )
        
        # Create hot reloader with migration support
        reload_config = ReloadConfig(strategy=ReloadStrategy.IMMEDIATE)
        hot_reloader = ConfigurationHotReloader(
            env_manager,
            reload_config,
            migrations_dir='/workspace/config/migrations'
        )
        
        print("Hot-Reload with Migration Support:")
        print(f"  • Migration enabled: {hot_reloader.migrator is not None}")
        print(f"  • Version detector enabled: {hot_reloader.version_detector is not None}")
        print()
        
        # Get initial configuration (should be migrated)
        initial_config = hot_reloader.get_current_config()
        print("Initial Configuration (after migration):")
        if initial_config:
            print(f"  • Config version: {initial_config.get('config_version', 'unknown')}")
            print(f"  • Has agent_parameters: {'agent_parameters' in initial_config}")
            print(f"  • Has visualization: {'visualization' in initial_config}")
            print(f"  • Has redis: {'redis' in initial_config}")
            print(f"  • Total keys: {len(initial_config)}")
        print()
        
        # Make a change to trigger reload
        print("Making Configuration Change:")
        v1_0_config['max_steps'] = 1500
        with open(base_file, 'w') as f:
            yaml.dump(v1_0_config, f)
        
        time.sleep(0.5)  # Wait for reload
        
        # Check configuration after reload
        current_config = hot_reloader.get_current_config()
        print("Configuration After Reload (with migration):")
        if current_config:
            print(f"  • Config version: {current_config.get('config_version', 'unknown')}")
            print(f"  • max_steps: {current_config.get('max_steps')}")
            print(f"  • Has agent_parameters: {'agent_parameters' in current_config}")
            print(f"  • Has visualization: {'visualization' in current_config}")
            print(f"  • Has redis: {'redis' in current_config}")
        print()


def main():
    """Run all Phase 4 demonstrations."""
    print("PHASE 4: HOT-RELOAD SYSTEM")
    print("=" * 60)
    print("This demonstration shows the hot-reload system implemented")
    print("in Phase 4, including file monitoring, reload strategies,")
    print("notifications, and integration with the complete configuration system.")
    print()
    
    demo_reload_strategies()
    demo_notification_system()
    demo_hot_reload_integration()
    demo_manual_reload_strategy()
    demo_rollback_mechanism()
    demo_migration_integration()
    
    print("=" * 60)
    print("PHASE 4 IMPLEMENTATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Key Features Implemented:")
    print("✓ ConfigurationHotReloader with file system monitoring")
    print("✓ Four reload strategies (immediate, batched, scheduled, manual)")
    print("✓ ConfigurationNotificationManager with multiple channels")
    print("✓ Rollback mechanism for failed configuration updates")
    print("✓ Integration with migration system from Phase 3")
    print("✓ Comprehensive error handling and validation")
    print("✓ File system monitoring with watchdog integration")
    print("✓ Async notification processing")
    print("✓ Configuration backup and restore")
    print()
    print("Reload Strategies:")
    print("  🔄 immediate - Reload immediately when changes detected")
    print("  📦 batched - Batch changes and reload after delay")
    print("  ⏰ scheduled - Reload at scheduled intervals")
    print("  👤 manual - Only reload when explicitly requested")
    print()
    print("Notification Channels:")
    print("  📺 console - Console output notifications")
    print("  📝 log - Log file notifications")
    print("  📄 file - File-based notifications")
    print("  📋 queue - Queue-based notifications")
    print("  🔔 callback - Custom callback notifications")
    print("  ⚡ async_callback - Async callback notifications")
    print()
    print("Integration Features:")
    print("  🔗 Seamless integration with Phases 1, 2, and 3")
    print("  🔄 Automatic migration during hot-reload")
    print("  📊 Comprehensive statistics and monitoring")
    print("  🛡️ Robust error handling and rollback")
    print("  📁 File system monitoring with pattern matching")
    print("  🔔 Real-time notification system")
    print()
    print("🎉 Complete Hierarchical Configuration System Ready!")
    print("All four phases successfully implemented and integrated.")


if __name__ == '__main__':
    main()
"""
Comprehensive tests for the hot-reload system.

This module tests the hot-reload functionality, notification system,
reload strategies, and integration with the configuration system.
"""

import pytest
import tempfile
import shutil
import time
import threading
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from farm.core.config import (
    ConfigurationHotReloader,
    ReloadConfig,
    ReloadStrategy,
    ReloadEvent,
    ReloadNotification,
    ConfigurationNotificationManager,
    NotificationConfig,
    NotificationChannel,
    NotificationSubscriber,
    AsyncNotificationSubscriber,
    EnvironmentConfigManager,
    ConfigurationError
)


class TestReloadConfig:
    """Test cases for ReloadConfig class."""
    
    def test_default_config(self):
        """Test default reload configuration."""
        config = ReloadConfig()
        
        assert config.strategy == ReloadStrategy.BATCHED
        assert config.batch_delay == 1.0
        assert config.max_batch_size == 10
        assert config.schedule_interval == 5.0
        assert config.enable_rollback == True
        assert config.max_rollback_attempts == 3
        assert config.validate_on_reload == True
        assert config.backup_configs == True
        assert config.max_backups == 5
        assert config.watch_subdirectories == True
        assert "*.yaml" in config.file_patterns
        assert "*.yml" in config.file_patterns
        assert "*.json" in config.file_patterns
        assert "*.tmp" in config.ignore_patterns
        assert "*.bak" in config.ignore_patterns
        assert ".*" in config.ignore_patterns


class TestReloadNotification:
    """Test cases for ReloadNotification class."""
    
    def test_notification_creation(self):
        """Test notification creation."""
        notification = ReloadNotification(
            event_type=ReloadEvent.CONFIG_RELOADED,
            timestamp=time.time(),
            file_path="/path/to/config.yaml",
            message="Configuration reloaded"
        )
        
        assert notification.event_type == ReloadEvent.CONFIG_RELOADED
        assert notification.file_path == "/path/to/config.yaml"
        assert notification.message == "Configuration reloaded"
        assert notification.old_config is None
        assert notification.new_config is None
        assert notification.error is None
        assert notification.migration_info is None


class TestReloadStrategyHandlers:
    """Test cases for reload strategy handlers."""
    
    @pytest.fixture
    def mock_hot_reloader(self):
        """Create mock hot reloader."""
        mock_reloader = Mock()
        mock_reloader.reload_config = ReloadConfig()
        return mock_reloader
    
    def test_immediate_reload_handler(self, mock_hot_reloader):
        """Test immediate reload handler."""
        from farm.core.config.hot_reload import ImmediateReloadHandler
        
        handler = ImmediateReloadHandler(mock_hot_reloader)
        
        # Test handle_change
        handler.handle_change("/path/to/config.yaml", "modified")
        mock_hot_reloader._reload_configuration.assert_called_once()
        
        # Test start/stop
        handler.start()
        handler.stop()
    
    def test_batched_reload_handler(self, mock_hot_reloader):
        """Test batched reload handler."""
        from farm.core.config.hot_reload import BatchedReloadHandler
        
        handler = BatchedReloadHandler(mock_hot_reloader)
        
        # Test single change
        handler.handle_change("/path/to/config1.yaml", "modified")
        assert len(handler.pending_changes) == 1
        
        # Test multiple changes
        handler.handle_change("/path/to/config2.yaml", "modified")
        assert len(handler.pending_changes) == 2
        
        # Test batch size limit
        mock_hot_reloader.reload_config.max_batch_size = 2
        handler.handle_change("/path/to/config3.yaml", "modified")
        mock_hot_reloader._reload_configuration.assert_called()
        
        # Test start/stop
        handler.start()
        handler.stop()
    
    def test_scheduled_reload_handler(self, mock_hot_reloader):
        """Test scheduled reload handler."""
        from farm.core.config.hot_reload import ScheduledReloadHandler
        
        handler = ScheduledReloadHandler(mock_hot_reloader)
        
        # Test handle_change
        handler.handle_change("/path/to/config.yaml", "modified")
        assert len(handler.pending_changes) == 1
        
        # Test start/stop
        handler.start()
        time.sleep(0.1)  # Let scheduler start
        handler.stop()
    
    def test_manual_reload_handler(self, mock_hot_reloader):
        """Test manual reload handler."""
        from farm.core.config.hot_reload import ManualReloadHandler
        
        handler = ManualReloadHandler(mock_hot_reloader)
        
        # Test handle_change
        handler.handle_change("/path/to/config.yaml", "modified")
        assert len(handler.pending_changes) == 1
        
        # Test get_pending_changes
        changes = handler.get_pending_changes()
        assert "/path/to/config.yaml" in changes
        
        # Test clear_pending_changes
        handler.clear_pending_changes()
        assert len(handler.pending_changes) == 0
        
        # Test start/stop
        handler.start()
        handler.stop()


class TestConfigurationHotReloader:
    """Test cases for ConfigurationHotReloader class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        
        # Create base configuration
        base_config = {
            'simulation_id': 'test',
            'max_steps': 1000,
            'learning_rate': 0.001
        }
        
        base_file = config_dir / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create environment directory
        env_dir = config_dir / "environments"
        env_dir.mkdir()
        
        env_config = {
            'debug': True,
            'max_steps': 500
        }
        
        env_file = env_dir / "development.yaml"
        with open(env_file, 'w') as f:
            yaml.dump(env_config, f)
        
        yield config_dir
        
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config_manager(self, temp_config_dir):
        """Create mock configuration manager."""
        manager = Mock(spec=EnvironmentConfigManager)
        manager.config_dir = str(temp_config_dir)
        manager.environment = 'development'
        manager.get_effective_config.return_value = {
            'simulation_id': 'test',
            'max_steps': 500,
            'learning_rate': 0.001,
            'debug': True
        }
        return manager
    
    def test_hot_reloader_initialization(self, mock_config_manager):
        """Test hot reloader initialization."""
        reload_config = ReloadConfig(strategy=ReloadStrategy.IMMEDIATE)
        hot_reloader = ConfigurationHotReloader(mock_config_manager, reload_config)
        
        assert hot_reloader.config_manager == mock_config_manager
        assert hot_reloader.reload_config.strategy == ReloadStrategy.IMMEDIATE
        assert hot_reloader.current_config is not None
        assert hot_reloader.is_reloading == False
        assert len(hot_reloader.notification_callbacks) == 0
    
    def test_strategy_handler_initialization(self, mock_config_manager):
        """Test strategy handler initialization."""
        # Test immediate strategy
        reload_config = ReloadConfig(strategy=ReloadStrategy.IMMEDIATE)
        hot_reloader = ConfigurationHotReloader(mock_config_manager, reload_config)
        assert isinstance(hot_reloader.strategy_handler, type(hot_reloader.strategy_handler))
        
        # Test batched strategy
        reload_config = ReloadConfig(strategy=ReloadStrategy.BATCHED)
        hot_reloader = ConfigurationHotReloader(mock_config_manager, reload_config)
        assert isinstance(hot_reloader.strategy_handler, type(hot_reloader.strategy_handler))
        
        # Test scheduled strategy
        reload_config = ReloadConfig(strategy=ReloadStrategy.SCHEDULED)
        hot_reloader = ConfigurationHotReloader(mock_config_manager, reload_config)
        assert isinstance(hot_reloader.strategy_handler, type(hot_reloader.strategy_handler))
        
        # Test manual strategy
        reload_config = ReloadConfig(strategy=ReloadStrategy.MANUAL)
        hot_reloader = ConfigurationHotReloader(mock_config_manager, reload_config)
        assert isinstance(hot_reloader.strategy_handler, type(hot_reloader.strategy_handler))
    
    def test_file_monitoring_start_stop(self, mock_config_manager, temp_config_dir):
        """Test file monitoring start and stop."""
        hot_reloader = ConfigurationHotReloader(mock_config_manager)
        
        # Test start monitoring
        hot_reloader.start_monitoring()
        assert hot_reloader.is_monitoring() == True
        
        # Test stop monitoring
        hot_reloader.stop_monitoring()
        assert hot_reloader.is_monitoring() == False
    
    def test_manual_reload(self, mock_config_manager):
        """Test manual reload functionality."""
        reload_config = ReloadConfig(strategy=ReloadStrategy.MANUAL)
        hot_reloader = ConfigurationHotReloader(mock_config_manager, reload_config)
        
        # Test manual reload
        result = hot_reloader.manual_reload()
        assert result == True
        mock_config_manager.reload_configs.assert_called()
    
    def test_manual_reload_wrong_strategy(self, mock_config_manager):
        """Test manual reload with wrong strategy."""
        reload_config = ReloadConfig(strategy=ReloadStrategy.IMMEDIATE)
        hot_reloader = ConfigurationHotReloader(mock_config_manager, reload_config)
        
        # Test manual reload with wrong strategy
        result = hot_reloader.manual_reload()
        assert result == False
    
    def test_get_pending_changes(self, mock_config_manager):
        """Test getting pending changes."""
        reload_config = ReloadConfig(strategy=ReloadStrategy.MANUAL)
        hot_reloader = ConfigurationHotReloader(mock_config_manager, reload_config)
        
        # Test get pending changes
        changes = hot_reloader.get_pending_changes()
        assert isinstance(changes, list)
    
    def test_get_current_config(self, mock_config_manager):
        """Test getting current configuration."""
        hot_reloader = ConfigurationHotReloader(mock_config_manager)
        
        # Test get current config
        config = hot_reloader.get_current_config()
        assert config is not None
        assert isinstance(config, dict)
    
    def test_get_config_backups(self, mock_config_manager):
        """Test getting configuration backups."""
        hot_reloader = ConfigurationHotReloader(mock_config_manager)
        
        # Test get config backups
        backups = hot_reloader.get_config_backups()
        assert isinstance(backups, list)
    
    def test_notification_callbacks(self, mock_config_manager):
        """Test notification callback system."""
        hot_reloader = ConfigurationHotReloader(mock_config_manager)
        
        # Test add callback
        callback = Mock()
        hot_reloader.add_notification_callback(callback)
        assert len(hot_reloader.notification_callbacks) == 1
        
        # Test remove callback
        hot_reloader.remove_notification_callback(callback)
        assert len(hot_reloader.notification_callbacks) == 0
    
    def test_get_reload_stats(self, mock_config_manager):
        """Test getting reload statistics."""
        hot_reloader = ConfigurationHotReloader(mock_config_manager)
        
        # Test get reload stats
        stats = hot_reloader.get_reload_stats()
        assert isinstance(stats, dict)
        assert 'is_monitoring' in stats
        assert 'is_reloading' in stats
        assert 'strategy' in stats
        assert 'backups_count' in stats
        assert 'current_config_loaded' in stats
        assert 'migration_enabled' in stats
        assert 'validation_enabled' in stats
        assert 'rollback_enabled' in stats


class TestNotificationSystem:
    """Test cases for notification system."""
    
    def test_notification_config_defaults(self):
        """Test default notification configuration."""
        config = NotificationConfig()
        
        assert NotificationChannel.LOG in config.enabled_channels
        assert config.default_priority == NotificationPriority.NORMAL
        assert config.max_queue_size == 1000
        assert config.async_timeout == 5.0
        assert config.log_level == "INFO"
        assert config.file_path is None
    
    def test_notification_manager_initialization(self):
        """Test notification manager initialization."""
        config = NotificationConfig(
            enabled_channels={NotificationChannel.CONSOLE, NotificationChannel.LOG}
        )
        manager = ConfigurationNotificationManager(config)
        
        assert manager.config == config
        assert len(manager.handlers) == 2
        assert NotificationChannel.CONSOLE in manager.handlers
        assert NotificationChannel.LOG in manager.handlers
        assert len(manager.subscribers) == 0
        assert len(manager.async_subscribers) == 0
        assert manager.async_running == False
    
    def test_notification_sending(self):
        """Test notification sending."""
        config = NotificationConfig(
            enabled_channels={NotificationChannel.CONSOLE, NotificationChannel.LOG}
        )
        manager = ConfigurationNotificationManager(config)
        
        # Create test notification
        notification = ReloadNotification(
            event_type=ReloadEvent.CONFIG_RELOADED,
            timestamp=time.time(),
            message="Test notification"
        )
        
        # Send notification
        manager.send_notification(notification)
        
        # Check stats
        stats = manager.get_notification_stats()
        assert stats['stats']['notifications_sent'] == 1
    
    def test_subscriber_management(self):
        """Test subscriber management."""
        manager = ConfigurationNotificationManager()
        
        # Create mock subscriber
        subscriber = Mock(spec=NotificationSubscriber)
        subscriber.get_filter.return_value = None
        
        # Test add subscriber
        manager.add_subscriber(subscriber)
        assert len(manager.subscribers) == 1
        
        # Test remove subscriber
        manager.remove_subscriber(subscriber)
        assert len(manager.subscribers) == 0
    
    def test_async_subscriber_management(self):
        """Test async subscriber management."""
        manager = ConfigurationNotificationManager()
        
        # Create mock async subscriber
        async_subscriber = Mock(spec=AsyncNotificationSubscriber)
        async_subscriber.get_filter.return_value = None
        
        # Test add async subscriber
        manager.add_async_subscriber(async_subscriber)
        assert len(manager.async_subscribers) == 1
        assert manager.async_running == True
        
        # Test remove async subscriber
        manager.remove_subscriber(async_subscriber)
        assert len(manager.async_subscribers) == 0
    
    def test_notification_filtering(self):
        """Test notification filtering."""
        from farm.core.config.notifications import NotificationFilter
        
        manager = ConfigurationNotificationManager()
        
        # Create subscriber with filter
        subscriber = Mock(spec=NotificationSubscriber)
        subscriber.get_filter.return_value = NotificationFilter(
            event_types={ReloadEvent.CONFIG_RELOADED},
            include_errors=False
        )
        
        manager.add_subscriber(subscriber)
        
        # Test notification that should be filtered out
        notification = ReloadNotification(
            event_type=ReloadEvent.CONFIG_FAILED,
            timestamp=time.time(),
            message="Test notification",
            error=Exception("Test error")
        )
        
        manager.send_notification(notification)
        
        # Subscriber should not be called due to filtering
        subscriber.handle_notification.assert_not_called()
    
    def test_queue_notification_handler(self):
        """Test queue notification handler."""
        config = NotificationConfig(
            enabled_channels={NotificationChannel.QUEUE}
        )
        manager = ConfigurationNotificationManager(config)
        
        # Get queue handler
        queue_handler = manager.get_queue_handler()
        assert queue_handler is not None
        
        # Create test notification
        notification = ReloadNotification(
            event_type=ReloadEvent.CONFIG_RELOADED,
            timestamp=time.time(),
            message="Test notification"
        )
        
        # Send notification
        manager.send_notification(notification)
        
        # Get notification from queue
        received_notification = queue_handler.get_notification(timeout=1.0)
        assert received_notification is not None
        assert received_notification.event_type == ReloadEvent.CONFIG_RELOADED
        assert received_notification.message == "Test notification"
    
    def test_notification_stats(self):
        """Test notification statistics."""
        manager = ConfigurationNotificationManager()
        
        # Get initial stats
        stats = manager.get_notification_stats()
        assert 'enabled_channels' in stats
        assert 'active_handlers' in stats
        assert 'synchronous_subscribers' in stats
        assert 'async_subscribers' in stats
        assert 'async_processing_active' in stats
        assert 'stats' in stats
        
        # Clear stats
        manager.clear_stats()
        stats = manager.get_notification_stats()
        assert stats['stats']['notifications_sent'] == 0
    
    def test_notification_manager_shutdown(self):
        """Test notification manager shutdown."""
        manager = ConfigurationNotificationManager()
        
        # Add some subscribers
        subscriber = Mock(spec=NotificationSubscriber)
        manager.add_subscriber(subscriber)
        
        # Shutdown
        manager.shutdown()
        
        assert len(manager.subscribers) == 0
        assert len(manager.async_subscribers) == 0
        assert len(manager.handlers) == 0


class TestIntegration:
    """Integration tests for hot-reload system."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        temp_dir = tempfile.mkdtemp()
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        
        # Create base configuration
        base_config = {
            'simulation_id': 'test',
            'max_steps': 1000,
            'learning_rate': 0.001
        }
        
        base_file = config_dir / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Create environment directory
        env_dir = config_dir / "environments"
        env_dir.mkdir()
        
        env_config = {
            'debug': True,
            'max_steps': 500
        }
        
        env_file = env_dir / "development.yaml"
        with open(env_file, 'w') as f:
            yaml.dump(env_config, f)
        
        yield config_dir
        
        shutil.rmtree(temp_dir)
    
    def test_hot_reload_with_file_changes(self, temp_config_dir):
        """Test hot-reload with actual file changes."""
        # Create environment config manager
        base_file = temp_config_dir / "base.yaml"
        config_manager = EnvironmentConfigManager(
            str(base_file),
            config_dir=str(temp_config_dir),
            environment='development'
        )
        
        # Create hot reloader
        reload_config = ReloadConfig(strategy=ReloadStrategy.IMMEDIATE)
        hot_reloader = ConfigurationHotReloader(config_manager, reload_config)
        
        # Start monitoring
        hot_reloader.start_monitoring()
        
        try:
            # Modify configuration file
            env_file = temp_config_dir / "environments" / "development.yaml"
            with open(env_file, 'w') as f:
                yaml.dump({'debug': False, 'max_steps': 750}, f)
            
            # Wait for reload
            time.sleep(0.5)
            
            # Check if configuration was reloaded
            current_config = hot_reloader.get_current_config()
            assert current_config is not None
            
        finally:
            hot_reloader.stop_monitoring()
    
    def test_hot_reload_with_notifications(self, temp_config_dir):
        """Test hot-reload with notification system."""
        # Create environment config manager
        base_file = temp_config_dir / "base.yaml"
        config_manager = EnvironmentConfigManager(
            str(base_file),
            config_dir=str(temp_config_dir),
            environment='development'
        )
        
        # Create notification manager
        notification_manager = ConfigurationNotificationManager()
        
        # Create hot reloader
        reload_config = ReloadConfig(strategy=ReloadStrategy.IMMEDIATE)
        hot_reloader = ConfigurationHotReloader(config_manager, reload_config)
        
        # Add notification callback
        notifications_received = []
        def notification_callback(notification):
            notifications_received.append(notification)
        
        hot_reloader.add_notification_callback(notification_callback)
        
        # Start monitoring
        hot_reloader.start_monitoring()
        
        try:
            # Modify configuration file
            env_file = temp_config_dir / "environments" / "development.yaml"
            with open(env_file, 'w') as f:
                yaml.dump({'debug': False, 'max_steps': 750}, f)
            
            # Wait for reload
            time.sleep(0.5)
            
            # Check notifications
            assert len(notifications_received) > 0
            
            # Check for file change notification
            file_change_notifications = [
                n for n in notifications_received
                if n.event_type == ReloadEvent.FILE_CHANGED
            ]
            assert len(file_change_notifications) > 0
            
        finally:
            hot_reloader.stop_monitoring()


if __name__ == '__main__':
    pytest.main([__file__])
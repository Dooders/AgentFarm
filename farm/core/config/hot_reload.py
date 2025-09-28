"""
Hot-reloading system for dynamic configuration updates.

This module provides the ConfigurationHotReloader class and related components
for monitoring configuration files and automatically reloading configurations
when changes are detected, with support for different reload strategies and
rollback mechanisms.
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
try:
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileMovedEvent, FileCreatedEvent
    from watchdog.observers import Observer
except ImportError:
    from .watchdog_mock import FileSystemEventHandler, FileModifiedEvent, FileMovedEvent, FileCreatedEvent, Observer

from .exceptions import ConfigurationError, ConfigurationLoadError
from .environment import EnvironmentConfigManager
from .hierarchical import HierarchicalConfig
from .migration import ConfigurationMigrator, ConfigurationVersionDetector

logger = logging.getLogger(__name__)


class ReloadStrategy(Enum):
    """Different strategies for handling configuration reloads."""
    
    IMMEDIATE = "immediate"  # Reload immediately when change is detected
    BATCHED = "batched"     # Batch changes and reload after a delay
    SCHEDULED = "scheduled" # Reload at scheduled intervals
    MANUAL = "manual"       # Only reload when explicitly requested


class ReloadEvent(Enum):
    """Types of reload events."""
    
    CONFIG_LOADED = "config_loaded"
    CONFIG_RELOADED = "config_reloaded"
    CONFIG_FAILED = "config_failed"
    CONFIG_ROLLED_BACK = "config_rolled_back"
    FILE_CHANGED = "file_changed"
    MIGRATION_APPLIED = "migration_applied"


@dataclass
class ReloadNotification:
    """Notification about a configuration reload event."""
    
    event_type: ReloadEvent
    timestamp: float
    file_path: Optional[str] = None
    old_config: Optional[Dict[str, Any]] = None
    new_config: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    migration_info: Optional[Dict[str, Any]] = None
    message: str = ""


@dataclass
class ReloadConfig:
    """Configuration for hot-reload behavior."""
    
    strategy: ReloadStrategy = ReloadStrategy.BATCHED
    batch_delay: float = 1.0  # Seconds to wait before batching changes
    max_batch_size: int = 10  # Maximum number of changes to batch
    schedule_interval: float = 5.0  # Seconds between scheduled reloads
    enable_rollback: bool = True  # Enable automatic rollback on failure
    max_rollback_attempts: int = 3  # Maximum rollback attempts
    validate_on_reload: bool = True  # Validate configuration after reload
    backup_configs: bool = True  # Keep backup of previous configurations
    max_backups: int = 5  # Maximum number of configuration backups
    watch_subdirectories: bool = True  # Watch subdirectories for changes
    file_patterns: List[str] = field(default_factory=lambda: ["*.yaml", "*.yml", "*.json"])
    ignore_patterns: List[str] = field(default_factory=lambda: ["*.tmp", "*.bak", ".*"])


class ConfigurationChangeHandler(FileSystemEventHandler):
    """Handles file system events for configuration files."""
    
    def __init__(self, hot_reloader: 'ConfigurationHotReloader'):
        """Initialize the change handler.
        
        Args:
            hot_reloader: The hot reloader instance to notify of changes
        """
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and self._should_handle_file(event.src_path):
            self.logger.debug(f"File modified: {event.src_path}")
            self.hot_reloader._handle_file_change(event.src_path, "modified")
    
    def on_moved(self, event):
        """Handle file move/rename events."""
        if not event.is_directory:
            if self._should_handle_file(event.dest_path):
                self.logger.debug(f"File moved to: {event.dest_path}")
                self.hot_reloader._handle_file_change(event.dest_path, "moved")
            elif self._should_handle_file(event.src_path):
                self.logger.debug(f"File moved from: {event.src_path}")
                self.hot_reloader._handle_file_change(event.src_path, "moved")
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and self._should_handle_file(event.src_path):
            self.logger.debug(f"File created: {event.src_path}")
            self.hot_reloader._handle_file_change(event.src_path, "created")
    
    def _should_handle_file(self, file_path: str) -> bool:
        """Check if a file should be handled based on patterns.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file should be handled, False otherwise
        """
        path = Path(file_path)
        
        # Check ignore patterns first
        for ignore_pattern in self.hot_reloader.reload_config.ignore_patterns:
            if path.match(ignore_pattern):
                return False
        
        # Check file patterns
        for file_pattern in self.hot_reloader.reload_config.file_patterns:
            if path.match(file_pattern):
                return True
        
        return False


class ReloadStrategyHandler(ABC):
    """Abstract base class for reload strategy handlers."""
    
    @abstractmethod
    def handle_change(self, file_path: str, change_type: str) -> None:
        """Handle a configuration file change.
        
        Args:
            file_path: Path to the changed file
            change_type: Type of change (modified, moved, created)
        """
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start the strategy handler."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the strategy handler."""
        pass


class ImmediateReloadHandler(ReloadStrategyHandler):
    """Handler for immediate reload strategy."""
    
    def __init__(self, hot_reloader: 'ConfigurationHotReloader'):
        """Initialize immediate reload handler.
        
        Args:
            hot_reloader: The hot reloader instance
        """
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def handle_change(self, file_path: str, change_type: str) -> None:
        """Handle change with immediate reload."""
        self.logger.debug(f"Immediate reload triggered by {change_type}: {file_path}")
        self.hot_reloader._reload_configuration()
    
    def start(self) -> None:
        """Start immediate reload handler."""
        self.logger.debug("Started immediate reload handler")
    
    def stop(self) -> None:
        """Stop immediate reload handler."""
        self.logger.debug("Stopped immediate reload handler")


class BatchedReloadHandler(ReloadStrategyHandler):
    """Handler for batched reload strategy."""
    
    def __init__(self, hot_reloader: 'ConfigurationHotReloader'):
        """Initialize batched reload handler.
        
        Args:
            hot_reloader: The hot reloader instance
        """
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.pending_changes: Set[str] = set()
        self.batch_timer: Optional[threading.Timer] = None
        self.lock = threading.Lock()
    
    def handle_change(self, file_path: str, change_type: str) -> None:
        """Handle change with batched reload."""
        with self.lock:
            self.pending_changes.add(file_path)
            self.logger.debug(f"Added to batch: {file_path} (total: {len(self.pending_changes)})")
            
            # Cancel existing timer
            if self.batch_timer:
                self.batch_timer.cancel()
            
            # Check if we should reload immediately
            if len(self.pending_changes) >= self.hot_reloader.reload_config.max_batch_size:
                self.logger.debug("Batch size limit reached, reloading immediately")
                self._reload_batch()
            else:
                # Set new timer
                self.batch_timer = threading.Timer(
                    self.hot_reloader.reload_config.batch_delay,
                    self._reload_batch
                )
                self.batch_timer.start()
    
    def _reload_batch(self) -> None:
        """Reload configuration with batched changes."""
        with self.lock:
            if self.pending_changes:
                self.logger.debug(f"Reloading batch of {len(self.pending_changes)} changes")
                self.hot_reloader._reload_configuration()
                self.pending_changes.clear()
            
            if self.batch_timer:
                self.batch_timer.cancel()
                self.batch_timer = None
    
    def start(self) -> None:
        """Start batched reload handler."""
        self.logger.debug("Started batched reload handler")
    
    def stop(self) -> None:
        """Stop batched reload handler."""
        with self.lock:
            if self.batch_timer:
                self.batch_timer.cancel()
                self.batch_timer = None
            self.pending_changes.clear()
        self.logger.debug("Stopped batched reload handler")


class ScheduledReloadHandler(ReloadStrategyHandler):
    """Handler for scheduled reload strategy."""
    
    def __init__(self, hot_reloader: 'ConfigurationHotReloader'):
        """Initialize scheduled reload handler.
        
        Args:
            hot_reloader: The hot reloader instance
        """
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pending_changes: Set[str] = set()
        self.lock = threading.Lock()
    
    def handle_change(self, file_path: str, change_type: str) -> None:
        """Handle change by adding to pending changes."""
        with self.lock:
            self.pending_changes.add(file_path)
            self.logger.debug(f"Added to scheduled batch: {file_path}")
    
    def start(self) -> None:
        """Start scheduled reload handler."""
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.debug("Started scheduled reload handler")
    
    def stop(self) -> None:
        """Stop scheduled reload handler."""
        self.stop_event.set()
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=1.0)
        self.logger.debug("Stopped scheduled reload handler")
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while not self.stop_event.is_set():
            if self.stop_event.wait(self.hot_reloader.reload_config.schedule_interval):
                break
            
            with self.lock:
                if self.pending_changes:
                    self.logger.debug(f"Scheduled reload with {len(self.pending_changes)} changes")
                    self.hot_reloader._reload_configuration()
                    self.pending_changes.clear()


class ManualReloadHandler(ReloadStrategyHandler):
    """Handler for manual reload strategy."""
    
    def __init__(self, hot_reloader: 'ConfigurationHotReloader'):
        """Initialize manual reload handler.
        
        Args:
            hot_reloader: The hot reloader instance
        """
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.pending_changes: Set[str] = set()
        self.lock = threading.Lock()
    
    def handle_change(self, file_path: str, change_type: str) -> None:
        """Handle change by adding to pending changes."""
        with self.lock:
            self.pending_changes.add(file_path)
            self.logger.debug(f"Added to manual batch: {file_path}")
    
    def start(self) -> None:
        """Start manual reload handler."""
        self.logger.debug("Started manual reload handler")
    
    def stop(self) -> None:
        """Stop manual reload handler."""
        with self.lock:
            self.pending_changes.clear()
        self.logger.debug("Stopped manual reload handler")
    
    def get_pending_changes(self) -> List[str]:
        """Get list of pending changes.
        
        Returns:
            List of file paths with pending changes
        """
        with self.lock:
            return list(self.pending_changes)
    
    def clear_pending_changes(self) -> None:
        """Clear pending changes."""
        with self.lock:
            self.pending_changes.clear()


class ConfigurationHotReloader:
    """Hot-reloading system for configuration files.
    
    This class provides automatic reloading of configuration files when changes
    are detected, with support for different reload strategies, rollback mechanisms,
    and change notifications.
    """
    
    def __init__(
        self,
        config_manager: EnvironmentConfigManager,
        reload_config: Optional[ReloadConfig] = None,
        migrations_dir: Optional[str] = None
    ):
        """Initialize the hot reloader.
        
        Args:
            config_manager: Environment configuration manager
            reload_config: Hot-reload configuration
            migrations_dir: Directory containing migration scripts
        """
        self.config_manager = config_manager
        self.reload_config = reload_config or ReloadConfig()
        self.migrations_dir = migrations_dir
        
        # File system monitoring
        self.observer: Optional[Observer] = None
        self.change_handler: Optional[ConfigurationChangeHandler] = None
        self.strategy_handler: Optional[ReloadStrategyHandler] = None
        
        # Configuration state
        self.current_config: Optional[Dict[str, Any]] = None
        self.config_backups: List[Dict[str, Any]] = []
        self.is_reloading = False
        self.reload_lock = threading.Lock()
        
        # Migration support
        self.migrator: Optional[ConfigurationMigrator] = None
        self.version_detector: Optional[ConfigurationVersionDetector] = None
        
        if migrations_dir:
            self.migrator = ConfigurationMigrator(migrations_dir)
            self.version_detector = ConfigurationVersionDetector()
        
        # Notification callbacks
        self.notification_callbacks: List[Callable[[ReloadNotification], None]] = []
        
        # Initialize
        self._initialize_strategy_handler()
        self._load_initial_configuration()
    
    def _initialize_strategy_handler(self) -> None:
        """Initialize the appropriate strategy handler."""
        if self.reload_config.strategy == ReloadStrategy.IMMEDIATE:
            self.strategy_handler = ImmediateReloadHandler(self)
        elif self.reload_config.strategy == ReloadStrategy.BATCHED:
            self.strategy_handler = BatchedReloadHandler(self)
        elif self.reload_config.strategy == ReloadStrategy.SCHEDULED:
            self.strategy_handler = ScheduledReloadHandler(self)
        elif self.reload_config.strategy == ReloadStrategy.MANUAL:
            self.strategy_handler = ManualReloadHandler(self)
        else:
            raise ValueError(f"Unknown reload strategy: {self.reload_config.strategy}")
    
    def _load_initial_configuration(self) -> None:
        """Load the initial configuration."""
        try:
            self.current_config = self.config_manager.get_effective_config()
            self._notify(ReloadEvent.CONFIG_LOADED, message="Initial configuration loaded")
            logger.info("Initial configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load initial configuration: {e}")
            self._notify(ReloadEvent.CONFIG_FAILED, error=e, message="Failed to load initial configuration")
    
    def start_monitoring(self) -> None:
        """Start monitoring configuration files for changes."""
        if self.observer and self.observer.is_alive():
            logger.warning("File monitoring is already active")
            return
        
        try:
            # Initialize file system observer
            self.observer = Observer()
            self.change_handler = ConfigurationChangeHandler(self)
            
            # Watch configuration directory
            config_dir = Path(self.config_manager.config_dir)
            if config_dir.exists():
                self.observer.schedule(
                    self.change_handler,
                    str(config_dir),
                    recursive=self.reload_config.watch_subdirectories
                )
                logger.info(f"Started monitoring configuration directory: {config_dir}")
            else:
                logger.warning(f"Configuration directory not found: {config_dir}")
            
            # Start strategy handler
            self.strategy_handler.start()
            
            # Start observer
            self.observer.start()
            
            logger.info("Configuration hot-reload monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start file monitoring: {e}")
            raise ConfigurationError(f"Failed to start file monitoring: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring configuration files."""
        try:
            # Stop strategy handler
            if self.strategy_handler:
                self.strategy_handler.stop()
            
            # Stop observer
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=1.0)
            
            logger.info("Configuration hot-reload monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping file monitoring: {e}")
    
    def _handle_file_change(self, file_path: str, change_type: str) -> None:
        """Handle a file change event.
        
        Args:
            file_path: Path to the changed file
            change_type: Type of change (modified, moved, created)
        """
        logger.debug(f"File change detected: {change_type} - {file_path}")
        
        # Notify about file change
        self._notify(
            ReloadEvent.FILE_CHANGED,
            file_path=file_path,
            message=f"File {change_type}: {file_path}"
        )
        
        # Handle change based on strategy
        if self.strategy_handler:
            self.strategy_handler.handle_change(file_path, change_type)
    
    def _reload_configuration(self) -> None:
        """Reload the configuration."""
        if self.is_reloading:
            logger.debug("Configuration reload already in progress, skipping")
            return
        
        with self.reload_lock:
            self.is_reloading = True
            
            try:
                logger.info("Starting configuration reload")
                
                # Backup current configuration
                if self.reload_config.backup_configs and self.current_config:
                    self._backup_configuration(self.current_config)
                
                # Reload configuration
                old_config = self.current_config
                new_config = self.config_manager.get_effective_config(force_reload=True)
                
                # Apply migrations if available
                if self.migrator and self.version_detector:
                    new_config = self._apply_migrations(new_config)
                
                # Validate configuration if enabled
                if self.reload_config.validate_on_reload:
                    self._validate_configuration(new_config)
                
                # Update current configuration
                self.current_config = new_config
                
                # Notify about successful reload
                self._notify(
                    ReloadEvent.CONFIG_RELOADED,
                    old_config=old_config,
                    new_config=new_config,
                    message="Configuration reloaded successfully"
                )
                
                logger.info("Configuration reload completed successfully")
                
            except Exception as e:
                logger.error(f"Configuration reload failed: {e}")
                
                # Attempt rollback if enabled
                if self.reload_config.enable_rollback and self.config_backups:
                    self._attempt_rollback(e)
                else:
                    self._notify(
                        ReloadEvent.CONFIG_FAILED,
                        error=e,
                        message=f"Configuration reload failed: {e}"
                    )
            
            finally:
                self.is_reloading = False
    
    def _apply_migrations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply migrations to configuration if needed.
        
        Args:
            config: Configuration to migrate
            
        Returns:
            Migrated configuration
        """
        try:
            current_version = self.version_detector.detect_version(config)
            latest_version = "2.1"  # Latest version from our migrations
            
            if current_version != latest_version:
                logger.info(f"Migrating configuration from {current_version} to {latest_version}")
                migrated_config = self.migrator.migrate_config(config, current_version, latest_version)
                
                # Notify about migration
                self._notify(
                    ReloadEvent.MIGRATION_APPLIED,
                    new_config=migrated_config,
                    migration_info={
                        'from_version': current_version,
                        'to_version': latest_version
                    },
                    message=f"Configuration migrated from {current_version} to {latest_version}"
                )
                
                return migrated_config
            
            return config
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise ConfigurationError(f"Migration failed: {e}")
    
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate configuration after reload.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ConfigurationError: If validation fails
        """
        try:
            # Use the environment manager's validation
            self.config_manager.validate_all_configs()
            logger.debug("Configuration validation passed")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ConfigurationError(f"Configuration validation failed: {e}")
    
    def _backup_configuration(self, config: Dict[str, Any]) -> None:
        """Backup current configuration.
        
        Args:
            config: Configuration to backup
        """
        try:
            # Add timestamp to backup
            backup = {
                'config': config.copy(),
                'timestamp': time.time(),
                'version': self.version_detector.detect_version(config) if self.version_detector else 'unknown'
            }
            
            self.config_backups.append(backup)
            
            # Limit number of backups
            if len(self.config_backups) > self.reload_config.max_backups:
                self.config_backups.pop(0)
            
            logger.debug(f"Configuration backed up (total backups: {len(self.config_backups)})")
            
        except Exception as e:
            logger.error(f"Failed to backup configuration: {e}")
    
    def _attempt_rollback(self, error: Exception) -> None:
        """Attempt to rollback to previous configuration.
        
        Args:
            error: The error that caused the rollback
        """
        if not self.config_backups:
            logger.error("No configuration backups available for rollback")
            self._notify(
                ReloadEvent.CONFIG_FAILED,
                error=error,
                message="Configuration reload failed and no backups available for rollback"
            )
            return
        
        try:
            # Get most recent backup
            backup = self.config_backups[-1]
            backup_config = backup['config']
            
            logger.info(f"Attempting rollback to backup from {backup['timestamp']}")
            
            # Restore configuration
            self.current_config = backup_config
            
            # Notify about rollback
            self._notify(
                ReloadEvent.CONFIG_ROLLED_BACK,
                new_config=backup_config,
                error=error,
                message=f"Configuration rolled back due to error: {error}"
            )
            
            logger.info("Configuration rollback completed successfully")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            self._notify(
                ReloadEvent.CONFIG_FAILED,
                error=e,
                message=f"Configuration reload and rollback both failed: {e}"
            )
    
    def manual_reload(self) -> bool:
        """Manually trigger configuration reload.
        
        Returns:
            True if reload was successful, False otherwise
        """
        if self.reload_config.strategy != ReloadStrategy.MANUAL:
            logger.warning("Manual reload called but strategy is not manual")
            return False
        
        # Clear pending changes for manual strategy
        if isinstance(self.strategy_handler, ManualReloadHandler):
            pending = self.strategy_handler.get_pending_changes()
            if pending:
                logger.info(f"Manual reload triggered with {len(pending)} pending changes")
                self.strategy_handler.clear_pending_changes()
            else:
                logger.info("Manual reload triggered with no pending changes")
        
        try:
            self._reload_configuration()
            return True
        except Exception as e:
            logger.error(f"Manual reload failed: {e}")
            return False
    
    def get_pending_changes(self) -> List[str]:
        """Get list of pending changes (for manual strategy).
        
        Returns:
            List of file paths with pending changes
        """
        if isinstance(self.strategy_handler, ManualReloadHandler):
            return self.strategy_handler.get_pending_changes()
        return []
    
    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """Get the current configuration.
        
        Returns:
            Current configuration dictionary or None if not loaded
        """
        return self.current_config.copy() if self.current_config else None
    
    def get_config_backups(self) -> List[Dict[str, Any]]:
        """Get configuration backups.
        
        Returns:
            List of configuration backups
        """
        return self.config_backups.copy()
    
    def add_notification_callback(self, callback: Callable[[ReloadNotification], None]) -> None:
        """Add a notification callback.
        
        Args:
            callback: Function to call when reload events occur
        """
        self.notification_callbacks.append(callback)
    
    def remove_notification_callback(self, callback: Callable[[ReloadNotification], None]) -> None:
        """Remove a notification callback.
        
        Args:
            callback: Function to remove
        """
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
    
    def _notify(
        self,
        event_type: ReloadEvent,
        file_path: Optional[str] = None,
        old_config: Optional[Dict[str, Any]] = None,
        new_config: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None,
        migration_info: Optional[Dict[str, Any]] = None,
        message: str = ""
    ) -> None:
        """Send notification to all registered callbacks.
        
        Args:
            event_type: Type of reload event
            file_path: Path to the changed file
            old_config: Previous configuration
            new_config: New configuration
            error: Error that occurred
            migration_info: Migration information
            message: Human-readable message
        """
        notification = ReloadNotification(
            event_type=event_type,
            timestamp=time.time(),
            file_path=file_path,
            old_config=old_config,
            new_config=new_config,
            error=error,
            migration_info=migration_info,
            message=message
        )
        
        for callback in self.notification_callbacks:
            try:
                callback(notification)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")
    
    def is_monitoring(self) -> bool:
        """Check if file monitoring is active.
        
        Returns:
            True if monitoring is active, False otherwise
        """
        return self.observer is not None and self.observer.is_alive()
    
    def get_reload_stats(self) -> Dict[str, Any]:
        """Get reload statistics.
        
        Returns:
            Dictionary with reload statistics
        """
        return {
            'is_monitoring': self.is_monitoring(),
            'is_reloading': self.is_reloading,
            'strategy': self.reload_config.strategy.value,
            'backups_count': len(self.config_backups),
            'current_config_loaded': self.current_config is not None,
            'migration_enabled': self.migrator is not None,
            'validation_enabled': self.reload_config.validate_on_reload,
            'rollback_enabled': self.reload_config.enable_rollback
        }
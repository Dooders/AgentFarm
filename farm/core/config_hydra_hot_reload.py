"""
Hydra-based hot-reloading configuration system.

This module provides a Hydra-based hot-reloading system that integrates with
the existing hot-reload infrastructure while using Hydra for configuration management.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .config_hydra_simple import SimpleHydraConfigManager
from .config.hot_reload import (
    ConfigurationHotReloader,
    ReloadConfig,
    ReloadStrategy,
    ReloadEvent,
    ReloadNotification,
    ConfigurationChangeHandler,
    ReloadStrategyHandler
)

logger = logging.getLogger(__name__)


class HydraConfigurationHotReloader:
    """Hydra-based hot-reloading system for configuration files.
    
    This class provides automatic reloading of Hydra configuration files when changes
    are detected, integrating with the existing hot-reload infrastructure.
    """
    
    def __init__(
        self,
        config_manager: SimpleHydraConfigManager,
        reload_config: Optional[ReloadConfig] = None
    ):
        """Initialize the Hydra hot reloader.
        
        Args:
            config_manager: Hydra configuration manager
            reload_config: Hot-reload configuration
        """
        self.config_manager = config_manager
        self.reload_config = reload_config or ReloadConfig()
        
        # File system monitoring (reuse existing infrastructure)
        self.observer = None
        self.change_handler = None
        self.strategy_handler = None
        
        # Configuration state
        self.current_config: Optional[Dict[str, Any]] = None
        self.config_backups: List[Dict[str, Any]] = []
        self.is_reloading = False
        self.reload_lock = threading.Lock()
        
        # Notification callbacks
        self.notification_callbacks: List[Callable[[ReloadNotification], None]] = []
        
        # Initialize
        self._initialize_strategy_handler()
        self._load_initial_configuration()
    
    def _initialize_strategy_handler(self) -> None:
        """Initialize the appropriate strategy handler."""
        if self.reload_config.strategy == ReloadStrategy.IMMEDIATE:
            self.strategy_handler = HydraImmediateReloadHandler(self)
        elif self.reload_config.strategy == ReloadStrategy.BATCHED:
            self.strategy_handler = HydraBatchedReloadHandler(self)
        elif self.reload_config.strategy == ReloadStrategy.SCHEDULED:
            self.strategy_handler = HydraScheduledReloadHandler(self)
        elif self.reload_config.strategy == ReloadStrategy.MANUAL:
            self.strategy_handler = HydraManualReloadHandler(self)
        else:
            raise ValueError(f"Unknown reload strategy: {self.reload_config.strategy}")
    
    def _load_initial_configuration(self) -> None:
        """Load the initial configuration."""
        try:
            self.current_config = self.config_manager.to_dict()
            self._notify(ReloadEvent.CONFIG_LOADED, message="Initial Hydra configuration loaded")
            logger.info("Initial Hydra configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load initial Hydra configuration: {e}")
            self._notify(ReloadEvent.CONFIG_FAILED, error=e, message="Failed to load initial Hydra configuration")
    
    def start_monitoring(self) -> None:
        """Start monitoring configuration files for changes."""
        if self.observer and self.observer.is_alive():
            logger.warning("File monitoring is already active")
            return
        
        try:
            # Initialize file system observer
            from watchdog.observers import Observer
            self.observer = Observer()
            self.change_handler = HydraConfigurationChangeHandler(self)
            
            # Watch configuration directory
            config_dir = Path(self.config_manager.config_dir)
            if config_dir.exists():
                self.observer.schedule(
                    self.change_handler,
                    str(config_dir),
                    recursive=self.reload_config.watch_subdirectories
                )
                logger.info(f"Started monitoring Hydra configuration directory: {config_dir}")
            else:
                logger.warning(f"Hydra configuration directory not found: {config_dir}")
            
            # Start strategy handler
            self.strategy_handler.start()
            
            # Start observer
            self.observer.start()
            
            logger.info("Hydra configuration hot-reload monitoring started")
            
        except Exception as e:
            logger.error(f"Failed to start Hydra file monitoring: {e}")
            raise RuntimeError(f"Failed to start Hydra file monitoring: {e}")
    
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
            
            logger.info("Hydra configuration hot-reload monitoring stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Hydra file monitoring: {e}")
    
    def _handle_file_change(self, file_path: str, change_type: str) -> None:
        """Handle a file change event.
        
        Args:
            file_path: Path to the changed file
            change_type: Type of change (modified, moved, created)
        """
        logger.debug(f"Hydra file change detected: {change_type} - {file_path}")
        
        # Notify about file change
        self._notify(
            ReloadEvent.FILE_CHANGED,
            file_path=file_path,
            message=f"Hydra file {change_type}: {file_path}"
        )
        
        # Handle change based on strategy
        if self.strategy_handler:
            self.strategy_handler.handle_change(file_path, change_type)
    
    def _reload_configuration(self) -> None:
        """Reload the Hydra configuration."""
        if self.is_reloading:
            logger.debug("Hydra configuration reload already in progress, skipping")
            return
        
        with self.reload_lock:
            self.is_reloading = True
            
            try:
                logger.info("Starting Hydra configuration reload")
                
                # Backup current configuration
                if self.reload_config.backup_configs and self.current_config:
                    self._backup_configuration(self.current_config)
                
                # Reload configuration
                old_config = self.current_config
                
                # Reinitialize Hydra with current settings
                self.config_manager.reload()
                new_config = self.config_manager.to_dict()
                
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
                    message="Hydra configuration reloaded successfully"
                )
                
                logger.info("Hydra configuration reload completed successfully")
                
            except Exception as e:
                logger.error(f"Hydra configuration reload failed: {e}")
                
                # Attempt rollback if enabled
                if self.reload_config.enable_rollback and self.config_backups:
                    self._attempt_rollback(e)
                else:
                    self._notify(
                        ReloadEvent.CONFIG_FAILED,
                        error=e,
                        message=f"Hydra configuration reload failed: {e}"
                    )
            
            finally:
                self.is_reloading = False
    
    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate configuration after reload.
        
        Args:
            config: Configuration to validate
            
        Raises:
            RuntimeError: If validation fails
        """
        try:
            # Use the Hydra config manager's validation
            errors = self.config_manager.validate_configuration()
            if errors:
                raise RuntimeError(f"Configuration validation failed: {errors}")
            logger.debug("Hydra configuration validation passed")
        except Exception as e:
            logger.error(f"Hydra configuration validation failed: {e}")
            raise RuntimeError(f"Hydra configuration validation failed: {e}")
    
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
                'version': 'hydra'
            }
            
            self.config_backups.append(backup)
            
            # Limit number of backups
            if len(self.config_backups) > self.reload_config.max_backups:
                self.config_backups.pop(0)
            
            logger.debug(f"Hydra configuration backed up (total backups: {len(self.config_backups)})")
            
        except Exception as e:
            logger.error(f"Failed to backup Hydra configuration: {e}")
    
    def _attempt_rollback(self, error: Exception) -> None:
        """Attempt to rollback to previous configuration.
        
        Args:
            error: The error that caused the rollback
        """
        if not self.config_backups:
            logger.error("No Hydra configuration backups available for rollback")
            self._notify(
                ReloadEvent.CONFIG_FAILED,
                error=error,
                message="Hydra configuration reload failed and no backups available for rollback"
            )
            return
        
        try:
            # Get most recent backup
            backup = self.config_backups[-1]
            backup_config = backup['config']
            
            logger.info(f"Attempting Hydra rollback to backup from {backup['timestamp']}")
            
            # Restore configuration
            self.current_config = backup_config
            
            # Notify about rollback
            self._notify(
                ReloadEvent.CONFIG_ROLLED_BACK,
                new_config=backup_config,
                error=error,
                message=f"Hydra configuration rolled back due to error: {error}"
            )
            
            logger.info("Hydra configuration rollback completed successfully")
            
        except Exception as e:
            logger.error(f"Hydra rollback failed: {e}")
            self._notify(
                ReloadEvent.CONFIG_FAILED,
                error=e,
                message=f"Hydra configuration reload and rollback both failed: {e}"
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
        if isinstance(self.strategy_handler, HydraManualReloadHandler):
            pending = self.strategy_handler.get_pending_changes()
            if pending:
                logger.info(f"Manual Hydra reload triggered with {len(pending)} pending changes")
                self.strategy_handler.clear_pending_changes()
            else:
                logger.info("Manual Hydra reload triggered with no pending changes")
        
        try:
            self._reload_configuration()
            return True
        except Exception as e:
            logger.error(f"Manual Hydra reload failed: {e}")
            return False
    
    def get_pending_changes(self) -> List[str]:
        """Get list of pending changes (for manual strategy).
        
        Returns:
            List of file paths with pending changes
        """
        if isinstance(self.strategy_handler, HydraManualReloadHandler):
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
                logger.error(f"Error in Hydra notification callback: {e}")
    
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
            'validation_enabled': self.reload_config.validate_on_reload,
            'rollback_enabled': self.reload_config.enable_rollback,
            'config_manager_type': 'Hydra'
        }


# Hydra-specific reload strategy handlers
class HydraImmediateReloadHandler(ReloadStrategyHandler):
    """Handler for immediate reload strategy with Hydra."""
    
    def __init__(self, hot_reloader: 'HydraConfigurationHotReloader'):
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def handle_change(self, file_path: str, change_type: str) -> None:
        self.logger.debug(f"Hydra immediate reload triggered by {change_type}: {file_path}")
        self.hot_reloader._reload_configuration()
    
    def start(self) -> None:
        self.logger.debug("Started Hydra immediate reload handler")
    
    def stop(self) -> None:
        self.logger.debug("Stopped Hydra immediate reload handler")


class HydraBatchedReloadHandler(ReloadStrategyHandler):
    """Handler for batched reload strategy with Hydra."""
    
    def __init__(self, hot_reloader: 'HydraConfigurationHotReloader'):
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.pending_changes: Set[str] = set()
        self.batch_timer: Optional[threading.Timer] = None
        self.lock = threading.Lock()
    
    def handle_change(self, file_path: str, change_type: str) -> None:
        with self.lock:
            self.pending_changes.add(file_path)
            self.logger.debug(f"Added to Hydra batch: {file_path} (total: {len(self.pending_changes)})")
            
            # Cancel existing timer
            if self.batch_timer:
                self.batch_timer.cancel()
            
            # Check if we should reload immediately
            if len(self.pending_changes) >= self.hot_reloader.reload_config.max_batch_size:
                self.logger.debug("Hydra batch size limit reached, reloading immediately")
                self._reload_batch()
            else:
                # Set new timer
                self.batch_timer = threading.Timer(
                    self.hot_reloader.reload_config.batch_delay,
                    self._reload_batch
                )
                self.batch_timer.start()
    
    def _reload_batch(self) -> None:
        with self.lock:
            if self.pending_changes:
                self.logger.debug(f"Reloading Hydra batch of {len(self.pending_changes)} changes")
                self.hot_reloader._reload_configuration()
                self.pending_changes.clear()
            
            if self.batch_timer:
                self.batch_timer.cancel()
                self.batch_timer = None
    
    def start(self) -> None:
        self.logger.debug("Started Hydra batched reload handler")
    
    def stop(self) -> None:
        with self.lock:
            if self.batch_timer:
                self.batch_timer.cancel()
                self.batch_timer = None
            self.pending_changes.clear()
        self.logger.debug("Stopped Hydra batched reload handler")


class HydraScheduledReloadHandler(ReloadStrategyHandler):
    """Handler for scheduled reload strategy with Hydra."""
    
    def __init__(self, hot_reloader: 'HydraConfigurationHotReloader'):
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.pending_changes: Set[str] = set()
        self.lock = threading.Lock()
    
    def handle_change(self, file_path: str, change_type: str) -> None:
        with self.lock:
            self.pending_changes.add(file_path)
            self.logger.debug(f"Added to Hydra scheduled batch: {file_path}")
    
    def start(self) -> None:
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        self.logger.debug("Started Hydra scheduled reload handler")
    
    def stop(self) -> None:
        self.stop_event.set()
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=1.0)
        self.logger.debug("Stopped Hydra scheduled reload handler")
    
    def _scheduler_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.stop_event.wait(self.hot_reloader.reload_config.schedule_interval):
                break
            
            with self.lock:
                if self.pending_changes:
                    self.logger.debug(f"Scheduled Hydra reload with {len(self.pending_changes)} changes")
                    self.hot_reloader._reload_configuration()
                    self.pending_changes.clear()


class HydraManualReloadHandler(ReloadStrategyHandler):
    """Handler for manual reload strategy with Hydra."""
    
    def __init__(self, hot_reloader: 'HydraConfigurationHotReloader'):
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.pending_changes: Set[str] = set()
        self.lock = threading.Lock()
    
    def handle_change(self, file_path: str, change_type: str) -> None:
        with self.lock:
            self.pending_changes.add(file_path)
            self.logger.debug(f"Added to Hydra manual batch: {file_path}")
    
    def start(self) -> None:
        self.logger.debug("Started Hydra manual reload handler")
    
    def stop(self) -> None:
        with self.lock:
            self.pending_changes.clear()
        self.logger.debug("Stopped Hydra manual reload handler")
    
    def get_pending_changes(self) -> List[str]:
        with self.lock:
            return list(self.pending_changes)
    
    def clear_pending_changes(self) -> None:
        with self.lock:
            self.pending_changes.clear()


class HydraConfigurationChangeHandler:
    """Handles file system events for Hydra configuration files."""
    
    def __init__(self, hot_reloader: 'HydraConfigurationHotReloader'):
        self.hot_reloader = hot_reloader
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def on_modified(self, event):
        if not event.is_directory and self._should_handle_file(event.src_path):
            self.logger.debug(f"Hydra file modified: {event.src_path}")
            self.hot_reloader._handle_file_change(event.src_path, "modified")
    
    def on_moved(self, event):
        if not event.is_directory:
            if self._should_handle_file(event.dest_path):
                self.logger.debug(f"Hydra file moved to: {event.dest_path}")
                self.hot_reloader._handle_file_change(event.dest_path, "moved")
            elif self._should_handle_file(event.src_path):
                self.logger.debug(f"Hydra file moved from: {event.src_path}")
                self.hot_reloader._handle_file_change(event.src_path, "moved")
    
    def on_created(self, event):
        if not event.is_directory and self._should_handle_file(event.src_path):
            self.logger.debug(f"Hydra file created: {event.src_path}")
            self.hot_reloader._handle_file_change(event.src_path, "created")
    
    def _should_handle_file(self, file_path: str) -> bool:
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
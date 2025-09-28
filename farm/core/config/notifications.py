"""
Configuration change notification system.

This module provides a comprehensive notification system for configuration
changes, including event broadcasting, subscription management, and
integration with the hot-reload system.
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from queue import Queue, Empty

from .hot_reload import ReloadEvent, ReloadNotification

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Priority levels for notifications."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Different notification channels."""
    
    CONSOLE = "console"
    LOG = "log"
    CALLBACK = "callback"
    ASYNC_CALLBACK = "async_callback"
    QUEUE = "queue"
    FILE = "file"


@dataclass
class NotificationFilter:
    """Filter for notifications."""
    
    event_types: Optional[Set[ReloadEvent]] = None
    file_patterns: Optional[List[str]] = None
    min_priority: NotificationPriority = NotificationPriority.LOW
    include_errors: bool = True
    include_success: bool = True


@dataclass
class NotificationConfig:
    """Configuration for notification system."""
    
    enabled_channels: Set[NotificationChannel] = field(default_factory=lambda: {NotificationChannel.LOG})
    default_priority: NotificationPriority = NotificationPriority.NORMAL
    max_queue_size: int = 1000
    async_timeout: float = 5.0
    log_level: str = "INFO"
    file_path: Optional[str] = None
    console_format: str = "[{timestamp}] {event_type}: {message}"
    log_format: str = "{timestamp} - {event_type} - {message}"
    file_format: str = "{timestamp},{event_type},{priority},{message}"


class NotificationSubscriber(ABC):
    """Abstract base class for notification subscribers."""
    
    @abstractmethod
    def handle_notification(self, notification: ReloadNotification) -> None:
        """Handle a configuration change notification.
        
        Args:
            notification: The notification to handle
        """
        pass
    
    @abstractmethod
    def get_filter(self) -> Optional[NotificationFilter]:
        """Get notification filter for this subscriber.
        
        Returns:
            Notification filter or None for no filtering
        """
        pass


class AsyncNotificationSubscriber(ABC):
    """Abstract base class for async notification subscribers."""
    
    @abstractmethod
    async def handle_notification_async(self, notification: ReloadNotification) -> None:
        """Handle a configuration change notification asynchronously.
        
        Args:
            notification: The notification to handle
        """
        pass
    
    @abstractmethod
    def get_filter(self) -> Optional[NotificationFilter]:
        """Get notification filter for this subscriber.
        
        Returns:
            Notification filter or None for no filtering
        """
        pass


class ConsoleNotificationHandler:
    """Handler for console notifications."""
    
    def __init__(self, config: NotificationConfig):
        """Initialize console notification handler.
        
        Args:
            config: Notification configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def handle_notification(self, notification: ReloadNotification) -> None:
        """Handle console notification.
        
        Args:
            notification: The notification to handle
        """
        try:
            message = self._format_notification(notification)
            print(message)
        except Exception as e:
            self.logger.error(f"Error handling console notification: {e}")
    
    def _format_notification(self, notification: ReloadNotification) -> str:
        """Format notification for console output.
        
        Args:
            notification: The notification to format
            
        Returns:
            Formatted notification string
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(notification.timestamp))
        
        return self.config.console_format.format(
            timestamp=timestamp,
            event_type=notification.event_type.value,
            message=notification.message,
            file_path=notification.file_path or "",
            error=str(notification.error) if notification.error else ""
        )


class LogNotificationHandler:
    """Handler for log notifications."""
    
    def __init__(self, config: NotificationConfig):
        """Initialize log notification handler.
        
        Args:
            config: Notification configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Set up logger level
        level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
    
    def handle_notification(self, notification: ReloadNotification) -> None:
        """Handle log notification.
        
        Args:
            notification: The notification to handle
        """
        try:
            message = self._format_notification(notification)
            
            # Choose log level based on event type
            if notification.event_type == ReloadEvent.CONFIG_FAILED:
                self.logger.error(message)
            elif notification.event_type == ReloadEvent.CONFIG_ROLLED_BACK:
                self.logger.warning(message)
            elif notification.event_type in [ReloadEvent.CONFIG_LOADED, ReloadEvent.CONFIG_RELOADED]:
                self.logger.info(message)
            else:
                self.logger.debug(message)
                
        except Exception as e:
            self.logger.error(f"Error handling log notification: {e}")
    
    def _format_notification(self, notification: ReloadNotification) -> str:
        """Format notification for log output.
        
        Args:
            notification: The notification to format
            
        Returns:
            Formatted notification string
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(notification.timestamp))
        
        return self.config.log_format.format(
            timestamp=timestamp,
            event_type=notification.event_type.value,
            message=notification.message,
            file_path=notification.file_path or "",
            error=str(notification.error) if notification.error else ""
        )


class FileNotificationHandler:
    """Handler for file notifications."""
    
    def __init__(self, config: NotificationConfig):
        """Initialize file notification handler.
        
        Args:
            config: Notification configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.file_path = config.file_path
        self.lock = threading.Lock()
    
    def handle_notification(self, notification: ReloadNotification) -> None:
        """Handle file notification.
        
        Args:
            notification: The notification to handle
        """
        if not self.file_path:
            return
        
        try:
            with self.lock:
                message = self._format_notification(notification)
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
        except Exception as e:
            self.logger.error(f"Error handling file notification: {e}")
    
    def _format_notification(self, notification: ReloadNotification) -> str:
        """Format notification for file output.
        
        Args:
            notification: The notification to format
            
        Returns:
            Formatted notification string
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(notification.timestamp))
        
        return self.config.file_format.format(
            timestamp=timestamp,
            event_type=notification.event_type.value,
            priority="normal",  # Could be enhanced to include priority
            message=notification.message.replace(',', ';'),  # Escape commas for CSV
            file_path=notification.file_path or "",
            error=str(notification.error).replace(',', ';') if notification.error else ""
        )


class QueueNotificationHandler:
    """Handler for queue notifications."""
    
    def __init__(self, config: NotificationConfig):
        """Initialize queue notification handler.
        
        Args:
            config: Notification configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.notification_queue: Queue = Queue(maxsize=config.max_queue_size)
    
    def handle_notification(self, notification: ReloadNotification) -> None:
        """Handle queue notification.
        
        Args:
            notification: The notification to handle
        """
        try:
            self.notification_queue.put_nowait(notification)
        except Exception as e:
            self.logger.error(f"Error adding notification to queue: {e}")
    
    def get_notification(self, timeout: Optional[float] = None) -> Optional[ReloadNotification]:
        """Get notification from queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Notification or None if timeout
        """
        try:
            return self.notification_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def get_all_notifications(self) -> List[ReloadNotification]:
        """Get all notifications from queue.
        
        Returns:
            List of all notifications in queue
        """
        notifications = []
        while True:
            try:
                notification = self.notification_queue.get_nowait()
                notifications.append(notification)
            except Empty:
                break
        return notifications
    
    def clear_queue(self) -> None:
        """Clear all notifications from queue."""
        while True:
            try:
                self.notification_queue.get_nowait()
            except Empty:
                break


class ConfigurationNotificationManager:
    """Manager for configuration change notifications.
    
    This class provides a centralized notification system for configuration
    changes, supporting multiple notification channels and filtering.
    """
    
    def __init__(self, config: Optional[NotificationConfig] = None):
        """Initialize notification manager.
        
        Args:
            config: Notification configuration
        """
        self.config = config or NotificationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Notification handlers
        self.handlers: Dict[NotificationChannel, Any] = {}
        self._initialize_handlers()
        
        # Subscribers
        self.subscribers: List[NotificationSubscriber] = []
        self.async_subscribers: List[AsyncNotificationSubscriber] = []
        
        # Async support
        self.async_loop: Optional[asyncio.AbstractEventLoop] = None
        self.async_thread: Optional[threading.Thread] = None
        self.async_queue: Queue = Queue()
        self.async_running = False
        
        # Statistics
        self.stats = {
            'notifications_sent': 0,
            'notifications_failed': 0,
            'subscribers_notified': 0,
            'async_notifications_sent': 0,
            'async_notifications_failed': 0
        }
    
    def _initialize_handlers(self) -> None:
        """Initialize notification handlers."""
        if NotificationChannel.CONSOLE in self.config.enabled_channels:
            self.handlers[NotificationChannel.CONSOLE] = ConsoleNotificationHandler(self.config)
        
        if NotificationChannel.LOG in self.config.enabled_channels:
            self.handlers[NotificationChannel.LOG] = LogNotificationHandler(self.config)
        
        if NotificationChannel.FILE in self.config.enabled_channels:
            self.handlers[NotificationChannel.FILE] = FileNotificationHandler(self.config)
        
        if NotificationChannel.QUEUE in self.config.enabled_channels:
            self.handlers[NotificationChannel.QUEUE] = QueueNotificationHandler(self.config)
    
    def start_async_processing(self) -> None:
        """Start async notification processing."""
        if self.async_running:
            return
        
        self.async_running = True
        self.async_thread = threading.Thread(target=self._async_worker, daemon=True)
        self.async_thread.start()
        self.logger.info("Started async notification processing")
    
    def stop_async_processing(self) -> None:
        """Stop async notification processing."""
        if not self.async_running:
            return
        
        self.async_running = False
        if self.async_thread and self.async_thread.is_alive():
            self.async_thread.join(timeout=1.0)
        self.logger.info("Stopped async notification processing")
    
    def _async_worker(self) -> None:
        """Async worker thread."""
        try:
            self.async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.async_loop)
            self.async_loop.run_until_complete(self._async_processor())
        except Exception as e:
            self.logger.error(f"Error in async worker: {e}")
        finally:
            if self.async_loop:
                self.async_loop.close()
    
    async def _async_processor(self) -> None:
        """Process async notifications without blocking the event loop."""
        while self.async_running:
            try:
                # Offload blocking queue.get to thread pool
                notification = await asyncio.to_thread(self.async_queue.get, timeout=0.1)
                
                # Process async subscribers
                for subscriber in self.async_subscribers:
                    try:
                        if self._should_notify_subscriber(subscriber, notification):
                            await subscriber.handle_notification_async(notification)
                            self.stats['async_notifications_sent'] += 1
                    except Exception as e:
                        self.logger.error(f"Error in async subscriber: {e}")
                        self.stats['async_notifications_failed'] += 1
                
            except Empty:
                await asyncio.sleep(0)
                continue
            except Exception as e:
                self.logger.error(f"Error processing async notification: {e}")
    
    def send_notification(self, notification: ReloadNotification) -> None:
        """Send notification through all enabled channels.
        
        Args:
            notification: The notification to send
        """
        try:
            # Send to handlers
            for channel, handler in self.handlers.items():
                try:
                    handler.handle_notification(notification)
                except Exception as e:
                    self.logger.error(f"Error in {channel.value} handler: {e}")
            
            # Send to synchronous subscribers
            for subscriber in self.subscribers:
                try:
                    if self._should_notify_subscriber(subscriber, notification):
                        subscriber.handle_notification(notification)
                        self.stats['subscribers_notified'] += 1
                except Exception as e:
                    self.logger.error(f"Error in subscriber: {e}")
            
            # Send to async subscribers
            if self.async_subscribers and self.async_running:
                try:
                    self.async_queue.put_nowait(notification)
                except Exception as e:
                    self.logger.error(f"Error queuing async notification: {e}")
            
            self.stats['notifications_sent'] += 1
            
        except Exception as e:
            self.logger.error(f"Error sending notification: {e}")
            self.stats['notifications_failed'] += 1
    
    def _should_notify_subscriber(self, subscriber: Union[NotificationSubscriber, AsyncNotificationSubscriber], notification: ReloadNotification) -> bool:
        """Check if subscriber should be notified.
        
        Args:
            subscriber: The subscriber to check
            notification: The notification
            
        Returns:
            True if subscriber should be notified
        """
        filter_config = subscriber.get_filter()
        if not filter_config:
            return True
        
        # Check event type filter
        if filter_config.event_types and notification.event_type not in filter_config.event_types:
            return False
        
        # Check file pattern filter
        if filter_config.file_patterns and notification.file_path:
            from pathlib import Path
            file_path = Path(notification.file_path)
            if not any(file_path.match(pattern) for pattern in filter_config.file_patterns):
                return False
        
        # Check error/success filter
        if notification.error and not filter_config.include_errors:
            return False
        if not notification.error and not filter_config.include_success:
            return False
        
        return True
    
    def add_subscriber(self, subscriber: NotificationSubscriber) -> None:
        """Add a synchronous notification subscriber.
        
        Args:
            subscriber: The subscriber to add
        """
        self.subscribers.append(subscriber)
        self.logger.debug(f"Added synchronous subscriber: {subscriber.__class__.__name__}")
    
    def add_async_subscriber(self, subscriber: AsyncNotificationSubscriber) -> None:
        """Add an async notification subscriber.
        
        Args:
            subscriber: The subscriber to add
        """
        self.async_subscribers.append(subscriber)
        self.logger.debug(f"Added async subscriber: {subscriber.__class__.__name__}")
        
        # Start async processing if not already running
        if not self.async_running:
            self.start_async_processing()
    
    def remove_subscriber(self, subscriber: Union[NotificationSubscriber, AsyncNotificationSubscriber]) -> None:
        """Remove a notification subscriber.
        
        Args:
            subscriber: The subscriber to remove
        """
        if subscriber in self.subscribers:
            self.subscribers.remove(subscriber)
            self.logger.debug(f"Removed synchronous subscriber: {subscriber.__class__.__name__}")
        
        if subscriber in self.async_subscribers:
            self.async_subscribers.remove(subscriber)
            self.logger.debug(f"Removed async subscriber: {subscriber.__class__.__name__}")
    
    def get_queue_handler(self) -> Optional[QueueNotificationHandler]:
        """Get the queue notification handler.
        
        Returns:
            Queue handler or None if not enabled
        """
        return self.handlers.get(NotificationChannel.QUEUE)
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics.
        
        Returns:
            Dictionary with notification statistics
        """
        return {
            'enabled_channels': [channel.value for channel in self.config.enabled_channels],
            'active_handlers': len(self.handlers),
            'synchronous_subscribers': len(self.subscribers),
            'async_subscribers': len(self.async_subscribers),
            'async_processing_active': self.async_running,
            'stats': self.stats.copy()
        }
    
    def clear_stats(self) -> None:
        """Clear notification statistics."""
        self.stats = {
            'notifications_sent': 0,
            'notifications_failed': 0,
            'subscribers_notified': 0,
            'async_notifications_sent': 0,
            'async_notifications_failed': 0
        }
    
    def shutdown(self) -> None:
        """Shutdown notification manager."""
        self.stop_async_processing()
        self.subscribers.clear()
        self.async_subscribers.clear()
        self.handlers.clear()
        self.logger.info("Notification manager shutdown complete")


# Convenience functions for common notification patterns

def create_console_notification_manager() -> ConfigurationNotificationManager:
    """Create a notification manager with console output.
    
    Returns:
        Notification manager configured for console output
    """
    config = NotificationConfig(
        enabled_channels={NotificationChannel.CONSOLE, NotificationChannel.LOG},
        default_priority=NotificationPriority.NORMAL
    )
    return ConfigurationNotificationManager(config)


def create_file_notification_manager(file_path: str) -> ConfigurationNotificationManager:
    """Create a notification manager with file output.
    
    Args:
        file_path: Path to notification log file
        
    Returns:
        Notification manager configured for file output
    """
    config = NotificationConfig(
        enabled_channels={NotificationChannel.FILE, NotificationChannel.LOG},
        file_path=file_path,
        default_priority=NotificationPriority.NORMAL
    )
    return ConfigurationNotificationManager(config)


def create_queue_notification_manager() -> ConfigurationNotificationManager:
    """Create a notification manager with queue output.
    
    Returns:
        Notification manager configured for queue output
    """
    config = NotificationConfig(
        enabled_channels={NotificationChannel.QUEUE, NotificationChannel.LOG},
        default_priority=NotificationPriority.NORMAL
    )
    return ConfigurationNotificationManager(config)
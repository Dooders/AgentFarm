"""
Mock implementation of watchdog for testing purposes.

This module provides mock implementations of watchdog classes when the
actual watchdog package is not available.
"""

import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path


class FileSystemEventHandler(ABC):
    """Abstract base class for file system event handlers."""
    
    @abstractmethod
    def on_modified(self, event):
        """Handle file modification events."""
        pass
    
    @abstractmethod
    def on_moved(self, event):
        """Handle file move/rename events."""
        pass
    
    @abstractmethod
    def on_created(self, event):
        """Handle file creation events."""
        pass


class FileSystemEvent:
    """Base class for file system events."""
    
    def __init__(self, src_path, is_directory=False):
        """Initialize file system event.
        
        Args:
            src_path: Source path of the event
            is_directory: Whether the path is a directory
        """
        self.src_path = src_path
        self.is_directory = is_directory


class FileModifiedEvent(FileSystemEvent):
    """Event for file modifications."""
    
    def __init__(self, src_path):
        """Initialize file modified event.
        
        Args:
            src_path: Path to the modified file
        """
        super().__init__(src_path, is_directory=False)


class FileMovedEvent(FileSystemEvent):
    """Event for file moves/renames."""
    
    def __init__(self, src_path, dest_path):
        """Initialize file moved event.
        
        Args:
            src_path: Source path of the moved file
            dest_path: Destination path of the moved file
        """
        super().__init__(src_path, is_directory=False)
        self.dest_path = dest_path


class FileCreatedEvent(FileSystemEvent):
    """Event for file creation."""
    
    def __init__(self, src_path):
        """Initialize file created event.
        
        Args:
            src_path: Path to the created file
        """
        super().__init__(src_path, is_directory=False)


class Observer:
    """Mock observer for file system monitoring."""
    
    def __init__(self):
        """Initialize observer."""
        self.handlers = []
        self.watches = []
        self.running = False
        self.thread = None
    
    def schedule(self, event_handler, path, recursive=False):
        """Schedule watching a path.
        
        Args:
            event_handler: Handler for file system events
            path: Path to watch
            recursive: Whether to watch recursively
        """
        self.handlers.append(event_handler)
        self.watches.append({
            'path': path,
            'recursive': recursive,
            'handler': event_handler
        })
    
    def start(self):
        """Start the observer."""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the observer."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
    
    def join(self, timeout=None):
        """Join the observer thread."""
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)
    
    def is_alive(self):
        """Check if observer is running."""
        return self.running and self.thread and self.thread.is_alive()
    
    def _monitor_loop(self):
        """Main monitoring loop (mock implementation)."""
        while self.running:
            time.sleep(0.1)  # Mock monitoring loop


# Try to import real watchdog, fall back to mock
try:
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileMovedEvent, FileCreatedEvent
    from watchdog.observers import Observer
except ImportError:
    # Use mock implementations
    pass
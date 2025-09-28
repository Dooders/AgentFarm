"""
Configuration file watcher for runtime reloading.

This module provides functionality to watch configuration files for changes
and automatically reload them when modified, useful for development and
long-running simulations.
"""

import hashlib
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Set, Union

import yaml

from farm.core.config import SimulationConfig


class ConfigWatcher:
    """
    Watches configuration files for changes and triggers reload callbacks.
    """

    def __init__(self, watch_interval: float = 2.0):
        """
        Initialize the configuration watcher.

        Args:
            watch_interval: How often to check for file changes (seconds)
        """
        self.watch_interval = watch_interval
        self.watched_files: Dict[str, str] = {}  # filepath -> hash
        self.callbacks: Dict[str, Set[Callable]] = {}  # filepath -> set of callbacks
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()

    def watch_file(self, filepath: str, callback: Callable[[SimulationConfig], None]) -> None:
        """
        Start watching a configuration file for changes.

        Args:
            filepath: Path to the configuration file to watch
            callback: Function to call when config changes (receives new SimulationConfig)
        """
        with self.lock:
            filepath = str(Path(filepath).resolve())

            if filepath not in self.callbacks:
                self.callbacks[filepath] = set()
                # Initialize hash
                try:
                    self.watched_files[filepath] = self._get_file_hash(filepath)
                except FileNotFoundError:
                    self.watched_files[filepath] = ""

            self.callbacks[filepath].add(callback)

    def unwatch_file(self, filepath: str, callback: Optional[Callable] = None) -> None:
        """
        Stop watching a configuration file.

        Args:
            filepath: Path to stop watching
            callback: Specific callback to remove (if None, removes all for this file)
        """
        with self.lock:
            filepath = str(Path(filepath).resolve())

            if callback is None:
                self.callbacks.pop(filepath, None)
                self.watched_files.pop(filepath, None)
            else:
                if filepath in self.callbacks:
                    self.callbacks[filepath].discard(callback)
                    if not self.callbacks[filepath]:
                        self.callbacks.pop(filepath, None)
                        self.watched_files.pop(filepath, None)

    def start(self) -> None:
        """Start the file watching thread."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stop the file watching thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)

    def _watch_loop(self) -> None:
        """Main watching loop that checks for file changes."""
        while self.running:
            try:
                self._check_files()
            except Exception as e:
                # Log error but continue watching
                print(f"ConfigWatcher error: {e}")
            time.sleep(self.watch_interval)

    def _check_files(self) -> None:
        """Check all watched files for changes."""
        with self.lock:
            files_to_check = list(self.watched_files.keys())

        for filepath in files_to_check:
            try:
                current_hash = self._get_file_hash(filepath)
                previous_hash = self.watched_files.get(filepath, "")

                if current_hash != previous_hash:
                    # File changed, reload and notify callbacks
                    self._handle_file_change(filepath, current_hash)

            except FileNotFoundError:
                # File was deleted
                if filepath in self.watched_files:
                    self.watched_files[filepath] = ""
            except Exception as e:
                print(f"Error checking file {filepath}: {e}")

    def _handle_file_change(self, filepath: str, new_hash: str) -> None:
        """Handle a file change by reloading config and calling callbacks."""
        try:
            # Reload configuration
            config = SimulationConfig.from_yaml(filepath)

            # Update hash
            with self.lock:
                self.watched_files[filepath] = new_hash
                callbacks = self.callbacks.get(filepath, set()).copy()

            # Call all callbacks
            for callback in callbacks:
                try:
                    callback(config)
                except Exception as e:
                    print(f"Error in config change callback: {e}")

        except Exception as e:
            print(f"Error reloading config from {filepath}: {e}")

    def _get_file_hash(self, filepath: str) -> str:
        """
        Get SHA256 hash of a file's contents.

        Args:
            filepath: Path to the file

        Returns:
            str: File hash

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()

    def get_watched_files(self) -> Dict[str, str]:
        """
        Get information about currently watched files.

        Returns:
            Dict[str, str]: filepath -> current_hash mapping
        """
        with self.lock:
            return self.watched_files.copy()


class ReloadableConfig:
    """
    A configuration wrapper that supports runtime reloading.
    """

    def __init__(self, config: SimulationConfig, watcher: Optional[ConfigWatcher] = None):
        """
        Initialize a reloadable configuration.

        Args:
            config: Initial configuration
            watcher: ConfigWatcher instance (created automatically if None)
        """
        self._config = config
        self._watcher = watcher or ConfigWatcher()
        self._callbacks: Set[Callable[[SimulationConfig], None]] = set()
        self._watcher_started = False

    def watch_file(self, filepath: str) -> None:
        """
        Watch a configuration file for changes.

        Args:
            filepath: Path to watch
        """
        def on_config_change(new_config: SimulationConfig):
            self._config = new_config
            self._notify_callbacks(new_config)

        self._watcher.watch_file(filepath, on_config_change)

        if not self._watcher_started:
            self._watcher.start()
            self._watcher_started = True

    def add_change_callback(self, callback: Callable[[SimulationConfig], None]) -> None:
        """
        Add a callback to be called when configuration changes.

        Args:
            callback: Function to call with new config
        """
        self._callbacks.add(callback)

    def remove_change_callback(self, callback: Callable[[SimulationConfig], None]) -> None:
        """
        Remove a change callback.

        Args:
            callback: Callback to remove
        """
        self._callbacks.discard(callback)

    def _notify_callbacks(self, new_config: SimulationConfig) -> None:
        """Notify all registered callbacks of config change."""
        for callback in self._callbacks.copy():
            try:
                callback(new_config)
            except Exception as e:
                print(f"Error in reload callback: {e}")

    @property
    def config(self) -> SimulationConfig:
        """Get the current configuration."""
        return self._config

    def reload_from_file(self, filepath: str) -> None:
        """
        Manually reload configuration from a file.

        Args:
            filepath: Path to load from
        """
        new_config = SimulationConfig.from_yaml(filepath)
        self._config = new_config
        self._notify_callbacks(new_config)

    def stop_watching(self) -> None:
        """Stop watching files for changes."""
        if self._watcher_started:
            self._watcher.stop()
            self._watcher_started = False

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying config."""
        return getattr(self._config, name)

    def __repr__(self) -> str:
        return f"ReloadableConfig({self._config!r})"


# Global watcher instance for shared use
_global_watcher = ConfigWatcher()

def get_global_watcher() -> ConfigWatcher:
    """Get the global configuration watcher instance."""
    return _global_watcher


def watch_config_file(filepath: str, callback: Callable[[SimulationConfig], None]) -> None:
    """
    Convenience function to watch a config file with the global watcher.

    Args:
        filepath: Path to watch
        callback: Callback function for config changes
    """
    _global_watcher.watch_file(filepath, callback)
    if not _global_watcher.running:
        _global_watcher.start()


def create_reloadable_config(
    config_or_path: Union[SimulationConfig, str],
    watch_path: Optional[str] = None
) -> ReloadableConfig:
    """
    Create a reloadable configuration.

    Args:
        config_or_path: Initial SimulationConfig or path to config file
        watch_path: Optional path to watch for changes (defaults to config file path)

    Returns:
        ReloadableConfig: Reloadable configuration wrapper
    """
    if isinstance(config_or_path, str):
        config = SimulationConfig.from_yaml(config_or_path)
        watch_path = watch_path or config_or_path
    else:
        config = config_or_path

    reloadable = ReloadableConfig(config, _global_watcher)

    if watch_path:
        reloadable.watch_file(watch_path)

    return reloadable

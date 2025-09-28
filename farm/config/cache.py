"""
Configuration caching and performance optimization.

This module provides caching mechanisms and performance optimizations
for the configuration system to ensure fast loading and efficient memory usage.
"""

import hashlib
import os
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .config import SimulationConfig


class ConfigCache:
    """
    Thread-safe configuration cache with automatic invalidation.

    Features:
    - LRU eviction policy
    - File modification time tracking
    - Memory usage limits
    - Thread-safe operations
    """

    def __init__(self, max_size: int = 50, max_memory_mb: float = 100.0):
        """
        Initialize the configuration cache.

        Args:
            max_size: Maximum number of cached configurations
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.file_mtimes: Dict[str, float] = {}
        self.memory_usage = 0.0
        self.lock = threading.RLock()

    def get(
        self, cache_key: str, filepath: Optional[str] = None
    ) -> Optional[SimulationConfig]:
        """
        Retrieve a configuration from cache.

        Args:
            cache_key: Unique cache key
            filepath: File path to check for modifications

        Returns:
            Cached configuration or None if not found or invalid
        """
        with self.lock:
            if cache_key not in self.cache:
                return None

            # Check if file has been modified
            if filepath and os.path.exists(filepath):
                current_mtime = os.path.getmtime(filepath)
                cached_mtime = self.file_mtimes.get(cache_key, 0)

                if current_mtime > cached_mtime:
                    # File modified, invalidate cache
                    self._remove_entry(cache_key)
                    return None

            # Update access time
            self.access_times[cache_key] = time.time()

            # Return cached config
            cached_data = self.cache[cache_key]
            try:
                return SimulationConfig.from_dict(cached_data["config"])
            except Exception:
                # Invalid cached data, remove it
                self._remove_entry(cache_key)
                return None

    def put(
        self, cache_key: str, config: SimulationConfig, filepath: Optional[str] = None
    ) -> None:
        """
        Store a configuration in cache.

        Args:
            cache_key: Unique cache key
            config: Configuration to cache
            filepath: Associated file path for modification tracking
        """
        with self.lock:
            # Estimate memory usage
            config_dict = config.to_dict()
            config_size = len(pickle.dumps(config_dict)) / (1024 * 1024)  # MB

            # Check memory limit
            if self.memory_usage + config_size > self.max_memory_mb:
                self._evict_to_fit(config_size)

            # Check size limit
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            # Store in cache
            self.cache[cache_key] = {
                "config": config_dict,
                "size": config_size,
                "created": time.time(),
            }
            self.access_times[cache_key] = time.time()
            self.memory_usage += config_size

            # Track file modification time
            if filepath and os.path.exists(filepath):
                self.file_mtimes[cache_key] = os.path.getmtime(filepath)

    def invalidate(self, cache_key: str) -> None:
        """
        Invalidate a specific cache entry.

        Args:
            cache_key: Cache key to invalidate
        """
        with self.lock:
            self._remove_entry(cache_key)

    def clear(self) -> None:
        """Clear all cached configurations."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.file_mtimes.clear()
            self.memory_usage = 0.0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            return {
                "entries": len(self.cache),
                "memory_usage_mb": self.memory_usage,
                "max_size": self.max_size,
                "max_memory_mb": self.max_memory_mb,
                "hit_rate": 0.0,  # Could be implemented with additional tracking
            }

    def _remove_entry(self, cache_key: str) -> None:
        """Remove a cache entry and update memory usage."""
        if cache_key in self.cache:
            self.memory_usage -= self.cache[cache_key]["size"]
            del self.cache[cache_key]

        self.access_times.pop(cache_key, None)
        self.file_mtimes.pop(cache_key, None)

    def _evict_lru(self) -> None:
        """Evict least recently used entries to make room."""
        if not self.access_times:
            return

        # Find least recently used
        lru_key = min(self.access_times, key=self.access_times.get)
        self._remove_entry(lru_key)

    def _evict_to_fit(self, required_size: float) -> None:
        """Evict entries until there's enough space for required_size."""
        while self.memory_usage + required_size > self.max_memory_mb and self.cache:
            self._evict_lru()


# Global cache instance
_global_cache = ConfigCache()


def get_global_cache() -> ConfigCache:
    """Get the global configuration cache instance."""
    return _global_cache


class OptimizedConfigLoader:
    """
    Optimized configuration loader with caching and performance enhancements.
    """

    def __init__(self, cache: Optional[ConfigCache] = None):
        """
        Initialize optimized loader.

        Args:
            cache: Cache instance to use (uses global cache if None)
        """
        self.cache = cache or _global_cache

    def load_centralized_config(
        self,
        environment: str = "development",
        profile: Optional[str] = None,
        config_dir: str = "farm/config",
        use_cache: bool = True,
    ) -> SimulationConfig:
        """
        Load centralized configuration with caching.

        Args:
            environment: Environment name
            profile: Optional profile name
            config_dir: Configuration directory
            use_cache: Whether to use caching

        Returns:
            Loaded configuration
        """
        if not use_cache:
            return self._load_from_files(environment, profile, config_dir)

        # Create cache key
        cache_key = self._create_cache_key(environment, profile, config_dir)

        # Try to get from cache
        base_config_path = os.path.join(config_dir, "default.yaml")
        config = self.cache.get(cache_key, base_config_path)
        if config is not None:
            return config

        # Load from files
        config = self._load_from_files(environment, profile, config_dir)

        # Cache the result with base config file path for modification tracking
        base_config_path = os.path.join(config_dir, "default.yaml")
        self.cache.put(cache_key, config, base_config_path)
        return config

    def _load_from_files(
        self, environment: str, profile: Optional[str], config_dir: str
    ) -> SimulationConfig:
        """
        Load configuration from files without caching.

        Args:
            environment: Environment name
            profile: Optional profile name
            config_dir: Configuration directory

        Returns:
            Loaded configuration
        """
        import os

        from .config import RedisMemoryConfig, VisualizationConfig

        # Load base configuration
        base_path = os.path.join(config_dir, "default.yaml")
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base configuration not found: {base_path}")

        with open(base_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)

        # Merge environment overrides
        env_path = os.path.join(config_dir, "environments", f"{environment}.yaml")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                env_config = yaml.safe_load(f)
            base_config = SimulationConfig._deep_merge(base_config, env_config)
        else:
            # If environment is not "default" and file doesn't exist, raise error
            if environment != "default":
                raise FileNotFoundError(
                    f"Environment configuration not found: {env_path}"
                )

        # Merge profile overrides (highest precedence)
        if profile:
            profile_path = os.path.join(config_dir, "profiles", f"{profile}.yaml")
            if os.path.exists(profile_path):
                with open(profile_path, "r", encoding="utf-8") as f:
                    profile_config = yaml.safe_load(f)
                base_config = SimulationConfig._deep_merge(base_config, profile_config)
            else:
                raise FileNotFoundError(
                    f"Profile configuration not found: {profile_path}"
                )

        # Handle nested configs
        vis_config = base_config.pop("visualization", {})
        base_config["visualization"] = VisualizationConfig(**vis_config)

        redis_config = base_config.pop("redis", {})
        base_config["redis"] = RedisMemoryConfig(**redis_config)

        obs_config = base_config.pop("observation", None)
        if obs_config:
            from farm.core.observations import ObservationConfig

            base_config["observation"] = ObservationConfig(**obs_config)

        return SimulationConfig(**base_config)

    def preload_common_configs(self, config_dir: str = "farm/config") -> None:
        """
        Preload commonly used configurations into cache.

        Args:
            config_dir: Configuration directory
        """
        common_configs = [
            ("development", None),
            ("production", None),
            ("testing", None),
            ("development", "benchmark"),
            ("production", "benchmark"),
        ]

        for environment, profile in common_configs:
            try:
                self.load_centralized_config(
                    environment=environment,
                    profile=profile,
                    config_dir=config_dir,
                    use_cache=True,
                )
            except Exception:
                # Skip configs that can't be loaded
                continue

    def _create_cache_key(
        self, environment: str, profile: Optional[str], config_dir: str
    ) -> str:
        """Create a unique cache key for the configuration."""
        key_parts = [config_dir, environment, profile or ""]
        key_string = "|".join(key_parts)

        # Include file modification times in key for automatic invalidation
        try:
            mtimes = []
            base_file = os.path.join(config_dir, "default.yaml")
            if os.path.exists(base_file):
                mtimes.append(str(os.path.getmtime(base_file)))

            env_file = os.path.join(config_dir, "environments", f"{environment}.yaml")
            if os.path.exists(env_file):
                mtimes.append(str(os.path.getmtime(env_file)))

            if profile:
                profile_file = os.path.join(config_dir, "profiles", f"{profile}.yaml")
                if os.path.exists(profile_file):
                    mtimes.append(str(os.path.getmtime(profile_file)))

            key_string += "|" + "|".join(mtimes)
        except Exception:
            # If we can't get mtimes, just use the basic key
            pass

        return hashlib.md5(key_string.encode()).hexdigest()


class LazyConfigLoader:
    """
    Lazy configuration loader that defers expensive operations.
    """

    def __init__(self, loader: Optional[OptimizedConfigLoader] = None):
        """
        Initialize lazy loader.

        Args:
            loader: Underlying loader to use
        """
        self.loader = loader or OptimizedConfigLoader()
        self._config: Optional[SimulationConfig] = None
        self._load_params: Optional[Dict[str, Any]] = None

    def configure(
        self,
        environment: str = "development",
        profile: Optional[str] = None,
        config_dir: str = "farm/config",
    ) -> "LazyConfigLoader":
        """
        Configure the loader parameters.

        Args:
            environment: Environment name
            profile: Optional profile name
            config_dir: Configuration directory

        Returns:
            Self for method chaining
        """
        self._load_params = {
            "environment": environment,
            "profile": profile,
            "config_dir": config_dir,
        }
        self._config = None  # Invalidate cached config
        return self

    def get_config(self) -> SimulationConfig:
        """
        Get the configuration, loading it if necessary.

        Returns:
            Configuration instance
        """
        if self._config is None:
            if self._load_params is None:
                raise ValueError("Loader not configured. Call configure() first.")

            self._config = self.loader.load_centralized_config(**self._load_params)

        return self._config

    def reload(self) -> SimulationConfig:
        """
        Force reload the configuration.

        Returns:
            Fresh configuration instance
        """
        if self._load_params is None:
            raise ValueError("Loader not configured. Call configure() first.")

        # Invalidate cache for this config
        cache_key = self.loader._create_cache_key(**self._load_params)
        self.loader.cache.invalidate(cache_key)

        # Reload
        self._config = self.loader.load_centralized_config(**self._load_params)
        return self._config

    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying config."""
        return getattr(self.get_config(), name)

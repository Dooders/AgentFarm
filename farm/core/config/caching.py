"""
Configuration caching and performance optimization system.

This module provides intelligent caching, lazy loading, and performance
optimization features for the hierarchical configuration system.
"""

import functools
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from weakref import WeakValueDictionary

from .exceptions import ConfigurationError
from .hierarchical import HierarchicalConfig

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Different caching strategies."""
    
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    SIZE = "size"  # Size-based eviction
    NONE = "none"  # No caching


class CacheEntry:
    """Individual cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, created_at: float = None):
        """Initialize cache entry.
        
        Args:
            key: Cache key
            value: Cached value
            created_at: Creation timestamp
        """
        self.key = key
        self.value = value
        self.created_at = created_at or time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.size_bytes = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """Calculate approximate size of the cached value.
        
        Returns:
            Size in bytes
        """
        try:
            import sys
            return sys.getsizeof(self.value)
        except Exception:
            return 0
    
    def access(self) -> None:
        """Record access to this cache entry."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def is_expired(self, ttl: float) -> bool:
        """Check if cache entry is expired.
        
        Args:
            ttl: Time to live in seconds
            
        Returns:
            True if expired, False otherwise
        """
        return time.time() - self.created_at > ttl


@dataclass
class CacheConfig:
    """Configuration for caching behavior."""
    
    strategy: CacheStrategy = CacheStrategy.LRU
    max_size: int = 1000  # Maximum number of entries
    max_memory_mb: int = 100  # Maximum memory usage in MB
    ttl_seconds: float = 3600  # Time to live in seconds
    enable_metrics: bool = True  # Enable performance metrics
    enable_compression: bool = False  # Enable value compression
    cleanup_interval: float = 300  # Cleanup interval in seconds


class CacheMetrics:
    """Performance metrics for caching system."""
    
    def __init__(self):
        """Initialize cache metrics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.insertions = 0
        self.updates = 0
        self.cleanups = 0
        self.total_access_time = 0.0
        self.total_size_bytes = 0
        self.lock = threading.Lock()
    
    def record_hit(self, access_time: float) -> None:
        """Record a cache hit.
        
        Args:
            access_time: Time taken to access the cache
        """
        with self.lock:
            self.hits += 1
            self.total_access_time += access_time
    
    def record_miss(self, access_time: float) -> None:
        """Record a cache miss.
        
        Args:
            access_time: Time taken to determine cache miss
        """
        with self.lock:
            self.misses += 1
            self.total_access_time += access_time
    
    def record_eviction(self) -> None:
        """Record a cache eviction."""
        with self.lock:
            self.evictions += 1
    
    def record_insertion(self, size_bytes: int) -> None:
        """Record a cache insertion.
        
        Args:
            size_bytes: Size of the inserted value
        """
        with self.lock:
            self.insertions += 1
            self.total_size_bytes += size_bytes
    
    def record_update(self, old_size: int, new_size: int) -> None:
        """Record a cache update.
        
        Args:
            old_size: Size of the old value
            new_size: Size of the new value
        """
        with self.lock:
            self.updates += 1
            self.total_size_bytes = self.total_size_bytes - old_size + new_size
    
    def record_cleanup(self) -> None:
        """Record a cache cleanup."""
        with self.lock:
            self.cleanups += 1
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate.
        
        Returns:
            Hit rate as a percentage (0.0 to 1.0)
        """
        with self.lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0
    
    def get_average_access_time(self) -> float:
        """Get average access time.
        
        Returns:
            Average access time in seconds
        """
        with self.lock:
            total = self.hits + self.misses
            return self.total_access_time / total if total > 0 else 0.0
    
    def get_memory_usage_mb(self) -> float:
        """Get memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        with self.lock:
            return self.total_size_bytes / (1024 * 1024)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            return {
                'hits': self.hits,
                'misses': self.misses,
                'evictions': self.evictions,
                'insertions': self.insertions,
                'updates': self.updates,
                'cleanups': self.cleanups,
                'hit_rate': self.get_hit_rate(),
                'average_access_time': self.get_average_access_time(),
                'memory_usage_mb': self.get_memory_usage_mb(),
                'total_size_bytes': self.total_size_bytes
            }


class ConfigurationCache:
    """Intelligent configuration cache with multiple eviction strategies."""
    
    def __init__(self, config: CacheConfig):
        """Initialize configuration cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.metrics = CacheMetrics() if config.enable_metrics else None
        self.lock = threading.RLock()
        self.cleanup_timer: Optional[threading.Timer] = None
        self._start_cleanup_timer()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        start_time = time.time()
        
        with self.lock:
            if key not in self.cache:
                if self.metrics:
                    self.metrics.record_miss(time.time() - start_time)
                return None
            
            entry = self.cache[key]
            
            # Check if expired
            if self.config.strategy == CacheStrategy.TTL and entry.is_expired(self.config.ttl_seconds):
                del self.cache[key]
                if self.metrics:
                    self.metrics.record_miss(time.time() - start_time)
                return None
            
            # Record access
            entry.access()
            
            if self.metrics:
                self.metrics.record_hit(time.time() - start_time)
            
            return entry.value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Check if we need to evict entries
            self._evict_if_needed()
            
            # Create cache entry
            entry = CacheEntry(key, value)
            
            # Check if updating existing entry
            if key in self.cache:
                old_entry = self.cache[key]
                if self.metrics:
                    self.metrics.record_update(old_entry.size_bytes, entry.size_bytes)
            else:
                if self.metrics:
                    self.metrics.record_insertion(entry.size_bytes)
            
            # Store entry
            self.cache[key] = entry
    
    def delete(self, key: str) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if key was deleted, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache limits are exceeded."""
        # Check size limit
        if len(self.cache) >= self.config.max_size:
            self._evict_by_strategy()
        
        # Check memory limit
        if self.metrics and self.metrics.get_memory_usage_mb() > self.config.max_memory_mb:
            self._evict_by_memory()
    
    def _evict_by_strategy(self) -> None:
        """Evict entries based on configured strategy."""
        if self.config.strategy == CacheStrategy.LRU:
            self._evict_lru()
        elif self.config.strategy == CacheStrategy.LFU:
            self._evict_lfu()
        elif self.config.strategy == CacheStrategy.SIZE:
            self._evict_largest()
        else:
            # Default to LRU
            self._evict_lru()
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_accessed)
        del self.cache[lru_key]
        
        if self.metrics:
            self.metrics.record_eviction()
    
    def _evict_lfu(self) -> None:
        """Evict least frequently used entry."""
        if not self.cache:
            return
        
        lfu_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        del self.cache[lfu_key]
        
        if self.metrics:
            self.metrics.record_eviction()
    
    def _evict_largest(self) -> None:
        """Evict largest entry by size."""
        if not self.cache:
            return
        
        largest_key = max(self.cache.keys(), key=lambda k: self.cache[k].size_bytes)
        del self.cache[largest_key]
        
        if self.metrics:
            self.metrics.record_eviction()
    
    def _evict_by_memory(self) -> None:
        """Evict entries to reduce memory usage."""
        # Sort by size (largest first) and evict until under limit
        sorted_entries = sorted(
            self.cache.items(),
            key=lambda x: x[1].size_bytes,
            reverse=True
        )
        
        target_memory_mb = self.config.max_memory_mb * 0.8  # Evict to 80% of limit
        
        for key, entry in sorted_entries:
            if self.metrics and self.metrics.get_memory_usage_mb() <= target_memory_mb:
                break
            
            del self.cache[key]
            if self.metrics:
                self.metrics.record_eviction()
    
    def _start_cleanup_timer(self) -> None:
        """Start periodic cleanup timer."""
        if self.config.strategy == CacheStrategy.TTL:
            self.cleanup_timer = threading.Timer(
                self.config.cleanup_interval,
                self._cleanup_expired
            )
            self.cleanup_timer.daemon = True
            self.cleanup_timer.start()
    
    def _cleanup_expired(self) -> None:
        """Clean up expired entries."""
        with self.lock:
            expired_keys = [
                key for key, entry in self.cache.items()
                if entry.is_expired(self.config.ttl_seconds)
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            if self.metrics and expired_keys:
                self.metrics.record_cleanup()
        
        # Schedule next cleanup
        self._start_cleanup_timer()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            stats = {
                'cache_size': len(self.cache),
                'max_size': self.config.max_size,
                'strategy': self.config.strategy.value,
                'ttl_seconds': self.config.ttl_seconds
            }
            
            if self.metrics:
                stats.update(self.metrics.get_stats())
            
            return stats
    
    def shutdown(self) -> None:
        """Shutdown cache and cleanup resources."""
        if self.cleanup_timer:
            self.cleanup_timer.cancel()
        
        with self.lock:
            self.cache.clear()


class LazyConfigurationLoader:
    """Lazy loader for configuration sections."""
    
    def __init__(self, loader_func: Callable[[], Any], cache: Optional[ConfigurationCache] = None):
        """Initialize lazy loader.
        
        Args:
            loader_func: Function to load configuration
            cache: Optional cache for loaded configurations
        """
        self.loader_func = loader_func
        self.cache = cache
        self._value: Optional[Any] = None
        self._loaded = False
        self.lock = threading.Lock()
    
    def get(self) -> Any:
        """Get configuration value, loading if necessary.
        
        Returns:
            Loaded configuration value
        """
        if self._loaded:
            return self._value
        
        with self.lock:
            if self._loaded:
                return self._value
            
            # Check cache first
            if self.cache:
                cache_key = f"lazy_loader_{id(self)}"
                cached_value = self.cache.get(cache_key)
                if cached_value is not None:
                    self._value = cached_value
                    self._loaded = True
                    return self._value
            
            # Load configuration
            try:
                self._value = self.loader_func()
                self._loaded = True
                
                # Cache the result
                if self.cache:
                    self.cache.put(cache_key, self._value)
                
                return self._value
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def reload(self) -> Any:
        """Force reload configuration.
        
        Returns:
            Reloaded configuration value
        """
        with self.lock:
            self._loaded = False
            self._value = None
            
            # Clear from cache
            if self.cache:
                cache_key = f"lazy_loader_{id(self)}"
                self.cache.delete(cache_key)
            
            return self.get()
    
    def is_loaded(self) -> bool:
        """Check if configuration is loaded.
        
        Returns:
            True if loaded, False otherwise
        """
        return self._loaded


class CachedHierarchicalConfig(HierarchicalConfig):
    """Hierarchical configuration with caching support."""
    
    def __init__(
        self,
        global_config: Dict[str, Any] = None,
        environment_config: Dict[str, Any] = None,
        agent_config: Dict[str, Any] = None,
        cache_config: Optional[CacheConfig] = None
    ):
        """Initialize cached hierarchical configuration.
        
        Args:
            global_config: Global configuration dictionary
            environment_config: Environment-specific configuration
            agent_config: Agent-specific configuration
            cache_config: Cache configuration
        """
        super().__init__(global_config, environment_config, agent_config)
        
        self.cache_config = cache_config or CacheConfig()
        self.cache = ConfigurationCache(self.cache_config)
        self.lazy_loaders: Dict[str, LazyConfigurationLoader] = {}
    
    def get(self, key: str, default=None) -> Any:
        """Get configuration value with caching.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Check cache first
        cached_value = self.cache.get(key)
        if cached_value is not None:
            return cached_value
        
        # Get value from hierarchical lookup
        value = super().get(key, default)
        
        # Cache the result
        self.cache.put(key, value)
        
        return value
    
    def get_nested(self, key: str, default=None) -> Any:
        """Get nested configuration value with caching.
        
        Args:
            key: Dot-separated key path
            default: Default value if key not found
            
        Returns:
            Nested configuration value
        """
        # Check cache first
        cache_key = f"nested_{key}"
        cached_value = self.cache.get(cache_key)
        if cached_value is not None:
            return cached_value
        
        # Get value from hierarchical lookup
        value = super().get_nested(key, default)
        
        # Cache the result
        self.cache.put(cache_key, value)
        
        return value
    
    def add_lazy_loader(self, key: str, loader_func: Callable[[], Any]) -> None:
        """Add lazy loader for a configuration key.
        
        Args:
            key: Configuration key
            loader_func: Function to load configuration value
        """
        self.lazy_loaders[key] = LazyConfigurationLoader(loader_func, self.cache)
    
    def get_lazy(self, key: str) -> Any:
        """Get configuration value using lazy loading.
        
        Args:
            key: Configuration key
            
        Returns:
            Loaded configuration value
        """
        if key in self.lazy_loaders:
            return self.lazy_loaders[key].get()
        
        # Fall back to regular get
        return self.get(key)
    
    def reload_lazy(self, key: str) -> Any:
        """Force reload lazy-loaded configuration.
        
        Args:
            key: Configuration key
            
        Returns:
            Reloaded configuration value
        """
        if key in self.lazy_loaders:
            return self.lazy_loaders[key].reload()
        
        # Clear from cache and reload
        self.cache.delete(key)
        return self.get(key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self.cache.clear()
    
    def shutdown(self) -> None:
        """Shutdown configuration and cleanup resources."""
        self.cache.shutdown()


def cached_config_method(cache_key: str = None):
    """Decorator for caching configuration method results.
    
    Args:
        cache_key: Optional custom cache key
        
    Returns:
        Decorated method
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key
            if cache_key:
                key = cache_key
            else:
                key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            
            # Check if instance has cache
            if hasattr(self, 'cache') and self.cache:
                cached_value = self.cache.get(key)
                if cached_value is not None:
                    return cached_value
            
            # Execute function
            result = func(self, *args, **kwargs)
            
            # Cache result
            if hasattr(self, 'cache') and self.cache:
                self.cache.put(key, result)
            
            return result
        
        return wrapper
    return decorator


class ConfigurationPerformanceProfiler:
    """Performance profiler for configuration operations."""
    
    def __init__(self):
        """Initialize performance profiler."""
        self.operation_times: Dict[str, List[float]] = {}
        self.operation_counts: Dict[str, int] = {}
        self.lock = threading.Lock()
    
    def profile_operation(self, operation_name: str):
        """Decorator to profile configuration operations.
        
        Args:
            operation_name: Name of the operation to profile
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    with self.lock:
                        if operation_name not in self.operation_times:
                            self.operation_times[operation_name] = []
                            self.operation_counts[operation_name] = 0
                        
                        self.operation_times[operation_name].append(duration)
                        self.operation_counts[operation_name] += 1
            
            return wrapper
        return decorator
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        with self.lock:
            stats = {}
            
            for operation, times in self.operation_times.items():
                if times:
                    stats[operation] = {
                        'count': self.operation_counts[operation],
                        'total_time': sum(times),
                        'average_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'last_time': times[-1] if times else 0
                    }
            
            return stats
    
    def clear_stats(self) -> None:
        """Clear performance statistics."""
        with self.lock:
            self.operation_times.clear()
            self.operation_counts.clear()
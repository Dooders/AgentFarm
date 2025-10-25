"""
Performance optimization system for large-scale simulation analysis.

This module provides performance optimization capabilities including caching,
memory management, parallel processing, and resource monitoring.
"""

import gc
import os
import psutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
import pickle
import hashlib
import json
import threading
from queue import Queue, Empty

from farm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Caching
    enable_caching: bool = True
    cache_dir: Union[str, Path] = "analysis_cache"
    cache_ttl: int = 3600  # seconds
    max_cache_size: int = 1024 * 1024 * 1024  # 1GB
    
    # Memory management
    max_memory_usage: float = 0.8  # 80% of available memory
    memory_cleanup_threshold: float = 0.7  # 70% of max memory
    enable_memory_monitoring: bool = True
    
    # Parallel processing
    max_workers: int = None  # Auto-detect
    use_multiprocessing: bool = True
    chunk_size: int = 100
    
    # Resource monitoring
    enable_resource_monitoring: bool = True
    monitoring_interval: float = 1.0  # seconds
    resource_log_file: Optional[str] = None
    
    # Performance tuning
    enable_profiling: bool = False
    profile_output_dir: Union[str, Path] = "profiles"
    optimization_level: int = 2  # 0=disabled, 1=basic, 2=aggressive


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    active_threads: int
    active_processes: int


@dataclass
class PerformanceProfile:
    """Performance profile for analysis operations."""
    
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_usage_percent: float
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_workers: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceOptimizer:
    """Performance optimization system for simulation analysis."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize the performance optimizer."""
        self.config = config or PerformanceConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize resource monitoring
        self._resource_monitor_thread = None
        self._resource_metrics: List[ResourceMetrics] = []
        self._monitoring_active = False
        self._resource_queue = Queue()
        
        # Performance tracking
        self._performance_profiles: List[PerformanceProfile] = []
        self._operation_timers: Dict[str, float] = {}
        
        # Memory management
        self._memory_cleanup_lock = threading.Lock()
        self._last_cleanup = datetime.now()
        
        # Auto-detect optimal settings
        self._auto_configure()
        
        logger.info("PerformanceOptimizer initialized")
    
    def _auto_configure(self):
        """Auto-configure performance settings based on system resources."""
        if self.config.max_workers is None:
            self.config.max_workers = min(os.cpu_count() or 4, 8)
        
        # Adjust chunk size based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if available_memory_gb < 4:
            self.config.chunk_size = 50
        elif available_memory_gb < 8:
            self.config.chunk_size = 100
        else:
            self.config.chunk_size = 200
        
        logger.info(f"Auto-configured: max_workers={self.config.max_workers}, chunk_size={self.config.chunk_size}")
    
    def start_resource_monitoring(self):
        """Start resource monitoring in background thread."""
        if not self.config.enable_resource_monitoring:
            return
        
        self._monitoring_active = True
        self._resource_monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self._resource_monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_resource_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring_active = False
        if self._resource_monitor_thread:
            self._resource_monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self):
        """Monitor system resources in background."""
        while self._monitoring_active:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                metrics = ResourceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used_mb=memory.used / (1024**2),
                    memory_available_mb=memory.available / (1024**2),
                    disk_usage_percent=disk.percent,
                    active_threads=threading.active_count(),
                    active_processes=len(psutil.pids())
                )
                
                self._resource_metrics.append(metrics)
                
                # Keep only recent metrics (last hour)
                cutoff_time = datetime.now() - timedelta(hours=1)
                self._resource_metrics = [
                    m for m in self._resource_metrics 
                    if m.timestamp > cutoff_time
                ]
                
                # Check memory usage and trigger cleanup if needed
                if memory.percent > self.config.memory_cleanup_threshold * 100:
                    self._trigger_memory_cleanup()
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _trigger_memory_cleanup(self):
        """Trigger memory cleanup if needed."""
        with self._memory_cleanup_lock:
            now = datetime.now()
            if (now - self._last_cleanup).total_seconds() < 30:  # Don't cleanup too frequently
                return
            
            logger.info("Triggering memory cleanup")
            gc.collect()
            self._last_cleanup = now
    
    def profile_operation(self, operation_name: str):
        """Context manager for profiling operations."""
        return OperationProfiler(self, operation_name)
    
    def optimize_parallel_execution(self, 
                                  tasks: List[Callable], 
                                  data_chunks: List[Any],
                                  operation_name: str = "parallel_execution") -> List[Any]:
        """Optimize parallel execution of tasks."""
        if not tasks or not data_chunks:
            return []
        
        start_time = time.time()
        memory_before = psutil.virtual_memory().used
        
        try:
            # Determine optimal execution strategy
            if len(tasks) == 1 or not self.config.use_multiprocessing:
                # Single task or multiprocessing disabled
                results = self._execute_sequential(tasks, data_chunks)
            else:
                # Parallel execution
                results = self._execute_parallel(tasks, data_chunks)
            
            # Record performance metrics
            end_time = time.time()
            memory_after = psutil.virtual_memory().used
            
            profile = PerformanceProfile(
                operation_name=operation_name,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.fromtimestamp(end_time),
                duration=end_time - start_time,
                memory_peak_mb=psutil.virtual_memory().used / (1024**2),
                memory_delta_mb=(memory_after - memory_before) / (1024**2),
                cpu_usage_percent=psutil.cpu_percent(),
                parallel_workers=self.config.max_workers if self.config.use_multiprocessing else 1
            )
            
            self._performance_profiles.append(profile)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in parallel execution: {e}")
            raise
    
    def _execute_sequential(self, tasks: List[Callable], data_chunks: List[Any]) -> List[Any]:
        """Execute tasks sequentially."""
        results = []
        for task, data in zip(tasks, data_chunks):
            result = task(data)
            results.append(result)
        return results
    
    def _execute_parallel(self, tasks: List[Callable], data_chunks: List[Any]) -> List[Any]:
        """Execute tasks in parallel."""
        results = [None] * len(tasks)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(task, data): i 
                for i, (task, data) in enumerate(zip(tasks, data_chunks))
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Task {index} failed: {e}")
                    results[index] = None
        
        return results
    
    def cache_result(self, key: str, result: Any, ttl: Optional[int] = None) -> bool:
        """Cache a result."""
        if not self.config.enable_caching:
            return False
        
        try:
            cache_ttl = ttl or self.config.cache_ttl
            cache_data = {
                'result': result,
                'timestamp': time.time(),
                'ttl': cache_ttl
            }
            
            # Generate cache file path
            cache_key = hashlib.md5(key.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            # Save to cache
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            # Cleanup old cache files if needed
            self._cleanup_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")
            return False
    
    def get_cached_result(self, key: str) -> Optional[Any]:
        """Get a cached result."""
        if not self.config.enable_caching:
            return None
        
        try:
            cache_key = hashlib.md5(key.encode()).hexdigest()
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if not cache_file.exists():
                return None
            
            # Load cache data
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check TTL
            if time.time() - cache_data['timestamp'] > cache_data['ttl']:
                cache_file.unlink()  # Remove expired cache
                return None
            
            return cache_data['result']
            
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
    
    def _cleanup_cache(self):
        """Cleanup old cache files."""
        try:
            current_time = time.time()
            total_size = 0
            files_to_remove = []
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    # Check file age
                    file_age = current_time - cache_file.stat().st_mtime
                    if file_age > self.config.cache_ttl:
                        files_to_remove.append(cache_file)
                        continue
                    
                    # Check file size
                    file_size = cache_file.stat().st_size
                    total_size += file_size
                    
                except OSError:
                    files_to_remove.append(cache_file)
            
            # Remove old files
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                except OSError:
                    pass
            
            # Remove largest files if cache is too big
            if total_size > self.config.max_cache_size:
                remaining_files = [
                    (f, f.stat().st_size) for f in self.cache_dir.glob("*.pkl")
                    if f not in files_to_remove
                ]
                remaining_files.sort(key=lambda x: x[1], reverse=True)
                
                current_size = total_size
                for file_path, file_size in remaining_files:
                    if current_size <= self.config.max_cache_size:
                        break
                    
                    try:
                        file_path.unlink()
                        current_size -= file_size
                    except OSError:
                        pass
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    def chunk_data(self, data: List[Any], chunk_size: Optional[int] = None) -> List[List[Any]]:
        """Split data into chunks for parallel processing."""
        chunk_size = chunk_size or self.config.chunk_size
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self._performance_profiles:
            return {"message": "No performance data available"}
        
        total_duration = sum(p.duration for p in self._performance_profiles)
        avg_duration = total_duration / len(self._performance_profiles)
        
        memory_peaks = [p.memory_peak_mb for p in self._performance_profiles]
        max_memory = max(memory_peaks) if memory_peaks else 0
        
        return {
            "total_operations": len(self._performance_profiles),
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "max_memory_usage_mb": max_memory,
            "operations": [
                {
                    "name": p.operation_name,
                    "duration": p.duration,
                    "memory_peak_mb": p.memory_peak_mb,
                    "parallel_workers": p.parallel_workers
                } for p in self._performance_profiles
            ],
            "resource_metrics": self._get_resource_summary()
        }
    
    def _get_resource_summary(self) -> Dict[str, Any]:
        """Get resource usage summary."""
        if not self._resource_metrics:
            return {"message": "No resource data available"}
        
        recent_metrics = self._resource_metrics[-100:]  # Last 100 measurements
        
        return {
            "avg_cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            "avg_memory_percent": sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            "max_memory_used_mb": max(m.memory_used_mb for m in recent_metrics),
            "avg_active_threads": sum(m.active_threads for m in recent_metrics) / len(recent_metrics),
            "measurements_count": len(recent_metrics)
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent
            },
            "disk": {
                "total_gb": disk.total / (1024**3),
                "free_gb": disk.free / (1024**3),
                "used_gb": disk.used / (1024**3),
                "percent": disk.percent
            },
            "processes": len(psutil.pids()),
            "threads": threading.active_count()
        }


class OperationProfiler:
    """Context manager for profiling operations."""
    
    def __init__(self, optimizer: PerformanceOptimizer, operation_name: str):
        self.optimizer = optimizer
        self.operation_name = operation_name
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            
            profile = PerformanceProfile(
                operation_name=self.operation_name,
                start_time=datetime.fromtimestamp(self.start_time),
                end_time=datetime.fromtimestamp(end_time),
                duration=end_time - self.start_time,
                memory_peak_mb=end_memory / (1024**2),
                memory_delta_mb=(end_memory - self.start_memory) / (1024**2),
                cpu_usage_percent=psutil.cpu_percent()
            )
            
            self.optimizer._performance_profiles.append(profile)
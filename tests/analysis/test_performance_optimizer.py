"""
Tests for performance optimizer.

This module contains unit tests for the performance optimization system
that handles caching, memory management, and parallel processing.
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from farm.analysis.comparative.performance_optimizer import (
    PerformanceOptimizer,
    PerformanceConfig,
    ResourceMetrics,
    PerformanceProfile,
    OperationProfiler
)


class TestPerformanceOptimizer:
    """Test cases for PerformanceOptimizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = PerformanceConfig(
            enable_caching=True,
            cache_dir="test_cache",
            max_workers=2,
            enable_resource_monitoring=True,
            monitoring_interval=0.1
        )
        
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network, \
             patch('psutil.getloadavg') as mock_load, \
             patch('psutil.pids') as mock_pids:
            
            # Setup mock return values
            mock_memory.return_value = Mock(
                percent=50.0,
                used=4 * 1024**3,  # 4GB
                available=4 * 1024**3  # 4GB
            )
            mock_cpu.return_value = 25.0
            mock_disk.return_value = Mock(percent=60.0, free=100 * 1024**3)
            mock_network.return_value = Mock(bytes_sent=1024**2, bytes_recv=1024**2)
            mock_load.return_value = (1.0, 1.5, 2.0)
            mock_pids.return_value = list(range(100))
            
            self.optimizer = PerformanceOptimizer(self.config)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        assert self.optimizer.config == self.config
        assert self.optimizer.cache_dir == Path("test_cache")
        assert self.optimizer.cache_dir.exists()
        assert self.optimizer._performance_profiles == []
        assert self.optimizer._operation_timers == {}
    
    def test_auto_configure(self):
        """Test automatic configuration."""
        # Test with different system configurations
        with patch('os.cpu_count', return_value=8), \
             patch('psutil.virtual_memory') as mock_memory:
            
            mock_memory.return_value = Mock(available=2 * 1024**3)  # 2GB
            
            config = PerformanceConfig()
            optimizer = PerformanceOptimizer(config)
            
            assert optimizer.config.max_workers == 8
            assert optimizer.config.chunk_size == 50  # Low memory
    
    def test_start_stop_resource_monitoring(self):
        """Test resource monitoring start/stop."""
        # Start monitoring
        self.optimizer.start_resource_monitoring()
        assert self.optimizer._monitoring_active is True
        assert self.optimizer._resource_monitor_thread is not None
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop monitoring
        self.optimizer.stop_resource_monitoring()
        assert self.optimizer._monitoring_active is False
    
    def test_collect_system_metrics(self):
        """Test system metrics collection."""
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_io_counters') as mock_network, \
             patch('psutil.getloadavg') as mock_load, \
             patch('psutil.pids') as mock_pids:
            
            # Setup mock return values
            mock_memory.return_value = Mock(
                percent=75.0,
                used=6 * 1024**3,
                available=2 * 1024**3,
                total=8 * 1024**3
            )
            mock_cpu.return_value = 50.0
            mock_disk.return_value = Mock(
                percent=80.0, 
                free=20 * 1024**3,
                total=100 * 1024**3,
                used=80 * 1024**3
            )
            mock_network.return_value = Mock(bytes_sent=2048**2, bytes_recv=1024**2)
            mock_load.return_value = (2.0, 2.5, 3.0)
            mock_pids.return_value = list(range(150))
            
            metrics = self.optimizer.get_system_resources()
            
        assert isinstance(metrics, dict)
        assert metrics["cpu_percent"] == 50.0
        assert metrics["memory"]["percent"] == 75.0
        assert metrics["disk"]["percent"] == 80.0
        assert metrics["processes"] == 150
    
    def test_optimize_parallel_execution(self):
        """Test parallel execution optimization."""
        def test_task(data):
            time.sleep(0.01)  # Small delay
            return data * 2
        
        tasks = [test_task] * 3
        data_chunks = [1, 2, 3]
        
        results = self.optimizer.optimize_parallel_execution(
            tasks, data_chunks, "test_operation"
        )
        
        assert len(results) == 3
        assert results[0] == 2
        assert results[1] == 4
        assert results[2] == 6
        
        # Check that performance profile was created
        assert len(self.optimizer._performance_profiles) == 1
        profile = self.optimizer._performance_profiles[0]
        assert profile.operation_name == "test_operation"
        assert profile.duration > 0
    
    def test_cache_result(self):
        """Test result caching."""
        key = "test_key"
        result = {"data": "test_value"}
        
        # Cache the result
        success = self.optimizer.cache_result(key, result)
        assert success is True
        
        # Retrieve the result
        cached_result = self.optimizer.get_cached_result(key)
        assert cached_result == result
    
    def test_get_cached_result_nonexistent(self):
        """Test getting non-existent cached result."""
        result = self.optimizer.get_cached_result("nonexistent_key")
        assert result is None
    
    def test_get_cached_result_expired(self):
        """Test getting expired cached result."""
        key = "expired_key"
        result = {"data": "test_value"}
        
        # Cache with short TTL
        success = self.optimizer.cache_result(key, result, ttl=1)
        assert success is True
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Try to retrieve
        cached_result = self.optimizer.get_cached_result(key)
        assert cached_result is None
    
    def test_chunk_data(self):
        """Test data chunking."""
        data = list(range(25))
        chunks = self.optimizer.chunk_data(data, chunk_size=10)
        
        assert len(chunks) == 3
        assert chunks[0] == list(range(10))
        assert chunks[1] == list(range(10, 20))
        assert chunks[2] == list(range(20, 25))
    
    def test_chunk_data_default_size(self):
        """Test data chunking with default chunk size."""
        data = list(range(100))
        chunks = self.optimizer.chunk_data(data)
        
        # Should use default chunk size from config
        expected_chunks = (100 + self.config.chunk_size - 1) // self.config.chunk_size
        assert len(chunks) == expected_chunks
    
    def test_get_performance_summary(self):
        """Test performance summary generation."""
        # Add some performance profiles
        profile1 = PerformanceProfile(
            operation_name="op1",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration=10.0,
            memory_peak_mb=100.0,
            memory_delta_mb=50.0,
            cpu_usage_percent=25.0
        )
        
        profile2 = PerformanceProfile(
            operation_name="op2",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration=5.0,
            memory_peak_mb=200.0,
            memory_delta_mb=100.0,
            cpu_usage_percent=50.0
        )
        
        self.optimizer._performance_profiles = [profile1, profile2]
        
        summary = self.optimizer.get_performance_summary()
        
        assert summary["total_operations"] == 2
        assert summary["total_duration"] == 15.0
        assert summary["average_duration"] == 7.5
        assert summary["max_memory_usage_mb"] == 200.0
        assert len(summary["operations"]) == 2
    
    def test_get_performance_summary_empty(self):
        """Test performance summary with no data."""
        summary = self.optimizer.get_performance_summary()
        
        assert "message" in summary
        assert "No performance data available" in summary["message"]
    
    def test_get_system_resources(self):
        """Test system resources retrieval."""
        with patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.pids') as mock_pids:
            
            mock_memory.return_value = Mock(
                total=8 * 1024**3,
                available=4 * 1024**3,
                used=4 * 1024**3,
                percent=50.0
            )
            mock_cpu.return_value = 30.0
            mock_disk.return_value = Mock(
                total=100 * 1024**3,
                free=40 * 1024**3,
                used=60 * 1024**3,
                percent=60.0
            )
            mock_pids.return_value = list(range(200))
            
            resources = self.optimizer.get_system_resources()
            
            assert "cpu_percent" in resources
            assert "memory" in resources
            assert "disk" in resources
            assert "processes" in resources
            assert "threads" in resources
            
            assert resources["cpu_percent"] == 30.0
            assert resources["memory"]["percent"] == 50.0
            assert resources["disk"]["percent"] == 60.0
            assert resources["processes"] == 200
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Add some cache files
        cache_file1 = self.optimizer.cache_dir / "test1.pkl"
        cache_file2 = self.optimizer.cache_dir / "test2.pkl"
        
        cache_file1.write_bytes(b"test1")
        cache_file2.write_bytes(b"test2")
        
        assert cache_file1.exists()
        assert cache_file2.exists()
        
        # Clear cache
        self.optimizer.clear_cache()
        
        # Check that cache files are gone
        assert not cache_file1.exists()
        assert not cache_file2.exists()
    
    def test_trigger_memory_cleanup(self):
        """Test memory cleanup triggering."""
        with patch('gc.collect') as mock_gc:
            # Reset the last cleanup time to ensure cleanup can be triggered
            self.optimizer._last_cleanup = datetime.now() - timedelta(seconds=60)
            self.optimizer._trigger_memory_cleanup()
            mock_gc.assert_called_once()
    
    def test_operation_profiler_context_manager(self):
        """Test operation profiler as context manager."""
        with self.optimizer.profile_operation("test_op") as profiler:
            time.sleep(0.01)
        
        # Check that profile was created
        assert len(self.optimizer._performance_profiles) == 1
        profile = self.optimizer._performance_profiles[0]
        assert profile.operation_name == "test_op"
        assert profile.duration > 0


class TestPerformanceConfig:
    """Test cases for PerformanceConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PerformanceConfig()
        
        assert config.enable_caching is True
        assert config.cache_dir == "analysis_cache"
        assert config.cache_ttl == 3600
        assert config.max_cache_size == 1024 * 1024 * 1024
        assert config.max_memory_usage == 0.8
        assert config.memory_cleanup_threshold == 0.7
        assert config.enable_memory_monitoring is True
        assert config.max_workers is None
        assert config.use_multiprocessing is True
        assert config.chunk_size == 100
        assert config.enable_resource_monitoring is True
        assert config.monitoring_interval == 1.0
        assert config.resource_log_file is None
        assert config.enable_profiling is False
        assert config.profile_output_dir == "profiles"
        assert config.optimization_level == 2
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PerformanceConfig(
            enable_caching=False,
            cache_dir="custom_cache",
            cache_ttl=7200,
            max_cache_size=2048 * 1024 * 1024,
            max_memory_usage=0.9,
            memory_cleanup_threshold=0.8,
            enable_memory_monitoring=False,
            max_workers=8,
            use_multiprocessing=False,
            chunk_size=200,
            enable_resource_monitoring=False,
            monitoring_interval=2.0,
            resource_log_file="custom.log",
            enable_profiling=True,
            profile_output_dir="custom_profiles",
            optimization_level=1
        )
        
        assert config.enable_caching is False
        assert config.cache_dir == "custom_cache"
        assert config.cache_ttl == 7200
        assert config.max_cache_size == 2048 * 1024 * 1024
        assert config.max_memory_usage == 0.9
        assert config.memory_cleanup_threshold == 0.8
        assert config.enable_memory_monitoring is False
        assert config.max_workers == 8
        assert config.use_multiprocessing is False
        assert config.chunk_size == 200
        assert config.enable_resource_monitoring is False
        assert config.monitoring_interval == 2.0
        assert config.resource_log_file == "custom.log"
        assert config.enable_profiling is True
        assert config.profile_output_dir == "custom_profiles"
        assert config.optimization_level == 1


class TestResourceMetrics:
    """Test cases for ResourceMetrics."""
    
    def test_resource_metrics_creation(self):
        """Test resource metrics creation."""
        timestamp = datetime.now()
        
        metrics = ResourceMetrics(
            timestamp=timestamp,
            cpu_percent=25.5,
            memory_percent=60.0,
            memory_used_mb=2048.0,
            memory_available_mb=1365.0,
            disk_usage_percent=45.0,
            active_threads=300,
            active_processes=150
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_percent == 25.5
        assert metrics.memory_percent == 60.0
        assert metrics.memory_used_mb == 2048.0
        assert metrics.memory_available_mb == 1365.0
        assert metrics.disk_usage_percent == 45.0
        assert metrics.active_threads == 300
        assert metrics.active_processes == 150


class TestPerformanceProfile:
    """Test cases for PerformanceProfile."""
    
    def test_performance_profile_creation(self):
        """Test performance profile creation."""
        start_time = datetime.now()
        end_time = datetime.now()
        
        profile = PerformanceProfile(
            operation_name="test_operation",
            start_time=start_time,
            end_time=end_time,
            duration=10.5,
            memory_peak_mb=512.0,
            memory_delta_mb=256.0,
            cpu_usage_percent=75.0,
            cache_hits=10,
            cache_misses=5,
            parallel_workers=4,
            metadata={"key": "value"}
        )
        
        assert profile.operation_name == "test_operation"
        assert profile.start_time == start_time
        assert profile.end_time == end_time
        assert profile.duration == 10.5
        assert profile.memory_peak_mb == 512.0
        assert profile.memory_delta_mb == 256.0
        assert profile.cpu_usage_percent == 75.0
        assert profile.cache_hits == 10
        assert profile.cache_misses == 5
        assert profile.parallel_workers == 4
        assert profile.metadata == {"key": "value"}


class TestOperationProfiler:
    """Test cases for OperationProfiler."""
    
    def test_operation_profiler_context_manager(self):
        """Test operation profiler as context manager."""
        optimizer = PerformanceOptimizer()
        
        with optimizer.profile_operation("test_op") as profiler:
            assert profiler.optimizer == optimizer
            assert profiler.operation_name == "test_op"
            assert profiler.start_time is not None
            assert profiler.start_memory is not None
        
        # Check that profile was added
        assert len(optimizer._performance_profiles) == 1
        profile = optimizer._performance_profiles[0]
        assert profile.operation_name == "test_op"
        assert profile.duration > 0
    
    def test_operation_profiler_with_exception(self):
        """Test operation profiler with exception."""
        optimizer = PerformanceOptimizer()
        
        try:
            with optimizer.profile_operation("test_op") as profiler:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # Check that profile was still added
        assert len(optimizer._performance_profiles) == 1
        profile = optimizer._performance_profiles[0]
        assert profile.operation_name == "test_op"
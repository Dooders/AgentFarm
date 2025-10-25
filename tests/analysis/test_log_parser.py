"""
Tests for LogParser class.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from farm.analysis.comparative.log_parser import LogParser, LogMetrics


class TestLogParser:
    """Test cases for LogParser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = LogParser()
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_log_file(self, content: str, filename: str = "test.log") -> Path:
        """Create a test log file with given content."""
        log_path = self.temp_dir / filename
        with open(log_path, 'w') as f:
            f.write(content)
        return log_path
    
    def test_init(self):
        """Test parser initialization."""
        assert self.parser.performance_patterns
        assert self.parser.error_patterns
        assert self.parser.timing_patterns
        
        # Check that patterns are compiled
        assert hasattr(self.parser.performance_patterns['execution_time'], 'search')
        assert hasattr(self.parser.error_patterns['error'], 'search')
        assert hasattr(self.parser.timing_patterns['timestamp'], 'search')
    
    def test_parse_performance_metrics_success(self):
        """Test successful performance metrics parsing."""
        log_content = """2024-01-01T00:00:00Z INFO: Simulation started
2024-01-01T00:00:01Z INFO: Execution time: 1.5 seconds
2024-01-01T00:00:02Z INFO: Memory usage: 128 MB
2024-01-01T00:00:03Z INFO: Throughput: 100 ops/sec
2024-01-01T00:00:04Z INFO: Iteration 1 completed
2024-01-01T00:00:05Z INFO: Agents: 100
2024-01-01T00:00:06Z INFO: Step 1 completed
"""
        log_path = self.create_test_log_file(log_content)
        
        metrics = self.parser.parse_performance_metrics([log_path])
        
        assert "file_metrics" in metrics
        assert "aggregated_metrics" in metrics
        assert "summary" in metrics
        
        file_metrics = metrics["file_metrics"]["test.log"]
        assert "execution_time" in file_metrics
        assert "memory_usage" in file_metrics
        assert "throughput" in file_metrics
        assert "iterations" in file_metrics
        assert "agents" in file_metrics
        assert "steps" in file_metrics
    
    def test_parse_performance_metrics_multiple_files(self):
        """Test parsing performance metrics from multiple files."""
        log1_content = "Execution time: 1.0 seconds\nMemory usage: 64 MB"
        log2_content = "Execution time: 2.0 seconds\nThroughput: 50 ops/sec"
        
        log1_path = self.create_test_log_file(log1_content, "log1.log")
        log2_path = self.create_test_log_file(log2_content, "log2.log")
        
        metrics = self.parser.parse_performance_metrics([log1_path, log2_path])
        
        assert len(metrics["file_metrics"]) == 2
        assert "log1.log" in metrics["file_metrics"]
        assert "log2.log" in metrics["file_metrics"]
        
        # Check aggregated metrics
        aggregated = metrics["aggregated_metrics"]
        assert "execution_time" in aggregated
        assert len(aggregated["execution_time"]) == 2  # Two values from two files
    
    def test_parse_performance_metrics_no_files(self):
        """Test parsing performance metrics with no files."""
        metrics = self.parser.parse_performance_metrics([])
        
        assert metrics == {"file_metrics": {}, "aggregated_metrics": {}, "summary": {}}
    
    def test_parse_error_metrics_success(self):
        """Test successful error metrics parsing."""
        log_content = """2024-01-01T00:00:00Z INFO: Simulation started
2024-01-01T00:00:01Z WARNING: Low memory warning
2024-01-01T00:00:02Z ERROR: Database connection failed
2024-01-01T00:00:03Z CRITICAL: Fatal error occurred
2024-01-01T00:00:04Z WARNING: Another warning
2024-01-01T00:00:05Z ERROR: Timeout occurred
"""
        log_path = self.create_test_log_file(log_content)
        
        metrics = self.parser.parse_error_metrics([log_path])
        
        assert "file_errors" in metrics
        assert "total_error_counts" in metrics
        assert "total_warning_counts" in metrics
        assert "error_details" in metrics
        assert "summary" in metrics
        
        # Check error counts
        assert metrics["total_error_counts"]["error"] == 2
        assert metrics["total_warning_counts"]["warning"] == 2
        assert metrics["total_error_counts"]["critical"] == 1
        
        # Check summary
        summary = metrics["summary"]
        assert summary["total_errors"] == 3  # 2 errors + 1 critical
        assert summary["total_warnings"] == 2
        assert summary["unique_error_types"] >= 2
    
    def test_parse_error_metrics_no_files(self):
        """Test parsing error metrics with no files."""
        metrics = self.parser.parse_error_metrics([])
        
        assert metrics == {
            "file_errors": {},
            "total_error_counts": {},
            "total_warning_counts": {},
            "error_details": {},
            "summary": {
                "total_errors": 0,
                "total_warnings": 0,
                "unique_error_types": 0,
                "unique_warning_types": 0
            }
        }
    
    def test_parse_log_file_success(self):
        """Test successful single log file parsing."""
        log_content = """2024-01-01T00:00:00Z INFO: Simulation started
2024-01-01T00:00:01Z INFO: Execution time: 1.5 seconds
2024-01-01T00:00:02Z WARNING: Low memory
2024-01-01T00:00:03Z ERROR: Connection failed
"""
        log_path = self.create_test_log_file(log_content)
        
        log_metrics = self.parser.parse_log_file(log_path)
        
        assert isinstance(log_metrics, LogMetrics)
        assert "execution_time" in log_metrics.performance_metrics
        assert "warning" in log_metrics.error_metrics["error_counts"]
        assert "error" in log_metrics.error_metrics["error_counts"]
        assert log_metrics.log_file_info["file_name"] == "test.log"
        assert log_metrics.log_file_info["total_lines"] == 4
    
    def test_parse_log_file_nonexistent(self):
        """Test parsing non-existent log file."""
        nonexistent_path = self.temp_dir / "nonexistent.log"
        
        log_metrics = self.parser.parse_log_file(nonexistent_path)
        
        assert isinstance(log_metrics, LogMetrics)
        assert log_metrics.performance_metrics == {}
        assert log_metrics.error_metrics == {}
        assert log_metrics.summary_stats == {}
        assert log_metrics.log_file_info == {}
    
    def test_parse_single_file_performance(self):
        """Test parsing performance metrics from single file."""
        log_content = "Execution time: 1.5 seconds\nMemory usage: 128 MB\nThroughput: 100 ops/sec"
        log_path = self.create_test_log_file(log_content)
        
        metrics = self.parser._parse_single_file_performance(log_path)
        
        assert "execution_time" in metrics
        assert "memory_usage" in metrics
        assert "throughput" in metrics
        assert metrics["execution_time"] == 1.5
        assert metrics["memory_usage"] == 128
        assert metrics["throughput"] == 100
    
    def test_parse_single_file_performance_milliseconds(self):
        """Test parsing performance metrics with milliseconds."""
        log_content = "Execution time: 1500 ms\nDuration: 500 ms"
        log_path = self.create_test_log_file(log_content)
        
        metrics = self.parser._parse_single_file_performance(log_path)
        
        assert "execution_time" in metrics
        assert "duration" in metrics
        assert metrics["execution_time"] == 1.5  # Converted to seconds
        assert metrics["duration"] == 0.5  # Converted to seconds
    
    def test_parse_single_file_errors(self):
        """Test parsing error metrics from single file."""
        log_content = """INFO: Normal message
WARNING: Warning message
ERROR: Error message
CRITICAL: Critical message
WARNING: Another warning
"""
        log_path = self.create_test_log_file(log_content)
        
        error_data = self.parser._parse_single_file_errors(log_path)
        
        assert "error_counts" in error_data
        assert "warning_counts" in error_data
        assert "error_details" in error_data
        
        assert error_data["error_counts"]["error"] == 1
        assert error_data["error_counts"]["critical"] == 1
        assert error_data["warning_counts"]["warning"] == 2
        
        # Check error details
        error_details = error_data["error_details"]
        assert len(error_details) == 4  # 2 warnings + 1 error + 1 critical
        
        # Check that details contain required fields
        for detail in error_details:
            assert "type" in detail
            assert "line_number" in detail
            assert "line_content" in detail
            assert "timestamp" in detail
    
    def test_extract_timing_metrics(self):
        """Test extracting timing metrics from content."""
        content = """2024-01-01T00:00:00Z INFO: Start
Duration: 1.5 seconds
Elapsed: 2.0 seconds
Runtime: 3.0 ms
"""
        
        timing_metrics = self.parser._extract_timing_metrics(content)
        
        assert "timestamps" in timing_metrics
        assert "duration" in timing_metrics
        assert "elapsed" in timing_metrics
        assert "runtime" in timing_metrics
        
        assert timing_metrics["duration"] == 1.5
        assert timing_metrics["elapsed"] == 2.0
        assert timing_metrics["runtime"] == 0.003  # Converted from ms to seconds
    
    def test_extract_timestamp_from_line(self):
        """Test extracting timestamp from log line."""
        line1 = "2024-01-01T00:00:00Z INFO: Message"
        line2 = "No timestamp here"
        
        timestamp1 = self.parser._extract_timestamp_from_line(line1)
        timestamp2 = self.parser._extract_timestamp_from_line(line2)
        
        assert timestamp1 == "2024-01-01T00:00:00Z"
        assert timestamp2 is None
    
    def test_calculate_performance_summary(self):
        """Test calculating performance summary statistics."""
        all_metrics = {
            "execution_time": [1.0, 2.0, 3.0],
            "memory_usage": [100, 200],
            "throughput": [50.5, 75.5, 100.5]
        }
        
        summary = self.parser._calculate_performance_summary(all_metrics)
        
        assert "execution_time" in summary
        assert "memory_usage" in summary
        assert "throughput" in summary
        
        # Check execution_time summary
        exec_summary = summary["execution_time"]
        assert exec_summary["count"] == 3
        assert exec_summary["min"] == 1.0
        assert exec_summary["max"] == 3.0
        assert exec_summary["mean"] == 2.0
        assert exec_summary["total"] == 6.0
        assert "std_dev" in exec_summary
    
    def test_calculate_file_summary(self):
        """Test calculating file summary statistics."""
        lines = [
            "INFO: Message 1",
            "DEBUG: Debug message",
            "WARNING: Warning message",
            "ERROR: Error message",
            "",  # Empty line
            "INFO: Another message"
        ]
        log_path = self.create_test_log_file("test content")
        
        summary = self.parser._calculate_file_summary(lines, log_path)
        
        assert summary["total_lines"] == 6
        assert summary["non_empty_lines"] == 5
        assert summary["empty_lines"] == 1
        assert summary["info_messages"] == 2
        assert summary["debug_messages"] == 1
        assert summary["warning_messages"] == 1
        assert summary["error_messages"] == 1
        assert summary["file_size_bytes"] > 0
    
    def test_performance_patterns(self):
        """Test that performance patterns are correctly configured."""
        patterns = self.parser.performance_patterns
        
        # Test execution time pattern
        exec_pattern = patterns["execution_time"]
        assert exec_pattern.search("Execution time: 1.5 seconds")
        assert exec_pattern.search("execution time: 1000 ms")
        
        # Test memory usage pattern
        memory_pattern = patterns["memory_usage"]
        assert memory_pattern.search("Memory usage: 128 MB")
        assert memory_pattern.search("memory: 1.5 GB")
        
        # Test throughput pattern
        throughput_pattern = patterns["throughput"]
        assert throughput_pattern.search("Throughput: 100 ops/sec")
        assert throughput_pattern.search("throughput: 50 operations/second")
    
    def test_error_patterns(self):
        """Test that error patterns are correctly configured."""
        patterns = self.parser.error_patterns
        
        # Test error pattern
        error_pattern = patterns["error"]
        assert error_pattern.search("ERROR: Something failed")
        assert error_pattern.search("Exception occurred")
        assert error_pattern.search("Failed to connect")
        
        # Test warning pattern
        warning_pattern = patterns["warning"]
        assert warning_pattern.search("WARNING: Low memory")
        assert warning_pattern.search("Warn: Check this")
        
        # Test critical pattern
        critical_pattern = patterns["critical"]
        assert critical_pattern.search("CRITICAL: Fatal error")
        assert critical_pattern.search("Fatal: System panic")
    
    def test_timing_patterns(self):
        """Test that timing patterns are correctly configured."""
        patterns = self.parser.timing_patterns
        
        # Test timestamp pattern
        timestamp_pattern = patterns["timestamp"]
        assert timestamp_pattern.search("2024-01-01T00:00:00Z")
        assert timestamp_pattern.search("2024-01-01T00:00:00.123Z")
        assert timestamp_pattern.search("2024-01-01T00:00:00+00:00")
        
        # Test duration pattern
        duration_pattern = patterns["duration"]
        assert duration_pattern.search("Duration: 1.5 seconds")
        assert duration_pattern.search("duration: 1000 ms")
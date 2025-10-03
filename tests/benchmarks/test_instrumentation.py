"""
Unit tests for the benchmarks.core.instrumentation module.

Tests timing, cProfile, and psutil instrumentation modules.
"""

import json
import os
import tempfile
import time
import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
from contextlib import contextmanager

from benchmarks.core.instrumentation.timing import time_block
from benchmarks.core.instrumentation.cprofile import cprofile_capture, _summarize_stats_from_file
from benchmarks.core.instrumentation.psutil_monitor import psutil_sampling, _Sampler


class TestTimeBlock(unittest.TestCase):
    """Test timing instrumentation."""

    def test_time_block_basic(self):
        """Test basic timing functionality."""
        metrics = {}
        
        with time_block(metrics, "test_duration"):
            time.sleep(0.01)  # Sleep for 10ms
        
        self.assertIn("test_duration", metrics)
        self.assertGreater(metrics["test_duration"], 0.005)  # Should be at least 5ms
        self.assertLess(metrics["test_duration"], 0.1)  # Should be less than 100ms

    def test_time_block_default_key(self):
        """Test timing with default key."""
        metrics = {}
        
        with time_block(metrics):
            time.sleep(0.01)
        
        self.assertIn("duration_s", metrics)
        self.assertGreater(metrics["duration_s"], 0.005)

    def test_time_block_exception_handling(self):
        """Test that timing works even when exception is raised."""
        metrics = {}
        
        with self.assertRaises(ValueError):
            with time_block(metrics, "test_duration"):
                time.sleep(0.01)
                raise ValueError("Test exception")
        
        # Should still record timing even with exception
        self.assertIn("test_duration", metrics)
        self.assertGreater(metrics["test_duration"], 0.005)

    def test_time_block_multiple_keys(self):
        """Test timing with multiple different keys."""
        metrics = {}
        
        with time_block(metrics, "first_duration"):
            time.sleep(0.01)
        
        with time_block(metrics, "second_duration"):
            time.sleep(0.01)
        
        self.assertIn("first_duration", metrics)
        self.assertIn("second_duration", metrics)
        self.assertGreater(metrics["first_duration"], 0.005)
        self.assertGreater(metrics["second_duration"], 0.005)

    def test_time_block_overwrites_existing_key(self):
        """Test that timing overwrites existing key value."""
        metrics = {"test_duration": 999.0}
        
        with time_block(metrics, "test_duration"):
            time.sleep(0.01)
        
        self.assertEqual(len(metrics), 1)
        self.assertIn("test_duration", metrics)
        self.assertLess(metrics["test_duration"], 1.0)  # Should be overwritten


class TestCProfileCapture(unittest.TestCase):
    """Test cProfile instrumentation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cprofile_capture_basic(self):
        """Test basic cProfile capture functionality."""
        metrics = {}
        
        with cprofile_capture(self.temp_dir, "test_run", 0, metrics):
            # Do some work to profile
            for i in range(1000):
                _ = i * i
        
        # Check that metrics were updated
        self.assertIn("cprofile_artifact", metrics)
        self.assertIn("cprofile_summary_path", metrics)
        
        # Check that files were created
        self.assertTrue(os.path.exists(metrics["cprofile_artifact"]))
        self.assertTrue(os.path.exists(metrics["cprofile_summary_path"]))
        
        # Check that summary file contains valid JSON
        with open(metrics["cprofile_summary_path"], "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        self.assertIn("top_cumulative", summary)
        self.assertIn("top_internal", summary)
        self.assertIn("top_calls", summary)

    def test_cprofile_capture_custom_top_n(self):
        """Test cProfile capture with custom top_n parameter."""
        metrics = {}
        
        with cprofile_capture(self.temp_dir, "test_run", 0, metrics, top_n=10):
            for i in range(1000):
                _ = i * i
        
        # Check that summary was created with custom top_n
        with open(metrics["cprofile_summary_path"], "r", encoding="utf-8") as f:
            summary = json.load(f)
        
        # Should have at most 10 entries in each category
        self.assertLessEqual(len(summary["top_cumulative"]), 10)
        self.assertLessEqual(len(summary["top_internal"]), 10)
        self.assertLessEqual(len(summary["top_calls"]), 10)

    def test_cprofile_capture_exception_handling(self):
        """Test cProfile capture with exception handling."""
        metrics = {}
        
        with self.assertRaises(ValueError):
            with cprofile_capture(self.temp_dir, "test_run", 0, metrics):
                raise ValueError("Test exception")
        
        # Should still record metrics even with exception
        self.assertIn("cprofile_artifact", metrics)
        self.assertIn("cprofile_summary_path", metrics)

    def test_cprofile_capture_file_naming(self):
        """Test that cProfile files are named correctly."""
        metrics = {}
        
        with cprofile_capture(self.temp_dir, "my_benchmark", 5, metrics):
            for i in range(100):
                _ = i * i
        
        # Check file naming convention
        expected_prof = os.path.join(self.temp_dir, "my_benchmark_iter005.prof")
        expected_summary = os.path.join(self.temp_dir, "my_benchmark_iter005_cprofile_summary.json")
        
        self.assertEqual(metrics["cprofile_artifact"], expected_prof)
        self.assertEqual(metrics["cprofile_summary_path"], expected_summary)

    def test_cprofile_capture_write_failure(self):
        """Test cProfile capture when file writing fails."""
        metrics = {}
        
        with patch('builtins.open', side_effect=IOError("Write failed")):
            with cprofile_capture(self.temp_dir, "test_run", 0, metrics):
                for i in range(100):
                    _ = i * i
        
        # Should not crash, but metrics might not be set
        # The function should handle the exception gracefully

    @patch('cProfile.Profile')
    def test_cprofile_capture_profiler_failure(self, mock_profile):
        """Test cProfile capture when profiler fails."""
        mock_profile.side_effect = Exception("Profiler failed")
        metrics = {}
        
        # Should not crash - the exception is handled in the context manager
        with self.assertRaises(Exception):
            with cprofile_capture(self.temp_dir, "test_run", 0, metrics):
                for i in range(100):
                    _ = i * i

    def test_summarize_stats_from_file(self):
        """Test _summarize_stats_from_file function."""
        # Create a mock stats file by running a simple cProfile
        prof_path = os.path.join(self.temp_dir, "test.prof")
        
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            for i in range(1000):
                _ = i * i
        finally:
            profiler.disable()
            profiler.dump_stats(prof_path)
        
        # Test summarization
        summary = _summarize_stats_from_file(prof_path, top_n=5)
        
        self.assertIn("top_cumulative", summary)
        self.assertIn("top_internal", summary)
        self.assertIn("top_calls", summary)
        
        # Check structure of summary entries
        for category in ["top_cumulative", "top_internal", "top_calls"]:
            self.assertIsInstance(summary[category], list)
            if summary[category]:  # If there are entries
                entry = summary[category][0]
                self.assertIn("function", entry)
                self.assertIn("file", entry)
                self.assertIn("line", entry)
                self.assertIn("calls", entry)
                self.assertIn("primitive_calls", entry)
                self.assertIn("internal_time", entry)
                self.assertIn("cumulative_time", entry)

    def test_summarize_stats_from_nonexistent_file(self):
        """Test _summarize_stats_from_file with nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            _summarize_stats_from_file("nonexistent.prof")


class TestPsutilSampling(unittest.TestCase):
    """Test psutil instrumentation."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('benchmarks.core.instrumentation.psutil_monitor.psutil')
    def test_psutil_sampling_basic(self, mock_psutil):
        """Test basic psutil sampling functionality."""
        # Mock psutil components
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 25.5
        mock_process.memory_info.return_value = Mock(rss=1024*1024*100)  # 100MB
        mock_process.io_counters.return_value = Mock(read_bytes=1000, write_bytes=2000)
        mock_psutil.Process.return_value = mock_process
        
        metrics = {}
        
        with psutil_sampling(self.temp_dir, "test_run", 0, metrics, interval_ms=100, max_samples=5):
            time.sleep(0.3)  # Sleep to allow sampling
        
        # Check that metrics were updated
        self.assertIn("psutil_artifact", metrics)
        self.assertIn("psutil_summary", metrics)
        
        # Check that file was created
        self.assertTrue(os.path.exists(metrics["psutil_artifact"]))
        
        # Check summary structure
        summary = metrics["psutil_summary"]
        self.assertIn("rss_bytes", summary)
        self.assertIn("cpu_percent", summary)
        self.assertIn("samples", summary)
        
        # Check that we got some samples
        self.assertGreater(summary["samples"], 0)

    @patch('benchmarks.core.instrumentation.psutil_monitor.psutil', None)
    def test_psutil_sampling_no_psutil(self):
        """Test psutil sampling when psutil is not available."""
        metrics = {}
        
        # Should not crash when psutil is None
        with psutil_sampling(self.temp_dir, "test_run", 0, metrics):
            time.sleep(0.1)
        
        # Should not have psutil metrics
        self.assertNotIn("psutil_artifact", metrics)
        self.assertNotIn("psutil_summary", metrics)

    @patch('benchmarks.core.instrumentation.psutil_monitor.psutil')
    def test_psutil_sampling_file_naming(self, mock_psutil):
        """Test that psutil files are named correctly."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 10.0
        mock_process.memory_info.return_value = Mock(rss=1024*1024*50)
        mock_process.io_counters.return_value = Mock(read_bytes=500, write_bytes=1000)
        mock_psutil.Process.return_value = mock_process
        
        metrics = {}
        
        with psutil_sampling(self.temp_dir, "my_benchmark", 3, metrics, interval_ms=50, max_samples=2):
            time.sleep(0.1)
        
        expected_path = os.path.join(self.temp_dir, "my_benchmark_iter003_psutil.jsonl")
        self.assertEqual(metrics["psutil_artifact"], expected_path)

    @patch('benchmarks.core.instrumentation.psutil_monitor.psutil')
    def test_psutil_sampling_exception_handling(self, mock_psutil):
        """Test psutil sampling with exception handling."""
        mock_process = Mock()
        mock_process.cpu_percent.side_effect = Exception("CPU percent failed")
        mock_psutil.Process.return_value = mock_process
        
        metrics = {}
        
        with self.assertRaises(ValueError):
            with psutil_sampling(self.temp_dir, "test_run", 0, metrics):
                raise ValueError("Test exception")
        
        # Should still record metrics even with exception
        self.assertIn("psutil_artifact", metrics)

    @patch('benchmarks.core.instrumentation.psutil_monitor.psutil')
    def test_psutil_sampling_write_failure(self, mock_psutil):
        """Test psutil sampling when file writing fails."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 10.0
        mock_process.memory_info.return_value = Mock(rss=1024*1024*50)
        mock_psutil.Process.return_value = mock_process
        
        metrics = {}
        
        with patch('builtins.open', side_effect=IOError("Write failed")):
            with psutil_sampling(self.temp_dir, "test_run", 0, metrics):
                time.sleep(0.1)
        
        # Should not crash, but metrics might not be set

    @patch('benchmarks.core.instrumentation.psutil_monitor.psutil')
    def test_psutil_sampling_max_samples(self, mock_psutil):
        """Test psutil sampling with max_samples limit."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 10.0
        mock_process.memory_info.return_value = Mock(rss=1024*1024*50)
        mock_psutil.Process.return_value = mock_process
        
        metrics = {}
        
        with psutil_sampling(self.temp_dir, "test_run", 0, metrics, interval_ms=10, max_samples=3):
            time.sleep(0.1)
        
        summary = metrics.get("psutil_summary", {})
        self.assertLessEqual(summary.get("samples", 0), 3)

    @patch('benchmarks.core.instrumentation.psutil_monitor.psutil')
    def test_psutil_sampling_no_io_counters(self, mock_psutil):
        """Test psutil sampling when io_counters is not available."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 10.0
        mock_process.memory_info.return_value = Mock(rss=1024*1024*50)
        # Remove io_counters method
        del mock_process.io_counters
        mock_psutil.Process.return_value = mock_process
        
        metrics = {}
        
        with psutil_sampling(self.temp_dir, "test_run", 0, metrics, interval_ms=50, max_samples=2):
            time.sleep(0.1)
        
        # Should still work without io_counters
        self.assertIn("psutil_artifact", metrics)


class TestSampler(unittest.TestCase):
    """Test _Sampler class."""

    def setUp(self):
        """Set up test fixtures."""
        self.samples = []

    @patch('benchmarks.core.instrumentation.psutil_monitor.psutil')
    def test_sampler_basic(self, mock_psutil):
        """Test basic sampler functionality."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 15.0
        mock_process.memory_info.return_value = Mock(rss=1024*1024*75)
        mock_process.io_counters.return_value = Mock(read_bytes=1500, write_bytes=3000)
        mock_psutil.Process.return_value = mock_process
        
        sampler = _Sampler(interval_s=0.01, out_list=self.samples, max_samples=3)
        sampler.start()
        
        # Let it run for a bit
        time.sleep(0.05)
        sampler.stop()
        sampler.join(timeout=1.0)
        
        # Should have collected some samples
        self.assertGreater(len(self.samples), 0)
        self.assertLessEqual(len(self.samples), 3)  # max_samples limit
        
        # Check sample structure
        if self.samples:
            sample = self.samples[0]
            self.assertIn("ts", sample)
            self.assertIn("cpu_percent", sample)
            self.assertIn("rss_bytes", sample)
            self.assertIn("read_bytes", sample)
            self.assertIn("write_bytes", sample)

    @patch('benchmarks.core.instrumentation.psutil_monitor.psutil', None)
    def test_sampler_no_psutil(self):
        """Test sampler when psutil is not available."""
        sampler = _Sampler(interval_s=0.01, out_list=self.samples, max_samples=3)
        sampler.start()
        
        time.sleep(0.05)
        sampler.stop()
        sampler.join(timeout=1.0)
        
        # Should not collect any samples
        self.assertEqual(len(self.samples), 0)

    @patch('benchmarks.core.instrumentation.psutil_monitor.psutil')
    def test_sampler_exception_handling(self, mock_psutil):
        """Test sampler with exception handling."""
        mock_process = Mock()
        mock_process.cpu_percent.side_effect = Exception("CPU percent failed")
        mock_psutil.Process.return_value = mock_process
        
        sampler = _Sampler(interval_s=0.01, out_list=self.samples, max_samples=3)
        sampler.start()
        
        time.sleep(0.05)
        sampler.stop()
        sampler.join(timeout=1.0)
        
        # Should not crash, but might not collect samples
        # The sampler should handle exceptions gracefully

    @patch('benchmarks.core.instrumentation.psutil_monitor.psutil')
    def test_sampler_max_samples(self, mock_psutil):
        """Test sampler with max_samples limit."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 10.0
        mock_process.memory_info.return_value = Mock(rss=1024*1024*50)
        mock_process.io_counters.return_value = Mock(read_bytes=1000, write_bytes=2000)
        mock_psutil.Process.return_value = mock_process
        
        sampler = _Sampler(interval_s=0.001, out_list=self.samples, max_samples=2)
        sampler.start()
        
        time.sleep(0.01)
        sampler.stop()
        sampler.join(timeout=1.0)
        
        # Should respect max_samples limit
        self.assertLessEqual(len(self.samples), 2)

    def test_sampler_stop(self):
        """Test sampler stop functionality."""
        sampler = _Sampler(interval_s=0.1, out_list=self.samples, max_samples=10)
        sampler.start()
        
        # Stop immediately
        sampler.stop()
        sampler.join(timeout=1.0)
        
        # Should stop quickly
        self.assertFalse(sampler.is_alive())


if __name__ == "__main__":
    unittest.main()

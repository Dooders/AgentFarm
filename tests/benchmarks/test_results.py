"""
Unit tests for the benchmarks.core.results module.

Tests RunResult, IterationResult, Artifact classes and utility functions.
"""

import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from benchmarks.core.results import (
    RunResult, IterationResult, Artifact,
    capture_environment, capture_vcs, _safe_git
)


class TestArtifact(unittest.TestCase):
    """Test Artifact dataclass."""

    def test_artifact_creation(self):
        """Test Artifact creation with all fields."""
        artifact = Artifact(
            name="test_artifact",
            type="json",
            path="/tmp/test.json"
        )
        
        self.assertEqual(artifact.name, "test_artifact")
        self.assertEqual(artifact.type, "json")
        self.assertEqual(artifact.path, "/tmp/test.json")

    def test_artifact_serialization(self):
        """Test Artifact can be converted to dict."""
        artifact = Artifact(
            name="test_artifact",
            type="profile",
            path="/tmp/test.prof"
        )
        
        artifact_dict = asdict(artifact)
        self.assertEqual(artifact_dict["name"], "test_artifact")
        self.assertEqual(artifact_dict["type"], "profile")
        self.assertEqual(artifact_dict["path"], "/tmp/test.prof")


class TestIterationResult(unittest.TestCase):
    """Test IterationResult dataclass."""

    def test_iteration_result_creation(self):
        """Test IterationResult creation with all fields."""
        metrics = {"duration": 1.5, "memory": 1024}
        result = IterationResult(
            index=0,
            duration_s=1.5,
            metrics=metrics
        )
        
        self.assertEqual(result.index, 0)
        self.assertEqual(result.duration_s, 1.5)
        self.assertEqual(result.metrics, metrics)

    def test_iteration_result_default_metrics(self):
        """Test IterationResult creation with default metrics."""
        result = IterationResult(index=1, duration_s=2.0)
        
        self.assertEqual(result.index, 1)
        self.assertEqual(result.duration_s, 2.0)
        self.assertEqual(result.metrics, {})

    def test_iteration_result_serialization(self):
        """Test IterationResult can be converted to dict."""
        metrics = {"test": "value"}
        result = IterationResult(
            index=2,
            duration_s=3.0,
            metrics=metrics
        )
        
        result_dict = asdict(result)
        self.assertEqual(result_dict["index"], 2)
        self.assertEqual(result_dict["duration_s"], 3.0)
        self.assertEqual(result_dict["metrics"], metrics)


class TestRunResult(unittest.TestCase):
    """Test RunResult dataclass."""

    def setUp(self):
        """Set up test fixtures."""
        self.run_result = RunResult(
            name="test_benchmark",
            run_id="test_run_123",
            parameters={"param1": "value1", "param2": 42},
            iterations={"warmup": 1, "measured": 3},
            tags=["test", "benchmark"],
            notes="Test run"
        )

    def test_run_result_creation(self):
        """Test RunResult creation with all fields."""
        self.assertEqual(self.run_result.name, "test_benchmark")
        self.assertEqual(self.run_result.run_id, "test_run_123")
        self.assertEqual(self.run_result.parameters, {"param1": "value1", "param2": 42})
        self.assertEqual(self.run_result.iterations, {"warmup": 1, "measured": 3})
        self.assertEqual(self.run_result.tags, ["test", "benchmark"])
        self.assertEqual(self.run_result.notes, "Test run")
        self.assertEqual(self.run_result.status, "success")

    def test_run_result_defaults(self):
        """Test RunResult creation with default values."""
        result = RunResult(
            name="minimal_test",
            run_id="minimal_123",
            parameters={},
            iterations={"warmup": 0, "measured": 1}
        )
        
        self.assertEqual(result.metrics, {})
        self.assertEqual(result.iteration_metrics, [])
        self.assertEqual(result.artifacts, [])
        self.assertEqual(result.tags, [])
        self.assertEqual(result.notes, "")
        self.assertEqual(result.status, "success")
        self.assertIsInstance(result.environment, dict)
        self.assertIsInstance(result.vcs, dict)

    def test_add_iteration(self):
        """Test adding iteration results."""
        metrics = {"duration": 1.5, "memory": 1024}
        self.run_result.add_iteration(0, 1.5, metrics)
        
        self.assertEqual(len(self.run_result.iteration_metrics), 1)
        iteration = self.run_result.iteration_metrics[0]
        self.assertEqual(iteration.index, 0)
        self.assertEqual(iteration.duration_s, 1.5)
        self.assertEqual(iteration.metrics, metrics)

    def test_add_multiple_iterations(self):
        """Test adding multiple iteration results."""
        for i in range(3):
            metrics = {"iteration": i, "duration": float(i + 1)}
            self.run_result.add_iteration(i, float(i + 1), metrics)
        
        self.assertEqual(len(self.run_result.iteration_metrics), 3)
        for i, iteration in enumerate(self.run_result.iteration_metrics):
            self.assertEqual(iteration.index, i)
            self.assertEqual(iteration.duration_s, float(i + 1))
            self.assertEqual(iteration.metrics["iteration"], i)

    def test_add_artifact(self):
        """Test adding artifacts."""
        self.run_result.add_artifact("test_artifact", "json", "/tmp/test.json")
        
        self.assertEqual(len(self.run_result.artifacts), 1)
        artifact = self.run_result.artifacts[0]
        self.assertEqual(artifact.name, "test_artifact")
        self.assertEqual(artifact.type, "json")
        self.assertEqual(artifact.path, "/tmp/test.json")

    def test_add_multiple_artifacts(self):
        """Test adding multiple artifacts."""
        artifacts = [
            ("artifact1", "json", "/tmp/artifact1.json"),
            ("artifact2", "profile", "/tmp/artifact2.prof"),
            ("artifact3", "log", "/tmp/artifact3.log")
        ]
        
        for name, type_, path in artifacts:
            self.run_result.add_artifact(name, type_, path)
        
        self.assertEqual(len(self.run_result.artifacts), 3)
        for i, (name, type_, path) in enumerate(artifacts):
            artifact = self.run_result.artifacts[i]
            self.assertEqual(artifact.name, name)
            self.assertEqual(artifact.type, type_)
            self.assertEqual(artifact.path, path)

    def test_get_mean_duration_empty(self):
        """Test get_mean_duration with no iterations."""
        mean_duration = self.run_result.get_mean_duration()
        self.assertEqual(mean_duration, 0.0)

    def test_get_mean_duration_with_iterations(self):
        """Test get_mean_duration with iterations."""
        durations = [1.0, 2.0, 3.0, 4.0]
        for i, duration in enumerate(durations):
            self.run_result.add_iteration(i, duration, {})
        
        mean_duration = self.run_result.get_mean_duration()
        self.assertEqual(mean_duration, 2.5)

    def test_get_median_duration_empty(self):
        """Test get_median_duration with no iterations."""
        median_duration = self.run_result.get_median_duration()
        self.assertEqual(median_duration, 0.0)

    def test_get_median_duration_odd_count(self):
        """Test get_median_duration with odd number of iterations."""
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, duration in enumerate(durations):
            self.run_result.add_iteration(i, duration, {})
        
        median_duration = self.run_result.get_median_duration()
        self.assertEqual(median_duration, 3.0)

    def test_get_median_duration_even_count(self):
        """Test get_median_duration with even number of iterations."""
        durations = [1.0, 2.0, 3.0, 4.0]
        for i, duration in enumerate(durations):
            self.run_result.add_iteration(i, duration, {})
        
        median_duration = self.run_result.get_median_duration()
        self.assertEqual(median_duration, 2.5)

    def test_to_dict(self):
        """Test converting RunResult to dictionary."""
        # Add some test data
        self.run_result.add_iteration(0, 1.5, {"test": "value"})
        self.run_result.add_artifact("test_artifact", "json", "/tmp/test.json")
        self.run_result.metrics["duration_s"] = {"mean": 1.5}
        
        result_dict = self.run_result.to_dict()
        
        # Check basic fields
        self.assertEqual(result_dict["name"], "test_benchmark")
        self.assertEqual(result_dict["run_id"], "test_run_123")
        self.assertEqual(result_dict["parameters"], {"param1": "value1", "param2": 42})
        self.assertEqual(result_dict["iterations"], {"warmup": 1, "measured": 3})
        self.assertEqual(result_dict["tags"], ["test", "benchmark"])
        self.assertEqual(result_dict["notes"], "Test run")
        self.assertEqual(result_dict["status"], "success")
        
        # Check iteration metrics
        self.assertEqual(len(result_dict["iteration_metrics"]), 1)
        iteration_dict = result_dict["iteration_metrics"][0]
        self.assertEqual(iteration_dict["index"], 0)
        self.assertEqual(iteration_dict["duration_s"], 1.5)
        self.assertEqual(iteration_dict["metrics"], {"test": "value"})
        
        # Check artifacts
        self.assertEqual(len(result_dict["artifacts"]), 1)
        artifact_dict = result_dict["artifacts"][0]
        self.assertEqual(artifact_dict["name"], "test_artifact")
        self.assertEqual(artifact_dict["type"], "json")
        self.assertEqual(artifact_dict["path"], "/tmp/test.json")
        
        # Check metrics
        self.assertEqual(result_dict["metrics"]["duration_s"]["mean"], 1.5)
        
        # Check environment and vcs are present
        self.assertIn("environment", result_dict)
        self.assertIn("vcs", result_dict)

    def test_save(self):
        """Test saving RunResult to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Add some test data
            self.run_result.add_iteration(0, 1.5, {"test": "value"})
            self.run_result.add_artifact("test_artifact", "json", "/tmp/test.json")
            
            # Save the result
            saved_path = self.run_result.save(tmpdir)
            
            # Check that file was created
            self.assertTrue(os.path.exists(saved_path))
            self.assertTrue(saved_path.endswith(".json"))
            
            # Check file contents
            with open(saved_path, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            
            self.assertEqual(saved_data["name"], "test_benchmark")
            self.assertEqual(saved_data["run_id"], "test_run_123")
            self.assertEqual(len(saved_data["iteration_metrics"]), 1)
            self.assertEqual(len(saved_data["artifacts"]), 1)

    def test_save_creates_directory(self):
        """Test that save creates the output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nonexistent", "subdir")
            
            # Save should create the directory
            saved_path = self.run_result.save(output_dir)
            
            self.assertTrue(os.path.exists(output_dir))
            self.assertTrue(os.path.exists(saved_path))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for environment and VCS capture."""

    @patch('subprocess.check_output')
    def test_safe_git_success(self, mock_check_output):
        """Test _safe_git with successful git command."""
        mock_check_output.return_value = b"abc123\n"
        
        result = _safe_git(["git", "rev-parse", "HEAD"])
        
        self.assertEqual(result, "abc123")
        mock_check_output.assert_called_once_with(
            ["git", "rev-parse", "HEAD"], 
            stderr=subprocess.DEVNULL
        )

    @patch('subprocess.check_output')
    def test_safe_git_failure(self, mock_check_output):
        """Test _safe_git with failed git command."""
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")
        
        result = _safe_git(["git", "rev-parse", "HEAD"])
        
        self.assertIsNone(result)

    @patch('platform.platform')
    @patch('platform.python_version')
    @patch('platform.machine')
    @patch('platform.processor')
    @patch('socket.gethostname')
    def test_capture_environment_basic(self, mock_hostname, mock_processor, 
                                      mock_machine, mock_python, mock_platform):
        """Test capture_environment with basic platform info."""
        mock_platform.return_value = "Windows-10-10.0.19041-SP0"
        mock_python.return_value = "3.9.7"
        mock_machine.return_value = "AMD64"
        mock_processor.return_value = "Intel64 Family 6 Model 142 Stepping 10, GenuineIntel"
        mock_hostname.return_value = "test-machine"
        
        env = capture_environment()
        
        self.assertEqual(env["os"], "Windows-10-10.0.19041-SP0")
        self.assertEqual(env["python"], "3.9.7")
        self.assertEqual(env["machine"], "AMD64")
        self.assertEqual(env["processor"], "Intel64 Family 6 Model 142 Stepping 10, GenuineIntel")
        self.assertEqual(env["hostname"], "test-machine")

    @patch('subprocess.check_output')
    @patch('platform.platform')
    @patch('platform.python_version')
    @patch('platform.machine')
    @patch('platform.processor')
    @patch('socket.gethostname')
    def test_capture_environment_with_gpu(self, mock_hostname, mock_processor,
                                         mock_machine, mock_python, mock_platform,
                                         mock_check_output):
        """Test capture_environment with GPU information."""
        mock_platform.return_value = "Linux-5.4.0-74-generic-x86_64-with-glibc2.29"
        mock_python.return_value = "3.8.10"
        mock_machine.return_value = "x86_64"
        mock_processor.return_value = "x86_64"
        mock_hostname.return_value = "gpu-machine"
        mock_check_output.return_value = b"NVIDIA GeForce RTX 3080, 10240 MiB\n"
        
        env = capture_environment()
        
        self.assertIn("gpus", env)
        self.assertEqual(len(env["gpus"]), 1)
        self.assertEqual(env["gpus"][0]["name"], "NVIDIA GeForce RTX 3080")
        self.assertEqual(env["gpus"][0]["memory_total"], "10240 MiB")

    @patch('subprocess.check_output')
    @patch('platform.platform')
    @patch('platform.python_version')
    @patch('platform.machine')
    @patch('platform.processor')
    @patch('socket.gethostname')
    def test_capture_environment_gpu_failure(self, mock_hostname, mock_processor,
                                            mock_machine, mock_python, mock_platform,
                                            mock_check_output):
        """Test capture_environment when GPU detection fails."""
        mock_platform.return_value = "Linux-5.4.0-74-generic-x86_64-with-glibc2.29"
        mock_python.return_value = "3.8.10"
        mock_machine.return_value = "x86_64"
        mock_processor.return_value = "x86_64"
        mock_hostname.return_value = "cpu-machine"
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "nvidia-smi")
        
        env = capture_environment()
        
        # Should not have gpus key when nvidia-smi fails
        self.assertNotIn("gpus", env)

    @patch('benchmarks.core.results._safe_git')
    def test_capture_vcs(self, mock_safe_git):
        """Test capture_vcs function."""
        mock_safe_git.side_effect = [
            "abc123def456",  # commit
            "main",          # branch
            "M file1.py\nA file2.py"  # status
        ]
        
        vcs = capture_vcs()
        
        self.assertEqual(vcs["commit"], "abc123def456")
        self.assertEqual(vcs["branch"], "main")
        self.assertTrue(vcs["dirty"])

    @patch('benchmarks.core.results._safe_git')
    def test_capture_vcs_clean_repo(self, mock_safe_git):
        """Test capture_vcs with clean repository."""
        mock_safe_git.side_effect = [
            "abc123def456",  # commit
            "main",          # branch
            ""               # status (empty = clean)
        ]
        
        vcs = capture_vcs()
        
        self.assertEqual(vcs["commit"], "abc123def456")
        self.assertEqual(vcs["branch"], "main")
        self.assertFalse(vcs["dirty"])

    @patch('benchmarks.core.results._safe_git')
    def test_capture_vcs_git_failure(self, mock_safe_git):
        """Test capture_vcs when git commands fail."""
        mock_safe_git.return_value = None
        
        vcs = capture_vcs()
        
        self.assertEqual(vcs["commit"], "")
        self.assertEqual(vcs["branch"], "")
        self.assertFalse(vcs["dirty"])


if __name__ == "__main__":
    unittest.main()

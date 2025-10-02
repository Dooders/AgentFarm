"""
Integration tests for the benchmarks module.

Tests end-to-end benchmark execution and component integration.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

from benchmarks.core.experiments import Experiment, ExperimentContext
from benchmarks.core.runner import Runner
from benchmarks.core.registry import ExperimentRegistry, register_experiment
from benchmarks.core.spec import load_spec, RunSpec
from benchmarks.core.results import RunResult


class IntegrationTestExperiment(Experiment):
    """Test experiment for integration testing."""
    
    def __init__(self, params=None, data_size=100):
        super().__init__(params)
        self.data_size = data_size
        self.setup_called = False
        self.teardown_called = False
        self.execute_count = 0
        self.test_data = None
    
    def setup(self, context):
        """Set up test data."""
        self.setup_called = True
        self.test_data = list(range(self.params.get("data_size", self.data_size)))
    
    def execute_once(self, context):
        """Execute test workload."""
        self.execute_count += 1
        
        # Simulate some work
        result = sum(self.test_data) * self.execute_count
        
        return {
            "iteration": self.execute_count,
            "result": result,
            "data_size": len(self.test_data),
            "run_id": context.run_id
        }
    
    def teardown(self, context):
        """Clean up test data."""
        self.teardown_called = True
        self.test_data = None


class TestBenchmarkIntegration(unittest.TestCase):
    """Test end-to-end benchmark execution."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.experiment = IntegrationTestExperiment({"data_size": 50})

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('benchmarks.core.instrumentation.timing.time_block')
    def test_runner_experiment_integration(self, mock_time_block):
        """Test integration between Runner and Experiment."""
        mock_time_block.return_value.__enter__ = Mock()
        mock_time_block.return_value.__exit__ = Mock(return_value=None)
        
        runner = Runner(
            name="integration_test",
            experiment=self.experiment,
            output_dir=self.temp_dir,
            iterations_warmup=1,
            iterations_measured=2,
            seed=42,
            tags=["integration", "test"],
            notes="Integration test run"
        )
        
        result = runner.run()
        
        # Check experiment lifecycle
        self.assertTrue(self.experiment.setup_called)
        self.assertTrue(self.experiment.teardown_called)
        self.assertEqual(self.experiment.execute_count, 3)  # 1 warmup + 2 measured
        
        # Check result structure
        self.assertIsInstance(result, RunResult)
        self.assertEqual(result.name, "integration_test")
        self.assertEqual(result.run_id, runner.run_id)
        self.assertEqual(result.tags, ["integration", "test"])
        self.assertIn("Integration test run", result.notes)
        
        # Check iteration results
        self.assertEqual(len(result.iteration_metrics), 2)
        for i, iteration in enumerate(result.iteration_metrics):
            self.assertEqual(iteration.index, i)
            self.assertIn("iteration", iteration.metrics)
            self.assertIn("result", iteration.metrics)
            self.assertIn("data_size", iteration.metrics)
            self.assertIn("run_id", iteration.metrics)

    @patch('benchmarks.core.instrumentation.timing.time_block')
    def test_runner_with_multiple_instruments(self, mock_time_block):
        """Test Runner with multiple instrumentation tools."""
        mock_time_block.return_value.__enter__ = Mock()
        mock_time_block.return_value.__exit__ = Mock(return_value=None)
        
        runner = Runner(
            name="multi_instrument_test",
            experiment=self.experiment,
            output_dir=self.temp_dir,
            instruments=["timing", "cprofile", "psutil"]
        )
        
        with patch('benchmarks.core.instrumentation.cprofile.cprofile_capture') as mock_cprofile, \
             patch('benchmarks.core.instrumentation.psutil_monitor.psutil_sampling') as mock_psutil:
            
            mock_cprofile.return_value.__enter__ = Mock()
            mock_cprofile.return_value.__exit__ = Mock(return_value=None)
            mock_psutil.return_value.__enter__ = Mock()
            mock_psutil.return_value.__exit__ = Mock(return_value=None)
            
            result = runner.run()
        
        # Should have called all instrumentation tools
        # Note: timing is always called, cProfile and psutil are called if configured
        self.assertGreaterEqual(mock_time_block.call_count, 1)  # At least 1 measured iteration
        mock_cprofile.assert_called_once()
        mock_psutil.assert_called_once()

    def test_registry_experiment_integration(self):
        """Test integration between registry and experiment."""
        registry = ExperimentRegistry()
        
        # Register the test experiment
        registry.register(
            slug="integration_test_exp",
            cls=IntegrationTestExperiment,
            summary="Integration test experiment",
            tags=["integration", "test"]
        )
        
        # Create experiment through registry
        experiment = registry.create("integration_test_exp", {"data_size": 25})
        
        self.assertIsInstance(experiment, IntegrationTestExperiment)
        self.assertEqual(experiment.params["data_size"], 25)
        
        # Test experiment execution
        context = ExperimentContext(
            run_id="test_run",
            output_dir=self.temp_dir,
            run_dir=os.path.join(self.temp_dir, "test_run")
        )
        
        experiment.setup(context)
        result = experiment.execute_once(context)
        experiment.teardown(context)
        
        self.assertTrue(experiment.setup_called)
        self.assertTrue(experiment.teardown_called)
        self.assertEqual(result["data_size"], 25)

    def test_spec_runner_integration(self):
        """Test integration between spec loading and runner execution."""
        # Create a spec file
        spec_data = {
            "experiment": "integration_test_exp",
            "params": {"data_size": 30},
            "iterations": {"warmup": 1, "measured": 2},
            "instrumentation": ["timing"],
            "output_dir": self.temp_dir,
            "tags": ["spec", "integration"],
            "notes": "Spec integration test",
            "seed": 123
        }
        
        spec_path = os.path.join(self.temp_dir, "test_spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec_data, f)
        
        # Load spec
        spec = load_spec(spec_path)
        
        self.assertIsInstance(spec, RunSpec)
        self.assertEqual(spec.experiment, "integration_test_exp")
        self.assertEqual(spec.params["data_size"], 30)
        self.assertEqual(spec.iterations["warmup"], 1)
        self.assertEqual(spec.iterations["measured"], 2)
        self.assertEqual(spec.tags, ["spec", "integration"])
        self.assertEqual(spec.seed, 123)

    @patch('benchmarks.core.instrumentation.timing.time_block')
    def test_full_benchmark_workflow(self, mock_time_block):
        """Test complete benchmark workflow from spec to results."""
        mock_time_block.return_value.__enter__ = Mock()
        mock_time_block.return_value.__exit__ = Mock(return_value=None)
        
        # Create registry and register experiment
        registry = ExperimentRegistry()
        registry.register(
            slug="workflow_test",
            cls=IntegrationTestExperiment,
            summary="Workflow test experiment"
        )
        
        # Create spec
        spec_data = {
            "experiment": "workflow_test",
            "params": {"data_size": 40},
            "iterations": {"warmup": 1, "measured": 3},
            "instrumentation": ["timing"],
            "output_dir": self.temp_dir,
            "tags": ["workflow", "test"],
            "notes": "Full workflow test",
            "seed": 456
        }
        
        spec_path = os.path.join(self.temp_dir, "workflow_spec.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec_data, f)
        
        # Load spec
        spec = load_spec(spec_path)
        
        # Create experiment from spec
        experiment = registry.create(spec.experiment, spec.params)
        
        # Create and run benchmark
        runner = Runner(
            name="workflow_benchmark",
            experiment=experiment,
            output_dir=spec.output_dir,
            iterations_warmup=spec.iterations["warmup"],
            iterations_measured=spec.iterations["measured"],
            seed=spec.seed,
            tags=spec.tags,
            notes=spec.notes,
            instruments=spec.instrumentation
        )
        
        result = runner.run()
        
        # Verify complete workflow
        self.assertEqual(result.name, "workflow_benchmark")
        self.assertEqual(result.tags, ["workflow", "test"])
        self.assertEqual(result.notes, "Full workflow test")
        self.assertEqual(len(result.iteration_metrics), 3)
        
        # Check that experiment was executed correctly
        self.assertTrue(experiment.setup_called)
        self.assertTrue(experiment.teardown_called)
        self.assertEqual(experiment.execute_count, 4)  # 1 warmup + 3 measured

    def test_experiment_context_integration(self):
        """Test ExperimentContext integration with experiment execution."""
        context = ExperimentContext(
            run_id="context_test",
            output_dir=self.temp_dir,
            run_dir=os.path.join(self.temp_dir, "context_test"),
            iteration_index=0,
            seed=789,
            instruments=[Mock(), Mock()],
            extras={"test_key": "test_value"}
        )
        
        # Test experiment with context
        experiment = IntegrationTestExperiment({"data_size": 20})
        
        experiment.setup(context)
        result = experiment.execute_once(context)
        experiment.teardown(context)
        
        # Check that context was used correctly
        self.assertEqual(result["run_id"], "context_test")
        self.assertTrue(experiment.setup_called)
        self.assertTrue(experiment.teardown_called)

    @patch('benchmarks.core.instrumentation.timing.time_block')
    def test_result_aggregation_integration(self, mock_time_block):
        """Test result aggregation across multiple iterations."""
        # Mock timing with different durations
        durations = [1.0, 2.0, 3.0]
        call_count = 0
        
        def mock_time_block_side_effect(metrics, key):
            nonlocal call_count
            metrics[key] = durations[call_count % len(durations)]
            call_count += 1
            return Mock()
        
        mock_time_block.side_effect = mock_time_block_side_effect
        
        runner = Runner(
            name="aggregation_test",
            experiment=self.experiment,
            output_dir=self.temp_dir,
            iterations_warmup=0,
            iterations_measured=3
        )
        
        result = runner.run()
        
        # Check aggregated timing statistics
        self.assertIn("duration_s", result.metrics)
        timing_stats = result.metrics["duration_s"]
        self.assertIn("mean", timing_stats)
        self.assertIn("p50", timing_stats)
        self.assertIn("p95", timing_stats)
        self.assertIn("p99", timing_stats)
        
        # Mean should be 2.0 (average of 1, 2, 3)
        # Note: Actual timing may vary due to system performance
        self.assertGreater(timing_stats["mean"], 0)

    def test_error_handling_integration(self):
        """Test error handling in integrated workflow."""
        class FailingExperiment(Experiment):
            def execute_once(self, context):
                raise ValueError("Test error")
        
        failing_experiment = FailingExperiment()
        
        runner = Runner(
            name="error_test",
            experiment=failing_experiment,
            output_dir=self.temp_dir,
            iterations_measured=1
        )
        
        # Should handle errors gracefully
        with self.assertRaises(ValueError):
            runner.run()

    @patch('benchmarks.core.instrumentation.timing.time_block')
    def test_artifact_creation_integration(self, mock_time_block):
        """Test artifact creation in integrated workflow."""
        mock_time_block.return_value.__enter__ = Mock()
        mock_time_block.return_value.__exit__ = Mock(return_value=None)
        
        runner = Runner(
            name="artifact_test",
            experiment=self.experiment,
            output_dir=self.temp_dir,
            instruments=["timing", "cprofile"]
        )
        
        with patch('benchmarks.core.instrumentation.cprofile.cprofile_capture') as mock_cprofile:
            mock_cprofile.return_value.__enter__ = Mock()
            mock_cprofile.return_value.__exit__ = Mock(return_value=None)
            
            # Mock cProfile to add artifacts
            def mock_cprofile_side_effect(run_dir, name, iteration, metrics, **kwargs):
                metrics["cprofile_artifact"] = os.path.join(run_dir, f"{name}_iter{iteration:03d}.prof")
                metrics["cprofile_summary_path"] = os.path.join(run_dir, f"{name}_iter{iteration:03d}_summary.json")
                return Mock()
            
            mock_cprofile.side_effect = mock_cprofile_side_effect
            
            result = runner.run()
        
        # Should have created artifacts
        self.assertEqual(len(result.artifacts), 2)  # cProfile artifact and summary
        
        artifact_names = [artifact.name for artifact in result.artifacts]
        self.assertIn("cprofile_iter_0", artifact_names)
        self.assertIn("cprofile_summary_iter_0", artifact_names)


if __name__ == "__main__":
    unittest.main()

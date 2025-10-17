"""
Pytest configuration and fixtures for benchmarks tests.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock

from benchmarks.core.experiments import Experiment, ExperimentContext


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_experiment():
    """Create a mock experiment for testing."""
    class MockExperiment(Experiment):
        def __init__(self, params=None):
            super().__init__(params)
            self.setup_called = False
            self.teardown_called = False
            self.execute_count = 0

        def setup(self, context):
            self.setup_called = True

        def execute_once(self, context):
            self.execute_count += 1
            return {"iteration": self.execute_count, "run_id": context.run_id}

        def teardown(self, context):
            self.teardown_called = True

    return MockExperiment


@pytest.fixture
def experiment_context(temp_dir):
    """Create an experiment context for testing."""
    return ExperimentContext(
        run_id="test_run_123",
        output_dir=temp_dir,
        run_dir=os.path.join(temp_dir, "test_run_123"),
        iteration_index=0,
        seed=42,
        instruments=[],
        extras={}
    )


@pytest.fixture
def mock_instrumentation():
    """Mock instrumentation tools."""
    with pytest.MonkeyPatch().context() as m:
        # Mock timing instrumentation
        m.setattr('benchmarks.core.instrumentation.timing.time_block',
                 lambda metrics, key: Mock())

        # Mock cProfile instrumentation
        m.setattr('benchmarks.core.instrumentation.cprofile.cprofile_capture',
                 lambda *args, **kwargs: Mock())

        # Mock psutil instrumentation
        m.setattr('benchmarks.core.instrumentation.psutil_monitor.psutil_sampling',
                 lambda *args, **kwargs: Mock())

        yield m

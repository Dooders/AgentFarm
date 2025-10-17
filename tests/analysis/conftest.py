"""
Pytest fixtures for analysis module tests.

Provides reusable test data and configurations.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, Any

from farm.analysis.common.context import AnalysisContext
from farm.analysis.core import BaseAnalysisModule, SimpleDataProcessor


@pytest.fixture
def sample_simulation_data() -> pd.DataFrame:
    """Create sample simulation data for testing.

    Returns:
        DataFrame with realistic simulation data
    """
    np.random.seed(42)

    iterations = 10
    agent_types = ['system', 'independent', 'control']

    data = []
    for iteration in range(iterations):
        for agent_type in agent_types:
            data.append({
                'iteration': iteration,
                'agent_type': agent_type,
                'final_population': np.random.randint(10, 100),
                'avg_survival': np.random.uniform(50, 200),
                'reproduction_count': np.random.randint(5, 50),
                'dominance_score': np.random.uniform(0, 1),
                'initial_distance_to_resource': np.random.uniform(5, 50),
            })

    return pd.DataFrame(data)


@pytest.fixture
def empty_dataframe() -> pd.DataFrame:
    """Create empty DataFrame."""
    return pd.DataFrame()


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Create temporary output directory.

    Args:
        tmp_path: Pytest tmp_path fixture

    Returns:
        Path to temporary directory
    """
    output_dir = tmp_path / "analysis_output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def analysis_context(temp_output_dir) -> AnalysisContext:
    """Create analysis context for testing.

    Args:
        temp_output_dir: Temporary output directory

    Returns:
        AnalysisContext instance
    """
    return AnalysisContext(
        output_path=temp_output_dir,
        config={'test_mode': True},
        metadata={'test': 'fixture'}
    )


@pytest.fixture
def mock_data_processor():
    """Create mock data processor."""
    def process_func(data, **kwargs):
        if isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame({'test': [1, 2, 3]})

    return SimpleDataProcessor(process_func)


@pytest.fixture
def minimal_module(mock_data_processor):
    """Create minimal test module.

    Args:
        mock_data_processor: Mock data processor

    Returns:
        Minimal test module instance
    """
    class TestModule(BaseAnalysisModule):
        def __init__(self):
            super().__init__(
                name="test_module",
                description="Test module for testing"
            )

        def register_functions(self):
            def test_func(df, ctx, **kwargs):
                return {'result': 'success'}

            test_func.__name__ = 'test_func'

            self._functions = {'test_func': test_func}
            self._groups = {'all': [test_func]}

        def get_data_processor(self):
            return mock_data_processor

    return TestModule()


@pytest.fixture
def sample_analysis_results() -> Dict[str, Any]:
    """Create sample analysis results."""
    return {
        'metrics': {
            'mean_survival': 125.5,
            'total_reproductions': 250,
            'dominance_rate': 0.75
        },
        'correlations': {
            'survival_vs_reproduction': 0.82,
            'distance_vs_dominance': -0.45
        },
        'summary': 'Analysis completed successfully'
    }


@pytest.fixture
def config_service_mock():
    """Create mock configuration service."""
    class MockConfigService:
        def __init__(self):
            self._paths = []

        def get_analysis_module_paths(self, env_var: str = "FARM_ANALYSIS_MODULES"):
            return self._paths

        def set_module_paths(self, paths):
            self._paths = paths

    return MockConfigService()


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset module registry before each test."""
    from farm.analysis.registry import registry
    registry.clear()
    yield
    registry.clear()

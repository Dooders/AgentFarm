"""
Tests for core analysis module functionality.
"""

import pytest
import pandas as pd
from pathlib import Path

from farm.analysis.core import (
    BaseAnalysisModule,
    SimpleDataProcessor,
    ChainedDataProcessor,
    make_analysis_function
)
from farm.analysis.common.context import AnalysisContext
from farm.analysis.exceptions import (
    AnalysisFunctionError,
    DataProcessingError
)


class TestBaseAnalysisModule:
    """Tests for BaseAnalysisModule."""

    def test_module_initialization(self, minimal_module):
        """Test module initialization."""
        assert minimal_module.name == "test_module"
        assert minimal_module.description == "Test module for testing"
        assert not minimal_module._registered

    def test_lazy_registration(self, minimal_module):
        """Test that functions are registered lazily."""
        assert not minimal_module._registered

        # First access triggers registration
        funcs = minimal_module.get_analysis_functions()
        assert minimal_module._registered
        assert len(funcs) > 0

    def test_get_function_groups(self, minimal_module):
        """Test getting function groups."""
        groups = minimal_module.get_function_groups()
        assert 'all' in groups

    def test_get_function_by_name(self, minimal_module):
        """Test getting specific function by name."""
        func = minimal_module.get_function('test_func')
        assert func is not None
        assert callable(func)

    def test_get_nonexistent_function(self, minimal_module):
        """Test getting non-existent function returns None."""
        func = minimal_module.get_function('nonexistent')
        assert func is None

    def test_get_function_names(self, minimal_module):
        """Test getting all function names."""
        names = minimal_module.get_function_names()
        assert 'test_func' in names

    def test_get_info(self, minimal_module):
        """Test getting module info."""
        info = minimal_module.get_info()

        assert info['name'] == 'test_module'
        assert info['description'] == 'Test module for testing'
        assert 'function_groups' in info
        assert 'functions' in info
        assert isinstance(info['supports_database'], bool)

    def test_database_support(self, minimal_module):
        """Test database support methods."""
        assert minimal_module.supports_database() == False
        assert minimal_module.get_db_filename() is None
        assert minimal_module.get_db_loader() is None

    def test_run_analysis_basic(self, minimal_module, temp_output_dir, tmp_path):
        """Test basic analysis run."""
        experiment_path = tmp_path / "experiment"
        experiment_path.mkdir()

        output_path, df = minimal_module.run_analysis(
            experiment_path=experiment_path,
            output_path=temp_output_dir
        )

        assert output_path == temp_output_dir
        # Data processor returns test dataframe
        assert df is not None or df.empty  # Depends on processor

    def test_run_analysis_with_progress(self, minimal_module, temp_output_dir, tmp_path):
        """Test analysis run with progress callback."""
        experiment_path = tmp_path / "experiment"
        experiment_path.mkdir()

        progress_calls = []

        def progress_callback(message: str, progress: float):
            progress_calls.append((message, progress))

        minimal_module.run_analysis(
            experiment_path=experiment_path,
            output_path=temp_output_dir,
            progress_callback=progress_callback
        )

        # Should have received progress updates
        assert len(progress_calls) > 0
        assert all(0.0 <= p <= 1.0 for _, p in progress_calls)

    def test_run_analysis_with_kwargs(self, minimal_module, temp_output_dir, tmp_path):
        """Test analysis run with function kwargs."""
        experiment_path = tmp_path / "experiment"
        experiment_path.mkdir()

        analysis_kwargs = {
            'test_func': {'custom_param': 'value'}
        }

        minimal_module.run_analysis(
            experiment_path=experiment_path,
            output_path=temp_output_dir,
            analysis_kwargs=analysis_kwargs
        )

        # Should complete without error


class TestSimpleDataProcessor:
    """Tests for SimpleDataProcessor."""

    def test_process_with_dataframe(self, sample_simulation_data):
        """Test processing with DataFrame input."""
        def process_func(data, **kwargs):
            return data

        processor = SimpleDataProcessor(process_func)
        result = processor.process(sample_simulation_data)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_simulation_data)

    def test_process_with_kwargs(self):
        """Test processing with additional kwargs."""
        def process_func(data, multiplier=1, **kwargs):
            df = pd.DataFrame({'value': [1, 2, 3]})
            df['value'] = df['value'] * multiplier
            return df

        processor = SimpleDataProcessor(process_func)
        result = processor.process(None, multiplier=10)

        assert result['value'].tolist() == [10, 20, 30]


class TestChainedDataProcessor:
    """Tests for ChainedDataProcessor."""

    def test_chain_processors(self):
        """Test chaining multiple processors."""
        def add_column(df, **kwargs):
            df = df.copy()
            df['new_col'] = 1
            return df

        def multiply_values(df, **kwargs):
            df = df.copy()
            df['value'] = df['value'] * 2
            return df

        processor1 = SimpleDataProcessor(add_column)
        processor2 = SimpleDataProcessor(multiply_values)

        chained = ChainedDataProcessor([processor1, processor2])

        df = pd.DataFrame({'value': [1, 2, 3]})
        result = chained.process(df)

        assert 'new_col' in result.columns
        assert result['value'].tolist() == [2, 4, 6]

    def test_empty_chain(self):
        """Test empty processor chain."""
        chained = ChainedDataProcessor([])
        df = pd.DataFrame({'value': [1, 2, 3]})
        result = chained.process(df)

        # Should return unchanged
        assert result.equals(df)


class TestMakeAnalysisFunction:
    """Tests for make_analysis_function wrapper."""

    def test_wrap_modern_function(self, sample_simulation_data, analysis_context):
        """Test wrapping function with ctx parameter."""
        def modern_func(df: pd.DataFrame, ctx: AnalysisContext, **kwargs):
            return {'data_len': len(df), 'output_path': str(ctx.output_path)}

        wrapped = make_analysis_function(modern_func)
        result = wrapped(sample_simulation_data, analysis_context)

        assert result['data_len'] == len(sample_simulation_data)
        assert 'output_path' in result

    def test_wrap_legacy_output_path_function(self, sample_simulation_data, analysis_context):
        """Test wrapping function with output_path parameter."""
        def legacy_func(df: pd.DataFrame, output_path: str, **kwargs):
            return {'path': output_path}

        wrapped = make_analysis_function(legacy_func)
        result = wrapped(sample_simulation_data, analysis_context)

        assert str(analysis_context.output_path) in result['path']

    def test_wrap_simple_function(self, sample_simulation_data, analysis_context):
        """Test wrapping simple function with only df parameter."""
        def simple_func(df, **kwargs):
            return len(df)

        wrapped = make_analysis_function(simple_func)
        result = wrapped(sample_simulation_data, analysis_context)

        assert result == len(sample_simulation_data)

    def test_custom_name(self, sample_simulation_data, analysis_context):
        """Test setting custom name for wrapped function."""
        def original_func(df, ctx, **kwargs):
            return None

        wrapped = make_analysis_function(original_func, name="custom_name")

        assert wrapped.__name__ == "custom_name"


class TestAnalysisContext:
    """Tests for AnalysisContext."""

    def test_context_initialization(self, temp_output_dir):
        """Test context initialization."""
        ctx = AnalysisContext(output_path=temp_output_dir)

        assert ctx.output_path == temp_output_dir
        assert isinstance(ctx.config, dict)
        assert isinstance(ctx.services, dict)
        assert ctx.logger is not None

    def test_string_path_conversion(self, tmp_path):
        """Test automatic conversion of string path to Path."""
        ctx = AnalysisContext(output_path=str(tmp_path))

        assert isinstance(ctx.output_path, Path)
        assert ctx.output_path == tmp_path

    def test_output_directory_creation(self, tmp_path):
        """Test automatic creation of output directory."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        ctx = AnalysisContext(output_path=new_dir)

        assert new_dir.exists()

    def test_get_output_file(self, temp_output_dir):
        """Test getting output file path."""
        ctx = AnalysisContext(output_path=temp_output_dir)

        file_path = ctx.get_output_file("test.csv")
        assert file_path == temp_output_dir / "test.csv"

        # With subdirectory
        subdir_path = ctx.get_output_file("test.png", subdir="plots")
        assert subdir_path == temp_output_dir / "plots" / "test.png"
        assert subdir_path.parent.exists()

    def test_report_progress(self, temp_output_dir):
        """Test progress reporting."""
        progress_calls = []

        def callback(message: str, progress: float):
            progress_calls.append((message, progress))

        ctx = AnalysisContext(
            output_path=temp_output_dir,
            progress_callback=callback
        )

        ctx.report_progress("Starting", 0.0)
        ctx.report_progress("Halfway", 0.5)
        ctx.report_progress("Done", 1.0)

        assert len(progress_calls) == 3
        assert progress_calls[0] == ("Starting", 0.0)
        assert progress_calls[1] == ("Halfway", 0.5)
        assert progress_calls[2] == ("Done", 1.0)

    def test_get_config(self, temp_output_dir):
        """Test getting configuration values."""
        ctx = AnalysisContext(
            output_path=temp_output_dir,
            config={'key1': 'value1', 'key2': 42}
        )

        assert ctx.get_config('key1') == 'value1'
        assert ctx.get_config('key2') == 42
        assert ctx.get_config('nonexistent', 'default') == 'default'

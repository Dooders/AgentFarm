"""
Tests for common analysis utilities.

Tests the utility functions in farm.analysis.common.utils.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from farm.analysis.common.utils import (
    calculate_statistics,
    calculate_trend,
    calculate_rolling_mean,
    normalize_dict,
    create_output_subdirs,
    validate_required_columns,
    align_time_series,
    find_database_path,
    convert_dict_to_dataframe,
    save_analysis_results,
    compute_basic_metrics,
    setup_plot_figure,
    save_plot_figure,
    get_agent_type_colors,
    normalize_agent_type_names,
    validate_data_quality,
    handle_missing_data,
)


class TestStatisticalFunctions:
    """Test statistical computation functions."""

    def test_calculate_statistics(self):
        """Test comprehensive statistics calculation."""
        data = np.array([1, 2, 3, 4, 5])

        stats = calculate_statistics(data)

        assert stats['mean'] == 3.0
        assert stats['median'] == 3.0
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['q25'] == 2.0
        assert stats['q75'] == 4.0
        assert 'std' in stats

    def test_calculate_trend(self):
        """Test linear trend calculation."""
        # Increasing data
        data = np.array([1, 2, 3, 4, 5])
        trend = calculate_trend(data)
        assert trend > 0

        # Decreasing data
        data = np.array([5, 4, 3, 2, 1])
        trend = calculate_trend(data)
        assert trend < 0

        # No trend
        data = np.array([1, 1, 1])
        trend = calculate_trend(data)
        assert abs(trend) < 1e-6

    def test_calculate_rolling_mean(self):
        """Test rolling mean calculation."""
        data = np.array([1, 2, 3, 4, 5])
        rolling = calculate_rolling_mean(data, window=3)

        # Should have length 3 (5 - 3 + 1)
        assert len(rolling) == 3
        assert abs(rolling[0] - (1 + 2 + 3) / 3) < 1e-10  # 2.0
        assert abs(rolling[1] - (2 + 3 + 4) / 3) < 1e-10  # 3.0
        assert abs(rolling[2] - (3 + 4 + 5) / 3) < 1e-10  # 4.0

    def test_normalize_dict(self):
        """Test dictionary normalization."""
        d = {'a': 10, 'b': 20, 'c': 30}
        normalized = normalize_dict(d)

        assert abs(normalized['a'] - 10/60) < 1e-10  # 10/60 = 1/6 ≈ 0.1667
        assert abs(normalized['b'] - 20/60) < 1e-10  # 20/60 = 1/3 ≈ 0.3333
        assert abs(normalized['c'] - 30/60) < 1e-10  # 30/60 = 0.5

    def test_normalize_empty_dict(self):
        """Test normalization with empty dict."""
        normalized = normalize_dict({})
        assert normalized == {}


class TestDataProcessingFunctions:
    """Test data processing utility functions."""

    def test_validate_required_columns_success(self):
        """Test successful column validation."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        # Should not raise
        validate_required_columns(df, ['a', 'b'])

    def test_validate_required_columns_missing(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({'a': [1, 2]})

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_required_columns(df, ['a', 'b', 'c'])

    def test_align_time_series(self):
        """Test time series alignment."""
        data_list = [
            np.array([1, 2, 3]),
            np.array([4, 5]),  # Shorter
            np.array([6, 7, 8, 9])  # Longer
        ]

        aligned = align_time_series(data_list)

        assert aligned.shape[0] == 3  # 3 series
        assert aligned.shape[1] == 4  # Max length
        assert aligned[0][3] == 3  # Padded with last value
        assert aligned[1][2] == 5  # Padded with last value
        assert aligned[1][3] == 5  # Padded with last value


class TestFileSystemFunctions:
    """Test file system utility functions."""

    def test_create_output_subdirs(self, tmp_path):
        """Test output subdirectory creation."""
        subdirs = ['plots', 'data', 'reports']
        paths = create_output_subdirs(tmp_path, subdirs)

        assert 'plots' in paths
        assert 'data' in paths
        assert 'reports' in paths

        assert (tmp_path / 'plots').exists()
        assert (tmp_path / 'data').exists()
        assert (tmp_path / 'reports').exists()

    def test_find_database_path_found(self, tmp_path):
        """Test finding database when it exists."""
        db_path = tmp_path / "simulation.db"
        db_path.touch()

        found_path = find_database_path(tmp_path)
        assert found_path == db_path

    def test_find_database_path_not_found(self, tmp_path):
        """Test error when database not found."""
        with pytest.raises(FileNotFoundError):
            find_database_path(tmp_path)


class TestDataFrameFunctions:
    """Test DataFrame utility functions."""

    def test_convert_dict_to_dataframe_time_series(self):
        """Test converting time series dict to DataFrame."""
        data = {
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'metric': 100
        }

        df = convert_dict_to_dataframe(data)

        assert len(df) == 3
        assert 'col1' in df.columns
        assert 'col2' in df.columns
        assert 'metric' in df.columns
        assert 'step' in df.columns  # Should be added

    def test_convert_dict_to_dataframe_single_values(self):
        """Test converting single values dict to DataFrame."""
        data = {
            'metric1': 100,
            'metric2': 200
        }

        df = convert_dict_to_dataframe(data)

        assert len(df) == 1
        assert df.iloc[0]['metric1'] == 100
        assert df.iloc[0]['metric2'] == 200

    def test_save_analysis_results(self, tmp_path):
        """Test saving analysis results."""
        results = {
            'metrics': {'mean': 42.0},
            'summary': 'Test completed'
        }

        saved_path = save_analysis_results(results, 'test_results.json', tmp_path)

        assert saved_path.exists()
        assert saved_path.name == 'test_results.json'

        # Verify content
        with open(saved_path) as f:
            loaded = json.load(f)
        assert loaded == results

    def test_compute_basic_metrics(self):
        """Test basic metrics computation."""
        df = pd.DataFrame({
            'numeric_col': [1, 2, 3, 4, 5],
            'string_col': ['a', 'b', 'c', 'd', 'e']
        })

        metrics = compute_basic_metrics(df, ['numeric_col', 'string_col'])

        assert 'numeric_col' in metrics
        assert metrics['numeric_col']['mean'] == 3.0
        assert 'string_col' not in metrics  # Not numeric


class TestPlottingFunctions:
    """Test plotting utility functions."""

    def test_get_agent_type_colors(self):
        """Test agent type color scheme."""
        colors = get_agent_type_colors()

        assert 'system' in colors
        assert 'independent' in colors
        assert 'control' in colors
        assert colors['system'] == 'blue'
        assert colors['independent'] == 'red'

    def test_normalize_agent_type_names(self):
        """Test agent type name normalization."""
        names = ['SystemAgent', 'IndependentAgent', 'ControlAgent', 'unknown']
        normalized = normalize_agent_type_names(names)

        assert normalized[0] == 'system'
        assert normalized[1] == 'independent'
        assert normalized[2] == 'control'
        assert normalized[3] == 'unknown'  # Unchanged

    def test_setup_plot_figure_single(self):
        """Test single plot figure setup."""
        fig, ax = setup_plot_figure(n_plots=1)

        assert fig is not None
        assert ax is not None

        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_setup_plot_figure_multiple(self):
        """Test multiple plot figure setup."""
        fig, axes = setup_plot_figure(n_plots=2)

        assert fig is not None
        assert len(axes) == 2

        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_save_plot_figure(self, tmp_path):
        """Test saving plot figure."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])

        saved_path = save_plot_figure(fig, tmp_path, 'test_plot.png')

        assert saved_path.exists()
        assert saved_path.name == 'test_plot.png'


class TestValidationFunctions:
    """Test data validation functions."""

    def test_validate_data_quality_success(self):
        """Test successful data quality validation."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4.0, 5.0, 6.0]
        })

        # Should not raise
        validate_data_quality(df, min_rows=2, required_numeric_cols=['col1', 'col2'])

    def test_validate_data_quality_insufficient_rows(self):
        """Test validation with insufficient rows."""
        df = pd.DataFrame({'col1': [1]})

        with pytest.raises(ValueError, match="must have at least"):
            validate_data_quality(df, min_rows=2)

    def test_validate_data_quality_missing_column(self):
        """Test validation with missing numeric column."""
        df = pd.DataFrame({'col1': [1, 2]})

        with pytest.raises(ValueError, match="Missing required column"):
            validate_data_quality(df, required_numeric_cols=['col1', 'col2'])

    def test_validate_data_quality_non_numeric(self):
        """Test validation with non-numeric column."""
        df = pd.DataFrame({
            'col1': [1, 2],
            'col2': ['a', 'b']
        })

        with pytest.raises(ValueError, match="must be numeric"):
            validate_data_quality(df, required_numeric_cols=['col1', 'col2'])

    def test_handle_missing_data_drop(self):
        """Test dropping missing data."""
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': [4, 5, None]
        })

        cleaned = handle_missing_data(df, strategy='drop')

        assert len(cleaned) == 1  # Only row with no missing values
        assert cleaned.iloc[0]['col1'] == 1
        assert cleaned.iloc[0]['col2'] == 4

    def test_handle_missing_data_fill_mean(self):
        """Test filling missing data with mean."""
        df = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': [4, 5, 6]
        })

        cleaned = handle_missing_data(df, strategy='fill_mean')

        assert cleaned.loc[1, 'col1'] == 2.0  # Mean of [1, 3]

    def test_handle_missing_data_fill_zero(self):
        """Test filling missing data with zero."""
        df = pd.DataFrame({
            'col1': [1, None, 3],
        })

        cleaned = handle_missing_data(df, strategy='fill_zero')

        assert cleaned.loc[1, 'col1'] == 0

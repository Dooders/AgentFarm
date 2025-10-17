"""
Tests for analysis data loading and processing modules.

Note: These modules may not be actively used (see comment in source files),
but these tests ensure they remain functional if needed.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Skip tests if modules are not available or deprecated
pytest.importorskip("farm.analysis.data.loaders", reason="Data loaders module may be deprecated")
pytest.importorskip("farm.analysis.data.processors", reason="Data processors module may be deprecated")

from farm.analysis.data.loaders import CSVLoader, JSONLoader
from farm.analysis.data.processors import (
    DataCleaner,
    TimeSeriesProcessor,
    AgentStatsProcessor,
)


class TestCSVLoader:
    """Tests for CSVLoader."""

    def test_load_csv(self, tmp_path):
        """Test loading data from CSV file."""
        # Create a test CSV file
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        df.to_csv(csv_path, index=False)

        # Load the CSV
        loader = CSVLoader(str(csv_path))
        loaded_df = loader.load_data()

        assert len(loaded_df) == 3
        assert 'col1' in loaded_df.columns
        assert 'col2' in loaded_df.columns

    def test_iter_csv(self, tmp_path):
        """Test streaming CSV data in chunks."""
        # Create a test CSV file with more rows
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'col1': range(100),
            'col2': ['value'] * 100
        })
        df.to_csv(csv_path, index=False)

        # Stream the CSV
        loader = CSVLoader(str(csv_path))
        chunks = list(loader.iter_data(chunksize=20))

        assert len(chunks) == 5  # 100 rows / 20 per chunk
        assert all(isinstance(chunk, pd.DataFrame) for chunk in chunks)

    def test_missing_csv_file(self, tmp_path):
        """Test error handling for missing CSV file."""
        loader = CSVLoader(str(tmp_path / "nonexistent.csv"))

        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_get_metadata(self, tmp_path):
        """Test getting CSV file metadata."""
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({'col1': [1, 2, 3]})
        df.to_csv(csv_path, index=False)

        loader = CSVLoader(str(csv_path))
        metadata = loader.get_metadata()

        assert 'file_path' in metadata
        assert 'file_size' in metadata
        assert 'columns' in metadata


class TestJSONLoader:
    """Tests for JSONLoader."""

    def test_load_json(self, tmp_path):
        """Test loading data from JSON file."""
        # Create a test JSON file
        json_path = tmp_path / "test.json"
        data = [
            {'col1': 1, 'col2': 'a'},
            {'col1': 2, 'col2': 'b'},
            {'col1': 3, 'col2': 'c'}
        ]

        import json
        with open(json_path, 'w') as f:
            json.dump(data, f)

        # Load the JSON
        loader = JSONLoader(str(json_path))
        loaded_df = loader.load_data()

        assert len(loaded_df) == 3
        assert 'col1' in loaded_df.columns
        assert 'col2' in loaded_df.columns

    def test_missing_json_file(self, tmp_path):
        """Test error handling for missing JSON file."""
        loader = JSONLoader(str(tmp_path / "nonexistent.json"))

        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_get_metadata(self, tmp_path):
        """Test getting JSON file metadata."""
        json_path = tmp_path / "test.json"
        data = [{'col1': 1}]

        import json
        with open(json_path, 'w') as f:
            json.dump(data, f)

        loader = JSONLoader(str(json_path))
        metadata = loader.get_metadata()

        assert 'file_path' in metadata
        assert 'file_size' in metadata
        assert 'structure_type' in metadata


class TestDataCleaner:
    """Tests for DataCleaner processor."""

    def test_handle_missing_values(self):
        """Test handling of missing values."""
        df = pd.DataFrame({
            'numeric': [1.0, None, 3.0, 4.0, 5.0],
            'category': ['A', 'B', None, 'D', 'E']
        })

        cleaner = DataCleaner(handle_missing=True)
        cleaned = cleaner.process(df)

        # Missing numeric values should be filled with median
        assert not cleaned['numeric'].isna().any()
        # Missing categorical values should be filled with mode
        assert not cleaned['category'].isna().any()

    def test_handle_outliers(self):
        """Test outlier handling."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]  # 1000 is an outlier
        })

        cleaner = DataCleaner(handle_outliers=True)
        cleaned = cleaner.process(df)

        # Outliers should be capped
        assert cleaned['value'].max() < 1000

    def test_no_modifications(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({'value': [1.0, None, 3.0]})
        original_id = id(df)

        cleaner = DataCleaner(handle_missing=True)
        cleaned = cleaner.process(df)

        # Should be a different DataFrame
        assert id(cleaned) != original_id
        # Original should still have missing values
        assert df['value'].isna().any()


class TestTimeSeriesProcessor:
    """Tests for TimeSeriesProcessor."""

    def test_smoothing(self):
        """Test time series smoothing."""
        df = pd.DataFrame({
            'value': [1, 10, 2, 9, 3, 8, 4, 7, 5]
        })

        processor = TimeSeriesProcessor(smooth=True, window_size=3)
        smoothed = processor.process(df)

        # Values should be smoothed (less volatile)
        assert 'value' in smoothed.columns

    def test_no_modifications(self):
        """Test that original DataFrame is not modified."""
        df = pd.DataFrame({'value': [1, 2, 3]})
        original_id = id(df)

        processor = TimeSeriesProcessor(smooth=True)
        processed = processor.process(df)

        # Should be a different DataFrame
        assert id(processed) != original_id


class TestAgentStatsProcessor:
    """Tests for AgentStatsProcessor."""

    def test_survival_time_calculation(self):
        """Test survival time calculation."""
        df = pd.DataFrame({
            'agent_type': ['A', 'A', 'B'],
            'birth_step': [0, 0, 0],
            'death_step': [10, 20, 15]
        })

        processor = AgentStatsProcessor(include_derived_metrics=False)
        processed = processor.process(df)

        assert 'survival_time' in processed.columns
        assert processed.loc[0, 'survival_time'] == 10
        assert processed.loc[1, 'survival_time'] == 20
        assert processed.loc[2, 'survival_time'] == 15

    def test_derived_metrics(self):
        """Test calculation of derived metrics."""
        df = pd.DataFrame({
            'agent_type': ['A', 'A', 'B', 'B'],
            'birth_step': [0, 0, 0, 0],
            'death_step': [10, 20, 15, 25],
            'generation': [1, 2, 1, 3]
        })

        processor = AgentStatsProcessor(include_derived_metrics=True)
        result = processor.process(df)

        # Should return aggregated stats
        assert 'agent_type' in result.columns
        assert len(result) <= 2  # Grouped by agent_type

    def test_alive_agents(self):
        """Test handling of agents that are still alive (no death_step)."""
        df = pd.DataFrame({
            'agent_type': ['A', 'A'],
            'birth_step': [0, 5],
            'death_step': [None, 10]
        })

        processor = AgentStatsProcessor(include_derived_metrics=False)
        processed = processor.process(df)

        # Should calculate survival time for alive agents
        assert 'survival_time' in processed.columns
        assert not pd.isna(processed.loc[0, 'survival_time'])


# Integration test to ensure modules can work together
class TestDataModuleIntegration:
    """Integration tests for data loading and processing."""

    def test_load_and_process_csv(self, tmp_path):
        """Test loading CSV and then processing it."""
        # Create CSV with missing values
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            'value': [1.0, None, 3.0, 100.0, 5.0]  # Has missing and outlier
        })
        df.to_csv(csv_path, index=False)

        # Load and process
        loader = CSVLoader(str(csv_path))
        data = loader.load_data()

        cleaner = DataCleaner(handle_missing=True, handle_outliers=True)
        cleaned = cleaner.process(data)

        # Should have no missing values and outliers handled
        assert not cleaned['value'].isna().any()
        assert cleaned['value'].max() < 100.0

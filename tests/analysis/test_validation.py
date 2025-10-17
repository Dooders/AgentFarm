"""
Tests for data validation.
"""

import pytest
import pandas as pd
import numpy as np

from farm.analysis.validation import (
    ColumnValidator,
    DataQualityValidator,
    CompositeValidator,
    validate_numeric_columns,
    validate_simulation_data
)
from farm.analysis.exceptions import DataValidationError, InsufficientDataError


class TestColumnValidator:
    """Tests for ColumnValidator."""

    def test_valid_data(self, sample_simulation_data):
        """Test validation of valid data."""
        validator = ColumnValidator(
            required_columns=['iteration', 'agent_type'],
            column_types={'iteration': int, 'agent_type': str}
        )

        # Should not raise
        validator.validate(sample_simulation_data)

    def test_missing_required_columns(self, sample_simulation_data):
        """Test detection of missing required columns."""
        validator = ColumnValidator(
            required_columns=['iteration', 'nonexistent_column']
        )

        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(sample_simulation_data)

        assert 'nonexistent_column' in exc_info.value.missing_columns

    def test_wrong_column_type(self):
        """Test detection of wrong column types."""
        df = pd.DataFrame({
            'numeric_col': ['a', 'b', 'c'],  # String instead of numeric
            'string_col': ['x', 'y', 'z']
        })

        validator = ColumnValidator(
            required_columns=['numeric_col'],
            column_types={'numeric_col': float}
        )

        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(df)

        assert 'numeric_col' in exc_info.value.invalid_columns

    def test_get_required_columns(self):
        """Test getting list of required columns."""
        validator = ColumnValidator(
            required_columns=['col1', 'col2', 'col3']
        )

        required = validator.get_required_columns()
        assert sorted(required) == ['col1', 'col2', 'col3']


class TestDataQualityValidator:
    """Tests for DataQualityValidator."""

    def test_minimum_rows(self, sample_simulation_data):
        """Test minimum row validation."""
        validator = DataQualityValidator(min_rows=5)
        validator.validate(sample_simulation_data)  # Should pass

        validator_strict = DataQualityValidator(min_rows=1000)
        with pytest.raises(InsufficientDataError) as exc_info:
            validator_strict.validate(sample_simulation_data)

        assert exc_info.value.required_rows == 1000
        assert exc_info.value.actual_rows == len(sample_simulation_data)

    def test_empty_dataframe(self, empty_dataframe):
        """Test validation of empty DataFrame."""
        validator = DataQualityValidator()

        with pytest.raises(InsufficientDataError) as exc_info:
            validator.validate(empty_dataframe)

        assert 'empty' in str(exc_info.value).lower()

    def test_excessive_nulls(self):
        """Test detection of excessive null values."""
        df = pd.DataFrame({
            'col1': [1, 2, None, None, None],
            'col2': [1, 2, 3, 4, 5]
        })

        validator = DataQualityValidator(max_null_fraction=0.3)

        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(df)

        assert 'null' in str(exc_info.value).lower()

    def test_duplicates(self):
        """Test duplicate detection."""
        df = pd.DataFrame({
            'col1': [1, 2, 2, 3],
            'col2': ['a', 'b', 'b', 'c']
        })

        # Should fail with allow_duplicates=False
        validator = DataQualityValidator(allow_duplicates=False)
        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(df)

        assert 'duplicate' in str(exc_info.value).lower()

        # Should pass with allow_duplicates=True
        validator_permissive = DataQualityValidator(allow_duplicates=True)
        validator_permissive.validate(df)  # Should not raise

    def test_value_ranges(self):
        """Test value range validation."""
        df = pd.DataFrame({
            'score': [0.5, 0.8, 0.3, 1.5],  # 1.5 is out of range
        })

        validator = DataQualityValidator(
            value_ranges={'score': (0.0, 1.0)}
        )

        with pytest.raises(DataValidationError) as exc_info:
            validator.validate(df)

        assert 'score' in str(exc_info.value)
        assert 'range' in str(exc_info.value).lower()

    def test_custom_checks(self):
        """Test custom validation checks."""
        def check_positive(df):
            if (df['value'] < 0).any():
                raise DataValidationError("Negative values not allowed")

        df_valid = pd.DataFrame({'value': [1, 2, 3]})
        df_invalid = pd.DataFrame({'value': [1, -2, 3]})

        validator = DataQualityValidator(custom_checks=[check_positive])

        validator.validate(df_valid)  # Should pass

        with pytest.raises(DataValidationError):
            validator.validate(df_invalid)


class TestCompositeValidator:
    """Tests for CompositeValidator."""

    def test_combines_validators(self, sample_simulation_data):
        """Test combining multiple validators."""
        col_validator = ColumnValidator(required_columns=['iteration'])
        quality_validator = DataQualityValidator(min_rows=1)

        composite = CompositeValidator([col_validator, quality_validator])
        composite.validate(sample_simulation_data)  # Should pass

    def test_fails_on_any_validator_failure(self):
        """Test that composite fails if any validator fails."""
        df = pd.DataFrame({'col1': [1, 2, 3]})

        col_validator = ColumnValidator(required_columns=['missing_col'])
        quality_validator = DataQualityValidator(min_rows=1)

        composite = CompositeValidator([col_validator, quality_validator])

        with pytest.raises(DataValidationError):
            composite.validate(df)

    def test_get_required_columns(self):
        """Test getting combined required columns."""
        col_validator1 = ColumnValidator(required_columns=['col1', 'col2'])
        col_validator2 = ColumnValidator(required_columns=['col2', 'col3'])

        composite = CompositeValidator([col_validator1, col_validator2])
        required = composite.get_required_columns()

        assert set(required) == {'col1', 'col2', 'col3'}


class TestValidateNumericColumns:
    """Tests for validate_numeric_columns function."""

    def test_valid_numeric_columns(self, sample_simulation_data):
        """Test filtering valid numeric columns."""
        columns = ['final_population', 'avg_survival', 'dominance_score']
        result = validate_numeric_columns(sample_simulation_data, columns)

        assert result == columns

    def test_mixed_columns(self, sample_simulation_data):
        """Test with mix of numeric and non-numeric."""
        columns = ['final_population', 'agent_type']  # agent_type is string

        with pytest.raises(DataValidationError):
            validate_numeric_columns(sample_simulation_data, columns)

    def test_allow_missing(self, sample_simulation_data):
        """Test allow_missing parameter."""
        columns = ['final_population', 'nonexistent_column']

        # With allow_missing=True, should skip nonexistent
        result = validate_numeric_columns(
            sample_simulation_data, columns, allow_missing=True
        )
        assert result == ['final_population']

        # With allow_missing=False, should raise
        with pytest.raises(DataValidationError):
            validate_numeric_columns(
                sample_simulation_data, columns, allow_missing=False
            )


class TestValidateSimulationData:
    """Tests for validate_simulation_data function."""

    def test_valid_simulation_data(self, sample_simulation_data):
        """Test validation of valid simulation data."""
        # Add simulation_id to make it valid
        df = sample_simulation_data.copy()
        df['simulation_id'] = 1

        validate_simulation_data(df)  # Should not raise

    def test_missing_simulation_id(self, sample_simulation_data):
        """Test detection of missing simulation_id."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_simulation_data(sample_simulation_data)

        assert 'simulation_id' in str(exc_info.value)

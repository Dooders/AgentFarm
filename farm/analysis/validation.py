"""
Data validation utilities for the analysis module system.

Provides validators to ensure data meets required schemas and constraints.
"""

from typing import List, Set, Dict, Optional, Any, Callable
import pandas as pd
import numpy as np
from farm.analysis.exceptions import DataValidationError, InsufficientDataError


class ColumnValidator:
    """Validates DataFrame columns meet requirements."""
    
    def __init__(
        self,
        required_columns: Optional[List[str]] = None,
        optional_columns: Optional[List[str]] = None,
        column_types: Optional[Dict[str, type]] = None
    ):
        """Initialize column validator.
        
        Args:
            required_columns: Columns that must be present
            optional_columns: Columns that may be present
            column_types: Expected types for columns {column_name: expected_type}
        """
        self.required_columns = set(required_columns or [])
        self.optional_columns = set(optional_columns or [])
        self.column_types = column_types or {}
    
    def validate(self, df: pd.DataFrame) -> None:
        """Validate DataFrame has required columns with correct types.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            DataValidationError: If validation fails
        """
        # Check for missing required columns
        missing = self.required_columns - set(df.columns)
        if missing:
            raise DataValidationError(
                f"Missing required columns",
                missing_columns=missing
            )
        
        # Check column types
        invalid_types = {}
        for col_name, expected_type in self.column_types.items():
            if col_name in df.columns:
                actual_dtype = df[col_name].dtype
                
                # Check numeric types
                if expected_type in (int, float, np.number):
                    if not pd.api.types.is_numeric_dtype(actual_dtype):
                        invalid_types[col_name] = f"Expected numeric, got {actual_dtype}"
                
                # Check string types
                elif expected_type == str:
                    if not pd.api.types.is_string_dtype(actual_dtype) and actual_dtype != object:
                        invalid_types[col_name] = f"Expected string, got {actual_dtype}"
                
                # Check datetime types
                elif expected_type in (pd.Timestamp, np.datetime64):
                    if not pd.api.types.is_datetime64_any_dtype(actual_dtype):
                        invalid_types[col_name] = f"Expected datetime, got {actual_dtype}"
        
        if invalid_types:
            raise DataValidationError(
                "Column type mismatches found",
                invalid_columns=invalid_types
            )
    
    def get_required_columns(self) -> List[str]:
        """Get list of required columns."""
        return sorted(self.required_columns)


class DataQualityValidator:
    """Validates data quality (nulls, duplicates, ranges, etc.)."""
    
    def __init__(
        self,
        min_rows: Optional[int] = None,
        max_null_fraction: float = 1.0,
        allow_duplicates: bool = True,
        value_ranges: Optional[Dict[str, tuple]] = None,
        custom_checks: Optional[List[Callable[[pd.DataFrame], None]]] = None
    ):
        """Initialize data quality validator.
        
        Args:
            min_rows: Minimum number of rows required
            max_null_fraction: Maximum fraction of null values allowed per column
            allow_duplicates: Whether duplicate rows are allowed
            value_ranges: Expected ranges for columns {column: (min, max)}
            custom_checks: Additional validation functions to run
        """
        self.min_rows = min_rows
        self.max_null_fraction = max_null_fraction
        self.allow_duplicates = allow_duplicates
        self.value_ranges = value_ranges or {}
        self.custom_checks = custom_checks or []
    
    def validate(self, df: pd.DataFrame) -> None:
        """Validate data quality.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            InsufficientDataError: If not enough data
            DataValidationError: If quality checks fail
        """
        # Check minimum rows
        if self.min_rows is not None and len(df) < self.min_rows:
            raise InsufficientDataError(
                "Insufficient data for analysis",
                required_rows=self.min_rows,
                actual_rows=len(df)
            )
        
        # Check for empty DataFrame
        if df.empty:
            raise InsufficientDataError("DataFrame is empty")
        
        # Check null fractions
        if self.max_null_fraction < 1.0:
            null_fractions = df.isnull().sum() / len(df)
            excessive_nulls = null_fractions[null_fractions > self.max_null_fraction]
            if not excessive_nulls.empty:
                raise DataValidationError(
                    f"Columns with excessive null values: {excessive_nulls.to_dict()}"
                )
        
        # Check duplicates
        if not self.allow_duplicates and df.duplicated().any():
            dup_count = df.duplicated().sum()
            raise DataValidationError(
                f"Found {dup_count} duplicate rows (duplicates not allowed)"
            )
        
        # Check value ranges
        for col, (min_val, max_val) in self.value_ranges.items():
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    out_of_range = (df[col] < min_val) | (df[col] > max_val)
                    if out_of_range.any():
                        count = out_of_range.sum()
                        raise DataValidationError(
                            f"Column '{col}' has {count} values outside range [{min_val}, {max_val}]"
                        )
        
        # Run custom checks
        for check in self.custom_checks:
            check(df)


class CompositeValidator:
    """Combines multiple validators into one."""
    
    def __init__(self, validators: List[Any]):
        """Initialize composite validator.
        
        Args:
            validators: List of validator objects (must have validate() method)
        """
        self.validators = validators
    
    def validate(self, df: pd.DataFrame) -> None:
        """Run all validators in sequence.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            DataValidationError: If any validator fails
        """
        for validator in self.validators:
            validator.validate(df)
    
    def get_required_columns(self) -> List[str]:
        """Get combined list of required columns from all validators."""
        all_required = set()
        for validator in self.validators:
            if hasattr(validator, 'get_required_columns'):
                all_required.update(validator.get_required_columns())
        return sorted(all_required)


def validate_numeric_columns(
    df: pd.DataFrame,
    columns: List[str],
    allow_missing: bool = False
) -> List[str]:
    """Validate and filter to only numeric columns.
    
    Args:
        df: DataFrame to check
        columns: Columns to validate
        allow_missing: If True, skip missing columns; if False, raise error
        
    Returns:
        List of column names that exist and are numeric
        
    Raises:
        DataValidationError: If required columns are missing or non-numeric
    """
    valid_columns = []
    missing_columns = []
    non_numeric_columns = []
    
    for col in columns:
        if col not in df.columns:
            missing_columns.append(col)
        elif not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric_columns.append(col)
        else:
            valid_columns.append(col)
    
    if not allow_missing and missing_columns:
        raise DataValidationError(
            f"Required numeric columns missing: {missing_columns}"
        )
    
    if non_numeric_columns:
        raise DataValidationError(
            f"Columns are not numeric: {non_numeric_columns}"
        )
    
    return valid_columns


def validate_simulation_data(df: pd.DataFrame) -> None:
    """Standard validator for simulation data.
    
    Args:
        df: Simulation DataFrame to validate
        
    Raises:
        DataValidationError: If validation fails
    """
    validator = ColumnValidator(
        required_columns=['simulation_id'],
        column_types={'simulation_id': int}
    )
    validator.validate(df)
    
    quality = DataQualityValidator(min_rows=1)
    quality.validate(df)

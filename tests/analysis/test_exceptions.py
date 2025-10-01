"""
Tests for custom exceptions.
"""

import pytest

from farm.analysis.exceptions import (
    AnalysisError,
    DataValidationError,
    ModuleNotFoundError,
    DataLoaderError,
    DataProcessingError,
    AnalysisFunctionError,
    ConfigurationError,
    InsufficientDataError,
    VisualizationError,
    DatabaseError
)


class TestExceptionHierarchy:
    """Test exception class hierarchy."""
    
    def test_base_exception(self):
        """Test base AnalysisError."""
        exc = AnalysisError("Test error")
        assert str(exc) == "Test error"
        assert isinstance(exc, Exception)
    
    def test_all_inherit_from_base(self):
        """Test all custom exceptions inherit from AnalysisError."""
        exceptions = [
            DataValidationError("test"),
            ModuleNotFoundError("module", []),
            DataLoaderError("test"),
            DataProcessingError("test"),
            ConfigurationError("test"),
            InsufficientDataError("test"),
            VisualizationError("test"),
            DatabaseError("test"),
        ]
        
        for exc in exceptions:
            assert isinstance(exc, AnalysisError)


class TestDataValidationError:
    """Tests for DataValidationError."""
    
    def test_basic_error(self):
        """Test basic validation error."""
        exc = DataValidationError("Validation failed")
        assert "Validation failed" in str(exc)
    
    def test_with_missing_columns(self):
        """Test error with missing columns."""
        exc = DataValidationError(
            "Missing columns",
            missing_columns={'col1', 'col2'}
        )
        
        error_str = str(exc)
        assert "Missing columns" in error_str
        assert "col1" in error_str
        assert "col2" in error_str
    
    def test_with_invalid_columns(self):
        """Test error with invalid column types."""
        exc = DataValidationError(
            "Invalid types",
            invalid_columns={'col1': 'Expected int, got str'}
        )
        
        error_str = str(exc)
        assert "Invalid types" in error_str
        assert "col1" in error_str
        assert "Expected int" in error_str
    
    def test_with_both_missing_and_invalid(self):
        """Test error with both missing and invalid columns."""
        exc = DataValidationError(
            "Multiple issues",
            missing_columns={'col1'},
            invalid_columns={'col2': 'wrong type'}
        )
        
        error_str = str(exc)
        assert "col1" in error_str
        assert "col2" in error_str


class TestModuleNotFoundError:
    """Tests for ModuleNotFoundError."""
    
    def test_error_message(self):
        """Test error message includes module name and available modules."""
        exc = ModuleNotFoundError(
            "missing_module",
            ["module1", "module2", "module3"]
        )
        
        error_str = str(exc)
        assert "missing_module" in error_str
        assert "module1" in error_str
        assert "Available modules" in error_str
    
    def test_attributes(self):
        """Test error attributes."""
        exc = ModuleNotFoundError(
            "missing",
            ["available1", "available2"]
        )
        
        assert exc.module_name == "missing"
        assert exc.available_modules == ["available1", "available2"]


class TestDataProcessingError:
    """Tests for DataProcessingError."""
    
    def test_basic_error(self):
        """Test basic processing error."""
        exc = DataProcessingError("Processing failed")
        assert "Processing failed" in str(exc)
        assert exc.step is None
    
    def test_with_step(self):
        """Test error with processing step."""
        exc = DataProcessingError("Error occurred", step="validation")
        
        error_str = str(exc)
        assert "[validation]" in error_str
        assert "Error occurred" in error_str
        assert exc.step == "validation"


class TestAnalysisFunctionError:
    """Tests for AnalysisFunctionError."""
    
    def test_wraps_original_error(self):
        """Test that error wraps original exception."""
        original = ValueError("Original error")
        exc = AnalysisFunctionError("my_function", original)
        
        error_str = str(exc)
        assert "my_function" in error_str
        assert "Original error" in error_str
        assert exc.function_name == "my_function"
        assert exc.original_error is original


class TestInsufficientDataError:
    """Tests for InsufficientDataError."""
    
    def test_basic_error(self):
        """Test basic insufficient data error."""
        exc = InsufficientDataError("Not enough data")
        assert "Not enough data" in str(exc)
    
    def test_with_row_counts(self):
        """Test error with required and actual row counts."""
        exc = InsufficientDataError(
            "Insufficient rows",
            required_rows=100,
            actual_rows=50
        )
        
        error_str = str(exc)
        assert "100" in error_str
        assert "50" in error_str
        assert exc.required_rows == 100
        assert exc.actual_rows == 50


class TestDatabaseError:
    """Tests for DatabaseError."""
    
    def test_basic_error(self):
        """Test basic database error."""
        exc = DatabaseError("Connection failed")
        assert "Connection failed" in str(exc)
        assert exc.db_path is None
    
    def test_with_db_path(self):
        """Test error with database path."""
        exc = DatabaseError("Query failed", db_path="/path/to/db.sqlite")
        
        error_str = str(exc)
        assert "Query failed" in error_str
        assert "/path/to/db.sqlite" in error_str
        assert exc.db_path == "/path/to/db.sqlite"


class TestExceptionUsage:
    """Test actual usage patterns of exceptions."""
    
    def test_catching_base_exception(self):
        """Test catching base AnalysisError catches all."""
        exceptions_to_test = [
            DataValidationError("test"),
            DataProcessingError("test"),
            InsufficientDataError("test"),
        ]
        
        for exc in exceptions_to_test:
            with pytest.raises(AnalysisError):
                raise exc
    
    def test_catching_specific_exception(self):
        """Test catching specific exception types."""
        with pytest.raises(DataValidationError):
            raise DataValidationError("test")
        
        with pytest.raises(ModuleNotFoundError):
            raise ModuleNotFoundError("test", [])

"""
Tests for analysis protocols.

Verifies that protocol definitions work correctly.
"""

import pytest
import pandas as pd
from pathlib import Path

from farm.analysis.protocols import (
    DataLoader,
    DataProcessor,
    DataValidator,
    AnalysisFunction,
    Analyzer,
    Visualizer,
    AnalysisModule
)
from farm.analysis.common.context import AnalysisContext


def test_data_loader_protocol():
    """Test DataLoader protocol detection."""
    class ValidLoader:
        def iter_data(self, **kwargs):
            yield pd.DataFrame()
        
        def load_data(self, **kwargs):
            return pd.DataFrame()
        
        def get_metadata(self):
            return {}
    
    loader = ValidLoader()
    assert isinstance(loader, DataLoader)


def test_data_processor_protocol():
    """Test DataProcessor protocol detection."""
    class ValidProcessor:
        def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
            return data
    
    processor = ValidProcessor()
    assert isinstance(processor, DataProcessor)


def test_data_validator_protocol():
    """Test DataValidator protocol detection."""
    class ValidValidator:
        def validate(self, data: pd.DataFrame) -> None:
            pass
        
        def get_required_columns(self) -> list:
            return ['col1', 'col2']
    
    validator = ValidValidator()
    assert isinstance(validator, DataValidator)


def test_analysis_function_protocol(sample_simulation_data, analysis_context):
    """Test AnalysisFunction protocol detection."""
    def valid_function(df: pd.DataFrame, ctx: AnalysisContext, **kwargs):
        return {'result': 'success'}
    
    # Functions are callable, so they match the protocol
    assert callable(valid_function)
    result = valid_function(sample_simulation_data, analysis_context)
    assert result['result'] == 'success'


def test_analyzer_protocol():
    """Test Analyzer protocol detection."""
    class ValidAnalyzer:
        def analyze(self, data: pd.DataFrame, **kwargs) -> dict:
            return {'analysis': 'complete'}
        
        def get_metrics(self) -> dict:
            return {'metric1': 1.0}
    
    analyzer = ValidAnalyzer()
    assert isinstance(analyzer, Analyzer)


def test_visualizer_protocol():
    """Test Visualizer protocol detection."""
    class ValidVisualizer:
        def create_charts(self, data: dict, **kwargs) -> dict:
            return {'chart1': 'data'}
        
        def save_charts(self, output_dir: Path, prefix: str = "") -> list:
            return []
    
    visualizer = ValidVisualizer()
    assert isinstance(visualizer, Visualizer)


def test_analysis_module_protocol(minimal_module):
    """Test AnalysisModule protocol detection."""
    # Check protocol compliance
    assert hasattr(minimal_module, 'name')
    assert hasattr(minimal_module, 'description')
    assert hasattr(minimal_module, 'get_data_processor')
    assert hasattr(minimal_module, 'get_validator')
    assert hasattr(minimal_module, 'get_analysis_functions')
    assert hasattr(minimal_module, 'get_function_groups')
    assert hasattr(minimal_module, 'supports_database')
    
    # Test protocol attributes
    assert isinstance(minimal_module.name, str)
    assert isinstance(minimal_module.description, str)
    assert callable(minimal_module.get_data_processor)
    assert callable(minimal_module.get_analysis_functions)


def test_protocol_mismatch():
    """Test that incomplete implementations don't match protocol."""
    class IncompleteLoader:
        def load_data(self, **kwargs):
            return pd.DataFrame()
        # Missing iter_data and get_metadata
    
    loader = IncompleteLoader()
    # Should not match DataLoader protocol
    assert not isinstance(loader, DataLoader)

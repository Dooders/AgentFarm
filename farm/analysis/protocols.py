"""
Core protocols and interfaces for the analysis module system.

This module defines the fundamental contracts that all analysis components must follow.
Uses Python's Protocol for structural typing (duck typing with type safety).
"""

from typing import Protocol, Iterator, Dict, Any, Optional, List, runtime_checkable
import pandas as pd
from pathlib import Path


@runtime_checkable
class DataLoader(Protocol):
    """Protocol for loading data from various sources.
    
    All data loaders must implement both streaming (iter_data) and batch (load_data) methods.
    """
    
    def iter_data(self, **kwargs) -> Iterator[pd.DataFrame]:
        """Stream data in chunks for memory-efficient processing.
        
        Yields:
            DataFrame chunks
        """
        ...
    
    def load_data(self, **kwargs) -> pd.DataFrame:
        """Load all data at once.
        
        Returns:
            Complete DataFrame
        """
        ...
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source.
        
        Returns:
            Metadata dictionary with information about the source
        """
        ...


@runtime_checkable
class DataProcessor(Protocol):
    """Protocol for transforming raw data into analysis-ready format."""
    
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process and transform input data.
        
        Args:
            data: Raw input DataFrame
            **kwargs: Additional processing parameters
            
        Returns:
            Processed DataFrame ready for analysis
        """
        ...


@runtime_checkable
class DataValidator(Protocol):
    """Protocol for validating data meets required schema/constraints."""
    
    def validate(self, data: pd.DataFrame) -> None:
        """Validate data, raising exceptions if invalid.
        
        Args:
            data: DataFrame to validate
            
        Raises:
            DataValidationError: If data doesn't meet requirements
        """
        ...
    
    def get_required_columns(self) -> List[str]:
        """Get list of required column names.
        
        Returns:
            List of required column names
        """
        ...


@runtime_checkable
class AnalysisFunction(Protocol):
    """Protocol for individual analysis functions.
    
    All analysis functions must accept DataFrame and context, return results.
    """
    
    def __call__(
        self,
        df: pd.DataFrame,
        ctx: "AnalysisContext",
        **kwargs: Any
    ) -> Optional[Any]:
        """Execute the analysis function.
        
        Args:
            df: Input DataFrame to analyze
            ctx: Analysis context with configuration and output paths
            **kwargs: Additional function-specific parameters
            
        Returns:
            Analysis results (can be None for side-effect functions like plots)
        """
        ...


@runtime_checkable
class Analyzer(Protocol):
    """Protocol for complete analysis implementations."""
    
    def analyze(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Perform complete analysis on data.
        
        Args:
            data: Input DataFrame to analyze
            **kwargs: Analysis parameters
            
        Returns:
            Dictionary of analysis results
        """
        ...
    
    def get_metrics(self) -> Dict[str, float]:
        """Get computed metrics from the analysis.
        
        Returns:
            Dictionary mapping metric names to values
        """
        ...


@runtime_checkable
class Visualizer(Protocol):
    """Protocol for creating visualizations from analysis results."""
    
    def create_charts(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Create charts from analysis results.
        
        Args:
            data: Analysis results to visualize
            **kwargs: Visualization parameters
            
        Returns:
            Dictionary of chart objects/handles
        """
        ...
    
    def save_charts(self, output_dir: Path, prefix: str = "") -> List[Path]:
        """Save charts to files.
        
        Args:
            output_dir: Directory to save charts
            prefix: Prefix for chart filenames
            
        Returns:
            List of paths to saved chart files
        """
        ...


@runtime_checkable
class AnalysisModule(Protocol):
    """Protocol for complete analysis modules.
    
    A module bundles together all components needed for a specific analysis type.
    """
    
    @property
    def name(self) -> str:
        """Module name (unique identifier)."""
        ...
    
    @property
    def description(self) -> str:
        """Human-readable description of what this module analyzes."""
        ...
    
    def get_data_processor(self) -> DataProcessor:
        """Get the data processor for this module.
        
        Returns:
            DataProcessor instance
        """
        ...
    
    def get_validator(self) -> Optional[DataValidator]:
        """Get the data validator for this module.
        
        Returns:
            DataValidator instance or None if no validation needed
        """
        ...
    
    def get_analysis_functions(self, group: str = "all") -> List[AnalysisFunction]:
        """Get analysis functions by group.
        
        Args:
            group: Function group name (e.g., 'plots', 'metrics', 'all')
            
        Returns:
            List of analysis functions
        """
        ...
    
    def get_function_groups(self) -> List[str]:
        """Get available function group names.
        
        Returns:
            List of group names
        """
        ...
    
    def supports_database(self) -> bool:
        """Whether this module uses database for intermediate storage.
        
        Returns:
            True if module uses database
        """
        ...


# Type aliases for clarity
AnalysisFunctionDict = Dict[str, AnalysisFunction]
AnalysisResults = Dict[str, Any]
ChartDict = Dict[str, Any]

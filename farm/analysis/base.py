"""
Analysis Framework Base Classes and Interfaces

This module provides the base classes and interfaces for the unified analysis framework.
It defines a consistent API for data loading, processing, visualization, and reporting.

The analysis framework is organized around these key concepts:
1. DataLoader - Responsible for loading data from databases and files
2. DataProcessor - Process and transform raw data for analysis
3. Analyzer - Performs specific analysis tasks on processed data
4. Visualizer - Creates visual representations of analysis results
5. Reporter - Generates reports from analysis results

Usage:
    # Example usage of a concrete analyzer:
    loader = SimulationLoader(db_path='path/to/simulation.db')
    analyzer = PopulationAnalyzer(data_loader=loader)
    results = analyzer.analyze()
    visualizer = PopulationVisualizer(data=results)
    visualizer.create_charts()
    reporter = HTMLReporter(data=results, charts=visualizer.charts)
    reporter.generate('population_report.html')
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Iterator, Callable

import pandas as pd


class DataLoader(ABC):
    """Base class for all data loaders.

    Data loaders are responsible for loading data from various sources
    (database, files, etc.) and providing it in a standardized format
    for processors and analyzers.
    """

    @abstractmethod
    def load_data(self, *args, **kwargs) -> pd.DataFrame:
        """Load data from the source.

        Returns:
            pd.DataFrame: Data loaded from the source
        """
        pass

    def iter_data(self, *args, **kwargs) -> Iterator[pd.DataFrame]:
        """Stream data in chunks.

        Default implementation loads all data and yields it as a single chunk.
        Concrete loaders should override for efficient chunked processing.

        Yields:
            Iterator[pd.DataFrame]: Data chunks
        """
        yield self.load_data(*args, **kwargs)

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the loaded data.

        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        pass


class DatabaseLoader(DataLoader):
    """Base class for database-specific data loaders."""

    def __init__(self, db_path: str):
        """Initialize the database loader.

        Args:
            db_path: Path to the database file
        """
        self.db_path = db_path
        self._connection = None

    @abstractmethod
    def connect(self):
        """Establish a connection to the database."""
        pass

    @abstractmethod
    def disconnect(self):
        """Close the database connection."""
        pass

    @abstractmethod
    def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            pd.DataFrame: Query results
        """
        pass


class DataProcessor(ABC):
    """Base class for data processors.

    Data processors transform raw data into forms suitable for analysis.
    This includes cleaning, normalization, feature extraction, etc.
    """

    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the input data.

        Args:
            data: Input data to process

        Returns:
            pd.DataFrame: Processed data
        """
        pass


class BaseAnalyzer(ABC):
    """Base class for all analyzers.

    Analyzers perform specific analysis tasks on data, such as:
    - Statistical analysis
    - Time series analysis
    - Population dynamics analysis
    - Behavioral pattern analysis
    - etc.
    """

    def __init__(
        self,
        data_loader: Optional[DataLoader] = None,
        data: Optional[pd.DataFrame] = None,
    ):
        """Initialize the analyzer.

        Args:
            data_loader: DataLoader instance to load data
            data: Preprocessed data (if data_loader is not provided)
        """
        if data_loader is None and data is None:
            raise ValueError("Either data_loader or data must be provided")

        self.data_loader = data_loader
        self._data = data
        self._results = {}

    @property
    def data(self) -> pd.DataFrame:
        """Get the data for analysis.

        Returns:
            pd.DataFrame: Data for analysis
        """
        if self._data is None and self.data_loader is not None:
            self._data = self.data_loader.load_data()
        return self._data or pd.DataFrame()

    @property
    def results(self) -> Dict[str, Any]:
        """Get the analysis results.

        Returns:
            Dict[str, Any]: Analysis results
        """
        return self._results

    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform the analysis on the provided data.

        Args:
            data: DataFrame to analyze

        Returns:
            Dict[str, Any]: Analysis results
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get metrics from the analysis.

        Returns:
            Dict[str, float]: Metrics dictionary
        """
        pass


class BaseVisualizer(ABC):
    """Base class for all visualizers.

    Visualizers create visual representations (charts, plots, etc.)
    of analysis results.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        """Initialize the visualizer.

        Args:
            data: Data to visualize
        """
        self._data = data or {}
        self._charts = {}

    @property
    def data(self) -> Dict[str, Any]:
        """Get the data for visualization.

        Returns:
            Dict[str, Any]: Data for visualization
        """
        return self._data

    @data.setter
    def data(self, value: Dict[str, Any]):
        """Set the data for visualization.

        Args:
            value: Data for visualization
        """
        self._data = value

    @property
    def charts(self) -> Dict[str, Any]:
        """Get the generated charts.

        Returns:
            Dict[str, Any]: Generated charts
        """
        return self._charts

    @abstractmethod
    def create_charts(self) -> Dict[str, Any]:
        """Create charts from the data.

        Returns:
            Dict[str, Any]: Generated charts
        """
        pass

    @abstractmethod
    def save_charts(self, output_dir: str, prefix: str = ""):
        """Save charts to files.

        Args:
            output_dir: Directory to save charts
            prefix: Prefix for chart filenames
        """
        pass


class BaseReporter(ABC):
    """Base class for all reporters.

    Reporters generate reports from analysis results and visualizations.
    """

    def __init__(
        self,
        data: Optional[Dict[str, Any]] = None,
        charts: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the reporter.

        Args:
            data: Analysis results data
            charts: Visualization charts
        """
        self._data = data or {}
        self._charts = charts or {}

    @abstractmethod
    def generate(self, output_path: str) -> str:
        """Generate a report and save it to the specified path.

        Args:
            output_path: Path to save the report

        Returns:
            str: Path to the generated report
        """
        pass


class AnalysisTask(ABC):
    """Base class for end-to-end analysis tasks.

    An AnalysisTask combines data loading, processing, analysis,
    visualization, and reporting into a single workflow.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        data_processor: Optional[DataProcessor] = None,
        analyzer: Optional[BaseAnalyzer] = None,
        visualizer: Optional[BaseVisualizer] = None,
        reporter: Optional[BaseReporter] = None,
    ):
        """Initialize the analysis task.

        Args:
            data_loader: DataLoader instance
            data_processor: DataProcessor instance
            analyzer: Analyzer instance
            visualizer: Visualizer instance
            reporter: Reporter instance
        """
        self.data_loader = data_loader
        self.data_processor = data_processor
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.reporter = reporter

        self._raw_data = None
        self._processed_data = None
        self._analysis_results = None

    def run(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete analysis task.

        Args:
            output_path: Path to save the report (if reporter is provided)

        Returns:
            Dict[str, Any]: Analysis results
        """
        # Load data
        self._raw_data = self.data_loader.load_data()

        # Process data
        if self.data_processor is not None:
            self._processed_data = self.data_processor.process(self._raw_data)
        else:
            self._processed_data = self._raw_data

        # Run analysis
        if self.analyzer is not None:
            # Run the analysis with explicit data parameter
            self._analysis_results = self.analyzer.analyze(self._processed_data)
        else:
            self._analysis_results = {}

        # Create visualizations
        if self.visualizer is not None and self._analysis_results:
            self.visualizer.data = self._analysis_results
            self.visualizer.create_charts()

        # Generate report
        if self.reporter is not None and output_path is not None:
            if self.visualizer is not None:
                self.reporter._charts = self.visualizer.charts
            self.reporter._data = self._analysis_results
            self.reporter.generate(output_path)

        return self._analysis_results

    def run_streaming(
        self,
        output_path: Optional[str] = None,
        process_chunk: Optional[Callable[[pd.DataFrame], None]] = None,
        loader_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the analysis task in a streaming fashion using data chunks.

        Args:
            output_path: Path to save report (if reporter provided)
            process_chunk: Optional callback called with each processed chunk
            loader_args: Optional arguments forwarded to data_loader.iter_data

        Returns:
            Dict[str, Any]: Final analysis results (may be empty if purely streaming)
        """
        if self.data_loader is None:
            raise ValueError("Streaming requires a data_loader")

        loader_args = loader_args or {}

        # Stream chunks
        aggregated_data = []
        for chunk in self.data_loader.iter_data(**loader_args):
            processed = self.data_processor.process(chunk) if self.data_processor else chunk
            if process_chunk is not None:
                process_chunk(processed)
            else:
                # Fallback accumulation when no callback provided
                aggregated_data.append(processed)

        # If we have aggregated data and an analyzer, run once on the combined data
        if aggregated_data and self.analyzer is not None:
            combined = pd.concat(aggregated_data, ignore_index=True) if len(aggregated_data) > 1 else aggregated_data[0]
            self._analysis_results = self.analyzer.analyze(combined)
        else:
            self._analysis_results = self._analysis_results or {}

        # Create visualizations
        if self.visualizer is not None and self._analysis_results:
            self.visualizer.data = self._analysis_results
            self.visualizer.create_charts()

        # Generate report
        if self.reporter is not None and output_path is not None:
            if self.visualizer is not None:
                self.reporter._charts = self.visualizer.charts
            self.reporter._data = self._analysis_results
            self.reporter.generate(output_path)

        return self._analysis_results


class CLIAnalysisTool(ABC):
    """Base class for command-line analysis tools.

    This provides a common interface for analysis tools that can be run
    from the command line.
    """

    @abstractmethod
    def parse_arguments(self):
        """Parse command-line arguments."""
        pass

    @abstractmethod
    def setup(self):
        """Set up the analysis components."""
        pass

    @abstractmethod
    def run(self):
        """Run the analysis."""
        pass

    @abstractmethod
    def output_results(self):
        """Output the results."""
        pass

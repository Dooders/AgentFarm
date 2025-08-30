"""
Base module for analysis modules.
This provides a common interface for all analysis modules.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from farm.analysis.common.context import AnalysisContext
from farm.analysis.common.metrics import (
    get_valid_numeric_columns as _common_get_valid_numeric_columns,
    split_and_compare_groups as _common_split_and_compare_groups,
    analyze_correlations as _common_analyze_correlations,
    group_and_analyze as _common_group_and_analyze,
    find_top_correlations as _common_find_top_correlations,
)


class AnalysisModule(ABC):
    """
    Base class for analysis modules.

    This class defines the interface that all analysis modules should implement.
    It provides methods for registering analysis functions, retrieving them,
    and getting module information.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize the analysis module.

        Parameters
        ----------
        name : str
            Name of the module
        description : str
            Description of the module
        """
        self.name = name
        self.description = description
        self._analysis_functions = {}
        self._analysis_groups = {}
        self._registered = False

    @abstractmethod
    def register_analysis(self) -> None:
        """
        Register all analysis functions for this module.
        This method should populate self._analysis_functions and self._analysis_groups.
        """
        pass

    @abstractmethod
    def get_data_processor(self) -> Callable:
        """
        Get the data processor function for this module.

        Returns
        -------
        Callable
            The data processor function
        """
        pass

    @abstractmethod
    def get_db_loader(self) -> Optional[Callable]:
        """
        Get the database loader function for this module.

        Returns
        -------
        Optional[Callable]
            The database loader function, or None if not using a database
        """
        pass

    @abstractmethod
    def get_db_filename(self) -> Optional[str]:
        """
        Get the database filename for this module.

        Returns
        -------
        Optional[str]
            The database filename, or None if not using a database
        """
        pass

    def supports_database(self) -> bool:
        """Whether this module uses a database for intermediate storage."""
        return self.get_db_filename() is not None

    def get_analysis_function(self, name: str) -> Optional[Callable]:
        """
        Get a specific analysis function by name.

        Parameters
        ----------
        name : str
            Name of the analysis function

        Returns
        -------
        Optional[Callable]
            The requested analysis function, or None if not found
        """
        if not self._registered:
            self.register_analysis()
            self._registered = True
        return self._analysis_functions.get(name)

    def get_analysis_functions(self, group: str = "all") -> List[Callable]:
        """
        Get a list of analysis functions by group.

        Parameters
        ----------
        group : str
            Name of the function group

        Returns
        -------
        List[Callable]
            List of analysis functions in the requested group
        """
        if not self._registered:
            self.register_analysis()
            self._registered = True
        return self._analysis_groups.get(group, [])

    def get_function_groups(self) -> List[str]:
        """
        Get a list of available function groups.

        Returns
        -------
        List[str]
            List of function group names
        """
        if not self._registered:
            self.register_analysis()
            self._registered = True
        return list(self._analysis_groups.keys())

    def get_function_names(self) -> List[str]:
        """
        Get a list of all available function names.

        Returns
        -------
        List[str]
            List of function names
        """
        if not self._registered:
            self.register_analysis()
            self._registered = True
        return list(self._analysis_functions.keys())

    def get_module_info(self) -> Dict[str, Any]:
        """
        Get information about this analysis module.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing module information
        """
        if not self._registered:
            self.register_analysis()
            self._registered = True

        return {
            "name": self.name,
            "description": self.description,
            "data_processor": self.get_data_processor(),
            "db_loader": self.get_db_loader(),
            "db_filename": self.get_db_filename(),
            "function_groups": self.get_function_groups(),
            "functions": self.get_function_names(),
        }

    def run_analysis(
        self,
        experiment_path: str,
        output_path: str,
        group: str = "all",
        processor_kwargs: Optional[Dict[str, Any]] = None,
        analysis_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Tuple[str, Optional[pd.DataFrame]]:
        """
        Run analysis functions for this module.

        Parameters
        ----------
        experiment_path : str
            Path to the experiment folder
        output_path : str
            Path to save analysis results
        group : str
            Name of the function group to run
        processor_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments for the data processor
        analysis_kwargs : Optional[Dict[str, Dict[str, Any]]]
            Dictionary mapping function names to their keyword arguments

        Returns
        -------
        Tuple[str, Optional[pd.DataFrame]]
            Tuple containing the output path and the processed DataFrame
        """
        import os

        # Initialize default kwargs if not provided
        if processor_kwargs is None:
            processor_kwargs = {}

        if analysis_kwargs is None:
            analysis_kwargs = {}

        # Get the data processor and analysis functions
        data_processor = self.get_data_processor()
        analysis_functions = self.get_analysis_functions(group)

        if not analysis_functions:
            logging.warning(f"No analysis functions found for group '{group}'")
            return output_path, None

        # Process data
        db_filename = self.get_db_filename()
        if db_filename:
            # If using a database, set up the database path
            db_path = os.path.join(output_path, db_filename)
            db_uri = f"sqlite:///{db_path}"
            logging.info(f"Processing data and inserting directly into {db_uri}")

            # Add database parameters to processor kwargs
            db_processor_kwargs = {
                "save_to_db": True,
                "db_path": db_uri,
                **processor_kwargs,
            }

            # Process data and save to database
            df = data_processor(experiment_path, **db_processor_kwargs)

            # Load data from database if processor doesn't return it
            if df is None:
                db_loader = self.get_db_loader()
                if db_loader:
                    logging.info("Loading data from database for visualization...")
                    df = db_loader(db_uri)
        else:
            # Process data without database
            df = data_processor(experiment_path, **processor_kwargs)

        if df is None or df.empty:
            logging.warning("No simulation data found.")
            return output_path, None

        # Log summary statistics
        logging.info(f"Analyzed {len(df)} simulations.")
        logging.info("\nSummary statistics:")
        logging.info(df.describe().to_string())

        # Prepare analysis context for standardized function calls
        ctx = AnalysisContext(output_path=output_path)

        # Run each analysis function
        for func in analysis_functions:
            try:
                # Get function name for logging
                func_name = func.__name__

                # Get kwargs for this function if available
                func_kwargs = analysis_kwargs.get(func_name, {})

                # Check if function expects output_path as an argument
                import inspect

                sig = inspect.signature(func)
                params = list(sig.parameters.keys())

                logging.info(f"Running {func_name}...")

                # Handle different function signatures properly
                if "ctx" in params:
                    # Preferred: function accepts standardized context
                    func(df=df, ctx=ctx, **func_kwargs)
                elif "context" in params:
                    # Alternate naming
                    func(df=df, context=ctx, **func_kwargs)
                elif "output_path" in params:
                    # Legacy: pass output_path directly
                    func(df=df, output_path=output_path, **func_kwargs)
                elif len(params) >= 2:
                    # Function expects positional args (old style)
                    func(df, output_path, **func_kwargs)
                else:
                    # Function only expects df
                    func(df, **func_kwargs)
            except Exception as e:
                logging.error(f"Error running {func.__name__}: {e}")
                import traceback

                logging.error(traceback.format_exc())

        # Log completion
        logging.info("\nAnalysis complete. Results saved.")
        if db_filename:
            logging.info(f"Database saved to: {db_path}")
        logging.info(f"All analysis files saved to: {output_path}")

        return output_path, df


def get_valid_numeric_columns(df: pd.DataFrame, column_list: List[str]) -> List[str]:
    """
    Delegates to the shared helper in `farm.analysis.common.metrics` to avoid duplication.

    See `farm.analysis.common.metrics.get_valid_numeric_columns` for details.
    """
    return _common_get_valid_numeric_columns(df, column_list)


def split_and_compare_groups(
    df: pd.DataFrame,
    split_column: str,
    split_value: Optional[float] = None,
    metrics: Optional[List[str]] = None,
    split_method: str = "median",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Split a DataFrame into high and low groups based on a column value and compare metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with analysis results
    split_column : str
        Column to use for splitting the DataFrame
    split_value : float, optional
        Value to use for splitting. If None, uses median or other split_method
    metrics : list, optional
        List of metric columns to compare. If None, uses all numeric columns
    split_method : str, optional
        Method to use for splitting if split_value is None ('median', 'mean', etc.)

    Returns
    -------
    dict
        Dictionary with comparison results for each metric
    """
    if df.empty or split_column not in df.columns:
        logging.warning(f"Cannot perform group comparison: missing {split_column}")
        return {}

    # Determine the split value if not provided
    if split_value is None:
        if split_method == "median":
            split_value = df[split_column].median()
        elif split_method == "mean":
            split_value = df[split_column].mean()
        else:
            split_value = df[split_column].median()  # Default to median

    # Split into high and low groups
    high_group = df[df[split_column] > split_value]
    low_group = df[df[split_column] <= split_value]

    logging.info(
        f"Analyzing {len(high_group)} high-{split_column} and {len(low_group)} low-{split_column} groups"
    )

    # Determine which metrics to compare
    if metrics is None:
        metrics = get_valid_numeric_columns(df, df.columns.tolist())
        # Remove the split column from metrics
        if split_column in metrics:
            metrics.remove(split_column)

    # Compare metrics between groups
    comparison = {}
    for metric in metrics:
        try:
            high_mean = high_group[metric].mean()
            low_mean = low_group[metric].mean()
            difference = high_mean - low_mean
            percent_diff = (
                (difference / low_mean * 100) if low_mean != 0 else float("inf")
            )

            comparison[metric] = {
                "high_group_mean": high_mean,
                "low_group_mean": low_mean,
                "difference": difference,
                "percent_difference": percent_diff,
            }
        except Exception as e:
            logging.warning(f"Error comparing metric {metric}: {e}")

    return {"comparison_results": comparison}


def analyze_correlations(
    df: pd.DataFrame,
    target_column: str,
    metric_columns: Optional[List[str]] = None,
    min_data_points: int = 5,
    filter_condition: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> Dict[str, float]:
    """Delegate to shared implementation in common.metrics."""
    return _common_analyze_correlations(
        df,
        target_column,
        metric_columns=metric_columns,
        min_data_points=min_data_points,
        filter_condition=filter_condition,
    )


def group_and_analyze(
    df: pd.DataFrame,
    group_column: str,
    group_values: List[str],
    analysis_func: Callable[[pd.DataFrame], Dict[str, Any]],
    min_group_size: int = 5,
) -> Dict[str, Dict[str, Any]]:
    return _common_group_and_analyze(
        df,
        group_column,
        group_values,
        analysis_func,
        min_group_size=min_group_size,
    )


def find_top_correlations(
    df: pd.DataFrame,
    target_column: str,
    metric_columns: Optional[List[str]] = None,
    top_n: int = 5,
    min_correlation: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    return _common_find_top_correlations(
        df,
        target_column,
        metric_columns=metric_columns,
        top_n=top_n,
        min_correlation=min_correlation,
    )


class BaseAnalysisModule:
    """
    Base class for analysis modules with common functionality.

    This class provides common methods for data analysis that can be reused
    across different analysis modules.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        Initialize the analysis module.

        Parameters
        ----------
        df : pandas.DataFrame, optional
            DataFrame with analysis results
        """
        self.df = df

    def set_data(self, df: pd.DataFrame) -> None:
        """
        Set the DataFrame for analysis.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with analysis results
        """
        self.df = df

    def get_valid_columns(self, pattern: str = "") -> List[str]:
        """
        Get valid numeric columns that match a pattern.

        Parameters
        ----------
        pattern : str, optional
            String pattern to match in column names

        Returns
        -------
        list
            List of column names that match the pattern and are numeric
        """
        if self.df is None:
            return []

        return get_valid_numeric_columns(
            self.df, [col for col in self.df.columns if pattern in col]
        )

    def split_and_compare(
        self,
        split_column: str,
        metrics: Optional[List[str]] = None,
        split_method: str = "median",
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Split data and compare metrics between groups.

        Parameters
        ----------
        split_column : str
            Column to use for splitting the DataFrame
        metrics : list, optional
            List of metric columns to compare
        split_method : str, optional
            Method to use for splitting ('median', 'mean', etc.)

        Returns
        -------
        dict
            Dictionary with comparison results
        """
        if self.df is None:
            return {}

        return _common_split_and_compare_groups(
            self.df, split_column, metrics=metrics, split_method=split_method
        )

    def analyze_correlations(
        self, target_column: str, metric_columns: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Analyze correlations between target and metrics.

        Parameters
        ----------
        target_column : str
            Target column to correlate with
        metric_columns : list, optional
            List of metrics to correlate

        Returns
        -------
        dict
            Dictionary mapping metrics to correlation coefficients
        """
        if self.df is None:
            return {}

        return analyze_correlations(self.df, target_column, metric_columns)

    def group_and_analyze(
        self,
        group_column: str,
        group_values: List[str],
        analysis_func: Callable[[pd.DataFrame], Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group data and perform analysis on each group.

        Parameters
        ----------
        group_column : str
            Column to group by
        group_values : list
            Values to group by
        analysis_func : callable
            Function to apply to each group

        Returns
        -------
        dict
            Dictionary mapping group values to analysis results
        """
        if self.df is None:
            return {}

        return group_and_analyze(self.df, group_column, group_values, analysis_func)

    def find_top_correlations(
        self,
        target_column: str,
        metric_columns: Optional[List[str]] = None,
        top_n: int = 5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Find top correlations with target column.

        Parameters
        ----------
        target_column : str
            Target column to correlate with
        metric_columns : list, optional
            List of metrics to correlate
        top_n : int, optional
            Number of top correlations to return

        Returns
        -------
        dict
            Dictionary with top positive and negative correlations
        """
        if self.df is None:
            return {"top_positive": {}, "top_negative": {}}

        return find_top_correlations(
            self.df, target_column, metric_columns, top_n=top_n
        )

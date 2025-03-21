"""
Base module for analysis modules.
This provides a common interface for all analysis modules.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


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
                if 'output_path' in params:
                    # Pass output_path as a named parameter
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
    Filter a list of column names to only include those that are numeric in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to check for numeric columns
    column_list : list
        List of column names to filter

    Returns
    -------
    list
        List of column names that exist in the DataFrame and are numeric
    """
    numeric_cols = []
    for col in column_list:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


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
    """
    Analyze correlations between a target column and multiple metric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with analysis results
    target_column : str
        Target column to correlate with metrics
    metric_columns : list, optional
        List of metric columns to correlate. If None, uses all numeric columns
    min_data_points : int, optional
        Minimum number of valid data points required for correlation analysis
    filter_condition : callable, optional
        Function to filter the DataFrame before analysis, accepts DataFrame and returns filtered DataFrame

    Returns
    -------
    dict
        Dictionary mapping metric column names to correlation coefficients
    """
    if df.empty or target_column not in df.columns:
        logging.warning(f"Cannot perform correlation analysis: missing {target_column}")
        return {}

    # Determine which metrics to correlate
    if metric_columns is None:
        metric_columns = get_valid_numeric_columns(df, df.columns.tolist())
        # Remove the target column from metrics
        if target_column in metric_columns:
            metric_columns.remove(target_column)

    # Apply filter condition if provided
    if filter_condition is not None:
        filtered_df = filter_condition(df)
    else:
        filtered_df = df

    # Calculate correlations
    correlations = {}
    for col in metric_columns:
        try:
            # Filter out rows with NaN values in either column
            valid_data = filtered_df[
                filtered_df[col].notna() & filtered_df[target_column].notna()
            ]

            if len(valid_data) >= min_data_points:
                corr = valid_data[[col, target_column]].corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr
            else:
                logging.debug(
                    f"Not enough valid data points for {col} correlation analysis"
                )
        except Exception as e:
            logging.warning(f"Error calculating correlation for {col}: {e}")

    return correlations


def group_and_analyze(
    df: pd.DataFrame,
    group_column: str,
    group_values: List[str],
    analysis_func: Callable[[pd.DataFrame], Dict[str, Any]],
    min_group_size: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """
    Group data by a categorical variable and perform analysis within each group.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with analysis results
    group_column : str
        Column to group by
    group_values : list
        List of values in the group_column to analyze
    analysis_func : callable
        Function to apply to each group, accepts DataFrame and returns dict of results
    min_group_size : int, optional
        Minimum group size required for analysis

    Returns
    -------
    dict
        Dictionary mapping group values to analysis results
    """
    if df.empty or group_column not in df.columns:
        logging.warning(f"Cannot perform group analysis: missing {group_column}")
        return {}

    results = {}
    for group_value in group_values:
        group_data = df[df[group_column] == group_value]
        if len(group_data) >= min_group_size:
            results[group_value] = analysis_func(group_data)
        else:
            logging.info(
                f"Not enough data points for {group_value} group analysis (n={len(group_data)})"
            )

    return results


def find_top_correlations(
    df: pd.DataFrame,
    target_column: str,
    metric_columns: Optional[List[str]] = None,
    top_n: int = 5,
    min_correlation: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    """
    Find the top positive and negative correlations between metrics and a target column.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with analysis results
    target_column : str
        Target column to correlate with metrics
    metric_columns : list, optional
        List of metric columns to correlate. If None, uses all numeric columns
    top_n : int, optional
        Number of top correlations to return
    min_correlation : float, optional
        Minimum absolute correlation value to include

    Returns
    -------
    dict
        Dictionary with top positive and negative correlations
    """
    correlations = analyze_correlations(df, target_column, metric_columns)

    if not correlations:
        return {"top_positive": {}, "top_negative": {}}

    # Sort correlations by absolute value
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    # Filter by minimum correlation and take top N
    positive_corrs = {k: v for k, v in sorted_corrs if v >= min_correlation}
    negative_corrs = {k: v for k, v in sorted_corrs if v <= -min_correlation}

    top_positive = dict(list(positive_corrs.items())[:top_n])
    top_negative = dict(list(negative_corrs.items())[:top_n])

    return {"top_positive": top_positive, "top_negative": top_negative}


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

        return split_and_compare_groups(
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

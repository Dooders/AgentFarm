"""
Common metrics and utility functions for analysis modules.

This module centralizes helpers that are shared across analysis packages
to avoid duplication and improve cohesion.
"""

from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from farm.utils.logging import get_logger

logger = get_logger(__name__)


def get_valid_numeric_columns(df: pd.DataFrame, column_list: List[str]) -> List[str]:
    """
    Filter a list of column names to only include those that are numeric
    in the provided DataFrame.

    Args:
        df: DataFrame to check for numeric columns
        column_list: List of column names to filter

    Returns:
        List[str]: List of column names that exist in the DataFrame and are numeric
    """
    numeric_columns: List[str] = []
    for column_name in column_list:
        if column_name in df.columns and pd.api.types.is_numeric_dtype(df[column_name]):
            numeric_columns.append(column_name)
    return numeric_columns


def split_and_compare_groups(
    df: pd.DataFrame,
    split_column: str,
    split_value: Optional[float] = None,
    metrics: Optional[List[str]] = None,
    split_method: str = "median",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Split a DataFrame into high and low groups based on a column value and compare metrics.

    Returns a dictionary with comparison results for each metric.
    """
    if df.empty or split_column not in df.columns:
        logger.warning("cannot_perform_group_comparison", missing_column=split_column)
        return {}

    if split_value is None:
        if split_method == "median":
            split_value = df[split_column].median()
        elif split_method == "mean":
            split_value = df[split_column].mean()
        else:
            split_value = df[split_column].median()

    high_group = df[df[split_column] > split_value]
    low_group = df[df[split_column] <= split_value]

    logger.info(
        f"Analyzing {len(high_group)} high-{split_column} and {len(low_group)} low-{split_column} groups"
    )

    if metrics is None:
        metrics = get_valid_numeric_columns(df, df.columns.tolist())
        if split_column in metrics:
            metrics.remove(split_column)

    comparison: Dict[str, Dict[str, float]] = {}
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
        except Exception as exc:
            logger.warning(f"Error comparing metric {metric}: {exc}")

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
    """
    if df.empty or target_column not in df.columns:
        logger.warning(
            "cannot_perform_correlation_analysis", missing_column=target_column
        )
        return {}

    if metric_columns is None:
        metric_columns = get_valid_numeric_columns(df, df.columns.tolist())
        if target_column in metric_columns:
            metric_columns.remove(target_column)

    filtered_df = filter_condition(df) if filter_condition is not None else df

    # Adjust threshold in presence of filter condition to be practical but safe
    # Allows correlations on smaller filtered subsets while keeping defaults intact
    threshold = min_data_points
    if filter_condition is not None:
        # Require at least 2 points to compute correlation
        threshold = max(2, min(min_data_points, len(filtered_df)))

    correlations: Dict[str, float] = {}
    for col in metric_columns:
        try:
            valid_data = filtered_df[
                filtered_df[col].notna() & filtered_df[target_column].notna()
            ]
            if len(valid_data) >= threshold:
                corr = valid_data[[col, target_column]].corr().iloc[0, 1]
                if not pd.isna(corr):
                    correlations[col] = pd.to_numeric(corr, errors="coerce")
        except Exception as exc:
            logger.warning(f"Error calculating correlation for {col}: {exc}")

    return correlations


def group_and_analyze(
    df: pd.DataFrame,
    group_column: str,
    group_values: List[str],
    analysis_func: Callable[[pd.DataFrame], Dict[str, Any]],
    min_group_size: int = 1,
) -> Dict[str, Dict[str, Any]]:
    if df.empty or group_column not in df.columns:
        return {}
    results: Dict[str, Dict[str, Any]] = {}
    for group_value in group_values:
        group_data = df[df[group_column] == group_value]
        if len(group_data) >= min_group_size:
            results[group_value] = analysis_func(group_data)
    return results


def find_top_correlations(
    df: pd.DataFrame,
    target_column: str,
    metric_columns: Optional[List[str]] = None,
    top_n: int = 5,
    min_correlation: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    correlations = analyze_correlations(df, target_column, metric_columns)
    if not correlations:
        return {"top_positive": {}, "top_negative": {}}
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    positive_corrs = {k: v for k, v in sorted_corrs if v >= min_correlation}
    negative_corrs = {k: v for k, v in sorted_corrs if v <= -min_correlation}
    top_positive = dict(list(positive_corrs.items())[:top_n])
    top_negative = dict(list(negative_corrs.items())[:top_n])
    return {"top_positive": top_positive, "top_negative": top_negative}

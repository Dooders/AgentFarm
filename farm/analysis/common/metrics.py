"""
Common metrics and utility functions for analysis modules.

This module centralizes helpers that are shared across analysis packages
to avoid duplication and improve cohesion.
"""

from typing import List

import pandas as pd


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


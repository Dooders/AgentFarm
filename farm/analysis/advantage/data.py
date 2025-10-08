"""
Data processing for advantage analysis module.
"""

import pandas as pd
from typing import Any, Dict, Optional
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def process_advantage_data(data: Any, **kwargs) -> pd.DataFrame:
    """
    Process advantage analysis data.

    This module primarily works with database sessions, but can also
    process DataFrame inputs for cross-simulation analysis.

    Parameters
    ----------
    data : Any
        Input data - can be a database session or DataFrame
    **kwargs
        Additional processing parameters

    Returns
    -------
    pd.DataFrame
        Processed data ready for analysis
    """
    if isinstance(data, pd.DataFrame):
        logger.info("Processing DataFrame input for advantage analysis")
        # DataFrame input - likely from cross-simulation analysis
        return data
    else:
        # Database session or other input - return empty DataFrame
        # The actual data loading happens in the analysis functions
        logger.info("Advantage analysis uses database sessions - no DataFrame processing needed")
        return pd.DataFrame()

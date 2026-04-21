"""
Data processing for the genetics analysis module.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from farm.utils.logging import get_logger

logger = get_logger(__name__)


def process_genetics_data(data: Any, **kwargs) -> pd.DataFrame:
    """Process input data for the genetics analysis module.

    Accepts either a :class:`pandas.DataFrame` (passed through unchanged) or
    an SQLAlchemy session (forwarded to the DB accessor).

    Parameters
    ----------
    data:
        A :class:`pandas.DataFrame`, an SQLAlchemy session, or an
        ``EvolutionExperimentResult``.
    **kwargs:
        Reserved for future use.

    Returns
    -------
    pd.DataFrame
        Processed data ready for analysis.
    """
    if isinstance(data, pd.DataFrame):
        logger.info("process_genetics_data: passing DataFrame through unchanged")
        return data

    # SQLAlchemy session duck-type check
    if hasattr(data, "query"):
        logger.info("process_genetics_data: loading agent genetics from database session")
        from farm.analysis.genetics.compute import build_agent_genetics_dataframe

        return build_agent_genetics_dataframe(data)

    # EvolutionExperimentResult duck-type check
    if hasattr(data, "evaluations") and hasattr(data, "generation_summaries"):
        logger.info("process_genetics_data: loading genetics from EvolutionExperimentResult")
        from farm.analysis.genetics.compute import build_evolution_experiment_dataframe

        return build_evolution_experiment_dataframe(data)

    raise TypeError(
        f"process_genetics_data: unsupported data type {type(data).__name__!r}. "
        "Expected a DataFrame, SQLAlchemy session, or EvolutionExperimentResult."
    )

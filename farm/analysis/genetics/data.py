"""
Data processing for the genetics analysis module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from farm.analysis.common.utils import find_database_path
from farm.analysis.genetics.compute import (
    build_agent_genetics_dataframe,
    build_evolution_experiment_dataframe,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def process_genetics_data(data: Any, use_database: bool = True, **kwargs) -> pd.DataFrame:
    """Process input data for the genetics analysis module.

    Supports:

    - ``Path`` / ``str`` experiment directories (standard module workflow)
    - :class:`pandas.DataFrame` passthrough
    - SQLAlchemy session objects
    - ``EvolutionExperimentResult``-like objects

    Parameters
    ----------
    data:
        Input data source.
    use_database:
        Whether path-based inputs should load from ``simulation.db``.
    **kwargs:
        Reserved for future use.

    Returns
    -------
    pd.DataFrame
        Processed data ready for analysis.
    """
    if isinstance(data, (Path, str)):
        if not use_database:
            raise ValueError("process_genetics_data requires use_database=True for path inputs")

        experiment_path = Path(data)
        db_path = find_database_path(experiment_path, "simulation.db")
        logger.info("process_genetics_data: loading agent genetics from database %s", db_path)

        engine = create_engine(f"sqlite:///{db_path}")
        with Session(engine) as session:
            df = build_agent_genetics_dataframe(session)
        engine.dispose()
        return df

    if isinstance(data, pd.DataFrame):
        logger.info("process_genetics_data: passing DataFrame through unchanged")
        return data

    # SQLAlchemy session duck-type check
    if hasattr(data, "query"):
        logger.info("process_genetics_data: loading agent genetics from provided session")
        return build_agent_genetics_dataframe(data)

    # EvolutionExperimentResult duck-type check
    if hasattr(data, "evaluations") and hasattr(data, "generation_summaries"):
        logger.info("process_genetics_data: loading genetics from EvolutionExperimentResult")
        return build_evolution_experiment_dataframe(data)

    raise TypeError(
        f"process_genetics_data: unsupported data type {type(data).__name__!r}. "
        "Expected experiment path, DataFrame, SQLAlchemy session, or EvolutionExperimentResult."
    )

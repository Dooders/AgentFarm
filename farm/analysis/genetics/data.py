"""
Data processing for the genetics analysis module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from farm.utils.logging import get_logger

logger = get_logger(__name__)


def process_genetics_data(data: Any, **kwargs) -> pd.DataFrame:
    """Process input data for the genetics analysis module.

    Accepts a :class:`~pathlib.Path` or ``str`` (experiment directory), a
    :class:`pandas.DataFrame` (passed through unchanged), an SQLAlchemy
    session, or an ``EvolutionExperimentResult``.

    When a ``Path`` or ``str`` is provided the function locates
    ``simulation.db`` inside the experiment directory (using
    :func:`~farm.analysis.common.utils.find_database_path`), opens a
    transient session via :class:`~farm.database.session_manager.SessionManager`,
    and delegates to :func:`~farm.analysis.genetics.compute.build_agent_genetics_dataframe`.

    Parameters
    ----------
    data:
        A :class:`~pathlib.Path` or ``str`` experiment directory, a
        :class:`pandas.DataFrame`, an SQLAlchemy session, or an
        ``EvolutionExperimentResult``.
    **kwargs:
        Reserved for future use.

    Returns
    -------
    pd.DataFrame
        Processed data ready for analysis.
    """
    # Path / str  ->  locate simulation.db and build the genetics frame
    if isinstance(data, (str, Path)):
        experiment_path = Path(data)
        logger.info("process_genetics_data: loading genetics from experiment path %s", experiment_path)
        # Deferred imports to avoid circular dependencies at module import time and
        # to keep heavy database dependencies from loading unless the Path branch is used.
        from farm.analysis.common.utils import find_database_path
        from farm.database.session_manager import SessionManager
        from farm.analysis.genetics.compute import build_agent_genetics_dataframe

        db_path = find_database_path(experiment_path)
        session_manager = SessionManager(f"sqlite:///{db_path}")
        with session_manager.session_scope() as session:
            return build_agent_genetics_dataframe(session)

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
        "Expected a Path/str experiment directory, DataFrame, SQLAlchemy session, "
        "or EvolutionExperimentResult."
    )

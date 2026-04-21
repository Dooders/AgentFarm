"""
Genetics Analysis Computation

Core computation functions for the genetics analysis module, including the
shared ``parse_parent_ids`` helper and population-level accessors for both
simulation-database and evolution-experiment data sources.
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import pandas as pd

from farm.analysis.genetics.utils import parse_parent_ids
from farm.utils.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from farm.runners.evolution_experiment import EvolutionExperimentResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# DB-backed population accessor
# ---------------------------------------------------------------------------

#: Columns produced by :func:`build_agent_genetics_dataframe`.
AGENT_GENETICS_COLUMNS = [
    "agent_id",
    "agent_type",
    "generation",
    "birth_time",
    "death_time",
    "genome_id",
    "parent_ids",
    "action_weights",
]


def build_agent_genetics_dataframe(session: "Session") -> pd.DataFrame:
    """Build a normalized genetics DataFrame from a simulation database session.

    Each row represents one agent.  The ``parent_ids`` column contains a
    Python list of parent agent-ID strings (empty list for genesis agents).
    Action weights are stored as a dict in the ``action_weights`` column.

    Parameters
    ----------
    session:
        An active SQLAlchemy session connected to a simulation database.

    Returns
    -------
    pd.DataFrame
        One row per agent with columns defined in :data:`AGENT_GENETICS_COLUMNS`.
        Returns an empty DataFrame (with correct columns) when no agents are
        found.
    """
    from farm.database.models import AgentModel

    agents: List[AgentModel] = session.query(AgentModel).all()

    if not agents:
        logger.info("build_agent_genetics_dataframe: no agents found in session")
        return pd.DataFrame(columns=AGENT_GENETICS_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for agent in agents:
        rows.append(
            {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "generation": agent.generation,
                "birth_time": agent.birth_time,
                "death_time": agent.death_time,
                "genome_id": agent.genome_id,
                "parent_ids": parse_parent_ids(agent.genome_id),
                "action_weights": agent.action_weights or {},
            }
        )

    df = pd.DataFrame(rows, columns=AGENT_GENETICS_COLUMNS)
    logger.info("build_agent_genetics_dataframe: built frame with %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Evolution-experiment-backed population accessor
# ---------------------------------------------------------------------------

#: Columns produced by :func:`build_evolution_experiment_dataframe`.
EVOLUTION_GENETICS_COLUMNS = [
    "candidate_id",
    "generation",
    "fitness",
    "parent_ids",
    "chromosome_values",
]


def build_evolution_experiment_dataframe(result: "EvolutionExperimentResult") -> pd.DataFrame:
    """Build a normalized genetics DataFrame from an
    :class:`~farm.runners.evolution_experiment.EvolutionExperimentResult`.

    Each row represents one evaluated candidate across all generations.

    Parameters
    ----------
    result:
        A completed evolution-experiment result object.

    Returns
    -------
    pd.DataFrame
        One row per evaluated candidate with columns defined in
        :data:`EVOLUTION_GENETICS_COLUMNS`.  Returns an empty DataFrame when
        *result* contains no evaluations.
    """
    evaluations = result.evaluations
    if not evaluations:
        logger.info("build_evolution_experiment_dataframe: no evaluations in result")
        return pd.DataFrame(columns=EVOLUTION_GENETICS_COLUMNS)

    rows: List[Dict[str, Any]] = []
    for ev in evaluations:
        rows.append(
            {
                "candidate_id": ev.candidate_id,
                "generation": ev.generation,
                "fitness": ev.fitness,
                "parent_ids": list(ev.parent_ids),
                "chromosome_values": dict(ev.chromosome_values),
            }
        )

    df = pd.DataFrame(rows, columns=EVOLUTION_GENETICS_COLUMNS)
    logger.info(
        "build_evolution_experiment_dataframe: built frame with %d rows across %d generation(s)",
        len(df),
        df["generation"].nunique(),
    )
    return df

"""
Shared SQL-backed loaders for legacy summaries and analysis module data processors.

Centralizes queries so :class:`farm.core.analysis.SimulationAnalyzer` and the
population/resources combat-related pipelines read the same definitions.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import pandas as pd
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from farm.database.data_types import Population
from farm.database.models import ActionModel, AgentModel, AgentStateModel, SimulationStepModel
from farm.database.repositories.population_repository import PopulationRepository
from farm.database.repositories.resource_repository import ResourceRepository
from farm.database.session_manager import SessionManager


def _normalize_db_url(db_path: str) -> str:
    """Return a SQLAlchemy SQLite URL for *db_path*.

    If *db_path* is already a full URL (contains ``://``), it is returned unchanged.
    Otherwise ``sqlite:///`` is prepended so the result is a valid SQLite URL
    that :class:`~farm.database.session_manager.SessionManager` will use directly.
    """
    if "://" in db_path:
        return db_path
    return f"sqlite:///{db_path}"


def survival_rates_from_session(session: Session, simulation_id: Optional[str] = None) -> pd.DataFrame:
    """Per-step counts of alive system vs independent agents (legacy analyzer schema)."""
    query = (
        session.query(
            SimulationStepModel.step_number,
            func.count(case((AgentModel.agent_type == "system", 1), else_=None)).label("system_alive"),
            func.count(case((AgentModel.agent_type == "independent", 1), else_=None)).label("independent_alive"),
        )
        .join(AgentStateModel, SimulationStepModel.step_number == AgentStateModel.step_number)
        .join(AgentModel, AgentStateModel.agent_id == AgentModel.agent_id)
    )
    if simulation_id is not None:
        query = query.filter(
            SimulationStepModel.simulation_id == simulation_id,
            AgentStateModel.simulation_id == simulation_id,
            AgentModel.simulation_id == simulation_id,
        )
    query = query.group_by(SimulationStepModel.step_number).order_by(SimulationStepModel.step_number)
    results = query.all()
    return pd.DataFrame(results, columns=["step", "system_alive", "independent_alive"])


def resource_distribution_from_session(session: Session, simulation_id: Optional[str] = None) -> pd.DataFrame:
    """Per-step per-agent-type resource stats (legacy analyzer schema)."""
    query = (
        session.query(
            SimulationStepModel.step_number,
            AgentModel.agent_type,
            func.avg(AgentStateModel.resource_level).label("avg_resources"),
            func.min(AgentStateModel.resource_level).label("min_resources"),
            func.max(AgentStateModel.resource_level).label("max_resources"),
            func.count().label("agent_count"),
        )
        .join(AgentStateModel, SimulationStepModel.step_number == AgentStateModel.step_number)
        .join(AgentModel, AgentStateModel.agent_id == AgentModel.agent_id)
    )
    if simulation_id is not None:
        query = query.filter(
            SimulationStepModel.simulation_id == simulation_id,
            AgentStateModel.simulation_id == simulation_id,
            AgentModel.simulation_id == simulation_id,
        )
    query = query.group_by(SimulationStepModel.step_number, AgentModel.agent_type).order_by(
        SimulationStepModel.step_number
    )
    results = query.all()
    return pd.DataFrame(
        results,
        columns=["step", "agent_type", "avg_resources", "min_resources", "max_resources", "agent_count"],
    )


def competitive_interactions_from_session(session: Session, simulation_id: Optional[str] = None) -> pd.DataFrame:
    """Attack counts per step (legacy analyzer schema)."""
    query = (
        session.query(
            ActionModel.step_number,
            func.count(ActionModel.action_id).label("competitive_interactions"),
        )
        .filter(ActionModel.action_type == "attack")
    )
    if simulation_id is not None:
        query = query.filter(ActionModel.simulation_id == simulation_id)
    query = query.group_by(ActionModel.step_number).order_by(ActionModel.step_number)
    results = query.all()
    return pd.DataFrame(results, columns=["step", "competitive_interactions"])


def resource_efficiency_from_session(session: Session, simulation_id: Optional[str] = None) -> pd.DataFrame:
    """Per-step resource_efficiency from simulation_steps (legacy analyzer schema)."""
    query = session.query(
        SimulationStepModel.step_number,
        SimulationStepModel.resource_efficiency.label("efficiency"),
    )
    if simulation_id is not None:
        query = query.filter(SimulationStepModel.simulation_id == simulation_id)
    query = query.order_by(SimulationStepModel.step_number)
    results = query.all()
    return pd.DataFrame(results, columns=["step", "efficiency"])


def run_dataframe_on_sqlite(db_path: str, query_fn: Callable[[Session], pd.DataFrame]) -> pd.DataFrame:
    """Run a session callback that returns a DataFrame against a SQLite file or URI path."""
    session_manager = SessionManager(_normalize_db_url(db_path))

    def _run(session: Session) -> pd.DataFrame:
        return query_fn(session)

    try:
        return session_manager.execute_with_retry(_run)
    finally:
        session_manager.cleanup()


def population_rows_from_sqlite(db_path: str) -> List[Population]:
    """Population time series used by the population analysis module."""
    session_manager = SessionManager(_normalize_db_url(db_path))
    try:
        repository = PopulationRepository(session_manager)
        return repository.get_population_over_time()
    finally:
        session_manager.cleanup()


def population_dataframe_from_sqlite(db_path: str) -> pd.DataFrame:
    """Build the population module DataFrame from a SQLite simulation database path."""
    population_data = population_rows_from_sqlite(db_path)
    return pd.DataFrame(
        {
            "step": [p.step_number for p in population_data],
            "total_agents": [p.total_agents for p in population_data],
            "system_agents": [p.system_agents if p.system_agents is not None else 0 for p in population_data],
            "independent_agents": [
                p.independent_agents if p.independent_agents is not None else 0 for p in population_data
            ],
            "control_agents": [p.control_agents if p.control_agents is not None else 0 for p in population_data],
            "avg_resources": [p.avg_resources if p.avg_resources is not None else 0.0 for p in population_data],
        }
    )


def resources_merged_with_positions_from_sqlite(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load merged resource metrics and per-cell resource positions in one DB session."""
    session_manager = SessionManager(_normalize_db_url(db_path))
    try:
        repository = ResourceRepository(session_manager)
        distribution = repository.resource_distribution()
        consumption = repository.consumption_patterns()
        efficiency = repository.efficiency_metrics()

        distribution_data = []
        for dist in distribution:
            distribution_data.append(
                {
                    "step": dist.step,
                    "total_resources": dist.total_resources,
                    "average_per_cell": dist.average_per_cell,
                    "distribution_entropy": dist.distribution_entropy,
                }
            )

        df = pd.DataFrame(distribution_data)
        df["total_consumed"] = consumption.total_consumed
        df["avg_consumption_rate"] = consumption.avg_consumption_rate
        df["peak_consumption"] = consumption.peak_consumption
        df["consumption_variance"] = consumption.consumption_variance
        df["utilization_rate"] = efficiency.utilization_rate
        df["distribution_efficiency"] = efficiency.distribution_efficiency
        df["consumption_efficiency"] = efficiency.consumption_efficiency
        df["regeneration_rate"] = efficiency.regeneration_rate

        raw_positions = repository.get_resource_positions_over_time()
        positions_df = pd.DataFrame(raw_positions) if raw_positions else pd.DataFrame()
        return df, positions_df
    finally:
        session_manager.cleanup()

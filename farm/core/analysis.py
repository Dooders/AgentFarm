import math
import os
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import case, func

from farm.database.database import SimulationDatabase
from farm.database.models import ActionModel, AgentModel, AgentStateModel, SimulationStepModel


class SimulationAnalyzer:
    def __init__(self, db_path: str = "simulation.db", simulation_id: str = None):
        self.db = SimulationDatabase(db_path, simulation_id=simulation_id)

    def calculate_survival_rates(self) -> pd.DataFrame:
        """Calculate survival rates for different agent types over time."""

        def _query(session):
            query = (
                session.query(
                    SimulationStepModel.step_number,
                    func.count(
                        case((AgentModel.agent_type == "system", 1), else_=None)
                    ).label("system_alive"),
                    func.count(
                        case(
                            (AgentModel.agent_type == "independent", 1), else_=None
                        )
                    ).label("independent_alive"),
                )
                .join(
                    AgentStateModel,
                    SimulationStepModel.step_number == AgentStateModel.step_number,
                )
                .join(AgentModel, AgentStateModel.agent_id == AgentModel.agent_id)
                .group_by(SimulationStepModel.step_number)
                .order_by(SimulationStepModel.step_number)
            )

            results = query.all()
            return pd.DataFrame(
                results, columns=["step", "system_alive", "independent_alive"]
            )

        return self.db._execute_in_transaction(_query)

    def analyze_resource_distribution(self) -> pd.DataFrame:
        """Analyze resource accumulation and distribution patterns."""

        def _query(session):
            query = (
                session.query(
                    SimulationStepModel.step_number,
                    AgentModel.agent_type,
                    func.avg(AgentStateModel.resource_level).label("avg_resources"),
                    func.min(AgentStateModel.resource_level).label("min_resources"),
                    func.max(AgentStateModel.resource_level).label("max_resources"),
                    func.count().label("agent_count"),
                )
                .join(
                    AgentStateModel,
                    SimulationStepModel.step_number == AgentStateModel.step_number,
                )
                .join(AgentModel, AgentStateModel.agent_id == AgentModel.agent_id)
                .group_by(SimulationStepModel.step_number, AgentModel.agent_type)
                .order_by(SimulationStepModel.step_number)
            )

            results = query.all()
            return pd.DataFrame(
                results,
                columns=[
                    "step",
                    "agent_type",
                    "avg_resources",
                    "min_resources",
                    "max_resources",
                    "agent_count",
                ],
            )

        return self.db._execute_in_transaction(_query)

    def analyze_competitive_interactions(self) -> pd.DataFrame:
        """Analyze patterns in competitive interactions.
        
        Derives combat encounters from the actions table by counting attack actions.
        """

        def _query(session):
            query = (
                session.query(
                    ActionModel.step_number,
                    func.count(ActionModel.action_id).label("competitive_interactions"),
                )
                .filter(ActionModel.action_type == "attack")
                .group_by(ActionModel.step_number)
                .order_by(ActionModel.step_number)
            )

            results = query.all()
            return pd.DataFrame(results, columns=["step", "competitive_interactions"])

        return self.db._execute_in_transaction(_query)

    def analyze_resource_efficiency(self) -> pd.DataFrame:
        """Analyze resource utilization efficiency over time."""

        def _query(session):
            query = session.query(
                SimulationStepModel.step_number,
                SimulationStepModel.resource_efficiency.label("efficiency"),
            ).order_by(SimulationStepModel.step_number)

            results = query.all()
            return pd.DataFrame(results, columns=["step", "efficiency"])

        return self.db._execute_in_transaction(_query)

    def generate_report(self, output_file: str = "simulation_report.html"):
        """Generate an HTML report with analysis results."""
        survival_rates = self.calculate_survival_rates()
        efficiency_data = self.analyze_resource_efficiency()

        # Create plots
        plt.figure(figsize=(10, 6))
        plt.plot(efficiency_data["step"], efficiency_data["efficiency"])
        plt.title("Resource Efficiency Over Time")
        plt.savefig("efficiency_plot.png")

        # Generate HTML report
        html = f"""
        <html>
        <head><title>Simulation Analysis Report</title></head>
        <body>
            <h1>Simulation Analysis Report</h1>

            <h2>Survival Rates</h2>
            <table>
                <tr><th>Agent Type</th><th>Survival Rate</th></tr>
                {''.join(f"<tr><td>{k}</td><td>{v:.2%}</td></tr>"
                        for k, v in survival_rates.iloc[-1].items())}
            </table>

            <h2>Resource Efficiency</h2>
            <img src="efficiency_plot.png" />

            <h2>Summary Statistics</h2>
            <pre>{efficiency_data.describe().to_string()}</pre>
        </body>
        </html>
        """

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)


def _json_safe_number(value: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-friendly Python numbers."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    try:
        if hasattr(value, "item"):
            value = value.item()
        fv = float(value)
        if math.isnan(fv) or math.isinf(fv):
            return None
        if float(int(fv)) == fv and abs(fv) < 2**53:
            return int(fv)
        return fv
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _series_describe_dict(series: pd.Series) -> Dict[str, Any]:
    if series.empty:
        return {}
    described = series.describe()
    return {str(k): _json_safe_number(v) for k, v in described.items()}


def _resolve_analyzer_target(simulation_data: Any) -> Tuple[str, Optional[str]]:
    if isinstance(simulation_data, SimulationDatabase):
        return simulation_data.db_path, simulation_data.simulation_id
    if isinstance(simulation_data, str):
        return simulation_data, None
    raise TypeError(
        "analyze_simulation expects a SimulationDatabase instance or a database file path str; "
        f"got {type(simulation_data).__name__}"
    )


def analyze_simulation(simulation_data: Any) -> Dict[str, Any]:
    """
    Analyze simulation data and return metrics.

    Runs the same SQL-backed summaries as :class:`SimulationAnalyzer` (survival,
    resource distribution, competitive interactions, resource efficiency).

    Args:
        simulation_data: :class:`SimulationDatabase` instance or path to the
            simulation SQLite database file.

    Returns:
        dict: ``metrics`` (counts and key aggregates) and ``statistics`` (describe_
        summaries for main series), using JSON-friendly values.

    Raises:
        TypeError: If ``simulation_data`` is not a database instance or path string.
    """
    db_path, simulation_id = _resolve_analyzer_target(simulation_data)
    analyzer = SimulationAnalyzer(db_path=db_path, simulation_id=simulation_id)

    survival = analyzer.calculate_survival_rates()
    resource_dist = analyzer.analyze_resource_distribution()
    combat = analyzer.analyze_competitive_interactions()
    efficiency = analyzer.analyze_resource_efficiency()

    metrics: Dict[str, Any] = {
        "survival_rates_row_count": len(survival),
        "resource_distribution_row_count": len(resource_dist),
        "competitive_interactions_row_count": len(combat),
        "resource_efficiency_row_count": len(efficiency),
    }
    if simulation_id is not None:
        metrics["simulation_id"] = simulation_id
    if db_path != ":memory:":
        metrics["database_basename"] = os.path.basename(db_path)

    if not survival.empty:
        last = survival.iloc[-1]
        metrics["last_step"] = int(last["step"])
        metrics["last_step_system_alive"] = int(last["system_alive"])
        metrics["last_step_independent_alive"] = int(last["independent_alive"])

    if not efficiency.empty and "efficiency" in efficiency.columns:
        metrics["final_resource_efficiency"] = _json_safe_number(
            efficiency["efficiency"].iloc[-1]
        )

    if not combat.empty and "competitive_interactions" in combat.columns:
        metrics["total_competitive_interactions"] = int(
            combat["competitive_interactions"].sum()
        )

    statistics: Dict[str, Any] = {}
    if not survival.empty:
        for col in ("system_alive", "independent_alive"):
            if col in survival.columns:
                statistics[f"survival_{col}"] = _series_describe_dict(survival[col])
    if not efficiency.empty and "efficiency" in efficiency.columns:
        statistics["resource_efficiency"] = _series_describe_dict(efficiency["efficiency"])
    if not resource_dist.empty and "avg_resources" in resource_dist.columns:
        statistics["avg_resources"] = _series_describe_dict(resource_dist["avg_resources"])

    return {"metrics": metrics, "statistics": statistics}

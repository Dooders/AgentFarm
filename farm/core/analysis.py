import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import case, func

from farm.database.database import SimulationDatabase
from farm.database.models import AgentModel, AgentStateModel, SimulationStepModel


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
        """Analyze patterns in competitive interactions."""

        def _query(session):
            query = session.query(
                SimulationStepModel.step_number,
                SimulationStepModel.combat_encounters.label("competitive_interactions"),
            ).order_by(SimulationStepModel.step_number)

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


def analyze_simulation(simulation_data):
    """
    Analyze simulation data and return metrics.

    Args:
        simulation_data: Data from the simulation to analyze

    Returns:
        dict: Analysis results and metrics
    """
    # Add your analysis logic here
    results = {
        "metrics": {},
        "statistics": {},
        # Add other analysis results
    }
    return results

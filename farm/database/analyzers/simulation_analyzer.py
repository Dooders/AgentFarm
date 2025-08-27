import pandas as pd

from farm.database.data_retrieval import DataRetriever
from farm.database.database import SimulationDatabase
from farm.database.session_manager import SessionManager


class SimulationAnalyzer:
    def __init__(self, db_path: str):
        self.db = SimulationDatabase(db_path)
        self.session_manager = SessionManager(db_path)
        self.retriever = DataRetriever(self.session_manager)

    def calculate_survival_rates(self) -> pd.DataFrame:
        """Calculate survival rates using DataRetriever."""
        population_stats = self.retriever.get_population_statistics()
        return pd.DataFrame(population_stats["population_over_time"])

    def analyze_resource_distribution(self) -> pd.DataFrame:
        """Analyze resource distribution using DataRetriever."""
        resource_stats = self.retriever.get_resource_statistics()
        return pd.DataFrame(
            {
                "steps": resource_stats["resource_distribution"]["steps"],
                "total_resources": resource_stats["resource_distribution"][
                    "total_resources"
                ],
                "average_per_agent": resource_stats["resource_distribution"][
                    "average_per_agent"
                ],
            }
        )

    def get_control_agent_stats(self) -> str:
        """Get control agent statistics using DataRetriever."""
        lifespan_stats = self.retriever.get_agent_lifespan_statistics()
        population_stats = self.retriever.get_population_statistics()

        return f"""
        Survival Rate: {lifespan_stats['survival_rate']:.2%}
        Average Resources: {population_stats['resource_metrics']['average_per_agent']:.2f}
        Total Agents: {population_stats['population_metrics']['total']}
        Average Lifespan: {lifespan_stats['average_lifespan']:.2f} steps
        """

    def analyze_population_balance(self) -> str:
        """Analyze the effectiveness of population balance."""
        query = """
        WITH AgentCounts AS (
            SELECT 
                step_number,
                system_agents::FLOAT / NULLIF(total_agents, 0) as system_ratio,
                independent_agents::FLOAT / NULLIF(total_agents, 0) as independent_ratio,
                control_agents::FLOAT / NULLIF(total_agents, 0) as control_ratio,
                total_resources,
                average_agent_resources
            FROM SimulationSteps
            WHERE total_agents > 0
        )
        SELECT 
            AVG(ABS(system_ratio - 0.33)) as system_deviation,
            AVG(ABS(independent_ratio - 0.33)) as independent_deviation,
            AVG(ABS(control_ratio - 0.34)) as control_deviation,
            AVG(total_resources) as avg_resources,
            AVG(average_agent_resources) as avg_agent_resources
        FROM AgentCounts
        """
        with self.db.engine.connect() as conn:
            result = pd.read_sql_query(query, conn)

        return f"""
        Population Balance Analysis:
        System Agent Deviation: {result['system_deviation'][0]:.2%}
        Independent Agent Deviation: {result['independent_deviation'][0]:.2%}
        Control Agent Deviation: {result['control_deviation'][0]:.2%}
        Average System Resources: {result['avg_resources'][0]:.2f}
        Average Agent Resources: {result['avg_agent_resources'][0]:.2f}
        """

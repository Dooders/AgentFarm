#!/usr/bin/env python3

"""
simulation_analysis.py

A comprehensive script for analyzing agent-based simulation results.
This script implements various analysis methods to understand agent behavior,
resource dynamics, and simulation outcomes.
"""
#! THIS SCRIPT MAY BE WRONG
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from farm.database.models import (
    AgentModel,
    ActionModel,
    ResourceModel,
    SimulationStepModel,
    Simulation,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimulationAnalyzer:
    """Class for analyzing simulation results."""

    def __init__(self, db_path: str):
        """Initialize the analyzer with database connection.

        Args:
            db_path: Path to the SQLite database file
        """
        self.engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

    def analyze_population_dynamics(self, simulation_id: int) -> pd.DataFrame:
        """Analyze how agent populations change throughout the simulation.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            DataFrame containing population dynamics data
        """
        logger.info(f"Analyzing population dynamics for simulation {simulation_id}")

        steps = (
            self.session.query(SimulationStepModel)
            .filter(SimulationStepModel.simulation_id == simulation_id)
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        step_data = [
            {
                "step": step.step_number,
                "system_agents": step.system_agents,
                "independent_agents": step.independent_agents,
                "control_agents": step.control_agents,
                "total_agents": step.total_agents,
                "resource_efficiency": step.resource_efficiency,
                "average_agent_health": step.average_agent_health,
                "average_reward": step.average_reward,
            }
            for step in steps
        ]

        df = pd.DataFrame(step_data)

        # Create population dynamics plot
        plt.figure(figsize=(12, 6))
        for agent_type in ["system_agents", "independent_agents", "control_agents"]:
            plt.plot(
                df["step"], df[agent_type], label=agent_type.replace("_agents", "")
            )

        plt.title("Population Dynamics Over Time")
        plt.xlabel("Simulation Step")
        plt.ylabel("Number of Agents")
        plt.legend()
        plt.savefig(f"population_dynamics_sim_{simulation_id}.png")
        plt.close()

        return df

    def analyze_resource_distribution(self, simulation_id: int) -> Dict[str, Any]:
        """Analyze resource distribution patterns and their effects.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            Dictionary containing resource distribution metrics
        """
        logger.info(f"Analyzing resource distribution for simulation {simulation_id}")

        # Sample time points for analysis
        time_points = [10, 100, 500]  # early, mid, late
        resource_stats = {}

        for step in time_points:
            resources = (
                self.session.query(ResourceModel)
                .filter(
                    ResourceModel.simulation_id == simulation_id,
                    ResourceModel.step_number == step,
                )
                .all()
            )

            if not resources:
                continue

            # Calculate resource clustering
            positions = np.array([(r.position_x, r.position_y) for r in resources])
            amounts = np.array([r.amount for r in resources])

            # Basic statistics
            resource_stats[f"step_{step}"] = {
                "total_resources": sum(amounts),
                "mean_amount": np.mean(amounts),
                "std_amount": np.std(amounts),
                "resource_count": len(resources),
            }

            # Calculate spatial distribution metrics
            if len(positions) > 1:
                from scipy.spatial import distance

                distances = distance.pdist(positions)
                resource_stats[f"step_{step}"].update(
                    {
                        "mean_distance": np.mean(distances),
                        "max_distance": np.max(distances),
                        "min_distance": np.min(distances),
                    }
                )

        return resource_stats

    def analyze_agent_interactions(self, simulation_id: int) -> Dict[str, int]:
        """Analyze interactions between different agent types.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            Dictionary containing interaction patterns
        """
        logger.info(f"Analyzing agent interactions for simulation {simulation_id}")

        attack_actions = (
            self.session.query(ActionModel)
            .filter(
                ActionModel.simulation_id == simulation_id,
                ActionModel.action_type == "attack",
            )
            .all()
        )

        interaction_patterns = {}
        for action in attack_actions:
            attacker = (
                self.session.query(AgentModel)
                .filter(AgentModel.agent_id == action.agent_id)
                .first()
            )
            target = (
                self.session.query(AgentModel)
                .filter(AgentModel.agent_id == action.action_target_id)
                .first()
            )

            if attacker and target:
                key = f"{attacker.agent_type}_attacks_{target.agent_type}"
                interaction_patterns[key] = interaction_patterns.get(key, 0) + 1

        # Create interaction heatmap
        interaction_matrix = pd.DataFrame(
            0,
            index=["system", "independent", "control"],
            columns=["system", "independent", "control"],
        )

        for key, value in interaction_patterns.items():
            attacker, target = key.split("_attacks_")
            interaction_matrix.loc[attacker, target] = value

        plt.figure(figsize=(8, 6))
        sns.heatmap(interaction_matrix, annot=True, fmt="d", cmap="YlOrRd")
        plt.title("Agent Interaction Patterns")
        plt.savefig(f"interaction_patterns_sim_{simulation_id}.png")
        plt.close()

        return interaction_patterns

    def analyze_generational_survival(
        self, simulation_id: int
    ) -> Dict[Tuple[int, str], float]:
        """Analyze survival rates across different generations.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            Dictionary containing survival rates by generation and agent type
        """
        logger.info(f"Analyzing generational survival for simulation {simulation_id}")

        agents = (
            self.session.query(AgentModel)
            .filter(AgentModel.simulation_id == simulation_id)
            .all()
        )

        generation_data = {}
        for agent in agents:
            key = (agent.generation, agent.agent_type)
            if key not in generation_data:
                generation_data[key] = {"count": 0, "survived": 0}

            generation_data[key]["count"] += 1
            if agent.death_time is None:
                generation_data[key]["survived"] += 1

        survival_rates = {
            key: data["survived"] / data["count"]
            for key, data in generation_data.items()
        }

        # Create survival rate plot
        generations = sorted(list(set(k[0] for k in survival_rates.keys())))
        agent_types = ["system", "independent", "control"]

        plt.figure(figsize=(12, 6))
        for agent_type in agent_types:
            rates = [survival_rates.get((gen, agent_type), 0) for gen in generations]
            plt.plot(generations, rates, label=agent_type, marker="o")

        plt.title("Survival Rates by Generation")
        plt.xlabel("Generation")
        plt.ylabel("Survival Rate")
        plt.legend()
        plt.savefig(f"survival_rates_sim_{simulation_id}.png")
        plt.close()

        return survival_rates

    def identify_critical_events(self, simulation_id: int) -> List[Dict[str, float]]:
        """Identify critical events that changed simulation trajectory.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            List of dictionaries containing critical events data
        """
        logger.info(f"Identifying critical events for simulation {simulation_id}")

        steps = (
            self.session.query(SimulationStepModel)
            .filter(SimulationStepModel.simulation_id == simulation_id)
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        critical_steps = []
        for i in range(1, len(steps)):
            prev, curr = steps[i - 1], steps[i]

            # Calculate population change rates
            system_change = (curr.system_agents - prev.system_agents) / max(
                prev.system_agents, 1
            )
            indep_change = (curr.independent_agents - prev.independent_agents) / max(
                prev.independent_agents, 1
            )
            control_change = (curr.control_agents - prev.control_agents) / max(
                prev.control_agents, 1
            )

            # If any population changed significantly
            if (
                abs(system_change) > 0.2
                or abs(indep_change) > 0.2
                or abs(control_change) > 0.2
            ):
                critical_steps.append(
                    {
                        "step": curr.step_number,
                        "system_change": system_change,
                        "independent_change": indep_change,
                        "control_change": control_change,
                        "total_agents": curr.total_agents,
                        "resource_efficiency": curr.resource_efficiency,
                    }
                )

        return critical_steps

    def analyze_agent_decisions(self, simulation_id: int) -> pd.DataFrame:
        """Analyze patterns in agent decision-making.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            DataFrame containing decision-making patterns
        """
        logger.info(f"Analyzing agent decisions for simulation {simulation_id}")

        actions = (
            self.session.query(ActionModel)
            .filter(ActionModel.simulation_id == simulation_id)
            .all()
        )

        action_counts = {}
        for action in actions:
            agent = (
                self.session.query(AgentModel)
                .filter(AgentModel.agent_id == action.agent_id)
                .first()
            )
            if agent:
                key = (agent.agent_type, action.action_type)
                action_counts[key] = action_counts.get(key, 0) + 1

        # Convert to DataFrame
        action_df = pd.DataFrame(
            [
                {"agent_type": k[0], "action_type": k[1], "count": v}
                for k, v in action_counts.items()
            ]
        )

        # Create decision pattern plot
        if not action_df.empty:
            plt.figure(figsize=(10, 6))
            action_pivot = action_df.pivot(
                index="agent_type", columns="action_type", values="count"
            ).fillna(0)

            sns.heatmap(action_pivot, annot=True, fmt="g", cmap="YlOrRd")
            plt.title("Agent Decision Patterns")
            plt.savefig(f"decision_patterns_sim_{simulation_id}.png")
            plt.close()

        return action_df

    def run_complete_analysis(self, simulation_id: int) -> Dict[str, Any]:
        """Run all analysis methods for a given simulation.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            Dictionary containing all analysis results
        """
        logger.info(f"Running complete analysis for simulation {simulation_id}")

        results = {
            "population_dynamics": self.analyze_population_dynamics(simulation_id),
            "resource_distribution": self.analyze_resource_distribution(simulation_id),
            "agent_interactions": self.analyze_agent_interactions(simulation_id),
            "generational_survival": self.analyze_generational_survival(simulation_id),
            "critical_events": self.identify_critical_events(simulation_id),
            "agent_decisions": self.analyze_agent_decisions(simulation_id),
        }

        # Save results to file
        output_dir = Path("analysis_results")
        output_dir.mkdir(exist_ok=True)

        results_file = output_dir / f"simulation_{simulation_id}_analysis.json"
        pd.io.json.to_json(results_file, results)

        logger.info(f"Analysis complete. Results saved to {results_file}")
        return results


def main():
    """Main function to run the analysis."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument(
        "--db-path", required=True, help="Path to the simulation database"
    )
    parser.add_argument(
        "--simulation-id",
        type=int,
        required=True,
        help="ID of the simulation to analyze",
    )

    args = parser.parse_args()

    analyzer = SimulationAnalyzer(args.db_path)
    results = analyzer.run_complete_analysis(args.simulation_id)

    logger.info("Analysis completed successfully")


if __name__ == "__main__":
    main()

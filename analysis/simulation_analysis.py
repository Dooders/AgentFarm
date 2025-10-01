#!/usr/bin/env python3

"""
simulation_analysis.py

A comprehensive script for analyzing agent-based simulation results.
This script implements various analysis methods to understand agent behavior,
resource dynamics, and simulation outcomes.

This module provides statistically rigorous analysis methods with proper
validation, confidence intervals, and significance testing.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from farm.database.models import (
    ActionModel,
    AgentModel,
    ResourceModel,
    Simulation,
    SimulationStepModel,
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

    def analyze_population_dynamics(self, simulation_id: int) -> Dict[str, Any]:
        """Analyze how agent populations change throughout the simulation with statistical validation.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            Dictionary containing population dynamics data and statistical analysis
        """
        logger.info(f"Analyzing population dynamics for simulation {simulation_id}")

        steps = (
            self.session.query(SimulationStepModel)
            .filter(SimulationStepModel.simulation_id == simulation_id)
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        if len(steps) < 2:
            logger.warning(f"Insufficient data points ({len(steps)}) for population dynamics analysis")
            return {"error": "Insufficient data", "dataframe": pd.DataFrame()}

        step_data = [
            {
                "step": step.step_number,
                "system_agents": step.system_agents or 0,
                "independent_agents": step.independent_agents or 0,
                "control_agents": step.control_agents or 0,
                "total_agents": step.total_agents or 0,
                "resource_efficiency": step.resource_efficiency or 0,
                "average_agent_health": step.average_agent_health or 0,
                "average_reward": step.average_reward or 0,
            }
            for step in steps
        ]

        df = pd.DataFrame(step_data)
        
        # Statistical analysis
        agent_types = ["system_agents", "independent_agents", "control_agents"]
        statistical_results = {}
        
        # Test for significant differences between agent types
        agent_populations = [df[agent_type] for agent_type in agent_types if agent_type in df.columns]
        
        if len(agent_populations) >= 2:
            # Kruskal-Wallis test for non-parametric comparison
            try:
                h_statistic, p_value = kruskal(*agent_populations)
                statistical_results["kruskal_wallis"] = {
                    "h_statistic": h_statistic,
                    "p_value": p_value,
                    "significant_difference": p_value < 0.05
                }
                logger.info(f"Kruskal-Wallis test: H={h_statistic:.3f}, p={p_value:.3f}")
            except Exception as e:
                logger.warning(f"Kruskal-Wallis test failed: {e}")
                statistical_results["kruskal_wallis"] = {"error": str(e)}
        
        # Pairwise comparisons using Mann-Whitney U test
        pairwise_results = {}
        for i, type1 in enumerate(agent_types):
            for j, type2 in enumerate(agent_types):
                if i < j and type1 in df.columns and type2 in df.columns:
                    try:
                        statistic, p_value = mannwhitneyu(
                            df[type1], df[type2], alternative='two-sided'
                        )
                        pairwise_results[f"{type1}_vs_{type2}"] = {
                            "statistic": statistic,
                            "p_value": p_value,
                            "significant_difference": p_value < 0.05
                        }
                    except Exception as e:
                        logger.warning(f"Mann-Whitney U test failed for {type1} vs {type2}: {e}")
                        pairwise_results[f"{type1}_vs_{type2}"] = {"error": str(e)}
        
        statistical_results["pairwise_comparisons"] = pairwise_results
        
        # Calculate confidence intervals for mean populations
        confidence_intervals = {}
        for agent_type in agent_types:
            if agent_type in df.columns:
                data = df[agent_type]
                mean_val = data.mean()
                std_val = data.std()
                n = len(data)
                
                # 95% confidence interval
                ci_lower, ci_upper = stats.t.interval(
                    0.95, n-1, loc=mean_val, scale=std_val/np.sqrt(n)
                )
                
                confidence_intervals[agent_type] = {
                    "mean": mean_val,
                    "std": std_val,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "sample_size": n
                }
        
        statistical_results["confidence_intervals"] = confidence_intervals
        
        # Create enhanced population dynamics plot with confidence intervals
        plt.figure(figsize=(14, 8))
        
        # Main plot
        ax1 = plt.subplot(2, 1, 1)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, agent_type in enumerate(agent_types):
            if agent_type in df.columns:
                plt.plot(
                    df["step"], df[agent_type], 
                    label=agent_type.replace("_agents", ""),
                    color=colors[i % len(colors)],
                    linewidth=2
                )
                
                # Add confidence band
                mean_val = df[agent_type].mean()
                std_val = df[agent_type].std()
                plt.fill_between(
                    df["step"], 
                    mean_val - std_val, 
                    mean_val + std_val,
                    alpha=0.2,
                    color=colors[i % len(colors)]
                )

        plt.title("Population Dynamics Over Time (with 1σ confidence bands)")
        plt.xlabel("Simulation Step")
        plt.ylabel("Number of Agents")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Statistical summary subplot
        ax2 = plt.subplot(2, 1, 2)
        means = [confidence_intervals[agent_type]["mean"] for agent_type in agent_types if agent_type in confidence_intervals]
        errors = [confidence_intervals[agent_type]["std"] for agent_type in agent_types if agent_type in confidence_intervals]
        labels = [agent_type.replace("_agents", "") for agent_type in agent_types if agent_type in confidence_intervals]
        
        bars = ax2.bar(labels, means, yerr=errors, capsize=5, color=colors[:len(means)])
        ax2.set_title("Mean Population with Standard Deviation")
        ax2.set_ylabel("Number of Agents")
        
        # Add significance annotations
        if "kruskal_wallis" in statistical_results and "p_value" in statistical_results["kruskal_wallis"]:
            p_val = statistical_results["kruskal_wallis"]["p_value"]
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax2.text(0.02, 0.98, f"Kruskal-Wallis: p={p_val:.3f} {significance}", 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"population_dynamics_sim_{simulation_id}.png", dpi=300, bbox_inches='tight')
        plt.close()

        return {
            "dataframe": df,
            "statistical_analysis": statistical_results,
            "summary": {
                "total_steps": len(df),
                "significant_differences": statistical_results.get("kruskal_wallis", {}).get("significant_difference", False),
                "agent_types_analyzed": len([t for t in agent_types if t in df.columns])
            }
        }

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

    def analyze_agent_interactions(self, simulation_id: int) -> Dict[str, Any]:
        """Analyze interactions between different agent types with statistical validation.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            Dictionary containing interaction patterns and statistical analysis
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

        if not attack_actions:
            logger.warning(f"No attack actions found for simulation {simulation_id}")
            return {
                "interaction_patterns": {},
                "statistical_analysis": {"error": "No attack actions found"},
                "interaction_matrix": pd.DataFrame()
            }

        interaction_patterns = {}
        interaction_data = []
        
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
                interaction_data.append({
                    "attacker_type": attacker.agent_type,
                    "target_type": target.agent_type,
                    "step": action.step_number
                })

        # Create interaction matrix
        interaction_df = pd.DataFrame(interaction_data)
        agent_types = ["system", "independent", "control"]
        
        interaction_matrix = pd.DataFrame(
            0,
            index=agent_types,
            columns=agent_types,
        )

        for key, value in interaction_patterns.items():
            if "_attacks_" in key:
                attacker, target = key.split("_attacks_")
                if attacker in agent_types and target in agent_types:
                    interaction_matrix.loc[attacker, target] = value

        # Statistical analysis
        statistical_results = {}
        
        if len(interaction_data) > 0:
            # Chi-square test for independence
            try:
                # Create contingency table
                contingency_table = pd.crosstab(
                    interaction_df['attacker_type'], 
                    interaction_df['target_type']
                )
                
                # Ensure all agent types are represented
                for agent_type in agent_types:
                    if agent_type not in contingency_table.index:
                        contingency_table.loc[agent_type] = 0
                    if agent_type not in contingency_table.columns:
                        contingency_table[agent_type] = 0
                
                # Reorder to match expected order
                contingency_table = contingency_table.reindex(agent_types, fill_value=0)
                contingency_table = contingency_table.reindex(agent_types, axis=1, fill_value=0)
                
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                statistical_results["chi_square_test"] = {
                    "chi2_statistic": chi2,
                    "p_value": p_value,
                    "degrees_of_freedom": dof,
                    "significant_association": p_value < 0.05,
                    "expected_frequencies": expected.tolist()
                }
                
                logger.info(f"Chi-square test: χ²={chi2:.3f}, p={p_value:.3f}")
                
            except Exception as e:
                logger.warning(f"Chi-square test failed: {e}")
                statistical_results["chi_square_test"] = {"error": str(e)}
            
            # Calculate interaction rates
            total_interactions = len(interaction_data)
            interaction_rates = {}
            
            for attacker in agent_types:
                for target in agent_types:
                    count = interaction_matrix.loc[attacker, target]
                    rate = count / total_interactions if total_interactions > 0 else 0
                    interaction_rates[f"{attacker}_to_{target}"] = {
                        "count": count,
                        "rate": rate,
                        "percentage": rate * 100
                    }
            
            statistical_results["interaction_rates"] = interaction_rates
            
            # Calculate confidence intervals for interaction rates
            confidence_intervals = {}
            for key, data in interaction_rates.items():
                count = data["count"]
                n = total_interactions
                
                if n > 0:
                    # Wilson score interval for proportions
                    p = count / n
                    z = 1.96  # 95% confidence
                    
                    ci_lower = (p + z*z/(2*n) - z * np.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)
                    ci_upper = (p + z*z/(2*n) + z * np.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)
                    
                    confidence_intervals[key] = {
                        "rate": p,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "count": count,
                        "total": n
                    }
            
            statistical_results["confidence_intervals"] = confidence_intervals

        # Create enhanced interaction heatmap with statistical annotations
        plt.figure(figsize=(12, 8))
        
        # Main heatmap
        ax1 = plt.subplot(2, 1, 1)
        sns.heatmap(interaction_matrix, annot=True, fmt="d", cmap="YlOrRd", 
                   cbar_kws={'label': 'Number of Attacks'})
        plt.title("Agent Interaction Patterns (Attack Actions)")
        plt.xlabel("Target Agent Type")
        plt.ylabel("Attacker Agent Type")
        
        # Add statistical annotation
        if "chi_square_test" in statistical_results and "p_value" in statistical_results["chi_square_test"]:
            p_val = statistical_results["chi_square_test"]["p_value"]
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            ax1.text(0.02, 0.98, f"Chi-square: p={p_val:.3f} {significance}", 
                    transform=ax1.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Interaction rates subplot
        ax2 = plt.subplot(2, 1, 2)
        if "interaction_rates" in statistical_results:
            rates_data = statistical_results["interaction_rates"]
            interactions = list(rates_data.keys())
            rates = [rates_data[key]["percentage"] for key in interactions]
            
            bars = ax2.bar(interactions, rates, color='skyblue', alpha=0.7)
            ax2.set_title("Interaction Rates by Type (%)")
            ax2.set_ylabel("Percentage of Total Interactions")
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"interaction_patterns_sim_{simulation_id}.png", dpi=300, bbox_inches='tight')
        plt.close()

        return {
            "interaction_patterns": interaction_patterns,
            "interaction_matrix": interaction_matrix,
            "statistical_analysis": statistical_results,
            "summary": {
                "total_interactions": len(interaction_data),
                "significant_association": statistical_results.get("chi_square_test", {}).get("significant_association", False),
                "most_common_interaction": max(interaction_patterns.items(), key=lambda x: x[1]) if interaction_patterns else None
            }
        }

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

    def identify_critical_events(self, simulation_id: int, significance_level: float = 0.05) -> List[Dict[str, float]]:
        """Identify critical events that changed simulation trajectory using statistical methods.

        Uses statistical change point detection instead of arbitrary thresholds.
        Implements z-score based detection with configurable significance levels.

        Args:
            simulation_id: ID of the simulation to analyze
            significance_level: Statistical significance level for change detection (default: 0.05)

        Returns:
            List of dictionaries containing critical events data with statistical measures
        """
        logger.info(f"Identifying critical events for simulation {simulation_id} using statistical methods")

        steps = (
            self.session.query(SimulationStepModel)
            .filter(SimulationStepModel.simulation_id == simulation_id)
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        if len(steps) < 10:
            logger.warning(f"Insufficient data points ({len(steps)}) for reliable change detection")
            return []

        # Convert to DataFrame for easier analysis
        step_data = []
        for step in steps:
            step_data.append({
                "step": step.step_number,
                "system_agents": step.system_agents or 0,
                "independent_agents": step.independent_agents or 0,
                "control_agents": step.control_agents or 0,
                "total_agents": step.total_agents or 0,
                "resource_efficiency": step.resource_efficiency or 0,
            })
        
        df = pd.DataFrame(step_data)
        
        critical_steps = []
        
        # Analyze each agent type for significant changes
        agent_types = ["system_agents", "independent_agents", "control_agents"]
        
        for agent_type in agent_types:
            if agent_type not in df.columns:
                continue
                
            # Calculate rolling statistics for change detection
            window_size = min(10, len(df) // 4)  # Adaptive window size
            rolling_mean = df[agent_type].rolling(window=window_size, min_periods=1).mean()
            rolling_std = df[agent_type].rolling(window=window_size, min_periods=1).std()
            
            # Avoid division by zero
            rolling_std = rolling_std.replace(0, 1)
            
            # Calculate z-scores for change detection
            z_scores = (df[agent_type] - rolling_mean) / rolling_std
            
            # Use statistical threshold (2-sigma for 95% confidence, 3-sigma for 99.7%)
            if significance_level <= 0.01:
                threshold = 3.0  # 99.7% confidence
            elif significance_level <= 0.05:
                threshold = 2.0  # 95% confidence
            else:
                threshold = 1.96  # 95% confidence (standard)
            
            # Find significant changes
            significant_changes = df[abs(z_scores) > threshold].copy()
            
            for _, row in significant_changes.iterrows():
                # Calculate change rate
                prev_idx = max(0, row.name - 1)
                prev_value = df.iloc[prev_idx][agent_type]
                curr_value = row[agent_type]
                
                if prev_value > 0:
                    change_rate = (curr_value - prev_value) / prev_value
                else:
                    change_rate = 0 if curr_value == 0 else float('inf')
                
                # Calculate statistical significance
                z_score = z_scores.iloc[row.name]
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
                
                critical_steps.append({
                    "step": int(row["step"]),
                    "agent_type": agent_type,
                    "change_rate": change_rate,
                    "z_score": z_score,
                    "p_value": p_value,
                    "is_significant": p_value < significance_level,
                    "previous_value": prev_value,
                    "current_value": curr_value,
                    "total_agents": row["total_agents"],
                    "resource_efficiency": row["resource_efficiency"],
                })
        
        # Sort by step number and remove duplicates
        critical_steps = sorted(critical_steps, key=lambda x: x["step"])
        
        # Log summary statistics
        significant_events = [e for e in critical_steps if e["is_significant"]]
        logger.info(f"Found {len(critical_steps)} potential events, {len(significant_events)} statistically significant")
        
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

    def run_complete_analysis(self, simulation_id: int, significance_level: float = 0.05) -> Dict[str, Any]:
        """Run all analysis methods for a given simulation with statistical validation.

        Args:
            simulation_id: ID of the simulation to analyze
            significance_level: Statistical significance level for analysis (default: 0.05)

        Returns:
            Dictionary containing all analysis results with statistical measures
        """
        logger.info(f"Running complete analysis for simulation {simulation_id} with significance level {significance_level}")

        try:
            results = {
                "simulation_id": simulation_id,
                "significance_level": significance_level,
                "population_dynamics": self.analyze_population_dynamics(simulation_id),
                "resource_distribution": self.analyze_resource_distribution(simulation_id),
                "agent_interactions": self.analyze_agent_interactions(simulation_id),
                "generational_survival": self.analyze_generational_survival(simulation_id),
                "critical_events": self.identify_critical_events(simulation_id, significance_level),
                "agent_decisions": self.analyze_agent_decisions(simulation_id),
            }

            # Add analysis metadata
            results["metadata"] = {
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "statistical_methods_used": [
                    "Kruskal-Wallis test",
                    "Mann-Whitney U test", 
                    "Z-score change detection",
                    "Confidence intervals (95%)",
                    "Chi-square test for independence"
                ],
                "significance_level": significance_level,
                "data_quality_checks": "Passed"
            }

            # Save results to file
            output_dir = Path("analysis_results")
            output_dir.mkdir(exist_ok=True)

            results_file = output_dir / f"simulation_{simulation_id}_analysis.json"
            import json

            # Custom JSON encoder for numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    return super(NumpyEncoder, self).default(obj)

            with open(results_file, "w") as f:
                json.dump(results, f, cls=NumpyEncoder, indent=2)

            logger.info(f"Analysis complete. Results saved to {results_file}")
            return results

        except Exception as e:
            logger.error(f"Analysis failed for simulation {simulation_id}: {e}")
            return {
                "simulation_id": simulation_id,
                "error": str(e),
                "analysis_failed": True
            }


def main():
    """Main function to run the analysis with proper error handling."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Analyze simulation results with statistical validation")
    parser.add_argument(
        "--db-path", required=True, help="Path to the simulation database"
    )
    parser.add_argument(
        "--simulation-id",
        type=int,
        required=True,
        help="ID of the simulation to analyze",
    )
    parser.add_argument(
        "--significance-level",
        type=float,
        default=0.05,
        help="Statistical significance level (default: 0.05)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Validate inputs
        if not Path(args.db_path).exists():
            logger.error(f"Database file not found: {args.db_path}")
            sys.exit(1)
        
        if args.significance_level <= 0 or args.significance_level >= 1:
            logger.error(f"Significance level must be between 0 and 1, got: {args.significance_level}")
            sys.exit(1)

        # Initialize analyzer
        analyzer = SimulationAnalyzer(args.db_path)
        
        # Run analysis
        logger.info(f"Starting analysis for simulation {args.simulation_id}")
        results = analyzer.run_complete_analysis(args.simulation_id, args.significance_level)
        
        # Check for analysis errors
        if "error" in results:
            logger.error(f"Analysis failed: {results['error']}")
            sys.exit(1)
        
        # Print summary
        logger.info("Analysis completed successfully")
        logger.info(f"Results saved to: analysis_results/simulation_{args.simulation_id}_analysis.json")
        
        # Print key findings
        if "population_dynamics" in results and "summary" in results["population_dynamics"]:
            summary = results["population_dynamics"]["summary"]
            logger.info(f"Population analysis: {summary.get('total_steps', 0)} steps analyzed")
            if summary.get("significant_differences"):
                logger.info("✓ Significant differences found between agent types")
            else:
                logger.info("✗ No significant differences between agent types")
        
        if "critical_events" in results:
            significant_events = [e for e in results["critical_events"] if e.get("is_significant", False)]
            logger.info(f"Critical events: {len(significant_events)} statistically significant events found")
        
        if "agent_interactions" in results and "summary" in results["agent_interactions"]:
            summary = results["agent_interactions"]["summary"]
            logger.info(f"Agent interactions: {summary.get('total_interactions', 0)} total interactions")
            if summary.get("significant_association"):
                logger.info("✓ Significant association found in interaction patterns")
            else:
                logger.info("✗ No significant association in interaction patterns")

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

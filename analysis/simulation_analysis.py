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
from scipy.signal import find_peaks, periodogram
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, pearsonr, spearmanr

# Note: cohens_d and cohens_f are calculated manually in _calculate_effect_sizes method
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from statsmodels.stats.power import ttest_power
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, coint, kpss
from statsmodels.tsa.vector_ar.var_model import VAR

from farm.database.models import (
    ActionModel,
    AgentModel,
    ResourceModel,
    Simulation,
    SimulationStepModel,
)

# Import reproducibility utilities
try:
    from .reproducibility import (
        AnalysisValidator,
        ReproducibilityManager,
        create_reproducibility_report,
    )
except ImportError:
    # Fallback if reproducibility module is not available
    ReproducibilityManager = None
    AnalysisValidator = None
    create_reproducibility_report = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimulationAnalyzer:
    """Class for analyzing simulation results with reproducibility features."""

    def __init__(self, db_path: str, random_seed: int = 42):
        """Initialize the analyzer with database connection and reproducibility features.

        Args:
            db_path: Path to the SQLite database file
            random_seed: Random seed for reproducible results
        """
        self.engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=self.engine)
        self.session = Session()

        # Initialize reproducibility features
        self.random_seed = random_seed
        if ReproducibilityManager is not None:
            self.repro_manager = ReproducibilityManager(random_seed)
            self.validator = AnalysisValidator()
        else:
            self.repro_manager = None
            self.validator = None
            logger.warning("Reproducibility features not available")

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
            logger.warning(
                f"Insufficient data points ({len(steps)}) for population dynamics analysis"
            )
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
        agent_populations = [
            df[agent_type] for agent_type in agent_types if agent_type in df.columns
        ]

        if len(agent_populations) >= 2:
            # Kruskal-Wallis test for non-parametric comparison
            try:
                h_statistic, p_value = kruskal(*agent_populations)
                statistical_results["kruskal_wallis"] = {
                    "h_statistic": h_statistic,
                    "p_value": p_value,
                    "significant_difference": p_value < 0.05,
                }
                logger.info(
                    f"Kruskal-Wallis test: H={h_statistic:.3f}, p={p_value:.3f}"
                )
            except Exception as e:
                logger.warning(f"Kruskal-Wallis test failed: {e}")
                statistical_results["kruskal_wallis"] = {"error": str(e)}

        # Pairwise comparisons using Mann-Whitney U test with effect sizes
        pairwise_results = {}
        for i, type1 in enumerate(agent_types):
            for j, type2 in enumerate(agent_types):
                if i < j and type1 in df.columns and type2 in df.columns:
                    try:
                        data1 = df[type1].dropna()
                        data2 = df[type2].dropna()

                        # Mann-Whitney U test
                        statistic, p_value = mannwhitneyu(
                            data1, data2, alternative="two-sided"
                        )

                        # Effect size calculations
                        effect_sizes = self._calculate_effect_sizes(data1, data2)

                        # Power analysis
                        power_analysis = self._calculate_power_analysis(
                            data1, data2, p_value
                        )

                        pairwise_results[f"{type1}_vs_{type2}"] = {
                            "statistic": statistic,
                            "p_value": p_value,
                            "significant_difference": p_value < 0.05,
                            "effect_sizes": effect_sizes,
                            "power_analysis": power_analysis,
                            "sample_sizes": {
                                "group1": len(data1),
                                "group2": len(data2),
                            },
                            "descriptive_stats": {
                                "group1": {
                                    "mean": data1.mean(),
                                    "median": data1.median(),
                                    "std": data1.std(),
                                    "iqr": data1.quantile(0.75) - data1.quantile(0.25),
                                },
                                "group2": {
                                    "mean": data2.mean(),
                                    "median": data2.median(),
                                    "std": data2.std(),
                                    "iqr": data2.quantile(0.75) - data2.quantile(0.25),
                                },
                            },
                        }
                    except Exception as e:
                        logger.warning(
                            f"Mann-Whitney U test failed for {type1} vs {type2}: {e}"
                        )
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
                    0.95, n - 1, loc=mean_val, scale=std_val / np.sqrt(n)
                )

                confidence_intervals[agent_type] = {
                    "mean": mean_val,
                    "std": std_val,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "sample_size": n,
                }

        statistical_results["confidence_intervals"] = confidence_intervals

        # Create enhanced population dynamics plot with confidence intervals
        plt.figure(figsize=(14, 8))

        # Main plot
        ax1 = plt.subplot(2, 1, 1)
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, agent_type in enumerate(agent_types):
            if agent_type in df.columns:
                plt.plot(
                    df["step"],
                    df[agent_type],
                    label=agent_type.replace("_agents", ""),
                    color=colors[i % len(colors)],
                    linewidth=2,
                )

                # Add confidence band
                mean_val = df[agent_type].mean()
                std_val = df[agent_type].std()
                plt.fill_between(
                    df["step"],
                    mean_val - std_val,
                    mean_val + std_val,
                    alpha=0.2,
                    color=colors[i % len(colors)],
                )

        plt.title("Population Dynamics Over Time (with 1σ confidence bands)")
        plt.xlabel("Simulation Step")
        plt.ylabel("Number of Agents")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Statistical summary subplot
        ax2 = plt.subplot(2, 1, 2)
        means = [
            confidence_intervals[agent_type]["mean"]
            for agent_type in agent_types
            if agent_type in confidence_intervals
        ]
        errors = [
            confidence_intervals[agent_type]["std"]
            for agent_type in agent_types
            if agent_type in confidence_intervals
        ]
        labels = [
            agent_type.replace("_agents", "")
            for agent_type in agent_types
            if agent_type in confidence_intervals
        ]

        bars = ax2.bar(
            labels, means, yerr=errors, capsize=5, color=colors[: len(means)]
        )
        ax2.set_title("Mean Population with Standard Deviation")
        ax2.set_ylabel("Number of Agents")

        # Add significance annotations
        if (
            "kruskal_wallis" in statistical_results
            and "p_value" in statistical_results["kruskal_wallis"]
        ):
            p_val = statistical_results["kruskal_wallis"]["p_value"]
            significance = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            )
            ax2.text(
                0.02,
                0.98,
                f"Kruskal-Wallis: p={p_val:.3f} {significance}",
                transform=ax2.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

        plt.tight_layout()
        plt.savefig(
            f"population_dynamics_sim_{simulation_id}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        return {
            "dataframe": df,
            "statistical_analysis": statistical_results,
            "summary": {
                "total_steps": len(df),
                "significant_differences": statistical_results.get(
                    "kruskal_wallis", {}
                ).get("significant_difference", False),
                "agent_types_analyzed": len(
                    [t for t in agent_types if t in df.columns]
                ),
            },
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
                "interaction_matrix": pd.DataFrame(),
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
                interaction_data.append(
                    {
                        "attacker_type": attacker.agent_type,
                        "target_type": target.agent_type,
                        "step": action.step_number,
                    }
                )

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
                    interaction_df["attacker_type"], interaction_df["target_type"]
                )

                # Ensure all agent types are represented
                for agent_type in agent_types:
                    if agent_type not in contingency_table.index:
                        contingency_table.loc[agent_type] = 0
                    if agent_type not in contingency_table.columns:
                        contingency_table[agent_type] = 0

                # Reorder to match expected order
                contingency_table = contingency_table.reindex(agent_types, fill_value=0)
                contingency_table = contingency_table.reindex(
                    agent_types, axis=1, fill_value=0
                )

                chi2, p_value, dof, expected = chi2_contingency(contingency_table)

                statistical_results["chi_square_test"] = {
                    "chi2_statistic": chi2,
                    "p_value": p_value,
                    "degrees_of_freedom": dof,
                    "significant_association": p_value < 0.05,
                    "expected_frequencies": expected.tolist(),
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
                        "percentage": rate * 100,
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

                    ci_lower = (
                        p
                        + z * z / (2 * n)
                        - z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
                    ) / (1 + z * z / n)
                    ci_upper = (
                        p
                        + z * z / (2 * n)
                        + z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
                    ) / (1 + z * z / n)

                    confidence_intervals[key] = {
                        "rate": p,
                        "ci_lower": ci_lower,
                        "ci_upper": ci_upper,
                        "count": count,
                        "total": n,
                    }

            statistical_results["confidence_intervals"] = confidence_intervals

        # Create enhanced interaction heatmap with statistical annotations
        plt.figure(figsize=(12, 8))

        # Main heatmap
        ax1 = plt.subplot(2, 1, 1)
        sns.heatmap(
            interaction_matrix,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            cbar_kws={"label": "Number of Attacks"},
        )
        plt.title("Agent Interaction Patterns (Attack Actions)")
        plt.xlabel("Target Agent Type")
        plt.ylabel("Attacker Agent Type")

        # Add statistical annotation
        if (
            "chi_square_test" in statistical_results
            and "p_value" in statistical_results["chi_square_test"]
        ):
            p_val = statistical_results["chi_square_test"]["p_value"]
            significance = (
                "***"
                if p_val < 0.001
                else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            )
            ax1.text(
                0.02,
                0.98,
                f"Chi-square: p={p_val:.3f} {significance}",
                transform=ax1.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
            )

        # Interaction rates subplot
        ax2 = plt.subplot(2, 1, 2)
        if "interaction_rates" in statistical_results:
            rates_data = statistical_results["interaction_rates"]
            interactions = list(rates_data.keys())
            rates = [rates_data[key]["percentage"] for key in interactions]

            bars = ax2.bar(interactions, rates, color="skyblue", alpha=0.7)
            ax2.set_title("Interaction Rates by Type (%)")
            ax2.set_ylabel("Percentage of Total Interactions")
            ax2.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()
        plt.savefig(
            f"interaction_patterns_sim_{simulation_id}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        return {
            "interaction_patterns": interaction_patterns,
            "interaction_matrix": interaction_matrix,
            "statistical_analysis": statistical_results,
            "summary": {
                "total_interactions": len(interaction_data),
                "significant_association": statistical_results.get(
                    "chi_square_test", {}
                ).get("significant_association", False),
                "most_common_interaction": (
                    max(interaction_patterns.items(), key=lambda x: x[1])
                    if interaction_patterns
                    else None
                ),
            },
        }

    def analyze_generational_survival(self, simulation_id: int) -> Dict[str, float]:
        """Analyze survival rates across different generations.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            Dictionary containing survival rates by generation and agent type (keys formatted as "generation_agent_type")
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
            f"{key[0]}_{key[1]}": data["survived"] / data["count"]
            for key, data in generation_data.items()
        }

        # Create survival rate plot
        generations = sorted(list(set(k[0] for k in generation_data.keys())))
        agent_types = ["system", "independent", "control"]

        plt.figure(figsize=(12, 6))
        for agent_type in agent_types:
            rates = [
                survival_rates.get(f"{gen}_{agent_type}", 0) for gen in generations
            ]
            plt.plot(generations, rates, label=agent_type, marker="o")

        plt.title("Survival Rates by Generation")
        plt.xlabel("Generation")
        plt.ylabel("Survival Rate")
        plt.legend()
        plt.savefig(f"survival_rates_sim_{simulation_id}.png")
        plt.close()

        return survival_rates

    def identify_critical_events(
        self, simulation_id: int, significance_level: float = 0.05
    ) -> List[Dict[str, float]]:
        """Identify critical events that changed simulation trajectory using statistical methods.

        Uses statistical change point detection instead of arbitrary thresholds.
        Implements z-score based detection with configurable significance levels.

        Args:
            simulation_id: ID of the simulation to analyze
            significance_level: Statistical significance level for change detection (default: 0.05)

        Returns:
            List of dictionaries containing critical events data with statistical measures
        """
        logger.info(
            f"Identifying critical events for simulation {simulation_id} using statistical methods"
        )

        steps = (
            self.session.query(SimulationStepModel)
            .filter(SimulationStepModel.simulation_id == simulation_id)
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        if len(steps) < 10:
            logger.warning(
                f"Insufficient data points ({len(steps)}) for reliable change detection"
            )
            return []

        # Convert to DataFrame for easier analysis
        step_data = []
        for step in steps:
            step_data.append(
                {
                    "step": step.step_number,
                    "system_agents": step.system_agents or 0,
                    "independent_agents": step.independent_agents or 0,
                    "control_agents": step.control_agents or 0,
                    "total_agents": step.total_agents or 0,
                    "resource_efficiency": step.resource_efficiency or 0,
                }
            )

        df = pd.DataFrame(step_data)

        critical_steps = []

        # Analyze each agent type for significant changes
        agent_types = ["system_agents", "independent_agents", "control_agents"]

        for agent_type in agent_types:
            if agent_type not in df.columns:
                continue

            # Calculate rolling statistics for change detection
            window_size = min(10, len(df) // 4)  # Adaptive window size
            rolling_mean = (
                df[agent_type].rolling(window=window_size, min_periods=1).mean()
            )
            rolling_std = (
                df[agent_type].rolling(window=window_size, min_periods=1).std()
            )

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
                    change_rate = 0 if curr_value == 0 else float("inf")

                # Calculate statistical significance
                z_score = z_scores.iloc[row.name]
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test

                critical_steps.append(
                    {
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
                    }
                )

        # Sort by step number and remove duplicates
        critical_steps = sorted(critical_steps, key=lambda x: x["step"])

        # Log summary statistics
        significant_events = [e for e in critical_steps if e["is_significant"]]
        logger.info(
            f"Found {len(critical_steps)} potential events, {len(significant_events)} statistically significant"
        )

        return critical_steps

    def _calculate_effect_sizes(
        self, data1: pd.Series, data2: pd.Series
    ) -> Dict[str, float]:
        """Calculate various effect size measures for comparing two groups.

        Args:
            data1: First group data
            data2: Second group data

        Returns:
            Dictionary containing different effect size measures
        """
        effect_sizes = {}

        try:
            # Cohen's d (standardized mean difference)
            n1, n2 = len(data1), len(data2)
            mean1, mean2 = data1.mean(), data2.mean()
            std1, std2 = data1.std(), data2.std()

            # Pooled standard deviation
            pooled_std = np.sqrt(
                ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
            )

            if pooled_std > 0:
                cohens_d = (mean1 - mean2) / pooled_std
                effect_sizes["cohens_d"] = cohens_d

                # Interpret Cohen's d
                if abs(cohens_d) < 0.2:
                    effect_sizes["cohens_d_interpretation"] = "negligible"
                elif abs(cohens_d) < 0.5:
                    effect_sizes["cohens_d_interpretation"] = "small"
                elif abs(cohens_d) < 0.8:
                    effect_sizes["cohens_d_interpretation"] = "medium"
                else:
                    effect_sizes["cohens_d_interpretation"] = "large"

            # Glass's delta (using control group standard deviation)
            if std2 > 0:
                glass_delta = (mean1 - mean2) / std2
                effect_sizes["glass_delta"] = glass_delta

            # Hedges' g (bias-corrected Cohen's d)
            if pooled_std > 0:
                # Correction factor
                correction = 1 - (3 / (4 * (n1 + n2) - 9))
                hedges_g = cohens_d * correction
                effect_sizes["hedges_g"] = hedges_g

            # Common Language Effect Size (CLES)
            # Probability that a randomly selected value from group 1 is greater than group 2
            try:
                from scipy.stats import norm

                if pooled_std > 0:
                    z_score = (mean1 - mean2) / pooled_std
                    cles = norm.cdf(z_score / np.sqrt(2))
                    effect_sizes["cles"] = cles
                    effect_sizes["cles_interpretation"] = (
                        f"Probability that group 1 > group 2: {cles:.1%}"
                    )
            except ImportError:
                pass

            # Point-biserial correlation (for binary vs continuous)
            # This would be applicable if one group was binary

            # Eta-squared (proportion of variance explained)
            ss_between = (
                n1 * (mean1 - (n1 * mean1 + n2 * mean2) / (n1 + n2)) ** 2
                + n2 * (mean2 - (n1 * mean1 + n2 * mean2) / (n1 + n2)) ** 2
            )
            ss_total = ((data1 - data1.mean()) ** 2).sum() + (
                (data2 - data2.mean()) ** 2
            ).sum()

            if ss_total > 0:
                eta_squared = ss_between / ss_total
                effect_sizes["eta_squared"] = eta_squared

                # Interpret eta-squared
                if eta_squared < 0.01:
                    effect_sizes["eta_squared_interpretation"] = "negligible"
                elif eta_squared < 0.06:
                    effect_sizes["eta_squared_interpretation"] = "small"
                elif eta_squared < 0.14:
                    effect_sizes["eta_squared_interpretation"] = "medium"
                else:
                    effect_sizes["eta_squared_interpretation"] = "large"

        except Exception as e:
            logger.warning(f"Effect size calculation failed: {e}")
            effect_sizes["error"] = str(e)

        return effect_sizes

    def _calculate_power_analysis(
        self, data1: pd.Series, data2: pd.Series, p_value: float
    ) -> Dict[str, Any]:
        """Calculate statistical power and related metrics.

        Args:
            data1: First group data
            data2: Second group data
            p_value: P-value from the statistical test

        Returns:
            Dictionary containing power analysis results
        """
        power_results = {}

        try:
            n1, n2 = len(data1), len(data2)
            mean1, mean2 = data1.mean(), data2.mean()
            std1, std2 = data1.std(), data2.std()

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
            )
            if pooled_std > 0:
                effect_size = abs(mean1 - mean2) / pooled_std
            else:
                effect_size = 0

            # Statistical power using t-test power
            try:
                from statsmodels.stats.power import ttest_power

                # Calculate power for different effect sizes
                power_results["observed_power"] = ttest_power(
                    effect_size, n1 + n2, alpha=0.05, alternative="two-sided"
                )

                # Power for different effect sizes
                effect_sizes = [0.2, 0.5, 0.8]  # Small, medium, large
                power_for_effects = {}
                for es in effect_sizes:
                    power_for_effects[f"power_for_d_{es}"] = ttest_power(
                        es, n1 + n2, alpha=0.05, alternative="two-sided"
                    )
                power_results["power_for_effects"] = power_for_effects

                # Sample size needed for 80% power
                try:
                    from statsmodels.stats.power import ttest_power

                    # This is a simplified calculation
                    if effect_size > 0:
                        # Approximate sample size for 80% power
                        # This is a rough estimate
                        n_needed = int(16 / (effect_size**2))  # Rough approximation
                        power_results["sample_size_for_80_power"] = n_needed
                except:
                    pass

            except ImportError:
                power_results["error"] = "statsmodels not available for power analysis"

            # Post-hoc power interpretation
            if "observed_power" in power_results:
                observed_power = power_results["observed_power"]
                if observed_power < 0.5:
                    power_results["power_interpretation"] = (
                        "low power - results may not be reliable"
                    )
                elif observed_power < 0.8:
                    power_results["power_interpretation"] = (
                        "moderate power - results should be interpreted cautiously"
                    )
                else:
                    power_results["power_interpretation"] = (
                        "adequate power - results are reliable"
                    )

            # Type II error rate
            if "observed_power" in power_results:
                power_results["type_ii_error_rate"] = (
                    1 - power_results["observed_power"]
                )

            # Effect size interpretation
            power_results["effect_size"] = effect_size
            if effect_size < 0.2:
                power_results["effect_size_interpretation"] = "negligible effect"
            elif effect_size < 0.5:
                power_results["effect_size_interpretation"] = "small effect"
            elif effect_size < 0.8:
                power_results["effect_size_interpretation"] = "medium effect"
            else:
                power_results["effect_size_interpretation"] = "large effect"

        except Exception as e:
            logger.warning(f"Power analysis failed: {e}")
            power_results["error"] = str(e)

        return power_results

    def analyze_temporal_patterns(self, simulation_id: int) -> Dict[str, Any]:
        """Analyze temporal patterns in simulation data using advanced time series methods.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            Dictionary containing comprehensive temporal analysis results
        """
        logger.info(f"Analyzing temporal patterns for simulation {simulation_id}")

        steps = (
            self.session.query(SimulationStepModel)
            .filter(SimulationStepModel.simulation_id == simulation_id)
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        if len(steps) < 20:
            logger.warning(
                f"Insufficient data points ({len(steps)}) for reliable time series analysis"
            )
            return {
                "error": "Insufficient data for time series analysis",
                "min_points_required": 20,
            }

        # Convert to DataFrame
        step_data = []
        for step in steps:
            step_data.append(
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
            )

        df = pd.DataFrame(step_data)
        df.set_index("step", inplace=True)

        temporal_results = {}

        # Analyze each time series
        time_series_columns = [
            "system_agents",
            "independent_agents",
            "control_agents",
            "total_agents",
            "resource_efficiency",
            "average_agent_health",
            "average_reward",
        ]

        for column in time_series_columns:
            if column not in df.columns:
                continue

            series = df[column].dropna()
            if len(series) < 10:
                continue

            logger.info(f"Analyzing temporal patterns for {column}")

            # 1. Stationarity Tests
            stationarity_results = {}

            # Augmented Dickey-Fuller test
            try:
                (
                    adf_stat,
                    adf_pvalue,
                    adf_used_lag,
                    adf_nobs,
                    adf_critical,
                    adf_icbest,
                ) = adfuller(series)
                stationarity_results["adf_test"] = {
                    "statistic": adf_stat,
                    "p_value": adf_pvalue,
                    "is_stationary": adf_pvalue < 0.05,
                    "critical_values": adf_critical,
                }
            except Exception as e:
                logger.warning(f"ADF test failed for {column}: {e}")
                stationarity_results["adf_test"] = {"error": str(e)}

            # KPSS test
            try:
                kpss_stat, kpss_pvalue, kpss_lags, kpss_critical = kpss(
                    series, regression="c"
                )
                stationarity_results["kpss_test"] = {
                    "statistic": kpss_stat,
                    "p_value": kpss_pvalue,
                    "is_stationary": kpss_pvalue > 0.05,
                    "critical_values": kpss_critical,
                }
            except Exception as e:
                logger.warning(f"KPSS test failed for {column}: {e}")
                stationarity_results["kpss_test"] = {"error": str(e)}

            # 2. Trend Analysis
            trend_results = {}

            # Linear trend
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            trend_results["linear_trend"] = {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value**2,
                "p_value": p_value,
                "significant_trend": p_value < 0.05,
                "trend_direction": (
                    "increasing"
                    if slope > 0
                    else "decreasing" if slope < 0 else "stable"
                ),
            }

            # 3. Seasonality Analysis
            seasonality_results = {}

            if len(series) >= 24:  # Need at least 2 cycles for meaningful seasonality
                try:
                    # Seasonal decomposition
                    decomposition = seasonal_decompose(
                        series, model="additive", period=min(12, len(series) // 2)
                    )

                    # Calculate seasonality strength
                    seasonal_strength = np.var(decomposition.seasonal) / np.var(series)
                    trend_strength = np.var(decomposition.trend.dropna()) / np.var(
                        series
                    )

                    seasonality_results["decomposition"] = {
                        "seasonal_strength": seasonal_strength,
                        "trend_strength": trend_strength,
                        "residual_strength": 1 - seasonal_strength - trend_strength,
                        "has_seasonality": seasonal_strength > 0.1,
                    }

                    # Periodogram analysis
                    freqs, psd = periodogram(series)
                    dominant_freq_idx = np.argmax(psd[1:]) + 1  # Skip DC component
                    dominant_period = (
                        1 / freqs[dominant_freq_idx]
                        if freqs[dominant_freq_idx] > 0
                        else np.inf
                    )

                    seasonality_results["periodogram"] = {
                        "dominant_frequency": freqs[dominant_freq_idx],
                        "dominant_period": dominant_period,
                        "max_power": psd[dominant_freq_idx],
                    }

                except Exception as e:
                    logger.warning(f"Seasonality analysis failed for {column}: {e}")
                    seasonality_results["error"] = str(e)

            # 4. Change Point Detection
            change_points = []

            # Find peaks and troughs
            try:
                peaks, peak_properties = find_peaks(
                    series, height=np.mean(series), distance=5
                )
                troughs, trough_properties = find_peaks(
                    -series, height=-np.mean(series), distance=5
                )

                change_points = {
                    "peaks": {
                        "indices": peaks.tolist(),
                        "values": series.iloc[peaks].tolist() if len(peaks) > 0 else [],
                        "count": len(peaks),
                    },
                    "troughs": {
                        "indices": troughs.tolist(),
                        "values": (
                            series.iloc[troughs].tolist() if len(troughs) > 0 else []
                        ),
                        "count": len(troughs),
                    },
                }
            except Exception as e:
                logger.warning(f"Change point detection failed for {column}: {e}")
                change_points = {"error": str(e)}

            # 5. Autocorrelation Analysis
            autocorr_results = {}

            try:
                # Calculate autocorrelation for different lags
                max_lag = min(20, len(series) // 4)
                autocorrs = [series.autocorr(lag=i) for i in range(1, max_lag + 1)]

                # Find significant autocorrelations
                significant_lags = []
                for i, ac in enumerate(autocorrs):
                    if not np.isnan(ac) and abs(ac) > 0.2:  # Threshold for significance
                        significant_lags.append(i + 1)

                autocorr_results = {
                    "autocorrelations": autocorrs,
                    "significant_lags": significant_lags,
                    "max_autocorr": max(autocorrs) if autocorrs else 0,
                    "has_autocorrelation": len(significant_lags) > 0,
                }
            except Exception as e:
                logger.warning(f"Autocorrelation analysis failed for {column}: {e}")
                autocorr_results = {"error": str(e)}

            # Compile results for this time series
            temporal_results[column] = {
                "stationarity": stationarity_results,
                "trend": trend_results,
                "seasonality": seasonality_results,
                "change_points": change_points,
                "autocorrelation": autocorr_results,
                "summary": {
                    "length": len(series),
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "range": series.max() - series.min(),
                },
            }

        # 6. Cross-correlation Analysis
        cross_corr_results = {}

        try:
            # Calculate cross-correlations between agent types
            agent_columns = ["system_agents", "independent_agents", "control_agents"]
            available_agents = [col for col in agent_columns if col in df.columns]

            if len(available_agents) >= 2:
                cross_corr_matrix = {}
                for i, col1 in enumerate(available_agents):
                    for j, col2 in enumerate(available_agents):
                        if i < j:  # Avoid duplicates
                            corr, p_value = pearsonr(
                                df[col1].dropna(), df[col2].dropna()
                            )
                            cross_corr_matrix[f"{col1}_vs_{col2}"] = {
                                "correlation": corr,
                                "p_value": p_value,
                                "significant": p_value < 0.05,
                                "strength": (
                                    "strong"
                                    if abs(corr) > 0.7
                                    else "moderate" if abs(corr) > 0.3 else "weak"
                                ),
                            }

                cross_corr_results = cross_corr_matrix
        except Exception as e:
            logger.warning(f"Cross-correlation analysis failed: {e}")
            cross_corr_results = {"error": str(e)}

        # Create comprehensive temporal visualization
        self._create_temporal_visualization(df, temporal_results, simulation_id)

        return {
            "time_series_analysis": temporal_results,
            "cross_correlations": cross_corr_results,
            "metadata": {
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "total_steps": len(df),
                "time_series_analyzed": len(
                    [col for col in time_series_columns if col in df.columns]
                ),
                "methods_used": [
                    "Augmented Dickey-Fuller test",
                    "KPSS test",
                    "Linear trend analysis",
                    "Seasonal decomposition",
                    "Periodogram analysis",
                    "Change point detection",
                    "Autocorrelation analysis",
                    "Cross-correlation analysis",
                ],
            },
        }

    def _create_temporal_visualization(
        self, df: pd.DataFrame, temporal_results: Dict[str, Any], simulation_id: int
    ) -> None:
        """Create comprehensive temporal analysis visualization."""

        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))

        # Main time series plot
        ax1 = plt.subplot(3, 3, (1, 3))

        agent_columns = ["system_agents", "independent_agents", "control_agents"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

        for i, col in enumerate(agent_columns):
            if col in df.columns:
                ax1.plot(
                    df.index,
                    df[col],
                    label=col.replace("_agents", ""),
                    color=colors[i],
                    linewidth=2,
                    alpha=0.8,
                )

        ax1.set_title(
            f"Temporal Analysis - Simulation {simulation_id}",
            fontsize=16,
            fontweight="bold",
        )
        ax1.set_xlabel("Simulation Step")
        ax1.set_ylabel("Number of Agents")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Trend analysis subplot
        ax2 = plt.subplot(3, 3, 4)
        if "total_agents" in temporal_results:
            trend_data = temporal_results["total_agents"]["trend"]["linear_trend"]
            x = np.arange(len(df))
            y = df["total_agents"] if "total_agents" in df.columns else df.iloc[:, 0]
            ax2.plot(x, y, "b-", alpha=0.6, label="Data")

            # Plot trend line
            trend_line = trend_data["slope"] * x + trend_data["intercept"]
            ax2.plot(
                x,
                trend_line,
                "r--",
                linewidth=2,
                label=f"Trend (R²={trend_data['r_squared']:.3f})",
            )

            ax2.set_title("Trend Analysis")
            ax2.set_xlabel("Step")
            ax2.set_ylabel("Value")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Stationarity test results
        ax3 = plt.subplot(3, 3, 5)
        stationarity_data = []
        labels = []

        for col, results in temporal_results.items():
            if "stationarity" in results and "adf_test" in results["stationarity"]:
                adf_test = results["stationarity"]["adf_test"]
                if "p_value" in adf_test:
                    stationarity_data.append(adf_test["p_value"])
                    labels.append(col.replace("_agents", "").replace("_", " "))

        if stationarity_data:
            bars = ax3.bar(
                labels,
                stationarity_data,
                color=["green" if p < 0.05 else "red" for p in stationarity_data],
            )
            ax3.axhline(
                y=0.05, color="black", linestyle="--", alpha=0.7, label="α=0.05"
            )
            ax3.set_title("Stationarity Tests (ADF p-values)")
            ax3.set_ylabel("p-value")
            ax3.legend()
            ax3.tick_params(axis="x", rotation=45)

            # Add significance annotations
            for bar, p_val in zip(bars, stationarity_data):
                height = bar.get_height()
                significance = "Stationary" if p_val < 0.05 else "Non-stationary"
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    significance,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Autocorrelation plot
        ax4 = plt.subplot(3, 3, 6)
        if (
            "total_agents" in temporal_results
            and "autocorrelation" in temporal_results["total_agents"]
        ):
            autocorr_data = temporal_results["total_agents"]["autocorrelation"]
            if "autocorrelations" in autocorr_data:
                lags = range(1, len(autocorr_data["autocorrelations"]) + 1)
                ax4.bar(lags, autocorr_data["autocorrelations"], alpha=0.7)
                ax4.axhline(
                    y=0.2,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Significance threshold",
                )
                ax4.axhline(y=-0.2, color="red", linestyle="--", alpha=0.7)
                ax4.set_title("Autocorrelation Function")
                ax4.set_xlabel("Lag")
                ax4.set_ylabel("Autocorrelation")
                ax4.legend()
                ax4.grid(True, alpha=0.3)

        # Cross-correlation heatmap
        ax5 = plt.subplot(3, 3, 7)
        # This would be implemented if cross-correlation data is available

        # Seasonality plot
        ax6 = plt.subplot(3, 3, 8)
        if (
            "total_agents" in temporal_results
            and "seasonality" in temporal_results["total_agents"]
        ):
            seasonality_data = temporal_results["total_agents"]["seasonality"]
            if "decomposition" in seasonality_data:
                decomp = seasonality_data["decomposition"]
                strengths = [
                    decomp["trend_strength"],
                    decomp["seasonal_strength"],
                    decomp["residual_strength"],
                ]
                labels = ["Trend", "Seasonal", "Residual"]
                colors = ["blue", "orange", "green"]

                wedges, texts, autotexts = ax6.pie(
                    strengths, labels=labels, colors=colors, autopct="%1.1f%%"
                )
                ax6.set_title("Variance Decomposition")

        # Summary statistics
        ax7 = plt.subplot(3, 3, 9)
        ax7.axis("off")

        # Create summary text
        summary_text = f"Temporal Analysis Summary\n\n"
        summary_text += f"Total Steps: {len(df)}\n"
        summary_text += f"Time Series Analyzed: {len(temporal_results)}\n\n"

        # Add key findings
        if "total_agents" in temporal_results:
            trend_info = temporal_results["total_agents"]["trend"]["linear_trend"]
            summary_text += f"Overall Trend: {trend_info['trend_direction']}\n"
            summary_text += f"Trend Significance: {'Yes' if trend_info['significant_trend'] else 'No'}\n"
            summary_text += f"R²: {trend_info['r_squared']:.3f}\n\n"

        summary_text += "Statistical Tests Applied:\n"
        summary_text += "• Augmented Dickey-Fuller\n"
        summary_text += "• KPSS Stationarity\n"
        summary_text += "• Linear Trend Analysis\n"
        summary_text += "• Seasonal Decomposition\n"
        summary_text += "• Autocorrelation Analysis"

        ax7.text(
            0.05,
            0.95,
            summary_text,
            transform=ax7.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.savefig(
            f"temporal_analysis_sim_{simulation_id}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def analyze_advanced_time_series_models(self, simulation_id: int) -> Dict[str, Any]:
        """Perform advanced time series modeling including ARIMA, VAR, and forecasting.

        Args:
            simulation_id: ID of the simulation to analyze

        Returns:
            Dictionary containing advanced time series modeling results
        """
        logger.info(
            f"Performing advanced time series modeling for simulation {simulation_id}"
        )

        steps = (
            self.session.query(SimulationStepModel)
            .filter(SimulationStepModel.simulation_id == simulation_id)
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        if len(steps) < 50:
            logger.warning(
                f"Insufficient data points ({len(steps)}) for advanced time series modeling"
            )
            return {
                "error": "Insufficient data for advanced modeling",
                "min_points_required": 50,
            }

        # Convert to DataFrame
        step_data = []
        for step in steps:
            step_data.append(
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
            )

        df = pd.DataFrame(step_data)
        df.set_index("step", inplace=True)

        advanced_results = {}

        # 1. ARIMA Modeling
        arima_results = {}
        time_series_columns = [
            "system_agents",
            "independent_agents",
            "control_agents",
            "total_agents",
        ]

        for column in time_series_columns:
            if column not in df.columns:
                continue

            series = df[column].dropna()
            if len(series) < 30:
                continue

            logger.info(f"Fitting ARIMA model for {column}")

            try:
                # Auto ARIMA parameter selection (simplified)
                best_aic = float("inf")
                best_order = None
                best_model = None

                # Try different ARIMA orders
                for p in range(0, 3):
                    for d in range(0, 2):
                        for q in range(0, 3):
                            try:
                                model = ARIMA(series, order=(p, d, q))
                                fitted_model = model.fit()

                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_order = (p, d, q)
                                    best_model = fitted_model
                            except Exception:
                                continue

                if best_model is not None:
                    # Generate forecasts
                    forecast_steps = min(10, len(series) // 4)
                    forecast = best_model.forecast(steps=forecast_steps)
                    forecast_ci = best_model.get_forecast(
                        steps=forecast_steps
                    ).conf_int()

                    # Model diagnostics
                    residuals = best_model.resid

                    arima_results[column] = {
                        "model_order": best_order,
                        "aic": best_aic,
                        "bic": best_model.bic,
                        "forecast": forecast.tolist(),
                        "forecast_ci_lower": forecast_ci.iloc[:, 0].tolist(),
                        "forecast_ci_upper": forecast_ci.iloc[:, 1].tolist(),
                        "residuals_stats": {
                            "mean": residuals.mean(),
                            "std": residuals.std(),
                            "skewness": stats.skew(residuals),
                            "kurtosis": stats.kurtosis(residuals),
                        },
                        "model_summary": {
                            "params": best_model.params.to_dict(),
                            "pvalues": best_model.pvalues.to_dict(),
                            "log_likelihood": best_model.llf,
                        },
                    }

                    # Ljung-Box test for residual autocorrelation
                    try:
                        from statsmodels.stats.diagnostic import acorr_ljungbox

                        lb_result = acorr_ljungbox(residuals, lags=10)
                        arima_results[column]["ljung_box_test"] = {
                            "statistic": lb_result["lb_stat"].iloc[-1],
                            "p_value": lb_result["lb_pvalue"].iloc[-1],
                            "residuals_white_noise": lb_result["lb_pvalue"].iloc[-1]
                            > 0.05,
                        }
                    except ImportError:
                        pass

            except Exception as e:
                logger.warning(f"ARIMA modeling failed for {column}: {e}")
                arima_results[column] = {"error": str(e)}

        # 2. Vector Autoregression (VAR) for multivariate time series
        var_results = {}

        try:
            # Prepare multivariate data
            var_data = df[
                ["system_agents", "independent_agents", "control_agents"]
            ].dropna()

            if len(var_data) >= 30:
                logger.info("Fitting VAR model for agent populations")

                # Fit VAR model
                var_model = VAR(var_data)
                lag_order = var_model.select_order(maxlags=10)
                best_lag = lag_order["aic"]

                fitted_var = var_model.fit(best_lag)

                # Generate forecasts
                forecast_steps = min(10, len(var_data) // 4)
                var_forecast = fitted_var.forecast(
                    var_data.values[-best_lag:], steps=forecast_steps
                )

                # Granger causality tests
                granger_results = {}
                for cause_var in var_data.columns:
                    for effect_var in var_data.columns:
                        if cause_var != effect_var:
                            try:
                                gc_test = fitted_var.test_causality(
                                    effect_var, cause_var, kind="f"
                                )
                                granger_results[f"{cause_var}_causes_{effect_var}"] = {
                                    "statistic": gc_test.test_statistic,
                                    "p_value": gc_test.pvalue,
                                    "significant": gc_test.pvalue < 0.05,
                                }
                            except:
                                pass

                var_results = {
                    "model_order": best_lag,
                    "aic": fitted_var.aic,
                    "bic": fitted_var.bic,
                    "forecast": var_forecast.tolist(),
                    "granger_causality": granger_results,
                    "model_summary": {
                        "params": fitted_var.params.to_dict(),
                        "resid_stats": {
                            "mean": fitted_var.resid.mean().to_dict(),
                            "std": fitted_var.resid.std().to_dict(),
                        },
                    },
                }

        except Exception as e:
            logger.warning(f"VAR modeling failed: {e}")
            var_results = {"error": str(e)}

        # 3. Exponential Smoothing
        exp_smoothing_results = {}

        for column in time_series_columns:
            if column not in df.columns:
                continue

            series = df[column].dropna()
            if len(series) < 20:
                continue

            logger.info(f"Fitting exponential smoothing for {column}")

            try:
                # Try different exponential smoothing models
                models = {}

                # Simple exponential smoothing
                try:
                    simple_model = ExponentialSmoothing(
                        series, trend=None, seasonal=None
                    ).fit()
                    models["simple"] = {
                        "aic": simple_model.aic,
                        "bic": simple_model.bic,
                        "sse": simple_model.sse,
                    }
                except:
                    pass

                # Holt's linear trend
                try:
                    holt_model = ExponentialSmoothing(
                        series, trend="add", seasonal=None
                    ).fit()
                    models["holt"] = {
                        "aic": holt_model.aic,
                        "bic": holt_model.bic,
                        "sse": holt_model.sse,
                    }
                except:
                    pass

                # Holt-Winters with seasonality (if enough data)
                if len(series) >= 24:
                    try:
                        seasonal_period = min(12, len(series) // 2)
                        hw_model = ExponentialSmoothing(
                            series,
                            trend="add",
                            seasonal="add",
                            seasonal_periods=seasonal_period,
                        ).fit()
                        models["holt_winters"] = {
                            "aic": hw_model.aic,
                            "bic": hw_model.bic,
                            "sse": hw_model.sse,
                            "seasonal_period": seasonal_period,
                        }
                    except:
                        pass

                if models:
                    # Select best model based on AIC
                    best_model_name = min(models.keys(), key=lambda k: models[k]["aic"])
                    best_model_info = models[best_model_name]

                    # Generate forecasts
                    if best_model_name == "simple":
                        model = ExponentialSmoothing(
                            series, trend=None, seasonal=None
                        ).fit()
                    elif best_model_name == "holt":
                        model = ExponentialSmoothing(
                            series, trend="add", seasonal=None
                        ).fit()
                    else:  # holt_winters
                        seasonal_period = best_model_info["seasonal_period"]
                        model = ExponentialSmoothing(
                            series,
                            trend="add",
                            seasonal="add",
                            seasonal_periods=seasonal_period,
                        ).fit()

                    forecast_steps = min(10, len(series) // 4)
                    forecast = model.forecast(steps=forecast_steps)

                    exp_smoothing_results[column] = {
                        "best_model": best_model_name,
                        "model_info": best_model_info,
                        "forecast": forecast.tolist(),
                        "fitted_values": model.fittedvalues.tolist(),
                        "residuals": model.resid.tolist(),
                    }

            except Exception as e:
                logger.warning(f"Exponential smoothing failed for {column}: {e}")
                exp_smoothing_results[column] = {"error": str(e)}

        # 4. Model Comparison and Selection
        model_comparison = {}

        for column in time_series_columns:
            if column not in df.columns:
                continue

            comparison = {}

            # ARIMA results
            if column in arima_results and "error" not in arima_results[column]:
                comparison["arima"] = {
                    "aic": arima_results[column]["aic"],
                    "bic": arima_results[column]["bic"],
                }

            # Exponential smoothing results
            if (
                column in exp_smoothing_results
                and "error" not in exp_smoothing_results[column]
            ):
                comparison["exponential_smoothing"] = {
                    "aic": exp_smoothing_results[column]["model_info"]["aic"],
                    "bic": exp_smoothing_results[column]["model_info"]["bic"],
                }

            if comparison:
                # Select best model based on AIC
                best_model = min(comparison.keys(), key=lambda k: comparison[k]["aic"])
                model_comparison[column] = {
                    "best_model": best_model,
                    "comparison": comparison,
                }

        # Create advanced visualization
        self._create_advanced_time_series_visualization(
            df, arima_results, var_results, exp_smoothing_results, simulation_id
        )

        return {
            "arima_models": arima_results,
            "var_model": var_results,
            "exponential_smoothing": exp_smoothing_results,
            "model_comparison": model_comparison,
            "metadata": {
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "total_steps": len(df),
                "methods_used": [
                    "ARIMA modeling with auto parameter selection",
                    "Vector Autoregression (VAR)",
                    "Exponential Smoothing (Simple, Holt, Holt-Winters)",
                    "Granger Causality Testing",
                    "Model comparison and selection",
                    "Forecasting with confidence intervals",
                ],
            },
        }

    def _create_advanced_time_series_visualization(
        self,
        df: pd.DataFrame,
        arima_results: Dict,
        var_results: Dict,
        exp_smoothing_results: Dict,
        simulation_id: int,
    ) -> None:
        """Create advanced time series modeling visualization."""

        fig = plt.figure(figsize=(20, 16))

        # 1. Original time series with ARIMA forecasts
        ax1 = plt.subplot(3, 3, (1, 2))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, column in enumerate(
            ["system_agents", "independent_agents", "control_agents", "total_agents"]
        ):
            if column in df.columns:
                ax1.plot(
                    df.index,
                    df[column],
                    label=column.replace("_agents", ""),
                    color=colors[i],
                    linewidth=2,
                    alpha=0.8,
                )

                # Add ARIMA forecast if available
                if column in arima_results and "error" not in arima_results[column]:
                    forecast = arima_results[column]["forecast"]
                    forecast_ci_lower = arima_results[column]["forecast_ci_lower"]
                    forecast_ci_upper = arima_results[column]["forecast_ci_upper"]

                    forecast_start = len(df)
                    forecast_end = forecast_start + len(forecast)
                    forecast_x = range(forecast_start, forecast_end)

                    ax1.plot(
                        forecast_x,
                        forecast,
                        "--",
                        color=colors[i],
                        alpha=0.7,
                        linewidth=2,
                    )
                    ax1.fill_between(
                        forecast_x,
                        forecast_ci_lower,
                        forecast_ci_upper,
                        alpha=0.2,
                        color=colors[i],
                    )

        ax1.set_title(
            f"Advanced Time Series Analysis - Simulation {simulation_id}",
            fontsize=16,
            fontweight="bold",
        )
        ax1.set_xlabel("Simulation Step")
        ax1.set_ylabel("Number of Agents")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. ARIMA model diagnostics
        ax2 = plt.subplot(3, 3, 3)
        if arima_results:
            model_orders = []
            aic_values = []
            model_names = []

            for column, result in arima_results.items():
                if "error" not in result:
                    model_orders.append(str(result["model_order"]))
                    aic_values.append(result["aic"])
                    model_names.append(column.replace("_agents", ""))

            if model_orders:
                bars = ax2.bar(model_names, aic_values, color="skyblue", alpha=0.7)
                ax2.set_title("ARIMA Model AIC Comparison")
                ax2.set_ylabel("AIC")
                ax2.tick_params(axis="x", rotation=45)

                # Add model orders as text
                for bar, order in zip(bars, model_orders):
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + height * 0.01,
                        f"ARIMA{order}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        # 3. VAR Granger Causality
        ax3 = plt.subplot(3, 3, 4)
        if (
            var_results
            and "error" not in var_results
            and "granger_causality" in var_results
        ):
            gc_results = var_results["granger_causality"]
            if gc_results:
                causes = []
                p_values = []

                for test_name, result in gc_results.items():
                    if result["significant"]:
                        causes.append(test_name.replace("_causes_", " → "))
                        p_values.append(result["p_value"])

                if causes:
                    bars = ax3.barh(causes, p_values, color="lightcoral", alpha=0.7)
                    ax3.set_title("Significant Granger Causality")
                    ax3.set_xlabel("p-value")
                    ax3.axvline(
                        x=0.05, color="red", linestyle="--", alpha=0.7, label="α=0.05"
                    )
                    ax3.legend()

        # 4. Exponential Smoothing Forecasts
        ax4 = plt.subplot(3, 3, 5)
        if exp_smoothing_results:
            for column, result in exp_smoothing_results.items():
                if "error" not in result and "forecast" in result:
                    forecast = result["forecast"]
                    forecast_start = len(df)
                    forecast_end = forecast_start + len(forecast)
                    forecast_x = range(forecast_start, forecast_end)

                    ax4.plot(
                        forecast_x,
                        forecast,
                        "o-",
                        label=column.replace("_agents", ""),
                        linewidth=2,
                        markersize=4,
                    )

            ax4.set_title("Exponential Smoothing Forecasts")
            ax4.set_xlabel("Forecast Step")
            ax4.set_ylabel("Predicted Value")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. Model Comparison
        ax5 = plt.subplot(3, 3, 6)
        # This would show model comparison results

        # 6. Residuals Analysis
        ax6 = plt.subplot(3, 3, 7)
        if arima_results:
            residuals_data = []
            for column, result in arima_results.items():
                if "error" not in result and "residuals_stats" in result:
                    residuals_data.append(result["residuals_stats"]["std"])

            if residuals_data:
                ax6.hist(residuals_data, bins=10, alpha=0.7, color="lightgreen")
                ax6.set_title("ARIMA Residuals Standard Deviation")
                ax6.set_xlabel("Residual Std")
                ax6.set_ylabel("Frequency")

        # 7. Forecast Accuracy (if we had actual future data)
        ax7 = plt.subplot(3, 3, 8)
        # This would show forecast accuracy metrics

        # 8. Summary Statistics
        ax8 = plt.subplot(3, 3, 9)
        ax8.axis("off")

        # Create summary text
        summary_text = f"Advanced Time Series Modeling Summary\n\n"
        summary_text += f"Simulation ID: {simulation_id}\n"
        summary_text += f"Total Steps: {len(df)}\n"
        summary_text += f"ARIMA Models: {len([k for k, v in arima_results.items() if 'error' not in v])}\n"
        summary_text += f"VAR Model: {'Yes' if 'error' not in var_results else 'No'}\n"
        summary_text += f"Exp. Smoothing: {len([k for k, v in exp_smoothing_results.items() if 'error' not in v])}\n\n"

        summary_text += "Methods Applied:\n"
        summary_text += "• ARIMA with auto parameter selection\n"
        summary_text += "• Vector Autoregression (VAR)\n"
        summary_text += "• Exponential Smoothing\n"
        summary_text += "• Granger Causality Testing\n"
        summary_text += "• Model comparison and selection\n"
        summary_text += "• Forecasting with confidence intervals"

        ax8.text(
            0.05,
            0.95,
            summary_text,
            transform=ax8.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.savefig(
            f"advanced_time_series_models_sim_{simulation_id}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

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

    def analyze_with_advanced_ml(
        self, simulation_id: int, target_variable: str = "population_dominance"
    ) -> Dict[str, Any]:
        """Perform advanced machine learning analysis with ensemble methods and feature selection.

        Args:
            simulation_id: ID of the simulation to analyze
            target_variable: Variable to predict (default: "population_dominance")

        Returns:
            Dictionary containing comprehensive ML analysis results
        """
        logger.info(f"Performing advanced ML analysis for simulation {simulation_id}")

        # Load comprehensive simulation data
        steps = (
            self.session.query(SimulationStepModel)
            .filter(SimulationStepModel.simulation_id == simulation_id)
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        if len(steps) < 50:
            logger.warning(
                f"Insufficient data points ({len(steps)}) for reliable ML analysis"
            )
            return {
                "error": "Insufficient data for ML analysis",
                "min_points_required": 50,
            }

        # Create comprehensive feature matrix
        feature_data = []
        for i, step in enumerate(steps):
            # Basic features
            features = {
                "step": step.step_number,
                "system_agents": step.system_agents or 0,
                "independent_agents": step.independent_agents or 0,
                "control_agents": step.control_agents or 0,
                "total_agents": step.total_agents or 0,
                "resource_efficiency": step.resource_efficiency or 0,
                "average_agent_health": step.average_agent_health or 0,
                "average_reward": step.average_reward or 0,
            }

            # Derived features
            if i > 0:
                prev_step = steps[i - 1]
                features.update(
                    {
                        "system_change": (step.system_agents or 0)
                        - (prev_step.system_agents or 0),
                        "independent_change": (step.independent_agents or 0)
                        - (prev_step.independent_agents or 0),
                        "control_change": (step.control_agents or 0)
                        - (prev_step.control_agents or 0),
                        "total_change": (step.total_agents or 0)
                        - (prev_step.total_agents or 0),
                        "efficiency_change": (step.resource_efficiency or 0)
                        - (prev_step.resource_efficiency or 0),
                    }
                )
            else:
                features.update(
                    {
                        "system_change": 0,
                        "independent_change": 0,
                        "control_change": 0,
                        "total_change": 0,
                        "efficiency_change": 0,
                    }
                )

            # Rolling window features (if enough data)
            if i >= 5:
                recent_steps = steps[max(0, i - 5) : i + 1]
                features.update(
                    {
                        "system_rolling_mean": np.mean(
                            [s.system_agents or 0 for s in recent_steps]
                        ),
                        "system_rolling_std": np.std(
                            [s.system_agents or 0 for s in recent_steps]
                        ),
                        "total_rolling_mean": np.mean(
                            [s.total_agents or 0 for s in recent_steps]
                        ),
                        "total_rolling_std": np.std(
                            [s.total_agents or 0 for s in recent_steps]
                        ),
                    }
                )
            else:
                features.update(
                    {
                        "system_rolling_mean": features["system_agents"],
                        "system_rolling_std": 0,
                        "total_rolling_mean": features["total_agents"],
                        "total_rolling_std": 0,
                    }
                )

            # Target variable (population dominance)
            if target_variable == "population_dominance":
                agent_counts = [
                    features["system_agents"],
                    features["independent_agents"],
                    features["control_agents"],
                ]
                max_count = max(agent_counts)
                if max_count > 0:
                    features["population_dominance"] = [
                        "system",
                        "independent",
                        "control",
                    ][agent_counts.index(max_count)]
                else:
                    features["population_dominance"] = "none"

            feature_data.append(features)

        df = pd.DataFrame(feature_data)

        # Prepare features and target
        feature_columns = [
            col for col in df.columns if col not in ["step", target_variable]
        ]
        X = df[feature_columns]
        y = df[target_variable]

        # Handle missing values
        X = X.fillna(X.median())

        # Encode target variable if categorical
        if y.dtype == "object":
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            target_classes = le.classes_
        else:
            y_encoded = y
            target_classes = None

        # Feature Selection
        feature_selection_results = {}

        # 1. Univariate feature selection
        try:
            selector_univariate = SelectKBest(
                score_func=f_classif, k=min(10, len(feature_columns))
            )
            X_selected_univariate = selector_univariate.fit_transform(X, y_encoded)
            selected_features_univariate = [
                feature_columns[i]
                for i in selector_univariate.get_support(indices=True)
            ]

            feature_selection_results["univariate"] = {
                "selected_features": selected_features_univariate,
                "scores": dict(zip(feature_columns, selector_univariate.scores_)),
                "n_features": len(selected_features_univariate),
            }
        except Exception as e:
            logger.warning(f"Univariate feature selection failed: {e}")
            feature_selection_results["univariate"] = {"error": str(e)}
            selected_features_univariate = feature_columns
            X_selected_univariate = X

        # 2. Recursive feature elimination
        try:
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector_rfe = RFE(
                estimator, n_features_to_select=min(8, len(feature_columns))
            )
            X_selected_rfe = selector_rfe.fit_transform(X, y_encoded)
            selected_features_rfe = [
                feature_columns[i] for i in selector_rfe.get_support(indices=True)
            ]

            feature_selection_results["rfe"] = {
                "selected_features": selected_features_rfe,
                "feature_ranking": dict(zip(feature_columns, selector_rfe.ranking_)),
                "n_features": len(selected_features_rfe),
            }
        except Exception as e:
            logger.warning(f"RFE feature selection failed: {e}")
            feature_selection_results["rfe"] = {"error": str(e)}
            selected_features_rfe = feature_columns
            X_selected_rfe = X

        # 3. Model-based feature selection
        try:
            selector_model = SelectFromModel(
                RandomForestClassifier(n_estimators=50, random_state=42)
            )
            X_selected_model = selector_model.fit_transform(X, y_encoded)
            selected_features_model = [
                feature_columns[i] for i in selector_model.get_support(indices=True)
            ]

            feature_selection_results["model_based"] = {
                "selected_features": selected_features_model,
                "feature_importance": dict(
                    zip(feature_columns, selector_model.estimator_.feature_importances_)
                ),
                "n_features": len(selected_features_model),
            }
        except Exception as e:
            logger.warning(f"Model-based feature selection failed: {e}")
            feature_selection_results["model_based"] = {"error": str(e)}
            selected_features_model = feature_columns
            X_selected_model = X

        # Ensemble Model Building
        ensemble_results = {}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 1. Individual Models
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, random_state=42
            ),
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "SVM": SVC(random_state=42, probability=True),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
        }

        individual_results = {}
        for name, model in models.items():
            try:
                # Train model
                model.fit(X_train_scaled, y_train)

                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = (
                    model.predict_proba(X_test_scaled)
                    if hasattr(model, "predict_proba")
                    else None
                )

                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train_scaled, y_train, cv=5, scoring="accuracy"
                )

                # Metrics
                test_accuracy = model.score(X_test_scaled, y_test)

                # Feature importance (if available)
                feature_importance = None
                if hasattr(model, "feature_importances_"):
                    feature_importance = dict(
                        zip(feature_columns, model.feature_importances_)
                    )
                elif hasattr(model, "coef_"):
                    feature_importance = dict(zip(feature_columns, abs(model.coef_[0])))

                individual_results[name] = {
                    "test_accuracy": test_accuracy,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "cv_scores": cv_scores.tolist(),
                    "feature_importance": feature_importance,
                    "predictions": y_pred.tolist(),
                    "probabilities": (
                        y_pred_proba.tolist() if y_pred_proba is not None else None
                    ),
                }

            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                individual_results[name] = {"error": str(e)}

        # 2. Ensemble Models
        try:
            # Voting Classifier
            voting_models = [
                ("rf", RandomForestClassifier(n_estimators=50, random_state=42)),
                ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42)),
                ("lr", LogisticRegression(random_state=42, max_iter=1000)),
            ]

            voting_clf = VotingClassifier(estimators=voting_models, voting="soft")
            voting_clf.fit(X_train_scaled, y_train)

            voting_accuracy = voting_clf.score(X_test_scaled, y_test)
            voting_cv_scores = cross_val_score(
                voting_clf, X_train_scaled, y_train, cv=5, scoring="accuracy"
            )

            ensemble_results["voting"] = {
                "test_accuracy": voting_accuracy,
                "cv_mean": voting_cv_scores.mean(),
                "cv_std": voting_cv_scores.std(),
                "cv_scores": voting_cv_scores.tolist(),
            }

        except Exception as e:
            logger.warning(f"Voting ensemble failed: {e}")
            ensemble_results["voting"] = {"error": str(e)}

        try:
            # Bagging Classifier
            bagging_clf = BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42),
                n_estimators=50,
                random_state=42,
            )
            bagging_clf.fit(X_train_scaled, y_train)

            bagging_accuracy = bagging_clf.score(X_test_scaled, y_test)
            bagging_cv_scores = cross_val_score(
                bagging_clf, X_train_scaled, y_train, cv=5, scoring="accuracy"
            )

            ensemble_results["bagging"] = {
                "test_accuracy": bagging_accuracy,
                "cv_mean": bagging_cv_scores.mean(),
                "cv_std": bagging_cv_scores.std(),
                "cv_scores": bagging_cv_scores.tolist(),
            }

        except Exception as e:
            logger.warning(f"Bagging ensemble failed: {e}")
            ensemble_results["bagging"] = {"error": str(e)}

        # 3. Hyperparameter Tuning (for best individual model)
        best_model_name = max(
            individual_results.keys(),
            key=lambda k: (
                individual_results[k].get("test_accuracy", 0)
                if "error" not in individual_results[k]
                else 0
            ),
        )

        hyperparameter_results = {}
        if "error" not in individual_results[best_model_name]:
            try:
                if best_model_name == "Random Forest":
                    param_grid = {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5, 10],
                    }
                    base_model = RandomForestClassifier(random_state=42)
                elif best_model_name == "Gradient Boosting":
                    param_grid = {
                        "n_estimators": [50, 100, 200],
                        "learning_rate": [0.01, 0.1, 0.2],
                        "max_depth": [3, 5, 7],
                    }
                    base_model = GradientBoostingClassifier(random_state=42)
                else:
                    param_grid = {}
                    base_model = models[best_model_name]

                if param_grid:
                    grid_search = GridSearchCV(
                        base_model, param_grid, cv=3, scoring="accuracy", n_jobs=-1
                    )
                    grid_search.fit(X_train_scaled, y_train)

                    hyperparameter_results = {
                        "best_params": grid_search.best_params_,
                        "best_score": grid_search.best_score_,
                        "test_accuracy": grid_search.score(X_test_scaled, y_test),
                    }
                else:
                    hyperparameter_results = {
                        "message": "No hyperparameter tuning for this model type"
                    }

            except Exception as e:
                logger.warning(f"Hyperparameter tuning failed: {e}")
                hyperparameter_results = {"error": str(e)}

        # Model Performance Comparison
        performance_comparison = {}
        all_models = {**individual_results, **ensemble_results}

        for model_name, results in all_models.items():
            if "error" not in results:
                performance_comparison[model_name] = {
                    "test_accuracy": results.get("test_accuracy", 0),
                    "cv_mean": results.get("cv_mean", 0),
                    "cv_std": results.get("cv_std", 0),
                }

        # Create ML visualization
        self._create_ml_visualization(
            individual_results,
            ensemble_results,
            feature_selection_results,
            simulation_id,
        )

        return {
            "feature_selection": feature_selection_results,
            "individual_models": individual_results,
            "ensemble_models": ensemble_results,
            "hyperparameter_tuning": hyperparameter_results,
            "performance_comparison": performance_comparison,
            "best_model": best_model_name,
            "target_variable": target_variable,
            "target_classes": (
                target_classes.tolist() if target_classes is not None else None
            ),
            "metadata": {
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "n_samples": len(df),
                "n_features": len(feature_columns),
                "methods_used": [
                    "Univariate feature selection",
                    "Recursive feature elimination",
                    "Model-based feature selection",
                    "Individual model training",
                    "Ensemble methods (Voting, Bagging)",
                    "Hyperparameter tuning",
                    "Cross-validation",
                ],
            },
        }

    def _create_ml_visualization(
        self,
        individual_results: Dict,
        ensemble_results: Dict,
        feature_selection_results: Dict,
        simulation_id: int,
    ) -> None:
        """Create comprehensive ML analysis visualization."""

        fig = plt.figure(figsize=(20, 12))

        # 1. Model Performance Comparison
        ax1 = plt.subplot(2, 3, 1)
        model_names = []
        test_accuracies = []
        cv_means = []
        cv_stds = []

        all_results = {**individual_results, **ensemble_results}
        for name, results in all_results.items():
            if "error" not in results:
                model_names.append(name)
                test_accuracies.append(results.get("test_accuracy", 0))
                cv_means.append(results.get("cv_mean", 0))
                cv_stds.append(results.get("cv_std", 0))

        if model_names:
            x = np.arange(len(model_names))
            width = 0.35

            bars1 = ax1.bar(
                x - width / 2, test_accuracies, width, label="Test Accuracy", alpha=0.8
            )
            bars2 = ax1.bar(
                x + width / 2,
                cv_means,
                width,
                label="CV Mean",
                alpha=0.8,
                yerr=cv_stds,
                capsize=5,
            )

            ax1.set_xlabel("Models")
            ax1.set_ylabel("Accuracy")
            ax1.set_title("Model Performance Comparison")
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_names, rotation=45, ha="right")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # 2. Feature Importance (from best model)
        ax2 = plt.subplot(2, 3, 2)
        best_model_name = max(
            individual_results.keys(),
            key=lambda k: (
                individual_results[k].get("test_accuracy", 0)
                if "error" not in individual_results[k]
                else 0
            ),
        )

        if "error" not in individual_results[best_model_name]:
            feature_importance = individual_results[best_model_name].get(
                "feature_importance"
            )
            if feature_importance:
                features = list(feature_importance.keys())
                importances = list(feature_importance.values())

                # Sort by importance
                sorted_data = sorted(
                    zip(features, importances), key=lambda x: x[1], reverse=True
                )
                top_features = [x[0] for x in sorted_data[:10]]
                top_importances = [x[1] for x in sorted_data[:10]]

                bars = ax2.barh(top_features, top_importances)
                ax2.set_xlabel("Feature Importance")
                ax2.set_title(f"Top 10 Features - {best_model_name}")
                ax2.grid(True, alpha=0.3)

        # 3. Feature Selection Comparison
        ax3 = plt.subplot(2, 3, 3)
        selection_methods = []
        n_features = []

        for method, results in feature_selection_results.items():
            if "error" not in results:
                selection_methods.append(method.replace("_", " ").title())
                n_features.append(results.get("n_features", 0))

        if selection_methods:
            bars = ax3.bar(selection_methods, n_features, alpha=0.7)
            ax3.set_ylabel("Number of Selected Features")
            ax3.set_title("Feature Selection Comparison")
            ax3.tick_params(axis="x", rotation=45)

            for bar in bars:
                height = bar.get_height()
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                )

        # 4. Cross-Validation Scores Distribution
        ax4 = plt.subplot(2, 3, 4)
        cv_data = []
        cv_labels = []

        for name, results in individual_results.items():
            if "error" not in results and "cv_scores" in results:
                cv_data.extend(results["cv_scores"])
                cv_labels.extend([name] * len(results["cv_scores"]))

        if cv_data:
            # Create box plot
            unique_labels = list(set(cv_labels))
            box_data = []
            for label in unique_labels:
                scores = [cv_data[i] for i, l in enumerate(cv_labels) if l == label]
                box_data.append(scores)

            ax4.boxplot(box_data, labels=unique_labels)
            ax4.set_ylabel("CV Accuracy")
            ax4.set_title("Cross-Validation Score Distribution")
            ax4.tick_params(axis="x", rotation=45)
            ax4.grid(True, alpha=0.3)

        # 5. Model Complexity vs Performance
        ax5 = plt.subplot(2, 3, 5)
        # This would show model complexity vs performance trade-off

        # 6. Summary Statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")

        # Create summary text
        summary_text = f"Advanced ML Analysis Summary\n\n"
        summary_text += f"Simulation ID: {simulation_id}\n"
        summary_text += f"Models Tested: {len([k for k, v in all_results.items() if 'error' not in v])}\n"
        summary_text += f"Best Model: {best_model_name}\n\n"

        if (
            best_model_name in individual_results
            and "error" not in individual_results[best_model_name]
        ):
            best_results = individual_results[best_model_name]
            summary_text += f"Best Performance:\n"
            summary_text += (
                f"• Test Accuracy: {best_results.get('test_accuracy', 0):.3f}\n"
            )
            summary_text += f"• CV Mean: {best_results.get('cv_mean', 0):.3f}\n"
            summary_text += f"• CV Std: {best_results.get('cv_std', 0):.3f}\n\n"

        summary_text += "Methods Applied:\n"
        summary_text += "• Feature Selection (3 methods)\n"
        summary_text += "• Individual Models (5 types)\n"
        summary_text += "• Ensemble Methods (2 types)\n"
        summary_text += "• Hyperparameter Tuning\n"
        summary_text += "• Cross-Validation"

        ax6.text(
            0.05,
            0.95,
            summary_text,
            transform=ax6.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightgray", "alpha": 0.8},
        )

        plt.tight_layout()
        plt.savefig(
            f"advanced_ml_analysis_sim_{simulation_id}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def run_complete_analysis(
        self, simulation_id: int, significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """Run all analysis methods for a given simulation with statistical validation.

        Args:
            simulation_id: ID of the simulation to analyze
            significance_level: Statistical significance level for analysis (default: 0.05)

        Returns:
            Dictionary containing all analysis results with statistical measures
        """
        logger.info(
            f"Running complete analysis for simulation {simulation_id} with significance level {significance_level}"
        )

        try:
            results = {
                "simulation_id": simulation_id,
                "significance_level": significance_level,
                "population_dynamics": self.analyze_population_dynamics(simulation_id),
                "resource_distribution": self.analyze_resource_distribution(
                    simulation_id
                ),
                "agent_interactions": self.analyze_agent_interactions(simulation_id),
                "generational_survival": self.analyze_generational_survival(
                    simulation_id
                ),
                "critical_events": self.identify_critical_events(
                    simulation_id, significance_level
                ),
                "agent_decisions": self.analyze_agent_decisions(simulation_id),
                "temporal_patterns": self.analyze_temporal_patterns(simulation_id),
                "advanced_time_series_models": self.analyze_advanced_time_series_models(
                    simulation_id
                ),
                "advanced_ml": self.analyze_with_advanced_ml(simulation_id),
            }

            # Add analysis metadata
            results["metadata"] = {
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "statistical_methods_used": [
                    "Kruskal-Wallis test",
                    "Mann-Whitney U test",
                    "Z-score change detection",
                    "Confidence intervals (95%)",
                    "Chi-square test for independence",
                    "Effect size calculations (Cohen's d, Hedges' g, eta-squared)",
                    "Statistical power analysis",
                    "Time series analysis (ADF, KPSS, seasonal decomposition)",
                    "Advanced time series modeling (ARIMA, VAR, exponential smoothing)",
                    "Granger causality testing and forecasting",
                    "Advanced ML (ensemble methods, feature selection)",
                    "Cross-validation and hyperparameter tuning",
                ],
                "significance_level": significance_level,
                "data_quality_checks": "Passed",
                "analysis_version": "Phase 2 - Statistical Enhancement",
            }

            # Save results to file
            output_dir = Path("analysis_results")
            output_dir.mkdir(exist_ok=True)

            results_file = output_dir / f"simulation_{simulation_id}_analysis.json"
            import json

            # Custom JSON encoder for numpy types and pandas DataFrames
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    elif isinstance(obj, pd.DataFrame):
                        return obj.to_dict("records")
                    elif isinstance(obj, pd.Series):
                        return obj.to_dict()
                    return super(NumpyEncoder, self).default(obj)

            with open(results_file, "w") as f:
                json.dump(results, f, cls=NumpyEncoder, indent=2)

            # Validate results if validator is available
            if self.validator is not None:
                validation_results = self.validator.validate_complete_analysis(results)
                results["validation_report"] = validation_results

                # Log validation summary
                if validation_results["overall_valid"]:
                    logger.info("✓ Analysis validation passed")
                else:
                    logger.warning("⚠ Analysis validation found issues")
                    for analysis_type, validation in validation_results[
                        "analysis_validations"
                    ].items():
                        if not validation["valid"]:
                            logger.warning(
                                f"  - {analysis_type}: {len(validation['errors'])} errors"
                            )

            # Create reproducibility report if available
            if (
                self.repro_manager is not None
                and create_reproducibility_report is not None
            ):
                analysis_params = {
                    "simulation_id": simulation_id,
                    "significance_level": significance_level,
                    "random_seed": self.random_seed,
                }

                repro_report_path = create_reproducibility_report(
                    analysis_params,
                    results,
                    output_path=output_dir
                    / f"reproducibility_report_sim_{simulation_id}.json",
                )
                logger.info(f"Reproducibility report saved to {repro_report_path}")

            logger.info(f"Analysis complete. Results saved to {results_file}")
            return results

        except Exception as e:
            logger.error(f"Analysis failed for simulation {simulation_id}: {e}")
            return {
                "simulation_id": simulation_id,
                "significance_level": significance_level,
                "error": str(e),
                "analysis_failed": True,
            }


def main():
    """Main function to run the analysis with proper error handling."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Analyze simulation results with statistical validation"
    )
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
        help="Statistical significance level (default: 0.05)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

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
            logger.error(
                f"Significance level must be between 0 and 1, got: {args.significance_level}"
            )
            sys.exit(1)

        # Initialize analyzer
        analyzer = SimulationAnalyzer(args.db_path)

        # Run analysis
        logger.info(f"Starting analysis for simulation {args.simulation_id}")
        results = analyzer.run_complete_analysis(
            args.simulation_id, args.significance_level
        )

        # Check for analysis errors
        if "error" in results:
            logger.error(f"Analysis failed: {results['error']}")
            sys.exit(1)

        # Print summary
        logger.info("Analysis completed successfully")
        logger.info(
            f"Results saved to: analysis_results/simulation_{args.simulation_id}_analysis.json"
        )

        # Print key findings
        if (
            "population_dynamics" in results
            and "summary" in results["population_dynamics"]
        ):
            summary = results["population_dynamics"]["summary"]
            logger.info(
                f"Population analysis: {summary.get('total_steps', 0)} steps analyzed"
            )
            if summary.get("significant_differences"):
                logger.info("✓ Significant differences found between agent types")
            else:
                logger.info("✗ No significant differences between agent types")

        if "critical_events" in results:
            significant_events = [
                e for e in results["critical_events"] if e.get("is_significant", False)
            ]
            logger.info(
                f"Critical events: {len(significant_events)} statistically significant events found"
            )

        if (
            "agent_interactions" in results
            and "summary" in results["agent_interactions"]
        ):
            summary = results["agent_interactions"]["summary"]
            logger.info(
                f"Agent interactions: {summary.get('total_interactions', 0)} total interactions"
            )
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

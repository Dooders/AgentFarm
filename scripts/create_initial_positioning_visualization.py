#!/usr/bin/env python3
"""
create_initial_positioning_visualization.py

This script creates visualizations that illustrate the relationship between
initial agent positioning relative to resources and dominance outcomes.

The analysis focuses only on the first agent of each type (SystemAgent,
IndependentAgent, ControlAgent) at the start of the simulation, not all agents.

It generates:
1. A correlation heatmap showing relationships between positioning metrics and dominance
2. A scatter plot showing how resource proximity relates to dominance duration
3. A comparative visualization of initial positions for different dominance outcomes
4. A time series analysis showing how initial advantages compound over time

Usage:
    python create_initial_positioning_visualization.py
"""

import glob
import json
import logging
import os
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import analysis configuration
from analysis_config import OUTPUT_PATH, setup_logging
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import database models
from farm.database.models import (
    AgentModel,
    AgentStateModel,
    Base,
    ResourceModel,
    SimulationStepModel,
)

# Define the specific experiment data path
EXPERIMENT_NAME = "one_of_a_kind_20250310_123136"
DATA_PATH = os.path.join(
    "results", "one_of_a_kind_500x3000", "experiments", "data", EXPERIMENT_NAME
)

# Create output directory
output_dir = os.path.join(OUTPUT_PATH, "initial_positioning")
os.makedirs(output_dir, exist_ok=True)

# Set up logging will be done in the main function


def get_initial_positions(db_path):
    """
    Extract initial positions of the first 3 agents (one of each type) and resources from the simulation database.
    """
    conn = sqlite3.connect(db_path)

    # Get initial agents (birth_time = 0)
    # Modified to ensure we only get the first agent of each type
    agents_query = """
    SELECT a.*
    FROM (
        SELECT agent_id, agent_type, position_x, position_y, initial_resources,
               ROW_NUMBER() OVER (PARTITION BY agent_type ORDER BY agent_id) as rn
        FROM agents
        WHERE birth_time = 0
    ) a
    WHERE a.rn = 1
    """
    agents = pd.read_sql_query(agents_query, conn)

    # Get initial resources (step_number = 0)
    resources_query = """
    SELECT resource_id, position_x, position_y, amount
    FROM resource_states
    WHERE step_number = 0
    """
    resources = pd.read_sql_query(resources_query, conn)

    conn.close()
    return agents, resources


def calculate_positioning_metrics(agents_df, resources_df, gathering_range=30):
    """
    Calculate positioning metrics for each agent type.
    Only considers the first agent of each type from the start of the simulation.
    """
    metrics = {}

    # Ensure we have at most one agent of each type
    unique_agents = agents_df.drop_duplicates("agent_type")

    for _, agent in unique_agents.iterrows():
        agent_type = agent["agent_type"]
        agent_pos = (agent["position_x"], agent["position_y"])

        # Calculate distances to all resources
        distances = []
        for _, resource in resources_df.iterrows():
            resource_pos = (resource["position_x"], resource["position_y"])
            distance = euclidean(agent_pos, resource_pos)
            distances.append((distance, resource["amount"]))

        # Calculate metrics
        if distances:
            # Nearest resource distance
            nearest_dist = min(distances, key=lambda x: x[0])[0]

            # Average distance to all resources
            avg_dist = sum(d[0] for d in distances) / len(distances)

            # Weighted distance (by resource amount)
            weighted_distances = [
                dist * (1 / (amount + 1)) for dist, amount in distances
            ]
            weighted_dist = sum(weighted_distances) / len(weighted_distances)

            # Resources within gathering range
            resources_in_range = sum(
                1 for dist, _ in distances if dist <= gathering_range
            )

            # Resource amount within gathering range
            resource_amount_in_range = sum(
                amount for dist, amount in distances if dist <= gathering_range
            )

            metrics[agent_type] = {
                "nearest_resource_dist": nearest_dist,
                "avg_resource_dist": avg_dist,
                "weighted_resource_dist": weighted_dist,
                "resources_in_range": resources_in_range,
                "resource_amount_in_range": resource_amount_in_range,
            }

    # Calculate relative advantages
    agent_types = list(metrics.keys())
    for i, type1 in enumerate(agent_types):
        for type2 in agent_types[i + 1 :]:
            # Nearest resource advantage
            metrics[f"{type1}_vs_{type2}_nearest_advantage"] = (
                metrics[type2]["nearest_resource_dist"]
                - metrics[type1]["nearest_resource_dist"]
            )

            # Resources in range advantage
            metrics[f"{type1}_vs_{type2}_resources_advantage"] = (
                metrics[type1]["resources_in_range"]
                - metrics[type2]["resources_in_range"]
            )

            # Resource amount advantage
            metrics[f"{type1}_vs_{type2}_amount_advantage"] = (
                metrics[type1]["resource_amount_in_range"]
                - metrics[type2]["resource_amount_in_range"]
            )

    return metrics


def get_dominance_data(db_path):
    """
    Extract dominance data from the simulation database.

    Calculates multiple dominance measures including:
    1. Population dominance (which agent type has the most agents at the end)
    2. Dominance duration (how many steps each agent type was dominant)
    3. Comprehensive dominance (a weighted combination of 5 metrics)
    """
    conn = sqlite3.connect(db_path)

    # Get population data for each step
    population_query = """
    SELECT s.step_number, 
           SUM(CASE WHEN a.agent_type = 'SystemAgent' THEN 1 ELSE 0 END) as system_agents,
           SUM(CASE WHEN a.agent_type = 'IndependentAgent' THEN 1 ELSE 0 END) as independent_agents,
           SUM(CASE WHEN a.agent_type = 'ControlAgent' THEN 1 ELSE 0 END) as control_agents
    FROM agent_states s
    JOIN agents a ON s.agent_id = a.agent_id
    GROUP BY s.step_number
    ORDER BY s.step_number
    """
    population_df = pd.read_sql_query(population_query, conn)

    conn.close()

    # Determine dominant agent type at each step
    population_df["dominant_type"] = population_df[
        ["system_agents", "independent_agents", "control_agents"]
    ].idxmax(axis=1)
    population_df["dominant_type"] = population_df["dominant_type"].map(
        {
            "system_agents": "SystemAgent",
            "independent_agents": "IndependentAgent",
            "control_agents": "ControlAgent",
        }
    )

    # Calculate dominance duration for each agent type
    dominance_counts = population_df["dominant_type"].value_counts()
    dominance_duration = {
        agent_type: dominance_counts.get(agent_type, 0)
        for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]
    }

    # Determine final dominant type (population dominance)
    final_step = population_df["step_number"].max()
    final_row = population_df[population_df["step_number"] == final_step].iloc[0]
    final_dominant = final_row["dominant_type"]

    # Calculate dominance switches
    switches = 0
    prev_dominant = None
    for dominant in population_df["dominant_type"]:
        if prev_dominant is not None and dominant != prev_dominant:
            switches += 1
        prev_dominant = dominant

    # Calculate comprehensive dominance components
    # 1. Area Under the Curve (AUC) - 20% weight
    auc = {
        "SystemAgent": population_df["system_agents"].sum(),
        "IndependentAgent": population_df["independent_agents"].sum(),
        "ControlAgent": population_df["control_agents"].sum(),
    }

    total_auc = sum(auc.values())
    normalized_auc = {
        agent_type: (count / total_auc if total_auc > 0 else 0)
        for agent_type, count in auc.items()
    }

    # 2. Recency-weighted AUC - 30% weight
    # Weight steps by their recency (later steps have more weight)
    steps = population_df["step_number"].to_numpy()
    max_step = steps.max() if len(steps) > 0 else 1
    weights = (steps / max_step) + 0.5  # Linear weights from 0.5 to 1.5

    recency_weighted_auc = {
        "SystemAgent": np.sum(population_df["system_agents"].to_numpy() * weights),
        "IndependentAgent": np.sum(
            population_df["independent_agents"].to_numpy() * weights
        ),
        "ControlAgent": np.sum(population_df["control_agents"].to_numpy() * weights),
    }

    total_recency_weighted_auc = sum(recency_weighted_auc.values())
    normalized_recency_weighted_auc = {
        agent_type: (
            count / total_recency_weighted_auc if total_recency_weighted_auc > 0 else 0
        )
        for agent_type, count in recency_weighted_auc.items()
    }

    # 3. Dominance Duration - 20% weight
    total_steps = len(population_df)
    normalized_dominance_duration = {
        agent_type: (duration / total_steps if total_steps > 0 else 0)
        for agent_type, duration in dominance_duration.items()
    }

    # 4. Growth Trend - 10% weight
    # Measure trend in the latter half of the simulation
    midpoint = total_steps // 2
    if midpoint > 0 and total_steps > midpoint:
        first_half = population_df.iloc[:midpoint]
        second_half = population_df.iloc[midpoint:]

        growth_trend = {}
        for agent_type, column in zip(
            ["SystemAgent", "IndependentAgent", "ControlAgent"],
            ["system_agents", "independent_agents", "control_agents"],
        ):
            first_half_avg = first_half[column].mean()
            second_half_avg = second_half[column].mean()

            # Calculate growth rate
            if first_half_avg > 0:
                growth_rate = (second_half_avg - first_half_avg) / first_half_avg
            elif second_half_avg > 0:
                growth_rate = 1.0  # If starting from 0, any growth is positive
            else:
                growth_rate = 0.0

            growth_trend[agent_type] = max(
                0, growth_rate
            )  # Only consider positive growth
    else:
        # If not enough data, assign equal growth trend
        growth_trend = {
            agent_type: 0
            for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]
        }

    # Normalize growth trend
    max_growth = max(growth_trend.values()) if growth_trend.values() else 1
    normalized_growth_trend = {
        agent_type: (trend / max_growth if max_growth > 0 else 0)
        for agent_type, trend in growth_trend.items()
    }

    # 5. Final Population Ratio - 20% weight
    final_total = (
        final_row["system_agents"]
        + final_row["independent_agents"]
        + final_row["control_agents"]
    )
    final_population_ratio = {
        "SystemAgent": (
            final_row["system_agents"] / final_total if final_total > 0 else 0
        ),
        "IndependentAgent": (
            final_row["independent_agents"] / final_total if final_total > 0 else 0
        ),
        "ControlAgent": (
            final_row["control_agents"] / final_total if final_total > 0 else 0
        ),
    }

    # Calculate comprehensive dominance score (weighted sum of all components)
    comprehensive_score = {}
    for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
        score = (
            0.20 * normalized_auc[agent_type]
            + 0.30 * normalized_recency_weighted_auc[agent_type]
            + 0.20 * normalized_dominance_duration[agent_type]
            + 0.10 * normalized_growth_trend[agent_type]
            + 0.20 * final_population_ratio[agent_type]
        )
        comprehensive_score[agent_type] = score

    # Determine comprehensive dominant type
    comprehensive_dominant = max(comprehensive_score.items(), key=lambda x: x[1])[0]

    # Store component scores for analysis
    component_scores = {
        agent_type: {
            "auc_score": normalized_auc[agent_type],
            "recency_weighted_auc_score": normalized_recency_weighted_auc[agent_type],
            "dominance_duration_score": normalized_dominance_duration[agent_type],
            "growth_trend_score": normalized_growth_trend[agent_type],
            "final_population_ratio_score": final_population_ratio[agent_type],
            "comprehensive_score": comprehensive_score[agent_type],
        }
        for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]
    }

    return {
        "dominance_duration": dominance_duration,
        "final_dominant": final_dominant,
        "dominance_switches": switches,
        "population_df": population_df,
        "comprehensive_dominant": comprehensive_dominant,
        "comprehensive_score": comprehensive_score,
        "component_scores": component_scores,
    }


def create_correlation_heatmap(data_df, output_path):
    """
    Create a heatmap showing correlations between positioning metrics and comprehensive dominance outcomes.
    """
    # Select positioning and dominance columns
    positioning_cols = [
        col
        for col in data_df.columns
        if any(
            x in col
            for x in [
                "resource_dist",
                "resources_in_range",
                "resource_amount",
                "advantage",
            ]
        )
    ]

    # Use comprehensive dominance metrics instead of population dominance
    dominance_cols = [
        "system_comprehensive_score",
        "independent_comprehensive_score",
        "control_comprehensive_score",
        "dominance_switches",
        "is_system_comprehensive_dominant",
        "is_independent_comprehensive_dominant",
        "is_control_comprehensive_dominant",
    ]

    # Calculate correlation matrix
    corr_df = data_df[positioning_cols + dominance_cols].corr()

    # Extract correlations between positioning metrics and dominance outcomes
    corr_subset = corr_df.loc[positioning_cols, dominance_cols]

    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_subset, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title(
        "Correlation between Initial Positioning and Comprehensive Dominance Outcomes",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "positioning_dominance_correlation.png"), dpi=150
    )
    plt.close()


def create_scatter_plot(data_df, output_path):
    """
    Create scatter plots showing the relationship between resource proximity and comprehensive dominance score.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
    colors = {
        "SystemAgent": "blue",
        "IndependentAgent": "green",
        "ControlAgent": "orange",
    }

    for i, agent_type in enumerate(agent_types):
        # Convert agent type to lowercase for column name matching
        short_name = agent_type.replace("Agent", "").lower()
        column_prefix = agent_type.lower()

        # Plot nearest resource distance vs comprehensive score (instead of dominance duration)
        axes[i].scatter(
            data_df[f"{column_prefix}_nearest_resource_dist"],
            data_df[f"{short_name}_comprehensive_score"],
            alpha=0.7,
            c=colors[agent_type],
            s=50,
        )

        # Add trend line
        x = data_df[f"{column_prefix}_nearest_resource_dist"]
        y = data_df[f"{short_name}_comprehensive_score"]

        # Count the number of valid data points (non-NaN values)
        n_simulations = x.notna().sum()

        # Only add trend line if we have enough data points
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axes[i].plot(x, p(x), "r--", alpha=0.8)

            # Calculate correlation
            corr = (
                data_df[
                    [
                        f"{column_prefix}_nearest_resource_dist",
                        f"{short_name}_comprehensive_score",
                    ]
                ]
                .corr()
                .iloc[0, 1]
            )
            corr_text = f"r={corr:.2f}"
        else:
            corr_text = "insufficient data"

        axes[i].set_title(f"{agent_type} ({corr_text}, n={n_simulations})")
        axes[i].set_xlabel("Nearest Resource Distance")
        axes[i].set_ylabel("Comprehensive Dominance Score")
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(
        "Relationship between Resource Proximity and Comprehensive Dominance",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "proximity_vs_duration_scatter.png"), dpi=150)
    plt.close()


def create_comparative_visualization(data_df, experiment_path, output_path):
    """
    Create a visualization comparing initial positions for different comprehensive dominance outcomes.
    Only visualizes the first agent of each type from the start of the simulation.
    """
    # Select a few representative simulations for each comprehensive dominant type
    examples = {}
    for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
        short_name = agent_type.replace("Agent", "").lower()
        column_prefix = agent_type.lower()
        # Use comprehensive dominance instead of population dominance
        col = f"is_{short_name}_comprehensive_dominant"
        dominant_sims = data_df[data_df[col] == 1].sort_values(
            by=f"{column_prefix}_resources_in_range", ascending=False
        )
        if len(dominant_sims) >= 2:
            examples[agent_type] = dominant_sims.iloc[:2]["iteration"].tolist()

    # Create a grid of visualizations
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))

    row = 0
    for agent_type, iterations in examples.items():
        for col, iteration in enumerate(iterations[:2]):
            ax = axes[row, col]

            # Get database path
            db_path = os.path.join(
                experiment_path, f"iteration_{iteration}", "simulation.db"
            )

            if os.path.exists(db_path):
                # Get initial positions
                agents, resources = get_initial_positions(db_path)

                # Plot resources
                for _, resource in resources.iterrows():
                    circle = Circle(
                        (resource["position_x"], resource["position_y"]),
                        resource["amount"] / 5,
                        color="green",
                        alpha=0.5,
                    )
                    ax.add_patch(circle)

                # Plot agents
                agent_colors = {
                    "SystemAgent": "blue",
                    "IndependentAgent": "red",
                    "ControlAgent": "orange",
                }

                for _, agent in agents.iterrows():
                    # Plot agent
                    ax.scatter(
                        agent["position_x"],
                        agent["position_y"],
                        color=agent_colors[agent["agent_type"]],
                        s=100,
                        zorder=10,
                        edgecolor="black",
                    )

                    # Plot gathering range
                    circle = Circle(
                        (agent["position_x"], agent["position_y"]),
                        30,  # gathering range
                        fill=False,
                        linestyle="--",
                        color=agent_colors[agent["agent_type"]],
                        alpha=0.5,
                    )
                    ax.add_patch(circle)

                # Set plot limits
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)

                # Add title
                ax.set_title(
                    f"Iteration {iteration}: {agent_type} Comprehensive Dominant"
                )

        row += 1

    plt.suptitle(
        "Initial Positions for Different Comprehensive Dominance Outcomes", fontsize=16
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "initial_positions_comparison.png"), dpi=150)
    plt.close()


def create_time_series_visualization(data_df, experiment_path, output_path):
    """
    Create a visualization showing how initial advantages compound over time leading to comprehensive dominance.
    Only considers the first agent of each type from the start of the simulation.
    """
    # Select a representative simulation with clear initial advantage and comprehensive dominance
    # Find a simulation where one agent type has significantly more resources in range
    # and achieved comprehensive dominance
    advantage_metrics = []

    for _, data in data_df.iterrows():
        system_resources = data.get("systemagent_resources_in_range", 0)
        independent_resources = data.get("independentagent_resources_in_range", 0)
        control_resources = data.get("controlagent_resources_in_range", 0)

        # Check which agent type achieved comprehensive dominance
        if data["is_system_comprehensive_dominant"] == 1:
            dominant_agent = "SystemAgent"
        elif data["is_independent_comprehensive_dominant"] == 1:
            dominant_agent = "IndependentAgent"
        elif data["is_control_comprehensive_dominant"] == 1:
            dominant_agent = "ControlAgent"
        else:
            continue  # Skip if no clear comprehensive dominance

        max_resources = max(system_resources, independent_resources, control_resources)
        if max_resources > 0:
            # Calculate the advantage ratio
            if max_resources == system_resources:
                advantage_agent = "SystemAgent"
                advantage_ratio = system_resources / (
                    max(independent_resources, control_resources, 1)
                )
            elif max_resources == independent_resources:
                advantage_agent = "IndependentAgent"
                advantage_ratio = independent_resources / (
                    max(system_resources, control_resources, 1)
                )
            else:
                advantage_agent = "ControlAgent"
                advantage_ratio = control_resources / (
                    max(system_resources, independent_resources, 1)
                )

            # Prioritize examples where the agent with resource advantage achieved comprehensive dominance
            if advantage_agent == dominant_agent:
                advantage_metrics.append(
                    (data["iteration"], advantage_agent, advantage_ratio, 2)
                )  # Higher priority
            else:
                advantage_metrics.append(
                    (data["iteration"], advantage_agent, advantage_ratio, 1)
                )  # Lower priority

    # Sort by priority first, then by advantage ratio
    advantage_metrics.sort(key=lambda x: (x[3], x[2]), reverse=True)
    if advantage_metrics:
        example_iteration, advantaged_agent, _, _ = advantage_metrics[0]

        # Get database path
        db_path = os.path.join(
            experiment_path, f"iteration_{example_iteration}", "simulation.db"
        )

        if os.path.exists(db_path):
            # Get population data
            conn = sqlite3.connect(db_path)
            population_query = """
            SELECT s.step_number, 
                   SUM(CASE WHEN a.agent_type = 'SystemAgent' THEN 1 ELSE 0 END) as system_agents,
                   SUM(CASE WHEN a.agent_type = 'IndependentAgent' THEN 1 ELSE 0 END) as independent_agents,
                   SUM(CASE WHEN a.agent_type = 'ControlAgent' THEN 1 ELSE 0 END) as control_agents
            FROM agent_states s
            JOIN agents a ON s.agent_id = a.agent_id
            GROUP BY s.step_number
            ORDER BY s.step_number
            """
            population_df = pd.read_sql_query(population_query, conn)
            conn.close()

            # Get initial positions and calculate advantages
            agents, resources = get_initial_positions(db_path)
            metrics = calculate_positioning_metrics(agents, resources)

            # Get comprehensive dominance data for this simulation
            dominance_data = get_dominance_data(db_path)
            comprehensive_scores = dominance_data["comprehensive_score"]
            comprehensive_dominant = dominance_data["comprehensive_dominant"]

            # Create the visualization
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot population time series
            ax.plot(
                population_df["step_number"],
                population_df["system_agents"],
                color="blue",
                label="System Agents",
            )
            ax.plot(
                population_df["step_number"],
                population_df["independent_agents"],
                color="red",
                label="Independent Agents",
            )
            ax.plot(
                population_df["step_number"],
                population_df["control_agents"],
                color="orange",
                label="Control Agents",
            )

            # Add vertical line at step 100 to mark early phase
            ax.axvline(x=100, color="gray", linestyle="--", alpha=0.7)
            ax.text(105, ax.get_ylim()[1] * 0.95, "Early Phase", fontsize=10)

            # Add annotations for initial advantages and comprehensive scores
            y_positions = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
            colors = {
                "SystemAgent": "blue",
                "IndependentAgent": "red",
                "ControlAgent": "orange",
            }

            # Add initial resource advantage information
            pos_idx = 0
            for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
                if agent_type in metrics:
                    agent_metrics = metrics[agent_type]
                    text = f"{agent_type}: {agent_metrics['resources_in_range']} resources in range ({agent_metrics['resource_amount_in_range']:.1f} total)"

                    # Highlight the advantaged agent
                    if agent_type == advantaged_agent:
                        text = f"★ {text} ★"

                    ax.text(
                        0.02,
                        y_positions[pos_idx],
                        text,
                        transform=ax.transAxes,
                        color=colors[agent_type],
                        fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
                    )
                    pos_idx += 1

            # Add comprehensive dominance information
            ax.text(
                0.02,
                y_positions[pos_idx],
                f"Comprehensive Dominant: {comprehensive_dominant}",
                transform=ax.transAxes,
                color=colors[comprehensive_dominant],
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
            )
            pos_idx += 1

            # Add comprehensive scores
            for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
                score = comprehensive_scores.get(agent_type, 0)
                ax.text(
                    0.02,
                    y_positions[pos_idx],
                    f"{agent_type} Comprehensive Score: {score:.3f}",
                    transform=ax.transAxes,
                    color=colors[agent_type],
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
                )
                pos_idx += 1

            # Set labels and title
            ax.set_xlabel("Simulation Step")
            ax.set_ylabel("Population Count")
            ax.set_title(
                f"Population Dynamics with Initial Resource Advantage (Iteration {example_iteration})"
            )
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

            # Save the figure
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_path, "advantage_compounding_time_series.png"),
                dpi=150,
            )
            plt.close()


def create_feature_importance_visualization(data_df, output_path):
    """
    Create a visualization showing the importance of positioning features for predicting comprehensive dominance.
    """
    # Prepare features and target
    positioning_cols = [
        col
        for col in data_df.columns
        if any(
            x in col
            for x in [
                "resource_dist",
                "resources_in_range",
                "resource_amount",
                "advantage",
            ]
        )
    ]

    # Prepare data for each dominance type, using comprehensive dominance
    for agent_type in ["system", "independent", "control"]:
        # Use comprehensive dominance instead of population dominance
        target_col = f"is_{agent_type}_comprehensive_dominant"

        if target_col in data_df.columns and len(positioning_cols) > 0:
            X = data_df[positioning_cols]
            y = data_df[target_col]

            # Handle categorical features with one-hot encoding
            X = pd.get_dummies(X, drop_first=True)

            # Train a Random Forest classifier
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X, y)

            # Get feature importances
            importances = clf.feature_importances_
            feature_names = X.columns

            # Sort features by importance
            indices = np.argsort(importances)[::-1]

            # Plot feature importances horizontally
            plt.figure(figsize=(12, 8))
            plt.title(
                f"Feature Importance for {agent_type.capitalize()} Comprehensive Dominance"
            )

            # Use only top 10 features for clarity
            top_n = min(10, len(indices))
            plt.barh(range(top_n), importances[indices[:top_n]], align="center")
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_path, f"{agent_type}_feature_importance.png"),
                dpi=150,
            )
            plt.close()


def create_comprehensive_dominance_scatter(data_df, output_path):
    """
    Create scatter plots showing the relationship between resource proximity and comprehensive dominance score.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
    colors = {
        "SystemAgent": "blue",
        "IndependentAgent": "green",
        "ControlAgent": "orange",
    }

    # For legend
    regular_points = []
    dominant_points = []

    for i, agent_type in enumerate(agent_types):
        # Convert agent type to lowercase for column name matching
        short_name = agent_type.replace("Agent", "").lower()
        column_prefix = agent_type.lower()

        # Plot nearest resource distance vs comprehensive score
        scatter = axes[i].scatter(
            data_df[f"{column_prefix}_nearest_resource_dist"],
            data_df[f"{short_name}_comprehensive_score"],
            alpha=0.7,
            c=colors[agent_type],
            s=50,
        )

        # Store for legend
        if i == 0:  # Only need to store once for the legend
            regular_points.append(scatter)

        # Highlight points where this agent type is the comprehensive dominant
        is_dominant = data_df[f"is_{short_name}_comprehensive_dominant"] == 1
        if any(is_dominant):
            dominant_scatter = axes[i].scatter(
                data_df.loc[is_dominant, f"{column_prefix}_nearest_resource_dist"],
                data_df.loc[is_dominant, f"{short_name}_comprehensive_score"],
                alpha=0.8,
                edgecolor="red",
                facecolor=colors[agent_type],
                s=80,
                marker="*",
            )

            # Store for legend
            if i == 0:  # Only need to store once for the legend
                dominant_points.append(dominant_scatter)

        # Add trend line
        x = data_df[f"{column_prefix}_nearest_resource_dist"]
        y = data_df[f"{short_name}_comprehensive_score"]

        # Count the number of valid data points (non-NaN values)
        n_simulations = x.notna().sum()

        # Only add trend line if we have enough data points
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axes[i].plot(x, p(x), "r--", alpha=0.8)

            # Calculate correlation
            corr = (
                data_df[
                    [
                        f"{column_prefix}_nearest_resource_dist",
                        f"{short_name}_comprehensive_score",
                    ]
                ]
                .corr()
                .iloc[0, 1]
            )
            corr_text = f"r={corr:.2f}"
        else:
            corr_text = "insufficient data"

        axes[i].set_title(f"{agent_type} ({corr_text}, n={n_simulations})")
        axes[i].set_xlabel("Nearest Resource Distance")
        axes[i].set_ylabel("Comprehensive Dominance Score")
        axes[i].grid(True, alpha=0.3)

    # Add a legend to explain the markers
    # Create a single legend for the figure
    if regular_points and dominant_points:
        # Create custom legend elements
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=10,
                label="All Simulations",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor="gray",
                markersize=15,
                label="Achieved Comprehensive Dominance",
            ),
            Line2D([0], [0], color="r", linestyle="--", alpha=0.8, label="Trend Line"),
        ]

        # Place legend at the top of the figure
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=3,
            bbox_to_anchor=(0.5, 0.98),
        )

    plt.suptitle(
        "Relationship between Resource Proximity and Comprehensive Dominance",
        fontsize=16,
        y=0.92,
    )
    plt.tight_layout(
        rect=(0, 0, 1, 0.92)
    )  # Adjust the layout to make room for the legend
    plt.savefig(
        os.path.join(output_path, "proximity_vs_comprehensive_scatter.png"), dpi=150
    )
    plt.close()


def create_component_breakdown_visualization(data_df, output_path):
    """
    Create a visualization showing how initial positioning affects each component
    of the comprehensive dominance score.
    """
    # Calculate correlations between initial positioning metrics and score components
    positioning_cols = [
        col
        for col in data_df.columns
        if any(
            x in col
            for x in [
                "resource_dist",
                "resources_in_range",
                "resource_amount",
                "advantage",
            ]
        )
    ]

    component_cols = []
    for agent_type in ["system", "independent", "control"]:
        for component in [
            "auc_score",
            "recency_weighted_auc_score",
            "dominance_duration_score",
            "growth_trend_score",
            "final_population_ratio_score",
        ]:
            component_cols.append(f"{agent_type}_{component}")

    # Create a figure with 3 rows (one per agent type) and 5 columns (one per component)
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    agent_types = ["system", "independent", "control"]
    agent_colors = {"system": "blue", "independent": "green", "control": "orange"}
    component_names = [
        "AUC",
        "Recency AUC",
        "Dominance Duration",
        "Growth Trend",
        "Final Ratio",
    ]

    # Define consistent metrics to use for each component column
    # This ensures all agent types use the same x-axis metric for each component
    consistent_metrics = {
        "auc_score": "weighted_resource_dist",
        "recency_weighted_auc_score": "weighted_resource_dist",
        "dominance_duration_score": "weighted_resource_dist",
        "growth_trend_score": "nearest_resource_dist",
        "final_population_ratio_score": "weighted_resource_dist",
    }

    for i, agent_type in enumerate(agent_types):
        for j, component in enumerate(
            [
                "auc_score",
                "recency_weighted_auc_score",
                "dominance_duration_score",
                "growth_trend_score",
                "final_population_ratio_score",
            ]
        ):
            ax = axes[i, j]

            # Use consistent metric for this component column
            metric_suffix = consistent_metrics[component]
            metric_col = f"{agent_type}agent_{metric_suffix}"

            # Find the component column
            component_col = f"{agent_type}_{component}"

            if component_col in data_df.columns and metric_col in data_df.columns:
                # Plot this relationship
                ax.scatter(
                    data_df[metric_col],
                    data_df[component_col],
                    alpha=0.7,
                    c=agent_colors[agent_type],
                    s=40,
                )

                # Add trend line
                x = data_df[metric_col].to_numpy()
                y = data_df[component_col].to_numpy()

                # Only add trend line if we have enough data
                if len(x) > 1:
                    # Calculate correlation
                    corr = data_df[[metric_col, component_col]].corr().iloc[0, 1]
                    if not np.isnan(corr):
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        ax.plot(x, p(x), "r--", alpha=0.8)

                        # Add correlation value
                        ax.text(
                            0.05,
                            0.95,
                            f"r={corr:.2f}",
                            transform=ax.transAxes,
                            fontsize=9,
                            va="top",
                        )

                # Format the axis label to make it more readable
                metric_label = metric_suffix.replace("_", " ")
                ax.set_xlabel(metric_label, fontsize=8)

                # Set consistent x-axis limits for each column to ensure comparability
                if j == 0:  # First column - set y-label
                    ax.set_ylabel(agent_type.capitalize(), fontsize=10)

            # Set title for each component
            if i == 0:
                ax.set_title(component_names[j], fontsize=10)

            ax.grid(True, alpha=0.3)

    # Set consistent x-axis limits for each column
    for j in range(5):  # For each component column
        x_min = float("inf")
        x_max = float("-inf")

        # Find min and max across all agent types for this component
        for i in range(3):  # For each agent type
            if axes[i, j].collections:  # If there's data plotted
                x_data = axes[i, j].collections[0].get_offsets()[:, 0]
                if len(x_data) > 0:
                    x_min = min(x_min, np.min(x_data))
                    x_max = max(x_max, np.max(x_data))

        # Apply consistent limits to all plots in this column
        if x_min != float("inf") and x_max != float("-inf"):
            for i in range(3):  # For each agent type
                axes[i, j].set_xlim(x_min, x_max)

    plt.suptitle(
        "Relationship between Initial Positioning and Comprehensive Score Components",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_path, "positioning_component_relationships.png"), dpi=150
    )
    plt.close()


def main():
    """
    Main function to create initial positioning visualizations.

    This analysis focuses only on the initial positioning of the first agent of each type
    (SystemAgent, IndependentAgent, ControlAgent) at the start of the simulation,
    not all agents with birth_time=0.
    """
    print(
        "Creating initial positioning visualizations for the first agent of each type..."
    )

    # Set up logging first
    log_file_path = setup_logging(output_dir)
    log_filename = os.path.basename(log_file_path)

    # Clear the output directory before running, but exclude the log file
    if os.path.exists(output_dir):
        print(f"Clearing output directory: {output_dir}")
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path) and file != log_filename:
                os.remove(file_path)
    else:
        os.makedirs(output_dir, exist_ok=True)

    # Create a new dataframe to store analysis results
    df = pd.DataFrame()

    # Find all simulation folders
    sim_folders = glob.glob(os.path.join(DATA_PATH, "iteration_*"))

    for folder in sim_folders:
        iteration = int(os.path.basename(folder).split("_")[1])
        db_path = os.path.join(folder, "simulation.db")

        if os.path.exists(db_path):
            print(f"Analyzing iteration {iteration}...")

            # Get initial positions
            agents, resources = get_initial_positions(db_path)

            # Calculate positioning metrics
            positioning_metrics = calculate_positioning_metrics(agents, resources)

            # Get dominance data
            dominance_data = get_dominance_data(db_path)

            # Combine data
            row_data = {
                "iteration": iteration,
                "dominance_switches": dominance_data["dominance_switches"],
                "is_system_dominant": (
                    1 if dominance_data["final_dominant"] == "SystemAgent" else 0
                ),
                "is_independent_dominant": (
                    1 if dominance_data["final_dominant"] == "IndependentAgent" else 0
                ),
                "is_control_dominant": (
                    1 if dominance_data["final_dominant"] == "ControlAgent" else 0
                ),
                "system_dominance_duration": dominance_data["dominance_duration"].get(
                    "SystemAgent", 0
                ),
                "independent_dominance_duration": dominance_data[
                    "dominance_duration"
                ].get("IndependentAgent", 0),
                "control_dominance_duration": dominance_data["dominance_duration"].get(
                    "ControlAgent", 0
                ),
                # Add comprehensive dominance metrics
                "is_system_comprehensive_dominant": (
                    1
                    if dominance_data["comprehensive_dominant"] == "SystemAgent"
                    else 0
                ),
                "is_independent_comprehensive_dominant": (
                    1
                    if dominance_data["comprehensive_dominant"] == "IndependentAgent"
                    else 0
                ),
                "is_control_comprehensive_dominant": (
                    1
                    if dominance_data["comprehensive_dominant"] == "ControlAgent"
                    else 0
                ),
                "system_comprehensive_score": dominance_data["comprehensive_score"].get(
                    "SystemAgent", 0
                ),
                "independent_comprehensive_score": dominance_data[
                    "comprehensive_score"
                ].get("IndependentAgent", 0),
                "control_comprehensive_score": dominance_data[
                    "comprehensive_score"
                ].get("ControlAgent", 0),
            }

            # Add positioning metrics
            for agent_type, metrics in positioning_metrics.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        key = f"{agent_type.lower()}_{metric}"
                        row_data[key] = value
                else:
                    row_data[agent_type] = metrics

            # Add component scores for each agent type
            for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
                short_name = agent_type.replace("Agent", "").lower()
                component_scores = dominance_data["component_scores"].get(
                    agent_type, {}
                )
                for component, score in component_scores.items():
                    row_data[f"{short_name}_{component}"] = score

            # Add to dataframe
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

    # Save analysis results for future use
    analysis_file = os.path.join(DATA_PATH, "simulation_analysis.csv")
    os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
    df.to_csv(analysis_file, index=False)

    # Create visualizations
    if not df.empty:
        create_correlation_heatmap(df, output_dir)
        create_scatter_plot(df, output_dir)
        create_comparative_visualization(df, DATA_PATH, output_dir)
        create_time_series_visualization(df, DATA_PATH, output_dir)
        create_feature_importance_visualization(df, output_dir)
        create_comprehensive_dominance_scatter(df, output_dir)
        create_component_breakdown_visualization(df, output_dir)

        print(f"Visualizations saved to {output_dir}")
        logging.info(f"Visualizations saved to {output_dir}")
        logging.info(f"Log file saved to: {log_file_path}")
    else:
        print("No data available for visualization")
        logging.warning("No data available for visualization")


if __name__ == "__main__":
    main()

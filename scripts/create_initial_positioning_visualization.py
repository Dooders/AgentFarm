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
import os
import sqlite3
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import analysis configuration
from analysis_config import OUTPUT_PATH, setup_logging

# Import database models
from farm.database.models import (
    AgentModel, 
    AgentStateModel, 
    ResourceModel, 
    SimulationStepModel,
    Base
)

# Define the specific experiment data path
EXPERIMENT_NAME = "one_of_a_kind_20250310_123136"
DATA_PATH = os.path.join("results", "one_of_a_kind_500x3000", "experiments", "data", EXPERIMENT_NAME)

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
    unique_agents = agents_df.drop_duplicates('agent_type')

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
            resources_in_range = sum(1 for dist, _ in distances if dist <= gathering_range)

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
        for type2 in agent_types[i + 1:]:
            # Nearest resource advantage
            metrics[f"{type1}_vs_{type2}_nearest_advantage"] = (
                metrics[type2]["nearest_resource_dist"] - metrics[type1]["nearest_resource_dist"]
            )

            # Resources in range advantage
            metrics[f"{type1}_vs_{type2}_resources_advantage"] = (
                metrics[type1]["resources_in_range"] - metrics[type2]["resources_in_range"]
            )

            # Resource amount advantage
            metrics[f"{type1}_vs_{type2}_amount_advantage"] = (
                metrics[type1]["resource_amount_in_range"] - metrics[type2]["resource_amount_in_range"]
            )

    return metrics


def get_dominance_data(db_path):
    """
    Extract dominance data from the simulation database.
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
    population_df["dominant_type"] = population_df[["system_agents", "independent_agents", "control_agents"]].idxmax(axis=1)
    population_df["dominant_type"] = population_df["dominant_type"].map({
        "system_agents": "SystemAgent",
        "independent_agents": "IndependentAgent",
        "control_agents": "ControlAgent"
    })

    # Calculate dominance duration for each agent type
    dominance_counts = population_df["dominant_type"].value_counts()
    dominance_duration = {
        agent_type: dominance_counts.get(agent_type, 0)
        for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]
    }

    # Determine final dominant type
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

    return {
        "dominance_duration": dominance_duration,
        "final_dominant": final_dominant,
        "dominance_switches": switches,
        "population_df": population_df
    }


def create_correlation_heatmap(data_df, output_path):
    """
    Create a heatmap showing correlations between positioning metrics and dominance outcomes.
    """
    # Select positioning and dominance columns
    positioning_cols = [col for col in data_df.columns if any(x in col for x in [
        "resource_dist", "resources_in_range", "resource_amount", "advantage"
    ])]
    
    dominance_cols = [
        "system_dominance_duration", "independent_dominance_duration", "control_dominance_duration",
        "dominance_switches", "is_system_dominant", "is_independent_dominant", "is_control_dominant"
    ]
    
    # Calculate correlation matrix
    corr_df = data_df[positioning_cols + dominance_cols].corr()
    
    # Extract correlations between positioning metrics and dominance outcomes
    corr_subset = corr_df.loc[positioning_cols, dominance_cols]
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_subset, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation between Initial Positioning and Dominance Outcomes", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "positioning_dominance_correlation.png"), dpi=150)
    plt.close()


def create_scatter_plot(data_df, output_path):
    """
    Create scatter plots showing the relationship between resource proximity and dominance duration.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
    colors = {"SystemAgent": "blue", "IndependentAgent": "green", "ControlAgent": "orange"}
    
    for i, agent_type in enumerate(agent_types):
        # Convert agent type to lowercase for column name matching
        column_prefix = agent_type.lower()
        
        # Plot nearest resource distance vs dominance duration
        axes[i].scatter(
            data_df[f"{column_prefix}_nearest_resource_dist"],
            data_df[f"{agent_type.replace('Agent', '').lower()}_dominance_duration"],
            alpha=0.7,
            c=colors[agent_type],
            s=50
        )
        
        # Add trend line
        x = data_df[f"{column_prefix}_nearest_resource_dist"]
        y = data_df[f"{agent_type.replace('Agent', '').lower()}_dominance_duration"]
        
        # Count the number of valid data points (non-NaN values)
        n_simulations = x.notna().sum()
        
        # Only add trend line if we have enough data points
        if len(x) > 1:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            axes[i].plot(x, p(x), "r--", alpha=0.8)
            
            # Calculate correlation
            corr = data_df[[f"{column_prefix}_nearest_resource_dist", 
                           f"{agent_type.replace('Agent', '').lower()}_dominance_duration"]].corr().iloc[0, 1]
            corr_text = f"r={corr:.2f}"
        else:
            corr_text = "insufficient data"
        
        axes[i].set_title(f"{agent_type} ({corr_text}, n={n_simulations})")
        axes[i].set_xlabel("Nearest Resource Distance")
        axes[i].set_ylabel("Dominance Duration (steps)")
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle("Relationship between Resource Proximity and Dominance Duration", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "proximity_vs_duration_scatter.png"), dpi=150)
    plt.close()


def create_comparative_visualization(data_df, experiment_path, output_path):
    """
    Create a visualization comparing initial positions for different dominance outcomes.
    Only visualizes the first agent of each type from the start of the simulation.
    """
    # Select a few representative simulations for each dominant type
    examples = {}
    for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
        col = f"is_{agent_type.replace('Agent', '').lower()}_dominant"
        column_prefix = agent_type.lower()
        dominant_sims = data_df[data_df[col] == 1].sort_values(
            by=f"{column_prefix}_resources_in_range", 
            ascending=False
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
            db_path = os.path.join(experiment_path, f"iteration_{iteration}", "simulation.db")
            
            if os.path.exists(db_path):
                # Get initial positions
                agents, resources = get_initial_positions(db_path)
                
                # Plot resources
                for _, resource in resources.iterrows():
                    circle = Circle(
                        (resource["position_x"], resource["position_y"]),
                        resource["amount"] / 5,
                        color="green",
                        alpha=0.5
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
                        edgecolor="black"
                    )
                    
                    # Plot gathering range
                    circle = Circle(
                        (agent["position_x"], agent["position_y"]),
                        30,  # gathering range
                        fill=False,
                        linestyle="--",
                        color=agent_colors[agent["agent_type"]],
                        alpha=0.5
                    )
                    ax.add_patch(circle)
                
                # Set plot limits
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                
                # Add title
                ax.set_title(f"Iteration {iteration}: {agent_type} Dominant")
            
        row += 1
    
    plt.suptitle("Initial Positions for Different Dominance Outcomes", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "initial_positions_comparison.png"), dpi=150)
    plt.close()


def create_time_series_visualization(data_df, experiment_path, output_path):
    """
    Create a visualization showing how initial advantages compound over time.
    Only considers the first agent of each type from the start of the simulation.
    """
    # Select a representative simulation with clear initial advantage
    # Find a simulation where one agent type has significantly more resources in range
    advantage_metrics = []
    for row in data_df.iterrows():
        data = row[1]
        system_resources = data.get("systemagent_resources_in_range", 0)
        independent_resources = data.get("independentagent_resources_in_range", 0)
        control_resources = data.get("controlagent_resources_in_range", 0)
        
        max_resources = max(system_resources, independent_resources, control_resources)
        if max_resources > 0:
            # Calculate the advantage ratio
            if max_resources == system_resources:
                advantage_agent = "SystemAgent"
                advantage_ratio = system_resources / (max(independent_resources, control_resources, 1))
            elif max_resources == independent_resources:
                advantage_agent = "IndependentAgent"
                advantage_ratio = independent_resources / (max(system_resources, control_resources, 1))
            else:
                advantage_agent = "ControlAgent"
                advantage_ratio = control_resources / (max(system_resources, independent_resources, 1))
            
            advantage_metrics.append((data["iteration"], advantage_agent, advantage_ratio))
    
    # Sort by advantage ratio and select the top example
    advantage_metrics.sort(key=lambda x: x[2], reverse=True)
    if advantage_metrics:
        example_iteration, advantaged_agent, _ = advantage_metrics[0]
        
        # Get database path
        db_path = os.path.join(experiment_path, f"iteration_{example_iteration}", "simulation.db")
        
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
            
            # Create the visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot population time series
            ax.plot(population_df["step_number"], population_df["system_agents"], color="blue", label="System Agents")
            ax.plot(population_df["step_number"], population_df["independent_agents"], color="red", label="Independent Agents")
            ax.plot(population_df["step_number"], population_df["control_agents"], color="orange", label="Control Agents")
            
            # Add vertical line at step 100 to mark early phase
            ax.axvline(x=100, color="gray", linestyle="--", alpha=0.7)
            ax.text(105, ax.get_ylim()[1] * 0.95, "Early Phase", fontsize=10)
            
            # Add annotations for initial advantages
            y_positions = [0.85, 0.80, 0.75]
            colors = {"SystemAgent": "blue", "IndependentAgent": "red", "ControlAgent": "orange"}
            
            for i, agent_type in enumerate(["SystemAgent", "IndependentAgent", "ControlAgent"]):
                if agent_type in metrics:
                    agent_metrics = metrics[agent_type]
                    text = f"{agent_type}: {agent_metrics['resources_in_range']} resources in range ({agent_metrics['resource_amount_in_range']:.1f} total)"
                    
                    # Highlight the advantaged agent
                    if agent_type == advantaged_agent:
                        text = f"★ {text} ★"
                    
                    ax.text(
                        0.02,
                        y_positions[i],
                        text,
                        transform=ax.transAxes,
                        color=colors[agent_type],
                        fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
                    )
            
            # Set labels and title
            ax.set_xlabel("Simulation Step")
            ax.set_ylabel("Population Count")
            ax.set_title(f"Population Dynamics with Initial Resource Advantage (Iteration {example_iteration})")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "advantage_compounding_time_series.png"), dpi=150)
            plt.close()


def create_feature_importance_visualization(data_df, output_path):
    """
    Create a visualization showing the importance of positioning features for predicting dominance.
    """
    # Prepare features and target
    positioning_cols = [col for col in data_df.columns if any(x in col for x in [
        "resource_dist", "resources_in_range", "resource_amount", "advantage"
    ])]
    
    # Prepare data for each dominance type
    for agent_type in ["system", "independent", "control"]:
        target_col = f"is_{agent_type}_dominant"
        
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
            plt.title(f"Feature Importance for {agent_type.capitalize()} Dominance")
            
            # Use only top 10 features for clarity
            top_n = min(10, len(indices))
            plt.barh(range(top_n), importances[indices[:top_n]], align="center")
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
            plt.xlabel("Importance")
            plt.ylabel("Features")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{agent_type}_feature_importance.png"), dpi=150)
            plt.close()


def main():
    """
    Main function to create initial positioning visualizations.
    
    This analysis focuses only on the initial positioning of the first agent of each type
    (SystemAgent, IndependentAgent, ControlAgent) at the start of the simulation,
    not all agents with birth_time=0.
    """
    print("Creating initial positioning visualizations for the first agent of each type...")
    
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
                "is_system_dominant": 1 if dominance_data["final_dominant"] == "SystemAgent" else 0,
                "is_independent_dominant": 1 if dominance_data["final_dominant"] == "IndependentAgent" else 0,
                "is_control_dominant": 1 if dominance_data["final_dominant"] == "ControlAgent" else 0,
                "system_dominance_duration": dominance_data["dominance_duration"].get("SystemAgent", 0),
                "independent_dominance_duration": dominance_data["dominance_duration"].get("IndependentAgent", 0),
                "control_dominance_duration": dominance_data["dominance_duration"].get("ControlAgent", 0),
            }
            
            # Add positioning metrics
            for agent_type, metrics in positioning_metrics.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        key = f"{agent_type.lower()}_{metric}"
                        row_data[key] = value
                else:
                    row_data[agent_type] = metrics
            
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
        
        print(f"Visualizations saved to {output_dir}")
        logging.info(f"Visualizations saved to {output_dir}")
        logging.info(f"Log file saved to: {log_file_path}")
    else:
        print("No data available for visualization")
        logging.warning("No data available for visualization")


if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
analyze_time_series.py

This script analyzes how initial conditions affect the evolution of agent populations over time.
It extracts time series data from simulation databases and visualizes population dynamics,
resource acquisition, and key events to understand how initial advantages translate into dominance.

Usage:
    python analyze_time_series.py
"""

import glob
import json
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.spatial.distance import euclidean

# Load the simulation analysis results
df = pd.read_csv("simulation_analysis.csv")


def extract_time_series(db_path):
    """
    Extract time series data from a simulation database.
    """
    conn = sqlite3.connect(db_path)

    # Get simulation steps data
    steps_query = """
    SELECT step_number, total_agents, system_agents, independent_agents, control_agents,
           total_resources, average_agent_resources, births, deaths, average_agent_health,
           average_agent_age, combat_encounters, successful_attacks, resources_shared_this_step AS resources_shared
    FROM simulation_steps
    ORDER BY step_number
    """
    steps_df = pd.read_sql_query(steps_query, conn)

    # Get reproduction events
    repro_query = """
    SELECT step_number, parent_id, offspring_id, success, parent_resources_before,
           parent_resources_after, offspring_initial_resources, parent_generation,
           offspring_generation, parent_position_x, parent_position_y
    FROM reproduction_events
    ORDER BY step_number
    """
    repro_df = pd.read_sql_query(repro_query, conn)

    # Get agent data to map IDs to types
    agents_query = """
    SELECT agent_id, agent_type, birth_time, death_time
    FROM agents
    """
    agents_df = pd.read_sql_query(agents_query, conn)

    conn.close()

    # Create a mapping of agent_id to agent_type
    agent_types = dict(zip(agents_df["agent_id"], agents_df["agent_type"]))

    # Add agent type information to reproduction events
    if not repro_df.empty:
        repro_df["parent_type"] = repro_df["parent_id"].map(agent_types)
        # Count successful reproductions by agent type and step
        if "success" in repro_df.columns:
            successful_repro = repro_df[repro_df["success"] == 1]
            repro_counts = (
                successful_repro.groupby(["step_number", "parent_type"])
                .size()
                .unstack(fill_value=0)
            )

            # Rename columns to standardize
            if "SystemAgent" in repro_counts.columns:
                repro_counts.rename(
                    columns={"SystemAgent": "system_reproduction"}, inplace=True
                )
            else:
                repro_counts["system_reproduction"] = 0

            if "IndependentAgent" in repro_counts.columns:
                repro_counts.rename(
                    columns={"IndependentAgent": "independent_reproduction"},
                    inplace=True,
                )
            else:
                repro_counts["independent_reproduction"] = 0

            if "ControlAgent" in repro_counts.columns:
                repro_counts.rename(
                    columns={"ControlAgent": "control_reproduction"}, inplace=True
                )
            else:
                repro_counts["control_reproduction"] = 0

            # Merge reproduction counts with steps data
            steps_df = pd.merge(
                steps_df, repro_counts.reset_index(), on="step_number", how="left"
            )
            steps_df.fillna(
                {
                    "system_reproduction": 0,
                    "independent_reproduction": 0,
                    "control_reproduction": 0,
                },
                inplace=True,
            )

    return steps_df, repro_df, agents_df


def get_initial_positions(db_path):
    """
    Extract initial positions of agents and resources from the simulation database.
    """
    conn = sqlite3.connect(db_path)

    # Get initial agents (birth_time = 0)
    agents_query = """
    SELECT agent_id, agent_type, position_x, position_y, initial_resources
    FROM agents
    WHERE birth_time = 0
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


def calculate_initial_advantages(agents_df, resources_df, gathering_range=30):
    """
    Calculate initial advantages based on agent positions relative to resources.
    """
    advantages = {}

    for _, agent in agents_df.iterrows():
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

            # Resources within gathering range
            resources_in_range = sum(1 for d, _ in distances if d <= gathering_range)

            # Resource amount within gathering range
            resource_amount_in_range = sum(
                amount for dist, amount in distances if dist <= gathering_range
            )

            advantages[agent_type] = {
                "nearest_resource_dist": nearest_dist,
                "resources_in_range": resources_in_range,
                "resource_amount_in_range": resource_amount_in_range,
            }

    return advantages


def plot_population_time_series(steps_df, initial_advantages, title, filename):
    """
    Plot population time series with initial advantage annotations.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot population counts
    ax.plot(
        steps_df["step_number"], steps_df["system_agents"], "b-", label="System Agents"
    )
    ax.plot(
        steps_df["step_number"],
        steps_df["independent_agents"],
        "r-",
        label="Independent Agents",
    )
    ax.plot(
        steps_df["step_number"],
        steps_df["control_agents"],
        "orange",
        label="Control Agents",
    )

    # Add annotations for initial advantages
    advantage_text = []
    for agent_type, metrics in initial_advantages.items():
        if agent_type == "SystemAgent":
            color = "blue"
            y_pos = 0.85
        elif agent_type == "IndependentAgent":
            color = "red"
            y_pos = 0.80
        else:  # ControlAgent
            color = "orange"
            y_pos = 0.75

        text = f"{agent_type}: {metrics['resources_in_range']} resources in range ({metrics['resource_amount_in_range']:.1f} total)"
        advantage_text.append(text)

        # Add text annotation
        ax.text(
            0.02,
            y_pos,
            text,
            transform=ax.transAxes,
            color=color,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
        )

    # Add vertical line at step 100 to mark early phase
    ax.axvline(x=100, color="gray", linestyle="--", alpha=0.7)
    ax.text(105, ax.get_ylim()[1] * 0.95, "Early Phase", fontsize=10)

    # Set labels and title
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Population Count")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reproduction_time_series(steps_df, repro_df, title, filename):
    """
    Plot reproduction events over time.
    """
    if "system_reproduction" not in steps_df.columns:
        return  # No reproduction data available

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot cumulative reproduction counts
    ax.plot(
        steps_df["step_number"],
        steps_df["system_reproduction"].cumsum(),
        "b-",
        label="System Reproduction",
    )
    ax.plot(
        steps_df["step_number"],
        steps_df["independent_reproduction"].cumsum(),
        "r-",
        label="Independent Reproduction",
    )
    ax.plot(
        steps_df["step_number"],
        steps_df["control_reproduction"].cumsum(),
        "orange",
        label="Control Reproduction",
    )

    # Find first reproduction for each agent type
    first_repro = {}
    for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
        type_repro = repro_df[repro_df["parent_type"] == agent_type]
        if not type_repro.empty and "success" in type_repro.columns:
            success_repro = type_repro[type_repro["success"] == 1]
            if not success_repro.empty:
                first_step = success_repro["step_number"].min()
                first_repro[agent_type] = first_step

                # Add marker for first reproduction
                if agent_type == "SystemAgent":
                    color = "blue"
                elif agent_type == "IndependentAgent":
                    color = "red"
                else:  # ControlAgent
                    color = "orange"

                ax.axvline(x=first_step, color=color, linestyle="--", alpha=0.7)
                ax.text(
                    first_step + 5,
                    ax.get_ylim()[1] * (0.1 + 0.05 * len(first_repro)),
                    f"First {agent_type} reproduction",
                    color=color,
                    fontsize=9,
                )

    # Set labels and title
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Cumulative Reproduction Count")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_resource_acquisition(steps_df, title, filename):
    """
    Plot resource acquisition metrics over time.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot total resources
    ax1.plot(
        steps_df["step_number"],
        steps_df["total_resources"],
        "g-",
        label="Total Resources",
    )
    ax1.set_ylabel("Total Resources")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Plot average agent resources
    ax2.plot(
        steps_df["step_number"],
        steps_df["average_agent_resources"],
        "b-",
        label="Avg Agent Resources",
    )

    # Add resource sharing if available
    if "resources_shared" in steps_df.columns:
        # Plot on secondary y-axis with bars
        ax3 = ax2.twinx()
        ax3.bar(
            steps_df["step_number"],
            steps_df["resources_shared"],
            color="r",
            alpha=0.3,
            label="Resources Shared",
            zorder=1,
        )
        ax3.set_ylabel("Resources Shared", color="r")
        ax3.tick_params(axis="y", labelcolor="r")

        # Ensure both axes have the same scale
        y_min = min(
            steps_df["average_agent_resources"].min(),
            steps_df["resources_shared"].min(),
        )
        y_max = (
            max(
                steps_df["average_agent_resources"].max(),
                steps_df["resources_shared"].max(),
            )
            + 1
        )
        ax2.set_ylim(y_min, y_max)
        ax3.set_ylim(y_min, y_max)

        # Ensure the line plot is on top of the bars
        ax2.set_zorder(ax3.get_zorder() + 1)
        ax2.patch.set_visible(False)  # Make ax2 background transparent

        # Create combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax2.legend(loc="upper right")

    ax2.set_xlabel("Simulation Step")
    ax2.set_ylabel("Average Resources per Agent")
    ax2.grid(True, alpha=0.3)

    # Set title
    fig.suptitle(title)

    # Save the figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for suptitle
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparative_analysis(simulations_data, title, filename):
    """
    Create a comparative analysis of multiple simulations.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Plot population ratios over time for each simulation
    for i, (sim_id, data) in enumerate(simulations_data.items()):
        steps_df = data["steps_df"]

        # Calculate population ratios
        total = steps_df["total_agents"].replace(0, 1)  # Avoid division by zero
        steps_df["system_ratio"] = steps_df["system_agents"] / total
        steps_df["independent_ratio"] = steps_df["independent_agents"] / total
        steps_df["control_ratio"] = steps_df["control_agents"] / total

        # Plot on the appropriate subplot
        ax_idx = i % 4
        ax = axes[ax_idx]

        ax.plot(steps_df["step_number"], steps_df["system_ratio"], "b-", label="System")
        ax.plot(
            steps_df["step_number"],
            steps_df["independent_ratio"],
            "r-",
            label="Independent",
        )
        ax.plot(
            steps_df["step_number"],
            steps_df["control_ratio"],
            "orange",
            label="Control",
        )

        # Add initial advantage annotation
        advantages = data["initial_advantages"]
        for agent_type, metrics in advantages.items():
            if agent_type == "SystemAgent":
                color = "blue"
                y_pos = 0.85
            elif agent_type == "IndependentAgent":
                color = "red"
                y_pos = 0.80
            else:  # ControlAgent
                color = "orange"
                y_pos = 0.75

            text = f"{agent_type}: {metrics['resources_in_range']} res."
            ax.text(
                0.02,
                y_pos,
                text,
                transform=ax.transAxes,
                color=color,
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
            )

        ax.set_title(f"Simulation {sim_id}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Population Ratio")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

        if ax_idx == 0:  # Only add legend to the first subplot
            ax.legend(loc="upper right")

    # Set overall title
    fig.suptitle(title, fontsize=16)

    # Save the figure
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust for suptitle
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    # Import analysis configuration
    from analysis_config import DATA_PATH, OUTPUT_PATH

    # Path to the experiment folder
    experiment_path = DATA_PATH
    # Create output directory for time series analysis
    os.makedirs(OUTPUT_PATH + "time_series_analysis", exist_ok=True)

    # Group simulations by dominance outcome
    dominance_groups = {
        "system": df[df["population_dominance"] == "system"]["iteration"].tolist(),
        "independent": df[df["population_dominance"] == "independent"][
            "iteration"
        ].tolist(),
        "control": df[df["population_dominance"] == "control"]["iteration"].tolist(),
    }

    # Select a few representative simulations from each group
    np.random.seed(42)  # For reproducibility
    selected_iterations = {}
    for dom_type, iterations in dominance_groups.items():
        if len(iterations) > 3:
            selected_iterations[dom_type] = np.random.choice(
                iterations, 3, replace=False
            ).tolist()
        else:
            selected_iterations[dom_type] = iterations

    # Analyze selected simulations
    simulations_data = {}

    for dom_type, iterations in selected_iterations.items():
        for iteration in iterations:
            print(
                f"Analyzing time series for iteration {iteration} (dominance: {dom_type})"
            )

            folder = os.path.join(experiment_path, f"iteration_{iteration}")
            db_path = os.path.join(folder, "simulation.db")
            config_path = os.path.join(folder, "config.json")

            if not os.path.exists(db_path) or not os.path.exists(config_path):
                print(f"Missing files for iteration {iteration}")
                continue

            # Load config
            with open(config_path, "r") as f:
                config = json.load(f)

            # Extract time series data
            steps_df, repro_df, agents_df = extract_time_series(db_path)

            # Get initial positions
            initial_agents, initial_resources = get_initial_positions(db_path)

            # Calculate initial advantages
            gathering_range = config.get("gathering_range", 30)
            initial_advantages = calculate_initial_advantages(
                initial_agents, initial_resources, gathering_range
            )

            # Store data for comparative analysis
            simulations_data[iteration] = {
                "steps_df": steps_df,
                "repro_df": repro_df,
                "agents_df": agents_df,
                "initial_advantages": initial_advantages,
                "dominance_type": dom_type,
            }

            # Plot population time series
            title = f"Population Dynamics - Iteration {iteration} ({dom_type.capitalize()} Dominance)"
            filename = f"time_series_analysis/population_time_series_{iteration}.png"
            plot_population_time_series(steps_df, initial_advantages, title, filename)

            # Plot reproduction time series
            title = f"Reproduction Events - Iteration {iteration} ({dom_type.capitalize()} Dominance)"
            filename = f"time_series_analysis/reproduction_time_series_{iteration}.png"
            plot_reproduction_time_series(steps_df, repro_df, title, filename)

            # Plot resource acquisition
            title = f"Resource Dynamics - Iteration {iteration} ({dom_type.capitalize()} Dominance)"
            filename = f"time_series_analysis/resource_dynamics_{iteration}.png"
            plot_resource_acquisition(steps_df, title, filename)

    # Create comparative analyses for each dominance type
    for dom_type, iterations in selected_iterations.items():
        # Filter to only include simulations that were successfully processed
        dom_simulations = {
            it: simulations_data[it] for it in iterations if it in simulations_data
        }

        if dom_simulations:
            title = (
                f"Comparative Analysis - {dom_type.capitalize()} Dominance Simulations"
            )
            filename = f"time_series_analysis/comparative_{dom_type}_dominance.png"
            plot_comparative_analysis(dom_simulations, title, filename)

    # Create a cross-dominance comparison
    cross_dom_simulations = {}
    for dom_type, iterations in selected_iterations.items():
        if iterations:
            # Take the first simulation from each dominance type
            it = iterations[0]
            if it in simulations_data:
                cross_dom_simulations[it] = simulations_data[it]

    if len(cross_dom_simulations) > 1:
        title = "Cross-Dominance Comparison - Initial Conditions Impact"
        filename = "time_series_analysis/cross_dominance_comparison.png"
        plot_comparative_analysis(cross_dom_simulations, title, filename)

    print(
        "Time series analysis complete. Results saved to 'time_series_analysis' directory."
    )


if __name__ == "__main__":
    main()

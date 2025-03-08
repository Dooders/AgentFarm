#!/usr/bin/env python3
"""
visualize_initial_conditions.py

This script visualizes the initial positions of agents and resources for selected simulations,
highlighting how initial spatial arrangements correlate with different dominance outcomes.

Usage:
    python visualize_initial_conditions.py
"""

import glob
import json
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from scipy.spatial.distance import euclidean

# Load the simulation analysis results
df = pd.read_csv("simulation_analysis.csv")


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


def visualize_simulation(iteration, experiment_path, dominance_type):
    """
    Create a visualization of initial positions for a specific simulation.
    """
    folder = os.path.join(experiment_path, f"iteration_{iteration}")
    db_path = os.path.join(folder, "simulation.db")
    config_path = os.path.join(folder, "config.json")

    if not os.path.exists(db_path) or not os.path.exists(config_path):
        print(f"Missing files for iteration {iteration}")
        return None

    # Load config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Get initial positions
    agents, resources = get_initial_positions(db_path)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot resources
    scatter = ax.scatter(
        resources["position_x"],
        resources["position_y"],
        s=resources["amount"] * 5,  # Size based on amount
        alpha=0.5,
        c="green",
        label="Resources",
    )

    # Plot agents with different colors by type
    agent_colors = {
        "SystemAgent": "blue",
        "IndependentAgent": "red",
        "ControlAgent": "orange",
    }

    for agent_type, color in agent_colors.items():
        agent_data = agents[agents["agent_type"] == agent_type]
        if not agent_data.empty:
            ax.scatter(
                agent_data["position_x"],
                agent_data["position_y"],
                s=100,
                c=color,
                edgecolors="black",
                label=agent_type,
            )

            # Add agent ID labels
            for _, agent in agent_data.iterrows():
                ax.annotate(
                    agent["agent_id"][-4:],  # Last 4 chars of ID
                    (agent["position_x"], agent["position_y"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                )

    # Draw gathering range circles around agents
    gathering_range = config.get("gathering_range", 30)
    for _, agent in agents.iterrows():
        circle = Circle(
            (agent["position_x"], agent["position_y"]),
            gathering_range,
            fill=False,
            linestyle="--",
            color=agent_colors[agent["agent_type"]],
            alpha=0.3,
        )
        ax.add_patch(circle)

    # Calculate and display distances to nearest resources
    for _, agent in agents.iterrows():
        agent_pos = (agent["position_x"], agent["position_y"])
        distances = [
            euclidean(agent_pos, (r["position_x"], r["position_y"]))
            for _, r in resources.iterrows()
        ]
        nearest_dist = min(distances) if distances else float("inf")

        # Find the nearest resource
        if distances:
            nearest_idx = np.argmin(distances)
            nearest_resource = resources.iloc[nearest_idx]
            nearest_pos = (
                nearest_resource["position_x"],
                nearest_resource["position_y"],
            )

            # Draw line to nearest resource
            ax.plot(
                [agent_pos[0], nearest_pos[0]],
                [agent_pos[1], nearest_pos[1]],
                color=agent_colors[agent["agent_type"]],
                linestyle=":",
                alpha=0.7,
            )

            # Add distance text
            mid_x = (agent_pos[0] + nearest_pos[0]) / 2
            mid_y = (agent_pos[1] + nearest_pos[1]) / 2
            ax.annotate(
                f"{nearest_dist:.1f}",
                (mid_x, mid_y),
                color=agent_colors[agent["agent_type"]],
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
            )

    # Set plot limits and labels
    width = config.get("width", 100)
    height = config.get("height", 100)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Get dominance information
    sim_data = df[df["iteration"] == iteration].iloc[0]
    pop_dom = sim_data["population_dominance"]
    surv_dom = sim_data["survival_dominance"]

    # Set title based on dominance type
    if dominance_type == "population":
        title = f"Iteration {iteration}: Population Dominance = {pop_dom}"
    else:
        title = f"Iteration {iteration}: Survival Dominance = {surv_dom}"

    ax.set_title(title)
    ax.legend(loc="upper right")

    # Add a text box with key metrics
    textstr = "\n".join(
        [f"Population Dominance: {pop_dom}", f"Survival Dominance: {surv_dom}"]
    )

    # Add agent-specific metrics
    for agent_type in ["system", "independent", "control"]:
        nearest_dist_key = f"{agent_type}agent_nearest_resource_dist"
        resources_in_range_key = f"{agent_type}agent_resources_in_range"

        if nearest_dist_key in sim_data and resources_in_range_key in sim_data:
            textstr += f"\n{agent_type.capitalize()} nearest resource: {sim_data[nearest_dist_key]:.1f}"
            textstr += f"\n{agent_type.capitalize()} resources in range: {sim_data[resources_in_range_key]}"

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )

    return fig


def main():
    # Path to the experiment folder
    experiment_path = (
        "results/one_of_a_kind/experiments/data/one_of_a_kind_20250307_193855"
    )

    # Create output directory for visualizations
    os.makedirs("initial_conditions_viz", exist_ok=True)

    # Select representative simulations for each dominance type
    dominance_types = {
        "system": df[df["population_dominance"] == "system"]["iteration"].tolist(),
        "independent": df[df["population_dominance"] == "independent"][
            "iteration"
        ].tolist(),
        "control": df[df["population_dominance"] == "control"]["iteration"].tolist(),
    }

    # Select 5 random examples from each dominance type
    np.random.seed(42)  # For reproducibility
    selected_iterations = {}
    for dom_type, iterations in dominance_types.items():
        if len(iterations) > 5:
            selected_iterations[dom_type] = np.random.choice(
                iterations, 5, replace=False
            ).tolist()
        else:
            selected_iterations[dom_type] = iterations

    # Visualize selected simulations
    for dom_type, iterations in selected_iterations.items():
        for iteration in iterations:
            print(f"Visualizing iteration {iteration} (dominance: {dom_type})")
            fig = visualize_simulation(iteration, experiment_path, "population")
            if fig:
                fig.savefig(
                    f"initial_conditions_viz/iteration_{iteration}_{dom_type}_dominance.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

    # Create a comparative visualization
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle("Initial Conditions by Dominance Type", fontsize=16)

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Plot one example from each dominance type in each row
    for i, dom_type in enumerate(["system", "independent", "control"]):
        iterations = selected_iterations[dom_type][:5]  # Take up to 5
        for j, iteration in enumerate(iterations):
            if j < 5:  # Ensure we don't exceed the grid
                ax_idx = i * 5 + j
                folder = os.path.join(experiment_path, f"iteration_{iteration}")
                db_path = os.path.join(folder, "simulation.db")

                if os.path.exists(db_path):
                    agents, resources = get_initial_positions(db_path)

                    # Plot resources
                    axes[ax_idx].scatter(
                        resources["position_x"],
                        resources["position_y"],
                        s=resources["amount"] * 3,
                        alpha=0.5,
                        c="green",
                    )

                    # Plot agents
                    agent_colors = {
                        "SystemAgent": "blue",
                        "IndependentAgent": "red",
                        "ControlAgent": "orange",
                    }

                    for agent_type, color in agent_colors.items():
                        agent_data = agents[agents["agent_type"] == agent_type]
                        if not agent_data.empty:
                            axes[ax_idx].scatter(
                                agent_data["position_x"],
                                agent_data["position_y"],
                                s=80,
                                c=color,
                                edgecolors="black",
                            )

                    axes[ax_idx].set_title(f"Iteration {iteration}")
                    axes[ax_idx].set_xlim(0, 100)
                    axes[ax_idx].set_ylim(0, 100)

                    # Add row labels
                    if j == 0:
                        axes[ax_idx].set_ylabel(
                            f"{dom_type.capitalize()} Dominance", fontsize=12
                        )

    # Add a legend to the figure
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="green",
            markersize=10,
            label="Resources",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="blue",
            markersize=10,
            label="System Agent",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=10,
            label="Independent Agent",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="orange",
            markersize=10,
            label="Control Agent",
        ),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the suptitle
    plt.savefig(
        "initial_conditions_viz/comparative_initial_conditions.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print("Visualizations saved to 'initial_conditions_viz' directory")


if __name__ == "__main__":
    main()

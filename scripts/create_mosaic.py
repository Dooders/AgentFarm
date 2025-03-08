#!/usr/bin/env python3
"""
create_mosaic.py

Creates a clean, minimalist mosaic of initial simulation states arranged in a grid.
"""

import json
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle


def get_initial_state(db_path):
    """Get the initial state (step 0) from the simulation database."""
    conn = sqlite3.connect(db_path)

    # Get initial agents (birth_time = 0)
    agents_query = """
    SELECT agent_id, agent_type, position_x, position_y, initial_resources as resources
    FROM agents
    WHERE birth_time = 0
    """
    agents = pd.read_sql_query(agents_query, conn)
    print(f"Found {len(agents)} agents")  # Debug print
    print(agents["agent_type"].value_counts())  # Debug print

    # Get initial resources (step 0)
    resources_query = """
    SELECT resource_id, position_x, position_y, amount
    FROM resource_states
    WHERE step_number = 0
    """
    resources = pd.read_sql_query(resources_query, conn)

    conn.close()
    return agents, resources


def create_clean_frame(ax, agents, resources, config):
    """Create a single clean frame without any decorations."""
    # Plot resources
    scatter = ax.scatter(
        resources["position_x"],
        resources["position_y"],
        s=resources["amount"] * 2,  # Smaller dots
        alpha=0.3,  # More transparent
        c="#2ecc71",  # Green
        zorder=1,
    )

    # Define colors for agents
    agent_colors = {
        "SystemAgent": "#3498db",  # Blue
        "IndependentAgent": "#e74c3c",  # Red
        "ControlAgent": "#f39c12",  # Orange
    }

    # Plot agents with larger, more visible dots
    for agent_type, color in agent_colors.items():
        agent_data = agents[agents["agent_type"] == agent_type]
        if not agent_data.empty:
            ax.scatter(
                agent_data["position_x"],
                agent_data["position_y"],
                s=120,  # Larger dots
                c=color,
                edgecolors="white",  # White edge for better visibility
                linewidth=1,
                alpha=1.0,
                zorder=2,
            )

    # Calculate padding based on the dot size
    dot_radius = np.sqrt(120 / np.pi)  # Approximate radius of the largest dots
    padding = dot_radius * 0.75  # Add 75% of dot radius as padding

    # Clean up the plot completely and add padding to prevent cutoff
    ax.set_xlim(-padding, 100 + padding)
    ax.set_ylim(-padding, 100 + padding)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_facecolor("white")
    ax.set_aspect("equal")  # Ensure square aspect ratio


def create_mosaic(experiment_path, selected_iterations, output_path="mosaic.png"):
    """Create a clean mosaic of initial states."""
    # Set up the figure
    plt.style.use("default")
    fig = plt.figure(figsize=(20, 20), facecolor="white")  # Perfect square

    # Create grid with minimal spacing
    n_rows = 5  # Now 5x5 grid
    n_cols = 5
    gs = GridSpec(n_rows, n_cols, figure=fig)
    gs.update(wspace=0.05, hspace=0.05)  # Slightly more spacing to prevent overlap

    # Process each iteration
    for idx, iteration in enumerate(selected_iterations):
        row = idx // n_cols
        col = idx % n_cols

        # Get paths
        folder = os.path.join(experiment_path, f"iteration_{iteration}")
        db_path = os.path.join(folder, "simulation.db")
        config_path = os.path.join(folder, "config.json")

        if not os.path.exists(db_path) or not os.path.exists(config_path):
            print(f"Missing files for iteration {iteration}")
            continue

        # Load data
        with open(config_path, "r") as f:
            config = json.load(f)
        agents, resources = get_initial_state(db_path)

        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        create_clean_frame(ax, agents, resources, config)

    # Save with high DPI and tight layout
    plt.savefig(
        output_path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
    )
    plt.close()
    print(f"Mosaic saved to: {output_path}")


def main():
    experiment_path = (
        "results/one_of_a_kind/experiments/data/one_of_a_kind_20250302_193353"
    )

    # Selected iterations from your image, now with a fifth row for perfect square
    selected_iterations = [
        249,
        123,
        110,
        37,
        225,  # First row
        71,
        233,
        61,
        24,
        19,  # Second row
        190,
        11,
        178,
        36,
        131,  # Third row
        42,
        85,
        156,
        201,
        95,  # Fourth row
        145,
        167,
        188,
        205,
        220,  # Fifth row (new)
    ]

    create_mosaic(experiment_path, selected_iterations, "artistic_mosaic.png")


if __name__ == "__main__":
    main()

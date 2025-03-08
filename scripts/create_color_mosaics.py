#!/usr/bin/env python3
"""
create_color_mosaics.py

Creates multiple mosaics with different artistic color schemes.
"""

import json
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

# Define various color schemes
COLOR_SCHEMES = {
    "default": {
        "background": "#ffffff",
        "resources": "#2ecc71",
        "agents": {
            "SystemAgent": "#3498db",
            "IndependentAgent": "#e74c3c",
            "ControlAgent": "#f39c12",
        },
    },
    "pastel": {
        "background": "#f6e6e4",
        "resources": "#95a5a6",
        "agents": {
            "SystemAgent": "#a8e6cf",
            "IndependentAgent": "#ffd3b6",
            "ControlAgent": "#ffaaa5",
        },
    },
    "monochrome": {
        "background": "#000000",
        "resources": "#ffffff",
        "agents": {
            "SystemAgent": "#cccccc",
            "IndependentAgent": "#999999",
            "ControlAgent": "#666666",
        },
    },
    "neon": {
        "background": "#000000",
        "resources": "#39ff14",
        "agents": {
            "SystemAgent": "#00ffff",
            "IndependentAgent": "#ff00ff",
            "ControlAgent": "#ffff00",
        },
    },
    "earth": {
        "background": "#2c1810",
        "resources": "#8b4513",
        "agents": {
            "SystemAgent": "#d4ac0d",
            "IndependentAgent": "#7d6608",
            "ControlAgent": "#b9770e",
        },
    },
    "ocean": {
        "background": "#000d1a",
        "resources": "#48d1cc",
        "agents": {
            "SystemAgent": "#00ffff",
            "IndependentAgent": "#4169e1",
            "ControlAgent": "#87ceeb",
        },
    },
    "sunset": {
        "background": "#1a0f1f",
        "resources": "#ff7e5f",
        "agents": {
            "SystemAgent": "#feb47b",
            "IndependentAgent": "#ff1361",
            "ControlAgent": "#662d8c",
        },
    },
    "vintage": {
        "background": "#2b1d0e",
        "resources": "#8c7355",
        "agents": {
            "SystemAgent": "#d7cdb4",
            "IndependentAgent": "#c5b9a0",
            "ControlAgent": "#9b8174",
        },
    },
    "cyberpunk": {
        "background": "#0b0221",
        "resources": "#0abdc6",
        "agents": {
            "SystemAgent": "#ea00d9",
            "IndependentAgent": "#711c91",
            "ControlAgent": "#133e7c",
        },
    },
    "autumn": {
        "background": "#1a0f00",
        "resources": "#2c5530",
        "agents": {
            "SystemAgent": "#c84c09",
            "IndependentAgent": "#8b0000",
            "ControlAgent": "#d35400",
        },
    },
}


def get_initial_state(db_path):
    """Get the initial state (step 0) from the simulation database."""
    conn = sqlite3.connect(db_path)

    agents_query = """
    SELECT agent_id, agent_type, position_x, position_y, initial_resources as resources
    FROM agents
    WHERE birth_time = 0
    """
    agents = pd.read_sql_query(agents_query, conn)

    resources_query = """
    SELECT resource_id, position_x, position_y, amount
    FROM resource_states
    WHERE step_number = 0
    """
    resources = pd.read_sql_query(resources_query, conn)

    conn.close()
    return agents, resources


def create_clean_frame(ax, agents, resources, config, color_scheme):
    """Create a single clean frame with specified color scheme."""
    # Set background color
    ax.set_facecolor(color_scheme["background"])

    # Plot resources with adjusted alpha based on background
    is_dark_bg = (
        sum(
            int(color_scheme["background"].lstrip("#")[i : i + 2], 16)
            for i in (0, 2, 4)
        )
        < 384
    )
    resource_alpha = 0.5 if is_dark_bg else 0.3

    scatter = ax.scatter(
        resources["position_x"],
        resources["position_y"],
        s=resources["amount"] * 2,
        alpha=resource_alpha,
        c=color_scheme["resources"],
        zorder=1,
    )

    # Plot agents
    edge_color = "black" if not is_dark_bg else "white"
    for agent_type, color in color_scheme["agents"].items():
        agent_data = agents[agents["agent_type"] == agent_type]
        if not agent_data.empty:
            ax.scatter(
                agent_data["position_x"],
                agent_data["position_y"],
                s=120,
                c=color,
                edgecolors=edge_color,
                linewidth=1,
                alpha=1.0,
                zorder=2,
            )

    # Calculate padding
    dot_radius = np.sqrt(120 / np.pi)
    padding = dot_radius * 0.75

    # Clean up the plot
    ax.set_xlim(-padding, 100 + padding)
    ax.set_ylim(-padding, 100 + padding)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_aspect("equal")


def create_mosaic(experiment_path, selected_iterations, color_scheme, output_path):
    """Create a mosaic with specified color scheme."""
    plt.style.use("default")
    fig = plt.figure(figsize=(20, 20), facecolor=color_scheme["background"])

    n_rows = 5
    n_cols = 5
    gs = GridSpec(n_rows, n_cols, figure=fig)
    gs.update(wspace=0.05, hspace=0.05)

    for idx, iteration in enumerate(selected_iterations):
        row = idx // n_cols
        col = idx % n_cols

        folder = os.path.join(experiment_path, f"iteration_{iteration}")
        db_path = os.path.join(folder, "simulation.db")
        config_path = os.path.join(folder, "config.json")

        if not os.path.exists(db_path) or not os.path.exists(config_path):
            print(f"Missing files for iteration {iteration}")
            continue

        with open(config_path, "r") as f:
            config = json.load(f)
        agents, resources = get_initial_state(db_path)

        ax = fig.add_subplot(gs[row, col])
        create_clean_frame(ax, agents, resources, config, color_scheme)

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor=color_scheme["background"],
        edgecolor="none",
    )
    plt.close()
    print(f"Mosaic saved to: {output_path}")


def main():
    experiment_path = (
        "results/one_of_a_kind/experiments/data/one_of_a_kind_20250302_193353"
    )

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
        220,  # Fifth row
    ]

    # Create output directory for color variations
    os.makedirs("mosaic_variations", exist_ok=True)

    # Generate a mosaic for each color scheme
    for scheme_name, color_scheme in COLOR_SCHEMES.items():
        output_path = f"mosaic_variations/mosaic_{scheme_name}.png"
        print(f"\nGenerating {scheme_name} color scheme...")
        create_mosaic(experiment_path, selected_iterations, color_scheme, output_path)


if __name__ == "__main__":
    main()

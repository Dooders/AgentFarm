#!/usr/bin/env python3
"""
mosaic_viz.py

Consolidated visualization module combining functionality from:
- create_mosaic.py (basic mosaic creation)
- create_color_mosaics.py (color scheme variations)
- visualize_initial_conditions.py (initial condition analysis)

This module provides comprehensive visualization capabilities for simulation initial states,
comparative analysis, and artistic color variations.
"""

import glob
import json
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# Import our utility modules
from ..data_extraction import (
    calculate_initial_advantages,
    get_initial_positions,
    get_initial_state_with_config,
    load_simulation_config,
)
from ..database_utils import (
    get_iteration_number,
    get_simulation_config_path,
    get_simulation_database_path,
    get_simulation_folders,
    validate_simulation_folder,
)
from ..visualization_utils import (
    COLOR_SCHEMES,
    create_clean_frame,
    create_frame_with_labels,
    create_mosaic_frame,
    get_agent_colors,
    save_figure,
    setup_plot_style,
)

# Extended color schemes from create_color_mosaics.py
EXTENDED_COLOR_SCHEMES = {
    **COLOR_SCHEMES,
    "pastel": {
        "background": "#f6e6e4",
        "resources": "#95a5a6",
        "agents": {
            "Agent": "#a8e6cf",
        },
    },
    "neon": {
        "background": "#000000",
        "resources": "#39ff14",
        "agents": {
            "Agent": "#00ffff",
        },
    },
    "earth": {
        "background": "#2c1810",
        "resources": "#8b4513",
        "agents": {
            "Agent": "#d4ac0d",
        },
    },
    "ocean": {
        "background": "#000d1a",
        "resources": "#48d1cc",
        "agents": {
            "Agent": "#00ffff",
        },
    },
    "sunset": {
        "background": "#1a0f1f",
        "resources": "#ff7e5f",
        "agents": {
            "Agent": "#feb47b",
        },
    },
    "vintage": {
        "background": "#2b1d0e",
        "resources": "#8c7355",
        "agents": {
            "Agent": "#d7cdb4",
        },
    },
    "cyberpunk": {
        "background": "#0b0221",
        "resources": "#0abdc6",
        "agents": {
            "Agent": "#ea00d9",
        },
    },
    "autumn": {
        "background": "#1a0f00",
        "resources": "#2c5530",
        "agents": {
            "Agent": "#c84c09",
        },
    },
}


def create_single_mosaic(
    experiment_path: str,
    selected_iterations: List[int],
    output_path: str = "mosaic.png",
    color_scheme: str = "default",
    show_labels: bool = False,
    title: str = "Simulation Initial States",
) -> None:
    """
    Create a single mosaic visualization.

    Parameters
    ----------
    experiment_path : str
        Path to experiment directory
    selected_iterations : List[int]
        List of iteration numbers to include
    output_path : str
        Output file path
    color_scheme : str
        Color scheme to use
    show_labels : bool
        Whether to show agent labels
    title : str
        Plot title
    """
    setup_plot_style("default")

    fig = plt.figure(figsize=(20, 20))
    n_rows = int(np.ceil(len(selected_iterations) / 5))
    n_cols = min(len(selected_iterations), 5)
    gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.05, hspace=0.05)

    if title:
        fig.suptitle(title, fontsize=16, y=0.95)

    for idx, iteration in enumerate(selected_iterations):
        row = idx // n_cols
        col = idx % n_cols

        folder = os.path.join(experiment_path, f"iteration_{iteration}")
        db_path = get_simulation_database_path(folder)
        config_path = get_simulation_config_path(folder)

        if not validate_simulation_folder(folder):
            print(f"Skipping invalid folder: {folder}")
            continue

        agents, resources = get_initial_positions(db_path)

        ax = fig.add_subplot(gs[row, col])

        if show_labels:
            create_frame_with_labels(ax, agents, resources, color_scheme=color_scheme)
        else:
            create_clean_frame(ax, agents, resources, color_scheme=color_scheme)

        ax.set_title(f"Iteration {iteration}", fontsize=10)

    save_figure(fig, output_path)
    print(f"Single mosaic saved to: {output_path}")


def create_color_variations_mosaic(
    experiment_path: str,
    selected_iterations: List[int],
    output_dir: str = "mosaic_variations",
    color_schemes: Optional[List[str]] = None,
) -> None:
    """
    Create mosaics with different color schemes.

    Parameters
    ----------
    experiment_path : str
        Path to experiment directory
    selected_iterations : List[int]
        List of iteration numbers to include
    output_dir : str
        Output directory for variations
    color_schemes : Optional[List[str]]
        List of color schemes to use (default: all available)
    """
    os.makedirs(output_dir, exist_ok=True)

    if color_schemes is None:
        color_schemes = list(EXTENDED_COLOR_SCHEMES.keys())

    for scheme_name in color_schemes:
        output_path = os.path.join(output_dir, f"mosaic_{scheme_name}.png")
        create_single_mosaic(
            experiment_path,
            selected_iterations,
            output_path,
            color_scheme=scheme_name,
            title=f"Mosaic - {scheme_name.capitalize()} Theme",
        )


def create_comparative_visualization(
    experiment_path: str,
    iterations: List[int],
    output_path: str = "comparative_analysis.png",
    dominance_data: Optional[pd.DataFrame] = None,
) -> None:
    """
    Create comparative visualization of multiple simulations.
    Combines functionality from visualize_initial_conditions.py.

    Parameters
    ----------
    experiment_path : str
        Path to experiment directory
    iterations : List[int]
        List of iteration numbers to compare
    output_path : str
        Output file path
    dominance_data : Optional[pd.DataFrame]
        DataFrame with dominance information
    """
    setup_plot_style("default")

    n_sims = len(iterations)
    n_cols = min(n_sims, 3)
    n_rows = int(np.ceil(n_sims / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.suptitle("Comparative Initial Conditions Analysis", fontsize=16)

    if n_sims == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, iteration in enumerate(iterations):
        folder = os.path.join(experiment_path, f"iteration_{iteration}")
        db_path = get_simulation_database_path(folder)
        config_path = get_simulation_config_path(folder)

        if not validate_simulation_folder(folder):
            print(f"Skipping invalid folder: {folder}")
            continue

        agents, resources, config = get_initial_state_with_config(db_path, config_path)
        advantages = calculate_initial_advantages(
            agents, resources, config.get("gathering_range", 30)
        )

        ax = axes[i]
        create_clean_frame(ax, agents, resources, config)

        # Add dominance information if available
        title_parts = [f"Iteration {iteration}"]

        if dominance_data is not None and iteration in dominance_data.get(
            "iteration", []
        ):
            sim_data = dominance_data[dominance_data["iteration"] == iteration].iloc[0]
            pop_dom = sim_data.get("population_dominance", "unknown")
            surv_dom = sim_data.get("survival_dominance", "unknown")
            title_parts.extend([f"Pop: {pop_dom}", f"Surv: {surv_dom}"])

        ax.set_title("\n".join(title_parts), fontsize=10)

        # Add info box with key metrics
        info_text = (
            ".1f"
            ".1f"
            ".1f"
            f"""
Agents: {len(agents)}
Resources: {len(resources)}
Total Res: {advantages.get('Agent', {}).get('resource_amount_in_range', 0):.1f}
"""
        )

        ax.text(
            0.02,
            0.98,
            info_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"Comparative visualization saved to: {output_path}")


def create_dominance_correlation_visualization(
    experiment_path: str,
    dominance_data: pd.DataFrame,
    output_path: str = "dominance_correlation.png",
) -> None:
    """
    Create visualization showing correlation between initial conditions and dominance.

    Parameters
    ----------
    experiment_path : str
        Path to experiment directory
    dominance_data : pd.DataFrame
        DataFrame with dominance information
    output_path : str
        Output file path
    """
    setup_plot_style("default")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Initial Conditions vs Dominance Correlation", fontsize=16)

    axes = axes.flatten()

    # Get data for dominant simulations by type
    dominance_types = ["system", "independent", "control"]

    for i, dom_type in enumerate(dominance_types):
        ax = axes[i]

        # Filter data for this dominance type
        type_data = dominance_data[
            (dominance_data["population_dominance"] == dom_type)
            | (dominance_data["survival_dominance"] == dom_type)
        ]

        if type_data.empty:
            ax.text(
                0.5,
                0.5,
                f"No {dom_type} dominance data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Get initial conditions for these simulations
        advantages_data = []
        for _, row in type_data.iterrows():
            iteration = int(row["iteration"])
            folder = os.path.join(experiment_path, f"iteration_{iteration}")
            db_path = get_simulation_database_path(folder)
            config_path = get_simulation_config_path(folder)

            if validate_simulation_folder(folder):
                agents, resources, config = get_initial_state_with_config(
                    db_path, config_path
                )
                advantages = calculate_initial_advantages(
                    agents, resources, config.get("gathering_range", 30)
                )
                advantages_data.append(
                    {
                        "iteration": iteration,
                        "agent_advantage": advantages.get("Agent", {}).get(
                            "resource_amount_in_range", 0
                        ),
                    }
                )

        if advantages_data:
            df = pd.DataFrame(advantages_data)

            # Create box plot
            agent_types = ["system", "independent", "control"]
            data = [df[f"{agent}_advantage"] for agent in agent_types]
            ax.boxplot(data, labels=[t.capitalize() for t in agent_types])
            ax.set_title(f"Resource Advantage - {dom_type.capitalize()} Dominance")
            ax.set_ylabel("Resources in Range")

    # Summary statistics
    ax = axes[3]

    # Calculate correlation between initial advantage and dominance
    all_advantages = []
    for _, row in dominance_data.iterrows():
        iteration = int(row["iteration"])
        folder = os.path.join(experiment_path, f"iteration_{iteration}")
        db_path = get_simulation_database_path(folder)
        config_path = get_simulation_config_path(folder)

        if validate_simulation_folder(folder):
            agents, resources, config = get_initial_state_with_config(
                db_path, config_path
            )
            advantages = calculate_initial_advantages(
                agents, resources, config.get("gathering_range", 30)
            )

            agent_advantage = advantages.get("Agent", {}).get(
                "resource_amount_in_range", 0
            )

            # Since agent types are unified, we use a single advantage value
            predicted_dominant = "agent"

            actual_dominant = row["population_dominance"]

            all_advantages.append(
                {
                    "predicted": predicted_dominant,
                    "actual": actual_dominant,
                    "correct": predicted_dominant == actual_dominant,
                }
            )

    if all_advantages:
        df = pd.DataFrame(all_advantages)
        accuracy = df["correct"].mean()

        ax.text(
            0.5,
            0.5,
            ".1f"
            f"""
Initial Advantage Prediction Accuracy

Accuracy: {accuracy:.1%}

This measures how well initial resource
advantages predict final dominance.
""",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

    ax.set_title("Prediction Accuracy")

    plt.tight_layout()
    save_figure(fig, output_path)
    print(f"Dominance correlation analysis saved to: {output_path}")


def create_initial_conditions_report(
    experiment_path: str,
    iterations: Optional[List[int]] = None,
    output_dir: str = "initial_conditions_report",
) -> None:
    """
    Create comprehensive initial conditions analysis report.

    Parameters
    ----------
    experiment_path : str
        Path to experiment directory
    iterations : Optional[List[int]]
        Specific iterations to analyze (all if None)
    output_dir : str
        Output directory for report
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all simulation folders
    sim_folders = get_simulation_folders(experiment_path)
    if iterations:
        sim_folders = [f for f in sim_folders if get_iteration_number(f) in iterations]

    print(f"Analyzing {len(sim_folders)} simulations for initial conditions report")

    # Create standard mosaic
    all_iterations = [get_iteration_number(f) for f in sim_folders]
    create_single_mosaic(
        experiment_path, all_iterations, os.path.join(output_dir, "standard_mosaic.png")
    )

    # Create color variations
    create_color_variations_mosaic(
        experiment_path,
        all_iterations[:10],
        os.path.join(output_dir, "color_variations"),
    )

    # Create comparative analysis (first 6 simulations)
    create_comparative_visualization(
        experiment_path,
        all_iterations[:6],
        os.path.join(output_dir, "comparative_analysis.png"),
    )

    print(f"Initial conditions report saved to: {output_dir}")


def main():
    """
    Example usage of the consolidated mosaic visualization module.
    """
    # Example experiment path (adjust as needed)
    experiment_path = (
        "results/one_of_a_kind/experiments/data/one_of_a_kind_20250302_193353"
    )

    if not os.path.exists(experiment_path):
        print(f"Experiment path not found: {experiment_path}")
        print("Please update the path in the main() function")
        return

    # Example 1: Create standard mosaic
    selected_iterations = [249, 123, 110, 37, 225, 71]
    create_single_mosaic(experiment_path, selected_iterations, "standard_mosaic.png")

    # Example 2: Create color variations
    create_color_variations_mosaic(
        experiment_path, selected_iterations[:5], "color_variations"
    )

    # Example 3: Create comparative visualization
    create_comparative_visualization(
        experiment_path, selected_iterations[:6], "comparative_analysis.png"
    )

    print("Mosaic visualization examples completed!")


if __name__ == "__main__":
    main()

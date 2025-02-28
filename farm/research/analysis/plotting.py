import logging
import os
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from pathlib import Path

from farm.research.analysis.analysis import (
    create_population_df,
    calculate_statistics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_mean_and_ci(ax, steps, mean, ci, color, label):
    """
    Plot mean line with confidence interval.

    Args:
        ax: Matplotlib axis
        steps: Array of steps
        mean: Array of mean values
        ci: Array of confidence interval values
        color: Color for the plot
        label: Label for the plot
    """
    ax.plot(steps, mean, color=color, label=label)
    ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.2)


def plot_median_line(ax, steps, median, color="g", style="--"):
    """
    Plot median line.

    Args:
        ax: Matplotlib axis
        steps: Array of steps
        median: Array of median values
        color: Color for the plot
        style: Line style
    """
    ax.plot(steps, median, color=color, linestyle=style, label="Median")


def plot_reference_line(ax, y_value, label, color="orange"):
    """
    Plot horizontal reference line.

    Args:
        ax: Matplotlib axis
        y_value: Y-value for the line
        label: Label for the line
        color: Color for the line
    """
    ax.axhline(y=y_value, color=color, linestyle="--", label=label)


def plot_marker_point(ax, x, y, label):
    """
    Plot marker point.

    Args:
        ax: Matplotlib axis
        x: X-coordinate
        y: Y-coordinate
        label: Label for the point
    """
    ax.plot(x, y, "ro", label=label)


def setup_plot_aesthetics(ax, title, experiment_name=None):
    """
    Set up plot aesthetics.

    Args:
        ax: Matplotlib axis
        title: Plot title
        experiment_name: Experiment name
    """
    # Set title with path effects for better visibility
    title_obj = ax.set_title(title)
    title_obj.set_path_effects(
        [path_effects.withStroke(linewidth=1, foreground="white")]
    )

    # Add experiment name as subtitle if provided
    if experiment_name:
        ax.text(
            0.5,
            0.98,
            f"Experiment: {experiment_name}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
            fontstyle="italic",
        )

    # Add grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add legend
    ax.legend(loc="best")


def plot_population_trends_across_simulations(
    all_populations: List[np.ndarray], max_steps: int, output_path: str
):
    """
    Plot population trends across simulations.

    Args:
        all_populations: List of population arrays
        max_steps: Maximum number of steps
        output_path: Path to save the plot
    """
    if not all_populations:
        logger.warning("No population data to plot")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame from population data
    df = create_population_df(all_populations, max_steps)

    # Calculate statistics
    mean, median, lower_ci, upper_ci = calculate_statistics(df)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot mean and confidence interval
    steps = np.arange(len(mean))
    plot_mean_and_ci(ax, steps, mean, upper_ci - mean, "b", "Mean")

    # Plot median line
    plot_median_line(ax, steps, median)

    # Set up plot aesthetics
    setup_plot_aesthetics(ax, "Population Trends Across Simulations")

    # Set axis labels
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Population")

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Population trends plot saved to {output_path}")


def plot_population_trends_by_agent_type(
    experiment_data: Dict[str, Dict], output_dir: str
):
    """
    Plot population trends by agent type.

    Args:
        experiment_data: Dictionary with experiment data by agent type
        output_dir: Directory to save the plots
    """
    if not experiment_data:
        logger.warning("No experiment data to plot")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot data for each agent type
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for i, (agent_type, data) in enumerate(experiment_data.items()):
        populations = data.get("populations", [])
        max_steps = data.get("max_steps", 0)

        if not populations or max_steps == 0:
            continue

        # Create DataFrame from population data
        df = create_population_df(populations, max_steps)

        # Calculate statistics
        mean, median, lower_ci, upper_ci = calculate_statistics(df)

        # Plot mean and confidence interval
        steps = np.arange(len(mean))
        color = colors[i % len(colors)]
        plot_mean_and_ci(ax, steps, mean, upper_ci - mean, color, agent_type)

    # Set up plot aesthetics
    setup_plot_aesthetics(ax, "Population Trends by Agent Type")

    # Set axis labels
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Population")

    # Save plot
    output_path = os.path.join(output_dir, "population_trends_by_agent_type.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Population trends by agent type plot saved to {output_path}")


def plot_resource_consumption_trends(experiment_data: Dict[str, Dict], output_dir: str):
    """
    Plot resource consumption trends by agent type.

    Args:
        experiment_data: Dictionary with experiment data by agent type
        output_dir: Directory to save the plots
    """
    if not experiment_data:
        logger.warning("No experiment data to plot")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot data for each agent type
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for i, (agent_type, data) in enumerate(experiment_data.items()):
        consumption = data.get("consumption", [])
        max_steps = data.get("max_steps", 0)

        if not consumption or max_steps == 0:
            continue

        # Create DataFrame from consumption data
        df = create_population_df(consumption, max_steps, is_resource_data=True)

        # Calculate statistics
        mean, median, lower_ci, upper_ci = calculate_statistics(df)

        # Plot mean and confidence interval
        steps = np.arange(len(mean))
        color = colors[i % len(colors)]
        plot_mean_and_ci(ax, steps, mean, upper_ci - mean, color, agent_type)

    # Set up plot aesthetics
    setup_plot_aesthetics(ax, "Resource Consumption Trends by Agent Type")

    # Set axis labels
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Resource Consumption")

    # Save plot
    output_path = os.path.join(output_dir, "resource_consumption_trends.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Resource consumption trends plot saved to {output_path}")


def plot_early_termination_analysis(
    early_terminations: Dict[str, Dict], output_dir: str
):
    """
    Plot early termination analysis.

    Args:
        early_terminations: Dictionary with early termination data
        output_dir: Directory to save the plots
    """
    if not early_terminations:
        logger.info("No early terminations detected")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for plotting
    completion_percentages = []
    last_populations = []
    max_populations = []
    labels = []

    for db_path, data in early_terminations.items():
        completion_percentages.append(data["completion_percentage"])
        last_populations.append(data["last_population"])
        max_populations.append(data["max_population"])
        labels.append(os.path.basename(db_path))

    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # Plot completion percentages
    axes[0].bar(labels, completion_percentages)
    axes[0].set_title("Completion Percentage")
    axes[0].set_ylabel("Percentage")
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # Plot last populations
    axes[1].bar(labels, last_populations)
    axes[1].set_title("Last Population")
    axes[1].set_ylabel("Population")
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].grid(True, linestyle="--", alpha=0.7)

    # Plot max populations
    axes[2].bar(labels, max_populations)
    axes[2].set_title("Max Population")
    axes[2].set_ylabel("Population")
    axes[2].set_xticklabels(labels, rotation=45, ha="right")
    axes[2].grid(True, linestyle="--", alpha=0.7)

    # Save plot
    output_path = os.path.join(output_dir, "early_termination_analysis.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Early termination analysis plot saved to {output_path}")


def plot_final_agent_counts(final_counts: Dict[str, Dict], output_dir: str):
    """
    Plot final agent counts.

    Args:
        final_counts: Dictionary with final agent count data
        output_dir: Directory to save the plots
    """
    if not final_counts:
        logger.warning("No final agent count data to plot")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for plotting
    agent_types = list(final_counts.keys())
    means = [data["mean"] for data in final_counts.values()]
    lower_cis = [data["lower_ci"] for data in final_counts.values()]
    upper_cis = [data["upper_ci"] for data in final_counts.values()]
    survival_rates = [data["survival_rate"] for data in final_counts.values()]

    # Calculate error bars
    yerr = np.array(
        [
            [mean - lower for mean, lower in zip(means, lower_cis)],
            [upper - mean for mean, upper in zip(means, upper_cis)],
        ]
    )

    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # Plot mean final counts with confidence intervals
    axes[0].bar(agent_types, means, yerr=yerr, capsize=5)
    axes[0].set_title("Mean Final Agent Counts with 95% Confidence Intervals")
    axes[0].set_ylabel("Agent Count")
    axes[0].grid(True, linestyle="--", alpha=0.7)

    # Plot survival rates
    axes[1].bar(agent_types, survival_rates)
    axes[1].set_title("Survival Rates")
    axes[1].set_ylabel("Survival Rate (%)")
    axes[1].grid(True, linestyle="--", alpha=0.7)

    # Save plot
    output_path = os.path.join(output_dir, "final_agent_counts.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Final agent counts plot saved to {output_path}")


def plot_rewards_by_generation(
    rewards_data: Dict[str, Dict[int, float]], output_dir: str
):
    """
    Plot rewards by generation.

    Args:
        rewards_data: Dictionary with rewards data
        output_dir: Directory to save the plots
    """
    if not rewards_data or not rewards_data.get("simulations"):
        logger.warning("No rewards data to plot")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot data for each simulation
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for i, (sim_name, rewards) in enumerate(rewards_data["simulations"].items()):
        if not rewards:
            continue

        # Extract generations and rewards
        generations = list(rewards.keys())
        reward_values = list(rewards.values())

        # Plot rewards
        color = colors[i % len(colors)]
        ax.plot(generations, reward_values, color=color, label=sim_name)

    # Set up plot aesthetics
    setup_plot_aesthetics(ax, "Rewards by Generation")

    # Set axis labels
    ax.set_xlabel("Generation")
    ax.set_ylabel("Average Reward")

    # Save plot
    output_path = os.path.join(output_dir, "rewards_by_generation.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Rewards by generation plot saved to {output_path}")


def plot_action_distributions(
    action_data: Dict[str, Dict[str, Dict[str, float]]], output_dir: str
):
    """
    Plot action distributions.

    Args:
        action_data: Dictionary with action distribution data
        output_dir: Directory to save the plots
    """
    if not action_data or not action_data.get("aggregated"):
        logger.warning("No action distribution data to plot")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot aggregated data for each agent type
    for agent_type, actions in action_data["aggregated"].items():
        if not actions:
            continue

        # Extract actions and percentages
        action_types = list(actions.keys())
        percentages = list(actions.values())

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot action distribution
        ax.bar(action_types, percentages)

        # Set up plot aesthetics
        setup_plot_aesthetics(ax, f"Action Distribution for {agent_type}")

        # Set axis labels
        ax.set_xlabel("Action Type")
        ax.set_ylabel("Percentage (%)")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right")

        # Save plot
        output_path = os.path.join(output_dir, f"action_distribution_{agent_type}.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Action distribution plot for {agent_type} saved to {output_path}")


def plot_resource_level_trends(resource_level_data: Dict[str, List], output_path: str):
    """
    Plot resource level trends.

    Args:
        resource_level_data: Dictionary with resource level data
        output_path: Path to save the plot
    """
    resource_levels = resource_level_data.get("resource_levels", [])
    max_steps = resource_level_data.get("max_steps", 0)

    if not resource_levels or max_steps == 0:
        logger.warning("No resource level data to plot")
        return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame from resource level data
    df = create_population_df(resource_levels, max_steps, is_resource_data=True)

    # Calculate statistics
    mean, median, lower_ci, upper_ci = calculate_statistics(df)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot mean and confidence interval
    steps = np.arange(len(mean))
    plot_mean_and_ci(ax, steps, mean, upper_ci - mean, "b", "Mean")

    # Plot median line
    plot_median_line(ax, steps, median)

    # Set up plot aesthetics
    setup_plot_aesthetics(ax, "Resource Level Trends")

    # Set axis labels
    ax.set_xlabel("Simulation Step")
    ax.set_ylabel("Resource Level")

    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Resource level trends plot saved to {output_path}")

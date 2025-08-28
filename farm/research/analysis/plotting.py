import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from farm.research.analysis.dataframes import create_population_df
from farm.research.analysis.util import calculate_statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def plot_mean_and_ci(ax, steps, mean, ci, color, label):
    """Plot mean line with confidence interval."""
    ax.plot(steps, mean, color=color, label=label, linewidth=2)
    ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.2)


def plot_median_line(ax, steps, median, color="g", style="--"):
    """Plot median line."""
    ax.plot(
        steps,
        median,
        color=color,
        linestyle=style,
        label="Median Population",
        linewidth=2,
    )


def plot_reference_line(ax, y_value, label, color="orange"):
    """Plot horizontal reference line."""
    ax.axhline(
        y=y_value,
        color=color,
        linestyle=":",
        alpha=0.8,
        label=f"{label}: {y_value:.1f}",
        linewidth=2,
    )


def plot_marker_point(ax, x, y, label):
    """Plot marker point with label."""
    ax.plot(x, y, "rx", markersize=10, label=label)


def setup_plot_aesthetics(ax, title, experiment_name=None):
    """Setup common plot aesthetics."""
    if experiment_name:
        ax.set_title(experiment_name, fontsize=12, pad=10)
    ax.set_xlabel("Simulation Step", fontsize=12)
    ax.set_ylabel("Number of Agents", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


# ---------------------
# Plotting Functions
# ---------------------


def plot_population_trends_across_simulations(
    all_populations: List[np.ndarray], max_steps: int, output_path: Union[str, Path]
):
    """Plot population trends using modularized plotting functions."""
    # Ensure output directory exists
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory {output_path.parent}: {str(e)}")
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    experiment_name = output_path.parent.name
    fig.suptitle(
        f"Population Trends Across All Simulations (N={len(all_populations)})",
        fontsize=14,
        y=0.95,
    )

    # Create DataFrame and calculate statistics
    df = create_population_df(all_populations, max_steps)
    mean_pop, median_pop, std_pop, confidence_interval = calculate_statistics(df)
    steps = np.arange(max_steps)

    # Calculate key metrics
    overall_median = np.nanmedian(median_pop)
    final_median = median_pop[-1]
    peak_step = np.nanargmax(mean_pop)
    peak_value = mean_pop[peak_step]

    # Use helper functions for plotting
    plot_mean_and_ci(ax, steps, mean_pop, confidence_interval, "b", "Mean Population")
    plot_median_line(ax, steps, median_pop)
    plot_reference_line(ax, overall_median, "Overall Median")
    plot_marker_point(ax, peak_step, peak_value, f"Peak at step {peak_step}")
    plot_marker_point(
        ax, max_steps - 1, final_median, f"Final Median: {final_median:.1f}"
    )

    setup_plot_aesthetics(ax, None, experiment_name)

    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    except Exception as e:
        logger.error(f"Error saving plot to {output_path}: {str(e)}")
    finally:
        plt.close()


def plot_population_trends_by_agent_type(
    experiment_data: Dict[str, Dict], output_dir: Union[str, Path]
):
    """Plot population trends comparison using modularized plotting functions."""
    # Ensure output directory exists
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {str(e)}")
        return

    # Check if we have any valid data to plot
    valid_data = False
    for data in experiment_data.values():
        if data["populations"] and len(data["populations"]) > 0:
            valid_data = True
            break

    if not valid_data:
        logger.error("No valid data to plot for any agent type")
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    fig.suptitle("Population Trends Comparison by Agent Type", fontsize=14, y=0.95)

    colors = {"system": "blue", "control": "green", "independent": "red"}

    for agent_type, data in experiment_data.items():
        if not data["populations"]:
            logger.warning(f"Skipping {agent_type} - no valid data")
            continue

        # Create DataFrame and calculate statistics for this agent type
        df = create_population_df(data["populations"], data["max_steps"])
        mean_pop, _, _, confidence_interval = calculate_statistics(df)
        steps = np.arange(data["max_steps"])

        display_name = agent_type.replace("_", " ").title()
        plot_mean_and_ci(
            ax,
            steps,
            mean_pop,
            confidence_interval,
            colors[agent_type],
            f"{display_name} Agent (n={len(data['populations'])})",
        )

    setup_plot_aesthetics(ax, None)

    output_path = output_dir / "population_trends_comparison.png"
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    except Exception as e:
        logger.error(f"Error saving plot to {output_path}: {str(e)}")
    finally:
        plt.close()


def plot_resource_consumption_trends(
    experiment_data: Dict[str, Dict], output_dir: Union[str, Path]
):
    """Plot resource consumption trends with separate subplots for each agent type."""
    # Ensure output directory exists
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {str(e)}")
        return

    # Check if we have any valid data to plot
    valid_data = False
    agent_types_with_data = []

    # Determine which agent types have valid data
    for agent_type in ["system", "control", "independent"]:
        data = experiment_data.get(agent_type, {"consumption": []})
        if data.get("consumption") and len(data["consumption"]) > 0:
            # Check if there's actual consumption (not all zeros)
            has_consumption = False
            for consumption in data["consumption"]:
                if (
                    np.sum(consumption) > 0.01
                ):  # Small threshold to account for floating point
                    has_consumption = True
                    break

            if has_consumption:
                valid_data = True
                agent_types_with_data.append(agent_type)
            else:
                logger.warning(f"Skipping {agent_type} - zero consumption")

    if not valid_data:
        logger.error("No valid consumption data to plot for any agent type")
        return

    # Create figure with dynamic number of subplots based on available data
    num_agent_types = len(agent_types_with_data)
    fig, axes = plt.subplots(
        num_agent_types, 1, figsize=(15, 5 * num_agent_types), sharex=True
    )

    # Handle case where there's only one subplot (axes is not an array)
    if num_agent_types == 1:
        axes = [axes]

    fig.suptitle("Resource Consumption Trends by Agent Type", fontsize=16, y=0.98)

    # Define colors for agent types
    colors = {"system": "blue", "control": "green", "independent": "red"}

    # Find maximum steps across all data
    max_steps = max(
        [data["max_steps"] for data in experiment_data.values() if "max_steps" in data]
    )
    steps = np.arange(max_steps)

    # Process and plot each agent type in its own subplot
    for i, agent_type in enumerate(agent_types_with_data):
        ax = axes[i]
        data = experiment_data.get(agent_type, {"consumption": []})

        # Calculate average consumption across all simulations
        all_consumption = []
        for consumption in data["consumption"]:
            # Pad shorter arrays to match max_steps
            padded = np.zeros(max_steps)
            padded[: len(consumption)] = consumption
            all_consumption.append(padded)

        # Convert to numpy array for easier calculations
        all_consumption = np.array(all_consumption)

        # Calculate mean and standard deviation
        mean_consumption = np.mean(all_consumption, axis=0)
        std_consumption = np.std(all_consumption, axis=0)

        # Apply smoothing for better visualization
        window_size = min(51, max_steps // 20)
        if window_size % 2 == 0:  # Ensure window size is odd
            window_size += 1

        try:
            from scipy.signal import savgol_filter

            # Use savgol_filter with appropriate window size and polynomial order
            smoothed_mean = savgol_filter(mean_consumption, window_size, 3)
            smoothed_std = savgol_filter(std_consumption, window_size, 3)
        except (ImportError, ValueError):
            # Fall back to simple moving average if scipy is not available or window size issues
            kernel = np.ones(window_size) / window_size
            smoothed_mean = np.convolve(mean_consumption, kernel, mode="same")
            smoothed_std = np.convolve(std_consumption, kernel, mode="same")

        # Plot mean line
        ax.plot(
            steps,
            smoothed_mean,
            color=colors[agent_type],
            linewidth=2,
            label=f"Mean Consumption",
        )

        # Plot confidence interval
        ax.fill_between(
            steps,
            smoothed_mean - smoothed_std,
            smoothed_mean + smoothed_std,
            color=colors[agent_type],
            alpha=0.2,
            label="±1 Std Dev",
        )

        # Calculate and display statistics
        avg_consumption = np.mean(smoothed_mean)
        max_consumption = np.max(smoothed_mean)
        total_consumption = np.sum(smoothed_mean)  # Total consumption across all steps

        # Calculate average consumption per simulation
        avg_per_simulation = total_consumption / len(data["consumption"])

        # Add statistics as text with total consumption
        stats_text = (
            f"Average: {avg_consumption:.2f}\n"
            f"Maximum: {max_consumption:.2f}\n"
            f"Total Consumed: {total_consumption:.2f}\n"
            f"Avg Total per Sim: {avg_per_simulation:.2f}\n"
            f"Simulations: {len(data['consumption'])}"
        )

        ax.text(
            0.02,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        # Set title and labels
        ax.set_title(f"{agent_type.title()} Agent Consumption", fontsize=12)
        ax.set_ylabel("Resources Consumed")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

        # Set y-axis to start from 0 with a small margin
        y_max = np.max(smoothed_mean + smoothed_std) * 1.1  # Add 10% margin
        ax.set_ylim(bottom=0, top=y_max)

    # Set common x-axis label on the bottom subplot
    axes[-1].set_xlabel("Simulation Step")

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(
        top=0.95 - 0.02 * num_agent_types
    )  # Adjust top margin based on number of subplots

    # Save the figure
    output_path = output_dir / "resource_consumption_trends.png"
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved resource consumption plot to {output_path}")
    except Exception as e:
        logger.error(f"Error saving plot to {output_path}: {str(e)}")
    finally:
        plt.close()


def plot_early_termination_analysis(
    early_terminations: Dict[str, Dict], output_dir: Union[str, Path]
):
    """
    Create visualizations for early termination analysis.

    Args:
        early_terminations: Dictionary of early termination data from detect_early_terminations()
        output_dir: Directory to save output plots
    """
    if not early_terminations:
        logger.warning("No early terminations to analyze")
        return

    # Ensure output directory exists
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {str(e)}")
        return

    # Extract data for plotting
    completion_percentages = [
        data["completion_percentage"] for data in early_terminations.values()
    ]
    causes = [data["likely_cause"] for data in early_terminations.values()]

    # 1. Create histogram of completion percentages
    plt.figure(figsize=(12, 6))
    plt.hist(completion_percentages, bins=20, color="skyblue", edgecolor="black")
    plt.title(
        "Distribution of Early Terminations by Completion Percentage", fontsize=14
    )
    plt.xlabel("Completion Percentage", fontsize=12)
    plt.ylabel("Number of Simulations", fontsize=12)
    plt.grid(alpha=0.3)

    # Add vertical line for the mean
    mean_completion = sum(completion_percentages) / len(completion_percentages)
    plt.axvline(
        mean_completion,
        color="red",
        linestyle="--",
        label=f"Mean: {mean_completion:.1f}%",
    )
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir / "early_termination_histogram.png", dpi=300)
    plt.close()

    # 2. Create pie chart of termination causes
    cause_counts = {}
    for cause in causes:
        cause_counts[cause] = cause_counts.get(cause, 0) + 1

    plt.figure(figsize=(10, 8))
    plt.pie(
        list(cause_counts.values()),
        labels=[f"{cause} ({count})" for cause, count in cause_counts.items()],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"],
    )
    plt.title("Causes of Early Termination", fontsize=14)

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_dir / "early_termination_causes.png", dpi=300)
    plt.close()

    # 3. Create scatter plot of final population vs. completion percentage
    plt.figure(figsize=(12, 8))

    # Extract data for each agent type
    data_points = []
    for data in early_terminations.values():
        for agent_type in ["system", "control", "independent"]:
            if data["final_populations"][agent_type] > 0:
                data_points.append(
                    {
                        "agent_type": agent_type,
                        "population": data["final_populations"][agent_type],
                        "completion": data["completion_percentage"],
                        "cause": data["likely_cause"],
                    }
                )

    # Create DataFrame for easier plotting
    df = pd.DataFrame(data_points)

    if not df.empty:
        # Define colors and markers for agent types and causes
        agent_colors = {"system": "blue", "control": "green", "independent": "red"}
        cause_markers = {
            "population_collapse": "x",
            "resource_depletion": "o",
            "unknown": "s",
        }

        # Plot each point
        for agent_type in df["agent_type"].unique():
            for cause in df["cause"].unique():
                subset = df[(df["agent_type"] == agent_type) & (df["cause"] == cause)]
                if not subset.empty:
                    plt.scatter(
                        subset["completion"],
                        subset["population"],
                        color=agent_colors[agent_type],
                        marker=cause_markers[cause],
                        alpha=0.7,
                        s=50,
                        label=f"{agent_type.title()} - {cause.replace('_', ' ').title()}",
                    )

        plt.title("Final Population vs. Completion Percentage", fontsize=14)
        plt.xlabel("Completion Percentage", fontsize=12)
        plt.ylabel("Final Population", fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_dir / "early_termination_population_scatter.png", dpi=300)

    plt.close()

    # 4. Create summary report
    with open(output_dir / "early_termination_summary.txt", "w") as f:
        f.write("Early Termination Analysis Summary\n")
        f.write("=================================\n\n")
        f.write(f"Total simulations analyzed: {len(early_terminations)}\n")
        f.write(f"Average completion percentage: {mean_completion:.1f}%\n\n")

        f.write("Termination causes:\n")
        for cause, count in cause_counts.items():
            f.write(
                f"  - {cause.replace('_', ' ').title()}: {count} ({count/len(early_terminations)*100:.1f}%)\n"
            )

        f.write("\nDetailed simulation data:\n")
        for i, (db_path, data) in enumerate(early_terminations.items()):
            f.write(f"\n{i+1}. Simulation: {Path(db_path).parent.name}\n")
            f.write(
                f"   Steps completed: {data['steps_completed']} / {data['expected_steps']} ({data['completion_percentage']}%)\n"
            )
            f.write(f"   Final population: {data['total_final_population']}\n")
            f.write(
                f"   Likely cause: {data['likely_cause'].replace('_', ' ').title()}\n"
            )

    logger.info(f"Early termination analysis saved to {output_dir}")


def plot_final_agent_counts(
    final_counts: Dict[str, Dict], output_dir: Union[str, Path]
):
    """
    Create visualizations for final agent counts analysis.

    Args:
        final_counts: Dictionary with final agent count statistics
        output_dir: Directory to save output plots
    """
    if not final_counts:
        logger.warning("No final count data to visualize")
        return

    # Ensure output directory exists
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {str(e)}")
        return

    # 1. Create bar chart of total final populations by agent type
    plt.figure(figsize=(12, 6))
    agent_types = ["system", "control", "independent"]
    totals = [final_counts[agent_type]["total"] for agent_type in agent_types]
    means = [final_counts[agent_type]["mean"] for agent_type in agent_types]

    # Create grouped bar chart
    x = np.arange(len(agent_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        totals,
        width,
        label="Total Final Population",
        color=["blue", "green", "red"],
    )

    # Add a second axis for the mean values
    ax2 = ax.twinx()
    ax2.bar(
        x + width / 2,
        means,
        width,
        label="Mean Final Population",
        color=["lightblue", "lightgreen", "lightcoral"],
    )

    # Add labels and title
    ax.set_xlabel("Agent Type", fontsize=12)
    ax.set_ylabel("Total Final Population", fontsize=12)
    ax2.set_ylabel("Mean Final Population", fontsize=12)
    ax.set_title("Final Agent Populations by Type", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([t.title() for t in agent_types])

    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Add simulation count as text
    for i, agent_type in enumerate(agent_types):
        ax.text(
            i,
            totals[i] + (max(totals) * 0.02),
            f"n={final_counts[agent_type]['simulations']}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(output_dir / "final_agent_populations.png", dpi=300)
    plt.close()

    # 2. Create pie chart of dominant agent types
    plt.figure(figsize=(10, 8))

    dominant_counts = final_counts["dominant_type_counts"]
    labels = [
        f"{t.title()} ({count})" for t, count in dominant_counts.items() if count > 0
    ]
    sizes = [count for count in dominant_counts.values() if count > 0]
    colors = ["blue", "green", "red", "gray"]

    if sum(sizes) > 0:  # Only create pie chart if there's data
        plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle
        plt.title("Dominant Agent Type Distribution", fontsize=14)

        plt.tight_layout()
        plt.savefig(output_dir / "dominant_agent_types.png", dpi=300)

    plt.close()

    # 3. Create a text summary file
    with open(output_dir / "final_agent_counts_summary.txt", "w") as f:
        f.write("Final Agent Counts Analysis\n")
        f.write("==========================\n\n")

        for agent_type in agent_types:
            stats = final_counts[agent_type]
            f.write(f"{agent_type.title()} Agents:\n")
            f.write(f"  Total across all simulations: {stats['total']:.1f}\n")
            f.write(f"  Mean per simulation: {stats['mean']:.2f}\n")
            f.write(f"  Median per simulation: {stats['median']:.1f}\n")
            f.write(f"  Maximum in any simulation: {stats['max']:.1f}\n")
            f.write(f"  Minimum in any simulation: {stats['min']:.1f}\n")
            f.write(f"  Number of simulations: {stats['simulations']}\n\n")

        f.write("Dominant Agent Type Distribution:\n")
        total_sims = sum(dominant_counts.values())
        for agent_type, count in dominant_counts.items():
            if count > 0:
                percentage = (count / total_sims) * 100
                f.write(
                    f"  {agent_type.title()}: {count} simulations ({percentage:.1f}%)\n"
                )


def plot_rewards_by_generation(
    rewards_data: Dict[str, Dict[int, float]], output_dir: Union[str, Path]
):
    """
    Create visualizations for rewards by generation.

    Args:
        rewards_data: Dictionary with reward data by generation for each agent type
        output_dir: Directory to save output plots
    """
    if not any(rewards_data[agent_type] for agent_type in rewards_data):
        logger.warning("No reward data by generation to visualize")
        return

    # Ensure output directory exists
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {str(e)}")
        return

    # Create figure
    plt.figure(figsize=(15, 8))
    plt.title("Average Rewards by Generation and Agent Type", fontsize=14)

    # Define colors for agent types
    colors = {"system": "blue", "control": "green", "independent": "red"}
    markers = {"system": "o", "control": "s", "independent": "^"}

    # Plot data for each agent type
    for agent_type, rewards in rewards_data.items():
        if not rewards:
            continue

        # Sort generations
        generations = sorted(rewards.keys())
        avg_rewards = [rewards[gen] for gen in generations]

        # Plot line
        plt.plot(
            generations,
            avg_rewards,
            color=colors[agent_type],
            marker=markers[agent_type],
            linestyle="-",
            linewidth=2,
            markersize=8,
            label=f"{agent_type.title()} Agents",
        )

        # Add trend line (polynomial fit)
        if len(generations) > 2:
            try:
                z = np.polyfit(generations, avg_rewards, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(min(generations), max(generations), 100)
                plt.plot(
                    x_trend,
                    p(x_trend),
                    color=colors[agent_type],
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1.5,
                )
            except Exception as e:
                logger.warning(
                    f"Could not create trend line for {agent_type}: {str(e)}"
                )

    # Set labels and grid
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Average Reward", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)

    # Add annotations for key points
    for agent_type, rewards in rewards_data.items():
        if not rewards:
            continue

        generations = sorted(rewards.keys())
        avg_rewards = [rewards[gen] for gen in generations]

        # Annotate maximum reward
        if avg_rewards:
            max_idx = np.argmax(avg_rewards)
            max_gen = generations[max_idx]
            max_reward = avg_rewards[max_idx]

            plt.annotate(
                f"Max: {max_reward:.2f}",
                xy=(max_gen, max_reward),
                xytext=(10, 10),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=colors[agent_type]),
                fontsize=9,
                color=colors[agent_type],
            )

    # Save the figure
    output_path = output_dir / "rewards_by_generation.png"
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved rewards by generation plot to {output_path}")
    except Exception as e:
        logger.error(f"Error saving plot to {output_path}: {str(e)}")
    finally:
        plt.close()

    # Create a text summary file
    with open(output_dir / "rewards_by_generation_summary.txt", "w") as f:
        f.write("Rewards by Generation Analysis\n")
        f.write("============================\n\n")

        for agent_type in ["system", "control", "independent"]:
            rewards = rewards_data[agent_type]
            f.write(f"{agent_type.title()} Agents:\n")

            if not rewards:
                f.write("  No reward data available\n\n")
                continue

            generations = sorted(rewards.keys())
            avg_rewards = [rewards[gen] for gen in generations]

            if avg_rewards:
                max_idx = np.argmax(avg_rewards)
                min_idx = np.argmin(avg_rewards)

                f.write(f"  Generations analyzed: {len(generations)}\n")
                f.write(
                    f"  Maximum reward: {avg_rewards[max_idx]:.4f} (Generation {generations[max_idx]})\n"
                )
                f.write(
                    f"  Minimum reward: {avg_rewards[min_idx]:.4f} (Generation {generations[min_idx]})\n"
                )

                if len(generations) > 1:
                    first_gen = generations[0]
                    last_gen = generations[-1]
                    first_reward = rewards[first_gen]
                    last_reward = rewards[last_gen]
                    change = (
                        ((last_reward - first_reward) / first_reward) * 100
                        if first_reward != 0
                        else float("inf")
                    )

                    f.write(
                        f"  First generation ({first_gen}) reward: {first_reward:.4f}\n"
                    )
                    f.write(
                        f"  Last generation ({last_gen}) reward: {last_reward:.4f}\n"
                    )
                    f.write(f"  Change from first to last: {change:.2f}%\n")

                f.write("\n  Generation -> Reward mapping:\n")
                for gen in generations:
                    f.write(f"    Generation {gen}: {rewards[gen]:.4f}\n")

            f.write("\n")


def plot_action_distributions(
    action_data: Dict[str, Dict[str, Dict[str, float]]], output_dir: Union[str, Path]
):
    """
    Create visualizations for action distributions by agent type.

    Args:
        action_data: Dictionary with action distribution data
        output_dir: Directory to save output plots
    """
    if not action_data:
        logger.warning("No action distribution data to visualize")
        return

    # Ensure output directory exists
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory {output_dir}: {str(e)}")
        return

    # Create a figure with subplots for each agent type
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Action Distributions by Agent Type", fontsize=16)

    agent_types = ["system", "control", "independent"]
    colors = {
        "move": "blue",
        "attack": "red",
        "defend": "green",
        "share": "purple",
        "reproduce": "orange",
        "eat": "brown",
        "rest": "gray",
    }

    # Process each agent type
    for i, agent_type in enumerate(agent_types):
        ax = axes[i]
        data = action_data[agent_type]

        if not data["actions"] or data["total_actions"] == 0:
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            ax.set_title(f"{agent_type.title()} Agents")
            continue

        # Sort actions by frequency
        sorted_actions = sorted(
            data["actions"].items(), key=lambda x: x[1], reverse=True
        )
        actions = [a[0] for a in sorted_actions]
        percentages = [a[1] * 100 for a in sorted_actions]  # Convert to percentages

        # Create bar colors
        bar_colors = [colors.get(action, "lightgray") for action in actions]

        # Create the bar chart
        bars = ax.bar(range(len(actions)), percentages, color=bar_colors)

        # Fix the x-axis ticks and labels - this addresses the warning
        ax.set_xticks(range(len(actions)))
        ax.set_xticklabels(actions, rotation=45, ha="right")

        # Add percentage labels on top of bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{percentage:.1f}%",
                ha="center",
                va="bottom",
                rotation=0,
                fontsize=8,
            )

        # Set title and labels
        ax.set_title(f"{agent_type.title()} Agents")
        ax.set_ylabel("Percentage of Actions (%)")
        ax.set_ylim(0, max(percentages) * 1.15)  # Add some space for labels

        # Add total actions count as text
        ax.text(
            0.5,
            0.95,
            f"Total Actions: {data['total_actions']:,}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for the suptitle

    # Save the figure
    output_path = output_dir / "action_distributions.png"
    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved action distribution plot to {output_path}")
    except Exception as e:
        logger.error(f"Error saving plot to {output_path}: {str(e)}")
    finally:
        plt.close()

    # Create a text summary file
    with open(output_dir / "action_distribution_summary.txt", "w") as f:
        f.write("Action Distribution Analysis\n")
        f.write("==========================\n\n")

        for agent_type in agent_types:
            data = action_data[agent_type]
            f.write(f"{agent_type.title()} Agents:\n")
            f.write(f"  Total actions: {data['total_actions']:,}\n")

            if data["actions"]:
                f.write("  Action breakdown:\n")
                sorted_actions = sorted(
                    data["actions"].items(), key=lambda x: x[1], reverse=True
                )
                for action, percentage in sorted_actions:
                    f.write(f"    {action}: {percentage*100:.2f}%\n")
            else:
                f.write("  No action data available\n")

            f.write("\n")


def plot_resource_level_trends(
    resource_level_data: Dict[str, Any], output_path: Union[str, Path]
):
    """
    Plot resource level trends with confidence intervals.

    Args:
        resource_level_data: Dictionary containing resource level data
        output_path: Path to save the output plot
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory {output_path.parent}: {str(e)}")
        return

    # Check if we have valid data
    if not resource_level_data["resource_levels"]:
        logger.error("No valid resource level data to plot")
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    experiment_name = output_path.parent.name
    fig.suptitle(
        f"Resource Level Trends Across All Simulations (N={len(resource_level_data['resource_levels'])})",
        fontsize=14,
        y=0.95,
    )

    # Create DataFrame and calculate statistics - specify this is resource data
    df = create_population_df(
        resource_level_data["resource_levels"],
        resource_level_data["max_steps"],
        is_resource_data=True,
    )
    mean_resources, median_resources, std_resources, confidence_interval = (
        calculate_statistics(df)
    )
    steps = np.arange(resource_level_data["max_steps"])

    # Check if we have valid statistics
    if len(mean_resources) == 0:
        logger.error("No valid statistics could be calculated")
        plt.close()
        return

    # Calculate key metrics with safety checks
    overall_median = np.nanmedian(median_resources) if len(median_resources) > 0 else 0
    final_median = median_resources[-1] if len(median_resources) > 0 else 0

    # Handle empty arrays for peak calculation
    if len(mean_resources) > 0:
        peak_step = np.nanargmax(mean_resources)
        peak_value = mean_resources[peak_step]
    else:
        peak_step = 0
        peak_value = 0

    # Use helper functions for plotting
    plot_mean_and_ci(
        ax, steps, mean_resources, confidence_interval, "purple", "Mean Resource Level"
    )
    plot_median_line(ax, steps, median_resources, color="darkgreen")
    plot_reference_line(ax, overall_median, "Overall Median", color="teal")
    plot_marker_point(ax, peak_step, peak_value, f"Peak at step {peak_step}")
    plot_marker_point(
        ax,
        resource_level_data["max_steps"] - 1,
        final_median,
        f"Final Median: {final_median:.1f}",
    )

    # Setup plot aesthetics
    ax.set_title(experiment_name, fontsize=12, pad=10)
    ax.set_xlabel("Simulation Step", fontsize=12)
    ax.set_ylabel("Average Agent Resource Level", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Ensure y-axis starts at 0 unless we have negative values
    if len(mean_resources) > 0 and np.min(mean_resources - confidence_interval) >= 0:
        ax.set_ylim(bottom=0)

    try:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved resource level plot to {output_path}")
    except Exception as e:
        logger.error(f"Error saving plot to {output_path}: {str(e)}")
    finally:
        plt.close()

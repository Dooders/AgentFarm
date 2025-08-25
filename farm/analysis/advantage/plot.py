"""
Advantage Plotting Module

This module provides functions to visualize advantage data and analysis results,
helping to understand the relationships between different types of advantages and dominance.
"""

import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_advantage_results(
    df: pd.DataFrame,
    analysis_results: Dict[str, Any],
    output_path: str,
    data_cleaned: bool = False,
):
    """
    Generate visualizations for advantage analysis results.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing advantage metrics for all simulations
    analysis_results : Dict
        Results from analyze_advantage_patterns
    output_path : str
        Directory where output files will be saved
    data_cleaned : bool, optional
        Flag indicating whether the data has been cleaned of NaN and infinity values
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # 1. Correlation matrix between advantages and dominance scores
    plot_advantage_correlation_matrix(df, output_path, data_cleaned)

    # 2. Advantage category importance by agent type
    plot_advantage_category_importance(analysis_results, output_path, data_cleaned)

    # 3. Top predictors of dominance for each agent type
    plot_top_dominance_predictors(analysis_results, output_path, data_cleaned)

    # 4. Advantage thresholds and dominance
    plot_advantage_thresholds(analysis_results, output_path, data_cleaned)

    # 5. Timing of advantages across simulation phases
    plot_advantage_timing(analysis_results, output_path, data_cleaned)

    # 6. Composite advantage breakdown
    plot_composite_advantage_breakdown(df, output_path, data_cleaned)

    # 7. Advantage trajectory over simulation
    plot_advantage_trajectories(df, output_path, data_cleaned)


def add_data_cleaning_watermark(fig, ax=None):
    """Add a watermark to indicate data has been cleaned."""
    if ax is None:
        ax = plt.gca()
    fig.text(
        0.5,
        0.01,
        "* Data has been cleaned of NaN and Infinity values for visualization purposes",
        ha="center",
        va="bottom",
        alpha=0.7,
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.1),
    )


def plot_advantage_correlation_matrix(
    df: pd.DataFrame, output_path: str, data_cleaned: bool = False
):
    """
    Plot correlation matrix between advantages and dominance scores.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing advantage metrics for all simulations
    output_path : str
        Directory where output files will be saved
    data_cleaned : bool, optional
        Flag indicating whether the data has been cleaned of NaN and infinity values
    """
    if df.empty:
        logging.warning("Empty DataFrame, skipping correlation matrix plot")
        return

    # Find relevant columns for correlation analysis
    advantage_cols = [
        col
        for col in df.columns
        if (
            ("advantage" in col or "trajectory" in col)
            and "composite" not in col
            and "ratio" not in col
        )
    ]

    dominance_cols = [
        "system_dominance_score",
        "independent_dominance_score",
        "control_dominance_score",
    ]

    # Limit to top 15 advantage columns to keep plot readable
    if len(advantage_cols) > 15:
        # Compute correlations with dominance scores to find most important advantages
        top_cols = set()
        for dom_col in dominance_cols:
            if dom_col in df.columns:
                corrs = (
                    df[advantage_cols + [dom_col]]
                    .corr()[dom_col]
                    .abs()
                    .sort_values(ascending=False)
                )
                top_cols.update(corrs.index[:5])  # Top 5 for each dominance score

        # Get the union of top advantages for all dominance scores, up to 15
        advantage_cols = list(top_cols.intersection(advantage_cols))[:15]

    if not advantage_cols or not all(col in df.columns for col in dominance_cols):
        logging.warning("No suitable columns for correlation matrix")
        return

    # Calculate correlation matrix
    corr_matrix = df[advantage_cols + dominance_cols].corr()

    # Plot correlation matrix
    fig = plt.figure(figsize=(12, 10))
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Mask upper triangle to avoid duplication

    # Plot only correlations with dominance scores
    mask_dominance = mask.copy()
    for i, col in enumerate(corr_matrix.columns):
        if col not in dominance_cols:
            mask_dominance[i, :] = True
            for j, row_col in enumerate(corr_matrix.index):
                if row_col not in dominance_cols:
                    mask_dominance[i, j] = True

    # Create a custom colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Plot the correlation matrix
    sns.heatmap(
        corr_matrix,
        mask=mask_dominance,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
    )

    # Format the axis labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    plt.title("Correlation Between Advantages and Dominance Scores", fontsize=14)
    plt.tight_layout()

    # Add watermark if data was cleaned
    if data_cleaned:
        add_data_cleaning_watermark(fig)

    # Save the figure
    output_file = os.path.join(output_path, "advantage_correlation_matrix.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved advantage correlation matrix to {output_file}")


def plot_advantage_category_importance(
    analysis_results: Dict[str, Any], output_path: str, data_cleaned: bool = False
):
    """
    Plot the importance of different advantage categories for each agent type.

    Parameters
    ----------
    analysis_results : Dict
        Results from analyze_advantage_patterns
    output_path : str
        Directory where output files will be saved
    data_cleaned : bool, optional
        Flag indicating whether the data has been cleaned of NaN and infinity values
    """
    if not analysis_results or "advantage_category_importance" not in analysis_results:
        logging.warning("No category importance data to plot")
        return

    category_data = analysis_results["advantage_category_importance"]["by_category"]

    # Prepare data for plotting
    categories = []
    system_scores = []
    independent_scores = []
    control_scores = []

    for category, agent_data in category_data.items():
        categories.append(category.replace("_", " ").title())

        system_scores.append(agent_data.get("system", {}).get("average_relevance", 0))
        independent_scores.append(
            agent_data.get("independent", {}).get("average_relevance", 0)
        )
        control_scores.append(agent_data.get("control", {}).get("average_relevance", 0))

    # Sort by overall importance
    if categories:
        sort_idx = np.argsort(
            [
                s + i + c
                for s, i, c in zip(system_scores, independent_scores, control_scores)
            ]
        )[::-1]
        categories = [categories[i] for i in sort_idx]
        system_scores = [system_scores[i] for i in sort_idx]
        independent_scores = [independent_scores[i] for i in sort_idx]
        control_scores = [control_scores[i] for i in sort_idx]

    # Create the plot
    fig = plt.figure(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.25

    plt.bar(x - width, system_scores, width, label="System", color="#1f77b4")
    plt.bar(x, independent_scores, width, label="Independent", color="#ff7f0e")
    plt.bar(x + width, control_scores, width, label="Control", color="#2ca02c")

    plt.xlabel("Advantage Category")
    plt.ylabel("Importance for Dominance")
    plt.title("Importance of Advantage Categories by Agent Type")
    plt.xticks(x, categories, rotation=45, ha="right")
    plt.legend()

    plt.tight_layout()

    # Add watermark if data was cleaned
    if data_cleaned:
        add_data_cleaning_watermark(fig)

    # Save the figure
    output_file = os.path.join(output_path, "advantage_category_importance.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved advantage category importance to {output_file}")


def plot_top_dominance_predictors(
    analysis_results: Dict[str, Any], output_path: str, data_cleaned: bool = False
):
    """
    Plot the top predictors of dominance for each agent type.

    Parameters
    ----------
    analysis_results : Dict
        Results from analyze_advantage_patterns
    output_path : str
        Directory where output files will be saved
    data_cleaned : bool, optional
        Flag indicating whether the data has been cleaned of NaN and infinity values
    """
    if not analysis_results or "agent_type_specific_analysis" not in analysis_results:
        logging.warning("No agent-specific analysis data to plot")
        return

    agent_type_data = analysis_results["agent_type_specific_analysis"]

    # Plot for each agent type
    for agent_type, data in agent_type_data.items():
        if "top_predictors" not in data or not data["top_predictors"]:
            continue

        # Get top 5 predictors
        top_predictors = list(data["top_predictors"].items())[:5]

        # Prepare data for plotting
        advantage_labels = []
        effect_sizes = []
        significant = []

        for adv_col, pred_data in top_predictors:
            # Format advantage label for readability
            label = adv_col.replace("_", " ")
            for cat in [
                "resource acquisition",
                "reproduction",
                "survival",
                "population growth",
                "combat",
                "initial positioning",
            ]:
                if cat in label:
                    parts = label.split(cat)
                    # Keep only the most informative part
                    label = f"{cat} - {parts[1].split('advantage')[0].strip()}"
                    break

            advantage_labels.append(label)
            effect_sizes.append(pred_data["effect_size"])
            significant.append(pred_data["significant"])

        # Create the plot
        fig = plt.figure(figsize=(10, 6))

        # Sort by absolute effect size
        sort_idx = np.argsort([abs(e) for e in effect_sizes])[::-1]
        advantage_labels = [advantage_labels[i] for i in sort_idx]
        effect_sizes = [effect_sizes[i] for i in sort_idx]
        significant = [significant[i] for i in sort_idx]

        # Create bars with different colors for significant vs. non-significant
        colors = ["#1f77b4" if sig else "#d62728" for sig in significant]
        plt.barh(range(len(advantage_labels)), effect_sizes, color=colors)

        plt.yticks(range(len(advantage_labels)), advantage_labels)
        plt.axvline(x=0, color="black", linestyle="-")

        plt.xlabel("Effect Size (Correlation with Dominance)")
        plt.ylabel("Advantage Type")
        plt.title(f"Top Predictors of {agent_type.capitalize()} Dominance")

        # Add a legend for significance
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#1f77b4", label="Significant (p<0.05)"),
            Patch(facecolor="#d62728", label="Non-significant"),
        ]
        plt.legend(handles=legend_elements, loc="best")

        plt.tight_layout()

        # Add watermark if data was cleaned
        if data_cleaned:
            add_data_cleaning_watermark(fig)

        # Save the figure
        output_file = os.path.join(
            output_path, f"{agent_type}_dominance_predictors.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved {agent_type} dominance predictors to {output_file}")


def plot_advantage_thresholds(
    analysis_results: Dict[str, Any], output_path: str, data_cleaned: bool = False
):
    """
    Plot advantage thresholds and their relationship with dominance likelihood.

    Parameters
    ----------
    analysis_results : Dict
        Results from analyze_advantage_patterns
    output_path : str
        Directory where output files will be saved
    data_cleaned : bool, optional
        Flag indicating whether the data has been cleaned of NaN and infinity values
    """
    if not analysis_results or "advantage_threshold_analysis" not in analysis_results:
        logging.warning("No threshold analysis data to plot")
        return

    threshold_data = analysis_results["advantage_threshold_analysis"]

    # Plot for each agent type
    for agent_type, thresholds in threshold_data.items():
        if not thresholds:
            continue

        # Get top 3 thresholds with highest dominance ratios
        top_thresholds = sorted(
            thresholds.items(), key=lambda x: x[1]["dominance_ratio"], reverse=True
        )[:3]

        if not top_thresholds:
            continue

        # Create a figure with multiple panels, one for each threshold
        fig, axes = plt.subplots(
            len(top_thresholds), 1, figsize=(10, 4 * len(top_thresholds))
        )
        if len(top_thresholds) == 1:
            axes = [axes]

        for i, (adv_col, threshold_info) in enumerate(top_thresholds):
            ax = axes[i]

            # Format advantage label for readability
            label = adv_col.replace("_", " ")
            for cat in [
                "resource acquisition",
                "reproduction",
                "survival",
                "population growth",
                "combat",
                "initial positioning",
            ]:
                if cat in label:
                    parts = label.split(cat)
                    # Keep only the most informative part
                    label = f"{cat} - {parts[1].split('advantage')[0].strip()}"
                    break

            # Extract threshold data points
            thresholds = [t["threshold"] for t in threshold_info["all_thresholds"]]
            ratios = [t["dominance_ratio"] for t in threshold_info["all_thresholds"]]

            # Plot threshold vs. dominance ratio
            ax.plot(thresholds, ratios, "o-", color="#1f77b4")

            # Highlight optimal threshold
            optimal = threshold_info["optimal_threshold"]
            optimal_ratio = threshold_info["dominance_ratio"]
            ax.plot(optimal, optimal_ratio, "o", color="red", markersize=10)

            # Add a horizontal line at ratio=1 (no effect)
            ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

            # Add labels and title
            ax.set_xlabel("Advantage Threshold")
            ax.set_ylabel("Dominance Likelihood Ratio")
            ax.set_title(
                f"{label} (Optimal Threshold: {optimal:.3f}, Ratio: {optimal_ratio:.2f}x)"
            )

            # Add grid
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Critical Advantage Thresholds for {agent_type.capitalize()} Dominance",
            fontsize=16,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.97))  # Adjust for suptitle

        # Add watermark if data was cleaned
        if data_cleaned:
            fig.text(
                0.5,
                0.01,
                "* Data has been cleaned of NaN and Infinity values for visualization purposes",
                ha="center",
                va="bottom",
                alpha=0.7,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.1),
            )

        # Save the figure
        output_file = os.path.join(
            output_path, f"{agent_type}_advantage_thresholds.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved {agent_type} advantage thresholds to {output_file}")


def plot_advantage_timing(
    analysis_results: Dict[str, Any], output_path: str, data_cleaned: bool = False
):
    """
    Plot the importance of advantages across different simulation phases.

    Parameters
    ----------
    analysis_results : Dict
        Results from analyze_advantage_patterns
    output_path : str
        Directory where output files will be saved
    data_cleaned : bool, optional
        Flag indicating whether the data has been cleaned of NaN and infinity values
    """
    if not analysis_results or "advantage_timing_analysis" not in analysis_results:
        logging.warning("No timing analysis data to plot")
        return

    timing_data = analysis_results["advantage_timing_analysis"]

    # Calculate the average strength of advantages for each agent type in each phase
    phase_strengths = {
        agent_type: {
            phase: sum(abs(adv["average_value"]) for adv in advantages.values())
            / max(len(advantages), 1)
            for phase, advantages in data.items()
            if advantages
        }
        for agent_type, data in timing_data.items()
    }

    # Prepare data for plotting
    phases = ["early", "mid", "late"]
    phase_labels = ["Early Phase", "Middle Phase", "Late Phase"]
    agent_types = list(phase_strengths.keys())

    # Create the plot
    fig = plt.figure(figsize=(10, 6))

    x = np.arange(len(phases))
    width = 0.25
    offsets = np.linspace(-width, width, len(agent_types))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, orange, green

    for i, agent_type in enumerate(agent_types):
        values = [phase_strengths[agent_type].get(phase, 0) for phase in phases]
        plt.bar(
            x + offsets[i],
            values,
            width,
            label=f"{agent_type.capitalize()}",
            color=colors[i % len(colors)],
        )

    plt.xlabel("Simulation Phase")
    plt.ylabel("Average Advantage Strength")
    plt.title("Advantage Strength by Simulation Phase")
    plt.xticks(x, phase_labels)
    plt.legend()

    plt.tight_layout()

    # Add watermark if data was cleaned
    if data_cleaned:
        add_data_cleaning_watermark(fig)

    # Save the figure
    output_file = os.path.join(output_path, "advantage_timing_analysis.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved advantage timing analysis to {output_file}")


def plot_composite_advantage_breakdown(
    df: pd.DataFrame, output_path: str, data_cleaned: bool = False
):
    """
    Plot breakdown of components contributing to composite advantage scores.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing advantage metrics for all simulations
    output_path : str
        Directory where output files will be saved
    data_cleaned : bool, optional
        Flag indicating whether the data has been cleaned of NaN and infinity values
    """
    if df.empty:
        logging.warning("Empty DataFrame, skipping composite advantage breakdown")
        return

    # Find columns with composite advantage component contributions
    contribution_cols = [col for col in df.columns if "contribution" in col]

    if not contribution_cols:
        logging.warning("No composite advantage component contributions found")
        return

    # Group contribution columns by agent pair
    pair_components = {}
    for col in contribution_cols:
        # Extract the agent pair and component from the column name
        parts = col.split("_contribution_")
        if len(parts) != 2:
            continue

        pair = parts[0]
        component = parts[1]

        if pair not in pair_components:
            pair_components[pair] = []

        pair_components[pair].append((component, col))

    # Create a plot for each agent pair
    for pair, components in pair_components.items():
        if not components:
            continue

        # Calculate average component values
        component_names = [c[0] for c in components]
        component_values = [df[c[1]].mean() for c in components]

        # Sort by absolute contribution
        sort_idx = np.argsort([abs(v) for v in component_values])[::-1]
        component_names = [component_names[i] for i in sort_idx]
        component_values = [component_values[i] for i in sort_idx]

        # Format agent pair label
        pair_label = pair.replace("_", " ").replace("vs", "vs.")

        # Create the plot
        fig = plt.figure(figsize=(10, 6))

        # Create bars with different colors for positive and negative contributions
        colors = ["#1f77b4" if v > 0 else "#d62728" for v in component_values]
        plt.barh(range(len(component_names)), component_values, color=colors)

        plt.yticks(
            range(len(component_names)),
            [c.replace("_", " ").title() for c in component_names],
        )
        plt.axvline(x=0, color="black", linestyle="-", alpha=0.3)

        plt.xlabel("Average Contribution to Composite Advantage")
        plt.ylabel("Advantage Component")
        plt.title(f"Composite Advantage Breakdown: {pair_label}")

        # Add total advantage score
        total_col = f"{pair}_composite_advantage"
        if total_col in df.columns:
            total_advantage = df[total_col].mean()
            plt.figtext(
                0.5,
                0.01,
                f"Total Composite Advantage: {total_advantage:.3f}",
                ha="center",
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.8),
            )

        plt.tight_layout()

        # Add watermark if data was cleaned
        if data_cleaned:
            add_data_cleaning_watermark(fig)

        # Save the figure
        output_file = os.path.join(output_path, f"{pair}_advantage_breakdown.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved {pair} advantage breakdown to {output_file}")


def plot_advantage_trajectories(
    df: pd.DataFrame, output_path: str, data_cleaned: bool = False
):
    """
    Plot advantage trajectories over the course of the simulation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing advantage metrics for all simulations
    output_path : str
        Directory where output files will be saved
    data_cleaned : bool, optional
        Flag indicating whether the data has been cleaned of NaN and infinity values
    """
    if df.empty:
        logging.warning("Empty DataFrame, skipping advantage trajectories plot")
        return

    # Find trajectory columns
    trajectory_cols = [col for col in df.columns if "trajectory" in col]

    if not trajectory_cols:
        logging.warning("No advantage trajectory data found")
        return

    # Group trajectory columns by category
    category_trajectories = {}
    for col in trajectory_cols:
        for category in [
            "resource_acquisition",
            "reproduction",
            "survival",
            "population_growth",
            "combat",
        ]:
            if category in col:
                if category not in category_trajectories:
                    category_trajectories[category] = []
                category_trajectories[category].append(col)
                break

    # Create a plot for each category
    for category, cols in category_trajectories.items():
        if not cols:
            continue

        # Get agent pairs from column names
        pairs = []
        for col in cols:
            if "_vs_" in col:
                pair = (
                    col.split("_vs_")[0] + "_vs_" + col.split("_vs_")[1].split("_")[0]
                )
                if pair not in pairs:
                    pairs.append(pair)

        # Create the plot
        fig = plt.figure(figsize=(10, 6))

        # Plot each pair's average trajectory
        for pair in pairs:
            pair_cols = [col for col in cols if pair in col]
            if not pair_cols:
                continue

            # Calculate average trajectory across all simulations
            pair_trajectories = df[pair_cols].mean(axis=1).mean()

            # Get the standard deviation for error bars
            pair_std = df[pair_cols].mean(axis=1).std()

            # Get the agent types
            type1, type2 = pair.split("_vs_")

            # For visualization, place each pair at a different x position
            if type1 == "system":
                x_pos = 0
            elif type1 == "independent":
                x_pos = 1
            else:  # control
                x_pos = 2

            # Add small offset based on second agent type
            if type2 == "system":
                x_pos += 0.1
            elif type2 == "independent":
                x_pos += 0.2
            else:  # control
                x_pos += 0.3

            # Plot the point with error bar
            plt.errorbar(
                x_pos,
                pair_trajectories,
                yerr=pair_std,
                fmt="o",
                capsize=5,
                label=f"{type1.capitalize()} vs {type2.capitalize()}",
            )

        # Add a horizontal line at y=0 (no change in advantage)
        plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # Format axis labels and title
        plt.xlabel("Agent Pair")
        plt.ylabel("Average Advantage Trajectory")
        plt.title(f'{category.replace("_", " ").title()} Advantage Trajectory')

        # Custom x-ticks
        plt.xticks([0.2, 1.2, 2.2], ["System", "Independent", "Control"])
        plt.xlim(-0.5, 3)

        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Add watermark if data was cleaned
        if data_cleaned:
            add_data_cleaning_watermark(fig)

        # Save the figure
        output_file = os.path.join(output_path, f"{category}_advantage_trajectory.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved {category} advantage trajectory to {output_file}")

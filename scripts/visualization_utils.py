#!/usr/bin/env python3
"""
visualization_utils.py

Shared utility functions for creating visualizations across analysis scripts.
Contains common plotting patterns, color schemes, and styling functions.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

# Standard color schemes used across analysis scripts
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
    "dark": {
        "background": "#2c3e50",
        "resources": "#27ae60",
        "agents": {
            "SystemAgent": "#3498db",
            "IndependentAgent": "#e74c3c",
            "ControlAgent": "#f39c12",
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
    "colorblind_friendly": {
        "background": "#ffffff",
        "resources": "#117733",
        "agents": {
            "SystemAgent": "#332288",
            "IndependentAgent": "#aa4499",
            "ControlAgent": "#ddcc77",
        },
    },
}


def get_agent_colors(scheme: str = "default") -> Dict[str, str]:
    """
    Get agent colors for a specific color scheme.

    Parameters
    ----------
    scheme : str
        Color scheme name

    Returns
    -------
    Dict[str, str]
        Mapping of agent types to colors
    """
    if scheme not in COLOR_SCHEMES:
        raise ValueError(f"Unknown color scheme: {scheme}")
    return COLOR_SCHEMES[scheme]["agents"]


def setup_plot_style(style: str = "default") -> None:
    """
    Set up matplotlib plot style and parameters.

    Parameters
    ----------
    style : str
        Plot style to use
    """
    plt.style.use(style)

    # Set common parameters
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def create_clean_frame(
    ax: Axes,
    agents: pd.DataFrame,
    resources: pd.DataFrame,
    config: Optional[Dict] = None,
    color_scheme: str = "default",
    show_gathering_range: bool = False,
) -> None:
    """
    Create a clean visualization frame of agents and resources.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    agents : pd.DataFrame
        DataFrame with agent data
    resources : pd.DataFrame
        DataFrame with resource data
    config : Optional[Dict]
        Simulation configuration
    color_scheme : str
        Color scheme to use
    show_gathering_range : bool
        Whether to show gathering range circles
    """
    colors = get_agent_colors(color_scheme)

    # Set background color
    scheme_config = COLOR_SCHEMES[color_scheme]
    ax.set_facecolor(scheme_config["background"])

    # Plot resources
    is_dark_bg = scheme_config["background"] in ["#000000", "#2c3e50"]
    resource_alpha = 0.5 if is_dark_bg else 0.3

    ax.scatter(
        resources["position_x"],
        resources["position_y"],
        s=resources["amount"] * 2,
        alpha=resource_alpha,
        c=scheme_config["resources"],
        zorder=1,
        label="Resources",
    )

    # Plot agents
    edge_color = "black" if not is_dark_bg else "white"

    for agent_type, color in colors.items():
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
                label=agent_type,
            )

    # Draw gathering range circles if enabled
    if show_gathering_range and config:
        gathering_range = config.get("gathering_range", 30)
        for _, agent in agents.iterrows():
            circle = Circle(
                (agent["position_x"], agent["position_y"]),
                gathering_range,
                fill=False,
                linestyle="--",
                color=colors[agent["agent_type"]],
                alpha=0.3,
            )
            ax.add_patch(circle)

    # Set plot limits and clean up
    width = config.get("width", 100) if config else 100
    height = config.get("height", 100) if config else 100

    # Calculate padding
    dot_radius = np.sqrt(120 / np.pi)
    padding = dot_radius * 0.75

    ax.set_xlim(-padding, width + padding)
    ax.set_ylim(-padding, height + padding)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.set_aspect("equal")


def create_frame_with_labels(
    ax: Axes,
    agents: pd.DataFrame,
    resources: pd.DataFrame,
    config: Optional[Dict] = None,
    color_scheme: str = "default",
) -> None:
    """
    Create a frame with agent ID labels and resource amounts.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    agents : pd.DataFrame
        DataFrame with agent data
    resources : pd.DataFrame
        DataFrame with resource data
    config : Optional[Dict]
        Simulation configuration
    color_scheme : str
        Color scheme to use
    """
    colors = get_agent_colors(color_scheme)

    # Set background color
    scheme_config = COLOR_SCHEMES[color_scheme]
    ax.set_facecolor(scheme_config["background"])

    # Plot resources
    ax.scatter(
        resources["position_x"],
        resources["position_y"],
        s=resources["amount"] * 5,
        alpha=0.5,
        c=scheme_config["resources"],
        label="Resources",
    )

    # Plot agents with labels
    for agent_type, color in colors.items():
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

            # Add agent ID labels and resource amounts
            for _, agent in agent_data.iterrows():
                ax.annotate(
                    f"{agent['agent_id'][-4:]}\n({agent['resources']:.1f})",
                    (agent["position_x"], agent["position_y"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

    # Set plot limits and labels
    width = config.get("width", 100) if config else 100
    height = config.get("height", 100) if config else 100
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")


def create_mosaic_frame(
    ax: Axes,
    agents: pd.DataFrame,
    resources: pd.DataFrame,
    iteration: int,
    color_scheme: str = "default",
) -> None:
    """
    Create a mosaic frame for initial state visualization.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    agents : pd.DataFrame
        DataFrame with agent data
    resources : pd.DataFrame
        DataFrame with resource data
    iteration : int
        Iteration number for title
    color_scheme : str
        Color scheme to use
    """
    create_clean_frame(ax, agents, resources, color_scheme=color_scheme)

    # Add iteration number to title
    ax.set_title(f"Iteration {iteration}")

    # Add legend to first subplot only (caller should handle this)
    ax.legend(loc="upper right")


def create_time_series_plot(
    ax: Axes,
    data: pd.DataFrame,
    x_column: str,
    y_columns: List[str],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
) -> None:
    """
    Create a time series plot with multiple lines.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    data : pd.DataFrame
        DataFrame with time series data
    x_column : str
        Name of x-axis column
    y_columns : List[str]
        List of y-axis column names
    labels : Optional[List[str]]
        Labels for each line
    colors : Optional[List[str]]
        Colors for each line
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    """
    if labels is None:
        labels = y_columns
    if colors is None:
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

    # Ensure colors is a list
    if not isinstance(colors, list):
        colors = list(colors)

    for i, (column, label) in enumerate(zip(y_columns, labels)):
        color = colors[i % len(colors)]
        ax.plot(data[x_column], data[column], color=color, label=label, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def create_histogram(
    ax: Axes,
    data: pd.Series,
    bins: int = 30,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Count",
    kde: bool = True,
) -> None:
    """
    Create a histogram plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    data : pd.Series
        Data to plot
    bins : int
        Number of bins
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    kde : bool
        Whether to show kernel density estimate
    """
    ax.hist(data, bins=bins, alpha=0.7, edgecolor="black")

    if kde:
        try:
            from scipy.stats import gaussian_kde

            density = gaussian_kde(data.dropna())
            x_range = np.linspace(data.min(), data.max(), 100)
            ax.plot(
                x_range,
                density(x_range) * len(data) * (data.max() - data.min()) / bins,
                color="red",
                linewidth=2,
                label="KDE",
            )
        except ImportError:
            pass

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def create_box_plot(
    ax: Axes,
    data: List[List[float]],
    labels: Optional[List[str]],
    title: str = "",
    ylabel: str = "",
) -> None:
    """
    Create a box plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    data : List[List[float]]
        Data for each box
    labels : List[str]
        Labels for each box
    title : str
        Plot title
    ylabel : str
        Y-axis label
    """
    # Create boxplot - matplotlib boxplot does accept labels parameter
    bp = ax.boxplot(data, patch_artist=True)  # type: ignore
    if labels is not None and len(labels) == len(data):
        ax.set_xticklabels(labels)

    # Color the boxes
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def create_correlation_heatmap(
    ax: Axes,
    data: pd.DataFrame,
    title: str = "Correlation Matrix",
    annot: bool = True,
    cmap: str = "coolwarm",
) -> None:
    """
    Create a correlation heatmap.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes object
    data : pd.DataFrame
        DataFrame to compute correlations from
    title : str
        Plot title
    annot : bool
        Whether to annotate cells with values
    cmap : str
        Colormap to use
    """
    corr_matrix = data.corr()

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.index)

    # Add annotations
    if annot:
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, ".2f", ha="center", va="center", color="black")

    ax.set_title(title)


def save_figure(
    fig: Any,
    output_path: str,
    dpi: int = 300,
    bbox_inches: str = "tight",
    facecolor: Optional[str] = None,
) -> None:
    """
    Save a matplotlib figure with consistent settings.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure object
    output_path : str
        Path to save the figure
    dpi : int
        Resolution in dots per inch
    bbox_inches : str
        Bounding box setting
    facecolor : Optional[str]
        Face color for the figure
    """
    save_kwargs = {"dpi": dpi, "bbox_inches": bbox_inches, "edgecolor": "none"}

    if facecolor:
        save_kwargs["facecolor"] = facecolor

    fig.savefig(output_path, **save_kwargs)
    plt.close(fig)


def create_legend_elements(
    agent_types: List[str], include_resources: bool = True
) -> List[Line2D]:
    """
    Create legend elements for agent types and resources.

    Parameters
    ----------
    agent_types : List[str]
        List of agent types
    include_resources : bool
        Whether to include resources in legend

    Returns
    -------
    List[Line2D]
        List of legend elements
    """
    colors = get_agent_colors("default")
    legend_elements = []

    if include_resources:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#2ecc71",
                markersize=10,
                label="Resources",
            )
        )

    for agent_type in agent_types:
        if agent_type in colors:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[agent_type],
                    markersize=10,
                    label=agent_type.replace("Agent", ""),
                )
            )

    return legend_elements


def get_agent_size(
    resource_level: float,
    agent_id: str,
    current_step: int,
    agent_first_seen: Dict[str, int],
    growth_period: int = 10,
) -> float:
    """
    Calculate agent marker size based on resources and birth status.

    Parameters
    ----------
    resource_level : float
        Agent's resource level
    agent_id : str
        Agent ID
    current_step : int
        Current simulation step
    agent_first_seen : Dict[str, int]
        Dictionary tracking when each agent was first seen
    growth_period : int
        Number of steps for agent growth animation

    Returns
    -------
    float
        Marker size for the agent
    """
    min_size = 50
    max_size = 600

    # For initial generation (step 0), show normal size
    if current_step == 0:
        normalized_level = max(0, resource_level) / 100
        return min_size + (max_size - min_size) * (normalized_level**0.7)

    # Track when we first see this agent
    if agent_id not in agent_first_seen:
        agent_first_seen[agent_id] = current_step

    # For newly born agents, scale size based on how many steps they've been alive
    steps_alive = current_step - agent_first_seen[agent_id]
    if steps_alive < growth_period:
        growth_factor = (steps_alive / growth_period) ** 0.5
        normalized_level = max(0, resource_level) / 100
        base_size = min_size + (max_size - min_size) * (normalized_level**0.7)
        start_size = min_size * 1.5
        return start_size + (base_size - start_size) * growth_factor

    # Normal size calculation for established agents
    normalized_level = max(0, resource_level) / 100
    return min_size + (max_size - min_size) * (normalized_level**0.7)

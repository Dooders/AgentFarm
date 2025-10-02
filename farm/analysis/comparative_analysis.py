import glob
import io
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
from sqlalchemy import func

from farm.database.database import SimulationDatabase
from farm.database.models import AgentModel, Simulation, SimulationStepModel


def find_simulation_databases(base_path: str) -> List[str]:
    """
    Find all SQLite database files in the specified directory and its subdirectories.

    Parameters
    ----------
    base_path : str
        Base directory to search for simulation databases

    Returns
    -------
    List[str]
        List of paths to found database files
    """
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    print(f"Searching for databases in: {os.path.abspath(base_path)}")

    # Find all simulation.db files in subdirectories
    db_files = []
    for root, dirs, files in os.walk(base_path):
        if "simulation.db" in files:
            db_path = os.path.join(root, "simulation.db")
            db_files.append(db_path)

    if not db_files:
        print(
            f"Warning: No simulation.db files found in {base_path} or its subdirectories"
        )
    else:
        print(f"Found {len(db_files)} database files:")
        for db_file in db_files:
            print(f"  - {db_file}")

    return sorted(db_files)


def gather_simulation_metadata(db_path: str) -> Dict[str, Any]:
    """
    Gather high-level metadata for a single simulation, including duration,
    configuration, and a summary of final-step metrics.
    """
    # Get a readable name from the database path
    sim_name = os.path.splitext(os.path.basename(db_path))[0]

    db = SimulationDatabase(db_path)
    session = db.Session()

    try:
        # Debug print to check table structure
        step = session.query(SimulationStepModel).first()
        if step:
            print(
                "Available columns:",
                [column.key for column in SimulationStepModel.__table__.columns],
            )

        # 1) Fetch simulation record
        sim = (
            session.query(Simulation).order_by(Simulation.simulation_id.desc()).first()
        )

        # 2) Calculate simulation duration in steps
        final_step_row = session.query(func.max(SimulationStepModel.step_number)).one()
        final_step = final_step_row[0] if final_step_row[0] is not None else 0

        # 3) Get step-wise population data
        population_data = (
            session.query(SimulationStepModel.total_agents)
            .order_by(SimulationStepModel.step_number)
            .all()
        )
        population_series = pd.Series([p[0] for p in population_data])

        # Calculate population statistics with safe defaults
        median_population = (
            population_series.median() if not population_series.empty else 0
        )
        # Handle case where mode() returns empty series
        mode_series = population_series.mode()
        mode_population = mode_series.iloc[0] if not mode_series.empty else 0
        mean_population = population_series.mean() if not population_series.empty else 0

        # 4) Calculate across-time averages
        avg_metrics = session.query(
            func.avg(SimulationStepModel.total_agents).label("avg_total_agents"),
            func.avg(SimulationStepModel.births).label("avg_births"),
            func.avg(SimulationStepModel.deaths).label("avg_deaths"),
            func.avg(SimulationStepModel.total_resources).label("avg_total_resources"),
            func.avg(SimulationStepModel.average_agent_health).label("avg_health"),
            func.avg(SimulationStepModel.average_reward).label("avg_reward"),
        ).first()

        # 5) Gather second-to-last step metrics
        final_step_row = session.query(func.max(SimulationStepModel.step_number)).one()
        final_step = final_step_row[0] if final_step_row[0] is not None else 0
        second_to_last_step = final_step - 1 if final_step > 0 else 0

        final_metrics = (
            session.query(SimulationStepModel)
            .filter(SimulationStepModel.step_number == second_to_last_step)
            .first()
        )

        if final_metrics:
            final_snapshot = {
                "final_total_agents": final_metrics.total_agents,
                "final_total_resources": final_metrics.total_resources,
                "final_average_health": final_metrics.average_agent_health,
                "final_average_reward": final_metrics.average_reward,
                "final_births": final_metrics.births,
                "final_deaths": final_metrics.deaths,
            }
        else:
            final_snapshot = {
                "final_total_agents": None,
                "final_total_resources": None,
                "final_average_health": None,
                "final_average_reward": None,
                "final_births": None,
                "final_deaths": None,
            }

        # 6) Extract config parameters
        config = (
            sim.parameters
            if sim and hasattr(sim, "parameters") and sim.parameters is not None
            else {}
        )

        # Add average age calculation
        avg_age_result = session.query(
            func.avg(SimulationStepModel.average_agent_age)
        ).scalar()

        # Add lifespan analysis
        lifespan_data = calculate_agent_lifespans(db_path)

        # Get phase lifespans with safe defaults
        phase_lifespans = lifespan_data.get("phase_lifespans", {})

        # 7) Build complete metadata dictionary
        return {
            "simulation_name": sim_name,
            "db_path": db_path,
            "simulation_id": sim.simulation_id if sim else None,
            "start_time": str(sim.start_time) if sim else None,
            "end_time": (
                str(sim.end_time)
                if sim and hasattr(sim, "end_time") and sim.end_time is not None
                else None
            ),
            "status": sim.status if sim else None,
            "duration_steps": final_step,
            "config": config,
            # Population statistics
            "median_population": (
                0.0 if median_population == 0 else float(str(median_population))
            ),
            "mode_population": (
                0.0 if mode_population == 0 else float(str(mode_population))
            ),
            "mean_population": (
                0.0 if mean_population == 0 else float(str(mean_population))
            ),
            # Across-time averages
            "avg_total_agents": (
                float(avg_metrics[0]) if avg_metrics and avg_metrics[0] else 0
            ),
            "avg_births": (
                float(avg_metrics[1]) if avg_metrics and avg_metrics[1] else 0
            ),
            "avg_deaths": (
                float(avg_metrics[2]) if avg_metrics and avg_metrics[2] else 0
            ),
            "avg_total_resources": (
                float(avg_metrics[3]) if avg_metrics and avg_metrics[3] else 0
            ),
            "avg_health": (
                float(avg_metrics[4]) if avg_metrics and avg_metrics[4] else 0
            ),
            "avg_reward": (
                float(avg_metrics[5]) if avg_metrics and avg_metrics[5] else 0
            ),
            "avg_agent_age": float(avg_age_result) if avg_age_result else 0,
            **final_snapshot,
            "lifespan_data": lifespan_data,
            "avg_lifespan": lifespan_data["avg_lifespan"],
            "median_lifespan": lifespan_data["median_lifespan"],
            "initial_phase_lifespan": phase_lifespans.get("initial", {}).get("mean", 0),
            "early_phase_lifespan": phase_lifespans.get("early", {}).get("mean", 0),
            "mid_phase_lifespan": phase_lifespans.get("mid", {}).get("mean", 0),
            "late_phase_lifespan": phase_lifespans.get("late", {}).get("mean", 0),
            "final_phase_lifespan": phase_lifespans.get("final", {}).get("mean", 0),
        }
    finally:
        session.close()
        db.close()


def plot_population_dynamics(df: pd.DataFrame, output_dir: str):
    """Plot population dynamics across simulations."""
    plt.figure(figsize=(12, 6))
    x = range(len(df))

    plt.plot(x, df["median_population"], "b--", label="Median Population")
    plt.plot(x, df["mode_population"], "g--", label="Mode Population")
    plt.plot(x, df["final_total_agents"], "r-", label="Final Population")
    plt.title("Population Comparison Across Simulations")
    plt.xlabel("Simulation Index")
    plt.ylabel("Number of Agents")
    plt.legend()

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, "temp_population_comparison.png")
    plt.savefig(temp_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Add the note using Pillow
    note_text = (
        "Note: This plot compares different population metrics across simulations. "
        "The median and mode show typical population levels, while the final population "
        "indicates how simulations ended."
    )

    final_path = os.path.join(output_dir, "population_comparison.png")
    add_note_to_image(temp_path, note_text)

    # Clean up temporary file
    os.remove(temp_path)


def plot_resource_utilization(df: pd.DataFrame, output_dir: str):
    """Plot resource utilization across simulations."""
    plt.figure(figsize=(12, 6))
    x = range(len(df))
    plt.plot(x, df["avg_total_resources"], "g-", label="Average Resources")
    plt.plot(x, df["final_total_resources"], "m--", label="Final Resources")
    plt.title("Resource Utilization Across Simulations")
    plt.xlabel("Simulation Index")
    plt.ylabel("Resource Level")
    plt.legend()

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, "temp_resource_comparison.png")
    plt.savefig(temp_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Add the note using Pillow
    note_text = (
        "Note: This visualization tracks resource levels across simulations. "
        "Average resources show typical availability, while final resources "
        "indicate end-state resource levels."
    )

    final_path = os.path.join(output_dir, "resource_comparison.png")
    add_note_to_image(temp_path, note_text)

    # Clean up temporary file
    os.remove(temp_path)


def plot_health_reward_distribution(df: pd.DataFrame, output_dir: str):
    """Plot health and reward distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(data=df[["avg_health", "final_average_health"]], ax=ax1)
    ax1.set_title("Health Distribution")
    ax1.set_ylabel("Health Level")

    sns.boxplot(data=df[["avg_reward", "final_average_reward"]], ax=ax2)
    ax2.set_title("Reward Distribution")
    ax2.set_ylabel("Reward Level")

    plt.tight_layout()

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, "temp_health_reward_distribution.png")
    plt.savefig(temp_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Add the note using Pillow
    note_text = (
        "Note: These box plots show the distribution of health and reward metrics. "
        "The boxes show the middle 50% of values, while whiskers extend to the full range "
        "excluding outliers."
    )

    final_path = os.path.join(output_dir, "health_reward_distribution.png")
    add_note_to_image(temp_path, note_text)

    # Clean up temporary file
    os.remove(temp_path)


def plot_population_vs_cycles(df: pd.DataFrame, output_dir: str):
    """Plot population vs simulation duration."""
    plt.figure(figsize=(12, 6))
    plt.scatter(
        df["avg_total_agents"], df["duration_steps"], alpha=0.6, c="blue", s=100
    )

    plt.title("Average Population vs Simulation Duration")
    plt.xlabel("Average Population")
    plt.ylabel("Number of Cycles (Steps)")

    z = np.polyfit(df["avg_total_agents"], df["duration_steps"], 1)
    p = np.poly1d(z)
    plt.plot(
        df["avg_total_agents"],
        p(df["avg_total_agents"]),
        "r--",
        alpha=0.8,
        label=f"Trend line (slope: {z[0]:.2f})",
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, "temp_population_vs_cycles.png")
    plt.savefig(temp_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Add the note using Pillow
    note_text = (
        "Note: This scatter plot reveals the relationship between population size and "
        "simulation duration. The trend line shows the general correlation between "
        "these variables."
    )

    final_path = os.path.join(output_dir, "population_vs_cycles.png")
    add_note_to_image(temp_path, note_text)

    # Clean up temporary file
    os.remove(temp_path)


def plot_population_vs_age(df: pd.DataFrame, output_dir: str):
    """Plot population vs agent age with birth rate color gradient."""
    plt.figure(figsize=(12, 6))

    scatter = plt.scatter(
        df["avg_total_agents"],
        df["avg_agent_age"],
        c=df["avg_births"],
        s=100,
        alpha=0.6,
        cmap="viridis",
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Average Birth Rate", rotation=270, labelpad=15)

    correlation = df["avg_total_agents"].corr(df["avg_agent_age"])
    plt.title(f"Average Population vs Average Agent Age (r = {correlation:.2f})")
    plt.xlabel("Average Population")
    plt.ylabel("Average Agent Age")

    z = np.polyfit(df["avg_total_agents"], df["avg_agent_age"], 1)
    p = np.poly1d(z)
    plt.plot(
        df["avg_total_agents"],
        p(df["avg_total_agents"]),
        "r--",
        alpha=0.8,
        label=f"Trend line (slope: {z[0]:.2f})",
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, "temp_population_vs_age.png")
    plt.savefig(temp_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Add the note using Pillow
    note_text = (
        "Note: This scatter plot shows how population size relates to average agent age. "
        "The color gradient indicates birth rates, while the trend line shows the overall "
        "relationship between population and age."
    )

    final_path = os.path.join(output_dir, "population_vs_age.png")
    add_note_to_image(temp_path, note_text)

    # Clean up temporary file
    os.remove(temp_path)


def plot_population_trends_across_simulations(df: pd.DataFrame, output_dir: str):
    """
    Create a visualization showing population trends across all simulations.
    """
    # Create figure with two subplots sharing x axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 2])
    fig.suptitle("Population Trends Across All Simulations", fontsize=14, y=0.95)

    # Get all simulation databases and collect step-wise data
    all_populations = []
    max_steps = 0

    for db_path in df["db_path"]:
        db = SimulationDatabase(db_path)
        session = db.Session()

        try:
            population_data = (
                session.query(
                    SimulationStepModel.step_number, SimulationStepModel.total_agents
                )
                .order_by(SimulationStepModel.step_number)
                .all()
            )

            steps = np.array([p[0] for p in population_data])
            pop = np.array([p[1] for p in population_data])

            max_steps = max(max_steps, len(steps))
            all_populations.append(pop)

        finally:
            session.close()
            db.close()

    # Pad shorter simulations with NaN values
    padded_populations = []
    for pop in all_populations:
        if len(pop) < max_steps:
            padded = np.pad(
                pop, (0, max_steps - len(pop)), mode="constant", constant_values=np.nan
            )
        else:
            padded = pop
        padded_populations.append(padded)

    population_array = np.array(padded_populations)

    # Calculate statistics
    mean_pop = np.nanmean(population_array, axis=0)
    median_pop = np.nanmedian(population_array, axis=0)
    std_pop = np.nanstd(population_array, axis=0)
    confidence_interval = 1.96 * std_pop / np.sqrt(len(all_populations))

    steps = np.arange(max_steps)

    # Plot full range in top subplot (linear scale)
    ax1.plot(steps, mean_pop, "b-", label="Mean Population", linewidth=2)
    ax1.plot(steps, median_pop, "g--", label="Median Population", linewidth=2)
    ax1.fill_between(
        steps,
        mean_pop - confidence_interval,
        mean_pop + confidence_interval,
        color="b",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    # Customize top subplot
    ax1.set_ylabel("Number of Agents", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Add text to explain the views
    ax1.text(
        0.02,
        0.98,
        "Full Range View",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    # Plot steady-state detail in bottom subplot (linear scale)
    # Find where the initial spike settles (simple heuristic)
    settling_point = 55  # Skip first 100 steps to avoid initial spike

    ax2.plot(
        steps[settling_point:],
        mean_pop[settling_point:],
        "b-",
        label="Mean Population",
        linewidth=2,
    )
    ax2.plot(
        steps[settling_point:],
        median_pop[settling_point:],
        "g--",
        label="Median Population",
        linewidth=2,
    )
    ax2.fill_between(
        steps[settling_point:],
        mean_pop[settling_point:] - confidence_interval[settling_point:],
        mean_pop[settling_point:] + confidence_interval[settling_point:],
        color="b",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    # Customize bottom subplot
    steady_state_min = np.nanmin(
        mean_pop[settling_point:] - confidence_interval[settling_point:]
    )
    steady_state_max = np.nanmax(
        mean_pop[settling_point:] + confidence_interval[settling_point:]
    )
    padding = (steady_state_max - steady_state_min) * 0.1
    ax2.set_ylim(steady_state_min - padding, steady_state_max + padding)
    ax2.set_xlabel("Simulation Step", fontsize=12)
    ax2.set_ylabel("Number of Agents", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Add text to explain the views
    ax2.text(
        0.02,
        0.98,
        "Steady State Detail",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, "temp_population_trends.png")
    plt.savefig(temp_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Add the note using Pillow
    note_text = (
        "Note: This visualization shows population changes over time across all simulations. "
        "The top panel displays the full range including initial population spikes, while "
        "the bottom panel focuses on the steady-state behavior after initial fluctuations settle."
    )

    final_path = os.path.join(output_dir, "population_trends.png")
    add_note_to_image(temp_path, note_text)

    # Clean up temporary file
    os.remove(temp_path)


def plot_population_trends_by_type_across_simulations(
    df: pd.DataFrame, output_dir: str
):
    """
    Create a visualization showing population trends by agent type across all simulations.
    """
    print(
        f"Starting plot_population_trends_by_type_across_simulations with output_dir: {output_dir}"
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with four subplots: one large at top, three below
    fig = plt.figure(figsize=(15, 20))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 1], hspace=0.3)

    ax_full = fig.add_subplot(gs[0])  # Top subplot for full view
    ax1 = fig.add_subplot(gs[1])  # System agents
    ax2 = fig.add_subplot(gs[2])  # Independent agents
    ax3 = fig.add_subplot(gs[3])  # Control agents

    fig.suptitle(
        "Population Trends by Agent Type Across All Simulations", fontsize=14, y=0.95
    )

    # Get all simulation databases and collect step-wise data
    agent_types = {"System Agents": [], "Independent Agents": [], "Control Agents": []}
    max_steps = 0

    for db_path in df["db_path"]:
        db = SimulationDatabase(db_path)
        session = db.Session()

        try:
            population_data = (
                session.query(
                    SimulationStepModel.step_number,
                    SimulationStepModel.system_agents,
                    SimulationStepModel.independent_agents,
                    SimulationStepModel.control_agents,
                )
                .order_by(SimulationStepModel.step_number)
                .all()
            )

            steps = np.array([p[0] for p in population_data])
            system_pop = np.array(
                [p[1] if p[1] is not None else 0 for p in population_data]
            )
            independent_pop = np.array(
                [p[2] if p[2] is not None else 0 for p in population_data]
            )
            control_pop = np.array(
                [p[3] if p[3] is not None else 0 for p in population_data]
            )

            max_steps = max(max_steps, len(steps))
            agent_types["System Agents"].append(system_pop)
            agent_types["Independent Agents"].append(independent_pop)
            agent_types["Control Agents"].append(control_pop)

        finally:
            session.close()
            db.close()

    # Pad shorter simulations with NaN values for each agent type
    padded_populations = {agent_type: [] for agent_type in agent_types.keys()}
    for agent_type, populations in agent_types.items():
        for pop in populations:
            if len(pop) < max_steps:
                padded = np.pad(
                    pop,
                    (0, max_steps - len(pop)),
                    mode="constant",
                    constant_values=np.nan,
                )
            else:
                padded = pop
            padded_populations[agent_type].append(padded)

    # Convert to numpy arrays and calculate statistics
    population_arrays = {
        agent_type: np.array(pops) for agent_type, pops in padded_populations.items()
    }

    steps = np.arange(max_steps)
    colors = {
        "System Agents": "blue",
        "Independent Agents": "green",
        "Control Agents": "red",
    }
    axes = {"System Agents": ax1, "Independent Agents": ax2, "Control Agents": ax3}

    # Plot full range view at the top
    for agent_type, pop_array in population_arrays.items():
        color = colors[agent_type]
        mean_pop = np.nanmean(pop_array, axis=0)
        median_pop = np.nanmedian(pop_array, axis=0)
        std_pop = np.nanstd(pop_array, axis=0)
        confidence_interval = 1.96 * std_pop / np.sqrt(len(pop_array))

        # Plot mean line for full view
        ax_full.plot(
            steps,
            mean_pop,
            color=color,
            linestyle="-",
            label=f"{agent_type} (Mean)",
            linewidth=2,
        )
        ax_full.fill_between(
            steps,
            mean_pop - confidence_interval,
            mean_pop + confidence_interval,
            color=color,
            alpha=0.2,
        )

    # Customize full view subplot
    ax_full.set_ylabel("Number of Agents", fontsize=12)
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(fontsize=10)
    ax_full.set_title("Full Range View - All Agent Types", fontsize=12, pad=10)

    # Plot stable view for each agent type
    settling_point = 55  # Skip first 55 steps to avoid initial spike
    axes = {"System Agents": ax1, "Independent Agents": ax2, "Control Agents": ax3}

    for agent_type, pop_array in population_arrays.items():
        ax = axes[agent_type]
        color = colors[agent_type]

        mean_pop = np.nanmean(pop_array, axis=0)
        median_pop = np.nanmedian(pop_array, axis=0)
        std_pop = np.nanstd(pop_array, axis=0)
        confidence_interval = 1.96 * std_pop / np.sqrt(len(pop_array))

        # Plot stable period only
        ax.plot(
            steps[settling_point:],
            mean_pop[settling_point:],
            color=color,
            linestyle="-",
            label="Mean Population",
            linewidth=2,
        )
        ax.plot(
            steps[settling_point:],
            median_pop[settling_point:],
            color=color,
            linestyle="--",
            label="Median Population",
            linewidth=2,
        )

        # Add confidence interval
        ax.fill_between(
            steps[settling_point:],
            mean_pop[settling_point:] - confidence_interval[settling_point:],
            mean_pop[settling_point:] + confidence_interval[settling_point:],
            color=color,
            alpha=0.2,
            label="95% Confidence Interval",
        )

        # Customize subplot
        ax.set_ylabel("Number of Agents", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_title(
            f"{agent_type} Population Trends (Stable Period)", fontsize=12, pad=10
        )

    # Set common x-axis label for bottom subplot only
    ax3.set_xlabel("Simulation Step", fontsize=12)

    # Adjust layout
    plt.tight_layout()

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, "temp_population_trends_by_type.png")
    final_path = os.path.join(output_dir, "population_trends_by_type.png")

    # Ensure we're using forward slashes for consistency
    temp_path = os.path.normpath(temp_path)
    final_path = os.path.normpath(final_path)

    print(f"Saving temporary file to: {temp_path}")
    try:
        plt.savefig(temp_path, dpi=300, bbox_inches="tight")
        print("Temporary file saved successfully")
        print(f"File size: {os.path.getsize(temp_path)} bytes")
    except Exception as e:
        print(f"Error saving temporary file: {str(e)}")
        return
    finally:
        plt.close()

    # Add the note using Pillow
    note_text = (
        "Note: This visualization shows population changes over time for each agent type "
        "across all simulations. The top panel shows the full range for all agent types, "
        "while the lower panels show detailed stable-period trends for each type. "
        "The lines show mean and median populations, while the shaded areas represent "
        "95% confidence intervals."
    )

    print(f"Adding note and saving final file to: {final_path}")
    add_note_to_image(temp_path, note_text)

    # Clean up temporary file
    try:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print(f"Temporary file removed: {temp_path}")
    except Exception as e:
        print(f"Error removing temporary file: {str(e)}")

    if os.path.exists(final_path):
        print(f"Final image saved successfully to: {final_path}")
        print(f"File size: {os.path.getsize(final_path)} bytes")
    else:
        print(f"Warning: Final image not found at: {final_path}")


def add_note_to_image(figure_path, note_text):
    """
    Add a note below the matplotlib figure using Pillow.
    """
    try:
        print(f"Opening image from: {figure_path}")
        # Open the original figure
        with Image.open(figure_path) as img:
            print(f"Original image size: {img.size}")

            # Try to load Arial font, fall back to default if not available
            try:
                font = ImageFont.truetype("arial.ttf", 14)
                print("Using Arial font")
            except OSError:
                font = ImageFont.load_default()
                print("Using default font")

            # Calculate wrapped text dimensions
            margin = 120
            max_text_width = img.width - (2 * margin)

            # Wrap text and calculate required height
            words = note_text.split()
            lines = []
            current_line = []
            current_width = 0

            for word in words:
                word_width = font.getlength(word + " ")
                if current_width + word_width <= max_text_width:
                    current_line.append(word)
                    current_width += word_width
                else:
                    lines.append(" ".join(current_line))
                    current_line = [word]
                    current_width = word_width

            if current_line:
                lines.append(" ".join(current_line))

            # Calculate required height for text
            line_height = 24
            text_height = len(lines) * line_height
            note_height = text_height + (3 * margin)

            print(
                f"Creating new image with dimensions: {img.width}x{img.height + note_height}"
            )
            # Create new image with space for text
            new_img = Image.new("RGB", (img.width, img.height + note_height), "white")
            new_img.paste(img, (0, 0))

            # Draw a light gray line to separate the note from the plot
            draw = ImageDraw.Draw(new_img)
            line_y = img.height + margin
            draw.line(
                [(margin, line_y), (img.width - margin, line_y)],
                fill="#CCCCCC",
                width=2,
            )

            # Add the text lines
            y = img.height + (margin * 1.5)

            for line in lines:
                # Center each line
                text_bbox = draw.textbbox((0, 0), line, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                x = (img.width - text_width) // 2

                draw.text((x, y), line, fill="black", font=font)
                y += line_height

            # Save the combined image to a new file to avoid any file locking issues
            final_path = figure_path.replace("temp_", "")
            print(f"Saving annotated image to: {final_path}")
            new_img.save(final_path, format="PNG")
            print("Image saved successfully")

            # Verify the file was created
            if os.path.exists(final_path):
                print(f"Verified: File exists at {final_path}")
                print(f"File size: {os.path.getsize(final_path)} bytes")
            else:
                print(f"Warning: File not found at {final_path}")

    except Exception as e:
        print(f"Error in add_note_to_image: {str(e)}")
        import traceback

        traceback.print_exc()


def plot_population_candlestick(df: pd.DataFrame, output_dir: str):
    """
    Create a candlestick-style visualization showing population trends across simulations.
    """
    # Create figure with two subplots sharing x axis
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig)

    # Get all simulation databases and collect step-wise data
    all_populations = []
    max_steps = 0

    for db_path in df["db_path"]:
        db = SimulationDatabase(db_path)
        session = db.Session()

        try:
            population_data = (
                session.query(
                    SimulationStepModel.step_number, SimulationStepModel.total_agents
                )
                .order_by(SimulationStepModel.step_number)
                .all()
            )

            steps = np.array([p[0] for p in population_data])
            pop = np.array([p[1] for p in population_data])

            max_steps = max(max_steps, len(steps))
            all_populations.append(pop)

        finally:
            session.close()
            db.close()

    # Pad shorter simulations with NaN values
    padded_populations = []
    for pop in all_populations:
        if len(pop) < max_steps:
            padded = np.pad(
                pop, (0, max_steps - len(pop)), mode="constant", constant_values=np.nan
            )
        else:
            padded = pop
        padded_populations.append(padded)

    population_array = np.array(padded_populations)

    # Calculate statistics for each time step
    steps = np.arange(max_steps)
    percentile_25 = np.nanpercentile(population_array, 25, axis=0)
    percentile_75 = np.nanpercentile(population_array, 75, axis=0)
    median_pop = np.nanmedian(population_array, axis=0)
    min_pop = np.nanmin(population_array, axis=0)
    max_pop = np.nanmax(population_array, axis=0)

    # Plot full range in top subplot
    # Plot min-max range
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(
        steps, min_pop, max_pop, alpha=0.2, color="gray", label="Min-Max Range"
    )
    # Plot interquartile range
    ax1.fill_between(
        steps,
        percentile_25,
        percentile_75,
        alpha=0.4,
        color="blue",
        label="Interquartile Range",
    )
    # Plot median line
    ax1.plot(steps, median_pop, "r-", label="Median", linewidth=1.5)

    # Customize top subplot
    ax1.set_ylabel("Number of Agents", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.text(
        0.02,
        0.98,
        "Full Range Distribution",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    # Plot steady-state detail in bottom subplot
    settling_point = 55

    # Plot min-max range for steady state
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(
        steps[settling_point:],
        min_pop[settling_point:],
        max_pop[settling_point:],
        alpha=0.2,
        color="gray",
        label="Min-Max Range",
    )
    # Plot interquartile range
    ax2.fill_between(
        steps[settling_point:],
        percentile_25[settling_point:],
        percentile_75[settling_point:],
        alpha=0.4,
        color="blue",
        label="Interquartile Range",
    )
    # Plot median line
    ax2.plot(
        steps[settling_point:],
        median_pop[settling_point:],
        "r-",
        label="Median",
        linewidth=1.5,
    )

    # Customize bottom subplot
    ax2.set_ylim(
        np.nanmin(min_pop[settling_point:]), np.nanmax(max_pop[settling_point:])
    )
    ax2.set_xlabel("Simulation Step", fontsize=12)
    ax2.set_ylabel("Number of Agents", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.text(
        0.02,
        0.98,
        "Steady State Distribution",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, "temp_population_distribution.png")
    plt.savefig(temp_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Add the note using Pillow
    note_text = (
        "Note: This chart shows the spread of population sizes across simulations. "
        "The gray area shows the full range (min to max), while the blue area shows where "
        "the middle 50% of populations fall. The red line tracks the median population."
    )

    final_path = os.path.join(output_dir, "population_distribution.png")
    add_note_to_image(temp_path, note_text)

    # Clean up temporary file
    os.remove(temp_path)


def plot_birth_death_rates(df: pd.DataFrame, output_dir: str):
    """
    Create a visualization showing average birth and death rates across all simulations by step.
    """
    # Get all simulation databases and collect step-wise data
    births_by_step = []
    deaths_by_step = []
    max_steps = 0

    for db_path in df["db_path"]:
        db = SimulationDatabase(db_path)
        session = db.Session()

        try:
            step_data = (
                session.query(
                    SimulationStepModel.step_number,
                    SimulationStepModel.births,
                    SimulationStepModel.deaths,
                )
                .order_by(SimulationStepModel.step_number)
                .all()
            )

            steps = [s[0] for s in step_data]
            births = [s[1] if s[1] is not None else 0 for s in step_data]
            deaths = [s[2] if s[2] is not None else 0 for s in step_data]

            max_steps = max(max_steps, len(steps))
            births_by_step.append(births)
            deaths_by_step.append(deaths)

        finally:
            session.close()
            db.close()

    # Pad shorter simulations with zeros
    padded_births = []
    padded_deaths = []
    for births, deaths in zip(births_by_step, deaths_by_step):
        if len(births) < max_steps:
            padded_births.append(
                np.pad(
                    births,
                    (0, max_steps - len(births)),
                    mode="constant",
                    constant_values=0,
                )
            )
            padded_deaths.append(
                np.pad(
                    deaths,
                    (0, max_steps - len(deaths)),
                    mode="constant",
                    constant_values=0,
                )
            )
        else:
            padded_births.append(births)
            padded_deaths.append(deaths)

    # Convert to numpy arrays for calculations
    births_array = np.array(padded_births)
    deaths_array = np.array(padded_deaths)

    # Calculate statistics
    mean_births = np.mean(births_array, axis=0)
    mean_deaths = np.mean(deaths_array, axis=0)
    std_births = np.std(births_array, axis=0)
    std_deaths = np.std(deaths_array, axis=0)

    # Create confidence intervals
    confidence_interval_births = 1.96 * std_births / np.sqrt(len(births_by_step))
    confidence_interval_deaths = 1.96 * std_deaths / np.sqrt(len(deaths_by_step))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[1, 1.5])
    fig.suptitle("Birth and Death Rates Across Simulations", fontsize=14, y=0.95)
    steps = np.arange(max_steps)

    # Plot full range in top subplot
    # Plot births
    ax1.plot(steps, mean_births, "g-", label="Birth Rate", linewidth=2)
    ax1.fill_between(
        steps,
        mean_births - confidence_interval_births,
        mean_births + confidence_interval_births,
        color="green",
        alpha=0.2,
        label="95% CI (Births)",
    )

    # Plot deaths
    ax1.plot(steps, mean_deaths, "r-", label="Death Rate", linewidth=2)
    ax1.fill_between(
        steps,
        mean_deaths - confidence_interval_deaths,
        mean_deaths + confidence_interval_deaths,
        color="red",
        alpha=0.2,
        label="95% CI (Deaths)",
    )

    # Plot net change
    net_change = mean_births - mean_deaths
    ax1.plot(steps, net_change, "b--", label="Net Population Change", linewidth=1.5)

    # Customize top subplot
    ax1.set_title("Full Range View", fontsize=12, pad=10)
    ax1.set_ylabel("Number of Agents", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot stable period in bottom subplot
    settling_point = 55  # Skip first 55 steps to avoid initial spike

    # Plot births for stable period
    ax2.plot(
        steps[settling_point:],
        mean_births[settling_point:],
        "g-",
        label="Birth Rate",
        linewidth=2,
    )
    ax2.fill_between(
        steps[settling_point:],
        mean_births[settling_point:] - confidence_interval_births[settling_point:],
        mean_births[settling_point:] + confidence_interval_births[settling_point:],
        color="green",
        alpha=0.2,
        label="95% CI (Births)",
    )

    # Plot deaths for stable period
    ax2.plot(
        steps[settling_point:],
        mean_deaths[settling_point:],
        "r-",
        label="Death Rate",
        linewidth=2,
    )
    ax2.fill_between(
        steps[settling_point:],
        mean_deaths[settling_point:] - confidence_interval_deaths[settling_point:],
        mean_deaths[settling_point:] + confidence_interval_deaths[settling_point:],
        color="red",
        alpha=0.2,
        label="95% CI (Deaths)",
    )

    # Plot net change for stable period
    ax2.plot(
        steps[settling_point:],
        net_change[settling_point:],
        "b--",
        label="Net Population Change",
        linewidth=1.5,
    )

    # Customize bottom subplot
    ax2.set_title("Stable Period View", fontsize=12, pad=10)
    ax2.set_xlabel("Simulation Step", fontsize=12)
    ax2.set_ylabel("Number of Agents", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, "temp_birth_death_rates.png")
    plt.savefig(temp_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Add the note using Pillow
    note_text = (
        "Note: This visualization shows the average birth and death rates across all simulations. "
        "The top panel shows the full range including initial fluctuations, while the bottom panel "
        "focuses on the stable period after initial population changes settle. The shaded areas "
        "represent 95% confidence intervals, and the dashed blue line shows the net population "
        "change (births minus deaths) at each step."
    )

    final_path = os.path.join(output_dir, "birth_death_rates.png")
    add_note_to_image(temp_path, note_text)

    # Clean up temporary file
    os.remove(temp_path)


def calculate_agent_lifespans(db_path: str) -> Dict[str, Any]:
    """Calculate lifespan statistics for agents in a simulation."""
    db = SimulationDatabase(db_path)
    session = db.Session()

    try:
        # Query all agents with both birth and death times
        agents = (
            session.query(AgentModel)
            .filter(AgentModel.birth_time.isnot(None))
            .filter(AgentModel.death_time.isnot(None))
        )
        if not agents:
            return {
                "avg_lifespan": 0,
                "median_lifespan": 0,
                "phase_lifespans": {},
                "lifespans_by_type": {},
            }

        # Calculate individual lifespans
        lifespans = [
            float(str(agent.death_time - agent.birth_time)) for agent in agents
        ]

        # Get simulation duration for phase analysis
        max_step = session.query(func.max(SimulationStepModel.step_number)).scalar()
        if max_step:
            # Split into 5 phases instead of 3
            phase_size = max_step // 5
            phases = {
                "initial": (0, phase_size),
                "early": (phase_size, 2 * phase_size),
                "mid": (2 * phase_size, 3 * phase_size),
                "late": (3 * phase_size, 4 * phase_size),
                "final": (4 * phase_size, max_step),
            }

            # Calculate lifespans for each phase
            phase_lifespans = {}
            for phase_name, (start, end) in phases.items():
                phase_agents = [
                    float(str(agent.death_time - agent.birth_time))
                    for agent in agents
                    if start <= agent.birth_time < end
                ]
                print(phase_agents)
                if phase_agents:
                    phase_lifespans[phase_name] = {
                        "mean": float(sum(phase_agents) / len(phase_agents)),
                        "count": len(phase_agents),
                    }
                else:
                    phase_lifespans[phase_name] = {"mean": 0.0, "count": 0}

            # Calculate lifespans by agent type - fixed to properly handle agent_type
            lifespans_by_type = {}
            for agent_type in ["system", "independent", "control"]:
                type_agents = [
                    agent
                    for agent in agents
                    if hasattr(agent, "agent_type")
                    and str(agent.agent_type) == agent_type
                ]
                if type_agents:
                    type_lifespans = [
                        float(str(agent.death_time - agent.birth_time))
                        for agent in type_agents
                    ]
                    lifespans_by_type[agent_type] = {
                        "mean": float(sum(type_lifespans) / len(type_lifespans)),
                        "median": float(np.median(type_lifespans)),
                        "std": float(np.std(type_lifespans)),
                        "count": len(type_lifespans),
                    }
                else:
                    lifespans_by_type[agent_type] = {
                        "mean": 0.0,
                        "median": 0.0,
                        "std": 0.0,
                        "count": 0,
                    }

            return {
                "avg_lifespan": float(sum(lifespans) / len(lifespans)),
                "median_lifespan": float(np.median(lifespans)),
                "phase_lifespans": phase_lifespans,
                "lifespans_by_type": lifespans_by_type,
            }

        return {
            "avg_lifespan": float(sum(lifespans) / len(lifespans)),
            "median_lifespan": float(np.median(lifespans)),
            "phase_lifespans": {},
            "lifespans_by_type": {},
        }

    finally:
        session.close()
        db.close()


def plot_agent_lifespans(df: pd.DataFrame, output_dir: str):
    """Create visualizations for agent lifespan analysis."""
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)

    # 1. Phase Analysis (top left)
    ax1 = fig.add_subplot(gs[0, 0])

    # Extract phase data from all simulations
    phase_names = ["initial", "early", "mid", "late", "final"]
    phase_means = []

    for phase in phase_names:
        phase_values = []
        for sim_data in df["lifespan_data"]:
            if sim_data.get("phase_lifespans"):
                phase_values.append(
                    sim_data["phase_lifespans"].get(phase, {}).get("mean", 0)
                )
        phase_means.append(np.mean(phase_values) if phase_values else 0)

    bars = ax1.bar(phase_names, phase_means)
    ax1.set_title("Average Lifespan by Simulation Phase")
    ax1.set_ylabel("Steps")
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    # 2. Agent Type Comparison (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    agent_types = ["system", "independent", "control"]

    # Calculate means and standard deviations for each agent type
    type_stats = {agent_type: {"means": [], "counts": []} for agent_type in agent_types}

    for sim_data in df["lifespan_data"]:
        for agent_type in agent_types:
            type_data = sim_data.get("lifespans_by_type", {}).get(agent_type, {})
            if type_data.get("count", 0) > 0:  # Only include if agents existed
                type_stats[agent_type]["means"].append(type_data.get("mean", 0))
                type_stats[agent_type]["counts"].append(type_data.get("count", 0))

    # Calculate weighted means and standard errors
    type_means = []
    type_errors = []

    for agent_type in agent_types:
        means = type_stats[agent_type]["means"]
        counts = type_stats[agent_type]["counts"]

        if means and counts:
            weighted_mean = np.average(means, weights=counts)
            type_means.append(weighted_mean)
            # Calculate weighted standard error
            variance = np.average(
                (np.array(means) - weighted_mean) ** 2, weights=counts
            )
            type_errors.append(
                np.sqrt(variance / sum(counts)) if sum(counts) > 0 else 0
            )
        else:
            type_means.append(0)
            type_errors.append(0)

    bars = ax2.bar(agent_types, type_means, yerr=type_errors, capsize=5)
    ax2.set_title("Average Lifespan by Agent Type")
    ax2.set_ylabel("Steps")

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    # 3. Lifespan Distribution (bottom)
    ax3 = fig.add_subplot(gs[1, :])

    # Create violin plot for lifespan distribution
    all_lifespans = []
    for sim_data in df["lifespan_data"]:
        if sim_data.get("lifespans_by_type"):
            for agent_type in agent_types:
                type_data = sim_data["lifespans_by_type"].get(agent_type, {})
                if type_data.get("count", 0) > 0:
                    all_lifespans.append(type_data["mean"])

    if all_lifespans:
        violin_parts = ax3.violinplot([all_lifespans], positions=[1])
        ax3.set_title("Distribution of Agent Lifespans")
        ax3.set_xticks([1])
        ax3.set_xticklabels(["All Agents"])
        ax3.set_ylabel("Steps")

        # Customize violin plot colors
        if violin_parts.get("bodies"):
            violin_parts["bodies"].set_facecolor("#3498db")
            violin_parts["bodies"].set_edgecolor("black")
            violin_parts["bodies"].set_alpha(0.7)

    # Adjust layout
    plt.tight_layout()

    # Save the figure without the note first
    temp_path = os.path.join(output_dir, "temp_agent_lifespans.png")
    plt.savefig(temp_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Add the note using Pillow
    note_text = (
        "Note: This visualization shows agent lifespan analysis across simulations. "
        "The top left panel shows average lifespans in different simulation phases, "
        "the top right panel compares lifespans across agent types, and the bottom "
        "panel shows the distribution of lifespans across all agents."
    )

    final_path = os.path.join(output_dir, "agent_lifespans.png")
    add_note_to_image(temp_path, note_text)

    # Clean up temporary file
    os.remove(temp_path)


def plot_comparative_metrics(
    df: pd.DataFrame, output_dir: str = "simulations/analysis"
):
    """Generate comparative visualizations of simulation metrics."""
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    plot_population_dynamics(df, output_dir)
    plot_resource_utilization(df, output_dir)
    plot_health_reward_distribution(df, output_dir)
    plot_population_vs_cycles(df, output_dir)
    plot_population_vs_age(df, output_dir)
    plot_population_trends_across_simulations(df, output_dir)
    plot_population_trends_by_type_across_simulations(df, output_dir)
    plot_population_candlestick(df, output_dir)
    plot_birth_death_rates(df, output_dir)
    plot_agent_lifespans(df, output_dir)


def compare_simulations(search_path: str, analysis_path: str) -> None:
    """
    Compare multiple simulations and generate analysis.

    Parameters
    ----------
    search_path : str
        Base path to search for simulation database files
    analysis_path : str
        Path where analysis results should be saved
    """
    print(f"Analyzing simulations from: {search_path}")
    print(f"Results will be saved to: {analysis_path}")

    # Find all simulation.db files in subdirectories
    db_paths = []
    for root, dirs, files in os.walk(search_path):
        if "simulation.db" in files:
            db_path = os.path.join(root, "simulation.db")
            db_paths.append(db_path)

    if not db_paths:
        print(f"No simulation.db files found in {search_path} or its subdirectories")
        return

    print(f"Found {len(db_paths)} database files")

    # Create results DataFrame
    results = []

    for path in db_paths:
        print(f"\nAnalyzing simulation: {path}")
        meta = gather_simulation_metadata(path)
        results.append(meta)

    results_df = pd.DataFrame(results)

    # Save results to CSV
    csv_path = os.path.join(analysis_path, "simulation_comparison.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved comparison results to: {csv_path}")

    # Create visualizations
    create_comparison_plots(results_df, analysis_path)

    # Create dashboard summary
    create_dashboard_summary(results_df, analysis_path)

    # Create experiment analysis
    create_experiment_analysis(results_df, analysis_path)


def create_comparison_plots(results_df: pd.DataFrame, analysis_path: str) -> None:
    """Create and save comparison visualizations."""
    plots_path = os.path.join(analysis_path, "plots")
    os.makedirs(plots_path, exist_ok=True)

    # 1. Population metrics plot
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["mean_population"], label="Mean")
    plt.plot(results_df["median_population"], label="Median")
    plt.plot(results_df["mode_population"], label="Mode")
    plt.title("Population Statistics Across Simulations")
    plt.xlabel("Simulation Index")
    plt.ylabel("Population")
    plt.legend()
    plt.savefig(os.path.join(plots_path, "population_metrics.png"))
    plt.close()

    # 2. Health and Resources
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["avg_health"], label="Average Health", color="green")
    plt.plot(results_df["avg_total_resources"], label="Average Resources", color="blue")
    plt.title("Health and Resources Over Time")
    plt.xlabel("Simulation Index")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig(os.path.join(plots_path, "health_resources.png"))
    plt.close()

    # 3. Birth and Death Rates
    plt.figure(figsize=(12, 6))
    plt.plot(results_df["avg_births"], label="Birth Rate", color="green")
    plt.plot(results_df["avg_deaths"], label="Death Rate", color="red")
    plt.title("Birth and Death Rates")
    plt.xlabel("Simulation Index")
    plt.ylabel("Rate")
    plt.legend()
    plt.savefig(os.path.join(plots_path, "birth_death_rates.png"))
    plt.close()

    # 4. Rewards Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(results_df["avg_reward"], bins=20, color="purple", alpha=0.7)
    plt.title("Distribution of Average Rewards")
    plt.xlabel("Average Reward")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(plots_path, "reward_distribution.png"))
    plt.close()

    # 5. Population vs Resources Scatter
    plt.figure(figsize=(12, 6))
    plt.scatter(
        results_df["avg_total_agents"],
        results_df["avg_total_resources"],
        alpha=0.6,
        c=results_df["avg_reward"],
        cmap="viridis",
    )
    plt.colorbar(label="Average Reward")
    plt.title("Population vs Resources (colored by reward)")
    plt.xlabel("Average Population")
    plt.ylabel("Average Resources")
    plt.savefig(os.path.join(plots_path, "population_resources_scatter.png"))
    plt.close()

    # 6. Simulation Duration Analysis
    plt.figure(figsize=(12, 6))
    plt.bar(
        range(len(results_df)), results_df["duration_steps"], color="skyblue", alpha=0.7
    )
    plt.title("Simulation Durations")
    plt.xlabel("Simulation Index")
    plt.ylabel("Number of Steps")
    plt.savefig(os.path.join(plots_path, "simulation_durations.png"))
    plt.close()

    print(f"Saved {6} plots to {plots_path}")


def create_dashboard_summary(results_df: pd.DataFrame, analysis_path: str) -> None:
    """Create and save a dashboard summary of the analysis."""
    summary_path = os.path.join(analysis_path, "dashboard_summary.html")

    # Create HTML summary
    html_content = f"""
    <html>
    <head>
        <title>Simulation Analysis Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .metric {{ margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }}
            .metric h3 {{ color: #333; margin-top: 0; }}
            .value {{ color: #2980b9; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Simulation Analysis Summary</h1>
        
        <div class="metric">
            <h3>Population Statistics</h3>
            <p>Mean Population: <span class="value">{results_df['mean_population'].mean():.2f}</span></p>
            <p>Median Population: <span class="value">{results_df['median_population'].mean():.2f}</span></p>
            <p>Mode Population: <span class="value">{results_df['mode_population'].mean():.2f}</span></p>
        </div>

        <div class="metric">
            <h3>Health and Resources</h3>
            <p>Average Health: <span class="value">{results_df['avg_health'].mean():.2f}</span></p>
            <p>Average Resources: <span class="value">{results_df['avg_total_resources'].mean():.2f}</span></p>
        </div>

        <div class="metric">
            <h3>Population Dynamics</h3>
            <p>Average Birth Rate: <span class="value">{results_df['avg_births'].mean():.2f}</span></p>
            <p>Average Death Rate: <span class="value">{results_df['avg_deaths'].mean():.2f}</span></p>
        </div>

        <div class="metric">
            <h3>Performance Metrics</h3>
            <p>Average Reward: <span class="value">{results_df['avg_reward'].mean():.2f}</span></p>
            <p>Average Duration: <span class="value">{results_df['duration_steps'].mean():.1f} steps</span></p>
        </div>
    </body>
    </html>
    """

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Saved dashboard summary to: {summary_path}")


def create_experiment_analysis(results_df: pd.DataFrame, analysis_path: str) -> None:
    """Create structured analysis of experiment results for both human and machine consumption."""

    # Calculate key metrics and their significance
    metrics = {
        "population": {
            "mean": float(results_df["mean_population"].mean()),
            "trend": (
                "increasing"
                if results_df["mean_population"].is_monotonic_increasing
                else (
                    "decreasing"
                    if results_df["mean_population"].is_monotonic_decreasing
                    else "fluctuating"
                )
            ),
            "stability": float(
                results_df["mean_population"].std()
                / results_df["mean_population"].mean()
            ),
            "expectation_met": bool(results_df["mean_population"].mean() > 10),
        },
        "resources": {
            "mean": float(results_df["avg_total_resources"].mean()),
            "efficiency": float(
                results_df["avg_total_resources"].mean()
                / results_df["mean_population"].mean()
            ),
            "sustainability": bool(
                results_df["avg_total_resources"].std()
                < results_df["avg_total_resources"].mean() * 0.5
            ),
        },
        "health": {
            "mean": float(results_df["avg_health"].mean()),
            "consistency": float(
                results_df["avg_health"].std() / results_df["avg_health"].mean()
            ),
        },
        "reproduction": {
            "birth_rate": float(results_df["avg_births"].mean()),
            "death_rate": float(results_df["avg_deaths"].mean()),
            "sustainability": bool(
                results_df["avg_births"].mean() > results_df["avg_deaths"].mean()
            ),
        },
    }

    # Generate analysis conclusions
    conclusions = []
    if metrics["population"]["expectation_met"]:
        conclusions.append("Population levels met or exceeded expectations")
    else:
        conclusions.append("Population levels were below target")

    if metrics["reproduction"]["sustainability"]:
        conclusions.append("Population showed sustainable growth")
    else:
        conclusions.append("Population growth may not be sustainable")

    if metrics["resources"]["sustainability"]:
        conclusions.append("Resource management was stable")
    else:
        conclusions.append("Resource management needs improvement")

    # Create machine-readable JSON
    analysis_dict = {
        "experiment_id": os.path.basename(analysis_path),
        "timestamp": pd.Timestamp.now().isoformat(),
        "metrics": metrics,
        "conclusions": conclusions,
        "recommendations": [
            (
                "Adjust reproduction threshold"
                if not metrics["reproduction"]["sustainability"]
                else None
            ),
            (
                "Modify resource distribution"
                if not metrics["resources"]["sustainability"]
                else None
            ),
            (
                "Review population control parameters"
                if not metrics["population"]["expectation_met"]
                else None
            ),
        ],
        "comparison_to_baseline": {
            "population_performance": (
                "above" if metrics["population"]["mean"] > 15 else "below"
            ),
            "resource_efficiency": (
                "efficient"
                if metrics["resources"]["efficiency"] > 0.8
                else "inefficient"
            ),
            "health_stability": (
                "stable" if metrics["health"]["consistency"] < 0.2 else "unstable"
            ),
        },
    }

    # Save JSON analysis
    json_path = os.path.join(analysis_path, "experiment_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_dict, f, indent=2)

    print(f"Saved experiment analysis to {json_path}")

    # Create human-readable markdown
    markdown_content = f"""# Experiment Analysis Report
    
## Overview
Experiment ID: {analysis_dict['experiment_id']}
Analysis Date: {analysis_dict['timestamp']}

## Key Metrics
### Population
- Mean Population: {metrics['population']['mean']:.2f}
- Population Trend: {metrics['population']['trend']}
- Population Stability: {metrics['population']['stability']:.2%}

### Resources
- Mean Resources: {metrics['resources']['mean']:.2f}
- Resource Efficiency: {metrics['resources']['efficiency']:.2f}
- Resource Sustainability: {'Yes' if metrics['resources']['sustainability'] else 'No'}

### Health
- Mean Health: {metrics['health']['mean']:.2f}
- Health Consistency: {metrics['health']['consistency']:.2%}

### Reproduction
- Birth Rate: {metrics['reproduction']['birth_rate']:.2f}
- Death Rate: {metrics['reproduction']['death_rate']:.2f}
- Sustainable: {'Yes' if metrics['reproduction']['sustainability'] else 'No'}

## Conclusions
{chr(10).join(f"- {conclusion}" for conclusion in conclusions)}

## Recommendations
{chr(10).join(f"- {rec}" for rec in analysis_dict['recommendations'] if rec)}

## Comparison to Baseline
- Population Performance: {analysis_dict['comparison_to_baseline']['population_performance']}
- Resource Efficiency: {analysis_dict['comparison_to_baseline']['resource_efficiency']}
- Health Stability: {analysis_dict['comparison_to_baseline']['health_stability']}
"""

    # Save markdown analysis
    markdown_path = os.path.join(analysis_path, "experiment_analysis.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    print(f"Saved experiment analysis to {json_path} and {markdown_path}")


def create_interactive_analysis_window(df: pd.DataFrame):
    """Create an interactive window for population vs age analysis."""
    import tkinter as tk
    from tkinter import ttk

    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Create the main window
    root = tk.Tk()
    root.title("Interactive Population Analysis")
    root.geometry("1000x800")

    # Create frame for controls
    control_frame = ttk.Frame(root, padding="5")
    control_frame.pack(fill=tk.X)

    # Create frame for the plot
    plot_frame = ttk.Frame(root, padding="5")
    plot_frame.pack(fill=tk.BOTH, expand=True)

    # Variables for plot customization
    color_var = tk.StringVar(value="avg_births")
    x_var = tk.StringVar(value="avg_total_agents")
    y_var = tk.StringVar(value="avg_agent_age")

    available_metrics = [
        "avg_births",
        "avg_deaths",
        "avg_health",
        "avg_reward",
        "avg_total_resources",
        "median_population",
        "mode_population",
        "avg_total_agents",
        "avg_agent_age",
        "duration_steps",
    ]

    # Create the figure and canvas
    fig, ax = plt.subplots(figsize=(10, 6))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_plot():
        """Update the plot based on current settings."""
        # Clear the entire figure
        fig.clear()
        ax = fig.add_subplot(111)

        # Create scatter plot with selected metrics
        scatter = ax.scatter(
            df[x_var.get()],
            df[y_var.get()],
            c=df[color_var.get()],
            s=100,
            alpha=0.6,
            cmap="viridis",
        )

        # Add colorbar
        fig.colorbar(scatter, ax=ax, label=color_var.get())

        # Calculate correlation
        correlation = df[x_var.get()].corr(df[y_var.get()])

        # Add trend line
        z = np.polyfit(df[x_var.get()], df[y_var.get()], 1)
        p = np.poly1d(z)
        ax.plot(
            df[x_var.get()],
            p(df[x_var.get()]),
            "r--",
            alpha=0.8,
            label=f"Trend line (slope: {z[0]:.2f})",
        )

        # Set labels and title
        ax.set_title(f"{x_var.get()} vs {y_var.get()} (r = {correlation:.2f})")
        ax.set_xlabel(x_var.get().replace("_", " ").title())
        ax.set_ylabel(y_var.get().replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Refresh canvas
        fig.tight_layout()
        canvas.draw()

    # Create controls
    ttk.Label(control_frame, text="X Axis:").pack(side=tk.LEFT, padx=5)
    x_menu = ttk.Combobox(
        control_frame, textvariable=x_var, values=available_metrics, width=20
    )
    x_menu.pack(side=tk.LEFT, padx=5)

    ttk.Label(control_frame, text="Y Axis:").pack(side=tk.LEFT, padx=5)
    y_menu = ttk.Combobox(
        control_frame, textvariable=y_var, values=available_metrics, width=20
    )
    y_menu.pack(side=tk.LEFT, padx=5)

    ttk.Label(control_frame, text="Color by:").pack(side=tk.LEFT, padx=5)
    color_menu = ttk.Combobox(
        control_frame, textvariable=color_var, values=available_metrics, width=20
    )
    color_menu.pack(side=tk.LEFT, padx=5)

    # Bind events
    x_menu.bind("<<ComboboxSelected>>", lambda e: update_plot())
    y_menu.bind("<<ComboboxSelected>>", lambda e: update_plot())
    color_menu.bind("<<ComboboxSelected>>", lambda e: update_plot())

    # Initial plot
    update_plot()

    # Start the GUI event loop
    root.mainloop()


def main(path: str):
    # Find all simulation databases
    db_paths = find_simulation_databases(path)

    if not db_paths:
        print(f"No simulation databases found in path: {path}")
        return

    print(f"Found {len(db_paths)} database files")

    # Compare and analyze simulations
    compare_simulations(path, path)


if __name__ == "__main__":
    main("experiments/initial_experiments/databases")

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.patheffects
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sqlalchemy import func

from farm.database.database import SimulationDatabase
from farm.database.models import (
    ActionModel,
    AgentModel,
    LearningExperienceModel,
    SimulationStepModel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------
# Helper Functions
# ---------------------


def find_simulation_databases(base_path: str) -> List[str]:
    """
    Find all simulation database files in the given directory and its subdirectories.
    Creates the base directory if it doesn't exist.
    """
    base = Path(base_path)
    try:
        base.mkdir(parents=True, exist_ok=True)
        logger.info(f"Searching for databases in: {base.resolve()}")

        db_files = list(base.rglob("simulation.db"))
        if not db_files:
            logger.warning(f"No simulation.db files found in {base}")
        else:
            logger.info(f"Found {len(db_files)} database files:")
            for db_file in db_files:
                logger.info(f"  - {db_file}")
        return sorted(str(path) for path in db_files)
    except Exception as e:
        logger.error(f"Error creating/accessing directory {base}: {str(e)}")
        return []


def create_population_df(
    all_populations: List[np.ndarray], max_steps: int, is_resource_data: bool = False
) -> pd.DataFrame:
    """
    Create a DataFrame from population or resource data with proper padding for missing steps.
    Includes data validation.

    Args:
        all_populations: List of data arrays
        max_steps: Maximum number of steps
        is_resource_data: Whether this is resource level data (allows negative values)

    Returns:
        DataFrame with the processed data
    """
    if not all_populations:
        logger.error("No data provided")
        return pd.DataFrame(columns=["simulation_id", "step", "population"])

    # Validate each data array
    valid_data = []
    for i, pop in enumerate(all_populations):
        if is_resource_data:
            # Use resource validation for resource data
            if validate_resource_level_data(pop):
                valid_data.append(pop)
            else:
                logger.warning(f"Skipping invalid resource data from simulation {i}")
        else:
            # Use population validation for population data
            if validate_population_data(pop):
                valid_data.append(pop)
            else:
                logger.warning(f"Skipping invalid population data from simulation {i}")

    if not valid_data:
        logger.error("No valid data after validation")
        return pd.DataFrame(columns=["simulation_id", "step", "population"])

    # Create DataFrame with valid data
    data = []
    for sim_idx, pop in enumerate(valid_data):
        for step in range(max_steps):
            population = pop[step] if step < len(pop) else np.nan
            data.append((f"sim_{sim_idx}", step, population))

    df = pd.DataFrame(data, columns=["simulation_id", "step", "population"])

    # Final validation of the DataFrame
    if df.empty:
        logger.warning("Created DataFrame is empty")
    elif df["population"].isna().all():
        logger.warning("All values are NaN in the DataFrame")

    return df


def calculate_statistics(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate population statistics with validation checks.
    """
    if df.empty:
        logger.error("Cannot calculate statistics: DataFrame is empty")
        return np.array([]), np.array([]), np.array([]), np.array([])

    n_simulations = df["simulation_id"].nunique()
    if n_simulations == 0:
        logger.error("No valid simulations found in data")
        return np.array([]), np.array([]), np.array([]), np.array([])

    grouped = df.groupby("step")["population"]

    # Calculate statistics with handling for all-NaN groups
    mean_pop = grouped.mean().fillna(0).values
    median_pop = grouped.median().fillna(0).values
    std_pop = grouped.std().fillna(0).values

    # Avoid division by zero in confidence interval calculation
    confidence_interval = np.where(
        n_simulations > 0, 1.96 * std_pop / np.sqrt(n_simulations), 0
    )

    return mean_pop, median_pop, std_pop, confidence_interval


def validate_population_data(population: np.ndarray, db_path: str = None) -> bool:
    """
    Validate population data array for integrity.

    Args:
        population: Array of population values to validate
        db_path: Optional database path for error logging

    Returns:
        bool: True if data is valid, False otherwise
    """
    if population is None:
        logger.warning(f"Population data is None{f' in {db_path}' if db_path else ''}")
        return False

    if len(population) == 0:
        logger.warning(f"Empty population data{f' in {db_path}' if db_path else ''}")
        return False

    if np.all(np.isnan(population)):
        logger.warning(
            f"Population data contains only NaN values{f' in {db_path}' if db_path else ''}"
        )
        return False

    if np.any(population < 0):
        logger.warning(
            f"Population data contains negative values{f' in {db_path}' if db_path else ''}"
        )
        return False

    return True


def validate_resource_level_data(
    resource_levels: np.ndarray, db_path: str = None
) -> bool:
    """
    Validate resource level data array for integrity.
    Unlike population data, resource levels can be negative.

    Args:
        resource_levels: Array of resource level values to validate
        db_path: Optional database path for error logging

    Returns:
        bool: True if data is valid, False otherwise
    """
    if resource_levels is None:
        logger.warning(
            f"Resource level data is None{f' in {db_path}' if db_path else ''}"
        )
        return False

    if len(resource_levels) == 0:
        logger.warning(
            f"Empty resource level data{f' in {db_path}' if db_path else ''}"
        )
        return False

    if np.all(np.isnan(resource_levels)):
        logger.warning(
            f"Resource level data contains only NaN values{f' in {db_path}' if db_path else ''}"
        )
        return False

    # We allow negative values for resource levels
    return True


# ---------------------
# Database Interaction
# ---------------------


def get_columns_data(
    experiment_db_path: str, columns: List[str]
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """
    Retrieve specified columns from the simulation database with robust error handling.

    Args:
        experiment_db_path: Path to the database file
        columns: List of column names to retrieve

    Returns:
        Tuple containing:
        - steps: Array of step numbers
        - populations: Dictionary mapping column names to their data arrays
        - max_steps: Number of steps in the simulation
    """
    if not os.path.exists(experiment_db_path):
        logger.error(f"Database file not found: {experiment_db_path}")
        return None, {}, 0

    db = None
    session = None
    try:
        db = SimulationDatabase(experiment_db_path)
        session = db.Session()

        # Validate requested columns exist
        for col in columns:
            if not hasattr(SimulationStepModel, col):
                logger.error(f"Column '{col}' not found in database schema")
                return None, {}, 0

        # Build query dynamically based on requested columns
        query_columns = [getattr(SimulationStepModel, col) for col in columns]
        query = (
            session.query(SimulationStepModel.step_number, *query_columns)
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        if not query:
            logger.warning(f"No data found in database: {experiment_db_path}")
            return None, {}, 0

        # Extract step numbers and column data
        steps = np.array([row[0] for row in query])
        populations = {
            col: np.array([row[i + 1] for row in query])
            for i, col in enumerate(columns)
        }

        # Validate each column's data with the appropriate validation function
        for col, data in populations.items():
            if col == "average_agent_resources":
                # Use resource level validation for resource data
                if not validate_resource_level_data(data, experiment_db_path):
                    logger.error(
                        f"Invalid resource level data for column '{col}' in {experiment_db_path}"
                    )
                    return None, {}, 0
            else:
                # Use population validation for other data
                if not validate_population_data(data, experiment_db_path):
                    logger.error(
                        f"Invalid data for column '{col}' in {experiment_db_path}"
                    )
                    return None, {}, 0

        max_steps = len(steps)

        # Validate steps
        if len(steps) == 0:
            logger.error(f"No steps found in database: {experiment_db_path}")
            return None, {}, 0

        if not np.all(np.diff(steps) >= 0):
            logger.error(
                f"Steps are not monotonically increasing in {experiment_db_path}"
            )
            return None, {}, 0

        return steps, populations, max_steps

    except Exception as e:
        logger.error(f"Error accessing database {experiment_db_path}: {str(e)}")
        return None, {}, 0
    finally:
        if session:
            session.close()
        if db:
            db.close()


def get_data(experiment_db_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Retrieve step numbers and total agents from the simulation database.
    This is a wrapper around get_columns_data for backward compatibility.
    """
    steps, populations, max_steps = get_columns_data(
        experiment_db_path, ["total_agents"]
    )
    if steps is not None:
        return steps, populations["total_agents"], max_steps
    return None, None, 0


def get_columns_data_by_agent_type(
    experiment_db_path: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """
    Retrieve population data for each agent type from the simulation database.
    """
    columns = ["system_agents", "control_agents", "independent_agents"]
    return get_columns_data(experiment_db_path, columns)


def get_resource_consumption_data(
    experiment_db_path: str,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """
    Retrieve resource consumption data for each agent type from the simulation database.

    Args:
        experiment_db_path: Path to the database file

    Returns:
        Tuple containing:
        - steps: Array of step numbers
        - consumption: Dictionary mapping agent types to their consumption data arrays
        - max_steps: Number of steps in the simulation
    """
    columns = [
        "system_agents",
        "control_agents",
        "independent_agents",
        "resources_consumed",
    ]
    steps, data, max_steps = get_columns_data(experiment_db_path, columns)

    if steps is None or not data:
        return None, {}, 0

    # Calculate consumption per agent type
    consumption = {}
    total_agents = np.zeros_like(steps, dtype=float)

    # Sum up all agent types to get total population
    for agent_type in ["system_agents", "control_agents", "independent_agents"]:
        if agent_type in data:
            total_agents += data[agent_type]

    # Calculate consumption proportionally for each agent type
    for agent_type in ["system", "control", "independent"]:
        db_column = f"{agent_type}_agents"
        if db_column in data and "resources_consumed" in data:
            # Avoid division by zero
            safe_total = np.where(total_agents > 0, total_agents, 1)
            consumption[agent_type] = (
                data[db_column] * data["resources_consumed"] / safe_total
            )

    return steps, consumption, max_steps


def get_action_distribution_data(experiment_db_path: str) -> Dict[str, Dict[str, int]]:
    """
    Retrieve action distribution data for each agent type from the simulation database.

    Args:
        experiment_db_path: Path to the database file

    Returns:
        Dictionary mapping agent types to their action distributions
    """
    if not os.path.exists(experiment_db_path):
        logger.error(f"Database file not found: {experiment_db_path}")
        return {}

    db = None
    session = None
    try:
        db = SimulationDatabase(experiment_db_path)
        session = db.Session()

        # Query to get agent types
        agent_types = session.query(AgentModel.agent_type).distinct().all()
        agent_types = [t[0] for t in agent_types]

        # Initialize result structure
        result = {agent_type: {} for agent_type in agent_types}

        # For each agent type, get action distribution
        for agent_type in agent_types:
            # Join agents with their actions and count by action type
            query = (
                session.query(
                    ActionModel.action_type,
                    func.count(ActionModel.action_id).label("count"),
                )
                .join(AgentModel, AgentModel.agent_id == ActionModel.agent_id)
                .filter(AgentModel.agent_type == agent_type)
                .group_by(ActionModel.action_type)
                .all()
            )

            # Store results
            action_counts = {action_type: count for action_type, count in query}
            result[agent_type] = action_counts

        return result

    except Exception as e:
        logger.error(f"Error accessing database {experiment_db_path}: {str(e)}")
        return {}
    finally:
        if session:
            session.close()
        if db:
            db.close()


def get_resource_level_data(
    experiment_db_path: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Retrieve resource level data from the simulation database.

    Args:
        experiment_db_path: Path to the database file

    Returns:
        Tuple containing:
        - steps: Array of step numbers
        - resource_levels: Array of average agent resource levels
        - max_steps: Number of steps in the simulation
    """
    columns = ["average_agent_resources"]
    steps, data, max_steps = get_columns_data(experiment_db_path, columns)

    if steps is None or not data:
        return None, None, 0

    return steps, data["average_agent_resources"], max_steps


def get_rewards_by_generation(experiment_db_path: str) -> Dict[int, float]:
    """
    Retrieve average rewards grouped by agent generation from the simulation database.

    Args:
        experiment_db_path: Path to the database file

    Returns:
        Dictionary mapping generation numbers to their average rewards
    """
    if not os.path.exists(experiment_db_path):
        logger.error(f"Database file not found: {experiment_db_path}")
        return {}

    db = None
    session = None
    try:
        db = SimulationDatabase(experiment_db_path)
        session = db.Session()

        # Query to get average reward by generation
        # Join agents with their learning experiences
        query = (
            session.query(
                AgentModel.generation,
                func.avg(LearningExperienceModel.reward).label("avg_reward"),
                func.count(LearningExperienceModel.experience_id).label(
                    "experience_count"
                ),
            )
            .join(
                LearningExperienceModel,
                LearningExperienceModel.agent_id == AgentModel.agent_id,
            )
            .group_by(AgentModel.generation)
            .having(
                func.count(LearningExperienceModel.experience_id) > 0
            )  # Only include generations with data
            .order_by(AgentModel.generation)
            .all()
        )

        if not query:
            logger.warning(
                f"No reward data by generation found in database: {experiment_db_path}"
            )
            return {}

        # Convert query results to dictionary
        rewards_by_generation = {
            generation: avg_reward for generation, avg_reward, _ in query
        }

        return rewards_by_generation

    except Exception as e:
        logger.error(f"Error accessing database {experiment_db_path}: {str(e)}")
        return {}
    finally:
        if session:
            session.close()
        if db:
            db.close()


# ---------------------
# Plotting Helpers
# ---------------------


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
    all_populations: List[np.ndarray], max_steps: int, output_path: str
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
    experiment_data: Dict[str, Dict], output_dir: str
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


def plot_resource_consumption_trends(experiment_data: Dict[str, Dict], output_dir: str):
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


def detect_early_terminations(
    db_paths: List[str], expected_steps: int = None
) -> Dict[str, Dict[str, Any]]:
    """
    Detect and analyze simulations that terminated earlier than expected.

    Args:
        db_paths: List of paths to simulation database files
        expected_steps: Expected number of steps (if None, will use max steps found)

    Returns:
        Dictionary mapping database paths to termination analysis
    """
    logger.info(f"Analyzing {len(db_paths)} simulations for early termination")

    results = {}
    all_step_counts = []

    # First pass: collect step counts for all simulations
    for db_path in db_paths:
        try:
            steps, _, steps_count = get_data(db_path)
            if steps is not None:
                all_step_counts.append(steps_count)
                results[db_path] = {"steps_completed": steps_count}
            else:
                logger.warning(f"Could not retrieve step data from {db_path}")
        except Exception as e:
            logger.error(f"Error analyzing {db_path}: {str(e)}")

    if not all_step_counts:
        logger.error("No valid step data found in any database")
        return {}

    # Determine expected step count if not provided
    if expected_steps is None:
        expected_steps = max(all_step_counts)
        logger.info(
            f"Using maximum observed steps ({expected_steps}) as expected duration"
        )

    # Set threshold for early termination (e.g., 90% of expected steps)
    early_threshold = int(expected_steps * 0.9)

    # Second pass: analyze early terminations
    early_terminations = {}
    for db_path, info in results.items():
        steps_completed = info["steps_completed"]

        # Check if this simulation ended early
        if steps_completed < early_threshold:
            try:
                # Get final state data
                steps, populations, _ = get_columns_data_by_agent_type(db_path)

                if steps is None or not populations:
                    logger.warning(f"Could not retrieve population data from {db_path}")
                    continue

                # Get resource consumption data
                _, consumption, _ = get_resource_consumption_data(db_path)

                # Analyze final state
                final_state = {
                    "steps_completed": steps_completed,
                    "expected_steps": expected_steps,
                    "completion_percentage": round(
                        steps_completed / expected_steps * 100, 1
                    ),
                    "final_populations": {
                        agent_type: (
                            populations.get(f"{agent_type}_agents", [])[-1]
                            if populations.get(f"{agent_type}_agents")
                            else 0
                        )
                        for agent_type in ["system", "control", "independent"]
                    },
                    "total_final_population": sum(
                        (
                            populations.get(f"{agent_type}_agents", [])[-1]
                            if populations.get(f"{agent_type}_agents")
                            else 0
                        )
                        for agent_type in ["system", "control", "independent"]
                    ),
                    "resource_consumption": {
                        agent_type: (
                            consumption.get(agent_type, [])[-1]
                            if agent_type in consumption and consumption[agent_type]
                            else 0
                        )
                        for agent_type in ["system", "control", "independent"]
                    },
                }

                # Determine likely cause of termination
                if final_state["total_final_population"] == 0:
                    final_state["likely_cause"] = "population_collapse"
                elif (
                    sum(final_state["resource_consumption"].values()) < 0.1
                ):  # Near-zero resource consumption
                    final_state["likely_cause"] = "resource_depletion"
                else:
                    final_state["likely_cause"] = "unknown"

                early_terminations[db_path] = final_state

            except Exception as e:
                logger.error(
                    f"Error analyzing early termination for {db_path}: {str(e)}"
                )

    logger.info(f"Found {len(early_terminations)} simulations that terminated early")
    return early_terminations


def plot_early_termination_analysis(
    early_terminations: Dict[str, Dict], output_dir: str
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
        cause_counts.values(),
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


def analyze_final_agent_counts(experiment_data: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Analyze the final agent counts by type across all simulations in an experiment.

    Args:
        experiment_data: Dictionary containing processed population data for each agent type

    Returns:
        Dictionary with summary statistics about final agent counts
    """
    logger.info("Analyzing final agent counts by agent type")

    result = {
        "system": {
            "total": 0,
            "mean": 0,
            "median": 0,
            "max": 0,
            "min": 0,
            "simulations": 0,
        },
        "control": {
            "total": 0,
            "mean": 0,
            "median": 0,
            "max": 0,
            "min": 0,
            "simulations": 0,
        },
        "independent": {
            "total": 0,
            "mean": 0,
            "median": 0,
            "max": 0,
            "min": 0,
            "simulations": 0,
        },
        "dominant_type_counts": {"system": 0, "control": 0, "independent": 0, "tie": 0},
    }

    # Count valid simulations for each agent type
    valid_simulations = 0

    # Extract final population values for each simulation
    final_populations = {"system": [], "control": [], "independent": []}

    # Process each agent type's data
    for agent_type, data in experiment_data.items():
        if not data["populations"] or len(data["populations"]) == 0:
            logger.warning(f"No valid population data for {agent_type} agents")
            continue

        # Extract final population count from each simulation
        for population in data["populations"]:
            if len(population) > 0:
                final_populations[agent_type].append(population[-1])

    # Count simulations where we have data for all agent types
    simulation_count = min(
        len(final_populations["system"]),
        len(final_populations["control"]),
        len(final_populations["independent"]),
    )

    if simulation_count == 0:
        logger.warning("No simulations with complete agent type data")
        return result

    # Determine dominant agent type for each simulation
    for i in range(simulation_count):
        counts = {
            "system": (
                final_populations["system"][i]
                if i < len(final_populations["system"])
                else 0
            ),
            "control": (
                final_populations["control"][i]
                if i < len(final_populations["control"])
                else 0
            ),
            "independent": (
                final_populations["independent"][i]
                if i < len(final_populations["independent"])
                else 0
            ),
        }

        # Find the dominant type
        max_count = max(counts.values())
        dominant_types = [t for t, c in counts.items() if c == max_count]

        if len(dominant_types) > 1:
            result["dominant_type_counts"]["tie"] += 1
        else:
            result["dominant_type_counts"][dominant_types[0]] += 1

    # Calculate statistics for each agent type
    for agent_type in ["system", "control", "independent"]:
        if final_populations[agent_type]:
            values = final_populations[agent_type]
            result[agent_type]["total"] = sum(values)
            result[agent_type]["mean"] = sum(values) / len(values)
            result[agent_type]["median"] = sorted(values)[len(values) // 2]
            result[agent_type]["max"] = max(values)
            result[agent_type]["min"] = min(values)
            result[agent_type]["simulations"] = len(values)

    return result


def plot_final_agent_counts(final_counts: Dict[str, Dict], output_dir: str):
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


def process_experiment_rewards_by_generation(
    experiment: str,
) -> Dict[str, Dict[int, float]]:
    """
    Process reward data by generation for each agent type in an experiment.

    Args:
        experiment: Name of the experiment

    Returns:
        Dictionary containing reward data by generation for each agent type
    """
    logger.info(f"Processing rewards by generation for experiment: {experiment}")

    result = {
        "system": {},
        "control": {},
        "independent": {},
    }

    try:
        experiment_path = f"results/one_of_a_kind/experiments/data/{experiment}"
        if not os.path.exists(experiment_path):
            logger.error(f"Experiment directory not found: {experiment_path}")
            return result

        db_paths = find_simulation_databases(experiment_path)
        if not db_paths:
            logger.error(f"No database files found in {experiment_path}")
            return result

        valid_dbs = 0
        failed_dbs = 0

        # Temporary storage for aggregating data across simulations
        all_rewards = {"system": {}, "control": {}, "independent": {}}
        reward_counts = {"system": {}, "control": {}, "independent": {}}

        for db_path in db_paths:
            try:
                # Get rewards by generation for all agents
                rewards_by_generation = get_rewards_by_generation(db_path)

                if not rewards_by_generation:
                    failed_dbs += 1
                    continue

                # Get agent types for each generation
                db = SimulationDatabase(db_path)
                session = db.Session()

                # Query to get generations and their agent types
                gen_types = (
                    session.query(AgentModel.generation, AgentModel.agent_type)
                    .distinct()
                    .all()
                )

                session.close()
                db.close()

                # Map generations to agent types
                gen_to_type = {}
                for gen, agent_type in gen_types:
                    # Normalize agent type names
                    if agent_type in ["system", "SystemAgent"]:
                        type_key = "system"
                    elif agent_type in ["control", "ControlAgent"]:
                        type_key = "control"
                    elif agent_type in ["independent", "IndependentAgent"]:
                        type_key = "independent"
                    else:
                        continue

                    gen_to_type[gen] = type_key

                # Aggregate rewards by agent type and generation
                for gen, reward in rewards_by_generation.items():
                    if gen in gen_to_type:
                        agent_type = gen_to_type[gen]

                        if gen not in all_rewards[agent_type]:
                            all_rewards[agent_type][gen] = 0
                            reward_counts[agent_type][gen] = 0

                        all_rewards[agent_type][gen] += reward
                        reward_counts[agent_type][gen] += 1

                valid_dbs += 1

            except Exception as e:
                logger.error(f"Error processing database {db_path}: {str(e)}")
                failed_dbs += 1

        # Calculate average rewards across all simulations
        for agent_type in result:
            for gen in all_rewards[agent_type]:
                if reward_counts[agent_type][gen] > 0:
                    result[agent_type][gen] = (
                        all_rewards[agent_type][gen] / reward_counts[agent_type][gen]
                    )

        if valid_dbs == 0:
            logger.error(
                f"No valid reward data found in experiment {experiment}. "
                f"All {failed_dbs} databases were corrupted or invalid."
            )
            return result

        logger.info(
            f"Successfully processed reward data from {valid_dbs} databases. "
            f"Skipped {failed_dbs} corrupted/invalid databases."
        )

        return result

    except Exception as e:
        logger.error(
            f"Unexpected error processing rewards by generation for {experiment}: {str(e)}"
        )
        return result


def plot_rewards_by_generation(
    rewards_data: Dict[str, Dict[int, float]], output_dir: str
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


# ---------------------
# Main Execution
# ---------------------


def process_experiment(agent_type: str, experiment: str) -> Dict[str, List]:
    """
    Process experiment data with comprehensive error handling.

    Args:
        agent_type: Type of agent being analyzed
        experiment: Name of the experiment

    Returns:
        Dictionary containing processed population data and metadata
    """
    logger.info(f"Processing experiment: {experiment}")

    try:
        experiment_path = f"results/one_of_a_kind/experiments/data/{experiment}"
        if not os.path.exists(experiment_path):
            logger.error(f"Experiment directory not found: {experiment_path}")
            return {"populations": [], "max_steps": 0}

        db_paths = find_simulation_databases(experiment_path)
        if not db_paths:
            logger.error(f"No database files found in {experiment_path}")
            return {"populations": [], "max_steps": 0}

        all_populations = []
        max_steps = 0
        valid_dbs = 0
        failed_dbs = 0

        for db_path in db_paths:
            try:
                steps, pop, steps_count = get_data(db_path)
                if steps is not None and pop is not None:
                    # Validate population data
                    if not validate_population_data(pop, db_path):
                        failed_dbs += 1
                        continue

                    all_populations.append(pop)
                    max_steps = max(max_steps, steps_count)
                    valid_dbs += 1
                else:
                    failed_dbs += 1
            except Exception as e:
                logger.error(f"Error processing database {db_path}: {str(e)}")
                failed_dbs += 1

        if not all_populations:
            logger.error(
                f"No valid data found in experiment {experiment}. "
                f"All {failed_dbs} databases were corrupted or invalid."
            )
            return {"populations": [], "max_steps": 0}

        logger.info(
            f"Successfully processed {valid_dbs} databases. "
            f"Skipped {failed_dbs} corrupted/invalid databases."
        )

        # Only create plot if we have valid data
        if valid_dbs > 0:
            try:
                output_path = Path(experiment_path) / "population_trends.png"
                plot_population_trends_across_simulations(
                    all_populations,
                    max_steps,
                    str(output_path),
                )
            except Exception as e:
                logger.error(f"Error creating plot for {experiment}: {str(e)}")

        return {"populations": all_populations, "max_steps": max_steps}

    except Exception as e:
        logger.error(f"Unexpected error processing experiment {experiment}: {str(e)}")
        return {"populations": [], "max_steps": 0}


def find_experiments(base_path: str) -> Dict[str, List[str]]:
    """Find all experiment directories and their iterations."""
    base = Path(base_path)
    experiments = {
        "single_agent": {},  # For single_*_agent experiments
        "one_of_a_kind": [],  # For one_of_a_kind experiments
    }

    # Look for directories that match the pattern single_*_agent_*
    for exp_dir in base.glob("single_*_agent_*"):
        if exp_dir.is_dir():
            agent_type = exp_dir.name.split("_")[1]  # Extract 'system', 'control', etc.
            if agent_type not in experiments["single_agent"]:
                experiments["single_agent"][agent_type] = []
            experiments["single_agent"][agent_type].append(exp_dir.name)

    # Look for directories that match the pattern one_of_a_kind_*
    for exp_dir in base.glob("one_of_a_kind_*"):
        if exp_dir.is_dir():
            experiments["one_of_a_kind"].append(exp_dir.name)

    return experiments


def process_experiment_by_agent_type(experiment: str) -> Dict[str, Dict]:
    """
    Process experiment data separated by agent type.

    Args:
        experiment: Name of the experiment

    Returns:
        Dictionary containing processed population data for each agent type
    """
    logger.info(f"Processing experiment by agent type: {experiment}")

    result = {
        "system": {"populations": [], "max_steps": 0},
        "control": {"populations": [], "max_steps": 0},
        "independent": {"populations": [], "max_steps": 0},
    }

    try:
        experiment_path = f"results/one_of_a_kind/experiments/data/{experiment}"
        if not os.path.exists(experiment_path):
            logger.error(f"Experiment directory not found: {experiment_path}")
            return result

        db_paths = find_simulation_databases(experiment_path)
        if not db_paths:
            logger.error(f"No database files found in {experiment_path}")
            return result

        max_steps = 0
        valid_dbs = 0
        failed_dbs = 0

        # Temporary storage for populations from each database
        temp_populations = {"system": [], "control": [], "independent": []}

        for db_path in db_paths:
            try:
                steps, pops, steps_count = get_columns_data_by_agent_type(db_path)
                if steps is not None and pops:
                    # Validate population data for each agent type
                    valid_data = True
                    for agent_type, pop in zip(
                        ["system", "control", "independent"],
                        ["system_agents", "control_agents", "independent_agents"],
                    ):
                        if pop in pops and validate_population_data(pops[pop], db_path):
                            temp_populations[agent_type].append(pops[pop])
                        else:
                            valid_data = False
                            break

                    if valid_data:
                        max_steps = max(max_steps, steps_count)
                        valid_dbs += 1
                    else:
                        failed_dbs += 1
                else:
                    failed_dbs += 1
            except Exception as e:
                logger.error(f"Error processing database {db_path}: {str(e)}")
                failed_dbs += 1

        # Combine results
        for agent_type in result:
            result[agent_type]["populations"] = temp_populations[agent_type]
            result[agent_type]["max_steps"] = max_steps

        if valid_dbs == 0:
            logger.error(
                f"No valid data found in experiment {experiment}. "
                f"All {failed_dbs} databases were corrupted or invalid."
            )
            return result

        logger.info(
            f"Successfully processed {valid_dbs} databases. "
            f"Skipped {failed_dbs} corrupted/invalid databases."
        )

        return result

    except Exception as e:
        logger.error(f"Unexpected error processing experiment {experiment}: {str(e)}")
        return result


def process_experiment_resource_consumption(experiment: str) -> Dict[str, Dict]:
    """
    Process experiment resource consumption data separated by agent type.

    Args:
        experiment: Name of the experiment

    Returns:
        Dictionary containing processed consumption data for each agent type
    """
    logger.info(f"Processing resource consumption for experiment: {experiment}")

    result = {
        "system": {"consumption": [], "max_steps": 0},
        "control": {"consumption": [], "max_steps": 0},
        "independent": {"consumption": [], "max_steps": 0},
    }

    try:
        experiment_path = f"results/one_of_a_kind/experiments/data/{experiment}"
        if not os.path.exists(experiment_path):
            logger.error(f"Experiment directory not found: {experiment_path}")
            return result

        db_paths = find_simulation_databases(experiment_path)
        if not db_paths:
            logger.error(f"No database files found in {experiment_path}")
            return result

        max_steps = 0
        valid_dbs = 0
        failed_dbs = 0

        # Temporary storage for consumption from each database
        temp_consumption = {"system": [], "control": [], "independent": []}

        for db_path in db_paths:
            try:
                steps, consumption, steps_count = get_resource_consumption_data(db_path)
                if steps is not None and consumption:
                    # Validate consumption data for each agent type
                    valid_data = True
                    for agent_type in ["system", "control", "independent"]:
                        if (
                            agent_type in consumption
                            and len(consumption[agent_type]) > 0
                        ):
                            temp_consumption[agent_type].append(consumption[agent_type])
                        else:
                            logger.warning(
                                f"Missing consumption data for {agent_type} in {db_path}"
                            )
                            valid_data = False
                            break

                    if valid_data:
                        max_steps = max(max_steps, steps_count)
                        valid_dbs += 1
                    else:
                        failed_dbs += 1
                else:
                    failed_dbs += 1
            except Exception as e:
                logger.error(f"Error processing database {db_path}: {str(e)}")
                failed_dbs += 1

        # Combine results
        for agent_type in result:
            result[agent_type]["consumption"] = temp_consumption[agent_type]
            result[agent_type]["max_steps"] = max_steps

        if valid_dbs == 0:
            logger.error(
                f"No valid consumption data found in experiment {experiment}. "
                f"All {failed_dbs} databases were corrupted or invalid."
            )
            return result

        logger.info(
            f"Successfully processed consumption data from {valid_dbs} databases. "
            f"Skipped {failed_dbs} corrupted/invalid databases."
        )

        return result

    except Exception as e:
        logger.error(
            f"Unexpected error processing consumption data for {experiment}: {str(e)}"
        )
        return result


def process_action_distributions(
    experiment: str,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Process action distribution data for an experiment.

    Args:
        experiment: Name of the experiment

    Returns:
        Dictionary containing processed action distribution data for each agent type
    """
    logger.info(f"Processing action distributions for experiment: {experiment}")

    result = {
        "system": {"actions": {}, "total_actions": 0},
        "control": {"actions": {}, "total_actions": 0},
        "independent": {"actions": {}, "total_actions": 0},
    }

    try:
        experiment_path = f"results/one_of_a_kind/experiments/data/{experiment}"
        if not os.path.exists(experiment_path):
            logger.error(f"Experiment directory not found: {experiment_path}")
            return result

        db_paths = find_simulation_databases(experiment_path)
        if not db_paths:
            logger.error(f"No database files found in {experiment_path}")
            return result

        valid_dbs = 0
        failed_dbs = 0

        # Process each database
        for db_path in db_paths:
            try:
                action_data = get_action_distribution_data(db_path)
                if action_data:
                    # Aggregate action counts across databases
                    for agent_type, actions in action_data.items():
                        # Map database agent_type to our standard types
                        if agent_type == "system" or agent_type == "SystemAgent":
                            type_key = "system"
                        elif agent_type == "control" or agent_type == "ControlAgent":
                            type_key = "control"
                        elif (
                            agent_type == "independent"
                            or agent_type == "IndependentAgent"
                        ):
                            type_key = "independent"
                        else:
                            logger.warning(f"Unknown agent type: {agent_type}")
                            continue

                        # Add action counts to our aggregated results
                        for action, count in actions.items():
                            if action not in result[type_key]["actions"]:
                                result[type_key]["actions"][action] = 0
                            result[type_key]["actions"][action] += count
                            result[type_key]["total_actions"] += count

                    valid_dbs += 1
                else:
                    failed_dbs += 1
            except Exception as e:
                logger.error(f"Error processing database {db_path}: {str(e)}")
                failed_dbs += 1

        # Calculate percentages
        for agent_type in result:
            if result[agent_type]["total_actions"] > 0:
                for action in result[agent_type]["actions"]:
                    result[agent_type]["actions"][action] = (
                        result[agent_type]["actions"][action]
                        / result[agent_type]["total_actions"]
                    )

        logger.info(
            f"Successfully processed action distributions from {valid_dbs} databases. "
            f"Skipped {failed_dbs} corrupted/invalid databases."
        )

        return result

    except Exception as e:
        logger.error(
            f"Unexpected error processing action distributions for {experiment}: {str(e)}"
        )
        return result


def plot_action_distributions(
    action_data: Dict[str, Dict[str, Dict[str, float]]], output_dir: str
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


def process_experiment_resource_levels(experiment: str) -> Dict[str, List]:
    """
    Process resource level data for an experiment.

    Args:
        experiment: Name of the experiment

    Returns:
        Dictionary containing processed resource level data
    """
    logger.info(f"Processing resource level data for experiment: {experiment}")

    result = {"resource_levels": [], "max_steps": 0}

    try:
        experiment_path = f"results/one_of_a_kind/experiments/data/{experiment}"
        if not os.path.exists(experiment_path):
            logger.error(f"Experiment directory not found: {experiment_path}")
            return result

        db_paths = find_simulation_databases(experiment_path)
        if not db_paths:
            logger.error(f"No database files found in {experiment_path}")
            return result

        max_steps = 0
        valid_dbs = 0
        failed_dbs = 0

        for db_path in db_paths:
            try:
                steps, resource_levels, steps_count = get_resource_level_data(db_path)
                if steps is not None and resource_levels is not None:
                    # Use the specialized validation function for resource levels
                    if validate_resource_level_data(resource_levels, db_path):
                        result["resource_levels"].append(resource_levels)
                        max_steps = max(max_steps, steps_count)
                        valid_dbs += 1
                    else:
                        failed_dbs += 1
                else:
                    failed_dbs += 1
            except Exception as e:
                logger.error(f"Error processing database {db_path}: {str(e)}")
                failed_dbs += 1

        result["max_steps"] = max_steps

        if valid_dbs == 0:
            logger.error(
                f"No valid resource level data found in experiment {experiment}. "
                f"All {failed_dbs} databases were corrupted or invalid."
            )
            return result

        logger.info(
            f"Successfully processed resource level data from {valid_dbs} databases. "
            f"Skipped {failed_dbs} corrupted/invalid databases."
        )

        return result

    except Exception as e:
        logger.error(
            f"Unexpected error processing resource level data for {experiment}: {str(e)}"
        )
        return result


def plot_resource_level_trends(resource_level_data: Dict[str, List], output_path: str):
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


def main():
    base_path = Path("results/one_of_a_kind/experiments/data")
    try:
        # Ensure base directory exists
        base_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating base directory {base_path}: {str(e)}")
        return

    experiments = find_experiments(str(base_path))
    logger.info(f"Found experiments: {experiments}")

    # Process single agent experiments
    all_experiment_data = {}
    all_consumption_data = {}
    all_resource_level_data = {"resource_levels": [], "max_steps": 0}

    for agent_type, experiment_list in experiments["single_agent"].items():
        all_experiment_data[agent_type] = {"populations": [], "max_steps": 0}
        all_consumption_data[agent_type] = {"consumption": [], "max_steps": 0}

        # Process each iteration of this experiment type
        for experiment in experiment_list:
            # Process population data
            data = process_experiment(agent_type, experiment)
            all_experiment_data[agent_type]["populations"].extend(data["populations"])
            all_experiment_data[agent_type]["max_steps"] = max(
                all_experiment_data[agent_type]["max_steps"], data["max_steps"]
            )

            # Process consumption data
            consumption_data = process_experiment_resource_consumption(experiment)
            if agent_type in consumption_data:
                all_consumption_data[agent_type]["consumption"].extend(
                    consumption_data[agent_type]["consumption"]
                )
                all_consumption_data[agent_type]["max_steps"] = max(
                    all_consumption_data[agent_type]["max_steps"],
                    consumption_data[agent_type]["max_steps"],
                )

            # Process resource level data
            resource_level_data = process_experiment_resource_levels(experiment)
            all_resource_level_data["resource_levels"].extend(
                resource_level_data["resource_levels"]
            )
            all_resource_level_data["max_steps"] = max(
                all_resource_level_data["max_steps"], resource_level_data["max_steps"]
            )

            # Create individual experiment resource consumption chart
            experiment_path = f"results/one_of_a_kind/experiments/data/{experiment}"
            if any(
                consumption_data[agent_type]["consumption"]
                for agent_type in consumption_data
            ):
                plot_resource_consumption_trends(consumption_data, experiment_path)

            # Create individual experiment resource level chart
            if resource_level_data["resource_levels"]:
                output_path = Path(experiment_path) / "resource_level_trends.png"
                plot_resource_level_trends(resource_level_data, str(output_path))

    # Create combined plots for single agent experiments
    plot_population_trends_by_agent_type(all_experiment_data, str(base_path))
    plot_resource_consumption_trends(all_consumption_data, str(base_path))

    # Create combined resource level plot
    if all_resource_level_data["resource_levels"]:
        output_path = base_path / "resource_level_trends.png"
        plot_resource_level_trends(all_resource_level_data, str(output_path))

    # Process one_of_a_kind experiments
    for experiment in experiments["one_of_a_kind"]:
        experiment_path = base_path / experiment

        # Create overall population trend plot
        data = process_experiment("one_of_a_kind", experiment)
        if data["populations"]:
            output_path = experiment_path / "population_trends.png"
            plot_population_trends_across_simulations(
                data["populations"], data["max_steps"], str(output_path)
            )

        # Create agent type comparison plots
        data_by_type = process_experiment_by_agent_type(experiment)
        if any(data_by_type[agent_type]["populations"] for agent_type in data_by_type):
            plot_population_trends_by_agent_type(data_by_type, str(experiment_path))

            # Add final agent counts analysis
            final_counts = analyze_final_agent_counts(data_by_type)
            plot_final_agent_counts(
                final_counts, str(experiment_path / "final_agent_analysis")
            )

        # Create resource consumption comparison plot
        consumption_by_type = process_experiment_resource_consumption(experiment)
        if any(
            consumption_by_type[agent_type]["consumption"]
            for agent_type in consumption_by_type
        ):
            plot_resource_consumption_trends(consumption_by_type, str(experiment_path))

        # Create resource level trend plot
        resource_level_data = process_experiment_resource_levels(experiment)
        if resource_level_data["resource_levels"]:
            output_path = experiment_path / "resource_level_trends.png"
            plot_resource_level_trends(resource_level_data, str(output_path))

        # Add action distribution analysis
        action_data = process_action_distributions(experiment)
        if any(data["total_actions"] > 0 for data in action_data.values()):
            action_analysis_dir = experiment_path / "action_analysis"
            plot_action_distributions(action_data, str(action_analysis_dir))

        # Analyze early terminations
        db_paths = find_simulation_databases(str(experiment_path))
        if db_paths:
            early_terminations = detect_early_terminations(db_paths)
            if early_terminations:
                early_term_dir = experiment_path / "early_termination_analysis"
                plot_early_termination_analysis(early_terminations, str(early_term_dir))

        # Process rewards by generation
        rewards_data = process_experiment_rewards_by_generation(experiment)
        if rewards_data:
            plot_rewards_by_generation(
                rewards_data, str(experiment_path / "rewards_by_generation")
            )

        # Add rewards by generation analysis
        rewards_by_generation = process_experiment_rewards_by_generation(experiment)
        if any(
            rewards_by_generation[agent_type] for agent_type in rewards_by_generation
        ):
            rewards_analysis_dir = experiment_path / "rewards_analysis"
            plot_rewards_by_generation(rewards_by_generation, str(rewards_analysis_dir))


if __name__ == "__main__":
    main()

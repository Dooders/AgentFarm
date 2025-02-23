import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from farm.database.database import SimulationDatabase
from farm.database.models import SimulationStepModel

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
    all_populations: List[np.ndarray], max_steps: int
) -> pd.DataFrame:
    """
    Create a DataFrame from population data with proper padding for missing steps.
    Includes data validation.
    """
    if not all_populations:
        logger.error("No population data provided")
        return pd.DataFrame(columns=["simulation_id", "step", "population"])

    # Validate each population array
    valid_populations = []
    for i, pop in enumerate(all_populations):
        if validate_population_data(pop):
            valid_populations.append(pop)
        else:
            logger.warning(f"Skipping invalid population data from simulation {i}")

    if not valid_populations:
        logger.error("No valid population data after validation")
        return pd.DataFrame(columns=["simulation_id", "step", "population"])

    # Create DataFrame with valid data
    data = []
    for sim_idx, pop in enumerate(valid_populations):
        for step in range(max_steps):
            population = pop[step] if step < len(pop) else np.nan
            data.append((f"sim_{sim_idx}", step, population))

    df = pd.DataFrame(data, columns=["simulation_id", "step", "population"])

    # Final validation of the DataFrame
    if df.empty:
        logger.warning("Created DataFrame is empty")
    elif df["population"].isna().all():
        logger.warning("All population values are NaN in the DataFrame")

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

        # Validate each population array
        for col, pop in populations.items():
            if not validate_population_data(pop, experiment_db_path):
                logger.error(f"Invalid data for column '{col}' in {experiment_db_path}")
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


# ---------------------
# Plotting Helpers
# ---------------------


def plot_mean_and_ci(ax, steps, mean, ci, color, label):
    """Plot mean line with confidence interval."""
    ax.plot(steps, mean, color=color, label=label, linewidth=2)
    ax.fill_between(steps, mean - ci, mean + ci, color=color, alpha=0.2)


def plot_median_line(ax, steps, median, color="g", style="--"):
    """Plot median line."""
    ax.plot(steps, median, f"{color}{style}", label="Median Population", linewidth=2)


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
    experiments = {}

    # Look for directories that match the pattern single_*_agent_*
    for exp_dir in base.glob("single_*_agent_*"):
        if exp_dir.is_dir():
            agent_type = exp_dir.name.split("_")[1]  # Extract 'system', 'control', etc.
            if agent_type not in experiments:
                experiments[agent_type] = []
            experiments[agent_type].append(exp_dir.name)

    return experiments


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

    all_experiment_data = {}
    for agent_type, experiment_list in experiments.items():
        all_experiment_data[agent_type] = {"populations": [], "max_steps": 0}

        # Process each iteration of this experiment type
        for experiment in experiment_list:
            data = process_experiment(agent_type, experiment)
            all_experiment_data[agent_type]["populations"].extend(data["populations"])
            all_experiment_data[agent_type]["max_steps"] = max(
                all_experiment_data[agent_type]["max_steps"], data["max_steps"]
            )

    # Create combined plot
    plot_population_trends_by_agent_type(all_experiment_data, str(base_path))


if __name__ == "__main__":
    main()

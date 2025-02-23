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
    base = Path(base_path)
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


def create_population_df(
    all_populations: List[np.ndarray], max_steps: int
) -> pd.DataFrame:
    """
    Create a DataFrame from population data with proper padding for missing steps.

    Args:
        all_populations: List of population arrays from different simulations
        max_steps: Maximum number of steps across all simulations

    Returns:
        DataFrame with columns: simulation_id, step, population
    """
    data = []
    for sim_idx, pop in enumerate(all_populations):
        for step in range(max_steps):
            population = pop[step] if step < len(pop) else np.nan
            data.append((f"sim_{sim_idx}", step, population))
    return pd.DataFrame(data, columns=["simulation_id", "step", "population"])


def calculate_statistics(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate population statistics using Pandas operations.

    Args:
        df: DataFrame with columns: simulation_id, step, population

    Returns:
        Tuple of (mean, median, std_dev, confidence_interval)
    """
    grouped = df.groupby("step")["population"]
    mean_pop = grouped.mean().values
    median_pop = grouped.median().values
    std_pop = grouped.std().values
    n_simulations = df["simulation_id"].nunique()
    confidence_interval = 1.96 * std_pop / np.sqrt(n_simulations)
    return mean_pop, median_pop, std_pop, confidence_interval


# ---------------------
# Database Interaction
# ---------------------


def get_columns_data(
    experiment_db_path: str, columns: List[str]
) -> Tuple[np.ndarray, Dict[str, np.ndarray], int]:
    """
    Retrieve specified columns from the simulation database.

    Args:
        experiment_db_path: Path to the database file
        columns: List of column names to retrieve

    Returns:
        Tuple containing:
        - steps: Array of step numbers
        - populations: Dictionary mapping column names to their data arrays
        - max_steps: Number of steps in the simulation
    """
    db = SimulationDatabase(experiment_db_path)
    session = db.Session()
    try:
        # Build query dynamically based on requested columns
        query_columns = [getattr(SimulationStepModel, col) for col in columns]
        query = (
            session.query(SimulationStepModel.step_number, *query_columns)
            .order_by(SimulationStepModel.step_number)
            .all()
        )

        # Extract step numbers and column data
        steps = np.array([row[0] for row in query])
        populations = {
            col: np.array([row[i + 1] for row in query])
            for i, col in enumerate(columns)
        }
        max_steps = len(steps)

        return steps, populations, max_steps
    except Exception as e:
        logger.error(f"Error retrieving data from {experiment_db_path}: {e}")
        return None, {}, 0
    finally:
        session.close()
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
# Plotting Functions
# ---------------------


def plot_population_trends_across_simulations(
    all_populations: List[np.ndarray], max_steps: int, output_path: str
):
    fig, ax = plt.subplots(figsize=(15, 8))
    experiment_name = Path(output_path).parent.name
    fig.suptitle(
        f"Population Trends Across All Simulations (N={len(all_populations)})",
        fontsize=14,
        y=0.95,
    )
    ax.set_title(experiment_name, fontsize=12, pad=10)

    # Create DataFrame and calculate statistics
    df = create_population_df(all_populations, max_steps)
    mean_pop, median_pop, std_pop, confidence_interval = calculate_statistics(df)
    steps = np.arange(max_steps)

    # Calculate key metrics
    overall_median = np.nanmedian(median_pop)
    final_median = median_pop[-1]
    peak_step = np.nanargmax(mean_pop)
    peak_value = mean_pop[peak_step]

    # Plot lines
    ax.plot(steps, mean_pop, "b-", label="Mean Population", linewidth=2)
    ax.plot(steps, median_pop, "g--", label="Median Population", linewidth=2)
    ax.axhline(
        y=overall_median,
        color="orange",
        linestyle=":",
        alpha=0.8,
        label=f"Overall Median: {overall_median:.1f}",
        linewidth=2,
    )
    ax.plot(
        peak_step, peak_value, "rx", markersize=10, label=f"Peak at step {peak_step}"
    )
    ax.plot(
        max_steps - 1,
        final_median,
        "rx",
        markersize=10,
        label=f"Final Median: {final_median:.1f}",
    )

    # Confidence interval
    ax.fill_between(
        steps,
        mean_pop - confidence_interval,
        mean_pop + confidence_interval,
        color="b",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    ax.set_xlabel("Simulation Step", fontsize=12)
    ax.set_ylabel("Number of Agents", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_population_trends_by_agent_type(
    experiment_data: Dict[str, Dict], output_dir: str
):
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

        ax.plot(
            steps,
            mean_pop,
            color=colors[agent_type],
            label=f"{display_name} Agent (n={len(data['populations'])})",
            linewidth=2,
        )
        ax.fill_between(
            steps,
            mean_pop - confidence_interval,
            mean_pop + confidence_interval,
            color=colors[agent_type],
            alpha=0.2,
        )

    ax.set_xlabel("Simulation Step", fontsize=12)
    ax.set_ylabel("Number of Agents", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    output_path = Path(output_dir) / "population_trends_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------
# Main Execution
# ---------------------


def process_experiment(agent_type: str, experiment: str) -> Dict[str, List]:
    logger.info(f"Processing experiment: {experiment}")
    db_paths = find_simulation_databases(
        f"results/one_of_a_kind/experiments/data/{experiment}"
    )

    all_populations = []
    max_steps = 0
    valid_dbs = 0
    failed_dbs = 0

    for db_path in db_paths:
        steps, pop, steps_count = get_data(db_path)
        if steps is not None and pop is not None:
            all_populations.append(pop)
            max_steps = max(max_steps, steps_count)
            valid_dbs += 1
        else:
            failed_dbs += 1

    if not all_populations:
        logger.error(
            f"No valid data found in experiment {experiment}. "
            f"All {failed_dbs} databases were corrupted."
        )
        return {"populations": [], "max_steps": 0}

    logger.info(
        f"Successfully processed {valid_dbs} databases. "
        f"Skipped {failed_dbs} corrupted databases."
    )

    # Only create plot if we have valid data
    if valid_dbs > 0:
        plot_population_trends_across_simulations(
            all_populations,
            max_steps,
            f"results/one_of_a_kind/experiments/data/{experiment}/population_trends.png",
        )

    return {"populations": all_populations, "max_steps": max_steps}


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
    base_path = "results/one_of_a_kind/experiments/data"
    experiments = find_experiments(base_path)

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
    plot_population_trends_by_agent_type(all_experiment_data, base_path)


if __name__ == "__main__":
    main()

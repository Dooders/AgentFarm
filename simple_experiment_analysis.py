import os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import logging
from matplotlib import pyplot as plt
from farm.database.database import SimulationDatabase
from farm.database.models import SimulationStepModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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

def pad_population(pop: np.ndarray, target_length: int) -> np.ndarray:
    """
    Convert population data to float and pad it with NaN values to match target_length.
    """
    pop = pop.astype(float)
    return np.pad(pop, (0, target_length - len(pop)), mode="constant", constant_values=np.nan)

def compute_statistics(population_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean_pop = np.nanmean(population_array, axis=0)
    median_pop = np.nanmedian(population_array, axis=0)
    std_pop = np.nanstd(population_array, axis=0)
    confidence_interval = 1.96 * std_pop / np.sqrt(population_array.shape[0])
    return mean_pop, median_pop, std_pop, confidence_interval

# ---------------------
# Database Interaction
# ---------------------

def get_data(experiment_db_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
    db = SimulationDatabase(experiment_db_path)
    session = db.Session()
    try:
        population_data = (session.query(SimulationStepModel.step_number, SimulationStepModel.total_agents)
                                    .order_by(SimulationStepModel.step_number)
                                    .all())
        steps = np.array([p[0] for p in population_data])
        pop = np.array([p[1] for p in population_data])
        max_steps = len(steps)
        return steps, pop, max_steps
    finally:
        session.close()
        db.close()

# ---------------------
# Plotting Functions
# ---------------------

def plot_population_trends_across_simulations(all_populations: List[np.ndarray], max_steps: int, output_path: str):
    fig, ax = plt.subplots(figsize=(15, 8))
    experiment_name = Path(output_path).parent.name
    fig.suptitle(f"Population Trends Across All Simulations (N={len(all_populations)})", fontsize=14, y=0.95)
    ax.set_title(experiment_name, fontsize=12, pad=10)

    # Pad populations and calculate statistics
    padded_populations = [pad_population(pop, max_steps) for pop in all_populations]
    population_array = np.array(padded_populations)
    steps = np.arange(max_steps)

    mean_pop, median_pop, _, confidence_interval = compute_statistics(population_array)
    overall_median = np.nanmedian(median_pop)
    final_median = median_pop[-1]
    peak_step = np.nanargmax(mean_pop)
    peak_value = mean_pop[peak_step]

    # Plot lines
    ax.plot(steps, mean_pop, "b-", label="Mean Population", linewidth=2)
    ax.plot(steps, median_pop, "g--", label="Median Population", linewidth=2)
    ax.axhline(y=overall_median, color='orange', linestyle=':', alpha=0.8,
               label=f'Overall Median: {overall_median:.1f}', linewidth=2)
    ax.plot(peak_step, peak_value, 'rx', markersize=10, label=f'Peak at step {peak_step}')
    ax.plot(max_steps - 1, final_median, 'rx', markersize=10,
            label=f'Final Median: {final_median:.1f}')

    # Confidence interval
    ax.fill_between(steps, mean_pop - confidence_interval, mean_pop + confidence_interval,
                    color="b", alpha=0.2, label="95% Confidence Interval")

    ax.set_xlabel("Simulation Step", fontsize=12)
    ax.set_ylabel("Number of Agents", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_population_trends_by_agent_type(experiment_data: Dict[str, Dict], output_dir: str):
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.suptitle("Population Trends Comparison by Agent Type", fontsize=14, y=0.95)

    colors = {'system': 'blue', 'control': 'green', 'independent': 'red'}

    for agent_type, data in experiment_data.items():
        populations = [pad_population(pop, data['max_steps']) for pop in data['populations']]
        population_array = np.array(populations)
        steps = np.arange(data['max_steps'])

        mean_pop, _, _, confidence_interval = compute_statistics(population_array)
        display_name = agent_type.replace('_', ' ').title()

        ax.plot(steps, mean_pop, color=colors[agent_type], label=f'{display_name} Agent (n={len(populations)})', linewidth=2)
        ax.fill_between(steps, mean_pop - confidence_interval, mean_pop + confidence_interval, color=colors[agent_type], alpha=0.2)

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
    db_paths = find_simulation_databases(f"results/one_of_a_kind/experiments/data/{experiment}")

    all_populations = []
    max_steps = 0

    for db_path in db_paths:
        _, pop, steps = get_data(db_path)
        all_populations.append(pop)
        max_steps = max(max_steps, steps)

    plot_population_trends_across_simulations(
        all_populations,
        max_steps,
        f"results/one_of_a_kind/experiments/data/{experiment}/population_trends.png"
    )

    return {'populations': all_populations, 'max_steps': max_steps}

def main():
    experiments = {
        "system": "single_system_agent_20250222_210756",
        "control": "single_control_agent_20250222_203526",
        "independent": "single_independent_agent_20250222_214838"
    }

    experiment_data = {agent_type: process_experiment(agent_type, exp) for agent_type, exp in experiments.items()}

    # Create combined plot
    plot_population_trends_by_agent_type(experiment_data, "results/one_of_a_kind/experiments/data")

if __name__ == "__main__":
    main()

import argparse
import logging
import os
from typing import Dict, List

from farm.research.analysis.database import find_simulation_databases
from farm.research.analysis.analysis import (
    process_experiment,
    process_experiment_by_agent_type,
    process_experiment_resource_consumption,
    process_action_distributions,
    process_experiment_resource_levels,
    process_experiment_rewards_by_generation,
    analyze_final_agent_counts,
    detect_early_terminations,
    find_experiments,
)
from farm.research.analysis.plotting import (
    plot_population_trends_across_simulations,
    plot_population_trends_by_agent_type,
    plot_resource_consumption_trends,
    plot_action_distributions,
    plot_resource_level_trends,
    plot_early_termination_analysis,
    plot_final_agent_counts,
    plot_rewards_by_generation,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_experiment(experiment: str, output_base_dir: str):
    """
    Analyze a single experiment.

    Args:
        experiment: Experiment path
        output_base_dir: Base directory for output
    """
    logger.info(f"Analyzing experiment: {experiment}")

    # Create output directory
    output_dir = os.path.join(output_base_dir, experiment)
    os.makedirs(output_dir, exist_ok=True)

    # Process data by agent type
    experiment_data = process_experiment_by_agent_type(experiment)
    if experiment_data:
        # Plot population trends by agent type
        plot_population_trends_by_agent_type(experiment_data, output_dir)

        # Analyze final agent counts
        final_counts = analyze_final_agent_counts(experiment_data)
        if final_counts:
            plot_final_agent_counts(final_counts, output_dir)

    # Process resource consumption data
    consumption_data = process_experiment_resource_consumption(experiment)
    if consumption_data:
        plot_resource_consumption_trends(consumption_data, output_dir)

    # Process action distribution data
    action_data = process_action_distributions(experiment)
    if action_data:
        plot_action_distributions(action_data, output_dir)

    # Process resource level data
    resource_level_data = process_experiment_resource_levels(experiment)
    if resource_level_data:
        resource_level_output_path = os.path.join(
            output_dir, "resource_level_trends.png"
        )
        plot_resource_level_trends(resource_level_data, resource_level_output_path)

    # Process rewards by generation
    rewards_data = process_experiment_rewards_by_generation(experiment)
    if rewards_data:
        plot_rewards_by_generation(rewards_data, output_dir)

    # Check for early terminations
    db_paths = find_simulation_databases(
        f"results/one_of_a_kind/experiments/data/{experiment}"
    )
    if db_paths:
        early_terminations = detect_early_terminations(db_paths)
        if early_terminations:
            plot_early_termination_analysis(early_terminations, output_dir)

    logger.info(f"Analysis complete for experiment: {experiment}")


def analyze_agent_type(agent_type: str, experiments: List[str], output_base_dir: str):
    """
    Analyze experiments for a specific agent type.

    Args:
        agent_type: Agent type
        experiments: List of experiment paths
        output_base_dir: Base directory for output
    """
    logger.info(
        f"Analyzing agent type: {agent_type} across {len(experiments)} experiments"
    )

    # Create output directory
    output_dir = os.path.join(output_base_dir, agent_type)
    os.makedirs(output_dir, exist_ok=True)

    # Process data for each experiment
    all_experiment_data = {agent_type: {"populations": [], "max_steps": 0}}
    all_consumption_data = {agent_type: {"consumption": [], "max_steps": 0}}
    all_resource_level_data = {"resource_levels": [], "max_steps": 0}

    for experiment in experiments:
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

    # Plot population trends
    if all_experiment_data[agent_type]["populations"]:
        population_output_path = os.path.join(output_dir, "population_trends.png")
        plot_population_trends_across_simulations(
            all_experiment_data[agent_type]["populations"],
            all_experiment_data[agent_type]["max_steps"],
            population_output_path,
        )

    # Plot resource consumption trends
    if all_consumption_data[agent_type]["consumption"]:
        plot_resource_consumption_trends(all_consumption_data, output_dir)

    # Plot resource level trends
    if all_resource_level_data["resource_levels"]:
        resource_level_output_path = os.path.join(
            output_dir, "resource_level_trends.png"
        )
        plot_resource_level_trends(all_resource_level_data, resource_level_output_path)

    logger.info(f"Analysis complete for agent type: {agent_type}")


def analyze_all_experiments(base_path: str, output_base_dir: str):
    """
    Analyze all experiments.

    Args:
        base_path: Base path for experiments
        output_base_dir: Base directory for output
    """
    logger.info(f"Analyzing all experiments in {base_path}")

    # Find all experiments
    experiments_by_agent_type = find_experiments(base_path)

    if not experiments_by_agent_type:
        logger.warning(f"No experiments found in {base_path}")
        return

    # Analyze each agent type
    for agent_type, experiments in experiments_by_agent_type.items():
        analyze_agent_type(agent_type, experiments, output_base_dir)

    # Analyze each experiment
    all_experiments = set()
    for experiments in experiments_by_agent_type.values():
        all_experiments.update(experiments)

    for experiment in all_experiments:
        analyze_experiment(experiment, output_base_dir)

    logger.info("Analysis complete for all experiments")


def main():
    """
    Main entry point for the analysis script.
    """
    parser = argparse.ArgumentParser(description="Analyze simulation results")
    parser.add_argument(
        "--base-path",
        type=str,
        default="results/one_of_a_kind/experiments/data",
        help="Base path for experiments",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/one_of_a_kind/experiments/analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Analyze a specific experiment",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        help="Analyze a specific agent type",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.experiment and args.agent_type:
        # Analyze a specific agent type in a specific experiment
        logger.info(
            f"Analyzing agent type {args.agent_type} in experiment {args.experiment}"
        )
        data = process_experiment(args.agent_type, args.experiment)
        if data["populations"]:
            output_dir = os.path.join(args.output_dir, args.experiment, args.agent_type)
            os.makedirs(output_dir, exist_ok=True)

            population_output_path = os.path.join(output_dir, "population_trends.png")
            plot_population_trends_across_simulations(
                data["populations"], data["max_steps"], population_output_path
            )
    elif args.experiment:
        # Analyze a specific experiment
        analyze_experiment(args.experiment, args.output_dir)
    elif args.agent_type:
        # Analyze a specific agent type across all experiments
        experiments_by_agent_type = find_experiments(args.base_path)
        if args.agent_type in experiments_by_agent_type:
            analyze_agent_type(
                args.agent_type,
                experiments_by_agent_type[args.agent_type],
                args.output_dir,
            )
        else:
            logger.warning(f"No experiments found for agent type {args.agent_type}")
    else:
        # Analyze all experiments
        analyze_all_experiments(args.base_path, args.output_dir)


if __name__ == "__main__":
    main()

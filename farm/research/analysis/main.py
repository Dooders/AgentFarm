import logging
from pathlib import Path

from farm.research.analysis.analysis import (
    analyze_final_agent_counts,
    detect_early_terminations,
    find_experiments,
    process_action_distributions,
    process_experiment,
    process_experiment_by_agent_type,
    process_experiment_resource_consumption,
    process_experiment_resource_levels,
    process_experiment_rewards_by_generation,
)
from farm.research.analysis.database import find_simulation_databases

# import results.one_of_a_kind.scripts.generational_fitness_analysis as generational_fitness_analysis  # Module not found
from farm.research.analysis.plotting import (
    plot_action_distributions,
    plot_early_termination_analysis,
    plot_final_agent_counts,
    plot_population_trends_across_simulations,
    plot_population_trends_by_agent_type,
    plot_resource_consumption_trends,
    plot_resource_level_trends,
    plot_rewards_by_generation,
)

# Constants for file paths
EXPERIMENT_DATA_PATH = "results/one_of_a_kind_v1/experiments/data"
EXPERIMENT_ANALYSIS_PATH = "results/one_of_a_kind_v1/experiments/analysis"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    base_path = Path(EXPERIMENT_DATA_PATH)
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

    # Type assertion to help the type checker
    single_agent_experiments = experiments["single_agent"]
    if isinstance(single_agent_experiments, dict):
        for agent_type, experiment_list in single_agent_experiments.items():
            all_experiment_data[agent_type] = {"populations": [], "max_steps": 0}
            all_consumption_data[agent_type] = {"consumption": [], "max_steps": 0}

            # Process each iteration of this experiment type
            for experiment in experiment_list:
                # Process population data
                data = process_experiment(agent_type, experiment)
                all_experiment_data[agent_type]["populations"].extend(
                    data["populations"]
                )
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
                    all_resource_level_data["max_steps"],
                    resource_level_data["max_steps"],
                )

                # Create individual experiment resource consumption chart
                experiment_path = (
                    f"results/one_of_a_kind_v1/experiments/data/{experiment}"
                )
                if any(
                    consumption_data[agent_type]["consumption"]
                    for agent_type in consumption_data
                ):
                    plot_resource_consumption_trends(consumption_data, experiment_path)

                # Create individual experiment resource level chart
                if resource_level_data["resource_levels"]:
                    output_path = Path(experiment_path) / "resource_level_trends.png"
                    plot_resource_level_trends(
                        {"resource_levels": resource_level_data["resource_levels"]}, str(output_path)  # type: ignore
                    )
    else:
        logger.warning("single_agent_experiments is not a dict as expected")

    # Create combined plots for single agent experiments
    plot_population_trends_by_agent_type(all_experiment_data, str(base_path))
    plot_resource_consumption_trends(all_consumption_data, str(base_path))

    # Create combined resource level plot
    if all_resource_level_data["resource_levels"]:
        output_path = base_path / "resource_level_trends.png"
        plot_resource_level_trends(
            {"resource_levels": all_resource_level_data["resource_levels"]},
            str(output_path),
        )

    # Process one_of_a_kind experiments
    for experiment in experiments["one_of_a_kind"]:
        experiment_path = base_path / experiment

        # Create overall population trend plot
        data = process_experiment("one_of_a_kind", experiment)
        if isinstance(data["populations"], list) and isinstance(data["max_steps"], int):
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
            plot_resource_level_trends(
                {"resource_levels": resource_level_data["resource_levels"]},  # type: ignore
                str(output_path),
            )

        # Add action distribution analysis
        action_data = process_action_distributions(experiment)
        if any(
            isinstance((total_actions := data.get("total_actions", 0)), int)
            and total_actions > 0
            for data in action_data.values()
        ):
            action_analysis_dir = experiment_path / "action_analysis"
            plot_action_distributions(action_data, str(action_analysis_dir))  # type: ignore

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

        # Add generational fitness analysis
        # TODO: Re-enable when generational_fitness_analysis module is available
        # logger.info(
        #     f"Running generational fitness analysis for experiment: {experiment}"
        # )
        # try:
        #     generational_fitness_results = (
        #         generational_fitness_analysis.process_experiment_generational_fitness(
        #             experiment,
        #             data_path=EXPERIMENT_DATA_PATH,
        #             analysis_path=EXPERIMENT_ANALYSIS_PATH,
        #         )
        #     )
        #     logger.info(f"Completed generational fitness analysis for {experiment}")
        # except Exception as e:
        #     logger.error(
        #         f"Error in generational fitness analysis for {experiment}: {str(e)}"
        #     )


if __name__ == "__main__":
    main()

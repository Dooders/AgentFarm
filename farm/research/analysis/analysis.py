import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from farm.database.database import SimulationDatabase
from farm.database.models import AgentModel
from farm.research.analysis.database import (
    find_simulation_databases,
    get_action_distribution_data,
    get_columns_data_by_agent_type,
    get_data,
    get_resource_consumption_data,
    get_resource_level_data,
    get_rewards_by_generation,
)
from farm.research.analysis.plotting import plot_population_trends_across_simulations
from farm.research.analysis.util import (
    validate_population_data,
    validate_resource_level_data,
)

# EXPERIMENT_DATA_PATH = "results/one_of_a_kind_v1/experiments/data"

logger = logging.getLogger(__name__)


def detect_early_terminations(
    db_paths: List[str], expected_steps: Optional[int] = None
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
            data = get_data(db_path)
            if data is not None:
                steps, _, steps_count = data
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

    # At this point expected_steps is guaranteed to be not None
    assert expected_steps is not None

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
                result = get_columns_data_by_agent_type(db_path)
                if result is None:
                    logger.warning(f"Could not retrieve population data from {db_path}")
                    continue

                steps, populations, _ = result

                # Get resource consumption data
                consumption_result = get_resource_consumption_data(db_path)
                if consumption_result is None:
                    logger.warning(
                        f"Could not retrieve consumption data from {db_path}"
                    )
                    continue

                _, consumption, _ = consumption_result

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
            "mean": 0.0,
            "median": 0,
            "max": 0,
            "min": 0,
            "simulations": 0,
        },
        "control": {
            "total": 0,
            "mean": 0.0,
            "median": 0,
            "max": 0,
            "min": 0,
            "simulations": 0,
        },
        "independent": {
            "total": 0,
            "mean": 0.0,
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


def process_experiment_rewards_by_generation(
    experiment: str,
) -> Dict[str, Dict[int, float]]:
    #! refactor
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
        experiment_path = os.path.join(
            "results/one_of_a_kind_v1/experiments/data", experiment
        )
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


def process_experiment(agent_type: str, experiment: str) -> Dict[str, Union[List, int]]:
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
        experiment_path = os.path.join(
            "results/one_of_a_kind_v1/experiments/data", experiment
        )
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
                result = get_data(db_path)
                if result is not None:
                    steps, pop, steps_count = result
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


def find_experiments(
    base_path: str,
) -> Dict[str, Union[Dict[str, List[str]], List[str]]]:
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
        experiment_path = f"results/one_of_a_kind_v1/experiments/data/{experiment}"
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
                db_result = get_columns_data_by_agent_type(db_path)
                if db_result is not None:
                    steps, pops, steps_count = db_result
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
        experiment_path = f"results/one_of_a_kind_v1/experiments/data/{experiment}"
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
                db_result = get_resource_consumption_data(db_path)
                if db_result is not None:
                    steps, consumption, steps_count = db_result
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
) -> Dict[str, Dict[str, Union[Dict[str, float], int]]]:
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
        experiment_path = f"results/one_of_a_kind_v1/experiments/data/{experiment}"
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


def process_experiment_resource_levels(experiment: str) -> Dict[str, Union[List, int]]:
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
        experiment_path = f"results/one_of_a_kind_v1/experiments/data/{experiment}"
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
                result_data = get_resource_level_data(db_path)
                if result_data is not None:
                    steps, resource_levels, steps_count = result_data
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

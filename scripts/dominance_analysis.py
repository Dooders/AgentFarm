#!/usr/bin/env python3
"""
analyze_simulations.py

This script analyzes simulation databases to determine what factors predict
which agent type becomes dominant during the simulation.

Key questions:
1. What initial conditions (e.g., proximity to resources) predict dominance?
2. Which agent parameters correlate with dominance?
3. What reproduction and resource acquisition patterns lead to dominance?

Each simulation starts with exactly one agent of each type (system, independent, control).

Usage:
    python analyze_simulations.py
"""

import glob
import json
import logging
import os
import shutil
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlalchemy

# Import analysis configuration
from analysis_config import DATA_PATH, OUTPUT_PATH
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import sessionmaker

# Import the database models
from farm.database.models import (
    AgentModel,
    ReproductionEventModel,
    ResourceModel,
    SimulationStepModel,
)

# Global variable to store the current output path
CURRENT_OUTPUT_PATH = None


# Setup logging to both console and file
def setup_logging(output_dir):
    """
    Set up logging to both console and file.

    Parameters
    ----------
    output_dir : str
        Directory to save the log file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"analysis_log_{timestamp}.txt")

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"Logging to {log_file}")
    return log_file


def compute_population_dominance(sim_session):
    """
    Compute the dominant agent type by final population.
    Query the final simulation step and choose the type with the highest count.
    """
    final_step = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.desc())
        .first()
    )
    if final_step is None:
        return None
    # Create a dictionary of agent counts
    counts = {
        "system": final_step.system_agents,
        "independent": final_step.independent_agents,
        "control": final_step.control_agents,
    }
    # Return the key with the maximum count
    return max(counts, key=counts.get)


def compute_survival_dominance(sim_session):
    """
    Compute the dominant agent type by average survival time.
    For each agent, compute survival time as (death_time - birth_time) if the agent has died.
    (For agents still alive, use the final step as a proxy)
    Then, for each agent type, compute the average survival time.
    Return the type with the highest average.
    """
    agents = sim_session.query(AgentModel).all()

    # Get the final step number for calculating survival of still-alive agents
    final_step = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.desc())
        .first()
    )
    final_step_number = final_step.step_number if final_step else 0

    survival_by_type = {}
    count_by_type = {}
    for agent in agents:
        # For alive agents, use the final step as the death time
        if agent.death_time is not None:
            survival = agent.death_time - agent.birth_time
        else:
            survival = final_step_number - agent.birth_time

        survival_by_type.setdefault(agent.agent_type, 0)
        count_by_type.setdefault(agent.agent_type, 0)
        survival_by_type[agent.agent_type] += survival
        count_by_type[agent.agent_type] += 1

    avg_survival = {
        agent_type: (survival_by_type[agent_type] / count_by_type[agent_type])
        for agent_type in survival_by_type
        if count_by_type[agent_type] > 0
    }
    if not avg_survival:
        return None
    return max(avg_survival, key=avg_survival.get)


def get_final_population_counts(sim_session):
    """
    Get the final population counts for each agent type.
    """
    final_step = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.desc())
        .first()
    )
    if final_step is None:
        return None

    return {
        "system_agents": final_step.system_agents,
        "independent_agents": final_step.independent_agents,
        "control_agents": final_step.control_agents,
        "total_agents": final_step.total_agents,
        "final_step": final_step.step_number,
    }


def get_agent_survival_stats(sim_session):
    """
    Get detailed survival statistics for each agent type.
    """
    agents = sim_session.query(AgentModel).all()

    # Initialize counters and accumulators
    stats = {
        "system": {"count": 0, "alive": 0, "dead": 0, "total_survival": 0},
        "independent": {"count": 0, "alive": 0, "dead": 0, "total_survival": 0},
        "control": {"count": 0, "alive": 0, "dead": 0, "total_survival": 0},
    }

    # Get the final step number for calculating survival of still-alive agents
    final_step = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.desc())
        .first()
    )
    final_step_number = final_step.step_number if final_step else 0

    for agent in agents:
        agent_type = agent.agent_type.lower()
        if agent_type not in stats:
            continue

        stats[agent_type]["count"] += 1

        if agent.death_time is not None:
            stats[agent_type]["dead"] += 1
            survival = agent.death_time - agent.birth_time
        else:
            stats[agent_type]["alive"] += 1
            # For alive agents, use the final step as the death time
            survival = final_step_number - agent.birth_time

        stats[agent_type]["total_survival"] += survival

    # Calculate averages
    result = {}
    for agent_type, data in stats.items():
        if data["count"] > 0:
            result[f"{agent_type}_count"] = data["count"]
            result[f"{agent_type}_alive"] = data["alive"]
            result[f"{agent_type}_dead"] = data["dead"]
            result[f"{agent_type}_avg_survival"] = (
                data["total_survival"] / data["count"]
            )
            if data["dead"] > 0:
                result[f"{agent_type}_dead_ratio"] = data["dead"] / data["count"]
            else:
                result[f"{agent_type}_dead_ratio"] = 0

    return result


def get_initial_positions_and_resources(sim_session, config):
    """
    Get the initial positions of agents and resources.
    Calculate distances between agents and resources.
    """
    # Get the initial agents (birth_time = 0)
    initial_agents = (
        sim_session.query(AgentModel).filter(AgentModel.birth_time == 0).all()
    )

    # Get the initial resources (step_number = 0)
    initial_resources = (
        sim_session.query(ResourceModel).filter(ResourceModel.step_number == 0).all()
    )

    if not initial_agents or not initial_resources:
        return {}

    # Extract positions
    agent_positions = {
        agent.agent_type.lower(): (agent.position_x, agent.position_y)
        for agent in initial_agents
    }

    resource_positions = [
        (resource.position_x, resource.position_y, resource.amount)
        for resource in initial_resources
    ]

    # Calculate distances to resources
    result = {}
    for agent_type, pos in agent_positions.items():
        # Calculate distances to all resources
        distances = [
            euclidean(pos, (r_pos[0], r_pos[1])) for r_pos in resource_positions
        ]

        # Calculate distance to nearest resource
        if distances:
            result[f"{agent_type}_nearest_resource_dist"] = min(distances)

            # Calculate average distance to all resources
            result[f"{agent_type}_avg_resource_dist"] = sum(distances) / len(distances)

            # Calculate weighted distance (by resource amount)
            weighted_distances = [
                distances[i]
                * (
                    1 / (resource_positions[i][2] + 1)
                )  # Add 1 to avoid division by zero
                for i in range(len(distances))
            ]
            result[f"{agent_type}_weighted_resource_dist"] = sum(
                weighted_distances
            ) / len(weighted_distances)

            # Count resources within gathering range
            gathering_range = config.get("gathering_range", 30)
            resources_in_range = sum(1 for d in distances if d <= gathering_range)
            result[f"{agent_type}_resources_in_range"] = resources_in_range

            # Calculate total resource amount within gathering range
            resource_amount_in_range = sum(
                resource_positions[i][2]
                for i in range(len(distances))
                if distances[i] <= gathering_range
            )
            result[f"{agent_type}_resource_amount_in_range"] = resource_amount_in_range

    # Calculate relative advantages (differences between agent types)
    agent_types = ["system", "independent", "control"]
    for i, type1 in enumerate(agent_types):
        for type2 in agent_types[i + 1 :]:
            # Difference in nearest resource distance
            key1 = f"{type1}_nearest_resource_dist"
            key2 = f"{type2}_nearest_resource_dist"
            if key1 in result and key2 in result:
                result[f"{type1}_vs_{type2}_nearest_resource_advantage"] = (
                    result[key2] - result[key1]
                )

            # Difference in resources in range
            key1 = f"{type1}_resources_in_range"
            key2 = f"{type2}_resources_in_range"
            if key1 in result and key2 in result:
                result[f"{type1}_vs_{type2}_resources_in_range_advantage"] = (
                    result[key1] - result[key2]
                )

            # Difference in resource amount in range
            key1 = f"{type1}_resource_amount_in_range"
            key2 = f"{type2}_resource_amount_in_range"
            if key1 in result and key2 in result:
                result[f"{type1}_vs_{type2}_resource_amount_advantage"] = (
                    result[key1] - result[key2]
                )

    return result


def get_reproduction_stats(sim_session):
    """
    Analyze reproduction patterns for each agent type.
    """
    try:
        # Check if ReproductionEventModel table exists
        inspector = sqlalchemy.inspect(sim_session.bind)
        if "reproduction_events" not in inspector.get_table_names():
            logging.warning("No reproduction_events table found in database")
            return {}

        # Query reproduction events
        try:
            reproduction_events = sim_session.query(ReproductionEventModel).all()
            logging.info(
                f"Found {len(reproduction_events)} reproduction events in database"
            )
        except Exception as e:
            logging.error(f"Error querying reproduction events: {e}")
            return {}

        if not reproduction_events:
            logging.warning("No reproduction events found in the database")
            return {}

        # Get all agents to determine their types
        try:
            # Create a mapping of agent IDs to normalized agent types
            agents_raw = {
                agent.agent_id: agent.agent_type
                for agent in sim_session.query(AgentModel).all()
            }

            # Normalize agent types (handle different case formats)
            agents = {}
            for agent_id, agent_type in agents_raw.items():
                # Convert to lowercase for comparison
                agent_type_lower = agent_type.lower()

                # Map to standard types based on substring matching
                if "system" in agent_type_lower:
                    normalized_type = "system"
                elif "independent" in agent_type_lower:
                    normalized_type = "independent"
                elif "control" in agent_type_lower:
                    normalized_type = "control"
                else:
                    normalized_type = "unknown"

                agents[agent_id] = normalized_type

            logging.info(f"Found {len(agents)} agents in database")

            # Log the agent type mapping for debugging
            agent_types_found = set(agents_raw.values())
            normalized_types = set(agents.values())
            logging.info(
                f"Original agent types in database: {', '.join(agent_types_found)}"
            )
            logging.info(f"Normalized to: {', '.join(normalized_types)}")

        except Exception as e:
            logging.error(f"Error querying agents: {e}")
            return {}

        # Initialize counters
        stats = {
            "system": {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "first_reproduction_time": float("inf"),
                "resources_spent": 0,
                "offspring_resources": 0,
            },
            "independent": {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "first_reproduction_time": float("inf"),
                "resources_spent": 0,
                "offspring_resources": 0,
            },
            "control": {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "first_reproduction_time": float("inf"),
                "resources_spent": 0,
                "offspring_resources": 0,
            },
        }

        # Process reproduction events
        unknown_agent_types = set()
        missing_resource_data = 0

        for event in reproduction_events:
            try:
                parent_id = event.parent_id
                parent_type = agents.get(parent_id, "unknown")

                if parent_type not in stats:
                    if parent_type not in unknown_agent_types:
                        unknown_agent_types.add(parent_type)
                        logging.warning(
                            f"Unknown agent type: {parent_type} for agent {parent_id}"
                        )
                    continue

                stats[parent_type]["attempts"] += 1

                # Calculate resources spent on reproduction
                try:
                    resources_spent = (
                        event.parent_resources_before - event.parent_resources_after
                    )
                    stats[parent_type]["resources_spent"] += resources_spent
                except (TypeError, AttributeError):
                    missing_resource_data += 1
                    # Skip resource calculation if data is missing

                if event.success:
                    stats[parent_type]["successes"] += 1
                    # Track first successful reproduction time
                    stats[parent_type]["first_reproduction_time"] = min(
                        stats[parent_type]["first_reproduction_time"], event.step_number
                    )
                    # Track resources given to offspring
                    if (
                        hasattr(event, "offspring_initial_resources")
                        and event.offspring_initial_resources is not None
                    ):
                        stats[parent_type][
                            "offspring_resources"
                        ] += event.offspring_initial_resources
                else:
                    stats[parent_type]["failures"] += 1
            except Exception as e:
                logging.error(f"Error processing reproduction event: {e}")
                continue

        if missing_resource_data > 0:
            logging.warning(
                f"Missing resource data for {missing_resource_data} reproduction events"
            )

        if unknown_agent_types:
            logging.warning(
                f"Found unknown agent types: {', '.join(unknown_agent_types)}"
            )

        # Calculate derived metrics
        result = {}
        for agent_type, data in stats.items():
            if data["attempts"] > 0:
                result[f"{agent_type}_reproduction_attempts"] = data["attempts"]
                result[f"{agent_type}_reproduction_successes"] = data["successes"]
                result[f"{agent_type}_reproduction_failures"] = data["failures"]

                # Calculate success rate
                result[f"{agent_type}_reproduction_success_rate"] = (
                    data["successes"] / data["attempts"]
                )

                # Calculate resource metrics if we have resource data
                if data["resources_spent"] > 0:
                    result[f"{agent_type}_avg_resources_per_reproduction"] = (
                        data["resources_spent"] / data["attempts"]
                    )

                    if data["successes"] > 0 and data["offspring_resources"] > 0:
                        result[f"{agent_type}_avg_offspring_resources"] = (
                            data["offspring_resources"] / data["successes"]
                        )
                        result[f"{agent_type}_reproduction_efficiency"] = (
                            data["offspring_resources"] / data["resources_spent"]
                        )

                # First reproduction time
                if data["first_reproduction_time"] != float("inf"):
                    result[f"{agent_type}_first_reproduction_time"] = data[
                        "first_reproduction_time"
                    ]
                else:
                    result[f"{agent_type}_first_reproduction_time"] = (
                        -1
                    )  # No successful reproduction

        # Calculate relative advantages (differences between agent types)
        agent_types = ["system", "independent", "control"]
        for i, type1 in enumerate(agent_types):
            for type2 in agent_types[i + 1 :]:
                # Difference in reproduction success rate
                key1 = f"{type1}_reproduction_success_rate"
                key2 = f"{type2}_reproduction_success_rate"
                if key1 in result and key2 in result:
                    result[f"{type1}_vs_{type2}_reproduction_rate_advantage"] = (
                        result[key1] - result[key2]
                    )

                # Difference in first reproduction time (negative is better - earlier reproduction)
                key1 = f"{type1}_first_reproduction_time"
                key2 = f"{type2}_first_reproduction_time"
                if (
                    key1 in result
                    and key2 in result
                    and result[key1] > 0
                    and result[key2] > 0
                ):
                    result[f"{type1}_vs_{type2}_first_reproduction_advantage"] = (
                        result[key2] - result[key1]
                    )

                # Difference in reproduction efficiency
                key1 = f"{type1}_reproduction_efficiency"
                key2 = f"{type2}_reproduction_efficiency"
                if key1 in result and key2 in result:
                    result[f"{type1}_vs_{type2}_reproduction_efficiency_advantage"] = (
                        result[key1] - result[key2]
                    )

        # Log summary of reproduction stats
        if result:
            logging.info("\nReproduction statistics summary:")
            for agent_type in agent_types:
                attempts_key = f"{agent_type}_reproduction_attempts"
                success_rate_key = f"{agent_type}_reproduction_success_rate"
                if attempts_key in result:
                    attempts = result[attempts_key]
                    success_rate = result.get(success_rate_key, 0) * 100
                    logging.info(
                        f"  {agent_type}: {attempts} attempts, {success_rate:.1f}% success rate"
                    )

            # Log all calculated metrics for debugging
            logging.info(f"Calculated {len(result)} reproduction metrics:")
            for key in sorted(result.keys())[:10]:  # Show first 10
                logging.info(f"  {key}: {result[key]}")
        else:
            logging.warning("No reproduction statistics could be calculated")

        return result

    except Exception as e:
        logging.error(f"Error in get_reproduction_stats: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return {}


def compute_dominance_switches(sim_session):
    """
    Analyze how often agent types switch dominance during a simulation.

    This function examines the entire simulation history to identify:
    1. Total number of dominance switches
    2. Average duration of dominance periods for each agent type
    3. Volatility of dominance (frequency of switches in different phases)
    4. Transition matrix showing which agent types tend to take over from others

    Returns a dictionary with dominance switching statistics.
    """
    # Query all simulation steps ordered by step number
    sim_steps = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.asc())
        .all()
    )

    if not sim_steps:
        return None

    # Initialize tracking variables
    agent_types = ["system", "independent", "control"]
    current_dominant = None
    previous_dominant = None
    dominance_periods = {agent_type: [] for agent_type in agent_types}
    switches = []
    transition_matrix = {
        from_type: {to_type: 0 for to_type in agent_types} for from_type in agent_types
    }

    # Track the current dominance period
    period_start_step = 0
    total_steps = len(sim_steps)

    # Process each simulation step
    for step_idx, step in enumerate(sim_steps):
        # Determine which type is dominant in this step
        counts = {
            "system": step.system_agents,
            "independent": step.independent_agents,
            "control": step.control_agents,
        }

        # Skip steps with no agents
        if sum(counts.values()) == 0:
            continue

        current_dominant = max(counts, key=counts.get)

        # If this is the first step with agents, initialize
        if previous_dominant is None:
            previous_dominant = current_dominant
            period_start_step = step_idx
            continue

        # Check if dominance has switched
        if current_dominant != previous_dominant:
            # Record the switch
            switches.append(
                {
                    "step": step.step_number,
                    "from": previous_dominant,
                    "to": current_dominant,
                    "phase": (
                        "early"
                        if step_idx < total_steps / 3
                        else "middle" if step_idx < 2 * total_steps / 3 else "late"
                    ),
                }
            )

            # Update transition matrix
            transition_matrix[previous_dominant][current_dominant] += 1

            # Record the duration of the completed dominance period
            period_duration = step_idx - period_start_step
            dominance_periods[previous_dominant].append(period_duration)

            # Reset for the new period
            period_start_step = step_idx
            previous_dominant = current_dominant

    # Record the final dominance period
    if previous_dominant is not None:
        final_period_duration = total_steps - period_start_step
        dominance_periods[previous_dominant].append(final_period_duration)

    # Calculate average dominance period durations
    avg_dominance_periods = {}
    for agent_type in agent_types:
        periods = dominance_periods[agent_type]
        avg_dominance_periods[agent_type] = (
            sum(periods) / len(periods) if periods else 0
        )

    # Calculate phase-specific switch counts
    phase_switches = {
        "early": sum(1 for s in switches if s["phase"] == "early"),
        "middle": sum(1 for s in switches if s["phase"] == "middle"),
        "late": sum(1 for s in switches if s["phase"] == "late"),
    }

    # Calculate normalized transition probabilities
    transition_probabilities = {from_type: {} for from_type in agent_types}

    for from_type in agent_types:
        total_transitions = sum(transition_matrix[from_type].values())
        for to_type in agent_types:
            transition_probabilities[from_type][to_type] = (
                transition_matrix[from_type][to_type] / total_transitions
                if total_transitions > 0
                else 0
            )

    # Return comprehensive results
    return {
        "total_switches": len(switches),
        "switches_per_step": len(switches) / total_steps if total_steps > 0 else 0,
        "switches_detail": switches,
        "avg_dominance_periods": avg_dominance_periods,
        "phase_switches": phase_switches,
        "transition_matrix": transition_matrix,
        "transition_probabilities": transition_probabilities,
    }


def analyze_simulations(experiment_path):
    """
    Analyze all simulation databases in the experiment folder.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment folder containing simulation databases

    Returns
    -------
    pandas.DataFrame
        DataFrame with analysis results for each simulation
    """
    data = []

    # Find all simulation folders
    sim_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))

    for folder in sim_folders:
        # Check if this is a simulation folder with a database
        db_path = os.path.join(folder, "simulation.db")
        config_path = os.path.join(folder, "config.json")

        if not (os.path.exists(db_path) and os.path.exists(config_path)):
            logging.warning(f"Skipping {folder}: Missing database or config file")
            continue

        try:
            # Extract the iteration number from the folder name
            folder_name = os.path.basename(folder)
            if folder_name.startswith("iteration_"):
                iteration = int(folder_name.split("_")[1])
            else:
                logging.warning(f"Skipping {folder}: Invalid folder name format")
                continue

            # Load the configuration
            with open(config_path, "r") as f:
                config = json.load(f)

            # Connect to the database
            engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
            Session = sessionmaker(bind=engine)
            session = Session()

            # Compute dominance metrics
            population_dominance = compute_population_dominance(session)
            survival_dominance = compute_survival_dominance(session)
            comprehensive_dominance = compute_comprehensive_dominance(session)

            # Compute dominance switching metrics
            dominance_switches = compute_dominance_switches(session)

            # Get initial positions and resource data
            initial_data = get_initial_positions_and_resources(session, config)

            # Get final population counts
            final_counts = get_final_population_counts(session)

            # Get agent survival statistics
            survival_stats = get_agent_survival_stats(session)

            # Get reproduction statistics
            reproduction_stats = get_reproduction_stats(session)

            # Combine all data
            sim_data = {
                "iteration": iteration,
                "population_dominance": population_dominance,
                "survival_dominance": survival_dominance,
                "comprehensive_dominance": comprehensive_dominance["dominant_type"],
            }

            # Add dominance scores
            for agent_type in ["system", "independent", "control"]:
                sim_data[f"{agent_type}_dominance_score"] = comprehensive_dominance[
                    "scores"
                ][agent_type]
                sim_data[f"{agent_type}_auc"] = comprehensive_dominance["metrics"][
                    "auc"
                ][agent_type]
                sim_data[f"{agent_type}_recency_weighted_auc"] = (
                    comprehensive_dominance["metrics"]["recency_weighted_auc"][
                        agent_type
                    ]
                )
                sim_data[f"{agent_type}_dominance_duration"] = comprehensive_dominance[
                    "metrics"
                ]["dominance_duration"][agent_type]
                sim_data[f"{agent_type}_growth_trend"] = comprehensive_dominance[
                    "metrics"
                ]["growth_trends"][agent_type]
                sim_data[f"{agent_type}_final_ratio"] = comprehensive_dominance[
                    "metrics"
                ]["final_ratios"][agent_type]

            # Add dominance switching data
            if dominance_switches:
                sim_data["total_switches"] = dominance_switches["total_switches"]
                sim_data["switches_per_step"] = dominance_switches["switches_per_step"]

                # Add average dominance periods
                for agent_type in ["system", "independent", "control"]:
                    sim_data[f"{agent_type}_avg_dominance_period"] = dominance_switches[
                        "avg_dominance_periods"
                    ][agent_type]

                # Add phase-specific switch counts
                for phase in ["early", "middle", "late"]:
                    sim_data[f"{phase}_phase_switches"] = dominance_switches[
                        "phase_switches"
                    ][phase]

                # Add transition matrix data
                for from_type in ["system", "independent", "control"]:
                    for to_type in ["system", "independent", "control"]:
                        sim_data[f"{from_type}_to_{to_type}"] = dominance_switches[
                            "transition_probabilities"
                        ][from_type][to_type]

            # Add all other data
            sim_data.update(initial_data)
            sim_data.update(final_counts)
            sim_data.update(survival_stats)
            sim_data.update(reproduction_stats)

            data.append(sim_data)

            # Close the session
            session.close()

        except Exception as e:
            logging.error(f"Error processing {folder}: {e}")

    # Convert to DataFrame
    return pd.DataFrame(data)


def train_classifier(X, y, label_name):
    """
    Train a Random Forest classifier and print a classification report and feature importances.
    """
    # Handle categorical features with one-hot encoding
    categorical_cols = X.select_dtypes(exclude=["number"]).columns
    if not categorical_cols.empty:
        logging.info(
            f"One-hot encoding {len(categorical_cols)} categorical features..."
        )
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logging.info(f"\n=== Classification Report for {label_name} Dominance ===")
    logging.info(classification_report(y_test, y_pred))

    # Print confusion matrix
    logging.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logging.info(cm)

    # Print feature importances
    importances = clf.feature_importances_
    feature_names = X.columns
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    logging.info("\nTop 15 Feature Importances:")
    for feat, imp in feat_imp[:15]:
        logging.info(f"{feat}: {imp:.3f}")

    return clf, feat_imp


def plot_dominance_distribution(df):
    """
    Plot the distribution of dominance types as percentages.
    """
    global CURRENT_OUTPUT_PATH

    # Determine how many plots we need
    dominance_measures = [
        "population_dominance",
        "survival_dominance",
        "comprehensive_dominance",
    ]
    available_measures = [m for m in dominance_measures if m in df.columns]
    n_plots = len(available_measures)

    if n_plots == 0:
        return

    # Create a figure with the appropriate number of subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    # If there's only one measure, axes will not be an array
    if n_plots == 1:
        axes = [axes]

    # Plot each dominance measure
    for i, measure in enumerate(available_measures):
        # Get counts and convert to percentages
        counts = df[measure].value_counts()
        total = counts.sum()
        percentages = (counts / total) * 100

        # Plot percentages
        axes[i].bar(percentages.index, percentages.values)
        axes[i].set_title(f"{measure.replace('_', ' ').title()} Distribution")
        axes[i].set_ylabel("Percentage (%)")
        axes[i].set_xlabel("Agent Type")

        # Add percentage labels on top of each bar
        for j, p in enumerate(percentages):
            axes[i].annotate(f"{p:.1f}%", (j, p), ha="center", va="bottom")

        # Set y-axis limit to slightly above 100% to make room for annotations
        axes[i].set_ylim(0, 105)

    # Add caption
    caption = (
        "This chart shows the percentage distribution of dominant agent types across simulations. "
        "Each bar represents the percentage of simulations where a particular agent type "
        "(system, independent, or control) was dominant according to different dominance measures."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_file = os.path.join(CURRENT_OUTPUT_PATH, "dominance_distribution.png")
    plt.savefig(output_file)
    logging.info(f"Saved dominance distribution plot to {output_file}")
    plt.close()


def plot_feature_importance(feat_imp, label_name):
    """
    Plot feature importance for a classifier.
    """
    global CURRENT_OUTPUT_PATH

    top_features = feat_imp[:15]
    features = [f[0] for f in top_features]
    importances = [f[1] for f in top_features]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), importances, align="center")
    plt.yticks(range(len(features)), features)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 15 Feature Importances for {label_name}")

    # Add caption
    caption = (
        f"This chart displays the top 15 most important features that predict {label_name}. "
        f"Features with higher importance values have a stronger influence on determining "
        f"which agent type becomes dominant in the simulation."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_file = os.path.join(
        CURRENT_OUTPUT_PATH, f"{label_name}_feature_importance.png"
    )
    plt.savefig(output_file)
    logging.info(f"Saved feature importance plot to {output_file}")
    plt.close()


def plot_resource_proximity_vs_dominance(df):
    """
    Plot the relationship between initial resource proximity and dominance.
    """
    global CURRENT_OUTPUT_PATH

    resource_metrics = [
        col
        for col in df.columns
        if "resource_distance" in col or "resource_proximity" in col
    ]

    if not resource_metrics:
        return

    # Create a figure with subplots for each metric
    n_metrics = len(resource_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5 * n_metrics))

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(resource_metrics):
        if metric in df.columns:
            sns.boxplot(x="population_dominance", y=metric, data=df, ax=axes[i])
            axes[i].set_title(f"{metric} vs Population Dominance")
            axes[i].set_xlabel("Dominant Agent Type")
            axes[i].set_ylabel(metric)

    # Add caption
    caption = (
        "This chart shows the relationship between initial resource proximity/distance and which agent type "
        "becomes dominant. The boxplots display the distribution of resource metrics for each dominant agent type, "
        "helping identify if certain agent types tend to dominate when resources are closer or further away."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_file = os.path.join(
        CURRENT_OUTPUT_PATH, "resource_proximity_vs_dominance.png"
    )
    plt.savefig(output_file)
    logging.info(f"Saved resource proximity plot to {output_file}")
    plt.close()


def plot_reproduction_vs_dominance(df):
    """
    Plot reproduction metrics vs dominance.
    """
    global CURRENT_OUTPUT_PATH

    reproduction_metrics = [
        col for col in df.columns if "reproduction" in col or "offspring" in col
    ]

    if not reproduction_metrics:
        return

    # Create a figure with subplots for each metric
    n_metrics = len(reproduction_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5 * n_metrics))

    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(reproduction_metrics):
        if metric in df.columns:
            sns.boxplot(x="population_dominance", y=metric, data=df, ax=axes[i])
            axes[i].set_title(f"{metric} vs Population Dominance")
            axes[i].set_xlabel("Dominant Agent Type")
            axes[i].set_ylabel(metric)

    # Add caption
    caption = (
        "This chart illustrates the relationship between reproduction metrics and population dominance. "
        "The boxplots show how reproduction rates and offspring counts differ across simulations where "
        "different agent types became dominant, revealing how reproductive success correlates with dominance."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    output_file = os.path.join(CURRENT_OUTPUT_PATH, "reproduction_metrics_boxplots.png")
    plt.savefig(output_file)
    logging.info(f"Saved reproduction metrics boxplots to {output_file}")
    plt.close()


def plot_correlation_matrix(df, label_name):
    """
    Plot correlation matrix between features and the target label.
    """
    global CURRENT_OUTPUT_PATH

    # Convert categorical target to numeric for correlation
    target_numeric = pd.get_dummies(df[label_name])

    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number])

    # Remove the target if it's in the numeric features
    if label_name in numeric_features.columns:
        numeric_features = numeric_features.drop(label_name, axis=1)

    # Filter out columns with zero standard deviation to avoid division by zero warnings
    std_dev = numeric_features.std()
    valid_columns = std_dev[std_dev > 0].index
    numeric_features = numeric_features[valid_columns]

    if numeric_features.empty:
        logging.warning(
            f"No valid numeric features with non-zero standard deviation for {label_name}"
        )
        return

    # Calculate correlation with each target class
    correlations = {}
    for target_class in target_numeric.columns:
        # Filter out any NaN values in the target
        valid_mask = ~target_numeric[target_class].isna()
        if valid_mask.sum() > 0:
            correlations[target_class] = numeric_features[valid_mask].corrwith(
                target_numeric[target_class][valid_mask]
            )

    if not correlations:
        logging.warning(f"Could not calculate correlations for {label_name}")
        return

    # Combine correlations into a single DataFrame
    corr_df = pd.DataFrame(correlations)

    # Sort by absolute correlation
    corr_df["max_abs_corr"] = corr_df.abs().max(axis=1)
    corr_df = corr_df.sort_values("max_abs_corr", ascending=False).drop(
        "max_abs_corr", axis=1
    )

    # Plot top correlations
    top_n = min(20, len(corr_df))  # Ensure we don't try to plot more rows than we have
    if top_n == 0:
        logging.warning(f"No correlations to plot for {label_name}")
        return

    top_corr = corr_df.head(top_n)

    plt.figure(figsize=(12, 10))
    sns.heatmap(top_corr, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Top {top_n} Feature Correlations with {label_name}")

    # Add caption
    caption = (
        f"This heatmap shows the top {top_n} features most correlated with {label_name}. "
        f"Red cells indicate positive correlation (as the feature increases, the likelihood of that agent type "
        f"being dominant increases), while blue cells indicate negative correlation. "
        f"The intensity of color represents the strength of correlation."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    output_file = os.path.join(
        CURRENT_OUTPUT_PATH, f"{label_name}_correlation_matrix.png"
    )
    plt.savefig(output_file)
    logging.info(f"Saved correlation matrix plot to {output_file}")
    plt.close()


def compute_comprehensive_dominance(sim_session):
    """
    Compute a comprehensive dominance score that considers the entire simulation history.

    This function uses multiple metrics to determine dominance:
    1. Area Under the Curve (AUC): Total agent-steps throughout the simulation
    2. Recency-weighted AUC: Gives more weight to later steps in the simulation
    3. Dominance duration: How many steps each agent type was dominant
    4. Growth trend: Positive growth trends in the latter half of simulation
    5. Final population ratio: The proportion of agents at the end of simulation

    Returns a dictionary with dominance scores for each agent type and the overall dominant type.
    """
    # Query all simulation steps ordered by step number
    sim_steps = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.asc())
        .all()
    )

    if not sim_steps:
        return None

    # Initialize metrics
    agent_types = ["system", "independent", "control"]
    total_steps = len(sim_steps)

    # Calculate Area Under the Curve (agent-steps)
    auc = {agent_type: 0 for agent_type in agent_types}

    # Recency-weighted AUC (recent steps count more)
    recency_weighted_auc = {agent_type: 0 for agent_type in agent_types}

    # Count how many steps each type was dominant
    dominance_duration = {agent_type: 0 for agent_type in agent_types}

    # Track agent counts for calculating trends
    agent_counts = {agent_type: [] for agent_type in agent_types}

    # Process each simulation step
    for i, step in enumerate(sim_steps):
        # Calculate recency weight (later steps count more)
        recency_weight = 1 + (i / total_steps)

        # Update metrics for each agent type
        for agent_type in agent_types:
            agent_count = getattr(step, f"{agent_type}_agents")

            # Basic AUC - sum of agent counts across all steps
            auc[agent_type] += agent_count

            # Recency-weighted AUC
            recency_weighted_auc[agent_type] += agent_count * recency_weight

            # Track agent counts for trend analysis
            agent_counts[agent_type].append(agent_count)

        # Determine which type was dominant in this step
        counts = {
            "system": step.system_agents,
            "independent": step.independent_agents,
            "control": step.control_agents,
        }
        dominant_type = max(counts, key=counts.get) if any(counts.values()) else None
        if dominant_type:
            dominance_duration[dominant_type] += 1

    # Calculate growth trends in the latter half
    growth_trends = {}
    for agent_type in agent_types:
        counts = agent_counts[agent_type]
        if len(counts) >= 4:  # Need at least a few points for meaningful trend
            # Focus on the latter half of the simulation
            latter_half = counts[len(counts) // 2 :]

            if all(x == 0 for x in latter_half):
                growth_trends[agent_type] = 0
            else:
                # Simple trend calculation: last value compared to average of latter half
                latter_half_avg = sum(latter_half) / len(latter_half)
                if latter_half_avg == 0:
                    growth_trends[agent_type] = 0
                else:
                    growth_trends[agent_type] = (
                        latter_half[-1] - latter_half_avg
                    ) / latter_half_avg
        else:
            growth_trends[agent_type] = 0

    # Calculate final population ratios
    final_step = sim_steps[-1]
    total_final_agents = final_step.total_agents
    final_ratios = {}

    if total_final_agents > 0:
        for agent_type in agent_types:
            agent_count = getattr(final_step, f"{agent_type}_agents")
            final_ratios[agent_type] = agent_count / total_final_agents
    else:
        final_ratios = {agent_type: 0 for agent_type in agent_types}

    # Normalize metrics to [0,1] scale
    normalized_metrics = {}

    for metric_name, metric_values in [
        ("auc", auc),
        ("recency_weighted_auc", recency_weighted_auc),
        ("dominance_duration", dominance_duration),
    ]:
        total = sum(metric_values.values())
        if total > 0:
            normalized_metrics[metric_name] = {
                agent_type: value / total for agent_type, value in metric_values.items()
            }
        else:
            normalized_metrics[metric_name] = {
                agent_type: 0 for agent_type in agent_types
            }

    # Calculate final composite score with weights for different metrics
    weights = {
        "auc": 0.2,  # Basic population persistence
        "recency_weighted_auc": 0.3,  # Emphasize later simulation stages
        "dominance_duration": 0.2,  # Reward consistent dominance
        "growth_trend": 0.1,  # Reward positive growth in latter half
        "final_ratio": 0.2,  # Reward final state
    }

    composite_scores = {agent_type: 0 for agent_type in agent_types}

    for agent_type in agent_types:
        composite_scores[agent_type] = (
            weights["auc"] * normalized_metrics["auc"][agent_type]
            + weights["recency_weighted_auc"]
            * normalized_metrics["recency_weighted_auc"][agent_type]
            + weights["dominance_duration"]
            * normalized_metrics["dominance_duration"][agent_type]
            + weights["growth_trend"]
            * (max(0, growth_trends[agent_type]))  # Only count positive growth
            + weights["final_ratio"] * final_ratios[agent_type]
        )

    # Determine overall dominant type
    dominant_type = (
        max(composite_scores, key=composite_scores.get)
        if any(composite_scores.values())
        else None
    )

    # Return comprehensive results
    return {
        "dominant_type": dominant_type,
        "scores": composite_scores,
        "metrics": {
            "auc": auc,
            "recency_weighted_auc": recency_weighted_auc,
            "dominance_duration": dominance_duration,
            "growth_trends": growth_trends,
            "final_ratios": final_ratios,
        },
        "normalized_metrics": normalized_metrics,
    }


def plot_dominance_comparison(df):
    """
    Create visualizations to compare different dominance measures.

    This function creates:
    1. A comparison of how often each agent type is dominant according to different measures (as percentages)
    2. A correlation heatmap between different dominance metrics
    3. A scatter plot showing the relationship between AUC and composite dominance scores

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the dominance metrics for each simulation iteration
    """
    global CURRENT_OUTPUT_PATH

    plt.figure(figsize=(15, 10))

    # 1. Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Define dominance measures and agent types
    dominance_measures = [
        "population_dominance",
        "survival_dominance",
        "comprehensive_dominance",
    ]
    agent_types = ["system", "independent", "control"]
    colors = {"system": "blue", "independent": "green", "control": "red"}

    # 2. Dominance distribution comparison (as percentages)
    ax = axes[0, 0]
    comparison_data = []

    # Calculate percentages for each measure
    for measure in dominance_measures:
        if measure in df.columns:
            counts = df[measure].value_counts()
            total = counts.sum()

            for agent_type in agent_types:
                if agent_type in counts:
                    percentage = (counts[agent_type] / total) * 100
                    comparison_data.append(
                        {
                            "Measure": measure.replace("_", " ").title(),
                            "Agent Type": agent_type,
                            "Percentage": percentage,
                        }
                    )

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        sns.barplot(
            x="Agent Type",
            y="Percentage",
            hue="Measure",
            data=comparison_df,
            ax=ax,
        )
        ax.set_title("Dominance by Different Measures")
        ax.set_xlabel("Agent Type")
        ax.set_ylabel("Percentage (%)")
        ax.legend(title="Dominance Measure")

        # Add percentage labels on top of each bar
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.1f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
            )

        # Set y-axis limit to slightly above 100% to make room for annotations
        ax.set_ylim(0, 105)

    # 3. Correlation between dominance scores
    ax = axes[0, 1]
    score_cols = [col for col in df.columns if col.endswith("_dominance_score")]

    if score_cols:
        corr = df[score_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Between Dominance Scores")

    # 4. AUC vs Dominance Score
    ax = axes[1, 0]
    for agent_type in agent_types:
        auc_col = f"{agent_type}_auc"
        score_col = f"{agent_type}_dominance_score"

        if auc_col in df.columns and score_col in df.columns:
            ax.scatter(
                df[auc_col],
                df[score_col],
                label=agent_type,
                color=colors[agent_type],
                alpha=0.6,
            )

    ax.set_xlabel("Total Agent-Steps (AUC)")
    ax.set_ylabel("Composite Dominance Score")
    ax.set_title("Relationship Between AUC and Composite Dominance Score")
    ax.legend()

    # 5. Population vs Dominance Score
    ax = axes[1, 1]
    for agent_type in agent_types:
        pop_col = f"{agent_type}_agents"  # Final population count
        score_col = f"{agent_type}_dominance_score"

        if pop_col in df.columns and score_col in df.columns:
            ax.scatter(
                df[pop_col],
                df[score_col],
                label=agent_type,
                color=colors[agent_type],
                alpha=0.6,
            )

    ax.set_xlabel("Final Population Count")
    ax.set_ylabel("Composite Dominance Score")
    ax.set_title("Final Population vs Composite Dominance Score")
    ax.legend()

    # Add caption
    caption = (
        "This multi-panel figure compares different dominance measures across simulations. "
        "Top left: Percentage of simulations where each agent type is dominant according to different measures. "
        "Top right: Correlation between different dominance scores. "
        "Bottom left: Relationship between total agent-steps (AUC) and composite dominance score. "
        "Bottom right: Relationship between final population count and composite dominance score."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    output_file = os.path.join(CURRENT_OUTPUT_PATH, "dominance_comparison.png")
    plt.savefig(output_file, dpi=300)
    logging.info(f"Saved dominance comparison plot to {output_file}")
    plt.close()


def plot_dominance_switches(df):
    """
    Create visualizations for dominance switching patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    """
    if df.empty or "total_switches" not in df.columns:
        logging.warning("No dominance switch data available for plotting")
        return

    # 1. Distribution of total switches
    plt.figure(figsize=(10, 6))
    sns.histplot(df["total_switches"], kde=True)
    plt.title("Distribution of Dominance Switches Across Simulations")
    plt.xlabel("Number of Dominance Switches")
    plt.ylabel("Count")

    # Add caption for the first plot
    caption = (
        "This histogram shows how many dominance switches occurred in each simulation. "
        "A dominance switch happens when the dominant agent type changes during the simulation. "
        "The distribution reveals whether simulations typically have few or many switches in dominance."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    output_file = os.path.join(
        CURRENT_OUTPUT_PATH, "dominance_switches_distribution.png"
    )
    plt.savefig(output_file)
    plt.close()

    # 2. Average dominance period duration by agent type
    plt.figure(figsize=(10, 6))
    agent_types = ["system", "independent", "control"]
    avg_periods = [
        df[f"{agent_type}_avg_dominance_period"].mean() for agent_type in agent_types
    ]

    bars = plt.bar(agent_types, avg_periods)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.title("Average Dominance Period Duration by Agent Type")
    plt.xlabel("Agent Type")
    plt.ylabel("Average Steps")

    # Add caption for the second plot
    caption = (
        "This bar chart shows how long each agent type typically remains dominant before being "
        "replaced by another type. Longer bars indicate that when this agent type becomes dominant, "
        "it tends to maintain dominance for more simulation steps."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    output_file = os.path.join(CURRENT_OUTPUT_PATH, "avg_dominance_period.png")
    plt.savefig(output_file)
    plt.close()

    # 3. Phase-specific switch frequency
    if "early_phase_switches" in df.columns:
        plt.figure(figsize=(10, 6))
        phases = ["early", "middle", "late"]
        phase_data = [df[f"{phase}_phase_switches"].mean() for phase in phases]

        bars = plt.bar(phases, phase_data)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
            )

        plt.title("Average Dominance Switches by Simulation Phase")
        plt.xlabel("Simulation Phase")
        plt.ylabel("Average Number of Switches")

        # Add caption for the third plot
        caption = (
            "This chart shows how dominance switching behavior changes throughout the simulation. "
            "It displays the average number of dominance switches that occur during each phase "
            "(early, middle, and late) of the simulations, revealing when dominance is most volatile."
        )
        plt.figtext(
            0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9
        )

        # Adjust layout to make room for caption
        plt.tight_layout(rect=[0, 0.07, 1, 0.95])
        output_file = os.path.join(CURRENT_OUTPUT_PATH, "phase_switches.png")
        plt.savefig(output_file)
        plt.close()

    # 4. Transition matrix heatmap (average across all simulations)
    if all(
        f"{from_type}_to_{to_type}" in df.columns
        for from_type in agent_types
        for to_type in agent_types
    ):
        plt.figure(figsize=(10, 8))
        transition_data = np.zeros((3, 3))

        for i, from_type in enumerate(agent_types):
            for j, to_type in enumerate(agent_types):
                transition_data[i, j] = df[f"{from_type}_to_{to_type}"].mean()

        # Normalize rows to show probabilities
        row_sums = transition_data.sum(axis=1, keepdims=True)
        transition_probs = np.divide(
            transition_data,
            row_sums,
            out=np.zeros_like(transition_data),
            where=row_sums != 0,
        )

        sns.heatmap(
            transition_probs,
            annot=True,
            cmap="YlGnBu",
            fmt=".2f",
            xticklabels=agent_types,
            yticklabels=agent_types,
        )
        plt.title("Dominance Transition Probabilities")
        plt.xlabel("To Agent Type")
        plt.ylabel("From Agent Type")

        # Add caption for the fourth plot
        caption = (
            "This heatmap shows the probability of transitioning from one dominant agent type to another. "
            "Each cell represents the probability that when the row agent type loses dominance, "
            "it will be replaced by the column agent type. Higher values (darker colors) indicate "
            "more common transitions between those agent types."
        )
        plt.figtext(
            0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9
        )

        # Adjust layout to make room for caption
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        output_file = os.path.join(CURRENT_OUTPUT_PATH, "dominance_transitions.png")
        plt.savefig(output_file)
        plt.close()

    # 5. Plot dominance stability vs dominance score
    plot_dominance_stability(df)


def plot_dominance_stability(df):
    """
    Create a scatter plot showing the relationship between dominance stability
    (inverse of switches per step) and dominance score for different agent types.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    """
    global CURRENT_OUTPUT_PATH

    if df.empty or "switches_per_step" not in df.columns:
        logging.warning("No dominance stability data available for plotting")
        return

    plt.figure(figsize=(14, 8))

    # Calculate dominance stability as inverse of switches per step
    # Higher values mean more stable (fewer switches)
    df["dominance_stability"] = 1 / (
        df["switches_per_step"] + 0.01
    )  # Add small constant to avoid division by zero

    agent_types = ["system", "independent", "control"]
    colors = {"system": "blue", "independent": "orange", "control": "green"}

    # Plot each agent type
    for agent_type in agent_types:
        score_col = f"{agent_type}_dominance_score"

        if score_col in df.columns:
            plt.scatter(
                df["dominance_stability"],
                df[score_col],
                label=agent_type,
                color=colors[agent_type],
                alpha=0.6,
            )

    plt.title("Relationship Between Dominance Stability and Final Dominance Score")
    plt.xlabel("Dominance Stability (inverse of switches per step)")
    plt.ylabel("Dominance Score")
    plt.legend()

    # Add caption
    caption = (
        "This scatter plot illustrates the relationship between dominance stability and final dominance scores "
        "for each agent type. Dominance stability (x-axis) is calculated as the inverse of switches per step, "
        "where higher values indicate more stable dominance patterns with fewer changes in the dominant agent type. "
        "The y-axis shows the final dominance score for each agent type. This visualization helps identify "
        "whether more stable dominance patterns correlate with higher dominance scores for different agent types."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    output_file = os.path.join(CURRENT_OUTPUT_PATH, "dominance_stability_analysis.png")
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Saved dominance stability analysis to {output_file}")

    # Plot reproduction advantage vs dominance stability
    plot_reproduction_advantage_stability(df)


def plot_reproduction_advantage_stability(df):
    """
    Create a scatter plot showing the relationship between reproduction advantage
    and dominance stability with trend lines for different agent type comparisons.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results
    """
    global CURRENT_OUTPUT_PATH

    if df.empty or "switches_per_step" not in df.columns:
        logging.warning("No dominance stability data available for plotting")
        return

    # Check if reproduction advantage columns exist
    repro_advantage_cols = [
        col for col in df.columns if "reproduction_advantage" in col
    ]
    if not repro_advantage_cols:
        logging.warning("No reproduction advantage data available for plotting")
        return

    plt.figure(figsize=(14, 8))

    # Calculate dominance stability if not already calculated
    if "dominance_stability" not in df.columns:
        df["dominance_stability"] = 1 / (
            df["switches_per_step"] + 0.01
        )  # Add small constant to avoid division by zero

    # Define comparison pairs and their colors
    comparisons = [
        ("system", "independent", "blue"),
        ("system", "control", "orange"),
        ("independent", "control", "red"),
    ]

    # Plot each comparison
    for type1, type2, color in comparisons:
        advantage_col = f"{type1}_vs_{type2}_reproduction_advantage"
        reverse_advantage_col = f"{type2}_vs_{type1}_reproduction_advantage"

        if advantage_col in df.columns:
            plt.scatter(
                df[advantage_col],
                df["dominance_stability"],
                label=f"{type1} vs {type2}",
                color=color,
                alpha=0.6,
            )

            # Add trend line
            mask = ~df[advantage_col].isna()
            if mask.sum() > 1:  # Need at least 2 points for a line
                x = df.loc[mask, advantage_col]
                y = df.loc[mask, "dominance_stability"]
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_range = np.linspace(x.min(), x.max(), 100)
                plt.plot(x_range, p(x_range), f"--{color}")

        elif reverse_advantage_col in df.columns:
            # If we have the reverse comparison, use negative values
            plt.scatter(
                -df[reverse_advantage_col],
                df["dominance_stability"],
                label=f"{type1} vs {type2}",
                color=color,
                alpha=0.6,
            )

            # Add trend line
            mask = ~df[reverse_advantage_col].isna()
            if mask.sum() > 1:  # Need at least 2 points for a line
                x = -df.loc[mask, reverse_advantage_col]
                y = df.loc[mask, "dominance_stability"]
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_range = np.linspace(x.min(), x.max(), 100)
                plt.plot(x_range, p(x_range), f"--{color}")

    plt.title("Reproduction Advantage vs. Dominance Stability")
    plt.xlabel("Reproduction Advantage")
    plt.ylabel("Dominance Stability")
    plt.legend()

    # Add caption
    caption = (
        "This scatter plot shows the relationship between reproduction advantage and dominance stability. "
        "Reproduction advantage (x-axis) measures the difference in reproduction rates between agent types, "
        "where positive values indicate the first agent type has a reproductive advantage over the second. "
        "Dominance stability (y-axis) is calculated as the inverse of switches per step, where higher values "
        "indicate more stable dominance patterns with fewer changes. The dashed trend lines show the general "
        "relationship between reproductive advantage and stability for each agent type comparison. "
        "This visualization helps identify whether reproductive advantages correlate with more stable "
        "dominance patterns in the simulation."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9)

    # Adjust layout to make room for caption
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    output_file = os.path.join(
        CURRENT_OUTPUT_PATH, "reproduction_advantage_stability.png"
    )
    plt.savefig(output_file)
    logging.info(f"Saved reproduction advantage vs stability plot to {output_file}")
    plt.close()


def safe_remove_directory(directory_path, max_retries=3, retry_delay=1):
    """
    Safely remove a directory with retries.

    Parameters
    ----------
    directory_path : str
        Path to the directory to remove
    max_retries : int
        Maximum number of removal attempts
    retry_delay : float
        Delay in seconds between retries

    Returns
    -------
    bool
        True if directory was successfully removed, False otherwise
    """
    for attempt in range(max_retries):
        try:
            if os.path.exists(directory_path):
                shutil.rmtree(directory_path)
            return True
        except (PermissionError, OSError) as e:
            logging.warning(
                f"Attempt {attempt+1}/{max_retries} to remove directory failed: {e}"
            )
            if attempt < max_retries - 1:
                logging.info(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)

    return False


def analyze_dominance_switch_factors(df):
    """
    Analyze what factors correlate with dominance switching patterns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results

    Returns
    -------
    dict
        Dictionary with analysis results
    """
    if df.empty or "total_switches" not in df.columns:
        logging.warning("No dominance switch data available for analysis")
        return None

    results = {}

    # 1. Correlation between initial conditions and switching frequency
    initial_condition_cols = [
        col
        for col in df.columns
        if any(x in col for x in ["initial_", "resource_", "proximity"])
    ]

    if initial_condition_cols and len(df) > 5:
        # Calculate correlations with total switches
        corr_with_switches = (
            df[initial_condition_cols + ["total_switches"]]
            .corr()["total_switches"]
            .drop("total_switches")
        )

        # Get top positive and negative correlations
        top_positive = corr_with_switches.sort_values(ascending=False).head(5)
        top_negative = corr_with_switches.sort_values().head(5)

        results["top_positive_correlations"] = top_positive.to_dict()
        results["top_negative_correlations"] = top_negative.to_dict()

        logging.info("\nFactors associated with MORE dominance switching:")
        for factor, corr in top_positive.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                logging.info(f"  {factor}: {corr:.3f}")

        logging.info("\nFactors associated with LESS dominance switching:")
        for factor, corr in top_negative.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                logging.info(f"  {factor}: {corr:.3f}")

    # 2. Relationship between switching and final dominance
    if "comprehensive_dominance" in df.columns:
        # Average switches by dominant type
        switches_by_dominant = df.groupby("comprehensive_dominance")[
            "total_switches"
        ].mean()
        results["switches_by_dominant_type"] = switches_by_dominant.to_dict()

        logging.info("\nAverage dominance switches by final dominant type:")
        for agent_type, avg_switches in switches_by_dominant.items():
            logging.info(f"  {agent_type}: {avg_switches:.2f}")

    # 3. Relationship between switching and reproduction metrics
    reproduction_cols = [col for col in df.columns if "reproduction" in col]
    if reproduction_cols and len(df) > 5:
        # Calculate correlations with total switches
        repro_corr = (
            df[reproduction_cols + ["total_switches"]]
            .corr()["total_switches"]
            .drop("total_switches")
        )

        # Get top correlations (absolute value)
        top_repro_corr = repro_corr.abs().sort_values(ascending=False).head(5)
        top_repro_factors = repro_corr[top_repro_corr.index]

        results["reproduction_correlations"] = top_repro_factors.to_dict()

        logging.info("\nReproduction factors most associated with dominance switching:")
        for factor, corr in top_repro_factors.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                direction = "more" if corr > 0 else "fewer"
                logging.info(f"  {factor}: {corr:.3f} ({direction} switches)")

    # 4. Create a plot showing the relationship between switching and dominance stability
    plt.figure(figsize=(10, 6))

    # Calculate stability metric (inverse of switches per step)
    df["dominance_stability"] = 1 / (
        df["switches_per_step"] + 0.01
    )  # Add small constant to avoid division by zero

    # Plot relationship between stability and dominance score for each agent type
    for agent_type in ["system", "independent", "control"]:
        score_col = f"{agent_type}_dominance_score"
        if score_col in df.columns:
            plt.scatter(
                df["dominance_stability"], df[score_col], label=agent_type, alpha=0.7
            )

    plt.xlabel("Dominance Stability (inverse of switches per step)")
    plt.ylabel("Dominance Score")
    plt.title("Relationship Between Dominance Stability and Final Dominance Score")
    plt.legend()
    plt.tight_layout()

    output_file = os.path.join(CURRENT_OUTPUT_PATH, "dominance_stability_analysis.png")
    plt.savefig(output_file)
    plt.close()
    logging.info(f"Saved dominance stability analysis to {output_file}")

    return results


def analyze_reproduction_dominance_switching(df):
    """
    Analyze the relationship between reproduction strategies and dominance switching patterns.

    This function examines how different reproduction metrics correlate with dominance
    switching patterns, including:
    1. How reproduction success rates affect dominance stability
    2. How reproduction timing relates to dominance switches
    3. Differences in reproduction strategies between simulations with high vs. low switching
    4. How reproduction efficiency impacts dominance patterns

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with simulation analysis results

    Returns
    -------
    dict
        Dictionary with analysis results
    """
    if df.empty or "total_switches" not in df.columns:
        logging.warning(
            "No dominance switch data available for reproduction-switching analysis"
        )
        return None

    # Check if we have reproduction data
    reproduction_cols = [col for col in df.columns if "reproduction" in col]

    # Log all available columns for debugging
    logging.info(f"Available columns in dataframe: {', '.join(df.columns)}")

    if not reproduction_cols:
        logging.warning("No reproduction data columns found in the analysis dataframe")
        # Check if we have any agent type columns that might indicate reproduction data
        agent_type_cols = [
            col
            for col in df.columns
            if any(agent in col for agent in ["system", "independent", "control"])
        ]
        logging.info(f"Agent-related columns: {', '.join(agent_type_cols)}")
        return None

    logging.info(f"Found reproduction columns: {', '.join(reproduction_cols)}")

    # Filter to only include numeric reproduction columns
    numeric_repro_cols = []
    for col in reproduction_cols:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check if column has enough non-zero values
                non_zero_count = (df[col] != 0).sum()
                if non_zero_count > 5:  # Need at least 5 non-zero values for analysis
                    numeric_repro_cols.append(col)
                else:
                    logging.info(
                        f"Skipping column {col} with only {non_zero_count} non-zero values"
                    )
            else:
                logging.info(f"Skipping non-numeric reproduction column: {col}")

    if not numeric_repro_cols:
        logging.warning("No numeric reproduction data columns found")
        return None

    results = {}

    # 1. Divide simulations into high and low switching groups
    median_switches = df["total_switches"].median()
    high_switching = df[df["total_switches"] > median_switches]
    low_switching = df[df["total_switches"] <= median_switches]

    logging.info(
        f"Analyzing {len(high_switching)} high-switching and {len(low_switching)} low-switching simulations"
    )

    # Compare reproduction metrics between high and low switching groups
    repro_comparison = {}
    for col in numeric_repro_cols:
        try:
            high_mean = high_switching[col].mean()
            low_mean = low_switching[col].mean()
            difference = high_mean - low_mean
            percent_diff = (
                (difference / low_mean * 100) if low_mean != 0 else float("inf")
            )

            repro_comparison[col] = {
                "high_switching_mean": high_mean,
                "low_switching_mean": low_mean,
                "difference": difference,
                "percent_difference": percent_diff,
            }
        except Exception as e:
            logging.warning(f"Error processing column {col}: {e}")

    results["reproduction_high_vs_low_switching"] = repro_comparison

    # Log the most significant differences
    logging.info(
        "\nReproduction differences between high and low switching simulations:"
    )
    sorted_diffs = sorted(
        repro_comparison.items(),
        key=lambda x: abs(x[1]["percent_difference"]),
        reverse=True,
    )

    for col, stats in sorted_diffs[:5]:  # Top 5 differences
        if abs(stats["percent_difference"]) > 10:  # Only report meaningful differences
            direction = "higher" if stats["difference"] > 0 else "lower"
            logging.info(
                f"  {col}: {stats['high_switching_mean']:.3f} vs {stats['low_switching_mean']:.3f} "
                f"({abs(stats['percent_difference']):.1f}% {direction} in high-switching simulations)"
            )

    # 2. Analyze how first reproduction timing relates to dominance switching
    first_repro_cols = [
        col for col in numeric_repro_cols if "first_reproduction_time" in col
    ]
    if first_repro_cols:
        first_repro_corr = {}
        for col in first_repro_cols:
            try:
                # Filter out -1 values (no reproduction)
                valid_data = df[df[col] > 0]
                if len(valid_data) > 5:  # Need enough data points
                    corr = valid_data[[col, "total_switches"]].corr().iloc[0, 1]
                    first_repro_corr[col] = corr
                else:
                    logging.info(
                        f"Not enough valid data points for {col} correlation analysis"
                    )
            except Exception as e:
                logging.warning(f"Error calculating correlation for {col}: {e}")

        results["first_reproduction_timing_correlation"] = first_repro_corr

        logging.info(
            "\nCorrelation between first reproduction timing and dominance switches:"
        )
        for col, corr in first_repro_corr.items():
            agent_type = col.split("_first_reproduction")[0]
            if abs(corr) > 0.1:  # Only report meaningful correlations
                direction = "more" if corr > 0 else "fewer"
                logging.info(
                    f"  Earlier {agent_type} reproduction → {direction} dominance switches (r={corr:.3f})"
                )
    else:
        logging.info("No first reproduction timing data available for analysis")

    # 3. Create visualization showing reproduction success rate vs. dominance switching
    success_rate_cols = [
        col for col in numeric_repro_cols if "reproduction_success_rate" in col
    ]
    if success_rate_cols:
        plt.figure(figsize=(12, 8))

        for i, col in enumerate(success_rate_cols):
            try:
                agent_type = col.split("_reproduction")[0]

                # Filter out NaN values before plotting
                valid_data = df.dropna(subset=[col, "total_switches"])

                if len(valid_data) < 5:  # Skip if not enough valid data
                    logging.warning(
                        f"Not enough valid data points for {col} visualization"
                    )
                    continue

                plt.subplot(1, len(success_rate_cols), i + 1)

                # Create scatter plot with only valid data
                plt.scatter(
                    valid_data[col],
                    valid_data["total_switches"],
                    alpha=0.7,
                    label=agent_type,
                    c=f"C{i}",
                )

                # Add trend line - with robust error handling
                if len(valid_data) > 5:
                    try:
                        # Check if we have enough variation in the data
                        if valid_data[col].std() > 0.001:  # Need some variation
                            # Try polynomial fit with regularization
                            from sklearn.linear_model import Ridge
                            from sklearn.pipeline import make_pipeline
                            from sklearn.preprocessing import PolynomialFeatures

                            # Create a simple linear model with regularization
                            X = valid_data[col].values.reshape(-1, 1)
                            y = valid_data["total_switches"].values

                            # Make sure there are no NaN values
                            if np.isnan(X).any() or np.isnan(y).any():
                                logging.warning(
                                    f"Data for {col} still contains NaN values after filtering"
                                )
                                # Fallback to simple mean line
                                plt.axhline(
                                    y=valid_data["total_switches"].mean(),
                                    color=f"C{i}",
                                    linestyle="--",
                                    alpha=0.5,
                                )
                            else:
                                # Use Ridge regression which is more stable
                                model = make_pipeline(
                                    PolynomialFeatures(degree=1), Ridge(alpha=1.0)
                                )
                                model.fit(X, y)

                                # Generate prediction points
                                x_plot = np.linspace(
                                    valid_data[col].min(), valid_data[col].max(), 100
                                ).reshape(-1, 1)
                                y_plot = model.predict(x_plot)

                                # Plot the trend line
                                plt.plot(x_plot, y_plot, f"C{i}--", alpha=0.8)
                        else:
                            logging.info(
                                f"Not enough variation in {col} for trend line"
                            )
                            # Fallback to simple mean line
                            plt.axhline(
                                y=valid_data["total_switches"].mean(),
                                color=f"C{i}",
                                linestyle="--",
                                alpha=0.5,
                            )
                    except Exception as e:
                        logging.warning(f"Error creating trend line for {col}: {e}")
                        # Fallback to simple mean line if trend calculation fails
                        plt.axhline(
                            y=valid_data["total_switches"].mean(),
                            color=f"C{i}",
                            linestyle="--",
                            alpha=0.5,
                        )

                plt.xlabel(f"{agent_type.capitalize()} Reproduction Success Rate")
                plt.ylabel("Total Dominance Switches")
                plt.title(f"{agent_type.capitalize()} Reproduction vs. Switching")
            except Exception as e:
                logging.warning(f"Error creating plot for {col}: {e}")

        # Add caption
        caption = (
            "This multi-panel figure shows the relationship between reproduction success rates and dominance switching "
            "for different agent types. Each panel displays a scatter plot of reproduction success rate (x-axis) versus "
            "the total number of dominance switches (y-axis) for a specific agent type. The dashed trend lines indicate "
            "the general relationship between reproductive success and dominance stability. This visualization helps identify "
            "whether higher reproduction success correlates with more or fewer changes in dominance throughout the simulation."
        )
        plt.figtext(
            0.5, 0.01, caption, wrap=True, horizontalalignment="center", fontsize=9
        )

        # Adjust layout to make room for caption
        plt.tight_layout(rect=[0, 0.07, 1, 0.95])
        output_file = os.path.join(CURRENT_OUTPUT_PATH, "reproduction_vs_switching.png")
        plt.savefig(output_file)
        plt.close()
        logging.info(f"Saved reproduction vs. switching analysis to {output_file}")
    else:
        logging.info("No reproduction success rate data available for visualization")

    # 4. Analyze if reproduction efficiency correlates with dominance stability
    efficiency_cols = [
        col for col in numeric_repro_cols if "reproduction_efficiency" in col
    ]
    if efficiency_cols and "switches_per_step" in df.columns:
        # Calculate stability metric (inverse of switches per step)
        df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)

        efficiency_stability_corr = {}
        for col in efficiency_cols:
            try:
                # Filter out rows with NaN or zero values
                valid_data = df[(df[col].notna()) & (df[col] != 0)]
                if len(valid_data) > 5:  # Need enough data points
                    corr = valid_data[[col, "dominance_stability"]].corr().iloc[0, 1]
                    efficiency_stability_corr[col] = corr
                else:
                    logging.info(
                        f"Not enough valid data points for {col} correlation analysis"
                    )
            except Exception as e:
                logging.warning(f"Error calculating correlation for {col}: {e}")

        results["reproduction_efficiency_stability_correlation"] = (
            efficiency_stability_corr
        )

        logging.info(
            "\nCorrelation between reproduction efficiency and dominance stability:"
        )
        for col, corr in efficiency_stability_corr.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                agent_type = col.split("_reproduction")[0]
                direction = "more" if corr > 0 else "less"
                logging.info(
                    f"  Higher {agent_type} reproduction efficiency → {direction} stable dominance (r={corr:.3f})"
                )

    # 5. Analyze reproduction advantage and dominance switching
    advantage_cols = [
        col
        for col in numeric_repro_cols
        if "reproduction_rate_advantage" in col
        or "reproduction_efficiency_advantage" in col
    ]
    if advantage_cols and "switches_per_step" in df.columns:
        # Calculate stability metric (inverse of switches per step)
        if "dominance_stability" not in df.columns:
            df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)

        advantage_stability_corr = {}
        for col in advantage_cols:
            try:
                # Filter out rows with NaN values
                valid_data = df[df[col].notna()]
                if len(valid_data) > 5:  # Need enough data points
                    corr = valid_data[[col, "dominance_stability"]].corr().iloc[0, 1]
                    advantage_stability_corr[col] = corr
                else:
                    logging.info(
                        f"Not enough valid data points for {col} correlation analysis"
                    )
            except Exception as e:
                logging.warning(f"Error calculating correlation for {col}: {e}")

        results["reproduction_advantage_stability_correlation"] = (
            advantage_stability_corr
        )

        logging.info(
            "\nCorrelation between reproduction advantage and dominance stability:"
        )
        for col, corr in advantage_stability_corr.items():
            if abs(corr) > 0.1:  # Only report meaningful correlations
                if "_vs_" in col:
                    types = (
                        col.split("_vs_")[0],
                        col.split("_vs_")[1].split("_reproduction")[0],
                    )
                    direction = "more" if corr > 0 else "less"
                    logging.info(
                        f"  {types[0]} advantage over {types[1]} → {direction} stable dominance (r={corr:.3f})"
                    )

        # Create visualization of reproduction advantage vs. stability
        plt.figure(figsize=(10, 6))

        # Count how many valid columns we have for plotting
        valid_advantage_cols = []
        for col in advantage_cols:
            # Check if column has enough non-NaN values
            valid_count = df[col].notna().sum()
            if valid_count > 5 and "_vs_" in col:
                valid_advantage_cols.append(col)

        if valid_advantage_cols:
            for i, col in enumerate(valid_advantage_cols):
                try:
                    if "_vs_" in col:
                        types = (
                            col.split("_vs_")[0],
                            col.split("_vs_")[1].split("_reproduction")[0],
                        )
                        label = f"{types[0]} vs {types[1]}"

                        # Filter out NaN values
                        valid_data = df[
                            df[col].notna() & df["dominance_stability"].notna()
                        ]

                        if len(valid_data) < 5:  # Skip if not enough valid data
                            logging.warning(
                                f"Not enough valid data points for {col} visualization"
                            )
                            continue

                        plt.scatter(
                            valid_data[col],
                            valid_data["dominance_stability"],
                            alpha=0.7,
                            label=label,
                        )

                        # Add trend line - with robust error handling
                        if len(valid_data) > 5:
                            try:
                                # Check if we have enough variation in the data
                                if valid_data[col].std() > 0.001:  # Need some variation
                                    # Use robust regression
                                    from sklearn.linear_model import RANSACRegressor

                                    # Create a robust regression model
                                    X = valid_data[col].values.reshape(-1, 1)
                                    y = valid_data["dominance_stability"].values

                                    # Double-check for NaN values
                                    if np.isnan(X).any() or np.isnan(y).any():
                                        logging.warning(
                                            f"Data for {col} still contains NaN values after filtering"
                                        )
                                        # Fallback to horizontal line at mean
                                        plt.axhline(
                                            y=valid_data["dominance_stability"].mean(),
                                            linestyle="--",
                                            alpha=0.3,
                                        )
                                    else:
                                        # RANSAC is robust to outliers
                                        model = RANSACRegressor(random_state=42)
                                        model.fit(X, y)

                                        # Generate prediction points
                                        x_sorted = np.sort(X, axis=0)
                                        y_pred = model.predict(x_sorted)

                                        # Plot the trend line
                                        plt.plot(x_sorted, y_pred, "--", alpha=0.6)
                                else:
                                    logging.info(
                                        f"Not enough variation in {col} for trend line"
                                    )
                                    # Fallback to horizontal line at mean
                                    plt.axhline(
                                        y=valid_data["dominance_stability"].mean(),
                                        linestyle="--",
                                        alpha=0.3,
                                    )
                            except Exception as e:
                                logging.warning(
                                    f"Error creating trend line for {col}: {e}"
                                )
                                # Fallback to horizontal line at mean
                                plt.axhline(
                                    y=valid_data["dominance_stability"].mean(),
                                    linestyle="--",
                                    alpha=0.3,
                                )
                except Exception as e:
                    logging.warning(f"Error creating plot for {col}: {e}")

            plt.xlabel("Reproduction Advantage")
            plt.ylabel("Dominance Stability")
            plt.title("Reproduction Advantage vs. Dominance Stability")
            plt.legend()
            plt.tight_layout()

            output_file = os.path.join(
                CURRENT_OUTPUT_PATH, "reproduction_advantage_stability.png"
            )
            plt.savefig(output_file)
            plt.close()
            logging.info(
                f"Saved reproduction advantage vs. stability analysis to {output_file}"
            )
        else:
            logging.warning("No valid advantage columns for plotting")
            plt.close()

    # 6. Analyze relationship between reproduction metrics and dominance switching by agent type
    if "comprehensive_dominance" in df.columns:
        # Group by dominant agent type
        for agent_type in ["system", "independent", "control"]:
            type_data = df[df["comprehensive_dominance"] == agent_type]
            if len(type_data) > 5:  # Need enough data points
                type_results = {}

                # Calculate correlations between reproduction metrics and switching for this agent type
                for col in numeric_repro_cols:
                    try:
                        # Filter out NaN values
                        valid_data = type_data[
                            type_data[col].notna() & type_data["total_switches"].notna()
                        ]
                        if len(valid_data) > 5:  # Need enough data points
                            corr = valid_data[[col, "total_switches"]].corr().iloc[0, 1]
                            if not np.isnan(corr):
                                type_results[col] = corr
                        else:
                            logging.info(
                                f"Not enough valid data points for {col} in {agent_type}-dominant simulations"
                            )
                    except Exception as e:
                        logging.warning(
                            f"Error calculating correlation for {col} in {agent_type}-dominant simulations: {e}"
                        )

                # Add to results
                results[f"{agent_type}_dominance_reproduction_correlations"] = (
                    type_results
                )

                # Log top correlations
                if type_results:
                    logging.info(
                        f"\nTop reproduction factors affecting switching in {agent_type}-dominant simulations:"
                    )
                    sorted_corrs = sorted(
                        type_results.items(), key=lambda x: abs(x[1]), reverse=True
                    )
                    for col, corr in sorted_corrs[:3]:  # Top 3
                        if abs(corr) > 0.2:  # Only report stronger correlations
                            direction = "more" if corr > 0 else "fewer"
                            logging.info(f"  {col}: {corr:.3f} ({direction} switches)")

    return results


def check_db_schema(engine, table_name):
    """
    Check the schema of a specific table in the database.

    Parameters
    ----------
    engine : sqlalchemy.engine.Engine
        SQLAlchemy engine connected to the database
    table_name : str
        Name of the table to check

    Returns
    -------
    dict
        Dictionary with table schema information
    """
    try:
        inspector = sqlalchemy.inspect(engine)

        if table_name not in inspector.get_table_names():
            return {"exists": False}

        columns = inspector.get_columns(table_name)
        column_names = [col["name"] for col in columns]

        # Get primary key
        pk = inspector.get_pk_constraint(table_name)

        # Get foreign keys
        fks = inspector.get_foreign_keys(table_name)

        # Get indexes
        indexes = inspector.get_indexes(table_name)

        return {
            "exists": True,
            "column_count": len(columns),
            "columns": column_names,
            "primary_key": pk,
            "foreign_keys": fks,
            "indexes": indexes,
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}


def check_reproduction_events(experiment_path):
    """
    Check if reproduction events exist in the simulation databases.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment folder containing simulation databases

    Returns
    -------
    bool
        True if any reproduction events are found, False otherwise
    """
    logging.info("Checking for reproduction events in databases...")

    # Find all simulation folders
    sim_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))
    total_events = 0
    checked_dbs = 0

    for folder in sim_folders[:5]:  # Check first 5 databases
        # Check if this is a simulation folder with a database
        db_path = os.path.join(folder, "simulation.db")

        if not os.path.exists(db_path):
            continue

        try:
            # Connect to the database
            engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
            Session = sessionmaker(bind=engine)
            session = Session()

            # Check if ReproductionEventModel table exists
            inspector = sqlalchemy.inspect(engine)
            if "reproduction_events" not in inspector.get_table_names():
                logging.warning(
                    f"Database {db_path} does not have a reproduction_events table"
                )

                # Check what tables do exist
                tables = inspector.get_table_names()
                logging.info(f"Available tables: {', '.join(tables)}")

                session.close()
                continue

            # Check the schema of the reproduction_events table
            schema_info = check_db_schema(engine, "reproduction_events")
            if schema_info["exists"]:
                logging.info(
                    f"reproduction_events table schema: {len(schema_info['columns'])} columns"
                )
                required_columns = [
                    "event_id",
                    "step_number",
                    "parent_id",
                    "offspring_id",
                    "success",
                    "parent_resources_before",
                    "parent_resources_after",
                ]
                missing_columns = [
                    col for col in required_columns if col not in schema_info["columns"]
                ]
                if missing_columns:
                    logging.warning(
                        f"Missing required columns in reproduction_events: {', '.join(missing_columns)}"
                    )

            # Count reproduction events
            event_count = session.query(ReproductionEventModel).count()
            total_events += event_count
            checked_dbs += 1

            logging.info(
                f"Database {os.path.basename(folder)}: {event_count} reproduction events"
            )

            # If we have events, check a sample
            if event_count > 0:
                sample_event = session.query(ReproductionEventModel).first()
                logging.info(
                    f"Sample event: parent_id={sample_event.parent_id}, success={sample_event.success}"
                )

                # Check if we have resource data
                has_resource_data = (
                    hasattr(sample_event, "parent_resources_before")
                    and sample_event.parent_resources_before is not None
                )
                if not has_resource_data:
                    logging.warning("Sample event is missing resource data")

            # Close the session
            session.close()

        except Exception as e:
            logging.error(f"Error checking reproduction events in {folder}: {e}")
            import traceback

            logging.error(traceback.format_exc())

    if checked_dbs > 0:
        logging.info(
            f"Found {total_events} total reproduction events in {checked_dbs} checked databases"
        )
        return total_events > 0
    else:
        logging.warning("Could not check any databases for reproduction events")
        return False


def main():
    global CURRENT_OUTPUT_PATH

    # Create dominance output directory
    dominance_output_path = os.path.join(OUTPUT_PATH, "dominance")

    # Clear the dominance directory if it exists
    if os.path.exists(dominance_output_path):
        logging.info(f"Clearing existing dominance directory: {dominance_output_path}")
        if not safe_remove_directory(dominance_output_path):
            # If we couldn't remove the directory after retries, create a new one with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dominance_output_path = os.path.join(OUTPUT_PATH, f"dominance_{timestamp}")
            logging.info(f"Using alternative directory: {dominance_output_path}")

    # Create the directory
    os.makedirs(dominance_output_path, exist_ok=True)

    # Set the global output path
    CURRENT_OUTPUT_PATH = dominance_output_path

    # Set up logging to the dominance directory
    log_file = setup_logging(dominance_output_path)

    logging.info(f"Saving results to {dominance_output_path}")

    # Find the most recent experiment folder in DATA_PATH
    experiment_folders = glob.glob(os.path.join(DATA_PATH, "*"))
    if not experiment_folders:
        logging.error(f"No experiment folders found in {DATA_PATH}")
        return

    # Sort by modification time (most recent first)
    experiment_folders.sort(key=os.path.getmtime, reverse=True)
    experiment_path = experiment_folders[0]

    # Check if reproduction events exist in the databases
    has_reproduction_events = check_reproduction_events(experiment_path)
    if not has_reproduction_events:
        logging.warning(
            "No reproduction events found in databases. Reproduction analysis may be limited."
        )

    logging.info(f"Analyzing simulations in {experiment_path}...")
    df = analyze_simulations(experiment_path)

    if df.empty:
        logging.warning("No simulation data found.")
        return

    # Save the raw data
    output_csv = os.path.join(dominance_output_path, "simulation_analysis.csv")
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved analysis data to {output_csv}")

    # Check if we have reproduction data in the dataframe
    reproduction_cols = [col for col in df.columns if "reproduction" in col]
    if reproduction_cols:
        logging.info(
            f"Found {len(reproduction_cols)} reproduction-related columns in analysis data"
        )
        for col in reproduction_cols[:10]:  # Show first 10
            non_null = df[col].count()
            logging.info(f"  {col}: {non_null} non-null values")
    else:
        logging.warning("No reproduction columns found in analysis data")

    # Continue with the rest of the analysis
    logging.info(f"Analyzed {len(df)} simulations.")
    logging.info("\nSummary statistics:")
    logging.info(df.describe().to_string())

    # Calculate and display dominance distributions as percentages
    logging.info("\nDominance distribution (percentages):")

    # Population dominance
    pop_counts = df["population_dominance"].value_counts()
    pop_percentages = (pop_counts / pop_counts.sum() * 100).round(1)
    logging.info("Population dominance:")
    for agent_type, percentage in pop_percentages.items():
        logging.info(f"  {agent_type}: {percentage}%")

    # Survival dominance
    surv_counts = df["survival_dominance"].value_counts()
    surv_percentages = (surv_counts / surv_counts.sum() * 100).round(1)
    logging.info("Survival dominance:")
    for agent_type, percentage in surv_percentages.items():
        logging.info(f"  {agent_type}: {percentage}%")

    # Comprehensive dominance
    if "comprehensive_dominance" in df.columns:
        comp_counts = df["comprehensive_dominance"].value_counts()
        comp_percentages = (comp_counts / comp_counts.sum() * 100).round(1)
        logging.info("Comprehensive dominance:")
        for agent_type, percentage in comp_percentages.items():
            logging.info(f"  {agent_type}: {percentage}%")

    # Dominance switching statistics
    if "total_switches" in df.columns:
        logging.info("\nDominance switching statistics:")
        logging.info(
            f"Average switches per simulation: {df['total_switches'].mean():.2f}"
        )
        logging.info(f"Average switches per step: {df['switches_per_step'].mean():.4f}")

        # Average dominance period by agent type
        logging.info("\nAverage dominance period duration (steps):")
        for agent_type in ["system", "independent", "control"]:
            avg_period = df[f"{agent_type}_avg_dominance_period"].mean()
            logging.info(f"  {agent_type}: {avg_period:.2f}")

        # Phase-specific switching
        if "early_phase_switches" in df.columns:
            logging.info("\nAverage switches by simulation phase:")
            for phase in ["early", "middle", "late"]:
                avg_switches = df[f"{phase}_phase_switches"].mean()
                logging.info(f"  {phase}: {avg_switches:.2f}")

        # Transition probabilities
        if all(
            f"{from_type}_to_{to_type}" in df.columns
            for from_type in ["system", "independent", "control"]
            for to_type in ["system", "independent", "control"]
        ):
            logging.info("\nDominance transition probabilities:")
            for from_type in ["system", "independent", "control"]:
                logging.info(f"  From {from_type}:")
                for to_type in ["system", "independent", "control"]:
                    if from_type != to_type:  # Skip self-transitions
                        prob = df[f"{from_type}_to_{to_type}"].mean()
                        logging.info(f"    To {to_type}: {prob:.2f}")

    # Plot dominance distribution
    plot_dominance_distribution(df)

    # Plot dominance switching patterns
    if "total_switches" in df.columns:
        plot_dominance_switches(df)

        # Analyze factors related to dominance switching
        analyze_dominance_switch_factors(df)

        # Analyze relationship between reproduction and dominance switching
        analyze_reproduction_dominance_switching(df)

    # Plot resource proximity vs dominance
    plot_resource_proximity_vs_dominance(df)

    # Plot reproduction metrics vs dominance
    plot_reproduction_vs_dominance(df)

    # Plot correlation matrices
    for label in ["population_dominance", "survival_dominance"]:
        if label in df.columns and df[label].nunique() > 1:
            plot_correlation_matrix(df, label)

    # Plot comparison of different dominance measures
    plot_dominance_comparison(df)

    # Prepare features for classification
    # Exclude non-feature columns and outcome variables
    exclude_cols = [
        "iteration",
        "population_dominance",
        "survival_dominance",
        "system_agents",
        "independent_agents",
        "control_agents",
        "total_agents",
        "final_step",
    ]

    # Also exclude derived statistics columns that are outcomes, not predictors
    for prefix in ["system_", "independent_", "control_"]:
        for suffix in ["count", "alive", "dead", "avg_survival", "dead_ratio"]:
            exclude_cols.append(f"{prefix}{suffix}")

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Check if we have enough data for classification
    if len(df) > 10 and len(feature_cols) > 0:
        # Create an explicit copy to avoid SettingWithCopyWarning
        X = df[feature_cols].copy()

        # Handle missing values - separate numeric and non-numeric columns
        numeric_cols = X.select_dtypes(include=["number"]).columns
        categorical_cols = X.select_dtypes(exclude=["number"]).columns

        # Fill numeric columns with mean
        if not numeric_cols.empty:
            for col in numeric_cols:
                X.loc[:, col] = X[col].fillna(X[col].mean())

        # Fill categorical columns with mode (most frequent value)
        if not categorical_cols.empty:
            for col in categorical_cols:
                X.loc[:, col] = X[col].fillna(
                    X[col].mode()[0] if not X[col].mode().empty else "unknown"
                )

        # Train classifiers for each dominance type
        for label in ["population_dominance", "survival_dominance"]:
            if df[label].nunique() > 1:  # Only if we have multiple classes
                logging.info(f"\nTraining classifier for {label}...")
                y = df[label]
                clf, feat_imp = train_classifier(X, y, label)

                # Plot feature importance
                plot_feature_importance(feat_imp, label)

    logging.info("\nAnalysis complete. Results saved to CSV and PNG files.")
    logging.info(f"Log file saved to: {log_file}")
    logging.info(f"All analysis files saved to: {dominance_output_path}")


if __name__ == "__main__":
    main()

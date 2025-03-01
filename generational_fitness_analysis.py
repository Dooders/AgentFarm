"""Generational Fitness Analysis Module.

This module analyzes the fitness and adaptation of agents across generations,
comparing metrics between earlier and later generations to measure evolutionary progress.

Key metrics analyzed:
- Resource acquisition efficiency
- Survival rates and resilience
- Action effectiveness
- Reward optimization
- Reproductive success

The module provides functions to:
1. Extract generational performance data from simulation databases
2. Compare first generation to later generations
3. Visualize fitness trends across generations
4. Generate statistical analysis of generational improvements
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import and_, case, desc, func
from sqlalchemy.orm import Session

from farm.database.database import SimulationDatabase
from farm.database.models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    LearningExperienceModel,
    ReproductionEventModel,
    SimulationStepModel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants for file paths
DEFAULT_DATA_PATH = "results/one_of_a_kind_v1/experiments/data"
DEFAULT_ANALYSIS_PATH = "results/one_of_a_kind_v1/experiments/analysis"


def get_agent_metrics_by_generation(
    experiment_db_path: str,
) -> Dict[int, Dict[str, float]]:
    """
    Extract performance metrics for agents grouped by generation.

    Args:
        experiment_db_path: Path to the simulation database

    Returns:
        Dictionary mapping generation numbers to their performance metrics
    """
    if not os.path.exists(experiment_db_path):
        logger.error(f"Database file not found: {experiment_db_path}")
        return {}

    db = None
    session = None
    try:
        db = SimulationDatabase(experiment_db_path)
        session = db.Session()

        # Get all generations
        generations = [
            gen[0]
            for gen in session.query(AgentModel.generation)
            .distinct()
            .order_by(AgentModel.generation)
            .all()
        ]

        metrics_by_generation = {}

        for gen in generations:
            # Get agents of this generation
            agents = (
                session.query(AgentModel).filter(AgentModel.generation == gen).all()
            )
            agent_ids = [agent.agent_id for agent in agents]

            if not agent_ids:
                continue

            # Calculate survival time (for agents that have died)
            survival_query = (
                session.query(func.avg(AgentModel.death_time - AgentModel.birth_time))
                .filter(AgentModel.generation == gen, AgentModel.death_time > 0)
                .scalar()
            )

            avg_survival_time = (
                float(survival_query) if survival_query is not None else 0
            )

            # Calculate resource acquisition metrics
            resource_metrics = (
                session.query(
                    func.avg(AgentStateModel.resource_level).label("avg_resources"),
                    func.max(AgentStateModel.resource_level).label("max_resources"),
                )
                .filter(AgentStateModel.agent_id.in_(agent_ids))
                .first()
            )

            # Calculate action effectiveness
            action_metrics = (
                session.query(
                    func.count(ActionModel.action_id).label("total_actions"),
                    func.avg(ActionModel.reward).label("avg_reward"),
                )
                .filter(ActionModel.agent_id.in_(agent_ids))
                .first()
            )

            # Calculate reproduction success rate
            reproduction_metrics = (
                session.query(
                    func.count(ReproductionEventModel.event_id).label("total_attempts"),
                    func.sum(
                        case((ReproductionEventModel.success == True, 1), else_=0)
                    ).label("successful_attempts"),
                )
                .filter(ReproductionEventModel.parent_id.in_(agent_ids))
                .first()
            )

            # Store metrics for this generation
            metrics_by_generation[gen] = {
                "avg_survival_time": avg_survival_time,
                "avg_resources": (
                    float(resource_metrics.avg_resources)
                    if resource_metrics.avg_resources is not None
                    else 0
                ),
                "max_resources": (
                    float(resource_metrics.max_resources)
                    if resource_metrics.max_resources is not None
                    else 0
                ),
                "total_actions": (
                    int(action_metrics.total_actions)
                    if action_metrics.total_actions is not None
                    else 0
                ),
                "avg_reward": (
                    float(action_metrics.avg_reward)
                    if action_metrics.avg_reward is not None
                    else 0
                ),
                "reproduction_attempts": (
                    int(reproduction_metrics.total_attempts)
                    if reproduction_metrics.total_attempts is not None
                    else 0
                ),
                "reproduction_success": (
                    int(reproduction_metrics.successful_attempts)
                    if reproduction_metrics.successful_attempts is not None
                    else 0
                ),
                "reproduction_rate": (
                    (
                        float(reproduction_metrics.successful_attempts)
                        / float(reproduction_metrics.total_attempts)
                    )
                    if reproduction_metrics.total_attempts
                    and reproduction_metrics.total_attempts > 0
                    else 0
                ),
                "agent_count": len(agent_ids),
            }

        return metrics_by_generation

    except Exception as e:
        logger.error(f"Error analyzing generational metrics: {str(e)}")
        return {}
    finally:
        if session:
            session.close()
        if db:
            db.close()


def get_resource_efficiency_by_generation(
    experiment_db_path: str,
) -> Dict[int, Dict[str, float]]:
    """
    Calculate resource acquisition and usage efficiency metrics by generation.

    Args:
        experiment_db_path: Path to the simulation database

    Returns:
        Dictionary mapping generation numbers to resource efficiency metrics
    """
    if not os.path.exists(experiment_db_path):
        logger.error(f"Database file not found: {experiment_db_path}")
        return {}

    db = None
    session = None
    try:
        db = SimulationDatabase(experiment_db_path)
        session = db.Session()

        # Get all generations
        generations = [
            gen[0]
            for gen in session.query(AgentModel.generation)
            .distinct()
            .order_by(AgentModel.generation)
            .all()
        ]

        efficiency_by_generation = {}

        for gen in generations:
            # Get agents of this generation
            agents = (
                session.query(AgentModel).filter(AgentModel.generation == gen).all()
            )
            agent_ids = [agent.agent_id for agent in agents]

            if not agent_ids:
                continue

            # Get resource-related actions (e.g., forage, consume)
            resource_actions = (
                session.query(
                    ActionModel.action_type,
                    func.avg(
                        ActionModel.resources_after - ActionModel.resources_before
                    ).label("avg_resource_change"),
                    func.count(ActionModel.action_id).label("action_count"),
                )
                .filter(
                    ActionModel.agent_id.in_(agent_ids),
                    ActionModel.action_type.in_(["forage", "consume", "share"]),
                )
                .group_by(ActionModel.action_type)
                .all()
            )

            # Calculate resource efficiency metrics
            efficiency_metrics = {}
            for action_type, avg_change, count in resource_actions:
                efficiency_metrics[f"{action_type}_efficiency"] = (
                    float(avg_change) if avg_change is not None else 0
                )
                efficiency_metrics[f"{action_type}_count"] = (
                    int(count) if count is not None else 0
                )

            # Get average resource level
            avg_resources = (
                session.query(func.avg(AgentStateModel.resource_level))
                .filter(AgentStateModel.agent_id.in_(agent_ids))
                .scalar()
            )

            efficiency_metrics["avg_resource_level"] = (
                float(avg_resources) if avg_resources is not None else 0
            )

            # Store metrics for this generation
            efficiency_by_generation[gen] = efficiency_metrics

        return efficiency_by_generation

    except Exception as e:
        logger.error(f"Error analyzing resource efficiency: {str(e)}")
        return {}
    finally:
        if session:
            session.close()
        if db:
            db.close()


def get_action_effectiveness_by_generation(
    experiment_db_path: str,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """
    Analyze the effectiveness of different actions across generations.

    Args:
        experiment_db_path: Path to the simulation database

    Returns:
        Dictionary mapping generation numbers to action effectiveness metrics
    """
    if not os.path.exists(experiment_db_path):
        logger.error(f"Database file not found: {experiment_db_path}")
        return {}

    db = None
    session = None
    try:
        db = SimulationDatabase(experiment_db_path)
        session = db.Session()

        # Get all generations
        generations = [
            gen[0]
            for gen in session.query(AgentModel.generation)
            .distinct()
            .order_by(AgentModel.generation)
            .all()
        ]

        action_metrics_by_generation = {}

        for gen in generations:
            # Get agents of this generation
            agents = (
                session.query(AgentModel).filter(AgentModel.generation == gen).all()
            )
            agent_ids = [agent.agent_id for agent in agents]

            if not agent_ids:
                continue

            # Get metrics for each action type
            action_metrics = (
                session.query(
                    ActionModel.action_type,
                    func.avg(ActionModel.reward).label("avg_reward"),
                    func.count(ActionModel.action_id).label("count"),
                    func.avg(
                        ActionModel.resources_after - ActionModel.resources_before
                    ).label("avg_resource_change"),
                )
                .filter(ActionModel.agent_id.in_(agent_ids))
                .group_by(ActionModel.action_type)
                .all()
            )

            # Store metrics for this generation
            gen_metrics = {}
            for action_type, avg_reward, count, avg_resource_change in action_metrics:
                gen_metrics[action_type] = {
                    "avg_reward": float(avg_reward) if avg_reward is not None else 0,
                    "count": int(count) if count is not None else 0,
                    "avg_resource_change": (
                        float(avg_resource_change)
                        if avg_resource_change is not None
                        else 0
                    ),
                    "frequency": (
                        int(count) / sum(m[2] for m in action_metrics)
                        if count is not None
                        else 0
                    ),
                }

            action_metrics_by_generation[gen] = gen_metrics

        return action_metrics_by_generation

    except Exception as e:
        logger.error(f"Error analyzing action effectiveness: {str(e)}")
        return {}
    finally:
        if session:
            session.close()
        if db:
            db.close()


def compare_generations(
    metrics_by_generation: Dict[int, Dict[str, float]],
    first_gen: int = 0,
    last_n: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Compare metrics between first generation and the last N generations.

    Args:
        metrics_by_generation: Dictionary of metrics by generation
        first_gen: Generation number to use as baseline (default: 0)
        last_n: Number of latest generations to compare against (default: 3)

    Returns:
        Dictionary with comparison results
    """
    if not metrics_by_generation or first_gen not in metrics_by_generation:
        return {}

    # Get the baseline metrics
    baseline = metrics_by_generation[first_gen]

    # Get the latest generations
    all_gens = sorted(metrics_by_generation.keys())
    latest_gens = all_gens[-last_n:] if len(all_gens) >= last_n else all_gens[1:]

    # Remove the first generation if it's in the latest
    if first_gen in latest_gens:
        latest_gens.remove(first_gen)

    if not latest_gens:
        return {}

    # Calculate average metrics for latest generations
    latest_metrics = {}
    for metric in baseline.keys():
        values = [
            metrics_by_generation[gen][metric]
            for gen in latest_gens
            if metric in metrics_by_generation[gen]
        ]
        if values:
            latest_metrics[metric] = sum(values) / len(values)
        else:
            latest_metrics[metric] = 0

    # Calculate differences and percent changes
    comparison = {
        "baseline": baseline,
        "latest_avg": latest_metrics,
        "absolute_diff": {},
        "percent_change": {},
    }

    for metric in baseline.keys():
        if metric in latest_metrics:
            comparison["absolute_diff"][metric] = (
                latest_metrics[metric] - baseline[metric]
            )
            if baseline[metric] != 0:
                comparison["percent_change"][metric] = (
                    (latest_metrics[metric] - baseline[metric]) / baseline[metric] * 100
                )
            else:
                comparison["percent_change"][metric] = (
                    float("inf") if latest_metrics[metric] > 0 else 0
                )

    return comparison


def plot_generational_trends(
    metrics_by_generation: Dict[int, Dict[str, float]],
    metrics_to_plot: List[str],
    title: str,
    output_path: str,
):
    """
    Plot trends of selected metrics across generations.

    Args:
        metrics_by_generation: Dictionary of metrics by generation
        metrics_to_plot: List of metric names to include in the plot
        title: Plot title
        output_path: Path to save the plot
    """
    if not metrics_by_generation:
        logger.warning("No metrics data to plot")
        return

    generations = sorted(metrics_by_generation.keys())

    plt.figure(figsize=(12, 8))

    for metric in metrics_to_plot:
        values = [metrics_by_generation[gen].get(metric, 0) for gen in generations]
        plt.plot(generations, values, marker="o", label=metric)

    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Plot saved to {output_path}")


def generate_fitness_report(comparison: Dict[str, Dict[str, float]], output_path: str):
    """
    Generate a text report of generational fitness comparison.

    Args:
        comparison: Dictionary with comparison results
        output_path: Path to save the report
    """
    if not comparison:
        logger.warning("No comparison data for report")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("Generational Fitness Analysis Report\n")
        f.write("===================================\n\n")

        f.write("Comparing first generation to latest generations\n\n")

        # Write metric comparisons
        f.write("Performance Metrics:\n")
        f.write("-----------------\n")

        for metric in comparison["baseline"].keys():
            baseline = comparison["baseline"].get(metric, 0)
            latest = comparison["latest_avg"].get(metric, 0)
            diff = comparison["absolute_diff"].get(metric, 0)
            pct_change = comparison["percent_change"].get(metric, 0)

            f.write(f"{metric}:\n")
            f.write(f"  First generation: {baseline:.4f}\n")
            f.write(f"  Latest generations (avg): {latest:.4f}\n")
            f.write(f"  Absolute difference: {diff:.4f}\n")

            if pct_change == float("inf"):
                f.write(f"  Percent change: Infinite (from zero)\n")
            else:
                f.write(f"  Percent change: {pct_change:.2f}%\n")

            # Add interpretation
            if pct_change > 10:
                f.write(f"  Interpretation: Significant improvement\n")
            elif pct_change > 0:
                f.write(f"  Interpretation: Slight improvement\n")
            elif pct_change < -10:
                f.write(f"  Interpretation: Significant decline\n")
            elif pct_change < 0:
                f.write(f"  Interpretation: Slight decline\n")
            else:
                f.write(f"  Interpretation: No change\n")

            f.write("\n")

        # Overall assessment
        f.write("\nOverall Fitness Assessment:\n")
        f.write("-------------------------\n")

        # Count improvements and declines
        improvements = sum(1 for v in comparison["percent_change"].values() if v > 0)
        declines = sum(1 for v in comparison["percent_change"].values() if v < 0)
        total = len(comparison["percent_change"])

        if improvements > declines:
            f.write(
                f"Later generations show improvement in {improvements}/{total} metrics ({improvements/total*100:.1f}%).\n"
            )
            f.write(
                "Overall, later generations demonstrate better fitness and adaptation.\n"
            )
        elif declines > improvements:
            f.write(
                f"Later generations show decline in {declines}/{total} metrics ({declines/total*100:.1f}%).\n"
            )
            f.write(
                "Overall, later generations demonstrate reduced fitness compared to the first generation.\n"
            )
        else:
            f.write(
                "Later generations show mixed results with no clear trend in fitness.\n"
            )

    logger.info(f"Fitness report saved to {output_path}")


def analyze_generational_fitness(experiment_db_path: str, output_dir: str):
    """
    Perform comprehensive generational fitness analysis on a simulation database.

    Args:
        experiment_db_path: Path to the simulation database
        output_dir: Directory to save analysis outputs
    """
    logger.info(f"Analyzing generational fitness for {experiment_db_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract metrics by generation
    metrics_by_generation = get_agent_metrics_by_generation(experiment_db_path)
    if not metrics_by_generation:
        logger.error("Failed to extract generational metrics")
        return

    # Get resource efficiency metrics
    resource_efficiency = get_resource_efficiency_by_generation(experiment_db_path)

    # Get action effectiveness metrics
    action_effectiveness = get_action_effectiveness_by_generation(experiment_db_path)

    # Compare first generation to latest generations
    first_gen = min(metrics_by_generation.keys())
    comparison = compare_generations(metrics_by_generation, first_gen=first_gen)

    # Plot survival metrics
    survival_metrics = ["avg_survival_time", "reproduction_rate"]
    plot_generational_trends(
        metrics_by_generation,
        survival_metrics,
        "Survival Metrics Across Generations",
        os.path.join(output_dir, "survival_trends.png"),
    )

    # Plot resource metrics
    resource_metrics = ["avg_resources", "max_resources"]
    plot_generational_trends(
        metrics_by_generation,
        resource_metrics,
        "Resource Acquisition Across Generations",
        os.path.join(output_dir, "resource_trends.png"),
    )

    # Plot reward metrics
    reward_metrics = ["avg_reward"]
    plot_generational_trends(
        metrics_by_generation,
        reward_metrics,
        "Reward Optimization Across Generations",
        os.path.join(output_dir, "reward_trends.png"),
    )

    # Generate fitness report
    generate_fitness_report(
        comparison, os.path.join(output_dir, "generational_fitness_report.txt")
    )

    # Plot action distribution changes
    if action_effectiveness:
        plot_action_distribution_evolution(
            action_effectiveness, os.path.join(output_dir, "action_evolution.png")
        )

    logger.info(
        f"Generational fitness analysis completed. Results saved to {output_dir}"
    )


def plot_action_distribution_evolution(
    action_metrics: Dict[int, Dict[str, Dict[str, float]]], output_path: str
):
    """
    Plot how action distribution evolves across generations.

    Args:
        action_metrics: Dictionary of action metrics by generation
        output_path: Path to save the plot
    """
    if not action_metrics:
        logger.warning("No action metrics data to plot")
        return

    generations = sorted(action_metrics.keys())

    # Get all unique action types
    action_types = set()
    for gen_metrics in action_metrics.values():
        action_types.update(gen_metrics.keys())

    # Create a DataFrame for plotting
    data = []
    for gen in generations:
        for action in action_types:
            if action in action_metrics[gen]:
                metrics = action_metrics[gen][action]
                data.append(
                    {
                        "Generation": gen,
                        "Action": action,
                        "Frequency": metrics.get("frequency", 0),
                        "Reward": metrics.get("avg_reward", 0),
                    }
                )

    df = pd.DataFrame(data)

    # Plot action frequency evolution
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    for action in action_types:
        action_data = df[df["Action"] == action]
        if not action_data.empty:
            plt.plot(
                action_data["Generation"],
                action_data["Frequency"],
                marker="o",
                label=action,
            )

    plt.title("Action Frequency Evolution Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Action Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot action reward evolution
    plt.subplot(2, 1, 2)
    for action in action_types:
        action_data = df[df["Action"] == action]
        if not action_data.empty:
            plt.plot(
                action_data["Generation"],
                action_data["Reward"],
                marker="o",
                label=action,
            )

    plt.title("Action Reward Evolution Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Action evolution plot saved to {output_path}")


def process_experiment_generational_fitness(
    experiment: str,
    data_path: str = DEFAULT_DATA_PATH,
    analysis_path: str = DEFAULT_ANALYSIS_PATH,
) -> Dict[str, Any]:
    """
    Process generational fitness analysis for an experiment.

    Args:
        experiment: Name of the experiment
        data_path: Base path for experiment data (default: DEFAULT_DATA_PATH)
        analysis_path: Base path for analysis output (default: DEFAULT_ANALYSIS_PATH)

    Returns:
        Dictionary with analysis results
    """
    logger.info(f"Processing generational fitness for experiment: {experiment}")

    result = {
        "experiment": experiment,
        "metrics_by_generation": {},
        "resource_efficiency": {},
        "action_effectiveness": {},
        "comparison": {},
    }

    try:
        # Find all database files for this experiment
        experiment_path = os.path.join(data_path, experiment)
        if not os.path.exists(experiment_path):
            logger.error(f"Experiment directory not found: {experiment_path}")
            return result

        # Find simulation databases
        db_paths = []
        for root, _, files in os.walk(experiment_path):
            for file in files:
                if file.endswith(".db"):
                    db_paths.append(os.path.join(root, file))

        if not db_paths:
            logger.error(f"No database files found in {experiment_path}")
            return result

        # Create output directory for the experiment-wide analysis
        output_dir = os.path.join(analysis_path, experiment, "generational_fitness")
        os.makedirs(output_dir, exist_ok=True)

        # Preload all data from databases
        logger.info(
            f"Preloading data from {len(db_paths)} databases for experiment {experiment}"
        )
        all_metrics = {}
        all_resource_efficiency = {}
        all_action_effectiveness = {}

        for db_path in db_paths:
            db_name = os.path.basename(db_path).replace(".db", "")

            # Extract metrics for this database
            metrics = get_agent_metrics_by_generation(db_path)
            resource_eff = get_resource_efficiency_by_generation(db_path)
            action_eff = get_action_effectiveness_by_generation(db_path)

            # Store in result dictionary
            all_metrics[db_name] = metrics
            all_resource_efficiency[db_name] = resource_eff
            all_action_effectiveness[db_name] = action_eff

            # Store in result
            if metrics:
                first_gen = min(metrics.keys())
                comparison = compare_generations(metrics, first_gen=first_gen)

                result["metrics_by_generation"][db_name] = metrics
                result["resource_efficiency"][db_name] = resource_eff
                result["action_effectiveness"][db_name] = action_eff
                result["comparison"][db_name] = comparison

        # Combine metrics for experiment-wide analysis
        combined_metrics_by_generation = {}
        combined_resource_efficiency = {}
        combined_action_effectiveness = {}

        # Process metrics
        for db_name, metrics in all_metrics.items():
            for gen, gen_metrics in metrics.items():
                if gen not in combined_metrics_by_generation:
                    combined_metrics_by_generation[gen] = {
                        k: 0 for k in gen_metrics.keys()
                    }
                    combined_metrics_by_generation[gen]["simulation_count"] = 0

                for metric, value in gen_metrics.items():
                    if metric != "simulation_count":
                        combined_metrics_by_generation[gen][metric] += value

                combined_metrics_by_generation[gen]["simulation_count"] += 1

        # Process resource efficiency
        for db_name, resource_eff in all_resource_efficiency.items():
            for gen, gen_metrics in resource_eff.items():
                if gen not in combined_resource_efficiency:
                    combined_resource_efficiency[gen] = {
                        k: 0 for k in gen_metrics.keys()
                    }
                    combined_resource_efficiency[gen]["simulation_count"] = 0

                for metric, value in gen_metrics.items():
                    if metric != "simulation_count":
                        combined_resource_efficiency[gen][metric] += value

                combined_resource_efficiency[gen]["simulation_count"] += 1

        # Process action effectiveness
        for db_name, action_eff in all_action_effectiveness.items():
            for gen, gen_actions in action_eff.items():
                if gen not in combined_action_effectiveness:
                    combined_action_effectiveness[gen] = {}

                for action_type, action_metrics in gen_actions.items():
                    if action_type not in combined_action_effectiveness.get(gen, {}):
                        combined_action_effectiveness.setdefault(gen, {})[
                            action_type
                        ] = {k: 0 for k in action_metrics.keys()}
                        combined_action_effectiveness[gen][action_type][
                            "simulation_count"
                        ] = 0

                    for metric, value in action_metrics.items():
                        if metric != "simulation_count":
                            combined_action_effectiveness[gen][action_type][
                                metric
                            ] += value

                    combined_action_effectiveness[gen][action_type][
                        "simulation_count"
                    ] += 1

        # Calculate averages for all metrics
        for gen in combined_metrics_by_generation:
            sim_count = combined_metrics_by_generation[gen]["simulation_count"]
            if sim_count > 0:
                for metric in combined_metrics_by_generation[gen]:
                    if metric != "simulation_count":
                        combined_metrics_by_generation[gen][metric] /= sim_count

        for gen in combined_resource_efficiency:
            sim_count = combined_resource_efficiency[gen]["simulation_count"]
            if sim_count > 0:
                for metric in combined_resource_efficiency[gen]:
                    if metric != "simulation_count":
                        combined_resource_efficiency[gen][metric] /= sim_count

        for gen in combined_action_effectiveness:
            for action_type in combined_action_effectiveness[gen]:
                sim_count = combined_action_effectiveness[gen][action_type][
                    "simulation_count"
                ]
                if sim_count > 0:
                    for metric in combined_action_effectiveness[gen][action_type]:
                        if metric != "simulation_count":
                            combined_action_effectiveness[gen][action_type][
                                metric
                            ] /= sim_count

        # Skip analysis if no data was found
        if not combined_metrics_by_generation:
            logger.error("No generational metrics found in any simulation")
            return result

        # Generate experiment-wide analysis
        logger.info(f"Generating experiment-wide analysis for {experiment}")

        # Compare first generation to latest generations
        first_gen = min(combined_metrics_by_generation.keys())
        comparison = compare_generations(
            combined_metrics_by_generation, first_gen=first_gen
        )

        # Plot survival metrics
        survival_metrics = ["avg_survival_time", "reproduction_rate"]
        plot_generational_trends(
            combined_metrics_by_generation,
            survival_metrics,
            "Survival Metrics Across Generations (Experiment Average)",
            os.path.join(output_dir, "survival_trends.png"),
        )

        # Plot resource metrics
        resource_metrics = ["avg_resources", "max_resources"]
        plot_generational_trends(
            combined_metrics_by_generation,
            resource_metrics,
            "Resource Acquisition Across Generations (Experiment Average)",
            os.path.join(output_dir, "resource_trends.png"),
        )

        # Plot reward metrics
        reward_metrics = ["avg_reward"]
        plot_generational_trends(
            combined_metrics_by_generation,
            reward_metrics,
            "Reward Optimization Across Generations (Experiment Average)",
            os.path.join(output_dir, "reward_trends.png"),
        )

        # Generate fitness report
        generate_fitness_report(
            comparison, os.path.join(output_dir, "generational_fitness_report.txt")
        )

        # Plot action distribution changes
        if combined_action_effectiveness:
            plot_action_distribution_evolution(
                combined_action_effectiveness,
                os.path.join(output_dir, "action_evolution.png"),
            )

        # Generate individual simulation analyses if needed
        for db_name, metrics in all_metrics.items():
            if metrics:
                # Create output directory for this simulation
                sim_output_dir = os.path.join(
                    output_dir, "individual_simulations", db_name
                )
                os.makedirs(sim_output_dir, exist_ok=True)

                # Get data for this simulation
                resource_eff = all_resource_efficiency.get(db_name, {})
                action_eff = all_action_effectiveness.get(db_name, {})

                # Compare first generation to latest generations
                first_gen = min(metrics.keys())
                comparison = compare_generations(metrics, first_gen=first_gen)

                # Plot survival metrics
                plot_generational_trends(
                    metrics,
                    survival_metrics,
                    "Survival Metrics Across Generations",
                    os.path.join(sim_output_dir, "survival_trends.png"),
                )

                # Plot resource metrics
                plot_generational_trends(
                    metrics,
                    resource_metrics,
                    "Resource Acquisition Across Generations",
                    os.path.join(sim_output_dir, "resource_trends.png"),
                )

                # Plot reward metrics
                plot_generational_trends(
                    metrics,
                    reward_metrics,
                    "Reward Optimization Across Generations",
                    os.path.join(sim_output_dir, "reward_trends.png"),
                )

                # Generate fitness report
                generate_fitness_report(
                    comparison,
                    os.path.join(sim_output_dir, "generational_fitness_report.txt"),
                )

                # Plot action distribution changes
                if action_eff:
                    plot_action_distribution_evolution(
                        action_eff, os.path.join(sim_output_dir, "action_evolution.png")
                    )

        logger.info(
            f"Generational fitness analysis completed for experiment {experiment}"
        )
        return result

    except Exception as e:
        logger.error(f"Error processing experiment {experiment}: {str(e)}")
        return result

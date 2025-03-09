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
    # Query reproduction events
    reproduction_events = sim_session.query(ReproductionEventModel).all()

    if not reproduction_events:
        return {}

    # Get all agents to determine their types
    agents = {
        agent.agent_id: agent.agent_type.lower()
        for agent in sim_session.query(AgentModel).all()
    }

    # Initialize counters
    stats = {
        "system": {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "first_reproduction_time": float("inf"),
        },
        "independent": {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "first_reproduction_time": float("inf"),
        },
        "control": {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
            "first_reproduction_time": float("inf"),
        },
    }

    # Process reproduction events
    for event in reproduction_events:
        parent_id = event.parent_id
        parent_type = agents.get(parent_id, "unknown").lower()

        if parent_type not in stats:
            continue

        stats[parent_type]["attempts"] += 1

        if event.success:
            stats[parent_type]["successes"] += 1
            # Track first successful reproduction time
            stats[parent_type]["first_reproduction_time"] = min(
                stats[parent_type]["first_reproduction_time"], event.step_number
            )
        else:
            stats[parent_type]["failures"] += 1

    # Calculate derived metrics
    result = {}
    for agent_type, data in stats.items():
        if data["attempts"] > 0:
            result[f"{agent_type}_reproduction_attempts"] = data["attempts"]
            result[f"{agent_type}_reproduction_successes"] = data["successes"]
            result[f"{agent_type}_reproduction_failures"] = data["failures"]
            result[f"{agent_type}_reproduction_success_rate"] = (
                data["successes"] / data["attempts"]
            )

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

    return result


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

    plt.tight_layout()
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
    plt.tight_layout()
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

    plt.tight_layout()
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

    plt.tight_layout()
    output_file = os.path.join(CURRENT_OUTPUT_PATH, "reproduction_vs_dominance.png")
    plt.savefig(output_file)
    logging.info(f"Saved reproduction metrics plot to {output_file}")
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
    plt.tight_layout()
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

    # Adjust layout and save
    plt.tight_layout()
    output_file = os.path.join(CURRENT_OUTPUT_PATH, "dominance_comparison.png")
    plt.savefig(output_file, dpi=300)
    logging.info(f"Saved dominance comparison plot to {output_file}")
    plt.close()


def main():
    global CURRENT_OUTPUT_PATH

    # Create dominance output directory
    dominance_output_path = os.path.join(OUTPUT_PATH, "dominance")

    # Clear the dominance directory if it exists
    if os.path.exists(dominance_output_path):
        logging.info(f"Clearing existing dominance directory: {dominance_output_path}")
        shutil.rmtree(dominance_output_path)

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

    logging.info(f"Analyzing simulations in {experiment_path}...")
    df = analyze_simulations(experiment_path)

    if df.empty:
        logging.warning("No simulation data found.")
        return

    # Save the raw data
    output_csv = os.path.join(dominance_output_path, "simulation_analysis.csv")
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved analysis data to {output_csv}")

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

    # Plot dominance distribution
    plot_dominance_distribution(df)

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

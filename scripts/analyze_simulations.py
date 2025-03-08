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
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlalchemy
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
    Analyze all simulations in the experiment folder.
    """
    # Get all iteration folders
    iteration_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))

    data = []
    for folder in iteration_folders:
        iteration_number = int(os.path.basename(folder).split("_")[1])
        db_path = os.path.join(folder, "simulation.db")
        config_path = os.path.join(folder, "config.json")

        if not os.path.exists(db_path) or not os.path.exists(config_path):
            print(f"Skipping {folder}: Missing database or config file")
            continue

        try:
            # Load config parameters
            with open(config_path, "r") as f:
                config = json.load(f)

            # Connect to the simulation database
            engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
            Session = sessionmaker(bind=engine)
            session = Session()

            # Compute dominance metrics
            pop_dom = compute_population_dominance(session)
            surv_dom = compute_survival_dominance(session)

            # Get additional statistics
            pop_counts = get_final_population_counts(session)
            survival_stats = get_agent_survival_stats(session)

            # Get initial positions and resource proximity
            initial_positions = get_initial_positions_and_resources(session, config)

            # Get reproduction statistics
            reproduction_stats = get_reproduction_stats(session)

            # Close the database session
            session.close()

            # Create entry with iteration number and dominance metrics
            entry = {
                "iteration": iteration_number,
                "population_dominance": pop_dom,
                "survival_dominance": surv_dom,
            }

            # Add population counts
            if pop_counts:
                entry.update(pop_counts)

            # Add survival statistics
            if survival_stats:
                entry.update(survival_stats)

            # Add initial position and resource data
            if initial_positions:
                entry.update(initial_positions)

            # Add reproduction statistics
            if reproduction_stats:
                entry.update(reproduction_stats)

            # Extract key parameters from config
            key_params = [
                "system_agents",
                "independent_agents",
                "control_agents",
                "initial_resource_level",
                "max_population",
                "starvation_threshold",
                "offspring_cost",
                "min_reproduction_resources",
                "offspring_initial_resources",
                "perception_radius",
                "base_attack_strength",
                "base_defense_strength",
                "resource_regen_rate",
                "resource_regen_amount",
                "max_resource_amount",
                "base_consumption_rate",
                "max_movement",
                "gathering_range",
                "max_gather_amount",
                "territory_range",
                "starting_health",
                "attack_range",
                "attack_base_damage",
                "attack_kill_reward",
                "simulation_steps",
            ]

            # Add key parameters to entry
            for param in key_params:
                if param in config:
                    entry[param] = config[param]

            # Add agent type parameters
            if "agent_parameters" in config:
                for agent_type, params in config["agent_parameters"].items():
                    for param_name, param_value in params.items():
                        entry[f"{agent_type.lower()}_{param_name}"] = param_value

            data.append(entry)

        except Exception as e:
            print(f"Error processing {folder}: {e}")

    # Convert to DataFrame
    return pd.DataFrame(data)


def train_classifier(X, y, label_name):
    """
    Train a Random Forest classifier and print a classification report and feature importances.
    """
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n=== Classification Report for {label_name} Dominance ===")
    print(classification_report(y_test, y_pred))

    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Print feature importances
    importances = clf.feature_importances_
    feature_names = X.columns
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("\nTop 15 Feature Importances:")
    for feat, imp in feat_imp[:15]:
        print(f"{feat}: {imp:.3f}")

    return clf, feat_imp


def plot_dominance_distribution(df):
    """
    Plot the distribution of dominance types.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Population dominance
    pop_counts = df["population_dominance"].value_counts()
    ax1.bar(pop_counts.index, pop_counts.values)
    ax1.set_title("Population Dominance Distribution")
    ax1.set_ylabel("Count")
    ax1.set_xlabel("Agent Type")

    # Survival dominance
    surv_counts = df["survival_dominance"].value_counts()
    ax2.bar(surv_counts.index, surv_counts.values)
    ax2.set_title("Survival Dominance Distribution")
    ax2.set_ylabel("Count")
    ax2.set_xlabel("Agent Type")

    plt.tight_layout()
    plt.savefig("dominance_distribution.png")
    plt.close()


def plot_feature_importance(feat_imp, label_name):
    """
    Plot feature importance for a classifier.
    """
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
    plt.savefig(f"{label_name}_feature_importance.png")
    plt.close()


def plot_resource_proximity_vs_dominance(df):
    """
    Plot resource proximity metrics vs dominance.
    """
    resource_metrics = [
        col for col in df.columns if "resource" in col and "dist" in col
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
    plt.savefig("resource_proximity_vs_dominance.png")
    plt.close()


def plot_reproduction_vs_dominance(df):
    """
    Plot reproduction metrics vs dominance.
    """
    reproduction_metrics = [
        col
        for col in df.columns
        if ("reproduction" in col or "first_reproduction_time" in col)
        and not "vs" in col
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
    plt.savefig("reproduction_vs_dominance.png")
    plt.close()


def plot_correlation_matrix(df, label_name):
    """
    Plot correlation matrix between features and the target label.
    """
    # Convert categorical target to numeric for correlation
    target_numeric = pd.get_dummies(df[label_name])

    # Select numeric features
    numeric_features = df.select_dtypes(include=[np.number])

    # Remove the target if it's in the numeric features
    if label_name in numeric_features.columns:
        numeric_features = numeric_features.drop(label_name, axis=1)

    # Calculate correlation with each target class
    correlations = {}
    for target_class in target_numeric.columns:
        correlations[target_class] = numeric_features.corrwith(
            target_numeric[target_class]
        )

    # Combine correlations into a single DataFrame
    corr_df = pd.DataFrame(correlations)

    # Sort by absolute correlation
    corr_df["max_abs_corr"] = corr_df.abs().max(axis=1)
    corr_df = corr_df.sort_values("max_abs_corr", ascending=False).drop(
        "max_abs_corr", axis=1
    )

    # Plot top correlations
    top_n = 20
    top_corr = corr_df.head(top_n)

    plt.figure(figsize=(12, 10))
    sns.heatmap(top_corr, annot=True, cmap="coolwarm", center=0)
    plt.title(f"Top {top_n} Feature Correlations with {label_name}")
    plt.tight_layout()
    plt.savefig(f"{label_name}_correlation_matrix.png")
    plt.close()


def main():
    # Path to the experiment folder
    experiment_path = (
        "results/one_of_a_kind/experiments/data/one_of_a_kind_20250307_193855"
    )

    print(f"Analyzing simulations in {experiment_path}...")
    df = analyze_simulations(experiment_path)

    if df.empty:
        print("No simulation data found.")
        return

    # Save the raw data
    df.to_csv("simulation_analysis.csv", index=False)

    print(f"Analyzed {len(df)} simulations.")
    print("\nSummary statistics:")
    print(df.describe())

    print("\nDominance distribution:")
    print("Population dominance:", df["population_dominance"].value_counts())
    print("Survival dominance:", df["survival_dominance"].value_counts())

    # Plot dominance distribution
    plot_dominance_distribution(df)

    # Plot resource proximity vs dominance
    plot_resource_proximity_vs_dominance(df)

    # Plot reproduction metrics vs dominance
    plot_reproduction_vs_dominance(df)

    # Plot correlation matrices
    for label in ["population_dominance", "survival_dominance"]:
        if df[label].nunique() > 1:
            plot_correlation_matrix(df, label)

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
        X = df[feature_cols]

        # Handle missing values
        X = X.fillna(X.mean())

        # Train classifiers for each dominance type
        for label in ["population_dominance", "survival_dominance"]:
            if df[label].nunique() > 1:  # Only if we have multiple classes
                print(f"\nTraining classifier for {label}...")
                y = df[label]
                clf, feat_imp = train_classifier(X, y, label)

                # Plot feature importance
                plot_feature_importance(feat_imp, label)

    print("\nAnalysis complete. Results saved to CSV and PNG files.")


if __name__ == "__main__":
    main()

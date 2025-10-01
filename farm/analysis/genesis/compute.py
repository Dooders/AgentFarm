"""
Genesis Computation Module

This module provides functions to compute metrics related to initial states and conditions
in simulations, and analyze how these genesis factors impact simulation outcomes.
"""

from farm.utils.logging_config import get_logger
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import and_, desc, func, or_
from sqlalchemy.orm import Session

from farm.database.models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    ReproductionEventModel,
    ResourceModel,
    SimulationStepModel,
    SocialInteractionModel,
)

logger = get_logger(__name__)


def transform_metrics_for_plotting(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform computed metrics into the format expected by plotting functions.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Raw computed metrics

    Returns
    -------
    Dict[str, Any]
        Transformed metrics in plotting-friendly format
    """
    transformed = {}

    # Transform agent type distribution and attributes
    agent_starting_attributes = {}
    if "agent_type_distribution" in metrics:
        for agent_type, count in metrics["agent_type_distribution"].items():
            agent_starting_attributes[agent_type] = {
                "count": count,
                "avg_initial_resources": 0.0,  # Will be updated if data exists
                "avg_starting_health": 0.0,  # Will be updated if data exists
            }

    # Add resource metrics by agent type
    if "agent_resource_proximity" in metrics:
        for agent_id, data in metrics["agent_resource_proximity"].items():
            agent_type = metrics.get("agent_types", {}).get(agent_id)
            if agent_type and agent_type in agent_starting_attributes:
                attrs = agent_starting_attributes[agent_type]
                if "initial_resources" not in attrs:
                    attrs["initial_resources"] = []
                attrs["initial_resources"].append(data.get("resources_within_30", 0))

    # Calculate averages
    for agent_type, attrs in agent_starting_attributes.items():
        if "initial_resources" in attrs:
            attrs["avg_initial_resources"] = np.mean(attrs["initial_resources"])
            del attrs["initial_resources"]  # Remove raw data

    transformed["agent_starting_attributes"] = agent_starting_attributes

    # Add resource proximity data
    if "agent_type_resource_proximity" in metrics:
        transformed["agent_type_resource_proximity"] = metrics[
            "agent_type_resource_proximity"
        ]

    # Transform critical period metrics
    if "early_deaths" in metrics:
        total_deaths = sum(metrics["early_deaths"].values())
        total_agents = sum(metrics.get("agent_type_distribution", {}).values())
        if total_agents > 0:
            transformed["survival_rate"] = 1.0 - (total_deaths / total_agents)
        else:
            transformed["survival_rate"] = 0.0

    # Extract reproduction rate from first reproduction events
    if "first_reproduction_events" in metrics:
        reproduction_events = len(metrics["first_reproduction_events"])
        total_agents = sum(metrics.get("agent_type_distribution", {}).values())
        if total_agents > 0:
            transformed["reproduction_rate"] = reproduction_events / total_agents
        else:
            transformed["reproduction_rate"] = 0.0

    # Extract resource efficiency from agent type averages
    resource_efficiency = 0.0
    resource_count = 0

    for key, value in metrics.items():
        if key.endswith("_avg_resources") and isinstance(value, (int, float)):
            resource_efficiency += value
            resource_count += 1

    if resource_count > 0:
        transformed["resource_efficiency"] = resource_efficiency / resource_count
    else:
        transformed["resource_efficiency"] = 0.0

    # Add growth rates if available
    for key, value in metrics.items():
        if key.endswith("_growth_rate"):
            transformed[key] = value

    return transformed


def compute_initial_state_metrics(session: Session) -> Dict[str, Any]:
    """
    Compute comprehensive metrics about the initial state of a simulation.

    This function analyzes the starting configuration of agents, resources, and their
    spatial relationships at step 0 of the simulation.

    Parameters
    ----------
    session : Session
        SQLAlchemy database session

    Returns
    -------
    Dict[str, Any]
        Dictionary containing various initial state metrics
    """
    metrics = {}

    # Get initial agents (birth_time = 0) - use explicit column selection
    initial_agents_data = (
        session.query(
            AgentModel.agent_id,
            AgentModel.agent_type,
            AgentModel.position_x,
            AgentModel.position_y,
            AgentModel.initial_resources,
            AgentModel.starting_health,
            AgentModel.starvation_counter,
            AgentModel.action_weights,
        )
        .filter(AgentModel.birth_time == 0)
        .all()
    )

    # Convert to list of dictionaries for easier handling
    initial_agents = [
        {
            "agent_id": row[0],
            "agent_type": row[1],
            "position_x": row[2],
            "position_y": row[3],
            "initial_resources": row[4],
            "starting_health": row[5],
            "starvation_counter": row[6],
            "action_weights": row[7],
        }
        for row in initial_agents_data
    ]

    # Get initial resources (step_number = 0) - use explicit column selection
    initial_resources_data = (
        session.query(
            ResourceModel.amount, ResourceModel.position_x, ResourceModel.position_y
        )
        .filter(ResourceModel.step_number == 0)
        .all()
    )

    # Convert to list of dictionaries for easier handling
    initial_resources = [
        {"amount": row[0], "position_x": row[1], "position_y": row[2]}
        for row in initial_resources_data
    ]

    # Store agent types for reference
    metrics["agent_types"] = {
        agent["agent_id"]: agent["agent_type"] for agent in initial_agents
    }

    # Basic counts
    metrics["initial_agent_count"] = len(initial_agents)
    metrics["initial_resource_count"] = len(initial_resources)

    # Agent type distribution
    agent_type_counts = defaultdict(int)
    agent_type_resources = defaultdict(list)
    agent_type_health = defaultdict(list)

    for agent in initial_agents:
        agent_type_counts[agent["agent_type"]] += 1
        if agent["initial_resources"] is not None:
            agent_type_resources[agent["agent_type"]].append(agent["initial_resources"])
        if agent["starting_health"] is not None:
            agent_type_health[agent["agent_type"]].append(agent["starting_health"])

    metrics["agent_type_distribution"] = dict(agent_type_counts)

    # Calculate average resources and health by agent type
    agent_type_stats = {}
    for agent_type in agent_type_counts.keys():
        agent_type_stats[agent_type] = {
            "count": agent_type_counts[agent_type],
            "avg_initial_resources": (
                np.mean(agent_type_resources[agent_type])
                if agent_type_resources[agent_type] is not None
                else 0.0
            ),
            "avg_starting_health": (
                np.mean(agent_type_health[agent_type])
                if agent_type_health[agent_type] is not None
                else 0.0
            ),
        }

    metrics["agent_starting_attributes"] = agent_type_stats

    # Total initial resources in environment
    total_resource_amount = sum(
        resource["amount"]
        for resource in initial_resources
        if resource["amount"] is not None
    )
    metrics["total_resource_amount"] = total_resource_amount

    # Average resource amount
    metrics["avg_resource_amount"] = (
        total_resource_amount / len(initial_resources) if initial_resources else 0
    )

    # Resource distribution metrics
    if initial_resources:
        resource_amounts = []
        resource_positions = []

        for resource in initial_resources:
            # Get actual values from dictionary
            if resource["amount"] is not None:
                try:
                    amount = float(resource["amount"])
                    resource_amounts.append(amount)
                except (ValueError, TypeError):
                    continue

            if (
                resource["position_x"] is not None
                and resource["position_y"] is not None
            ):
                try:
                    pos_x = float(resource["position_x"])
                    pos_y = float(resource["position_y"])
                    resource_positions.append((pos_x, pos_y))
                except (ValueError, TypeError):
                    continue

        if resource_amounts:
            metrics["resource_amount_std"] = np.std(resource_amounts)
            metrics["resource_amount_min"] = min(resource_amounts)
            metrics["resource_amount_max"] = max(resource_amounts)

        if len(resource_positions) >= 2:
            # Calculate distances between all pairs of resources
            distances = pdist(resource_positions)
            metrics["resource_avg_distance"] = np.mean(distances)
            metrics["resource_min_distance"] = np.min(distances)
            metrics["resource_max_distance"] = np.max(distances)

            # Resource clustering coefficient (using a distance threshold)
            distance_matrix = squareform(distances)
            threshold = 20  # Consider resources within 20 units as clustered
            adjacency_matrix = distance_matrix < threshold

            # Count connections for each resource
            connections = (
                np.sum(adjacency_matrix, axis=1, out=None) - 1
            )  # Subtract 1 to exclude self-connection
            possible_connections = len(resource_positions) - 1

            # Calculate clustering coefficient for each resource
            clustering_coeffs = []
            for i in range(len(resource_positions)):
                if connections[i] <= 1:  # Not enough connections for triangles
                    continue

                # Count triangles (closed triplets)
                neighbors = np.where(adjacency_matrix[i])[0]
                triangles = 0

                for j in range(len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        if adjacency_matrix[neighbors[j], neighbors[k]]:
                            triangles += 1

                max_triangles = (connections[i] * (connections[i] - 1)) / 2
                if max_triangles > 0:
                    clustering_coeffs.append(triangles / max_triangles)

            if clustering_coeffs:
                metrics["resource_clustering_coefficient"] = np.mean(clustering_coeffs)
            else:
                metrics["resource_clustering_coefficient"] = 0

    # Agent-resource proximity metrics
    agent_resource_metrics = compute_agent_resource_proximity(
        initial_agents, initial_resources
    )
    metrics.update(agent_resource_metrics)

    # Agent-agent proximity metrics
    agent_agent_metrics = compute_agent_agent_proximity(initial_agents)
    metrics.update(agent_agent_metrics)

    # Agent starting attributes
    agent_attribute_metrics = compute_agent_starting_attributes(initial_agents)
    metrics.update(agent_attribute_metrics)

    # Relative advantage metrics
    advantage_metrics = compute_initial_relative_advantages(
        initial_agents, initial_resources
    )
    metrics.update(advantage_metrics)

    return metrics


def compute_agent_resource_proximity(
    agents: List[Dict[str, Any]], resources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute proximity metrics between agents and resources.

    Parameters
    ----------
    agents : List[AgentModel]
        List of agent models
    resources : List[ResourceModel]
        List of resource models

    Returns
    -------
    Dict[str, Any]
        Dictionary containing agent-resource proximity metrics
    """
    metrics = {}

    # Skip if no agents or resources
    if not agents or not resources:
        return {"agent_resource_proximity": {}, "agent_type_resource_proximity": {}}

    # Calculate distances from each agent to each resource
    agent_resource_distances = {}
    agent_type_distances = defaultdict(list)

    for agent in agents:
        agent_pos = (agent["position_x"], agent["position_y"])

        # Calculate distances to all resources
        distances = []
        weighted_distances = []

        for resource in resources:
            resource_pos = (resource["position_x"], resource["position_y"])
            distance = euclidean(agent_pos, resource_pos)
            distances.append(distance)

            # Weight distance by resource amount (higher amount = more important)
            weighted_distances.append(distance / (1 + resource["amount"]))

        # Store metrics for this agent
        agent_resource_distances[agent["agent_id"]] = {
            "min_distance": min(distances) if distances else float("inf"),
            "avg_distance": np.mean(distances) if distances else float("inf"),
            "weighted_avg_distance": (
                np.mean(weighted_distances) if weighted_distances else float("inf")
            ),
            "resources_within_30": sum(1 for d in distances if d <= 30),
            "resource_amount_within_30": sum(
                r["amount"] for r, d in zip(resources, distances) if d <= 30
            ),
        }

        # Aggregate by agent type
        agent_type_distances[agent["agent_type"]].extend(distances)

    # Calculate metrics by agent type
    agent_type_metrics = {}
    for agent_type, distances in agent_type_distances.items():
        agent_type_metrics[agent_type] = {
            "min_distance": min(distances) if distances else float("inf"),
            "avg_distance": np.mean(distances) if distances else float("inf"),
            "median_distance": np.median(distances) if distances else float("inf"),
        }

    metrics["agent_resource_proximity"] = agent_resource_distances
    metrics["agent_type_resource_proximity"] = agent_type_metrics

    return metrics


def compute_agent_agent_proximity(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute proximity metrics between agents.

    Parameters
    ----------
    agents : List[AgentModel]
        List of agent models

    Returns
    -------
    Dict[str, Any]
        Dictionary containing agent-agent proximity metrics
    """
    metrics = {}

    # Skip if not enough agents
    if len(agents) < 2:
        return {"agent_agent_proximity": {}, "agent_type_proximity": {}}

    # Calculate distances between all pairs of agents
    agent_positions = [(agent["position_x"], agent["position_y"]) for agent in agents]
    agent_types = [agent["agent_type"] for agent in agents]

    # Calculate pairwise distances
    distances = squareform(pdist(agent_positions))

    # Agent-agent proximity
    agent_agent_proximity = {}
    for i, agent in enumerate(agents):
        # Exclude self-distance (diagonal)
        other_distances = [distances[i, j] for j in range(len(agents)) if i != j]

        agent_agent_proximity[agent["agent_id"]] = {
            "nearest_agent_distance": (
                min(other_distances) if other_distances else float("inf")
            ),
            "avg_agent_distance": (
                np.mean(other_distances) if other_distances else float("inf")
            ),
            "agents_within_30": sum(1 for d in other_distances if d <= 30),
        }

    # Agent type proximity (average distance between agent types)
    agent_type_proximity = {}
    unique_agent_types = set(agent_types)

    for type1 in unique_agent_types:
        agent_type_proximity[type1] = {}

        for type2 in unique_agent_types:
            type_distances = []

            for i, agent_i in enumerate(agents):
                if agent_i["agent_type"] == type1:
                    for j, agent_j in enumerate(agents):
                        if i != j and agent_j["agent_type"] == type2:
                            type_distances.append(distances[i, j])

            if type_distances:
                agent_type_proximity[type1][type2] = {
                    "min_distance": min(type_distances),
                    "avg_distance": np.mean(type_distances),
                    "median_distance": np.median(type_distances),
                }

    metrics["agent_agent_proximity"] = agent_agent_proximity
    metrics["agent_type_proximity"] = agent_type_proximity

    return metrics


def compute_agent_starting_attributes(agents: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute metrics about agents' starting attributes.

    Parameters
    ----------
    agents : List[AgentModel]
        List of agent models

    Returns
    -------
    Dict[str, Any]
        Dictionary containing agent starting attribute metrics
    """
    metrics = {}

    # Skip if no agents
    if not agents:
        return {"agent_starting_attributes": {}}

    # Check if action_weights exist on any agent
    has_action_weights = any(
        "action_weights" in agent and agent["action_weights"] is not None
        for agent in agents
    )

    # Aggregate attributes by agent type
    agent_type_attributes = defaultdict(lambda: defaultdict(list))

    for agent in agents:
        agent_type_attributes[agent["agent_type"]]["initial_resources"].append(
            agent["initial_resources"]
        )
        agent_type_attributes[agent["agent_type"]]["starting_health"].append(
            agent["starting_health"]
        )
        agent_type_attributes[agent["agent_type"]]["starvation_counter"].append(
            agent["starvation_counter"]
        )

        # Process action_weights if they exist
        if (
            has_action_weights
            and "action_weights" in agent
            and agent["action_weights"] is not None
        ):
            if "action_weights_data" not in agent_type_attributes[agent["agent_type"]]:
                agent_type_attributes[agent["agent_type"]]["action_weights_data"] = []

            # Store the action weights data for later analysis
            agent_type_attributes[agent["agent_type"]]["action_weights_data"].append(
                agent["action_weights"]
            )

            try:
                # If action_weights is a dictionary, analyze its properties
                if isinstance(agent["action_weights"], dict):
                    # Number of actions with weights
                    if "action_count" not in agent_type_attributes[agent["agent_type"]]:
                        agent_type_attributes[agent["agent_type"]]["action_count"] = []
                    agent_type_attributes[agent["agent_type"]]["action_count"].append(
                        len(agent["action_weights"])
                    )

                    # Average weight value
                    if (
                        "avg_action_weight"
                        not in agent_type_attributes[agent["agent_type"]]
                    ):
                        agent_type_attributes[agent["agent_type"]][
                            "avg_action_weight"
                        ] = []
                    if agent["action_weights"]:
                        avg_weight = sum(
                            float(w) for w in agent["action_weights"].values()
                        ) / len(agent["action_weights"])
                        agent_type_attributes[agent["agent_type"]][
                            "avg_action_weight"
                        ].append(avg_weight)

                    # Maximum weight
                    if (
                        "max_action_weight"
                        not in agent_type_attributes[agent["agent_type"]]
                    ):
                        agent_type_attributes[agent["agent_type"]][
                            "max_action_weight"
                        ] = []
                    if agent["action_weights"]:
                        max_weight = max(
                            float(w) for w in agent["action_weights"].values()
                        )
                        agent_type_attributes[agent["agent_type"]][
                            "max_action_weight"
                        ].append(max_weight)
            except Exception as e:
                # Log but continue if there's an error processing action_weights
                logger.warning(
                    f"Error processing action_weights for agent {agent['agent_id']}: {e}"
                )

    # Calculate statistics for each attribute by agent type
    agent_starting_attributes = {}

    for agent_type, attributes in agent_type_attributes.items():
        agent_starting_attributes[agent_type] = {}

        for attr_name, values in attributes.items():
            # Skip raw action_weights_data from the output metrics
            if attr_name == "action_weights_data":
                continue

            if values:  # Only compute statistics if we have values
                try:
                    agent_starting_attributes[agent_type][attr_name] = {
                        "mean": np.mean(values),
                        "median": np.median(values),
                        "min": min(values),
                        "max": max(values),
                        "std": np.std(values),
                    }
                except Exception as e:
                    logger.warning(f"Error computing statistics for {attr_name}: {e}")
                    # Store the raw values instead
                    agent_starting_attributes[agent_type][attr_name] = {
                        "values": values,
                        "error": str(e),
                    }

    # Analyze action weights in more detail if available
    if has_action_weights:
        for agent_type, attributes in agent_type_attributes.items():
            if (
                "action_weights_data" in attributes
                and attributes["action_weights_data"]
            ):
                # Initialize action weight analysis
                if (
                    "action_weights_analysis"
                    not in agent_starting_attributes[agent_type]
                ):
                    agent_starting_attributes[agent_type][
                        "action_weights_analysis"
                    ] = {}

                # Collect all action types across agents of this type
                action_types = set()
                for weights in attributes["action_weights_data"]:
                    if isinstance(weights, dict):
                        action_types.update(weights.keys())

                # Analyze each action type
                for action in action_types:
                    action_values = []
                    for weights in attributes["action_weights_data"]:
                        if isinstance(weights, dict) and action in weights:
                            try:
                                action_values.append(float(weights[action]))
                            except (ValueError, TypeError):
                                pass  # Skip non-numeric values

                    if action_values:
                        agent_starting_attributes[agent_type][
                            "action_weights_analysis"
                        ][action] = {
                            "mean": np.mean(action_values),
                            "median": np.median(action_values),
                            "min": min(action_values),
                            "max": max(action_values),
                            "std": (
                                np.std(action_values) if len(action_values) > 1 else 0.0
                            ),
                        }

    metrics["agent_starting_attributes"] = agent_starting_attributes

    return metrics


def compute_initial_relative_advantages(
    agents: List[Dict[str, Any]], resources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute relative advantages between agent types based on initial conditions.

    Parameters
    ----------
    agents : List[AgentModel]
        List of agent models
    resources : List[ResourceModel]
        List of resource models

    Returns
    -------
    Dict[str, Any]
        Dictionary containing relative advantage metrics
    """
    metrics = {}

    # Skip if no agents
    if not agents:
        return {"initial_relative_advantages": {}}

    # Group agents by type
    agents_by_type = defaultdict(list)
    for agent in agents:
        agents_by_type[agent["agent_type"]].append(agent)

    # Calculate resource proximity advantage
    resource_proximity_advantage = {}

    # Calculate resources within gathering range (30 units) for each agent type
    resources_in_range_by_type = {}
    resource_amount_in_range_by_type = {}

    for agent_type, agent_list in agents_by_type.items():
        resources_in_range = 0
        resource_amount_in_range = 0

        for agent in agent_list:
            agent_pos = (agent["position_x"], agent["position_y"])

            for resource in resources:
                resource_pos = (resource["position_x"], resource["position_y"])
                distance = euclidean(agent_pos, resource_pos)

                if distance <= 30:  # Gathering range
                    resources_in_range += 1
                    resource_amount_in_range += resource["amount"]

        # Normalize by number of agents of this type
        if agent_list:
            resources_in_range_by_type[agent_type] = resources_in_range / len(
                agent_list
            )
            resource_amount_in_range_by_type[agent_type] = (
                resource_amount_in_range / len(agent_list)
            )

    # Calculate pairwise advantages
    agent_types = list(agents_by_type.keys())

    for i, type1 in enumerate(agent_types):
        for j in range(i + 1, len(agent_types)):
            type2 = agent_types[j]

            # Skip if either type has no resources in range data
            if (
                type1 not in resources_in_range_by_type
                or type2 not in resources_in_range_by_type
            ):
                continue

            # Calculate resource count advantage
            count_advantage = (
                resources_in_range_by_type[type1] - resources_in_range_by_type[type2]
            )

            # Calculate resource amount advantage
            amount_advantage = (
                resource_amount_in_range_by_type[type1]
                - resource_amount_in_range_by_type[type2]
            )

            # Store advantages
            key = f"{type1}_vs_{type2}"
            resource_proximity_advantage[key] = {
                "resource_count_advantage": count_advantage,
                "resource_amount_advantage": amount_advantage,
            }

    # Calculate starting attribute advantages
    attribute_advantage = {}

    for i, type1 in enumerate(agent_types):
        for j in range(i + 1, len(agent_types)):
            type2 = agent_types[j]

            # Skip if either type has no agents
            if not agents_by_type[type1] or not agents_by_type[type2]:
                continue

            # Calculate average initial resources for each type
            avg_resources_type1 = np.mean(
                [agent["initial_resources"] for agent in agents_by_type[type1]]
            )
            avg_resources_type2 = np.mean(
                [agent["initial_resources"] for agent in agents_by_type[type2]]
            )

            # Calculate average starting health for each type
            avg_health_type1 = np.mean(
                [agent["starting_health"] for agent in agents_by_type[type1]]
            )
            avg_health_type2 = np.mean(
                [agent["starting_health"] for agent in agents_by_type[type2]]
            )

            # Store advantages
            key = f"{type1}_vs_{type2}"
            attribute_advantage[key] = {
                "initial_resources_advantage": avg_resources_type1
                - avg_resources_type2,
                "starting_health_advantage": avg_health_type1 - avg_health_type2,
            }

    metrics["resource_proximity_advantage"] = resource_proximity_advantage
    metrics["attribute_advantage"] = attribute_advantage

    return {"initial_relative_advantages": metrics}


def compute_genesis_impact_scores(session: Session) -> Dict[str, Any]:
    """
    Compute impact scores that quantify how initial conditions affect simulation outcomes.

    This function uses machine learning to determine which initial factors have the
    greatest impact on various outcome metrics.

    Parameters
    ----------
    session : Session
        SQLAlchemy database session

    Returns
    -------
    Dict[str, Any]
        Dictionary containing impact scores for initial conditions
    """
    # Get initial state metrics
    initial_metrics = compute_initial_state_metrics(session)

    # Get outcome metrics
    outcome_metrics = compute_simulation_outcomes(session)

    # Prepare feature set from initial metrics
    features = extract_features_from_metrics(initial_metrics)

    # Calculate impact scores for each outcome
    impact_scores = {}

    for outcome_name, outcome_value in outcome_metrics.items():
        # Skip non-numeric outcomes
        if not isinstance(outcome_value, (int, float)) or math.isnan(outcome_value):
            continue

        # Calculate feature importance for this outcome
        importance = calculate_feature_importance(features, outcome_value)

        if importance:
            impact_scores[outcome_name] = importance

    # Calculate overall impact scores (average across outcomes)
    overall_impact = {}

    for feature in features.keys():
        scores = [
            outcome_scores.get(feature, 0)
            for outcome_scores in impact_scores.values()
            if feature in outcome_scores
        ]

        if scores:
            overall_impact[feature] = np.mean(scores)

    return {
        "outcome_specific_impacts": impact_scores,
        "overall_impact_scores": overall_impact,
    }


def compute_simulation_outcomes(session: Session) -> Dict[str, Any]:
    """
    Compute various outcome metrics for the simulation.

    Parameters
    ----------
    session : Session
        SQLAlchemy database session

    Returns
    -------
    Dict[str, Any]
        Dictionary containing outcome metrics
    """
    outcomes = {}

    # Get the final simulation step
    final_step = (
        session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.desc())
        .first()
    )

    if not final_step:
        return outcomes

    # Population outcomes
    outcomes["final_total_agents"] = final_step.total_agents
    outcomes["final_system_agents"] = final_step.system_agents
    outcomes["final_independent_agents"] = final_step.independent_agents
    outcomes["final_control_agents"] = final_step.control_agents

    # Determine population dominance
    step_dict = final_step.as_dict()
    agent_counts = {
        "SystemAgent": step_dict["system_agents"] or 0,
        "IndependentAgent": step_dict["independent_agents"] or 0,
        "ControlAgent": step_dict["control_agents"] or 0,
    }

    if any(agent_counts.values()):
        outcomes["population_dominant_type"] = max(
            agent_counts, key=lambda k: agent_counts[k]
        )

        # Calculate dominance margin (difference between highest and second highest)
        sorted_counts = sorted(agent_counts.values(), reverse=True)
        if len(sorted_counts) >= 2:
            outcomes["dominance_margin"] = sorted_counts[0] - sorted_counts[1]

    # Resource efficiency
    outcomes["final_resource_efficiency"] = final_step.resource_efficiency

    # Reproduction outcomes
    reproduction_events = session.query(ReproductionEventModel).all()

    if reproduction_events:
        # Total successful reproductions
        successful_reproductions = [
            event for event in reproduction_events if event.success is True
        ]
        outcomes["total_successful_reproductions"] = len(successful_reproductions)

        # Reproduction by agent type
        reproduction_by_type = defaultdict(int)
        for event in successful_reproductions:
            parent = (
                session.query(AgentModel)
                .filter(AgentModel.agent_id == event.parent_id)
                .first()
            )
            if parent:
                reproduction_by_type[parent.agent_type] += 1

        for agent_type, count in reproduction_by_type.items():
            outcomes[f"{agent_type}_reproductions"] = count

    # Survival outcomes
    agents = session.query(AgentModel).all()

    if agents:
        # Calculate average lifespan for dead agents
        dead_agents = [agent for agent in agents if agent["death_time"] is not None]
        if dead_agents:
            lifespans = [
                agent["death_time"] - agent["birth_time"] for agent in dead_agents
            ]
            outcomes["average_lifespan"] = np.mean(lifespans)

        # Calculate survival rate
        outcomes["survival_rate"] = 1 - (len(dead_agents) / len(agents))

        # Survival by agent type
        survival_by_type = defaultdict(lambda: {"total": 0, "alive": 0})

        for agent in agents:
            survival_by_type[agent["agent_type"]]["total"] += 1
            if agent["death_time"] is None:
                survival_by_type[agent["agent_type"]]["alive"] += 1

        for agent_type, counts in survival_by_type.items():
            if counts["total"] > 0:
                outcomes[f"{agent_type}_survival_rate"] = (
                    counts["alive"] / counts["total"]
                )

    return outcomes


def extract_features_from_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract a flat dictionary of numeric features from the hierarchical metrics.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Hierarchical metrics dictionary

    Returns
    -------
    Dict[str, float]
        Flat dictionary of numeric features
    """
    features = {}

    # Helper function to flatten nested dictionaries
    def flatten_dict(d, prefix=""):
        for key, value in d.items():
            if isinstance(value, dict):
                flatten_dict(value, f"{prefix}{key}_")
            elif (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and not math.isnan(value)
            ):
                features[f"{prefix}{key}"] = value

    flatten_dict(metrics)
    return features


def calculate_feature_importance(
    features: Dict[str, float], outcome: float
) -> Dict[str, float]:
    """
    Calculate the importance of each feature for predicting the outcome.

    Parameters
    ----------
    features : Dict[str, float]
        Dictionary of feature values
    outcome : float
        Outcome value to predict

    Returns
    -------
    Dict[str, float]
        Dictionary mapping feature names to importance scores
    """
    # Convert features to numpy array
    feature_names = list(features.keys())
    X = np.array([features[name] for name in feature_names]).reshape(1, -1)
    y = np.array([outcome])

    # Skip if not enough data
    if len(X) < 1 or len(feature_names) < 1:
        return {}

    try:
        # Train a simple model to get feature importance
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Get feature importance
        importance = model.feature_importances_

        # Create dictionary of feature importance
        importance_dict = {
            name: float(imp) for name, imp in zip(feature_names, importance)
        }

        return importance_dict
    except Exception as e:
        logger.warning(f"Error calculating feature importance: {e}")
        return {}


def compute_critical_period_metrics(
    session: Session, critical_period_end: int = 100
) -> Dict[str, Any]:
    """
    Compute metrics for the critical early period of the simulation.

    This function analyzes the early phase of the simulation (up to critical_period_end steps)
    to identify patterns and events that significantly impact the final outcome.

    Parameters
    ----------
    session : Session
        SQLAlchemy database session
    critical_period_end : int, optional
        The step number that marks the end of the critical period, by default 100

    Returns
    -------
    Dict[str, Any]
        Dictionary containing critical period metrics
    """
    metrics = {}

    # Get initial state
    initial_metrics = compute_initial_state_metrics(session)
    initial_agent_count = sum(
        initial_metrics.get("agent_type_distribution", {}).values()
    )

    # Get simulation steps in the critical period
    critical_steps = (
        session.query(SimulationStepModel)
        .filter(SimulationStepModel.step_number <= critical_period_end)
        .order_by(SimulationStepModel.step_number)
        .all()
    )

    if not critical_steps:
        return {"error": "No steps found in critical period"}

    # Population metrics during critical period
    population_metrics = {
        "system_agents": [step.as_dict()["system_agents"] for step in critical_steps],
        "independent_agents": [
            step.as_dict()["independent_agents"] for step in critical_steps
        ],
        "control_agents": [step.as_dict()["control_agents"] for step in critical_steps],
        "total_agents": [step.as_dict()["total_agents"] for step in critical_steps],
    }

    # Calculate population growth rates
    for agent_type in ["system_agents", "independent_agents", "control_agents"]:
        if len(population_metrics[agent_type]) >= 2:
            initial_pop = population_metrics[agent_type][0] or 0
            final_pop = population_metrics[agent_type][-1] or 0

            if initial_pop > 0:
                growth_rate = (final_pop - initial_pop) / initial_pop
            else:
                growth_rate = float("inf") if final_pop > 0 else 0

            metrics[f"{agent_type}_growth_rate"] = growth_rate

    # Calculate survival rate based on deaths during critical period
    dead_agents = (
        session.query(AgentModel)
        .filter(AgentModel.death_time <= critical_period_end)
        .all()
    )

    early_deaths = defaultdict(int)
    for agent in dead_agents:
        early_deaths[agent.agent_type] += 1

    total_early_deaths = sum(early_deaths.values())
    metrics["early_deaths"] = dict(early_deaths)
    metrics["survival_rate"] = (
        1.0 - (total_early_deaths / initial_agent_count)
        if initial_agent_count > 0
        else 0.0
    )

    # Calculate reproduction rate based on successful reproduction events
    reproduction_events = (
        session.query(ReproductionEventModel)
        .filter(ReproductionEventModel.step_number <= critical_period_end)
        .filter(ReproductionEventModel.success == True)
        .all()
    )

    first_reproductions = {}
    reproduction_count = 0
    for event in reproduction_events:
        agent = (
            session.query(AgentModel)
            .filter(AgentModel.agent_id == event.parent_id)
            .first()
        )
        if agent:
            if agent.agent_type not in first_reproductions:
                first_reproductions[agent.agent_type] = event.step_number
                reproduction_count += 1

    metrics["first_reproduction_events"] = first_reproductions
    metrics["reproduction_rate"] = (
        reproduction_count / initial_agent_count if initial_agent_count > 0 else 0.0
    )

    # Calculate resource efficiency based on average resource levels
    resource_metrics = defaultdict(list)

    agent_states = (
        session.query(AgentStateModel)
        .filter(AgentStateModel.step_number <= critical_period_end)
        .all()
    )

    for state in agent_states:
        agent = (
            session.query(AgentModel)
            .filter(AgentModel.agent_id == state.agent_id)
            .first()
        )
        if agent and state.resource_level is not None:
            resource_metrics[agent.agent_type].append(state.resource_level)

    # Calculate average resources by agent type
    avg_resources = {}
    for agent_type, resources in resource_metrics.items():
        if resources:
            avg_resources[agent_type] = np.mean(resources)
            metrics[f"{agent_type}_avg_resources"] = avg_resources[agent_type]

    # Overall resource efficiency
    if avg_resources:
        metrics["resource_efficiency"] = np.mean(list(avg_resources.values()))
    else:
        metrics["resource_efficiency"] = 0.0

    # Early social interactions
    social_interactions = (
        session.query(SocialInteractionModel)
        .filter(SocialInteractionModel.step_number <= critical_period_end)
        .all()
    )

    if social_interactions:
        interaction_counts = defaultdict(lambda: defaultdict(int))

        for interaction in social_interactions:
            initiator = (
                session.query(AgentModel)
                .filter(AgentModel.agent_id == interaction.initiator_id)
                .first()
            )
            recipient = (
                session.query(AgentModel)
                .filter(AgentModel.agent_id == interaction.recipient_id)
                .first()
            )

            if initiator and recipient:
                interaction_counts[initiator.agent_type][
                    interaction.interaction_type
                ] += 1

        metrics["early_social_interactions"] = {
            agent_type: dict(type_counts)
            for agent_type, type_counts in interaction_counts.items()
        }

        # Calculate critical period dominance
        if critical_steps:
            # Get the last step in the critical period
            last_critical_step = critical_steps[-1]
            step_dict = last_critical_step.as_dict()

            agent_counts = {
                "SystemAgent": step_dict["system_agents"] or 0,
                "IndependentAgent": step_dict["independent_agents"] or 0,
                "ControlAgent": step_dict["control_agents"] or 0,
            }

            if any(agent_counts.values()):
                metrics["critical_period_dominant"] = max(
                    agent_counts, key=lambda k: agent_counts[k]
                )

    # Add logging to debug metric values
    logger.info(f"Critical Period Metrics:")
    logger.info(f"Survival Rate: {metrics.get('survival_rate', 0.0)}")
    logger.info(f"Reproduction Rate: {metrics.get('reproduction_rate', 0.0)}")
    logger.info(f"Resource Efficiency: {metrics.get('resource_efficiency', 0.0)}")

    return metrics

"""
Social Behavior Analysis Module

This module provides functions to compute metrics related to social behaviors
between agents, including cooperation, competition, and group dynamics.
"""

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import and_, desc, func, or_
from sqlalchemy.orm import Session

from farm.database.models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    HealthIncident,
    ReproductionEventModel,
    SocialInteractionModel,
)

logger = logging.getLogger(__name__)


def compute_social_network_metrics(session: Session) -> Dict[str, Any]:
    """
    Compute social network metrics from agent interactions.

    Parameters
    ----------
    session : Session
        SQLAlchemy database session

    Returns
    -------
    Dict[str, Any]
        Dictionary containing various social network metrics
    """
    # Query all interactions between agents
    interactions = (
        session.query(
            ActionModel.agent_id,
            ActionModel.action_target_id,
            ActionModel.action_type,
            ActionModel.step_number,
        )
        .filter(
            ActionModel.action_target_id.isnot(None),
            ActionModel.action_type.in_(["share", "attack", "defend", "assist"]),
        )
        .all()
    )

    if not interactions:
        logger.warning("No social interactions found in the database")
        return {"error": "No social interactions found"}

    # Build adjacency matrix and interaction counts
    agent_ids = set()
    interaction_counts = defaultdict(int)
    interaction_types = defaultdict(lambda: defaultdict(int))

    for agent_id, target_id, action_type, _ in interactions:
        agent_ids.add(agent_id)
        agent_ids.add(target_id)

        # Count interactions between each pair
        interaction_key = (agent_id, target_id)
        interaction_counts[interaction_key] += 1

        # Count by type
        interaction_types[action_type][interaction_key] += 1

    # Convert to list for indexing
    agent_id_list = list(agent_ids)
    n = len(agent_id_list)

    # Create adjacency matrix
    adjacency_matrix = np.zeros((n, n))
    for i, agent1 in enumerate(agent_id_list):
        for j, agent2 in enumerate(agent_id_list):
            if (agent1, agent2) in interaction_counts:
                adjacency_matrix[i, j] = interaction_counts[(agent1, agent2)]

    # Compute network metrics
    results = {
        "total_interactions": sum(interaction_counts.values()),
        "unique_interaction_pairs": len(interaction_counts),
        "network_density": (
            sum(adjacency_matrix.flatten()) / (n * (n - 1)) if n > 1 else 0
        ),
        "interaction_types": {k: sum(v.values()) for k, v in interaction_types.items()},
        "agent_interaction_counts": {},
    }

    # Compute degree centrality (number of connections per agent)
    for i, agent_id in enumerate(agent_id_list):
        # Out-degree: number of agents this agent interacted with
        out_degree = np.sum(adjacency_matrix[i, :] > 0, out=None)
        # In-degree: number of agents that interacted with this agent
        in_degree = np.sum(adjacency_matrix[:, i] > 0, out=None)

        agent_type = (
            session.query(AgentModel.agent_type)
            .filter(AgentModel.agent_id == agent_id)
            .first()
        )
        if agent_type:
            agent_type = agent_type[0]
        else:
            agent_type = "unknown"

        results["agent_interaction_counts"][agent_id] = {
            "out_degree": int(out_degree),
            "in_degree": int(in_degree),
            "total_outgoing": int(np.sum(adjacency_matrix[i, :])),
            "total_incoming": int(np.sum(adjacency_matrix[:, i])),
            "agent_type": agent_type,
        }

    # Compute average metrics by agent type
    agent_type_metrics = defaultdict(lambda: defaultdict(list))
    for agent_id, metrics in results["agent_interaction_counts"].items():
        agent_type = metrics["agent_type"]
        agent_type_metrics[agent_type]["out_degree"].append(metrics["out_degree"])
        agent_type_metrics[agent_type]["in_degree"].append(metrics["in_degree"])
        agent_type_metrics[agent_type]["total_outgoing"].append(
            metrics["total_outgoing"]
        )
        agent_type_metrics[agent_type]["total_incoming"].append(
            metrics["total_incoming"]
        )

    results["agent_type_averages"] = {}
    for agent_type, metrics in agent_type_metrics.items():
        results["agent_type_averages"][agent_type] = {
            "avg_out_degree": (
                np.mean(metrics["out_degree"]) if metrics["out_degree"] else 0
            ),
            "avg_in_degree": (
                np.mean(metrics["in_degree"]) if metrics["in_degree"] else 0
            ),
            "avg_total_outgoing": (
                np.mean(metrics["total_outgoing"]) if metrics["total_outgoing"] else 0
            ),
            "avg_total_incoming": (
                np.mean(metrics["total_incoming"]) if metrics["total_incoming"] else 0
            ),
            "count": len(metrics["out_degree"]),
        }

    return results


def compute_resource_sharing_metrics(session: Session) -> Dict[str, Any]:
    """
    Compute metrics related to resource sharing behavior.

    Parameters
    ----------
    session : Session
        SQLAlchemy database session

    Returns
    -------
    Dict[str, Any]
        Dictionary containing resource sharing metrics
    """
    # Query resource sharing actions
    share_actions = (
        session.query(
            ActionModel.agent_id,
            ActionModel.action_target_id,
            ActionModel.resources_before,
            ActionModel.resources_after,
            ActionModel.step_number,
            AgentModel.agent_type.label("initiator_type"),
        )
        .join(AgentModel, AgentModel.agent_id == ActionModel.agent_id)
        .filter(
            ActionModel.action_type == "share", ActionModel.action_target_id.isnot(None)
        )
        .all()
    )

    if not share_actions:
        logger.warning("No resource sharing actions found in the database")
        return {"error": "No resource sharing actions found"}

    # Get agent types for targets
    agent_types = {
        a.agent_id: a.agent_type
        for a in session.query(AgentModel.agent_id, AgentModel.agent_type).all()
    }

    # Analyze sharing patterns
    results = {
        "total_sharing_actions": len(share_actions),
        "total_resources_shared": 0,
        "by_agent_type": defaultdict(lambda: defaultdict(int)),
        "sharing_matrix": defaultdict(lambda: defaultdict(float)),
        "time_distribution": defaultdict(int),
    }

    for action in share_actions:
        # Calculate amount shared
        resources_shared = action.resources_before - action.resources_after
        if resources_shared <= 0:
            continue  # Skip if no resources were actually shared

        target_type = agent_types.get(action.action_target_id, "unknown")

        # Update total
        results["total_resources_shared"] += resources_shared

        # Update by agent type
        results["by_agent_type"][action.initiator_type]["actions"] += 1
        results["by_agent_type"][action.initiator_type]["resources"] += resources_shared

        # Update sharing matrix (who shares with whom)
        results["sharing_matrix"][action.initiator_type][
            target_type
        ] += resources_shared

        # Update time distribution (bucketed by steps)
        step_bucket = action.step_number // 100  # Group by every 100 steps
        results["time_distribution"][step_bucket] += 1

    # Calculate average resources shared per action
    if results["total_sharing_actions"] > 0:
        results["avg_resources_per_share"] = (
            results["total_resources_shared"] / results["total_sharing_actions"]
        )
    else:
        results["avg_resources_per_share"] = 0

    # Convert sharing_matrix from defaultdict to regular dict for serialization
    results["sharing_matrix"] = {
        k: dict(v) for k, v in results["sharing_matrix"].items()
    }
    results["by_agent_type"] = dict(results["by_agent_type"])
    results["time_distribution"] = dict(results["time_distribution"])

    # Calculate altruism metrics: sharing when own resources are low
    altruistic_actions = (
        session.query(
            ActionModel.agent_id,
            ActionModel.resources_before,
            AgentModel.agent_type,
        )
        .join(AgentModel, AgentModel.agent_id == ActionModel.agent_id)
        .filter(
            ActionModel.action_type == "share",
            ActionModel.action_target_id.isnot(None),
            ActionModel.resources_before < 200,  # Define "low resources" threshold
        )
        .all()
    )

    results["altruistic_sharing"] = {
        "count": len(altruistic_actions),
        "by_agent_type": defaultdict(int),
    }

    for action in altruistic_actions:
        results["altruistic_sharing"]["by_agent_type"][action.agent_type] += 1

    results["altruistic_sharing"]["by_agent_type"] = dict(
        results["altruistic_sharing"]["by_agent_type"]
    )

    return results


def compute_spatial_clustering(
    session: Session, step: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze spatial clustering and group formation behaviors.

    Parameters
    ----------
    session : Session
        SQLAlchemy database session
    step : Optional[int]
        Specific step to analyze, if None analyzes the last step

    Returns
    -------
    Dict[str, Any]
        Dictionary containing spatial clustering metrics
    """
    # If step not provided, use the last step
    if step is None:
        max_step = session.query(func.max(AgentStateModel.step_number)).scalar()
        if max_step is None:
            logger.warning("No agent states found in the database")
            return {"error": "No agent states found"}
        step = max_step

    # Get all agent positions at the given step
    agent_positions = (
        session.query(
            AgentStateModel.agent_id,
            AgentStateModel.position_x,
            AgentStateModel.position_y,
            AgentModel.agent_type,
        )
        .join(AgentModel, AgentModel.agent_id == AgentStateModel.agent_id)
        .filter(AgentStateModel.step_number == step)
        .all()
    )

    if not agent_positions:
        logger.warning(f"No agent positions found for step {step}")
        return {"error": f"No agent positions found for step {step}"}

    # Format agent positions
    agents = []
    for agent_id, x, y, agent_type in agent_positions:
        agents.append(
            {"agent_id": agent_id, "position": (x, y), "agent_type": agent_type}
        )

    # Perform DBSCAN clustering to identify spatial groups
    from sklearn.cluster import DBSCAN

    # Extract coordinates for clustering
    coordinates = np.array(
        [[agent["position"][0], agent["position"][1]] for agent in agents]
    )

    # Determine epsilon (max distance between points in a cluster)
    # This should be calibrated based on your simulation's scale
    epsilon = 50.0  # Adjust based on your simulation's spatial scale

    # Run DBSCAN
    clustering = DBSCAN(eps=epsilon, min_samples=2).fit(coordinates)

    # Get cluster labels (-1 means noise/no cluster)
    labels = clustering.labels_

    # Count clusters and assign to agents
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    for i, agent in enumerate(agents):
        agent["cluster"] = int(labels[i])

    # Analyze clusters
    clusters = defaultdict(list)
    for agent in agents:
        if agent["cluster"] >= 0:  # Skip noise points
            clusters[agent["cluster"]].append(agent)

    # Calculate cluster stats
    cluster_stats = []
    for cluster_id, members in clusters.items():
        # Count by agent type
        type_counts = defaultdict(int)
        for agent in members:
            type_counts[agent["agent_type"]] += 1

        # Calculate centroid
        centroid_x = sum(agent["position"][0] for agent in members) / len(members)
        centroid_y = sum(agent["position"][1] for agent in members) / len(members)

        # Calculate diversity index (Shannon entropy)
        total = len(members)
        diversity = 0
        for count in type_counts.values():
            p = count / total
            diversity -= p * math.log(p) if p > 0 else 0

        cluster_stats.append(
            {
                "cluster_id": cluster_id,
                "size": len(members),
                "centroid": (centroid_x, centroid_y),
                "type_composition": dict(type_counts),
                "diversity_index": diversity,
            }
        )

    # Count isolated agents (no cluster)
    isolated_count = sum(1 for label in labels if label == -1)
    isolated_by_type = defaultdict(int)
    for i, agent in enumerate(agents):
        if labels[i] == -1:
            isolated_by_type[agent["agent_type"]] += 1

    # Prepare results
    results = {
        "step": step,
        "total_agents": len(agents),
        "total_clusters": n_clusters,
        "isolated_agents": isolated_count,
        "isolated_by_type": dict(isolated_by_type),
        "clustering_ratio": (
            (len(agents) - isolated_count) / len(agents) if len(agents) > 0 else 0
        ),
        "avg_cluster_size": (
            sum(len(members) for members in clusters.values()) / n_clusters
            if n_clusters > 0
            else 0
        ),
        "cluster_stats": cluster_stats,
    }

    # Add agent type clustering metrics
    agent_type_clusters = defaultdict(list)
    for i, agent in enumerate(agents):
        if labels[i] != -1:  # Skip noise points
            agent_type_clusters[agent["agent_type"]].append(labels[i])

    results["agent_type_clustering"] = {}
    for agent_type, cluster_ids in agent_type_clusters.items():
        # Count how many unique clusters this agent type participates in
        unique_clusters = len(set(cluster_ids))
        # Count agents of this type
        type_count = sum(1 for agent in agents if agent["agent_type"] == agent_type)
        # Calculate clustering ratio for this type
        clustering_ratio = len(cluster_ids) / type_count if type_count > 0 else 0

        results["agent_type_clustering"][agent_type] = {
            "unique_clusters": unique_clusters,
            "agents_in_clusters": len(cluster_ids),
            "total_agents": type_count,
            "clustering_ratio": clustering_ratio,
        }

    return results


def compute_cooperation_competition_metrics(session: Session) -> Dict[str, Any]:
    """
    Compute metrics related to cooperative and competitive behaviors.

    Parameters
    ----------
    session : Session
        SQLAlchemy database session

    Returns
    -------
    Dict[str, Any]
        Dictionary containing cooperation and competition metrics
    """
    # Cooperation includes sharing resources, helping defenses
    # Competition includes attacks, resource competition

    # Get cooperation actions
    cooperation_actions = (
        session.query(
            ActionModel.agent_id,
            ActionModel.action_target_id,
            ActionModel.action_type,
            ActionModel.step_number,
            AgentModel.agent_type.label("initiator_type"),
        )
        .join(AgentModel, AgentModel.agent_id == ActionModel.agent_id)
        .filter(
            ActionModel.action_type.in_(["share", "assist", "defend"]),
            ActionModel.action_target_id.isnot(None),
        )
        .all()
    )

    # Get competition actions (attacks)
    competition_actions = (
        session.query(
            ActionModel.agent_id,
            ActionModel.action_target_id,
            ActionModel.action_type,
            ActionModel.step_number,
            AgentModel.agent_type.label("initiator_type"),
        )
        .join(AgentModel, AgentModel.agent_id == ActionModel.agent_id)
        .filter(
            ActionModel.action_type.in_(["attack", "steal"]),
            ActionModel.action_target_id.isnot(None),
        )
        .all()
    )

    # Get agent types
    agent_types = {
        a.agent_id: a.agent_type
        for a in session.query(AgentModel.agent_id, AgentModel.agent_type).all()
    }

    # Initialize results
    results = {
        "cooperation": {
            "total_actions": len(cooperation_actions),
            "by_agent_type": defaultdict(int),
            "with_agent_type": defaultdict(lambda: defaultdict(int)),
            "action_types": defaultdict(int),
            "time_series": defaultdict(int),
        },
        "competition": {
            "total_actions": len(competition_actions),
            "by_agent_type": defaultdict(int),
            "with_agent_type": defaultdict(lambda: defaultdict(int)),
            "action_types": defaultdict(int),
            "time_series": defaultdict(int),
        },
    }

    # Process cooperation actions
    for action in cooperation_actions:
        target_type = agent_types.get(action.action_target_id, "unknown")

        results["cooperation"]["by_agent_type"][action.initiator_type] += 1
        results["cooperation"]["with_agent_type"][action.initiator_type][
            target_type
        ] += 1
        results["cooperation"]["action_types"][action.action_type] += 1

        step_bucket = action.step_number // 100  # Group by every 100 steps
        results["cooperation"]["time_series"][step_bucket] += 1

    # Process competition actions
    for action in competition_actions:
        target_type = agent_types.get(action.action_target_id, "unknown")

        results["competition"]["by_agent_type"][action.initiator_type] += 1
        results["competition"]["with_agent_type"][action.initiator_type][
            target_type
        ] += 1
        results["competition"]["action_types"][action.action_type] += 1

        step_bucket = action.step_number // 100
        results["competition"]["time_series"][step_bucket] += 1

    # Calculate cooperation-competition ratios by agent type
    results["coop_comp_ratio"] = {}
    for agent_type in set(results["cooperation"]["by_agent_type"].keys()) | set(
        results["competition"]["by_agent_type"].keys()
    ):
        coop_actions = results["cooperation"]["by_agent_type"].get(agent_type, 0)
        comp_actions = results["competition"]["by_agent_type"].get(agent_type, 0)

        if comp_actions > 0:
            ratio = coop_actions / comp_actions
        else:
            ratio = float("inf") if coop_actions > 0 else 0

        results["coop_comp_ratio"][agent_type] = {
            "cooperation": coop_actions,
            "competition": comp_actions,
            "ratio": ratio,
        }

    # Convert defaultdicts to regular dicts for serialization
    results["cooperation"]["by_agent_type"] = dict(
        results["cooperation"]["by_agent_type"]
    )
    results["cooperation"]["action_types"] = dict(
        results["cooperation"]["action_types"]
    )
    results["cooperation"]["time_series"] = dict(results["cooperation"]["time_series"])
    results["cooperation"]["with_agent_type"] = {
        k: dict(v) for k, v in results["cooperation"]["with_agent_type"].items()
    }

    results["competition"]["by_agent_type"] = dict(
        results["competition"]["by_agent_type"]
    )
    results["competition"]["action_types"] = dict(
        results["competition"]["action_types"]
    )
    results["competition"]["time_series"] = dict(results["competition"]["time_series"])
    results["competition"]["with_agent_type"] = {
        k: dict(v) for k, v in results["competition"]["with_agent_type"].items()
    }

    return results


def compute_reproduction_social_patterns(session: Session) -> Dict[str, Any]:
    """
    Analyze social patterns in reproduction.

    Parameters
    ----------
    session : Session
        SQLAlchemy database session

    Returns
    -------
    Dict[str, Any]
        Dictionary containing reproduction social metrics
    """
    # Get successful reproduction events
    reproduction_events = (
        session.query(
            ReproductionEventModel.parent_id,
            ReproductionEventModel.offspring_id,
            ReproductionEventModel.parent_position_x,
            ReproductionEventModel.parent_position_y,
            ReproductionEventModel.step_number,
            ReproductionEventModel.parent_resources_before,
            AgentModel.agent_type.label("parent_type"),
        )
        .join(AgentModel, AgentModel.agent_id == ReproductionEventModel.parent_id)
        .filter(
            ReproductionEventModel.success == True,
            ReproductionEventModel.offspring_id.isnot(None),
        )
        .all()
    )

    if not reproduction_events:
        logger.warning("No successful reproduction events found")
        return {"error": "No successful reproduction events found"}

    # Get agent positions for proximity analysis
    step_positions = defaultdict(list)

    # Query positions for steps where reproduction occurred
    reproduction_steps = set(event.step_number for event in reproduction_events)

    for step in reproduction_steps:
        positions = (
            session.query(
                AgentStateModel.agent_id,
                AgentStateModel.position_x,
                AgentStateModel.position_y,
                AgentModel.agent_type,
            )
            .join(AgentModel, AgentModel.agent_id == AgentStateModel.agent_id)
            .filter(AgentStateModel.step_number == step)
            .all()
        )

        for agent_id, x, y, agent_type in positions:
            step_positions[step].append(
                {"agent_id": agent_id, "position": (x, y), "agent_type": agent_type}
            )

    # Initialize results
    results = {
        "total_events": len(reproduction_events),
        "by_agent_type": defaultdict(int),
        "social_context": {
            "isolation": 0,  # No nearby agents
            "homogeneous": 0,  # Nearby agents of same type
            "heterogeneous": 0,  # Nearby agents of different types
            "by_agent_type": defaultdict(lambda: defaultdict(int)),
        },
        "time_series": defaultdict(int),
    }

    # Analyze each reproduction event
    for event in reproduction_events:
        parent_type = event.parent_type
        results["by_agent_type"][parent_type] += 1

        step_bucket = event.step_number // 100
        results["time_series"][step_bucket] += 1

        # Analyze social context (nearby agents)
        nearby_agents = []
        parent_pos = (event.parent_position_x, event.parent_position_y)

        for agent in step_positions[event.step_number]:
            if agent["agent_id"] == event.parent_id:
                continue  # Skip the parent

            agent_pos = agent["position"]
            distance = math.sqrt(
                (parent_pos[0] - agent_pos[0]) ** 2
                + (parent_pos[1] - agent_pos[1]) ** 2
            )

            # Define "nearby" threshold
            if distance < 100:  # Adjust based on your simulation scale
                nearby_agents.append(agent)

        # Categorize social context
        if not nearby_agents:
            results["social_context"]["isolation"] += 1
            results["social_context"]["by_agent_type"][parent_type]["isolation"] += 1
        else:
            # Check if nearby agents are same type or mixed
            nearby_types = set(agent["agent_type"] for agent in nearby_agents)
            if len(nearby_types) == 1 and parent_type in nearby_types:
                results["social_context"]["homogeneous"] += 1
                results["social_context"]["by_agent_type"][parent_type][
                    "homogeneous"
                ] += 1
            else:
                results["social_context"]["heterogeneous"] += 1
                results["social_context"]["by_agent_type"][parent_type][
                    "heterogeneous"
                ] += 1

    # Convert defaultdicts to regular dicts for serialization
    results["by_agent_type"] = dict(results["by_agent_type"])
    results["time_series"] = dict(results["time_series"])

    results["social_context"]["by_agent_type"] = {
        k: dict(v) for k, v in results["social_context"]["by_agent_type"].items()
    }

    # Calculate percentages
    total = results["total_events"]
    if total > 0:
        results["social_context"]["isolation_pct"] = (
            results["social_context"]["isolation"] / total * 100
        )
        results["social_context"]["homogeneous_pct"] = (
            results["social_context"]["homogeneous"] / total * 100
        )
        results["social_context"]["heterogeneous_pct"] = (
            results["social_context"]["heterogeneous"] / total * 100
        )

    return results


def compute_all_social_metrics(session: Session) -> Dict[str, Any]:
    """
    Compute all social behavior metrics in one function.

    Parameters
    ----------
    session : Session
        SQLAlchemy database session

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all social behavior metrics
    """
    results = {
        "social_network": compute_social_network_metrics(session),
        "resource_sharing": compute_resource_sharing_metrics(session),
        "spatial_clustering": compute_spatial_clustering(session),
        "cooperation_competition": compute_cooperation_competition_metrics(session),
        "reproduction_patterns": compute_reproduction_social_patterns(session),
    }

    # Add summary metrics
    summary = {}

    # Cooperation vs competition summary
    coop_actions = results["cooperation_competition"]["cooperation"]["total_actions"]
    comp_actions = results["cooperation_competition"]["competition"]["total_actions"]

    if comp_actions > 0:
        summary["overall_coop_comp_ratio"] = coop_actions / comp_actions
    else:
        summary["overall_coop_comp_ratio"] = float("inf") if coop_actions > 0 else 0

    # Add more overall summary metrics as needed
    summary["total_social_interactions"] = coop_actions + comp_actions

    if "error" not in results["spatial_clustering"]:
        summary["clustering_ratio"] = results["spatial_clustering"]["clustering_ratio"]

    # Agent type summaries
    agent_types = set()
    for metric in ["social_network", "resource_sharing", "cooperation_competition"]:
        if "agent_type_averages" in results[metric]:
            agent_types.update(results[metric]["agent_type_averages"].keys())
        elif "by_agent_type" in results[metric]:
            agent_types.update(results[metric]["by_agent_type"].keys())

    summary["agent_type_summary"] = {}
    for agent_type in agent_types:
        type_summary = {}

        # Get metrics for each agent type where available
        if "agent_type_averages" in results["social_network"]:
            network_data = results["social_network"]["agent_type_averages"].get(
                agent_type, {}
            )
            type_summary["avg_outgoing_connections"] = network_data.get(
                "avg_out_degree", 0
            )
            type_summary["avg_incoming_connections"] = network_data.get(
                "avg_in_degree", 0
            )

        if "coop_comp_ratio" in results["cooperation_competition"]:
            coop_comp = results["cooperation_competition"]["coop_comp_ratio"].get(
                agent_type, {}
            )
            type_summary["cooperation_competition_ratio"] = coop_comp.get("ratio", 0)

        if "agent_type_clustering" in results["spatial_clustering"]:
            clustering = results["spatial_clustering"]["agent_type_clustering"].get(
                agent_type, {}
            )
            type_summary["clustering_ratio"] = clustering.get("clustering_ratio", 0)

        if type_summary:
            summary["agent_type_summary"][agent_type] = type_summary

    results["summary"] = summary

    return results

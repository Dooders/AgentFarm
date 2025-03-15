"""
Social Behavior Analysis Module

This module provides functions to analyze social behavior patterns across simulations,
identify trends, and generate insights.
"""

import logging
import os
from typing import Dict, List, Any, Optional
import json
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from farm.analysis.social_behavior.compute import (
    compute_all_social_metrics,
    compute_social_network_metrics,
    compute_resource_sharing_metrics,
    compute_spatial_clustering,
    compute_cooperation_competition_metrics,
    compute_reproduction_social_patterns
)

logger = logging.getLogger(__name__)


def analyze_social_behaviors(session: Session) -> Dict[str, Any]:
    """
    Analyze social behaviors in a simulation.
    
    Parameters
    ----------
    session : Session
        SQLAlchemy database session
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing analysis results
    """
    # Compute comprehensive social metrics
    metrics = compute_all_social_metrics(session)
    
    # Identify key patterns and insights
    insights = extract_social_behavior_insights(metrics)
    
    # Return combined results
    return {
        "metrics": metrics,
        "insights": insights
    }


def extract_social_behavior_insights(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key insights from social metrics.
    
    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary of social behavior metrics
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing key insights
    """
    insights = {
        "key_findings": [],
        "agent_type_insights": {},
        "emergent_patterns": [],
        "recommendations": []
    }
    
    # Check for missing data
    if any("error" in section for section in metrics.values()):
        insights["warnings"] = [
            f"Missing data in section: {section}" 
            for section, data in metrics.items() 
            if isinstance(data, dict) and "error" in data
        ]
    
    # Agent type differences in social network connectedness
    if "social_network" in metrics and "agent_type_averages" in metrics["social_network"]:
        agent_types = metrics["social_network"]["agent_type_averages"].keys()
        
        if len(agent_types) > 1:
            # Find type with highest/lowest connectivity
            connectivity_scores = {
                agent_type: data["avg_out_degree"] + data["avg_in_degree"]
                for agent_type, data in metrics["social_network"]["agent_type_averages"].items()
            }
            
            most_connected = max(connectivity_scores.items(), key=lambda x: x[1])
            least_connected = min(connectivity_scores.items(), key=lambda x: x[1])
            
            insights["key_findings"].append({
                "type": "connectivity_difference",
                "description": f"{most_connected[0]} agents are the most socially connected (score: {most_connected[1]:.2f}), " +
                               f"while {least_connected[0]} agents are the least connected (score: {least_connected[1]:.2f})."
            })
    
    # Cooperation vs competition patterns
    if "cooperation_competition" in metrics and "coop_comp_ratio" in metrics["cooperation_competition"]:
        ratios = metrics["cooperation_competition"]["coop_comp_ratio"]
        
        # Is one type significantly more cooperative?
        if len(ratios) > 1:
            most_cooperative = max(ratios.items(), key=lambda x: x[1]["ratio"])
            least_cooperative = min(ratios.items(), key=lambda x: x[1]["ratio"])
            
            insights["key_findings"].append({
                "type": "cooperation_tendency",
                "description": f"{most_cooperative[0]} agents tend to be more cooperative (ratio: {most_cooperative[1]['ratio']:.2f}), " +
                               f"while {least_cooperative[0]} agents are more competitive (ratio: {least_cooperative[1]['ratio']:.2f})."
            })
    
    # Spatial clustering analysis
    if "spatial_clustering" in metrics and "error" not in metrics["spatial_clustering"]:
        clustering = metrics["spatial_clustering"]
        
        # Check if there are significant clusters
        if clustering["total_clusters"] > 1:
            insights["key_findings"].append({
                "type": "spatial_organization",
                "description": f"Agents form {clustering['total_clusters']} distinct spatial clusters with " +
                               f"average size of {clustering['avg_cluster_size']:.1f} agents."
            })
            
            # Is there segregation by agent type?
            if "agent_type_clustering" in clustering:
                type_clusters = clustering["agent_type_clustering"]
                
                homogeneous_clusters = sum(
                    1 for cluster in clustering["cluster_stats"]
                    if len(cluster["type_composition"]) == 1
                )
                
                if homogeneous_clusters > 0:
                    percent_homogeneous = (homogeneous_clusters / clustering["total_clusters"]) * 100
                    
                    if percent_homogeneous > 50:
                        insights["emergent_patterns"].append({
                            "type": "type_segregation",
                            "description": f"{percent_homogeneous:.1f}% of clusters are homogeneous, " +
                                        f"suggesting strong type segregation in the population."
                        })
                    elif percent_homogeneous < 20:
                        insights["emergent_patterns"].append({
                            "type": "type_integration",
                            "description": f"Only {percent_homogeneous:.1f}% of clusters are homogeneous, " +
                                        f"suggesting strong integration between agent types."
                        })
    
    # Resource sharing insights
    if "resource_sharing" in metrics and "error" not in metrics["resource_sharing"]:
        sharing = metrics["resource_sharing"]
        
        # Check if resource sharing is common
        if sharing["total_sharing_actions"] > 0:
            # Who shares the most?
            if "by_agent_type" in sharing:
                sharing_amounts = {
                    agent_type: data["resources"]
                    for agent_type, data in sharing["by_agent_type"].items()
                }
                
                if sharing_amounts:
                    most_generous = max(sharing_amounts.items(), key=lambda x: x[1])
                    
                    insights["key_findings"].append({
                        "type": "resource_sharing",
                        "description": f"{most_generous[0]} agents are the most generous, " +
                                      f"sharing {most_generous[1]:.1f} total resources."
                    })
            
            # Check for altruistic sharing
            if "altruistic_sharing" in sharing and sharing["altruistic_sharing"]["count"] > 0:
                pct_altruistic = (sharing["altruistic_sharing"]["count"] / sharing["total_sharing_actions"]) * 100
                
                if pct_altruistic > 10:
                    insights["emergent_patterns"].append({
                        "type": "altruism",
                        "description": f"{pct_altruistic:.1f}% of sharing actions are altruistic " +
                                      f"(when resources are scarce)."
                    })
    
    # Reproduction social context
    if "reproduction_patterns" in metrics and "error" not in metrics["reproduction_patterns"]:
        reproduction = metrics["reproduction_patterns"]
        
        if "social_context" in reproduction:
            context = reproduction["social_context"]
            
            # Do agents prefer to reproduce in isolation or in groups?
            if context.get("isolation_pct", 0) > 60:
                insights["emergent_patterns"].append({
                    "type": "isolated_reproduction",
                    "description": f"{context['isolation_pct']:.1f}% of reproduction occurs in isolation, " +
                                  f"suggesting agents avoid reproducing near others."
                })
            elif context.get("homogeneous_pct", 0) > 60:
                insights["emergent_patterns"].append({
                    "type": "in-group_reproduction",
                    "description": f"{context['homogeneous_pct']:.1f}% of reproduction occurs among same-type agents, " +
                                  f"suggesting strong in-group preference."
                })
    
    # Generate agent-type specific insights
    agent_types = set()
    for section in metrics.values():
        if isinstance(section, dict):
            if "agent_type_averages" in section:
                agent_types.update(section["agent_type_averages"].keys())
            elif "by_agent_type" in section:
                agent_types.update(section["by_agent_type"].keys())
    
    for agent_type in agent_types:
        type_insights = []
        
        # Check each metric section for agent-type specific insights
        for section_name, section in metrics.items():
            if isinstance(section, dict) and "error" not in section:
                if section_name == "social_network" and "agent_type_averages" in section:
                    if agent_type in section["agent_type_averages"]:
                        data = section["agent_type_averages"][agent_type]
                        if data["avg_out_degree"] > 3:  # Threshold can be adjusted
                            type_insights.append(
                                f"Forms many outgoing connections ({data['avg_out_degree']:.1f} average)"
                            )
                        if data["avg_in_degree"] > data["avg_out_degree"] * 1.5:
                            type_insights.append(
                                f"Receives more interactions than initiates (in/out ratio: {data['avg_in_degree']/data['avg_out_degree']:.1f})"
                            )
                
                elif section_name == "cooperation_competition" and "coop_comp_ratio" in section:
                    if agent_type in section["coop_comp_ratio"]:
                        ratio = section["coop_comp_ratio"][agent_type]["ratio"]
                        if ratio > 2:
                            type_insights.append(
                                f"Strongly cooperative behavior (coop/comp ratio: {ratio:.1f})"
                            )
                        elif ratio < 0.5:
                            type_insights.append(
                                f"Strongly competitive behavior (coop/comp ratio: {ratio:.1f})"
                            )
                
                elif section_name == "resource_sharing" and "by_agent_type" in section:
                    if agent_type in section["by_agent_type"]:
                        data = section["by_agent_type"][agent_type]
                        if data.get("actions", 0) > 10:  # Threshold can be adjusted
                            type_insights.append(
                                f"Frequent resource sharer ({data['actions']} actions, {data.get('resources', 0):.1f} resources)"
                            )
                
                elif section_name == "spatial_clustering" and "agent_type_clustering" in section:
                    if agent_type in section["agent_type_clustering"]:
                        data = section["agent_type_clustering"][agent_type]
                        if data["clustering_ratio"] > 0.8:
                            type_insights.append(
                                f"Strong tendency to form groups ({data['clustering_ratio']*100:.1f}% in clusters)"
                            )
                        elif data["clustering_ratio"] < 0.3:
                            type_insights.append(
                                f"Tendency to remain isolated ({(1-data['clustering_ratio'])*100:.1f}% isolated)"
                            )
        
        if type_insights:
            insights["agent_type_insights"][agent_type] = type_insights
    
    # Generate recommendations based on insights
    if "social_network" in metrics and "error" not in metrics["social_network"]:
        if metrics["social_network"].get("network_density", 0) < 0.1:
            insights["recommendations"].append(
                "Social interactions are sparse. Consider mechanisms to encourage more agent interactions."
            )
    
    if "cooperation_competition" in metrics:
        coop_actions = metrics["cooperation_competition"].get("cooperation", {}).get("total_actions", 0)
        comp_actions = metrics["cooperation_competition"].get("competition", {}).get("total_actions", 0)
        
        if coop_actions + comp_actions > 0:
            ratio = coop_actions / (comp_actions + 0.001)  # Avoid div by zero
            
            if ratio < 0.2:
                insights["recommendations"].append(
                    "Environment is highly competitive. Consider incentives for cooperation."
                )
            elif ratio > 5:
                insights["recommendations"].append(
                    "Environment is highly cooperative. Consider introducing competitive pressures."
                )
    
    if "spatial_clustering" in metrics and "error" not in metrics["spatial_clustering"]:
        if metrics["spatial_clustering"].get("clustering_ratio", 0) > 0.9:
            insights["recommendations"].append(
                "Agents are forming tight clusters. Consider resource distribution patterns to encourage exploration."
            )
    
    return insights


def analyze_social_behaviors_across_simulations(experiment_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze social behaviors across multiple simulations in an experiment.
    
    Parameters
    ----------
    experiment_path : str
        Path to the directory containing simulation folders
    output_path : Optional[str]
        Path to save results, if None results are not saved
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing aggregated social behavior analysis
    """
    import glob
    import sqlalchemy
    from sqlalchemy.orm import sessionmaker
    
    # Find all simulation databases
    sim_paths = glob.glob(os.path.join(experiment_path, "iteration_*/simulation.db"))
    
    if not sim_paths:
        logger.error(f"No simulation databases found in {experiment_path}")
        return {"error": f"No simulation databases found in {experiment_path}"}
    
    logger.info(f"Found {len(sim_paths)} simulation databases to analyze")
    
    # Initialize results storage
    all_results = []
    
    # Process each simulation
    for i, db_path in enumerate(sim_paths):
        logger.info(f"Processing simulation {i+1}/{len(sim_paths)}: {db_path}")
        
        try:
            # Connect to the database
            engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
            Session = sessionmaker(bind=engine)
            session = Session()
            
            # Run analysis
            results = analyze_social_behaviors(session)
            
            # Add simulation identifier
            sim_id = os.path.basename(os.path.dirname(db_path))
            results["simulation_id"] = sim_id
            
            all_results.append(results)
            
            # Close the session
            session.close()
            
        except Exception as e:
            logger.error(f"Error analyzing simulation {db_path}: {e}")
    
    # Aggregate results across simulations
    aggregated = aggregate_social_behavior_results(all_results)
    
    # Save results if requested
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        
        # Save as JSON
        output_file = os.path.join(output_path, "social_behavior_analysis.json")
        
        try:
            # Convert numpy types for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                return obj
            
            # Process the results to make them JSON-serializable
            def process_dict(d):
                if isinstance(d, dict):
                    return {k: process_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [process_dict(item) for item in d]
                else:
                    return convert_for_json(d)
            
            with open(output_file, "w") as f:
                json.dump(process_dict(aggregated), f, indent=2)
                
            logger.info(f"Saved analysis results to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
    
    return aggregated


def aggregate_social_behavior_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate social behavior results across multiple simulations.
    
    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of social behavior analysis results, one per simulation
        
    Returns
    -------
    Dict[str, Any]
        Aggregated results
    """
    if not results:
        return {"error": "No results to aggregate"}
    
    # Initialize aggregated results
    aggregated = {
        "total_simulations": len(results),
        "metrics": {
            "social_network": {
                "network_density": [],
                "total_interactions": [],
                "unique_interaction_pairs": [],
                "by_agent_type": {}
            },
            "resource_sharing": {
                "total_sharing_actions": [],
                "total_resources_shared": [],
                "avg_resources_per_share": [],
                "by_agent_type": {}
            },
            "cooperation_competition": {
                "cooperation_total": [],
                "competition_total": [],
                "coop_comp_ratio": [],
                "by_agent_type": {}
            },
            "spatial_clustering": {
                "total_clusters": [],
                "clustering_ratio": [],
                "avg_cluster_size": [],
                "by_agent_type": {}
            },
            "reproduction_patterns": {
                "total_events": [],
                "isolation_pct": [],
                "homogeneous_pct": [],
                "heterogeneous_pct": [],
                "by_agent_type": {}
            }
        },
        "common_insights": {
            "key_findings": {},
            "emergent_patterns": {},
            "agent_type_insights": {},
            "recommendations": {}
        }
    }
    
    # Collect all agent types across all simulations
    agent_types = set()
    for result in results:
        if "metrics" in result:
            metrics = result["metrics"]
            
            # Social network agent types
            if "social_network" in metrics and "agent_type_averages" in metrics["social_network"]:
                agent_types.update(metrics["social_network"]["agent_type_averages"].keys())
            
            # Resource sharing agent types
            if "resource_sharing" in metrics and "by_agent_type" in metrics["resource_sharing"]:
                agent_types.update(metrics["resource_sharing"]["by_agent_type"].keys())
            
            # Cooperation/competition agent types
            if "cooperation_competition" in metrics and "coop_comp_ratio" in metrics["cooperation_competition"]:
                agent_types.update(metrics["cooperation_competition"]["coop_comp_ratio"].keys())
            
            # Spatial clustering agent types
            if "spatial_clustering" in metrics and "agent_type_clustering" in metrics["spatial_clustering"]:
                agent_types.update(metrics["spatial_clustering"]["agent_type_clustering"].keys())
            
            # Reproduction agent types
            if "reproduction_patterns" in metrics and "by_agent_type" in metrics["reproduction_patterns"]:
                agent_types.update(metrics["reproduction_patterns"]["by_agent_type"].keys())
    
    # Initialize agent-type specific structures
    for agent_type in agent_types:
        aggregated["metrics"]["social_network"]["by_agent_type"][agent_type] = {
            "avg_out_degree": [],
            "avg_in_degree": [],
            "avg_total_outgoing": [],
            "avg_total_incoming": []
        }
        
        aggregated["metrics"]["resource_sharing"]["by_agent_type"][agent_type] = {
            "actions": [],
            "resources": []
        }
        
        aggregated["metrics"]["cooperation_competition"]["by_agent_type"][agent_type] = {
            "cooperation": [],
            "competition": [],
            "ratio": []
        }
        
        aggregated["metrics"]["spatial_clustering"]["by_agent_type"][agent_type] = {
            "clustering_ratio": []
        }
        
        aggregated["metrics"]["reproduction_patterns"]["by_agent_type"][agent_type] = {
            "count": []
        }
    
    # Aggregate metrics
    for result in results:
        if "metrics" not in result:
            continue
            
        metrics = result["metrics"]
        
        # Social network metrics
        if "social_network" in metrics and "error" not in metrics["social_network"]:
            sn = metrics["social_network"]
            
            aggregated["metrics"]["social_network"]["network_density"].append(sn.get("network_density", 0))
            aggregated["metrics"]["social_network"]["total_interactions"].append(sn.get("total_interactions", 0))
            aggregated["metrics"]["social_network"]["unique_interaction_pairs"].append(sn.get("unique_interaction_pairs", 0))
            
            # Agent type metrics
            if "agent_type_averages" in sn:
                for agent_type, data in sn["agent_type_averages"].items():
                    for metric in ["avg_out_degree", "avg_in_degree", "avg_total_outgoing", "avg_total_incoming"]:
                        if metric in data:
                            aggregated["metrics"]["social_network"]["by_agent_type"][agent_type][metric].append(data[metric])
        
        # Resource sharing metrics
        if "resource_sharing" in metrics and "error" not in metrics["resource_sharing"]:
            rs = metrics["resource_sharing"]
            
            aggregated["metrics"]["resource_sharing"]["total_sharing_actions"].append(rs.get("total_sharing_actions", 0))
            aggregated["metrics"]["resource_sharing"]["total_resources_shared"].append(rs.get("total_resources_shared", 0))
            aggregated["metrics"]["resource_sharing"]["avg_resources_per_share"].append(rs.get("avg_resources_per_share", 0))
            
            # Agent type metrics
            if "by_agent_type" in rs:
                for agent_type, data in rs["by_agent_type"].items():
                    aggregated["metrics"]["resource_sharing"]["by_agent_type"][agent_type]["actions"].append(data.get("actions", 0))
                    aggregated["metrics"]["resource_sharing"]["by_agent_type"][agent_type]["resources"].append(data.get("resources", 0))
        
        # Cooperation/competition metrics
        if "cooperation_competition" in metrics and "coop_comp_ratio" in metrics["cooperation_competition"]:
            cc = metrics["cooperation_competition"]
            
            coop_total = cc.get("cooperation", {}).get("total_actions", 0)
            comp_total = cc.get("competition", {}).get("total_actions", 0)
            
            aggregated["metrics"]["cooperation_competition"]["cooperation_total"].append(coop_total)
            aggregated["metrics"]["cooperation_competition"]["competition_total"].append(comp_total)
            
            # Calculate overall ratio
            if comp_total > 0:
                ratio = coop_total / comp_total
            else:
                ratio = float('inf') if coop_total > 0 else 0
            
            aggregated["metrics"]["cooperation_competition"]["coop_comp_ratio"].append(ratio)
            
            # Agent type metrics
            if "coop_comp_ratio" in cc:
                for agent_type, data in cc["coop_comp_ratio"].items():
                    aggregated["metrics"]["cooperation_competition"]["by_agent_type"][agent_type]["cooperation"].append(data.get("cooperation", 0))
                    aggregated["metrics"]["cooperation_competition"]["by_agent_type"][agent_type]["competition"].append(data.get("competition", 0))
                    aggregated["metrics"]["cooperation_competition"]["by_agent_type"][agent_type]["ratio"].append(data.get("ratio", 0))
        
        # Spatial clustering metrics
        if "spatial_clustering" in metrics and "error" not in metrics["spatial_clustering"]:
            sc = metrics["spatial_clustering"]
            
            aggregated["metrics"]["spatial_clustering"]["total_clusters"].append(sc.get("total_clusters", 0))
            aggregated["metrics"]["spatial_clustering"]["clustering_ratio"].append(sc.get("clustering_ratio", 0))
            aggregated["metrics"]["spatial_clustering"]["avg_cluster_size"].append(sc.get("avg_cluster_size", 0))
            
            # Agent type metrics
            if "agent_type_clustering" in sc:
                for agent_type, data in sc["agent_type_clustering"].items():
                    aggregated["metrics"]["spatial_clustering"]["by_agent_type"][agent_type]["clustering_ratio"].append(data.get("clustering_ratio", 0))
        
        # Reproduction patterns
        if "reproduction_patterns" in metrics and "error" not in metrics["reproduction_patterns"]:
            rp = metrics["reproduction_patterns"]
            
            aggregated["metrics"]["reproduction_patterns"]["total_events"].append(rp.get("total_events", 0))
            
            if "social_context" in rp:
                context = rp["social_context"]
                aggregated["metrics"]["reproduction_patterns"]["isolation_pct"].append(context.get("isolation_pct", 0))
                aggregated["metrics"]["reproduction_patterns"]["homogeneous_pct"].append(context.get("homogeneous_pct", 0))
                aggregated["metrics"]["reproduction_patterns"]["heterogeneous_pct"].append(context.get("heterogeneous_pct", 0))
            
            # Agent type metrics
            if "by_agent_type" in rp:
                for agent_type, count in rp["by_agent_type"].items():
                    aggregated["metrics"]["reproduction_patterns"]["by_agent_type"][agent_type]["count"].append(count)
    
    # Aggregate insights
    for result in results:
        if "insights" not in result:
            continue
            
        insights = result["insights"]
        
        # Key findings
        if "key_findings" in insights:
            for finding in insights["key_findings"]:
                finding_type = finding.get("type", "unknown")
                
                if finding_type not in aggregated["common_insights"]["key_findings"]:
                    aggregated["common_insights"]["key_findings"][finding_type] = {
                        "count": 0,
                        "descriptions": []
                    }
                
                aggregated["common_insights"]["key_findings"][finding_type]["count"] += 1
                aggregated["common_insights"]["key_findings"][finding_type]["descriptions"].append(finding.get("description", ""))
        
        # Emergent patterns
        if "emergent_patterns" in insights:
            for pattern in insights["emergent_patterns"]:
                pattern_type = pattern.get("type", "unknown")
                
                if pattern_type not in aggregated["common_insights"]["emergent_patterns"]:
                    aggregated["common_insights"]["emergent_patterns"][pattern_type] = {
                        "count": 0,
                        "descriptions": []
                    }
                
                aggregated["common_insights"]["emergent_patterns"][pattern_type]["count"] += 1
                aggregated["common_insights"]["emergent_patterns"][pattern_type]["descriptions"].append(pattern.get("description", ""))
        
        # Agent type insights
        if "agent_type_insights" in insights:
            for agent_type, type_insights in insights["agent_type_insights"].items():
                if agent_type not in aggregated["common_insights"]["agent_type_insights"]:
                    aggregated["common_insights"]["agent_type_insights"][agent_type] = {}
                
                for insight in type_insights:
                    # Extract a key from the insight text
                    key = insight.split('(')[0].strip().lower()
                    
                    if key not in aggregated["common_insights"]["agent_type_insights"][agent_type]:
                        aggregated["common_insights"]["agent_type_insights"][agent_type][key] = {
                            "count": 0,
                            "texts": []
                        }
                    
                    aggregated["common_insights"]["agent_type_insights"][agent_type][key]["count"] += 1
                    aggregated["common_insights"]["agent_type_insights"][agent_type][key]["texts"].append(insight)
        
        # Recommendations
        if "recommendations" in insights:
            for recommendation in insights["recommendations"]:
                # Extract a key from the recommendation text
                key = recommendation.split('.')[0].lower()
                
                if key not in aggregated["common_insights"]["recommendations"]:
                    aggregated["common_insights"]["recommendations"][key] = {
                        "count": 0,
                        "texts": []
                    }
                
                aggregated["common_insights"]["recommendations"][key]["count"] += 1
                aggregated["common_insights"]["recommendations"][key]["texts"].append(recommendation)
    
    # Calculate averages
    aggregated["averages"] = {
        "social_network": {
            "network_density": np.mean(aggregated["metrics"]["social_network"]["network_density"]) if aggregated["metrics"]["social_network"]["network_density"] else 0,
            "total_interactions": np.mean(aggregated["metrics"]["social_network"]["total_interactions"]) if aggregated["metrics"]["social_network"]["total_interactions"] else 0,
            "by_agent_type": {}
        },
        "resource_sharing": {
            "total_sharing_actions": np.mean(aggregated["metrics"]["resource_sharing"]["total_sharing_actions"]) if aggregated["metrics"]["resource_sharing"]["total_sharing_actions"] else 0,
            "total_resources_shared": np.mean(aggregated["metrics"]["resource_sharing"]["total_resources_shared"]) if aggregated["metrics"]["resource_sharing"]["total_resources_shared"] else 0,
            "by_agent_type": {}
        },
        "cooperation_competition": {
            "cooperation_total": np.mean(aggregated["metrics"]["cooperation_competition"]["cooperation_total"]) if aggregated["metrics"]["cooperation_competition"]["cooperation_total"] else 0,
            "competition_total": np.mean(aggregated["metrics"]["cooperation_competition"]["competition_total"]) if aggregated["metrics"]["cooperation_competition"]["competition_total"] else 0,
            "coop_comp_ratio": np.mean(aggregated["metrics"]["cooperation_competition"]["coop_comp_ratio"]) if aggregated["metrics"]["cooperation_competition"]["coop_comp_ratio"] else 0,
            "by_agent_type": {}
        },
        "spatial_clustering": {
            "total_clusters": np.mean(aggregated["metrics"]["spatial_clustering"]["total_clusters"]) if aggregated["metrics"]["spatial_clustering"]["total_clusters"] else 0,
            "clustering_ratio": np.mean(aggregated["metrics"]["spatial_clustering"]["clustering_ratio"]) if aggregated["metrics"]["spatial_clustering"]["clustering_ratio"] else 0,
            "by_agent_type": {}
        },
        "reproduction_patterns": {
            "total_events": np.mean(aggregated["metrics"]["reproduction_patterns"]["total_events"]) if aggregated["metrics"]["reproduction_patterns"]["total_events"] else 0,
            "isolation_pct": np.mean(aggregated["metrics"]["reproduction_patterns"]["isolation_pct"]) if aggregated["metrics"]["reproduction_patterns"]["isolation_pct"] else 0,
            "by_agent_type": {}
        }
    }
    
    # Calculate agent type averages
    for agent_type in agent_types:
        # Social network
        aggregated["averages"]["social_network"]["by_agent_type"][agent_type] = {}
        for metric in ["avg_out_degree", "avg_in_degree", "avg_total_outgoing", "avg_total_incoming"]:
            values = aggregated["metrics"]["social_network"]["by_agent_type"][agent_type][metric]
            aggregated["averages"]["social_network"]["by_agent_type"][agent_type][metric] = np.mean(values) if values else 0
        
        # Resource sharing
        aggregated["averages"]["resource_sharing"]["by_agent_type"][agent_type] = {}
        for metric in ["actions", "resources"]:
            values = aggregated["metrics"]["resource_sharing"]["by_agent_type"][agent_type][metric]
            aggregated["averages"]["resource_sharing"]["by_agent_type"][agent_type][metric] = np.mean(values) if values else 0
        
        # Cooperation/competition
        aggregated["averages"]["cooperation_competition"]["by_agent_type"][agent_type] = {}
        for metric in ["cooperation", "competition", "ratio"]:
            values = aggregated["metrics"]["cooperation_competition"]["by_agent_type"][agent_type][metric]
            aggregated["averages"]["cooperation_competition"]["by_agent_type"][agent_type][metric] = np.mean(values) if values else 0
        
        # Spatial clustering
        aggregated["averages"]["spatial_clustering"]["by_agent_type"][agent_type] = {}
        values = aggregated["metrics"]["spatial_clustering"]["by_agent_type"][agent_type]["clustering_ratio"]
        aggregated["averages"]["spatial_clustering"]["by_agent_type"][agent_type]["clustering_ratio"] = np.mean(values) if values else 0
        
        # Reproduction
        aggregated["averages"]["reproduction_patterns"]["by_agent_type"][agent_type] = {}
        values = aggregated["metrics"]["reproduction_patterns"]["by_agent_type"][agent_type]["count"]
        aggregated["averages"]["reproduction_patterns"]["by_agent_type"][agent_type]["count"] = np.mean(values) if values else 0
    
    # Add most common insights overall
    aggregated["most_common"] = {
        "key_findings": [],
        "emergent_patterns": [],
        "recommendations": []
    }
    
    # Most common key findings
    for finding_type, data in aggregated["common_insights"]["key_findings"].items():
        percentage = (data["count"] / len(results)) * 100
        if percentage >= 25:  # Appears in at least 25% of simulations
            most_frequent_description = max(set(data["descriptions"]), key=data["descriptions"].count)
            
            aggregated["most_common"]["key_findings"].append({
                "type": finding_type,
                "frequency_pct": percentage,
                "description": most_frequent_description
            })
    
    # Most common emergent patterns
    for pattern_type, data in aggregated["common_insights"]["emergent_patterns"].items():
        percentage = (data["count"] / len(results)) * 100
        if percentage >= 25:  # Appears in at least 25% of simulations
            most_frequent_description = max(set(data["descriptions"]), key=data["descriptions"].count)
            
            aggregated["most_common"]["emergent_patterns"].append({
                "type": pattern_type,
                "frequency_pct": percentage,
                "description": most_frequent_description
            })
    
    # Most common recommendations
    for rec_key, data in aggregated["common_insights"]["recommendations"].items():
        percentage = (data["count"] / len(results)) * 100
        if percentage >= 25:  # Appears in at least 25% of simulations
            most_frequent_text = max(set(data["texts"]), key=data["texts"].count)
            
            aggregated["most_common"]["recommendations"].append({
                "key": rec_key,
                "frequency_pct": percentage,
                "recommendation": most_frequent_text
            })
    
    # Sort by frequency
    for section in ["key_findings", "emergent_patterns", "recommendations"]:
        aggregated["most_common"][section].sort(key=lambda x: x["frequency_pct"], reverse=True)
    
    return aggregated 
#!/usr/bin/env python
"""
Social Behavior Analysis Script

This script analyzes social behaviors across agents in simulations, generating
metrics and insights about cooperation, competition, resource sharing, and spatial clustering.

Usage:
    python analyze_social_behaviors.py [--experiment_path PATH] [--output_path PATH]

The script will:
1. Analyze social behaviors in the most recent experiment or specified experiment path
2. Generate visualizations for different social behavior patterns
3. Create a comprehensive analysis report
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import analysis configuration
from analysis_config import (
    DATA_PATH,
    OUTPUT_PATH,
    safe_remove_directory,
    setup_logging,
    setup_analysis_directory,
    find_latest_experiment_path,
)

from farm.analysis.social_behavior.analyze import (
    analyze_social_behaviors,
    analyze_social_behaviors_across_simulations,
)

from farm.database.models import (
    AgentModel,
    AgentStateModel,
    ActionModel,
    HealthIncident,
    ReproductionEventModel,
    SocialInteractionModel,
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze social behaviors in simulations")
    parser.add_argument(
        "--experiment_path",
        type=str,
        help="Path to the experiment directory (default: most recent in DATA_PATH)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save results (default: social_behavior in OUTPUT_PATH)",
    )
    return parser.parse_args()


def plot_social_network(metrics, output_path):
    """Generate a visualization of the social network."""
    plt.figure(figsize=(10, 8))
    
    # Extract agent types and interaction counts
    agent_types = metrics.get("social_network", {}).get("agent_type_averages", {})
    if not agent_types or not isinstance(agent_types, dict):
        plt.title("No social network data available")
        plt.savefig(os.path.join(output_path, "social_network.png"))
        plt.close()
        return
    
    # Check if agent_types is properly structured
    if not all(isinstance(k, str) for k in agent_types.keys()):
        logging.warning("Social network agent_type_averages is not properly structured, skipping agent type plots")
        
        # Create a simple overall plot instead
        network_density = metrics.get("social_network", {}).get("network_density", 0)
        plt.figure(figsize=(8, 6))
        plt.pie([network_density, 1-network_density], 
                labels=['Connected', 'Potential Connections'], 
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F5F5F5'],
                startangle=90)
        plt.title(f'Social Network Density: {network_density:.3f}')
        plt.axis('equal')
        plt.savefig(os.path.join(output_path, "social_network_density.png"))
        plt.close()
        return
    
    # Create bar chart for average connection counts by agent type
    agent_types_list = list(agent_types.keys())
    out_degrees = [agent_types[at].get("avg_out_degree", 0) for at in agent_types_list]
    in_degrees = [agent_types[at].get("avg_in_degree", 0) for at in agent_types_list]
    
    x = np.arange(len(agent_types_list))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, out_degrees, width, label='Outgoing Connections')
    ax.bar(x + width/2, in_degrees, width, label='Incoming Connections')
    
    ax.set_xlabel('Agent Type')
    ax.set_ylabel('Average Connections')
    ax.set_title('Social Network Connections by Agent Type')
    ax.set_xticks(x)
    ax.set_xticklabels(agent_types_list)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "social_network_connections.png"))
    plt.close()
    
    # Create network density visualization
    network_density = metrics.get("social_network", {}).get("network_density", 0)
    
    plt.figure(figsize=(8, 6))
    plt.pie([network_density, 1-network_density], 
            labels=['Connected', 'Potential Connections'], 
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F5F5F5'],
            startangle=90)
    plt.title(f'Social Network Density: {network_density:.3f}')
    plt.axis('equal')
    plt.savefig(os.path.join(output_path, "social_network_density.png"))
    plt.close()


def plot_resource_sharing(metrics, output_path):
    """Generate visualizations of resource sharing patterns."""
    # Check if data is available
    if "resource_sharing" not in metrics or "error" in metrics["resource_sharing"]:
        plt.figure(figsize=(8, 6))
        plt.title("No resource sharing data available")
        plt.savefig(os.path.join(output_path, "resource_sharing.png"))
        plt.close()
        return
    
    resource_sharing = metrics["resource_sharing"]
    
    # Plot sharing by agent type
    if "by_agent_type" in resource_sharing:
        by_agent_type = resource_sharing["by_agent_type"]
        
        # Check if by_agent_type is properly structured
        if not isinstance(by_agent_type, dict) or not all(isinstance(k, str) for k in by_agent_type.keys()):
            logging.warning("Resource sharing by_agent_type is not properly structured, skipping agent type plots")
            
            # Create a simple overall plot instead
            plt.figure(figsize=(8, 6))
            total_resources = resource_sharing.get("total_resources_shared", 0)
            total_actions = resource_sharing.get("total_sharing_actions", 0)
            
            plt.bar(["Resources", "Actions"], [total_resources, total_actions], color=['#2196F3', '#FF9800'])
            plt.ylabel('Count')
            plt.title('Overall Resource Sharing')
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "resource_sharing_overall.png"))
            plt.close()
            return
        
        # Total resources shared by agent type
        agent_types = list(by_agent_type.keys())
        resources = [by_agent_type[at].get("resources", 0) for at in agent_types]
        actions = [by_agent_type[at].get("actions", 0) for at in agent_types]
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot resources
        ax1.bar(agent_types, resources, alpha=0.7, label='Resources Shared')
        ax1.set_xlabel('Agent Type')
        ax1.set_ylabel('Total Resources Shared')
        ax1.tick_params(axis='y')
        
        # Plot actions on secondary Y-axis
        ax2 = ax1.twinx()
        ax2.plot(agent_types, actions, 'ro-', label='Sharing Actions')
        ax2.set_ylabel('Number of Sharing Actions')
        ax2.tick_params(axis='y', labelcolor='r')
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.title('Resource Sharing by Agent Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "resource_sharing_by_type.png"))
        plt.close()
    
    # Plot sharing matrix (who shares with whom)
    if "sharing_matrix" in resource_sharing:
        sharing_matrix = resource_sharing["sharing_matrix"]
        
        # Check if sharing_matrix is properly structured
        if not isinstance(sharing_matrix, dict):
            logging.warning("Resource sharing matrix is not properly structured, skipping matrix plot")
            return
        
        # Convert to DataFrame for heatmap
        agent_types = set()
        for giver in sharing_matrix:
            if not isinstance(sharing_matrix[giver], dict):
                logging.warning(f"Resource sharing matrix entry for {giver} is not a dictionary, skipping matrix plot")
                return
                
            agent_types.add(giver)
            for receiver in sharing_matrix[giver]:
                agent_types.add(receiver)
        
        agent_types = sorted(list(agent_types))
        matrix_data = np.zeros((len(agent_types), len(agent_types)))
        
        for i, giver in enumerate(agent_types):
            for j, receiver in enumerate(agent_types):
                if giver in sharing_matrix and receiver in sharing_matrix[giver]:
                    matrix_data[i, j] = sharing_matrix[giver][receiver]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix_data, annot=True, fmt=".1f", xticklabels=agent_types, yticklabels=agent_types, cmap="YlGnBu")
        plt.title("Resource Sharing Matrix (From → To)")
        plt.xlabel("Receiver")
        plt.ylabel("Giver")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "resource_sharing_matrix.png"))
        plt.close()
    
    # Plot time distribution
    if "time_distribution" in resource_sharing:
        time_dist = resource_sharing["time_distribution"]
        
        # Check if time_dist is properly structured
        if not isinstance(time_dist, dict):
            logging.warning("Resource sharing time_distribution is not properly structured, skipping time distribution plot")
            return
        
        # Convert to sorted list of tuples
        try:
            time_points = sorted([(int(step_bucket)*100, count) for step_bucket, count in time_dist.items()])
            if time_points:
                steps, counts = zip(*time_points)
                
                plt.figure(figsize=(12, 6))
                plt.plot(steps, counts, 'b-', marker='o')
                plt.xlabel('Simulation Step')
                plt.ylabel('Number of Sharing Actions')
                plt.title('Resource Sharing Over Time')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_path, "resource_sharing_time.png"))
                plt.close()
        except Exception as e:
            logging.warning(f"Error plotting time distribution: {e}")


def plot_cooperation_competition(metrics, output_path):
    """Generate visualizations of cooperation and competition patterns."""
    # Check if data is available
    if "cooperation_competition" not in metrics:
        plt.figure(figsize=(8, 6))
        plt.title("No cooperation/competition data available")
        plt.savefig(os.path.join(output_path, "cooperation_competition.png"))
        plt.close()
        return
    
    coop_comp = metrics["cooperation_competition"]
    
    # Plot overall cooperation vs competition
    coop_total = coop_comp.get("cooperation_total", 0) if isinstance(coop_comp.get("cooperation"), dict) else coop_comp.get("cooperation_total", 0)
    comp_total = coop_comp.get("competition_total", 0) if isinstance(coop_comp.get("competition"), dict) else coop_comp.get("competition_total", 0)
    
    if coop_total > 0 or comp_total > 0:
        plt.figure(figsize=(8, 6))
        plt.pie([coop_total, comp_total], 
                labels=['Cooperation', 'Competition'], 
                autopct='%1.1f%%',
                colors=['#4CAF50', '#F44336'],
                explode=(0.1, 0),
                startangle=90)
        plt.title(f'Cooperation vs Competition Actions (Total: {coop_total + comp_total})')
        plt.axis('equal')
        plt.savefig(os.path.join(output_path, "coop_vs_comp_overall.png"))
        plt.close()
    
    # Plot cooperation/competition ratio by agent type
    if "coop_comp_ratio" in coop_comp:
        ratios = coop_comp["coop_comp_ratio"]
        
        # Check if ratios is a dictionary (with agent types as keys)
        if not isinstance(ratios, dict):
            # If it's not a dictionary (e.g., it's a numpy.float64), we can't plot by agent type
            logging.warning("Cooperation/competition ratio is not a dictionary, skipping agent type plots")
            
            # Plot the overall ratio as a single value
            plt.figure(figsize=(8, 6))
            plt.bar(["Overall"], [min(float(ratios), 10)], color='#2196F3')  # Cap at 10 for visualization
            plt.axhline(y=1, color='r', linestyle='-', alpha=0.5)
            plt.ylabel('Cooperation/Competition Ratio (capped at 10)')
            plt.title('Overall Cooperation to Competition Ratio')
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "coop_comp_ratio_overall.png"))
            plt.close()
            return
        
        agent_types = list(ratios.keys())
        ratio_values = []
        coop_values = []
        comp_values = []
        
        for agent_type in agent_types:
            ratio_values.append(min(ratios[agent_type]["ratio"], 10))  # Cap at 10 for visualization
            coop_values.append(ratios[agent_type]["cooperation"])
            comp_values.append(ratios[agent_type]["competition"])
        
        # Plot cooperation/competition by agent type
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(agent_types))
        width = 0.35
        
        ax.bar(x - width/2, coop_values, width, label='Cooperation Actions', color='#4CAF50')
        ax.bar(x + width/2, comp_values, width, label='Competition Actions', color='#F44336')
        
        ax.set_xlabel('Agent Type')
        ax.set_ylabel('Number of Actions')
        ax.set_title('Cooperation vs Competition by Agent Type')
        ax.set_xticks(x)
        ax.set_xticklabels(agent_types)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "coop_comp_by_type.png"))
        plt.close()
        
        # Plot cooperation/competition ratio
        plt.figure(figsize=(10, 6))
        plt.bar(agent_types, ratio_values, color='#2196F3')
        plt.axhline(y=1, color='r', linestyle='-', alpha=0.5)
        plt.xlabel('Agent Type')
        plt.ylabel('Cooperation/Competition Ratio (capped at 10)')
        plt.title('Cooperation to Competition Ratio by Agent Type')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "coop_comp_ratio.png"))
        plt.close()


def plot_spatial_clustering(metrics, output_path):
    """Generate visualizations of spatial clustering patterns."""
    # Check if data is available
    if "spatial_clustering" not in metrics or "error" in metrics["spatial_clustering"]:
        plt.figure(figsize=(8, 6))
        plt.title("No spatial clustering data available")
        plt.savefig(os.path.join(output_path, "spatial_clustering.png"))
        plt.close()
        return
    
    clustering = metrics["spatial_clustering"]
    
    # Plot clustering ratio by agent type
    if "agent_type_clustering" in clustering:
        clustering_by_type = clustering["agent_type_clustering"]
        
        # Check if clustering_by_type is properly structured
        if not isinstance(clustering_by_type, dict) or not all(isinstance(k, str) for k in clustering_by_type.keys()):
            logging.warning("Spatial clustering agent_type_clustering is not properly structured, skipping agent type plots")
            
            # Create a simple overall plot instead
            plt.figure(figsize=(8, 6))
            clustering_ratio = clustering.get("clustering_ratio", 0)
            plt.pie([clustering_ratio, 1-clustering_ratio], 
                    labels=['In Clusters', 'Isolated'], 
                    autopct='%1.1f%%',
                    colors=['#4CAF50', '#FFC107'],
                    startangle=90)
            plt.title('Overall Clustering vs Isolation')
            plt.axis('equal')
            plt.savefig(os.path.join(output_path, "clustering_overall.png"))
            plt.close()
            return
        
        agent_types = list(clustering_by_type.keys())
        clustering_ratios = [clustering_by_type[at].get("clustering_ratio", 0) for at in agent_types]
        isolated_ratios = [1 - clustering_by_type[at].get("clustering_ratio", 0) for at in agent_types]
        
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(agent_types))
        width = 0.35
        
        plt.bar(x, clustering_ratios, width, label='In Clusters', color='#4CAF50')
        plt.bar(x, isolated_ratios, width, bottom=clustering_ratios, label='Isolated', color='#FFC107')
        
        plt.xlabel('Agent Type')
        plt.ylabel('Proportion')
        plt.title('Clustering vs Isolation by Agent Type')
        plt.xticks(x, agent_types)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "clustering_by_type.png"))
        plt.close()
    
    # Plot cluster composition if available
    if "cluster_stats" in clustering:
        cluster_stats = clustering["cluster_stats"]
        
        # Check if cluster_stats is a list
        if not isinstance(cluster_stats, list):
            logging.warning("Spatial clustering cluster_stats is not properly structured, skipping cluster stats plots")
            return
        
        # Extract diversity index for each cluster
        try:
            cluster_ids = [stat.get("cluster_id", i) for i, stat in enumerate(cluster_stats)]
            diversity_indices = [stat.get("diversity_index", 0) for stat in cluster_stats]
            cluster_sizes = [stat.get("size", 0) for stat in cluster_stats]
            
            # Plot diversity index against cluster size
            plt.figure(figsize=(10, 6))
            plt.scatter(cluster_sizes, diversity_indices, alpha=0.7, s=100)
            
            plt.xlabel('Cluster Size')
            plt.ylabel('Diversity Index (Shannon Entropy)')
            plt.title('Cluster Size vs Diversity')
            plt.grid(True, alpha=0.3)
            
            # Add annotations for cluster IDs
            for i, cluster_id in enumerate(cluster_ids):
                plt.annotate(f"Cluster {cluster_id}", 
                            (cluster_sizes[i], diversity_indices[i]),
                            xytext=(5, 5),
                            textcoords='offset points')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "cluster_diversity.png"))
            plt.close()
            
            # Visualize the composition of largest clusters
            largest_clusters = sorted(cluster_stats, key=lambda x: x.get("size", 0), reverse=True)[:5]
            
            if largest_clusters:
                fig, axes = plt.subplots(1, len(largest_clusters), figsize=(4*len(largest_clusters), 5))
                if len(largest_clusters) == 1:
                    axes = [axes]
                
                for i, cluster in enumerate(largest_clusters):
                    if "type_composition" not in cluster or not isinstance(cluster["type_composition"], dict):
                        axes[i].text(0.5, 0.5, "No composition data", 
                                    horizontalalignment='center', verticalalignment='center')
                        axes[i].set_title(f'Cluster {cluster.get("cluster_id", i)}')
                        continue
                        
                    composition = cluster["type_composition"]
                    agent_types = list(composition.keys())
                    counts = list(composition.values())
                    
                    # Create pie chart
                    axes[i].pie(counts, labels=agent_types, autopct='%1.1f%%', startangle=90)
                    axes[i].set_title(f'Cluster {cluster.get("cluster_id", i)}\n(Size: {cluster.get("size", 0)})')
                    axes[i].axis('equal')
                
                plt.suptitle('Composition of Largest Clusters', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, "largest_clusters_composition.png"))
                plt.close()
        except Exception as e:
            logging.warning(f"Error plotting cluster stats: {e}")


def plot_reproduction_patterns(metrics, output_path):
    """Generate visualizations of reproduction social patterns."""
    # Check if data is available
    if "reproduction_patterns" not in metrics or "error" in metrics["reproduction_patterns"]:
        plt.figure(figsize=(8, 6))
        plt.title("No reproduction pattern data available")
        plt.savefig(os.path.join(output_path, "reproduction_patterns.png"))
        plt.close()
        return
    
    reproduction = metrics["reproduction_patterns"]
    
    # Plot reproduction events by agent type
    if "by_agent_type" in reproduction:
        by_agent_type = reproduction["by_agent_type"]
        
        # Check if by_agent_type is a dictionary with string keys and numeric values
        if not isinstance(by_agent_type, dict) or not all(isinstance(k, str) and (isinstance(v, (int, float)) or isinstance(v, dict)) for k, v in by_agent_type.items()):
            logging.warning("Reproduction by_agent_type is not properly structured, skipping agent type plots")
            
            # Create a simple overall plot instead
            plt.figure(figsize=(8, 6))
            plt.bar(["Overall"], [reproduction.get("total_events", 0)], color='#9C27B0')
            plt.ylabel('Number of Reproduction Events')
            plt.title('Total Reproduction Events')
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "reproduction_by_type_overall.png"))
            plt.close()
        else:
            # Normal processing when data is properly structured
            try:
                agent_types = []
                counts = []
                
                # Handle both simple dict {type: count} and nested dict {type: {count: value}}
                for agent_type, value in by_agent_type.items():
                    agent_types.append(agent_type)
                    if isinstance(value, dict):
                        # If it's a nested dictionary, try to extract a count value
                        if "count" in value:
                            counts.append(value["count"])
                        else:
                            # Use the first numeric value found
                            numeric_values = [v for k, v in value.items() if isinstance(v, (int, float))]
                            counts.append(numeric_values[0] if numeric_values else 0)
                    else:
                        # If it's a direct value
                        counts.append(value)
                
                plt.figure(figsize=(10, 6))
                plt.bar(agent_types, counts, color='#9C27B0')
                plt.xlabel('Agent Type')
                plt.ylabel('Number of Reproduction Events')
                plt.title('Reproduction Events by Agent Type')
                plt.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, "reproduction_by_type.png"))
                plt.close()
            except Exception as e:
                logging.warning(f"Error plotting reproduction by agent type: {e}")
                # Fallback to simple plot
                plt.figure(figsize=(8, 6))
                plt.bar(["Overall"], [reproduction.get("total_events", 0)], color='#9C27B0')
                plt.ylabel('Number of Reproduction Events')
                plt.title('Total Reproduction Events')
                plt.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_path, "reproduction_by_type_overall.png"))
                plt.close()
    
    # Plot social context of reproduction
    if "social_context" in reproduction:
        context = reproduction["social_context"]
        
        # Check if context is a dictionary
        if not isinstance(context, dict):
            logging.warning("Reproduction social_context is not properly structured, skipping context plots")
            return
        
        try:
            # Get values
            isolation = context.get("isolation", 0)
            homogeneous = context.get("homogeneous", 0)
            heterogeneous = context.get("heterogeneous", 0)
            
            labels = ['Isolation', 'Same-Type Group', 'Mixed Group']
            values = [isolation, homogeneous, heterogeneous]
            
            plt.figure(figsize=(10, 6))
            plt.pie(values, labels=labels, autopct='%1.1f%%', 
                    colors=['#FFC107', '#4CAF50', '#2196F3'], 
                    startangle=90)
            plt.title('Social Context of Reproduction')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "reproduction_social_context.png"))
            plt.close()
            
            # Plot social context by agent type if available
            if "by_agent_type" in context:
                by_type = context["by_agent_type"]
                
                # Check if by_type is properly structured
                if not isinstance(by_type, dict) or not all(isinstance(k, str) for k in by_type.keys()):
                    logging.warning("Reproduction social_context by_agent_type is not properly structured, skipping context by type plots")
                    return
                
                try:
                    agent_types = list(by_type.keys())
                    
                    # Create stacked bar chart
                    isolation_by_type = [by_type[at].get("isolation", 0) for at in agent_types]
                    homogeneous_by_type = [by_type[at].get("homogeneous", 0) for at in agent_types]
                    heterogeneous_by_type = [by_type[at].get("heterogeneous", 0) for at in agent_types]
                    
                    plt.figure(figsize=(12, 6))
                    width = 0.8
                    
                    plt.bar(agent_types, isolation_by_type, width, label='Isolation', color='#FFC107')
                    plt.bar(agent_types, homogeneous_by_type, width, bottom=isolation_by_type, 
                            label='Same-Type Group', color='#4CAF50')
                    
                    bottom_values = [i + h for i, h in zip(isolation_by_type, homogeneous_by_type)]
                    plt.bar(agent_types, heterogeneous_by_type, width, bottom=bottom_values, 
                            label='Mixed Group', color='#2196F3')
                    
                    plt.xlabel('Agent Type')
                    plt.ylabel('Number of Reproduction Events')
                    plt.title('Social Context of Reproduction by Agent Type')
                    plt.legend()
                    plt.grid(True, axis='y', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_path, "reproduction_context_by_type.png"))
                    plt.close()
                except Exception as e:
                    logging.warning(f"Error plotting reproduction social context by agent type: {e}")
        except Exception as e:
            logging.warning(f"Error plotting reproduction social context: {e}")


def plot_all_visualizations(metrics, output_path):
    """Generate all social behavior visualizations."""
    os.makedirs(output_path, exist_ok=True)
    
    plot_social_network(metrics, output_path)
    plot_resource_sharing(metrics, output_path)
    plot_cooperation_competition(metrics, output_path)
    plot_spatial_clustering(metrics, output_path)
    plot_reproduction_patterns(metrics, output_path)


def generate_summary_report(analysis_results, output_path):
    """Generate a summary report of the social behavior analysis."""
    metrics = analysis_results.get("metrics", {})
    insights = analysis_results.get("insights", {})
    
    report_path = os.path.join(output_path, "social_behavior_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Social Behavior Analysis Report\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # 1. Overview
        f.write("## 1. Overview\n\n")
        f.write("This report analyzes social behaviors between agents in the simulation, focusing on patterns of cooperation, competition, resource sharing, spatial clustering, and reproduction behaviors.\n\n")
        
        # 2. Key Findings
        f.write("## 2. Key Findings\n\n")
        
        if "key_findings" in insights and insights["key_findings"]:
            for finding in insights["key_findings"]:
                f.write(f"- **{finding['type'].replace('_', ' ').title()}**: {finding['description']}\n")
        else:
            f.write("No key findings identified.\n")
        f.write("\n")
        
        # 3. Social Network Analysis
        f.write("## 3. Social Network Analysis\n\n")
        
        if "social_network" in metrics and "error" not in metrics["social_network"]:
            sn = metrics["social_network"]
            
            f.write(f"- **Total Interactions**: {sn.get('total_interactions', 0)}\n")
            f.write(f"- **Unique Interaction Pairs**: {sn.get('unique_interaction_pairs', 0)}\n")
            
            # Fix for network_density potentially being a list
            network_density = sn.get('network_density', 0)
            if isinstance(network_density, list):
                network_density = network_density[0] if network_density else 0
            f.write(f"- **Network Density**: {network_density:.3f}\n\n")
            
            if "agent_type_averages" in sn:
                f.write("### Agent Type Connection Metrics\n\n")
                f.write("| Agent Type | Outgoing Connections | Incoming Connections | Total Outgoing | Total Incoming |\n")
                f.write("|------------|---------------------|---------------------|---------------|---------------|\n")
                
                for agent_type, data in sn["agent_type_averages"].items():
                    # Handle potential list values
                    avg_out = data.get('avg_out_degree', 0)
                    avg_in = data.get('avg_in_degree', 0)
                    total_out = data.get('avg_total_outgoing', 0)
                    total_in = data.get('avg_total_incoming', 0)
                    
                    # Convert lists to single values
                    if isinstance(avg_out, list): avg_out = avg_out[0] if avg_out else 0
                    if isinstance(avg_in, list): avg_in = avg_in[0] if avg_in else 0
                    if isinstance(total_out, list): total_out = total_out[0] if total_out else 0
                    if isinstance(total_in, list): total_in = total_in[0] if total_in else 0
                    
                    f.write(f"| {agent_type} | {avg_out:.2f} | {avg_in:.2f} | {total_out:.2f} | {total_in:.2f} |\n")
                
                f.write("\n")
        else:
            f.write("No social network data available.\n\n")
        
        # 4. Resource Sharing
        f.write("## 4. Resource Sharing\n\n")
        
        if "resource_sharing" in metrics and "error" not in metrics["resource_sharing"]:
            rs = metrics["resource_sharing"]
            
            # Handle potential list values
            total_actions = rs.get('total_sharing_actions', 0)
            total_resources = rs.get('total_resources_shared', 0)
            avg_resources = rs.get('avg_resources_per_share', 0)
            
            # Convert lists to single values
            if isinstance(total_actions, list): total_actions = total_actions[0] if total_actions else 0
            if isinstance(total_resources, list): total_resources = total_resources[0] if total_resources else 0
            if isinstance(avg_resources, list): avg_resources = avg_resources[0] if avg_resources else 0
            
            f.write(f"- **Total Sharing Actions**: {total_actions}\n")
            f.write(f"- **Total Resources Shared**: {total_resources:.2f}\n")
            f.write(f"- **Average Resources per Share**: {avg_resources:.2f}\n\n")
            
            if "by_agent_type" in rs:
                f.write("### Resource Sharing by Agent Type\n\n")
                f.write("| Agent Type | Actions | Resources Shared |\n")
                f.write("|------------|---------|------------------|\n")
                
                for agent_type, data in rs["by_agent_type"].items():
                    # Handle potential list values
                    actions = data.get('actions', 0)
                    resources = data.get('resources', 0)
                    
                    # Convert lists to single values
                    if isinstance(actions, list): actions = actions[0] if actions else 0
                    if isinstance(resources, list): resources = resources[0] if resources else 0
                    
                    f.write(f"| {agent_type} | {actions} | {resources:.2f} |\n")
                
                f.write("\n")
            
            if "altruistic_sharing" in rs:
                alt = rs["altruistic_sharing"]
                total = total_actions  # Using the already processed value
                
                # Handle potential list values
                alt_count = alt.get('count', 0)
                if isinstance(alt_count, list): alt_count = alt_count[0] if alt_count else 0
                
                if total > 0:
                    percentage = (alt_count / total) * 100
                    f.write(f"- **Altruistic Sharing**: {alt_count} actions ({percentage:.1f}% of all sharing)\n\n")
        else:
            f.write("No resource sharing data available.\n\n")
        
        # 5. Cooperation vs Competition
        f.write("## 5. Cooperation vs Competition\n\n")
        
        if "cooperation_competition" in metrics:
            cc = metrics["cooperation_competition"]
            
            # Handle potential nested dictionaries or direct values
            if isinstance(cc.get("cooperation"), dict):
                coop_total = cc.get("cooperation", {}).get("total_actions", 0)
            else:
                coop_total = cc.get("cooperation_total", 0)
                
            if isinstance(cc.get("competition"), dict):
                comp_total = cc.get("competition", {}).get("total_actions", 0)
            else:
                comp_total = cc.get("competition_total", 0)
            
            # Handle potential list values
            if isinstance(coop_total, list): coop_total = coop_total[0] if coop_total else 0
            if isinstance(comp_total, list): comp_total = comp_total[0] if comp_total else 0
            
            f.write(f"- **Cooperation Actions**: {coop_total}\n")
            f.write(f"- **Competition Actions**: {comp_total}\n")
            
            if comp_total > 0:
                ratio = coop_total / comp_total
                f.write(f"- **Cooperation-Competition Ratio**: {ratio:.2f}\n\n")
            else:
                f.write("- **Cooperation-Competition Ratio**: ∞ (No competition actions)\n\n")
            
            if "coop_comp_ratio" in cc:
                ratios = cc["coop_comp_ratio"]
                
                # Check if ratios is a dictionary (with agent types as keys)
                if isinstance(ratios, dict):
                    f.write("### Cooperation-Competition by Agent Type\n\n")
                    f.write("| Agent Type | Cooperation | Competition | Ratio |\n")
                    f.write("|------------|-------------|-------------|-------|\n")
                    
                    for agent_type, data in ratios.items():
                        # Handle potential list values
                        coop = data.get('cooperation', 0)
                        comp = data.get('competition', 0)
                        ratio_value = data.get('ratio', 0)
                        
                        # Convert lists to single values
                        if isinstance(coop, list): coop = coop[0] if coop else 0
                        if isinstance(comp, list): comp = comp[0] if comp else 0
                        if isinstance(ratio_value, list): ratio_value = ratio_value[0] if ratio_value else 0
                        
                        ratio_str = f"{ratio_value:.2f}" if ratio_value != float('inf') else "∞"
                        
                        f.write(f"| {agent_type} | {coop} | {comp} | {ratio_str} |\n")
                    
                    f.write("\n")
        else:
            f.write("No cooperation/competition data available.\n\n")
        
        # 6. Spatial Clustering
        f.write("## 6. Spatial Clustering\n\n")
        
        if "spatial_clustering" in metrics and "error" not in metrics["spatial_clustering"]:
            sc = metrics["spatial_clustering"]
            
            # Handle potential list values
            total_clusters = sc.get('total_clusters', 0)
            avg_cluster_size = sc.get('avg_cluster_size', 0)
            clustering_ratio = sc.get('clustering_ratio', 0)
            isolated_agents = sc.get('isolated_agents', 0)
            total_agents = sc.get('total_agents', 1)
            
            # Convert lists to single values
            if isinstance(total_clusters, list): total_clusters = total_clusters[0] if total_clusters else 0
            if isinstance(avg_cluster_size, list): avg_cluster_size = avg_cluster_size[0] if avg_cluster_size else 0
            if isinstance(clustering_ratio, list): clustering_ratio = clustering_ratio[0] if clustering_ratio else 0
            if isinstance(isolated_agents, list): isolated_agents = isolated_agents[0] if isolated_agents else 0
            if isinstance(total_agents, list): total_agents = total_agents[0] if total_agents else 1
            
            f.write(f"- **Total Clusters**: {total_clusters}\n")
            f.write(f"- **Average Cluster Size**: {avg_cluster_size:.2f}\n")
            f.write(f"- **Clustering Ratio**: {clustering_ratio:.2f}\n")
            f.write(f"- **Isolated Agents**: {isolated_agents} ({(isolated_agents/total_agents)*100:.1f}% of total)\n\n")
            
            if "agent_type_clustering" in sc:
                f.write("### Clustering by Agent Type\n\n")
                f.write("| Agent Type | Agents in Clusters | Total Agents | Clustering Ratio |\n")
                f.write("|------------|-------------------|--------------|------------------|\n")
                
                for agent_type, data in sc["agent_type_clustering"].items():
                    # Handle potential list values
                    agents_in_clusters = data.get('agents_in_clusters', 0)
                    total_agents_type = data.get('total_agents', 0)
                    clustering_ratio_type = data.get('clustering_ratio', 0)
                    
                    # Convert lists to single values
                    if isinstance(agents_in_clusters, list): agents_in_clusters = agents_in_clusters[0] if agents_in_clusters else 0
                    if isinstance(total_agents_type, list): total_agents_type = total_agents_type[0] if total_agents_type else 0
                    if isinstance(clustering_ratio_type, list): clustering_ratio_type = clustering_ratio_type[0] if clustering_ratio_type else 0
                    
                    f.write(f"| {agent_type} | {agents_in_clusters} | {total_agents_type} | {clustering_ratio_type:.2f} |\n")
                
                f.write("\n")
            
            if "cluster_stats" in sc:
                f.write("### Largest Clusters\n\n")
                
                # Ensure cluster_stats is a list
                cluster_stats = sc["cluster_stats"]
                if not isinstance(cluster_stats, list):
                    if isinstance(cluster_stats, dict):
                        # Convert dict to list if needed
                        cluster_stats = [cluster_stats]
                    else:
                        # Skip if not convertible
                        f.write("Cluster statistics data is not properly structured.\n\n")
                        return
                
                try:
                    largest_clusters = sorted(cluster_stats, key=lambda x: x.get("size", 0), reverse=True)[:5]
                    
                    for i, cluster in enumerate(largest_clusters, 1):
                        cluster_id = cluster.get('cluster_id', i)
                        size = cluster.get('size', 0)
                        diversity_index = cluster.get('diversity_index', 0)
                        
                        # Convert lists to single values
                        if isinstance(cluster_id, list): cluster_id = cluster_id[0] if cluster_id else i
                        if isinstance(size, list): size = size[0] if size else 0
                        if isinstance(diversity_index, list): diversity_index = diversity_index[0] if diversity_index else 0
                        
                        f.write(f"#### Cluster {cluster_id} (Size: {size})\n\n")
                        f.write(f"- **Diversity Index**: {diversity_index:.3f}\n")
                        f.write("- **Composition**:\n")
                        
                        if "type_composition" in cluster and isinstance(cluster["type_composition"], dict):
                            for agent_type, count in cluster["type_composition"].items():
                                # Convert list to single value
                                if isinstance(count, list): count = count[0] if count else 0
                                
                                percentage = (count / size) * 100 if size > 0 else 0
                                f.write(f"  - {agent_type}: {count} agents ({percentage:.1f}%)\n")
                        else:
                            f.write("  - Composition data not available\n")
                        
                        f.write("\n")
                except Exception as e:
                    f.write(f"Error processing cluster statistics: {str(e)}\n\n")
        else:
            f.write("No spatial clustering data available.\n\n")
        
        # 7. Reproduction Patterns
        f.write("## 7. Reproduction Social Patterns\n\n")
        
        if "reproduction_patterns" in metrics and "error" not in metrics["reproduction_patterns"]:
            rp = metrics["reproduction_patterns"]
            
            # Handle potential list values
            total_events = rp.get('total_events', 0)
            if isinstance(total_events, list): total_events = total_events[0] if total_events else 0
            
            f.write(f"- **Total Reproduction Events**: {total_events}\n\n")
            
            if "social_context" in rp:
                context = rp["social_context"]
                
                f.write("### Social Context of Reproduction\n\n")
                
                # Handle potential list values
                isolation = context.get('isolation', 0)
                homogeneous = context.get('homogeneous', 0)
                heterogeneous = context.get('heterogeneous', 0)
                isolation_pct = context.get('isolation_pct', 0)
                homogeneous_pct = context.get('homogeneous_pct', 0)
                heterogeneous_pct = context.get('heterogeneous_pct', 0)
                
                # Convert lists to single values
                if isinstance(isolation, list): isolation = isolation[0] if isolation else 0
                if isinstance(homogeneous, list): homogeneous = homogeneous[0] if homogeneous else 0
                if isinstance(heterogeneous, list): heterogeneous = heterogeneous[0] if heterogeneous else 0
                if isinstance(isolation_pct, list): isolation_pct = isolation_pct[0] if isolation_pct else 0
                if isinstance(homogeneous_pct, list): homogeneous_pct = homogeneous_pct[0] if homogeneous_pct else 0
                if isinstance(heterogeneous_pct, list): heterogeneous_pct = heterogeneous_pct[0] if heterogeneous_pct else 0
                
                f.write(f"- **Isolation**: {isolation} events ({isolation_pct:.1f}%)\n")
                f.write(f"- **Same-Type Group**: {homogeneous} events ({homogeneous_pct:.1f}%)\n")
                f.write(f"- **Mixed Group**: {heterogeneous} events ({heterogeneous_pct:.1f}%)\n\n")
            
            if "by_agent_type" in rp:
                f.write("### Reproduction by Agent Type\n\n")
                f.write("| Agent Type | Reproduction Events |\n")
                f.write("|------------|---------------------|\n")
                
                by_agent_type = rp["by_agent_type"]
                if isinstance(by_agent_type, dict):
                    for agent_type, count in by_agent_type.items():
                        # Convert list to single value
                        if isinstance(count, list): count = count[0] if count else 0
                        elif isinstance(count, dict):
                            # If it's a nested dictionary, try to extract a count value
                            if "count" in count:
                                count = count["count"]
                            else:
                                # Use the first numeric value found
                                numeric_values = [v for k, v in count.items() if isinstance(v, (int, float))]
                                count = numeric_values[0] if numeric_values else 0
                        
                        f.write(f"| {agent_type} | {count} |\n")
                
                f.write("\n")
        else:
            f.write("No reproduction pattern data available.\n\n")
        
        # 8. Emergent Patterns
        f.write("## 8. Emergent Patterns\n\n")
        
        if "emergent_patterns" in insights and insights["emergent_patterns"]:
            # Ensure emergent_patterns is a list
            emergent_patterns = insights["emergent_patterns"]
            if not isinstance(emergent_patterns, list):
                if isinstance(emergent_patterns, dict):
                    # Convert dict to list if needed
                    emergent_patterns = [emergent_patterns]
                else:
                    emergent_patterns = []
            
            for pattern in emergent_patterns:
                if isinstance(pattern, dict):
                    pattern_type = pattern.get('type', 'Unknown')
                    description = pattern.get('description', 'No description available')
                    
                    # Convert lists to single values
                    if isinstance(pattern_type, list): pattern_type = pattern_type[0] if pattern_type else 'Unknown'
                    if isinstance(description, list): description = description[0] if description else 'No description available'
                    
                    f.write(f"- **{pattern_type.replace('_', ' ').title()}**: {description}\n")
                else:
                    f.write(f"- {pattern}\n")
        else:
            f.write("No emergent patterns identified.\n")
        f.write("\n")
        
        # 9. Agent Type Insights
        f.write("## 9. Agent Type Insights\n\n")
        
        if "agent_type_insights" in insights and insights["agent_type_insights"]:
            agent_type_insights = insights["agent_type_insights"]
            
            # Check if agent_type_insights is a dictionary
            if isinstance(agent_type_insights, dict):
                for agent_type, type_insights in agent_type_insights.items():
                    f.write(f"### {agent_type.capitalize()} Agent\n\n")
                    
                    # Ensure type_insights is a list
                    if not isinstance(type_insights, list):
                        if isinstance(type_insights, dict):
                            # If it's a dictionary, extract values
                            type_insights = list(type_insights.values())
                        elif isinstance(type_insights, str):
                            # If it's a string, make it a single-item list
                            type_insights = [type_insights]
                        else:
                            type_insights = []
                    
                    for insight in type_insights:
                        # Handle if insight is a dict with a description field
                        if isinstance(insight, dict) and 'description' in insight:
                            insight_text = insight['description']
                            if isinstance(insight_text, list): insight_text = insight_text[0] if insight_text else ''
                            f.write(f"- {insight_text}\n")
                        else:
                            # Handle if insight is a string or convert to string
                            if isinstance(insight, list): insight = insight[0] if insight else ''
                            f.write(f"- {insight}\n")
                    
                    f.write("\n")
            else:
                f.write("Agent type insights data is not properly structured.\n\n")
        else:
            f.write("No agent type insights available.\n\n")
        
        # 10. Recommendations
        f.write("## 10. Recommendations\n\n")
        
        if "recommendations" in insights and insights["recommendations"]:
            # Ensure recommendations is a list
            recommendations = insights["recommendations"]
            if not isinstance(recommendations, list):
                if isinstance(recommendations, dict):
                    # If it's a dictionary, extract values
                    recommendations = list(recommendations.values())
                elif isinstance(recommendations, str):
                    # If it's a string, make it a single-item list
                    recommendations = [recommendations]
                else:
                    recommendations = []
            
            for recommendation in recommendations:
                # Handle if recommendation is a dict with a description field
                if isinstance(recommendation, dict) and 'description' in recommendation:
                    rec_text = recommendation['description']
                    if isinstance(rec_text, list): rec_text = rec_text[0] if rec_text else ''
                    f.write(f"- {rec_text}\n")
                else:
                    # Handle if recommendation is a string or convert to string
                    if isinstance(recommendation, list): recommendation = recommendation[0] if recommendation else ''
                    f.write(f"- {recommendation}\n")
        else:
            f.write("No recommendations available.\n")
    
    return report_path


def main():
    """Run the social behavior analysis."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup output directory
    social_behavior_output_path = args.output_path
    if not social_behavior_output_path:
        # Use the setup_analysis_directory function
        social_behavior_output_path, log_file = setup_analysis_directory("social")
    else:
        # If custom output path is provided, still set up logging
        os.makedirs(social_behavior_output_path, exist_ok=True)
        log_file = setup_logging(social_behavior_output_path)
    
    start_time = time.time()
    logging.info("Starting social behavior analysis script")
    
    try:
        # Determine the experiment path
        experiment_path = args.experiment_path
        if not experiment_path:
            # Use the find_latest_experiment_path function
            experiment_path = find_latest_experiment_path()
            if not experiment_path:
                return
        else:
            logging.info(f"Using experiment path: {experiment_path}")
        
        # Run analysis across simulations
        logging.info("Starting analysis across simulations")
        analysis_results = analyze_social_behaviors_across_simulations(
            experiment_path, social_behavior_output_path
        )
        
        if "error" in analysis_results:
            logging.error(f"Error in analysis: {analysis_results['error']}")
            return
        
        # Generate visualizations
        logging.info("Generating visualizations")
        plot_all_visualizations(analysis_results["averages"], social_behavior_output_path)
        
        # Generate summary report
        logging.info("Generating summary report")
        report_path = generate_summary_report(analysis_results, social_behavior_output_path)
        
        # Log completion
        duration = time.time() - start_time
        logging.info(f"Completed social behavior analysis in {duration:.2f} seconds")
        logging.info(f"Results saved to {social_behavior_output_path}")
        logging.info(f"Summary report saved to {report_path}")
        
    except Exception as e:
        logging.error(f"Error in social behavior analysis: {e}")
        import traceback
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main() 
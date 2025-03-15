"""
Genesis Analysis Visualization Module

This module provides functions to visualize analysis results from the Genesis module,
including initial state comparisons, critical period analysis, and overall simulation trends.
"""

import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Set the style globally for all plots
plt.style.use('seaborn-v0_8')  # Using a valid style name

def plot_initial_state_comparison(simulations: List[Dict[str, Any]], output_path: str):
    """
    Generate visualizations comparing initial states across simulations.
    
    Parameters
    ----------
    simulations : List[Dict[str, Any]]
        List of simulation results
    output_path : str
        Directory to save visualizations
    """
    logger.info(f"Starting initial state comparison plot with {len(simulations)} simulations")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Debug: Print simulation data structure
    for i, sim in enumerate(simulations):
        logger.info(f"Simulation {i} keys: {list(sim.keys())}")
        if "results" in sim:
            logger.info(f"Simulation {i} results keys: {list(sim['results'].keys())}")
            if "initial_metrics" in sim["results"]:
                logger.info(f"Simulation {i} initial_metrics keys: {list(sim['results']['initial_metrics'].keys())}")
    
    # 1. Initial Resource Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    resource_data = []
    for sim in simulations:
        if "results" in sim and "initial_metrics" in sim["results"]:
            metrics = sim["results"]["initial_metrics"]
            if "agent_starting_attributes" in metrics:
                for agent_type, attrs in metrics["agent_starting_attributes"].items():
                    if "avg_initial_resources" in attrs:
                        resource_data.append({
                            "Simulation": f"Sim {sim.get('iteration', 'unknown')}",
                            "Agent Type": agent_type,
                            "Initial Resources": attrs["avg_initial_resources"]
                        })
    
    logger.info(f"Resource distribution data points: {len(resource_data)}")
    if resource_data:
        df_resources = pd.DataFrame(resource_data)
        sns.boxplot(data=df_resources, x="Agent Type", y="Initial Resources", ax=ax1)
        ax1.set_title("Initial Resource Distribution by Agent Type")
        ax1.tick_params(axis='x', rotation=45)
    else:
        logger.warning("No resource distribution data available for plotting")
    
    # 2. Initial Population Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    pop_data = []
    for sim in simulations:
        if "initial_metrics" in sim["results"]:
            metrics = sim["results"]["initial_metrics"]
            if "agent_starting_attributes" in metrics:
                for agent_type, attrs in metrics["agent_starting_attributes"].items():
                    if "count" in attrs:
                        pop_data.append({
                            "Simulation": f"Sim {sim['iteration']}",
                            "Agent Type": agent_type,
                            "Count": attrs["count"]
                        })
    
    if pop_data:
        df_pop = pd.DataFrame(pop_data)
        sns.barplot(data=df_pop, x="Agent Type", y="Count", ax=ax2)
        ax2.set_title("Initial Population Distribution")
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Initial Health Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    health_data = []
    for sim in simulations:
        if "initial_metrics" in sim["results"]:
            metrics = sim["results"]["initial_metrics"]
            if "agent_starting_attributes" in metrics:
                for agent_type, attrs in metrics["agent_starting_attributes"].items():
                    if "avg_starting_health" in attrs:
                        health_data.append({
                            "Simulation": f"Sim {sim['iteration']}",
                            "Agent Type": agent_type,
                            "Starting Health": attrs["avg_starting_health"]
                        })
    
    if health_data:
        df_health = pd.DataFrame(health_data)
        sns.boxplot(data=df_health, x="Agent Type", y="Starting Health", ax=ax3)
        ax3.set_title("Initial Health Distribution by Agent Type")
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Resource Proximity
    ax4 = fig.add_subplot(gs[1, 1])
    proximity_data = []
    for sim in simulations:
        if "initial_metrics" in sim["results"]:
            metrics = sim["results"]["initial_metrics"]
            if "agent_type_resource_proximity" in metrics:
                for agent_type, prox in metrics["agent_type_resource_proximity"].items():
                    if "avg_distance" in prox:
                        proximity_data.append({
                            "Simulation": f"Sim {sim['iteration']}",
                            "Agent Type": agent_type,
                            "Average Distance": prox["avg_distance"]
                        })
    
    if proximity_data:
        df_prox = pd.DataFrame(proximity_data)
        sns.boxplot(data=df_prox, x="Agent Type", y="Average Distance", ax=ax4)
        ax4.set_title("Initial Resource Proximity by Agent Type")
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "initial_state_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_critical_period_analysis(simulations: List[Dict[str, Any]], output_path: str):
    """
    Generate visualizations analyzing the critical period across simulations.
    
    Parameters
    ----------
    simulations : List[Dict[str, Any]]
        List of simulation results
    output_path : str
        Directory to save visualizations
    """
    logger.info(f"Starting critical period analysis plot with {len(simulations)} simulations")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Debug: Print data structure
    for i, sim in enumerate(simulations):
        logger.info(f"Simulation {i} keys: {list(sim.keys())}")
        if "results" in sim:
            logger.info(f"Simulation {i} results keys: {list(sim['results'].keys())}")
            if "critical_period" in sim["results"]:
                logger.info(f"Simulation {i} critical_period keys: {list(sim['results']['critical_period'].keys())}")
    
    # 1. Population Growth Rates
    ax1 = fig.add_subplot(gs[0, 0])
    growth_data = []
    for sim in simulations:
        if "results" in sim and "critical_period" in sim["results"]:
            metrics = sim["results"]["critical_period"]
            for key, value in metrics.items():
                if key.endswith("_growth_rate"):
                    agent_type = key.replace("_growth_rate", "")
                    growth_data.append({
                        "Simulation": f"Sim {sim.get('iteration', 'unknown')}",
                        "Agent Type": agent_type,
                        "Growth Rate": value
                    })
    
    logger.info(f"Growth rate data points: {len(growth_data)}")
    if growth_data:
        df_growth = pd.DataFrame(growth_data)
        sns.boxplot(data=df_growth, x="Agent Type", y="Growth Rate", ax=ax1)
        ax1.set_title("Population Growth Rates in Critical Period")
        ax1.tick_params(axis='x', rotation=45)
    else:
        logger.warning("No growth rate data available for plotting")
    
    # 2. Early Deaths
    ax2 = fig.add_subplot(gs[0, 1])
    death_data = []
    for sim in simulations:
        if "results" in sim and "critical_period" in sim["results"]:
            metrics = sim["results"]["critical_period"]
            if "early_deaths" in metrics:
                for agent_type, deaths in metrics["early_deaths"].items():
                    death_data.append({
                        "Simulation": f"Sim {sim.get('iteration', 'unknown')}",
                        "Agent Type": agent_type,
                        "Early Deaths": deaths
                    })
    
    logger.info(f"Early deaths data points: {len(death_data)}")
    if death_data:
        df_deaths = pd.DataFrame(death_data)
        sns.barplot(data=df_deaths, x="Agent Type", y="Early Deaths", ax=ax2)
        ax2.set_title("Early Deaths by Agent Type")
        ax2.tick_params(axis='x', rotation=45)
    else:
        logger.warning("No early deaths data available for plotting")
    
    # 3. Resource Acquisition
    ax3 = fig.add_subplot(gs[1, 0])
    resource_data = []
    for sim in simulations:
        if "results" in sim and "critical_period" in sim["results"]:
            metrics = sim["results"]["critical_period"]
            for key, value in metrics.items():
                if key.endswith("_avg_resources"):
                    agent_type = key.replace("_avg_resources", "")
                    resource_data.append({
                        "Simulation": f"Sim {sim.get('iteration', 'unknown')}",
                        "Agent Type": agent_type,
                        "Average Resources": value
                    })
    
    logger.info(f"Resource acquisition data points: {len(resource_data)}")
    if resource_data:
        df_resources = pd.DataFrame(resource_data)
        sns.boxplot(data=df_resources, x="Agent Type", y="Average Resources", ax=ax3)
        ax3.set_title("Resource Acquisition in Critical Period")
        ax3.tick_params(axis='x', rotation=45)
    else:
        logger.warning("No resource acquisition data available for plotting")
    
    # 4. First Reproduction Timing
    ax4 = fig.add_subplot(gs[1, 1])
    repro_data = []
    for sim in simulations:
        if "results" in sim and "critical_period" in sim["results"]:
            metrics = sim["results"]["critical_period"]
            if "first_reproduction_events" in metrics:
                for agent_type, step in metrics["first_reproduction_events"].items():
                    repro_data.append({
                        "Simulation": f"Sim {sim.get('iteration', 'unknown')}",
                        "Agent Type": agent_type,
                        "First Reproduction Step": step
                    })
    
    logger.info(f"First reproduction data points: {len(repro_data)}")
    if repro_data:
        df_repro = pd.DataFrame(repro_data)
        sns.boxplot(data=df_repro, x="Agent Type", y="First Reproduction Step", ax=ax4)
        ax4.set_title("First Reproduction Timing by Agent Type")
        ax4.tick_params(axis='x', rotation=45)
    else:
        logger.warning("No first reproduction data available for plotting")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "critical_period_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_genesis_analysis_results(results: Dict[str, Any], output_path: str):
    """
    Generate comprehensive visualizations of Genesis analysis results.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Genesis analysis results
    output_path : str
        Directory to save visualizations
    """
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(3, 2, figure=fig)
    
    # 1. Critical Period Trends
    ax1 = fig.add_subplot(gs[0, :])
    
    # Debug data structure
    logger.info(f"Results keys: {list(results.keys())}")
    if 'simulations' in results:
        logger.info(f"Number of simulations: {len(results['simulations'])}")
        for i, sim in enumerate(results['simulations']):
            logger.info(f"Simulation {i} keys: {list(sim.keys())}")
            if 'results' in sim:
                logger.info(f"Simulation {i} results keys: {list(sim['results'].keys())}")
                if 'critical_period' in sim['results']:
                    logger.info(f"Simulation {i} critical_period keys: {list(sim['results']['critical_period'].keys())}")
    
    # Collect trend data
    trend_data = {
        'Survival Rate': [],
        'Reproduction Rate': [],
        'Resource Efficiency': []
    }
    
    # Get simulation indices
    steps = range(len(results.get('simulations', [])))
    
    # Collect metrics from each simulation
    for sim in results.get('simulations', []):
        if 'results' in sim and 'critical_period' in sim['results']:
            metrics = sim['results']['critical_period']
            
            # Get survival rate directly
            trend_data['Survival Rate'].append(metrics.get('survival_rate', 0.0))
            
            # Get reproduction rate directly
            trend_data['Reproduction Rate'].append(metrics.get('reproduction_rate', 0.0))
            
            # Get resource efficiency directly
            trend_data['Resource Efficiency'].append(metrics.get('resource_efficiency', 0.0))
    
    # Log the collected data for debugging
    logger.info(f"Trend data points: {[len(v) for k,v in trend_data.items()]}")
    logger.info(f"Survival Rate values: {trend_data['Survival Rate']}")
    logger.info(f"Reproduction Rate values: {trend_data['Reproduction Rate']}")
    logger.info(f"Resource Efficiency values: {trend_data['Resource Efficiency']}")
    
    # Plot trends
    if steps and any(len(data) > 0 for data in trend_data.values()):
        for label, data in trend_data.items():
            if data:  # Only plot if we have data
                ax1.plot(steps, data, label=label)
        
        ax1.set_xlabel('Simulation Index')
        ax1.set_ylabel('Rate')
        ax1.set_title('Critical Period Metrics Across Simulations')
        ax1.legend()
        ax1.grid(True)
        
        # Set y-axis limits to show variation better
        min_val = min(min(data) for data in trend_data.values() if data)
        max_val = max(max(data) for data in trend_data.values() if data)
        if min_val != max_val:
            padding = (max_val - min_val) * 0.1
            ax1.set_ylim(min_val - padding, max_val + padding)
    else:
        logger.warning("No trend data available for plotting")
    
    # Save critical period trends separately
    plt.figure(figsize=(12, 6))
    if steps and any(len(data) > 0 for data in trend_data.values()):
        for label, data in trend_data.items():
            if data:  # Only plot if we have data
                plt.plot(steps, data, label=label)
        
        plt.xlabel('Simulation Index')
        plt.ylabel('Rate')
        plt.title('Critical Period Metrics Across Simulations')
        plt.legend()
        plt.grid(True)
        
        # Set y-axis limits to show variation better
        min_val = min(min(data) for data in trend_data.values() if data)
        max_val = max(max(data) for data in trend_data.values() if data)
        if min_val != max_val:
            padding = (max_val - min_val) * 0.1
            plt.ylim(min_val - padding, max_val + padding)
        
        plt.savefig(os.path.join(output_path, 'critical_period_trends.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Continue with the rest of the plots...
    # 2. Initial Resource Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    resource_data = []
    
    for sim in results.get('simulations', []):
        if 'results' in sim and 'initial_metrics' in sim['results']:
            metrics = sim['results']['initial_metrics']
            if 'agent_starting_attributes' in metrics:
                for agent_type, attrs in metrics['agent_starting_attributes'].items():
                    if 'avg_initial_resources' in attrs:
                        resource_data.append({
                            'Agent Type': agent_type,
                            'Initial Resources': attrs['avg_initial_resources']
                        })
    
    if resource_data:
        df_resources = pd.DataFrame(resource_data)
        sns.boxplot(data=df_resources, x='Agent Type', y='Initial Resources', ax=ax2)
        ax2.set_title('Initial Resource Distribution by Agent Type')
        ax2.tick_params(axis='x', rotation=45)
    
    # 3. Initial Population Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    pop_data = []
    
    for sim in results.get('simulations', []):
        if 'results' in sim and 'initial_metrics' in sim['results']:
            metrics = sim['results']['initial_metrics']
            if 'agent_starting_attributes' in metrics:
                for agent_type, attrs in metrics['agent_starting_attributes'].items():
                    if 'count' in attrs:
                        pop_data.append({
                            'Agent Type': agent_type,
                            'Count': attrs['count']
                        })
    
    if pop_data:
        df_pop = pd.DataFrame(pop_data)
        sns.barplot(data=df_pop, x='Agent Type', y='Count', ax=ax3)
        ax3.set_title('Initial Population Distribution')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Resource Proximity
    ax4 = fig.add_subplot(gs[2, :])
    proximity_data = []
    
    for sim in results.get('simulations', []):
        if 'results' in sim and 'initial_metrics' in sim['results']:
            metrics = sim['results']['initial_metrics']
            if 'agent_type_resource_proximity' in metrics:
                for agent_type, prox in metrics['agent_type_resource_proximity'].items():
                    if 'avg_distance' in prox:
                        proximity_data.append({
                            'Agent Type': agent_type,
                            'Average Distance': prox['avg_distance']
                        })
    
    if proximity_data:
        df_prox = pd.DataFrame(proximity_data)
        sns.boxplot(data=df_prox, x='Agent Type', y='Average Distance', ax=ax4)
        ax4.set_title('Initial Resource Proximity by Agent Type')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'genesis_analysis_results.png'), dpi=300, bbox_inches='tight')
    plt.close() 
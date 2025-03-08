#!/usr/bin/env python3
"""
controlled_initial_conditions.py

This script creates a set of controlled experiments to systematically test
how different initial conditions affect agent dominance outcomes.

It generates configuration files with controlled variations in:
1. Initial agent positions relative to resources
2. Resource distribution patterns
3. Equal vs. unequal resource access

Usage:
    python controlled_initial_conditions.py
"""

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.spatial.distance import euclidean

# Base configuration template
BASE_CONFIG = {
    "width": 100,
    "height": 100,
    "system_agents": 1,
    "independent_agents": 1,
    "control_agents": 1,
    "initial_resource_level": 5,
    "max_population": 300,
    "starvation_threshold": 0,
    "max_starvation_time": 15,
    "offspring_cost": 3,
    "min_reproduction_resources": 8,
    "offspring_initial_resources": 5,
    "perception_radius": 2,
    "base_attack_strength": 2,
    "base_defense_strength": 2,
    "seed": None,
    "initial_resources": 20,
    "resource_regen_rate": 0.1,
    "resource_regen_amount": 2,
    "max_resource_amount": 30,
    "base_consumption_rate": 0.15,
    "max_movement": 8,
    "gathering_range": 30,
    "max_gather_amount": 3,
    "territory_range": 30,
    "simulation_steps": 2000,
    "agent_parameters": {
        "SystemAgent": {
            "gather_efficiency_multiplier": 0.4,
            "gather_cost_multiplier": 0.4,
            "min_resource_threshold": 0.2,
            "share_weight": 0.3,
            "attack_weight": 0.05,
        },
        "IndependentAgent": {
            "gather_efficiency_multiplier": 0.7,
            "gather_cost_multiplier": 0.2,
            "min_resource_threshold": 0.05,
            "share_weight": 0.05,
            "attack_weight": 0.25,
        },
        "ControlAgent": {
            "gather_efficiency_multiplier": 0.55,
            "gather_cost_multiplier": 0.3,
            "min_resource_threshold": 0.125,
            "share_weight": 0.15,
            "attack_weight": 0.15,
        },
    },
}


def generate_resource_positions(pattern, num_resources=20, grid_size=(100, 100)):
    """
    Generate resource positions based on different patterns.

    Patterns:
    - 'random': Randomly distributed resources
    - 'clustered': Resources clustered in specific areas
    - 'uniform_grid': Resources arranged in a uniform grid
    - 'central': Resources concentrated in the center
    - 'corners': Resources concentrated in the corners
    """
    width, height = grid_size
    positions = []

    if pattern == "random":
        # Random distribution
        for _ in range(num_resources):
            x = random.uniform(10, width - 10)
            y = random.uniform(10, height - 10)
            amount = random.uniform(10, 30)
            positions.append((x, y, amount))

    elif pattern == "clustered":
        # Create 3-4 clusters
        num_clusters = random.randint(3, 4)
        cluster_centers = []

        # Generate cluster centers
        for _ in range(num_clusters):
            x = random.uniform(20, width - 20)
            y = random.uniform(20, height - 20)
            cluster_centers.append((x, y))

        # Distribute resources around cluster centers
        for _ in range(num_resources):
            center = random.choice(cluster_centers)
            # Generate position with normal distribution around center
            x = np.random.normal(center[0], 10)
            y = np.random.normal(center[1], 10)
            # Ensure within bounds
            x = max(5, min(width - 5, x))
            y = max(5, min(height - 5, y))
            amount = random.uniform(10, 30)
            positions.append((x, y, amount))

    elif pattern == "uniform_grid":
        # Arrange in a grid
        grid_dim = int(np.ceil(np.sqrt(num_resources)))
        x_step = width / (grid_dim + 1)
        y_step = height / (grid_dim + 1)

        count = 0
        for i in range(1, grid_dim + 1):
            for j in range(1, grid_dim + 1):
                if count < num_resources:
                    x = i * x_step
                    y = j * y_step
                    # Add small random offset
                    x += random.uniform(-5, 5)
                    y += random.uniform(-5, 5)
                    amount = random.uniform(10, 30)
                    positions.append((x, y, amount))
                    count += 1

    elif pattern == "central":
        # Concentrate resources in the center
        center_x, center_y = width / 2, height / 2
        for _ in range(num_resources):
            # Distance from center follows exponential distribution
            distance = random.expovariate(1 / 15)  # Mean distance of 15
            angle = random.uniform(0, 2 * np.pi)
            x = center_x + distance * np.cos(angle)
            y = center_y + distance * np.sin(angle)
            # Ensure within bounds
            x = max(5, min(width - 5, x))
            y = max(5, min(height - 5, y))
            amount = random.uniform(10, 30)
            positions.append((x, y, amount))

    elif pattern == "corners":
        # Concentrate resources in the corners
        corners = [
            (10, 10),
            (10, height - 10),
            (width - 10, 10),
            (width - 10, height - 10),
        ]
        resources_per_corner = num_resources // 4
        extra = num_resources % 4

        for i, corner in enumerate(corners):
            corner_resources = resources_per_corner + (1 if i < extra else 0)
            for _ in range(corner_resources):
                # Distance from corner follows exponential distribution
                distance = random.expovariate(1 / 15)  # Mean distance of 15
                angle = random.uniform(0, 2 * np.pi)
                x = corner[0] + distance * np.cos(angle)
                y = corner[1] + distance * np.sin(angle)
                # Ensure within bounds
                x = max(5, min(width - 5, x))
                y = max(5, min(height - 5, y))
                amount = random.uniform(10, 30)
                positions.append((x, y, amount))

    return positions


def generate_agent_positions(scenario, resource_positions, grid_size=(100, 100)):
    """
    Generate agent positions based on different scenarios.

    Scenarios:
    - 'equal_distance': All agents at equal distances from resources
    - 'one_advantaged': One agent type has advantage (closer to resources)
    - 'one_disadvantaged': One agent type has disadvantage (farther from resources)
    - 'gradient_advantage': Gradient of advantage (system > control > independent)
    - 'random': Random positions for all agents
    """
    width, height = grid_size
    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]

    # Calculate resource centroid
    if resource_positions:
        centroid_x = sum(pos[0] for pos in resource_positions) / len(resource_positions)
        centroid_y = sum(pos[1] for pos in resource_positions) / len(resource_positions)
        resource_centroid = (centroid_x, centroid_y)
    else:
        resource_centroid = (width / 2, height / 2)

    if scenario == "equal_distance":
        # Place all agents at equal distance from resource centroid
        distance = 20  # Fixed distance from centroid
        positions = {}

        # Place agents at equal angles around the centroid
        for i, agent_type in enumerate(agent_types):
            angle = i * (2 * np.pi / 3)  # 120 degrees apart
            x = resource_centroid[0] + distance * np.cos(angle)
            y = resource_centroid[1] + distance * np.sin(angle)
            # Ensure within bounds
            x = max(5, min(width - 5, x))
            y = max(5, min(height - 5, y))
            positions[agent_type] = (x, y)

    elif scenario == "one_advantaged":
        # One agent type (randomly chosen) has advantage
        advantaged_type = random.choice(agent_types)
        positions = {}

        for agent_type in agent_types:
            if agent_type == advantaged_type:
                # Place closer to resources
                distance = 10
            else:
                # Place farther from resources
                distance = 30

            angle = random.uniform(0, 2 * np.pi)
            x = resource_centroid[0] + distance * np.cos(angle)
            y = resource_centroid[1] + distance * np.sin(angle)
            # Ensure within bounds
            x = max(5, min(width - 5, x))
            y = max(5, min(height - 5, y))
            positions[agent_type] = (x, y)

    elif scenario == "one_disadvantaged":
        # One agent type (randomly chosen) has disadvantage
        disadvantaged_type = random.choice(agent_types)
        positions = {}

        for agent_type in agent_types:
            if agent_type == disadvantaged_type:
                # Place farther from resources
                distance = 40
            else:
                # Place closer to resources
                distance = 15

            angle = random.uniform(0, 2 * np.pi)
            x = resource_centroid[0] + distance * np.cos(angle)
            y = resource_centroid[1] + distance * np.sin(angle)
            # Ensure within bounds
            x = max(5, min(width - 5, x))
            y = max(5, min(height - 5, y))
            positions[agent_type] = (x, y)

    elif scenario == "gradient_advantage":
        # Gradient of advantage: System > Control > Independent
        positions = {}
        distances = {"SystemAgent": 10, "ControlAgent": 25, "IndependentAgent": 40}

        for agent_type, distance in distances.items():
            angle = random.uniform(0, 2 * np.pi)
            x = resource_centroid[0] + distance * np.cos(angle)
            y = resource_centroid[1] + distance * np.sin(angle)
            # Ensure within bounds
            x = max(5, min(width - 5, x))
            y = max(5, min(height - 5, y))
            positions[agent_type] = (x, y)

    elif scenario == "random":
        # Random positions for all agents
        positions = {}

        for agent_type in agent_types:
            x = random.uniform(10, width - 10)
            y = random.uniform(10, height - 10)
            positions[agent_type] = (x, y)

    return positions


def visualize_experiment_setup(resource_positions, agent_positions, title, filename):
    """
    Create a visualization of the experiment setup.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot resources
    for x, y, amount in resource_positions:
        ax.scatter(x, y, s=amount * 3, alpha=0.6, c="green")

    # Plot agents
    agent_colors = {
        "SystemAgent": "blue",
        "IndependentAgent": "red",
        "ControlAgent": "orange",
    }

    for agent_type, (x, y) in agent_positions.items():
        ax.scatter(
            x,
            y,
            s=100,
            c=agent_colors[agent_type],
            edgecolors="black",
            label=agent_type,
        )

        # Draw gathering range circle
        circle = Circle(
            (x, y),
            BASE_CONFIG["gathering_range"],
            fill=False,
            linestyle="--",
            color=agent_colors[agent_type],
            alpha=0.3,
        )
        ax.add_patch(circle)

    # Set plot limits and labels
    ax.set_xlim(0, BASE_CONFIG["width"])
    ax.set_ylim(0, BASE_CONFIG["height"])
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(title)
    ax.legend(loc="upper right")

    # Save the figure
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def create_experiment_config(resource_pattern, agent_scenario, experiment_id):
    """
    Create a configuration file for a specific experiment.
    """
    # Copy the base config
    config = BASE_CONFIG.copy()

    # Set a specific seed for reproducibility
    config["seed"] = experiment_id

    # Generate resource positions
    resource_positions = generate_resource_positions(resource_pattern)

    # Generate agent positions
    agent_positions = generate_agent_positions(agent_scenario, resource_positions)

    # Add initial positions to config
    config["initial_positions"] = {
        "resources": [
            {"position_x": x, "position_y": y, "amount": amount}
            for x, y, amount in resource_positions
        ],
        "agents": {
            agent_type: {"position_x": pos[0], "position_y": pos[1]}
            for agent_type, pos in agent_positions.items()
        },
    }

    return config, resource_positions, agent_positions


def main():
    # Create output directories
    os.makedirs("controlled_experiments", exist_ok=True)
    os.makedirs("controlled_experiments/configs", exist_ok=True)
    os.makedirs("controlled_experiments/visualizations", exist_ok=True)

    # Define experiment combinations
    resource_patterns = ["random", "clustered", "uniform_grid", "central", "corners"]
    agent_scenarios = [
        "equal_distance",
        "one_advantaged",
        "one_disadvantaged",
        "gradient_advantage",
        "random",
    ]

    # Number of repetitions for each combination
    repetitions = 3

    # Generate experiments
    experiment_id = 1
    experiments = []

    for pattern in resource_patterns:
        for scenario in agent_scenarios:
            for rep in range(repetitions):
                # Create experiment config
                config, resource_positions, agent_positions = create_experiment_config(
                    pattern, scenario, experiment_id
                )

                # Save config to file
                config_filename = (
                    f"controlled_experiments/configs/experiment_{experiment_id}.json"
                )
                with open(config_filename, "w") as f:
                    json.dump(config, f, indent=2)

                # Create visualization
                title = f"Experiment {experiment_id}: {pattern.capitalize()} resources, {scenario} agents"
                viz_filename = f"controlled_experiments/visualizations/experiment_{experiment_id}.png"
                visualize_experiment_setup(
                    resource_positions, agent_positions, title, viz_filename
                )

                # Record experiment details
                experiments.append(
                    {
                        "id": experiment_id,
                        "resource_pattern": pattern,
                        "agent_scenario": scenario,
                        "repetition": rep + 1,
                        "config_file": config_filename,
                        "visualization": viz_filename,
                    }
                )

                experiment_id += 1

    # Save experiment index
    with open("controlled_experiments/experiment_index.json", "w") as f:
        json.dump(experiments, f, indent=2)

    print(f"Generated {len(experiments)} controlled experiments")
    print("Configuration files saved to: controlled_experiments/configs/")
    print("Visualizations saved to: controlled_experiments/visualizations/")
    print("Experiment index saved to: controlled_experiments/experiment_index.json")


if __name__ == "__main__":
    main()

"""Command-line interface for the unified AgentFarm API.

This module provides a CLI for testing and interacting with the unified API
without needing to implement MCP integration.
"""

import argparse
import json
import time
from typing import Any, Dict

from farm.api import AgentFarmController
from farm.utils.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def create_basic_simulation_config() -> Dict[str, Any]:
    """Create a basic simulation configuration."""
    return {
        "name": "CLI Test Simulation",
        "steps": 100,
        "environment": {"width": 50, "height": 50, "resources": 25},
        "agents": {"system_agents": 5, "independent_agents": 5, "control_agents": 0},
    }


def create_basic_experiment_config() -> Dict[str, Any]:
    """Create a basic experiment configuration."""
    return {
        "name": "CLI Test Experiment",
        "description": "Testing parameter variations",
        "iterations": 3,
        "base_config": {
            "steps": 50,
            "environment": {"width": 30, "height": 30, "resources": 15},
        },
        "variations": [
            {"agents": {"system_agents": 3, "independent_agents": 7}},
            {"agents": {"system_agents": 5, "independent_agents": 5}},
            {"agents": {"system_agents": 7, "independent_agents": 3}},
        ],
    }


def run_simulation_demo(controller: AgentFarmController):
    """Run a simulation demo."""
    print("=== Simulation Demo ===")

    # Create session
    session_id = controller.create_session(
        "CLI Demo", "Testing simulation functionality"
    )
    print(f"Created session: {session_id}")

    # Create simulation
    config = create_basic_simulation_config()
    simulation_id = controller.create_simulation(session_id, config)
    print(f"Created simulation: {simulation_id}")

    # Start simulation
    print("Starting simulation...")
    controller.start_simulation(session_id, simulation_id)

    # Monitor progress
    print("Monitoring progress...")
    while True:
        status = controller.get_simulation_status(session_id, simulation_id)
        print(
            f"Step {status.current_step}/{status.total_steps} ({status.progress_percentage:.1f}%) - {status.status.value}"
        )

        if status.status.value in ["completed", "error", "stopped"]:
            break

        time.sleep(1)

    # Get results
    results = controller.get_simulation_results(session_id, simulation_id)
    print("\nSimulation completed!")
    print(f"Final agents: {results.final_agent_count}")
    print(f"Final resources: {results.final_resource_count}")
    print(f"Data files: {len(results.data_files)}")


def run_experiment_demo(controller: AgentFarmController):
    """Run an experiment demo."""
    print("=== Experiment Demo ===")

    # Create session
    session_id = controller.create_session(
        "CLI Experiment Demo", "Testing experiment functionality"
    )
    print(f"Created session: {session_id}")

    # Create experiment
    config = create_basic_experiment_config()
    experiment_id = controller.create_experiment(session_id, config)
    print(f"Created experiment: {experiment_id}")

    # Start experiment
    print("Starting experiment...")
    controller.start_experiment(session_id, experiment_id)

    # Monitor progress
    print("Monitoring progress...")
    while True:
        status = controller.get_experiment_status(session_id, experiment_id)
        print(
            f"Iteration {status.current_iteration}/{status.total_iterations} ({status.progress_percentage:.1f}%) - {status.status.value}"
        )

        if status.status.value in ["completed", "error", "stopped"]:
            break

        time.sleep(2)

    # Get results
    results = controller.get_experiment_results(session_id, experiment_id)
    print("\nExperiment completed!")
    print(
        f"Completed iterations: {results.completed_iterations}/{results.total_iterations}"
    )
    print(f"Data files: {len(results.data_files)}")


def list_configs_demo(controller: AgentFarmController):
    """List available configurations."""
    print("=== Available Configurations ===")

    configs = controller.get_available_configs()
    for config in configs:
        print(f"\n{config.name}:")
        print(f"  Description: {config.description}")
        print(f"  Category: {config.category.value}")
        print(f"  Required fields: {config.required_fields}")
        print(f"  Optional fields: {config.optional_fields}")


def validate_config_demo(controller: AgentFarmController):
    """Demonstrate configuration validation."""
    print("=== Configuration Validation Demo ===")

    # Valid config
    valid_config = create_basic_simulation_config()
    result = controller.validate_config(valid_config)
    print(f"Valid config: {result.is_valid}")
    if result.warnings:
        print(f"Warnings: {result.warnings}")

    # Invalid config
    invalid_config = {"name": "Test"}  # Missing required 'steps' field
    result = controller.validate_config(invalid_config)
    print(f"Invalid config: {result.is_valid}")
    if result.errors:
        print(f"Errors: {result.errors}")


def create_config_from_template_demo(controller: AgentFarmController):
    """Demonstrate creating config from template."""
    print("=== Create Config from Template Demo ===")

    # Create config from template
    config = controller.create_config_from_template(
        "basic_simulation",
        {"steps": 200, "agents": {"system_agents": 8, "independent_agents": 12}},
    )

    print("Generated config:")
    print(json.dumps(config, indent=2))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="AgentFarm Unified API CLI")
    parser.add_argument(
        "--workspace", default="cli_workspace", help="Workspace directory"
    )
    parser.add_argument(
        "--demo",
        choices=["simulation", "experiment", "configs", "validate", "template"],
        help="Run a specific demo",
    )
    parser.add_argument("--all", action="store_true", help="Run all demos")

    args = parser.parse_args()

    # Configure logging
    configure_logging(environment="development", log_level="INFO")

    # Initialize controller
    controller = AgentFarmController(workspace_path=args.workspace)

    try:
        if args.all:
            print("Running all demos...\n")
            list_configs_demo(controller)
            print("\n" + "=" * 50 + "\n")
            validate_config_demo(controller)
            print("\n" + "=" * 50 + "\n")
            create_config_from_template_demo(controller)
            print("\n" + "=" * 50 + "\n")
            run_simulation_demo(controller)
            print("\n" + "=" * 50 + "\n")
            run_experiment_demo(controller)
        elif args.demo == "simulation":
            run_simulation_demo(controller)
        elif args.demo == "experiment":
            run_experiment_demo(controller)
        elif args.demo == "configs":
            list_configs_demo(controller)
        elif args.demo == "validate":
            validate_config_demo(controller)
        elif args.demo == "template":
            create_config_from_template_demo(controller)
        else:
            print("No demo specified. Use --help for options.")
            print(
                "Available demos: simulation, experiment, configs, validate, template"
            )
            print("Use --all to run all demos")

    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()

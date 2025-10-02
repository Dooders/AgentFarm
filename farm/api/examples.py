"""Example usage of the unified AgentFarm API.

This module provides comprehensive examples of how to use the unified API
for various simulation and experiment scenarios.
"""

import time

from farm.api import AgentFarmController


def basic_simulation_example():
    """Basic simulation example."""
    print("=== Basic Simulation Example ===")

    # Initialize controller
    controller = AgentFarmController(workspace_path="examples_workspace")

    try:
        # Create session
        session_id = controller.create_session(
            "Basic Simulation", "Simple simulation example"
        )
        print(f"Created session: {session_id}")

        # Create simulation configuration
        config = {
            "name": "Basic Test",
            "steps": 500,
            "environment": {"width": 100, "height": 100, "resources": 50},
            "agents": {
                "system_agents": 10,
                "independent_agents": 10,
                "control_agents": 0,
            },
            "learning": {"enabled": True, "algorithm": "dqn"},
        }

        # Create simulation
        simulation_id = controller.create_simulation(session_id, config)
        print(f"Created simulation: {simulation_id}")

        # Start simulation
        print("Starting simulation...")
        controller.start_simulation(session_id, simulation_id)

        # Monitor progress
        while True:
            status = controller.get_simulation_status(session_id, simulation_id)
            print(
                f"Step {status.current_step}/{status.total_steps} ({status.progress_percentage:.1f}%)"
            )

            if status.status.value in ["completed", "error", "stopped"]:
                break

            time.sleep(0.5)

        # Get results
        results = controller.get_simulation_results(session_id, simulation_id)
        print("\nSimulation completed!")
        print(f"Final agents: {results.final_agent_count}")
        print(f"Final resources: {results.final_resource_count}")
        print(f"Duration: {results.metrics.get('duration_seconds', 0):.2f} seconds")

    finally:
        controller.cleanup()


def parameter_variation_experiment_example():
    """Example of running an experiment with parameter variations."""
    print("=== Parameter Variation Experiment Example ===")

    controller = AgentFarmController(workspace_path="examples_workspace")

    try:
        # Create session
        session_id = controller.create_session(
            "Parameter Study", "Testing different agent populations"
        )
        print(f"Created session: {session_id}")

        # Create experiment configuration
        config = {
            "name": "Agent Population Study",
            "description": "Compare different agent population ratios",
            "iterations": 5,
            "base_config": {
                "steps": 300,
                "environment": {"width": 80, "height": 80, "resources": 40},
            },
            "variations": [
                {
                    "agents": {"system_agents": 5, "independent_agents": 15}
                },  # 25% system
                {
                    "agents": {"system_agents": 8, "independent_agents": 12}
                },  # 40% system
                {
                    "agents": {"system_agents": 10, "independent_agents": 10}
                },  # 50% system
                {
                    "agents": {"system_agents": 12, "independent_agents": 8}
                },  # 60% system
                {
                    "agents": {"system_agents": 15, "independent_agents": 5}
                },  # 75% system
            ],
        }

        # Create experiment
        experiment_id = controller.create_experiment(session_id, config)
        print(f"Created experiment: {experiment_id}")

        # Start experiment
        print("Starting experiment...")
        controller.start_experiment(session_id, experiment_id)

        # Monitor progress
        while True:
            status = controller.get_experiment_status(session_id, experiment_id)
            print(
                f"Iteration {status.current_iteration}/{status.total_iterations} ({status.progress_percentage:.1f}%)"
            )

            if status.status.value in ["completed", "error", "stopped"]:
                break

            time.sleep(1)

        # Get results
        results = controller.get_experiment_results(session_id, experiment_id)
        print("\nExperiment completed!")
        print(
            f"Completed iterations: {results.completed_iterations}/{results.total_iterations}"
        )
        print(f"Data files generated: {len(results.data_files)}")

    finally:
        controller.cleanup()


def configuration_templates_example():
    """Example of using configuration templates."""
    print("=== Configuration Templates Example ===")

    controller = AgentFarmController(workspace_path="examples_workspace")

    try:
        # List available templates
        templates = controller.get_available_configs()
        print("Available configuration templates:")
        for template in templates:
            print(f"- {template.name}: {template.description}")

        # Create config from template
        print("\nCreating config from 'combat_simulation' template...")
        config = controller.create_config_from_template(
            "combat_simulation",
            {"steps": 1000, "agents": {"system_agents": 20, "independent_agents": 20}},
        )

        print("Generated configuration:")
        print(f"Name: {config['name']}")
        print(f"Steps: {config['steps']}")
        print(f"Environment: {config['environment']}")
        print(f"Agents: {config['agents']}")

        # Validate configuration
        validation = controller.validate_config(config)
        print(f"\nConfiguration valid: {validation.is_valid}")
        if validation.warnings:
            print(f"Warnings: {validation.warnings}")

    finally:
        controller.cleanup()


def analysis_and_comparison_example():
    """Example of running analysis and comparisons."""
    print("=== Analysis and Comparison Example ===")

    controller = AgentFarmController(workspace_path="examples_workspace")

    try:
        # Create session
        session_id = controller.create_session(
            "Analysis Demo", "Testing analysis features"
        )
        print(f"Created session: {session_id}")

        simulation_ids = []

        # Create multiple simulations with different configurations
        configs = [
            {
                "name": "High Resources",
                "steps": 200,
                "environment": {"width": 50, "height": 50, "resources": 50},
                "agents": {"system_agents": 10, "independent_agents": 10},
            },
            {
                "name": "Low Resources",
                "steps": 200,
                "environment": {"width": 50, "height": 50, "resources": 10},
                "agents": {"system_agents": 10, "independent_agents": 10},
            },
            {
                "name": "Large Population",
                "steps": 200,
                "environment": {"width": 50, "height": 50, "resources": 30},
                "agents": {"system_agents": 20, "independent_agents": 20},
            },
        ]

        # Run simulations
        for i, config in enumerate(configs):
            print(f"Running simulation {i+1}: {config['name']}")

            simulation_id = controller.create_simulation(session_id, config)
            controller.start_simulation(session_id, simulation_id)

            # Wait for completion
            while True:
                status = controller.get_simulation_status(session_id, simulation_id)
                if status.status.value in ["completed", "error", "stopped"]:
                    break
                time.sleep(0.1)

            simulation_ids.append(simulation_id)
            print(f"Completed simulation {i+1}")

        # Run analysis on first simulation
        print("\nRunning analysis on first simulation...")
        analysis = controller.analyze_simulation(session_id, simulation_ids[0])
        print(f"Analysis completed: {analysis.analysis_id}")
        print(f"Output files: {len(analysis.output_files)}")

        # Compare simulations
        print("\nComparing all simulations...")
        comparison = controller.compare_simulations(session_id, simulation_ids)
        print(f"Comparison completed: {comparison.comparison_id}")
        print(f"Output files: {len(comparison.output_files)}")

    finally:
        controller.cleanup()


def event_monitoring_example():
    """Example of event monitoring and subscriptions."""
    print("=== Event Monitoring Example ===")

    controller = AgentFarmController(workspace_path="examples_workspace")

    try:
        # Create session
        session_id = controller.create_session("Event Demo", "Testing event system")
        print(f"Created session: {session_id}")

        # Subscribe to events
        subscription_id = controller.subscribe_to_events(
            session_id,
            ["simulation_created", "simulation_started", "simulation_completed"],
        )
        print(f"Created event subscription: {subscription_id}")

        # Create and run simulation
        config = {
            "name": "Event Test",
            "steps": 100,
            "environment": {"width": 30, "height": 30, "resources": 15},
            "agents": {"system_agents": 5, "independent_agents": 5},
        }

        simulation_id = controller.create_simulation(session_id, config)
        controller.start_simulation(session_id, simulation_id)

        # Wait for completion
        while True:
            status = controller.get_simulation_status(session_id, simulation_id)
            if status.status.value in ["completed", "error", "stopped"]:
                break
            time.sleep(0.1)

        # Get event history
        events = controller.get_event_history(session_id, subscription_id)
        print(f"\nCaptured {len(events)} events:")
        for event in events:
            print(f"- {event.event_type}: {event.message}")

    finally:
        controller.cleanup()


def main():
    """Run all examples."""
    examples = [
        ("Basic Simulation", basic_simulation_example),
        ("Parameter Variation Experiment", parameter_variation_experiment_example),
        ("Configuration Templates", configuration_templates_example),
        ("Analysis and Comparison", analysis_and_comparison_example),
        ("Event Monitoring", event_monitoring_example),
    ]

    print("AgentFarm Unified API Examples")
    print("=" * 50)

    for name, example_func in examples:
        print(f"\n{name}")
        print("-" * len(name))
        try:
            example_func()
        except Exception as e:
            print(f"Error in {name}: {e}")
        print()


if __name__ == "__main__":
    main()

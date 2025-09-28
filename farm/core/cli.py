import argparse
import json
import logging
import os
import tkinter as tk
from datetime import datetime

from farm.core.analysis import SimulationAnalyzer
from farm.core.cli import run_simulation
from farm.core.config_hydra_bridge import HydraSimulationConfig
from farm.core.visualization import SimulationVisualizer
from farm.runners.experiment_runner import ExperimentRunner


def setup_logging(log_dir="logs"):
    """Setup logging configuration."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def run_experiment(args):
    """Run experiment with specified parameters."""
    config = SimulationConfig.from_yaml(args.config)
    experiment = ExperimentRunner(config, args.experiment_name)

    if args.variations:
        # Load variations from JSON file
        with open(args.variations) as f:
            variations = json.load(f)
        experiment.run_iterations(args.iterations, variations)
    else:
        experiment.run_iterations(args.iterations)

    # experiment.generate_report()


def main():
    parser = argparse.ArgumentParser(description="Agent-Based Simulation CLI")
    parser.add_argument(
        "--mode",
        choices=["simulate", "visualize", "analyze", "experiment"],
        default="simulate",
        help="Mode of operation",
    )
    parser.add_argument(
        "--db-path", default="simulation.db", help="Path to the simulation database"
    )
    parser.add_argument(
        "--report-path",
        default="simulation_report.html",
        help="Path for the analysis report",
    )
    parser.add_argument(
        "--export-path", default="simulation_data.csv", help="Path for exported data"
    )

    # Simulation parameters
    parser.add_argument(
        "--steps", type=int, default=5000, help="Number of simulation steps"
    )
    parser.add_argument(
        "--system-agents", type=int, default=25, help="Initial number of system agents"
    )
    parser.add_argument(
        "--independent-agents",
        type=int,
        default=25,
        help="Initial number of independent agents",
    )
    parser.add_argument(
        "--resources", type=int, default=60, help="Initial number of resources"
    )

    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to configuration file"
    )
    parser.add_argument(
        "--save-config", type=str, help="Save current configuration to file"
    )

    # Experiment parameters
    parser.add_argument("--experiment-name", help="Name of the experiment")
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of iterations to run"
    )
    parser.add_argument(
        "--variations", help="Path to JSON file containing parameter variations"
    )

    args = parser.parse_args()

    setup_logging()

    if args.mode == "simulate":
        # Load configuration
        config = SimulationConfig.from_yaml(args.config)

        # Override config with command line arguments if provided
        if args.system_agents:
            config.system_agents = args.system_agents
        if args.independent_agents:
            config.independent_agents = args.independent_agents
        if args.resources:
            config.initial_resources = args.resources

        # Save configuration if requested
        if args.save_config:
            config.to_yaml(args.save_config)

        # Run simulation with configuration
        # run_simulation(num_steps=args.steps, config=config, db_path=args.db_path)

    elif args.mode == "visualize":
        # Open visualization for existing simulation
        if not os.path.exists(args.db_path):
            logging.error(f"Database file not found: {args.db_path}")
            return

        root = tk.Tk()
        visualizer = SimulationVisualizer(root, db_path=args.db_path)
        visualizer.run()

    elif args.mode == "analyze":
        # Generate analysis report
        if not os.path.exists(args.db_path):
            logging.error(f"Database file not found: {args.db_path}")
            return

        analyzer = SimulationAnalyzer(db_path=args.db_path)
        analyzer.generate_report(output_file=args.report_path)
        logging.info(f"Analysis report generated: {args.report_path}")

        # Export data if requested
        if args.export_path:
            analyzer.db.export_data(args.export_path)
            logging.info(f"Data exported to: {args.export_path}")

    elif args.mode == "experiment":
        run_experiment(args)


if __name__ == "__main__":
    main()

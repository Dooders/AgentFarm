#!/usr/bin/env python3
"""
Genesis Analysis Script

This script analyzes how initial states and conditions impact simulation outcomes
using the Genesis analysis module.

Usage:
    python genesis_analysis.py

The script automatically finds the most recent experiment in the DATA_PATH
defined in analysis_config.py and saves results to the OUTPUT_PATH.
"""

import glob
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Import analysis configuration
from analysis_config import (DATA_PATH, OUTPUT_PATH, safe_remove_directory,
                             setup_logging)
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

# Add the parent directory to sys.path to import farm modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from farm.analysis.genesis.analyze import (
        analyze_critical_period, analyze_genesis_across_simulations,
        analyze_genesis_factors)
    from farm.analysis.genesis.plot import (plot_critical_period_analysis,
                                            plot_genesis_analysis_results,
                                            plot_initial_state_comparison)
except ImportError as e:
    print(f"Error importing Genesis module: {e}")
    print("Make sure the Genesis module is properly installed.")
    sys.exit(1)


def check_database_schema(engine) -> bool:
    """
    Check if the database has the required tables for Genesis analysis.

    Parameters
    ----------
    engine : SQLAlchemy engine
        Database engine

    Returns
    -------
    bool
        True if the database has the required tables, False otherwise
    """
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    logging.info(f"Found tables in database: {', '.join(table_names)}")

    # Define required tables with possible alternative names
    required_tables = {
        "agents": ["agents"],
        "resources": ["resource_states"],
        "simulation_steps": ["simulation_steps"],
    }

    # Check each required table
    for table_type, possible_names in required_tables.items():
        found = False
        for name in possible_names:
            if name in table_names:
                found = True
                logging.info(f"Found required table '{table_type}' as '{name}'")
                break
        if not found:
            logging.error(
                f"Missing required table '{table_type}'. Expected one of: {', '.join(possible_names)}"
            )
            return False

    # Additional schema validation
    for table_name in table_names:
        columns = inspector.get_columns(table_name)
        logging.info(
            f"Table '{table_name}' has {len(columns)} columns: {', '.join(col['name'] for col in columns)}"
        )

    return True


def check_action_weights_column(engine) -> bool:
    """
    Check if the agents table has the action_weights column.

    Parameters
    ----------
    engine : SQLAlchemy engine
        Database engine

    Returns
    -------
    bool
        True if the action_weights column exists, False otherwise
    """
    inspector = inspect(engine)
    columns = [col["name"] for col in inspector.get_columns("agents")]
    return "action_weights" in columns


def analyze_single_simulation(
    sim_path: str, output_path: str, critical_period: int = 100
) -> Dict[str, Any]:
    """
    Analyze a single simulation using the Genesis module.

    Parameters
    ----------
    sim_path : str
        Path to the simulation directory
    output_path : str
        Path where analysis results will be saved
    critical_period : int, optional
        Number of steps to consider as the critical period, by default 100

    Returns
    -------
    Dict[str, Any]
        Analysis results
    """
    logging.info(f"Analyzing simulation at {sim_path}")

    # Create database connection
    db_path = os.path.join(sim_path, "simulation.db")

    if not os.path.exists(db_path):
        logging.error(f"Database not found at {db_path}")
        return {"error": f"Database not found at {db_path}"}

    try:
        engine = create_engine(f"sqlite:///{db_path}")

        # Check if the database has the required schema
        if not check_database_schema(engine):
            logging.error(f"Database at {db_path} does not have the required tables")
            return {"error": f"Database schema mismatch at {db_path}"}

        # Check if action_weights column exists
        has_action_weights = check_action_weights_column(engine)
        if has_action_weights:
            logging.info(
                f"Database at {db_path} has action_weights column. Full agent analysis will be performed."
            )
        else:
            logging.warning(
                f"Database at {db_path} does not have the action_weights column. "
                "Action weights analysis will be skipped."
            )

        Session = sessionmaker(bind=engine)
        session = Session()

        try:
            # Analyze genesis factors
            logging.info("Computing genesis factors...")
            genesis_results = analyze_genesis_factors(session)

            # Analyze critical period
            logging.info(
                f"Analyzing critical period (first {critical_period} steps)..."
            )
            critical_period_results = analyze_critical_period(session, critical_period)

            # Combine results
            results = {
                "genesis_analysis": genesis_results,
                "critical_period_analysis": critical_period_results,
                "database_info": {"has_action_weights_column": has_action_weights},
            }

            # Create simulation-specific output directory
            sim_name = os.path.basename(sim_path)
            sim_output_path = os.path.join(output_path, sim_name)
            os.makedirs(sim_output_path, exist_ok=True)

            # Generate visualizations
            logging.info("Generating visualizations...")
            try:
                plot_genesis_analysis_results(genesis_results, sim_output_path)
            except Exception as e:
                logging.error(f"Error generating visualizations: {e}")
                logging.debug(traceback.format_exc())

            # Save results as JSON
            try:
                with open(
                    os.path.join(sim_output_path, "genesis_analysis_results.json"), "w"
                ) as f:
                    json.dump(results, f, indent=2, default=str)
            except Exception as e:
                logging.error(f"Error saving results to JSON: {e}")
                logging.debug(traceback.format_exc())

            logging.info(f"Analysis complete. Results saved to {sim_output_path}")

            return results

        except Exception as e:
            logging.error(f"Error analyzing simulation: {e}")
            logging.debug(traceback.format_exc())
            return {"error": str(e)}

        finally:
            session.close()

    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        logging.debug(traceback.format_exc())
        return {"error": f"Database connection error: {str(e)}"}


def analyze_experiment(
    experiment_path: str, output_path: str, critical_period: int = 100
) -> Dict[str, Any]:
    """
    Analyze all simulations in an experiment using the Genesis module.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment directory
    output_path : str
        Path where analysis results will be saved
    critical_period : int, optional
        Number of steps to consider as the critical period, by default 100

    Returns
    -------
    Dict[str, Any]
        Analysis results
    """
    logging.info(f"Analyzing experiment at {experiment_path}")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    try:
        # First, validate all databases have required schema
        sim_dbs = []
        for root, _, files in os.walk(experiment_path):
            if "simulation.db" in files:
                db_path = os.path.join(root, "simulation.db")
                engine = create_engine(f"sqlite:///{db_path}")
                if check_database_schema(engine):
                    sim_dbs.append(db_path)
                else:
                    logging.warning(f"Skipping {db_path} due to invalid schema")

        if not sim_dbs:
            raise ValueError("No valid simulation databases found")

        logging.info(f"Found {len(sim_dbs)} valid simulation databases")

        # Analyze across simulations
        logging.info("Performing cross-simulation analysis...")
        cross_sim_results = analyze_genesis_across_simulations(experiment_path)

        # Enhance cross-simulation results with additional metrics
        cross_sim_results["experiment_summary"] = {
            "total_simulations": len(sim_dbs),
            "critical_period": critical_period,
            "experiment_path": experiment_path,
            "analysis_timestamp": datetime.now().isoformat(),
        }

        # Aggregate critical period data across all simulations
        logging.info("Aggregating critical period data...")
        critical_period_data = []
        for db_path in sim_dbs:
            engine = create_engine(f"sqlite:///{db_path}")
            Session = sessionmaker(bind=engine)
            session = Session()
            try:
                period_results = analyze_critical_period(session, critical_period)
                critical_period_data.append(period_results)
            except Exception as e:
                logging.warning(f"Error analyzing critical period for {db_path}: {e}")
            finally:
                session.close()

        if critical_period_data:
            cross_sim_results["aggregated_critical_period"] = {
                "data": critical_period_data,
                "summary": {
                    "mean_survival_rate": np.mean(
                        [d.get("survival_rate", 0) for d in critical_period_data]
                    ),
                    "mean_reproduction_rate": np.mean(
                        [d.get("reproduction_rate", 0) for d in critical_period_data]
                    ),
                    "mean_resource_efficiency": np.mean(
                        [d.get("resource_efficiency", 0) for d in critical_period_data]
                    ),
                },
            }

        # Generate all visualizations with proper error handling
        logging.info("Generating comprehensive visualizations...")

        visualization_functions = [
            (
                "Initial State Comparison",
                plot_initial_state_comparison,
                [cross_sim_results.get("simulations", []), output_path],
            ),
            (
                "Critical Period Analysis",
                plot_critical_period_analysis,
                [cross_sim_results.get("simulations", []), output_path],
            ),
            (
                "Genesis Analysis Results",
                plot_genesis_analysis_results,
                [cross_sim_results, output_path],
            ),
        ]

        for viz_name, viz_func, viz_args in visualization_functions:
            try:
                logging.info(f"Generating {viz_name} visualization...")
                viz_func(*viz_args)
                logging.info(f"Successfully generated {viz_name} visualization")
            except Exception as e:
                logging.error(f"Error generating {viz_name} visualization: {e}")
                logging.debug(traceback.format_exc())

        # Additional visualization for critical period trends
        if critical_period_data:
            try:
                logging.info("Generating Critical Period Trends visualization...")
                plt.figure(figsize=(12, 6))
                metrics = ["survival_rate", "reproduction_rate", "resource_efficiency"]
                for metric in metrics:
                    values = [d.get(metric, 0) for d in critical_period_data]
                    plt.plot(values, label=metric.replace("_", " ").title())
                plt.title("Critical Period Metrics Across Simulations")
                plt.xlabel("Simulation Index")
                plt.ylabel("Rate")
                plt.legend()
                plt.grid(True)
                plt.savefig(
                    os.path.join(output_path, "critical_period_trends.png"), dpi=150
                )
                plt.close()
                logging.info(
                    "Successfully generated Critical Period Trends visualization"
                )
            except Exception as e:
                logging.error(
                    f"Error generating Critical Period Trends visualization: {e}"
                )
                logging.debug(traceback.format_exc())

        # Save comprehensive cross-simulation results
        try:
            results_file = os.path.join(output_path, "cross_simulation_analysis.json")
            with open(results_file, "w") as f:
                json.dump(cross_sim_results, f, indent=2, default=str)
            logging.info(f"Saved comprehensive analysis results to {results_file}")

            # Generate summary report
            summary_file = os.path.join(output_path, "analysis_summary.txt")
            with open(summary_file, "w") as f:
                f.write("Genesis Analysis Summary\n")
                f.write("======================\n\n")
                f.write(
                    f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Experiment Path: {experiment_path}\n")
                f.write(f"Total Simulations Analyzed: {len(sim_dbs)}\n")
                f.write(f"Critical Period Length: {critical_period} steps\n\n")

                if "determinism_analysis" in cross_sim_results:
                    f.write("Determinism Analysis\n")
                    f.write("-----------------\n")
                    det = cross_sim_results["determinism_analysis"]
                    if "determinism_level" in det:
                        f.write(f"Determinism Level: {det['determinism_level']}\n")
                    if "initial_advantage_realization_rate" in det:
                        f.write(
                            f"Initial Advantage Realization Rate: {det['initial_advantage_realization_rate']:.2f}\n"
                        )
                    f.write("\n")

                if "cross_simulation_patterns" in cross_sim_results:
                    f.write("Cross-Simulation Patterns\n")
                    f.write("----------------------\n")
                    patterns = cross_sim_results["cross_simulation_patterns"]
                    if "advantage_consistency" in patterns:
                        f.write("Advantage Consistency:\n")
                        for agent_type, value in patterns[
                            "advantage_consistency"
                        ].items():
                            f.write(f"  {agent_type}: {value:.2f}\n")
                    f.write("\n")

                if "aggregated_critical_period" in cross_sim_results:
                    f.write("Critical Period Summary\n")
                    f.write("--------------------\n")
                    summary = cross_sim_results["aggregated_critical_period"]["summary"]
                    for metric, value in summary.items():
                        f.write(f"{metric.replace('_', ' ').title()}: {value:.3f}\n")

            logging.info(f"Generated analysis summary at {summary_file}")

        except Exception as e:
            logging.error(f"Error saving analysis results: {e}")
            logging.debug(traceback.format_exc())

        logging.info(
            f"Cross-simulation analysis complete. Results saved to {output_path}"
        )

        return cross_sim_results

    except Exception as e:
        logging.error(f"Error in cross-simulation analysis: {e}")
        logging.debug(traceback.format_exc())
        return {"error": str(e)}


def main():
    """Main function to run the Genesis analysis."""
    try:
        # Create genesis output directory
        genesis_output_path = os.path.join(OUTPUT_PATH, "genesis")

        # Clear the genesis directory if it exists
        if os.path.exists(genesis_output_path):
            logging.info(f"Clearing existing genesis directory: {genesis_output_path}")
            if not safe_remove_directory(genesis_output_path):
                # If we couldn't remove the directory after retries, create a new one with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                genesis_output_path = os.path.join(OUTPUT_PATH, f"genesis_{timestamp}")
                logging.info(f"Using alternative directory: {genesis_output_path}")

        # Create the directory
        os.makedirs(genesis_output_path, exist_ok=True)

        # Set up logging to the genesis directory
        log_file = setup_logging(genesis_output_path)

        logging.info(f"Saving results to {genesis_output_path}")

        # Find the most recent experiment folder in DATA_PATH
        if not os.path.exists(DATA_PATH):
            logging.error(f"DATA_PATH does not exist: {DATA_PATH}")
            return

        experiment_folders = [
            d for d in glob.glob(os.path.join(DATA_PATH, "*")) if os.path.isdir(d)
        ]
        if not experiment_folders:
            logging.error(f"No experiment folders found in {DATA_PATH}")
            return

        # Sort by modification time (most recent first)
        experiment_folders.sort(key=os.path.getmtime, reverse=True)
        experiment_path = experiment_folders[0]

        # Check if experiment_path contains iteration folders directly
        iteration_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))
        if not iteration_folders:
            # If no iteration folders found directly, look for subdirectories that might contain them
            subdirs = [
                d
                for d in glob.glob(os.path.join(experiment_path, "*"))
                if os.path.isdir(d)
            ]
            if subdirs:
                # Sort by modification time (most recent first)
                subdirs.sort(key=os.path.getmtime, reverse=True)
                experiment_path = subdirs[0]
                logging.info(
                    f"Using subdirectory as experiment path: {experiment_path}"
                )

                # Verify that this subdirectory contains iteration folders
                iteration_folders = glob.glob(
                    os.path.join(experiment_path, "iteration_*")
                )
                if not iteration_folders:
                    logging.error(f"No iteration folders found in {experiment_path}")
                    return
            else:
                logging.error(f"No subdirectories found in {experiment_path}")
                return

        # Check if reproduction events exist in the databases
        # For now, assume they exist and continue
        has_reproduction_events = True
        if not has_reproduction_events:
            logging.warning(
                "No reproduction events found in databases. Reproduction analysis may be limited."
            )

        # Define critical period length
        critical_period = 100  # Default value, can be adjusted if needed

        # Log analysis parameters
        logging.info(f"Genesis Analysis started")
        logging.info(f"Experiment path: {experiment_path}")
        logging.info(f"Output path: {genesis_output_path}")
        logging.info(f"Critical period: {critical_period} steps")
        logging.info(f"Number of simulations to analyze: {len(iteration_folders)}")

        # Analyze experiment as a whole
        logging.info("Performing experiment-wide analysis...")
        cross_sim_results = analyze_experiment(
            experiment_path, genesis_output_path, critical_period
        )

        # Log summary of findings
        if "simulations" in cross_sim_results:
            sim_count = len(cross_sim_results["simulations"])
            logging.info(f"Cross-simulation analysis included {sim_count} simulations.")

            # Log determinism analysis if available
            if "determinism_analysis" in cross_sim_results:
                determinism = cross_sim_results["determinism_analysis"]
                if "determinism_level" in determinism:
                    logging.info(
                        f"Determinism level: {determinism['determinism_level']}"
                    )
                if "initial_advantage_realization_rate" in determinism:
                    rate = determinism["initial_advantage_realization_rate"]
                    logging.info(f"Initial advantage realization rate: {rate:.2f}")

            # Log predictive model performance if available
            if "predictive_models" in cross_sim_results:
                models = cross_sim_results["predictive_models"]
                if "dominance_prediction" in models:
                    accuracy = models["dominance_prediction"].get("accuracy", 0)
                    logging.info(f"Dominance prediction accuracy: {accuracy:.2f}")
                if "survival_prediction" in models:
                    r2 = models["survival_prediction"].get("r2_score", 0)
                    logging.info(f"Survival prediction R² score: {r2:.2f}")

            # Log cross-simulation patterns if available
            if "cross_simulation_patterns" in cross_sim_results:
                patterns = cross_sim_results["cross_simulation_patterns"]
                if "advantage_consistency" in patterns:
                    consistency = patterns["advantage_consistency"]
                    logging.info("\nAdvantage consistency across simulations:")
                    for agent_type, value in consistency.items():
                        logging.info(f"  {agent_type}: {value:.2f}")

        logging.info(
            f"\nGenesis Analysis completed. All results saved to {genesis_output_path}"
        )
        logging.info(f"Log file saved to: {log_file}")

    except Exception as e:
        logging.error(f"Unhandled exception in main: {e}")
        logging.debug(traceback.format_exc())


if __name__ == "__main__":
    main()

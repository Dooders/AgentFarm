#!/usr/bin/env python3
"""
core_analysis.py

Consolidated analysis module combining functionality from:
- simple_research_analysis.py (population, resource, action analysis)
- experiment_analysis.py (comprehensive experiment analysis)

This module provides unified analysis capabilities for simulation experiments,
eliminating code duplication while maintaining all functionality.
"""

import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlalchemy
from matplotlib import pyplot as plt
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker

# Import database models
from farm.database.models import (
    ActionModel,
    AgentModel,
    AgentStateModel,
    HealthIncident,
    LearningExperienceModel,
    ReproductionEventModel,
    SimulationStepModel,
)

# Import our utility modules
from ..data_extraction import (
    extract_time_series,
    get_column_data_at_steps,
    get_initial_positions,
    validate_dataframe,
)
from ..database_utils import (
    create_database_session,
    get_agent_counts_by_type,
    get_final_step_number,
    get_iteration_number,
    get_simulation_database_path,
    get_simulation_folders,
    validate_simulation_folder,
)
from ..visualization_utils import (
    create_box_plot,
    create_histogram,
    create_time_series_plot,
    save_figure,
    setup_plot_style,
)

logger = logging.getLogger(__name__)


class UnifiedExperimentAnalyzer:
    """
    Unified analyzer combining functionality from simple_research_analysis and experiment_analysis.
    """

    def __init__(self, experiment_path: str, output_dir: str = "analysis_results"):
        """
        Initialize the unified analyzer.

        Parameters
        ----------
        experiment_path : str
            Path to experiment directory containing simulation folders
        output_dir : str
            Output directory for results
        """
        self.experiment_path = experiment_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Get all simulation folders
        self.sim_folders = get_simulation_folders(experiment_path)
        self.valid_folders = [
            f for f in self.sim_folders if validate_simulation_folder(f)
        ]

        logger.info(
            f"Found {len(self.sim_folders)} simulation folders, {len(self.valid_folders)} valid"
        )

    def analyze_population_dynamics(self) -> Dict[str, Any]:
        """
        Analyze population dynamics across all simulations.
        Combines functionality from both analysis scripts.
        """
        logger.info("Analyzing population dynamics...")

        all_populations = []
        agent_type_populations = {
            "SystemAgent": [],
            "IndependentAgent": [],
            "ControlAgent": [],
        }
        max_steps = 0

        for folder in self.valid_folders:
            db_path = get_simulation_database_path(folder)

            try:
                session = create_database_session(db_path)

                # Get population data
                steps_df, _, _ = extract_time_series(db_path)
                validate_dataframe(steps_df, ["step_number"])

                # Total population
                total_pop = steps_df["total_agents"].values
                all_populations.append(total_pop)

                # Agent type populations
                for agent_type in agent_type_populations.keys():
                    col_name = f"{agent_type.lower().replace('agent', '_agents')}"
                    if col_name in steps_df.columns:
                        agent_type_populations[agent_type].append(
                            steps_df[col_name].values
                        )

                max_steps = max(max_steps, len(total_pop))
                session.close()

            except Exception as e:
                logger.error(f"Error analyzing population in {folder}: {e}")
                continue

        # Calculate statistics
        results = self._calculate_population_statistics(all_populations, max_steps)

        # Add agent type analysis
        results["agent_types"] = {}
        for agent_type, populations in agent_type_populations.items():
            if populations:
                results["agent_types"][agent_type] = (
                    self._calculate_population_statistics(populations, max_steps)
                )

        return results

    def analyze_resource_dynamics(self) -> Dict[str, Any]:
        """
        Analyze resource dynamics across all simulations.
        """
        logger.info("Analyzing resource dynamics...")

        all_resources = []
        max_steps = 0

        for folder in self.valid_folders:
            db_path = get_simulation_database_path(folder)

            try:
                session = create_database_session(db_path)

                # Get resource data
                steps_df, _, _ = extract_time_series(db_path)
                validate_dataframe(steps_df, ["step_number"])

                if "total_resources" in steps_df.columns:
                    resources = steps_df["total_resources"].values
                    all_resources.append(resources)
                    max_steps = max(max_steps, len(resources))

                session.close()

            except Exception as e:
                logger.error(f"Error analyzing resources in {folder}: {e}")
                continue

        return self._calculate_resource_statistics(all_resources, max_steps)

    def analyze_action_patterns(self) -> Dict[str, Any]:
        """
        Analyze action patterns across simulations.
        """
        logger.info("Analyzing action patterns...")

        action_counts = {}
        action_success_rates = {}

        for folder in self.valid_folders:
            db_path = get_simulation_database_path(folder)

            try:
                session = create_database_session(db_path)

                # Get action data
                actions = session.query(ActionModel).all()

                for action in actions:
                    action_type = action.action_type
                    if action_type not in action_counts:
                        action_counts[action_type] = 0
                        action_success_rates[action_type] = {"success": 0, "total": 0}

                    action_counts[action_type] += 1
                    action_success_rates[action_type]["total"] += 1

                    # Check if action was successful
                    if hasattr(action, "details") and action.details:
                        try:
                            details = (
                                json.loads(action.details)
                                if isinstance(action.details, str)
                                else action.details
                            )
                            if details.get("success", False):
                                action_success_rates[action_type]["success"] += 1
                        except (json.JSONDecodeError, KeyError):
                            pass

                session.close()

            except Exception as e:
                logger.error(f"Error analyzing actions in {folder}: {e}")
                continue

        # Calculate success rates
        for action_type in action_success_rates:
            total = action_success_rates[action_type]["total"]
            if total > 0:
                action_success_rates[action_type]["rate"] = (
                    action_success_rates[action_type]["success"] / total
                )
            else:
                action_success_rates[action_type]["rate"] = 0

        return {
            "action_counts": action_counts,
            "action_success_rates": action_success_rates,
        }

    def analyze_reproduction(self) -> Dict[str, Any]:
        """
        Analyze reproduction patterns across simulations.
        """
        logger.info("Analyzing reproduction patterns...")

        reproduction_stats = {
            "total_events": 0,
            "successful_events": 0,
            "by_agent_type": {},
            "success_rates": {},
        }

        for folder in self.valid_folders:
            db_path = get_simulation_database_path(folder)

            try:
                session = create_database_session(db_path)

                # Get reproduction data
                reproductions = session.query(ReproductionEventModel).all()

                for repro in reproductions:
                    reproduction_stats["total_events"] += 1
                    if repro.success:
                        reproduction_stats["successful_events"] += 1

                    # Track by agent type
                    if repro.parent_id:
                        # Get agent type from agents table
                        agent = (
                            session.query(AgentModel)
                            .filter(AgentModel.agent_id == repro.parent_id)
                            .first()
                        )
                        if agent:
                            agent_type = agent.agent_type
                            if agent_type not in reproduction_stats["by_agent_type"]:
                                reproduction_stats["by_agent_type"][agent_type] = {
                                    "total": 0,
                                    "successful": 0,
                                }

                            reproduction_stats["by_agent_type"][agent_type][
                                "total"
                            ] += 1
                            if repro.success:
                                reproduction_stats["by_agent_type"][agent_type][
                                    "successful"
                                ] += 1

                session.close()

            except Exception as e:
                logger.error(f"Error analyzing reproduction in {folder}: {e}")
                continue

        # Calculate success rates
        if reproduction_stats["total_events"] > 0:
            reproduction_stats["overall_success_rate"] = (
                reproduction_stats["successful_events"]
                / reproduction_stats["total_events"]
            )

        for agent_type in reproduction_stats["by_agent_type"]:
            total = reproduction_stats["by_agent_type"][agent_type]["total"]
            successful = reproduction_stats["by_agent_type"][agent_type]["successful"]
            if total > 0:
                reproduction_stats["success_rates"][agent_type] = successful / total
            else:
                reproduction_stats["success_rates"][agent_type] = 0

        return reproduction_stats

    def _calculate_population_statistics(
        self, all_populations: List[np.ndarray], max_steps: int
    ) -> Dict[str, Any]:
        """Calculate comprehensive population statistics."""
        if not all_populations:
            return {"error": "No population data available"}

        # Create aligned population data
        aligned_populations = []
        for pop in all_populations:
            if len(pop) < max_steps:
                # Pad with last value
                padded = np.pad(pop, (0, max_steps - len(pop)), "edge")
            else:
                padded = pop[:max_steps]
            aligned_populations.append(padded)

        populations_array = np.array(aligned_populations)
        steps = np.arange(max_steps)

        # Calculate statistics
        mean_pop = np.nanmean(populations_array, axis=0)
        median_pop = np.nanmedian(populations_array, axis=0)
        std_pop = np.nanstd(populations_array, axis=0)

        # Confidence intervals
        confidence_interval = 1.96 * std_pop / np.sqrt(len(aligned_populations))

        # Key metrics
        final_populations = populations_array[:, -1]
        peak_step = np.nanargmax(mean_pop)
        peak_value = mean_pop[peak_step]

        return {
            "mean": mean_pop,
            "median": median_pop,
            "std": std_pop,
            "confidence_interval": confidence_interval,
            "steps": steps,
            "final_populations": final_populations,
            "peak_step": int(peak_step),
            "peak_value": float(peak_value),
            "n_simulations": len(aligned_populations),
            "survival_rate": np.mean(final_populations > 0),
        }

    def _calculate_resource_statistics(
        self, all_resources: List[np.ndarray], max_steps: int
    ) -> Dict[str, Any]:
        """Calculate comprehensive resource statistics."""
        if not all_resources:
            return {"error": "No resource data available"}

        # Create aligned resource data
        aligned_resources = []
        for res in all_resources:
            if len(res) < max_steps:
                padded = np.pad(res, (0, max_steps - len(res)), "edge")
            else:
                padded = res[:max_steps]
            aligned_resources.append(padded)

        resources_array = np.array(aligned_resources)
        steps = np.arange(max_steps)

        # Calculate statistics
        mean_res = np.nanmean(resources_array, axis=0)
        median_res = np.nanmedian(resources_array, axis=0)
        std_res = np.nanstd(resources_array, axis=0)

        return {
            "mean": mean_res,
            "median": median_res,
            "std": std_res,
            "steps": steps,
            "n_simulations": len(aligned_resources),
            "final_resources": resources_array[:, -1],
        }

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report combining all metrics.
        """
        logger.info("Generating comprehensive analysis report...")

        report = {
            "experiment_info": {
                "total_simulations": len(self.sim_folders),
                "valid_simulations": len(self.valid_folders),
                "experiment_path": str(self.experiment_path),
            },
            "population_analysis": self.analyze_population_dynamics(),
            "resource_analysis": self.analyze_resource_dynamics(),
            "action_analysis": self.analyze_action_patterns(),
            "reproduction_analysis": self.analyze_reproduction(),
        }

        # Save report
        report_path = self.output_dir / "comprehensive_analysis_report.json"
        with open(report_path, "w") as f:
            # Convert numpy types to JSON serializable
            json_report = self._make_json_serializable(report)
            json.dump(json_report, f, indent=2)

        logger.info(f"Comprehensive report saved to: {report_path}")
        return report

    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to JSON serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def create_visualization_report(self) -> None:
        """
        Create comprehensive visualization report.
        """
        logger.info("Creating visualization report...")

        # Population dynamics visualization
        pop_data = self.analyze_population_dynamics()
        if "mean" in pop_data:
            fig, ax = plt.subplots(figsize=(12, 8))
            create_time_series_plot(
                ax,
                pd.DataFrame(
                    {
                        "step_number": pop_data["steps"],
                        "mean": pop_data["mean"],
                        "median": pop_data["median"],
                    }
                ),
                "step_number",
                ["mean", "median"],
                title="Population Dynamics Across Simulations",
                xlabel="Step Number",
                ylabel="Population Count",
            )
            save_figure(fig, str(self.output_dir / "population_dynamics.png"))

        # Resource dynamics visualization
        res_data = self.analyze_resource_dynamics()
        if "mean" in res_data:
            fig, ax = plt.subplots(figsize=(12, 8))
            create_time_series_plot(
                ax,
                pd.DataFrame(
                    {
                        "step_number": res_data["steps"],
                        "mean": res_data["mean"],
                        "median": res_data["median"],
                    }
                ),
                "step_number",
                ["mean", "median"],
                title="Resource Dynamics Across Simulations",
                xlabel="Step Number",
                ylabel="Total Resources",
            )
            save_figure(fig, str(self.output_dir / "resource_dynamics.png"))

        logger.info(f"Visualization report saved to: {self.output_dir}")


def analyze_single_experiment(
    experiment_path: str,
    output_dir: str = "experiment_analysis",
    create_visualizations: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to analyze a single experiment.

    Parameters
    ----------
    experiment_path : str
        Path to experiment directory
    output_dir : str
        Output directory for results
    create_visualizations : bool
        Whether to create visualizations

    Returns
    -------
    Dict[str, Any]
        Comprehensive analysis results
    """
    analyzer = UnifiedExperimentAnalyzer(experiment_path, output_dir)

    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()

    # Create visualizations if requested
    if create_visualizations:
        analyzer.create_visualization_report()

    return report


def main():
    """
    Example usage of the unified experiment analyzer.
    """
    # Example experiment path (adjust as needed)
    experiment_path = (
        "results/one_of_a_kind/experiments/data/one_of_a_kind_20250302_193353"
    )

    if not os.path.exists(experiment_path):
        print(f"Experiment path not found: {experiment_path}")
        print("Please update the path in the main() function")
        return

    # Create comprehensive analysis
    print("Starting unified experiment analysis...")
    report = analyze_single_experiment(
        experiment_path, "unified_analysis", create_visualizations=True
    )

    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Total simulations: {report['experiment_info']['total_simulations']}")
    print(f"Valid simulations: {report['experiment_info']['valid_simulations']}")

    if (
        "population_analysis" in report
        and "peak_value" in report["population_analysis"]
    ):
        pop_analysis = report["population_analysis"]
        print(f"Peak population: {pop_analysis['peak_value']:.1f}")
        print(f"Survival rate: {pop_analysis['survival_rate']:.1f}")
    if "action_analysis" in report:
        action_analysis = report["action_analysis"]
        if "action_counts" in action_analysis:
            total_actions = sum(action_analysis["action_counts"].values())
            print(f"Total actions across simulations: {total_actions}")

    if "reproduction_analysis" in report:
        repro_analysis = report["reproduction_analysis"]
        if "overall_success_rate" in repro_analysis:
            print(".1%")

    print("\nDetailed results saved to: unified_analysis/")
    print("Run completed successfully!")


if __name__ == "__main__":
    main()

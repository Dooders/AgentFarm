"""Experiment analysis module for analyzing multiple simulation runs.

This module provides tools for analyzing and comparing multiple simulation runs,
generating visualizations and reports of key metrics and patterns.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy import Integer, create_engine, func
from sqlalchemy.orm import sessionmaker

from farm.database.models import (
    ActionModel,
    HealthIncident,
    LearningExperienceModel,
    ReproductionEventModel,
    Simulation,
    SimulationComparison,
    SimulationDifference,
    SimulationStepModel,
)
from farm.database.research_db_client import ResearchDBClient
from farm.database.research_models import ExperimentStats

logger = logging.getLogger(__name__)


class ExperimentAnalyzer:
    """Analyzes multiple simulation experiments and generates comparative analysis."""

    def __init__(self, db_paths: List[str], output_dir: str):
        """Initialize analyzer with database paths and output directory.

        Parameters
        ----------
        db_paths : List[str]
            List of paths to simulation databases to analyze
        output_dir : str
            Directory where analysis results will be saved
        """
        self.db_paths = db_paths
        self.output_dir = output_dir
        self.engines = {path: create_engine(f"sqlite:///{path}") for path in db_paths}
        self.sessions = {
            path: sessionmaker(bind=engine)() for path, engine in self.engines.items()
        }

        # Setup logging
        os.makedirs(output_dir, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the analyzer."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Create handlers
        log_path = os.path.join(self.output_dir, "experiment_analysis.log")
        file_handler = logging.FileHandler(log_path)
        console_handler = logging.StreamHandler()

        # Create formatters
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def analyze_experiments(self) -> Dict[str, Any]:
        """Analyze all experiments and generate comparative analysis.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing analysis results
        """
        try:
            self.logger.info(f"Starting analysis of {len(self.db_paths)} experiments")

            results = {
                "metadata": self._analyze_metadata(),
                "population": self._analyze_population_dynamics(),
                "resources": self._analyze_resource_dynamics(),
                "learning": self._analyze_learning_outcomes(),
                "reproduction": self._analyze_reproduction(),
                "health": self._analyze_health_incidents(),
                "actions": self._analyze_actions(),
                "comparisons": self._generate_comparisons(),
            }

            # Save results
            self._save_results(results)

            return results

        except Exception as e:
            self.logger.error(
                f"Error during experiment analysis: {str(e)}", exc_info=True
            )
            raise
        finally:
            self._cleanup()

    def _analyze_metadata(self) -> Dict[str, Any]:
        """Analyze metadata across all experiments."""
        metadata = {}

        for db_path, session in self.sessions.items():
            simulation = session.query(Simulation).first()
            if simulation:
                metadata[db_path] = {
                    "simulation_id": simulation.simulation_id,
                    "start_time": simulation.start_time.isoformat(),
                    "end_time": (
                        simulation.end_time.isoformat() if simulation.end_time else None
                    ),
                    "status": simulation.status,
                    "parameters": simulation.parameters,
                    "duration": (
                        (simulation.end_time - simulation.start_time).total_seconds()
                        if simulation.end_time
                        else None
                    ),
                }

        return metadata

    def _analyze_population_dynamics(self) -> Dict[str, Any]:
        """Analyze population dynamics across experiments."""
        data = {"populations": {}, "agent_types": {}, "statistics": {}}

        # Collect population data from each experiment
        for db_path, session in self.sessions.items():
            steps = (
                session.query(SimulationStepModel)
                .order_by(SimulationStepModel.step_number)
                .all()
            )

            data["populations"][db_path] = {
                "total_agents": [step.total_agents for step in steps],
                "steps": [step.step_number for step in steps],
            }

            data["agent_types"][db_path] = {
                "system": [step.system_agents for step in steps],
                "independent": [step.independent_agents for step in steps],
                "control": [step.control_agents for step in steps],
            }

        # Calculate statistics
        all_populations = [
            pop for exp in data["populations"].values() for pop in exp["total_agents"]
        ]

        data["statistics"] = {
            "mean_population": np.mean(all_populations),
            "std_population": np.std(all_populations),
            "max_population": np.max(all_populations),
            "min_population": np.min(all_populations),
        }

        # Generate visualization
        plt.figure(figsize=(12, 6))
        for db_path, pop_data in data["populations"].items():
            plt.plot(
                pop_data["steps"],
                pop_data["total_agents"],
                label=os.path.basename(db_path),
                alpha=0.7,
            )
        plt.title("Population Dynamics Across Experiments")
        plt.xlabel("Simulation Step")
        plt.ylabel("Total Agents")
        plt.legend()
        plt.grid(True, alpha=0.3)

        data["plot"] = plt.gcf()

        return data

    def _analyze_resource_dynamics(self) -> Dict[str, Any]:
        """Analyze resource dynamics across experiments."""
        data = {"resources": {}, "efficiency": {}, "statistics": {}}

        # Collect resource data from each experiment
        for db_path, session in self.sessions.items():
            steps = (
                session.query(SimulationStepModel)
                .order_by(SimulationStepModel.step_number)
                .all()
            )

            data["resources"][db_path] = {
                "total": [step.total_resources for step in steps],
                "average": [step.average_agent_resources for step in steps],
                "steps": [step.step_number for step in steps],
            }

            data["efficiency"][db_path] = [step.resource_efficiency for step in steps]

        # Calculate statistics
        all_resources = [
            res for exp in data["resources"].values() for res in exp["total"]
        ]
        all_efficiency = [eff for exp in data["efficiency"].values() for eff in exp]

        data["statistics"] = {
            "mean_resources": np.mean(all_resources),
            "std_resources": np.std(all_resources),
            "mean_efficiency": np.mean(all_efficiency),
            "std_efficiency": np.std(all_efficiency),
        }

        # Generate visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot total resources
        for db_path, res_data in data["resources"].items():
            ax1.plot(
                res_data["steps"],
                res_data["total"],
                label=os.path.basename(db_path),
                alpha=0.7,
            )
        ax1.set_title("Total Resources Across Experiments")
        ax1.set_xlabel("Simulation Step")
        ax1.set_ylabel("Total Resources")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot efficiency
        for db_path, eff_data in data["efficiency"].items():
            ax2.plot(
                data["resources"][db_path]["steps"],
                eff_data,
                label=os.path.basename(db_path),
                alpha=0.7,
            )
        ax2.set_title("Resource Efficiency Across Experiments")
        ax2.set_xlabel("Simulation Step")
        ax2.set_ylabel("Efficiency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        data["plot"] = plt.gcf()

        return data

    def _analyze_learning_outcomes(self) -> Dict[str, Any]:
        """Analyze learning outcomes across experiments."""
        data = {"experiences": {}, "rewards": {}, "statistics": {}}

        # Collect learning data from each experiment
        for db_path, session in self.sessions.items():
            experiences = session.query(LearningExperienceModel).all()

            data["experiences"][db_path] = {
                exp.module_type: {"rewards": [], "steps": []} for exp in experiences
            }

            for exp in experiences:
                data["experiences"][db_path][exp.module_type]["rewards"].append(
                    exp.reward
                )
                data["experiences"][db_path][exp.module_type]["steps"].append(
                    exp.step_number
                )

        # Calculate statistics
        all_rewards = [
            reward
            for exp in data["experiences"].values()
            for module in exp.values()
            for reward in module["rewards"]
        ]

        data["statistics"] = {
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "max_reward": np.max(all_rewards),
            "min_reward": np.min(all_rewards),
        }

        # Generate visualization
        plt.figure(figsize=(12, 6))
        for db_path, exp_data in data["experiences"].items():
            for module_type, module_data in exp_data.items():
                plt.plot(
                    module_data["steps"],
                    module_data["rewards"],
                    label=f"{os.path.basename(db_path)} - {module_type}",
                    alpha=0.7,
                )
        plt.title("Learning Rewards Across Experiments")
        plt.xlabel("Simulation Step")
        plt.ylabel("Reward")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        data["plot"] = plt.gcf()

        return data

    def _analyze_reproduction(self) -> Dict[str, Any]:
        """Analyze reproduction patterns across experiments."""
        data = {"events": {}, "success_rates": {}, "statistics": {}}

        # Collect reproduction data from each experiment
        for db_path, session in self.sessions.items():
            events = session.query(ReproductionEventModel).all()

            data["events"][db_path] = {
                "success": [event for event in events if event.success],
                "failure": [event for event in events if not event.success],
                "steps": [event.step_number for event in events],
            }

            total_events = len(events)
            if total_events > 0:
                data["success_rates"][db_path] = (
                    len(data["events"][db_path]["success"]) / total_events
                )
            else:
                data["success_rates"][db_path] = 0

        # Calculate statistics
        data["statistics"] = {
            "mean_success_rate": np.mean(list(data["success_rates"].values())),
            "std_success_rate": np.std(list(data["success_rates"].values())),
            "total_events": sum(
                len(events["success"]) + len(events["failure"])
                for events in data["events"].values()
            ),
        }

        # Generate visualization
        plt.figure(figsize=(12, 6))
        x = range(len(data["success_rates"]))
        plt.bar(
            x,
            data["success_rates"].values(),
            tick_label=[
                os.path.basename(path) for path in data["success_rates"].keys()
            ],
        )
        plt.title("Reproduction Success Rates Across Experiments")
        plt.xlabel("Experiment")
        plt.ylabel("Success Rate")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        data["plot"] = plt.gcf()

        return data

    def _analyze_health_incidents(self) -> Dict[str, Any]:
        """Analyze health incidents across experiments."""
        data = {"incidents": {}, "causes": {}, "statistics": {}}

        # Collect health incident data from each experiment
        for db_path, session in self.sessions.items():
            incidents = session.query(HealthIncident).all()

            data["incidents"][db_path] = {
                "impacts": [
                    incident.health_after - incident.health_before
                    for incident in incidents
                ],
                "steps": [incident.step_number for incident in incidents],
            }

            # Count causes
            data["causes"][db_path] = {}
            for incident in incidents:
                data["causes"][db_path][incident.cause] = (
                    data["causes"][db_path].get(incident.cause, 0) + 1
                )

        # Calculate statistics
        all_impacts = [
            impact for exp in data["incidents"].values() for impact in exp["impacts"]
        ]

        data["statistics"] = {
            "mean_impact": np.mean(all_impacts),
            "std_impact": np.std(all_impacts),
            "total_incidents": len(all_impacts),
        }

        # Generate visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot impact distribution
        for db_path, inc_data in data["incidents"].items():
            ax1.hist(
                inc_data["impacts"], label=os.path.basename(db_path), alpha=0.5, bins=20
            )
        ax1.set_title("Health Impact Distribution")
        ax1.set_xlabel("Health Impact")
        ax1.set_ylabel("Frequency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot cause distribution
        cause_data = {}
        for exp_causes in data["causes"].values():
            for cause, count in exp_causes.items():
                cause_data[cause] = cause_data.get(cause, 0) + count

        ax2.bar(cause_data.keys(), cause_data.values())
        ax2.set_title("Incident Causes Across All Experiments")
        ax2.set_xlabel("Cause")
        ax2.set_ylabel("Count")
        plt.xticks(rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        data["plot"] = plt.gcf()

        return data

    def _analyze_actions(self) -> Dict[str, Any]:
        """Analyze actions across experiments."""
        data = {"actions": {}, "rewards": {}, "statistics": {}}

        # Collect action data from each experiment
        for db_path, session in self.sessions.items():
            actions = session.query(ActionModel).all()

            # Group by action type
            data["actions"][db_path] = {}
            data["rewards"][db_path] = {}

            for action in actions:
                if action.action_type not in data["actions"][db_path]:
                    data["actions"][db_path][action.action_type] = []
                    data["rewards"][db_path][action.action_type] = []

                data["actions"][db_path][action.action_type].append(action)
                if action.reward is not None:
                    data["rewards"][db_path][action.action_type].append(action.reward)

        # Calculate statistics
        data["statistics"] = {"action_counts": {}, "reward_stats": {}}

        for exp_actions in data["actions"].values():
            for action_type, actions in exp_actions.items():
                if action_type not in data["statistics"]["action_counts"]:
                    data["statistics"]["action_counts"][action_type] = 0
                data["statistics"]["action_counts"][action_type] += len(actions)

        for exp_rewards in data["rewards"].values():
            for action_type, rewards in exp_rewards.items():
                if rewards:  # Only calculate if we have rewards
                    if action_type not in data["statistics"]["reward_stats"]:
                        data["statistics"]["reward_stats"][action_type] = {
                            "mean": np.mean(rewards),
                            "std": np.std(rewards),
                            "max": np.max(rewards),
                            "min": np.min(rewards),
                        }

        # Generate visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot action distribution
        action_types = list(data["statistics"]["action_counts"].keys())
        counts = list(data["statistics"]["action_counts"].values())
        ax1.bar(action_types, counts)
        ax1.set_title("Action Distribution Across All Experiments")
        ax1.set_xlabel("Action Type")
        ax1.set_ylabel("Count")
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        ax1.grid(True, alpha=0.3)

        # Plot reward distribution by action type
        for action_type in action_types:
            if action_type in data["statistics"]["reward_stats"]:
                stats = data["statistics"]["reward_stats"][action_type]
                ax2.boxplot(
                    [
                        rewards
                        for exp_rewards in data["rewards"].values()
                        for rewards in exp_rewards.get(action_type, [])
                    ],
                    positions=[action_types.index(action_type)],
                    labels=[action_type],
                )
        ax2.set_title("Reward Distribution by Action Type")
        ax2.set_xlabel("Action Type")
        ax2.set_ylabel("Reward")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        data["plot"] = plt.gcf()

        return data

    def _generate_comparisons(self) -> List[SimulationDifference]:
        """Generate pairwise comparisons between simulations."""
        comparisons = []

        # Get all pairs of simulations
        sim_pairs = [
            (path1, path2)
            for i, path1 in enumerate(self.db_paths)
            for path2 in self.db_paths[i + 1 :]
        ]

        for path1, path2 in sim_pairs:
            session1 = self.sessions[path1]
            session2 = self.sessions[path2]

            sim1 = session1.query(Simulation).first()
            sim2 = session2.query(Simulation).first()

            if sim1 and sim2:
                comparison = SimulationComparison(sim1, sim2)
                diff = comparison.compare(session1)
                comparisons.append(diff)

        return comparisons

    def _save_results(self, results: Dict[str, Any]):
        """Save analysis results to output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create experiment-specific directory
        exp_dir = os.path.join(self.output_dir, f"experiment_analysis_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)

        # Save plots
        for analysis_type, result in results.items():
            if isinstance(result, dict) and "plot" in result:
                plot_path = os.path.join(exp_dir, f"{analysis_type}_plot.png")
                result["plot"].savefig(plot_path)
                plt.close(result["plot"])
                result["plot_path"] = plot_path
                del result["plot"]

        # Save data
        data_path = os.path.join(exp_dir, "analysis_data.json")
        with open(data_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Generate report
        report_path = os.path.join(exp_dir, "analysis_report.md")
        self._generate_report(results, report_path)

        self.logger.info(f"Analysis results saved to {exp_dir}")

    def _generate_report(self, results: Dict[str, Any], output_path: str):
        """Generate markdown report from analysis results."""
        with open(output_path, "w") as f:
            f.write("# Experiment Analysis Report\n\n")

            # Write experiment overview
            f.write("## Overview\n")
            f.write(f"- Number of experiments analyzed: {len(self.db_paths)}\n")
            f.write(f"- Analysis timestamp: {datetime.now().isoformat()}\n\n")

            # Write section for each analysis type
            for analysis_type, result in results.items():
                if isinstance(result, dict) and "statistics" in result:
                    f.write(f"## {analysis_type.replace('_', ' ').title()}\n")
                    f.write("\n### Statistics\n")
                    for stat_name, stat_value in result["statistics"].items():
                        f.write(
                            f"- {stat_name.replace('_', ' ').title()}: {stat_value}\n"
                        )
                    f.write("\n")

            # Write comparison summary
            if "comparisons" in results:
                f.write("## Experiment Comparisons\n")
                for i, comparison in enumerate(results["comparisons"]):
                    f.write(f"\n### Comparison {i+1}\n")
                    for field, diff in comparison.metadata_diff.items():
                        f.write(f"- {field}: {diff[0]} vs {diff[1]}\n")

            # Write artifacts section
            f.write("\n## Analysis Artifacts\n")
            f.write("The following files contain detailed analysis data:\n\n")
            f.write("- analysis_data.json: Raw analysis data\n")
            for analysis_type, result in results.items():
                if isinstance(result, dict) and "plot_path" in result:
                    f.write(
                        f"- {analysis_type}_plot.png: {analysis_type.replace('_', ' ').title()} visualization\n"
                    )

    def _cleanup(self):
        """Clean up database sessions."""
        for session in self.sessions.values():
            session.close()


def analyze_experiment_iterations(
    experiment_path: str, research_name: str = None, output_dir: str = None
) -> None:
    """
    Analyze multiple simulation iterations of an experiment.

    Parameters
    ----------
    experiment_path : str
        Path to experiment directory
    research_name : str, optional
        Name of the research project this experiment belongs to.
        If not provided, will use parent directory name
    output_dir : str, optional
        Directory to save analysis results
    """
    # Normalize paths to use forward slashes
    experiment_path = os.path.normpath(experiment_path).replace("\\", "/")

    # Use experiment directory if no output_dir specified
    if output_dir is None:
        output_dir = os.path.join(experiment_path, "analysis").replace("\\", "/")
    else:
        output_dir = os.path.normpath(output_dir).replace("\\", "/")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create/connect to research database
    research_db_path = os.path.join(os.path.dirname(__file__), "research.db")
    research_client = ResearchDBClient(research_db_path)

    try:
        # Get or create research project
        if research_name is None:
            research_name = os.path.basename(os.path.dirname(experiment_path))

        research = research_client.get_or_create_research(research_name)

        # Find all simulation databases
        sim_dbs = []
        for root, dirs, files in os.walk(experiment_path):
            if "simulation.db" in files:
                db_path = os.path.join(root, "simulation.db").replace("\\", "/")
                sim_dbs.append(db_path)
                logger.info(f"Found simulation database: {db_path}")

        if not sim_dbs:
            raise ValueError(f"No simulation databases found in {experiment_path}")

        logger.info(f"Found {len(sim_dbs)} simulation iterations")

        # Process each simulation database
        aggregated_data = {
            "population_stats": [],
            "resource_stats": [],
            "reproduction_stats": [],
        }

        for sim_db in sim_dbs:
            iteration_id = os.path.basename(os.path.dirname(sim_db))
            logger.info(f"Processing iteration {iteration_id}")

            # Connect to simulation database
            sim_engine = create_engine(f"sqlite:///{sim_db}")
            SimSession = sessionmaker(bind=sim_engine)
            sim_session = SimSession()

            try:
                # Get population statistics
                pop_stats = sim_session.query(
                    func.avg(SimulationStepModel.total_agents).label("avg_population"),
                    func.max(SimulationStepModel.total_agents).label("max_population"),
                    func.min(SimulationStepModel.total_agents).label("min_population"),
                ).first()

                # Get resource statistics
                resource_stats = sim_session.query(
                    func.avg(SimulationStepModel.total_resources).label(
                        "avg_resources"
                    ),
                    func.avg(SimulationStepModel.resource_efficiency).label(
                        "avg_efficiency"
                    ),
                ).first()

                # Get reproduction statistics
                repro_stats = sim_session.query(
                    func.count(ReproductionEventModel.event_id).label("total_attempts"),
                    func.sum(ReproductionEventModel.success.cast(Integer)).label(
                        "successes"
                    ),
                ).first()

                # Add to aggregated data
                aggregated_data["population_stats"].append(
                    {
                        "iteration_id": iteration_id,
                        "avg_population": float(pop_stats.avg_population or 0),
                        "max_population": int(pop_stats.max_population or 0),
                        "min_population": int(pop_stats.min_population or 0),
                    }
                )

                aggregated_data["resource_stats"].append(
                    {
                        "iteration_id": iteration_id,
                        "avg_resources": float(resource_stats.avg_resources or 0),
                        "avg_efficiency": float(resource_stats.avg_efficiency or 0),
                    }
                )

                total_attempts = repro_stats.total_attempts or 0
                successes = repro_stats.successes or 0
                aggregated_data["reproduction_stats"].append(
                    {
                        "iteration_id": iteration_id,
                        "total_attempts": total_attempts,
                        "successes": successes,
                        "success_rate": float(
                            successes / total_attempts if total_attempts > 0 else 0
                        ),
                    }
                )

            finally:
                sim_session.close()

        # Create experiment stats record
        experiment_stats = research_client.add_experiment_stats(
            research_id=research.id,
            experiment_id=os.path.basename(experiment_path),
            num_iterations=len(sim_dbs),
            population_stats={
                "mean": np.mean(
                    [s["avg_population"] for s in aggregated_data["population_stats"]]
                ),
                "std": np.std(
                    [s["avg_population"] for s in aggregated_data["population_stats"]]
                ),
                "max": max(
                    s["max_population"] for s in aggregated_data["population_stats"]
                ),
                "min": min(
                    s["min_population"] for s in aggregated_data["population_stats"]
                ),
            },
            resource_stats={
                "mean_resources": np.mean(
                    [s["avg_resources"] for s in aggregated_data["resource_stats"]]
                ),
                "std_resources": np.std(
                    [s["avg_resources"] for s in aggregated_data["resource_stats"]]
                ),
                "mean_efficiency": np.mean(
                    [s["avg_efficiency"] for s in aggregated_data["resource_stats"]]
                ),
                "std_efficiency": np.std(
                    [s["avg_efficiency"] for s in aggregated_data["resource_stats"]]
                ),
            },
            reproduction_stats={
                "mean_success_rate": np.mean(
                    [s["success_rate"] for s in aggregated_data["reproduction_stats"]]
                ),
                "std_success_rate": np.std(
                    [s["success_rate"] for s in aggregated_data["reproduction_stats"]]
                ),
                "total_attempts": sum(
                    s["total_attempts"] for s in aggregated_data["reproduction_stats"]
                ),
                "total_successes": sum(
                    s["successes"] for s in aggregated_data["reproduction_stats"]
                ),
            },
        )

        # Add iteration stats
        for pop_stat, res_stat, repro_stat in zip(
            aggregated_data["population_stats"],
            aggregated_data["resource_stats"],
            aggregated_data["reproduction_stats"],
        ):
            research_client.add_iteration_stats(
                experiment_id=experiment_stats.id,
                iteration_id=pop_stat["iteration_id"],
                population_stats={
                    "avg": pop_stat["avg_population"],
                    "max": pop_stat["max_population"],
                    "min": pop_stat["min_population"],
                },
                resource_stats={
                    "avg_resources": res_stat["avg_resources"],
                    "efficiency": res_stat["avg_efficiency"],
                },
                reproduction_stats={
                    "attempts": repro_stat["total_attempts"],
                    "successes": repro_stat["successes"],
                    "success_rate": repro_stat["success_rate"],
                },
            )

        # Generate report
        report_path = os.path.join(output_dir, "experiment_report.md")
        _generate_experiment_report(experiment_stats, aggregated_data, report_path)

        logger.info(f"Analysis complete. Results saved to {output_dir}")

    finally:
        # No need to close sessions as the client handles that
        pass


def _generate_experiment_report(
    experiment_stats: ExperimentStats,
    aggregated_data: Dict[str, List[Dict]],
    output_path: str,
) -> None:
    """Generate markdown report for experiment analysis."""
    with open(output_path, "w") as f:
        f.write("# Experiment Analysis Report\n\n")

        # Overview
        f.write("## Overview\n")
        f.write(f"- Total iterations: {experiment_stats.num_iterations}\n")
        f.write(f"- Analysis timestamp: {datetime.now().isoformat()}\n\n")

        # Population statistics
        f.write("## Population Statistics\n")
        f.write(f"- Mean: {experiment_stats.mean_population:.2f}\n")
        f.write(f"- Standard Deviation: {experiment_stats.std_population:.2f}\n")
        f.write(f"- Maximum: {experiment_stats.max_population}\n")
        f.write(f"- Minimum: {experiment_stats.min_population}\n")
        f.write("\n")

        # Resource statistics
        f.write("## Resource Statistics\n")
        f.write(f"- Mean Resources: {experiment_stats.mean_resources:.2f}\n")
        f.write(f"- Standard Deviation: {experiment_stats.std_resources:.2f}\n")
        f.write(f"- Mean Efficiency: {experiment_stats.mean_efficiency:.2f}\n")
        f.write(
            f"- Efficiency Standard Deviation: {experiment_stats.std_efficiency:.2f}\n"
        )
        f.write("\n")

        # Reproduction statistics
        f.write("## Reproduction Statistics\n")
        f.write(f"- Mean Success Rate: {experiment_stats.mean_success_rate:.2%}\n")
        f.write(
            f"- Success Rate Standard Deviation: {experiment_stats.std_success_rate:.2%}\n"
        )
        f.write(f"- Total Attempts: {experiment_stats.total_reproduction_attempts}\n")
        f.write(
            f"- Total Successes: {experiment_stats.total_successful_reproductions}\n"
        )
        f.write("\n")

        # Iteration details
        f.write("## Individual Iteration Summary\n")
        for i, pop_stat in enumerate(aggregated_data["population_stats"]):
            f.write(f"\n### Iteration {pop_stat['iteration_id']}\n")
            f.write(f"- Average population: {pop_stat['avg_population']:.2f}\n")
            f.write(f"- Peak population: {pop_stat['max_population']}\n")

            # Add reproduction stats
            repro_stat = aggregated_data["reproduction_stats"][i]
            f.write(f"- Reproduction success rate: {repro_stat['success_rate']:.2%}\n")

            # Add resource stats
            resource_stat = aggregated_data["resource_stats"][i]
            f.write(f"- Average resources: {resource_stat['avg_resources']:.2f}\n")


def main():
    """Run experiment analysis from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze simulation experiments")
    parser.add_argument("experiment_path", help="Path to experiment directory")
    parser.add_argument(
        "--output-dir", help="Optional directory to save analysis results"
    )

    args = parser.parse_args()

    analyze_experiment_iterations(args.experiment_path, args.output_dir)


if __name__ == "__main__":
    main()

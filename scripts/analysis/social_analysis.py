#!/usr/bin/env python3
"""
social_analysis.py

Consolidated social analysis module combining functionality from:
- competition_analysis.py (combat and competitive behaviors)
- cooperation_analysis.py (sharing and cooperative behaviors)
- social_analysis.py (social networks and interactions)

This module provides unified analysis of social dynamics in agent simulations,
eliminating code duplication while maintaining all functionality.
"""

import glob
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

# Import our utility modules
from ..data_extraction import (
    get_initial_positions,
    extract_time_series,
    get_simulation_steps_range
)
from ..database_utils import (
    create_database_session,
    get_simulation_folders,
    get_simulation_database_path,
    get_simulation_config_path,
    validate_simulation_folder,
    get_iteration_number,
    get_final_step_number,
    get_agent_counts_by_type,
    get_action_types,
    find_action_type,
    safe_close_session
)
from ..visualization_utils import (
    setup_plot_style,
    create_time_series_plot,
    create_histogram,
    create_box_plot,
    save_figure
)

# Import database models
from farm.database.models import (
    ActionModel,
    AgentModel,
    HealthIncident,
    ReproductionEventModel,
    SimulationStepModel,
)


class SocialDynamicsAnalyzer:
    """
    Unified analyzer for social dynamics including competition, cooperation, and social networks.
    """

    def __init__(self, experiment_path: str, output_dir: str = "social_analysis_results"):
        """
        Initialize the social dynamics analyzer.

        Parameters
        ----------
        experiment_path : str
            Path to experiment directory containing simulation folders
        output_dir : str
            Output directory for results
        """
        self.experiment_path = experiment_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Get simulation folders
        self.sim_folders = get_simulation_folders(experiment_path)
        self.valid_folders = [f for f in self.sim_folders if validate_simulation_folder(f)]

        self.logger.info(f"Found {len(self.sim_folders)} simulation folders, {len(self.valid_folders)} valid")

    def analyze_competition_metrics(self) -> pd.DataFrame:
        """
        Analyze competition metrics from all simulations.
        Based on competition_analysis.py functionality.
        """
        self.logger.info("Analyzing competition metrics...")

        results = []

        for folder in self.valid_folders:
            try:
                iteration = get_iteration_number(folder)
                db_path = get_simulation_database_path(folder)

                session = create_database_session(db_path)
                sim_metrics = self._collect_competition_metrics(session, iteration)
                results.append(sim_metrics)
                session.close()

                self.logger.info(f"Analyzed competition in simulation {iteration}")

            except Exception as e:
                self.logger.error(f"Error analyzing competition in {folder}: {e}")
                continue

        return pd.DataFrame(results) if results else pd.DataFrame()

    def analyze_cooperation_metrics(self) -> pd.DataFrame:
        """
        Analyze cooperation metrics from all simulations.
        Based on cooperation_analysis.py functionality.
        """
        self.logger.info("Analyzing cooperation metrics...")

        results = []

        for folder in self.valid_folders:
            try:
                iteration = get_iteration_number(folder)
                db_path = get_simulation_database_path(folder)

                session = create_database_session(db_path)
                sim_metrics = self._collect_cooperation_metrics(session, iteration)
                results.append(sim_metrics)
                session.close()

                self.logger.info(f"Analyzed cooperation in simulation {iteration}")

            except Exception as e:
                self.logger.error(f"Error analyzing cooperation in {folder}: {e}")
                continue

        return pd.DataFrame(results) if results else pd.DataFrame()

    def _collect_competition_metrics(self, session: sqlalchemy.orm.Session, iteration: int) -> Dict[str, Any]:
        """Collect competition metrics from a single simulation."""
        metrics = {"iteration": iteration}

        try:
            # Get final step number
            final_step = get_final_step_number(session)
            metrics["final_step"] = final_step

            # Get agent counts by type
            agent_counts = get_agent_counts_by_type(session)
            metrics.update(agent_counts)
            total_agents = sum(agent_counts.values())
            metrics["total_agents"] = total_agents

            # Get combat metrics from simulation steps
            final_step_data = session.query(SimulationStepModel).filter(
                SimulationStepModel.step_number == final_step
            ).first()

            if final_step_data:
                # Combat encounters
                if hasattr(final_step_data, "combat_encounters"):
                    metrics["total_combat_encounters"] = final_step_data.combat_encounters
                    metrics["avg_combat_encounters_per_step"] = (
                        final_step_data.combat_encounters / final_step
                        if final_step > 0
                        else 0
                    )

                # Successful attacks
                if hasattr(final_step_data, "successful_attacks"):
                    metrics["total_successful_attacks"] = final_step_data.successful_attacks
                    metrics["avg_successful_attacks_per_step"] = (
                        final_step_data.successful_attacks / final_step
                        if final_step > 0
                        else 0
                    )
                    metrics["attack_success_rate"] = (
                        final_step_data.successful_attacks / final_step_data.combat_encounters
                        if hasattr(final_step_data, "combat_encounters") and final_step_data.combat_encounters > 0
                        else 0
                    )

            # Find and analyze attack actions
            attack_action_type = find_action_type(session, ["attack", "ATTACK", "combat", "fight", "Attack"])

            if attack_action_type:
                # General attack statistics
                attack_actions = session.query(ActionModel).filter(
                    ActionModel.action_type == attack_action_type
                ).count()

                metrics["attack_actions"] = attack_actions
                metrics["attack_actions_per_step"] = attack_actions / final_step if final_step > 0 else 0
                metrics["attack_actions_per_agent"] = attack_actions / total_agents if total_agents > 0 else 0

                # Successful attacks from action details
                successful_attacks = session.query(ActionModel).filter(
                    ActionModel.action_type == attack_action_type,
                    ActionModel.details.like('%"success": true%'),
                ).count()

                metrics["successful_attack_actions"] = successful_attacks
                metrics["attack_success_rate_from_actions"] = (
                    successful_attacks / attack_actions if attack_actions > 0 else 0
                )

                # Analyze attack behavior by agent type
                agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]

                for agent_type in agent_types:
                    # Attacks initiated by this agent type
                    type_attack_actions = session.query(ActionModel).join(
                        AgentModel, ActionModel.agent_id == AgentModel.agent_id
                    ).filter(
                        AgentModel.agent_type == agent_type,
                        ActionModel.action_type == attack_action_type,
                    ).count()

                    metrics[f"{agent_type}_attack_actions"] = type_attack_actions

                    agent_count = metrics.get(f"{agent_type}_agents", 0)
                    if agent_count > 0:
                        metrics[f"{agent_type}_attack_actions_per_agent"] = type_attack_actions / agent_count
                    else:
                        metrics[f"{agent_type}_attack_actions_per_agent"] = 0

                    # Successful attacks by this agent type
                    type_successful_attacks = session.query(ActionModel).join(
                        AgentModel, ActionModel.agent_id == AgentModel.agent_id
                    ).filter(
                        AgentModel.agent_type == agent_type,
                        ActionModel.action_type == attack_action_type,
                        ActionModel.details.like('%"success": true%'),
                    ).count()

                    metrics[f"{agent_type}_successful_attacks"] = type_successful_attacks
                    metrics[f"{agent_type}_attack_success_rate"] = (
                        type_successful_attacks / type_attack_actions
                        if type_attack_actions > 0
                        else 0
                    )

                    # Attacks targeting this agent type
                    targeted_attacks = session.query(ActionModel).join(
                        AgentModel, ActionModel.action_target_id == AgentModel.agent_id
                    ).filter(
                        AgentModel.agent_type == agent_type,
                        ActionModel.action_type == attack_action_type,
                    ).count()

                    metrics[f"{agent_type}_targeted_attacks"] = targeted_attacks

                    if agent_count > 0:
                        metrics[f"{agent_type}_targeted_attacks_per_agent"] = targeted_attacks / agent_count
                    else:
                        metrics[f"{agent_type}_targeted_attacks_per_agent"] = 0
            else:
                self.logger.warning(f"Could not find attack action type for iteration {iteration}")
                # Set default values
                agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
                for agent_type in agent_types:
                    metrics[f"{agent_type}_attack_actions"] = 0
                    metrics[f"{agent_type}_attack_actions_per_agent"] = 0
                    metrics[f"{agent_type}_successful_attacks"] = 0
                    metrics[f"{agent_type}_attack_success_rate"] = 0
                    metrics[f"{agent_type}_targeted_attacks"] = 0
                    metrics[f"{agent_type}_targeted_attacks_per_agent"] = 0

            # Analyze health incidents related to combat
            self._analyze_combat_health_incidents(session, metrics, total_agents)

            # Calculate competition index
            if total_agents > 0:
                metrics["competition_index"] = (
                    metrics.get("attack_actions_per_agent", 0) * 10 +
                    metrics.get("attack_success_rate_from_actions", 0) * 50
                )
            else:
                metrics["competition_index"] = 0

        except Exception as e:
            self.logger.error(f"Error collecting competition metrics for iteration {iteration}: {e}")

        return metrics

    def _collect_cooperation_metrics(self, session: sqlalchemy.orm.Session, iteration: int) -> Dict[str, Any]:
        """Collect cooperation metrics from a single simulation."""
        metrics = {"iteration": iteration}

        try:
            # Get final step number
            final_step = get_final_step_number(session)
            metrics["final_step"] = final_step

            # Get agent counts by type
            agent_counts = get_agent_counts_by_type(session)
            metrics.update(agent_counts)
            total_agents = sum(agent_counts.values())
            metrics["total_agents"] = total_agents

            # Get resource sharing metrics from simulation steps
            final_step_data = session.query(SimulationStepModel).filter(
                SimulationStepModel.step_number == final_step
            ).first()

            if final_step_data and hasattr(final_step_data, "resources_shared"):
                metrics["total_resources_shared"] = final_step_data.resources_shared
                metrics["avg_resources_shared_per_step"] = (
                    final_step_data.resources_shared / final_step if final_step > 0 else 0
                )

            # Find and analyze sharing actions
            share_action_type = find_action_type(session, ["share", "SHARE", "give_resources", "transfer_resource", "give", "Share"])

            if share_action_type:
                # General sharing statistics
                share_actions = session.query(ActionModel).filter(
                    ActionModel.action_type == share_action_type
                ).count()

                metrics["share_actions"] = share_actions
                metrics["share_actions_per_step"] = share_actions / final_step if final_step > 0 else 0
                metrics["share_actions_per_agent"] = share_actions / total_agents if total_agents > 0 else 0

                # Analyze sharing behavior by agent type
                agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]

                for agent_type in agent_types:
                    # Shares initiated by this agent type
                    type_share_actions = session.query(ActionModel).join(
                        AgentModel, ActionModel.agent_id == AgentModel.agent_id
                    ).filter(
                        AgentModel.agent_type == agent_type,
                        ActionModel.action_type == share_action_type,
                    ).count()

                    metrics[f"{agent_type}_share_actions"] = type_share_actions

                    agent_count = metrics.get(f"{agent_type}_agents", 0)
                    if agent_count > 0:
                        metrics[f"{agent_type}_share_actions_per_agent"] = type_share_actions / agent_count
                    else:
                        metrics[f"{agent_type}_share_actions_per_agent"] = 0
            else:
                self.logger.warning(f"Could not find share action type for iteration {iteration}")
                # Set default values
                agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
                for agent_type in agent_types:
                    metrics[f"{agent_type}_share_actions"] = 0
                    metrics[f"{agent_type}_share_actions_per_agent"] = 0

            # Analyze reproduction cooperation
            self._analyze_reproduction_cooperation(session, metrics, agent_types)

            # Calculate cooperation index
            if total_agents > 0:
                metrics["cooperation_index"] = (
                    metrics.get("share_actions_per_agent", 0) * 10 +
                    metrics.get("reproduction_success_rate", 0) * 5
                )
            else:
                metrics["cooperation_index"] = 0

        except Exception as e:
            self.logger.error(f"Error collecting cooperation metrics for iteration {iteration}: {e}")

        return metrics

    def _analyze_combat_health_incidents(self, session: sqlalchemy.orm.Session, metrics: Dict[str, Any], total_agents: int) -> None:
        """Analyze health incidents related to combat."""
        try:
            # Get all health incidents
            health_incidents = session.query(HealthIncident).count()
            metrics["health_incidents"] = health_incidents

            # Get combat-related health incidents
            combat_health_incidents = session.query(HealthIncident).filter(
                HealthIncident.cause.like("%attack%")
            ).count()

            if combat_health_incidents == 0:
                # Try other possible causes
                for cause in ["combat", "fight", "damage"]:
                    combat_health_incidents = session.query(HealthIncident).filter(
                        HealthIncident.cause.like(f"%{cause}%")
                    ).count()
                    if combat_health_incidents > 0:
                        break

            metrics["combat_health_incidents"] = combat_health_incidents
            metrics["combat_health_ratio"] = (
                combat_health_incidents / health_incidents if health_incidents > 0 else 0
            )

            # Get average health loss from combat incidents
            avg_health_loss_result = session.query(
                func.avg(HealthIncident.health_before - HealthIncident.health_after)
            ).filter(HealthIncident.cause.like("%attack%")).scalar()

            metrics["avg_combat_health_loss"] = avg_health_loss_result or 0

            # Get health incidents by agent type
            agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
            for agent_type in agent_types:
                type_combat_incidents = session.query(HealthIncident).join(
                    AgentModel, HealthIncident.agent_id == AgentModel.agent_id
                ).filter(
                    AgentModel.agent_type == agent_type,
                    HealthIncident.cause.like("%attack%"),
                ).count()

                metrics[f"{agent_type}_combat_incidents"] = type_combat_incidents

                agent_count = metrics.get(f"{agent_type}_agents", 0)
                if agent_count > 0:
                    metrics[f"{agent_type}_combat_incidents_per_agent"] = type_combat_incidents / agent_count
                else:
                    metrics[f"{agent_type}_combat_incidents_per_agent"] = 0

        except Exception as e:
            self.logger.warning(f"Error analyzing health incidents: {e}")

    def _analyze_reproduction_cooperation(self, session: sqlalchemy.orm.Session, metrics: Dict[str, Any], agent_types: List[str]) -> None:
        """Analyze reproduction cooperation."""
        try:
            # Get reproduction data
            reproduction_events = session.query(ReproductionEventModel).count()
            successful_reproductions = session.query(ReproductionEventModel).filter(
                ReproductionEventModel.success == True
            ).count()

            metrics["reproduction_events"] = reproduction_events
            metrics["successful_reproductions"] = successful_reproductions
            metrics["reproduction_success_rate"] = (
                successful_reproductions / reproduction_events if reproduction_events > 0 else 0
            )

            # Calculate average resource transfer in successful reproductions
            if successful_reproductions > 0:
                avg_resource_transfer_result = session.query(
                    func.avg(ReproductionEventModel.parent_resources_before - ReproductionEventModel.parent_resources_after)
                ).filter(ReproductionEventModel.success == True).scalar()

                metrics["avg_reproduction_resource_transfer"] = avg_resource_transfer_result or 0

            # Analyze reproduction by agent type
            for agent_type in agent_types:
                type_reproductions = session.query(ReproductionEventModel).join(
                    AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id
                ).filter(AgentModel.agent_type == agent_type).count()

                type_successful = session.query(ReproductionEventModel).join(
                    AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id
                ).filter(
                    AgentModel.agent_type == agent_type,
                    ReproductionEventModel.success == True
                ).count()

                metrics[f"{agent_type}_reproduction_events"] = type_reproductions
                metrics[f"{agent_type}_successful_reproductions"] = type_successful
                metrics[f"{agent_type}_reproduction_success_rate"] = (
                    type_successful / type_reproductions if type_reproductions > 0 else 0
                )

        except Exception as e:
            self.logger.warning(f"Error analyzing reproduction events: {e}")

    def analyze_social_dynamics(self) -> Dict[str, Any]:
        """
        Perform comprehensive social dynamics analysis.
        """
        self.logger.info("Performing comprehensive social dynamics analysis...")

        # Analyze competition and cooperation
        competition_df = self.analyze_competition_metrics()
        cooperation_df = self.analyze_cooperation_metrics()

        # Combine results
        results = {
            "competition_analysis": {
                "data": competition_df.to_dict('records') if not competition_df.empty else [],
                "summary": self._calculate_social_summary(competition_df, "competition")
            },
            "cooperation_analysis": {
                "data": cooperation_df.to_dict('records') if not cooperation_df.empty else [],
                "summary": self._calculate_social_summary(cooperation_df, "cooperation")
            },
            "social_balance": self._calculate_social_balance(competition_df, cooperation_df)
        }

        return results

    def _calculate_social_summary(self, df: pd.DataFrame, analysis_type: str) -> Dict[str, Any]:
        """Calculate summary statistics for social analysis."""
        if df.empty:
            return {"error": "No data available"}

        summary = {
            "n_simulations": len(df),
            "mean_score": df.get(f"{analysis_type}_index", pd.Series()).mean(),
            "std_score": df.get(f"{analysis_type}_index", pd.Series()).std(),
        }

        # Agent type specific statistics
        agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
        for agent_type in agent_types:
            agent_col = f"{agent_type}_{analysis_type.replace('ion', '')}s_per_agent"
            if agent_col in df.columns:
                summary[f"{agent_type.lower()}_mean"] = df[agent_col].mean()
                summary[f"{agent_type.lower()}_std"] = df[agent_col].std()

        return summary

    def _calculate_social_balance(self, competition_df: pd.DataFrame, cooperation_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate social balance metrics."""
        if competition_df.empty or cooperation_df.empty:
            return {"error": "Insufficient data for balance calculation"}

        # Calculate average competition vs cooperation indices
        avg_competition = competition_df.get("competition_index", pd.Series()).mean()
        avg_cooperation = cooperation_df.get("cooperation_index", pd.Series()).mean()

        balance_ratio = avg_cooperation / (avg_competition + avg_cooperation) if (avg_competition + avg_cooperation) > 0 else 0

        return {
            "avg_competition_index": avg_competition,
            "avg_cooperation_index": avg_cooperation,
            "balance_ratio": balance_ratio,
            "social_climate": "cooperative" if balance_ratio > 0.6 else "competitive" if balance_ratio < 0.4 else "balanced"
        }

    def create_social_visualizations(self) -> None:
        """
        Create comprehensive social dynamics visualizations.
        """
        self.logger.info("Creating social dynamics visualizations...")

        # Analyze data
        competition_df = self.analyze_competition_metrics()
        cooperation_df = self.analyze_cooperation_metrics()

        # Create competition visualizations
        if not competition_df.empty:
            self._create_competition_visualizations(competition_df)

        # Create cooperation visualizations
        if not cooperation_df.empty:
            self._create_cooperation_visualizations(cooperation_df)

        # Create comparative visualizations
        if not competition_df.empty and not cooperation_df.empty:
            self._create_social_balance_visualizations(competition_df, cooperation_df)

    def _create_competition_visualizations(self, df: pd.DataFrame) -> None:
        """Create competition-focused visualizations."""
        setup_plot_style("default")

        # Competition index distribution
        if "competition_index" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            create_histogram(
                ax, df["competition_index"],
                title="Competition Index Distribution",
                xlabel="Competition Index",
                ylabel="Count"
            )
            save_figure(fig, os.path.join(self.output_dir, "competition_index_distribution.png"))

        # Attack actions by agent type
        agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
        attack_cols = [f"{agent_type}_attack_actions_per_agent" for agent_type in agent_types]

        if all(col in df.columns for col in attack_cols):
            fig, ax = plt.subplots(figsize=(10, 6))
            attack_data = [df[col].values for col in attack_cols]
            create_box_plot(
                ax, attack_data, agent_types,
                title="Attack Actions Per Agent by Type",
                ylabel="Actions Per Agent"
            )
            save_figure(fig, os.path.join(self.output_dir, "attack_actions_by_type.png"))

    def _create_cooperation_visualizations(self, df: pd.DataFrame) -> None:
        """Create cooperation-focused visualizations."""
        setup_plot_style("default")

        # Cooperation index distribution
        if "cooperation_index" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            create_histogram(
                ax, df["cooperation_index"],
                title="Cooperation Index Distribution",
                xlabel="Cooperation Index",
                ylabel="Count"
            )
            save_figure(fig, os.path.join(self.output_dir, "cooperation_index_distribution.png"))

        # Sharing actions by agent type
        agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
        share_cols = [f"{agent_type}_share_actions_per_agent" for agent_type in agent_types]

        if all(col in df.columns for col in share_cols):
            fig, ax = plt.subplots(figsize=(10, 6))
            share_data = [df[col].values for col in share_cols]
            create_box_plot(
                ax, share_data, agent_types,
                title="Share Actions Per Agent by Type",
                ylabel="Actions Per Agent"
            )
            save_figure(fig, os.path.join(self.output_dir, "share_actions_by_type.png"))

    def _create_social_balance_visualizations(self, competition_df: pd.DataFrame, cooperation_df: pd.DataFrame) -> None:
        """Create social balance comparative visualizations."""
        setup_plot_style("default")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Competition vs Cooperation indices
        if "competition_index" in competition_df.columns and "cooperation_index" in cooperation_df.columns:
            comp_means = [competition_df[f"{agent_type}_attack_actions_per_agent"].mean()
                         for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]]
            coop_means = [cooperation_df[f"{agent_type}_share_actions_per_agent"].mean()
                         for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]]

            agent_types = ["System", "Independent", "Control"]
            x = np.arange(len(agent_types))

            ax1.bar(x - 0.2, comp_means, 0.4, label="Competition", alpha=0.8)
            ax1.bar(x + 0.2, coop_means, 0.4, label="Cooperation", alpha=0.8)
            ax1.set_xlabel("Agent Type")
            ax1.set_ylabel("Actions Per Agent")
            ax1.set_title("Competition vs Cooperation by Agent Type")
            ax1.set_xticks(x)
            ax1.set_xticklabels(agent_types)
            ax1.legend()

        # Social balance radar chart
        balance_data = self._calculate_social_balance(competition_df, cooperation_df)

        if "balance_ratio" in balance_data:
            categories = ["Competition", "Cooperation", "Balance"]
            values = [
                balance_data.get("avg_competition_index", 0),
                balance_data.get("avg_cooperation_index", 0),
                balance_data.get("balance_ratio", 0) * 100
            ]

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # Close the polygon
            angles += angles[:1]

            ax2.plot(angles, values, 'o-', linewidth=2, label="Social Dynamics")
            ax2.fill(angles, values, alpha=0.25)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(categories)
            ax2.set_ylim(0, max(values) * 1.1)
            ax2.set_title("Social Dynamics Balance")
            ax2.grid(True)

        plt.tight_layout()
        save_figure(fig, os.path.join(self.output_dir, "social_balance_comparison.png"))


def analyze_social_dynamics(
    experiment_path: str,
    output_dir: str = "social_analysis_results",
    create_visualizations: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to analyze social dynamics in an experiment.

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
        Comprehensive social dynamics analysis
    """
    analyzer = SocialDynamicsAnalyzer(experiment_path, output_dir)

    # Perform comprehensive analysis
    results = analyzer.analyze_social_dynamics()

    # Create visualizations if requested
    if create_visualizations:
        analyzer.create_social_visualizations()

    # Save results
    results_path = os.path.join(output_dir, "social_dynamics_analysis.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    """
    Example usage of the social dynamics analyzer.
    """
    # Example experiment path (adjust as needed)
    experiment_path = "results/one_of_a_kind/experiments/data/one_of_a_kind_20250302_193353"

    if not os.path.exists(experiment_path):
        print(f"Experiment path not found: {experiment_path}")
        print("Please update the path in the main() function")
        return

    print("Starting social dynamics analysis...")
    results = analyze_social_dynamics(experiment_path, "social_dynamics_analysis", create_visualizations=True)

    # Print summary
    print("\n=== SOCIAL DYNAMICS SUMMARY ===")

    if "competition_analysis" in results and "summary" in results["competition_analysis"]:
        comp_summary = results["competition_analysis"]["summary"]
        print(".3f"        print(f"Number of simulations: {comp_summary.get('n_simulations', 0)}")

    if "cooperation_analysis" in results and "summary" in results["cooperation_analysis"]:
        coop_summary = results["cooperation_analysis"]["summary"]
        print(".3f"
    if "social_balance" in results:
        balance = results["social_balance"]
        if "social_climate" in balance:
            print(f"Social climate: {balance['social_climate']}")
        if "balance_ratio" in balance:
            print(".1%")

    print("\nDetailed results saved to: social_dynamics_analysis/")
    print("Run completed successfully!")


if __name__ == "__main__":
    main()

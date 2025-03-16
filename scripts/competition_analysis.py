import glob
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import analysis configuration
from analysis_config import (
    DATA_PATH,
    OUTPUT_PATH,
    safe_remove_directory,
    setup_logging,
    setup_analysis_directory,
    find_latest_experiment_path,
)
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from farm.database.models import (
    ActionModel,
    AgentModel,
    HealthIncident,
    SimulationStepModel,
)


def analyze_competition_metrics(experiment_path):
    """
    Analyze competition metrics from simulation databases.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment folder containing simulation databases

    Returns
    -------
    pandas.DataFrame
        DataFrame containing competition metrics for each simulation
    """
    logging.info(f"Analyzing competition metrics in {experiment_path}...")

    # Find all simulation folders
    sim_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))

    results = []

    for folder in sim_folders:
        # Check if this is a simulation folder with a database
        db_path = os.path.join(folder, "simulation.db")

        if not os.path.exists(db_path):
            continue

        try:
            # Get iteration number from folder name
            iteration = int(os.path.basename(folder).split("_")[1])

            # Connect to the database
            engine = create_engine(f"sqlite:///{db_path}")
            Session = sessionmaker(bind=engine)
            session = Session()

            # Collect simulation metrics
            sim_metrics = collect_competition_metrics(session, iteration)
            results.append(sim_metrics)

            # Close the session
            session.close()

            logging.info(f"Analyzed simulation {iteration}")

        except Exception as e:
            logging.error(f"Error analyzing simulation in {folder}: {e}")
            import traceback

            logging.error(traceback.format_exc())

    # Convert results to DataFrame
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        logging.warning("No valid simulation data found")
        return pd.DataFrame()


def collect_competition_metrics(session, iteration):
    """
    Collect competition metrics from a single simulation database.

    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        SQLAlchemy session for database access
    iteration : int
        Iteration number of the simulation

    Returns
    -------
    dict
        Dictionary of competition metrics
    """
    metrics = {"iteration": iteration}

    # Get final step number
    try:
        final_step = (
            session.query(SimulationStepModel.step_number)
            .order_by(SimulationStepModel.step_number.desc())
            .first()
        )

        if final_step is None:
            logging.warning(f"No simulation steps found for iteration {iteration}")
            return metrics

        final_step = final_step[0]
        metrics["final_step"] = final_step

        # Get agent counts by type
        agent_counts = (
            session.query(AgentModel.agent_type, func.count(AgentModel.agent_id))
            .group_by(AgentModel.agent_type)
            .all()
        )

        total_agents = 0
        for agent_type, count in agent_counts:
            metrics[f"{agent_type}_agents"] = count
            total_agents += count

        metrics["total_agents"] = total_agents

        # Get combat metrics from simulation steps
        final_step_data = (
            session.query(SimulationStepModel)
            .filter(SimulationStepModel.step_number == final_step)
            .first()
        )

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
                    final_step_data.successful_attacks
                    / final_step_data.combat_encounters
                    if hasattr(final_step_data, "combat_encounters")
                    and final_step_data.combat_encounters > 0
                    else 0
                )

        # Try to determine the correct action type for attacks
        possible_attack_actions = ["attack", "ATTACK", "combat", "fight", "Attack"]

        # First, let's check what action types exist in the database
        action_types = session.query(ActionModel.action_type).distinct().all()
        action_types = [a[0] for a in action_types]
        logging.info(f"Available action types in database: {action_types}")

        # Find the attack action type
        attack_action_type = None
        for action_type in possible_attack_actions:
            if action_type in action_types:
                attack_action_type = action_type
                logging.info(f"Found attack action type: {attack_action_type}")
                break

        if not attack_action_type:
            # If we didn't find a specific attack type, try to infer from the action types
            for action_type in action_types:
                if (
                    "attack" in action_type.lower()
                    or "combat" in action_type.lower()
                    or "fight" in action_type.lower()
                ):
                    attack_action_type = action_type
                    logging.info(f"Inferred attack action type: {attack_action_type}")
                    break

        # Analyze attack actions
        if attack_action_type:
            attack_actions = (
                session.query(ActionModel)
                .filter(ActionModel.action_type == attack_action_type)
                .count()
            )

            metrics["attack_actions"] = attack_actions
            metrics["attack_actions_per_step"] = (
                attack_actions / final_step if final_step > 0 else 0
            )
            metrics["attack_actions_per_agent"] = (
                attack_actions / total_agents if total_agents > 0 else 0
            )

            # Get successful attacks from action details
            successful_attacks = (
                session.query(ActionModel)
                .filter(
                    ActionModel.action_type == attack_action_type,
                    ActionModel.details.like('%"success": true%'),
                )
                .count()
            )

            metrics["successful_attack_actions"] = successful_attacks
            metrics["attack_success_rate_from_actions"] = (
                successful_attacks / attack_actions if attack_actions > 0 else 0
            )

            # Analyze attack behavior by agent type
            agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]

            for agent_type in agent_types:
                # Attacks initiated by this agent type
                type_attack_actions = (
                    session.query(ActionModel)
                    .join(AgentModel, ActionModel.agent_id == AgentModel.agent_id)
                    .filter(
                        AgentModel.agent_type == agent_type,
                        ActionModel.action_type == attack_action_type,
                    )
                    .count()
                )

                metrics[f"{agent_type}_attack_actions"] = type_attack_actions

                # Calculate per agent of this type
                agent_count = metrics.get(f"{agent_type}_agents", 0)
                if agent_count > 0:
                    metrics[f"{agent_type}_attack_actions_per_agent"] = (
                        type_attack_actions / agent_count
                    )
                else:
                    metrics[f"{agent_type}_attack_actions_per_agent"] = 0

                # Successful attacks by this agent type
                type_successful_attacks = (
                    session.query(ActionModel)
                    .join(AgentModel, ActionModel.agent_id == AgentModel.agent_id)
                    .filter(
                        AgentModel.agent_type == agent_type,
                        ActionModel.action_type == attack_action_type,
                        ActionModel.details.like('%"success": true%'),
                    )
                    .count()
                )

                metrics[f"{agent_type}_successful_attacks"] = type_successful_attacks
                metrics[f"{agent_type}_attack_success_rate"] = (
                    type_successful_attacks / type_attack_actions
                    if type_attack_actions > 0
                    else 0
                )

                # Attacks targeting this agent type
                targeted_attacks = 0
                try:
                    targeted_attacks = (
                        session.query(ActionModel)
                        .join(
                            AgentModel,
                            ActionModel.action_target_id == AgentModel.agent_id,
                        )
                        .filter(
                            AgentModel.agent_type == agent_type,
                            ActionModel.action_type == attack_action_type,
                        )
                        .count()
                    )
                except Exception as e:
                    logging.warning(
                        f"Could not query targeted attacks for {agent_type}: {e}"
                    )

                metrics[f"{agent_type}_targeted_attacks"] = targeted_attacks

                if agent_count > 0:
                    metrics[f"{agent_type}_targeted_attacks_per_agent"] = (
                        targeted_attacks / agent_count
                    )
                else:
                    metrics[f"{agent_type}_targeted_attacks_per_agent"] = 0
        else:
            logging.warning("Could not find attack action type in database")
            metrics["attack_actions"] = 0
            metrics["attack_actions_per_step"] = 0
            metrics["attack_actions_per_agent"] = 0

            for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
                metrics[f"{agent_type}_attack_actions"] = 0
                metrics[f"{agent_type}_attack_actions_per_agent"] = 0
                metrics[f"{agent_type}_successful_attacks"] = 0
                metrics[f"{agent_type}_attack_success_rate"] = 0
                metrics[f"{agent_type}_targeted_attacks"] = 0
                metrics[f"{agent_type}_targeted_attacks_per_agent"] = 0

        # Analyze health incidents related to combat
        try:
            # Get all health incidents
            health_incidents = session.query(HealthIncident).count()
            metrics["health_incidents"] = health_incidents

            # Get combat-related health incidents
            combat_health_incidents = (
                session.query(HealthIncident)
                .filter(HealthIncident.cause.like("%attack%"))
                .count()
            )

            if combat_health_incidents == 0:
                # Try other possible causes
                for cause in ["combat", "fight", "damage"]:
                    combat_health_incidents = (
                        session.query(HealthIncident)
                        .filter(HealthIncident.cause.like(f"%{cause}%"))
                        .count()
                    )

                    if combat_health_incidents > 0:
                        break

            metrics["combat_health_incidents"] = combat_health_incidents
            metrics["combat_health_ratio"] = (
                combat_health_incidents / health_incidents
                if health_incidents > 0
                else 0
            )

            # Get average health loss from combat incidents
            avg_health_loss = (
                session.query(
                    func.avg(HealthIncident.health_before - HealthIncident.health_after)
                )
                .filter(HealthIncident.cause.like("%attack%"))
                .scalar()
            )

            metrics["avg_combat_health_loss"] = avg_health_loss or 0

            # Get health incidents by agent type
            for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
                type_combat_incidents = (
                    session.query(HealthIncident)
                    .join(AgentModel, HealthIncident.agent_id == AgentModel.agent_id)
                    .filter(
                        AgentModel.agent_type == agent_type,
                        HealthIncident.cause.like("%attack%"),
                    )
                    .count()
                )

                metrics[f"{agent_type}_combat_incidents"] = type_combat_incidents

                agent_count = metrics.get(f"{agent_type}_agents", 0)
                if agent_count > 0:
                    metrics[f"{agent_type}_combat_incidents_per_agent"] = (
                        type_combat_incidents / agent_count
                    )
                else:
                    metrics[f"{agent_type}_combat_incidents_per_agent"] = 0

        except Exception as e:
            logging.warning(f"Error analyzing health incidents: {e}")

        # Calculate competition index (a synthetic metric combining attack actions and success)
        if total_agents > 0:
            metrics["competition_index"] = (
                metrics.get("attack_actions_per_agent", 0) * 10
            ) + (  # Weighted attack actions
                metrics.get("attack_success_rate_from_actions", 0) * 50
            )  # Weighted attack success
        else:
            metrics["competition_index"] = 0

    except Exception as e:
        logging.error(
            f"Error collecting competition metrics for iteration {iteration}: {e}"
        )
        import traceback

        logging.error(traceback.format_exc())

    return metrics


def plot_competition_distribution(df, output_path):
    """
    Plot the distribution of competition metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing competition metrics
    output_path : str
        Path to save the output plots
    """
    if "competition_index" not in df.columns or df.empty:
        logging.warning("Competition index not available for plotting")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df["competition_index"], kde=True)
    plt.title("Distribution of Competition Index Across Simulations")
    plt.xlabel("Competition Index")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_path, "competition_index_distribution.png"))
    plt.close()

    # Plot competition metrics by agent type
    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
    metrics = ["attack_actions_per_agent", "attack_success_rate"]

    for metric in metrics:
        metric_by_type = [f"{agent_type}_{metric}" for agent_type in agent_types]

        if not all(col in df.columns for col in metric_by_type):
            continue

        plt.figure(figsize=(10, 6))
        data = []

        for agent_type, col in zip(agent_types, metric_by_type):
            if col in df.columns:
                data.append(df[col].tolist())

        plt.boxplot(data, labels=[t.replace("Agent", "") for t in agent_types])
        plt.title(f"{metric.replace('_', ' ').title()} by Agent Type")
        plt.ylabel(metric.replace("_", " ").title())
        plt.savefig(os.path.join(output_path, f"{metric}_by_agent_type.png"))
        plt.close()


def plot_combat_time_series(experiment_path, output_path, max_sims=5):
    """
    Plot combat metrics over time for a sample of simulations.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment folder containing simulation databases
    output_path : str
        Path to save the output plots
    max_sims : int
        Maximum number of simulations to include in the plot
    """
    sim_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))

    if not sim_folders:
        logging.warning("No simulation folders found for time series plots")
        return

    # Sort by modification time (most recent first) and limit to max_sims
    sim_folders.sort(key=os.path.getmtime, reverse=True)
    sim_folders = sim_folders[:max_sims]

    # Plot combat encounters over time
    plt.figure(figsize=(12, 8))

    for folder in sim_folders:
        iteration = os.path.basename(folder).split("_")[1]
        db_path = os.path.join(folder, "simulation.db")

        if not os.path.exists(db_path):
            continue

        try:
            # Connect to the database
            engine = create_engine(f"sqlite:///{db_path}")

            # Query combat encounters over time
            try:
                time_series = pd.read_sql_query(
                    "SELECT step_number, combat_encounters_this_step FROM simulation_steps",
                    engine,
                )
            except:
                # Try alternative column name if needed
                try:
                    time_series = pd.read_sql_query(
                        "SELECT step_number, combat_encounters FROM simulation_steps",
                        engine,
                    )
                    time_series.rename(
                        columns={"combat_encounters": "combat_encounters_this_step"},
                        inplace=True,
                    )
                except:
                    logging.warning(
                        f"Could not find combat encounters data for iteration {iteration}"
                    )
                    continue

            if not time_series.empty:
                plt.plot(
                    time_series["step_number"],
                    time_series["combat_encounters_this_step"],
                    label=f"Sim {iteration}",
                )
        except Exception as e:
            logging.error(f"Error creating time series plot for {folder}: {e}")

    plt.title("Combat Encounters Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Combat Encounters This Step")
    plt.legend()
    plt.savefig(os.path.join(output_path, "combat_encounters_time_series.png"))
    plt.close()


def analyze_competition_factors(df, output_path):
    """
    Analyze factors that correlate with competition.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing competition metrics
    output_path : str
        Path to save the output plots
    """
    if df.empty or "competition_index" not in df.columns:
        logging.warning("Insufficient data for competition factors analysis")
        return

    # Calculate correlations with competition index
    competition_cols = [
        col
        for col in df.columns
        if any(term in col for term in ["attack", "combat", "competition"])
        and "competition_index" != col
    ]

    # Add basic simulation metrics
    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
    base_cols = ["final_step", "total_agents"] + [
        f"{agent_type}_agents" for agent_type in agent_types
    ]

    # Combine all columns for correlation analysis
    all_cols = competition_cols + [col for col in base_cols if col in df.columns]
    all_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(df[col])]

    if "competition_index" in df.columns and all_cols:
        # Calculate correlations
        correlations = (
            df[all_cols + ["competition_index"]]
            .corr()["competition_index"]
            .drop("competition_index")
        )

        # Plot correlations
        plt.figure(figsize=(12, 8))
        correlations.sort_values().plot(kind="barh")
        plt.title("Correlation with Competition Index")
        plt.xlabel("Correlation Coefficient")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "competition_correlations.png"))
        plt.close()

        # Create a correlation matrix for all competition metrics
        if len(all_cols) > 1:
            plt.figure(figsize=(14, 12))
            sns.heatmap(df[all_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix of Competition Metrics")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "competition_correlation_matrix.png"))
            plt.close()

    # Calculate attack targeting preference (who attacks whom)
    try:
        if all(
            f"{agent_type}_attack_actions" in df.columns for agent_type in agent_types
        ) and all(
            f"{agent_type}_targeted_attacks" in df.columns for agent_type in agent_types
        ):

            # Create a matrix of attack actions by and against each agent type
            agent_types_short = [t.replace("Agent", "") for t in agent_types]
            attack_matrix = pd.DataFrame(
                index=agent_types_short, columns=agent_types_short
            )

            # We can only know the total attacks by each type and against each type
            # We'll need to estimate the full matrix
            for i, attacker in enumerate(agent_types):
                for j, target in enumerate(agent_types):
                    if i == j:
                        # Self-attacks are likely rare and can be set to the minimum value
                        attack_matrix.iloc[i, j] = 0.0
                    else:
                        # For now, distribute the attacks proportionally based on population
                        attacker_attacks = float(
                            df[f"{attacker}_attack_actions"].mean()
                        )
                        target_population = float(
                            df[f"{target}_agents"].mean()
                        ) / float(df["total_agents"].mean())
                        attack_matrix.iloc[i, j] = attacker_attacks * target_population

            # Ensure the matrix contains only numeric values
            attack_matrix = attack_matrix.astype(float)

            # Normalize the matrix to show attack preferences
            row_sums = attack_matrix.sum(axis=1)
            # Handle zero row sums to avoid division by zero
            for i in range(len(row_sums)):
                if row_sums.iloc[i] == 0:
                    row_sums.iloc[i] = 1.0

            attack_matrix_normalized = attack_matrix.div(row_sums, axis=0)

            # Plot the attack matrix as a heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                attack_matrix_normalized,
                annot=True,
                cmap="Reds",
                fmt=".2f",
                vmin=0,
                vmax=1,
            )
            plt.title("Attack Targeting Preferences (Estimated)")
            plt.xlabel("Target Agent Type")
            plt.ylabel("Attacker Agent Type")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "attack_targeting_preferences.png"))
            plt.close()
    except Exception as e:
        logging.error(f"Error creating attack targeting heatmap: {e}")
        import traceback

        logging.error(traceback.format_exc())


def plot_agent_type_comparison(df, output_path):
    """
    Plot comparison of competition metrics across agent types.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing competition metrics
    output_path : str
        Path to save the output plots
    """
    if df.empty:
        logging.warning("No data for agent type comparison")
        return

    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]

    # Check for attack metrics by agent type
    attack_cols = [
        f"{agent_type}_attack_actions_per_agent" for agent_type in agent_types
    ]
    if all(col in df.columns for col in attack_cols):
        attack_data = df[attack_cols].mean().reset_index()
        attack_data.columns = ["Agent Type", "Attack Actions Per Agent"]
        attack_data["Agent Type"] = (
            attack_data["Agent Type"]
            .str.replace("_attack_actions_per_agent", "")
            .str.replace("Agent", "")
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Agent Type", y="Attack Actions Per Agent", data=attack_data)
        plt.title("Average Attack Actions Per Agent by Agent Type")
        plt.savefig(os.path.join(output_path, "attack_actions_by_agent_type.png"))
        plt.close()

    # Check for attack success rate by agent type
    success_cols = [f"{agent_type}_attack_success_rate" for agent_type in agent_types]
    if all(col in df.columns for col in success_cols):
        success_data = df[success_cols].mean().reset_index()
        success_data.columns = ["Agent Type", "Attack Success Rate"]
        success_data["Agent Type"] = (
            success_data["Agent Type"]
            .str.replace("_attack_success_rate", "")
            .str.replace("Agent", "")
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Agent Type", y="Attack Success Rate", data=success_data)
        plt.title("Average Attack Success Rate by Agent Type")
        plt.savefig(os.path.join(output_path, "attack_success_by_agent_type.png"))
        plt.close()

    # Check for targets of attacks by agent type
    target_cols = [
        f"{agent_type}_targeted_attacks_per_agent" for agent_type in agent_types
    ]
    if all(col in df.columns for col in target_cols):
        target_data = df[target_cols].mean().reset_index()
        target_data.columns = ["Agent Type", "Targeted Attacks Per Agent"]
        target_data["Agent Type"] = (
            target_data["Agent Type"]
            .str.replace("_targeted_attacks_per_agent", "")
            .str.replace("Agent", "")
        )

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Agent Type", y="Targeted Attacks Per Agent", data=target_data)
        plt.title("Average Targeted Attacks Per Agent by Agent Type")
        plt.savefig(os.path.join(output_path, "targeted_attacks_by_agent_type.png"))
        plt.close()


def analyze_competitive_action_details(df, experiment_path, output_path):
    """
    Analyze the details of competitive actions across simulations
    by examining action types and details.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing competition metrics
    experiment_path : str
        Path to experiment directory containing simulations
    output_path : str
        Path to save analysis outputs
    """
    if df.empty:
        logging.warning("No data for competitive action details analysis")
        return

    # Select a sample of simulation databases to analyze
    sim_iterations = df["iteration"].sample(min(5, len(df))).tolist()

    action_types = {}
    action_details = {}

    for iteration in sim_iterations:
        db_path = os.path.join(
            experiment_path, f"iteration_{iteration}", "simulation.db"
        )
        if not os.path.exists(db_path):
            continue

        try:
            # Connect to the database
            engine = create_engine(f"sqlite:///{db_path}")
            Session = sessionmaker(bind=engine)
            session = Session()

            # Get all action types
            all_actions = session.query(ActionModel.action_type).distinct().all()

            for action_type in [a[0] for a in all_actions]:
                if action_type not in action_types:
                    action_types[action_type] = 0

                # Count this action type
                count = (
                    session.query(ActionModel)
                    .filter(ActionModel.action_type == action_type)
                    .count()
                )

                action_types[action_type] += count

                # For potential competitive actions, sample some details
                if any(
                    term in action_type.lower()
                    for term in ["attack", "fight", "combat", "defend"]
                ):
                    # Sample some actions with details
                    sample_actions = (
                        session.query(ActionModel)
                        .filter(
                            ActionModel.action_type == action_type,
                            ActionModel.details.isnot(None),
                        )
                        .limit(10)
                        .all()
                    )

                    if sample_actions:
                        if action_type not in action_details:
                            action_details[action_type] = []

                        for action in sample_actions:
                            if (
                                action.details
                                and action.details not in action_details[action_type]
                            ):
                                action_details[action_type].append(action.details)

            session.close()

        except Exception as e:
            logging.error(
                f"Error analyzing competitive action details for iteration {iteration}: {e}"
            )

    # Log the findings
    logging.info("\nCompetitive Action Analysis:")
    logging.info("---------------------------")

    if action_types:
        logging.info("Action types found in the database:")
        for action_type, count in sorted(
            action_types.items(), key=lambda x: x[1], reverse=True
        ):
            logging.info(f"  {action_type}: {count} occurrences")

        # Identify potentially competitive actions
        comp_actions = [
            (action, count)
            for action, count in action_types.items()
            if any(
                term in action.lower()
                for term in ["attack", "fight", "combat", "defend"]
            )
        ]

        if comp_actions:
            logging.info("\nPotentially competitive actions:")
            for action, count in comp_actions:
                logging.info(f"  {action}: {count} occurrences")

                # Log sample details if available
                if action in action_details and action_details[action]:
                    logging.info(f"    Sample details:")
                    for i, detail in enumerate(action_details[action][:3]):
                        logging.info(f"      {i+1}. {detail}")
        else:
            logging.info(
                "No explicitly competitive actions identified in the database."
            )
            logging.info(
                "You may need to modify the script to look for other action types specific to your simulation."
            )
    else:
        logging.info("No action types found in the database.")

    # Also analyze health incidents to understand competition outcomes
    try:
        health_incident_causes = {}

        for iteration in sim_iterations:
            db_path = os.path.join(
                experiment_path, f"iteration_{iteration}", "simulation.db"
            )
            if not os.path.exists(db_path):
                continue

            engine = create_engine(f"sqlite:///{db_path}")
            Session = sessionmaker(bind=engine)
            session = Session()

            # Get distinct health incident causes
            causes = session.query(HealthIncident.cause).distinct().all()

            for cause in [c[0] for c in causes]:
                if cause not in health_incident_causes:
                    health_incident_causes[cause] = 0

                count = (
                    session.query(HealthIncident)
                    .filter(HealthIncident.cause == cause)
                    .count()
                )

                health_incident_causes[cause] += count

            session.close()

        if health_incident_causes:
            logging.info("\nHealth incident causes:")
            for cause, count in sorted(
                health_incident_causes.items(), key=lambda x: x[1], reverse=True
            ):
                logging.info(f"  {cause}: {count} occurrences")

            # Identify combat-related health incidents
            combat_causes = [
                (cause, count)
                for cause, count in health_incident_causes.items()
                if any(
                    term in cause.lower()
                    for term in ["attack", "combat", "fight", "damage"]
                )
            ]

            if combat_causes:
                logging.info("\nCombat-related health incidents:")
                for cause, count in combat_causes:
                    logging.info(f"  {cause}: {count} occurrences")

                    # Sample some details for this cause
                    for iteration in sim_iterations[
                        :1
                    ]:  # Just use the first simulation for details
                        db_path = os.path.join(
                            experiment_path, f"iteration_{iteration}", "simulation.db"
                        )
                        if not os.path.exists(db_path):
                            continue

                        engine = create_engine(f"sqlite:///{db_path}")
                        Session = sessionmaker(bind=engine)
                        session = Session()

                        sample_incidents = (
                            session.query(HealthIncident)
                            .filter(HealthIncident.cause == cause)
                            .limit(3)
                            .all()
                        )

                        if sample_incidents:
                            logging.info(f"    Sample health changes:")
                            for i, incident in enumerate(sample_incidents):
                                health_change = (
                                    incident.health_before - incident.health_after
                                )
                                logging.info(
                                    f"      {i+1}. Health before: {incident.health_before}, after: {incident.health_after}, change: {health_change}"
                                )

                        session.close()
                        break
    except Exception as e:
        logging.error(f"Error analyzing health incidents: {e}")


def validate_data(df):
    """
    Validate the collected data and log warnings about data quality issues.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing competition metrics

    Returns
    -------
    bool
        True if data passes basic validation, False otherwise
    """
    if df.empty:
        logging.error("No data collected for analysis")
        return False

    # Check if we have data by agent type
    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
    agent_type_cols = [f"{agent_type}_agents" for agent_type in agent_types]

    if not all(col in df.columns for col in agent_type_cols):
        logging.warning("Missing some agent type data columns")

    # Check if we have attack action data
    if "attack_actions" in df.columns and df["attack_actions"].sum() == 0:
        logging.warning("No attack actions found across all simulations")

        # Check available action types to troubleshoot
        logging.warning(
            "This may indicate that 'attack' actions are recorded differently in the database"
        )
        logging.warning("Check logs above for available action types")

    # Check if we have attack data by agent type
    attack_by_type_cols = [f"{agent_type}_attack_actions" for agent_type in agent_types]
    if all(col in df.columns for col in attack_by_type_cols):
        if df[attack_by_type_cols].sum().sum() == 0:
            logging.warning("No attack actions found by any agent type")

    # Check combat health incidents data
    if "combat_health_incidents" in df.columns:
        if df["combat_health_incidents"].sum() > 0:
            logging.info(
                f"Found {df['combat_health_incidents'].sum()} combat health incidents across all simulations"
            )
        else:
            logging.warning("No combat health incidents found in the data")

    # Check data consistency
    for col in df.columns:
        # Check for completely null columns
        if df[col].isnull().all():
            logging.warning(f"Column {col} has all null values")
        # Check for columns with all zeros when we expect non-zero values
        elif (
            col not in ["iteration", "final_step"]
            and df[col].dtype != object
            and (df[col] == 0).all()
        ):
            logging.warning(f"Column {col} has all zero values")

    return True


def main():
    # Set up the competition analysis directory
    competition_output_path, log_file = setup_analysis_directory("competition")

    # Find the most recent experiment folder
    experiment_path = find_latest_experiment_path()
    if not experiment_path:
        return

    logging.info(f"Analyzing competition in simulations in {experiment_path}...")
    df = analyze_competition_metrics(experiment_path)

    if df.empty:
        logging.warning("No simulation data found.")
        return

    # Validate the data quality
    data_valid = validate_data(df)
    if not data_valid:
        logging.warning(
            "Data validation issues detected. Analysis results may be limited."
        )

    # Analyze competitive action details
    analyze_competitive_action_details(df, experiment_path, competition_output_path)

    # Save the raw data
    output_csv = os.path.join(competition_output_path, "competition_analysis.csv")
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved analysis data to {output_csv}")

    # Display summary statistics
    logging.info("\nSummary statistics:")
    logging.info(df.describe().to_string())

    # Display average competition metrics
    logging.info("\nAverage competition metrics across simulations:")
    for metric in [
        "attack_actions_per_agent",
        "attack_success_rate_from_actions",
        "competition_index",
    ]:
        if metric in df.columns:
            avg_value = df[metric].mean()
            logging.info(f"  Average {metric}: {avg_value:.4f}")

    # Generate and save plots
    plot_competition_distribution(df, competition_output_path)
    plot_combat_time_series(experiment_path, competition_output_path)
    analyze_competition_factors(df, competition_output_path)
    plot_agent_type_comparison(df, competition_output_path)

    logging.info("\nAnalysis complete. Results saved to CSV and PNG files.")
    logging.info(f"Log file saved to: {log_file}")
    logging.info(f"All analysis files saved to: {competition_output_path}")


if __name__ == "__main__":
    main()

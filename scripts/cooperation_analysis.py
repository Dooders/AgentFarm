import glob
import logging
import os
from datetime import datetime

# Import analysis configuration
from analysis_config import DATA_PATH, OUTPUT_PATH

from analysis_config import (
    check_reproduction_events,
    safe_remove_directory,
    setup_logging,
    setup_analysis_directory,
    find_latest_experiment_path,
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker

from farm.database.models import (
    ActionModel,
    AgentModel,
    ReproductionEventModel,
    SimulationStepModel,
)


def analyze_cooperation_metrics(experiment_path):
    """
    Analyze cooperation metrics from simulation databases.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment folder containing simulation databases

    Returns
    -------
    pandas.DataFrame
        DataFrame containing cooperation metrics for each simulation
    """
    logging.info(f"Analyzing cooperation metrics in {experiment_path}...")
    
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
            sim_metrics = collect_simulation_metrics(session, iteration)
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


def collect_simulation_metrics(session, iteration):
    """
    Collect cooperation metrics from a single simulation database.
    
    Parameters
    ----------
    session : sqlalchemy.orm.session.Session
        SQLAlchemy session for database access
    iteration : int
        Iteration number of the simulation
        
    Returns
    -------
    dict
        Dictionary of cooperation metrics
    """
    metrics = {"iteration": iteration}
    
    # Get final step number
    try:
        final_step = session.query(SimulationStepModel.step_number).order_by(
            SimulationStepModel.step_number.desc()
        ).first()
        
        if final_step is None:
            logging.warning(f"No simulation steps found for iteration {iteration}")
            return metrics
            
        final_step = final_step[0]
        metrics["final_step"] = final_step
        
        # Get agent counts by type
        agent_counts = session.query(
            AgentModel.agent_type, 
            func.count(AgentModel.agent_id)
        ).group_by(AgentModel.agent_type).all()
        
        total_agents = 0
        for agent_type, count in agent_counts:
            metrics[f"{agent_type}_agents"] = count
            total_agents += count
        
        metrics["total_agents"] = total_agents
        
        # Get resource sharing metrics from simulation steps
        final_step_data = session.query(SimulationStepModel).filter(
            SimulationStepModel.step_number == final_step
        ).first()
        
        if final_step_data:
            metrics["total_resources_shared"] = final_step_data.resources_shared
            metrics["avg_resources_shared_per_step"] = final_step_data.resources_shared / final_step if final_step > 0 else 0
        
        # Try to determine the correct action type for sharing
        possible_share_actions = ["share", "SHARE", "give_resources", "transfer_resource", "give", "Share"]
        
        # First, let's check what action types exist in the database
        action_types = session.query(ActionModel.action_type).distinct().all()
        action_types = [a[0] for a in action_types]
        logging.info(f"Available action types in database: {action_types}")
        
        # Find the share action type
        share_action_type = None
        for action_type in possible_share_actions:
            if action_type in action_types:
                share_action_type = action_type
                logging.info(f"Found share action type: {share_action_type}")
                break
        
        if not share_action_type:
            # If we didn't find a specific share type, try to infer from the action types
            for action_type in action_types:
                if "share" in action_type.lower() or "give" in action_type.lower() or "transfer" in action_type.lower():
                    share_action_type = action_type
                    logging.info(f"Inferred share action type: {share_action_type}")
                    break
        
        # Analyze sharing actions from agent actions
        if share_action_type:
            share_actions = session.query(ActionModel).filter(
                ActionModel.action_type == share_action_type
            ).count()
            
            metrics["share_actions"] = share_actions
            metrics["share_actions_per_step"] = share_actions / final_step if final_step > 0 else 0
            metrics["share_actions_per_agent"] = share_actions / total_agents if total_agents > 0 else 0
            
            # Analyze sharing behavior by agent type
            agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
            
            for agent_type in agent_types:
                type_share_actions = session.query(ActionModel).join(
                    AgentModel, ActionModel.agent_id == AgentModel.agent_id
                ).filter(
                    AgentModel.agent_type == agent_type,
                    ActionModel.action_type == share_action_type
                ).count()
                
                metrics[f"{agent_type}_share_actions"] = type_share_actions
                
                # Calculate per agent of this type
                agent_count = metrics.get(f"{agent_type}_agents", 0)
                if agent_count > 0:
                    metrics[f"{agent_type}_share_actions_per_agent"] = type_share_actions / agent_count
                else:
                    metrics[f"{agent_type}_share_actions_per_agent"] = 0
        else:
            logging.warning("Could not find share action type in database")
            metrics["share_actions"] = 0
            metrics["share_actions_per_step"] = 0
            metrics["share_actions_per_agent"] = 0
            
            for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
                metrics[f"{agent_type}_share_actions"] = 0
                metrics[f"{agent_type}_share_actions_per_agent"] = 0
        
        # Analyze reproduction cooperation (if ReproductionEventModel exists)
        try:
            reproduction_events = session.query(ReproductionEventModel).count()
            successful_reproductions = session.query(ReproductionEventModel).filter(
                ReproductionEventModel.success == True
            ).count()
            
            metrics["reproduction_events"] = reproduction_events
            metrics["successful_reproductions"] = successful_reproductions
            metrics["reproduction_success_rate"] = successful_reproductions / reproduction_events if reproduction_events > 0 else 0
            
            # Calculate average resource transfer in successful reproductions
            if successful_reproductions > 0:
                avg_resource_transfer = session.query(
                    func.avg(ReproductionEventModel.parent_resources_before - ReproductionEventModel.parent_resources_after)
                ).filter(
                    ReproductionEventModel.success == True
                ).scalar()
                
                metrics["avg_reproduction_resource_transfer"] = avg_resource_transfer or 0
            
            # Analyze reproduction by agent type
            for agent_type in agent_types:
                type_reproductions = session.query(ReproductionEventModel).join(
                    AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id
                ).filter(
                    AgentModel.agent_type == agent_type
                ).count()
                
                type_successful = session.query(ReproductionEventModel).join(
                    AgentModel, ReproductionEventModel.parent_id == AgentModel.agent_id
                ).filter(
                    AgentModel.agent_type == agent_type,
                    ReproductionEventModel.success == True
                ).count()
                
                metrics[f"{agent_type}_reproduction_events"] = type_reproductions
                metrics[f"{agent_type}_successful_reproductions"] = type_successful
                metrics[f"{agent_type}_reproduction_success_rate"] = type_successful / type_reproductions if type_reproductions > 0 else 0
                
        except Exception as e:
            logging.warning(f"Error analyzing reproduction events: {e}")
            
        # Calculate cooperation index (a synthetic metric combining sharing and reproduction)
        # This is a simple example - could be more sophisticated
        if total_agents > 0:
            metrics["cooperation_index"] = (
                (metrics.get("share_actions_per_agent", 0) * 10) +  # Weighted sharing actions
                (metrics.get("reproduction_success_rate", 0) * 5)    # Weighted reproduction success
            )
        else:
            metrics["cooperation_index"] = 0
            
    except Exception as e:
        logging.error(f"Error collecting metrics for iteration {iteration}: {e}")
        import traceback
        logging.error(traceback.format_exc())
    
    return metrics


def plot_cooperation_distribution(df, output_path):
    """
    Plot the distribution of cooperation metrics.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing cooperation metrics
    output_path : str
        Path to save the output plots
    """
    if "cooperation_index" not in df.columns or df.empty:
        logging.warning("Cooperation index not available for plotting")
        return
        
    plt.figure(figsize=(10, 6))
    sns.histplot(df["cooperation_index"], kde=True)
    plt.title("Distribution of Cooperation Index Across Simulations")
    plt.xlabel("Cooperation Index")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_path, "cooperation_index_distribution.png"))
    plt.close()
    
    # Plot cooperation metrics by agent type
    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
    metrics = ["share_actions_per_agent", "reproduction_success_rate"]
    
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
        plt.ylabel(metric.replace('_', ' ').title())
        plt.savefig(os.path.join(output_path, f"{metric}_by_agent_type.png"))
        plt.close()


def plot_cooperation_time_series(experiment_path, output_path, max_sims=5):
    """
    Plot cooperation metrics over time for a sample of simulations.
    
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
    
    # Plot resources shared over time
    plt.figure(figsize=(12, 8))
    
    for folder in sim_folders:
        iteration = os.path.basename(folder).split("_")[1]
        db_path = os.path.join(folder, "simulation.db")
        
        if not os.path.exists(db_path):
            continue
            
        try:
            # Connect to the database
            engine = create_engine(f"sqlite:///{db_path}")
            
            # Query resources shared over time
            try:
                time_series = pd.read_sql_query(
                    "SELECT step_number, resources_shared_this_step FROM simulation_steps",
                    engine
                )
            except:
                # Try alternative column name if needed
                try:
                    time_series = pd.read_sql_query(
                        "SELECT step_number, resources_shared FROM simulation_steps",
                        engine
                    )
                    time_series.rename(columns={'resources_shared': 'resources_shared_this_step'}, inplace=True)
                except:
                    logging.warning(f"Could not find resources shared data for iteration {iteration}")
                    continue
            
            if not time_series.empty:
                plt.plot(
                    time_series["step_number"], 
                    time_series["resources_shared_this_step"],
                    label=f"Sim {iteration}"
                )
        except Exception as e:
            logging.error(f"Error creating time series plot for {folder}: {e}")
    
    plt.title("Resources Shared Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Resources Shared This Step")
    plt.legend()
    plt.savefig(os.path.join(output_path, "resources_shared_time_series.png"))
    plt.close()


def analyze_cooperation_factors(df, output_path):
    """
    Analyze factors that correlate with cooperation.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing cooperation metrics
    output_path : str
        Path to save the output plots
    """
    if df.empty or "cooperation_index" not in df.columns:
        logging.warning("Insufficient data for cooperation factors analysis")
        return
        
    # Calculate correlations with cooperation index
    cooperation_cols = [
        col for col in df.columns if any(term in col for term in 
        ["share", "reproduction", "cooperation"]) and 
        "cooperation_index" != col
    ]
    
    # Add basic simulation metrics
    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
    base_cols = ["final_step", "total_agents"] + [f"{agent_type}_agents" for agent_type in agent_types]
    
    # Combine all columns for correlation analysis
    all_cols = cooperation_cols + [col for col in base_cols if col in df.columns]
    all_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(df[col])]
    
    if "cooperation_index" in df.columns and all_cols:
        # Calculate correlations
        correlations = df[all_cols + ["cooperation_index"]].corr()["cooperation_index"].drop("cooperation_index")
        
        # Plot correlations
        plt.figure(figsize=(12, 8))
        correlations.sort_values().plot(kind="barh")
        plt.title("Correlation with Cooperation Index")
        plt.xlabel("Correlation Coefficient")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, "cooperation_correlations.png"))
        plt.close()
        
        # Create a correlation matrix for all cooperation metrics
        if len(all_cols) > 1:
            plt.figure(figsize=(14, 12))
            sns.heatmap(df[all_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix of Cooperation Metrics")
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "cooperation_correlation_matrix.png"))
            plt.close()


def plot_agent_type_comparison(df, output_path):
    """
    Plot comparison of cooperation metrics across agent types.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing cooperation metrics
    output_path : str
        Path to save the output plots
    """
    if df.empty:
        logging.warning("No data for agent type comparison")
        return
        
    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
    
    # Check for sharing metrics by agent type
    sharing_cols = [f"{agent_type}_share_actions_per_agent" for agent_type in agent_types]
    if all(col in df.columns for col in sharing_cols):
        sharing_data = df[sharing_cols].mean().reset_index()
        sharing_data.columns = ["Agent Type", "Share Actions Per Agent"]
        sharing_data["Agent Type"] = sharing_data["Agent Type"].str.replace("_share_actions_per_agent", "").str.replace("Agent", "")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Agent Type", y="Share Actions Per Agent", data=sharing_data)
        plt.title("Average Share Actions Per Agent by Agent Type")
        plt.savefig(os.path.join(output_path, "share_actions_by_agent_type.png"))
        plt.close()
    
    # Check for reproduction metrics by agent type
    repro_cols = [f"{agent_type}_reproduction_success_rate" for agent_type in agent_types]
    if all(col in df.columns for col in repro_cols):
        repro_data = df[repro_cols].mean().reset_index()
        repro_data.columns = ["Agent Type", "Reproduction Success Rate"]
        repro_data["Agent Type"] = repro_data["Agent Type"].str.replace("_reproduction_success_rate", "").str.replace("Agent", "")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Agent Type", y="Reproduction Success Rate", data=repro_data)
        plt.title("Average Reproduction Success Rate by Agent Type")
        plt.savefig(os.path.join(output_path, "reproduction_success_by_agent_type.png"))
        plt.close()


def validate_data(df):
    """
    Validate the collected data and log warnings about data quality issues.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing cooperation metrics
        
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
    
    # Check if we have share action data
    if "share_actions" in df.columns and df["share_actions"].sum() == 0:
        logging.warning("No share actions found across all simulations")
        
        # Check available action types to troubleshoot
        logging.warning("This may indicate that 'share' actions are recorded differently in the database")
        logging.warning("Check logs above for available action types")
    
    # Check if we have sharing data by agent type
    share_by_type_cols = [f"{agent_type}_share_actions" for agent_type in agent_types]
    if all(col in df.columns for col in share_by_type_cols):
        if df[share_by_type_cols].sum().sum() == 0:
            logging.warning("No share actions found by any agent type")
    
    # Check reproduction data
    if "reproduction_events" in df.columns:
        if df["reproduction_events"].sum() > 0:
            logging.info(f"Found {df['reproduction_events'].sum()} reproduction events across all simulations")
            if "successful_reproductions" in df.columns:
                avg_success_rate = df["successful_reproductions"].sum() / df["reproduction_events"].sum()
                logging.info(f"Average reproduction success rate: {avg_success_rate:.4f}")
        else:
            logging.warning("No reproduction events found in the data")
    
    # Check data consistency
    for col in df.columns:
        # Check for completely null columns
        if df[col].isnull().all():
            logging.warning(f"Column {col} has all null values")
        # Check for columns with all zeros when we expect non-zero values
        elif col not in ["iteration", "final_step"] and df[col].dtype != object and (df[col] == 0).all():
            logging.warning(f"Column {col} has all zero values")
    
    return True


def analyze_cooperative_action_details(df, experiment_path, output_path):
    """
    Analyze the details of cooperative actions across simulations
    by examining action types and details.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing cooperation metrics
    experiment_path : str
        Path to experiment directory containing simulations
    output_path : str
        Path to save analysis outputs
    """
    if df.empty:
        logging.warning("No data for cooperative action details analysis")
        return
    
    # Select a sample of simulation databases to analyze
    sim_iterations = df["iteration"].sample(min(5, len(df))).tolist()
    
    action_types = {}
    action_details = {}
    
    for iteration in sim_iterations:
        db_path = os.path.join(experiment_path, f"iteration_{iteration}", "simulation.db")
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
                count = session.query(ActionModel).filter(
                    ActionModel.action_type == action_type
                ).count()
                
                action_types[action_type] += count
                
                # For potential cooperative actions, sample some details
                if any(term in action_type.lower() for term in ["share", "give", "help", "cooperate", "transfer"]):
                    # Sample some actions with details
                    sample_actions = session.query(ActionModel).filter(
                        ActionModel.action_type == action_type,
                        ActionModel.details.isnot(None)
                    ).limit(10).all()
                    
                    if sample_actions:
                        if action_type not in action_details:
                            action_details[action_type] = []
                        
                        for action in sample_actions:
                            if action.details and action.details not in action_details[action_type]:
                                action_details[action_type].append(action.details)
            
            session.close()
            
        except Exception as e:
            logging.error(f"Error analyzing cooperative action details for iteration {iteration}: {e}")
    
    # Log the findings
    logging.info("\nCooperative Action Analysis:")
    logging.info("---------------------------")
    
    if action_types:
        logging.info("Action types found in the database:")
        for action_type, count in sorted(action_types.items(), key=lambda x: x[1], reverse=True):
            logging.info(f"  {action_type}: {count} occurrences")
        
        # Identify potentially cooperative actions
        coop_actions = [(action, count) for action, count in action_types.items() 
                       if any(term in action.lower() for term in ["share", "give", "help", "cooperate", "transfer"])]
        
        if coop_actions:
            logging.info("\nPotentially cooperative actions:")
            for action, count in coop_actions:
                logging.info(f"  {action}: {count} occurrences")
                
                # Log sample details if available
                if action in action_details and action_details[action]:
                    logging.info(f"    Sample details:")
                    for i, detail in enumerate(action_details[action][:3]):
                        logging.info(f"      {i+1}. {detail}")
        else:
            logging.info("No explicitly cooperative actions identified in the database.")
            logging.info("You may need to modify the script to look for other action types specific to your simulation.")
    else:
        logging.info("No action types found in the database.")


def main():
    # Set up the cooperation analysis directory
    cooperation_output_path, log_file = setup_analysis_directory("cooperation")
    
    # Find the most recent experiment folder
    experiment_path = find_latest_experiment_path()
    if not experiment_path:
        return
    
    # Check if reproduction events exist in the databases
    has_reproduction_events = check_reproduction_events(experiment_path)
    if not has_reproduction_events:
        logging.warning(
            "No reproduction events found in databases. Reproduction analysis may be limited."
        )
    
    logging.info(f"Analyzing cooperation in simulations in {experiment_path}...")
    df = analyze_cooperation_metrics(experiment_path)
    
    if df.empty:
        logging.warning("No simulation data found.")
        return
    
    # Validate the data quality
    data_valid = validate_data(df)
    if not data_valid:
        logging.warning("Data validation issues detected. Analysis results may be limited.")
    
    # Analyze cooperative action details
    analyze_cooperative_action_details(df, experiment_path, cooperation_output_path)
    
    # Save the raw data
    output_csv = os.path.join(cooperation_output_path, "cooperation_analysis.csv")
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved analysis data to {output_csv}")
    
    # Display summary statistics
    logging.info("\nSummary statistics:")
    logging.info(df.describe().to_string())
    
    # Display average cooperation metrics
    logging.info("\nAverage cooperation metrics across simulations:")
    for metric in ["share_actions_per_agent", "reproduction_success_rate", "cooperation_index"]:
        if metric in df.columns:
            avg_value = df[metric].mean()
            logging.info(f"  Average {metric}: {avg_value:.4f}")
    
    # Generate and save plots
    plot_cooperation_distribution(df, cooperation_output_path)
    plot_cooperation_time_series(experiment_path, cooperation_output_path)
    analyze_cooperation_factors(df, cooperation_output_path)
    plot_agent_type_comparison(df, cooperation_output_path)
    
    logging.info("\nAnalysis complete. Results saved to CSV and PNG files.")
    logging.info(f"Log file saved to: {log_file}")
    logging.info(f"All analysis files saved to: {cooperation_output_path}")


if __name__ == "__main__":
    main() 
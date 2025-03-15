import glob
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import analysis configuration
from analysis_config import (
    DATA_PATH,
    OUTPUT_PATH,
    check_reproduction_events,
    safe_remove_directory,
    setup_logging,
    setup_analysis_directory,
    find_latest_experiment_path,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from farm.database.models import AgentModel, ReproductionEventModel, SimulationStepModel


def analyze_reproduction_rate_changes(sim_session, window_size=20):
    """
    Analyze changes in reproduction rates over time for each agent type.

    Parameters
    ----------
    sim_session : SQLAlchemy session
        Database session
    window_size : int
        Size of the sliding window for calculating reproduction rates

    Returns
    -------
    dict
        Dictionary with reproduction rate changes data
    """
    # Get all reproduction events ordered by step
    reproduction_events = (
        sim_session.query(ReproductionEventModel)
        .order_by(ReproductionEventModel.step_number)
        .all()
    )

    # Get agent types
    agent_types = {}
    for agent in sim_session.query(AgentModel).all():
        agent_types[agent.agent_id] = agent.agent_type

    # Initialize data structures
    steps = []
    events_by_step = {}

    # Group events by step
    for event in reproduction_events:
        step = event.step_number
        if step not in events_by_step:
            events_by_step[step] = []
            steps.append(step)
        events_by_step[step].append(event)

    # Sort steps
    steps.sort()

    # Calculate reproduction rates in sliding windows
    reproduction_rates = {"SystemAgent": [], "IndependentAgent": [], "ControlAgent": []}

    # Count agents by type at each step
    agent_counts = {}
    for step in steps:
        # Get simulation step data
        sim_step = (
            sim_session.query(SimulationStepModel).filter_by(step_number=step).first()
        )
        if sim_step:
            agent_counts[step] = {
                "SystemAgent": sim_step.system_agents,
                "IndependentAgent": sim_step.independent_agents,
                "ControlAgent": sim_step.control_agents,
            }

    # Calculate reproduction rates in sliding windows
    for i in range(len(steps) - window_size + 1):
        window_steps = steps[i : i + window_size]
        window_center = window_steps[window_size // 2]

        # Count successful reproductions by agent type in this window
        reproductions = {"SystemAgent": 0, "IndependentAgent": 0, "ControlAgent": 0}

        for step in window_steps:
            for event in events_by_step.get(step, []):
                if event.success and event.parent_id in agent_types:
                    agent_type = agent_types[event.parent_id]
                    # Handle case where agent_type might not be one of our expected types
                    if agent_type in reproductions:
                        reproductions[agent_type] += 1

        # Calculate average agent counts in this window
        avg_counts = {"SystemAgent": 0, "IndependentAgent": 0, "ControlAgent": 0}

        count_steps = 0
        for step in window_steps:
            if step in agent_counts:
                for agent_type in avg_counts:
                    avg_counts[agent_type] += agent_counts[step][agent_type]
                count_steps += 1

        # Avoid division by zero
        if count_steps > 0:
            for agent_type in avg_counts:
                avg_counts[agent_type] /= count_steps

        # Calculate reproduction rate (reproductions per agent)
        for agent_type in reproduction_rates:
            if avg_counts[agent_type] > 0:
                rate = reproductions[agent_type] / avg_counts[agent_type]
            else:
                rate = 0

            reproduction_rates[agent_type].append((window_center, rate))

    # Calculate rate of change (derivative) of reproduction rates
    rate_changes = {"SystemAgent": [], "IndependentAgent": [], "ControlAgent": []}

    for agent_type in reproduction_rates:
        rates = reproduction_rates[agent_type]
        for i in range(1, len(rates)):
            prev_step, prev_rate = rates[i - 1]
            curr_step, curr_rate = rates[i]

            # Calculate rate of change
            if curr_step > prev_step:
                change = (curr_rate - prev_rate) / (curr_step - prev_step)
                rate_changes[agent_type].append((curr_step, change))

    return {"reproduction_rates": reproduction_rates, "rate_changes": rate_changes}


def identify_reproduction_strategy_changes(sim_session, threshold=0.1):
    """
    Identify significant changes in reproduction strategies.

    Parameters
    ----------
    sim_session : SQLAlchemy session
        Database session
    threshold : float
        Threshold for significant change in reproduction rate

    Returns
    -------
    dict
        Dictionary with significant reproduction strategy changes
    """
    # Get reproduction rate changes
    repro_data = analyze_reproduction_rate_changes(sim_session)
    rate_changes = repro_data["rate_changes"]

    # Identify significant changes
    significant_changes = {
        "SystemAgent": [],
        "IndependentAgent": [],
        "ControlAgent": [],
    }

    for agent_type in rate_changes:
        changes = rate_changes[agent_type]
        for step, change in changes:
            if abs(change) > threshold:
                significant_changes[agent_type].append(
                    {
                        "step": step,
                        "change": change,
                        "direction": "increase" if change > 0 else "decrease",
                    }
                )

    return significant_changes


def analyze_reproduction_resource_allocation(sim_session):
    """
    Analyze changes in how agents allocate resources to reproduction.

    Parameters
    ----------
    sim_session : SQLAlchemy session
        Database session

    Returns
    -------
    dict
        Dictionary with resource allocation data
    """
    # Get all successful reproduction events
    reproduction_events = (
        sim_session.query(ReproductionEventModel)
        .filter(ReproductionEventModel.success == True)
        .order_by(ReproductionEventModel.step_number)
        .all()
    )

    # Get agent types
    agent_types = {}
    for agent in sim_session.query(AgentModel).all():
        agent_types[agent.agent_id] = agent.agent_type

    # Initialize data structures
    resource_allocation = {
        "SystemAgent": [],
        "IndependentAgent": [],
        "ControlAgent": [],
    }

    # Calculate resource allocation for each reproduction event
    for event in reproduction_events:
        if event.parent_id in agent_types:
            agent_type = agent_types[event.parent_id]

            # Only process if agent_type is one of our expected types
            if agent_type in resource_allocation:
                # Calculate percentage of resources given to offspring
                if (
                    event.parent_resources_before > 0
                    and event.offspring_initial_resources is not None
                ):
                    allocation_percentage = (
                        event.offspring_initial_resources
                        / event.parent_resources_before
                    ) * 100
                    resource_allocation[agent_type].append(
                        {
                            "step": event.step_number,
                            "allocation_percentage": allocation_percentage,
                            "parent_resources_before": event.parent_resources_before,
                            "parent_resources_after": event.parent_resources_after,
                            "offspring_resources": event.offspring_initial_resources,
                        }
                    )

    # Calculate moving average of resource allocation
    window_size = 10
    moving_avg = {"SystemAgent": [], "IndependentAgent": [], "ControlAgent": []}

    for agent_type in resource_allocation:
        allocations = resource_allocation[agent_type]
        if len(allocations) >= window_size:
            for i in range(len(allocations) - window_size + 1):
                window = allocations[i : i + window_size]
                avg_allocation = (
                    sum(item["allocation_percentage"] for item in window) / window_size
                )
                center_step = window[window_size // 2]["step"]
                moving_avg[agent_type].append(
                    {"step": center_step, "avg_allocation": avg_allocation}
                )

    return {
        "resource_allocation": resource_allocation,
        "moving_avg_allocation": moving_avg,
    }


def analyze_reproduction_timing_adaptations(sim_session):
    """
    Analyze adaptations in reproduction timing.

    Parameters
    ----------
    sim_session : SQLAlchemy session
        Database session

    Returns
    -------
    dict
        Dictionary with reproduction timing adaptation data
    """
    # Get all reproduction events
    reproduction_events = (
        sim_session.query(ReproductionEventModel)
        .order_by(ReproductionEventModel.step_number)
        .all()
    )

    # Get agent types
    agent_types = {}
    for agent in sim_session.query(AgentModel).all():
        agent_types[agent.agent_id] = agent.agent_type

    # Initialize data structures
    reproduction_intervals = {
        "SystemAgent": {},
        "IndependentAgent": {},
        "ControlAgent": {},
    }

    # Calculate intervals between reproduction attempts for each agent
    for event in reproduction_events:
        if event.parent_id in agent_types:
            agent_type = agent_types[event.parent_id]

            # Only process if agent_type is one of our expected types
            if agent_type in reproduction_intervals:
                parent_id = event.parent_id

                if parent_id not in reproduction_intervals[agent_type]:
                    reproduction_intervals[agent_type][parent_id] = []

                reproduction_intervals[agent_type][parent_id].append(event.step_number)

    # Calculate average intervals and changes over time
    interval_changes = {"SystemAgent": [], "IndependentAgent": [], "ControlAgent": []}

    for agent_type in reproduction_intervals:
        for parent_id, steps in reproduction_intervals[agent_type].items():
            if (
                len(steps) >= 3
            ):  # Need at least 3 reproduction attempts to calculate changes
                steps.sort()

                # Calculate intervals
                intervals = [steps[i] - steps[i - 1] for i in range(1, len(steps))]

                # Calculate changes in intervals
                for i in range(1, len(intervals)):
                    change = intervals[i] - intervals[i - 1]
                    interval_changes[agent_type].append(
                        {
                            "step": steps[i + 1],
                            "interval_change": change,
                            "relative_change": (
                                change / intervals[i - 1] if intervals[i - 1] > 0 else 0
                            ),
                        }
                    )

    return {
        "reproduction_intervals": reproduction_intervals,
        "interval_changes": interval_changes,
    }


def calculate_reproduction_strategy_volatility(sim_session):
    """
    Calculate the volatility of reproduction strategies for each agent type.

    Parameters
    ----------
    sim_session : SQLAlchemy session
        Database session

    Returns
    -------
    dict
        Dictionary with reproduction strategy volatility data
    """
    # Get reproduction rate changes
    repro_data = analyze_reproduction_rate_changes(sim_session)
    rate_changes = repro_data["rate_changes"]

    # Calculate volatility (standard deviation of rate changes)
    volatility = {}

    for agent_type in rate_changes:
        changes = [abs(change) for _, change in rate_changes[agent_type]]

        if changes:
            mean_change = sum(changes) / len(changes)
            variance = sum((change - mean_change) ** 2 for change in changes) / len(
                changes
            )
            std_dev = variance**0.5

            volatility[agent_type] = {
                "mean_absolute_change": mean_change,
                "standard_deviation": std_dev,
                "coefficient_of_variation": (
                    std_dev / mean_change if mean_change > 0 else 0
                ),
                "sample_size": len(changes),
            }
        else:
            volatility[agent_type] = {
                "mean_absolute_change": 0,
                "standard_deviation": 0,
                "coefficient_of_variation": 0,
                "sample_size": 0,
            }

    return volatility


def analyze_reproduction_strategies(sim_path):
    """
    Analyze reproduction strategies for a single simulation.

    Parameters
    ----------
    sim_path : str
        Path to the simulation database

    Returns
    -------
    dict
        Dictionary with reproduction strategy analysis
    """
    try:
        # Connect to the database
        engine = create_engine(f"sqlite:///{sim_path}")
        Session = sessionmaker(bind=engine)
        session = Session()

        # Get simulation metadata
        sim_id = os.path.basename(os.path.dirname(sim_path))

        # Get final step number
        final_step = (
            session.query(SimulationStepModel)
            .order_by(SimulationStepModel.step_number.desc())
            .first()
        )

        if not final_step:
            logging.warning(f"No simulation steps found in {sim_path}")
            session.close()
            return None

        final_step_number = final_step.step_number

        # Analyze reproduction rates
        repro_rates = analyze_reproduction_rate_changes(session)

        # Identify significant strategy changes
        strategy_changes = identify_reproduction_strategy_changes(session)

        # Analyze resource allocation
        resource_allocation = analyze_reproduction_resource_allocation(session)

        # Analyze timing adaptations
        timing_adaptations = analyze_reproduction_timing_adaptations(session)

        # Calculate strategy volatility
        volatility = calculate_reproduction_strategy_volatility(session)

        # Count total reproduction events and success rates
        total_events = session.query(ReproductionEventModel).count()
        successful_events = (
            session.query(ReproductionEventModel)
            .filter(ReproductionEventModel.success == True)
            .count()
        )

        success_rate = (
            (successful_events / total_events * 100) if total_events > 0 else 0
        )

        # Count reproduction events by agent type
        events_by_type = {"SystemAgent": 0, "IndependentAgent": 0, "ControlAgent": 0}
        success_by_type = {"SystemAgent": 0, "IndependentAgent": 0, "ControlAgent": 0}

        # Get agent types
        agent_types = {}
        for agent in session.query(AgentModel).all():
            agent_types[agent.agent_id] = agent.agent_type

        # Count events by type
        for event in session.query(ReproductionEventModel).all():
            if event.parent_id in agent_types:
                agent_type = agent_types[event.parent_id]
                if agent_type in events_by_type:
                    events_by_type[agent_type] += 1
                    if event.success:
                        success_by_type[agent_type] += 1

        # Calculate success rates by type
        success_rates_by_type = {}
        for agent_type in events_by_type:
            if events_by_type[agent_type] > 0:
                success_rates_by_type[agent_type] = (
                    success_by_type[agent_type] / events_by_type[agent_type] * 100
                )
            else:
                success_rates_by_type[agent_type] = 0

        # Close the session
        session.close()

        # Compile results
        results = {
            "simulation_id": sim_id,
            "final_step": final_step_number,
            "total_reproduction_events": total_events,
            "successful_reproduction_events": successful_events,
            "overall_success_rate": success_rate,
            "events_by_agent_type": events_by_type,
            "success_by_agent_type": success_by_type,
            "success_rates_by_agent_type": success_rates_by_type,
            "reproduction_rates": repro_rates,
            "strategy_changes": strategy_changes,
            "resource_allocation": resource_allocation,
            "timing_adaptations": timing_adaptations,
            "strategy_volatility": volatility,
        }

        return results

    except Exception as e:
        logging.error(f"Error analyzing reproduction strategies in {sim_path}: {e}")
        import traceback

        logging.error(traceback.format_exc())
        return None


def analyze_simulations(experiment_path):
    """
    Analyze reproduction strategies across all simulations in an experiment.

    Parameters
    ----------
    experiment_path : str
        Path to the experiment folder

    Returns
    -------
    pandas.DataFrame
        DataFrame with reproduction strategy analysis for all simulations
    """
    # Find all simulation folders
    sim_folders = glob.glob(os.path.join(experiment_path, "iteration_*"))
    logging.info(f"Found {len(sim_folders)} simulation folders")

    # Initialize list to store results
    results = []

    # Analyze each simulation
    for folder in sim_folders:
        db_path = os.path.join(folder, "simulation.db")

        if not os.path.exists(db_path):
            logging.warning(f"No database found in {folder}")
            continue

        logging.info(
            f"Analyzing reproduction strategies in {os.path.basename(folder)}..."
        )

        # Analyze reproduction strategies
        sim_results = analyze_reproduction_strategies(db_path)

        if sim_results:
            # Extract key metrics for the DataFrame
            row = {
                "iteration": os.path.basename(folder),
                "final_step": sim_results["final_step"],
                "total_reproduction_events": sim_results["total_reproduction_events"],
                "successful_reproduction_events": sim_results[
                    "successful_reproduction_events"
                ],
                "overall_success_rate": sim_results["overall_success_rate"],
                "SystemAgent_reproduction_events": sim_results["events_by_agent_type"][
                    "SystemAgent"
                ],
                "IndependentAgent_reproduction_events": sim_results[
                    "events_by_agent_type"
                ]["IndependentAgent"],
                "ControlAgent_reproduction_events": sim_results["events_by_agent_type"][
                    "ControlAgent"
                ],
                "SystemAgent_success_rate": sim_results["success_rates_by_agent_type"][
                    "SystemAgent"
                ],
                "IndependentAgent_success_rate": sim_results[
                    "success_rates_by_agent_type"
                ]["IndependentAgent"],
                "ControlAgent_success_rate": sim_results["success_rates_by_agent_type"][
                    "ControlAgent"
                ],
            }

            # Add volatility metrics
            for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
                volatility = sim_results["strategy_volatility"].get(agent_type, {})
                row[f"{agent_type}_repro_volatility"] = volatility.get(
                    "coefficient_of_variation", 0
                )
                row[f"{agent_type}_repro_mean_change"] = volatility.get(
                    "mean_absolute_change", 0
                )

            # Add strategy change counts
            for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
                changes = sim_results["strategy_changes"].get(agent_type, [])
                row[f"{agent_type}_strategy_changes"] = len(changes)

                # Count increases and decreases
                increases = sum(
                    1 for change in changes if change["direction"] == "increase"
                )
                decreases = sum(
                    1 for change in changes if change["direction"] == "decrease"
                )

                row[f"{agent_type}_strategy_increases"] = increases
                row[f"{agent_type}_strategy_decreases"] = decreases

            results.append(row)

    # Create DataFrame
    if results:
        df = pd.DataFrame(results)
        return df
    else:
        logging.warning("No valid simulation results found")
        return pd.DataFrame()


def plot_reproduction_rates(df, output_path):
    """
    Plot reproduction rates and success rates by agent type.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with reproduction analysis
    output_path : str
        Path to save the plots
    """
    plt.figure(figsize=(12, 8))

    # Plot reproduction events by agent type
    plt.subplot(2, 2, 1)
    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
    events = [
        df[f"{agent_type}_reproduction_events"].mean() for agent_type in agent_types
    ]
    plt.bar(agent_types, events, color=["blue", "green", "red"])
    plt.title("Average Reproduction Events by Agent Type")
    plt.ylabel("Number of Events")
    plt.xticks(rotation=45)

    # Plot success rates by agent type
    plt.subplot(2, 2, 2)
    success_rates = [
        df[f"{agent_type}_success_rate"].mean() for agent_type in agent_types
    ]
    plt.bar(agent_types, success_rates, color=["blue", "green", "red"])
    plt.title("Average Reproduction Success Rate by Agent Type")
    plt.ylabel("Success Rate (%)")
    plt.xticks(rotation=45)

    # Plot reproduction volatility by agent type
    plt.subplot(2, 2, 3)
    volatility = [
        df[f"{agent_type}_repro_volatility"].mean() for agent_type in agent_types
    ]
    plt.bar(agent_types, volatility, color=["blue", "green", "red"])
    plt.title("Reproduction Strategy Volatility by Agent Type")
    plt.ylabel("Coefficient of Variation")
    plt.xticks(rotation=45)

    # Plot strategy changes by agent type
    plt.subplot(2, 2, 4)
    changes = [
        df[f"{agent_type}_strategy_changes"].mean() for agent_type in agent_types
    ]
    plt.bar(agent_types, changes, color=["blue", "green", "red"])
    plt.title("Average Strategy Changes by Agent Type")
    plt.ylabel("Number of Changes")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "reproduction_rates.png"))
    plt.close()


def plot_strategy_changes(df, output_path):
    """
    Plot reproduction strategy changes by agent type.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with reproduction analysis
    output_path : str
        Path to save the plots
    """
    plt.figure(figsize=(12, 6))

    # Plot strategy increases and decreases by agent type
    agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
    increases = [
        df[f"{agent_type}_strategy_increases"].mean() for agent_type in agent_types
    ]
    decreases = [
        df[f"{agent_type}_strategy_decreases"].mean() for agent_type in agent_types
    ]

    x = np.arange(len(agent_types))
    width = 0.35

    plt.bar(x - width / 2, increases, width, label="Increases", color="green")
    plt.bar(x + width / 2, decreases, width, label="Decreases", color="red")

    plt.xlabel("Agent Type")
    plt.ylabel("Average Number of Changes")
    plt.title("Reproduction Strategy Changes by Direction")
    plt.xticks(x, agent_types, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "strategy_changes.png"))
    plt.close()


def plot_reproduction_correlation(df, output_path):
    """
    Plot correlation between reproduction metrics and other simulation metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with reproduction analysis
    output_path : str
        Path to save the plots
    """
    # Select reproduction-related columns
    repro_cols = [
        col
        for col in df.columns
        if "reproduction" in col or "success_rate" in col or "strategy" in col
    ]

    # Select other numeric columns
    other_cols = [
        col
        for col in df.columns
        if col not in repro_cols and df[col].dtype in [np.int64, np.float64]
    ]

    if not repro_cols or not other_cols:
        logging.warning("Not enough columns for correlation analysis")
        return

    # Calculate correlation matrix
    correlation_matrix = df[repro_cols + other_cols].corr()

    # Plot heatmap of correlations
    plt.figure(figsize=(14, 12))
    plt.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation Coefficient")
    plt.xticks(
        range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90
    )
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title("Correlation Between Reproduction Metrics and Other Simulation Metrics")

    # Add correlation values
    for i in range(len(correlation_matrix.columns)):
        for j in range(len(correlation_matrix.columns)):
            plt.text(
                j,
                i,
                f"{correlation_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(correlation_matrix.iloc[i, j]) > 0.5 else "black",
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "reproduction_correlation.png"))
    plt.close()


def main():
    # Set up the reproduction analysis directory
    reproduction_output_path, log_file = setup_analysis_directory("reproduction")

    # Find the most recent experiment folder
    experiment_path = find_latest_experiment_path()
    if not experiment_path:
        return

    # Check if reproduction events exist in the databases
    has_reproduction_events = check_reproduction_events(experiment_path)
    if not has_reproduction_events:
        logging.error(
            "No reproduction events found in databases. Cannot perform reproduction analysis."
        )
        return

    logging.info(f"Analyzing reproduction strategies in {experiment_path}...")
    df = analyze_simulations(experiment_path)

    if df.empty:
        logging.warning("No simulation data found.")
        return

    # Save the raw data
    output_csv = os.path.join(reproduction_output_path, "reproduction_analysis.csv")
    df.to_csv(output_csv, index=False)
    logging.info(f"Saved analysis data to {output_csv}")

    # Log summary statistics
    logging.info(f"Analyzed {len(df)} simulations.")
    logging.info("\nSummary statistics:")
    logging.info(df.describe().to_string())

    # Log reproduction event counts
    logging.info("\nAverage reproduction events by agent type:")
    for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
        avg_events = df[f"{agent_type}_reproduction_events"].mean()
        logging.info(f"  {agent_type}: {avg_events:.2f}")

    # Log success rates
    logging.info("\nAverage reproduction success rates by agent type:")
    for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
        avg_rate = df[f"{agent_type}_success_rate"].mean()
        logging.info(f"  {agent_type}: {avg_rate:.2f}%")

    # Log strategy changes
    logging.info("\nAverage strategy changes by agent type:")
    for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
        avg_changes = df[f"{agent_type}_strategy_changes"].mean()
        logging.info(f"  {agent_type}: {avg_changes:.2f}")

    # Log strategy volatility
    logging.info("\nAverage reproduction strategy volatility by agent type:")
    for agent_type in ["SystemAgent", "IndependentAgent", "ControlAgent"]:
        avg_volatility = df[f"{agent_type}_repro_volatility"].mean()
        logging.info(f"  {agent_type}: {avg_volatility:.4f}")

    # Generate plots
    logging.info("\nGenerating plots...")
    plot_reproduction_rates(df, reproduction_output_path)
    plot_strategy_changes(df, reproduction_output_path)
    plot_reproduction_correlation(df, reproduction_output_path)

    logging.info("\nAnalysis complete. Results saved to CSV and PNG files.")
    logging.info(f"Log file saved to: {log_file}")
    logging.info(f"All analysis files saved to: {reproduction_output_path}")


if __name__ == "__main__":
    main()

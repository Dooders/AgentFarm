import os
import glob
import pandas as pd
from typing import List, Dict, Any
from sqlalchemy import create_engine, func
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from farm.database.models import Simulation, SimulationStepModel
from farm.database.database import SimulationDatabase


def find_simulation_databases(base_path: str) -> List[str]:
    """
    Find all SQLite database files in the specified directory.

    Parameters
    ----------
    base_path : str
        Base directory to search for simulation databases

    Returns
    -------
    List[str]
        List of paths to found database files
    """
    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    # Find all .db files in the directory
    db_pattern = os.path.join(base_path, "*.db")
    db_files = glob.glob(db_pattern)

    if not db_files:
        print(f"Warning: No database files found in {base_path}")

    return sorted(db_files)


def gather_simulation_metadata(db_path: str) -> Dict[str, Any]:
    """
    Gather high-level metadata for a single simulation, including duration,
    configuration, and a summary of final-step metrics.
    """
    # Get a readable name from the database path
    sim_name = os.path.splitext(os.path.basename(db_path))[0]

    db = SimulationDatabase(db_path)
    session = db.Session()

    try:
        # 1) Fetch simulation record
        sim = (
            session.query(Simulation).order_by(Simulation.simulation_id.desc()).first()
        )

        # 2) Calculate simulation duration in steps
        final_step_row = session.query(func.max(SimulationStepModel.step_number)).one()
        final_step = final_step_row[0] if final_step_row[0] is not None else 0

        # 3) Get step-wise population data
        population_data = (
            session.query(SimulationStepModel.total_agents)
            .order_by(SimulationStepModel.step_number)
            .all()
        )
        population_series = pd.Series([p[0] for p in population_data])

        # Calculate population statistics
        median_population = population_series.median()
        mode_population = population_series.mode().iloc[0]
        mean_population = population_series.mean()

        # 4) Calculate across-time averages
        avg_metrics = session.query(
            func.avg(SimulationStepModel.total_agents).label("avg_total_agents"),
            func.avg(SimulationStepModel.births).label("avg_births"),
            func.avg(SimulationStepModel.deaths).label("avg_deaths"),
            func.avg(SimulationStepModel.total_resources).label("avg_total_resources"),
            func.avg(SimulationStepModel.average_agent_health).label("avg_health"),
            func.avg(SimulationStepModel.average_reward).label("avg_reward"),
        ).first()

        # 5) Gather second-to-last step metrics
        final_step_row = session.query(func.max(SimulationStepModel.step_number)).one()
        final_step = final_step_row[0] if final_step_row[0] is not None else 0
        second_to_last_step = final_step - 1 if final_step > 0 else 0

        final_metrics = (
            session.query(SimulationStepModel)
            .filter(SimulationStepModel.step_number == second_to_last_step)
            .first()
        )

        if final_metrics:
            final_snapshot = {
                "final_total_agents": final_metrics.total_agents,
                "final_total_resources": final_metrics.total_resources,
                "final_average_health": final_metrics.average_agent_health,
                "final_average_reward": final_metrics.average_reward,
                "final_births": final_metrics.births,
                "final_deaths": final_metrics.deaths,
            }
        else:
            final_snapshot = {
                "final_total_agents": None,
                "final_total_resources": None,
                "final_average_health": None,
                "final_average_reward": None,
                "final_births": None,
                "final_deaths": None,
            }

        # 6) Extract config parameters
        config = sim.parameters if sim and sim.parameters else {}

        # 7) Build complete metadata dictionary
        return {
            "simulation_name": sim_name,
            "db_path": db_path,
            "simulation_id": sim.simulation_id if sim else None,
            "start_time": str(sim.start_time) if sim else None,
            "end_time": str(sim.end_time) if sim and sim.end_time else None,
            "status": sim.status if sim else None,
            "duration_steps": final_step,
            "config": config,
            # Population statistics
            "median_population": float(median_population),
            "mode_population": float(mode_population),
            "mean_population": float(mean_population),
            # Across-time averages
            "avg_total_agents": float(avg_metrics[0]) if avg_metrics[0] else 0,
            "avg_births": float(avg_metrics[1]) if avg_metrics[1] else 0,
            "avg_deaths": float(avg_metrics[2]) if avg_metrics[2] else 0,
            "avg_total_resources": float(avg_metrics[3]) if avg_metrics[3] else 0,
            "avg_health": float(avg_metrics[4]) if avg_metrics[4] else 0,
            "avg_reward": float(avg_metrics[5]) if avg_metrics[5] else 0,
            **final_snapshot,
        }
    finally:
        session.close()
        db.close()


def plot_comparative_metrics(
    df: pd.DataFrame, output_dir: str = "simulations/analysis"
):
    """
    Generate comparative visualizations of simulation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing simulation metrics
    output_dir : str
        Directory to save generated plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Population Dynamics
    plt.figure(figsize=(12, 6))
    x = range(len(df))

    plt.plot(x, df["median_population"], "b--", label="Median Population")
    plt.plot(x, df["mode_population"], "g--", label="Mode Population")
    plt.plot(x, df["final_total_agents"], "r-", label="Final Population")
    plt.title("Population Comparison Across Simulations")
    plt.xlabel("Simulation Index")
    plt.ylabel("Number of Agents")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "population_comparison.png"))
    plt.close()

    # 2. Resource Utilization
    plt.figure(figsize=(12, 6))
    plt.plot(x, df["avg_total_resources"], "g-", label="Average Resources")
    plt.plot(x, df["final_total_resources"], "m--", label="Final Resources")
    plt.title("Resource Utilization Across Simulations")
    plt.xlabel("Simulation Index")
    plt.ylabel("Resource Level")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "resource_comparison.png"))
    plt.close()

    # 3. Health and Reward Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(data=df[["avg_health", "final_average_health"]], ax=ax1)
    ax1.set_title("Health Distribution")
    ax1.set_ylabel("Health Level")

    sns.boxplot(data=df[["avg_reward", "final_average_reward"]], ax=ax2)
    ax2.set_title("Reward Distribution")
    ax2.set_ylabel("Reward Level")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "health_reward_distribution.png"))
    plt.close()


def compare_simulations(db_paths: List[str]) -> pd.DataFrame:
    """
    Compare N simulations by collecting key metadata and stats into a pandas DataFrame.

    Parameters
    ----------
    db_paths : List[str]
        List of paths to simulation database files

    Returns
    -------
    pd.DataFrame
        DataFrame containing comparative metrics
    """
    records = []
    for path in db_paths:
        meta = gather_simulation_metadata(path)
        records.append(meta)

    df = pd.DataFrame(records)

    # Generate summary statistics
    print("\nComparative Simulation Summary:")
    print("\nBasic Statistics:")
    print(df[["duration_steps", "avg_total_agents", "avg_total_resources"]].describe())

    print("\nConfiguration Differences:")
    if "config" in df.columns:
        configs = pd.json_normalize(df["config"].apply(lambda x: x or {}))
        if not configs.empty:
            print(configs.describe())

    # Generate plots
    plot_comparative_metrics(df)

    return df


def create_dashboard_summary(
    df: pd.DataFrame, output_dir: str = "simulations/analysis"
):
    """
    Create a dashboard-style summary image highlighting key metrics across simulations.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing simulation metrics
    output_dir : str
        Directory to save the dashboard image
    """
    # Use a built-in style instead of 'seaborn'
    plt.style.use("fivethirtyeight")  # Alternative: 'ggplot'

    # Create figure with subplots in a 3x2 grid
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2, figure=fig)

    # 1. Population Metrics (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = {
        "Avg Population": df["avg_total_agents"].mean(),
        "Peak Population": df["final_total_agents"].max(),
        "Avg Birth Rate": df["avg_births"].mean(),
        "Avg Death Rate": df["avg_deaths"].mean(),
    }

    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f1c40f"]
    ax1.bar(range(len(metrics)), metrics.values(), color=colors)
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metrics.keys(), rotation=45)
    ax1.set_title("Population Metrics", fontsize=12, pad=20)

    # Add value labels on bars
    for i, v in enumerate(metrics.values()):
        ax1.text(i, v, f"{v:.1f}", ha="center", va="bottom")

    # 2. Resource Metrics (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    resource_data = {
        "Avg Resources": df["avg_total_resources"].mean(),
        "Final Resources": df["final_total_resources"].mean(),
    }

    ax2.pie(
        resource_data.values(),
        labels=resource_data.keys(),
        autopct="%1.1f%%",
        colors=["#2ecc71", "#e74c3c"],
    )
    ax2.set_title("Resource Distribution", fontsize=12, pad=20)

    # 3. Health & Reward Trends (Middle)
    ax3 = fig.add_subplot(gs[1, :])

    # Use simulation names for x-axis if available
    if "simulation_name" in df.columns:
        x = df["simulation_name"]
    else:
        x = range(len(df))

    ax3.plot(
        x, df["final_average_health"], "r-", label="Pre-Final Agent Health", linewidth=2
    )
    ax3.plot(x, df["avg_health"], "g-", label="Average Agent Health", linewidth=2)

    ax3.set_title(
        "Health Metrics Across Simulations (Pre-Final Step)", fontsize=12, pad=20
    )
    ax3.set_xlabel("Simulation Name")
    ax3.set_ylabel("Health")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Rotate x-axis labels if they're text
    if "simulation_name" in df.columns:
        plt.xticks(rotation=45, ha="right")

    # 4. Key Statistics Table (Bottom)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("tight")
    ax4.axis("off")

    stats_data = [
        ["Metric", "Mean", "Min", "Max", "Std Dev"],
        [
            "Duration (steps)",
            f"{df['duration_steps'].mean():.1f}",
            f"{df['duration_steps'].min():.1f}",
            f"{df['duration_steps'].max():.1f}",
            f"{df['duration_steps'].std():.1f}",
        ],
        [
            "Total Agents",
            f"{df['avg_total_agents'].mean():.1f}",
            f"{df['avg_total_agents'].min():.1f}",
            f"{df['avg_total_agents'].max():.1f}",
            f"{df['avg_total_agents'].std():.1f}",
        ],
        [
            "Final Health",
            f"{df['final_average_health'].mean():.2f}",
            f"{df['final_average_health'].min():.2f}",
            f"{df['final_average_health'].max():.2f}",
            f"{df['final_average_health'].std():.2f}",
        ],
        [
            "Average Health",
            f"{df['avg_health'].mean():.2f}",
            f"{df['avg_health'].min():.2f}",
            f"{df['avg_health'].max():.2f}",
            f"{df['avg_health'].std():.2f}",
        ],
    ]

    table = ax4.table(cellText=stats_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)

    # Style header row
    for i in range(len(stats_data[0])):
        table[(0, i)].set_facecolor("#3498db")
        table[(0, i)].set_text_props(color="white")

    plt.suptitle("Simulation Analysis Dashboard", fontsize=16, y=0.95)
    plt.tight_layout()

    # Save the dashboard
    plt.savefig(
        os.path.join(output_dir, "simulation_dashboard.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main(path: str):
    # Find all simulation databases
    db_paths = find_simulation_databases(path)

    if not db_paths:
        print(
            "No simulation databases found. Please ensure databases exist in 'simulations/databases/'"
        )
        return

    # Compare and analyze simulations
    results_df = compare_simulations(db_paths)

    # Save results to CSV
    output_dir = "simulations/analysis"
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(
        os.path.join(output_dir, "simulation_comparison.csv"), index=False
    )
    print(f"\nResults saved to {output_dir}")

    # Create dashboard summary
    create_dashboard_summary(results_df)


if __name__ == "__main__":
    main("experiments/initial_experiments/databases")

import os

import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine, inspect


# Define the analysis functions
def plot_lifespan_distribution(dataframe):
    """Plot the distribution of agent lifespans."""
    dataframe["lifespan"] = dataframe["death_time"] - dataframe["birth_time"]
    plt.figure(figsize=(10, 6))
    plt.hist(dataframe["lifespan"], bins=30, edgecolor="k", alpha=0.7)
    plt.title("Lifespan Distribution")
    plt.xlabel("Lifespan (Time Units)")
    plt.ylabel("Number of Agents")
    return plt


def plot_lineage_size(dataframe):
    """Plot the distribution of lineage sizes."""
    lineage_sizes = dataframe["genome_id"].value_counts()
    plt.figure(figsize=(10, 6))
    plt.hist(lineage_sizes, bins=20, edgecolor="k", alpha=0.7)
    plt.title("Lineage Size Distribution")
    plt.xlabel("Number of Descendants")
    plt.ylabel("Number of Parents")
    return plt


def plot_agent_types_over_time(dataframe):
    """Plot the number of agents of each type over time."""
    dataframe["lifetime"] = dataframe["death_time"] - dataframe["birth_time"]
    agent_counts = (
        dataframe.groupby(["agent_type", "birth_time"]).size().unstack(fill_value=0)
    )
    agent_counts = agent_counts.cumsum(axis=1)  # Cumulative count over time
    plt.figure(figsize=(12, 6))
    agent_counts.T.plot(kind="line", linewidth=2)
    plt.title("Number of Agents Over Time by Type")
    plt.xlabel("Time")
    plt.ylabel("Number of Agents")
    plt.legend(title="Agent Type")
    return plt


def plot_reproduction_success_rate(dataframe, connection_string=None):
    """Plot the reproduction rate over generations."""
    plt.figure(figsize=(10, 6))

    try:
        if connection_string is None:
            connection_string = "sqlite:///simulations/simulation_results.db"

        engine = create_engine(connection_string)

        # Query reproduction events data grouped by parent generation
        repro_query = """
        SELECT 
            parent_generation,
            COUNT(*) as attempts,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes
        FROM reproduction_events
        GROUP BY parent_generation
        ORDER BY parent_generation
        """

        repro_data = pd.read_sql(repro_query, engine)

        if not repro_data.empty:
            # Plot success rate by generation
            success_rate = repro_data["successes"] / repro_data["attempts"] * 100
            plt.plot(
                repro_data["parent_generation"],
                success_rate,
                marker="o",
                alpha=0.6,
                label="Success Rate",
            )

            # Plot total attempts as a filled area
            plt.fill_between(
                repro_data["parent_generation"],
                repro_data["attempts"],
                alpha=0.2,
                label="Total Attempts",
            )

            plt.title("Reproduction Success Rate by Generation")
            plt.xlabel("Parent Generation")
            plt.ylabel("Success Rate (%)")
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            # Create a placeholder plot if no data
            plt.text(
                0.5,
                0.5,
                "No reproduction data available",
                horizontalalignment="center",
                verticalalignment="center",
            )
            plt.title("Reproduction Success Rate")
            plt.xlabel("Generation")
            plt.ylabel("Success Rate (%)")

    except Exception as e:
        print(f"Error in reproduction success rate plot: {e}")
        # Create a simple placeholder plot
        plt.text(
            0.5,
            0.5,
            "Error plotting reproduction data",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.title("Reproduction Success Rate")
        plt.xlabel("Generation")
        plt.ylabel("Success Rate (%)")

    return plt


# Load the dataset
def main(dataframe):
    try:
        # Create output directory
        output_dir = "chart_analysis"
        os.makedirs(output_dir, exist_ok=True)

        # Dictionary of plot functions and their names
        plot_functions = {
            "lifespan_distribution": plot_lifespan_distribution,
            "lineage_size": plot_lineage_size,
            "agent_types_over_time": plot_agent_types_over_time,
            "reproduction_success_rate": plot_reproduction_success_rate,
        }

        # Generate and save each plot
        for name, func in plot_functions.items():
            print(f"Plotting {name}...")
            plt = func(dataframe)
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


# Run the analysis
if __name__ == "__main__":
    # connection_string = "sqlite:///simulations/simulation_20241110_122335.db"
    connection_string = "sqlite:///simulations/simulation_results.db"

    # Create engine
    engine = create_engine(connection_string)

    df = pd.read_sql("SELECT * FROM Agents", engine)

    main(df)

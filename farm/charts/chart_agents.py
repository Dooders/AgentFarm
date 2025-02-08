import os

import matplotlib.pyplot as plt
import pandas as pd


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


def plot_reproduction_success_rate(dataframe):
    """Plot the reproduction rate over generations."""
    plt.figure(figsize=(10, 6))
    # Count number of agents per generation
    reproduction_counts = dataframe.groupby("generation").size()
    plt.plot(
        reproduction_counts.index, reproduction_counts.values, marker="o", alpha=0.6
    )
    plt.title("Population Size by Generation")
    plt.xlabel("Generation")
    plt.ylabel("Number of Agents")
    plt.grid(True, alpha=0.3)
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
    import pandas as pd
    from sqlalchemy import create_engine, inspect

    # connection_string = "sqlite:///simulations/simulation_20241110_122335.db"
    connection_string = "sqlite:///simulations/simulation_results.db"

    # Create engine
    engine = create_engine(connection_string)

    df = pd.read_sql("SELECT * FROM Agents", engine)

    main(df)

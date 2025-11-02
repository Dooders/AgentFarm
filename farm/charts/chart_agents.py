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
    # Extract base genome_id (without counter) for lineage grouping
    # Format: parent1:parent2:counter (counter >= 1) -> group by parent1:parent2
    def get_base_genome_id(genome_id: str) -> str:
        """Extract base genome_id without counter."""
        if pd.isna(genome_id) or genome_id == "":
            return "::"
        parts = str(genome_id).split(":")
        # If has counter (3 parts with digit in third), return first 2 parts joined
        if len(parts) == 3 and parts[2].isdigit():
            # Preserve trailing colon for empty parents case (::counter -> ::)
            if parts[0] == "" and parts[1] == "":
                return "::"
            elif parts[1] == "":
                # Single parent with counter: agent_a:counter -> agent_a:
                return f"{parts[0]}:"
            else:
                # Two parents with counter: agent_a:agent_b:counter -> agent_a:agent_b
                return f"{parts[0]}:{parts[1]}"
        elif len(parts) == 2:
            # Could be ::, agent_a:, agent_a:counter, or agent_a:agent_b
            parent1, parent2 = parts
            if parent1 == "" and parent2 == "":
                # Initial agent: ::
                return "::"
            elif parent2.isdigit():
                # Single parent with counter: agent_a:counter -> agent_a:
                return f"{parent1}:"
            elif parent2 == "":
                # Single parent without counter: agent_a:
                return f"{parent1}:"
            else:
                # Two parents without counter: agent_a:agent_b
                return f"{parent1}:{parent2}"
        # Otherwise return as-is (already base or malformed, possibly legacy format)
        return ":".join(parts[:2]) if len(parts) >= 2 else str(genome_id)
    
    dataframe["base_genome_id"] = dataframe["genome_id"].apply(get_base_genome_id)
    lineage_sizes = dataframe["base_genome_id"].value_counts()
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
    connection_string = "sqlite:///simulations/simulation.db"

    # Create engine
    engine = create_engine(connection_string)

    df = pd.read_sql("SELECT * FROM Agents", engine)

    main(df)

import matplotlib.pyplot as plt
import pandas as pd


# Define the analysis functions
def plot_total_reward_distribution(dataframe):
    """Plot the distribution of total rewards across all agents."""
    plt.figure(figsize=(10, 6))
    plt.hist(dataframe["total_reward"], bins=30, edgecolor="k", alpha=0.7)
    plt.title("Total Reward Distribution")
    plt.xlabel("Total Reward")
    plt.ylabel("Number of Agents")
    return plt


def plot_average_health_vs_age(dataframe):
    """Plot the relationship between agent age and average health."""
    avg_health = dataframe.groupby("age")["current_health"].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(avg_health.index, avg_health.values, label="Average Health", color="blue")
    plt.title("Average Health vs. Age")
    plt.xlabel("Age")
    plt.ylabel("Average Health")
    plt.legend()
    return plt


def plot_average_resource_vs_age(dataframe):
    """Plot the relationship between agent age and average resource level."""
    avg_resources = dataframe.groupby("age")["resource_level"].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(
        avg_resources.index,
        avg_resources.values,
        label="Average Resource Level",
        color="green",
    )
    plt.title("Average Resource Level vs. Age")
    plt.xlabel("Age")
    plt.ylabel("Resource Level")
    plt.legend()
    return plt


def plot_average_health_over_time(dataframe):
    """Plot the average health of all agents over time with standard deviation bands."""
    health_stats = (
        dataframe.groupby("step_number")["current_health"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        health_stats["step_number"],
        health_stats["mean"],
        label="Average Health",
        color="blue",
    )

    plt.fill_between(
        health_stats["step_number"],
        health_stats["mean"] - health_stats["std"],
        health_stats["mean"] + health_stats["std"],
        alpha=0.2,
        color="blue",
        label="±1 Standard Deviation",
    )

    plt.title("Average Health Over Time (with Standard Deviation)")
    plt.xlabel("Step Number")
    plt.ylabel("Health")
    plt.legend()
    return plt


def plot_average_resource_over_time(dataframe):
    """Plot the average resource level of all agents over time with standard deviation bands."""
    resource_stats = (
        dataframe.groupby("step_number")["resource_level"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        resource_stats["step_number"],
        resource_stats["mean"],
        label="Average Resource Level",
        color="green",
    )

    plt.fill_between(
        resource_stats["step_number"],
        resource_stats["mean"] - resource_stats["std"],
        resource_stats["mean"] + resource_stats["std"],
        alpha=0.2,
        color="green",
        label="±1 Standard Deviation",
    )

    plt.title("Average Resource Level Over Time (with Standard Deviation)")
    plt.xlabel("Step Number")
    plt.ylabel("Resource Level")
    plt.legend()
    return plt


def main(dataframe):
    try:
        # Dictionary of plot functions and their names
        plot_functions = {
            "total_reward_distribution": plot_total_reward_distribution,
            "average_health_vs_age": plot_average_health_vs_age,
            "average_resource_vs_age": plot_average_resource_vs_age,
            "average_health_over_time": plot_average_health_over_time,
            "average_resource_over_time": plot_average_resource_over_time,
        }

        # Generate and show each plot
        for name, func in plot_functions.items():
            print(f"Plotting {name}...")
            plt = func(dataframe)
            if plt is not None:
                plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"DataFrame shape: {dataframe.shape}")
        print(f"DataFrame columns: {dataframe.columns.tolist()}")


if __name__ == "__main__":
    import pandas as pd
    from sqlalchemy import create_engine

    connection_string = "sqlite:///simulations/simulation_results.db"

    # Create engine
    engine = create_engine(connection_string)

    try:
        # Modified query to include starting_health from agents table
        query = """
        SELECT s.*, a.starting_health, a.starvation_threshold 
        FROM agent_states s
        JOIN agents a ON s.agent_id = a.agent_id
        """
        df = pd.read_sql(query, engine)

        if df.empty:
            print("No data found in the database")
        else:
            main(df)

    except Exception as e:
        print(f"Database error: {e}")

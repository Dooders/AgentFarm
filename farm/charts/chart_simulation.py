import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text

from .chart_utils import (
    save_plot,
)  # Import from new utilities module instead of chart_agents

# Define database connection string once at module level
CONNECTION_STRING = "sqlite:///simulations/simulation.db"

# Define the analysis functions


def plot_population_dynamics(dataframe):
    """Plot total agents, system agents, and independent agents over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe["step_number"],
        dataframe["total_agents"],
        label="Total Agents",
        color="blue",
    )
    plt.plot(
        dataframe["step_number"],
        dataframe["system_agents"],
        label="System Agents",
        color="green",
    )
    plt.plot(
        dataframe["step_number"],
        dataframe["independent_agents"],
        label="Independent Agents",
        color="orange",
    )
    plt.plot(
        dataframe["step_number"],
        dataframe["control_agents"],
        label="Control Agents",
        color="red",
    )
    plt.title("Population Dynamics Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Number of Agents")
    plt.legend()
    return plt


def plot_births_and_deaths(dataframe):
    """Plot births and deaths over time, with deaths shown as negative values, starting from step 20."""
    plt.figure(figsize=(10, 6))

    # Filter data starting from step 20
    df_filtered = dataframe[dataframe["step_number"] >= 20]

    # Make deaths negative but keep actual numbers
    births = df_filtered["births"]
    deaths = -df_filtered["deaths"]  # Make deaths negative

    plt.fill_between(
        df_filtered["step_number"], births, 0, label="Births", color="green", alpha=0.3
    )
    plt.fill_between(
        df_filtered["step_number"], deaths, 0, label="Deaths", color="red", alpha=0.3
    )

    plt.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    plt.title("Population Changes Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Number of Agents (Births +, Deaths -)")
    plt.legend()

    # Set y-axis limits to be symmetrical around zero
    max_value = max(births.max(), abs(deaths.min()))
    plt.ylim(-max_value * 1.1, max_value * 1.1)
    return plt


def plot_births_and_deaths_by_type(dataframe, connection_string=None):
    """Plot births and deaths over time separated by agent type."""
    if connection_string is None:
        connection_string = CONNECTION_STRING

    try:
        engine = create_engine(connection_string)
        events_query = """
        SELECT
            time_point as step_number,
            agent_type,
            COUNT(CASE WHEN event_type = 'birth' THEN 1 END) as births,
            COUNT(CASE WHEN event_type = 'death' THEN 1 END) as deaths
        FROM (
            SELECT
                birth_time as time_point,
                agent_type,
                'birth' as event_type
            FROM agents
            WHERE birth_time >= 20
            UNION ALL
            SELECT
                death_time as time_point,
                agent_type,
                'death' as event_type
            FROM agents
            WHERE death_time IS NOT NULL
            AND death_time >= 20
        ) events
        GROUP BY time_point, agent_type
        ORDER BY time_point
        """

        events_df = pd.read_sql(events_query, engine)

        if events_df.empty:
            print("No birth/death data available to plot")
            return None

        # Create subplots for each agent type
        agent_types = ["SystemAgent", "IndependentAgent", "ControlAgent"]
        active_agent_types = [
            at for at in agent_types if at in events_df["agent_type"].unique()
        ]

        if not active_agent_types:
            print("No data available for any agent type")
            return None

        fig, axes = plt.subplots(
            len(active_agent_types), 1, figsize=(15, 12), sharex=True
        )
        fig.suptitle("Population Changes by Agent Type Over Time", fontsize=14, y=0.95)

        # Handle case where there's only one subplot
        if len(active_agent_types) == 1:
            axes = [axes]

        birth_color = "green"
        death_color = "red"

        # Find global min and max values
        global_min_step = events_df["step_number"].min()
        global_max_step = events_df["step_number"].max()
        global_max_value = max(
            events_df["births"].fillna(0).max(), events_df["deaths"].fillna(0).max()
        )

        for idx, agent_type in enumerate(active_agent_types):
            ax = axes[idx]
            agent_data = events_df[events_df["agent_type"] == agent_type]

            if not agent_data.empty:
                ax.fill_between(
                    agent_data["step_number"],
                    agent_data["births"].fillna(0),
                    0,
                    label="Births",
                    color=birth_color,
                    alpha=0.3,
                )
                ax.fill_between(
                    agent_data["step_number"],
                    -agent_data["deaths"].fillna(0),
                    0,
                    label="Deaths",
                    color=death_color,
                    alpha=0.3,
                )
                ax.legend()
                ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

            ax.set_xlim(global_min_step, global_max_step)
            ax.set_ylim(-global_max_value * 1.1, global_max_value * 1.1)
            ax.grid(True, alpha=0.3)

            display_name = agent_type.replace("Agent", "")
            ax.set_title(f"{display_name} Agents", pad=5)
            ax.set_ylabel("Count")

        axes[-1].set_xlabel("Step Number")
        plt.tight_layout()
        return plt

    except Exception as e:
        print(f"Error plotting births and deaths by type: {e}")
        return None


def plot_resource_efficiency(dataframe):
    """Plot resource efficiency and total resources over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe["step_number"],
        dataframe["resource_efficiency"],
        label="Resource Efficiency",
        color="purple",
    )
    plt.plot(
        dataframe["step_number"],
        dataframe["total_resources"],
        label="Total Resources",
        color="blue",
        linestyle="--",
    )
    plt.title("Resource Efficiency and Total Resources Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Value")
    plt.legend()
    return plt


def plot_agent_health_and_age(dataframe):
    """Plot average agent health and age over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe["step_number"],
        dataframe["average_agent_health"],
        label="Average Health",
        color="cyan",
    )
    plt.plot(
        dataframe["step_number"],
        dataframe["average_agent_age"],
        label="Average Age",
        color="magenta",
    )
    plt.title("Agent Health and Age Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Value")
    plt.legend()
    return plt


def plot_combat_metrics(dataframe):
    """Plot combat encounters and successful attacks over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe["step_number"],
        dataframe["combat_encounters"],
        label="Combat Encounters",
        color="orange",
    )
    plt.plot(
        dataframe["step_number"],
        dataframe["successful_attacks"],
        label="Successful Attacks",
        color="green",
    )
    plt.title("Combat Metrics Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Count")
    plt.legend()
    return plt


def plot_resource_sharing(dataframe):
    """Plot the amount of resources shared over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe["step_number"],
        dataframe["resources_shared"],
        label="Resources Shared",
        color="gold",
    )
    plt.title("Resources Shared Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Resources Shared")
    plt.legend()
    return plt


def plot_evolutionary_metrics(dataframe):
    """Plot genetic diversity and dominant genome ratio over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe["step_number"],
        dataframe["genetic_diversity"],
        label="Genetic Diversity",
        color="blue",
    )
    plt.plot(
        dataframe["step_number"],
        dataframe["dominant_genome_ratio"],
        label="Dominant Genome Ratio",
        color="purple",
    )
    plt.title("Evolutionary Metrics Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Value")
    plt.legend()
    return plt


def plot_resource_distribution_entropy(dataframe):
    """Plot resource distribution entropy over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe["step_number"],
        dataframe["resource_distribution_entropy"],
        label="Resource Distribution Entropy",
        color="darkred",
    )
    plt.title("Resource Distribution Entropy Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Entropy")
    plt.legend()
    return plt


def plot_rewards(dataframe):
    """Plot average reward over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe["step_number"],
        dataframe["average_reward"],
        label="Average Reward",
        color="teal",
    )
    plt.title("Average Reward Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Average Reward")
    plt.legend()
    return plt


def plot_average_resources(dataframe):
    """Plot average agent resource levels over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        dataframe["step_number"],
        dataframe["average_agent_resources"],
        label="Average Agent Resources",
        color="green",
    )
    plt.title("Average Agent Resources Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Average Resources")
    plt.legend()
    return plt


def plot_agent_lifespan_histogram(dataframe, connection_string=None):
    """Plot histogram of agent lifespans."""
    if connection_string is None:
        connection_string = CONNECTION_STRING

    plt.figure(figsize=(10, 6))
    max_step = dataframe["step_number"].max()

    try:
        engine = create_engine(connection_string)
        agents_df = pd.read_sql("SELECT birth_time, death_time FROM agents", engine)

        lifespans = agents_df.apply(
            lambda x: (
                x["death_time"] - x["birth_time"]
                if pd.notnull(x["death_time"])
                else max_step - x["birth_time"]
            ),
            axis=1,
        )

        plt.hist(lifespans, bins=30, color="blue", alpha=0.7)
        plt.title("Distribution of Agent Lifespans")
        plt.xlabel("Lifespan (steps)")
        plt.ylabel("Number of Agents")

        mean_lifespan = lifespans.mean()
        median_lifespan = lifespans.median()
        plt.axvline(
            mean_lifespan,
            color="red",
            linestyle="--",
            label=f"Mean: {mean_lifespan:.1f}",
        )
        plt.axvline(
            median_lifespan,
            color="green",
            linestyle="--",
            label=f"Median: {median_lifespan:.1f}",
        )

        plt.legend()
        return plt

    except Exception as e:
        print(f"Error plotting lifespan histogram: {e}")
        return None


def plot_agent_type_comparison(dataframe, connection_string=None):
    """Create a radar chart comparing different agent types on key metrics."""
    if connection_string is None:
        connection_string = CONNECTION_STRING

    # Get the last step's data
    final_step = dataframe.iloc[-1]

    # Calculate metrics per agent type
    metrics = {
        "Population": {
            "System": float(final_step["system_agents"]),
            "Independent": float(final_step["independent_agents"]),
            "Control": float(final_step["control_agents"]),
        }
    }

    # Query additional metrics from the database
    engine = create_engine(connection_string)

    # Calculate average resources per agent type
    resources_query = """
    SELECT
        a.agent_type,
        COALESCE(AVG(CAST(s.resource_level AS FLOAT)), 0) as avg_resources,
        COALESCE(AVG(CAST(s.current_health AS FLOAT)), 0) as avg_health,
        COALESCE(AVG(CAST(s.age AS FLOAT)), 0) as avg_age,
        COALESCE(AVG(CAST(s.total_reward AS FLOAT)), 0) as avg_reward
    FROM agents a
    JOIN agent_states s ON a.agent_id = s.agent_id
    WHERE s.step_number = (SELECT MAX(step_number) FROM agent_states)
    GROUP BY a.agent_type
    """

    agent_metrics = pd.read_sql(resources_query, engine)
    agent_metrics.set_index("agent_type", inplace=True)

    # Add metrics to our dictionary
    for metric in ["avg_resources", "avg_health", "avg_age", "avg_reward"]:
        metrics[metric.replace("avg_", "").title()] = {
            "System": float(
                agent_metrics.at["SystemAgent", metric]
                if "SystemAgent" in agent_metrics.index
                else 0
            ),
            "Independent": float(
                agent_metrics.at["IndependentAgent", metric]
                if "IndependentAgent" in agent_metrics.index
                else 0
            ),
            "Control": float(
                agent_metrics.at["ControlAgent", metric]
                if "ControlAgent" in agent_metrics.index
                else 0
            ),
        }

    # Convert to numpy arrays for plotting
    categories = list(metrics.keys())
    agent_types = ["System", "Independent", "Control"]

    # Normalize values to 0-1 scale for each metric
    values = np.zeros((len(agent_types), len(categories)))
    for i, agent_type in enumerate(agent_types):
        for j, category in enumerate(categories):
            try:
                values[i, j] = float(metrics[category][agent_type])
            except (ValueError, TypeError):
                values[i, j] = 0.0

    # Normalize each metric
    for j in range(values.shape[1]):
        max_val = values[:, j].max()
        if max_val != 0:
            values[:, j] = values[:, j] / max_val

    # Set up the angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)

    # Close the plot by appending the first value to the end
    values = np.concatenate((values, values[:, [0]]), axis=1)
    angles = np.concatenate((angles, [angles[0]]))
    categories = categories + [categories[0]]

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))

    # Plot data
    colors = ["green", "orange", "red"]
    for i, agent_type in enumerate(agent_types):
        ax.plot(angles, values[i], "o-", linewidth=2, label=agent_type, color=colors[i])
        ax.fill(angles, values[i], alpha=0.25, color=colors[i])

    # Fix axis to go in the right order and start at 12 o'clock
    # Type: ignore for polar axes methods
    ax.set_theta_offset(np.pi / 2)  # type: ignore
    ax.set_theta_direction(-1)  # type: ignore

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])

    # Add gridlines and adjust their appearance
    ax.grid(True, alpha=0.3)

    # Set the radial limits and ticks
    ax.set_ylim(0, 1)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])  # type: ignore

    # Add legend with better positioning
    plt.legend(loc="center left", bbox_to_anchor=(1.2, 0.5))

    plt.title("Agent Type Performance Comparison\n(Normalized Metrics)", pad=20)
    plt.tight_layout()
    return plt


def plot_reproduction_success_rate(dataframe, connection_string=None):
    """Plot reproduction success rate over time."""
    if connection_string is None:
        connection_string = CONNECTION_STRING

    plt.figure(figsize=(10, 6))

    # Query reproduction events data
    engine = create_engine(connection_string)
    repro_query = """
    SELECT
        step_number,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attempts
    FROM reproduction_events
    GROUP BY step_number
    ORDER BY step_number
    """

    repro_data = pd.read_sql(repro_query, engine)

    # Calculate success rate
    repro_data["success_rate"] = (
        repro_data["successful_attempts"] / repro_data["total_attempts"] * 100
    )

    # Plot success rate
    plt.plot(
        repro_data["step_number"],
        repro_data["success_rate"],
        label="Success Rate",
        color="green",
    )

    # Plot total attempts as a light fill
    plt.fill_between(
        repro_data["step_number"],
        repro_data["total_attempts"],
        alpha=0.2,
        color="blue",
        label="Total Attempts",
    )

    plt.title("Reproduction Success Rate Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Success Rate (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    return plt


def plot_reproduction_resources(dataframe, connection_string=None):
    """Plot resource distribution in reproduction events."""
    if connection_string is None:
        connection_string = CONNECTION_STRING

    plt.figure(figsize=(12, 6))

    # Query reproduction events data
    engine = create_engine(connection_string)
    resource_query = """
    SELECT
        parent_resources_before,
        parent_resources_after,
        offspring_initial_resources
    FROM reproduction_events
    WHERE success = 1
    """

    resource_data = pd.read_sql(resource_query, engine)

    # Create box plots
    data = [
        resource_data["parent_resources_before"],
        resource_data["parent_resources_after"],
        resource_data["offspring_initial_resources"],
    ]

    plt.boxplot(
        data, tick_labels=["Parent Before", "Parent After", "Offspring Initial"]
    )

    plt.title("Resource Distribution in Successful Reproduction Events")
    plt.ylabel("Resource Amount")
    plt.grid(True, alpha=0.3)
    return plt


def plot_generational_analysis(dataframe, connection_string=None):
    """Plot analysis of reproduction across generations."""
    if connection_string is None:
        connection_string = CONNECTION_STRING

    plt.figure(figsize=(12, 8))

    # Query generation data
    engine = create_engine(connection_string)
    gen_query = """
    SELECT
        parent_generation,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_attempts,
        AVG(CASE WHEN success = 1 THEN offspring_initial_resources ELSE NULL END) as avg_offspring_resources
    FROM reproduction_events
    GROUP BY parent_generation
    ORDER BY parent_generation
    """

    gen_data = pd.read_sql(gen_query, engine)

    # Create subplot for success rate
    plt.subplot(2, 1, 1)
    success_rate = gen_data["successful_attempts"] / gen_data["total_attempts"] * 100
    plt.bar(gen_data["parent_generation"], success_rate, color="green", alpha=0.6)
    plt.title("Reproduction Success Rate by Generation")
    plt.ylabel("Success Rate (%)")
    plt.grid(True, alpha=0.3)

    # Create subplot for average offspring resources
    plt.subplot(2, 1, 2)
    plt.bar(
        gen_data["parent_generation"],
        gen_data["avg_offspring_resources"],
        color="blue",
        alpha=0.6,
    )
    plt.title("Average Offspring Initial Resources by Generation")
    plt.xlabel("Parent Generation")
    plt.ylabel("Average Resources")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    return plt


def plot_reproduction_failure_reasons(dataframe, connection_string=None):
    """Plot reproduction failure reasons over time."""
    if connection_string is None:
        connection_string = CONNECTION_STRING

    plt.figure(figsize=(12, 6))

    # Query reproduction events data for failures
    engine = create_engine(connection_string)
    failure_query = """
    SELECT
        step_number,
        failure_reason,
        COUNT(*) as count
    FROM reproduction_events
    WHERE success = 0
        AND failure_reason IS NOT NULL
    GROUP BY step_number, failure_reason
    ORDER BY step_number
    """

    failure_data = pd.read_sql(failure_query, engine)

    # Pivot the data to get failure reasons as columns
    pivot_data = failure_data.pivot(
        index="step_number", columns="failure_reason", values="count"
    ).fillna(0)

    # Create stacked area plot
    plt.stackplot(pivot_data.index, pivot_data.T, labels=pivot_data.columns, alpha=0.6)

    plt.title("Reproduction Failure Reasons Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Number of Failures")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    return plt


def check_database(connection_string):
    try:
        engine = create_engine(connection_string)
        inspector = inspect(engine)

        print("Available tables:", inspector.get_table_names())

        if "agents" in inspector.get_table_names():
            print("Columns in agents table:", inspector.get_columns("agents"))

            # Check for some data
            with engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM agents")).scalar()
                print(f"Number of agents in database: {result}")

    except Exception as e:
        print(f"Database connection error: {e}")


# Load the dataset
def main(dataframe):
    try:
        print("Plotting population dynamics...")
        plt = plot_population_dynamics(dataframe)
        save_plot(plt, "population_dynamics")

        print("Plotting births and deaths...")
        plt = plot_births_and_deaths(dataframe)
        save_plot(plt, "births_and_deaths")

        print("Plotting births and deaths by type...")
        plt = plot_births_and_deaths_by_type(dataframe)
        save_plot(plt, "births_and_deaths_by_type")

        print("Plotting resource efficiency...")
        plt = plot_resource_efficiency(dataframe)
        save_plot(plt, "resource_efficiency")

        print("Plotting agent health and age...")
        plt = plot_agent_health_and_age(dataframe)
        save_plot(plt, "agent_health_and_age")

        print("Plotting combat metrics...")
        plt = plot_combat_metrics(dataframe)
        save_plot(plt, "combat_metrics")

        print("Plotting resource sharing...")
        plt = plot_resource_sharing(dataframe)
        save_plot(plt, "resource_sharing")

        print("Plotting evolutionary metrics...")
        plt = plot_evolutionary_metrics(dataframe)
        save_plot(plt, "evolutionary_metrics")

        print("Plotting resource distribution entropy...")
        plt = plot_resource_distribution_entropy(dataframe)
        save_plot(plt, "resource_distribution_entropy")

        print("Plotting average agent resources...")
        plt = plot_average_resources(dataframe)
        save_plot(plt, "average_resources")

        print("Plotting average rewards...")
        plt = plot_rewards(dataframe)
        save_plot(plt, "rewards")

        print("Plotting agent lifespan histogram...")
        plt = plot_agent_lifespan_histogram(dataframe)
        save_plot(plt, "agent_lifespan_histogram")

        print("Plotting agent type comparison...")
        plt = plot_agent_type_comparison(dataframe)
        save_plot(plt, "agent_type_comparison")

        print("Plotting reproduction success rate...")
        plt = plot_reproduction_success_rate(dataframe)
        save_plot(plt, "reproduction_success_rate")

        print("Plotting reproduction resources...")
        plt = plot_reproduction_resources(dataframe)
        save_plot(plt, "reproduction_resources")

        print("Plotting generational analysis...")
        plt = plot_generational_analysis(dataframe)
        save_plot(plt, "generational_analysis")

        print("Plotting reproduction failure reasons...")
        plt = plot_reproduction_failure_reasons(dataframe)
        save_plot(plt, "reproduction_failure_reasons")

    except Exception as e:
        print(f"An error occurred: {e}")


# Run the analysis
if __name__ == "__main__":
    check_database(CONNECTION_STRING)

    # Create engine
    engine = create_engine(CONNECTION_STRING)

    df = pd.read_sql("SELECT * FROM Simulation_Steps", engine)

    main(df)

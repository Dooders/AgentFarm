import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect

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
    plt.show()


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

    plt.show()


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
    plt.show()


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
    plt.show()


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
    plt.show()


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
    plt.show()


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
    plt.show()


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
    plt.show()


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
    plt.show()


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
    plt.show()


def plot_agent_lifespan_histogram(dataframe):
    """Plot histogram of agent lifespans."""
    plt.figure(figsize=(10, 6))

    # Calculate lifespans - for living agents, use the last step as death_time
    max_step = dataframe["step_number"].max()
    lifespans = []

    # Query the agents table directly since we need birth/death data
    engine = create_engine(connection_string)
    agents_df = pd.read_sql("SELECT birth_time, death_time FROM agents", engine)

    # Calculate lifespan for each agent
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

    # Add mean and median lines
    mean_lifespan = lifespans.mean()
    median_lifespan = lifespans.median()
    plt.axvline(
        mean_lifespan, color="red", linestyle="--", label=f"Mean: {mean_lifespan:.1f}"
    )
    plt.axvline(
        median_lifespan,
        color="green",
        linestyle="--",
        label=f"Median: {median_lifespan:.1f}",
    )

    plt.legend()
    plt.show()


def plot_agent_type_comparison(dataframe):
    """Create a radar chart comparing different agent types on key metrics at the end of simulation."""
    # Get the last step's data
    final_step = dataframe.iloc[-1]

    # Calculate metrics per agent type
    metrics = {
        "Population": {
            "System": final_step["system_agents"],
            "Independent": final_step["independent_agents"],
            "Control": final_step["control_agents"],
        }
    }

    # Query additional metrics from the database
    engine = create_engine(connection_string)

    # Calculate average resources per agent type
    resources_query = """
    SELECT 
        a.agent_type,
        AVG(s.resource_level) as avg_resources,
        AVG(s.current_health) as avg_health,
        AVG(s.age) as avg_age,
        AVG(s.total_reward) as avg_reward
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
            "System": (
                agent_metrics.loc["system", metric]
                if "system" in agent_metrics.index
                else 0
            ),
            "Independent": (
                agent_metrics.loc["independent", metric]
                if "independent" in agent_metrics.index
                else 0
            ),
            "Control": (
                agent_metrics.loc["control", metric]
                if "control" in agent_metrics.index
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
            values[i, j] = metrics[category][agent_type]

    # Normalize each metric
    for j in range(values.shape[1]):
        if values[:, j].max() != 0:
            values[:, j] = values[:, j] / values[:, j].max()

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
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories[:-1])

    # Add gridlines and adjust their appearance
    ax.grid(True, alpha=0.3)

    # Set the radial limits and ticks
    ax.set_ylim(0, 1)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])

    # Add legend with better positioning
    plt.legend(loc="center left", bbox_to_anchor=(1.2, 0.5))

    plt.title("Agent Type Performance Comparison\n(Normalized Metrics)", pad=20)
    plt.tight_layout()
    plt.show()


def plot_reproduction_success_rate(dataframe):
    """Plot reproduction success rate over time."""
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
    repro_data['success_rate'] = (repro_data['successful_attempts'] / 
                                 repro_data['total_attempts'] * 100)
    
    # Plot success rate
    plt.plot(repro_data['step_number'], 
             repro_data['success_rate'],
             label='Success Rate',
             color='green')
    
    # Plot total attempts as a light fill
    plt.fill_between(repro_data['step_number'],
                     repro_data['total_attempts'],
                     alpha=0.2,
                     color='blue',
                     label='Total Attempts')
    
    plt.title('Reproduction Success Rate Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Success Rate (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_reproduction_resources(dataframe):
    """Plot resource distribution in reproduction events."""
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
        resource_data['parent_resources_before'],
        resource_data['parent_resources_after'],
        resource_data['offspring_initial_resources']
    ]
    
    plt.boxplot(data, labels=['Parent Before', 'Parent After', 'Offspring Initial'])
    
    plt.title('Resource Distribution in Successful Reproduction Events')
    plt.ylabel('Resource Amount')
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_generational_analysis(dataframe):
    """Plot analysis of reproduction across generations."""
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
    success_rate = (gen_data['successful_attempts'] / gen_data['total_attempts'] * 100)
    plt.bar(gen_data['parent_generation'], success_rate, color='green', alpha=0.6)
    plt.title('Reproduction Success Rate by Generation')
    plt.ylabel('Success Rate (%)')
    plt.grid(True, alpha=0.3)
    
    # Create subplot for average offspring resources
    plt.subplot(2, 1, 2)
    plt.bar(gen_data['parent_generation'], 
            gen_data['avg_offspring_resources'],
            color='blue',
            alpha=0.6)
    plt.title('Average Offspring Initial Resources by Generation')
    plt.xlabel('Parent Generation')
    plt.ylabel('Average Resources')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_reproduction_failure_reasons(dataframe):
    """Plot reproduction failure reasons over time.
    
    Creates a stacked area chart showing the distribution of different
    failure reasons for reproduction attempts across simulation steps.
    """
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
        index='step_number',
        columns='failure_reason',
        values='count'
    ).fillna(0)
    
    # Create stacked area plot
    plt.stackplot(
        pivot_data.index,
        pivot_data.T,
        labels=pivot_data.columns,
        alpha=0.6
    )
    
    plt.title('Reproduction Failure Reasons Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Number of Failures')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Load the dataset
def main(dataframe):

    try:

        # Call each function to analyze and visualize
        print("Plotting population dynamics...")
        plot_population_dynamics(dataframe)

        print("Plotting births and deaths...")
        plot_births_and_deaths(dataframe)

        print("Plotting resource efficiency...")
        plot_resource_efficiency(dataframe)

        print("Plotting agent health and age...")
        plot_agent_health_and_age(dataframe)

        print("Plotting combat metrics...")
        plot_combat_metrics(dataframe)

        print("Plotting resource sharing...")
        plot_resource_sharing(dataframe)

        print("Plotting evolutionary metrics...")
        plot_evolutionary_metrics(dataframe)

        print("Plotting resource distribution entropy...")
        plot_resource_distribution_entropy(dataframe)

        print("Plotting average agent resources...")
        plot_average_resources(dataframe)

        print("Plotting average rewards...")
        plot_rewards(dataframe)

        print("Plotting agent lifespan histogram...")
        plot_agent_lifespan_histogram(dataframe)

        print("Plotting agent type comparison...")
        plot_agent_type_comparison(dataframe)

        print("Plotting reproduction success rate...")
        plot_reproduction_success_rate(dataframe)
        
        print("Plotting reproduction resources...")
        plot_reproduction_resources(dataframe)
        
        print("Plotting generational analysis...")
        plot_generational_analysis(dataframe)

        print("Plotting reproduction failure reasons...")
        plot_reproduction_failure_reasons(dataframe)

    except Exception as e:
        print(f"An error occurred: {e}")


# Run the analysis
if __name__ == "__main__":
    # connection_string = "sqlite:///simulations/simulation_20241110_122335.db"
    connection_string = "sqlite:///simulations/simulation_results.db"

    # Create engine
    engine = create_engine(connection_string)

    df = pd.read_sql("SELECT * FROM Simulation_Steps", engine)

    main(df)

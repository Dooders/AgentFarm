import pandas as pd
import matplotlib.pyplot as plt

# Define the analysis functions

def plot_lifespan_distribution(dataframe):
    dataframe['lifespan'] = dataframe['death_time'] - dataframe['birth_time']
    plt.figure(figsize=(10, 6))
    plt.hist(dataframe['lifespan'], bins=30, edgecolor='k', alpha=0.7)
    plt.title('Lifespan Distribution')
    plt.xlabel('Lifespan (Time Units)')
    plt.ylabel('Number of Agents')
    plt.show()

def plot_spatial_distribution(dataframe):
    plt.figure(figsize=(10, 6))
    plt.scatter(dataframe['position_x'], dataframe['position_y'], alpha=0.6)
    plt.title('Spatial Distribution of Agents')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

def plot_resources_by_generation(dataframe):
    generation_means = dataframe.groupby('generation')['initial_resources'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(generation_means.index, generation_means.values, marker='o')
    plt.title('Initial Resources by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Initial Resources')
    plt.show()

def plot_starvation_thresholds(dataframe):
    thresholds = dataframe.groupby('agent_type')['starvation_threshold'].mean()
    plt.figure(figsize=(10, 6))
    thresholds.plot(kind='bar', color='skyblue', edgecolor='k')
    plt.title('Starvation Thresholds by Agent Type')
    plt.xlabel('Agent Type')
    plt.ylabel('Average Starvation Threshold')
    plt.xticks(rotation=45)
    plt.show()

def plot_lineage_size(dataframe):
    lineage_sizes = dataframe['genome_id'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.hist(lineage_sizes, bins=20, edgecolor='k', alpha=0.7)
    plt.title('Lineage Size Distribution')
    plt.xlabel('Number of Descendants')
    plt.ylabel('Number of Parents')
    plt.show()

def plot_health_vs_resources(dataframe):
    plt.figure(figsize=(10, 6))
    plt.scatter(dataframe['initial_resources'], dataframe['starting_health'], alpha=0.6)
    plt.title('Starting Health vs. Initial Resources')
    plt.xlabel('Initial Resources')
    plt.ylabel('Starting Health')
    plt.show()

def plot_agent_types_over_time(dataframe):
    dataframe['lifetime'] = dataframe['death_time'] - dataframe['birth_time']
    agent_counts = dataframe.groupby(['agent_type', 'birth_time']).size().unstack(fill_value=0)
    agent_counts = agent_counts.cumsum(axis=1)  # Cumulative count over time
    plt.figure(figsize=(12, 6))
    agent_counts.T.plot(kind='line', linewidth=2)
    plt.title('Number of Agents Over Time by Type')
    plt.xlabel('Time')
    plt.ylabel('Number of Agents')
    plt.legend(title='Agent Type')
    plt.show()

# Load the dataset
def main(dataframe):

    try:
        # Call each function to analyze and visualize
        print("Plotting lifespan distribution...")
        plot_lifespan_distribution(dataframe)

        print("Plotting spatial distribution...")
        plot_spatial_distribution(dataframe)

        print("Plotting resources by generation...")
        plot_resources_by_generation(dataframe)

        print("Plotting starvation thresholds by agent type...")
        plot_starvation_thresholds(dataframe)

        print("Plotting lineage size distribution...")
        plot_lineage_size(dataframe)

        print("Plotting health vs. resources...")
        plot_health_vs_resources(dataframe)

        print("Plotting agent types over time...")
        plot_agent_types_over_time(dataframe)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the analysis
if __name__ == "__main__":
    import pandas as pd
    from sqlalchemy import create_engine, inspect

    # connection_string = "sqlite:///simulations/simulation_20241110_122335.db"
    connection_string = "sqlite:///simulations/simulation.db"

    # Create engine
    engine = create_engine(connection_string)
    
    df = pd.read_sql("SELECT * FROM Agents", engine)

    main(df)

import pandas as pd
import matplotlib.pyplot as plt

# Define the analysis functions

def plot_health_over_time(dataframe, agent_id):
    agent_data = dataframe[dataframe['agent_id'] == agent_id]
    plt.figure(figsize=(10, 6))
    plt.plot(agent_data['step_number'], agent_data['current_health'], label='Current Health')
    plt.plot(agent_data['step_number'], agent_data['starting_health'], label='Starting Health', linestyle='--')
    plt.axhline(agent_data['starvation_threshold'].iloc[0], color='red', linestyle=':', label='Starvation Threshold')
    plt.title(f'Health Over Time for Agent {agent_id}')
    plt.xlabel('Step Number')
    plt.ylabel('Health')
    plt.legend()
    plt.show()

def plot_resource_level_over_time(dataframe, agent_id):
    agent_data = dataframe[dataframe['agent_id'] == agent_id]
    plt.figure(figsize=(10, 6))
    plt.plot(agent_data['step_number'], agent_data['resource_level'], label='Resource Level', color='green')
    plt.title(f'Resource Level Over Time for Agent {agent_id}')
    plt.xlabel('Step Number')
    plt.ylabel('Resource Level')
    plt.legend()
    plt.show()

def plot_spatial_trajectory(dataframe, agent_id):
    agent_data = dataframe[dataframe['agent_id'] == agent_id]
    plt.figure(figsize=(10, 6))
    plt.plot(agent_data['position_x'], agent_data['position_y'], marker='o', linestyle='-', label='Trajectory')
    plt.title(f'Spatial Trajectory of Agent {agent_id}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.show()

def plot_total_reward_distribution(dataframe):
    plt.figure(figsize=(10, 6))
    plt.hist(dataframe['total_reward'], bins=30, edgecolor='k', alpha=0.7)
    plt.title('Total Reward Distribution')
    plt.xlabel('Total Reward')
    plt.ylabel('Number of Agents')
    plt.show()

def plot_defending_agents_over_time(dataframe):
    defending_counts = dataframe.groupby('step_number')['is_defending'].sum()
    plt.figure(figsize=(10, 6))
    plt.plot(defending_counts.index, defending_counts.values, label='Number of Defending Agents', color='purple')
    plt.title('Defending Agents Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Number of Defending Agents')
    plt.legend()
    plt.show()

def plot_average_health_vs_age(dataframe):
    avg_health = dataframe.groupby('age')['current_health'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(avg_health.index, avg_health.values, label='Average Health', color='blue')
    plt.title('Average Health vs. Age')
    plt.xlabel('Age')
    plt.ylabel('Average Health')
    plt.legend()
    plt.show()

def plot_average_resource_vs_age(dataframe):
    avg_resources = dataframe.groupby('age')['resource_level'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(avg_resources.index, avg_resources.values, label='Average Resource Level', color='green')
    plt.title('Average Resource Level vs. Age')
    plt.xlabel('Age')
    plt.ylabel('Resource Level')
    plt.legend()
    plt.show()

def plot_average_health_over_time(dataframe):
    # Calculate mean and std of health for each step
    health_stats = dataframe.groupby('step_number')['current_health'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot mean line
    plt.plot(health_stats['step_number'], health_stats['mean'], 
            label='Average Health', color='blue')
    
    # Plot std bands
    plt.fill_between(health_stats['step_number'],
                     health_stats['mean'] - health_stats['std'],
                     health_stats['mean'] + health_stats['std'],
                     alpha=0.2, color='blue',
                     label='±1 Standard Deviation')
    
    plt.title('Average Health Over Time (with Standard Deviation)')
    plt.xlabel('Step Number')
    plt.ylabel('Health')
    plt.legend()
    plt.show()

def plot_average_resource_over_time(dataframe):
    # Calculate mean and std of resource level for each step
    resource_stats = dataframe.groupby('step_number')['resource_level'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Plot mean line
    plt.plot(resource_stats['step_number'], resource_stats['mean'], 
            label='Average Resource Level', color='green')
    
    # Plot std bands
    plt.fill_between(resource_stats['step_number'],
                     resource_stats['mean'] - resource_stats['std'],
                     resource_stats['mean'] + resource_stats['std'],
                     alpha=0.2, color='green',
                     label='±1 Standard Deviation')
    
    plt.title('Average Resource Level Over Time (with Standard Deviation)')
    plt.xlabel('Step Number')
    plt.ylabel('Resource Level')
    plt.legend()
    plt.show()

# Load the dataset
def main(dataframe):
    try:
        # Call each function to analyze and visualize
        agent_id = 1  # Change to the agent_id you want to analyze

        print(f"Plotting health over time for agent {agent_id}...")
        plot_health_over_time(dataframe, agent_id)

        print(f"Plotting resource level over time for agent {agent_id}...")
        plot_resource_level_over_time(dataframe, agent_id)

        print(f"Plotting spatial trajectory for agent {agent_id}...")
        plot_spatial_trajectory(dataframe, agent_id)

        print("Plotting total reward distribution...")
        plot_total_reward_distribution(dataframe)

        print("Plotting defending agents over time...")
        plot_defending_agents_over_time(dataframe)

        print("Plotting average health vs. age...")
        plot_average_health_vs_age(dataframe)

        print("Plotting average resource level vs. age...")
        plot_average_resource_vs_age(dataframe)

        print("Plotting average health over time...")
        plot_average_health_over_time(dataframe)

        print("Plotting average resource level over time...")
        plot_average_resource_over_time(dataframe)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the analysis
if __name__ == "__main__":
    import pandas as pd
    from sqlalchemy import create_engine

    # connection_string = "sqlite:///simulations/simulation_20241110_122335.db"
    connection_string = "sqlite:///simulations/simulation.db"

    # Create engine
    engine = create_engine(connection_string)
    
    df = pd.read_sql("SELECT * FROM AgentStates", engine)

    main(df)
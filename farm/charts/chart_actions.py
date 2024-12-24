import pandas as pd
import matplotlib.pyplot as plt
import json

# Define the analysis functions

def plot_action_type_distribution(dataframe):
    """Plot the distribution of different action types."""
    action_counts = dataframe['action_type'].value_counts()
    plt.figure(figsize=(10, 6))
    action_counts.plot(kind='bar', color='skyblue', edgecolor='k')
    plt.title('Action Type Distribution')
    plt.xlabel('Action Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

def plot_rewards_by_action_type(dataframe):
    """Plot average rewards for each action type."""
    avg_rewards = dataframe.groupby('action_type')['reward'].mean()
    plt.figure(figsize=(10, 6))
    avg_rewards.plot(kind='bar', color='orange', edgecolor='k')
    plt.title('Average Rewards by Action Type')
    plt.xlabel('Action Type')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)
    plt.show()

def plot_resource_changes(dataframe):
    """Plot resource changes (before vs after) across actions."""
    dataframe['resource_change'] = dataframe['resources_after'] - dataframe['resources_before']
    plt.figure(figsize=(10, 6))
    plt.hist(dataframe['resource_change'], bins=30, edgecolor='k', alpha=0.7, color='purple')
    plt.title('Resource Change Distribution')
    plt.xlabel('Resource Change')
    plt.ylabel('Frequency')
    plt.show()

def plot_action_frequency_over_time(dataframe):
    """Plot the frequency of actions over time."""
    action_counts = dataframe.groupby('step_number').size()
    plt.figure(figsize=(10, 6))
    plt.plot(action_counts.index, action_counts.values, marker='o', label='Action Frequency')
    plt.title('Action Frequency Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Number of Actions')
    plt.legend()
    plt.show()

def plot_position_changes(dataframe, agent_id):
    """Plot the position changes for a specific agent."""
    agent_data = dataframe[dataframe['agent_id'] == agent_id]
    positions_before = agent_data['position_before'].apply(eval)  # Convert string to tuple
    positions_after = agent_data['position_after'].apply(eval)    # Convert string to tuple
    x_before, y_before = zip(*positions_before)
    x_after, y_after = zip(*positions_after)

    plt.figure(figsize=(10, 6))
    plt.scatter(x_before, y_before, label='Before', alpha=0.7, color='blue')
    plt.scatter(x_after, y_after, label='After', alpha=0.7, color='red')
    plt.plot(x_after, y_after, linestyle='--', alpha=0.5, color='gray')
    plt.title(f'Position Changes for Agent {agent_id}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.show()

def plot_rewards_over_time(dataframe):
    """Plot cumulative rewards over time."""
    rewards = dataframe.groupby('step_number')['reward'].sum().cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(rewards.index, rewards.values, marker='o', color='green', label='Cumulative Rewards')
    plt.title('Cumulative Rewards Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()

def plot_action_target_distribution(dataframe):
    """Plot the distribution of action targets."""
    target_counts = dataframe['action_target_id'].value_counts()
    plt.figure(figsize=(10, 6))
    target_counts.plot(kind='bar', color='coral', edgecolor='k')
    plt.title('Action Target Distribution')
    plt.xlabel('Target ID')
    plt.ylabel('Frequency')
    plt.show()

def analyze_action_details(dataframe):
    """Analyze and print details from the JSON-encoded 'details' field."""
    print("Action Details Analysis:")
    for index, row in dataframe.iterrows():
        if pd.notnull(row['details']):
            details = json.loads(row['details'])
            print(f"Action ID: {row['action_id']}, Details: {details}")

# Load the dataset
def main(dataframe):

    try:

        # Call each function to analyze and visualize
        agent_id = '1'  # Specify an agent ID to analyze

        print("Plotting action type distribution...")
        plot_action_type_distribution(dataframe)

        print("Plotting rewards by action type...")
        plot_rewards_by_action_type(dataframe)

        print("Plotting resource changes...")
        plot_resource_changes(dataframe)

        print("Plotting action frequency over time...")
        plot_action_frequency_over_time(dataframe)

        print(f"Plotting position changes for agent {agent_id}...")
        plot_position_changes(dataframe, agent_id)

        print("Plotting cumulative rewards over time...")
        plot_rewards_over_time(dataframe)

        print("Plotting action target distribution...")
        plot_action_target_distribution(dataframe)

        print("Analyzing action details...")
        analyze_action_details(dataframe)

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
    
    df = pd.read_sql("SELECT * FROM AgentActions", engine)

    main(df)
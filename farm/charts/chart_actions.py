import pandas as pd
import matplotlib.pyplot as plt
import json

# Define the analysis functions

def save_plot_to_file(plt, filename: str):
    """Save the current plot to a file."""
    plt.savefig(filename)
    plt.close()

def plot_action_type_distribution(dataframe):
    """Plot the distribution of different action types."""
    action_counts = dataframe['action_type'].value_counts()
    plt.figure(figsize=(10, 6))
    action_counts.plot(kind='bar', color='skyblue', edgecolor='k')
    plt.title('Action Type Distribution')
    plt.xlabel('Action Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    filename = "action_type_distribution.png"
    save_plot_to_file(plt, filename)
    return filename

def plot_rewards_by_action_type(dataframe):
    """Plot average rewards for each action type."""
    avg_rewards = dataframe.groupby('action_type')['reward'].mean()
    plt.figure(figsize=(10, 6))
    avg_rewards.plot(kind='bar', color='orange', edgecolor='k')
    plt.title('Average Rewards by Action Type')
    plt.xlabel('Action Type')
    plt.ylabel('Average Reward')
    plt.xticks(rotation=45)
    filename = "rewards_by_action_type.png"
    save_plot_to_file(plt, filename)
    return filename

def plot_resource_changes(dataframe):
    """Plot resource changes (before vs after) across actions."""
    dataframe['resource_change'] = dataframe['resources_after'] - dataframe['resources_before']
    plt.figure(figsize=(10, 6))
    plt.hist(dataframe['resource_change'], bins=30, edgecolor='k', alpha=0.7, color='purple')
    plt.title('Resource Change Distribution')
    plt.xlabel('Resource Change')
    plt.ylabel('Frequency')
    filename = "resource_changes.png"
    save_plot_to_file(plt, filename)
    return filename

def plot_action_frequency_over_time(dataframe):
    """Plot the frequency of actions over time as a stacked area chart."""
    # Group by both step number and action type to get counts
    action_counts = dataframe.groupby(['step_number', 'action_type']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(12, 6))
    plt.stackplot(action_counts.index, 
                 action_counts.T.values,
                 labels=action_counts.columns,
                 alpha=0.8)
    
    plt.title('Action Frequency Over Time by Action Type')
    plt.xlabel('Step Number')
    plt.ylabel('Number of Actions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Adjust layout to prevent legend cutoff
    filename = "action_frequency_over_time.png"
    save_plot_to_file(plt, filename)
    return filename

def plot_position_changes(dataframe, agent_id):
    """Plot the position changes for a specific agent."""
    # Get agent actions ordered by step number
    agent_actions = dataframe[dataframe['agent_id'] == agent_id].sort_values('step_number')
    
    # Extract positions from details
    try:
        positions = []
        for _, row in agent_actions.iterrows():
            if pd.notnull(row['details']):
                details = json.loads(row['details'])
                if 'target_position' in details:
                    positions.append(tuple(details['target_position']))
                elif 'distance_moved' in details:
                    # For move actions, we need to track the actual positions
                    # For now, we'll skip these since we don't have the absolute position
                    continue
        
        if not positions:
            print("No position data found in action details")
            return
            
        # Convert positions to arrays
        x_positions = [p[0] for p in positions]
        y_positions = [p[1] for p in positions]
        
        # Create before/after pairs
        x_before = x_positions[:-1]  # All positions except the last
        y_before = y_positions[:-1]  # All positions except the last
        x_after = x_positions[1:]    # All positions except the first
        y_after = y_positions[1:]    # All positions except the first

        plt.figure(figsize=(10, 6))
        plt.scatter(x_before, y_before, label='Target Positions', alpha=0.7, color='blue')
        plt.scatter(x_after, y_after, label='Next Target', alpha=0.7, color='red')
        plt.plot(x_positions, y_positions, linestyle='--', alpha=0.5, color='gray', label='Path')
        plt.title(f'Target Positions for Agent {agent_id}')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True, alpha=0.3)
        filename = f"position_changes_agent_{agent_id}.png"
        save_plot_to_file(plt, filename)
        return filename
        
    except Exception as e:
        print(f"Error processing position data: {e}")
        print("Please ensure the action details contain position information")

def plot_rewards_over_time(dataframe):
    """Plot cumulative rewards over time."""
    rewards = dataframe.groupby('step_number')['reward'].sum().cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(rewards.index, rewards.values, marker='o', color='green', label='Cumulative Rewards')
    plt.title('Cumulative Rewards Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    filename = "rewards_over_time.png"
    save_plot_to_file(plt, filename)
    return filename

def plot_action_target_distribution(dataframe):
    """Plot the distribution of action targets."""
    target_counts = dataframe['action_target_id'].value_counts()
    plt.figure(figsize=(10, 6))
    target_counts.plot(kind='bar', color='coral', edgecolor='k')
    plt.title('Action Target Distribution')
    plt.xlabel('Target ID')
    plt.ylabel('Frequency')
    filename = "action_target_distribution.png"
    save_plot_to_file(plt, filename)
    return filename

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
    
    df = pd.read_sql("SELECT * FROM agent_actions", engine)

    main(df)
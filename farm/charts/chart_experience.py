import pandas as pd
import matplotlib.pyplot as plt
import json

# Define the analysis functions

def plot_action_rewards(dataframe):
    """Plot rewards associated with actions."""
    plt.figure(figsize=(10, 6))
    plt.scatter(dataframe['action_taken'], dataframe['reward'], alpha=0.7, color='blue')
    plt.title('Rewards Associated with Actions')
    plt.xlabel('Action Taken')
    plt.ylabel('Reward')
    plt.show()

def plot_module_type_distribution(dataframe):
    """Plot the distribution of experiences by module type."""
    module_counts = dataframe['module_type'].value_counts()
    plt.figure(figsize=(10, 6))
    module_counts.plot(kind='bar', color='green', edgecolor='k')
    plt.title('Experience Distribution by Module Type')
    plt.xlabel('Module Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.show()

def plot_loss_over_time(dataframe):
    """Plot the average loss over time."""
    avg_loss = dataframe.groupby('step_number')['loss'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(avg_loss.index, avg_loss.values, marker='o', color='red', label='Average Loss')
    plt.title('Average Loss Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.show()

def plot_rewards_over_time(dataframe):
    """Plot cumulative rewards over time."""
    cumulative_rewards = dataframe.groupby('step_number')['reward'].sum().cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_rewards.index, cumulative_rewards.values, marker='o', color='purple', label='Cumulative Rewards')
    plt.title('Cumulative Rewards Over Time')
    plt.xlabel('Step Number')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.show()

def plot_state_transitions(dataframe, agent_id):
    """Visualize state transitions for a specific agent."""
    agent_data = dataframe[dataframe['agent_id'] == agent_id]
    states_before = agent_data['state_before'].apply(eval)  # Convert string to tuple/dict
    states_after = agent_data['state_after'].apply(eval)    # Convert string to tuple/dict

    plt.figure(figsize=(10, 6))
    plt.plot(states_before, label='State Before', linestyle='--', alpha=0.7)
    plt.plot(states_after, label='State After', linestyle='-', alpha=0.7)
    plt.title(f'State Transitions for Agent {agent_id}')
    plt.xlabel('Experience Index')
    plt.ylabel('State Value')
    plt.legend()
    plt.show()

def plot_loss_vs_rewards(dataframe):
    """Plot loss against rewards to examine correlation."""
    plt.figure(figsize=(10, 6))
    plt.scatter(dataframe['loss'], dataframe['reward'], alpha=0.7, color='orange')
    plt.title('Loss vs Rewards')
    plt.xlabel('Loss')
    plt.ylabel('Reward')
    plt.show()

def analyze_experience_details(dataframe):
    """Print details of specific experiences for debugging."""
    print("Experience Details Analysis:")
    for index, row in dataframe.iterrows():
        print(f"Experience ID: {row['experience_id']}, Module Type: {row['module_type']}, Action: {row['action_taken']}, Reward: {row['reward']}")

# Load the dataset
def main(dataframe):
    
    try:

        # Specify an agent ID for focused analysis
        agent_id = 1

        print("Plotting rewards associated with actions...")
        plot_action_rewards(dataframe)

        print("Plotting experience distribution by module type...")
        plot_module_type_distribution(dataframe)

        print("Plotting average loss over time...")
        plot_loss_over_time(dataframe)

        print("Plotting cumulative rewards over time...")
        plot_rewards_over_time(dataframe)

        print(f"Plotting state transitions for agent {agent_id}...")
        plot_state_transitions(dataframe, agent_id)

        print("Plotting loss vs rewards...")
        plot_loss_vs_rewards(dataframe)

        print("Analyzing experience details...")
        analyze_experience_details(dataframe)

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
    
    df = pd.read_sql("SELECT * FROM LearningExperiences", engine)

    main(df)

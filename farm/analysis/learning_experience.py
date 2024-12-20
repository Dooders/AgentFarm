import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from database.database import SimulationDatabase
from database.data_retrieval import DataRetriever

#! Table not populating currently


def analyze_learning_experiences(db_path: str):
    """Analyze learning experiences from the simulation database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file
    """
    # Initialize database and retriever
    db = SimulationDatabase(db_path)
    retriever = DataRetriever(db)

    try:
        # Get learning statistics using DataRetriever
        learning_stats = retriever.get_learning_statistics()
        
        # Convert to DataFrame for analysis
        learning_data = pd.DataFrame({
            'step': list(learning_stats['learning_progress']['average_reward'].keys()),
            'reward': list(learning_stats['learning_progress']['average_reward'].values()),
            'loss': list(learning_stats['learning_progress']['average_loss'].values())
        })

        # Module performance analysis
        module_performance = pd.DataFrame.from_dict(
            learning_stats['module_performance'], 
            orient='index'
        )

        # Create visualizations
        plot_learning_metrics(learning_data)

    finally:
        # Close database connection
        db.close()


def plot_learning_metrics(rl_data: pd.DataFrame):
    """Generate plots for learning metrics analysis.

    Parameters
    ----------
    rl_data : pd.DataFrame
        DataFrame containing learning experience data
    """
    # Check if the dataframe is empty
    if rl_data.empty:
        print("No learning experience data found in the database.")
        return

    # Plot 1: Reward vs. Loss
    plt.figure(figsize=(10, 6))
    plt.scatter(rl_data["reward"], rl_data["loss"], alpha=0.6)
    plt.title("Reward vs. Loss")
    plt.xlabel("Reward")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Plot 2: State Delta vs. Reward
    state_deltas = np.array(rl_data["state_delta"].tolist())
    if len(state_deltas) > 0:  # Only plot if we have data
        plt.figure(figsize=(10, 6))
        plt.scatter(
            [delta[0] for delta in state_deltas],  # Position X Delta
            rl_data["reward"],
            alpha=0.6,
            label="Position X Delta"
        )
        plt.scatter(
            [delta[1] for delta in state_deltas],  # Position Y Delta
            rl_data["reward"],
            alpha=0.6,
            label="Position Y Delta"
        )
        plt.title("State Change vs. Reward")
        plt.xlabel("State Change")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot 3: Reward Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rl_data["reward"], bins=50, alpha=0.7, color="blue", label="Rewards")
    plt.axvline(
        rl_data["reward"].mean(), color="red", linestyle="--", label="Mean Reward"
    )
    plt.title("Reward Distribution")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

    # Plot 4: Action Frequencies
    plt.figure(figsize=(10, 6))
    rl_data["action_taken"].value_counts().plot(kind="bar", color="purple", alpha=0.7)
    plt.title("Action Frequencies")
    plt.xlabel("Action")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Plot 5: Reward Over Time
    plt.figure(figsize=(10, 6))
    reward_time = rl_data.groupby("step_number")["reward"].mean()
    plt.plot(reward_time, label="Average Reward")
    plt.title("Reward Over Time")
    plt.xlabel("Simulation Step")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    db_path = "simulations/simulation.db"  # Replace with your database path
    analyze_learning_experiences(db_path)

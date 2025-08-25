import json

import matplotlib.pyplot as plt
import pandas as pd

# Define the analysis functions


def plot_action_type_distribution(dataframe):
    """Plot the distribution of different action types."""
    action_counts = dataframe["action_type"].value_counts()
    plt.figure(figsize=(10, 6))
    action_counts.plot(kind="bar", color="skyblue", edgecolor="k")
    plt.title("Action Type Distribution")
    plt.xlabel("Action Type")
    plt.ylabel("Frequency")
    plt.xticks(rotation=45)
    return plt


def plot_rewards_by_action_type(dataframe):
    """Plot average rewards for each action type."""
    avg_rewards = dataframe.groupby("action_type")["reward"].mean()
    plt.figure(figsize=(10, 6))
    avg_rewards.plot(kind="bar", color="orange", edgecolor="k")
    plt.title("Average Rewards by Action Type")
    plt.xlabel("Action Type")
    plt.ylabel("Average Reward")
    plt.xticks(rotation=45)
    return plt


def plot_resource_changes(dataframe):
    """Plot resource changes (before vs after) across actions."""
    dataframe["resource_change"] = (
        dataframe["resources_after"] - dataframe["resources_before"]
    )
    plt.figure(figsize=(10, 6))
    plt.hist(
        dataframe["resource_change"], bins=30, edgecolor="k", alpha=0.7, color="purple"
    )
    plt.title("Resource Change Distribution")
    plt.xlabel("Resource Change")
    plt.ylabel("Frequency")
    return plt


def plot_action_frequency_over_time(dataframe):
    """Plot the frequency of actions over time as a stacked area chart."""
    action_counts = (
        dataframe.groupby(["step_number", "action_type"]).size().unstack(fill_value=0)
    )

    plt.figure(figsize=(12, 6))
    plt.stackplot(
        action_counts.index,
        action_counts.T.values,
        labels=action_counts.columns,
        alpha=0.8,
    )

    plt.title("Action Frequency Over Time by Action Type")
    plt.xlabel("Step Number")
    plt.ylabel("Number of Actions")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return plt


def plot_position_changes(dataframe, agent_id):
    """Plot the position changes for a specific agent."""
    agent_actions = dataframe[dataframe["agent_id"] == agent_id].sort_values(
        "step_number"
    )

    try:
        positions = []
        for _, row in agent_actions.iterrows():
            if pd.notnull(row["details"]):
                details = json.loads(row["details"])
                if "target_position" in details:
                    positions.append(tuple(details["target_position"]))

        if not positions:
            print("No position data found in action details")
            return None

        x_positions = [p[0] for p in positions]
        y_positions = [p[1] for p in positions]
        x_before = x_positions[:-1]
        y_before = y_positions[:-1]
        x_after = x_positions[1:]
        y_after = y_positions[1:]

        plt.figure(figsize=(10, 6))
        plt.scatter(
            x_before, y_before, label="Target Positions", alpha=0.7, color="blue"
        )
        plt.scatter(x_after, y_after, label="Next Target", alpha=0.7, color="red")
        plt.plot(
            x_positions,
            y_positions,
            linestyle="--",
            alpha=0.5,
            color="gray",
            label="Path",
        )
        plt.title(f"Target Positions for Agent {agent_id}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt

    except Exception as e:
        print(f"Error processing position data: {e}")
        return None


def plot_rewards_over_time(dataframe):
    """Plot cumulative rewards over time."""
    rewards = dataframe.groupby("step_number")["reward"].sum().cumsum()
    plt.figure(figsize=(10, 6))
    plt.plot(
        rewards.index,
        rewards.values,
        marker="o",
        color="green",
        label="Cumulative Rewards",
    )
    plt.title("Cumulative Rewards Over Time")
    plt.xlabel("Step Number")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    return plt


def plot_action_target_distribution(dataframe):
    """Plot the distribution of action targets."""
    plt.figure(figsize=(10, 6))

    # Extract target information from the details column
    targets = []
    for _, row in dataframe.iterrows():
        if pd.notnull(row["details"]):
            try:
                details = json.loads(row["details"])
                if "target_id" in details:
                    targets.append(details["target_id"])
                elif "target_position" in details:
                    targets.append("position")
                else:
                    targets.append("no_target")
            except json.JSONDecodeError:
                targets.append("invalid_details")

    # Create distribution plot
    target_counts = pd.Series(targets).value_counts()
    plt.bar(range(len(target_counts)), target_counts.values.tolist(), alpha=0.7)
    plt.xticks(range(len(target_counts)), target_counts.index.tolist(), rotation=45)
    plt.title("Action Target Distribution")
    plt.xlabel("Target Type")
    plt.ylabel("Frequency")
    plt.tight_layout()
    return plt


# Load the dataset
def main(dataframe):
    try:
        # Dictionary of plot functions and their names
        plot_functions = {
            "action_type_distribution": plot_action_type_distribution,
            "rewards_by_action_type": plot_rewards_by_action_type,
            "resource_changes": plot_resource_changes,
            "action_frequency_over_time": plot_action_frequency_over_time,
            "rewards_over_time": plot_rewards_over_time,
            "action_target_distribution": plot_action_target_distribution,
        }

        # Generate and show each plot
        for name, func in plot_functions.items():
            print(f"Plotting {name}...")
            plt = func(dataframe)
            if plt is not None:
                plt.show()

        # Handle position changes separately since it needs an agent_id
        agent_id = "1"
        print(f"Plotting position changes for agent {agent_id}...")
        plt = plot_position_changes(dataframe, agent_id)
        if plt is not None:
            plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


# Run the analysis
if __name__ == "__main__":
    import pandas as pd
    from sqlalchemy import create_engine

    connection_string = "sqlite:///simulations/simulation.db"
    engine = create_engine(connection_string)
    df = pd.read_sql("SELECT * FROM agent_actions", engine)
    main(df)

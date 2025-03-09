"""
#! SIMULATION LEVEL ANALYSIS
Objective
The purpose of Action Type Distribution Analysis is to investigate the relationship
between the types of actions agents perform and their overall success or behavior
in the simulation. It helps to uncover:

Which actions are most or least common.
Whether certain actions contribute more significantly to agent success (e.g., longer lifespans, higher rewards).
Patterns or biases in agent behavior related to the action types.
Steps Involved
Frequency Analysis

Count the frequency of each action type to identify which actions dominate the simulation.
Use this to understand the balance of behaviors in the simulation (e.g., gathering vs. attacking).
Correlation Analysis

Investigate whether there is a statistical relationship between action frequencies and other metrics like lifespan or reward.
For example, do agents that perform "gather" more frequently live longer or earn higher rewards?
Chi-Square Test

Conduct a Chi-Square test to evaluate whether action type and agent success are independent or related.
Example: Determine if agents that "attack" are more likely to receive higher rewards than agents that "gather."
Visualization

Create bar charts or histograms to visualize the distribution of action types.
Highlight any disparities or trends in the data.

Why It's Important
This analysis provides critical insights into agent behaviors:

Helps to optimize the simulation by adjusting action rewards or probabilities.
Reveals dominant or underutilized strategies, guiding potential improvements in agent design or environment rules.
Identifies correlations that could indicate emergent behavior or unintended consequences in the simulation.
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency


def calculate_action_frequencies(actions_df):
    """
    Calculate frequencies of each action type.
    :param actions_df: DataFrame with columns ['action_type']
    :return: Series with action frequencies
    """
    return actions_df["action_type"].value_counts()


def calculate_action_correlations(actions_df):
    """
    Calculate correlation between action frequencies and agent success metrics.
    :param actions_df: DataFrame with columns ['agent_id', 'action_type', 'reward']
    :return: DataFrame with correlation analysis
    """
    # Group by agent_id and action_type to get action frequencies per agent
    action_frequencies = pd.crosstab(actions_df["agent_id"], actions_df["action_type"])

    # Calculate average reward per agent
    agent_rewards = actions_df.groupby("agent_id")["reward"].mean()

    # Combine frequencies with rewards
    combined_data = pd.concat([action_frequencies, agent_rewards], axis=1)

    # Calculate correlations between action frequencies and rewards
    correlations = combined_data.corr()["reward"].drop("reward")

    return correlations


def chi_square_test(actions_df):
    """
    Perform Chi-Square test between action type and success (high/low reward).
    :param actions_df: DataFrame with columns ['action_type', 'reward']
    :return: Dict with Chi-Square test results
    """
    # Create binary success indicator based on median reward
    median_reward = actions_df["reward"].median()
    high_reward = actions_df["reward"] > median_reward

    contingency_table = pd.crosstab(actions_df["action_type"], high_reward)
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    return {
        "chi2": chi2,
        "p_value": p,
        "degrees_of_freedom": dof,
        "contingency_table": contingency_table,
    }


def plot_action_distribution(actions_df):
    """
    Plot the frequency of each action type with reward information.
    :param actions_df: DataFrame with columns ['action_type', 'reward']
    """
    # Create a figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Action frequency
    action_counts = calculate_action_frequencies(actions_df)
    action_counts.plot(kind="bar", ax=ax1, color="skyblue", edgecolor="black")
    # Add value labels on bars
    for i, v in enumerate(action_counts):
        ax1.text(i, v, str(v), ha="center", va="bottom")
    ax1.set_title("Action Type Distribution")
    ax1.set_xlabel("Action Type")
    ax1.set_ylabel("Frequency")

    # Plot 2: Average reward by action type
    avg_rewards = actions_df.groupby("action_type")["reward"].mean()
    colors = ["red" if x < 0 else "lightgreen" for x in avg_rewards]
    avg_rewards.plot(kind="bar", ax=ax2, color=colors, edgecolor="black")
    # Add value labels on bars
    for i, v in enumerate(avg_rewards):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top")
    ax2.set_title("Average Reward by Action Type")
    ax2.set_xlabel("Action Type")
    ax2.set_ylabel("Average Reward")

    plt.tight_layout()
    plt.show()


def action_type_distribution(engine):
    # Load data from AgentActions table
    query = """
    SELECT action_type, reward, agent_id
    FROM agent_actions
    WHERE reward IS NOT NULL
    """
    actions_df = pd.read_sql(query, engine)

    # Calculate and display basic statistics
    print("\nAction Frequencies:")
    print(calculate_action_frequencies(actions_df))

    print("\nAction-Reward Correlations:")
    print(calculate_action_correlations(actions_df))

    print("\nChi-Square Test Results:")
    chi_square_results = chi_square_test(actions_df)
    print(f"Chi-Square Statistic: {chi_square_results['chi2']:.2f}")
    print(f"P-value: {chi_square_results['p_value']:.4f}")
    print("\nContingency Table:")
    print(chi_square_results["contingency_table"])

    # Plot distributions
    plot_action_distribution(actions_df)


if __name__ == "__main__":
    from sqlalchemy import create_engine

    connection_string = "sqlite:///simulations/simulation.db"

    # Create engine
    engine = create_engine(connection_string)
    action_type_distribution(engine)

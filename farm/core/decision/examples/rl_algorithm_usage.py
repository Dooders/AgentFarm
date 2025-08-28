"""Example usage of RL algorithms in AgentFarm.

This example demonstrates how to use the newly integrated Stable Baselines
algorithms (PPO, SAC, A2C, TD3) alongside traditional algorithms for
enhanced RL capabilities.
"""

from pathlib import Path

import numpy as np
import torch

from farm.core.decision.algorithms import (
    A2CWrapper,
    AlgorithmBenchmark,
    AlgorithmComparison,
    PPOWrapper,
    SACWrapper,
    TD3Wrapper,
)
from farm.core.decision.config import DecisionConfig
from farm.core.decision.decision import DecisionModule


def example_basic_rl_usage():
    """Demonstrate basic usage of RL algorithms."""
    print("=== Basic RL Algorithm Usage ===")

    # Configuration for RL algorithms
    config = DecisionConfig(
        algorithm_type="ppo",  # Can be 'ppo', 'sac', 'a2c', 'td3'
        rl_state_dim=8,
        rl_buffer_size=1000,
        rl_batch_size=32,
        rl_train_freq=4,
        algorithm_params={
            "learning_rate": 3e-4,
            "n_steps": 128,  # For PPO
        },
    )

    # Create mock agent
    from farm.core.agent import BaseAgent

    mock_agent = BaseAgent.__new__(BaseAgent)
    mock_agent.agent_id = "test"

    # Create SelectModule with PPO
    select_module = DecisionModule(agent=mock_agent, config=config)

    # Create a sample state
    state = torch.randn(8)

    # Select action (this will use PPO internally)
    # Note: In practice, you'd pass actual agent and actions objects
    print(f"SelectModule created with algorithm: {config.algorithm_type}")

    # Store some example experiences
    for i in range(10):
        action = np.random.randint(0, 6)
        reward = np.random.normal(0, 1)
        next_state = torch.randn(8)
        done = i == 9

        select_module.update(state, action, reward, next_state, done)

    print("Stored 10 experiences for training")


def example_algorithm_comparison():
    """Demonstrate algorithm comparison and benchmarking."""
    print("\n=== Algorithm Comparison ===")

    # Define algorithms to compare
    algorithms = [
        ("ppo", {"learning_rate": 3e-4}),
        ("sac", {"learning_rate": 3e-4}),
        ("a2c", {"learning_rate": 7e-4}),
        ("td3", {"learning_rate": 1e-3}),
    ]

    # Create benchmark
    benchmark = AlgorithmBenchmark(
        algorithms=algorithms,
        num_actions=6,
        state_dim=8,
        num_episodes=50,  # Small number for demonstration
        max_steps_per_episode=200,
        save_path=Path("benchmark_results"),
    )

    # Run benchmark
    print("Running benchmark comparison...")
    results = benchmark.run_benchmark()

    # Compare results
    comparison_df = AlgorithmComparison.compare_results(results)
    print("\nBenchmark Results:")
    print(comparison_df)

    # Find best algorithm
    best_algo, best_score = AlgorithmComparison.find_best_algorithm(results)
    print(f"\nBest algorithm: {best_algo} (score: {best_score:.3f})")

    # Create comparison plots
    try:
        AlgorithmComparison.plot_comparison(
            results, save_path=Path("benchmark_results")
        )
        print("Comparison plots saved to benchmark_results/")
    except ImportError:
        print("Matplotlib/seaborn not available - skipping plots")


def example_configuration_options():
    """Demonstrate different configuration options."""
    print("\n=== Configuration Examples ===")

    configs = [
        {
            "name": "PPO with custom learning rate",
            "config": DecisionConfig(
                algorithm_type="ppo",
                algorithm_params={"learning_rate": 1e-3, "n_steps": 256},
            ),
        },
        {
            "name": "SAC with entropy tuning",
            "config": DecisionConfig(
                algorithm_type="sac",
                algorithm_params={"ent_coef": "auto", "learning_starts": 100},
            ),
        },
        {
            "name": "TD3 with policy delay",
            "config": DecisionConfig(
                algorithm_type="td3",
                algorithm_params={"policy_delay": 3, "target_policy_noise": 0.3},
            ),
        },
    ]

    for example in configs:
        print(f"\n{example['name']}:")
        print(f"  Algorithm: {example['config'].algorithm_type}")
        print(f"  Parameters: {example['config'].algorithm_params}")


def example_standalone_rl_algorithms():
    """Demonstrate using RL algorithms directly."""
    print("\n=== Standalone RL Algorithm Usage ===")

    # Create PPO algorithm directly
    ppo = PPOWrapper(
        num_actions=6,
        state_dim=8,
        algorithm_kwargs={"learning_rate": 3e-4, "n_steps": 128, "batch_size": 32},
    )

    # Create sample state
    state = np.random.randn(8)

    # Select actions
    actions = []
    for i in range(5):
        action = ppo.select_action(state)
        actions.append(action)
        print(f"Step {i+1}: Selected action {action}")

    # Store some experiences
    for i in range(10):
        action = np.random.randint(0, 6)
        reward = np.random.normal(0, 1)
        next_state = np.random.randn(8)
        done = i == 9

        ppo.store_experience(state, action, reward, next_state, done)
        ppo.update_step_count()

        if ppo.should_train():
            batch = ppo.replay_buffer.sample(min(32, len(ppo.replay_buffer)))
            metrics = ppo.train_on_batch(batch)
            print(f"Training metrics at step {ppo.step_count}: {metrics}")

    print(f"Total experiences stored: {len(ppo.replay_buffer)}")


if __name__ == "__main__":
    print("AgentFarm RL Algorithms Integration Examples")
    print("=" * 50)

    try:
        example_basic_rl_usage()
        example_configuration_options()
        example_standalone_rl_algorithms()

        # Only run comparison if user wants (can be computationally intensive)
        run_comparison = (
            input("\nRun algorithm comparison benchmark? (y/n): ").lower().strip()
        )
        if run_comparison == "y":
            example_algorithm_comparison()
        else:
            print("Skipping benchmark comparison")

        print("\n=== Examples completed successfully! ===")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure Stable Baselines dependencies are installed:")
        print("  pip install stable-baselines3 gymnasium shimmy")

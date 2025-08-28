"""Example usage of traditional ML algorithms in AgentFarm.

This example demonstrates how to use the implemented traditional machine learning
algorithms (MLP, SVM, Random Forest, Gradient Boosting, Naive Bayes, KNN)
alongside reinforcement learning algorithms for enhanced decision making.
"""

from pathlib import Path

import numpy as np
import torch

from farm.actions.algorithms import (
    AlgorithmBenchmark,
    AlgorithmComparison,
    GradientBoostActionSelector,
    KNNActionSelector,
    MLPActionSelector,
    NaiveBayesActionSelector,
    RandomForestActionSelector,
    SVMActionSelector,
)
from farm.actions.config import SelectConfig
from farm.actions.select import SelectModule
from farm.actions.training import AlgorithmTrainer, ExperienceCollector


def example_basic_ml_usage():
    """Demonstrate basic usage of ML algorithms."""
    print("=== Basic ML Algorithm Usage ===")

    # Example configurations for different ML algorithms
    configs = [
        {
            "name": "MLP Neural Network",
            "config": SelectConfig(
                algorithm_type="mlp",
                algorithm_params={
                    "hidden_layer_sizes": (64, 32),
                    "max_iter": 1000,
                    "random_state": 42,
                },
            ),
        },
        {
            "name": "Support Vector Machine",
            "config": SelectConfig(
                algorithm_type="svm",
                algorithm_params={"kernel": "rbf", "C": 1.0, "random_state": 42},
            ),
        },
        {
            "name": "Random Forest",
            "config": SelectConfig(
                algorithm_type="random_forest",
                algorithm_params={"n_estimators": 100, "random_state": 42},
            ),
        },
        {
            "name": "Gradient Boosting",
            "config": SelectConfig(
                algorithm_type="gradient_boost",
                algorithm_params={
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "random_state": 42,
                },
            ),
        },
        {
            "name": "Naive Bayes",
            "config": SelectConfig(algorithm_type="naive_bayes", algorithm_params={}),
        },
        {
            "name": "K-Nearest Neighbors",
            "config": SelectConfig(
                algorithm_type="knn", algorithm_params={"n_neighbors": 5}
            ),
        },
    ]

    # Test each algorithm
    for example in configs:
        print(f"\n{example['name']}:")
        try:
            select_module = SelectModule(num_actions=6, config=example["config"])
            print(
                f"  ✅ Successfully created SelectModule with {example['config'].algorithm_type}"
            )

            # Test with sample data
            sample_state = torch.randn(8)

            # Create minimal mock objects
            from farm.core.action import Action
            from farm.core.agent import BaseAgent

            mock_agent = BaseAgent.__new__(BaseAgent)
            mock_agent.agent_id = "test"
            mock_actions = [
                Action(f"action_{i}", 1.0, lambda a, **kwargs: None) for i in range(6)
            ]

            action = select_module.select_action(mock_agent, mock_actions, sample_state)
            print(f"  ✅ Selected action: {action}")

        except Exception as e:
            print(f"  ❌ Error: {e}")


def example_ml_training():
    """Demonstrate ML algorithm training."""
    print("\n=== ML Algorithm Training ===")

    # Create training data
    collector = ExperienceCollector()
    trainer = AlgorithmTrainer()

    # Create sample training data (normally this would come from actual gameplay)
    np.random.seed(42)
    training_data = []

    for i in range(100):
        # Generate random state features
        state = np.random.randn(8)
        # Generate random action (0-5)
        action = np.random.randint(0, 6)
        # Generate random reward
        reward = np.random.normal(0, 1)

        training_data.append((state, action, reward))

    # Train different algorithms
    algorithms = [
        ("mlp", MLPActionSelector(num_actions=6)),
        ("svm", SVMActionSelector(num_actions=6)),
        ("random_forest", RandomForestActionSelector(num_actions=6)),
        ("knn", KNNActionSelector(num_actions=6)),
        ("naive_bayes", NaiveBayesActionSelector(num_actions=6)),
    ]

    print(f"Training on {len(training_data)} samples...")

    for name, algorithm in algorithms:
        try:
            trainer.train_algorithm(algorithm, training_data)
            print(f"  ✅ Trained {name} successfully")

            # Test prediction
            test_state = np.random.randn(8)
            action = algorithm.select_action(test_state)
            probs = algorithm.predict_proba(test_state)
            print(f"    Selected action: {action}, Confidence: {probs[action]:.3f}")

        except Exception as e:
            print(f"  ❌ Error training {name}: {e}")


def example_algorithm_comparison():
    """Demonstrate ML algorithm comparison and benchmarking."""
    print("\n=== ML Algorithm Comparison ===")

    # Define algorithms to compare (focus on traditional ML)
    algorithms = [
        ("mlp", {"hidden_layer_sizes": (32, 16), "max_iter": 500}),
        ("svm", {"kernel": "rbf", "C": 1.0}),
        ("random_forest", {"n_estimators": 50}),
        ("knn", {"n_neighbors": 5}),
        ("naive_bayes", {}),
    ]

    # Create benchmark
    benchmark = AlgorithmBenchmark(
        algorithms=algorithms,
        num_actions=6,
        state_dim=8,
        num_episodes=20,  # Smaller number for demonstration
        max_steps_per_episode=100,
        save_path=Path("ml_benchmark_results"),
    )

    # Run benchmark
    print("Running ML algorithm comparison...")
    results = benchmark.run_benchmark()

    # Compare results
    comparison_df = AlgorithmComparison.compare_results(results)
    print("\nML Benchmark Results:")
    print(comparison_df)

    # Find best algorithm
    best_algo, best_score = AlgorithmComparison.find_best_algorithm(results)
    print(f"\nBest ML algorithm: {best_algo} (score: {best_score:.3f})")

    # Create comparison plots
    try:
        AlgorithmComparison.plot_comparison(
            results, save_path=Path("ml_benchmark_results")
        )
        print("Comparison plots saved to ml_benchmark_results/")
    except ImportError:
        print("Matplotlib/seaborn not available - skipping plots")


def example_feature_engineering():
    """Demonstrate feature engineering capabilities."""
    print("\n=== Feature Engineering ===")

    from farm.actions.feature_engineering import FeatureEngineer

    # Create a mock agent and environment for demonstration
    from farm.core.agent import BaseAgent
    from farm.core.environment import Environment

    class MockAgent(BaseAgent):
        def __init__(self):
            self.current_health = 80.0
            self.starting_health = 100.0
            self.resource_level = 15.0
            self.position = (50, 50)
            self.starvation_threshold = 10.0
            self.max_starvation = 50.0
            self.is_defending = False
            self.config = None

    class MockEnvironment(Environment):
        def __init__(self):
            super().__init__(width=100, height=100, resource_distribution={})
            self.time = 100
            self.resources = [1, 2, 3, 4, 5]  # 5 resources
            self.agents = [1, 2, 3]  # 3 agents

        def get_nearby_resources(self, position, radius):
            return [1, 2]  # 2 nearby resources

        def get_nearby_agents(self, position, radius):
            return [1]  # 1 nearby agent

    # Create feature engineer
    feature_engineer = FeatureEngineer()
    agent = MockAgent()
    environment = MockEnvironment()

    # Extract features
    features = feature_engineer.extract_features(agent, environment)

    print(f"Extracted {len(features)} features:")
    feature_names = [
        "health_ratio",
        "resource_ratio",
        "pos_x_norm",
        "pos_y_norm",
        "resource_density",
        "starvation_ratio",
        "agent_density",
        "is_defending",
        "time_norm",
    ]

    for name, value in zip(feature_names, features):
        print(f"{name}: {value:.4f}")


def example_hybrid_approach():
    """Demonstrate hybrid ML + RL approach."""
    print("\n=== Hybrid ML + RL Approach ===")

    # Create configurations that combine ML with RL
    hybrid_configs = [
        {"name": "DQN (existing RL)", "config": SelectConfig(algorithm_type="dqn")},
        {
            "name": "PPO (new RL)",
            "config": SelectConfig(
                algorithm_type="ppo",
                rl_state_dim=8,
                algorithm_params={"learning_rate": 3e-4},
            ),
        },
        {
            "name": "Random Forest (ML)",
            "config": SelectConfig(
                algorithm_type="random_forest", algorithm_params={"n_estimators": 100}
            ),
        },
    ]

    print("Available hybrid approaches:")
    for config in hybrid_configs:
        print(f"  • {config['name']}: {config['config'].algorithm_type}")


def main():
    """Run all ML algorithm examples."""
    print("AgentFarm Traditional ML Algorithms Integration Examples")
    print("=" * 60)

    try:
        example_basic_ml_usage()
        example_feature_engineering()
        example_ml_training()

        # Only run comparison if user wants (can be computationally intensive)
        run_comparison = (
            input("\nRun ML algorithm comparison benchmark? (y/n): ").lower().strip()
        )
        if run_comparison == "y":
            example_algorithm_comparison()
        else:
            print("Skipping benchmark comparison")

        example_hybrid_approach()

        print("\n=== All examples completed successfully! ===")
        print("\nSummary of available ML algorithms:")
        print("  ✅ Multi-Layer Perceptron (MLP)")
        print("  ✅ Support Vector Machine (SVM)")
        print("  ✅ Random Forest")
        print("  ✅ Gradient Boosting (XGBoost/LightGBM)")
        print("  ✅ Naive Bayes")
        print("  ✅ K-Nearest Neighbors (KNN)")
        print("  ✅ Integration with RL algorithms (PPO, SAC, A2C, TD3)")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure scikit-learn and other ML dependencies are installed:")
        print("  pip install scikit-learn xgboost lightgbm")


if __name__ == "__main__":
    main()

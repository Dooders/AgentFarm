"""Algorithm comparison and benchmarking utilities for RL algorithms.

This module provides tools for comparing different reinforcement learning algorithms,
benchmarking their performance, and tracking training metrics over time.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .base import ActionAlgorithm, AlgorithmRegistry
from .rl_base import RLAlgorithm

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    algorithm_name: str
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    training_metrics: Dict[str, List[float]] = field(default_factory=dict)
    training_time: float = 0.0
    total_steps: int = 0
    final_score: float = 0.0
    convergence_step: Optional[int] = None
    # Computed fields (can be set during deserialization)
    mean_reward: float = field(default=0.0)
    std_reward: float = field(default=0.0)
    max_reward: float = field(default=0.0)

    def __post_init__(self):
        """Compute derived fields after initialization."""
        if self.episode_rewards:
            self.mean_reward = float(np.mean(self.episode_rewards))
            self.std_reward = float(np.std(self.episode_rewards))
            self.max_reward = float(max(self.episode_rewards))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Ensure computed fields are up to date
        if self.episode_rewards:
            self.mean_reward = float(np.mean(self.episode_rewards))
            self.std_reward = float(np.std(self.episode_rewards))
            self.max_reward = float(max(self.episode_rewards))
        else:
            self.mean_reward = 0.0
            self.std_reward = 0.0
            self.max_reward = 0.0

        base_data = {
            "algorithm_name": self.algorithm_name,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_metrics": self.training_metrics,
            "training_time": self.training_time,
            "total_steps": self.total_steps,
            "final_score": self.final_score,
            "convergence_step": self.convergence_step,
        }

        # Always include computed fields
        base_data.update(
            {
                "mean_reward": self.mean_reward,
                "std_reward": self.std_reward,
                "max_reward": self.max_reward,
            }
        )

        return base_data


class AlgorithmBenchmark:
    """Benchmark different algorithms on a common task."""

    def __init__(
        self,
        algorithms: List[Tuple[str, Dict[str, Any]]],
        num_actions: int,
        state_dim: int = 8,
        num_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        save_path: Optional[Path] = None,
    ):
        """Initialize the benchmark.

        Args:
            algorithms: List of (algorithm_name, kwargs) tuples
            num_actions: Number of possible actions
            state_dim: State dimension for RL algorithms
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode
            save_path: Path to save results
        """
        self.algorithms = algorithms
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.save_path = Path(save_path) if save_path else None

        self.results: Dict[str, BenchmarkResult] = {}

    def create_algorithm(
        self, name: str, kwargs: Dict[str, Any], seed: Optional[int] = None
    ) -> ActionAlgorithm:
        """Create an algorithm instance."""
        kwargs = kwargs.copy()

        if name in ["ppo", "sac", "a2c", "dqn", "ddpg"]:
            # RL algorithms need state_dim
            kwargs.setdefault("state_dim", self.state_dim)

        # Always ensure num_actions is in kwargs
        kwargs.setdefault("num_actions", self.num_actions)

        # Set random_state if seed is provided and algorithm supports it
        if seed is not None and name not in ["ppo", "sac", "a2c", "dqn", "ddpg"]:
            kwargs.setdefault("random_state", seed)

        return AlgorithmRegistry.create(name, **kwargs)

    def run_single_algorithm(
        self,
        algorithm_name: str,
        algorithm_kwargs: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> BenchmarkResult:
        """Run benchmark for a single algorithm.

        Args:
            algorithm_name: Name of the algorithm to test
            algorithm_kwargs: Algorithm-specific parameters
            seed: Random seed for reproducibility

        Returns:
            BenchmarkResult containing performance metrics
        """
        logger.info(f"Running benchmark for {algorithm_name}")

        if seed is not None:
            np.random.seed(seed)

        # Create algorithm
        algorithm = self.create_algorithm(algorithm_name, algorithm_kwargs, seed)
        result = BenchmarkResult(algorithm_name)

        start_time = time.time()

        # Run episodes
        for episode in range(self.num_episodes):
            episode_reward = 0.0
            episode_steps = 0

            # Reset episode state (simplified)
            state = np.random.randn(self.state_dim)

            for step in range(self.max_steps_per_episode):
                # Select action
                action = algorithm.select_action(state)

                # Simulate environment step (simplified)
                next_state = state + 0.1 * np.random.randn(self.state_dim)
                reward = np.random.normal(0, 1)  # Random reward for demonstration
                done = step >= self.max_steps_per_episode - 1

                episode_reward += reward
                episode_steps += 1

                # Store experience if RL algorithm
                if isinstance(algorithm, RLAlgorithm):
                    algorithm.store_experience(state, action, reward, next_state, done)
                    algorithm.update_step_count()

                    # Train if needed
                    if algorithm.should_train():
                        batch = algorithm.replay_buffer.sample(
                            min(32, len(algorithm.replay_buffer))
                        )
                        metrics = algorithm.train_on_batch(batch)

                        # Record training metrics
                        for key, value in metrics.items():
                            if key not in result.training_metrics:
                                result.training_metrics[key] = []
                            result.training_metrics[key].append(value)

                # Check termination
                if done:
                    break

                state = next_state

            # Record episode results
            result.episode_rewards.append(episode_reward)
            result.episode_lengths.append(episode_steps)
            result.total_steps += episode_steps

        result.training_time = time.time() - start_time
        result.final_score = float(
            np.mean(result.episode_rewards[-10:])
        )  # Average of last 10 episodes

        logger.info(
            f"Completed {algorithm_name} benchmark with final score: {result.final_score:.2f}"
        )

        return result

    def run_benchmark(
        self, seeds: Optional[List[int]] = None
    ) -> Dict[str, BenchmarkResult]:
        """Run benchmark for all algorithms.

        Args:
            seeds: List of random seeds for reproducibility

        Returns:
            Dictionary mapping algorithm names to results
        """
        if seeds is None:
            seeds = [42] * len(self.algorithms)

        for i, ((algorithm_name, algorithm_kwargs), seed) in enumerate(
            zip(self.algorithms, seeds)
        ):
            result = self.run_single_algorithm(algorithm_name, algorithm_kwargs, seed)
            self.results[algorithm_name] = result

        # Save results if path provided
        if self.save_path:
            self.save_results()

        return self.results

    def save_results(self, path: Optional[Path] = None) -> None:
        """Save benchmark results to disk.

        Args:
            path: Save path (uses self.save_path if None)
        """
        save_path = path or self.save_path
        if not save_path:
            return

        save_path.mkdir(parents=True, exist_ok=True)

        # Save individual results
        for algorithm_name, result in self.results.items():
            result_path = save_path / f"{algorithm_name}_results.json"
            with open(result_path, "w") as f:
                json.dump(result.to_dict(), f, indent=2, default=str)

        # Save summary
        summary_path = save_path / "benchmark_summary.json"
        summary = {
            algorithm_name: result.to_dict()
            for algorithm_name, result in self.results.items()
        }

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def load_results(self, path: Path) -> Dict[str, BenchmarkResult]:
        """Load benchmark results from disk.

        Args:
            path: Path to results directory

        Returns:
            Dictionary mapping algorithm names to results
        """
        results = {}

        # Load individual results
        for result_file in path.glob("*_results.json"):
            algorithm_name = result_file.stem.replace("_results", "")
            with open(result_file, "r") as f:
                data = json.load(f)
                results[algorithm_name] = BenchmarkResult(**data)

        self.results = results
        return results


class AlgorithmComparison:
    """Compare performance of different algorithms."""

    @staticmethod
    def compare_results(results: Dict[str, BenchmarkResult]) -> pd.DataFrame:
        """Compare algorithm results in a DataFrame.

        Args:
            results: Dictionary of benchmark results

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []

        for algorithm_name, result in results.items():
            data = result.to_dict()
            comparison_data.append(
                {
                    "Algorithm": algorithm_name,
                    "Mean Reward": data["mean_reward"],
                    "Std Reward": data["std_reward"],
                    "Max Reward": data["max_reward"],
                    "Training Time (s)": data["training_time"],
                    "Total Steps": data["total_steps"],
                    "Final Score": data["final_score"],
                    "Episodes": len(data["episode_rewards"]),
                }
            )

        return pd.DataFrame(comparison_data)

    @staticmethod
    def find_best_algorithm(
        results: Dict[str, BenchmarkResult], metric: str = "mean_reward"
    ) -> Tuple[Optional[str], float]:
        """Find the best performing algorithm for a given metric.

        Args:
            results: Dictionary of benchmark results
            metric: Metric to optimize ('mean_reward', 'final_score', etc.)

        Returns:
            Tuple of (algorithm_name, metric_value) or (None, best_value) if no results
        """
        best_algorithm = None
        best_value = float("-inf")

        for algorithm_name, result in results.items():
            data = result.to_dict()
            value = data.get(metric, 0.0)

            if value > best_value:
                best_value = value
                best_algorithm = algorithm_name

        return best_algorithm, best_value

    @staticmethod
    def plot_comparison(
        results: Dict[str, BenchmarkResult], save_path: Optional[Path] = None
    ) -> None:
        """Create comparison plots (requires matplotlib).

        Args:
            results: Dictionary of benchmark results
            save_path: Path to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            sns.set_style("whitegrid")

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle("Algorithm Comparison", fontsize=16)

            # Plot 1: Mean rewards over episodes
            ax1 = axes[0, 0]
            for algorithm_name, result in results.items():
                rewards = result.episode_rewards
                ax1.plot(rewards, label=algorithm_name, alpha=0.7)
            ax1.set_xlabel("Episode")
            ax1.set_ylabel("Reward")
            ax1.set_title("Reward Progression")
            ax1.legend()

            # Plot 2: Training time comparison
            ax2 = axes[0, 1]
            algorithms = list(results.keys())
            times = [result.training_time for result in results.values()]
            ax2.bar(algorithms, times)
            ax2.set_ylabel("Training Time (s)")
            ax2.set_title("Training Time Comparison")
            plt.xticks(rotation=45)

            # Plot 3: Final scores
            ax3 = axes[1, 0]
            scores = [result.final_score for result in results.values()]
            ax3.bar(algorithms, scores)
            ax3.set_ylabel("Final Score")
            ax3.set_title("Final Performance")
            plt.xticks(rotation=45)

            # Plot 4: Episode lengths
            ax4 = axes[1, 1]
            for algorithm_name, result in results.items():
                lengths = result.episode_lengths
                ax4.plot(lengths, label=algorithm_name, alpha=0.7)
            ax4.set_xlabel("Episode")
            ax4.set_ylabel("Length")
            ax4.set_title("Episode Lengths")
            ax4.legend()

            plt.tight_layout()

            if save_path:
                plt.savefig(
                    save_path / "algorithm_comparison.png", dpi=300, bbox_inches="tight"
                )
            else:
                plt.show()

        except ImportError:
            logger.warning(
                "matplotlib and/or seaborn not available. Skipping plot generation."
            )

    @staticmethod
    def statistical_test(
        results1: BenchmarkResult,
        results2: BenchmarkResult,
        metric: str = "episode_rewards",
    ) -> Dict[str, Union[float, str]]:
        """Perform statistical comparison between two algorithms.

        Args:
            results1: Results from first algorithm
            results2: Results from second algorithm
            metric: Metric to compare

        Returns:
            Dictionary with statistical test results
        """
        data1 = getattr(results1, metric, [])
        data2 = getattr(results2, metric, [])

        if not data1 or not data2:
            return {"error": "Insufficient data for comparison"}

        # Perform t-test
        try:
            from scipy import stats

            t_stat, p_value = stats.ttest_ind(data1, data2)

            return {
                "t_statistic": float(t_stat),  # type: ignore
                "p_value": float(p_value),  # type: ignore
                "significant": float(p_value) < 0.05,  # type: ignore
                "mean_diff": float(np.mean(data1) - np.mean(data2)),
                "std1": float(np.std(data1)),
                "std2": float(np.std(data2)),
            }
        except ImportError:
            # Fallback to simple comparison
            mean1, mean2 = np.mean(data1), np.mean(data2)
            return {
                "mean1": float(mean1),
                "mean2": float(mean2),
                "mean_diff": float(mean1 - mean2),
                "std1": float(np.std(data1)),
                "std2": float(np.std(data2)),
                "note": "Install scipy for proper statistical testing",
            }

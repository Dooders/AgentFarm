import unittest
from typing import cast
from unittest.mock import Mock, patch

import numpy as np

from farm.core.decision.algorithms import (
    GradientBoostActionSelector,
    KNNActionSelector,
    MLPActionSelector,
    NaiveBayesActionSelector,
    RandomForestActionSelector,
    SVMActionSelector,
)
from farm.core.decision.algorithms.base import ActionAlgorithm, AlgorithmRegistry
from farm.core.decision.algorithms.benchmark import (
    AlgorithmBenchmark,
    AlgorithmComparison,
)


class TestAlgorithmRegistryIntegration(unittest.TestCase):
    """Integration tests for algorithm registry functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Store original registry
        self.original_algorithms = AlgorithmRegistry._algorithms.copy()

    def tearDown(self):
        """Restore original registry."""
        AlgorithmRegistry._algorithms = self.original_algorithms

    def test_registry_contains_expected_algorithms(self):
        """Test that registry contains all expected algorithms."""
        expected_algorithms = {
            "mlp",
            "svm",
            "random_forest",
            "naive_bayes",
            "knn",
            "gradient_boost",
            "ppo",
            "sac",
            "a2c",
            "ddpg",
            "dqn",
        }

        registered_algorithms = set(AlgorithmRegistry._algorithms.keys())
        self.assertTrue(expected_algorithms.issubset(registered_algorithms))

    def test_create_all_standard_algorithms(self):
        """Test creating instances of all standard algorithms."""
        num_actions = 4
        test_state = np.array([0.5, 0.3, 0.8, 0.1])

        algorithms_to_test = ["mlp", "svm", "random_forest", "naive_bayes", "knn"]

        for algo_name in algorithms_to_test:
            with self.subTest(algorithm=algo_name):
                try:
                    algo = AlgorithmRegistry.create(algo_name, num_actions=num_actions)

                    # Test basic functionality
                    self.assertEqual(algo.num_actions, num_actions)

                    # Test action selection
                    action = algo.select_action(test_state)
                    self.assertIsInstance(action, int)
                    self.assertTrue(0 <= action < num_actions)

                    # Test probability prediction
                    probs = algo.predict_proba(test_state)
                    self.assertEqual(len(probs), num_actions)
                    self.assertAlmostEqual(np.sum(probs), 1.0, places=6)

                except Exception as e:
                    self.fail(f"Failed to create or use {algo_name}: {e}")

    def test_create_gradient_boost_algorithms(self):
        """Test creating gradient boost algorithms with different backends."""
        num_actions = 3

        # Check if we have at least one gradient boosting library available
        has_xgboost = False
        has_lightgbm = False

        try:
            import xgboost  # type: ignore

            has_xgboost = True
        except ImportError:
            pass

        try:
            import lightgbm  # type: ignore

            has_lightgbm = True
        except ImportError:
            pass

        if not has_xgboost and not has_lightgbm:
            self.skipTest(
                "Neither xgboost nor lightgbm is available. Install one to test gradient boosting."
            )

        # Test that we can create the algorithm
        try:
            algo = AlgorithmRegistry.create("gradient_boost", num_actions=num_actions)
            self.assertIsInstance(algo, GradientBoostActionSelector)

            # Check that the correct backend was selected
            gb_algo = cast(GradientBoostActionSelector, algo)
            if has_xgboost:
                self.assertEqual(gb_algo._backend, "xgboost")
            elif has_lightgbm:
                self.assertEqual(gb_algo._backend, "lightgbm")

        except ImportError as e:
            self.fail(f"Failed to create gradient boost algorithm: {e}")

    def test_algorithm_training_and_prediction_cycle(self):
        """Test full training and prediction cycle for different algorithms."""
        num_actions = 3
        num_samples = 20

        # Generate synthetic training data
        states = np.random.randn(num_samples, 4)
        actions = np.random.randint(0, num_actions, num_samples)
        rewards = np.random.randn(num_samples)

        test_algorithms = ["mlp", "svm", "random_forest", "naive_bayes", "knn"]

        for algo_name in test_algorithms:
            with self.subTest(algorithm=algo_name):
                algo = AlgorithmRegistry.create(algo_name, num_actions=num_actions)

                # Train algorithm
                algo.train(states, actions, rewards)

                # Test prediction on new data
                test_state = np.array([0.1, 0.2, 0.3, 0.4])
                action = algo.select_action(test_state)
                probs = algo.predict_proba(test_state)

                self.assertIsInstance(action, int)
                self.assertTrue(0 <= action < num_actions)
                self.assertEqual(len(probs), num_actions)
                self.assertAlmostEqual(np.sum(probs), 1.0, places=6)

    def test_algorithm_model_persistence(self):
        """Test model saving and loading for algorithms."""
        import os
        import tempfile

        num_actions = 3
        algo = AlgorithmRegistry.create("mlp", num_actions=num_actions)

        # Train briefly
        states = np.random.randn(10, 4)
        actions = np.random.randint(0, num_actions, 10)
        algo.train(states, actions)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp_path = tmp.name
                # Save model
                algo.save_model(tmp.name)

            # Load model (outside the context manager so file is closed)
            loaded_algo = AlgorithmRegistry.create("mlp", num_actions=num_actions)
            loaded_algo = ActionAlgorithm.load_model(tmp_path)

            # Test that loaded model works
            test_state = np.array([0.1, 0.2, 0.3, 0.4])
            original_probs = algo.predict_proba(test_state)
            loaded_probs = loaded_algo.predict_proba(test_state)

            # Probabilities should be very similar
            np.testing.assert_array_almost_equal(
                original_probs, loaded_probs, decimal=5
            )
            self.assertTrue(0 <= loaded_algo.select_action(test_state) < num_actions)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                # Force garbage collection to ensure file handles are released
                import gc

                gc.collect()
                os.unlink(tmp_path)

    def test_algorithm_comparison_integration(self):
        """Test algorithm comparison functionality."""
        algorithms = [
            ("mlp", {"hidden_layer_sizes": (8,), "max_iter": 50}),
            ("random_forest", {"n_estimators": 10, "random_state": 42}),
        ]

        num_actions = 3

        with patch("time.time", side_effect=[0.0, 1.0, 2.0, 3.0]):
            benchmark = AlgorithmBenchmark(
                algorithms=algorithms,
                num_actions=num_actions,
                num_episodes=2,
                max_steps_per_episode=3,
            )

            results = benchmark.run_benchmark()

        # Check results
        self.assertEqual(len(results), 2)
        self.assertIn("mlp", results)
        self.assertIn("random_forest", results)

        # Test comparison
        df = AlgorithmComparison.compare_results(results)
        self.assertEqual(len(df), 2)
        self.assertIn("Algorithm", df.columns)

        # Test best algorithm finding
        best_name, best_value = AlgorithmComparison.find_best_algorithm(
            results, "mean_reward"
        )
        self.assertIn(best_name, ["mlp", "random_forest"])


class TestAlgorithmBenchmarkingIntegration(unittest.TestCase):
    """Integration tests for algorithm benchmarking."""

    def test_benchmark_multiple_algorithms(self):
        """Test benchmarking multiple algorithms together."""
        algorithms = [
            ("mlp", {"hidden_layer_sizes": (4,), "max_iter": 20}),
            ("svm", {"C": 1.0, "random_state": 42}),
            ("random_forest", {"n_estimators": 5, "random_state": 42}),
        ]

        num_actions = 3

        benchmark = AlgorithmBenchmark(
            algorithms=algorithms,
            num_actions=num_actions,
            num_episodes=2,
            max_steps_per_episode=4,
            state_dim=4,
        )

        # Mock time to avoid delays
        with patch("time.time", side_effect=range(10)):
            results = benchmark.run_benchmark()

        self.assertEqual(len(results), 3)

        # All algorithms should have results
        for algo_name in ["mlp", "svm", "random_forest"]:
            self.assertIn(algo_name, results)
            result = results[algo_name]
            self.assertGreater(result.total_steps, 0)
            self.assertEqual(len(result.episode_rewards), 2)
            self.assertEqual(len(result.episode_lengths), 2)

    def test_benchmark_with_different_seeds(self):
        """Test benchmarking with different random seeds."""
        algorithms = [("mlp", {})]  # Don't specify random_state, let benchmark set it

        benchmark = AlgorithmBenchmark(
            algorithms=algorithms,
            num_actions=3,
            num_episodes=2,
            max_steps_per_episode=3,
        )

        # Run with different seeds - these should produce different results
        # because the algorithm will use different random states
        with patch("time.time", side_effect=range(10)):
            results1 = benchmark.run_benchmark(seeds=[42])

        with patch("time.time", side_effect=range(10, 20)):
            results2 = benchmark.run_benchmark(seeds=[123])

        # Results should be different due to different seeds affecting algorithm initialization
        # and environment randomness
        rewards1 = results1["mlp"].episode_rewards
        rewards2 = results2["mlp"].episode_rewards

        # Test that algorithms with different seeds behave differently
        # by creating fresh algorithms with explicit seeds
        test_state = np.array([0.1, 0.2, 0.3, 0.4])

        algo1 = AlgorithmRegistry.create("mlp", num_actions=3, random_state=42)
        algo2 = AlgorithmRegistry.create("mlp", num_actions=3, random_state=123)

        # Train with same data
        states = np.random.RandomState(0).randn(20, 4)
        actions = np.random.RandomState(0).randint(0, 3, 20)
        algo1.train(states, actions)
        algo2.train(states, actions)

        # Check that probability distributions are different
        probs1 = algo1.predict_proba(test_state)
        probs2 = algo2.predict_proba(test_state)

        # With different random states, the models should be different
        probs_differ = not np.allclose(probs1, probs2, atol=1e-10)
        self.assertTrue(
            probs_differ,
            f"Algorithms with different seeds should produce different results: {probs1} vs {probs2}",
        )


class TestAlgorithmErrorHandling(unittest.TestCase):
    """Test error handling in algorithm operations."""

    def test_invalid_algorithm_creation(self):
        """Test creating invalid algorithms."""
        with self.assertRaises(ValueError):
            AlgorithmRegistry.create("nonexistent_algorithm", num_actions=3)

    def test_algorithm_with_wrong_parameters(self):
        """Test algorithm creation with wrong parameters."""
        # This should work since algorithms should handle extra parameters gracefully
        algo = AlgorithmRegistry.create(
            "mlp", num_actions=3, invalid_param="value", another_invalid_param=123
        )
        self.assertIsInstance(algo, MLPActionSelector)

    def test_algorithm_with_insufficient_data(self):
        """Test algorithms with insufficient training data."""
        algo = AlgorithmRegistry.create("mlp", num_actions=3)

        # Try to train with no data
        with self.assertRaises(Exception):  # Should raise some error
            algo.train(np.array([]), np.array([]))

    def test_probability_prediction_unfitted_algorithm(self):
        """Test probability prediction for unfitted algorithms."""
        algo = AlgorithmRegistry.create("mlp", num_actions=4)
        state = np.array([0.1, 0.2, 0.3, 0.4])

        # Should return uniform distribution
        probs = algo.predict_proba(state)
        expected_prob = 1.0 / 4
        np.testing.assert_array_almost_equal(probs, [expected_prob] * 4)


class TestAlgorithmPerformanceCharacteristics(unittest.TestCase):
    """Test performance characteristics of different algorithms."""

    def test_algorithm_consistency(self):
        """Test that algorithms produce consistent results with same input."""
        num_actions = 3
        state = np.array([0.5, 0.3, 0.8, 0.1])

        # Test deterministic algorithms
        deterministic_algos = ["svm", "random_forest", "naive_bayes", "knn"]

        for algo_name in deterministic_algos:
            with self.subTest(algorithm=algo_name):
                # Create two instances with same parameters
                algo1 = AlgorithmRegistry.create(
                    algo_name, num_actions=num_actions, random_state=42
                )
                algo2 = AlgorithmRegistry.create(
                    algo_name, num_actions=num_actions, random_state=42
                )

                # Train both with same data
                states = np.random.RandomState(42).randn(20, 4)
                actions = np.random.RandomState(42).randint(0, num_actions, 20)

                algo1.train(states, actions)
                algo2.train(states, actions)

                # They should produce same probability distributions
                probs1 = algo1.predict_proba(state)
                probs2 = algo2.predict_proba(state)

                np.testing.assert_array_almost_equal(
                    probs1,
                    probs2,
                    decimal=10,
                    err_msg=f"{algo_name} produced inconsistent probability distributions",
                )

                # The most probable action should also be the same
                action1 = np.argmax(probs1)
                action2 = np.argmax(probs2)
                self.assertEqual(
                    action1,
                    action2,
                    f"{algo_name} produced inconsistent most probable actions",
                )

    def test_algorithm_training_speed(self):
        """Test that algorithms train within reasonable time."""
        import time

        num_actions = 3
        states = np.random.randn(50, 4)
        actions = np.random.randint(0, num_actions, 50)

        algorithms = ["mlp", "svm", "random_forest", "naive_bayes", "knn"]

        for algo_name in algorithms:
            with self.subTest(algorithm=algo_name):
                algo = AlgorithmRegistry.create(algo_name, num_actions=num_actions)

                start_time = time.time()
                algo.train(states, actions)
                training_time = time.time() - start_time

                # Should train reasonably quickly (less than 1 second for small data)
                self.assertLess(
                    training_time,
                    1.0,
                    f"{algo_name} training took too long: {training_time}s",
                )


if __name__ == "__main__":
    unittest.main()

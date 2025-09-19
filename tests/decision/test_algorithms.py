import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

# Import all algorithm classes
from farm.core.decision.algorithms.base import ActionAlgorithm, AlgorithmRegistry
from farm.core.decision.algorithms.benchmark import (
    AlgorithmBenchmark,
    AlgorithmComparison,
    BenchmarkResult,
)
from farm.core.decision.algorithms.ensemble import (
    GradientBoostActionSelector,
    KNNActionSelector,
    NaiveBayesActionSelector,
    RandomForestActionSelector,
)
from farm.core.decision.algorithms.mlp import MLPActionSelector
from farm.core.decision.algorithms.svm import SVMActionSelector


class TestActionAlgorithm(unittest.TestCase):
    """Test the abstract base ActionAlgorithm class."""

    def test_abstract_methods(self):
        """Test that ActionAlgorithm is properly abstract."""
        from abc import ABCMeta

        # Check that it's an abstract class
        self.assertIsInstance(ActionAlgorithm, ABCMeta)
        self.assertTrue(hasattr(ActionAlgorithm, '__abstractmethods__'))

        # Check that required abstract methods are present
        abstract_methods = ActionAlgorithm.__abstractmethods__
        expected_methods = {'select_action', 'train', 'predict_proba'}
        self.assertTrue(expected_methods.issubset(abstract_methods))

        # Verify that all required methods are abstract
        for method_name in expected_methods:
            method = getattr(ActionAlgorithm, method_name)
            # Check if method has __isabstractmethod__ attribute
            self.assertTrue(getattr(method, '__isabstractmethod__', False),
                          f"Method {method_name} should be abstract")

    def test_save_load_model(self):
        """Test model serialization functionality."""
        # Use MLPActionSelector for testing since it's serializable
        algo = MLPActionSelector(num_actions=3, max_iter=10)

        # Use a unique filename to avoid conflicts
        import uuid

        temp_filename = f"test_model_{uuid.uuid4().hex}.pkl"

        try:
            # Test save
            algo.save_model(temp_filename)
            self.assertTrue(os.path.exists(temp_filename))

            # Test load
            loaded_algo = ActionAlgorithm.load_model(temp_filename)
            self.assertIsInstance(loaded_algo, MLPActionSelector)
            self.assertEqual(loaded_algo.num_actions, 3)

        finally:
            # Clean up with retry for Windows
            try:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            except OSError:
                pass  # Ignore cleanup errors on Windows


class TestAlgorithmRegistry(unittest.TestCase):
    """Test the AlgorithmRegistry functionality."""

    def setUp(self):
        """Reset registry before each test."""
        # Store original registry and restore after test
        self.original_algorithms = AlgorithmRegistry._algorithms.copy()

    def tearDown(self):
        """Restore original registry."""
        AlgorithmRegistry._algorithms = self.original_algorithms

    def test_register_algorithm(self):
        """Test registering a new algorithm."""
        AlgorithmRegistry.register("test_algo", "test.module:TestClass")

        self.assertIn("test_algo", AlgorithmRegistry._algorithms)
        self.assertEqual(
            AlgorithmRegistry._algorithms["test_algo"], "test.module:TestClass"
        )

    def test_create_algorithm_success(self):
        """Test successfully creating an algorithm."""

        class MockAlgorithm(ActionAlgorithm):
            def __init__(self, num_actions, **kwargs):
                super().__init__(num_actions=num_actions, **kwargs)
                self.test_param = kwargs.get("test_param", "default")

            def select_action(self, state):
                return 0

            def train(self, states, actions, rewards=None):
                pass

            def predict_proba(self, state):
                return np.full(self.num_actions, 1.0 / self.num_actions)

        # Mock the import_module and getattr
        with patch("farm.core.decision.algorithms.base.import_module") as mock_import:
            mock_module = Mock()
            mock_module.MockAlgorithm = MockAlgorithm
            mock_import.return_value = mock_module

            # Temporarily add to registry
            AlgorithmRegistry._algorithms["mock"] = "mock.module:MockAlgorithm"

            try:
                algo = AlgorithmRegistry.create(
                    "mock", num_actions=3, test_param="value"
                )
                self.assertIsInstance(algo, MockAlgorithm)
                self.assertEqual(algo.num_actions, 3)
                self.assertEqual(algo.test_param, "value")  # type: ignore
            finally:
                # Clean up
                if "mock" in AlgorithmRegistry._algorithms:
                    del AlgorithmRegistry._algorithms["mock"]

    def test_create_algorithm_unknown(self):
        """Test creating an unknown algorithm raises ValueError."""
        with self.assertRaises(ValueError) as context:
            AlgorithmRegistry.create("nonexistent_algorithm", num_actions=3)

        self.assertIn(
            "Unknown algorithm 'nonexistent_algorithm'", str(context.exception)
        )

    def test_create_algorithm_invalid_path(self):
        """Test creating algorithm with invalid module path."""
        AlgorithmRegistry._algorithms["invalid"] = "invalid_path"

        with self.assertRaises(ValueError) as context:
            AlgorithmRegistry.create("invalid", num_actions=3)

        self.assertIn("Invalid registry entry", str(context.exception))


class TestMLPActionSelector(unittest.TestCase):
    """Test MLP-based action selector."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_actions = 4
        self.state_dim = 6
        self.algo = MLPActionSelector(
            num_actions=self.num_actions,
            hidden_layer_sizes=(8,),
            random_state=42,
            max_iter=100,
        )

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.algo.num_actions, self.num_actions)
        self.assertFalse(self.algo._fitted)

    def test_unfitted_returns_uniform(self):
        """Test that unfitted algorithm returns uniform probabilities."""
        state = np.zeros(self.state_dim)
        proba = self.algo.predict_proba(state)

        self.assertEqual(len(proba), self.num_actions)
        self.assertAlmostEqual(float(np.sum(proba, out=None)), 1.0, places=6)
        expected_prob = 1.0 / self.num_actions
        np.testing.assert_array_almost_equal(proba, [expected_prob] * self.num_actions)

    def test_train_and_select(self):
        """Test training and action selection."""
        # Simple synthetic dataset mapping quadrant to action
        X = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # class 0
                [0.0, 1.0, 0.0, 0.0],  # class 1
                [0.0, 0.0, 1.0, 0.0],  # class 2
                [0.0, 0.0, 0.0, 1.0],  # class 3
            ]
        )
        y = np.array([0, 1, 2, 3])

        self.algo.train(X, y)
        self.assertTrue(self.algo._fitted)

        test_state = np.array([1.0, 0.0, 0.0, 0.0])
        proba = self.algo.predict_proba(test_state)

        self.assertEqual(len(proba), self.num_actions)
        self.assertAlmostEqual(float(np.sum(proba, out=None)), 1.0, places=6)

        action = self.algo.select_action(test_state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.num_actions)

    def test_predict_proba_with_rewards(self):
        """Test that rewards parameter is ignored in MLP training."""
        X = np.random.randn(10, 4)
        y = np.random.randint(0, self.num_actions, 10)
        rewards = np.random.randn(10)

        # Should not raise an error
        self.algo.train(X, y, rewards)
        self.assertTrue(self.algo._fitted)


class TestRandomForestActionSelector(unittest.TestCase):
    """Test Random Forest action selector."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_actions = 3
        self.algo = RandomForestActionSelector(
            num_actions=self.num_actions, n_estimators=10, random_state=0
        )

    def test_unfitted_returns_uniform(self):
        """Test that unfitted algorithm returns uniform probabilities."""
        state = np.ones(5)
        proba = self.algo.predict_proba(state)

        self.assertEqual(len(proba), self.num_actions)
        self.assertAlmostEqual(float(np.sum(proba, out=None)), 1.0, places=6)

    def test_train_and_predict(self):
        """Test training and prediction."""
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        y = np.array([0, 1, 2])

        self.algo.train(X, y)
        self.assertTrue(self.algo._fitted)

        proba = self.algo.predict_proba(np.array([1.0, 0.0]))
        self.assertEqual(len(proba), self.num_actions)
        self.assertAlmostEqual(float(np.sum(proba, out=None)), 1.0, places=6)

        action = self.algo.select_action(np.array([1.0, 0.0]))
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.num_actions)

    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        algo = RandomForestActionSelector(
            num_actions=2, n_estimators=50, max_depth=5, random_state=42
        )
        self.assertEqual(algo.num_actions, 2)


class TestNaiveBayesActionSelector(unittest.TestCase):
    """Test Naive Bayes action selector."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_actions = 3
        self.algo = NaiveBayesActionSelector(num_actions=self.num_actions)

    def test_unfitted_returns_uniform(self):
        """Test that unfitted algorithm returns uniform probabilities."""
        state = np.ones(4)
        proba = self.algo.predict_proba(state)

        self.assertEqual(len(proba), self.num_actions)
        self.assertAlmostEqual(float(np.sum(proba, out=None)), 1.0, places=6)

    def test_train_and_predict(self):
        """Test training and prediction."""
        X = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        y = np.array([0, 1, 2])

        self.algo.train(X, y)
        self.assertTrue(self.algo._fitted)

        proba = self.algo.predict_proba(np.array([1.0, 0.0]))
        self.assertEqual(len(proba), self.num_actions)
        self.assertAlmostEqual(float(np.sum(proba)), 1.0, places=6)


class TestKNNActionSelector(unittest.TestCase):
    """Test KNN action selector."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_actions = 3
        self.algo = KNNActionSelector(num_actions=self.num_actions, n_neighbors=3)

    def test_custom_neighbors(self):
        """Test initialization with custom n_neighbors."""
        algo = KNNActionSelector(num_actions=4, n_neighbors=5)
        self.assertEqual(algo.num_actions, 4)

    def test_train_and_predict(self):
        """Test training and prediction."""
        X = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        y = np.array([0, 1, 2, 0])

        self.algo.train(X, y)
        self.assertTrue(self.algo._fitted)

        proba = self.algo.predict_proba(np.array([0.5, 0.5]))
        self.assertEqual(len(proba), self.num_actions)
        self.assertAlmostEqual(float(np.sum(proba, out=None)), 1.0, places=6)


class TestGradientBoostActionSelector(unittest.TestCase):
    """Test Gradient Boosting action selector."""

    def test_xgboost_backend(self):
        """Test XGBoost backend initialization."""
        # Skip this test if xgboost is not available
        try:
            import xgboost

            algo = GradientBoostActionSelector(num_actions=3, random_state=42)
            self.assertEqual(algo._backend, "xgboost")
            self.assertEqual(algo.num_actions, 3)
        except ImportError:
            self.skipTest("XGBoost not available")

    def test_lightgbm_backend_fallback(self):
        """Test LightGBM backend fallback when XGBoost is unavailable."""
        # This test would require mocking at the import level, skip for now
        self.skipTest("Complex import mocking required")

    def test_no_backend_available(self):
        """Test error when neither XGBoost nor LightGBM is available."""
        # This test would require mocking at the import level, skip for now
        self.skipTest("Complex import mocking required")


class TestSVMActionSelector(unittest.TestCase):
    """Test SVM action selector."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_actions = 3
        self.algo = SVMActionSelector(num_actions=self.num_actions, random_state=42)

    def test_initialization(self):
        """Test proper initialization."""
        self.assertEqual(self.algo.num_actions, self.num_actions)
        self.assertFalse(self.algo._fitted)

    def test_unfitted_returns_uniform(self):
        """Test that unfitted algorithm returns uniform probabilities."""
        state = np.ones(4)
        proba = self.algo.predict_proba(state)

        self.assertEqual(len(proba), self.num_actions)
        self.assertAlmostEqual(float(np.sum(proba)), 1.0, places=6)

    def test_train_and_predict(self):
        """Test training and prediction."""
        X = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        y = np.array([0, 1, 2])

        self.algo.train(X, y)
        self.assertTrue(self.algo._fitted)

        proba = self.algo.predict_proba(np.array([1.0, 0.0]))
        self.assertEqual(len(proba), self.num_actions)
        self.assertAlmostEqual(float(np.sum(proba, out=None)), 1.0, places=6)

    def test_custom_kernel(self):
        """Test SVM with different kernel."""
        algo = SVMActionSelector(
            num_actions=2, kernel="rbf", C=2.0, gamma="scale", random_state=42
        )
        self.assertEqual(algo.num_actions, 2)


class TestBenchmarkResult(unittest.TestCase):
    """Test BenchmarkResult data structure."""

    def test_initialization(self):
        """Test BenchmarkResult initialization."""
        result = BenchmarkResult("test_algorithm")

        self.assertEqual(result.algorithm_name, "test_algorithm")
        self.assertEqual(result.episode_rewards, [])
        self.assertEqual(result.episode_lengths, [])
        self.assertEqual(result.training_metrics, {})
        self.assertEqual(result.training_time, 0.0)
        self.assertEqual(result.total_steps, 0)
        self.assertEqual(result.final_score, 0.0)
        self.assertIsNone(result.convergence_step)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = BenchmarkResult("test_algorithm")
        result.episode_rewards = [1.0, 2.0, 3.0]
        result.episode_lengths = [10, 20, 30]
        result.training_time = 5.5
        result.total_steps = 60
        result.final_score = 2.5

        data = result.to_dict()

        self.assertEqual(data["algorithm_name"], "test_algorithm")
        self.assertEqual(data["episode_rewards"], [1.0, 2.0, 3.0])
        self.assertEqual(data["mean_reward"], 2.0)
        # Population standard deviation: sqrt(((1-2)^2 + (2-2)^2 + (3-2)^2)/3) = sqrt(2/3) ≈ 0.8165
        self.assertAlmostEqual(data["std_reward"], 0.816496580927726, places=6)
        self.assertEqual(data["max_reward"], 3.0)
        self.assertEqual(data["training_time"], 5.5)
        self.assertEqual(data["total_steps"], 60)
        self.assertEqual(data["final_score"], 2.5)


class TestAlgorithmBenchmark(unittest.TestCase):
    """Test AlgorithmBenchmark class."""

    def setUp(self):
        """Set up test fixtures."""
        self.algorithms = [
            ("mlp", {"hidden_layer_sizes": (4,)}),
            ("random_forest", {"n_estimators": 5}),
        ]
        self.num_actions = 3
        self.num_episodes = 2
        self.max_steps = 5

        self.benchmark = AlgorithmBenchmark(
            algorithms=self.algorithms,
            num_actions=self.num_actions,
            num_episodes=self.num_episodes,
            max_steps_per_episode=self.max_steps,
        )

    def test_initialization(self):
        """Test AlgorithmBenchmark initialization."""
        self.assertEqual(len(self.benchmark.algorithms), 2)
        self.assertEqual(self.benchmark.num_actions, self.num_actions)
        self.assertEqual(self.benchmark.num_episodes, self.num_episodes)
        self.assertEqual(self.benchmark.max_steps_per_episode, self.max_steps)
        self.assertEqual(self.benchmark.results, {})

    def test_create_algorithm_standard(self):
        """Test creating standard algorithms."""
        algo = self.benchmark.create_algorithm("mlp", {})
        self.assertEqual(algo.num_actions, self.num_actions)
        self.assertIsInstance(algo, MLPActionSelector)

    def test_create_algorithm_rl(self):
        """Test creating RL algorithms with proper 2D observation shapes."""
        # PPO algorithm expects 2D spatial observations, so we provide observation_shape
        # instead of just state_dim. For a simple 4-element observation, we can use (1, 2, 2)
        # which represents 1 channel with 2x2 spatial dimensions
        benchmark = AlgorithmBenchmark(
            algorithms=[("ppo", {"observation_shape": (1, 2, 2)})],
            num_actions=self.num_actions,
            state_dim=4,  # Total flattened size should match observation_shape product
        )

        # Add debugging to check if observation_shape is properly set
        try:
            algo = benchmark.create_algorithm("ppo", {"observation_shape": (1, 2, 2)})
            print(
                f"Algorithm created successfully. Observation shape: {getattr(algo, 'observation_shape', 'NOT_SET')}"
            )
            self.assertEqual(algo.num_actions, self.num_actions)
        except Exception as e:
            print(f"Error creating algorithm: {e}")
            print(f"Benchmark state_dim: {benchmark.state_dim}")
            raise

    @patch("time.time")
    def test_run_single_algorithm(self, mock_time):
        """Test running benchmark for a single algorithm."""
        mock_time.side_effect = [0.0, 1.5]  # Start and end times

        result = self.benchmark.run_single_algorithm(
            "mlp", {"hidden_layer_sizes": (4,), "max_iter": 10}, seed=42
        )

        self.assertEqual(result.algorithm_name, "mlp")
        self.assertEqual(len(result.episode_rewards), self.num_episodes)
        self.assertEqual(len(result.episode_lengths), self.num_episodes)
        self.assertEqual(result.training_time, 1.5)
        self.assertGreater(result.total_steps, 0)

    def test_run_benchmark(self):
        """Test running full benchmark."""
        with patch("time.time", side_effect=[0.0, 1.0, 2.0, 3.0]):
            results = self.benchmark.run_benchmark(seeds=[42, 43])

        self.assertEqual(len(results), 2)
        self.assertIn("mlp", results)
        self.assertIn("random_forest", results)

    def test_save_and_load_results(self):
        """Test saving and loading benchmark results."""
        # Create mock results
        result = BenchmarkResult("test_algo")
        result.episode_rewards = [1.0, 2.0]
        result.training_time = 1.0
        self.benchmark.results["test_algo"] = result

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            self.benchmark.save_path = save_path
            self.benchmark.save_results()

            # Check files were created
            self.assertTrue((save_path / "test_algo_results.json").exists())
            self.assertTrue((save_path / "benchmark_summary.json").exists())

            # Load results
            loaded_results = self.benchmark.load_results(save_path)
            self.assertIn("test_algo", loaded_results)
            self.assertEqual(loaded_results["test_algo"].algorithm_name, "test_algo")


class TestAlgorithmComparison(unittest.TestCase):
    """Test AlgorithmComparison utility class."""

    def setUp(self):
        """Set up test fixtures."""
        self.results = {
            "algo1": BenchmarkResult("algo1"),
            "algo2": BenchmarkResult("algo2"),
        }

        # Populate results with data
        self.results["algo1"].episode_rewards = [1.0, 2.0, 3.0]
        self.results["algo1"].training_time = 1.0
        self.results["algo1"].total_steps = 100

        self.results["algo2"].episode_rewards = [2.0, 3.0, 4.0]
        self.results["algo2"].training_time = 2.0
        self.results["algo2"].total_steps = 150

    def test_compare_results(self):
        """Test comparing algorithm results."""
        df = AlgorithmComparison.compare_results(self.results)

        self.assertEqual(len(df), 2)
        self.assertIn("Algorithm", df.columns)
        self.assertIn("Mean Reward", df.columns)
        self.assertIn("Training Time (s)", df.columns)

        # Check algo1 data
        algo1_row = df[df["Algorithm"] == "algo1"].iloc[0]
        self.assertAlmostEqual(algo1_row["Mean Reward"], 2.0)
        self.assertEqual(algo1_row["Training Time (s)"], 1.0)

    def test_find_best_algorithm(self):
        """Test finding the best algorithm by metric."""
        best_name, best_value = AlgorithmComparison.find_best_algorithm(
            self.results, "mean_reward"
        )

        self.assertEqual(best_name, "algo2")  # algo2 has higher mean reward
        self.assertAlmostEqual(best_value, 3.0)

    def test_find_best_algorithm_empty(self):
        """Test finding best algorithm with empty results."""
        best_name, best_value = AlgorithmComparison.find_best_algorithm(
            {}, "mean_reward"
        )

        self.assertIsNone(best_name)
        self.assertEqual(best_value, float("-inf"))

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.tight_layout")
    @patch("matplotlib.pyplot.xticks")
    @patch("seaborn.set_style")
    @patch("matplotlib.pyplot.subplots")
    def test_plot_comparison(
        self,
        mock_subplots,
        mock_set_style,
        mock_xticks,
        mock_tight_layout,
        mock_show,
        mock_savefig,
    ):
        """Test plotting comparison (mocked)."""
        # Create a mock axes array that supports tuple indexing
        mock_axes = Mock()
        mock_ax = Mock()
        # Make axes support tuple indexing like a 2x2 array
        mock_axes.__getitem__ = Mock(return_value=mock_ax)
        mock_fig = Mock()
        mock_subplots.return_value = (mock_fig, mock_axes)

        # Test without save path
        AlgorithmComparison.plot_comparison(self.results)
        mock_show.assert_called_once()

        # Test with save path
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            AlgorithmComparison.plot_comparison(self.results, save_path)
            mock_savefig.assert_called_once()

    def test_plot_comparison_no_matplotlib(self):
        """Test plotting when matplotlib is not available."""
        # Mock the import of matplotlib itself
        with patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None}):
            with patch("farm.core.decision.algorithms.benchmark.logger") as mock_logger:
                AlgorithmComparison.plot_comparison(self.results)
                mock_logger.warning.assert_called_once()

    def test_statistical_test_success(self):
        """Test statistical comparison between algorithms."""
        result1 = BenchmarkResult("algo1")
        result1.episode_rewards = [1.0, 2.0, 3.0, 4.0, 5.0]

        result2 = BenchmarkResult("algo2")
        result2.episode_rewards = [2.0, 3.0, 4.0, 5.0, 6.0]

        with patch("scipy.stats.ttest_ind") as mock_ttest:
            mock_ttest.return_value = (2.0, 0.01)  # t-stat, p-value

            result = AlgorithmComparison.statistical_test(result1, result2)

            self.assertIn("t_statistic", result)
            self.assertIn("p_value", result)
            self.assertTrue(result["significant"])
            self.assertAlmostEqual(result["mean_diff"], -1.0)  # type: ignore

    def test_statistical_test_no_scipy(self):
        """Test statistical comparison when scipy is not available."""
        result1 = BenchmarkResult("algo1")
        result1.episode_rewards = [1.0, 2.0, 3.0]

        result2 = BenchmarkResult("algo2")
        result2.episode_rewards = [2.0, 3.0, 4.0]

        with patch.dict("sys.modules", {"scipy": None}):
            result = AlgorithmComparison.statistical_test(result1, result2)

            self.assertIn("mean1", result)
            self.assertIn("mean2", result)
            self.assertIn("note", result)

    def test_statistical_test_insufficient_data(self):
        """Test statistical test with insufficient data."""
        result1 = BenchmarkResult("algo1")
        result1.episode_rewards = []

        result2 = BenchmarkResult("algo2")
        result2.episode_rewards = [1.0, 2.0]

        result = AlgorithmComparison.statistical_test(result1, result2)
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()

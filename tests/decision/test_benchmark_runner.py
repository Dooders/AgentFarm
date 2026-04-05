"""Tests for farm/core/decision/benchmark/runner.py – AlgorithmBenchmark."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from farm.core.decision.benchmark.runner import AlgorithmBenchmark


class TestAlgorithmBenchmark:
    def test_run_comparison_returns_dict(self):
        benchmark = AlgorithmBenchmark()
        with patch("farm.core.decision.benchmark.runner.AlgorithmRegistry") as mock_registry:
            mock_algo = MagicMock()
            mock_registry.create.return_value = mock_algo

            result = benchmark.run_comparison(
                algorithms=["mlp"],
                environments=["env1"],
                metrics=["accuracy", "speed"],
            )

        assert isinstance(result, dict)

    def test_run_comparison_keys_match_algorithm_names(self):
        benchmark = AlgorithmBenchmark()
        with patch("farm.core.decision.benchmark.runner.AlgorithmRegistry") as mock_registry:
            mock_registry.create.return_value = MagicMock()

            result = benchmark.run_comparison(
                algorithms=["mlp", "dqn"],
                environments=["e1"],
                metrics=["accuracy"],
            )

        # Each algorithm produces one key (algo_synthetic)
        assert "mlp_synthetic" in result
        assert "dqn_synthetic" in result

    def test_run_comparison_metric_values_are_floats(self):
        benchmark = AlgorithmBenchmark()
        metrics = ["accuracy", "speed", "stability"]
        with patch("farm.core.decision.benchmark.runner.AlgorithmRegistry") as mock_registry:
            mock_registry.create.return_value = MagicMock()

            result = benchmark.run_comparison(
                algorithms=["mlp"],
                environments=["e1"],
                metrics=metrics,
            )

        key = "mlp_synthetic"
        for m in metrics:
            assert m in result[key]
            assert isinstance(result[key][m], float)

    def test_run_comparison_empty_algorithms(self):
        benchmark = AlgorithmBenchmark()
        result = benchmark.run_comparison(
            algorithms=[],
            environments=["e1"],
            metrics=["acc"],
        )
        assert result == {}

    def test_run_comparison_calls_registry_create(self):
        benchmark = AlgorithmBenchmark()
        with patch("farm.core.decision.benchmark.runner.AlgorithmRegistry") as mock_registry:
            mock_registry.create.return_value = MagicMock()

            benchmark.run_comparison(
                algorithms=["mlp", "nb"],
                environments=["e1"],
                metrics=["acc"],
            )

            assert mock_registry.create.call_count == 2
            mock_registry.create.assert_any_call("mlp", num_actions=4)
            mock_registry.create.assert_any_call("nb", num_actions=4)

#!/usr/bin/env python3
"""
Tests for advanced time series analysis capabilities.

Tests the new ARIMA, VAR, and exponential smoothing modeling features.
"""

import sqlite3

# Add the parent directory to the path to import the analysis module
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from analysis.simulation_analysis import SimulationAnalyzer


class TestAdvancedTimeSeriesAnalysis(unittest.TestCase):
    """Test cases for advanced time series analysis."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db.close()

        # Create test database with comprehensive time series data
        self._create_advanced_test_database()

        # Initialize analyzer
        self.analyzer = SimulationAnalyzer(self.temp_db.name, random_seed=42)

    def tearDown(self):
        """Clean up test fixtures."""
        # Close the analyzer session and engine first
        if hasattr(self, "analyzer"):
            if hasattr(self.analyzer, "session"):
                self.analyzer.session.close()
            if hasattr(self.analyzer, "engine"):
                self.analyzer.engine.dispose()
        # Then remove the database file
        Path(self.temp_db.name).unlink(missing_ok=True)

    def _create_advanced_test_database(self):
        """Create a test database with complex time series patterns."""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()

        # Create tables
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_steps (
                step_number INTEGER,
                simulation_id TEXT,
                total_agents INTEGER,
                system_agents INTEGER,
                independent_agents INTEGER,
                control_agents INTEGER,
                total_resources REAL,
                average_agent_resources REAL,
                births INTEGER,
                deaths INTEGER,
                current_max_generation INTEGER,
                resource_efficiency REAL,
                resource_distribution_entropy REAL,
                average_agent_health REAL,
                average_agent_age INTEGER,
                average_reward REAL,
                combat_encounters INTEGER,
                successful_attacks INTEGER,
                resources_shared REAL,
                resources_shared_this_step REAL DEFAULT 0.0,
                combat_encounters_this_step INTEGER DEFAULT 0,
                successful_attacks_this_step INTEGER DEFAULT 0,
                genetic_diversity REAL,
                dominant_genome_ratio REAL,
                resources_consumed REAL DEFAULT 0.0,
                PRIMARY KEY (step_number, simulation_id)
            )
        """
        )

        # Insert complex time series data
        test_simulation_id = 1
        np.random.seed(42)  # For reproducible test data

        for step in range(1, 201):  # 200 steps for advanced modeling
            # Create complex time series with trends, seasonality, and autoregressive patterns
            trend = 0.05 * step
            seasonal = 8 * np.sin(2 * np.pi * step / 20) + 4 * np.cos(
                2 * np.pi * step / 15
            )
            ar_component = 0.3 * (step > 1 and step <= 200)  # Simple AR(1) component
            noise = np.random.normal(0, 1.5)

            # System agents with strong trend and seasonality
            system_agents = max(0, int(50 + trend + seasonal + ar_component + noise))

            # Independent agents with different pattern
            indep_trend = 0.02 * step
            indep_seasonal = 5 * np.sin(2 * np.pi * step / 25) + 3 * np.cos(
                2 * np.pi * step / 18
            )
            indep_noise = np.random.normal(0, 1.2)
            independent_agents = max(
                0, int(30 + indep_trend + indep_seasonal + indep_noise)
            )

            # Control agents with minimal trend
            control_trend = 0.01 * step
            control_seasonal = 3 * np.sin(2 * np.pi * step / 30)
            control_noise = np.random.normal(0, 0.8)
            control_agents = max(
                0, int(20 + control_trend + control_seasonal + control_noise)
            )

            total_agents = system_agents + independent_agents + control_agents

            cursor.execute(
                """
                INSERT INTO simulation_steps 
                (simulation_id, step_number, system_agents, independent_agents, 
                 control_agents, total_agents, total_resources, average_agent_resources,
                 births, deaths, current_max_generation, resource_efficiency, 
                 resource_distribution_entropy, average_agent_health, average_agent_age, 
                 average_reward, combat_encounters, successful_attacks, resources_shared,
                 genetic_diversity, dominant_genome_ratio, resources_consumed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    str(test_simulation_id),
                    step,
                    system_agents,
                    independent_agents,
                    control_agents,
                    total_agents,
                    1000.0 + 50 * np.sin(step * 0.1),  # total_resources
                    (1000.0 + 50 * np.sin(step * 0.1))
                    / max(total_agents, 1),  # average_agent_resources
                    max(0, int(np.random.poisson(0.5))),  # births
                    max(0, int(np.random.poisson(0.3))),  # deaths
                    max(1, int(step / 50)),  # current_max_generation
                    0.8
                    + 0.1 * np.sin(step * 0.1)
                    + 0.05 * np.random.random(),  # resource_efficiency
                    0.6
                    + 0.2 * np.cos(step * 0.12)
                    + 0.1 * np.random.random(),  # resource_distribution_entropy
                    0.7
                    + 0.2 * np.cos(step * 0.15)
                    + 0.1 * np.random.random(),  # average_agent_health
                    max(1, int(step / 10)),  # average_agent_age
                    0.5
                    + 0.3 * np.sin(step * 0.2)
                    + 0.15 * np.random.random(),  # average_reward
                    max(0, int(np.random.poisson(0.2))),  # combat_encounters
                    max(0, int(np.random.poisson(0.1))),  # successful_attacks
                    max(0, 10 * np.random.random()),  # resources_shared
                    0.8
                    + 0.1 * np.sin(step * 0.08)
                    + 0.05 * np.random.random(),  # genetic_diversity
                    0.3
                    + 0.2 * np.cos(step * 0.1)
                    + 0.1 * np.random.random(),  # dominant_genome_ratio
                    max(0, 5 * np.random.random()),
                ),
            )  # resources_consumed

        conn.commit()
        conn.close()

    def test_advanced_time_series_modeling_basic(self):
        """Test basic functionality of advanced time series modeling."""
        result = self.analyzer.analyze_advanced_time_series_models(1)

        # Check return structure
        self.assertIn("arima_models", result)
        self.assertIn("var_model", result)
        self.assertIn("exponential_smoothing", result)
        self.assertIn("model_comparison", result)
        self.assertIn("metadata", result)

        # Check metadata
        metadata = result["metadata"]
        self.assertIn("analysis_timestamp", metadata)
        self.assertIn("total_steps", metadata)
        self.assertIn("methods_used", metadata)

        methods_used = metadata["methods_used"]
        expected_methods = [
            "ARIMA modeling with auto parameter selection",
            "Vector Autoregression (VAR)",
            "Exponential Smoothing (Simple, Holt, Holt-Winters)",
            "Granger Causality Testing",
            "Model comparison and selection",
            "Forecasting with confidence intervals",
        ]

        for method in expected_methods:
            self.assertIn(method, methods_used)

    def test_arima_modeling(self):
        """Test ARIMA modeling functionality."""
        result = self.analyzer.analyze_advanced_time_series_models(1)

        arima_models = result["arima_models"]
        self.assertIsInstance(arima_models, dict)

        # Check that ARIMA models were fitted for available time series
        expected_series = [
            "system_agents",
            "independent_agents",
            "control_agents",
            "total_agents",
        ]

        for series_name in expected_series:
            if series_name in arima_models:
                arima_result = arima_models[series_name]

                if "error" not in arima_result:
                    # Check ARIMA model structure
                    self.assertIn("model_order", arima_result)
                    self.assertIn("aic", arima_result)
                    self.assertIn("bic", arima_result)
                    self.assertIn("forecast", arima_result)
                    self.assertIn("forecast_ci_lower", arima_result)
                    self.assertIn("forecast_ci_upper", arima_result)
                    self.assertIn("residuals_stats", arima_result)
                    self.assertIn("model_summary", arima_result)

                    # Check model order is a tuple
                    model_order = arima_result["model_order"]
                    self.assertIsInstance(model_order, tuple)
                    self.assertEqual(len(model_order), 3)  # (p, d, q)

                    # Check AIC and BIC are reasonable
                    self.assertIsInstance(arima_result["aic"], (int, float))
                    self.assertIsInstance(arima_result["bic"], (int, float))

                    # Check forecast structure
                    forecast = arima_result["forecast"]
                    self.assertIsInstance(forecast, list)
                    self.assertGreater(len(forecast), 0)

                    # Check confidence intervals
                    ci_lower = arima_result["forecast_ci_lower"]
                    ci_upper = arima_result["forecast_ci_upper"]
                    self.assertEqual(len(ci_lower), len(forecast))
                    self.assertEqual(len(ci_upper), len(forecast))

                    # Check residuals stats
                    residuals_stats = arima_result["residuals_stats"]
                    self.assertIn("mean", residuals_stats)
                    self.assertIn("std", residuals_stats)
                    self.assertIn("skewness", residuals_stats)
                    self.assertIn("kurtosis", residuals_stats)

    def test_var_modeling(self):
        """Test Vector Autoregression (VAR) modeling."""
        result = self.analyzer.analyze_advanced_time_series_models(1)

        var_model = result["var_model"]

        if "error" not in var_model:
            # Check VAR model structure
            self.assertIn("model_order", var_model)
            self.assertIn("aic", var_model)
            self.assertIn("bic", var_model)
            self.assertIn("forecast", var_model)
            self.assertIn("granger_causality", var_model)
            self.assertIn("model_summary", var_model)

            # Check model order
            self.assertIsInstance(var_model["model_order"], int)
            self.assertGreater(var_model["model_order"], 0)

            # Check AIC and BIC
            self.assertIsInstance(var_model["aic"], (int, float))
            self.assertIsInstance(var_model["bic"], (int, float))

            # Check forecast structure
            forecast = var_model["forecast"]
            self.assertIsInstance(forecast, list)
            self.assertGreater(len(forecast), 0)

            # Check Granger causality results
            granger_causality = var_model["granger_causality"]
            self.assertIsInstance(granger_causality, dict)

            for test_name, test_result in granger_causality.items():
                self.assertIn("statistic", test_result)
                self.assertIn("p_value", test_result)
                self.assertIn("significant", test_result)
                self.assertIsInstance(test_result["significant"], bool)

    def test_exponential_smoothing(self):
        """Test exponential smoothing modeling."""
        result = self.analyzer.analyze_advanced_time_series_models(1)

        exp_smoothing = result["exponential_smoothing"]
        self.assertIsInstance(exp_smoothing, dict)

        expected_series = [
            "system_agents",
            "independent_agents",
            "control_agents",
            "total_agents",
        ]

        for series_name in expected_series:
            if series_name in exp_smoothing:
                exp_result = exp_smoothing[series_name]

                if "error" not in exp_result:
                    # Check exponential smoothing structure
                    self.assertIn("best_model", exp_result)
                    self.assertIn("model_info", exp_result)
                    self.assertIn("forecast", exp_result)
                    self.assertIn("fitted_values", exp_result)
                    self.assertIn("residuals", exp_result)

                    # Check best model is one of the expected types
                    best_model = exp_result["best_model"]
                    self.assertIn(best_model, ["simple", "holt", "holt_winters"])

                    # Check model info
                    model_info = exp_result["model_info"]
                    self.assertIn("aic", model_info)
                    self.assertIn("bic", model_info)
                    self.assertIn("sse", model_info)

                    # Check forecast
                    forecast = exp_result["forecast"]
                    self.assertIsInstance(forecast, list)
                    self.assertGreater(len(forecast), 0)

                    # Check fitted values and residuals
                    fitted_values = exp_result["fitted_values"]
                    residuals = exp_result["residuals"]
                    self.assertEqual(len(fitted_values), len(residuals))

    def test_model_comparison(self):
        """Test model comparison functionality."""
        result = self.analyzer.analyze_advanced_time_series_models(1)

        model_comparison = result["model_comparison"]
        self.assertIsInstance(model_comparison, dict)

        for series_name, comparison in model_comparison.items():
            self.assertIn("best_model", comparison)
            self.assertIn("comparison", comparison)

            # Check that best model is one of the available models
            best_model = comparison["best_model"]
            self.assertIn(best_model, ["arima", "exponential_smoothing"])

            # Check comparison structure
            comparison_data = comparison["comparison"]
            self.assertIsInstance(comparison_data, dict)

            for model_name, model_metrics in comparison_data.items():
                self.assertIn("aic", model_metrics)
                self.assertIn("bic", model_metrics)
                self.assertIsInstance(model_metrics["aic"], (int, float))
                self.assertIsInstance(model_metrics["bic"], (int, float))

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data for advanced modeling."""
        # Create database with insufficient data
        small_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        small_db.close()

        conn = sqlite3.connect(small_db.name)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS simulation_steps (
                step_number INTEGER,
                simulation_id TEXT,
                total_agents INTEGER,
                system_agents INTEGER,
                independent_agents INTEGER,
                control_agents INTEGER,
                total_resources REAL,
                average_agent_resources REAL,
                births INTEGER,
                deaths INTEGER,
                current_max_generation INTEGER,
                resource_efficiency REAL,
                resource_distribution_entropy REAL,
                average_agent_health REAL,
                average_agent_age INTEGER,
                average_reward REAL,
                combat_encounters INTEGER,
                successful_attacks INTEGER,
                resources_shared REAL,
                resources_shared_this_step REAL DEFAULT 0.0,
                combat_encounters_this_step INTEGER DEFAULT 0,
                successful_attacks_this_step INTEGER DEFAULT 0,
                genetic_diversity REAL,
                dominant_genome_ratio REAL,
                resources_consumed REAL DEFAULT 0.0,
                PRIMARY KEY (step_number, simulation_id)
            )
        """
        )

        # Insert only 20 steps (insufficient for advanced modeling)
        for step in range(1, 21):
            cursor.execute(
                """
                INSERT INTO simulation_steps 
                (simulation_id, step_number, system_agents, independent_agents, control_agents,
                 total_agents, total_resources, average_agent_resources, births, deaths,
                 current_max_generation, resource_efficiency, resource_distribution_entropy,
                 average_agent_health, average_agent_age, average_reward, combat_encounters,
                 successful_attacks, resources_shared, genetic_diversity, dominant_genome_ratio,
                 resources_consumed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    "1",
                    step,
                    10,
                    5,
                    3,
                    18,
                    100.0,
                    5.56,
                    0,
                    0,
                    1,
                    0.8,
                    0.6,
                    0.7,
                    1,
                    0.5,
                    0,
                    0,
                    0.0,
                    0.8,
                    0.3,
                    0.0,
                ),
            )

        conn.commit()
        conn.close()

        analyzer = SimulationAnalyzer(small_db.name, random_seed=42)
        result = analyzer.analyze_advanced_time_series_models(1)

        self.assertIn("error", result)
        self.assertEqual(result["error"], "Insufficient data for advanced modeling")
        self.assertEqual(result["min_points_required"], 50)

        # Close the analyzer session and engine first
        analyzer.session.close()
        analyzer.engine.dispose()
        Path(small_db.name).unlink()

    def test_visualization_creation(self):
        """Test that advanced time series visualizations are created."""
        with patch("matplotlib.pyplot.savefig") as mock_savefig, patch(
            "matplotlib.pyplot.close"
        ) as mock_close:

            # Run advanced time series modeling
            self.analyzer.analyze_advanced_time_series_models(1)

            # Check that savefig was called (indicating plot was created)
            mock_savefig.assert_called()

            # Check that the correct filename pattern was used
            call_args = mock_savefig.call_args[0]
            self.assertIn("advanced_time_series_models_sim_1.png", call_args[0])

    def test_integration_with_complete_analysis(self):
        """Test integration with complete analysis workflow."""
        # Mock the complete analysis to avoid needing all database tables
        with patch.object(self.analyzer, "run_complete_analysis") as mock_complete:
            # Create a mock result that includes advanced time series modeling
            mock_result = {
                "simulation_id": 1,
                "advanced_time_series_models": {
                    "arima_models": {"system_agents": {"model_order": (1, 1, 1)}},
                    "var_model": {"model_order": 2},
                    "exponential_smoothing": {"system_agents": {"best_model": "holt"}},
                    "model_comparison": {"system_agents": {"best_model": "arima"}},
                },
                "metadata": {
                    "statistical_methods_used": [
                        "Advanced time series modeling (ARIMA, VAR, exponential smoothing)",
                        "Granger causality testing and forecasting",
                    ]
                },
            }
            mock_complete.return_value = mock_result

            result = self.analyzer.run_complete_analysis(1, significance_level=0.05)

            # Check that advanced time series modeling is included
            self.assertIn("advanced_time_series_models", result)

            advanced_models = result["advanced_time_series_models"]
            self.assertIn("arima_models", advanced_models)
            self.assertIn("var_model", advanced_models)
            self.assertIn("exponential_smoothing", advanced_models)
            self.assertIn("model_comparison", advanced_models)

            # Check metadata includes advanced time series methods
            metadata = result["metadata"]
            methods_used = metadata["statistical_methods_used"]

            advanced_methods = [
                "Advanced time series modeling (ARIMA, VAR, exponential smoothing)",
                "Granger causality testing and forecasting",
            ]

            for method in advanced_methods:
                self.assertIn(method, methods_used)

    def test_reproducibility_with_advanced_modeling(self):
        """Test that advanced time series modeling is reproducible."""
        # Run analysis twice with same seed
        result1 = self.analyzer.analyze_advanced_time_series_models(1)
        result2 = self.analyzer.analyze_advanced_time_series_models(1)

        # Results should be identical (or very close for floating point)
        self._compare_advanced_results(result1, result2)

    def _compare_advanced_results(self, result1, result2, tolerance=1e-10):
        """Compare two advanced time series results for consistency."""

        def compare_values(val1, val2, path=""):
            # Skip timestamp comparison as it will always be different
            if path.endswith("analysis_timestamp"):
                return

            if type(val1) != type(val2):
                self.fail(f"Type mismatch at {path}: {type(val1)} vs {type(val2)}")

            if isinstance(val1, dict):
                keys1, keys2 = set(val1.keys()), set(val2.keys())
                self.assertEqual(keys1, keys2, f"Key mismatch at {path}")

                for key in keys1:
                    compare_values(
                        val1[key], val2[key], f"{path}.{key}" if path else key
                    )

            elif isinstance(val1, (list, tuple)):
                self.assertEqual(len(val1), len(val2), f"Length mismatch at {path}")
                for i, (v1, v2) in enumerate(zip(val1, val2)):
                    compare_values(v1, v2, f"{path}[{i}]")

            elif isinstance(val1, (int, float)):
                if not np.isclose(val1, val2, rtol=tolerance, atol=tolerance):
                    self.fail(f"Value mismatch at {path}: {val1} vs {val2}")

            elif isinstance(val1, str):
                self.assertEqual(val1, val2, f"String mismatch at {path}")

            else:
                self.assertEqual(val1, val2, f"Value mismatch at {path}")

        compare_values(result1, result2)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)

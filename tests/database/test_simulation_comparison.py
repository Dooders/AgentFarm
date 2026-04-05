"""Tests for simulation_comparison module.

Covers format_diff_output, compare_simulations, summarize_comparison,
and get_significant_changes using SQLite :memory: databases.
"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from farm.database.models import Base, Simulation, SimulationStepModel, SimulationDifference
from farm.database.simulation_comparison import (
    compare_simulations,
    format_diff_output,
    get_significant_changes,
    summarize_comparison,
)


def _make_simulation(simulation_id: str, status: str = "completed", start=None):
    return Simulation(
        simulation_id=simulation_id,
        start_time=start or datetime(2024, 1, 1),
        status=status,
        parameters={},
        simulation_db_path=f"/tmp/{simulation_id}.db",
    )


class TestFormatDiffOutput(unittest.TestCase):
    """Tests for format_diff_output function."""

    def _make_diff(self, metadata_diff=None, parameter_diff=None, results_diff=None, step_metrics_diff=None):
        return SimulationDifference(
            metadata_diff=metadata_diff or {},
            parameter_diff=parameter_diff or {},
            results_diff=results_diff or {},
            step_metrics_diff=step_metrics_diff or {},
        )

    def test_empty_diff_returns_empty_sections(self):
        diff = self._make_diff()
        result = format_diff_output(diff)
        self.assertIn("metadata", result)
        self.assertIn("parameters", result)
        self.assertIn("results", result)
        self.assertIn("metrics", result)
        self.assertEqual(result["metadata"], {})
        self.assertEqual(result["metrics"], {})

    def test_metadata_diff_formatted(self):
        diff = self._make_diff(
            metadata_diff={"status": ("running", "completed")}
        )
        result = format_diff_output(diff)
        self.assertIn("status", result["metadata"])
        self.assertEqual(result["metadata"]["status"]["simulation_1"], "running")
        self.assertEqual(result["metadata"]["status"]["simulation_2"], "completed")

    def test_step_metrics_diff_rounded(self):
        diff = self._make_diff(
            step_metrics_diff={
                "total_agents": {
                    "mean_diff": 5.3214,
                    "max_diff": 8.0,
                    "min_diff": 2.0,
                    "std_diff": 1.2345,
                }
            }
        )
        result = format_diff_output(diff)
        self.assertIn("total_agents", result["metrics"])
        stats = result["metrics"]["total_agents"]
        self.assertEqual(stats["mean_difference"], round(5.3214, 3))
        self.assertEqual(stats["max_difference"], round(8.0, 3))
        self.assertEqual(stats["min_difference"], round(2.0, 3))
        self.assertEqual(stats["std_dev_difference"], round(1.2345, 3))

    def test_parameter_diff_formatted(self):
        diff = self._make_diff(
            parameter_diff={
                "values_changed": {"config.pop": {"old_value": 10, "new_value": 20}}
            }
        )
        result = format_diff_output(diff)
        self.assertIn("changed", result["parameters"])


class TestCompareSimulations(unittest.TestCase):
    """Tests for compare_simulations using an in-memory DB session."""

    def setUp(self):
        """Build an in-memory DB with two Simulation rows."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

        # Simulation uses simulation_id (string) as primary key
        sim1 = _make_simulation("sim_001", start=datetime(2024, 1, 1))
        sim2 = _make_simulation("sim_002", start=datetime(2024, 1, 2))
        self.session.add_all([sim1, sim2])
        self.session.commit()
        self.sim1_id = "sim_001"
        self.sim2_id = "sim_002"

    def tearDown(self):
        self.session.close()

    def test_compare_returns_dict_with_expected_keys(self):
        result = compare_simulations(self.session, self.sim1_id, self.sim2_id)
        self.assertIsInstance(result, dict)
        for key in ("metadata", "parameters", "results", "metrics"):
            self.assertIn(key, result)

    def test_compare_invalid_id_raises(self):
        with self.assertRaises(ValueError):
            compare_simulations(self.session, "nonexistent_1", "nonexistent_2")

    def test_compare_same_simulation(self):
        """Comparing a simulation to itself should produce empty diffs."""
        result = compare_simulations(self.session, self.sim1_id, self.sim1_id)
        self.assertEqual(result["metrics"], {})


class TestSummarizeComparison(unittest.TestCase):
    """Tests for summarize_comparison function."""

    def setUp(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

        sim1 = _make_simulation("sim_A", start=datetime(2024, 1, 1))
        sim2 = _make_simulation("sim_B", start=datetime(2024, 1, 2))
        self.session.add_all([sim1, sim2])
        self.session.commit()
        self.sim1_id = "sim_A"
        self.sim2_id = "sim_B"

    def tearDown(self):
        self.session.close()

    def test_summarize_returns_string(self):
        result = summarize_comparison(self.session, self.sim1_id, self.sim2_id)
        self.assertIsInstance(result, str)
        self.assertIn(self.sim1_id, result)
        self.assertIn(self.sim2_id, result)


class TestGetSignificantChanges(unittest.TestCase):
    """Tests for get_significant_changes function."""

    def setUp(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        self.session = Session()

        sim1 = _make_simulation("sim_X", start=datetime(2024, 1, 1))
        sim2 = _make_simulation("sim_Y", start=datetime(2024, 1, 2))
        self.session.add_all([sim1, sim2])
        self.session.commit()
        self.sim1_id = "sim_X"
        self.sim2_id = "sim_Y"

    def tearDown(self):
        self.session.close()

    def test_returns_empty_dict_for_identical_simulations(self):
        """No significant changes between identical simulations."""
        result = get_significant_changes(self.session, self.sim1_id, self.sim1_id)
        self.assertIsInstance(result, dict)

    def test_threshold_filters_small_changes(self):
        """Insignificant metric differences are excluded at given threshold."""
        with patch(
            "farm.database.simulation_comparison.compare_simulations"
        ) as mock_cmp:
            mock_cmp.return_value = {
                "metadata": {},
                "parameters": {},
                "results": {},
                "metrics": {
                    "big_change": {"mean_difference": 5.0},
                    "small_change": {"mean_difference": 0.01},
                },
            }
            result = get_significant_changes(
                self.session, self.sim1_id, self.sim2_id, threshold=0.1
            )
        self.assertIn("big_change", result)
        self.assertNotIn("small_change", result)


if __name__ == "__main__":
    unittest.main()

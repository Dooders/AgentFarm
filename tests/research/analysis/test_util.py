"""Tests for farm/research/analysis/util.py."""

import numpy as np
import pandas as pd

from farm.research.analysis.util import (
    calculate_statistics,
    validate_population_data,
    validate_resource_level_data,
)


class TestValidatePopulationData:
    def test_valid_data(self):
        assert validate_population_data(np.array([1.0, 2.0, 3.0]))

    def test_none_is_invalid(self):
        assert not validate_population_data(None)

    def test_empty_is_invalid(self):
        assert not validate_population_data(np.array([]))

    def test_all_nan_is_invalid(self):
        assert not validate_population_data(np.array([np.nan, np.nan]))

    def test_negative_values_are_invalid(self):
        assert not validate_population_data(np.array([1.0, -2.0]))

    def test_db_path_only_affects_logging(self):
        assert validate_population_data(np.array([1.0]), db_path="some.db")


class TestValidateResourceLevelData:
    def test_valid_data(self):
        assert validate_resource_level_data(np.array([1.0, 2.0]))

    def test_negative_values_are_allowed(self):
        assert validate_resource_level_data(np.array([-1.0, 2.0]))

    def test_none_is_invalid(self):
        assert not validate_resource_level_data(None)

    def test_empty_is_invalid(self):
        assert not validate_resource_level_data(np.array([]))

    def test_all_nan_is_invalid(self):
        assert not validate_resource_level_data(np.array([np.nan]))


class TestCalculateStatistics:
    def _make_df(self):
        # Two simulations, three steps each.
        return pd.DataFrame(
            {
                "simulation_id": ["a"] * 3 + ["b"] * 3,
                "step": [0, 1, 2, 0, 1, 2],
                "population": [10.0, 20.0, 30.0, 20.0, 30.0, 40.0],
            }
        )

    def test_mean_and_median(self):
        mean, median, std, ci = calculate_statistics(self._make_df())
        np.testing.assert_allclose(mean, [15.0, 25.0, 35.0])
        np.testing.assert_allclose(median, [15.0, 25.0, 35.0])
        assert len(std) == 3
        assert len(ci) == 3

    def test_confidence_interval_formula(self):
        df = self._make_df()
        _, _, std, ci = calculate_statistics(df)
        np.testing.assert_allclose(ci, 1.96 * std / np.sqrt(2))

    def test_empty_dataframe_returns_empty_arrays(self):
        result = calculate_statistics(pd.DataFrame())
        assert all(len(arr) == 0 for arr in result)

    def test_no_valid_simulations_returns_empty_arrays(self):
        df = pd.DataFrame(
            {"simulation_id": [np.nan], "step": [0], "population": [1.0]}
        )
        result = calculate_statistics(df)
        assert all(len(arr) == 0 for arr in result)

    def test_single_simulation_std_is_zero_filled(self):
        df = pd.DataFrame(
            {"simulation_id": ["a"] * 2, "step": [0, 1], "population": [5.0, 6.0]}
        )
        _, _, std, _ = calculate_statistics(df)
        np.testing.assert_allclose(std, [0.0, 0.0])

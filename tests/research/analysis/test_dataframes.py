"""Tests for farm/research/analysis/dataframes.py."""

import numpy as np

from farm.research.analysis.dataframes import create_population_df


class TestCreatePopulationDf:
    def test_basic_dataframe(self):
        df = create_population_df([np.array([1.0, 2.0]), np.array([3.0, 4.0])], max_steps=2)
        assert len(df) == 4
        assert set(df.columns) == {"simulation_id", "step", "population"}
        assert df[df["simulation_id"] == "sim_0"]["population"].tolist() == [1.0, 2.0]

    def test_shorter_series_padded_with_nan(self):
        df = create_population_df([np.array([1.0])], max_steps=3)
        values = df["population"].tolist()
        assert values[0] == 1.0
        assert np.isnan(values[1]) and np.isnan(values[2])

    def test_empty_input_returns_empty_dataframe(self):
        df = create_population_df([], max_steps=5)
        assert df.empty
        assert set(df.columns) == {"simulation_id", "step", "population"}

    def test_invalid_population_data_is_skipped(self):
        df = create_population_df(
            [np.array([-1.0, 2.0]), np.array([1.0, 2.0])], max_steps=2
        )
        # The negative-valued array fails population validation.
        assert df["simulation_id"].nunique() == 1

    def test_all_invalid_returns_empty_dataframe(self):
        df = create_population_df([np.array([-1.0])], max_steps=1)
        assert df.empty

    def test_zero_max_steps_returns_empty_dataframe(self):
        df = create_population_df([np.array([1.0])], max_steps=0)
        assert df.empty

    def test_resource_data_allows_negative_values(self):
        df = create_population_df(
            [np.array([-1.0, 2.0])], max_steps=2, is_resource_data=True
        )
        assert len(df) == 2
        assert df["population"].tolist() == [-1.0, 2.0]

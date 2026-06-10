"""Tests for farm/research/analysis/database.py against a real seeded SQLite db."""

import numpy as np
import pytest

from farm.research.analysis.database import (
    find_simulation_databases,
    get_action_distribution_data,
    get_columns_data,
    get_columns_data_by_agent_type,
    get_data,
    get_resource_consumption_data,
    get_resource_level_data,
    get_rewards_by_generation,
)

from .conftest import NUM_STEPS


class TestFindSimulationDatabases:
    def test_finds_nested_databases(self, tmp_path, seeded_db_path):
        found = find_simulation_databases(str(tmp_path))
        assert found == [seeded_db_path]

    def test_creates_missing_directory_and_returns_empty(self, tmp_path):
        target = tmp_path / "missing"
        assert find_simulation_databases(str(target)) == []
        assert target.is_dir()


class TestGetColumnsData:
    def test_returns_steps_and_columns(self, seeded_db_path):
        result = get_columns_data(seeded_db_path, ["total_agents"])
        assert result is not None
        steps, populations, max_steps = result
        assert max_steps == NUM_STEPS
        np.testing.assert_array_equal(steps, np.arange(NUM_STEPS))
        np.testing.assert_array_equal(populations["total_agents"], [10, 11, 12, 13, 14])

    def test_missing_file_returns_none(self, tmp_path):
        assert get_columns_data(str(tmp_path / "absent.db"), ["total_agents"]) is None

    def test_unknown_column_returns_none(self, seeded_db_path):
        assert get_columns_data(seeded_db_path, ["not_a_column"]) is None

    def test_resource_column_uses_resource_validation(self, seeded_db_path):
        # average_agent_resources contains negative values in the seed data,
        # which only the resource-level validator accepts.
        result = get_columns_data(seeded_db_path, ["average_agent_resources"])
        assert result is not None


class TestGetData:
    def test_returns_total_agents(self, seeded_db_path):
        result = get_data(seeded_db_path)
        assert result is not None
        _, population, max_steps = result
        assert max_steps == NUM_STEPS
        assert population[0] == 10

    def test_missing_file_returns_none(self, tmp_path):
        assert get_data(str(tmp_path / "absent.db")) is None


class TestGetColumnsDataByAgentType:
    def test_extracts_counts_from_json(self, seeded_db_path):
        result = get_columns_data_by_agent_type(seeded_db_path)
        assert result is not None
        steps, populations, max_steps = result
        assert max_steps == NUM_STEPS
        np.testing.assert_array_equal(populations["system_agents"], [4, 5, 6, 7, 8])
        np.testing.assert_array_equal(populations["independent_agents"], [3] * NUM_STEPS)
        np.testing.assert_array_equal(populations["order_agents"], [0] * NUM_STEPS)

    def test_missing_file_returns_none(self, tmp_path):
        assert get_columns_data_by_agent_type(str(tmp_path / "absent.db")) is None


class TestGetResourceConsumptionData:
    def test_consumption_split_proportionally(self, seeded_db_path):
        result = get_resource_consumption_data(seeded_db_path)
        assert result is not None
        steps, consumption, max_steps = result
        assert max_steps == NUM_STEPS
        assert set(consumption) == {"system", "independent", "control"}
        # Step 1: total counts = 5+3+3 = 11, resources consumed = 2.0
        assert consumption["system"][1] == pytest.approx(5 * 2.0 / 11)
        # Totals per step must add up to resources consumed.
        totals = consumption["system"] + consumption["independent"] + consumption["control"]
        np.testing.assert_allclose(totals, [0.0, 2.0, 4.0, 6.0, 8.0])

    def test_missing_file_returns_none(self, tmp_path):
        assert get_resource_consumption_data(str(tmp_path / "absent.db")) is None


class TestGetActionDistributionData:
    def test_counts_actions_per_agent_type(self, seeded_db_path):
        result = get_action_distribution_data(seeded_db_path)
        assert result == {
            "system": {"move": 1, "gather": 1},
            "independent": {"move": 1},
        }

    def test_missing_file_returns_empty(self, tmp_path):
        assert get_action_distribution_data(str(tmp_path / "absent.db")) == {}


class TestGetResourceLevelData:
    def test_returns_average_agent_resources(self, seeded_db_path):
        result = get_resource_level_data(seeded_db_path)
        assert result is not None
        _, resource_levels, max_steps = result
        assert max_steps == NUM_STEPS
        np.testing.assert_allclose(resource_levels, [5.0, 4.0, 3.0, 2.0, 1.0])

    def test_missing_file_returns_none(self, tmp_path):
        assert get_resource_level_data(str(tmp_path / "absent.db")) is None


class TestGetRewardsByGeneration:
    def test_averages_rewards_per_generation(self, seeded_db_path):
        result = get_rewards_by_generation(seeded_db_path)
        # Generation 0 (agent_sys): rewards 1.0 and 3.0 -> avg 2.0
        # Generation 1 (agent_ind): reward 2.0
        assert result == {0: 2.0, 1: 2.0}

    def test_missing_file_returns_empty(self, tmp_path):
        assert get_rewards_by_generation(str(tmp_path / "absent.db")) == {}


class TestEmptyDatabaseHandling:
    """All readers must degrade gracefully on a schema-only database."""

    def test_get_columns_data(self, empty_db_path):
        assert get_columns_data(empty_db_path, ["total_agents"]) is None

    def test_get_columns_data_by_agent_type(self, empty_db_path):
        assert get_columns_data_by_agent_type(empty_db_path) is None

    def test_get_resource_consumption_data(self, empty_db_path):
        assert get_resource_consumption_data(empty_db_path) is None

    def test_get_action_distribution_data(self, empty_db_path):
        assert get_action_distribution_data(empty_db_path) == {}

    def test_get_rewards_by_generation(self, empty_db_path):
        assert get_rewards_by_generation(empty_db_path) == {}


class TestCorruptDatabaseHandling:
    """All readers must degrade gracefully on a non-SQLite file."""

    def test_get_columns_data(self, corrupt_db_path):
        assert get_columns_data(corrupt_db_path, ["total_agents"]) is None

    def test_get_columns_data_by_agent_type(self, corrupt_db_path):
        assert get_columns_data_by_agent_type(corrupt_db_path) is None

    def test_get_resource_consumption_data(self, corrupt_db_path):
        assert get_resource_consumption_data(corrupt_db_path) is None

    def test_get_action_distribution_data(self, corrupt_db_path):
        assert get_action_distribution_data(corrupt_db_path) == {}

    def test_get_rewards_by_generation(self, corrupt_db_path):
        assert get_rewards_by_generation(corrupt_db_path) == {}

"""Tests for farm/research/analysis/analysis.py.

Experiment-processing functions resolve paths relative to the working
directory; the `experiment_dir` fixture provides a seeded layout under a
temporary cwd. Plot generation is patched out to keep tests fast.
"""

from unittest.mock import patch

import numpy as np

from farm.research.analysis.analysis import (
    analyze_final_agent_counts,
    detect_early_terminations,
    find_experiments,
    process_action_distributions,
    process_experiment,
    process_experiment_by_agent_type,
    process_experiment_resource_consumption,
    process_experiment_resource_levels,
    process_experiment_rewards_by_generation,
)

from .conftest import NUM_STEPS


class TestFindExperiments:
    def test_groups_single_agent_and_one_of_a_kind(self, tmp_path):
        (tmp_path / "single_system_agent_v1").mkdir()
        (tmp_path / "single_control_agent_v1").mkdir()
        (tmp_path / "single_system_agent_v2").mkdir()
        (tmp_path / "one_of_a_kind_v1").mkdir()
        (tmp_path / "unrelated_dir").mkdir()

        experiments = find_experiments(str(tmp_path))

        assert sorted(experiments["single_agent"]["system"]) == [
            "single_system_agent_v1",
            "single_system_agent_v2",
        ]
        assert experiments["single_agent"]["control"] == ["single_control_agent_v1"]
        assert experiments["one_of_a_kind"] == ["one_of_a_kind_v1"]

    def test_empty_directory(self, tmp_path):
        experiments = find_experiments(str(tmp_path))
        assert experiments == {"single_agent": {}, "one_of_a_kind": []}


class TestAnalyzeFinalAgentCounts:
    def _experiment_data(self):
        return {
            "system": {"populations": [np.array([1, 5]), np.array([2, 8])]},
            "control": {"populations": [np.array([1, 3]), np.array([2, 8])]},
            "independent": {"populations": [np.array([1, 2]), np.array([2, 4])]},
        }

    def test_statistics_per_agent_type(self):
        result = analyze_final_agent_counts(self._experiment_data())
        assert result["system"]["total"] == 13
        assert result["system"]["mean"] == 6.5
        assert result["system"]["max"] == 8
        assert result["system"]["min"] == 5
        assert result["system"]["simulations"] == 2

    def test_dominant_type_counting(self):
        result = analyze_final_agent_counts(self._experiment_data())
        # Simulation 0: system=5 dominant; simulation 1: system/control tie at 8.
        assert result["dominant_type_counts"]["system"] == 1
        assert result["dominant_type_counts"]["tie"] == 1

    def test_empty_data_returns_zeroed_result(self):
        empty = {t: {"populations": []} for t in ["system", "control", "independent"]}
        result = analyze_final_agent_counts(empty)
        assert result["system"]["simulations"] == 0
        assert result["dominant_type_counts"]["tie"] == 0


class TestDetectEarlyTerminations:
    def test_flags_simulation_below_threshold(self, seeded_db_path):
        # The seeded db has NUM_STEPS steps; expecting far more marks it early.
        result = detect_early_terminations([seeded_db_path], expected_steps=100)
        assert seeded_db_path in result
        info = result[seeded_db_path]
        assert info["steps_completed"] == NUM_STEPS
        assert info["completion_percentage"] == 5.0
        assert info["likely_cause"] in {"population_collapse", "resource_depletion", "unknown"}

    def test_no_early_termination_with_default_expected_steps(self, seeded_db_path):
        # With expected_steps derived from the max observed, nothing is early.
        assert detect_early_terminations([seeded_db_path]) == {}

    def test_no_valid_databases_returns_empty(self, tmp_path):
        assert detect_early_terminations([str(tmp_path / "absent.db")]) == {}


class TestProcessExperiment:
    def test_processes_seeded_experiment(self, experiment_dir):
        with patch(
            "farm.research.analysis.analysis.plot_population_trends_across_simulations"
        ) as mock_plot:
            result = process_experiment("system", experiment_dir)
        assert result["max_steps"] == NUM_STEPS
        assert len(result["populations"]) == 1
        np.testing.assert_array_equal(result["populations"][0], [10, 11, 12, 13, 14])
        mock_plot.assert_called_once()

    def test_missing_experiment_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert process_experiment("system", "no_such_experiment") == {
            "populations": [],
            "max_steps": 0,
        }


class TestProcessExperimentByAgentType:
    def test_populations_split_by_type(self, experiment_dir):
        result = process_experiment_by_agent_type(experiment_dir)
        assert result["system"]["max_steps"] == NUM_STEPS
        np.testing.assert_array_equal(result["system"]["populations"][0], [4, 5, 6, 7, 8])
        np.testing.assert_array_equal(result["control"]["populations"][0], [3] * NUM_STEPS)

    def test_missing_experiment_returns_empty_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = process_experiment_by_agent_type("no_such_experiment")
        assert result["system"]["populations"] == []


class TestProcessExperimentResourceConsumption:
    def test_consumption_per_agent_type(self, experiment_dir):
        result = process_experiment_resource_consumption(experiment_dir)
        assert result["system"]["max_steps"] == NUM_STEPS
        assert len(result["system"]["consumption"]) == 1
        assert result["system"]["consumption"][0][0] == 0.0

    def test_missing_experiment_returns_empty_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = process_experiment_resource_consumption("no_such_experiment")
        assert result["system"]["consumption"] == []


class TestProcessActionDistributions:
    def test_action_percentages(self, experiment_dir):
        result = process_action_distributions(experiment_dir)
        assert result["system"]["total_actions"] == 2
        assert result["system"]["actions"]["move"] == 0.5
        assert result["system"]["actions"]["gather"] == 0.5
        assert result["independent"]["actions"]["move"] == 1.0

    def test_missing_experiment_returns_empty_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = process_action_distributions("no_such_experiment")
        assert result["system"]["total_actions"] == 0


class TestProcessExperimentResourceLevels:
    def test_resource_levels_extracted(self, experiment_dir):
        result = process_experiment_resource_levels(experiment_dir)
        assert result["max_steps"] == NUM_STEPS
        np.testing.assert_allclose(result["resource_levels"][0], [5.0, 4.0, 3.0, 2.0, 1.0])

    def test_missing_experiment_returns_empty_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = process_experiment_resource_levels("no_such_experiment")
        assert result == {"resource_levels": [], "max_steps": 0}


class TestExperimentWithoutDatabases:
    """Experiment directories with no simulation.db must return empty results."""

    def test_process_experiment(self, experiment_dir_without_dbs):
        result = process_experiment("system", experiment_dir_without_dbs)
        assert result == {"populations": [], "max_steps": 0}

    def test_process_experiment_by_agent_type(self, experiment_dir_without_dbs):
        result = process_experiment_by_agent_type(experiment_dir_without_dbs)
        assert result["system"]["populations"] == []

    def test_process_experiment_resource_consumption(self, experiment_dir_without_dbs):
        result = process_experiment_resource_consumption(experiment_dir_without_dbs)
        assert result["system"]["consumption"] == []

    def test_process_action_distributions(self, experiment_dir_without_dbs):
        result = process_action_distributions(experiment_dir_without_dbs)
        assert result["system"]["total_actions"] == 0

    def test_process_experiment_resource_levels(self, experiment_dir_without_dbs):
        result = process_experiment_resource_levels(experiment_dir_without_dbs)
        assert result == {"resource_levels": [], "max_steps": 0}

    def test_process_experiment_rewards_by_generation(self, experiment_dir_without_dbs):
        result = process_experiment_rewards_by_generation(experiment_dir_without_dbs)
        assert result == {"system": {}, "control": {}, "independent": {}}


class TestExperimentWithCorruptDatabase:
    """Corrupt databases must be skipped without raising."""

    def test_process_experiment(self, experiment_dir_with_corrupt_db):
        result = process_experiment("system", experiment_dir_with_corrupt_db)
        assert result == {"populations": [], "max_steps": 0}

    def test_process_experiment_by_agent_type(self, experiment_dir_with_corrupt_db):
        result = process_experiment_by_agent_type(experiment_dir_with_corrupt_db)
        assert result["system"]["populations"] == []

    def test_process_experiment_resource_consumption(self, experiment_dir_with_corrupt_db):
        result = process_experiment_resource_consumption(experiment_dir_with_corrupt_db)
        assert result["system"]["consumption"] == []

    def test_process_action_distributions(self, experiment_dir_with_corrupt_db):
        result = process_action_distributions(experiment_dir_with_corrupt_db)
        assert result["system"]["total_actions"] == 0

    def test_process_experiment_resource_levels(self, experiment_dir_with_corrupt_db):
        result = process_experiment_resource_levels(experiment_dir_with_corrupt_db)
        assert result == {"resource_levels": [], "max_steps": 0}

    def test_process_experiment_rewards_by_generation(self, experiment_dir_with_corrupt_db):
        result = process_experiment_rewards_by_generation(experiment_dir_with_corrupt_db)
        assert result == {"system": {}, "control": {}, "independent": {}}


class TestProcessExperimentRewardsByGeneration:
    def test_rewards_grouped_by_agent_type(self, experiment_dir):
        result = process_experiment_rewards_by_generation(experiment_dir)
        # Generation 0 belongs to a system agent, generation 1 to an independent.
        assert result["system"] == {0: 2.0}
        assert result["independent"] == {1: 2.0}
        assert result["control"] == {}

    def test_missing_experiment_returns_empty_result(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = process_experiment_rewards_by_generation("no_such_experiment")
        assert result == {"system": {}, "control": {}, "independent": {}}

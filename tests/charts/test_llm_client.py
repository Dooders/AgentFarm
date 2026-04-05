"""Tests for llm_client module with mocked OpenAI."""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from farm.charts.llm_client import LLMClient


def _make_config(api_key="test-api-key"):
    """Return a mock IConfigService that provides a fake API key."""
    cfg = MagicMock()
    cfg.get_openai_api_key.return_value = api_key
    return cfg


def _make_actions_df():
    return pd.DataFrame(
        {
            "action_type": ["move", "eat", "move", "attack", "eat"],
            "reward": [0.1, 0.5, 0.2, -0.1, 0.4],
            "step_number": [1, 1, 2, 2, 3],
            "resources_before": [10.0, 10.0, 9.5, 9.0, 9.5],
            "resources_after": [9.5, 11.0, 9.0, 8.0, 11.0],
            "agent_id": ["a1", "a1", "a1", "a2", "a2"],
            "action_target_id": ["t1", "t2", "t1", "t3", "t2"],
        }
    )


def _make_agents_df():
    return pd.DataFrame(
        {
            "agent_id": ["a1", "a2", "a3"],
            "agent_type": ["SystemAgent", "IndependentAgent", "SystemAgent"],
            "birth_time": [0, 5, 10],
            "death_time": [20, 15, 30],
            "genome_id": ["::", "a1::1", "a2:a1:1"],
            "starting_health": [1.0, 0.9, 1.0],
            "initial_resources": [10.0, 8.0, 12.0],
            "generation": [1, 2, 2],
            "position_x": [5.0, 10.0, 15.0],
            "position_y": [5.0, 10.0, 15.0],
        }
    )


class TestLLMClientInit(unittest.TestCase):
    """Tests for LLMClient initialization."""

    @patch("farm.charts.llm_client.OpenAI")
    def test_init_with_explicit_api_key(self, mock_openai):
        """LLMClient can be initialized with an explicit API key."""
        client = LLMClient(api_key="sk-test-key", config_service=_make_config())
        self.assertEqual(client.api_key, "sk-test-key")
        mock_openai.assert_called_once_with(api_key="sk-test-key")

    @patch("farm.charts.llm_client.OpenAI")
    def test_init_with_config_service_key(self, mock_openai):
        """LLMClient falls back to config service when api_key is None."""
        client = LLMClient(api_key=None, config_service=_make_config())
        self.assertEqual(client.api_key, "test-api-key")

    @patch("farm.charts.llm_client.OpenAI")
    def test_init_no_key_raises(self, mock_openai):
        """LLMClient raises ValueError when no API key is available."""
        bad_cfg = MagicMock()
        bad_cfg.get_openai_api_key.return_value = None
        with self.assertRaises(ValueError):
            LLMClient(api_key=None, config_service=bad_cfg)

    @patch("farm.charts.llm_client.OpenAI")
    def test_set_data_actions(self, mock_openai):
        """set_data stores actions DataFrame."""
        client = LLMClient(api_key="sk-test", config_service=_make_config())
        df = _make_actions_df()
        client.set_data(df, data_type="actions")
        self.assertIsNotNone(client.actions_df)

    @patch("farm.charts.llm_client.OpenAI")
    def test_set_data_agents(self, mock_openai):
        """set_data stores agents DataFrame."""
        client = LLMClient(api_key="sk-test", config_service=_make_config())
        df = _make_agents_df()
        client.set_data(df, data_type="agents")
        self.assertIsNotNone(client.agents_df)

    @patch("farm.charts.llm_client.OpenAI")
    def test_set_data_invalid_type_raises(self, mock_openai):
        """set_data raises ValueError for unknown data_type."""
        client = LLMClient(api_key="sk-test", config_service=_make_config())
        with self.assertRaises(ValueError):
            client.set_data(_make_actions_df(), data_type="invalid")


class TestLLMClientActionAnalysis(unittest.TestCase):
    """Tests for LLMClient action-based analysis methods."""

    def setUp(self):
        with patch("farm.charts.llm_client.OpenAI"):
            self.client = LLMClient(api_key="sk-test", config_service=_make_config())
        self.client.set_data(_make_actions_df(), data_type="actions")

    def test_analyze_action_distribution(self):
        result = self.client._analyze_action_distribution()
        self.assertIn("Action Distribution Analysis", result)
        self.assertIn("Most frequent action", result)

    def test_analyze_rewards_by_action(self):
        result = self.client._analyze_rewards_by_action()
        self.assertIn("Reward Analysis by Action", result)

    def test_analyze_resource_changes(self):
        result = self.client._analyze_resource_changes()
        self.assertIn("Resource Impact Analysis", result)

    def test_analyze_temporal_patterns(self):
        result = self.client._analyze_temporal_patterns()
        self.assertIn("Temporal Pattern Analysis", result)

    def test_analyze_reward_progression(self):
        result = self.client._analyze_reward_progression()
        self.assertIn("Reward Progression Analysis", result)

    def test_analyze_target_distribution(self):
        result = self.client._analyze_target_distribution()
        self.assertIn("Target Selection Analysis", result)

    def test_analyze_action_distribution_no_data(self):
        self.client.actions_df = None
        result = self.client._analyze_action_distribution()
        self.assertIn("Error", result)


class TestLLMClientAgentAnalysis(unittest.TestCase):
    """Tests for LLMClient agent-based analysis methods."""

    def setUp(self):
        with patch("farm.charts.llm_client.OpenAI"):
            self.client = LLMClient(api_key="sk-test", config_service=_make_config())
        self.client.set_data(_make_agents_df(), data_type="agents")

    def test_analyze_lifespan_distribution(self):
        result = self.client._analyze_lifespan_distribution()
        self.assertIn("Lifespan Distribution Analysis", result)

    def test_analyze_lifespan_distribution_no_data(self):
        self.client.agents_df = None
        result = self.client._analyze_lifespan_distribution()
        self.assertIn("Error", result)

    def test_analyze_spatial_distribution(self):
        result = self.client._analyze_spatial_distribution()
        self.assertIn("Spatial Distribution Analysis", result)

    def test_analyze_resources_by_generation(self):
        result = self.client._analyze_resources_by_generation()
        self.assertIn("Resource Evolution Analysis", result)

    def test_analyze_starvation_counters_no_column(self):
        """_analyze_starvation_counters handles missing column gracefully."""
        result = self.client._analyze_starvation_counters()
        # Should handle missing column gracefully
        self.assertIsInstance(result, str)

    def test_analyze_starvation_counters_with_column(self):
        """_analyze_starvation_counters works when column is present."""
        df = _make_agents_df().copy()
        df["starvation_counter"] = [0, 5, 3]
        self.client.agents_df = df
        result = self.client._analyze_starvation_counters()
        self.assertIsInstance(result, str)

    def test_analyze_health_vs_resources(self):
        result = self.client._analyze_health_vs_resources()
        self.assertIsInstance(result, str)
        # Should contain correlation analysis
        self.assertIn("Health-Resource Relationship", result)

    def test_analyze_health_vs_resources_constant_values(self):
        """Handles constant health/resource values gracefully."""
        df = _make_agents_df().copy()
        df["starting_health"] = 1.0  # All same
        df["initial_resources"] = 10.0  # All same
        self.client.agents_df = df
        result = self.client._analyze_health_vs_resources()
        self.assertIn("Health-Resource Relationship", result)

    def test_analyze_agent_types_over_time(self):
        result = self.client._analyze_agent_types_over_time()
        self.assertIn("Population Evolution Analysis", result)

    def test_analyze_lineage_size(self):
        result = self.client._analyze_lineage_size()
        self.assertIn("Lineage Analysis", result)

    def test_analyze_reproduction_success_rate_no_success_column(self):
        result = self.client._analyze_reproduction_success_rate()
        self.assertIn("Reproduction data not available", result)

    def test_analyze_reproduction_success_rate_with_column(self):
        df = _make_agents_df().copy()
        df["success"] = [1, 0, 1]
        df["step_number"] = [1, 2, 3]
        self.client.agents_df = df
        result = self.client._analyze_reproduction_success_rate()
        self.assertIn("Reproduction Success Analysis", result)


class TestLLMClientAnalyzeChart(unittest.TestCase):
    """Tests for LLMClient.analyze_chart dispatch method."""

    def setUp(self):
        with patch("farm.charts.llm_client.OpenAI"):
            self.client = LLMClient(api_key="sk-test", config_service=_make_config())
        self.client.set_data(_make_actions_df(), data_type="actions")
        self.client.set_data(_make_agents_df(), data_type="agents")

    def test_analyze_chart_action_type_distribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "action_type_distribution.png")
            # Create a dummy file so os.path.basename works
            open(image_path, "w").close()
            with patch.object(self.client, "_save_analyses"):
                result = self.client.analyze_chart(image_path)
        self.assertIn("Action Distribution Analysis", result)

    def test_analyze_chart_lifespan_distribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "lifespan_distribution.png")
            open(image_path, "w").close()
            with patch.object(self.client, "_save_analyses"):
                result = self.client.analyze_chart(image_path)
        self.assertIn("Lifespan Distribution Analysis", result)

    def test_analyze_chart_rewards_by_action(self):
        """Covers the rewards_by_action action_analyses branch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "rewards_by_action_type.png")
            open(image_path, "w").close()
            with patch.object(self.client, "_save_analyses"):
                result = self.client.analyze_chart(image_path)
        self.assertIn("Reward Analysis by Action", result)

    def test_analyze_chart_resource_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "resource_changes.png")
            open(image_path, "w").close()
            with patch.object(self.client, "_save_analyses"):
                result = self.client.analyze_chart(image_path)
        self.assertIn("Resource Impact Analysis", result)

    def test_analyze_chart_spatial_distribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "spatial_distribution.png")
            open(image_path, "w").close()
            with patch.object(self.client, "_save_analyses"):
                result = self.client.analyze_chart(image_path)
        self.assertIn("Spatial Distribution Analysis", result)

    def test_analyze_chart_starvation_counters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "starvation_counters.png")
            open(image_path, "w").close()
            with patch.object(self.client, "_save_analyses"):
                result = self.client.analyze_chart(image_path)
        # Should return some string (either analysis or "Data not available")
        self.assertIsInstance(result, str)

    def test_analyze_chart_health_vs_resources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "health_vs_resources.png")
            open(image_path, "w").close()
            with patch.object(self.client, "_save_analyses"):
                result = self.client.analyze_chart(image_path)
        self.assertIn("Health-Resource Relationship", result)

    def test_analyze_chart_agent_types_over_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "agent_types_over_time.png")
            open(image_path, "w").close()
            with patch.object(self.client, "_save_analyses"):
                result = self.client.analyze_chart(image_path)
        self.assertIn("Population Evolution Analysis", result)

    def test_analyze_chart_reproduction_success_rate(self):
        df = _make_agents_df().copy()
        df["success"] = [1, 0, 1]
        df["step_number"] = [1, 2, 3]
        self.client.set_data(df, data_type="agents")
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "reproduction_success_rate.png")
            open(image_path, "w").close()
            with patch.object(self.client, "_save_analyses"):
                result = self.client.analyze_chart(image_path)
        self.assertIsInstance(result, str)

    def test_analyze_chart_lineage_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "lineage_size.png")
            open(image_path, "w").close()
            with patch.object(self.client, "_save_analyses"):
                result = self.client.analyze_chart(image_path)
        self.assertIn("Lineage Analysis", result)

    def test_analyze_chart_exception_returns_error_string(self):
        """analyze_chart returns error string when an exception occurs."""
        with patch.object(self.client, "_analyze_action_distribution") as mock_method:
            mock_method.side_effect = Exception("Unexpected error")
            with tempfile.TemporaryDirectory() as tmpdir:
                image_path = os.path.join(tmpdir, "action_type_distribution.png")
                open(image_path, "w").close()
                result = self.client.analyze_chart(image_path)
        self.assertIn("Error analyzing chart", result)

    def test_analyze_chart_unknown_returns_not_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "unknown_chart.png")
            open(image_path, "w").close()
            result = self.client.analyze_chart(image_path)
        self.assertIn("Analysis not available", result)

    def test_analyze_chart_action_no_data_returns_error(self):
        self.client.actions_df = None
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "action_type_distribution.png")
            open(image_path, "w").close()
            result = self.client.analyze_chart(image_path)
        self.assertIn("Error", result)

    def test_save_analyses_creates_files(self):
        """_save_analyses writes JSON and text files."""
        self.client.analyses = {"test_chart": "Some analysis text"}
        with tempfile.TemporaryDirectory() as tmpdir:
            self.client._save_analyses(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "chart_analyses.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "chart_analyses.txt")))


if __name__ == "__main__":
    unittest.main()

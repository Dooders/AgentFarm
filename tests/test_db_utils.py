"""Tests for farm/database/utils.py."""
import unittest

import pandas as pd

from farm.database.utils import (
    extract_agent_counts_from_json,
    normalize_simulation_steps_dataframe,
)


class TestExtractAgentCountsFromJson(unittest.TestCase):
    def test_none_returns_empty_dict(self):
        self.assertEqual(extract_agent_counts_from_json(None), {})

    def test_dict_returns_dict_unchanged(self):
        data = {"system": 3, "independent": 2}
        self.assertEqual(extract_agent_counts_from_json(data), data)

    def test_valid_json_string(self):
        json_str = '{"system": 5, "independent": 3, "control": 1}'
        result = extract_agent_counts_from_json(json_str)
        self.assertEqual(result["system"], 5)
        self.assertEqual(result["independent"], 3)
        self.assertEqual(result["control"], 1)

    def test_invalid_json_string_returns_empty(self):
        result = extract_agent_counts_from_json("not_valid_json{{{")
        self.assertEqual(result, {})

    def test_empty_json_object_string(self):
        result = extract_agent_counts_from_json("{}")
        self.assertEqual(result, {})

    def test_non_string_non_dict_non_none_returns_empty(self):
        result = extract_agent_counts_from_json(42)
        self.assertEqual(result, {})

    def test_list_returns_empty(self):
        result = extract_agent_counts_from_json([1, 2, 3])
        self.assertEqual(result, {})

    def test_empty_string_returns_empty(self):
        # json.loads("") raises JSONDecodeError → should return {}
        result = extract_agent_counts_from_json("")
        self.assertEqual(result, {})


class TestNormalizeSimulationStepsDataframe(unittest.TestCase):
    def test_empty_dataframe_returned_unchanged(self):
        df = pd.DataFrame()
        result = normalize_simulation_steps_dataframe(df)
        self.assertTrue(result.empty)

    def test_adds_agent_columns_from_json(self):
        df = pd.DataFrame(
            {
                "step": [1, 2],
                "agent_type_counts": [
                    '{"system": 3, "independent": 2, "control": 1}',
                    '{"system": 4, "independent": 1, "control": 0}',
                ],
            }
        )
        result = normalize_simulation_steps_dataframe(df)
        self.assertIn("system_agents", result.columns)
        self.assertIn("independent_agents", result.columns)
        self.assertIn("control_agents", result.columns)
        self.assertEqual(result.iloc[0]["system_agents"], 3)
        self.assertEqual(result.iloc[0]["independent_agents"], 2)
        self.assertEqual(result.iloc[0]["control_agents"], 1)
        self.assertEqual(result.iloc[1]["system_agents"], 4)

    def test_no_agent_type_counts_column_no_old_columns_adds_zeros(self):
        df = pd.DataFrame({"step": [1, 2], "population": [10, 8]})
        result = normalize_simulation_steps_dataframe(df)
        self.assertIn("system_agents", result.columns)
        self.assertTrue((result["system_agents"] == 0).all())
        self.assertTrue((result["independent_agents"] == 0).all())
        self.assertTrue((result["control_agents"] == 0).all())

    def test_no_agent_type_counts_but_old_columns_present(self):
        df = pd.DataFrame(
            {
                "step": [1],
                "system_agents": [5],
                "independent_agents": [3],
                "control_agents": [2],
            }
        )
        result = normalize_simulation_steps_dataframe(df)
        self.assertEqual(result.iloc[0]["system_agents"], 5)
        self.assertIn("order_agents", result.columns)
        self.assertIn("chaos_agents", result.columns)
        self.assertEqual(result.iloc[0]["order_agents"], 0)
        self.assertEqual(result.iloc[0]["chaos_agents"], 0)

    def test_missing_agent_types_default_to_zero(self):
        df = pd.DataFrame(
            {
                "step": [1],
                "agent_type_counts": ['{"system": 2}'],
            }
        )
        result = normalize_simulation_steps_dataframe(df)
        self.assertEqual(result.iloc[0]["system_agents"], 2)
        self.assertEqual(result.iloc[0]["independent_agents"], 0)
        self.assertEqual(result.iloc[0]["control_agents"], 0)

    def test_dict_in_agent_type_counts(self):
        df = pd.DataFrame(
            {
                "step": [1],
                "agent_type_counts": [{"system": 7, "independent": 3}],
            }
        )
        result = normalize_simulation_steps_dataframe(df)
        self.assertEqual(result.iloc[0]["system_agents"], 7)
        self.assertEqual(result.iloc[0]["independent_agents"], 3)

    def test_invalid_json_defaults_to_zero(self):
        df = pd.DataFrame(
            {
                "step": [1],
                "agent_type_counts": ["invalid_json"],
            }
        )
        result = normalize_simulation_steps_dataframe(df)
        self.assertEqual(result.iloc[0]["system_agents"], 0)
        self.assertEqual(result.iloc[0]["independent_agents"], 0)
        self.assertEqual(result.iloc[0]["control_agents"], 0)

    def test_existing_columns_overwritten_when_json_present(self):
        df = pd.DataFrame(
            {
                "step": [1],
                "agent_type_counts": ['{"system": 10, "independent": 5, "control": 2}'],
                "system_agents": [0],
                "independent_agents": [0],
                "control_agents": [0],
            }
        )
        result = normalize_simulation_steps_dataframe(df)
        self.assertEqual(result.iloc[0]["system_agents"], 10)

"""Extended tests for database utilities covering setup and helper functions."""

import unittest

from farm.database.utilities import (
    format_position,
    parse_position,
    safe_json_loads,
    validate_export_format,
)


class TestDatabaseUtilitiesExtended(unittest.TestCase):
    """Extended tests for database utilities."""

    def test_safe_json_loads_valid(self):
        """Test safe_json_loads with valid JSON."""
        json_str = '{"key": "value", "number": 42}'
        result = safe_json_loads(json_str)

        self.assertIsNotNone(result)
        self.assertEqual(result["key"], "value")
        self.assertEqual(result["number"], 42)

    def test_safe_json_loads_invalid(self):
        """Test safe_json_loads with invalid JSON."""
        json_str = "invalid json {"
        result = safe_json_loads(json_str)

        self.assertIsNone(result)

    def test_safe_json_loads_none(self):
        """Test safe_json_loads with None."""
        result = safe_json_loads(None)

        self.assertIsNone(result)

    def test_format_position(self):
        """Test position formatting."""
        position = (10.5, 20.7)
        formatted = format_position(position)

        self.assertEqual(formatted, "10.5, 20.7")  # Note: includes space after comma

    def test_parse_position(self):
        """Test position parsing."""
        position_str = "10.5,20.7"
        parsed = parse_position(position_str)

        self.assertEqual(parsed, (10.5, 20.7))

    def test_validate_export_format(self):
        """Test export format validation."""
        self.assertTrue(validate_export_format("csv"))
        self.assertTrue(validate_export_format("json"))
        self.assertFalse(validate_export_format("invalid"))


if __name__ == "__main__":
    unittest.main()


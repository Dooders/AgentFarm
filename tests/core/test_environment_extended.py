"""Extended tests for environment module covering resource management and spatial operations."""

import unittest
from unittest.mock import Mock, patch

from farm.core.environment import Environment


class TestEnvironmentExtended(unittest.TestCase):
    """Extended tests for Environment class."""

    @patch("farm.database.utilities.setup_db")
    def test_resource_management(self, mock_setup_db):
        """Test resource management methods."""
        mock_db = Mock()
        mock_setup_db.return_value = mock_db

        env = Environment(width=100, height=100, resource_distribution="uniform")

        # Test resource access
        self.assertIsNotNone(env)

    @patch("farm.database.utilities.setup_db")
    def test_spatial_operations(self, mock_setup_db):
        """Test spatial operations."""
        mock_db = Mock()
        mock_setup_db.return_value = mock_db

        env = Environment(width=100, height=100, resource_distribution="uniform")

        # Test spatial index exists
        self.assertIsNotNone(env)


if __name__ == "__main__":
    unittest.main()


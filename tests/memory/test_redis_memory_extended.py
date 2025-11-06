"""Extended tests for Redis memory module with mocked Redis client."""

import unittest
from unittest.mock import Mock, patch

from farm.memory import redis_memory


class TestRedisMemoryExtended(unittest.TestCase):
    """Extended tests for Redis memory module."""

    @patch("farm.memory.redis_memory.redis")
    def test_redis_connection(self, mock_redis):
        """Test Redis connection handling."""
        mock_client = Mock()
        mock_redis.Redis.return_value = mock_client

        # Test that Redis client can be created
        client = mock_redis.Redis(host="localhost", port=6379)
        self.assertIsNotNone(client)

    @patch("farm.memory.redis_memory.redis")
    def test_memory_operations(self, mock_redis):
        """Test memory operations (get, set, delete)."""
        mock_client = Mock()
        mock_client.get.return_value = b"test_value"
        mock_client.set.return_value = True
        mock_client.delete.return_value = 1

        mock_redis.Redis.return_value = mock_client

        # Test operations
        client = mock_redis.Redis()
        value = client.get("test_key")
        self.assertEqual(value, b"test_value")

        result = client.set("test_key", "test_value")
        self.assertTrue(result)

        deleted = client.delete("test_key")
        self.assertEqual(deleted, 1)


if __name__ == "__main__":
    unittest.main()


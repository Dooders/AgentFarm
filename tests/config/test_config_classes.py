"""
Unit tests for configuration classes in farm.config.

This module tests the core configuration dataclasses including:
- VisualizationConfig
- RedisMemoryConfig
- Basic config serialization/deserialization
"""

import unittest
from farm.config import VisualizationConfig, RedisMemoryConfig


class TestVisualizationConfig(unittest.TestCase):
    """Test cases for VisualizationConfig class."""

    def test_default_initialization(self):
        """Test that VisualizationConfig initializes with correct defaults."""
        config = VisualizationConfig()

        self.assertEqual(config.canvas_size, (400, 400))
        self.assertEqual(config.padding, 20)
        self.assertEqual(config.background_color, "black")
        self.assertEqual(config.max_animation_frames, 5)
        self.assertEqual(config.animation_min_delay, 50)
        self.assertEqual(config.max_resource_amount, 30)
        self.assertEqual(config.resource_colors, {"glow_red": 150, "glow_green": 255, "glow_blue": 50})
        self.assertEqual(config.resource_size, 2)
        self.assertEqual(config.agent_radius_scale, 2)
        self.assertEqual(config.birth_radius_scale, 4)
        self.assertEqual(config.death_mark_scale, 1.5)
        self.assertEqual(config.agent_colors, {"SystemAgent": "blue", "IndependentAgent": "red"})
        self.assertEqual(config.min_font_size, 10)
        self.assertEqual(config.font_scale_factor, 40)
        self.assertEqual(config.font_family, "arial")
        self.assertEqual(config.death_mark_color, [255, 0, 0])
        self.assertEqual(config.birth_mark_color, [255, 255, 255])
        expected_metric_colors = {
            "total_agents": "#4a90e2",
            "system_agents": "#50c878",
            "independent_agents": "#e74c3c",
            "total_resources": "#f39c12",
            "average_agent_resources": "#9b59b6",
        }
        self.assertEqual(config.metric_colors, expected_metric_colors)

    def test_custom_initialization(self):
        """Test that VisualizationConfig can be initialized with custom values."""
        custom_colors = {"SystemAgent": "green", "IndependentAgent": "purple"}
        custom_metric_colors = {"total_agents": "#ff0000"}

        config = VisualizationConfig(
            canvas_size=(800, 600),
            padding=50,
            background_color="white",
            agent_colors=custom_colors,
            metric_colors=custom_metric_colors
        )

        self.assertEqual(config.canvas_size, (800, 600))
        self.assertEqual(config.padding, 50)
        self.assertEqual(config.background_color, "white")
        self.assertEqual(config.agent_colors, custom_colors)
        self.assertEqual(config.metric_colors, custom_metric_colors)

    def test_to_dict_conversion(self):
        """Test that to_dict converts config to proper dictionary format."""
        config = VisualizationConfig(
            canvas_size=(800, 600),
            death_mark_color=[255, 100, 100],
            birth_mark_color=[200, 200, 200]
        )

        config_dict = config.to_dict()

        # Check that tuples are converted to lists
        self.assertEqual(config_dict["canvas_size"], [800, 600])

        # Check that lists remain lists
        self.assertEqual(config_dict["death_mark_color"], [255, 100, 100])
        self.assertEqual(config_dict["birth_mark_color"], [200, 200, 200])

        # Check that nested dicts are preserved
        self.assertEqual(config_dict["agent_colors"], {"SystemAgent": "blue", "IndependentAgent": "red"})

        # Check basic types
        self.assertEqual(config_dict["padding"], 20)
        self.assertEqual(config_dict["background_color"], "black")

    def test_from_dict_creation(self):
        """Test that from_dict creates config from dictionary."""
        config_dict = {
            "canvas_size": [1024, 768],
            "padding": 30,
            "background_color": "gray",
            "max_animation_frames": 10,
            "agent_colors": {"SystemAgent": "cyan", "IndependentAgent": "magenta"},
            "death_mark_color": [128, 128, 128],
            "birth_mark_color": [255, 255, 0]
        }

        config = VisualizationConfig.from_dict(config_dict)

        # Check tuple conversion from list
        self.assertEqual(config.canvas_size, (1024, 768))

        # Check basic fields
        self.assertEqual(config.padding, 30)
        self.assertEqual(config.background_color, "gray")
        self.assertEqual(config.max_animation_frames, 10)

        # Check dict fields
        self.assertEqual(config.agent_colors, {"SystemAgent": "cyan", "IndependentAgent": "magenta"})

        # Check list fields
        self.assertEqual(config.death_mark_color, [128, 128, 128])
        self.assertEqual(config.birth_mark_color, [255, 255, 0])

    def test_round_trip_serialization(self):
        """Test that config can be serialized and deserialized without data loss."""
        original_config = VisualizationConfig(
            canvas_size=(1920, 1080),
            padding=40,
            background_color="navy",
            agent_colors={"SystemAgent": "yellow", "IndependentAgent": "orange"}
        )

        # Serialize to dict
        config_dict = original_config.to_dict()

        # Deserialize from dict
        restored_config = VisualizationConfig.from_dict(config_dict)

        # Check that all fields match
        self.assertEqual(original_config.canvas_size, restored_config.canvas_size)
        self.assertEqual(original_config.padding, restored_config.padding)
        self.assertEqual(original_config.background_color, restored_config.background_color)
        self.assertEqual(original_config.agent_colors, restored_config.agent_colors)
        self.assertEqual(original_config.death_mark_color, restored_config.death_mark_color)
        self.assertEqual(original_config.metric_colors, restored_config.metric_colors)


class TestRedisMemoryConfig(unittest.TestCase):
    """Test cases for RedisMemoryConfig class."""

    def test_default_initialization(self):
        """Test that RedisMemoryConfig initializes with correct defaults."""
        config = RedisMemoryConfig()

        self.assertEqual(config.host, "localhost")
        self.assertEqual(config.port, 6379)
        self.assertEqual(config.db, 0)
        self.assertEqual(config.password, None)
        self.assertEqual(config.decode_responses, True)
        self.assertEqual(config.environment, "default")

    def test_custom_initialization(self):
        """Test that RedisMemoryConfig can be initialized with custom values."""
        config = RedisMemoryConfig(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            decode_responses=False,
            environment="production"
        )

        self.assertEqual(config.host, "redis.example.com")
        self.assertEqual(config.port, 6380)
        self.assertEqual(config.db, 1)
        self.assertEqual(config.password, "secret")
        self.assertEqual(config.decode_responses, False)
        self.assertEqual(config.environment, "production")

    def test_to_dict_conversion(self):
        """Test that to_dict converts Redis config to dictionary."""
        config = RedisMemoryConfig(
            host="custom.redis.host",
            port=6380,
            db=2,
            password="testpass",
            decode_responses=False,
            environment="staging"
        )

        config_dict = config.to_dict()

        expected_dict = {
            "host": "custom.redis.host",
            "port": 6380,
            "db": 2,
            "password": "testpass",
            "decode_responses": False,
            "environment": "staging"
        }

        self.assertEqual(config_dict, expected_dict)

    def test_to_dict_with_none_password(self):
        """Test that to_dict handles None password correctly."""
        config = RedisMemoryConfig(password=None)
        config_dict = config.to_dict()

        self.assertIsNone(config_dict["password"])


if __name__ == '__main__':
    unittest.main()

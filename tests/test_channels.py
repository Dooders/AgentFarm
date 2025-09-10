"""Unit tests for the channels module.

This module tests the dynamic channel system including channel behaviors,
channel handlers, channel registry, and all core channel implementations.
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from farm.core.channels import (
    NUM_CHANNELS,
    AlliesHPHandler,
    Channel,
    ChannelBehavior,
    ChannelHandler,
    ChannelRegistry,
    EnemiesHPHandler,
    GoalHandler,
    KnownEmptyHandler,
    LandmarkHandler,
    SelfHPHandler,
    TransientEventHandler,
    VisibilityHandler,
    WorldLayerHandler,
    get_channel_registry,
    register_channel,
)


class TestChannelBehavior(unittest.TestCase):
    """Test cases for ChannelBehavior enum."""

    def test_channel_behavior_values(self):
        """Test that ChannelBehavior enum has correct values."""
        self.assertEqual(ChannelBehavior.INSTANT.value, 0)
        self.assertEqual(ChannelBehavior.DYNAMIC.value, 1)
        self.assertEqual(ChannelBehavior.PERSISTENT.value, 2)

    def test_channel_behavior_names(self):
        """Test that ChannelBehavior enum has correct names."""
        self.assertEqual(ChannelBehavior.INSTANT.name, "INSTANT")
        self.assertEqual(ChannelBehavior.DYNAMIC.name, "DYNAMIC")
        self.assertEqual(ChannelBehavior.PERSISTENT.name, "PERSISTENT")

    def test_channel_behavior_order(self):
        """Test that ChannelBehavior values are in ascending order."""
        behaviors = [
            ChannelBehavior.INSTANT,
            ChannelBehavior.DYNAMIC,
            ChannelBehavior.PERSISTENT,
        ]
        values = [b.value for b in behaviors]
        self.assertEqual(values, sorted(values))


class TestChannelHandler(unittest.TestCase):
    """Test cases for ChannelHandler abstract base class."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a concrete implementation for testing
        class TestChannelHandler(ChannelHandler):
            def __init__(
                self, name="TEST", behavior=ChannelBehavior.INSTANT, gamma=None
            ):
                super().__init__(name, behavior, gamma)

            def process(
                self, observation, channel_idx, config, agent_world_pos, **kwargs
            ):
                # Simple test implementation using utility method
                self._safe_store_sparse_point(observation, channel_idx, 0, 0, 1.0)

        self.TestChannelHandler = TestChannelHandler
        self.config_mock = Mock()
        self.config_mock.R = 5

    def test_initialization(self):
        """Test ChannelHandler initialization."""
        handler = self.TestChannelHandler(
            "TEST_CHANNEL", ChannelBehavior.DYNAMIC, gamma=0.9
        )

        self.assertEqual(handler.name, "TEST_CHANNEL")
        self.assertEqual(handler.behavior, ChannelBehavior.DYNAMIC)
        self.assertEqual(handler.gamma, 0.9)

    def test_initialization_default_gamma(self):
        """Test ChannelHandler initialization with default gamma."""
        handler = self.TestChannelHandler()

        self.assertIsNone(handler.gamma)

    def test_abstract_process_method(self):
        """Test that ChannelHandler requires process method implementation."""
        # This should work since we implemented process in TestChannelHandler
        handler = self.TestChannelHandler()
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        handler.process(observation, 0, self.config_mock, agent_world_pos)

        self.assertEqual(observation[0, 0, 0].item(), 1.0)

    def test_decay_instant_channel(self):
        """Test decay method for INSTANT channel."""
        handler = self.TestChannelHandler("TEST", ChannelBehavior.INSTANT)
        observation = torch.ones(1, 11, 11)

        handler.decay(observation, 0)

        # INSTANT channels should not decay
        self.assertTrue(torch.allclose(observation, torch.ones(1, 11, 11)))

    def test_decay_dynamic_channel_with_gamma(self):
        """Test decay method for DYNAMIC channel with gamma."""
        handler = self.TestChannelHandler("TEST", ChannelBehavior.DYNAMIC, gamma=0.8)
        observation = torch.ones(1, 11, 11)

        handler.decay(observation, 0)

        # DYNAMIC channels should decay by gamma factor
        expected = torch.ones(1, 11, 11) * 0.8
        self.assertTrue(torch.allclose(observation, expected))

    def test_decay_dynamic_channel_without_gamma(self):
        """Test decay method for DYNAMIC channel without gamma."""
        handler = self.TestChannelHandler("TEST", ChannelBehavior.DYNAMIC)
        observation = torch.ones(1, 11, 11)

        handler.decay(observation, 0)

        # Should not decay if no gamma
        self.assertTrue(torch.allclose(observation, torch.ones(1, 11, 11)))

    def test_clear_instant_channel(self):
        """Test clear method for INSTANT channel."""
        handler = self.TestChannelHandler("TEST", ChannelBehavior.INSTANT)
        observation = torch.ones(1, 11, 11)

        handler.clear(observation, 0)

        # INSTANT channels should be cleared to zeros
        self.assertTrue(torch.allclose(observation, torch.zeros(1, 11, 11)))

    def test_clear_dynamic_channel(self):
        """Test clear method for DYNAMIC channel."""
        handler = self.TestChannelHandler("TEST", ChannelBehavior.DYNAMIC)
        observation = torch.ones(1, 11, 11)

        handler.clear(observation, 0)

        # DYNAMIC channels should not be cleared
        self.assertTrue(torch.allclose(observation, torch.ones(1, 11, 11)))


class TestChannelRegistry(unittest.TestCase):
    """Test cases for ChannelRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ChannelRegistry()

        # Create a test handler
        class TestHandler(ChannelHandler):
            def __init__(self, name):
                super().__init__(name, ChannelBehavior.INSTANT)

            def process(
                self, observation, channel_idx, config, agent_world_pos, **kwargs
            ):
                # Test handler - no-op implementation
                pass

        self.TestHandler = TestHandler

    def test_initialization(self):
        """Test ChannelRegistry initialization."""
        registry = ChannelRegistry()

        self.assertEqual(len(registry._handlers), 0)
        self.assertEqual(len(registry._name_to_index), 0)
        self.assertEqual(len(registry._index_to_name), 0)
        self.assertEqual(registry._next_index, 0)

    def test_register_handler(self):
        """Test registering a handler."""
        handler = self.TestHandler("TEST_CHANNEL")

        index = self.registry.register(handler)

        self.assertEqual(index, 0)
        self.assertIn("TEST_CHANNEL", self.registry._handlers)
        self.assertEqual(self.registry._name_to_index["TEST_CHANNEL"], 0)
        self.assertEqual(self.registry._index_to_name[0], "TEST_CHANNEL")

    def test_register_handler_with_specific_index(self):
        """Test registering a handler with specific index."""
        handler = self.TestHandler("TEST_CHANNEL")

        index = self.registry.register(handler, index=5)

        self.assertEqual(index, 5)
        self.assertEqual(
            self.registry._next_index, 6
        )  # Should be updated to max(index+1, current)

    def test_register_duplicate_handler(self):
        """Test registering a handler with duplicate name."""
        handler1 = self.TestHandler("TEST_CHANNEL")
        handler2 = self.TestHandler("TEST_CHANNEL")

        self.registry.register(handler1)

        with self.assertRaises(ValueError) as cm:
            self.registry.register(handler2)

        self.assertIn("already registered", str(cm.exception))

    def test_register_handler_conflicting_index(self):
        """Test registering a handler with conflicting index."""
        handler1 = self.TestHandler("TEST_CHANNEL_1")
        handler2 = self.TestHandler("TEST_CHANNEL_2")

        self.registry.register(handler1, index=0)

        with self.assertRaises(ValueError) as cm:
            self.registry.register(handler2, index=0)

        self.assertIn("already assigned", str(cm.exception))

    def test_get_handler(self):
        """Test getting a handler by name."""
        handler = self.TestHandler("TEST_CHANNEL")
        self.registry.register(handler)

        retrieved = self.registry.get_handler("TEST_CHANNEL")

        self.assertEqual(retrieved, handler)

    def test_get_handler_not_found(self):
        """Test getting a handler that doesn't exist."""
        with self.assertRaises(KeyError) as cm:
            self.registry.get_handler("NONEXISTENT")

        self.assertIn("not registered", str(cm.exception))

    def test_get_index(self):
        """Test getting index by name."""
        handler = self.TestHandler("TEST_CHANNEL")
        self.registry.register(handler, index=3)

        index = self.registry.get_index("TEST_CHANNEL")

        self.assertEqual(index, 3)

    def test_get_index_not_found(self):
        """Test getting index for nonexistent handler."""
        with self.assertRaises(KeyError) as cm:
            self.registry.get_index("NONEXISTENT")

        self.assertIn("not registered", str(cm.exception))

    def test_get_name(self):
        """Test getting name by index."""
        handler = self.TestHandler("TEST_CHANNEL")
        self.registry.register(handler, index=7)

        name = self.registry.get_name(7)

        self.assertEqual(name, "TEST_CHANNEL")

    def test_get_name_not_found(self):
        """Test getting name for nonexistent index."""
        with self.assertRaises(KeyError) as cm:
            self.registry.get_name(999)

        self.assertIn("not registered", str(cm.exception))

    def test_get_all_handlers(self):
        """Test getting all handlers."""
        handler1 = self.TestHandler("CHANNEL_1")
        handler2 = self.TestHandler("CHANNEL_2")

        self.registry.register(handler1)
        self.registry.register(handler2)

        handlers = self.registry.get_all_handlers()

        self.assertEqual(len(handlers), 2)
        self.assertIn("CHANNEL_1", handlers)
        self.assertIn("CHANNEL_2", handlers)

    def test_num_channels(self):
        """Test num_channels property."""
        self.assertEqual(self.registry.num_channels, 0)

        handler = self.TestHandler("TEST_CHANNEL")
        self.registry.register(handler)

        self.assertEqual(self.registry.num_channels, 1)

    def test_apply_decay(self):
        """Test applying decay to all DYNAMIC channels."""

        # Create a dynamic handler
        class DynamicHandler(ChannelHandler):
            def __init__(self, name):
                super().__init__(name, ChannelBehavior.DYNAMIC, gamma=0.8)

            def process(
                self, observation, channel_idx, config, agent_world_pos, **kwargs
            ):
                # Test dynamic handler - no-op implementation
                pass

        dynamic_handler = DynamicHandler("DYNAMIC_CHANNEL")
        static_handler = self.TestHandler("STATIC_CHANNEL")

        self.registry.register(dynamic_handler)
        self.registry.register(static_handler)

        observation = torch.ones(2, 11, 11)
        config_mock = Mock()

        self.registry.apply_decay(observation, config_mock)

        # Only the dynamic channel should be decayed
        self.assertAlmostEqual(
            observation[0, 0, 0].item(), 0.8
        )  # Dynamic channel decayed
        self.assertEqual(observation[1, 0, 0].item(), 1.0)  # Static channel unchanged

    def test_clear_instant(self):
        """Test clearing all INSTANT channels."""
        handler1 = self.TestHandler("INSTANT_1")
        handler2 = self.TestHandler("INSTANT_2")

        # Create a dynamic handler that shouldn't be cleared
        class DynamicHandler(ChannelHandler):
            def __init__(self, name):
                super().__init__(name, ChannelBehavior.DYNAMIC)

            def process(
                self, observation, channel_idx, config, agent_world_pos, **kwargs
            ):
                # Test dynamic handler - no-op implementation
                pass

        dynamic_handler = DynamicHandler("DYNAMIC")

        self.registry.register(handler1)
        self.registry.register(handler2)
        self.registry.register(dynamic_handler)

        observation = torch.ones(3, 11, 11)

        self.registry.clear_instant(observation)

        # Only INSTANT channels should be cleared
        self.assertEqual(observation[0, 0, 0].item(), 0.0)  # Cleared
        self.assertEqual(observation[1, 0, 0].item(), 0.0)  # Cleared
        self.assertEqual(observation[2, 0, 0].item(), 1.0)  # Not cleared (DYNAMIC)


class TestGlobalFunctions(unittest.TestCase):
    """Test cases for global functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the global registry for each test
        from farm.core.channels import _global_registry

        _global_registry._handlers.clear()
        _global_registry._name_to_index.clear()
        _global_registry._index_to_name.clear()
        _global_registry._next_index = 0

        class TestHandler(ChannelHandler):
            def __init__(self, name):
                super().__init__(name, ChannelBehavior.INSTANT)

            def process(
                self, observation, channel_idx, config, agent_world_pos, **kwargs
            ):
                # Test handler - no-op implementation
                pass

        self.TestHandler = TestHandler

    def test_register_channel(self):
        """Test register_channel global function."""
        handler = self.TestHandler("GLOBAL_TEST")

        index = register_channel(handler)

        self.assertEqual(index, 0)
        registry = get_channel_registry()
        self.assertIn("GLOBAL_TEST", registry._handlers)

    def test_register_channel_with_index(self):
        """Test register_channel with specific index."""
        handler = self.TestHandler("GLOBAL_TEST_INDEXED")

        index = register_channel(handler, index=10)

        self.assertEqual(index, 10)
        registry = get_channel_registry()
        self.assertEqual(registry._name_to_index["GLOBAL_TEST_INDEXED"], 10)

    def test_get_channel_registry(self):
        """Test get_channel_registry function."""
        registry1 = get_channel_registry()
        registry2 = get_channel_registry()

        # Should return the same instance
        self.assertIs(registry1, registry2)
        self.assertIsInstance(registry1, ChannelRegistry)


class TestSelfHPHandler(unittest.TestCase):
    """Test cases for SelfHPHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = SelfHPHandler()
        self.config_mock = Mock()
        self.config_mock.R = 5

    def test_initialization(self):
        """Test SelfHPHandler initialization."""
        self.assertEqual(self.handler.name, "SELF_HP")
        self.assertEqual(self.handler.behavior, ChannelBehavior.INSTANT)

    def test_process(self):
        """Test SelfHPHandler process method."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, self_hp01=0.75
        )

        # Should set center position to health value
        self.assertEqual(observation[0, 5, 5].item(), 0.75)

    def test_process_no_health_data(self):
        """Test SelfHPHandler with no health data."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        self.handler.process(observation, 0, self.config_mock, agent_world_pos)

        # Should default to 0.0
        self.assertEqual(observation[0, 5, 5].item(), 0.0)


class TestAlliesHPHandler(unittest.TestCase):
    """Test cases for AlliesHPHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = AlliesHPHandler()
        self.config_mock = Mock()
        self.config_mock.R = 5

    def test_initialization(self):
        """Test AlliesHPHandler initialization."""
        self.assertEqual(self.handler.name, "ALLIES_HP")
        self.assertEqual(self.handler.behavior, ChannelBehavior.INSTANT)

    def test_process_with_allies(self):
        """Test AlliesHPHandler process method with allies."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        allies = [
            (7, 5, 0.8),  # Ally at relative (2, 0) from agent
            (5, 7, 0.6),  # Ally at relative (0, 2) from agent
        ]

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, allies=allies
        )

        # Check that allies' health is written at correct positions
        self.assertAlmostEqual(observation[0, 7, 5].item(), 0.8)  # First ally
        self.assertAlmostEqual(observation[0, 5, 7].item(), 0.6)  # Second ally

    def test_process_no_allies(self):
        """Test AlliesHPHandler with no allies."""
        observation = torch.ones(1, 11, 11)
        agent_world_pos = (5, 5)

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, allies=[]
        )

        # Observation should remain unchanged
        self.assertTrue(torch.allclose(observation, torch.ones(1, 11, 11)))

    def test_process_ally_outside_bounds(self):
        """Test AlliesHPHandler with ally outside observation bounds."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        allies = [
            (20, 20, 0.8),  # Way outside bounds
        ]

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, allies=allies
        )

        # Should not write anything (all values remain 0)
        self.assertTrue(torch.allclose(observation, torch.zeros(1, 11, 11)))

    def test_process_multiple_allies_same_position(self):
        """Test AlliesHPHandler with multiple allies at same position."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        allies = [
            (7, 5, 0.6),  # Ally at relative (2, 0) with lower health
            (7, 5, 0.8),  # Same position with higher health
        ]

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, allies=allies
        )

        # Should take the maximum health value
        self.assertAlmostEqual(observation[0, 7, 5].item(), 0.8)


class TestEnemiesHPHandler(unittest.TestCase):
    """Test cases for EnemiesHPHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = EnemiesHPHandler()
        self.config_mock = Mock()
        self.config_mock.R = 5

    def test_initialization(self):
        """Test EnemiesHPHandler initialization."""
        self.assertEqual(self.handler.name, "ENEMIES_HP")
        self.assertEqual(self.handler.behavior, ChannelBehavior.INSTANT)

    def test_process_with_enemies(self):
        """Test EnemiesHPHandler process method with enemies."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        enemies = [
            (3, 5, 0.7),  # Enemy at relative (-2, 0) from agent
            (5, 3, 0.9),  # Enemy at relative (0, -2) from agent
        ]

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, enemies=enemies
        )

        # Check that enemies' health is written at correct positions
        self.assertAlmostEqual(observation[0, 3, 5].item(), 0.7)  # First enemy
        self.assertAlmostEqual(observation[0, 5, 3].item(), 0.9)  # Second enemy

    def test_process_no_enemies(self):
        """Test EnemiesHPHandler with no enemies."""
        observation = torch.ones(1, 11, 11)
        agent_world_pos = (5, 5)

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, enemies=[]
        )

        # Observation should remain unchanged
        self.assertTrue(torch.allclose(observation, torch.ones(1, 11, 11)))


class TestWorldLayerHandler(unittest.TestCase):
    """Test cases for WorldLayerHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = WorldLayerHandler("TEST_LAYER", "TEST_LAYER")
        self.config_mock = Mock()
        self.config_mock.R = 5
        self.config_mock.pad_val = 0.0
        self.config_mock.device = torch.device("cpu")
        self.config_mock.torch_dtype = torch.float32
        self.config_mock.get_local_observation_size = Mock(return_value=(11, 11))

    def test_initialization(self):
        """Test WorldLayerHandler initialization."""
        self.assertEqual(self.handler.name, "TEST_LAYER")
        self.assertEqual(self.handler.behavior, ChannelBehavior.INSTANT)
        self.assertEqual(self.handler.layer_key, "TEST_LAYER")

    def test_process_with_layer_data(self):
        """Test WorldLayerHandler process method with layer data."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        # Mock world layer data
        world_layers = {"TEST_LAYER": torch.ones(20, 20)}  # 20x20 world layer

        with patch("farm.core.observations.crop_local") as mock_crop:
            mock_crop.return_value = torch.full((11, 11), 0.5)

            self.handler.process(
                observation,
                0,
                self.config_mock,
                agent_world_pos,
                world_layers=world_layers,
            )

            # Check that crop_local was called correctly
            mock_crop.assert_called_once()
            args = mock_crop.call_args[0]
            self.assertEqual(args[1], agent_world_pos)  # agent_world_pos
            self.assertEqual(args[2], 5)  # R

            # Check that observation was updated
            self.assertTrue(torch.allclose(observation[0], torch.full((11, 11), 0.5)))

    def test_process_missing_layer_data(self):
        """Test WorldLayerHandler with missing layer data."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        world_layers = {}  # Empty world layers

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, world_layers=world_layers
        )

        # Should not change observation
        self.assertTrue(torch.allclose(observation, torch.zeros(1, 11, 11)))


class TestVisibilityHandler(unittest.TestCase):
    """Test cases for VisibilityHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = VisibilityHandler()
        self.config_mock = Mock()
        self.config_mock.R = 5
        self.config_mock.fov_radius = 5
        self.config_mock.device = torch.device("cpu")
        self.config_mock.torch_dtype = torch.float32

    def test_initialization(self):
        """Test VisibilityHandler initialization."""
        self.assertEqual(self.handler.name, "VISIBILITY")
        self.assertEqual(self.handler.behavior, ChannelBehavior.INSTANT)

    def test_process(self):
        """Test VisibilityHandler process method."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        with patch("farm.core.observations.make_disk_mask") as mock_disk:
            mock_disk.return_value = torch.full((11, 11), 0.7)

            self.handler.process(observation, 0, self.config_mock, agent_world_pos)

            # Check that make_disk_mask was called correctly
            mock_disk.assert_called_once()
            args = mock_disk.call_args[0]
            self.assertEqual(args[0], 11)  # 2*R + 1
            self.assertEqual(args[1], 5)  # min(fov_radius, R)

            # Check that observation was updated
            self.assertTrue(torch.allclose(observation[0], torch.full((11, 11), 0.7)))


class TestKnownEmptyHandler(unittest.TestCase):
    """Test cases for KnownEmptyHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = KnownEmptyHandler()
        self.config_mock = Mock()
        self.config_mock.gamma_known = 0.9

    def test_initialization(self):
        """Test KnownEmptyHandler initialization."""
        self.assertEqual(self.handler.name, "KNOWN_EMPTY")
        self.assertEqual(self.handler.behavior, ChannelBehavior.DYNAMIC)
        self.assertIsNone(self.handler.gamma)  # Uses config gamma

    def test_decay_with_config_gamma(self):
        """Test decay method with config gamma."""
        observation = torch.ones(1, 11, 11)

        self.handler.decay(observation, 0, config=self.config_mock)

        expected = torch.ones(1, 11, 11) * 0.9
        self.assertTrue(torch.allclose(observation, expected))

    def test_decay_without_config_gamma(self):
        """Test decay method without config gamma."""
        observation = torch.ones(1, 11, 11)

        # Test with config that doesn't have gamma_known attribute
        config_no_gamma = Mock()
        del config_no_gamma.gamma_known  # Remove the attribute entirely

        self.handler.decay(observation, 0, config=config_no_gamma)

        # Should not decay without gamma
        self.assertTrue(torch.allclose(observation, torch.ones(1, 11, 11)))


class TestTransientEventHandler(unittest.TestCase):
    """Test cases for TransientEventHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = TransientEventHandler("TEST_EVENT", "test_events", "gamma_test")
        self.config_mock = Mock()
        self.config_mock.R = 5

    def test_initialization(self):
        """Test TransientEventHandler initialization."""
        self.assertEqual(self.handler.name, "TEST_EVENT")
        self.assertEqual(self.handler.behavior, ChannelBehavior.DYNAMIC)
        self.assertEqual(self.handler.data_key, "test_events")
        self.assertEqual(self.handler.config_gamma_key, "gamma_test")

    def test_decay_with_config_gamma(self):
        """Test decay method with config gamma."""
        observation = torch.ones(1, 11, 11)
        config_with_gamma = Mock()
        config_with_gamma.gamma_test = 0.7

        self.handler.decay(observation, 0, config=config_with_gamma)

        expected = torch.ones(1, 11, 11) * 0.7
        self.assertTrue(torch.allclose(observation, expected))

    def test_process_with_events(self):
        """Test TransientEventHandler process method with events."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        events = [
            (7, 5, 0.8),  # Event at relative (2, 0) from agent
            (5, 7, 0.6),  # Event at relative (0, 2) from agent
        ]

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, test_events=events
        )

        # Check that events are written at correct positions
        self.assertAlmostEqual(observation[0, 7, 5].item(), 0.8)  # First event
        self.assertAlmostEqual(observation[0, 5, 7].item(), 0.6)  # Second event

    def test_process_multiple_events_same_position(self):
        """Test TransientEventHandler with multiple events at same position."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        events = [
            (7, 5, 0.5),  # Event at relative (2, 0) with lower intensity
            (7, 5, 0.8),  # Same position with higher intensity
        ]

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, test_events=events
        )

        # Should take the maximum intensity
        self.assertAlmostEqual(observation[0, 7, 5].item(), 0.8)


class TestGoalHandler(unittest.TestCase):
    """Test cases for GoalHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = GoalHandler()
        self.config_mock = Mock()
        self.config_mock.R = 5

    def test_initialization(self):
        """Test GoalHandler initialization."""
        self.assertEqual(self.handler.name, "GOAL")
        self.assertEqual(self.handler.behavior, ChannelBehavior.INSTANT)

    def test_process_with_goal(self):
        """Test GoalHandler process method with goal."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)
        goal_world_pos = (7, 5)  # Goal at relative (2, 0) from agent

        self.handler.process(
            observation,
            0,
            self.config_mock,
            agent_world_pos,
            goal_world_pos=goal_world_pos,
        )

        # Should set goal position to 1.0
        self.assertEqual(observation[0, 7, 5].item(), 1.0)

    def test_process_no_goal(self):
        """Test GoalHandler with no goal."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        self.handler.process(observation, 0, self.config_mock, agent_world_pos)

        # Should not change observation
        self.assertTrue(torch.allclose(observation, torch.zeros(1, 11, 11)))

    def test_process_goal_outside_bounds(self):
        """Test GoalHandler with goal outside observation bounds."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)
        goal_world_pos = (20, 20)  # Way outside bounds

        self.handler.process(
            observation,
            0,
            self.config_mock,
            agent_world_pos,
            goal_world_pos=goal_world_pos,
        )

        # Should not write anything
        self.assertTrue(torch.allclose(observation, torch.zeros(1, 11, 11)))


class TestLandmarkHandler(unittest.TestCase):
    """Test cases for LandmarkHandler."""

    def setUp(self):
        """Set up test fixtures."""
        self.handler = LandmarkHandler()
        self.config_mock = Mock()
        self.config_mock.R = 5

    def test_initialization(self):
        """Test LandmarkHandler initialization."""
        self.assertEqual(self.handler.name, "LANDMARKS")
        self.assertEqual(self.handler.behavior, ChannelBehavior.PERSISTENT)

    def test_process_with_landmarks(self):
        """Test LandmarkHandler process method with landmarks."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        landmarks = [
            (7, 5, 0.8),  # Landmark at relative (2, 0) from agent
            (5, 7, 0.6),  # Landmark at relative (0, 2) from agent
        ]

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, landmarks_world=landmarks
        )

        # Check that landmarks are written at correct positions
        self.assertAlmostEqual(observation[0, 7, 5].item(), 0.8)  # First landmark
        self.assertAlmostEqual(observation[0, 5, 7].item(), 0.6)  # Second landmark

    def test_process_no_landmarks(self):
        """Test LandmarkHandler with no landmarks."""
        observation = torch.ones(1, 11, 11)
        agent_world_pos = (5, 5)

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, landmarks_world=[]
        )

        # Observation should remain unchanged
        self.assertTrue(torch.allclose(observation, torch.ones(1, 11, 11)))

    def test_process_landmark_outside_bounds(self):
        """Test LandmarkHandler with landmark outside observation bounds."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        landmarks = [
            (20, 20, 0.8),  # Way outside bounds
        ]

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, landmarks_world=landmarks
        )

        # Should not write anything (all values remain 0)
        self.assertTrue(torch.allclose(observation, torch.zeros(1, 11, 11)))

    def test_process_multiple_landmarks_same_position(self):
        """Test LandmarkHandler with multiple landmarks at same position."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        landmarks = [
            (7, 5, 0.6),  # Landmark at relative (2, 0) with lower importance
            (7, 5, 0.8),  # Same position with higher importance
        ]

        self.handler.process(
            observation, 0, self.config_mock, agent_world_pos, landmarks_world=landmarks
        )

        # Should take the maximum importance
        self.assertAlmostEqual(observation[0, 7, 5].item(), 0.8)

    def test_process_accumulates_landmarks(self):
        """Test LandmarkHandler accumulates landmarks across multiple calls."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        # First call with one landmark
        landmarks1 = [(7, 5, 0.6)]
        self.handler.process(
            observation,
            0,
            self.config_mock,
            agent_world_pos,
            landmarks_world=landmarks1,
        )

        # Second call with landmark at different position
        landmarks2 = [(5, 7, 0.8)]
        self.handler.process(
            observation,
            0,
            self.config_mock,
            agent_world_pos,
            landmarks_world=landmarks2,
        )

        # Both landmarks should be present
        self.assertAlmostEqual(observation[0, 7, 5].item(), 0.6)
        self.assertAlmostEqual(observation[0, 5, 7].item(), 0.8)

    def test_process_accumulates_at_same_position(self):
        """Test LandmarkHandler accumulates importance at same position."""
        observation = torch.zeros(1, 11, 11)
        agent_world_pos = (5, 5)

        # First call with landmark at position
        landmarks1 = [(7, 5, 0.6)]
        self.handler.process(
            observation,
            0,
            self.config_mock,
            agent_world_pos,
            landmarks_world=landmarks1,
        )

        # Second call with higher importance at same position
        landmarks2 = [(7, 5, 0.8)]
        self.handler.process(
            observation,
            0,
            self.config_mock,
            agent_world_pos,
            landmarks_world=landmarks2,
        )

        # Should have the maximum importance
        self.assertAlmostEqual(observation[0, 7, 5].item(), 0.8)


class TestChannelEnum(unittest.TestCase):
    """Test cases for backward compatibility Channel enum."""

    def test_channel_enum_values(self):
        """Test that Channel enum has correct values."""
        self.assertEqual(Channel.SELF_HP.value, 0)
        self.assertEqual(Channel.ALLIES_HP.value, 1)
        self.assertEqual(Channel.ENEMIES_HP.value, 2)
        self.assertEqual(Channel.RESOURCES.value, 3)
        self.assertEqual(Channel.OBSTACLES.value, 4)
        self.assertEqual(Channel.TERRAIN_COST.value, 5)
        self.assertEqual(Channel.VISIBILITY.value, 6)
        self.assertEqual(Channel.KNOWN_EMPTY.value, 7)
        self.assertEqual(Channel.DAMAGE_HEAT.value, 8)
        self.assertEqual(Channel.TRAILS.value, 9)
        self.assertEqual(Channel.ALLY_SIGNAL.value, 10)
        self.assertEqual(Channel.GOAL.value, 11)

    def test_channel_enum_order(self):
        """Test that Channel enum values are in sequential order."""
        channels = [
            Channel.SELF_HP,
            Channel.ALLIES_HP,
            Channel.ENEMIES_HP,
            Channel.RESOURCES,
            Channel.OBSTACLES,
            Channel.TERRAIN_COST,
            Channel.VISIBILITY,
            Channel.KNOWN_EMPTY,
            Channel.DAMAGE_HEAT,
            Channel.TRAILS,
            Channel.ALLY_SIGNAL,
            Channel.GOAL,
        ]

        for i, channel in enumerate(channels):
            self.assertEqual(channel.value, i)


class TestNUM_CHANNELS(unittest.TestCase):
    """Test cases for NUM_CHANNELS constant."""

    def test_num_channels_value(self):
        """Test that NUM_CHANNELS has correct value."""
        # Since we register core channels on import, it should be 13
        # (includes the new LANDMARKS channel added for persistent landmarks)
        self.assertEqual(NUM_CHANNELS, 13)

    def test_num_channels_matches_registry(self):
        """Test that NUM_CHANNELS matches the registry's num_channels."""
        registry = get_channel_registry()
        # The registry may have core channels pre-registered from module import
        # So we check that NUM_CHANNELS is at least the registry's count
        self.assertGreaterEqual(NUM_CHANNELS, registry.num_channels)


class TestIntegration(unittest.TestCase):
    """Integration tests for the channels module."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset the global registry
        from farm.core.channels import _global_registry

        _global_registry._handlers.clear()
        _global_registry._name_to_index.clear()
        _global_registry._index_to_name.clear()
        _global_registry._next_index = 0

    def test_full_channel_system_workflow(self):
        """Test a complete workflow of the channel system."""

        # Register a custom handler
        class CustomHandler(ChannelHandler):
            def __init__(self):
                super().__init__("CUSTOM", ChannelBehavior.DYNAMIC, gamma=0.9)

            def process(
                self, observation, channel_idx, config, agent_world_pos, **kwargs
            ):
                # Use utility method for consistent sparse storage handling
                self._safe_store_sparse_point(observation, channel_idx, 0, 0, 42.0)

        custom_handler = CustomHandler()
        register_channel(custom_handler)

        # Get the registry and verify registration
        registry = get_channel_registry()
        self.assertIn("CUSTOM", registry._handlers)
        # Index may not be 0 if core channels are already registered
        custom_index = registry.get_index("CUSTOM")
        self.assertIsInstance(custom_index, int)

        # Test processing
        observation = torch.zeros(1, 11, 11)
        config_mock = Mock()
        config_mock.R = 5
        agent_world_pos = (5, 5)

        custom_handler.process(observation, custom_index, config_mock, agent_world_pos)
        self.assertEqual(observation[custom_index, 0, 0].item(), 42.0)

        # Test decay
        custom_handler.decay(observation, custom_index)
        self.assertAlmostEqual(
            observation[custom_index, 0, 0].item(), 42.0 * 0.9, places=5
        )

    def test_registry_operations_workflow(self):
        """Test registry operations in sequence."""

        # Create and register multiple handlers
        class TestHandler(ChannelHandler):
            def __init__(self, name):
                super().__init__(name, ChannelBehavior.INSTANT)

            def process(
                self, observation, channel_idx, config, agent_world_pos, **kwargs
            ):
                # Test handler - no-op implementation
                pass

        handler1 = TestHandler("HANDLER_1")
        handler2 = TestHandler("HANDLER_2")
        handler3 = TestHandler("HANDLER_3")

        # Register handlers
        index1 = register_channel(handler1)
        index2 = register_channel(
            handler2, index=15
        )  # Specific index far from core channels
        index3 = register_channel(handler3)  # Auto-assigned

        # Verify indices are valid
        self.assertIsInstance(index1, int)
        self.assertEqual(index2, 15)
        self.assertIsInstance(index3, int)
        self.assertGreater(index3, index2)  # Should be after the specific index

        # Test lookup operations
        registry = get_channel_registry()
        self.assertEqual(registry.get_index("HANDLER_1"), index1)
        self.assertEqual(registry.get_name(15), "HANDLER_2")
        self.assertEqual(registry.get_handler("HANDLER_3"), handler3)

        # Test batch operations
        # Create observation tensor large enough for all registered channels
        observation = torch.ones(registry.max_index + 1, 11, 11)
        config_mock = Mock()

        registry.apply_decay(
            observation, config_mock
        )  # Should not change anything (all INSTANT)
        registry.clear_instant(observation)  # Should clear all channels

        # Check that the registered channels were cleared to zero
        for name, handler in registry.get_all_handlers().items():
            channel_idx = registry.get_index(name)
            if handler.behavior == ChannelBehavior.INSTANT:
                # INSTANT channels should be cleared
                self.assertTrue(
                    torch.allclose(observation[channel_idx], torch.zeros(11, 11))
                )


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for StateManager.

Tests verify:
- Position management works correctly
- Orientation management works correctly
- Lifecycle tracking is accurate
- Genealogy data is stored properly
- State serialization works
"""

import pytest
from unittest.mock import Mock, MagicMock
from farm.core.agent.state.state_manager import StateManager


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock()
    agent.agent_id = "test_agent_001"
    agent._spatial_service = Mock()
    agent._time_service = Mock()
    agent._time_service.current_time.return_value = 100
    return agent


@pytest.fixture
def state_manager(mock_agent):
    """Create a StateManager with mock agent."""
    return StateManager(mock_agent)


class TestPositionManagement:
    """Tests for position management."""

    def test_initial_position(self, state_manager):
        """Test initial position is (0, 0)."""
        assert state_manager.position == (0.0, 0.0)
        assert state_manager.position_3d == (0.0, 0.0, 0.0)

    def test_set_position_2d(self, state_manager):
        """Test setting 2D position."""
        state_manager.set_position((10.5, 20.3))
        assert state_manager.position == (10.5, 20.3)
        assert state_manager.position_3d == (10.5, 20.3, 0.0)

    def test_set_position_3d(self, state_manager):
        """Test setting 3D position."""
        state_manager.set_position((10.5, 20.3, 5.0))
        assert state_manager.position == (10.5, 20.3)
        assert state_manager.position_3d == (10.5, 20.3, 5.0)

    def test_position_change_marks_spatial_dirty(self, state_manager, mock_agent):
        """Test that position change marks spatial index dirty."""
        state_manager.set_position((5.0, 5.0))
        mock_agent._spatial_service.mark_positions_dirty.assert_called_once()

    def test_same_position_marks_spatial_dirty(self, state_manager, mock_agent):
        """Test that setting same position doesn't mark spatial dirty."""
        state_manager.set_position((0.0, 0.0, 0.0))
        # Should not be called since position didn't change
        mock_agent._spatial_service.mark_positions_dirty.assert_not_called()


class TestOrientationManagement:
    """Tests for orientation management."""

    def test_initial_orientation(self, state_manager):
        """Test initial orientation is 0."""
        assert state_manager.orientation == 0.0

    def test_set_orientation(self, state_manager):
        """Test setting orientation."""
        state_manager.set_orientation(90.0)
        assert state_manager.orientation == 90.0

    def test_orientation_normalization(self, state_manager):
        """Test orientation is normalized to 0-360."""
        state_manager.set_orientation(450.0)  # 450 % 360 = 90
        assert state_manager.orientation == 90.0

        state_manager.set_orientation(-90.0)  # -90 % 360 = 270
        assert state_manager.orientation == 270.0

    def test_rotate(self, state_manager):
        """Test rotation by delta."""
        state_manager.set_orientation(45.0)
        state_manager.rotate(90.0)
        assert state_manager.orientation == 135.0

    def test_rotate_with_wraparound(self, state_manager):
        """Test rotation wraps around at 360."""
        state_manager.set_orientation(350.0)
        state_manager.rotate(20.0)
        assert state_manager.orientation == 10.0


class TestLifecycleManagement:
    """Tests for lifecycle tracking."""

    def test_initial_birth_time(self, state_manager):
        """Test initial birth time is 0."""
        assert state_manager.birth_time == 0

    def test_set_birth_time(self, state_manager):
        """Test setting birth time."""
        state_manager.set_birth_time(50)
        assert state_manager.birth_time == 50

    def test_initial_death_time(self, state_manager):
        """Test initial death time is None."""
        assert state_manager.death_time is None

    def test_set_death_time(self, state_manager):
        """Test setting death time."""
        state_manager.set_death_time(200)
        assert state_manager.death_time == 200

    def test_age_while_alive(self, state_manager, mock_agent):
        """Test age calculation for living agent."""
        state_manager.set_birth_time(50)
        mock_agent._time_service.current_time.return_value = 150

        assert state_manager.age == 100  # 150 - 50

    def test_age_when_dead(self, state_manager):
        """Test age calculation for dead agent."""
        state_manager.set_birth_time(50)
        state_manager.set_death_time(200)

        assert state_manager.age == 150  # 200 - 50


class TestGenealogyManagement:
    """Tests for genealogy tracking."""

    def test_initial_generation(self, state_manager):
        """Test initial generation is 0."""
        assert state_manager.generation == 0

    def test_set_generation(self, state_manager):
        """Test setting generation."""
        state_manager.set_generation(3)
        assert state_manager.generation == 3

    def test_initial_genome_id(self, state_manager):
        """Test initial genome ID is empty."""
        assert state_manager.genome_id == ""

    def test_set_genome_id(self, state_manager):
        """Test setting genome ID."""
        state_manager.set_genome_id("genome_abc123")
        assert state_manager.genome_id == "genome_abc123"

    def test_initial_parent_ids(self, state_manager):
        """Test initial parent IDs is empty list."""
        assert state_manager.parent_ids == []

    def test_set_parent_ids(self, state_manager):
        """Test setting parent IDs."""
        parents = ["parent_1", "parent_2"]
        state_manager.set_parent_ids(parents)
        assert state_manager.parent_ids == parents

    def test_parent_ids_are_copied(self, state_manager):
        """Test that parent IDs are copied, not referenced."""
        parents = ["parent_1"]
        state_manager.set_parent_ids(parents)

        # Modify original list
        parents.append("parent_2")

        # Should not affect stored list
        assert state_manager.parent_ids == ["parent_1"]

    def test_parent_ids_return_copy(self, state_manager):
        """Test that getting parent IDs returns a copy."""
        state_manager.set_parent_ids(["parent_1"])

        # Get parent IDs and modify
        parents = state_manager.parent_ids
        parents.append("parent_2")

        # Should not affect stored list
        assert state_manager.parent_ids == ["parent_1"]


class TestStateSerialization:
    """Tests for state serialization."""

    def test_get_state_dict(self, state_manager):
        """Test getting state as dictionary."""
        state_manager.set_position((10.0, 20.0, 5.0))
        state_manager.set_orientation(45.0)
        state_manager.set_birth_time(50)
        state_manager.set_generation(2)
        state_manager.set_genome_id("genome_123")
        state_manager.set_parent_ids(["parent_1"])

        state = state_manager.get_state_dict()

        assert state["position"] == (10.0, 20.0, 5.0)
        assert state["orientation"] == 45.0
        assert state["birth_time"] == 50
        assert state["death_time"] is None
        assert state["generation"] == 2
        assert state["genome_id"] == "genome_123"
        assert state["parent_ids"] == ["parent_1"]

    def test_load_state_dict(self, state_manager):
        """Test loading state from dictionary."""
        state = {
            "position": (15.0, 25.0, 3.0),
            "orientation": 90.0,
            "birth_time": 100,
            "death_time": 200,
            "generation": 3,
            "genome_id": "genome_xyz",
            "parent_ids": ["parent_a", "parent_b"],
        }

        state_manager.load_state_dict(state)

        assert state_manager.position_3d == (15.0, 25.0, 3.0)
        assert state_manager.orientation == 90.0
        assert state_manager.birth_time == 100
        assert state_manager.death_time == 200
        assert state_manager.generation == 3
        assert state_manager.genome_id == "genome_xyz"
        assert state_manager.parent_ids == ["parent_a", "parent_b"]

    def test_round_trip_serialization(self, state_manager):
        """Test that save/load preserves state."""
        # Set up state
        state_manager.set_position((10.0, 20.0, 5.0))
        state_manager.set_orientation(45.0)
        state_manager.set_birth_time(50)

        # Save state
        state = state_manager.get_state_dict()

        # Create new manager and load
        new_manager = StateManager(Mock())
        new_manager.load_state_dict(state)

        # Verify
        assert new_manager.position_3d == (10.0, 20.0, 5.0)
        assert new_manager.orientation == 45.0
        assert new_manager.birth_time == 50
"""
Unit tests for agent configuration value objects.

Tests verify:
- Default values are correct
- Validation works properly
- Immutability is enforced
- from_dict() creates correct configs
"""

import pytest
from farm.core.agent.config.agent_config import (
    AgentConfig,
    MovementConfig,
    ResourceConfig,
    CombatConfig,
    ReproductionConfig,
    PerceptionConfig,
)


class TestMovementConfig:
    """Tests for MovementConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MovementConfig()
        assert config.max_movement == 8.0
        assert config.position_discretization_method == "floor"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MovementConfig(
            max_movement=12.0, position_discretization_method="round"
        )
        assert config.max_movement == 12.0
        assert config.position_discretization_method == "round"

    def test_immutability(self):
        """Test that config is immutable."""
        config = MovementConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.max_movement = 10.0

    def test_validation_negative_movement(self):
        """Test validation rejects negative movement."""
        with pytest.raises(ValueError, match="max_movement must be non-negative"):
            MovementConfig(max_movement=-1.0)

    def test_validation_invalid_discretization(self):
        """Test validation rejects invalid discretization method."""
        with pytest.raises(ValueError, match="Invalid discretization method"):
            MovementConfig(position_discretization_method="invalid")


class TestResourceConfig:
    """Tests for ResourceConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ResourceConfig()
        assert config.base_consumption_rate == 1
        assert config.starvation_threshold == 100
        assert config.initial_resources == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ResourceConfig(
            base_consumption_rate=2, starvation_threshold=50, initial_resources=20
        )
        assert config.base_consumption_rate == 2
        assert config.starvation_threshold == 50
        assert config.initial_resources == 20

    def test_validation_negative_consumption(self):
        """Test validation rejects negative consumption."""
        with pytest.raises(
            ValueError, match="base_consumption_rate must be non-negative"
        ):
            ResourceConfig(base_consumption_rate=-1)

    def test_validation_negative_threshold(self):
        """Test validation rejects negative threshold."""
        with pytest.raises(
            ValueError, match="starvation_threshold must be non-negative"
        ):
            ResourceConfig(starvation_threshold=-1)


class TestCombatConfig:
    """Tests for CombatConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CombatConfig()
        assert config.starting_health == 100.0
        assert config.base_attack_strength == 10.0
        assert config.base_defense_strength == 5.0
        assert config.defense_reduction == 0.5
        assert config.defense_duration == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CombatConfig(
            starting_health=150.0,
            base_attack_strength=15.0,
            defense_reduction=0.3,
        )
        assert config.starting_health == 150.0
        assert config.base_attack_strength == 15.0
        assert config.defense_reduction == 0.3

    def test_validation_zero_health(self):
        """Test validation rejects zero or negative health."""
        with pytest.raises(ValueError, match="starting_health must be positive"):
            CombatConfig(starting_health=0)

    def test_validation_defense_reduction_out_of_range(self):
        """Test validation rejects defense reduction outside 0-1."""
        with pytest.raises(
            ValueError, match="defense_reduction must be between 0 and 1"
        ):
            CombatConfig(defense_reduction=1.5)


class TestReproductionConfig:
    """Tests for ReproductionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ReproductionConfig()
        assert config.offspring_cost == 5
        assert config.offspring_initial_resources == 10
        assert config.reproduction_threshold == 8

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ReproductionConfig(
            offspring_cost=10, offspring_initial_resources=15, reproduction_threshold=30
        )
        assert config.offspring_cost == 10
        assert config.offspring_initial_resources == 15
        assert config.reproduction_threshold == 30


class TestPerceptionConfig:
    """Tests for PerceptionConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PerceptionConfig()
        assert config.perception_radius == 5

    def test_grid_size_calculation(self):
        """Test that grid size is calculated correctly from radius."""
        config = PerceptionConfig(perception_radius=5)
        assert config.perception_grid_size == 11  # 2 * 5 + 1

        config = PerceptionConfig(perception_radius=10)
        assert config.perception_grid_size == 21  # 2 * 10 + 1


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_values(self):
        """Test default configuration with all sub-configs."""
        config = AgentConfig()
        assert isinstance(config.movement, MovementConfig)
        assert isinstance(config.resource, ResourceConfig)
        assert isinstance(config.combat, CombatConfig)
        assert isinstance(config.reproduction, ReproductionConfig)
        assert isinstance(config.perception, PerceptionConfig)

    def test_custom_sub_configs(self):
        """Test creating config with custom sub-configs."""
        config = AgentConfig(
            movement=MovementConfig(max_movement=15.0),
            resource=ResourceConfig(base_consumption_rate=3),
        )
        assert config.movement.max_movement == 15.0
        assert config.resource.base_consumption_rate == 3

    def test_from_dict_empty(self):
        """Test from_dict with empty dictionary uses defaults."""
        config = AgentConfig.from_dict({})
        assert config.movement.max_movement == 8.0
        assert config.resource.base_consumption_rate == 1

    def test_from_dict_partial(self):
        """Test from_dict with partial configuration."""
        config = AgentConfig.from_dict(
            {
                "movement": {"max_movement": 12.0},
                "resource": {"base_consumption_rate": 2},
            }
        )
        assert config.movement.max_movement == 12.0
        assert config.resource.base_consumption_rate == 2
        # Other configs should use defaults
        assert config.combat.starting_health == 100.0

    def test_from_dict_complete(self):
        """Test from_dict with complete configuration."""
        config = AgentConfig.from_dict(
            {
                "movement": {"max_movement": 12.0},
                "resource": {"base_consumption_rate": 2},
                "combat": {"starting_health": 150.0},
                "reproduction": {"offspring_cost": 10},
                "perception": {"perception_radius": 8},
            }
        )
        assert config.movement.max_movement == 12.0
        assert config.resource.base_consumption_rate == 2
        assert config.combat.starting_health == 150.0
        assert config.reproduction.offspring_cost == 10
        assert config.perception.perception_radius == 8

    def test_immutability(self):
        """Test that AgentConfig is immutable."""
        config = AgentConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.movement = MovementConfig(max_movement=20.0)
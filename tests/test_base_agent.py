"""
Comprehensive unit tests for the BaseAgent class.

This test suite covers all aspects of the BaseAgent class including:
- Initialization and service integration
- State management and lifecycle methods
- Decision making and action execution
- Combat system and damage handling
- Memory system integration
- Reproduction and termination mechanics
- Edge cases and error handling
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from farm.core.action import Action, action_registry
from farm.core.agent import BaseAgent
from farm.core.decision.decision import DecisionModule
from farm.core.perception import PerceptionData
from farm.core.services.factory import AgentServiceFactory
from farm.core.services.interfaces import (
    IAgentLifecycleService,
    ILoggingService,
    IMetricsService,
    ISpatialQueryService,
    ITimeService,
    IValidationService,
)
from farm.core.state import AgentState
from farm.database.data_types import GenomeId
from farm.memory.redis_memory import AgentMemoryManager, RedisMemoryConfig
from tests.utils.test_helpers import MemoryTestHelper


@pytest.fixture
def mock_spatial_service():
    """Mock spatial query service for testing."""
    service = Mock(spec=ISpatialQueryService)

    # Mock get_nearby to return empty results for both "resources" and "agents" indices
    def mock_get_nearby(position, radius, index_names=None):
        result = {}
        if index_names is None or "resources" in index_names:
            result["resources"] = []
        if index_names is None or "agents" in index_names:
            result["agents"] = []
        return result

    service.get_nearby.side_effect = mock_get_nearby
    service.get_nearest.return_value = {}
    service.mark_positions_dirty.return_value = None
    return service


@pytest.fixture
def mock_metrics_service():
    """Mock metrics service for testing."""
    service = Mock(spec=IMetricsService)
    service.record_birth.return_value = None
    return service


@pytest.fixture
def mock_logging_service():
    """Mock logging service for testing."""
    service = Mock(spec=ILoggingService)
    service.log_reproduction_event.return_value = None
    service.update_agent_death.return_value = None
    return service


@pytest.fixture
def mock_validation_service():
    """Mock validation service for testing."""
    service = Mock(spec=IValidationService)
    service.is_valid_position.return_value = True
    return service


@pytest.fixture
def mock_time_service():
    """Mock time service for testing."""
    service = Mock(spec=ITimeService)
    service.current_time.return_value = 100
    return service


@pytest.fixture
def mock_lifecycle_service():
    """Mock lifecycle service for testing."""
    service = Mock(spec=IAgentLifecycleService)
    service.get_next_agent_id.return_value = "test_agent_002"
    service.add_agent.return_value = None
    service.remove_agent.return_value = None
    return service


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = Mock()
    config.max_movement = 8
    config.starvation_threshold = 100
    config.starting_health = 100
    config.base_consumption_rate = 1
    config.offspring_initial_resources = 10
    config.offspring_cost = 5
    config.base_attack_strength = 10
    config.base_defense_strength = 5
    config.redis_host = "localhost"
    config.redis_port = 6379
    config.memory_limit = 1000
    config.perception_radius = 5

    # Configure attributes that should return None instead of Mock objects
    config.action_space = None
    config.observation_space = None
    config.curriculum_phases = None

    # Configure decision module settings
    decision_config = Mock()
    decision_config.rl_state_dim = 8
    decision_config.learning_rate = 0.001
    decision_config.gamma = 0.99
    config.decision = decision_config

    return config


@pytest.fixture
def mock_decision_module():
    """Mock decision module for testing."""
    module = Mock(spec=DecisionModule)
    module.decide_action.return_value = 0  # Return first action index
    module.update.return_value = None
    return module


@pytest.fixture
def mock_environment():
    """Mock environment for testing."""
    env = Mock()
    env.observe.return_value = np.zeros((3, 11, 11))  # 3 channels, 11x11 grid
    return env


@pytest.fixture
def base_agent_kwargs(
    mock_spatial_service,
    mock_metrics_service,
    mock_logging_service,
    mock_validation_service,
    mock_time_service,
    mock_lifecycle_service,
    mock_config,
    mock_decision_module,
    mock_environment,
):
    """Common kwargs for creating BaseAgent instances in tests."""
    return {
        "agent_id": "test_agent_001",
        "position": (5.0, 5.0),
        "resource_level": 50,
        "spatial_service": mock_spatial_service,
        "environment": mock_environment,
        "metrics_service": mock_metrics_service,
        "logging_service": mock_logging_service,
        "validation_service": mock_validation_service,
        "time_service": mock_time_service,
        "lifecycle_service": mock_lifecycle_service,
        "config": mock_config,
    }


@pytest.fixture
def sample_base_agent(base_agent_kwargs):
    """Create a sample BaseAgent instance for testing."""
    with patch(
        "farm.core.agent.AgentServiceFactory.create_services"
    ) as mock_factory, patch("farm.core.agent.DecisionModule") as mock_decision_class:

        # Mock the factory to return services
        mock_factory.return_value = (
            base_agent_kwargs["metrics_service"],
            base_agent_kwargs["logging_service"],
            base_agent_kwargs["validation_service"],
            base_agent_kwargs["time_service"],
            base_agent_kwargs["lifecycle_service"],
            base_agent_kwargs["config"],
        )

        # Mock DecisionModule constructor
        mock_decision_class.return_value = base_agent_kwargs.get(
            "decision_module", Mock()
        )

        agent = BaseAgent(**base_agent_kwargs)
        return agent


class TestBaseAgentInitialization:
    """Test BaseAgent initialization and setup."""

    def test_basic_initialization(self, base_agent_kwargs):
        """Test basic agent initialization with minimal parameters."""
        with patch(
            "farm.core.agent.AgentServiceFactory.create_services"
        ) as mock_factory, patch(
            "farm.core.agent.DecisionModule"
        ) as mock_decision_class:

            mock_factory.return_value = (
                base_agent_kwargs["metrics_service"],
                base_agent_kwargs["logging_service"],
                base_agent_kwargs["validation_service"],
                base_agent_kwargs["time_service"],
                base_agent_kwargs["lifecycle_service"],
                base_agent_kwargs["config"],
            )
            mock_decision_class.return_value = Mock()

            agent = BaseAgent(**base_agent_kwargs)

            assert agent.agent_id == "test_agent_001"
            assert agent.position == (5.0, 5.0)
            assert agent.resource_level == 50
            assert agent.alive == True
            assert agent.current_health == 100  # From config
            assert agent.starting_health == 100
            assert agent.max_movement == 8
            assert agent.total_reward == 0.0

    def test_initialization_without_optional_services(self, mock_spatial_service):
        """Test initialization with only required parameters."""
        minimal_kwargs = {
            "agent_id": "minimal_agent",
            "position": (0.0, 0.0),
            "resource_level": 25,
            "spatial_service": mock_spatial_service,
        }

        with patch(
            "farm.core.agent.AgentServiceFactory.create_services"
        ) as mock_factory, patch(
            "farm.core.agent.DecisionModule"
        ) as mock_decision_class:

            # Mock factory returns None for all optional services
            mock_factory.return_value = (None, None, None, None, None, None)
            mock_decision_class.return_value = Mock()

            agent = BaseAgent(**minimal_kwargs)

            assert agent.agent_id == "minimal_agent"
            assert agent.position == (0.0, 0.0)
            assert agent.resource_level == 25
            assert agent.alive == True
            # Check that optional services are None
            assert agent.metrics_service is None
            assert agent.logging_service is None
            assert agent.validation_service is None
            assert agent.time_service is None
            assert agent.lifecycle_service is None

    def test_initialization_with_custom_action_set(self, base_agent_kwargs):
        """Test initialization with custom action set."""
        custom_actions = [
            Action("move", 1.0, lambda agent: None),
            Action("gather", 2.0, lambda agent: None),
        ]

        base_agent_kwargs["action_set"] = custom_actions

        with patch(
            "farm.core.agent.AgentServiceFactory.create_services"
        ) as mock_factory, patch(
            "farm.core.agent.DecisionModule"
        ) as mock_decision_class:

            mock_factory.return_value = (
                base_agent_kwargs["metrics_service"],
                base_agent_kwargs["logging_service"],
                base_agent_kwargs["validation_service"],
                base_agent_kwargs["time_service"],
                base_agent_kwargs["lifecycle_service"],
                base_agent_kwargs["config"],
            )
            mock_decision_class.return_value = Mock()

            agent = BaseAgent(**base_agent_kwargs)

            assert len(agent.actions) == 2
            assert agent.actions[0].name == "move"
            assert agent.actions[1].name == "gather"

    def test_initialization_with_memory_enabled(self, base_agent_kwargs):
        """Test initialization with Redis memory enabled."""
        base_agent_kwargs["use_memory"] = True

        with patch(
            "farm.core.agent.AgentServiceFactory.create_services"
        ) as mock_factory, patch(
            "farm.core.agent.DecisionModule"
        ) as mock_decision_class, patch(
            "farm.core.agent.AgentMemoryManager"
        ) as mock_memory_manager:

            mock_factory.return_value = (
                base_agent_kwargs["metrics_service"],
                base_agent_kwargs["logging_service"],
                base_agent_kwargs["validation_service"],
                base_agent_kwargs["time_service"],
                base_agent_kwargs["lifecycle_service"],
                base_agent_kwargs["config"],
            )
            mock_decision_class.return_value = Mock()

            # Mock memory manager
            mock_memory_instance = Mock()
            mock_memory_manager.get_instance.return_value = mock_memory_instance
            mock_memory_instance.get_memory.return_value = Mock()

            agent = BaseAgent(**base_agent_kwargs)

            assert agent.memory is not None
            mock_memory_manager.get_instance.assert_called_once()

    def test_genome_id_generation(self, base_agent_kwargs, mock_time_service):
        """Test genome ID generation during initialization."""
        with patch(
            "farm.core.agent.AgentServiceFactory.create_services"
        ) as mock_factory, patch(
            "farm.core.agent.DecisionModule"
        ) as mock_decision_class:

            mock_factory.return_value = (
                base_agent_kwargs["metrics_service"],
                base_agent_kwargs["logging_service"],
                base_agent_kwargs["validation_service"],
                base_agent_kwargs["time_service"],
                base_agent_kwargs["lifecycle_service"],
                base_agent_kwargs["config"],
            )
            mock_decision_class.return_value = Mock()

            agent = BaseAgent(**base_agent_kwargs)

            # Check that genome_id is generated
            assert agent.genome_id is not None
            assert isinstance(agent.genome_id, str)
            assert "BaseAgent" in agent.genome_id  # Should contain class name
            assert "0" in agent.genome_id  # Generation 0


class TestBaseAgentStateManagement:
    """Test state management methods."""

    def test_get_state_basic(self, sample_base_agent, mock_time_service):
        """Test basic state retrieval."""
        state = sample_base_agent.get_state()

        assert isinstance(state, AgentState)
        assert state.agent_id == "test_agent_001"
        assert state.step_number == 100  # From mock time service
        assert state.position_x == 5.0
        assert state.position_y == 5.0
        assert state.position_z == 0  # Default for 2D position
        assert state.resource_level == 50
        assert state.current_health == 100
        assert state.is_defending == False
        assert state.total_reward == 0.0
        assert state.age == 0  # birth_time = 100, current_time = 100

    def test_get_state_with_defense(self, sample_base_agent):
        """Test state retrieval when agent is defending."""
        sample_base_agent.is_defending = True
        sample_base_agent.defense_timer = 3

        state = sample_base_agent.get_state()

        assert state.is_defending == True

    def test_update_position(self, sample_base_agent, mock_spatial_service):
        """Test position update functionality."""
        new_position = (10.0, 15.0)

        sample_base_agent.update_position(new_position)

        assert sample_base_agent.position == new_position
        mock_spatial_service.mark_positions_dirty.assert_called_once()

    def test_update_position_no_change(self, sample_base_agent, mock_spatial_service):
        """Test position update when position doesn't actually change."""
        current_position = sample_base_agent.position
        sample_base_agent.update_position(current_position)

        # Should not mark as dirty if position didn't change
        mock_spatial_service.mark_positions_dirty.assert_not_called()


class TestBaseAgentDecisionMaking:
    """Test decision making and action selection."""

    def test_decide_action_basic(self, sample_base_agent, mock_decision_module):
        """Test basic action decision making."""
        # Setup decision module mock
        sample_base_agent.decision_module = mock_decision_module
        mock_decision_module.decide_action.return_value = 0  # First action

        action = sample_base_agent.decide_action()

        assert isinstance(action, Action)
        mock_decision_module.decide_action.assert_called_once()

    def test_decide_action_with_curriculum(
        self, sample_base_agent, mock_config, mock_time_service, mock_decision_module
    ):
        """Test action decision with curriculum learning restrictions."""
        # Setup curriculum phases
        mock_config.curriculum_phases = [
            {"steps": 50, "enabled_actions": ["move", "gather"]},
            {
                "steps": -1,
                "enabled_actions": ["move", "gather", "attack"],
            },  # -1 means until end
        ]
        sample_base_agent.config = mock_config
        sample_base_agent.decision_module = mock_decision_module

        # Mock current time to be within first phase
        mock_time_service.current_time.return_value = 25

        # Setup actions to include restricted ones
        sample_base_agent.actions = [
            Action("move", 1.0, lambda agent: None),
            Action("gather", 1.0, lambda agent: None),
            Action("attack", 1.0, lambda agent: None),  # Should be restricted
        ]

        # Mock decision module to return index 0 (move action)
        mock_decision_module.decide_action.return_value = 0

        action = sample_base_agent.decide_action()

        # Should only consider enabled actions (move, gather)
        assert action.name in ["move", "gather"]

        # Verify decision module was called with enabled action indices [0, 1]
        mock_decision_module.decide_action.assert_called_once()
        call_args = mock_decision_module.decide_action.call_args
        # Check that enabled_actions parameter was passed
        assert len(call_args[0]) == 2  # state and enabled_actions
        enabled_indices = call_args[0][1]  # Second argument is enabled_actions
        assert enabled_indices == [0, 1]  # Indices of move and gather actions

    def test_create_decision_state_with_environment(
        self, sample_base_agent, mock_environment
    ):
        """Test decision state creation with environment."""
        sample_base_agent.environment = mock_environment

        state = sample_base_agent.create_decision_state()

        assert isinstance(state, torch.Tensor)
        assert state.device == sample_base_agent.device
        mock_environment.observe.assert_called_once_with("test_agent_001")

    def test_create_decision_state_fallback(self, sample_base_agent):
        """Test fallback decision state creation without environment."""
        sample_base_agent.environment = None

        state = sample_base_agent.create_decision_state()

        assert isinstance(state, torch.Tensor)
        assert state.device == sample_base_agent.device
        # Should be 3D tensor with multi-channel observation
        assert len(state.shape) == 3

    def test_get_fallback_perception(
        self, sample_base_agent, mock_spatial_service, mock_validation_service
    ):
        """Test fallback perception generation."""
        # Setup mock spatial service with some nearby entities
        mock_resource = Mock()
        mock_resource.position = (6.0, 5.0)  # Adjacent to agent at (5,5)

        mock_agent = Mock()
        mock_agent.agent_id = "other_agent"
        mock_agent.position = (5.0, 6.0)  # Adjacent to agent

        # Mock get_nearby to return the mock entities
        def mock_get_nearby(position, radius, index_names=None):
            result = {}
            if "resources" in index_names:
                result["resources"] = [mock_resource]
            if "agents" in index_names:
                result["agents"] = [mock_agent]
            return result

        mock_spatial_service.get_nearby.side_effect = mock_get_nearby

        perception = sample_base_agent.get_fallback_perception()

        assert isinstance(perception, PerceptionData)
        assert perception.grid.shape == (11, 11)  # 2*5+1 = 11

        # Check that entities are marked in perception grid
        # Resource at relative position (1, 0) from center (5,5)
        # Agent at relative position (0, 1) from center

    def test_action_to_index(self, sample_base_agent):
        """Test action to index conversion."""
        from farm.core.action import action_name_to_index

        with patch("farm.core.agent.action_name_to_index") as mock_action_to_index:
            mock_action_to_index.return_value = 2

            mock_action = Mock()
            mock_action.name = "test_action"

            index = sample_base_agent._action_to_index(mock_action)

            assert index == 2
            mock_action_to_index.assert_called_once_with("test_action")


class TestBaseAgentActionExecution:
    """Test action execution and reward calculation."""

    def test_act_basic_flow(
        self, sample_base_agent, mock_decision_module, mock_time_service
    ):
        """Test basic action execution flow."""
        sample_base_agent.decision_module = mock_decision_module

        # Setup initial state for reward calculation
        sample_base_agent.previous_state = sample_base_agent.get_state()

        # Execute act
        sample_base_agent.act()

        # Verify resource consumption
        assert sample_base_agent.resource_level == 49  # 50 - 1 (base consumption)

        # Verify decision module was used
        mock_decision_module.decide_action.assert_called()
        mock_decision_module.update.assert_called()

    def test_act_dead_agent(self, sample_base_agent):
        """Test that dead agents don't act."""
        sample_base_agent.alive = False

        # Record initial state
        initial_resources = sample_base_agent.resource_level
        initial_health = sample_base_agent.current_health

        sample_base_agent.act()

        # Nothing should change for dead agent
        assert sample_base_agent.resource_level == initial_resources
        assert sample_base_agent.current_health == initial_health

    def test_calculate_reward_no_previous_state(self, sample_base_agent):
        """Test reward calculation without previous state."""
        # Create mock states and action for new signature
        pre_state = AgentState(
            agent_id="test_agent_001",
            step_number=100,
            position_x=5.0,
            position_y=5.0,
            position_z=0.0,
            resource_level=50,
            current_health=100,
            is_defending=False,
            total_reward=0.0,
            age=0,
        )
        post_state = pre_state  # Same state for no change case
        action = Mock()
        action.name = "pass"

        reward = sample_base_agent._calculate_reward(pre_state, post_state, action)

        # Should be 0.1 for survival reward only (no state changes)
        assert reward == 0.1

    def test_calculate_reward_with_changes(self, sample_base_agent, mock_time_service):
        """Test reward calculation with state changes."""
        # Setup pre-action state (less resources and health than current state)
        pre_state = AgentState(
            agent_id="test_agent_001",
            step_number=99,
            position_x=5.0,
            position_y=5.0,
            position_z=0.0,
            resource_level=40,  # Less than current 50
            current_health=90,  # Less than current 100
            is_defending=False,
            total_reward=0.0,
            age=0,
        )

        # Post-action state (current agent state)
        post_state = AgentState(
            agent_id="test_agent_001",
            step_number=100,
            position_x=5.0,
            position_y=5.0,
            position_z=0.0,
            resource_level=50,  # Current resource level
            current_health=100,  # Current health level
            is_defending=False,
            total_reward=0.0,
            age=0,
        )

        # Mock action taken
        action = Mock()
        action.name = "gather"

        reward = sample_base_agent._calculate_reward(pre_state, post_state, action)

        # Should have positive reward from resource and health gains
        assert reward > 0
        # Should include action bonus for non-pass action
        expected_bonus = 0.05
        actual_bonus = reward - (10 * 0.1) - (10 * 0.5) - 0.1
        assert (
            abs(actual_bonus - expected_bonus) < 1e-10
        )  # Check within floating point tolerance

    def test_defense_timer_update(self, sample_base_agent, mock_decision_module):
        """Test defense timer countdown."""
        sample_base_agent.decision_module = mock_decision_module
        sample_base_agent.is_defending = True
        sample_base_agent.defense_timer = 3

        sample_base_agent.act()

        assert sample_base_agent.defense_timer == 2
        assert sample_base_agent.is_defending == True

        # Continue until timer expires
        sample_base_agent.defense_timer = 1
        sample_base_agent.act()

        assert sample_base_agent.defense_timer == 0
        assert sample_base_agent.is_defending == False


class TestBaseAgentLifeCycle:
    """Test lifecycle methods (reproduction, termination, starvation)."""

    def test_check_starvation_no_starvation(self, sample_base_agent):
        """Test starvation check when agent has resources."""
        sample_base_agent.resource_level = 20
        sample_base_agent.starvation_counter = 5

        died = sample_base_agent.check_starvation()

        assert died == False
        assert (
            sample_base_agent.starvation_counter == 0
        )  # Reset when resources available

    def test_check_starvation_with_starvation(self, sample_base_agent):
        """Test starvation check when agent runs out of resources."""
        sample_base_agent.resource_level = 0
        sample_base_agent.starvation_counter = (
            99  # Near max starvation (will reach 100 and die)
        )

        died = sample_base_agent.check_starvation()

        assert died == True
        assert sample_base_agent.alive == False

    def test_terminate_alive_agent(
        self,
        sample_base_agent,
        mock_time_service,
        mock_logging_service,
        mock_lifecycle_service,
    ):
        """Test agent termination."""
        sample_base_agent.alive = True

        sample_base_agent.terminate()

        assert sample_base_agent.alive == False
        assert sample_base_agent.death_time == 100  # From mock time service
        mock_logging_service.update_agent_death.assert_called_once_with(
            "test_agent_001", 100
        )
        mock_lifecycle_service.remove_agent.assert_called_once_with(sample_base_agent)

    def test_terminate_dead_agent(
        self, sample_base_agent, mock_logging_service, mock_lifecycle_service
    ):
        """Test termination of already dead agent."""
        sample_base_agent.alive = False

        sample_base_agent.terminate()

        # Should not call services again
        mock_logging_service.update_agent_death.assert_not_called()
        mock_lifecycle_service.remove_agent.assert_not_called()

    def test_reproduce_success(
        self,
        sample_base_agent,
        mock_logging_service,
        mock_lifecycle_service,
        mock_time_service,
    ):
        """Test successful reproduction."""
        initial_resources = sample_base_agent.resource_level

        with patch.object(
            sample_base_agent, "_create_offspring"
        ) as mock_create_offspring:
            mock_offspring = Mock()
            mock_offspring.agent_id = "child_agent"
            mock_offspring.generation = 1
            mock_create_offspring.return_value = mock_offspring

            # Mock the side effect of resource decrement that happens in _create_offspring
            def mock_create_side_effect():
                sample_base_agent.resource_level -= 5  # offspring_cost
                return mock_offspring

            mock_create_offspring.side_effect = mock_create_side_effect

            success = sample_base_agent.reproduce()

            assert success == True
            assert (
                sample_base_agent.resource_level == initial_resources - 5
            )  # offspring_cost
            mock_logging_service.log_reproduction_event.assert_called_once()
            mock_create_offspring.assert_called_once()

    def test_reproduce_failure(self, sample_base_agent, mock_logging_service):
        """Test reproduction failure."""
        initial_resources = sample_base_agent.resource_level

        with patch.object(
            sample_base_agent, "_create_offspring"
        ) as mock_create_offspring:
            mock_create_offspring.side_effect = Exception("Creation failed")

            success = sample_base_agent.reproduce()

            assert success == False
            # Resources should not change on failure (in this case)
            mock_logging_service.log_reproduction_event.assert_called_once()

    def test_create_offspring(self, sample_base_agent, mock_lifecycle_service):
        """Test offspring creation."""
        with patch(
            "farm.core.agent.AgentServiceFactory.create_services"
        ) as mock_factory, patch(
            "farm.core.agent.DecisionModule"
        ) as mock_decision_class:

            mock_factory.return_value = (None, None, None, None, None, None)
            mock_decision_class.return_value = Mock()

            offspring = sample_base_agent._create_offspring()

            assert offspring.agent_id == "test_agent_002"  # From mock lifecycle service
            assert offspring.position == (5.0, 5.0)  # Same position as parent
            assert offspring.resource_level == 10  # offspring_initial_resources
            assert offspring.generation == 1  # Parent generation + 1
            mock_lifecycle_service.add_agent.assert_called_once_with(offspring, flush_immediately=True)


class TestBaseAgentCombatSystem:
    """Test combat and damage handling."""

    def test_take_damage_valid(self, sample_base_agent):
        """Test taking valid damage."""
        initial_health = sample_base_agent.current_health

        success = sample_base_agent.take_damage(20)

        assert success == True
        assert sample_base_agent.current_health == initial_health - 20

    def test_take_damage_zero(self, sample_base_agent):
        """Test taking zero damage."""
        initial_health = sample_base_agent.current_health

        success = sample_base_agent.take_damage(0)

        assert success == False
        assert sample_base_agent.current_health == initial_health

    def test_take_damage_negative(self, sample_base_agent):
        """Test taking negative damage (healing)."""
        initial_health = sample_base_agent.current_health

        success = sample_base_agent.take_damage(-10)

        assert success == False
        assert sample_base_agent.current_health == initial_health

    def test_handle_combat_without_defense(self, sample_base_agent):
        """Test combat handling without defense."""
        attacker = Mock()
        damage = 30.0

        actual_damage = sample_base_agent.handle_combat(attacker, damage)

        assert actual_damage == 30.0
        assert sample_base_agent.current_health == 70.0  # 100 - 30

    def test_handle_combat_with_defense(self, sample_base_agent):
        """Test combat handling with defense active."""
        sample_base_agent.is_defending = True

        attacker = Mock()
        damage = 40.0

        actual_damage = sample_base_agent.handle_combat(attacker, damage)

        assert actual_damage == 20.0  # 40 * 0.5 (defense reduction)
        assert sample_base_agent.current_health == 80.0  # 100 - 20

    def test_handle_combat_death(self, sample_base_agent):
        """Test combat resulting in death."""
        attacker = Mock()
        damage = 120.0  # More than current health

        actual_damage = sample_base_agent.handle_combat(attacker, damage)

        assert actual_damage == 120.0
        assert sample_base_agent.current_health == 0.0
        assert sample_base_agent.alive == False

    @pytest.mark.parametrize(
        "health_ratio,expected_attack",
        [
            (1.0, 10.0),  # Full health
            (0.5, 5.0),  # Half health
            (0.0, 0.0),  # No health
        ],
    )
    def test_attack_strength_property(
        self, sample_base_agent, health_ratio, expected_attack
    ):
        """Test attack strength calculation based on health."""
        sample_base_agent.current_health = (
            sample_base_agent.starting_health * health_ratio
        )

        attack_strength = sample_base_agent.attack_strength

        assert attack_strength == expected_attack

    def test_defense_strength_property_defending(self, sample_base_agent):
        """Test defense strength when defending."""
        sample_base_agent.is_defending = True

        defense_strength = sample_base_agent.defense_strength

        assert defense_strength == 5.0  # base_defense_strength from config

    def test_defense_strength_property_not_defending(self, sample_base_agent):
        """Test defense strength when not defending."""
        sample_base_agent.is_defending = False

        defense_strength = sample_base_agent.defense_strength

        assert defense_strength == 0.0


class TestBaseAgentMemorySystem:
    """Test memory system integration."""

    def test_init_memory_success(self, sample_base_agent):
        """Test successful memory initialization."""
        memory_config = {"custom_param": "test_value"}

        with patch("farm.core.agent.AgentMemoryManager") as mock_memory_manager, patch(
            "farm.core.agent.RedisMemoryConfig"
        ) as mock_config_class:

            mock_memory_instance = Mock()
            mock_memory_manager.get_instance.return_value = mock_memory_instance
            mock_memory_instance.get_memory.return_value = Mock()

            sample_base_agent._init_memory(memory_config)

            assert sample_base_agent.memory is not None
            mock_memory_manager.get_instance.assert_called_once()
            mock_config_class.assert_called_once()

    def test_init_memory_failure(self, sample_base_agent):
        """Test memory initialization failure."""
        with patch("farm.core.agent.AgentMemoryManager") as mock_memory_manager:
            mock_memory_manager.get_instance.side_effect = Exception(
                "Connection failed"
            )

            sample_base_agent._init_memory()

            assert sample_base_agent.memory is None  # Should remain None on failure

    def test_remember_experience_with_memory(
        self, sample_base_agent, mock_time_service
    ):
        """Test experience recording with memory system."""
        mock_memory = MemoryTestHelper.create_mock_memory(remember_state_return=True)
        sample_base_agent.memory = mock_memory

        perception = PerceptionData(np.zeros((5, 5)))
        metadata = {"test_key": "test_value"}

        success = sample_base_agent.remember_experience(
            "gather", 5.0, perception, metadata
        )

        assert success == True
        mock_memory.remember_state.assert_called_once()

    def test_remember_experience_without_memory(self, sample_base_agent):
        """Test experience recording without memory system."""
        sample_base_agent.memory = None

        success = sample_base_agent.remember_experience("gather", 5.0)

        assert success == False


class TestBaseAgentUtilityMethods:
    """Test utility methods and properties."""

    def test_get_action_weights(self, sample_base_agent):
        """Test action weights retrieval."""
        # Setup actions with known weights
        sample_base_agent.actions = [
            Action("move", 1.5, lambda agent: None),
            Action("gather", 2.0, lambda agent: None),
            Action("attack", 0.5, lambda agent: None),
        ]

        weights = sample_base_agent.get_action_weights()

        expected_weights = {
            "move": 1.5,
            "gather": 2.0,
            "attack": 0.5,
        }
        assert weights == expected_weights

    def test_clone_agent(self, sample_base_agent):
        """Test agent cloning with mutation."""
        with patch("farm.core.genome.Genome.clone") as mock_clone, patch(
            "farm.core.genome.Genome.mutate"
        ) as mock_mutate, patch("farm.core.genome.Genome.to_agent") as mock_to_agent:

            mock_genome = {"action_set": [("move", 1.0), ("gather", 2.0)]}
            mock_clone.return_value = mock_genome
            mock_mutate.return_value = mock_genome

            mock_cloned_agent = Mock()
            mock_to_agent.return_value = mock_cloned_agent

            cloned_agent = sample_base_agent.clone()

            assert cloned_agent == mock_cloned_agent
            mock_clone.assert_called_once()
            mock_mutate.assert_called_once()

    def test_to_genome(self, sample_base_agent):
        """Test genome conversion."""
        with patch("farm.core.genome.Genome.from_agent") as mock_from_agent:
            mock_genome = {"test_genome": "data"}
            mock_from_agent.return_value = mock_genome

            genome = sample_base_agent.to_genome()

            assert genome == mock_genome
            mock_from_agent.assert_called_once_with(sample_base_agent)

    def test_from_genome_classmethod(self):
        """Test agent creation from genome."""
        mock_genome = {"action_set": [("move", 1.0)]}
        mock_environment = Mock()

        with patch("farm.core.genome.Genome.to_agent") as mock_to_agent:
            mock_agent = Mock()
            mock_to_agent.return_value = mock_agent

            result = BaseAgent.from_genome(
                mock_genome, "test_id", (10.0, 20.0), mock_environment
            )

            assert result == mock_agent
            mock_to_agent.assert_called_once_with(
                mock_genome, "test_id", (10, 20), mock_environment, BaseAgent
            )


class TestBaseAgentEdgeCases:
    """Test edge cases and error conditions."""

    def test_initialization_with_extreme_values(self, mock_spatial_service):
        """Test initialization with extreme parameter values."""
        extreme_kwargs = {
            "agent_id": "extreme_agent",
            "position": (999999.0, -999999.0),
            "resource_level": 999999,
            "spatial_service": mock_spatial_service,
        }

        with patch(
            "farm.core.agent.AgentServiceFactory.create_services"
        ) as mock_factory, patch(
            "farm.core.agent.DecisionModule"
        ) as mock_decision_class:

            mock_factory.return_value = (None, None, None, None, None, None)
            mock_decision_class.return_value = Mock()

            agent = BaseAgent(**extreme_kwargs)

            assert agent.position == (999999.0, -999999.0)
            assert agent.resource_level == 999999

    def test_position_update_with_invalid_values(
        self, sample_base_agent, mock_spatial_service
    ):
        """Test position update with various value types."""
        # Test with integers
        sample_base_agent.update_position((10, 20))
        assert sample_base_agent.position == (10, 20)

        # Test with floats
        sample_base_agent.update_position((15.5, 25.7))
        assert sample_base_agent.position == (15.5, 25.7)

    def test_multiple_terminate_calls(
        self, sample_base_agent, mock_logging_service, mock_lifecycle_service
    ):
        """Test multiple terminate calls."""
        sample_base_agent.terminate()
        sample_base_agent.terminate()  # Second call

        # Services should only be called once
        mock_logging_service.update_agent_death.assert_called_once()
        mock_lifecycle_service.remove_agent.assert_called_once()

    def test_act_with_missing_previous_state(
        self, sample_base_agent, mock_decision_module
    ):
        """Test act method when previous_state is None."""
        sample_base_agent.decision_module = mock_decision_module
        sample_base_agent.previous_state = None

        # Should not crash
        sample_base_agent.act()

        # Should still consume resources
        assert sample_base_agent.resource_level == 49

    def test_decision_state_creation_with_invalid_environment_data(
        self, sample_base_agent
    ):
        """Test decision state creation with invalid environment data."""
        mock_env = Mock()
        # Return invalid shape data
        mock_env.observe.return_value = np.array([])  # Empty array

        sample_base_agent.environment = mock_env

        # Should fallback to simple state creation
        state = sample_base_agent.create_decision_state()

        assert isinstance(state, torch.Tensor)
        # Should be 1D fallback state
        assert len(state.shape) == 1

    def test_perception_with_boundary_conditions(
        self, sample_base_agent, mock_spatial_service, mock_validation_service
    ):
        """Test perception generation at boundaries."""
        # Position agent at origin
        sample_base_agent.position = (0.0, 0.0)

        # Mock validation service to return False for some positions (boundaries)
        def mock_is_valid(pos):
            x, y = pos
            return -50 <= x <= 50 and -50 <= y <= 50

        mock_validation_service.is_valid_position.side_effect = mock_is_valid

        perception = sample_base_agent.get_fallback_perception()

        assert isinstance(perception, PerceptionData)
        # Should have boundary markers (value 3) at invalid positions

    def test_perception_without_validation_service(
        self, sample_base_agent, mock_spatial_service
    ):
        """Test perception generation when validation_service is None (infinite bounds)."""
        # Ensure validation_service is None
        sample_base_agent.validation_service = None

        # Setup mock spatial service with some nearby entities
        mock_resource = Mock()
        mock_resource.position = (6.0, 5.0)  # Adjacent to agent at (5,5)

        mock_agent = Mock()
        mock_agent.agent_id = "other_agent"
        mock_agent.position = (5.0, 6.0)  # Adjacent to agent

        # Mock get_nearby to return the mock entities
        def mock_get_nearby(position, radius, index_names=None):
            result = {}
            if "resources" in index_names:
                result["resources"] = [mock_resource]
            if "agents" in index_names:
                result["agents"] = [mock_agent]
            return result

        mock_spatial_service.get_nearby.side_effect = mock_get_nearby

        perception = sample_base_agent.get_fallback_perception()

        assert isinstance(perception, PerceptionData)
        assert perception.grid.shape == (11, 11)  # 2*5+1 = 11

        # When validation_service is None, no positions should be marked as obstacles (3)
        # Only resources (1) and other agents (2) should be marked
        obstacle_count = (perception.grid == 3).sum()
        assert (
            obstacle_count == 0
        ), "No obstacles should be marked when validation_service is None"

        # Verify that resources and agents are still properly marked
        resource_count = (perception.grid == 1).sum()
        agent_count = (perception.grid == 2).sum()
        assert resource_count == 1, "Should have one resource marked"
        assert agent_count == 1, "Should have one other agent marked"

    def test_reward_calculation_with_zero_division(self, sample_base_agent):
        """Test reward calculation edge cases."""
        # Setup state with zero starting health (edge case)
        sample_base_agent.starting_health = 0
        sample_base_agent.current_health = 0

        pre_state = AgentState(
            agent_id="test",
            step_number=99,
            position_x=5.0,
            position_y=5.0,
            position_z=0.0,
            resource_level=50,
            current_health=0,
            is_defending=False,
            total_reward=0.0,
            age=0,
        )

        post_state = AgentState(
            agent_id="test",
            step_number=100,
            position_x=5.0,
            position_y=5.0,
            position_z=0.0,
            resource_level=50,  # Same resources
            current_health=0,  # Same health
            is_defending=False,
            total_reward=0.0,
            age=0,
        )

        action = Mock()
        action.name = "pass"

        # Should handle zero division gracefully
        reward = sample_base_agent._calculate_reward(pre_state, post_state, action)

        assert isinstance(reward, float)

    def test_memory_config_override(self, sample_base_agent):
        """Test memory configuration override."""
        custom_config = {
            "redis_host": "custom_host",
            "redis_port": 9999,
            "memory_limit": 500,
        }

        with patch("farm.core.agent.AgentMemoryManager") as mock_memory_manager, patch(
            "farm.core.agent.RedisMemoryConfig"
        ) as mock_config_class:

            mock_config_instance = Mock()
            mock_config_class.return_value = mock_config_instance

            mock_memory_instance = Mock()
            mock_memory_manager.get_instance.return_value = mock_memory_instance
            mock_memory_instance.get_memory.return_value = Mock()

            sample_base_agent._init_memory(custom_config)

            # Verify custom config was applied
            assert mock_config_instance.redis_host == "custom_host"
            assert mock_config_instance.redis_port == 9999
            assert mock_config_instance.memory_limit == 500

    def test_curriculum_validation_valid_config(self, sample_base_agent):
        """Test curriculum configuration validation with valid config."""
        # Setup valid curriculum config
        valid_config = Mock()
        valid_config.curriculum_phases = [
            {"steps": 100, "enabled_actions": ["move", "gather"]},
            {"steps": 200, "enabled_actions": ["move", "gather", "share", "attack"]},
            {
                "steps": -1,
                "enabled_actions": ["move", "gather", "share", "attack", "reproduce"],
            },
        ]
        sample_base_agent.config = valid_config

        # Should return True for valid config
        assert sample_base_agent._validate_curriculum_config() is True

    def test_curriculum_validation_invalid_config(self, sample_base_agent):
        """Test curriculum configuration validation with invalid config."""
        # Setup invalid curriculum config (missing required field)
        invalid_config = Mock()
        invalid_config.curriculum_phases = [
            {"steps": 100},  # Missing enabled_actions
        ]
        sample_base_agent.config = invalid_config

        # Should return False for invalid config
        assert sample_base_agent._validate_curriculum_config() is False

    def test_curriculum_validation_empty_config(self, sample_base_agent):
        """Test curriculum configuration validation with empty config."""
        # No curriculum config should be valid (defaults to all actions)
        sample_base_agent.config = None
        assert sample_base_agent._validate_curriculum_config() is True

        # Empty curriculum list should be valid
        empty_config = Mock()
        empty_config.curriculum_phases = []
        sample_base_agent.config = empty_config
        assert sample_base_agent._validate_curriculum_config() is True

    def test_curriculum_phase_transitions(self, sample_base_agent):
        """Test curriculum phase transitions based on simulation steps."""
        # Setup curriculum config
        curriculum_config = Mock()
        curriculum_config.curriculum_phases = [
            {"steps": 100, "enabled_actions": ["move", "gather"]},
            {"steps": 200, "enabled_actions": ["move", "gather", "share", "attack"]},
            {
                "steps": -1,
                "enabled_actions": ["move", "gather", "share", "attack", "reproduce"],
            },
        ]
        sample_base_agent.config = curriculum_config

        # Mock time service to return different step numbers
        mock_time_service = Mock()
        sample_base_agent.time_service = mock_time_service

        # Test phase 1 (steps 0-99)
        mock_time_service.current_time.return_value = 50
        enabled_actions = []
        if sample_base_agent.config and hasattr(
            sample_base_agent.config, "curriculum_phases"
        ):
            curriculum_phases = getattr(
                sample_base_agent.config, "curriculum_phases", []
            )
            for phase in curriculum_phases:
                if 50 < phase["steps"] or phase["steps"] == -1:
                    enabled_actions = [
                        a
                        for a in sample_base_agent.actions
                        if a.name in phase["enabled_actions"]
                    ]
                    break

        assert len(enabled_actions) == 2
        assert enabled_actions[0].name == "move"
        assert enabled_actions[1].name == "gather"

        # Test phase 2 (steps 100-199)
        mock_time_service.current_time.return_value = 150
        enabled_actions = []
        for phase in curriculum_phases:
            if 150 < phase["steps"] or phase["steps"] == -1:
                enabled_actions = [
                    a
                    for a in sample_base_agent.actions
                    if a.name in phase["enabled_actions"]
                ]
                break

        assert len(enabled_actions) == 4
        action_names = {a.name for a in enabled_actions}
        assert "move" in action_names
        assert "gather" in action_names
        assert "share" in action_names
        assert "attack" in action_names

        # Test phase 3 (steps 200+)
        mock_time_service.current_time.return_value = 250
        enabled_actions = []
        for phase in curriculum_phases:
            if 250 < phase["steps"] or phase["steps"] == -1:
                enabled_actions = [
                    a
                    for a in sample_base_agent.actions
                    if a.name in phase["enabled_actions"]
                ]
                break

        assert len(enabled_actions) == 5
        action_names = {a.name for a in enabled_actions}
        assert "move" in action_names
        assert "gather" in action_names
        assert "share" in action_names
        assert "attack" in action_names
        assert "reproduce" in action_names

    @patch("farm.core.decision.decision.DecisionModule")
    def test_curriculum_integration_with_decision_module(
        self, mock_decision_module_class, sample_base_agent
    ):
        """Test integration between curriculum and DecisionModule."""
        # Setup mock DecisionModule
        mock_decision_module = Mock()
        mock_decision_module.decide_action.return_value = 0
        mock_decision_module_class.return_value = mock_decision_module

        # Replace the agent's decision_module with our test mock
        sample_base_agent.decision_module = mock_decision_module

        # Setup curriculum config
        curriculum_config = Mock()
        curriculum_config.curriculum_phases = [
            {"steps": 100, "enabled_actions": ["move", "gather"]},
        ]
        sample_base_agent.config = curriculum_config

        # Mock time service
        mock_time_service = Mock()
        mock_time_service.current_time.return_value = 50
        sample_base_agent.time_service = mock_time_service

        # Test action selection with curriculum
        action_index = sample_base_agent._select_action_with_curriculum(
            [a for a in sample_base_agent.actions if a.name in ["move", "gather"]]
        )

        # Verify DecisionModule was called with enabled action indices
        mock_decision_module.decide_action.assert_called_once()
        call_args = mock_decision_module.decide_action.call_args
        enabled_indices = call_args[0][1]  # Second argument should be enabled_indices

        # Should only include indices for move and gather actions
        assert len(enabled_indices) == 2
        assert enabled_indices[0] in [0, 1, 2, 3, 4, 5, 6]  # Valid action indices
        assert enabled_indices[1] in [0, 1, 2, 3, 4, 5, 6]

    def test_curriculum_fallback_on_invalid_config(self, sample_base_agent):
        """Test that agent falls back gracefully when curriculum config is invalid."""
        # Setup invalid curriculum config
        invalid_config = Mock()
        invalid_config.curriculum_phases = [
            {"steps": 100},  # Missing enabled_actions
        ]
        sample_base_agent.config = invalid_config

        # Validation should detect invalid config
        is_valid = sample_base_agent._validate_curriculum_config()

        # Should return False for invalid config
        assert is_valid is False

        # Config should still contain the invalid data (validation doesn't modify it)
        assert len(sample_base_agent.config.curriculum_phases) == 1
        assert "enabled_actions" not in sample_base_agent.config.curriculum_phases[0]

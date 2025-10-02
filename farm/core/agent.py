"""
Agent System for Multi-Agent Simulations

This module provides the core agent implementation for AgentFarm simulations.
It defines the BaseAgent class and related agent types that form the foundation
of autonomous entities in the simulation environment.

The agent system is designed around the concept of autonomous decision-making
entities that can perceive their environment, make decisions, and take actions.
Agents maintain their own state, interact with other agents, and adapt to their
environment through various decision-making modules.

Key Components:
    - BaseAgent: Core agent class with full functionality
    - Agent lifecycle management through services
    - Decision-making integration with various algorithms
    - Memory systems for learning and adaptation
    - Spatial awareness and interaction capabilities
    - Resource management and combat mechanics

Agent Capabilities:
    - Movement and navigation in 2D space
    - Resource gathering and sharing
    - Combat and defense mechanics
    - Communication with other agents
    - Learning through various decision algorithms
    - Memory persistence across simulation runs

Integration Points:
    - Environment: Agents operate within simulation environments
    - Services: Various services provide specialized functionality
    - Decision Modules: Pluggable decision-making algorithms
    - Memory Systems: Persistent learning and adaptation
    - Spatial Index: Efficient queries for nearby entities
"""

import math
import random
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from farm.core.action import Action, action_name_to_index, action_registry
from farm.core.decision.config import DecisionConfig
from farm.core.decision.decision import DecisionModule
from farm.core.device_utils import create_device_from_config
from farm.core.genome import Genome
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
from farm.utils.config_utils import get_nested_then_flat
from farm.utils.logging_config import get_logger
from farm.utils.logging_utils import AgentLogger

try:
    from farm.memory.redis_memory import AgentMemoryManager, RedisMemoryConfig
except Exception:  # pragma: no cover - optional at runtime
    AgentMemoryManager = None  # type: ignore
    RedisMemoryConfig = None  # type: ignore

if TYPE_CHECKING:
    from farm.core.environment import Environment

logger = get_logger(__name__)


class BaseAgent:
    """Base agent class representing an autonomous entity in the simulation environment.

    This agent can move, gather resources, share with others, and engage in combat.
    It maintains its own state including position, resources, and health, while making
    decisions through various specialized modules.

    Attributes:
        actions (list[Action]): Available actions the agent can take
        agent_id (str): Unique identifier for this agent
        position (tuple[float, float]): Current (x,y) coordinates
        resource_level (int): Current amount of resources held
        alive (bool): Whether the agent is currently alive
        environment (Environment): Reference to the simulation environment
        device (torch.device): Computing device (CPU/GPU) for neural operations
        total_reward (float): Cumulative reward earned
        current_health (float): Current health points
        starting_health (float): Maximum possible health points
        is_defending (bool): Whether the agent is currently in defensive stance
        defense_timer (int): Turns remaining in defensive stance
    """

    def __init__(
        self,
        agent_id: str,
        position: tuple[float, float],
        resource_level: int,
        spatial_service: ISpatialQueryService,
        environment: Optional["Environment"] = None,
        *,
        agent_type: str = "BaseAgent",
        metrics_service: IMetricsService | None = None,
        logging_service: ILoggingService | None = None,
        validation_service: IValidationService | None = None,
        time_service: ITimeService | None = None,
        lifecycle_service: IAgentLifecycleService | None = None,
        config: object | None = None,
        action_set: Optional[list[Action]] = None,
        device: Optional[torch.device] = None,
        parent_ids: Optional[list[str]] = None,
        generation: int = 0,
        use_memory: bool = False,
        memory_config: Optional[dict] = None,
    ):
        """Initialize a new BaseAgent with the specified parameters and services.

        This constructor sets up a complete agent instance with all necessary services,
        state management, and decision-making capabilities. The agent is configured
        with spatial awareness, optional services for metrics/logging/validation,
        and memory systems if requested.

        Args:
            agent_id: Unique string identifier for this agent
            position: Initial (x, y) coordinates as a tuple of floats
            resource_level: Starting amount of resources the agent possesses
            spatial_service: Service for performing spatial queries on nearby entities
            environment: Optional reference to the simulation environment
            agent_type: Type identifier for this agent (e.g., "SystemAgent", "IndependentAgent")
            metrics_service: Optional service for recording simulation metrics
            logging_service: Optional service for logging agent activities and events
            validation_service: Optional service for validating agent actions and positions
            time_service: Optional service for accessing current simulation time
            lifecycle_service: Optional service for managing agent creation/removal
            config: Optional configuration object containing agent parameters
            action_set: List of available actions (uses defaults if empty)
            device: Optional device for neural network computations (auto-detected if None)
            parent_ids: List of parent agent IDs for genome tracking
            generation: Generation number for evolutionary tracking
            use_memory: Whether to initialize Redis-based memory system
            memory_config: Configuration dictionary for memory system
        """
        # Add default actions (already normalized by default)
        self.actions = (
            action_set
            if action_set is not None
            else action_registry.get_all(normalized=True)
        )

        self.agent_id = agent_id
        self.position = position
        self.resource_level = resource_level
        self.agent_type = agent_type
        self.alive = True

        # Initialize services
        self._initialize_services(
            environment=environment,
            spatial_service=spatial_service,
            metrics_service=metrics_service,
            logging_service=logging_service,
            validation_service=validation_service,
            time_service=time_service,
            lifecycle_service=lifecycle_service,
            config=config,
        )

        # Set up device for neural network computations
        if device is not None:
            self.device = device
        elif config is not None:
            self.device = create_device_from_config(config)
        else:
            # Fallback to auto-detection if no config provided
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_agent_state()

        # Generate genome info
        self.generation = generation
        self.genome_id = self._generate_genome_id(parent_ids or [])

        # Initialize DecisionModule for action selection
        self._initialize_decision_module()

        # Validate curriculum configuration if present
        if not self._validate_curriculum_config():
            logger.warning(
                f"Agent {self.agent_id}: Invalid curriculum configuration detected. "
                "Falling back to allowing all actions."
            )
            # Remove invalid curriculum config to prevent issues
            if self.config and hasattr(self.config, "curriculum_phases"):
                setattr(self.config, "curriculum_phases", [])

        # Initialize Redis memory if requested
        self.memory = None
        if use_memory:
            self._init_memory(memory_config)

        if self.metrics_service:
            try:
                self.metrics_service.record_birth()
            except Exception as e:
                logger.warning(
                    f"Failed to record birth metrics for agent {self.agent_id}: {e}"
                )

    def _initialize_agent_state(self) -> None:
        """Initialize agent state variables and configuration."""
        self.previous_state: AgentState | None = None
        self.previous_action = None
        # Movement distance from nested agent behavior when available
        self.max_movement = get_nested_then_flat(
            config=self.config,
            nested_parent_attr="agent_behavior",
            nested_attr_name="max_movement",
            flat_attr_name="max_movement",
            default_value=8,
            expected_types=(int, float),
        )
        self.total_reward = 0.0
        self.episode_rewards = []
        self.losses = []
        self.starvation_counter = 0  # Counter for consecutive steps with zero resources
        # Max steps agent can survive without resources
        self.starvation_threshold = get_nested_then_flat(
            config=self.config,
            nested_parent_attr="agent_behavior",
            nested_attr_name="starvation_threshold",
            flat_attr_name="starvation_threshold",
            default_value=100,
            expected_types=(int, float),
        )
        self.birth_time = self.time_service.current_time() if self.time_service else 0

        # Initialize health tracking first
        # Starting health from combat configuration when available
        self.starting_health = get_nested_then_flat(
            config=self.config,
            nested_parent_attr="combat",
            nested_attr_name="starting_health",
            flat_attr_name="starting_health",
            default_value=100,
            expected_types=(int, float),
        )
        self.current_health = self.starting_health
        self.is_defending = False
        self.defense_timer = 0

        # Orientation (degrees, clockwise): 0 = north/up, 90 = east/right
        # Used by the perception system to align egocentric observations
        self.orientation = 0.0

    def _initialize_services(
        self,
        environment: Optional["Environment"],
        spatial_service: ISpatialQueryService,
        metrics_service: IMetricsService | None,
        logging_service: ILoggingService | None,
        validation_service: IValidationService | None,
        time_service: ITimeService | None,
        lifecycle_service: IAgentLifecycleService | None,
        config: object | None,
    ) -> None:
        """Initialize agent services using the factory pattern.

        Args:
            environment: Reference to the simulation environment
            spatial_service: Service for spatial queries
            metrics_service: Service for metrics collection
            logging_service: Service for logging
            validation_service: Service for validation
            time_service: Service for time management
            lifecycle_service: Service for lifecycle management
            config: Configuration object
        """
        # Create services using factory - use local variables for performance
        services = AgentServiceFactory.create_services(
            environment=environment,
            metrics_service=metrics_service,
            logging_service=logging_service,
            validation_service=validation_service,
            time_service=time_service,
            lifecycle_service=lifecycle_service,
            config=config,
        )

        # Unpack services to local variables for efficient assignment
        (
            metrics_service,
            logging_service,
            validation_service,
            time_service,
            lifecycle_service,
            config,
        ) = services

        # Assign to instance attributes
        self.spatial_service = spatial_service
        self.metrics_service = metrics_service
        self.logging_service = logging_service
        self.validation_service = validation_service
        self.time_service = time_service
        self.lifecycle_service = lifecycle_service
        self.config = config
        self.environment = environment

    def _generate_genome_id(self, parent_ids: list[str]) -> str:
        """Generate a unique genome ID for this agent.

        Args:
            parent_ids (list[str]): List of parent agent IDs, if any

        Returns:
            str: Formatted genome ID string in format 'AgentType:generation:parents:time'
        """
        genome_id = GenomeId(
            agent_type=self.__class__.__name__,
            generation=self.generation,
            parent_ids=parent_ids,
            creation_time=self.time_service.current_time() if self.time_service else 0,
        )
        return genome_id.to_string()

    def _initialize_decision_module(self):
        """Initialize the DecisionModule for intelligent action selection.

        Sets up the decision-making system that uses reinforcement learning
        algorithms (DDQN, PPO, etc.) to select optimal actions based on the
        current state. The module is configured with action/observation spaces
        and uses the agent's current configuration for hyperparameters.

        The decision module integrates with the environment's multi-channel
        observation system and supports curriculum learning through configurable
        action restrictions based on simulation progress.
        """
        decision_config = DecisionConfig()
        if self.config and hasattr(self.config, "decision"):
            # Use config from environment if available
            decision_config = getattr(self.config, "decision", DecisionConfig())

        # Get action and observation spaces from config if available
        action_space = (
            getattr(self.config, "action_space", None) if self.config else None
        )
        observation_space = (
            getattr(self.config, "observation_space", None) if self.config else None
        )

        # If spaces are not in config, get them from the environment
        if (
            action_space is None
            and hasattr(self, "environment")
            and self.environment is not None
        ):
            action_space = self.environment.action_space()
        if (
            observation_space is None
            and hasattr(self, "environment")
            and self.environment is not None
        ):
            observation_space = self.environment.observation_space()

        self.decision_module = DecisionModule(
            agent=self,
            config=decision_config,
            action_space=action_space,
            observation_space=observation_space,
        )

    def get_fallback_perception(self) -> PerceptionData:
        """Get agent's perception of nearby environment elements.

        Creates a grid representation of the agent's surroundings within its perception radius.
        The grid uses the following encoding:
        - 0: Empty space
        - 1: Resource
        - 2: Other agent
        - 3: Boundary/obstacle

        Returns:
            PerceptionData: Structured perception data centered on agent, with dimensions
                (2 * perception_radius + 1) x (2 * perception_radius + 1)
        """
        # Get perception radius from config
        try:
            # Perception radius from nested agent behavior when available
            radius = get_nested_then_flat(
                config=self.config,
                nested_parent_attr="agent_behavior",
                nested_attr_name="perception_radius",
                flat_attr_name="perception_radius",
                default_value=5,
                expected_types=(int, float),
            )
            # Create perception grid centered on agent
            size = 2 * radius + 1
        except TypeError:
            # Handle cases where config attributes return non-numeric values
            radius = 5
            size = 11
        perception = np.zeros((size, size), dtype=np.int8)

        # Get nearby entities using spatial service
        try:
            nearby = self.spatial_service.get_nearby(
                self.position, radius, ["resources"]
            )
            nearby_resources = nearby.get("resources", [])
        except Exception as e:
            logger.warning(
                f"Failed to get nearby resources for agent {self.agent_id}: {e}"
            )
            nearby_resources = []  # Use empty list as fallback

        try:
            nearby = self.spatial_service.get_nearby(self.position, radius, ["agents"])
            nearby_agents = nearby.get("agents", [])
        except Exception as e:
            logger.warning(
                f"Failed to get nearby agents for agent {self.agent_id}: {e}"
            )
            nearby_agents = []  # Use empty list as fallback

        # Helper function to convert world coordinates to grid coordinates
        def world_to_grid(wx: float, wy: float) -> tuple[int, int]:
            # Convert world position to grid position relative to agent
            # Use configurable discretization method for consistency
            # Discretization method from nested environment config when available
            if self.config and getattr(self.config, "environment", None) is not None:
                discretization_method = getattr(
                    self.config.environment, "position_discretization_method", "floor"
                )
            else:
                discretization_method = (
                    getattr(self.config, "position_discretization_method", "floor")
                    if self.config
                    else "floor"
                )

            if discretization_method == "round":
                gx = int(round(wx - self.position[0] + radius))
                gy = int(round(wy - self.position[1] + radius))
            elif discretization_method == "ceil":
                gx = int(math.ceil(wx - self.position[0] + radius))
                gy = int(math.ceil(wy - self.position[1] + radius))
            else:  # "floor" (default)
                gx = int(math.floor(wx - self.position[0] + radius))
                gy = int(math.floor(wy - self.position[1] + radius))
            return gx, gy

        # Add resources to perception
        for resource in nearby_resources:
            gx, gy = world_to_grid(resource.position[0], resource.position[1])
            if 0 <= gx < size and 0 <= gy < size:
                perception[gy, gx] = 1

        # Add other agents to perception
        for agent in nearby_agents:
            if agent.agent_id != self.agent_id:  # Don't include self
                gx, gy = world_to_grid(agent.position[0], agent.position[1])
                if 0 <= gx < size and 0 <= gy < size:
                    perception[gy, gx] = 2

        # Add boundary/obstacle markers
        x_min = self.position[0] - radius
        y_min = self.position[1] - radius

        # Mark cells outside environment bounds as obstacles
        for i in range(size):
            for j in range(size):
                world_x = x_min + j
                world_y = y_min + i
                try:
                    # If validation_service is None, assume infinite bounds (all positions valid)
                    # Only mark as obstacle if validation service exists AND position is invalid
                    if (
                        self.validation_service
                        and not self.validation_service.is_valid_position(
                            (world_x, world_y)
                        )
                    ):
                        perception[i, j] = 3
                except Exception as e:
                    logger.warning(
                        f"Failed to validate position ({world_x}, {world_y}) for agent {self.agent_id}: {e}"
                    )
                    # Mark as obstacle by default when validation fails
                    perception[i, j] = 3

        return PerceptionData(perception)

    def create_decision_state(self):
        """Create a state representation suitable for the DecisionModule.

        Uses the environment's multi-channel observation system that provides
        rich information about the agent's surroundings including health,
        allies, enemies, resources, obstacles, and dynamic channels like
        trails and damage heat.

        When environment is not available, falls back to a simplified state
        representation that matches the expected observation space shape.

        Returns:
            torch.Tensor: Multi-channel observation tensor for decision making
        """
        if not hasattr(self, "environment") or self.environment is None:
            # Log fallback usage for debugging
            logger.warning(
                f"Agent {self.agent_id} using fallback state - no environment available. "
                "This may impact decision quality."
            )
            # Fallback to simple state if no environment available
            return self._create_fallback_state()

        # Get the multi-channel observation from the environment
        observation_np = self.environment.observe(self.agent_id)

        # Convert to torch tensor and ensure proper device/dtype
        # Keep multi-dimensional structure for CNN backbones
        # Shape: (NUM_CHANNELS, S, S)
        observation_tensor = torch.from_numpy(observation_np).to(
            device=self.device, dtype=torch.float32
        )

        return observation_tensor

    def _create_fallback_state(self):
        """Create a fallback state representation when environment is not available.

        Creates a multi-channel observation tensor that matches the expected shape
        from the DecisionModule's observation space. This ensures compatibility
        with CNN backbones and other multi-dimensional input architectures.

        The fallback uses agent properties and perception data arranged in channels
        to maintain shape consistency with the environment's multi-channel observations.

        Returns:
            torch.Tensor: Multi-channel observation tensor matching expected shape
        """
        # Get expected observation shape from DecisionModule if available
        if hasattr(self, "decision_module") and hasattr(
            self.decision_module, "observation_shape"
        ):
            expected_shape = self.decision_module.observation_shape
            # Handle case where observation_shape might be a Mock (for testing)
            try:
                # Try to get the length of the expected shape - this will fail for Mock objects
                shape_len = len(expected_shape)
                if shape_len >= 3 and len(expected_shape) > 1:
                    # Multi-channel case: (channels, height, width)
                    num_channels, size = expected_shape[0], expected_shape[1]
                elif shape_len == 2:
                    # 2D case: (height, width) - assume single channel
                    num_channels, size = 1, expected_shape[0]
                elif shape_len == 1:
                    # 1D case: (features,) - reshape to square grid
                    feature_count = expected_shape[0]
                    size = int(np.ceil(np.sqrt(feature_count)))
                    num_channels = 1
                else:
                    # Fallback for unexpected shape format
                    raise TypeError("Unexpected observation shape format")
            except (TypeError, AttributeError):
                # Fallback when observation_shape is not a proper sequence
                from farm.core.channels import NUM_CHANNELS

                try:
                    radius = get_nested_then_flat(
                        config=self.config,
                        nested_parent_attr="agent_behavior",
                        nested_attr_name="perception_radius",
                        flat_attr_name="perception_radius",
                        default_value=5,
                        expected_types=(int, float),
                    )
                    size = 2 * radius + 1
                except TypeError:
                    # Handle cases where config attributes return non-numeric values
                    radius = 5
                    size = 11
                num_channels = NUM_CHANNELS
        else:
            # Fallback to default values if DecisionModule not initialized
            from farm.core.channels import NUM_CHANNELS

            try:
                radius = get_nested_then_flat(
                    config=self.config,
                    nested_parent_attr="agent_behavior",
                    nested_attr_name="perception_radius",
                    flat_attr_name="perception_radius",
                    default_value=5,
                    expected_types=(int, float),
                )
                size = 2 * radius + 1
            except TypeError:
                # Handle cases where config attributes return non-numeric values
                radius = 5
                size = 11
            num_channels = NUM_CHANNELS

        # Create multi-channel observation array
        observation = np.zeros((num_channels, size, size), dtype=np.float32)

        # Channel 0: Agent properties (position, health, resources, etc.)
        # Normalize and place in center of the grid
        center = size // 2

        # Only store agent properties if we have enough channels
        agent_properties = [
            self.position[0] / 100.0,  # X position
            self.position[1] / 100.0,  # Y position
            self.resource_level / 100.0,  # Resources
            self.current_health / self.starting_health,  # Health ratio
            float(self.is_defending),  # Defense status
            min(self.total_reward / 100.0, 1.0),  # Reward (capped)
        ]

        # Store as many agent properties as we have channels
        for i, prop_value in enumerate(agent_properties):
            if i < num_channels:
                observation[i, center, center] = prop_value

        # Get perception data - use grid size that matches expected shape
        # Temporarily override config to get correct perception radius
        original_config = getattr(self, "config", None)
        try:
            if hasattr(self, "decision_module") and hasattr(
                self.decision_module, "observation_shape"
            ):
                expected_shape = self.decision_module.observation_shape
                # Try to get the length and handle Mock objects properly
                try:
                    shape_len = len(expected_shape)
                    if shape_len >= 3 and len(expected_shape) > 1:
                        # Calculate radius from expected grid size: radius = (size - 1) / 2
                        expected_size = expected_shape[1]  # Assuming square grid
                        expected_radius = (expected_size - 1) // 2

                        # Create a temporary config with the correct radius
                        class TempConfig:
                            perception_radius = expected_radius

                        self.config = TempConfig()
                except (TypeError, AttributeError):
                    # Skip config override if observation_shape is a Mock or invalid
                    pass

            perception = self.get_fallback_perception()
        finally:
            # Always restore original config to prevent inconsistent state
            self.config = original_config

        # Map perception grid to remaining channels (up to available channels)
        # 0: Empty, 1: Resource, 2: Agent, 3: Obstacle
        available_channels = max(
            0, num_channels - 6
        )  # Reserve channels for agent properties
        perception_channels = min(4, available_channels)

        for c in range(perception_channels):
            channel_idx = 6 + c
            if channel_idx < num_channels:
                # Extract specific perception type and normalize
                mask = (perception.grid == c).astype(np.float32)
                # Resize mask to fit the expected grid size if necessary
                if mask.shape != (size, size):
                    # Use center crop/pad to match expected size
                    mask_resized = np.zeros((size, size), dtype=np.float32)
                    min_size = min(mask.shape[0], size)
                    offset = (size - min_size) // 2
                    mask_resized[
                        offset : offset + min_size, offset : offset + min_size
                    ] = mask[:min_size, :min_size]
                    mask = mask_resized
                observation[channel_idx] = mask

        # Fill remaining channels with zeros (for future extensibility)
        for c in range(6 + perception_channels, num_channels):
            observation[c] = 0.0

        # Convert to torch tensor
        return torch.from_numpy(observation).to(device=self.device, dtype=torch.float32)

    def get_state(self) -> AgentState:
        """Returns the current state of the agent as an AgentState object.

        This method captures the agent's current state including:
        - Unique identifier
        - Current simulation step
        - 3D position coordinates
        - Resource level
        - Health status
        - Defense status
        - Cumulative reward
        - Agent age

        Returns:
            AgentState: A structured object containing all current state information
        """
        return AgentState(
            agent_id=self.agent_id,
            step_number=self.time_service.current_time() if self.time_service else 0,
            position_x=self.position[0],
            position_y=self.position[1],
            position_z=self.position[2] if len(self.position) > 2 else 0,
            resource_level=self.resource_level,
            current_health=self.current_health,
            is_defending=self.is_defending,
            total_reward=self.total_reward,
            age=(self.time_service.current_time() if self.time_service else 0)
            - self.birth_time,
        )

    def decide_action(self):
        """Select an action using the DecisionModule's intelligent decision making.

        The selection process involves:
        1. Getting current state representation
        2. Passing state through DecisionModule's algorithm (DDQN, PPO, etc.)
        3. Mapping action index to Action object
        4. Handling curriculum phases if configured

        Returns:
            Action: Selected action object to execute
        """
        # Cache state tensor to avoid recreating it multiple times
        current_time = self.time_service.current_time() if self.time_service else -1
        if not hasattr(self, "_cached_selection_state") or current_time != getattr(
            self, "_cached_selection_time", -1
        ):
            self._cached_selection_state = self.create_decision_state()
            self._cached_selection_time = current_time

        # Get enabled actions based on curriculum phases if configured
        current_step = current_time if current_time != -1 else 0
        enabled_actions = self.actions  # Default all
        if self.config and hasattr(self.config, "curriculum_phases"):
            curriculum_phases = getattr(self.config, "curriculum_phases", [])
            if isinstance(curriculum_phases, (list, tuple)):
                for phase in curriculum_phases:
                    if current_step < phase["steps"] or phase["steps"] == -1:
                        enabled_actions = [
                            a
                            for a in self.actions
                            if a.name in phase["enabled_actions"]
                        ]
                        break

        # Use DecisionModule to select action index, passing enabled actions for curriculum support
        enabled_action_indices = [
            self.actions.index(action) for action in enabled_actions
        ]
        action_index = self.decision_module.decide_action(
            self._cached_selection_state, enabled_action_indices
        )

        # Map action index to Action object
        # Since decision_module now respects enabled_actions, action_index should always be valid
        if enabled_actions:
            selected_action = enabled_actions[action_index]
        else:
            selected_action = self.actions[action_index]

        return selected_action

    def _select_action_with_curriculum(self, enabled_actions):
        """Select action index with curriculum restrictions.

        Args:
            enabled_actions: List of enabled Action objects

        Returns:
            int: Action index within enabled_actions list
        """
        # Cache state tensor to avoid recreating it multiple times
        current_time = self.time_service.current_time() if self.time_service else -1
        if not hasattr(self, "_cached_selection_state") or current_time != getattr(
            self, "_cached_selection_time", -1
        ):
            self._cached_selection_state = self.create_decision_state()
            self._cached_selection_time = current_time

        # Convert enabled actions to indices for DecisionModule
        if enabled_actions:
            enabled_action_indices = [
                self.actions.index(action) for action in enabled_actions
            ]
        else:
            enabled_action_indices = None

        # Use DecisionModule to select action index with curriculum support
        action_index = self.decision_module.decide_action(
            self._cached_selection_state, enabled_action_indices
        )

        return action_index

    def _validate_curriculum_config(self):
        """Validate curriculum configuration for consistency and completeness.

        Returns:
            bool: True if curriculum config is valid, False otherwise
        """
        if not self.config or not hasattr(self.config, "curriculum_phases"):
            return True  # No curriculum config is valid (defaults to all actions)

        curriculum_phases = getattr(self.config, "curriculum_phases", [])
        if not isinstance(curriculum_phases, (list, tuple)):
            logger.warning(
                f"Agent {self.agent_id}: curriculum_phases must be a list or tuple"
            )
            return False

        if not curriculum_phases:
            return True  # Empty curriculum is valid

        # Check each phase has required fields
        required_fields = ["steps", "enabled_actions"]
        action_names = {action.name for action in self.actions}

        for i, phase in enumerate(curriculum_phases):
            if not isinstance(phase, dict):
                logger.warning(f"Agent {self.agent_id}: Phase {i} must be a dictionary")
                return False

            # Check required fields
            for field in required_fields:
                if field not in phase:
                    logger.warning(
                        f"Agent {self.agent_id}: Phase {i} missing required field '{field}'"
                    )
                    return False

            # Validate steps
            steps = phase["steps"]
            if not isinstance(steps, int) or (steps != -1 and steps < 0):
                logger.warning(
                    f"Agent {self.agent_id}: Phase {i} steps must be non-negative integer or -1"
                )
                return False

            # Validate enabled_actions
            enabled_actions = phase["enabled_actions"]
            if not isinstance(enabled_actions, (list, tuple)):
                logger.warning(
                    f"Agent {self.agent_id}: Phase {i} enabled_actions must be a list or tuple"
                )
                return False

            # Check all enabled actions exist
            for action_name in enabled_actions:
                if action_name not in action_names:
                    logger.warning(
                        f"Agent {self.agent_id}: Phase {i} references unknown action '{action_name}'"
                    )
                    return False

        # Check phases are in ascending order (except -1 which should be last)
        prev_steps = -1
        for i, phase in enumerate(curriculum_phases):
            steps = phase["steps"]
            if steps == -1:
                # -1 should be the last phase
                if i != len(curriculum_phases) - 1:
                    logger.warning(
                        f"Agent {self.agent_id}: Phase with steps=-1 must be the last phase"
                    )
                    return False
            elif steps <= prev_steps:
                logger.warning(
                    f"Agent {self.agent_id}: Phase {i} steps ({steps}) not greater than previous ({prev_steps})"
                )
                return False
            prev_steps = steps

        return True

    def _calculate_reward(
        self, pre_action_state: AgentState, post_action_state: AgentState, action_taken
    ) -> float:
        """Calculate reward for the current state transition.

        Args:
            pre_action_state: Agent state before action execution
            post_action_state: Agent state after action execution
            action_taken: Action that was executed

        Returns:
            float: Calculated reward based on state changes from pre to post action

        Notes:
        - This method calculates reward based on the immediate effect of the action
        - Reward reflects changes within a single turn, not across turns
        - TODO: Separate rewards logic from agent
        """
        # Basic reward components based on state deltas
        resource_reward = (
            post_action_state.resource_level - pre_action_state.resource_level
        ) * 0.1
        health_reward = (
            post_action_state.current_health - pre_action_state.current_health
        ) * 0.5
        survival_reward = 0.1 if self.alive else -10.0

        # Action-specific bonuses (simplified)
        action_bonus = 0.0
        if action_taken and action_taken.name != "pass":
            # Small bonus for non-idle actions
            action_bonus = 0.05

        total_reward = resource_reward + health_reward + survival_reward + action_bonus

        # Update total reward tracking
        self.total_reward += total_reward

        return total_reward

    def _action_to_index(self, action) -> int:
        """Convert Action object to index for DecisionModule.

        Args:
            action: Action object to convert

        Returns:
            int: Action index
        """
        # Use centralized action space mapping
        return action_name_to_index(action.name)

    def check_starvation(self) -> bool:
        """Check and handle agent starvation state.

        Manages the agent's starvation counter based on resource levels:
        - Increments counter when resources are depleted
        - Resets counter when resources are available
        - Triggers death if counter exceeds starvation threshold

        Returns:
            bool: True if agent died from starvation, False otherwise
        """
        if self.resource_level <= 0:
            self.starvation_counter += 1
            if self.starvation_counter >= self.starvation_threshold:
                self.terminate()
                return True
        else:
            self.starvation_counter = 0
        return False

    def act(self) -> None:
        """Execute the agent's complete turn in the simulation.

        This method orchestrates the full agent lifecycle for a single simulation step,
        including resource management, decision-making, action execution, and learning.
        The agent will not act if it's not alive.

        The execution flow follows this sequence:
        1. Defense timer countdown and status updates
        2. Base resource consumption for turn maintenance
        3. Starvation check - agent dies if resources are depleted beyond threshold
        4. State observation and caching for decision-making efficiency
        5. Intelligent action selection using DecisionModule (DDQN/PPO algorithms)
        6. Action execution with environmental interaction
        7. Reward calculation based on state transitions
        8. Experience storage for reinforcement learning updates
        9. DecisionModule training with new experience tuple

        Each turn consumes configurable base resources and can lead to death
        through starvation mechanics. The method integrates curriculum learning
        by restricting available actions based on simulation progress when configured.
        """
        if not self.alive:
            return

        # Resource consumption
        consumption = get_nested_then_flat(
            config=self.config,
            nested_parent_attr="agent_behavior",
            nested_attr_name="base_consumption_rate",
            flat_attr_name="base_consumption_rate",
            default_value=1,
            expected_types=(int, float),
        )
        self.resource_level -= consumption

        # Check starvation state - exit early if agent dies
        if self.check_starvation():
            return

        # Get current state before action for learning
        current_state = self.get_state()
        current_state_tensor = self.create_decision_state()

        # Get enabled actions based on curriculum phases if configured
        current_step = self.time_service.current_time() if self.time_service else 0
        enabled_actions = self.actions  # Default all
        if self.config and hasattr(self.config, "curriculum_phases"):
            curriculum_phases = getattr(self.config, "curriculum_phases", [])
            if isinstance(curriculum_phases, (list, tuple)):
                for phase in curriculum_phases:
                    if current_step < phase["steps"] or phase["steps"] == -1:
                        enabled_actions = [
                            a
                            for a in self.actions
                            if a.name in phase["enabled_actions"]
                        ]
                        break

        # Select and execute action with curriculum restrictions
        action_index = self._select_action_with_curriculum(enabled_actions)
        if enabled_actions:
            action = enabled_actions[action_index]
        else:
            action = self.actions[action_index]

        # Capture resource level before action execution for accurate logging
        resources_before = self.resource_level

        # Execute action and capture result
        action_result = action.execute(self)

        # Capture resource level after action execution
        resources_after = self.resource_level

        # Log action to database if logger is available
        if (
            hasattr(self, "environment")
            and self.environment
            and hasattr(self.environment, "db")
            and self.environment.db
        ):
            try:
                # Get current time step
                current_time = (
                    self.time_service.current_time()
                    if hasattr(self, "time_service") and self.time_service
                    else 0
                )

                # Log the agent action
                self.environment.db.logger.log_agent_action(
                    step_number=current_time,
                    agent_id=self.agent_id,
                    action_type=action.name,
                    resources_before=resources_before,
                    resources_after=resources_after,
                    reward=0,  # Reward will be calculated later
                    details=action_result.get("details", {}),
                )
            except Exception as e:
                logger.warning(f"Failed to log agent action {action.name}: {e}")

        # Validate action result if validation service is available
        validation_result = None
        if hasattr(self, "validation_service") and self.validation_service:
            try:
                from farm.core.action import validate_action_result

                validation_result = validate_action_result(
                    self, action.name, action_result
                )
            except Exception as e:
                logger.warning(f"Action validation failed for {action.name}: {str(e)}")

        # Log warnings and errors from validation
        if validation_result and not validation_result.get("valid", True):
            logger.warning(
                f"Action validation failed for {action.name} on agent {self.agent_id}: "
                f"Issues: {validation_result.get('issues', [])}"
            )

        # Log action success/failure for debugging
        if action_result.get("success", False):
            logger.debug(f"Agent {self.agent_id} successfully executed {action.name}")
        else:
            logger.info(
                f"Agent {self.agent_id} failed to execute {action.name}: "
                f"{action_result.get('error', 'Unknown error')}"
            )

        # Store the action index for learning (relative to enabled actions)
        self._current_action_index = action_index

        # Store enabled actions for learning update
        self._current_enabled_actions = enabled_actions

        # Update defense status based on timer AFTER action execution
        if self.defense_timer > 0:
            self.defense_timer -= 1
            self.is_defending = self.defense_timer > 0
        else:
            self.is_defending = False

        # Get post-action state for reward calculation
        post_action_state = self.get_state()

        # Calculate reward based on state changes from pre-action to post-action
        reward = self._calculate_reward(current_state, post_action_state, action)

        # Store state and action for learning
        self.previous_state = current_state
        self.previous_state_tensor = current_state_tensor
        self.previous_action = action

        # Get next state after action
        next_state_tensor = self.create_decision_state()
        done = not self.alive

        # Update DecisionModule with experience
        if (
            hasattr(self, "previous_state_tensor")
            and self.previous_state_tensor is not None
        ):
            # Get enabled actions from previous turn for consistent learning
            previous_enabled_actions = getattr(self, "_previous_enabled_actions", None)

            # Use stored action index for consistency with curriculum
            previous_action_index = getattr(
                self, "_previous_action_index", self._current_action_index
            )

            self.decision_module.update(
                state=self.previous_state_tensor,
                action=previous_action_index,
                reward=reward,
                next_state=next_state_tensor,
                done=done,
                enabled_actions=previous_enabled_actions,
            )

            # Store current action index for next turn's learning
            # Always store index relative to enabled actions for curriculum consistency
            if hasattr(self, "_current_action_index"):
                self._previous_action_index = self._current_action_index
            elif enabled_actions:
                # Fallback: find current action's index within enabled actions
                try:
                    self._previous_action_index = enabled_actions.index(action)
                except ValueError:
                    # Action not in enabled actions - this shouldn't happen but handle gracefully
                    logger.warning(
                        f"Action {action.name} not found in enabled actions for agent {self.agent_id}. "
                        f"Using index 0 as fallback."
                    )
                    self._previous_action_index = 0
            else:
                # No enabled actions restriction - use full action space index
                self._previous_action_index = self._action_to_index(action)

            # Store current enabled actions for next turn's learning
            self._previous_enabled_actions = getattr(
                self, "_current_enabled_actions", None
            )

    def clone(self, environment: Optional["Environment"] = None) -> "BaseAgent":
        """Create a mutated copy of this agent.

        Creates a new agent by:
        1. Cloning the current agent's genome
        2. Applying random mutations with 10% probability
        3. Converting mutated genome back to agent instance

        Args:
            environment: Optional environment reference for the cloned agent.
                        If None, uses self.environment. Required for genome reconstruction.

        Returns:
            BaseAgent: A new agent with slightly modified characteristics

        Raises:
            ValueError: If neither environment parameter nor self.environment is available
        """
        # Determine which environment to use
        target_environment = environment or (
            self.environment if hasattr(self, "environment") else None
        )

        if target_environment is None:
            raise ValueError("Cannot clone agent without environment reference")

        # Clone and mutate the genome using Genome class operations
        cloned_genome = Genome.clone(self.to_genome())
        mutated_genome = Genome.mutate(cloned_genome, mutation_rate=0.1)

        # Generate a new agent ID for the clone (should be different from original)
        if self.lifecycle_service:
            new_agent_id = self.lifecycle_service.get_next_agent_id()
        else:
            # Fallback ID generation when lifecycle_service is unavailable
            current_time = self.time_service.current_time() if self.time_service else 0
            random_suffix = random.randint(1000, 9999)
            new_agent_id = f"{self.agent_id}_clone_{current_time}_{random_suffix}"

        # Use Genome.to_agent to properly reconstruct the agent with all module states
        new_agent = Genome.to_agent(
            mutated_genome,
            new_agent_id,
            (int(self.position[0]), int(self.position[1])),  # Convert to int tuple
            target_environment,
            type(self),
        )

        # Preserve additional services and configuration that aren't in the genome
        new_agent.metrics_service = self.metrics_service
        new_agent.logging_service = self.logging_service
        new_agent.validation_service = self.validation_service
        new_agent.time_service = self.time_service
        new_agent.lifecycle_service = self.lifecycle_service
        new_agent.config = self.config

        # Generate proper genome ID with parent relationship
        new_agent.genome_id = new_agent._generate_genome_id([self.agent_id])

        return new_agent

    def reproduce(self) -> bool:
        """Create offspring agent. Assumes resource requirements already checked by action."""
        # Store initial resources for tracking
        initial_resources = self.resource_level

        try:
            # Attempt to create offspring
            new_agent = self._create_offspring()

            # Record successful reproduction
            if self.logging_service:
                self.logging_service.log_reproduction_event(
                    step_number=(
                        self.time_service.current_time() if self.time_service else 0
                    ),
                    parent_id=self.agent_id,
                    offspring_id=new_agent.agent_id,
                    success=True,
                    parent_resources_before=initial_resources,
                    parent_resources_after=self.resource_level,
                    offspring_initial_resources=(
                        getattr(self.config, "offspring_initial_resources", 10)
                        if self.config
                        else 10
                    ),
                    failure_reason="",  # Empty string for successful reproduction
                    parent_generation=self.generation,
                    offspring_generation=new_agent.generation,
                    parent_position=self.position,
                )

            logger.info(
                f"Agent {self.agent_id} reproduced at {self.position} during step {self.time_service.current_time() if self.time_service else 0} creating agent {new_agent.agent_id}"
            )
            return True

        except Exception as e:
            # Log failed reproduction attempt
            if self.logging_service:
                self.logging_service.log_reproduction_event(
                    step_number=(
                        self.time_service.current_time() if self.time_service else 0
                    ),
                    parent_id=self.agent_id,
                    offspring_id="",  # Empty string for failed reproduction
                    success=False,
                    parent_resources_before=initial_resources,
                    parent_resources_after=self.resource_level,  # May have changed if partial creation occurred
                    offspring_initial_resources=0.0,  # Default value for failed reproduction
                    failure_reason=str(e),
                    parent_generation=self.generation,
                    offspring_generation=0,  # Default value for failed reproduction
                    parent_position=self.position,
                )

            logger.error(
                "reproduction_failed",
                agent_id=self.agent_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return False

    def _create_offspring(self):
        """Create a new agent as offspring."""
        # Get the agent's class (IndependentAgent, SystemAgent, etc)
        agent_class = type(self)

        # Generate unique ID and genome info first
        try:
            new_id = (
                self.lifecycle_service.get_next_agent_id()
                if self.lifecycle_service
                else self.agent_id + "_child"
            )
        except Exception as e:
            logger.error(
                "offspring_id_generation_failed",
                parent_agent_id=self.agent_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise RuntimeError(f"Failed to obtain agent ID for offspring: {e}") from e

        generation = self.generation + 1

        # Create new agent with all info
        try:
            new_agent = agent_class(
                agent_id=new_id,
                position=self.position,
                resource_level=get_nested_then_flat(
                    config=self.config,
                    nested_parent_attr="agent_behavior",
                    nested_attr_name="offspring_initial_resources",
                    flat_attr_name="offspring_initial_resources",
                    default_value=10,
                    expected_types=(int, float),
                ),
                spatial_service=self.spatial_service,
                metrics_service=self.metrics_service,
                logging_service=self.logging_service,
                validation_service=self.validation_service,
                time_service=self.time_service,
                lifecycle_service=self.lifecycle_service,
                config=self.config,
                generation=generation,
            )
        except Exception as e:
            logger.error(
                "offspring_creation_failed",
                parent_agent_id=self.agent_id,
                offspring_id=new_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise RuntimeError(f"Failed to instantiate offspring agent: {e}") from e

        # Add new agent to environment
        if self.lifecycle_service:
            try:
                self.lifecycle_service.add_agent(new_agent, flush_immediately=True)
            except Exception as e:
                logger.error(
                    "offspring_registration_failed",
                    parent_agent_id=self.agent_id,
                    offspring_id=new_agent.agent_id,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                # Agent was created but not added to lifecycle service
                # This is a critical failure that should prevent reproduction
                raise RuntimeError(f"Failed to register offspring agent: {e}") from e

        # Subtract offspring cost from parent's resources
        try:
            offspring_cost = get_nested_then_flat(
                config=self.config,
                nested_parent_attr="agent_behavior",
                nested_attr_name="offspring_cost",
                flat_attr_name="offspring_cost",
                default_value=5,
                expected_types=(int, float),
            )
            self.resource_level -= offspring_cost
        except Exception as e:
            logger.error(
                "parent_resource_update_failed",
                parent_agent_id=self.agent_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            # Don't raise here as the offspring is already created and registered
            # Just log the issue

        # Log creation
        logger.info(
            "offspring_created",
            offspring_id=new_id,
            parent_id=self.agent_id,
            position=self.position,
            step=self.time_service.current_time() if self.time_service else 0,
            agent_type=agent_class.__name__,
            generation=generation,
        )

        return new_agent

    def terminate(self):
        """Handle agent death."""

        if self.alive:
            self.alive = False
            self.death_time = (
                self.time_service.current_time() if self.time_service else 0
            )
            # Record the death in environment
            # Log death in database
            if self.logging_service:
                try:
                    self.logging_service.update_agent_death(
                        self.agent_id, self.death_time
                    )
                except Exception as e:
                    logger.error(
                        "agent_death_logging_failed",
                        agent_id=self.agent_id,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )

            logger.info(
                "agent_died",
                agent_id=self.agent_id,
                position=self.position,
                step=self.time_service.current_time() if self.time_service else 0,
                lifetime_steps=(
                    self.death_time - self.birth_time
                    if hasattr(self, "birth_time")
                    else None
                ),
            )
            if self.lifecycle_service:
                try:
                    self.lifecycle_service.remove_agent(self)
                except Exception as e:
                    logger.error(
                        "agent_removal_failed",
                        agent_id=self.agent_id,
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )

    def update_position(self, new_position):
        """Update agent position and mark spatial index as dirty.

        Args:
            new_position (tuple): New (x, y) position
        """
        if self.position != new_position:
            self.position = new_position
            # Mark spatial structures as dirty when position changes
            try:
                self.spatial_service.mark_positions_dirty()
            except Exception as e:
                logger.warning(
                    f"Failed to mark spatial positions as dirty for agent {self.agent_id}: {e}"
                )
                # Position was updated but spatial index wasn't marked dirty
                # This is not critical but should be logged

    def handle_combat(self, attacker: "BaseAgent", damage: float) -> float:
        """Handle incoming attack and calculate actual damage taken.

        Processes combat mechanics including:
        - Damage reduction from defensive stance (50% reduction when defending)
        - Health reduction (clamped to minimum of 0)
        - Death checking if health drops to 0

        Args:
            attacker (BaseAgent): Agent performing the attack
            damage (float): Base damage amount before modifications

        Returns:
            float: Actual damage dealt after defensive calculations

        Notes:
        - TO-DO: Implement more realistic combat mechanics as a separate module
        """
        # Reduce damage if defending
        if self.is_defending:
            damage *= 0.5  # 50% damage reduction when defending

        # Apply damage
        self.current_health = max(0, self.current_health - damage)

        # Check for death
        if self.current_health <= 0:
            self.terminate()

        return damage

    def to_genome(self) -> dict:
        """Convert agent's current state into a genome representation.

        Creates a genetic encoding of the agent's:
        - Neural network weights
        - Action preferences
        - Other learnable parameters

        Returns:
            Genome: Complete genetic representation of agent
        """
        return Genome.from_agent(self)

    @classmethod
    def from_genome(
        cls,
        genome: dict,
        agent_id: str,
        position: tuple[float, float],
        environment: "Environment",
    ) -> "BaseAgent":
        """Create a new agent instance from a genome dictionary.

        Factory method that reconstructs an agent from its serialized genome representation.
        The genome dictionary contains all necessary information to recreate the agent's
        state, including action preferences, neural network parameters, and physical attributes.

        Args:
            genome: Dictionary containing serialized agent genome with action_set,
                   module_states, agent_type, resource_level, and current_health
            agent_id: Unique string identifier for the new agent
            position: Starting (x, y) coordinates as floats
            environment: Simulation environment reference for agent integration

        Returns:
            BaseAgent: New agent instance with characteristics decoded from the genome
        """
        return Genome.to_agent(
            genome, agent_id, (int(position[0]), int(position[1])), environment, cls
        )

    def take_damage(self, damage: float) -> bool:
        """Apply damage to the agent.

        Args:
            damage: Amount of damage to apply

        Returns:
            bool: True if damage was successfully applied
        """
        if damage <= 0:
            return False

        self.current_health -= damage
        if self.current_health < 0:
            self.current_health = 0
        return True

    @property
    def attack_strength(self) -> float:
        """Calculate the agent's current attack strength."""
        base_attack = get_nested_then_flat(
            config=self.config,
            nested_parent_attr="agent_behavior",
            nested_attr_name="base_attack_strength",
            flat_attr_name="base_attack_strength",
            default_value=10,
            expected_types=(int, float),
        )
        return base_attack * (self.current_health / self.starting_health)

    @property
    def defense_strength(self) -> float:
        """Calculate the agent's current defense strength."""
        if not self.is_defending:
            return 0.0
        base_defense = get_nested_then_flat(
            config=self.config,
            nested_parent_attr="agent_behavior",
            nested_attr_name="base_defense_strength",
            flat_attr_name="base_defense_strength",
            default_value=5,
            expected_types=(int, float),
        )
        return base_defense

    def get_action_weights(self) -> dict:
        """Get a dictionary of action weights.

        Returns:
            dict: A dictionary mapping action names to their weights
        """
        return {action.name: action.weight for action in self.actions}

    def _init_memory(self, memory_config: Optional[dict] = None):
        """Initialize the Redis-based memory system for this agent.

        Args:
            memory_config (dict, optional): Configuration parameters for memory
        """
        try:
            # Create memory configuration
            if RedisMemoryConfig is None or AgentMemoryManager is None:
                raise RuntimeError("Redis memory is unavailable (redis not installed)")
            redis_config = RedisMemoryConfig(
                host=getattr(self.config, "redis_host", "localhost"),
                port=getattr(self.config, "redis_port", 6379),
                memory_limit=getattr(self.config, "memory_limit", 1000),
            )

            # Override with custom config if provided
            if memory_config:
                for key, value in memory_config.items():
                    if hasattr(redis_config, key):
                        setattr(redis_config, key, value)

            # Get memory manager instance
            memory_manager = AgentMemoryManager.get_instance(redis_config)

            # Get this agent's memory
            self.memory = memory_manager.get_memory(self.agent_id)
            logger.info(
                "redis_memory_initialized",
                agent_id=self.agent_id,
                host=redis_config.host,
                port=redis_config.port,
            )

        except Exception as e:
            logger.error(
                "redis_memory_init_failed",
                agent_id=self.agent_id,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            # Memory remains None, agent will function without memory

    def remember_experience(
        self,
        action_name: str,
        reward: float,
        perception_data: Optional[PerceptionData] = None,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Record current experience in agent memory.

        Args:
            action_name (str): Name of the action taken
            reward (float): Reward received for the action
            perception_data (PerceptionData, optional): Agent's perception data
            metadata (dict, optional): Additional information to store

        Returns:
            bool: True if successfully recorded, False otherwise

        Notes:
        - #! NOT USED: This is a placeholder for future memory implementation
        """
        if not self.memory:
            return False

        try:
            # Create state representation
            current_state = AgentState(
                agent_id=self.agent_id,
                step_number=(
                    self.time_service.current_time() if self.time_service else 0
                ),
                position_x=self.position[0],
                position_y=self.position[1],
                position_z=self.position[2] if len(self.position) > 2 else 0,
                resource_level=self.resource_level,
                current_health=self.current_health,
                is_defending=self.is_defending,
                total_reward=self.total_reward,
                age=(self.time_service.current_time() if self.time_service else 0)
                - self.birth_time,
            )

            # Add default metadata if not provided
            if metadata is None:
                metadata = {}

            metadata.update(
                {
                    "health_percent": self.current_health / self.starting_health,
                    "genome_id": self.genome_id,
                }
            )

            # Remember in Redis
            return self.memory.remember_state(
                step=self.time_service.current_time() if self.time_service else 0,
                state=current_state,
                action=action_name,
                reward=reward,
                perception=perception_data,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(
                "experience_memory_failed",
                agent_id=self.agent_id,
                action=action_name,
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return False

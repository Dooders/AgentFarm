"""Environment for AgentFarm multi-agent simulations.

Overview
--------
The Environment orchestrates a 2D multi-agent world using the PettingZoo AEC
pattern. It manages agents, resources, spatial indexing, metrics, and optional
SQLite-based logging. The design favors composability (services) and clear
integration points for RL.

Key Responsibilities
--------------------
- Agent lifecycle: add/remove agents, selection order, and step cycle
- Resource lifecycle: initialization, regeneration, consumption (via ResourceManager)
- Spatial queries: KD-tree accelerated nearby/nearest via `SpatialIndex`
- Observation/Action spaces: multi-channel observations and dynamic action mapping
- Reward calculation: per-step, delta-aware rewards with survival handling
- Metrics: step and cumulative metrics via `MetricsTracker` (optional DB logging)

Core Integrations
-----------------
- Channels/Observations: multi-channel perception buffers with decay and visibility
- Services: adapters for validation, time, lifecycle, metrics, and logging
- Database: structured step, agent, resource, and interaction logging when enabled

Determinism
-----------
A seed (explicit or from `SimulationConfig`) controls deterministic aspects
(e.g., identities, resources, and torch RNG when available).

Notes
-----
- Action mapping is dynamic and may be updated at runtime (curriculum/pruning)
- Spatial index rebuilds are optimized with dirty flags and hashing
- In-memory DB mode is supported for tests or ephemeral runs
"""

import logging
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time as _time

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import AECEnv

# Use action registry for cleaner action management
from farm.core.action import ActionType, action_registry
from farm.core.channels import NUM_CHANNELS
from farm.config import SimulationConfig
from farm.core.metrics_tracker import MetricsTracker
from farm.core.observations import AgentObservation, ObservationConfig
from farm.core.resource_manager import ResourceManager
from farm.core.services.implementations import (
    EnvironmentAgentLifecycleService,
    EnvironmentLoggingService,
    EnvironmentMetricsService,
    EnvironmentTimeService,
    EnvironmentValidationService,
    SpatialIndexAdapter,
)
from farm.core.spatial_index import SpatialIndex
from farm.core.state import EnvironmentState
from farm.database.utilities import setup_db
from farm.utils.identity import Identity, IdentityConfig

logger = logging.getLogger(__name__)


def discretize_position_continuous(
    position: Tuple[float, float], grid_size: Tuple[int, int], method: str = "floor"
) -> Tuple[int, int]:
    """
    Convert continuous position to discrete grid coordinates using specified method.

    Args:
        position: (x, y) continuous coordinates
        grid_size: (width, height) of the grid
        method: Discretization method - "floor", "round", or "ceil"

    Returns:
        (x_idx, y_idx) discrete grid coordinates
    """
    x, y = position
    width, height = grid_size

    if method == "round":
        x_idx = max(0, min(int(round(x)), width - 1))
        y_idx = max(0, min(int(round(y)), height - 1))
    elif method == "ceil":
        x_idx = max(0, min(int(math.ceil(x)), width - 1))
        y_idx = max(0, min(int(math.ceil(y)), height - 1))
    else:  # "floor" (default)
        x_idx = max(0, min(int(math.floor(x)), width - 1))
        y_idx = max(0, min(int(math.floor(y)), height - 1))

    return x_idx, y_idx


def bilinear_distribute_value(
    position: Tuple[float, float],
    value: float,
    grid: torch.Tensor,
    grid_size: Tuple[int, int],
) -> None:
    """
    Distribute a value across grid cells using bilinear interpolation.

    This preserves continuous position information by distributing values
    across the four nearest grid cells based on the fractional position components.

    Args:
        position: (x, y) continuous coordinates
        value: Value to distribute
        grid: Target grid tensor of shape (H, W)
        grid_size: (width, height) of the grid
    """
    x, y = position
    width, height = grid_size

    # Get the four nearest grid cells
    x_floor = int(math.floor(x))
    y_floor = int(math.floor(y))
    x_ceil = min(x_floor + 1, width - 1)
    y_ceil = min(y_floor + 1, height - 1)

    # Calculate interpolation weights
    x_frac = x - x_floor
    y_frac = y - y_floor

    # Ensure we don't go out of bounds
    x_floor = max(0, min(x_floor, width - 1))
    y_floor = max(0, min(y_floor, height - 1))
    x_ceil = max(0, min(x_ceil, width - 1))
    y_ceil = max(0, min(y_ceil, height - 1))

    # Bilinear interpolation weights
    w00 = (1 - x_frac) * (1 - y_frac)  # bottom-left
    w01 = (1 - x_frac) * y_frac  # top-left
    w10 = x_frac * (1 - y_frac)  # bottom-right
    w11 = x_frac * y_frac  # top-right

    # Distribute the value
    grid[y_floor, x_floor] += value * w00
    grid[y_ceil, x_floor] += value * w01
    grid[y_floor, x_ceil] += value * w10
    grid[y_ceil, x_ceil] += value * w11


class Environment(AECEnv):
    """Multi-agent simulation environment for AgentFarm.

    The Environment class manages a 2D world containing agents and resources,
    supporting multi-agent reinforcement learning through the PettingZoo API.
    It handles agent lifecycle, resource management, spatial relationships,
    combat interactions, and evolutionary dynamics.

    The environment provides:
    - Spatial indexing for efficient proximity queries
    - Resource spawning and regeneration
    - Agent birth, death, and reproduction
    - Combat and cooperation mechanics
    - Observation generation for RL training
    - Comprehensive metrics tracking
    - Database logging for analysis

    Attributes:
        width (int): Environment width in grid units
        height (int): Environment height in grid units
        agents (list): List of active agent IDs (PettingZoo requirement)
        resources (list): List of resource nodes in the environment
        time (int): Current simulation time step
        simulation_id (str): Unique identifier for this simulation run
        spatial_index (SpatialIndex): Efficient spatial query system
        resource_manager (ResourceManager): Handles resource lifecycle
        metrics_tracker (MetricsTracker): Tracks simulation statistics
        db (Database): Optional database for logging simulation data

    Inherits from:
        AECEnv: PettingZoo's Agent-Environment-Cycle environment base class
    """

    def __init__(
        self,
        width: int,
        height: int,
        resource_distribution: Union[Dict[str, Any], Callable],
        db_path: str = "simulation.db",
        max_resource: Optional[float] = None,
        config: Optional[Any] = None,
        simulation_id: Optional[str] = None,
        seed: Optional[int] = None,
        initial_agents: Optional[List[Any]] = None,
    ) -> None:
        """Initialize the AgentFarm environment.

        Creates a new simulation environment with specified dimensions and
        configuration. Sets up spatial indexing, resource management, metrics
        tracking, and database logging.

        Parameters
        ----------
        width : int
            Width of the environment in grid units
        height : int
            Height of the environment in grid units
        resource_distribution : dict or callable
            Configuration for initial resource placement. Can be a dictionary
            specifying resource parameters or a callable that generates resources.
        db_path : str, optional
            Path to SQLite database file for logging simulation data.
            Defaults to "simulation.db".
        max_resource : float, optional
            Maximum resource amount for normalization. If None, uses config
            value or defaults to reasonable value.
        config : object, optional
            Configuration object containing simulation parameters like
            max_steps, agent counts, observation settings, etc.
        simulation_id : str, optional
            Unique identifier for this simulation. If None, generates a new
            short UUID.
        seed : int, optional
            Random seed for deterministic simulation. If None, uses config
            seed or remains non-deterministic.
        initial_agents : list, optional
            Pre-instantiated agents to add to the environment at initialization.
            If provided, these agents will be added after environment setup.

        Raises
        ------
        ValueError
            If width or height are non-positive
        Exception
            If database setup fails or resource initialization fails
        """
        super().__init__()
        # Set seed if provided
        self.seed_value = (
            seed
            if seed is not None
            else config.seed if config and config.seed else None
        )
        if self.seed_value is not None:
            random.seed(self.seed_value)
            np.random.seed(self.seed_value)
            try:
                torch.manual_seed(self.seed_value)
            except RuntimeError as e:
                logger.warning(
                    "Failed to seed torch with value %s: %s", self.seed_value, e
                )

        # Initialize basic attributes
        self.width = width
        self.height = height
        self.agents = []
        self._agent_objects = {}  # Internal mapping: agent_id -> agent object
        self.resources = []
        self.time = 0

        # Initialize identity service (deterministic if seed provided)
        self.identity = Identity(IdentityConfig(deterministic_seed=self.seed_value))

        # Store simulation ID
        self.simulation_id = simulation_id or self.identity.simulation_id()

        # Setup database and get initialized database instance
        db_result = setup_db(db_path, self.simulation_id, config.to_dict() if config else None)
        if isinstance(db_result, tuple):
            self.db = db_result[0]  # Extract database object from tuple
        else:
            self.db = db_result

        # Use self.identity for all ID needs
        self.max_resource = max_resource
        self.config = config
        self.resource_distribution = resource_distribution
        self.max_steps = (
            config.max_steps if config and hasattr(config, "max_steps") else 1000
        )

        # Initialize action mapping based on configuration and available actions
        self._initialize_action_mapping()

        # Initialize PettingZoo required attributes
        self.agent_selection = None
        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self.observations = {}

        # Initialize spatial index for efficient spatial queries
        self.spatial_index = SpatialIndex(self.width, self.height)
        # Provide spatial service via adapter around spatial_index
        self.spatial_service = SpatialIndexAdapter(self.spatial_index)

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()
        # Provide metrics service delegating to the environment
        self.metrics_service = EnvironmentMetricsService(self)

        # Initialize resource sharing counters
        self.resources_shared = 0.0
        self.resources_shared_this_step = 0.0

        # Cache total resources to avoid recomputing on every step
        self.cached_total_resources = 0.0

        # Initialize cycle tracking for proper timestep semantics
        self._agents_acted_this_cycle = 0
        self._cycle_complete = False

        # Initialize resource manager
        self.resource_manager = ResourceManager(
            width=self.width,
            height=self.height,
            config=self.config,
            seed=self.seed_value,
            database_logger=self.db.logger if self.db else None,
            spatial_index=self.spatial_index,
        )

        # Initialize environment
        self.initialize_resources(self.resource_distribution)

        # Add observation space setup:
        self._setup_observation_space(self.config)

        # Add action space setup call:
        self._setup_action_space()

        # Initialize agent observations mapping before adding any agents
        self.agent_observations = {}

        # If pre-instantiated agents are provided, add them now
        if initial_agents:
            for agent in initial_agents:
                self.add_agent(agent)

        # Update spatial index references now that resources and agents are initialized
        # Pass a live view of agent objects to avoid accidental string ID lists
        self.spatial_index.set_references(
            list(self._agent_objects.values()), self.resources
        )
        self.spatial_index.update()

        # Perception profiler accumulators
        self._perception_profile = {
            "spatial_query_time_s": 0.0,
            "bilinear_time_s": 0.0,
            "nearest_time_s": 0.0,
            "bilinear_points": 0,
            "nearest_points": 0,
        }

    def _initialize_action_mapping(self) -> None:
        """Initialize the action mapping based on configuration and available actions.

        Creates a dynamic mapping between Action enum values and action registry names.
        Allows for flexible action configuration where simulations can enable/disable
        specific actions and handle missing actions gracefully.

        The mapping supports:
        - Configuration-driven action enabling/disabling
        - Dynamic discovery of available actions in registry
        - Validation of required actions
        - Graceful handling of missing actions
        """
        # Default mapping from ActionType enum to action registry names
        default_action_mapping = {
            ActionType.DEFEND: "defend",
            ActionType.ATTACK: "attack",
            ActionType.GATHER: "gather",
            ActionType.SHARE: "share",
            ActionType.MOVE: "move",
            ActionType.REPRODUCE: "reproduce",
            ActionType.PASS: "pass",
        }

        # Get enabled actions from config, or use all available if not specified
        if self.config and hasattr(self.config, "enabled_actions"):
            enabled_actions = getattr(self.config, "enabled_actions")
            if isinstance(enabled_actions, list):
                # Convert list of action names to mapping
                self._action_mapping = {}
                for action_name in enabled_actions:
                    # Find the corresponding Action enum value
                    for action_enum, registry_name in default_action_mapping.items():
                        if registry_name == action_name:
                            self._action_mapping[action_enum] = action_name
                            break
            else:
                self._action_mapping = default_action_mapping
        else:
            self._action_mapping = default_action_mapping

        # Validate that all mapped actions exist in the registry
        missing_actions = []
        for action_enum, action_name in self._action_mapping.items():
            if not action_registry.get(action_name):
                missing_actions.append(action_name)

        if missing_actions:
            logging.warning("Missing actions in registry: %s", missing_actions)
            # Remove missing actions from mapping
            self._action_mapping = {
                k: v
                for k, v in self._action_mapping.items()
                if v not in missing_actions
            }

        # Log the final action mapping
        available_actions = list(self._action_mapping.values())
        logging.info(
            "Initialized action mapping with %s actions: %s",
            len(available_actions),
            available_actions,
        )

    @property
    def agent_objects(self) -> List[Any]:
        """Backward compatibility property to get all agent objects as a list."""
        return list(self._agent_objects.values())

    def mark_positions_dirty(self) -> None:
        """Public method for agents to mark positions as dirty when they move."""
        self.spatial_index.mark_positions_dirty()

    def get_nearby_agents(
        self, position: Tuple[float, float], radius: float
    ) -> List[Any]:
        """Find all agents within radius of position.

        Parameters
        ----------
        position : tuple
            (x, y) coordinates to search around
        radius : float
            Search radius

        Returns
        -------
        list
            List of agents within radius
        """
        # Use generic method with "agents" index
        nearby = self.spatial_index.get_nearby(position, radius, ["agents"])
        return nearby.get("agents", [])

    def get_nearby_resources(
        self, position: Tuple[float, float], radius: float
    ) -> List[Any]:
        """Find all resources within radius of position.

        Parameters
        ----------
        position : tuple
            (x, y) coordinates to search around
        radius : float
            Search radius

        Returns
        -------
        list
            List of resources within radius
        """
        # Use generic method with "resources" index
        nearby = self.spatial_index.get_nearby(position, radius, ["resources"])
        return nearby.get("resources", [])

    def get_nearest_resource(self, position: Tuple[float, float]) -> Optional[Any]:
        """Find nearest resource to position.

        Parameters
        ----------
        position : tuple
            (x, y) coordinates to search from

        Returns
        -------
        Resource or None
            Nearest resource if any exist
        """
        # Use generic method with "resources" index
        nearest = self.spatial_index.get_nearest(position, ["resources"])
        return nearest.get("resources")

    # Resource IDs are managed by ResourceManager

    def consume_resource(self, resource: Any, amount: float) -> float:
        """Consume resources from a specific resource node.

        Parameters
        ----------
        resource : Resource
            Resource to consume from
        amount : float
            Amount to consume

        Returns
        -------
        float
            Actual amount consumed
        """
        return self.resource_manager.consume_resource(resource, amount)

    def initialize_resources(
        self, distribution: Union[Dict[str, Any], Callable]
    ) -> None:
        """Initialize resources in the environment using ResourceManager.

        Creates initial resource nodes based on the provided distribution
        configuration. Resources are placed according to the distribution
        parameters and assigned proper amounts.

        Parameters
        ----------
        distribution : dict or callable
            Resource distribution configuration specifying how resources
            should be placed in the environment. Can include parameters
            like density, clustering, amount ranges, etc.

        Notes
        -----
        This method delegates to ResourceManager for actual resource creation
        and then synchronizes the environment's resource list with the manager.
        The next_resource_id counter is also synchronized to ensure unique IDs.
        """
        # Use ResourceManager to initialize resources (passes through to original logic)
        resources = self.resource_manager.initialize_resources(distribution)

        # Update environment's resource list to match ResourceManager
        self.resources = self.resource_manager.resources

        # Update cached total resources after initialization
        self.cached_total_resources = sum(r.amount for r in self.resources)

        # Resource IDs are fully managed by ResourceManager

    def remove_agent(self, agent: Any) -> None:
        """Remove an agent from the environment.

        Handles complete agent removal including death recording, cleanup of
        internal data structures, and spatial index updates. This is typically
        called when an agent dies or is otherwise removed from the simulation.

        Parameters
        ----------
        agent : Agent
            The agent object to remove. Must have an agent_id attribute.

        Notes
        -----
        This method:
        - Records the death event for metrics tracking
        - Removes the agent from internal object mapping
        - Removes the agent from PettingZoo's agent list
        - Marks spatial index as dirty for next update
        - Cleans up agent observation data
        """
        self.record_death()
        if agent.agent_id in self._agent_objects:
            del self._agent_objects[agent.agent_id]
        if agent.agent_id in self.agents:
            self.agents.remove(agent.agent_id)  # Remove from PettingZoo agents list
        self.spatial_index.mark_positions_dirty()  # Mark positions as dirty when agent is removed
        if agent.agent_id in self.agent_observations:
            del self.agent_observations[agent.agent_id]

    def log_interaction_edge(
        self,
        source_type: str,
        source_id: str,
        target_type: str,
        target_id: str,
        interaction_type: str,
        action_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an interaction as an edge between nodes if database is enabled.

        Records interactions between agents and other entities (agents, resources)
        as graph edges in the database for network analysis and relationship tracking.

        Parameters
        ----------
        source_type : str
            Type of the source entity (e.g., 'agent', 'resource')
        source_id : str
            Unique identifier of the source entity
        target_type : str
            Type of the target entity (e.g., 'agent', 'resource')
        target_id : str
            Unique identifier of the target entity
        interaction_type : str
            Type of interaction (e.g., 'attack', 'share', 'gather')
        action_type : str, optional
            Specific action type if different from interaction_type
        details : dict, optional
            Additional interaction details (e.g., amount transferred, damage dealt)

        Notes
        -----
        If no database is configured, this method returns silently without logging.
        Errors during logging are caught and logged as warnings to prevent
        simulation disruption.
        """
        if self.db is None:
            return
        try:
            self.db.logger.log_interaction_edge(
                step_number=self.time,
                source_type=source_type,
                source_id=source_id,
                target_type=target_type,
                target_id=target_id,
                interaction_type=interaction_type,
                action_type=action_type,
                details=details,
            )
        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Failed to log interaction edge: %s", e)

    def log_reproduction_event(
        self,
        step_number: int,
        parent_id: str,
        success: bool,
        parent_resources_before: float,
        parent_resources_after: float,
        offspring_id: Optional[str] = None,
        offspring_initial_resources: Optional[float] = None,
        failure_reason: Optional[str] = None,
        parent_position: Optional[Tuple[float, float]] = None,
        parent_generation: Optional[int] = None,
        offspring_generation: Optional[int] = None,
    ) -> None:
        """Log a reproduction event if database is enabled.

        Records reproduction attempts and outcomes in the database for analysis
        of evolutionary dynamics, resource costs, and population growth patterns.

        Parameters
        ----------
        step_number : int
            Current simulation step when reproduction occurred
        parent_id : str
            Unique identifier of the parent agent attempting reproduction
        success : bool
            Whether the reproduction attempt was successful
        parent_resources_before : float
            Parent's resource level before reproduction attempt
        parent_resources_after : float
            Parent's resource level after reproduction attempt
        offspring_id : str, optional
            Unique identifier of the offspring if reproduction succeeded
        offspring_initial_resources : float, optional
            Initial resource level assigned to offspring
        failure_reason : str, optional
            Description of why reproduction failed (if applicable)
        parent_position : tuple[float, float], optional
            Position of the parent agent at time of reproduction
        parent_generation : int, optional
            Generation number of the parent agent
        offspring_generation : int, optional
            Generation number assigned to the offspring

        Notes
        -----
        If no database is configured, this method returns silently without logging.
        Errors during logging are caught and logged as warnings to prevent
        simulation disruption.
        """
        if self.db is None:
            return
        try:
            self.db.log_reproduction_event(
                step_number=step_number,
                parent_id=parent_id,
                success=success,
                parent_resources_before=parent_resources_before,
                parent_resources_after=parent_resources_after,
                offspring_id=offspring_id,
                offspring_initial_resources=offspring_initial_resources,
                failure_reason=failure_reason,
                parent_position=parent_position,
                parent_generation=parent_generation,
                offspring_generation=offspring_generation,
            )
        except (ValueError, TypeError, AttributeError) as e:
            logger.error("Failed to log reproduction event: %s", e)

    def update(self) -> None:
        """Update environment state for current time step.

        Performs a full environment update including resource regeneration,
        metrics calculation, spatial index updates, and time advancement.
        This method is called once per simulation step to advance the world state.

        The update process includes:
        - Resource regeneration and decay using ResourceManager
        - Metrics calculation and logging for current state
        - Spatial index updates for efficient proximity queries
        - Counter resets for step-specific tracking
        - Time step increment

        Raises
        ------
        Exception
            If any critical update operation fails, the exception is logged
            and re-raised to halt the simulation.

        Notes
        -----
        This method should be called after all agents have taken their actions
        for the current time step but before the next step begins.
        """
        try:
            # Update resources using ResourceManager
            resource_stats = self.resource_manager.update_resources(self.time)

            # Update cached total resources after resource regeneration
            self.cached_total_resources = sum(r.amount for r in self.resources)

            # Log resource update statistics if needed
            if resource_stats["regeneration_events"] > 0:
                logger.debug(
                    "Resource update: %s resources regenerated",
                    resource_stats["regeneration_events"],
                )

            # Calculate and log metrics
            metrics = self._calculate_metrics()
            self.metrics_tracker.update_metrics(
                metrics,
                db=self.db,
                time=self.time,
                agent_objects=self._agent_objects,
                resources=self.resources,
            )

            # Update spatial index
            self.spatial_index.update()

            # Reset counters for next step
            self.resources_shared_this_step = 0
            self.combat_encounters_this_step = 0
            self.successful_attacks_this_step = 0

            # Increment time step
            self.time += 1

        except (RuntimeError, ValueError, AttributeError) as e:
            logging.error("Error in environment update: %s", e)
            raise

    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate various metrics for the current simulation state.

        Computes comprehensive metrics about the current state of the simulation
        including agent statistics, resource distribution, population dynamics,
        and interaction patterns. These metrics are used for analysis and
        database logging.

        Returns
        -------
        dict
            Dictionary containing calculated metrics with keys like:
            - agent_count: Number of active agents
            - resource_count: Number of resource nodes
            - total_resources: Sum of all resource amounts
            - average_health: Mean agent health
            - population_density: Agents per unit area
            - And other simulation-specific metrics

        Notes
        -----
        This method delegates to MetricsTracker for actual calculations,
        passing the current agent objects, resources, time, and configuration.
        """
        return self.metrics_tracker.calculate_metrics(
            self._agent_objects, self.resources, self.time, self.config
        )

    def get_next_agent_id(self) -> str:
        """Generate a unique short ID for an agent using environment's seed.

        Returns
        -------
        str
            A unique short ID string
        """
        # Delegate to identity service (respects deterministic seed if set)
        return self.identity.agent_id()

    def state(self) -> EnvironmentState:
        """Get current environment state (PettingZoo AECEnv requirement).

        Creates an EnvironmentState object containing the current state of all
        agents, resources, and environment properties for serialization and
        analysis purposes.

        Returns
        -------
        EnvironmentState
            Immutable snapshot of the current environment state containing:
            - All agent states and positions
            - Resource locations and amounts
            - Environment dimensions and time
            - Spatial index state
        """
        return EnvironmentState.from_environment(self)

    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        """Check if a position is valid within the environment bounds.

        Validates that the given coordinates are within the rectangular bounds
        of the environment grid, inclusive of the boundary values. This is used
        for boundary checking before agent movement and resource placement.

        Parameters
        ----------
        position : tuple of float
            (x, y) coordinates to check, where x is the horizontal position
            and y is the vertical position

        Returns
        -------
        bool
            True if position is within bounds [0, width] x [0, height] inclusive,
            False otherwise. Note that boundary values (0, width, height) are
            considered valid positions.
        """
        x, y = position
        return (0 <= x <= self.width) and (0 <= y <= self.height)

    def record_birth(self) -> None:
        """Record a birth event in the metrics tracker.

        Increments the birth counter in the metrics tracker to track population
        dynamics and reproduction statistics. This is called whenever a new
        agent is created through reproduction or initial spawning.
        """
        self.metrics_tracker.record_birth()

    def record_death(self) -> None:
        """Record a death event in the metrics tracker.

        Increments the death counter in the metrics tracker to track population
        dynamics and mortality statistics. This is called whenever an agent
        dies due to starvation, combat, or other causes.
        """
        self.metrics_tracker.record_death()

    def record_combat_encounter(self) -> None:
        """Record a combat encounter event.

        Increments the combat encounter counter to track the frequency of
        combat interactions between agents. Used for analyzing conflict
        patterns and agent behavior.
        """
        self.metrics_tracker.record_combat_encounter()

    def record_successful_attack(self) -> None:
        """Record a successful attack event.

        Increments the successful attack counter to track combat effectiveness
        and agent combat success rates. This helps analyze combat dynamics
        and evolutionary fitness.
        """
        self.metrics_tracker.record_successful_attack()

    def record_resources_shared(self, amount: float) -> None:
        """Record resources shared between agents.

        Tracks cooperative resource sharing behavior by recording the amount
        of resources transferred between agents. Updates both the metrics
        tracker and internal counters for step-level and total tracking.

        Parameters
        ----------
        amount : float
            The amount of resources that were shared between agents. Must be
            a positive value representing the resource transfer amount.

        Notes
        -----
        This method updates three tracking variables:
        - MetricsTracker for long-term statistics and database logging
        - self.resources_shared for total resources shared across simulation
        - self.resources_shared_this_step for per-step tracking and analysis
        """
        self.metrics_tracker.record_resources_shared(amount)
        self.resources_shared += amount
        self.resources_shared_this_step += amount

    def close(self) -> None:
        """Clean up environment resources and close database connections.

        Properly closes any open database connections to ensure data integrity
        and prevent resource leaks. This method should be called when the
        environment is no longer needed or when shutting down the simulation.

        Notes
        -----
        This method safely handles cases where the database connection may
        not exist or may already be closed, preventing exceptions during
        cleanup operations.
        """
        if hasattr(self, "db") and self.db is not None:
            self.db.close()

    def add_agent(self, agent: Any) -> None:
        """Add an agent to the environment with efficient database logging.

        Registers a new agent in the environment, adding it to internal tracking
        structures, the spatial index, and optionally logging its creation to
        the database. This method handles all necessary setup for a new agent.

        Parameters
        ----------
        agent : Agent
            The agent object to add. Must have attributes like agent_id, position,
            resource_level, etc. The agent should be properly initialized before
            being added to the environment.

        Notes
        -----
        This method:
        - Extracts agent data for database logging
        - Adds agent to internal object mapping and PettingZoo agent list
        - Marks spatial index as dirty for next update
        - Batch logs agent data to database if available
        - Creates observation tracking for the agent

        The agent data logged includes birth time, position, resources, health,
        genome information (if applicable), and action weights.
        """
        # If the agent supports dependency injection, supply services and config
        try:
            # Spatial service is required for agent act()/perception; ensure present
            if hasattr(agent, "spatial_service"):
                agent.spatial_service = self.spatial_service
            else:
                setattr(agent, "spatial_service", self.spatial_service)

            # Optional services: inject only when missing/None
            if hasattr(agent, "metrics_service") and agent.metrics_service is None:
                agent.metrics_service = self.metrics_service
            if hasattr(agent, "logging_service") and agent.logging_service is None:
                agent.logging_service = EnvironmentLoggingService(self)
            if (
                hasattr(agent, "validation_service")
                and agent.validation_service is None
            ):
                agent.validation_service = EnvironmentValidationService(self)
            if hasattr(agent, "time_service") and agent.time_service is None:
                agent.time_service = EnvironmentTimeService(self)
            if hasattr(agent, "lifecycle_service") and agent.lifecycle_service is None:
                agent.lifecycle_service = EnvironmentAgentLifecycleService(self)
            if hasattr(agent, "config") and getattr(agent, "config", None) is None:
                agent.config = self.config

            # Environment reference for action/observation space access
            if hasattr(agent, "environment"):
                agent.environment = self
            else:
                setattr(agent, "environment", self)

            # Validate required dependencies after injection
            if getattr(agent, "spatial_service", None) is None:
                raise ValueError(
                    "Agent %s missing spatial_service after injection"
                    % getattr(agent, "agent_id", "?")
                )
        except (AttributeError, ValueError, TypeError) as e:
            logger.error(
                "Failed to inject services for agent %s: %s",
                getattr(agent, "agent_id", "?"),
                e,
            )
            raise

        agent_data = [
            {
                "simulation_id": self.simulation_id,
                "agent_id": agent.agent_id,
                "birth_time": self.time,
                "agent_type": agent.__class__.__name__,
                "position": agent.position,
                "initial_resources": agent.resource_level,
                "starting_health": agent.starting_health,
                "starvation_counter": agent.starvation_counter,
                "genome_id": getattr(agent, "genome_id", None),
                "generation": getattr(agent, "generation", 0),
                "action_weights": agent.get_action_weights(),
            }
        ]

        # Add to environment
        self._agent_objects[agent.agent_id] = agent
        self.agents.append(agent.agent_id)  # Add to PettingZoo agents list

        # Mark positions as dirty when new agent is added
        self.spatial_index.mark_positions_dirty()

        # Batch log to database using SQLAlchemy
        if self.db is not None:
            self.db.logger.log_agents_batch(agent_data)

        self.agent_observations[agent.agent_id] = AgentObservation(
            self.observation_config
        )

    def cleanup(self) -> None:
        """Clean up environment resources.

        Properly closes database connections and flushes any pending data
        to ensure no data loss. This method should be called when the
        simulation is finished or the environment is being destroyed.

        Notes
        -----
        This method:
        - Flushes all pending database buffers
        - Closes database connections gracefully
        - Handles errors during cleanup to prevent crashes

        Cleanup errors are logged but do not raise exceptions to allow
        for graceful shutdown even if some resources fail to close properly.
        """
        try:
            if hasattr(self, "db") and self.db is not None:
                # Use logger for buffer flushing
                if hasattr(self.db, "logger"):
                    self.db.logger.flush_all_buffers()
                self.db.close()
        except (OSError, AttributeError, ValueError) as e:
            logger.error("Error during environment cleanup: %s", e)

    def __del__(self) -> None:
        """Ensure cleanup on deletion.

        Destructor that calls cleanup() to ensure proper resource cleanup
        when the Environment object is garbage collected. This provides
        a safety net in case cleanup() is not called explicitly.

        Notes
        -----
        While this provides a backup cleanup mechanism, it's better practice
        to call cleanup() explicitly rather than relying on the destructor,
        as the timing of garbage collection is not guaranteed in Python.
        This destructor ensures database connections are properly closed and
        any buffered data is flushed to prevent data loss.
        """
        self.cleanup()

    def action_space(self, agent: Optional[str] = None) -> spaces.Discrete:
        """Get the action space for an agent (PettingZoo API).

        Returns the action space defining all possible actions an agent can take.
        The action space is dynamically configured based on enabled actions and
        may change during curriculum learning or action pruning scenarios.

        Parameters
        ----------
        agent : str, optional
            Agent ID. If None, returns the general action space that applies
            to all agents. The action space is the same for all agents in
            this environment.

        Returns
        -------
        gymnasium.spaces.Discrete
            The action space containing all enabled actions. Each action is
            represented by an integer index from 0 to n_actions-1, where
            n_actions is the number of currently enabled actions.

        Notes
        -----
        - The mapping from indices → actions follows the current `_action_mapping`
          order, which can change at runtime via `update_action_space`.
        - RL agents should be resilient to dynamic action-space size changes or
          update policies that keep the space fixed during training.
        """
        return self._action_space

    def observation_space(self, agent: Optional[str] = None) -> spaces.Box:
        """Get the observation space for an agent (PettingZoo API).

        Returns the observation space defining the shape and data type of
        observations provided to agents. The observation space is configured
        based on the observation configuration and includes multiple channels
        for different types of environmental information.

        Parameters
        ----------
        agent : str, optional
            Agent ID. If None, returns the general observation space that applies
            to all agents. All agents receive observations with the same shape
            and structure.

        Returns
        -------
        gymnasium.spaces.Box
            The observation space defining the shape and bounds of observations.
            Shape is (NUM_CHANNELS, S, S) where S = 2*R + 1 and R is the
            observation radius. Values are normalized to [0, 1] range.

        Notes
        -----
        - The dtype reflects the configured torch dtype mapped to a numpy dtype
          for space definition (bfloat16 maps to float32 for numpy compatibility).
        - Channel layout is defined by the channel registry in `farm.core.channels`.
        """
        return self._observation_space

    def observe(self, agent: str) -> np.ndarray:
        """Returns the observation an agent currently can make.

        Required by PettingZoo API. Generates a multi-channel observation tensor
        containing the agent's local view of the environment, including nearby
        resources, agents, and the agent's own state.

        Parameters
        ----------
        agent : str
            Agent identifier for which to generate the observation

        Returns
        -------
        np.ndarray
            Observation tensor for the agent with shape (channels, height, width).
            If the agent is dead or doesn't exist, returns a zero tensor of
            the appropriate shape. Values are normalized to [0, 1] range.
        """
        return self._get_observation(agent)

    def _setup_observation_space(self, config: Optional[Any]) -> None:
        """Setup the observation space based on configuration.

        Parameters
        ----------
        config : object, optional
            Configuration object containing observation settings.
        """
        if config and hasattr(config, "observation") and config.observation is not None:
            self.observation_config = config.observation
        else:
            self.observation_config = ObservationConfig()
        S = 2 * self.observation_config.R + 1
        # Robust numpy dtype mapping from torch dtype or string
        torch_dtype = (
            getattr(torch, self.observation_config.dtype)
            if isinstance(self.observation_config.dtype, str)
            else self.observation_config.dtype
        )
        if torch_dtype in (torch.float32, torch.float):
            np_dtype = np.float32
        elif torch_dtype in (torch.float64, torch.double):
            np_dtype = np.float64
        elif torch_dtype in (torch.float16, torch.half):
            np_dtype = np.float16
        elif torch_dtype == torch.bfloat16:
            # numpy has no bfloat16; use float32 for the observation space dtype
            np_dtype = np.float32
        else:
            np_dtype = np.float32
        self._observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(NUM_CHANNELS, S, S), dtype=np_dtype
        )

    def _setup_action_space(self) -> None:
        """Setup the action space with all available actions."""
        # Use only the enabled actions from the mapping instead of full ActionType enum
        self._action_space = spaces.Discrete(len(self._action_mapping))

        # Create a list of enabled ActionType values for dynamic remapping
        # This ensures consistent ordering: action index 0 maps to first enabled action, etc.
        self._enabled_action_types = list(self._action_mapping.keys())

    def update_action_space(
        self, new_enabled_actions: Optional[List[str]] = None
    ) -> None:
        """Update the action space when enabled actions configuration changes.

        This method should be called when the curriculum or configuration
        changes the set of enabled actions mid-simulation. It updates the
        action mapping, recreates the enabled action types list, and resizes
        the action space accordingly.

        Parameters
        ----------
        new_enabled_actions : list of str, optional
            New list of action names to enable. If None, restores full action space.
            Each action name should correspond to an action in the registry
            (e.g., ["move", "gather", "attack"]).

        Notes
        -----
        This method:
        - Updates self.config.enabled_actions if new_enabled_actions provided
        - Reinitializes the action mapping with the new configuration
        - Recreates the enabled action types list for dynamic remapping
        - Resizes the action space to match the new number of enabled actions
        - Logs the updated action space size and available actions

        Warning: This changes the action space size, which may affect RL agents
        that are not designed to handle dynamic action spaces.
        """
        # Update config if new enabled actions provided
        if new_enabled_actions is not None:
            if self.config is None:
                # Create a basic config object if none exists
                self.config = SimulationConfig()
            # Use setattr for dynamic attribute assignment (same pattern as original code)
            setattr(self.config, "enabled_actions", new_enabled_actions)
        else:
            # If None is passed, remove the enabled_actions attribute to restore full space
            if self.config and hasattr(self.config, "enabled_actions"):
                delattr(self.config, "enabled_actions")

        # Reinitialize action mapping with new configuration
        self._initialize_action_mapping()

        # Update action space and enabled action types list
        self._setup_action_space()

        # Log the update
        available_actions = list(self._action_mapping.values())
        logging.info(
            "Action space updated: %s actions available: %s",
            len(available_actions),
            available_actions,
        )

    def get_initial_agent_count(self) -> int:
        """Calculate the number of initial agents (born at time 0) dynamically.

        Counts how many agents were present at the start of the simulation
        by checking their birth_time attribute. This is useful for analysis
        and metrics that need to distinguish between initial population and
        agents born during the simulation.

        Returns
        -------
        int
            Number of agents with birth_time == 0, representing the initial
            population that was present when the simulation started.

        Notes
        -----
        This count is calculated dynamically by iterating through all current
        agents, so it only includes agents that are still alive. Dead agents
        are not counted even if they were part of the initial population.
        """
        return len(
            [
                agent
                for agent in self._agent_objects.values()
                if getattr(agent, "birth_time", 0) == 0
            ]
        )

    # _create_initial_agents removed: agents should be created outside the environment

    def _get_observation(self, agent_id: str) -> np.ndarray:
        """Generate an observation for a specific agent.

        Creates a multi-channel observation tensor containing information about
        the local environment around the agent, including resources, nearby agents,
        and the agent's own state. The observation follows the configured format
        and dimensions for reinforcement learning.

        Parameters
        ----------
        agent_id : str
            The ID of the agent to generate an observation for.

        Returns
        -------
        np.ndarray
            The observation tensor for the agent with shape (channels, height, width).
            Returns zero tensor if agent is None or not alive.

        Notes
        -----
        TODO: #! Deprecate. The observation will come from the spatial index.

        The observation includes:
        - Resource distribution in the agent's field of view
        - Positions and health of nearby allies and enemies
        - Agent's own health and position
        - Empty layers for obstacles and terrain cost (future features)

        The observation is generated using the AgentObservation class and
        follows the perception system defined in the observation configuration.
        """
        agent = self._agent_objects.get(agent_id)
        if agent is None or not agent.alive:
            return np.zeros(
                self._observation_space.shape, dtype=self._observation_space.dtype
            )

        # Assume width and height are integers for grid
        height, width = int(self.height), int(self.width)

        # Get discretization method from config
        # Resolve discretization/interpolation from nested environment config when available
        if self.config and getattr(self.config, "environment", None) is not None:
            discretization_method = getattr(
                self.config.environment, "position_discretization_method", "floor"
            )
            use_bilinear = getattr(
                self.config.environment, "use_bilinear_interpolation", True
            )
        else:
            discretization_method = (
                getattr(self.config, "position_discretization_method", "floor")
                if self.config
                else "floor"
            )
            use_bilinear = (
                getattr(self.config, "use_bilinear_interpolation", True)
                if self.config
                else True
            )

        # Agent position as (y, x) using configured discretization method
        grid_size = (width, height)
        ax, ay = discretize_position_continuous(
            agent.position, grid_size, discretization_method
        )

        # Ensure spatial index is up to date before observation generation
        self.spatial_index.update()

        # Build local resource layer directly (avoid full-world grids)
        R = self.observation_config.R
        S = 2 * R + 1
        resource_local = torch.zeros(
            (S, S),
            dtype=self.observation_config.torch_dtype,
            device=self.observation_config.device,
        )
        # Max amount resolution prefers nested resources config when available
        if self.config and getattr(self.config, "resources", None) is not None:
            max_amount = self.max_resource or self.config.resources.max_resource_amount
        else:
            max_amount = self.max_resource or (
                getattr(self.config, "max_resource_amount", 10) if self.config else 10
            )

        # Query nearby resources within a radius covering the local window
        # Use slightly larger than R to capture bilinear spread near the boundary
        try:
            _tq0 = _time.perf_counter()
            nearby = self.spatial_index.get_nearby(agent.position, R + 1, ["resources"])
            nearby_resources = nearby.get("resources", [])
            _tq1 = _time.perf_counter()
            self._perception_profile["spatial_query_time_s"] += max(0.0, _tq1 - _tq0)
        except AttributeError as e:
            # Handle case where spatial_index or its attributes are None
            logger.warning("Spatial index not properly initialized: %s", e)
            nearby_resources = []
        except (ValueError, TypeError) as e:
            # Handle invalid input parameters (position format, radius type)
            logger.warning("Invalid parameters for spatial query: %s", e)
            nearby_resources = []
        except IndexError as e:
            # Handle case where KD-tree indices are out of bounds
            logger.warning("Index error in spatial query: %s", e)
            nearby_resources = []
        except (RuntimeError, KeyError) as e:
            # Catch any other unexpected errors for debugging
            logger.exception(
                "Unexpected error querying nearby resources in spatial index"
            )
            nearby_resources = []

        if use_bilinear:
            _tb0 = _time.perf_counter()
            for res in nearby_resources:
                # Convert world to local continuous coords where (R, R) is agent center
                lx = float(res.position[0]) - (ax - R)
                ly = float(res.position[1]) - (ay - R)
                bilinear_distribute_value(
                    (lx, ly),
                    float(res.amount) / float(max_amount),
                    resource_local,
                    (S, S),
                )
                # 4 target points per bilinear distribution
                self._perception_profile["bilinear_points"] += 4
            _tb1 = _time.perf_counter()
            self._perception_profile["bilinear_time_s"] += max(0.0, _tb1 - _tb0)
        else:
            _tn0 = _time.perf_counter()
            for res in nearby_resources:
                rx, ry = discretize_position_continuous(
                    res.position, (width, height), discretization_method
                )
                lx = rx - (ax - R)
                ly = ry - (ay - R)
                if 0 <= lx < S and 0 <= ly < S:
                    resource_local[int(ly), int(lx)] += float(res.amount) / float(
                        max_amount
                    )
                    self._perception_profile["nearest_points"] += 1
            _tn1 = _time.perf_counter()
            self._perception_profile["nearest_time_s"] += max(0.0, _tn1 - _tn0)

        # Empty local layers
        obstacles_local = torch.zeros_like(resource_local)
        terrain_cost_local = torch.zeros_like(resource_local)

        world_layers = {
            "RESOURCES": resource_local,
            "OBSTACLES": obstacles_local,
            "TERRAIN_COST": terrain_cost_local,
        }

        self_hp01 = agent.current_health / agent.starting_health

        obs = self.agent_observations[agent_id]
        obs.perceive_world(
            world_layers=world_layers,
            agent_world_pos=(ay, ax),
            self_hp01=self_hp01,
            allies=None,  # Let observation system use spatial index for efficiency
            enemies=None,  # Let observation system use spatial index for efficiency
            goal_world_pos=None,  # TODO: Set if needed
            recent_damage_world=[],  # TODO: Implement if needed
            ally_signals_world=[],  # TODO: Implement if needed
            trails_world_points=[],  # TODO: Implement if needed
            spatial_index=self.spatial_index,
            agent_object=agent,
            agent_orientation=getattr(agent, "orientation", 0.0),
        )

        tensor = obs.tensor().cpu().numpy()
        return tensor

    def _process_action(self, agent_id: str, action: Optional[int]) -> None:
        """Process an action for a specific agent.

        Executes the specified action for the given agent by calling the
        appropriate action function. Actions are dynamically remapped from
        the enabled action space to their corresponding ActionType values.

        Parameters
        ----------
        agent_id : str
            The ID of the agent performing the action.
        action : int
            The action index within the enabled action space (0 to len(enabled_actions)-1).
            This gets remapped to the corresponding ActionType for execution.

        Notes
        -----
        This method:
        - Validates that the agent exists and is alive
        - Dynamically remaps action indices to enabled ActionType values
        - Maps actions to their implementation functions
        - Executes the action with the agent as parameter
        - Logs warnings for invalid actions
        - Returns silently if agent is dead or doesn't exist

        Action implementations are imported from the farm.actions module and
        handle the specific logic for each action type including validation,
        effects, and side effects like resource transfer or combat.

        PettingZoo Compliance
        ---------------------
        Database logging of actions (when enabled) does not alter the AEC step
        semantics. Reward is calculated later in `step`, after action execution.
        """
        agent = self._agent_objects.get(agent_id)
        if agent is None or not agent.alive or action is None:
            return

        # Validate action is within enabled action space bounds
        if action < 0 or action >= len(self._enabled_action_types):
            logging.debug(
                "Action %s is out of bounds for enabled action space (size: %s)",
                action,
                len(self._enabled_action_types),
            )
            return

        # Dynamically remap action index to corresponding ActionType
        action_type = self._enabled_action_types[action]

        # Get action name from dynamic mapping
        action_name = self._action_mapping.get(action_type)
        if action_name:
            action_obj = action_registry.get(action_name)
            if action_obj:
                # Capture resource level before action
                resources_before = agent.resource_level if agent else None

                # Execute the action
                action_result = action_obj.execute(agent)

                # Log action to database if available
                if self.db and agent:
                    try:
                        self.db.logger.log_agent_action(
                            step_number=self.time,
                            agent_id=agent_id,
                            action_type=action_name,
                            resources_before=resources_before,
                            resources_after=agent.resource_level,
                            reward=0,  # Reward will be calculated later
                            details=action_result.get("details", {}) if isinstance(action_result, dict) else {}
                        )
                    except Exception as e:
                        logging.warning(f"Failed to log agent action {action_name}: {e}")
            else:
                logging.warning("Action '%s' not found in action registry", action_name)
        else:
            logging.debug(
                "Action %s (mapped to %s) not available in current simulation configuration",
                action,
                action_type,
            )

    def _calculate_reward(
        self, agent_id: str, pre_action_state: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate the reward for a specific agent.

        Computes a reward signal for reinforcement learning based on the agent's
        state changes and current status. Uses delta-based rewards when pre-action
        state is available (better for RL), otherwise falls back to state-based rewards.

        Parameters
        ----------
        agent_id : str
            The ID of the agent to calculate reward for.
        pre_action_state : dict, optional
            Agent's state before the action was processed. If provided, uses
            delta-based rewards; otherwise uses state-based rewards.

        Returns
        -------
        float
            The calculated reward value. Returns -10.0 if agent is dead or missing.
            Uses delta rewards when pre_action_state is available, otherwise state-based.

        Notes
        -----
        Delta-based rewards (when pre_action_state provided):
        - Resource delta: direct resource change from action
        - Health delta: health change (scaled by 0.5)
        - Survival bonus: +0.1 for staying alive, -10 penalty for death
        - Better for reinforcement learning as it directly measures action impact

        State-based rewards (fallback when no pre_action_state):
        - Resource reward: 0.1 * resource_level
        - Survival reward: 0.1 for being alive
        - Health reward: current_health / starting_health ratio
        - Used for initial state or when delta calculation unavailable
        """
        agent = self._agent_objects.get(agent_id)
        if agent is None:
            return -10.0

        # Use delta-based rewards if pre-action state is available
        if pre_action_state is not None:
            resource_delta = agent.resource_level - pre_action_state["resource_level"]
            health_delta = agent.current_health - pre_action_state["health"]
            was_alive = pre_action_state["alive"]

            # Base delta rewards
            reward = resource_delta + health_delta * 0.5

            # Survival handling
            if agent.alive:
                reward += 0.1  # Survival bonus for staying alive
            else:
                reward -= 10.0  # Death penalty
                return reward  # Early return for dead agents

            # TODO: Add action-specific bonuses here when action tracking is implemented
            # For example:
            # - Combat success bonus if agent.last_action_success == ActionType.ATTACK
            # - Cooperation bonus if agent.last_action_success == ActionType.SHARE

            return reward

        # Fallback to state-based rewards when no pre-action state
        if not agent.alive:
            return -10.0

        resource_reward = agent.resource_level * 0.1
        survival_reward = 0.1
        health_reward = agent.current_health / agent.starting_health

        reward = resource_reward + survival_reward + health_reward

        # TODO: Add state-based action bonuses here when action tracking is implemented
        # This would require tracking last action and success status on the agent

        return reward

    def _next_agent(self) -> None:
        """Select the next agent to act in the environment.

        Implements the Agent-Environment-Cycle (AEC) pattern by selecting the
        next agent in round-robin order. Skips agents that are terminated or
        truncated, ensuring only active agents are selected for actions.

        Notes
        -----
        This method:
        - Finds the current agent's position in the agent list
        - Cycles through agents in order starting from the next position
        - Skips agents marked as terminated or truncated
        - Sets agent_selection to None if no active agents remain
        - Detects when a complete cycle of all agents has occurred

        The round-robin scheduling ensures fair time allocation among all
        active agents. Dead or removed agents are automatically skipped.
        """
        if not self.agents:
            self.agent_selection = None
            return

        # Store previous agent for cycle detection
        previous_agent = self.agent_selection

        try:
            current_idx = self.agents.index(self.agent_selection)
        except ValueError:
            current_idx = -1

        for i in range(1, len(self.agents) + 1):
            next_idx = (current_idx + i) % len(self.agents)
            next_agent = self.agents[next_idx]
            if not (
                self.terminations.get(next_agent, False)
                or self.truncations.get(next_agent, False)
            ):
                self.agent_selection = next_agent

                # Detect if we've completed a full cycle
                # This happens when we wrap around to the first agent
                if previous_agent is not None and next_idx < current_idx:
                    self._cycle_complete = True
                else:
                    self._cycle_complete = False

                return
        self.agent_selection = None

    def _update_agent_state(
        self,
        agent_id: str,
        agent: Optional[Any],
        observation: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> Tuple[np.ndarray, float, bool, bool]:
        """Update the agent's state in the PettingZoo environment.

        Handles updating all internal state dictionaries required for the AECEnv API,
        including observations, terminations, truncations, infos, and cumulative rewards.
        Also advances to the next agent in the cycle.

        Parameters
        ----------
        agent_id : str
            The ID of the agent to update
        agent : Optional[Any]
            The agent object (can be None for dead agents)
        observation : np.ndarray
            The observation for the current agent
        reward : float
            The reward received by the agent
        terminated : bool
            Whether the episode has terminated
        truncated : bool
            Whether the episode was truncated

        Returns
        -------
        Tuple[np.ndarray, float, bool, bool]
            Updated observation, reward, terminated, and truncated values

        PettingZoo Compliance
        ---------------------
        - `self.rewards[agent_id]` stores cumulative rewards as required by AECEnv.
        - `self._cumulative_rewards[agent_id]` is the internal running sum used
          to keep PettingZoo-compatible behavior across steps.
        """
        # Handle case where agent_id is None (no active agents)
        if agent_id is None:
            return observation, reward, terminated, truncated

        # Update internal PettingZoo state dictionaries (required for AECEnv API)
        self.observations[agent_id] = observation
        self.terminations[agent_id] = terminated
        self.truncations[agent_id] = truncated
        self.infos[agent_id] = {}

        # Handle cumulative rewards - PettingZoo expects cumulative rewards in rewards dict
        self._cumulative_rewards[agent_id] += reward
        # Update rewards dict with cumulative reward for PettingZoo compatibility
        self.rewards[agent_id] = self._cumulative_rewards[agent_id]

        # Advance to next agent in the cycle (required for AECEnv API)
        self._next_agent()

        return observation, reward, terminated, truncated

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # TODO: Reduce code duplication.
        """Reset the environment to its initial state.

        Parameters
        ----------
        seed : int, optional
            Random seed to set for deterministic behavior. If None, uses existing seed.
        options : dict, optional
            Additional options for reset (currently unused).

        Returns
        -------
        tuple
            A 2-tuple containing:
            - observation (np.ndarray): The initial observation for the first agent
            - info (dict): Additional information about the reset

        Notes
        -----
        - Rebuilds PettingZoo bookkeeping dictionaries and sets `agent_selection`
          to the first alive agent ID (if any).
        - If `options['agents']` is provided, the current population is replaced
          with those agents prior to rebuilding the PettingZoo state.
        """
        if seed is not None:
            self.seed_value = seed
            random.seed(seed)
            np.random.seed(seed)
            try:
                torch.manual_seed(seed)
            except RuntimeError:
                pass

        self.time = 0
        # Reset metrics tracker
        self.metrics_tracker.reset()

        self.resources = []
        self.initialize_resources(self.resource_distribution)

        # Optionally replace agents if provided via options
        if options and isinstance(options, dict) and options.get("agents") is not None:
            # Clear existing agents and re-add provided ones
            self._agent_objects = {}
            self.agents = []
            self.agent_observations = {}
            for agent in options.get("agents", []):
                self.add_agent(agent)
        else:
            # Preserve existing agents and refresh observations
            self.agent_observations = {}
            for agent in self._agent_objects.values():
                self.agent_observations[agent.agent_id] = AgentObservation(
                    self.observation_config
                )

        self.spatial_index.set_references(
            list(self._agent_objects.values()), self.resources
        )
        self.spatial_index.update()

        # Reset cycle tracking for proper timestep semantics
        self._agents_acted_this_cycle = 0
        self._cycle_complete = False

        # Rebuild PettingZoo agent lists from current alive agents
        self.agents = [a.agent_id for a in self._agent_objects.values() if a.alive]
        self.agent_selection = self.agents[0] if self.agents else None
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.observations = {a: self._get_observation(a) for a in self.agents}

        if self.agent_selection is None:
            dummy_obs = np.zeros(
                self._observation_space.shape, dtype=self._observation_space.dtype
            )
            return dummy_obs, {}

        return self.observations[self.agent_selection], self.infos[self.agent_selection]

    def step(
        self, action: Optional[int] = None
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment for the currently selected agent.

        Parameters
        ----------
        action : int, optional
            The action to take. Must be one of the valid actions defined in Action enum.
            If None, no action is taken. #! Change to ActionIntent

        Returns
        -------
        tuple
            A 5-tuple containing:
            - observation (np.ndarray): The observation for the current agent
            - reward (float): The reward received by the agent
            - terminated (bool): Whether the episode has terminated
            - truncated (bool): Whether the episode was truncated (e.g., max steps reached)
            - info (dict): Additional information about the step

        AEC Semantics
        -------------
        - The environment's global update (regeneration, metrics, time advance) occurs
          once per full cycle, when `_cycle_complete` is detected.
        - `terminated` may be triggered when all agents are gone or resources reach zero
          (after a cycle completes); `truncated` is based on `max_steps`.
        """
        agent_id = self.agent_selection
        agent = self._agent_objects.get(agent_id)

        # Capture pre-action state for delta calculations
        #! This will likely depend on the reward system
        pre_action_state = None
        if agent:
            pre_action_state = {
                "resource_level": agent.resource_level,
                "health": agent.current_health,
                "alive": agent.alive,
            }

        # Process the action
        self._process_action(agent_id, action)

        # Initialize terminated to False (will be updated when cycle completes)
        terminated = False

        # Check for immediate termination conditions (no alive agents)
        if len(self.agents) == 0:
            terminated = True

        # Only update environment state when all agents have acted (cycle complete)
        # This ensures proper timestep semantics in AECEnv
        if self._cycle_complete and not terminated:
            self.update()

            # Check termination conditions only after all agents have acted
            # This prevents premature termination if resources hit zero mid-cycle
            terminated = terminated or self.cached_total_resources == 0

            self._cycle_complete = False  # Reset for next cycle

        # Check truncation after potential environment update
        truncated = self.time >= self.max_steps

        # Calculate reward using consolidated method
        reward = self._calculate_reward(agent_id, pre_action_state)

        # Get observation for current agent
        observation = (
            self._get_observation(agent_id)
            if agent
            else np.zeros(
                self._observation_space.shape, dtype=self._observation_space.dtype
            )
        )

        # Update agent state and advance to next agent
        observation, reward, terminated, truncated = self._update_agent_state(
            agent_id, agent, observation, reward, terminated, truncated
        )

        return observation, reward, terminated, truncated, {}

    def get_perception_profile(self, reset: bool = False) -> Dict[str, float]:
        """Return aggregated perception profiling stats.

        Args:
            reset: If True, reset accumulators after returning.
        """
        prof = dict(self._perception_profile)
        if reset:
            for k in self._perception_profile.keys():
                self._perception_profile[k] = 0 if "points" in k else 0.0
        return prof

    def render(self, mode: str = "human") -> None:
        """Render the current state of the environment.

        Provides a simple text-based rendering of the current simulation state
        for monitoring and debugging purposes. Currently only supports human-
        readable console output with basic statistics.

        Parameters
        ----------
        mode : str, default="human"
            The rendering mode. Currently only supports "human" for console output.
            Future implementations may support graphical rendering modes.

        Notes
        -----
        Current implementation prints:
        - Current simulation time step
        - Number of active agents
        - Total cached resources in the environment

        TODO: Implement proper graphical visualization for better analysis
        and debugging capabilities.
        """
        if mode == "human":
            print(f"Time: {self.time}")
            print(f"Active agents: {len(self.agents)}")
            print(f"Total resources: {self.cached_total_resources}")
        # TODO: Implement proper visualization if needed

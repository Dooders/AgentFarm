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

import math
import random
import time as _time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import AECEnv

from farm.config import ResourceConfig, SimulationConfig

# Use action registry for cleaner action management
from farm.core.action import ActionType, action_registry
from farm.core.channels import NUM_CHANNELS
from farm.core.geometry import discretize_position_continuous
from farm.core.interfaces import DatabaseFactoryProtocol, DatabaseProtocol
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
from farm.core.spatial import SpatialIndex
from farm.core.state import EnvironmentState
from farm.utils.identity import Identity, IdentityConfig
from farm.utils.logging import get_logger
from farm.utils.spatial import bilinear_distribute_value

logger = get_logger(__name__)


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
        db (DatabaseProtocol): Optional database implementing DatabaseProtocol

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
        db_factory: Optional[DatabaseFactoryProtocol] = None,
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
        db_factory : DatabaseFactoryProtocol, optional
            Factory for creating database instances. If None, uses default
            database setup.

        Raises
        ------
        ValueError
            If width or height are non-positive
        Exception
            If database setup fails or resource initialization fails
        """
        super().__init__()
        # Set seed if provided
        self.seed_value = seed if seed is not None else config.seed if config and config.seed else None
        if self.seed_value is not None:
            random.seed(self.seed_value)
            np.random.seed(self.seed_value)
            try:
                torch.manual_seed(self.seed_value)
            except RuntimeError as e:
                logger.warning("Failed to seed torch with value %s: %s", self.seed_value, e)

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
        if db_factory is not None:
            self.db = db_factory.setup_db(db_path, self.simulation_id, config.to_dict() if config else None)
        else:
            # Import setup_db only when needed to avoid circular imports
            from farm.database.utilities import setup_db

            self.db = setup_db(db_path, self.simulation_id, config.to_dict() if config else None)

        # Use self.identity for all ID needs
        self.max_resource = max_resource
        self.config = config

        # Warn if config is None as it may cause issues in some scenarios
        if config is None:
            logger.warning(
                "environment_created_without_config",
                message="Environment created with config=None. Some features may not work as expected.",
                simulation_id=self.simulation_id,
            )

        self.resource_distribution = resource_distribution
        self.max_steps = config.max_steps if config and hasattr(config, "max_steps") else 1000

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

        # Initialize spatial index attributes
        self._quadtree_enabled = False
        self._spatial_hash_enabled = False

        # Initialize spatial index for efficient spatial queries with batch updates
        from farm.utils.config_utils import resolve_spatial_index_config

        spatial_config = resolve_spatial_index_config(config)

        if spatial_config:
            self.spatial_index = SpatialIndex(
                self.width,
                self.height,
                enable_batch_updates=spatial_config.enable_batch_updates,
                region_size=spatial_config.region_size,
                max_batch_size=spatial_config.max_batch_size,
                dirty_region_batch_size=getattr(spatial_config, "dirty_region_batch_size", 10),
            )

            # Enable additional index types if configured
            if spatial_config.enable_quadtree_indices:
                self.enable_quadtree_indices()
            if spatial_config.enable_spatial_hash_indices:
                self.enable_spatial_hash_indices(spatial_config.spatial_hash_cell_size)
        else:
            # Default configuration with batch updates enabled
            self.spatial_index = SpatialIndex(
                self.width,
                self.height,
                enable_batch_updates=True,
                region_size=50.0,
                max_batch_size=100,
                dirty_region_batch_size=10,
            )

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
            simulation_id=self.simulation_id,
            identity_service=self.identity,
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
        self.spatial_index.set_references(list(self._agent_objects.values()), self.resources)
        self.spatial_index.update()

        # Quadtree and spatial hash indices are already initialized above

        # Perception profiler accumulators
        self._perception_profile = {
            "spatial_query_time_s": 0.0,
            "bilinear_time_s": 0.0,
            "nearest_time_s": 0.0,
            "bilinear_points": 0,
            "nearest_points": 0,
        }

        # Population milestone tracking
        self._logged_population_milestones = set()

        # Resource depletion warning tracking
        self._initial_total_resources = None
        self._warned_10_percent = False
        self._warned_25_percent = False

        # Log environment initialization completion
        logger.info(
            "environment_initialized",
            simulation_id=self.simulation_id,
            dimensions=(self.width, self.height),
            initial_agents=len(self.agents),
            initial_resources=len(self.resources),
            total_resource_amount=sum(r.amount for r in self.resources),
            seed=self.seed_value,
            max_steps=self.max_steps,
            database_path=self.db.db_path if self.db else None,
            observation_channels=getattr(self, "NUM_CHANNELS", None),
            action_count=(len(self._action_mapping) if hasattr(self, "_action_mapping") else None),
        )

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
            logger.warning("missing_actions_in_registry", missing_actions=missing_actions)
            # Remove missing actions from mapping
            self._action_mapping = {k: v for k, v in self._action_mapping.items() if v not in missing_actions}

        # Log the final action mapping
        available_actions = list(self._action_mapping.values())
        logger.info(
            "action_mapping_initialized",
            action_count=len(available_actions),
            available_actions=available_actions,
        )

    @property
    def agent_objects(self) -> List[Any]:
        """Backward compatibility property to get all agent objects as a list."""
        return list(self._agent_objects.values())

    def mark_positions_dirty(self) -> None:
        """Public method for agents to mark positions as dirty when they move."""
        self.spatial_index.mark_positions_dirty()

    def process_batch_spatial_updates(self, force: bool = False) -> None:
        """
        Process any pending batch spatial updates.

        Parameters
        ----------
        force : bool
            Force processing even if batch is not full
        """
        if hasattr(self.spatial_index, "process_batch_updates"):
            self.spatial_index.process_batch_updates(force=force)

    def get_spatial_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for spatial indexing and batch updates."""
        stats = {}

        # Get basic spatial index stats
        if hasattr(self.spatial_index, "get_stats"):
            stats.update(self.spatial_index.get_stats())

        # Get batch update stats if available
        if hasattr(self.spatial_index, "get_batch_update_stats"):
            batch_stats = self.spatial_index.get_batch_update_stats()
            stats["batch_updates"] = batch_stats

        # Get perception profile stats
        if hasattr(self, "get_perception_profile"):
            perception_stats = self.get_perception_profile()
            stats["perception"] = perception_stats

        return stats

    def enable_batch_spatial_updates(self, region_size: float = 50.0, max_batch_size: int = 100) -> None:
        """Enable batch spatial updates with the specified configuration."""
        if hasattr(self.spatial_index, "enable_batch_updates"):
            self.spatial_index.enable_batch_updates(region_size, max_batch_size)

    def disable_batch_spatial_updates(self) -> None:
        """Disable batch spatial updates and process any pending updates."""
        if hasattr(self.spatial_index, "disable_batch_updates"):
            self.spatial_index.disable_batch_updates()

    def get_nearby_agents(self, position: Tuple[float, float], radius: float) -> List[Any]:
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

    def get_nearby_resources(self, position: Tuple[float, float], radius: float) -> List[Any]:
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

    def enable_quadtree_indices(self) -> None:
        """Enable Quadtree indices alongside existing KD-tree indices.

        This creates additional Quadtree-based spatial indices for performance
        comparison and hierarchical spatial operations. The Quadtrees will be
        available alongside the existing KD-tree indices.
        """
        if self._quadtree_enabled:
            return  # Already enabled

        # Register Quadtree versions of the default indices
        self.spatial_index.register_index(
            name="agents_quadtree",
            data_getter=lambda: list(self._agent_objects.values()),
            position_getter=lambda a: a.position,
            filter_func=lambda a: getattr(a, "alive", True),
            index_type="quadtree",
        )

        self.spatial_index.register_index(
            name="resources_quadtree",
            data_reference=self.resources,
            position_getter=lambda r: r.position,
            filter_func=None,
            index_type="quadtree",
        )

        self._quadtree_enabled = True
        logger.info("Quadtree indices enabled for spatial queries")

    def enable_spatial_hash_indices(self, cell_size: Optional[float] = None) -> None:
        """Enable Spatial Hash Grid indices alongside existing KD-tree indices.

        Registers spatial-hash-based indices for agents and resources. Spatial hash
        indices provide near-constant-time neighborhood queries by inspecting only
        nearby buckets and support efficient dynamic updates.

        Parameters
        ----------
        cell_size : float, optional
            Size of each grid cell. If None, a heuristic based on environment
            size is used to choose a reasonable default.
        """
        if self._spatial_hash_enabled:
            return

        self.spatial_index.register_index(
            name="agents_hash",
            data_getter=lambda: list(self._agent_objects.values()),
            position_getter=lambda a: a.position,
            filter_func=lambda a: getattr(a, "alive", True),
            index_type="spatial_hash",
            cell_size=cell_size,
        )

        self.spatial_index.register_index(
            name="resources_hash",
            data_reference=self.resources,
            position_getter=lambda r: r.position,
            filter_func=None,
            index_type="spatial_hash",
            cell_size=cell_size,
        )

        self._spatial_hash_enabled = True
        logger.info("Spatial hash indices enabled for spatial queries (cell_size=%s)", cell_size)

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

    def initialize_resources(self, distribution: Union[Dict[str, Any], Callable]) -> None:
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
        - Cleans up all PettingZoo state dictionaries
        - Marks spatial index as dirty for next update
        - Cleans up agent observation data
        """
        # Log agent removal before processing
        logger.info(
            "agent_removed",
            agent_id=agent.agent_id,
            agent_type=agent.__class__.__name__,
            cause=getattr(agent, "death_cause", "unknown"),
            lifespan=self.time - getattr(agent, "birth_time", 0),
            final_resources=getattr(agent, "resource_level", 0),
            final_health=getattr(agent, "current_health", 0),
            generation=getattr(agent, "generation", 0),
            step=self.time,
            remaining_agents=len(self.agents) - 1,
        )

        agent_id = agent.agent_id
        death_time = self.time
        death_cause = getattr(agent, "death_cause", "unknown")
        
        self.record_death()
        
        # Update agent death in database
        if hasattr(self, "db") and self.db is not None:
            try:
                self.db.update_agent_death(agent_id=agent_id, death_time=death_time, cause=death_cause)
            except Exception as e:
                logger.warning(
                    "Failed to update agent death in database",
                    agent_id=agent_id,
                    death_time=death_time,
                    error=str(e),
                )
        
        if agent_id in self._agent_objects:
            del self._agent_objects[agent_id]
        if agent_id in self.agents:
            self.agents.remove(agent_id)  # Remove from PettingZoo agents list

        # Check for population milestones after removal
        current_population = len(self.agents)
        milestones = [1, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000]

        # Find the closest milestone (checking for decline)
        for milestone in milestones:
            if milestone not in self._logged_population_milestones:
                if current_population < milestone:
                    # Log this milestone
                    agent_type_counts = {}
                    for agent in self._agent_objects.values():
                        agent_type = agent.__class__.__name__
                        agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

                    logger.info(
                        "population_milestone_reached",
                        milestone=milestone,
                        current_population=current_population,
                        step=self.time,
                        agent_types=agent_type_counts,
                        direction="decline",
                    )

                    self._logged_population_milestones.add(milestone)
                    break

        # Clean up PettingZoo state dictionaries to prevent stale references
        if agent_id in self._cumulative_rewards:
            del self._cumulative_rewards[agent_id]
        if agent_id in self.rewards:
            del self.rewards[agent_id]
        if agent_id in self.terminations:
            del self.terminations[agent_id]
        if agent_id in self.truncations:
            del self.truncations[agent_id]
        if agent_id in self.infos:
            del self.infos[agent_id]
        if agent_id in self.observations:
            del self.observations[agent_id]

        # Update agent_selection if necessary
        if self.agent_selection == agent_id or not self.agents:
            self._next_agent()

        self.spatial_index.mark_positions_dirty()  # Mark positions as dirty when agent is removed
        if agent_id in self.agent_observations:
            del self.agent_observations[agent_id]

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
            logger.error(
                "interaction_logging_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )

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
            logger.error(
                "reproduction_logging_failed",
                error_type=type(e).__name__,
                error_message=str(e),
            )

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

            # Check resource levels periodically
            if self.time % 50 == 0:  # Check every 50 steps
                if self._initial_total_resources is None:
                    self._initial_total_resources = self.cached_total_resources

                if self._initial_total_resources > 0:
                    resource_ratio = self.cached_total_resources / self._initial_total_resources
                    active_resources = len([r for r in self.resources if r.amount > 0])

                    # Warn at different thresholds
                    if resource_ratio < 0.1 and not self._warned_10_percent:
                        logger.debug(
                            "resources_critically_low",
                            remaining_ratio=round(resource_ratio, 3),
                            remaining_total=round(self.cached_total_resources, 2),
                            active_nodes=active_resources,
                            total_nodes=len(self.resources),
                            step=self.time,
                            agents=len(self.agents),
                        )
                        self._warned_10_percent = True

                    elif resource_ratio < 0.25 and not self._warned_25_percent:
                        logger.debug(
                            "resources_running_low",
                            remaining_ratio=round(resource_ratio, 3),
                            remaining_total=round(self.cached_total_resources, 2),
                            active_nodes=active_resources,
                            step=self.time,
                        )
                        self._warned_25_percent = True

            # Calculate and log metrics
            metrics = self._calculate_metrics()
            self.metrics_tracker.update_metrics(
                metrics,
                db=self.db,
                time=self.time,
                agent_objects=self._agent_objects,
                resources=self.resources,
            )

            # Update spatial performance metrics
            spatial_stats = self.get_spatial_performance_stats()
            self.metrics_tracker.update_spatial_performance_metrics(spatial_stats)

            # Update spatial index (this will process any pending batch updates)
            self.spatial_index.update()

            # Process any remaining batch updates to ensure all position changes are applied
            self.process_batch_spatial_updates(force=True)

            # Reset counters for next step
            self.resources_shared_this_step = 0
            self.combat_encounters_this_step = 0
            self.successful_attacks_this_step = 0

            # Log milestone every 100 steps
            if self.time % 100 == 0 and self.time > 0:
                # Calculate agent statistics
                agents_alive = len(self.agents)
                # Get health from combat component to ensure proper capping
                health_values = []
                for a in self._agent_objects.values():
                    combat_comp = a.get_component("combat")
                    if combat_comp:
                        health_values.append(combat_comp.health)
                    else:
                        health_values.append(0.0)
                avg_health = np.mean(health_values) if health_values else 0
                avg_resources = (
                    np.mean([a.resource_level for a in self._agent_objects.values()]) if self._agent_objects else 0
                )

                # Agent type distribution
                agent_type_counts = {}
                for agent in self._agent_objects.values():
                    agent_type = agent.__class__.__name__
                    agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

                logger.info(
                    "simulation_milestone",
                    step=self.time,
                    agents_alive=agents_alive,
                    total_resources=self.cached_total_resources,
                    resource_nodes=len([r for r in self.resources if r.amount > 0]),
                    avg_agent_health=round(avg_health, 2),
                    avg_agent_resources=round(avg_resources, 2),
                    agent_types=agent_type_counts,
                    combat_encounters=getattr(self, "combat_encounters_this_step", 0),
                    resources_shared=getattr(self, "resources_shared_this_step", 0),
                )

            # Increment time step
            self.time += 1

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("environment_update_error", error=str(e), exc_info=True)
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
        return self.metrics_tracker.calculate_metrics(self._agent_objects, self.resources, self.time, self.config)

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
        if hasattr(self, "resource_manager") and self.resource_manager is not None:
            # Ensure memmap file is flushed; delete based on config (default: keep for reuse)
            delete_memmap = getattr(self.config, "resources", ResourceConfig()).memmap_delete_on_close
            try:
                self.resource_manager.cleanup_memmap(delete_file=delete_memmap)
            except Exception as e:
                logger.error(
                    "memmap_cleanup_failed",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
        if hasattr(self, "db") and self.db is not None:
            self.db.close()

    def add_agent(self, agent: Any, flush_immediately: bool = False) -> None:
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
        flush_immediately : bool, optional
            If True, immediately flush the agent buffer to the database to ensure
            the agent is committed before any actions are processed. This is useful
            for agents created during simulation (e.g., through reproduction) to
            prevent foreign key constraint violations. Default is False.

        Notes
        -----
        This method:
        - Extracts agent data for database logging
        - Adds agent to internal object mapping and PettingZoo agent list
        - Marks spatial index as dirty for next update
        - Batch logs agent data to database if available
        - Creates observation tracking for the agent
        - Optionally flushes agent buffer immediately if flush_immediately=True

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
            if hasattr(agent, "validation_service") and agent.validation_service is None:
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
                raise ValueError("Agent %s missing spatial_service after injection" % getattr(agent, "agent_id", "?"))
        except (AttributeError, ValueError, TypeError) as e:
            logger.error(
                "service_injection_failed",
                agent_id=getattr(agent, "agent_id", "unknown"),
                error_type=type(e).__name__,
                error_message=str(e),
            )
            raise

        agent_data = [
            {
                "simulation_id": self.simulation_id,
                "agent_id": agent.agent_id,
                "birth_time": self.time,
                "agent_type": getattr(agent, "agent_type", agent.__class__.__name__),
                "position": agent.position,
                "initial_resources": agent.resource_level,
                "starting_health": self._get_agent_starting_health(agent),
                "starvation_counter": self._get_agent_starvation_counter(agent),
                "genome_id": getattr(agent, "genome_id", None),
                "generation": getattr(agent, "generation", 0),
                "action_weights": self._get_agent_action_weights(agent),
            }
        ]

        # Add to environment
        self._agent_objects[agent.agent_id] = agent
        self.agents.append(agent.agent_id)  # Add to PettingZoo agents list

        # Initialize PettingZoo state dictionaries for the new agent
        self.rewards[agent.agent_id] = 0
        self._cumulative_rewards[agent.agent_id] = 0
        self.terminations[agent.agent_id] = False
        self.truncations[agent.agent_id] = False
        self.infos[agent.agent_id] = {}
        # Initialize observation with zeros since agent_observations isn't set up yet
        self.observations[agent.agent_id] = np.zeros(self._observation_space.shape, dtype=self._observation_space.dtype)

        # Mark positions as dirty when new agent is added
        self.spatial_index.mark_positions_dirty()

        # Update spatial index references to include the new agent
        self.spatial_index.set_references(list(self._agent_objects.values()), self.resources)

        # Batch log to database using SQLAlchemy
        if self.db is not None:
            self.db.logger.log_agents_batch(agent_data)

            # Optionally flush immediately to ensure agent is committed before actions
            if flush_immediately:
                self.db.logger.flush_all_buffers()

        self.agent_observations[agent.agent_id] = AgentObservation(self.observation_config)

        # Record birth in metrics tracker (only for agents created during simulation, not initial population)
        if self.time > 0:  # Only record births for agents created after simulation starts
            self.record_birth()

        # Log agent addition
        logger.info(
            "agent_added",
            agent_id=agent.agent_id,
            agent_type=agent.__class__.__name__,
            position=agent.position,
            initial_resources=agent.resource_level,
            # Avoid redundant lookups for combat component
            initial_health=(agent.get_component("combat").health if agent.get_component("combat") else 0.0),
            generation=getattr(agent, "generation", 0),
            genome_id=getattr(agent, "genome_id", None),
            step=self.time,
            total_agents=len(self.agents),
        )

        # Check for population milestones
        current_population = len(self.agents)
        milestones = [1, 10, 25, 50, 100, 250, 500, 1000, 5000, 10000]

        # Find the closest milestone
        for milestone in milestones:
            # Check if we just crossed this milestone (either direction)
            if milestone not in self._logged_population_milestones:
                if current_population >= milestone:
                    # Log this milestone
                    agent_type_counts = {}
                    for agent in self._agent_objects.values():
                        agent_type = agent.__class__.__name__
                        agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

                    logger.info(
                        "population_milestone_reached",
                        milestone=milestone,
                        current_population=current_population,
                        step=self.time,
                        agent_types=agent_type_counts,
                        direction="growth",
                    )

                    self._logged_population_milestones.add(milestone)
                    break

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
            logger.error(
                "environment_cleanup_error",
                error_type=type(e).__name__,
                error_message=str(e),
            )

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
        self._observation_space = spaces.Box(low=0.0, high=1.0, shape=(NUM_CHANNELS, S, S), dtype=np_dtype)

    def _setup_action_space(self) -> None:
        """Setup the action space with all available actions."""
        # Use only the enabled actions from the mapping instead of full ActionType enum
        self._action_space = spaces.Discrete(len(self._action_mapping))

        # Create a list of enabled ActionType values for dynamic remapping
        # This ensures consistent ordering: action index 0 maps to first enabled action, etc.
        self._enabled_action_types = list(self._action_mapping.keys())

    def update_action_space(self, new_enabled_actions: Optional[List[str]] = None) -> None:
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
        logger.info(
            "action_space_updated",
            action_count=len(available_actions),
            available_actions=available_actions,
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
        return len([agent for agent in self._agent_objects.values() if getattr(agent, "birth_time", 0) == 0])

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
            return np.zeros(self._observation_space.shape, dtype=self._observation_space.dtype)

        # Assume width and height are integers for grid
        height, width = int(self.height), int(self.width)

        # Get discretization method from config
        # Resolve discretization/interpolation from nested environment config when available
        if self.config and getattr(self.config, "environment", None) is not None:
            discretization_method = getattr(self.config.environment, "position_discretization_method", "floor")
            use_bilinear = getattr(self.config.environment, "use_bilinear_interpolation", True)
        else:
            discretization_method = (
                getattr(self.config, "position_discretization_method", "floor") if self.config else "floor"
            )
            use_bilinear = getattr(self.config, "use_bilinear_interpolation", True) if self.config else True

        # Agent position as (y, x) using configured discretization method
        grid_size = (width, height)
        ax, ay = discretize_position_continuous(agent.position, grid_size, discretization_method)

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
        if self.max_resource is not None:
            max_amount = self.max_resource
        else:
            from farm.utils.config_utils import get_nested_then_flat

            max_amount = get_nested_then_flat(
                config=self.config,
                nested_parent_attr="resources",
                nested_attr_name="max_resource_amount",
                flat_attr_name="max_resource_amount",
                default_value=10,
                expected_types=(int, float),
            )

        # Query nearby resources within a radius covering the local window
        # Use slightly larger than R to capture bilinear spread near the boundary
        # If ResourceManager has a memmap grid, slice directly for the window; else use spatial queries
        nearby_resources = []
        used_memmap = False
        try:
            if (
                hasattr(self, "resource_manager")
                and getattr(self.resource_manager, "has_memmap", False)
                and self.resource_manager.has_memmap
            ):
                used_memmap = True
                # Compute world-space window bounds (y,x) centered at (ay, ax)
                y0 = ay - R
                y1 = ay + R + 1
                x0 = ax - R
                x1 = ax + R + 1
                window_np = self.resource_manager.get_resource_window(y0, y1, x0, x1, normalize=True)
                # Convert to torch tensor of correct dtype/device with minimal copies
                if (
                    self.observation_config.device == "cpu"
                    and self.observation_config.torch_dtype == torch.float32
                    and window_np.dtype == np.float32
                ):
                    resource_local = torch.from_numpy(window_np)
                else:
                    resource_local = torch.tensor(
                        window_np,
                        dtype=self.observation_config.torch_dtype,
                        device=self.observation_config.device,
                        copy=False,
                    )
            else:
                _tq0 = _time.perf_counter()
                nearby = self.spatial_index.get_nearby(agent.position, R + 1, ["resources"])
                nearby_resources = nearby.get("resources", [])
                _tq1 = _time.perf_counter()
                self._perception_profile["spatial_query_time_s"] += max(0.0, _tq1 - _tq0)
        except AttributeError as e:
            logger.warning(
                "spatial_resource_init_issue",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            nearby_resources = []
        except (ValueError, TypeError) as e:
            logger.warning(
                "invalid_observation_parameters",
                error_type=type(e).__name__,
                error_message=str(e),
            )
            nearby_resources = []
        except Exception as e:
            logger.error(
                "resource_layer_build_error",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
            nearby_resources = []

        if not used_memmap and use_bilinear:
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
        elif not used_memmap:
            _tn0 = _time.perf_counter()
            for res in nearby_resources:
                rx, ry = discretize_position_continuous(res.position, (width, height), discretization_method)
                lx = rx - (ax - R)
                ly = ry - (ay - R)
                if 0 <= lx < S and 0 <= ly < S:
                    resource_local[int(ly), int(lx)] += float(res.amount) / float(max_amount)
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

        # Get health from combat component to ensure proper capping
        combat_comp = agent.get_component("combat")
        current_health = combat_comp.health if combat_comp else 0.0
        starting_health = combat_comp.config.starting_health if combat_comp else 100.0
        self_hp01 = current_health / starting_health

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
            logger.debug(
                "action_out_of_bounds",
                action=action,
                action_space_size=len(self._enabled_action_types),
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

                # Log the action to the database
                # Note: This is necessary because _process_action directly calls action_obj.execute()
                # which bypasses agent._execute_action() where logging would normally occur
                if (
                    self.db
                    and hasattr(self.db, "logger")
                    and self.db.logger
                    and action_result
                    and isinstance(action_result, dict)
                ):
                    try:
                        # Extract target_id from action result if available
                        action_target_id = None
                        if "details" in action_result:
                            details = action_result["details"]
                            if isinstance(details, dict):
                                # For gather actions, use resource_id as target_id
                                if action_name == "gather" and "resource_id" in details:
                                    action_target_id = details["resource_id"]
                                # For other actions, use target_id if available
                                elif "target_id" in details:
                                    action_target_id = details["target_id"]

                        # Log the agent action
                        self.db.logger.log_agent_action(
                            step_number=self.time,
                            agent_id=agent_id,
                            action_type=action_name,
                            action_target_id=action_target_id,
                            reward=None,  # Reward is not available at this point; it will be calculated elsewhere in the learning system. None indicates the value is not yet available.
                            details=action_result.get("details", {}),
                        )
                    except Exception as e:
                        # Log warning but don't crash on database logging failure
                        logger.warning(f"Failed to log agent action {action_name} for agent {agent_id}: {e}")
            else:
                logger.warning("action_not_found_in_action_registry", action_name=action_name)
        else:
            logger.debug(
                "Action %s (mapped to %s) not available in current simulation configuration",
                action,
                action_type,
            )

    def _get_agent_reward(self, agent_id: str, pre_action_state: Optional[Dict[str, Any]] = None) -> float:
        """
        Get reward from agent's reward system.

        Args:
            agent_id: ID of the agent to get reward for
            pre_action_state: Unused, kept for backward compatibility

        Returns:
            Calculated reward value from agent's reward component
        """
        agent = self._agent_objects.get(agent_id)
        if agent is None:
            return -10.0

        # Use agent's public reward API with safe attribute access
        return getattr(agent, "step_reward", 0.0)

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
            if not (self.terminations.get(next_agent, False) or self.truncations.get(next_agent, False)):
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
                self.agent_observations[agent.agent_id] = AgentObservation(self.observation_config)

        self.spatial_index.set_references(list(self._agent_objects.values()), self.resources)
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
            dummy_obs = np.zeros(self._observation_space.shape, dtype=self._observation_space.dtype)
            return dummy_obs, {}

        return self.observations[self.agent_selection], self.infos[self.agent_selection]

    def step(self, action: Optional[int] = None) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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

        # Calculate reward using agent's reward component if available, otherwise fallback
        reward = self._get_agent_reward(agent_id)

        # Get observation for current agent
        observation = (
            self._get_observation(agent_id)
            if agent
            else np.zeros(self._observation_space.shape, dtype=self._observation_space.dtype)
        )

        # Update agent state and advance to next agent
        observation, reward, terminated, truncated = self._update_agent_state(
            agent_id, agent, observation, reward, terminated, truncated
        )

        return observation, reward, terminated, truncated, {}

    def _get_agent_starting_health(self, agent: Any) -> float:
        """Get starting health from an AgentCore component-based agent.

        Args:
            agent: AgentCore instance to get starting health from

        Returns:
            Starting health value, or 100.0 as default
        """
        try:
            combat_component = agent.get_component("combat")
            if (
                combat_component
                and hasattr(combat_component, "config")
                and hasattr(combat_component.config, "starting_health")
            ):
                return combat_component.config.starting_health
            return 100.0
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(
                "failed_to_get_starting_health",
                agent_id=getattr(agent, "agent_id", "unknown"),
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return 100.0

    def _get_agent_starvation_counter(self, agent: Any) -> int:
        """Get starvation counter from an AgentCore component-based agent.

        Args:
            agent: AgentCore instance to get starvation counter from

        Returns:
            Starvation counter value, or 0 as default
        """
        try:
            resource_component = agent.get_component("resource")
            if resource_component and hasattr(resource_component, "starvation_counter"):
                return resource_component.starvation_counter
            return 0
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(
                "failed_to_get_starvation_counter",
                agent_id=getattr(agent, "agent_id", "unknown"),
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return 0

    def _get_agent_action_weights(self, agent: Any) -> Dict[str, Any]:
        """Get action weights from an AgentCore component-based agent.

        Args:
            agent: AgentCore instance to get action weights from

        Returns:
            Dict containing action weights, or empty dict if not available
        """
        try:
            if hasattr(agent, "get_action_weights") and callable(agent.get_action_weights):
                return agent.get_action_weights()
            else:
                return {}
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug(
                "failed_to_get_action_weights",
                agent_id=getattr(agent, "agent_id", "unknown"),
                error_type=type(e).__name__,
                error_message=str(e),
            )
            return {}

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

"""Environment module for AgentFarm simulation.

This module contains the core Environment class that manages the simulation world,
including agents, resources, spatial relationships, and the simulation loop.
The Environment class extends PettingZoo's AECEnv to provide multi-agent
reinforcement learning capabilities.

Key Components:
    - Environment: Main simulation environment class
    - Action: Enumeration of available agent actions
    - Spatial indexing for efficient proximity queries
    - Resource management and regeneration
    - Metrics tracking and database logging
    - Agent lifecycle management

The environment supports various agent types (SystemAgent, IndependentAgent,
ControlAgent) and provides observation spaces for reinforcement learning
training and evaluation.
"""

import logging
import random
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import AECEnv

# TODO: Simplify actions import.
from farm.actions.attack import attack_action
from farm.actions.gather import gather_action
from farm.actions.move import move_action
from farm.actions.reproduce import reproduce_action
from farm.actions.share import share_action
from farm.agents import ControlAgent, IndependentAgent, SystemAgent
from farm.core.channels import NUM_CHANNELS
from farm.core.metrics_tracker import MetricsTracker
from farm.core.observations import AgentObservation, ObservationConfig
from farm.core.resource_manager import ResourceManager
from farm.core.spatial_index import SpatialIndex
from farm.core.state import EnvironmentState
from farm.database.utilities import setup_db
from farm.utils.short_id import ShortUUID


class Action(IntEnum):
    """Enumeration of available agent actions.

    Actions define the possible behaviors agents can take in the environment.
    Each action has specific effects on the agent and its surroundings.

    Attributes:
        DEFEND (0): Agent enters defensive stance, reducing damage from attacks
        ATTACK (1): Agent attempts to attack nearby enemies
        GATHER (2): Agent collects resources from nearby resource nodes
        SHARE (3): Agent shares resources with nearby allies
        MOVE (4): Agent moves to a new position within the environment
        REPRODUCE (5): Agent attempts to create offspring if conditions are met

    Note:
        TODO: Move to actions.py for better organization
    """

    DEFEND = 0
    ATTACK = 1
    GATHER = 2
    SHARE = 3
    MOVE = 4
    REPRODUCE = 5


logger = logging.getLogger(__name__)


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

        # Initialize basic attributes
        self.width = width
        self.height = height
        self.agents = []
        self._agent_objects = {}  # Internal mapping: agent_id -> agent object
        self.resources = []
        self.time = 0

        # Store simulation ID
        self.simulation_id = simulation_id or ShortUUID().uuid()

        # Setup database and get initialized database instance
        self.db = setup_db(db_path, self.simulation_id)

        self.id_generator = ShortUUID()
        self.next_resource_id = 0
        self.max_resource = max_resource
        self.config = config
        self.resource_distribution = resource_distribution
        self.max_steps = (
            config.max_steps if config and hasattr(config, "max_steps") else 1000
        )

        # Initialize PettingZoo required attributes
        self.agent_selection = None
        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}
        self.observations = {}

        # Initialize spatial index for efficient spatial queries
        self.spatial_index = SpatialIndex(width, height)

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker()

        # Initialize resource sharing counters
        self.resources_shared = 0.0
        self.resources_shared_this_step = 0.0

        # Initialize resource manager
        self.resource_manager = ResourceManager(
            width=width,
            height=height,
            config=config,
            seed=self.seed_value,
            database_logger=self.db.logger if self.db else None,
            spatial_index=self.spatial_index,
        )

        # Initialize environment
        self.initialize_resources(resource_distribution)
        # Set references and initialize spatial index
        self.spatial_index.set_references(
            list(self._agent_objects.values()), self.resources
        )
        self.spatial_index.update()

        # Add observation space setup:
        self._setup_observation_space(config)

        # Add action space setup call:
        self._setup_action_space()

        # Add call to create agents:
        self._create_initial_agents()

        self.agent_observations = {}
        for agent in self.agent_objects:
            self.agent_observations[agent.agent_id] = AgentObservation(
                self.observation_config
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
        return self.spatial_index.get_nearby_agents(position, radius)

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
        # Use spatial index for efficient O(log n) queries
        return self.spatial_index.get_nearby_resources(position, radius)

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
        # Use spatial index for efficient O(log n) queries
        return self.spatial_index.get_nearest_resource(position)

    def get_next_resource_id(self) -> int:
        """Generate the next unique resource ID.

        Returns a monotonically increasing integer ID for new resources
        and increments the internal counter.

        Returns
        -------
        int
            Unique resource ID
        """
        resource_id = self.next_resource_id
        self.next_resource_id += 1
        return resource_id

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

        # Update next_resource_id to match ResourceManager
        self.next_resource_id = self.resource_manager.next_resource_id

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
        source_id: Union[str, int],
        target_type: str,
        target_id: Union[str, int],
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
        source_id : str or int
            Unique identifier of the source entity
        target_type : str
            Type of the target entity (e.g., 'agent', 'resource')
        target_id : str or int
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
                source_id=str(source_id),
                target_type=target_type,
                target_id=str(target_id),
                interaction_type=interaction_type,
                action_type=action_type,
                details=details,
            )
        except Exception as e:
            logger.error(f"Failed to log interaction edge: {e}")

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

            # Log resource update statistics if needed
            if resource_stats["regeneration_events"] > 0:
                logger.debug(
                    f"Resource update: {resource_stats['regeneration_events']} resources regenerated"
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

        except Exception as e:
            logging.error(f"Error in environment update: {e}")
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
        if hasattr(self, "seed_value") and self.seed_value is not None:
            # For deterministic mode, create IDs based on a counter
            if not hasattr(self, "agent_id_counter"):
                self.agent_id_counter = 0

            # Use agent counter and seed to create a deterministic ID
            agent_seed = f"{self.seed_value}_{self.agent_id_counter}"
            # Increment counter for next agent
            self.agent_id_counter += 1

            # Use a deterministic hash function
            import hashlib

            # Create a hash and use the first 10 characters
            agent_hash = hashlib.md5(agent_seed.encode()).hexdigest()[:10]
            return f"agent_{agent_hash}"
        else:
            # Non-deterministic mode uses random short ID
            return self.id_generator.id()

    def get_state(self) -> EnvironmentState:
        """Get current environment state."""
        return EnvironmentState.from_environment(self)

    def is_valid_position(self, position: Tuple[float, float]) -> bool:
        """Check if a position is valid within the environment bounds.

        Parameters
        ----------
        position : tuple
            (x, y) coordinates to check

        Returns
        -------
        bool
            True if position is within bounds, False otherwise
        """
        x, y = position
        return (0 <= x <= self.width) and (0 <= y <= self.height)

    def record_birth(self) -> None:
        """Record a birth event."""
        self.metrics_tracker.record_birth()

    def record_death(self) -> None:
        """Record a death event."""
        self.metrics_tracker.record_death()

    def record_combat_encounter(self) -> None:
        """Record a combat encounter."""
        self.metrics_tracker.record_combat_encounter()

    def record_successful_attack(self) -> None:
        """Record a successful attack."""
        self.metrics_tracker.record_successful_attack()

    def record_resources_shared(self, amount: float) -> None:
        """Record resources shared between agents."""
        self.metrics_tracker.record_resources_shared(amount)
        self.resources_shared += amount
        self.resources_shared_this_step += amount

    def close(self) -> None:
        """Clean up environment resources."""
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
        agent_data = [
            {
                "simulation_id": self.simulation_id,
                "agent_id": agent.agent_id,
                "birth_time": self.time,
                "agent_type": agent.__class__.__name__,
                "position": agent.position,
                "initial_resources": agent.resource_level,
                "starting_health": agent.starting_health,
                "starvation_threshold": agent.starvation_threshold,
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
        except Exception as e:
            logger.error(f"Error during environment cleanup: {str(e)}")

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
        """
        self.cleanup()

    def action_space(self, agent: Optional[str] = None) -> spaces.Discrete:
        """Get the action space for an agent (PettingZoo API).

        Parameters
        ----------
        agent : str, optional
            Agent ID. If None, returns the general action space.

        Returns
        -------
        gymnasium.spaces.Discrete
            The action space containing all possible actions.
        """
        return self._action_space

    def observation_space(self, agent: Optional[str] = None) -> spaces.Box:
        """Get the observation space for an agent (PettingZoo API).

        Parameters
        ----------
        agent : str, optional
            Agent ID. If None, returns the general observation space.

        Returns
        -------
        gymnasium.spaces.Box
            The observation space defining the shape and bounds of observations.
        """
        return self._observation_space

    def observe(self, agent: str) -> np.ndarray:
        """Returns the observation an agent currently can make.

        Required by PettingZoo API.

        Parameters
        ----------
        agent : str
            Agent identifier

        Returns
        -------
        np.ndarray
            Observation for the agent
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
        np_dtype = getattr(np, self.observation_config.dtype)
        self._observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(NUM_CHANNELS, S, S), dtype=np_dtype
        )

    def _setup_action_space(self) -> None:
        """Setup the action space with all available actions."""
        self._action_space = spaces.Discrete(len(Action))

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

    def _create_initial_agents(self) -> None:
        """Create and add initial agents to the environment based on configuration.

        Creates the starting population of agents according to the configuration
        settings. Supports multiple agent types (SystemAgent, IndependentAgent,
        ControlAgent) with randomized initial positions and default attributes.

        Notes
        -----
        TODO: Agents will be created outside the environment, so this is not needed.

        This method:
        - Reads agent counts from config (system_agents, independent_agents, control_agents)
        - Creates agents with random positions within environment bounds
        - Uses deterministic randomization if seed is set
        - Adds each agent to the environment with default starting resources
        - Sets generation to 0 for all initial agents

        Agent positioning uses uniform random distribution within the environment
        bounds. All agents start with resource_level=1 and generation=0.
        """
        num_system = self.config.system_agents if self.config else 0
        num_independent = self.config.independent_agents if self.config else 0
        num_control = (
            self.config.control_agents if self.config else 1
        )  # At least one for RL

        if self.seed_value is not None:
            rng = random.Random(self.seed_value)
        else:
            rng = random

        for _ in range(num_system):
            position = (
                int(rng.uniform(0, self.width)),
                int(rng.uniform(0, self.height)),
            )
            agent = SystemAgent(
                agent_id=self.get_next_agent_id(),
                position=position,
                resource_level=1,
                environment=self,
                generation=0,
            )
            self.add_agent(agent)

        for _ in range(num_independent):
            position = (
                int(rng.uniform(0, self.width)),
                int(rng.uniform(0, self.height)),
            )
            agent = IndependentAgent(
                agent_id=self.get_next_agent_id(),
                position=position,
                resource_level=1,
                environment=self,
                generation=0,
            )
            self.add_agent(agent)

        for _ in range(num_control):
            position = (
                int(rng.uniform(0, self.width)),
                int(rng.uniform(0, self.height)),
            )
            agent = ControlAgent(
                agent_id=self.get_next_agent_id(),
                position=position,
                resource_level=1,
                environment=self,
                generation=0,
            )
            self.add_agent(agent)

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
        TODO: Deprecate. The observation will come from the spatial index.

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

        # Create resource grid
        resource_grid = torch.zeros(
            (height, width),
            dtype=self.observation_config.torch_dtype,
            device=self.observation_config.device,
        )
        max_amount = self.max_resource or (
            self.config.max_resource_amount if self.config else 10
        )
        for r in self.resources:
            y = int(round(r.position[1]))
            x = int(round(r.position[0]))
            if 0 <= y < height and 0 <= x < width:
                resource_grid[y, x] = r.amount / max_amount

        # Empty layers
        obstacles = torch.zeros_like(resource_grid)
        terrain_cost = torch.zeros_like(resource_grid)

        world_layers = {
            "RESOURCES": resource_grid,
            "OBSTACLES": obstacles,
            "TERRAIN_COST": terrain_cost,
        }

        # Agent position as (y, x)
        ay, ax = int(round(agent.position[1])), int(round(agent.position[0]))

        # Get nearby agents
        nearby = self.get_nearby_agents(
            agent.position, self.observation_config.fov_radius
        )

        allies = []
        enemies = []
        agent_type = type(agent)
        for na in nearby:
            if na == agent or not na.alive:
                continue
            ny = int(round(na.position[1]))
            nx = int(round(na.position[0]))
            hp01 = na.current_health / na.starting_health
            if isinstance(na, agent_type):
                allies.append((ny, nx, hp01))
            else:
                enemies.append((ny, nx, hp01))

        self_hp01 = agent.current_health / agent.starting_health

        obs = self.agent_observations[agent_id]  #! What is this for?
        obs.perceive_world(
            world_layers=world_layers,
            agent_world_pos=(ay, ax),
            self_hp01=self_hp01,
            allies=allies,
            enemies=enemies,
            goal_world_pos=None,  # TODO: Set if needed
            recent_damage_world=[],  # TODO: Implement if needed
            ally_signals_world=[],  # TODO: Implement if needed
            trails_world_points=[],  # TODO: Implement if needed
        )

        tensor = obs.tensor().cpu().numpy()
        return tensor

    def _process_action(self, agent_id: str, action: Optional[int]) -> None:
        """Process an action for a specific agent.

        Executes the specified action for the given agent by calling the
        appropriate action function. Actions are mapped to their corresponding
        implementation functions and executed if the agent is alive.

        Parameters
        ----------
        agent_id : str
            The ID of the agent performing the action.
        action : int
            The action to perform (from Action enum). Must be one of:
            DEFEND, ATTACK, GATHER, SHARE, MOVE, or REPRODUCE.

        Notes
        -----
        This method:
        - Validates that the agent exists and is alive
        - Maps actions to their implementation functions
        - Executes the action with the agent as parameter
        - Logs warnings for invalid actions
        - Returns silently if agent is dead or doesn't exist

        Action implementations are imported from the farm.actions module and
        handle the specific logic for each action type including validation,
        effects, and side effects like resource transfer or combat.
        """
        agent = self._agent_objects.get(agent_id)
        if agent is None or not agent.alive or action is None:
            return

        def defend_action(ag):
            ag.is_defending = True

        action_map = {
            Action.DEFEND: defend_action,
            Action.ATTACK: attack_action,
            Action.GATHER: gather_action,
            Action.SHARE: share_action,
            Action.MOVE: move_action,
            Action.REPRODUCE: reproduce_action,
        }

        func = action_map.get(Action(action))
        if func:
            func(agent)
        else:
            logging.warning(f"Invalid action {action} for agent {agent_id}")

    def _calculate_reward(self, agent_id: str) -> float:
        """Calculate the reward for a specific agent.

        Computes a reward signal for reinforcement learning based on the agent's
        current state including resource level, health, and survival status.
        The reward encourages resource accumulation, health maintenance, and survival.

        Parameters
        ----------
        agent_id : str
            The ID of the agent to calculate reward for.

        Returns
        -------
        float
            The calculated reward value. Returns -10.0 if agent is dead or missing.
            Positive rewards for resource accumulation, health, and survival.

        Notes
        -----
        The reward function includes:
        - Resource reward: 0.1 * resource_level
        - Survival reward: 0.1 for being alive
        - Health reward: current_health / starting_health ratio

        Additional rewards for combat success and cooperation could be added
        by tracking agent-specific metrics in action implementations.
        """
        agent = self._agent_objects.get(agent_id)
        if agent is None or not agent.alive:
            return -10.0

        resource_reward = agent.resource_level * 0.1
        survival_reward = 0.1
        health_reward = agent.current_health / agent.starting_health
        # For combat and cooperation, add if successful_attacks_this_step or resources_shared_this_step, but since global, divide by num agents or something, but poor.
        # Assume 0 for now.

        reward = resource_reward + survival_reward + health_reward

        # Add bonuses if did combat or share, but to detect, perhaps add agent metrics like agent.successful_attacks = 0, incremented in action.

        # Since can't modify actions, for this edit, use the simple formula.

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

        The round-robin scheduling ensures fair time allocation among all
        active agents. Dead or removed agents are automatically skipped.
        """
        if not self.agents:
            self.agent_selection = None
            return
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
                return
        self.agent_selection = None

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
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
        """
        if seed is not None:
            self.seed_value = seed
            random.seed(seed)
            np.random.seed(seed)

        self.time = 0
        # Reset metrics tracker
        self.metrics_tracker.reset()

        self.resources = []
        self.initialize_resources(self.resource_distribution)

        self._agent_objects = {}
        self._create_initial_agents()

        self.agent_observations = {}
        for agent in self._agent_objects.values():
            self.agent_observations[agent.agent_id] = AgentObservation(
                self.observation_config
            )

        self.spatial_index.set_references(
            list(self._agent_objects.values()), self.resources
        )
        self.spatial_index.update()

        self.agents = [a.agent_id for a in self._agent_objects.values() if a.alive]
        self.agent_selection = self.agents[0] if self.agents else None
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.observations = {a: self._get_observation(a) for a in self.agents}

        self.previous_agent_states = {}

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
            If None, no action is taken.

        Returns
        -------
        tuple
            A 5-tuple containing:
            - observation (np.ndarray): The observation for the current agent
            - reward (float): The reward received by the agent
            - terminated (bool): Whether the episode has terminated
            - truncated (bool): Whether the episode was truncated (e.g., max steps reached)
            - info (dict): Additional information about the step
        """
        if self.agent_selection is None or not self.agents:
            dummy_obs = np.zeros(
                self._observation_space.shape, dtype=self._observation_space.dtype
            )
            return dummy_obs, 0.0, True, True, {}

        agent_id = self.agent_selection
        agent = self._agent_objects.get(agent_id)
        if agent:
            self.previous_agent_states[agent_id] = {
                "resource_level": agent.resource_level,
                "health": agent.current_health,
                "alive": agent.alive,
            }
        self._process_action(agent_id, action)
        self.update()

        # Add resource check to terminated:
        alive_agents = [a for a in self._agent_objects.values() if a.alive]
        total_resources = sum(r.amount for r in self.resources)
        terminated = len(alive_agents) == 0 or total_resources == 0
        truncated = self.time >= self.max_steps

        reward = self._calculate_reward(agent_id)

        if agent:
            resource_delta = (
                agent.resource_level
                - self.previous_agent_states[agent_id]["resource_level"]
            )
            health_delta = (
                agent.current_health - self.previous_agent_states[agent_id]["health"]
            )
            survival_bonus = 0.1 if agent.alive else 0

            # Assume for combat success, if resource_delta >0 and health_delta >=0, +2 (possible successful attack)
            # For cooperation, if resource_delta <0 and health_delta >=0, +1 (possible share)

            reward = resource_delta + health_delta * 0.5 + survival_bonus

            if not agent.alive:
                reward -= 10

            # Add more if needed.

        # Get observation for current agent
        observation = (
            self._get_observation(agent_id)
            if agent
            else np.zeros(
                self._observation_space.shape, dtype=self._observation_space.dtype
            )
        )

        return observation, reward, terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        """Render the current state of the environment.

        Parameters
        ----------
        mode : str
            The rendering mode. Currently only supports "human".
        """
        if mode == "human":
            print(f"Time: {self.time}")
            print(f"Active agents: {len(self.agents)}")
            print(f"Total resources: {sum(r.amount for r in self.resources)}")
        # TODO: Implement proper visualization if needed

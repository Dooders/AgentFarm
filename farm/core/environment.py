import logging
import random
from enum import IntEnum
from typing import Dict

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import AECEnv

from farm.actions.attack import attack_action
from farm.actions.gather import gather_action
from farm.actions.move import move_action
from farm.actions.reproduce import reproduce_action
from farm.actions.share import share_action
from farm.agents import ControlAgent, IndependentAgent, SystemAgent
from farm.core.observations import NUM_CHANNELS, AgentObservation, ObservationConfig
from farm.core.resource_manager import ResourceManager
from farm.core.spatial_index import SpatialIndex
from farm.core.state import EnvironmentState
from farm.database.database import SimulationDatabase
from farm.database.utilities import setup_db
from farm.utils.short_id import ShortUUID


class Action(IntEnum):
    DEFEND = 0
    ATTACK = 1
    GATHER = 2
    SHARE = 3
    MOVE = 4
    REPRODUCE = 5


logger = logging.getLogger(__name__)


class Environment(AECEnv):
    def __init__(
        self,
        width,
        height,
        resource_distribution,
        db_path="simulation.db",
        max_resource=None,
        config=None,
        simulation_id=None,
        seed=None,
    ):
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

        # Setup database and get potentially modified path
        final_db_path = setup_db(db_path)

        # Initialize basic attributes
        self.width = width
        self.height = height
        self.agent_objects = []
        self.agents = []  # for PettingZoo
        self.resources = []
        self.time = 0

        # Store simulation ID
        self.simulation_id = simulation_id or ShortUUID().uuid()

        # Only initialize database if db_path is provided (not for in-memory DB)
        if final_db_path is not None:
            self.db = SimulationDatabase(
                final_db_path, simulation_id=self.simulation_id
            )
        else:
            # Will be set to InMemorySimulationDatabase later
            self.db = None

        self.id_generator = ShortUUID()
        self.next_resource_id = 0
        self.max_resource = max_resource
        self.config = config
        self.initial_agent_count = 0  #! Is this really needed?
        self.resource_distribution = resource_distribution
        self.max_steps = (
            config.max_steps if config and hasattr(config, "max_steps") else 1000
        )
        self.agent_observations = {}  # Initialize agent observations dictionary

        # Initialize spatial index for efficient spatial queries
        self.spatial_index = SpatialIndex(width, height)

        # Add tracking for births and deaths
        self.births_this_step = 0
        self.deaths_this_step = 0

        # Add tracking for resources shared
        self.resources_shared = 0
        self.resources_shared_this_step = 0

        # Add tracking for combat metrics
        self.combat_encounters = 0
        self.successful_attacks = 0
        self.combat_encounters_this_step = 0
        self.successful_attacks_this_step = 0

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
        self.spatial_index.set_references(self.agent_objects, self.resources)
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

    def mark_positions_dirty(self):
        """Public method for agents to mark positions as dirty when they move."""
        self.spatial_index.mark_positions_dirty()

    def get_nearby_agents(self, position, radius):
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

    def get_nearby_resources(self, position, radius):
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

    def get_nearest_resource(self, position):
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

    def get_next_resource_id(self):
        resource_id = self.next_resource_id
        self.next_resource_id += 1
        return resource_id

    def consume_resource(self, resource, amount):
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

    def initialize_resources(self, distribution):
        """Initialize resources with proper amounts using ResourceManager."""
        # Use ResourceManager to initialize resources (passes through to original logic)
        resources = self.resource_manager.initialize_resources(distribution)

        # Update environment's resource list to match ResourceManager
        self.resources = self.resource_manager.resources

        # Update next_resource_id to match ResourceManager
        self.next_resource_id = self.resource_manager.next_resource_id

    def remove_agent(self, agent):
        self.record_death()
        self.agent_objects.remove(agent)
        self.spatial_index.mark_positions_dirty()  # Mark positions as dirty when agent is removed
        if agent.agent_id in self.agent_observations:
            del self.agent_observations[agent.agent_id]

    def collect_action(self, **action_data):
        """Collect an action for batch processing and database logging.

        Parameters
        ----------
        **action_data : dict
            Action data including step_number, agent_id, action_type, etc.
        """

        if self.db is not None:
            self.db.logger.log_agent_action(
                step_number=action_data["step_number"],
                agent_id=action_data["agent_id"],
                action_type=action_data["action_type"],
                action_target_id=action_data.get("action_target_id"),
                resources_before=action_data.get("resources_before"),
                resources_after=action_data.get("resources_after"),
                reward=action_data.get("reward"),
                details=action_data.get("details"),
            )

    def log_interaction_edge(
        self,
        source_type,
        source_id,
        target_type,
        target_id,
        interaction_type,
        action_type=None,
        details=None,
    ):
        """Log an interaction as an edge between nodes if database is enabled."""
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

    def update(self):
        """Update environment state for current time step."""
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
            self.update_metrics(metrics)

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

    def _calculate_metrics(self):
        """Calculate various metrics for the current simulation state."""
        #! resources_shared is now being calculated
        try:
            # Get alive agents
            alive_agents = [agent for agent in self.agent_objects if agent.alive]
            total_agents = len(alive_agents)

            # Calculate agent type counts
            system_agents = len([a for a in alive_agents if isinstance(a, SystemAgent)])
            independent_agents = len(
                [a for a in alive_agents if isinstance(a, IndependentAgent)]
            )
            control_agents = len(
                [a for a in alive_agents if isinstance(a, ControlAgent)]
            )

            # Get births and deaths from tracking
            births = self.births_this_step
            deaths = self.deaths_this_step

            # Calculate generation metrics
            current_max_generation = (
                max([a.generation for a in alive_agents]) if alive_agents else 0
            )

            # Calculate health and age metrics
            average_health = (
                sum(a.current_health for a in alive_agents) / total_agents
                if total_agents > 0
                else 0
            )
            average_age = (
                sum(self.time - a.birth_time for a in alive_agents) / total_agents
                if total_agents > 0
                else 0
            )
            average_reward = (
                sum(a.total_reward for a in alive_agents) / total_agents
                if total_agents > 0
                else 0
            )

            # Calculate resource metrics
            total_resources = sum(r.amount for r in self.resources)
            average_agent_resources = (
                sum(a.resource_level for a in alive_agents) / total_agents
                if total_agents > 0
                else 0
            )
            resource_efficiency = (
                total_resources
                / (
                    len(self.resources)
                    * (self.config.max_resource_amount if self.config else 30)
                )
                if self.resources
                else 0
            )

            # Calculate genetic diversity
            genome_counts = {}
            for agent in alive_agents:
                genome_counts[agent.genome_id] = (
                    genome_counts.get(agent.genome_id, 0) + 1
                )
            genetic_diversity = (
                len(genome_counts) / total_agents if total_agents > 0 else 0
            )
            dominant_genome_ratio = (
                max(genome_counts.values()) / total_agents if genome_counts else 0
            )

            # Get combat and sharing metrics
            combat_encounters = getattr(self, "combat_encounters", 0)
            successful_attacks = getattr(self, "successful_attacks", 0)
            resources_shared = getattr(self, "resources_shared", 0)
            resources_shared_this_step = getattr(self, "resources_shared_this_step", 0)
            combat_encounters_this_step = getattr(
                self, "combat_encounters_this_step", 0
            )
            successful_attacks_this_step = getattr(
                self, "successful_attacks_this_step", 0
            )

            # Calculate resource distribution entropy
            resource_amounts = [r.amount for r in self.resources]
            if resource_amounts:
                total = sum(resource_amounts)
                if total > 0:
                    probabilities = [amt / total for amt in resource_amounts]
                    resource_distribution_entropy = -sum(
                        p * np.log(p) if p > 0 else 0 for p in probabilities
                    )
                else:
                    resource_distribution_entropy = 0.0
            else:
                resource_distribution_entropy = 0.0

            # Calculate resource consumption for this step
            previous_resources = getattr(self, "previous_total_resources", 0)
            current_resources = sum(r.amount for r in self.resources)
            resources_consumed = max(0, previous_resources - current_resources)
            self.previous_total_resources = current_resources

            # Add births and deaths to metrics
            metrics = {
                "total_agents": total_agents,
                "system_agents": system_agents,
                "independent_agents": independent_agents,
                "control_agents": control_agents,
                "total_resources": total_resources,
                "average_agent_resources": average_agent_resources,
                "resources_consumed": resources_consumed,
                "births": births,
                "deaths": deaths,
                "current_max_generation": current_max_generation,
                "resource_efficiency": resource_efficiency,
                "resource_distribution_entropy": resource_distribution_entropy,
                "average_agent_health": average_health,
                "average_agent_age": average_age,
                "average_reward": average_reward,
                "combat_encounters": combat_encounters,
                "successful_attacks": successful_attacks,
                "resources_shared": resources_shared,
                "resources_shared_this_step": resources_shared_this_step,
                "genetic_diversity": genetic_diversity,
                "dominant_genome_ratio": dominant_genome_ratio,
                "combat_encounters_this_step": combat_encounters_this_step,
                "successful_attacks_this_step": successful_attacks_this_step,
            }

            # Reset counters for next step
            self.births_this_step = 0
            self.deaths_this_step = 0

            return metrics
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
            return {}  # Return empty metrics on error

    def get_next_agent_id(self):
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

    def is_valid_position(self, position):
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

    def _get_random_position(self):
        """Generate a random position within the environment bounds.

        Uses a seeded random number generator for deterministic behavior if a seed is available.

        Returns
        -------
        tuple
            A (x, y) coordinate tuple within the environment bounds.
        """
        if hasattr(self, "seed_value") and self.seed_value is not None:
            # Create a separate random state to avoid affecting the global state
            rng = random.Random(
                self.seed_value + self.time
            )  # Add current time to avoid same positions
        else:
            rng = random

        x = rng.uniform(0, self.width)
        y = rng.uniform(0, self.height)
        return (x, y)

    def record_birth(self):
        """Record a birth event."""
        self.births_this_step += 1

    def record_death(self):
        """Record a death event."""
        self.deaths_this_step += 1

    def close(self):
        """Clean up environment resources."""
        if hasattr(self, "db") and self.db is not None:
            self.db.close()

    def add_agent(self, agent):
        """Add an agent to the environment with efficient database logging."""
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
        self.agent_objects.append(agent)
        if self.time == 0:
            self.initial_agent_count += 1

        # Mark positions as dirty when new agent is added
        self.spatial_index.mark_positions_dirty()

        # Batch log to database using SQLAlchemy
        if self.db is not None:
            self.db.logger.log_agents_batch(agent_data)

        self.agent_observations[agent.agent_id] = AgentObservation(
            self.observation_config
        )

    def cleanup(self):
        """Clean up environment resources."""
        try:
            if hasattr(self, "db") and self.db is not None:
                # Use logger for buffer flushing
                if hasattr(self.db, "logger"):
                    self.db.logger.flush_all_buffers()
                self.db.close()
        except Exception as e:
            logger.error(f"Error during environment cleanup: {str(e)}")

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()

    def update_metrics(self, metrics: Dict):
        """Update environment metrics and log to database.

        Parameters
        ----------
        metrics : Dict
            Dictionary containing metrics to update and log
        """
        try:
            # Log metrics to database
            if self.db:
                self.db.logger.log_step(
                    step_number=self.time,
                    agent_states=[
                        self._prepare_agent_state(agent)
                        for agent in self.agent_objects
                        if agent.alive
                    ],
                    resource_states=[
                        self._prepare_resource_state(resource)
                        for resource in self.resources
                    ],
                    metrics=metrics,
                )
        except Exception as e:
            logging.error(f"Error updating metrics: {e}")

    def _prepare_agent_state(self, agent) -> tuple:
        """Prepare agent state data for database logging.

        Parameters
        ----------
        agent : BaseAgent
            Agent to prepare state data for

        Returns
        -------
        tuple
            State data in format expected by database
        """
        return (
            agent.agent_id,
            agent.position[0],  # x coordinate
            agent.position[1],  # y coordinate
            agent.resource_level,
            agent.current_health,
            agent.starting_health,
            agent.starvation_threshold,
            int(agent.is_defending),
            agent.total_reward,
            self.time - agent.birth_time,  # age
        )

    def _prepare_resource_state(self, resource) -> tuple:
        """Prepare resource state data for database logging.

        Parameters
        ----------
        resource : Resource
            Resource to prepare state data for

        Returns
        -------
        tuple
            State data in format expected by database
        """
        return (
            resource.resource_id,
            resource.amount,
            resource.position[0],  # x coordinate
            resource.position[1],  # y coordinate
        )

    #! part of context manager, commented out for now
    # def register_active_context(self, agent: BaseAgent) -> None:
    #     """Register an active agent context."""
    #     with self._context_lock:
    #         self._active_contexts.add(agent)

    # def unregister_active_context(self, agent: BaseAgent) -> None:
    #     """Unregister an active agent context."""
    #     with self._context_lock:
    #         self._active_contexts.discard(agent)

    # def get_active_contexts(self) -> Set[BaseAgent]:
    #     """Get currently active agent contexts."""
    #     with self._context_lock:
    #         return self._active_contexts.copy()

    def action_space(self, agent):
        """Get the action space for an agent.

        Parameters
        ----------
        agent : str
            Agent identifier (unused but required by PettingZoo interface).

        Returns
        -------
        gymnasium.spaces.Discrete
            The action space containing all possible actions.
        """
        return self._action_space

    def observation_space(self, agent):
        """Get the observation space for an agent.

        Parameters
        ----------
        agent : str
            Agent identifier (unused but required by PettingZoo interface).

        Returns
        -------
        gymnasium.spaces.Box
            The observation space defining the shape and bounds of observations.
        """
        return self._observation_space

    def _setup_observation_space(self, config):
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

    def _setup_action_space(self):
        """Setup the action space with all available actions."""
        self._action_space = spaces.Discrete(
            len(Action)
        )  # Actions: DEFEND, ATTACK, GATHER, SHARE, MOVE, REPRODUCE

    def _create_initial_agents(self):
        """Create and add initial agents to the environment based on configuration."""
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

    def _get_observation(self, agent_id):
        """Generate an observation for a specific agent.

        Parameters
        ----------
        agent_id : str
            The ID of the agent to generate an observation for.

        Returns
        -------
        np.ndarray
            The observation tensor for the agent.
        """
        agent = next((a for a in self.agent_objects if a.agent_id == agent_id), None)
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

        obs = self.agent_observations[agent_id]
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

    def _process_action(self, agent_id, action):
        """Process an action for a specific agent.

        Parameters
        ----------
        agent_id : str
            The ID of the agent performing the action.
        action : int
            The action to perform (from Action enum).
        """
        agent = next((a for a in self.agent_objects if a.agent_id == agent_id), None)
        if agent is None or not agent.alive:
            return

        def defend_action(ag):
            ag.is_defending = True
            self.collect_action(
                step_number=self.time,
                agent_id=ag.agent_id,
                action_type="defend",
                details="Set defending flag",
            )

        action_map = {
            Action.DEFEND: defend_action,
            Action.ATTACK: attack_action,
            Action.GATHER: gather_action,
            Action.SHARE: share_action,
            Action.MOVE: move_action,
            Action.REPRODUCE: reproduce_action,
        }

        func = action_map.get(action)
        if func:
            func(agent)
        else:
            logging.warning(f"Invalid action {action} for agent {agent_id}")

    def _calculate_reward(self, agent_id):
        """Calculate the reward for a specific agent.

        Parameters
        ----------
        agent_id : str
            The ID of the agent to calculate reward for.

        Returns
        -------
        float
            The calculated reward value.
        """
        agent = next((a for a in self.agent_objects if a.agent_id == agent_id), None)
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

    def _next_agent(self):
        """Select the next agent to act in the environment."""
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

    def reset(self, *, seed=None, options=None):
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
        self.births_this_step = 0
        self.deaths_this_step = 0
        self.resources_shared = 0
        self.resources_shared_this_step = 0
        self.combat_encounters = 0
        self.successful_attacks = 0
        self.combat_encounters_this_step = 0
        self.successful_attacks_this_step = 0

        self.resources = []
        self.initialize_resources(self.resource_distribution)

        self.agent_objects = []
        self.initial_agent_count = 0
        self._create_initial_agents()

        self.agent_observations = {}
        for agent in self.agent_objects:
            self.agent_observations[agent.agent_id] = AgentObservation(
                self.observation_config
            )

        self.spatial_index.set_references(self.agent_objects, self.resources)
        self.spatial_index.update()

        self.agents = [a.agent_id for a in self.agent_objects if a.alive]
        self.agent_selection = self.agents[0] if self.agents else None
        self.rewards = {a: 0 for a in self.agents}
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

    def step(self, action=None) -> tuple[np.ndarray, float, bool, bool, dict]:
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
        agent = next((a for a in self.agent_objects if a.agent_id == agent_id), None)
        if agent:
            self.previous_agent_states[agent_id] = {
                "resource_level": agent.resource_level,
                "health": agent.current_health,
                "alive": agent.alive,
            }
        self._process_action(agent_id, action)
        self.update()

        # Add resource check to terminated:
        alive_agents = [a for a in self.agent_objects if a.alive]
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

    def render(self, mode="human"):
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

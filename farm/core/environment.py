import logging
import random
from typing import Dict

import numpy as np
from scipy.spatial import KDTree

from farm.agents import ControlAgent, IndependentAgent, SystemAgent
from farm.core.resources import Resource
from farm.core.state import EnvironmentState

# from farm.agents.base_agent import BaseAgent
from farm.database.database import SimulationDatabase
from farm.database.utilities import setup_db
from farm.utils.short_id import ShortUUID

logger = logging.getLogger(__name__)


class Environment:
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
        self.agents = []
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

        self.seed = ShortUUID()
        self.next_resource_id = 0
        # self.max_resource = max_resource or (config.max_resource_amount if config else None)
        self.max_resource = max_resource
        self.config = config
        self.initial_agent_count = 0
        self.pending_actions = []  # Initialize pending_actions list

        # Add KD-tree attributes
        self.agent_kdtree = None
        self.resource_kdtree = None
        self.agent_positions = None
        self.resource_positions = None

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

        # Context tracking
        # self._active_contexts: Set[BaseAgent] = set()
        # self._context_lock = threading.Lock()

        # Initialize environment
        self.initialize_resources(resource_distribution)
        self._update_kdtrees()

    def _update_kdtrees(self):
        """Update KD-trees for efficient spatial queries."""
        # Update agent KD-tree
        alive_agents = [agent for agent in self.agents if agent.alive]
        if alive_agents:
            self.agent_positions = np.array([agent.position for agent in alive_agents])
            self.agent_kdtree = KDTree(self.agent_positions)
        else:
            self.agent_kdtree = None
            self.agent_positions = None

        # Update resource KD-tree
        if self.resources:
            self.resource_positions = np.array(
                [resource.position for resource in self.resources]
            )
            self.resource_kdtree = KDTree(self.resource_positions)
        else:
            self.resource_kdtree = None
            self.resource_positions = None

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
        if self.agent_kdtree is None:
            return []

        indices = self.agent_kdtree.query_ball_point(position, radius)
        return [
            agent for i, agent in enumerate(self.agents) if agent.alive and i in indices
        ]

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
        if self.resource_kdtree is None:
            return []

        indices = self.resource_kdtree.query_ball_point(position, radius)
        return [self.resources[i] for i in indices]

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
        if self.resource_kdtree is None:
            return None

        distance, index = self.resource_kdtree.query(position)
        return self.resources[index]

    def get_next_resource_id(self):
        resource_id = self.next_resource_id
        self.next_resource_id += 1
        return resource_id

    def initialize_resources(self, distribution):
        """Initialize resources with proper amounts."""
        # Use a seeded random generator for resource initialization
        if self.seed_value is not None:
            # Create a separate random state to avoid affecting the global state
            rng = random.Random(self.seed_value)
            np_rng = np.random.RandomState(self.seed_value)
        else:
            rng = random
            np_rng = np.random

        for _ in range(distribution["amount"]):
            position = (rng.uniform(0, self.width), rng.uniform(0, self.height))

            # Deterministic resource amount
            if self.config and hasattr(self.config, "max_resource_amount"):
                if self.seed_value is not None:
                    # Use a fixed formula based on position for initial amount
                    pos_sum = position[0] + position[1]
                    amount = 3 + int(pos_sum % 6)  # Range from 3 to 8
                else:
                    # Random as before
                    amount = random.randint(3, 8)
            else:
                amount = 5  # Default if no config specified

            resource = Resource(
                resource_id=self.get_next_resource_id(),
                position=position,
                amount=amount,
                max_amount=self.config.max_resource_amount if self.config else 10,
                regeneration_rate=(
                    self.config.resource_regen_rate if self.config else 0.1
                ),
            )
            self.resources.append(resource)
            # Log resource to database
            if self.db:
                self.db.logger.log_resource(
                    resource_id=resource.resource_id,
                    initial_amount=resource.amount,
                    position=resource.position,
                )

    # def add_agent(self, agent):
    #     self.agents.append(agent)
    #     # Update initial count only during setup (time=0)
    #     if self.time == 0:
    #         self.initial_agent_count += 1

    def remove_agent(self, agent):
        self.record_death()
        self.agents.remove(agent)

    def collect_action(self, **action_data):
        """Collect an action for batch processing."""

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

    def update(self):
        """Update environment state for current time step."""
        try:
            # Update resources with deterministic regeneration if seed is set
            if hasattr(self, "seed_value") and self.seed_value is not None:
                # Create deterministic RNG based on seed and current time
                rng = random.Random(self.seed_value + self.time)

                # Deterministically decide which resources regenerate
                for resource in self.resources:
                    # Use resource ID and position as additional entropy sources
                    decision_seed = hash(
                        (
                            resource.resource_id,
                            resource.position[0],
                            resource.position[1],
                            self.time,
                        )
                    )
                    # Mix with simulation seed
                    combined_seed = (self.seed_value * 100000) + decision_seed
                    # Create a deterministic random generator for this resource
                    resource_rng = random.Random(combined_seed)

                    # Check if this resource should regenerate
                    if (
                        resource_rng.random() < self.config.resource_regen_rate
                        if self.config
                        else 0.1
                        and (
                            self.max_resource is None
                            or resource.amount < self.max_resource
                        )
                    ):
                        resource.amount = min(
                            (
                                resource.amount + self.config.resource_regen_amount
                                if self.config
                                else 2
                            ),
                            self.max_resource or float("inf"),
                        )
            else:
                # Use standard random method if no seed is set (non-deterministic mode)
                regen_mask = np.random.random(len(self.resources)) < (
                    self.config.resource_regen_rate if self.config else 0.1
                )
                for resource, should_regen in zip(self.resources, regen_mask):
                    if should_regen and (
                        self.max_resource is None or resource.amount < self.max_resource
                    ):
                        resource.amount = min(
                            (
                                resource.amount + self.config.resource_regen_amount
                                if self.config
                                else 2
                            ),
                            self.max_resource or float("inf"),
                        )

            # Calculate and log metrics
            metrics = self._calculate_metrics()
            self.update_metrics(metrics)

            # Update KD trees
            self._update_kdtrees()

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
            alive_agents = [agent for agent in self.agents if agent.alive]
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
            return self.seed.id()

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

    def _initialize_resources(self):
        """Initialize resources in the environment."""
        for i in range(self.config.initial_resources if self.config else 20):
            # Random position
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)

            # Create resource with regeneration parameters
            resource = Resource(
                resource_id=i,
                position=(x, y),
                amount=self.config.max_resource_amount if self.config else 10,
                max_amount=self.config.max_resource_amount if self.config else 10,
                regeneration_rate=(
                    self.config.resource_regen_rate if self.config else 0.1
                ),
            )
            self.resources.append(resource)

    def _get_random_position(self):
        """
        Generate a random position within the environment bounds.
        Uses a seeded random number generator for deterministic behavior if a seed is available.
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
        self.agents.append(agent)
        if self.time == 0:
            self.initial_agent_count += 1

        # Batch log to database using SQLAlchemy
        if self.db is not None:
            self.db.logger.log_agents_batch(agent_data)

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
                        for agent in self.agents
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

import logging
import os
import random
import threading
import time
from typing import Dict, Set

import numpy as np
from scipy.spatial import cKDTree
from sqlalchemy import create_engine

from farm.agents import ControlAgent, IndependentAgent, SystemAgent
from farm.core.resources import Resource
from farm.core.state import EnvironmentState

# from farm.agents.base_agent import BaseAgent
from farm.database.database import SimulationDatabase
from farm.utils.short_id import ShortUUID

logger = logging.getLogger(__name__)


def setup_db(db_path):
    """Initialize database if it doesn't exist.
    
    Parameters
    ----------
    db_path : str
        Path to the database file
        
    Returns
    -------
    str
        Path to the database file
    """
    # Skip setup for in-memory database
    if db_path is None:
        return None
        
    try:
        # Create database if it doesn't exist
        if not os.path.exists(db_path):
            engine = create_engine(f"sqlite:///{db_path}")
            Base.metadata.create_all(engine)
            logger.info(f"Created new database at {db_path}")
        return db_path
        
    except Exception as e:
        logger.error(f"Failed to setup database: {e}")
        raise


class Environment:
    def __init__(self, config, database=None, experiment_id=None, iteration_number=None):
        """Initialize environment.
        
        Parameters
        ----------
        config : SimulationConfig
            Configuration object
        database : SimulationDatabase, optional
            Database instance to use
        experiment_id : int, optional
            ID of experiment this environment belongs to
        iteration_number : int, optional
            Iteration number within the experiment
        """
        self.config = config
        self.db = database
        self.time = 0  # Initialize time to 0
        self.iteration_number = iteration_number
        
        # Initialize seed for ID generation
        import random
        import string
        
        class Seed:
            def __init__(self):
                self.counter = 0
                # Generate a random prefix for this environment to avoid ID collisions
                self.prefix = ''.join(random.choices(string.ascii_lowercase, k=5))
                
            def id(self):
                self.counter += 1
                return f"{self.prefix}_{self.counter}"
        
        self.seed = Seed()
        
        if self.db and experiment_id:
            self.db.current_experiment_id = experiment_id
            self.db.log_experiment_event('environment_initialized', {
                'config': config.to_dict()
            })
            
        # Initialize environment state
        self._initialize_state()

    def _initialize_state(self):
        """Initialize environment state."""
        # Initialize with current simulation context
        self.agents = {}
        self.resources = {}
        self.step_number = 0
        
        # Initialize KD trees
        self.agent_kdtree = None
        self.resource_kdtree = None
        self.agent_positions = None
        self.resource_positions = None
        
        # Initialize resource parameters
        self.max_resource = self.config.max_resource_amount if hasattr(self.config, 'max_resource_amount') else None
        
        # Initialize counters
        self.births_this_step = 0
        self.deaths_this_step = 0
        self.total_births = 0
        self.total_deaths = 0
        self.resources_shared = 0
        self.resources_shared_this_step = 0
        self.combat_encounters = 0
        self.combat_encounters_this_step = 0
        self.successful_attacks = 0
        self.successful_attacks_this_step = 0
        
        # Create simulation record if we have a database and experiment
        if self.db and self.db.current_experiment_id:
            # Get the iteration number from the environment if available
            iteration_number = getattr(self, 'iteration_number', None)
            
            self.simulation_id = self.db.create_simulation_in_experiment(
                experiment_id=self.db.current_experiment_id,
                config_variation=self.config.to_dict(),
                iteration_number=iteration_number
            )
        
        self._setup_initial_agents()
        self._setup_initial_resources()

    def _setup_initial_agents(self):
        """Initialize agents in the environment based on configuration."""
        from farm.core.simulation import create_initial_agents
        
        # Set width and height attributes from config for compatibility with create_initial_agents
        self.width = self.config.width
        self.height = self.config.height
        
        # Create initial agents based on configuration
        create_initial_agents(
            environment=self,
            num_system_agents=self.config.system_agents,
            num_independent_agents=self.config.independent_agents,
            num_control_agents=self.config.control_agents
        )
        
    def _setup_initial_resources(self):
        """Initialize resources in the environment."""
        # Initialize resources based on configuration
        self._initialize_resources()

    def _update_kdtrees(self):
        """Update KD-trees for efficient spatial queries."""
        try:
            from scipy.spatial import cKDTree
            import numpy as np
        except ImportError:
            self.agent_kdtree = None
            self.resource_kdtree = None
            return
            
        # Update agent KD-tree
        alive_agents = [agent for agent in self.agents.values() if agent.alive]
        if alive_agents:
            self.agent_positions = np.array([agent.position for agent in alive_agents])
            self.agent_kdtree = cKDTree(self.agent_positions)
        else:
            self.agent_kdtree = None
            self.agent_positions = None

        # Update resource KD-tree
        if self.resources:
            self.resource_positions = np.array(
                [resource.position for resource in self.resources.values()]
            )
            self.resource_kdtree = cKDTree(self.resource_positions)
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
        alive_agents = [agent for agent in self.agents.values() if agent.alive]
        return [alive_agents[i] for i in indices if i < len(alive_agents)]

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
        resources_list = list(self.resources.values())
        return [resources_list[i] for i in indices if i < len(resources_list)]

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
        resources_list = list(self.resources.values())
        if index < len(resources_list):
            return resources_list[index]
        return None

    def get_next_resource_id(self):
        resource_id = self.next_resource_id
        self.next_resource_id += 1
        return resource_id

    def initialize_resources(self, distribution):
        """Initialize resources with proper amounts."""
        for _ in range(distribution["amount"]):
            position = (random.uniform(0, self.width), random.uniform(0, self.height))
            resource = Resource(
                resource_id=self.get_next_resource_id(),
                position=position,
                # amount=self.config.max_resource_amount,  # Use config value instead of random
                amount=random.randint(3, 8),
                max_amount=self.config.max_resource_amount,
                regeneration_rate=self.config.resource_regen_rate,
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
        if agent.agent_id in self.agents:
            del self.agents[agent.agent_id]

    def collect_action(self, **action_data):
        """Collect an action for batch processing."""

        if self.logger is not None:
            self.logger.log_agent_action(
                step_number=action_data["step_number"],
                agent_id=action_data["agent_id"],
                action_type=action_data["action_type"],
                action_target_id=action_data.get("action_target_id"),
                position_before=action_data.get("position_before"),
                position_after=action_data.get("position_after"),
                resources_before=action_data.get("resources_before"),
                resources_after=action_data.get("resources_after"),
                reward=action_data.get("reward"),
                details=action_data.get("details"),
            )

    def update(self):
        """Update environment state for current time step."""
        try:
            # Update resources with proper regeneration
            resources_list = list(self.resources.values())
            regen_mask = (
                np.random.random(len(resources_list)) < self.config.resource_regen_rate
            )
            for resource, should_regen in zip(resources_list, regen_mask):
                if should_regen and (
                    self.max_resource is None or resource.amount < self.max_resource
                ):
                    resource.amount = min(
                        resource.amount + self.config.resource_regen_amount,
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
            alive_agents = [agent for agent in self.agents.values() if agent.alive]
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
            total_resources = sum(r.amount for r in self.resources.values())
            average_agent_resources = (
                sum(a.resource_level for a in alive_agents) / total_agents
                if total_agents > 0
                else 0
            )
            resource_efficiency = (
                total_resources
                / (len(self.resources) * self.config.max_resource_amount)
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
            combat_encounters_this_step = getattr(self, "combat_encounters_this_step", 0)
            successful_attacks_this_step = getattr(self, "successful_attacks_this_step", 0)

            # Calculate resource distribution entropy
            resource_amounts = [r.amount for r in self.resources.values()]
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
            current_resources = sum(r.amount for r in self.resources.values())
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
        for i in range(self.config.initial_resources):
            # Random position
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)

            # Create resource with regeneration parameters
            resource = Resource(
                resource_id=i,
                position=(x, y),
                amount=self.config.max_resource_amount,
                max_amount=self.config.max_resource_amount,
                regeneration_rate=self.config.resource_regen_rate,
            )
            # Add resource to dictionary with resource_id as key
            self.resources[resource.resource_id] = resource

    def _get_random_position(self):
        """Get a random position within the environment bounds."""
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        return (x, y)

    def record_birth(self):
        """Record a birth event."""
        self.births_this_step += 1

    def record_death(self):
        """Record a death event."""
        self.deaths_this_step += 1

    def step(self):
        """Execute one step of the simulation."""
        # Increment step counter
        self.step_number += 1
        self.time += 1
        
        # Reset step-specific counters
        self.births_this_step = 0
        self.deaths_this_step = 0
        self.resources_shared_this_step = 0
        self.combat_encounters_this_step = 0
        self.successful_attacks_this_step = 0
        
        # Log step metrics if we have a database
        if self.db and hasattr(self, 'simulation_id'):
            metrics = self.get_metrics()
            for name, value in metrics.items():
                self.db.log_experiment_metric(
                    metric_name=name,
                    metric_value=value,
                    metric_type='step',
                    metadata={'step': self.step_number}
                )
        
        return self.get_metrics()

    def close(self):
        """Clean up environment resources."""
        if hasattr(self, "db"):
            self.db.close()

    def add_agent(self, agent):
        """Add an agent to the environment with efficient database logging."""
        agent_data = [
            {
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

        # Add to environment (agents is a dictionary with agent_id as key)
        self.agents[agent.agent_id] = agent
        if self.time == 0:
            self.initial_agent_count = self.initial_agent_count + 1 if hasattr(self, 'initial_agent_count') else 1

        # Batch log to database using SQLAlchemy
        if self.db is not None:
            self.db.logger.log_agents_batch(agent_data)

    def cleanup(self):
        """Clean up environment resources."""
        try:
            if hasattr(self, 'db') and self.db is not None:
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

    def get_metrics(self):
        """Get current environment metrics.
        
        Returns
        -------
        Dict
            Dictionary of metrics
        """
        # Basic metrics
        metrics = {
            "step": self.step_number,
            "time": self.time,
            "total_agents": len(self.agents),
            "total_resources": sum(resource.amount for resource in self.resources.values()),
            "births_this_step": self.births_this_step,
            "deaths_this_step": self.deaths_this_step,
            "total_births": self.total_births if hasattr(self, 'total_births') else 0,
            "total_deaths": self.total_deaths if hasattr(self, 'total_deaths') else 0,
            "resources_shared": self.resources_shared,
            "resources_shared_this_step": self.resources_shared_this_step,
            "combat_encounters": self.combat_encounters,
            "combat_encounters_this_step": self.combat_encounters_this_step,
            "successful_attacks": self.successful_attacks,
            "successful_attacks_this_step": self.successful_attacks_this_step,
        }
        
        return metrics

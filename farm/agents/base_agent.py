import logging
import random
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch

from farm.actions.attack import DEFAULT_ATTACK_CONFIG, AttackActionSpace, AttackModule
from farm.actions.base_dqn import SharedEncoder
from farm.actions.gather import DEFAULT_GATHER_CONFIG, GatherModule
from farm.actions.move import DEFAULT_MOVE_CONFIG, MoveModule
from farm.actions.reproduce import DEFAULT_REPRODUCE_CONFIG, ReproduceModule
from farm.actions.select import SelectConfig, SelectModule, create_selection_state
from farm.actions.share import DEFAULT_SHARE_CONFIG, ShareModule
from farm.core.action import *
from farm.core.genome import Genome
from farm.core.perception import PerceptionData
from farm.core.state import AgentState
from farm.database.data_types import GenomeId
from farm.memory.redis_memory import AgentMemoryManager, RedisMemoryConfig

if TYPE_CHECKING:
    from farm.core.environment import Environment

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        agent_id: str,
        position: tuple[float, float],
        resource_level: int,
        environment: "Environment",
        action_set: list[Action] = [],
        parent_ids: list[str] = [],
        generation: int = 0,
        use_memory: bool = False,
        memory_config: Optional[dict] = None,
    ):
        """Initialize a new agent with given parameters."""
        # Add default actions
        self.actions = action_set if action_set else action_registry.get_all()

        # Normalize weights
        total_weight = sum(action.weight for action in self.actions)
        for action in self.actions:
            action.weight /= total_weight

        self.agent_id = agent_id
        self.position = position
        self.resource_level = resource_level
        self.alive = True
        self.environment = environment
        self.config = environment.config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.previous_state: AgentState | None = None
        self.previous_action = None
        self.max_movement = (
            self.config.max_movement if self.config else 8
        )  # Default value
        self.total_reward = 0.0
        self.episode_rewards = []
        self.losses = []
        self.starvation_threshold = (
            self.config.starvation_threshold if self.config else 10
        )
        self.max_starvation = self.config.max_starvation_time if self.config else 100
        self.birth_time = environment.time

        # Initialize health tracking first
        self.starting_health = self.config.starting_health if self.config else 100
        self.current_health = self.starting_health
        self.is_defending = False

        # Generate genome info
        self.generation = generation
        self.genome_id = self._generate_genome_id(parent_ids)

        # Initialize all modules first
        #! make this a list of action modules that can be provided to the agent at init
        # Use the config's dqn_hidden_size to match the modules
        hidden_size = self.config.dqn_hidden_size if self.config else 64
        self.shared_encoder = SharedEncoder(
            input_dim=8, hidden_size=hidden_size
        )  # Use 8-dimensional input to match selection state
        self.move_module = MoveModule(
            self.config if self.config else DEFAULT_MOVE_CONFIG,
            db=environment.db,
            shared_encoder=self.shared_encoder,
        )
        self.attack_module = AttackModule(
            self.config if self.config else DEFAULT_ATTACK_CONFIG,
            shared_encoder=self.shared_encoder,
        )
        self.share_module = ShareModule(
            self.config if self.config else DEFAULT_SHARE_CONFIG,
            shared_encoder=self.shared_encoder,
        )
        self.gather_module = GatherModule(
            self.config if self.config else DEFAULT_GATHER_CONFIG,
            shared_encoder=self.shared_encoder,
        )
        self.reproduce_module = ReproduceModule(
            self.config if self.config else DEFAULT_REPRODUCE_CONFIG,
            shared_encoder=self.shared_encoder,
        )
        #! change to ChoiceModule
        self.select_module = SelectModule(
            num_actions=len(self.actions),
            config=SelectConfig(),
            device=self.device,
            shared_encoder=self.shared_encoder,
        )

        # Initialize Redis memory if requested
        self.memory = None
        if use_memory:
            self._init_memory(memory_config)

        # self.environment.batch_add_agents([self])
        self.environment.record_birth()

        #! part of context manager, commented out for now
        # Context management
        # self._active = False  # Track if agent is in context
        # self._parent_context = None  # Track parent agent context
        # self._child_contexts = set()  # Track child agent contexts
        # self._context_depth = 0  # Track nesting level

        # # Context-specific logging
        # self._context_logger = logging.getLogger(f"agent.{agent_id}.context")

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
            creation_time=self.environment.time,
        )
        return genome_id.to_string()

    def get_perception(self) -> PerceptionData:
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
        radius = self.config.perception_radius if self.config else 5

        # Create perception grid centered on agent
        size = 2 * radius + 1
        perception = np.zeros((size, size), dtype=np.int8)

        # Get nearby entities using environment's spatial indexing
        nearby_resources = self.environment.get_nearby_resources(self.position, radius)
        nearby_agents = self.environment.get_nearby_agents(self.position, radius)

        # Helper function to convert world coordinates to grid coordinates
        def world_to_grid(wx: float, wy: float) -> tuple[int, int]:
            # Convert world position to grid position relative to agent
            gx = int(round(wx - self.position[0] + radius))
            gy = int(round(wy - self.position[1] + radius))
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
                if not self.environment.is_valid_position((world_x, world_y)):
                    perception[i, j] = 3

        return PerceptionData(perception)

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
            step_number=self.environment.time,
            position_x=self.position[0],
            position_y=self.position[1],
            position_z=self.position[2] if len(self.position) > 2 else 0,
            resource_level=self.resource_level,
            current_health=self.current_health,
            is_defending=self.is_defending,
            total_reward=self.total_reward,
            age=self.environment.time - self.birth_time,
        )

    def decide_action(self):
        """Select an action using the SelectModule's intelligent decision making.

        The selection process involves:
        1. Getting current state representation
        2. Passing state through SelectModule's neural network
        3. Combining learned preferences with predefined action weights
        4. Choosing optimal action based on current circumstances

        Returns:
            Action: Selected action object to execute
        """
        # Cache state tensor to avoid recreating it multiple times
        if not hasattr(
            self, "_cached_selection_state"
        ) or self.environment.time != getattr(self, "_cached_selection_time", -1):
            self._cached_selection_state = create_selection_state(self)
            self._cached_selection_time = self.environment.time

        current_step = self.environment.time
        enabled_actions = self.actions  # Default all
        if self.config and hasattr(self.config, "curriculum_phases"):
            for phase in self.config.curriculum_phases:
                if current_step < phase["steps"] or phase["steps"] == -1:
                    enabled_actions = [
                        a for a in self.actions if a.name in phase["enabled_actions"]
                    ]
                    break
        selected_action = self.select_module.select_action(
            agent=self, actions=enabled_actions, state=self._cached_selection_state
        )
        return selected_action

    def check_starvation(self) -> bool:
        """Check and handle agent starvation state.

        Manages the agent's starvation threshold based on resource levels:
        - Increments threshold when resources are depleted
        - Resets threshold when resources are available
        - Triggers death if threshold exceeds maximum starvation time

        Returns:
            bool: True if agent died from starvation, False otherwise
        """
        if self.resource_level <= 0:
            self.starvation_threshold += 1
            if self.starvation_threshold >= self.max_starvation:
                self.die()
                return True
        else:
            self.starvation_threshold = 0
        return False

    def act(self) -> None:
        """Execute the agent's turn in the simulation.

        This method handles the core action loop including:
        1. Resource consumption and starvation checks
        2. State observation
        3. Action selection and execution
        4. State/action memory for learning

        The agent will not act if it's not alive. Each turn consumes base resources
        and can potentially lead to death if resources are depleted.
        """
        if not self.alive:
            return

        # Reset defense status at start of turn
        self.is_defending = False

        # Resource consumption
        self.resource_level -= self.config.base_consumption_rate if self.config else 1

        # Check starvation state - exit early if agent dies
        if self.check_starvation():
            return

        # Get current state before action
        current_state = self.get_state()

        # Select and execute action
        action = self.decide_action()
        action.execute(self)

        # Store state for learning
        self.previous_state = current_state
        self.previous_action = action

        # Train all modules
        self.train_all_modules()

    def clone(self) -> "BaseAgent":
        """Create a mutated copy of this agent.

        Creates a new agent by:
        1. Cloning the current agent's genome
        2. Applying random mutations with 10% probability
        3. Converting mutated genome back to agent instance

        Returns:
            BaseAgent: A new agent with slightly modified characteristics
        """
        cloned_genome = Genome.clone(self.to_genome())
        mutated_genome = Genome.mutate(cloned_genome, mutation_rate=0.1)
        return Genome.to_agent(
            mutated_genome,
            self.agent_id,
            (int(self.position[0]), int(self.position[1])),
            self.environment,
        )

    def reproduce(self) -> bool:
        """Attempt reproduction. Returns True if successful."""
        # Store initial resources for tracking
        initial_resources = self.resource_level
        failure_reason = None

        # Check resource requirements
        if (
            self.resource_level < self.config.min_reproduction_resources
            if self.config
            else 10
        ):
            failure_reason = "Insufficient resources"

            # Record failed reproduction attempt
            if self.environment.db:
                self.environment.db.log_reproduction_event(
                    step_number=self.environment.time,
                    parent_id=self.agent_id,
                    offspring_id="",  # Empty string for failed reproduction
                    success=False,
                    parent_resources_before=initial_resources,
                    parent_resources_after=initial_resources,
                    offspring_initial_resources=0.0,  # Default value for failed reproduction
                    failure_reason=failure_reason,
                    parent_generation=self.generation,
                    offspring_generation=0,  # Default value for failed reproduction
                    parent_position=self.position,
                )
            return False

        # Check if enough resources for offspring cost
        if self.resource_level < self.config.offspring_cost + 2 if self.config else 10:
            failure_reason = "Insufficient resources for offspring cost"

            # Record failed reproduction attempt
            if self.environment.db:
                self.environment.db.log_reproduction_event(
                    step_number=self.environment.time,
                    parent_id=self.agent_id,
                    offspring_id="",  # Empty string for failed reproduction
                    success=False,
                    parent_resources_before=initial_resources,
                    parent_resources_after=initial_resources,
                    offspring_initial_resources=0.0,  # Default value for failed reproduction
                    failure_reason=failure_reason,
                    parent_generation=self.generation,
                    offspring_generation=0,  # Default value for failed reproduction
                    parent_position=self.position,
                )
            return False

        # Attempt reproduction
        new_agent = self.create_offspring()

        # Record successful reproduction
        if self.environment.db:
            self.environment.db.log_reproduction_event(
                step_number=self.environment.time,
                parent_id=self.agent_id,
                offspring_id=new_agent.agent_id,
                success=True,
                parent_resources_before=initial_resources,
                parent_resources_after=self.resource_level,
                offspring_initial_resources=(
                    self.config.offspring_initial_resources if self.config else 10
                ),
                failure_reason="",  # Empty string for successful reproduction
                parent_generation=self.generation,
                offspring_generation=new_agent.generation,
                parent_position=self.position,
            )

        logger.info(
            f"Agent {self.agent_id} reproduced at {self.position} during step {self.environment.time} creating agent {new_agent.agent_id}"
        )
        return True

    def create_offspring(self):
        """Create a new agent as offspring."""
        # Get the agent's class (IndependentAgent, SystemAgent, etc)
        agent_class = type(self)

        # Generate unique ID and genome info first
        #! need to update this since we are using strings now
        new_id = self.environment.get_next_agent_id()
        generation = self.generation + 1

        # Create new agent with all info
        new_agent = agent_class(
            agent_id=new_id,
            position=self.position,
            resource_level=(
                self.config.offspring_initial_resources if self.config else 10
            ),
            environment=self.environment,
            generation=generation,
        )

        # Add new agent to environment
        self.environment.add_agent(new_agent)

        # Subtract offspring cost from parent's resources
        self.resource_level -= self.config.offspring_cost if self.config else 5

        # Log creation
        logger.info(
            f"Agent {new_id} created at {self.position} during step {self.environment.time} of type {agent_class.__name__}"
        )

        return new_agent

    def die(self):
        """Handle agent death."""

        if self.alive:
            self.alive = False
            self.death_time = self.environment.time
            # Record the death in environment
            # Log death in database
            if self.environment.db:
                self.environment.db.update_agent_death(self.agent_id, self.death_time)

            logger.info(
                f"Agent {self.agent_id} died at {self.position} during step {self.environment.time}"
            )
            self.environment.remove_agent(self)

    def get_environment(self) -> "Environment":
        return self._environment

    def set_environment(self, environment: "Environment") -> None:
        self._environment = environment

    def calculate_new_position(self, action):
        """Calculate new position based on movement action.

        Takes into account environment boundaries and maximum movement distance.
        Movement is grid-based with four possible directions.

        Args:
            action (int): Movement direction index
                0: Right  (+x direction)
                1: Left   (-x direction)
                2: Up     (+y direction)
                3: Down   (-y direction)

        Returns:
            tuple[float, float]: New (x, y) position coordinates, bounded by environment limits
        """
        # Define movement vectors for each action
        action_vectors = {
            0: (1, 0),  # Right
            1: (-1, 0),  # Left
            2: (0, 1),  # Up
            3: (0, -1),  # Down
        }

        # Get movement vector for the action
        dx, dy = action_vectors[action]

        # Scale by max_movement
        dx *= self.config.max_movement if self.config else 1
        dy *= self.config.max_movement if self.config else 1

        # Calculate new position
        new_x = max(0, min(self.environment.width, self.position[0] + dx))
        new_y = max(0, min(self.environment.height, self.position[1] + dy))

        return (new_x, new_y)

    def calculate_move_reward(self, old_pos, new_pos):
        """Calculate reward for a movement action.

        Reward calculation considers:
        1. Base movement cost (-0.1)
        2. Distance to nearest resource before and after move
        3. Positive reward (0.3) for moving closer to resources
        4. Negative reward (-0.2) for moving away from resources

        Args:
            old_pos (tuple): Previous (x, y) position
            new_pos (tuple): New (x, y) position

        Returns:
            float: Movement reward value
        """
        # Base cost for moving
        reward = -0.1

        # Calculate movement distance
        distance_moved = np.sqrt(
            (new_pos[0] - old_pos[0]) ** 2 + (new_pos[1] - old_pos[1]) ** 2
        )

        if distance_moved > 0:
            # Find closest non-depleted resource
            closest_resource = min(
                [r for r in self.environment.resources if not r.is_depleted()],
                key=lambda r: np.sqrt(
                    (r.position[0] - new_pos[0]) ** 2
                    + (r.position[1] - new_pos[1]) ** 2
                ),
                default=None,
            )

            if closest_resource:
                # Calculate distances to resource before and after move
                old_distance = np.sqrt(
                    (closest_resource.position[0] - old_pos[0]) ** 2
                    + (closest_resource.position[1] - old_pos[1]) ** 2
                )
                new_distance = np.sqrt(
                    (closest_resource.position[0] - new_pos[0]) ** 2
                    + (closest_resource.position[1] - new_pos[1]) ** 2
                )

                # Reward for moving closer to resources, penalty for moving away
                reward += 0.3 if new_distance < old_distance else -0.2

        return reward

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
        """
        # Reduce damage if defending
        if self.is_defending:
            damage *= 0.5  # 50% damage reduction when defending

        # Apply damage
        self.current_health = max(0, self.current_health - damage)

        # Check for death
        if self.current_health <= 0:
            self.die()

        return damage

    def calculate_attack_reward(
        self, target: "BaseAgent", damage_dealt: float, action: int
    ) -> float:
        """Calculate reward for an attack action based on outcome.

        Reward components:
        - Base cost: Negative value from config.attack_base_cost
        - Successful hits: Positive reward scaled by damage ratio
        - Killing blows: Additional bonus from config.attack_kill_reward
        - Defensive actions: Positive when health below threshold, negative otherwise
        - Missed attacks: Penalty from config.attack_failure_penalty

        Args:
            target (BaseAgent): The agent that was attacked
            damage_dealt (float): Amount of damage successfully dealt
            action (int): The attack action that was taken (from AttackActionSpace)

        Returns:
            float: The calculated reward value, combining all applicable components
        """
        # Base reward starts with the attack cost
        reward = self.config.attack_base_cost if self.config else 1

        # Defensive action reward
        if action == AttackActionSpace.DEFEND:
            if (
                self.current_health
                < self.starting_health * self.config.attack_defense_threshold
                if self.config
                else 0.5
            ):
                reward += (
                    self.config.attack_success_reward if self.config else 1
                )  # Good decision to defend when health is low
            else:
                reward += self.config.attack_failure_penalty if self.config else -1
            return reward

        # Attack success reward
        if damage_dealt > 0:
            reward += (self.config.attack_success_reward if self.config else 1) * (
                damage_dealt / (self.config.attack_base_damage if self.config else 1)
            )
            if not target.alive:
                reward += self.config.attack_kill_reward if self.config else 10
        else:
            reward += self.config.attack_failure_penalty if self.config else -1

        return reward

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
        """Create a new agent instance from a genome.

        Factory method that:
        1. Decodes genome into agent parameters
        2. Initializes new agent with those parameters
        3. Sets up required environment connections

        Args:
            genome (Genome): Genetic encoding of agent parameters
            agent_id (str): Unique identifier for new agent
            position (tuple[int, int]): Starting coordinates
            environment (Environment): Simulation environment reference

        Returns:
            BaseAgent: New agent instance with genome's characteristics
        """
        return Genome.to_agent(
            genome, agent_id, (int(position[0]), int(position[1])), environment
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

    def calculate_attack_position(self, action: int) -> Tuple[float, float]:
        """Calculate the target position for an attack based on action.

        Args:
            action: Attack action index

        Returns:
            Tuple[float, float]: Target position coordinates
        """
        dx, dy = self.attack_module.action_space[action]
        return (
            self.position[0] + dx * self.config.attack_range if self.config else 10,
            self.position[1] + dy * self.config.attack_range if self.config else 10,
        )

    @property
    def attack_strength(self) -> float:
        """Calculate the agent's current attack strength."""
        return (self.config.base_attack_strength if self.config else 10) * (
            self.current_health / self.starting_health
        )

    @property
    def defense_strength(self) -> float:
        """Calculate the agent's current defense strength."""
        return (
            (self.config.base_defense_strength if self.config else 5)
            if self.is_defending
            else 0.0
        )

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
            logger.info(f"Initialized Redis memory for agent {self.agent_id}")

        except Exception as e:
            logger.error(
                f"Failed to initialize Redis memory for agent {self.agent_id}: {e}"
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
        """
        if not self.memory:
            return False

        try:
            # Create state representation
            current_state = AgentState(
                agent_id=self.agent_id,
                step_number=self.environment.time,
                position_x=self.position[0],
                position_y=self.position[1],
                position_z=self.position[2] if len(self.position) > 2 else 0,
                resource_level=self.resource_level,
                current_health=self.current_health,
                is_defending=self.is_defending,
                total_reward=self.total_reward,
                age=self.environment.time - self.birth_time,
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
                step=self.environment.time,
                state=current_state,
                action=action_name,
                reward=reward,
                perception=perception_data,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(
                f"Failed to remember experience for agent {self.agent_id}: {e}"
            )
            return False

    def recall_similar_situations(self, position=None, limit=5):
        """Retrieve memories similar to current situation.

        Args:
            position (tuple, optional): Position to search around, or current position if None
            limit (int): Maximum number of memories to retrieve

        Returns:
            list: List of similar memories, or empty list if memory not available
        """
        if not self.memory:
            return []

        try:
            # Use provided position or current position
            pos = position or self.position

            # Search memories by position
            return self.memory.search_by_position(pos, radius=10.0, limit=limit)

        except Exception as e:
            logger.error(f"Failed to recall memories for agent {self.agent_id}: {e}")
            return []

    def train_all_modules(self):
        modules = [
            self.move_module,
            self.attack_module,
            self.share_module,
            self.gather_module,
            self.reproduce_module,
            self.select_module,
        ]
        for module in modules:
            if len(module.memory) >= module.config.batch_size:
                batch = random.sample(module.memory, module.config.batch_size)
                module.train(batch)

    #! part of context manager, commented out for now
    # def __enter__(self):
    #     """Enter agent context.

    #     Activates agent in environment, initializes resources, and sets up context tracking.

    #     Raises:
    #         RuntimeError: If agent is already in an active context
    #     """
    #     if self._active:
    #         raise RuntimeError(f"Agent {self.agent_id} is already in an active context")

    #     self._active = True
    #     self._context_depth += 1

    #     # Register with environment's context tracker
    #     self.environment.register_active_context(self)

    #     # Add to environment and record birth
    #     self.environment.batch_add_agents([self])
    #     self.environment.record_birth()

    #     self._context_logger.info(
    #         f"Agent {self.agent_id} entered context (depth: {self._context_depth})"
    #     )

    #     return self

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     """Exit agent context.

    #     Ensures proper cleanup of agent resources and state. Also handles nested contexts
    #     and relationship cleanup.

    #     Args:
    #         exc_type: Type of exception that occurred, if any
    #         exc_val: Exception instance that occurred, if any
    #         exc_tb: Exception traceback, if any
    #     """
    #     try:
    #         # Clean up child contexts first
    #         for child in self._child_contexts.copy():
    #             if child._active:
    #                 child.__exit__(None, None, None)

    #         # Clean up agent state
    #         if self.alive:
    #             self.die()

    #         # Remove from environment's context tracker
    #         self.environment.unregister_active_context(self)

    #         # Clear relationship tracking
    #         if self._parent_context:
    #             self._parent_context._child_contexts.remove(self)
    #         self._parent_context = None
    #         self._child_contexts.clear()

    #         self._context_logger.info(
    #             f"Agent {self.agent_id} exited context (depth: {self._context_depth})"
    #         )

    #         if exc_type:
    #             self._context_logger.error(
    #                 f"Agent {self.agent_id} context exited with error: {exc_val}"
    #             )

    #     finally:
    #         self._active = False
    #         self._context_depth -= 1

    #     return False  # Don't suppress exceptions

    # def create_child_context(self, child_agent: "BaseAgent") -> None:
    #     """Create parent-child relationship between agent contexts.

    #     Args:
    #         child_agent: Agent to establish as child context

    #     Raises:
    #         RuntimeError: If either agent is not in an active context
    #     """
    #     if not self._active:
    #         raise RuntimeError("Parent agent must be in active context")
    #     if not child_agent._active:
    #         raise RuntimeError("Child agent must be in active context")

    #     child_agent._parent_context = self
    #     self._child_contexts.add(child_agent)

    #     self._context_logger.info(
    #         f"Established parent-child context: {self.agent_id} -> {child_agent.agent_id}"
    #     )

    # def validate_context(self) -> None:
    #     """Validate agent's context state.

    #     Raises:
    #         RuntimeError: If agent's context state is invalid
    #     """
    #     if not self._active:
    #         raise RuntimeError("Agent must be used within context manager")

    #     if self._context_depth <= 0:
    #         raise RuntimeError("Invalid context depth")

    #     # Validate parent-child relationships
    #     if self._parent_context and self not in self._parent_context._child_contexts:
    #         raise RuntimeError("Inconsistent parent-child relationship")

    #     for child in self._child_contexts:
    #         if child._parent_context is not self:
    #             raise RuntimeError("Inconsistent child-parent relationship")

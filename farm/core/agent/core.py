"""AgentCore - Minimal agent coordinating components and behavior.

Overview
--------
The AgentCore module provides the foundational agent class for AgentFarm multi-agent
simulations. It implements a component-based architecture where agents are composed
of pluggable components (movement, combat, resources, etc.) and behaviors (decision-making
logic), following SOLID design principles and enabling flexible agent composition.

Key Responsibilities
--------------------
- Agent identity management: unique IDs, alive status, and lifecycle tracking
- Component coordination: attaching, accessing, and managing pluggable components
- Behavior execution: delegating decision-making to behavior strategies
- State management: maintaining agent state through StateManager
- Service integration: dependency injection for spatial queries, time, metrics, etc.
- Metrics compatibility: exposing component state as properties for monitoring

Core Architecture
-----------------
- **Component-based Design**: Agents are composed of specialized components rather than
  inheriting from monolithic base classes. Each component handles one specific capability
  (movement, combat, resources, perception, reproduction).
- **Behavior Separation**: Decision-making logic is separated into behavior objects that
  can be swapped or extended without modifying the core agent.
- **Service-oriented Dependencies**: Agents receive services (spatial queries, time,
  metrics, validation) through dependency injection, enabling testability and flexibility.
- **State Management**: Centralized state management through StateManager for position,
  generation, birth time, and genome information.

Design Principles
-----------------
- **Single Responsibility Principle (SRP)**: AgentCore only coordinates components and
  behaviors; it doesn't implement specific agent logic.
- **Open-Closed Principle (OCP)**: New components and behaviors can be added without
  modifying existing code.
- **Liskov Substitution Principle (LSP)**: All components implement IAgentComponent
  interface and can be substituted.
- **Interface Segregation Principle (ISP)**: Components have focused interfaces for
  their specific responsibilities.
- **Dependency Inversion Principle (DIP)**: AgentCore depends on abstractions (interfaces)
  rather than concrete implementations.

Component System
----------------
Components provide specific agent capabilities:
- **MovementComponent**: Handles agent locomotion and position validation
- **CombatComponent**: Manages health, attacks, defense, and combat mechanics
- **ResourceComponent**: Tracks resources, consumption, and starvation
- **PerceptionComponent**: Provides spatial awareness and observation capabilities
- **ReproductionComponent**: Handles agent reproduction and offspring creation

Behavior System
---------------
Behaviors implement decision-making logic:
- **IAgentBehavior**: Base interface for all agent behaviors
- **Behavior Strategies**: Can be swapped at runtime for different decision-making approaches
- **Action Selection**: Behaviors determine which actions agents take based on observations

Service Integration
-------------------
Agents receive services through dependency injection:
- **ISpatialQueryService**: Spatial queries for nearby agents and resources
- **ITimeService**: Access to simulation time and timestamps
- **IAgentLifecycleService**: Agent creation, removal, and lifecycle management
- **IMetricsService**: Metrics recording and performance tracking
- **IValidationService**: Position and action validation
- **ILoggingService**: Event logging and debugging

Metrics Compatibility
---------------------
AgentCore exposes component state as properties to maintain compatibility with
existing metrics tracking systems:
- **resource_level**: Current resource amount from ResourceComponent
- **current_health**: Current health from CombatComponent
- **starting_health**: Maximum health from CombatComponent
- **generation**: Generation number from StateManager
- **birth_time**: Birth timestamp from StateManager
- **genome_id**: Genome identifier from StateManager
- **total_reward**: Accumulated reward (TODO: implement)

Usage Example
-------------
```python
# Create agent with components
agent = AgentCore(agent_id="agent_001", config=agent_config)

# Add components
agent.add_component(MovementComponent(movement_config))
agent.add_component(ResourceComponent(100, resource_config))
agent.add_component(CombatComponent(combat_config))

# Set behavior
agent.set_behavior(MyBehavior())

# Access component state
health = agent.current_health  # Delegates to CombatComponent
resources = agent.resource_level  # Delegates to ResourceComponent
```

Notes
-----
- Agents are created through AgentFactory for proper dependency injection
- Component lifecycle is managed automatically (attach/detach)
- State is persisted through StateManager for serialization and analysis
- All agent interactions go through the component system for modularity
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from farm.core.agent.behaviors.base_behavior import IAgentBehavior
from farm.core.agent.components.base import IAgentComponent
from farm.core.agent.state.state_manager import StateManager

if TYPE_CHECKING:
    from farm.core.services.interfaces import (
        IAgentLifecycleService,
        ISpatialQueryService,
        ITimeService,
    )


from farm.utils.logging import get_logger

logger = get_logger(__name__)


class AgentCore:
    """
    Minimal agent core that coordinates components and executes behavior.

    Responsibilities (Single Responsibility Principle):
    - Maintain agent identity (ID, alive status)
    - Coordinate component lifecycle
    - Execute behavior strategy
    - Provide component access
    - Expose metrics-compatible properties for monitoring

    This class follows:
    - SRP: Only coordinates, doesn't implement logic
    - OCP: Add components/behaviors without modification
    - DIP: Depends on abstractions (interfaces)
    - Composition: Components composed, not inherited

    The actual agent capabilities (movement, combat, etc.) are implemented
    in components. The decision-making logic is implemented in behaviors.
    AgentCore simply coordinates these parts and provides a unified interface
    for accessing component state (e.g., resource_level, health, etc.).

    Metrics Compatibility:
    The class provides properties that delegate to component state to maintain
    compatibility with existing metrics tracking systems:
    - resource_level: Current resource amount from ResourceComponent
    - current_health: Current health from CombatComponent
    - starting_health: Maximum health from CombatComponent
    - generation: Generation number from StateManager
    - birth_time: Birth timestamp from StateManager
    - genome_id: Genome identifier from StateManager
    - total_reward: Accumulated reward (TODO: implement)
    """

    def __init__(
        self,
        agent_id: str,
        position: tuple[float, float],
        spatial_service: "ISpatialQueryService",
        behavior: IAgentBehavior,
        components: Optional[List[IAgentComponent]] = None,
        time_service: Optional["ITimeService"] = None,
        lifecycle_service: Optional["IAgentLifecycleService"] = None,
    ):
        """
        Initialize agent core with identity and pluggable components.

        Args:
            agent_id: Unique identifier for this agent
            position: Initial (x, y) position
            spatial_service: Service for spatial queries (required)
            behavior: Behavior strategy for decision-making (required)
            components: List of components to attach (optional)
            time_service: Service for accessing simulation time (optional)
            lifecycle_service: Service for agent lifecycle management (optional)

        Example:
            >>> agent = AgentCore(
            ...     agent_id="agent_001",
            ...     position=(10.0, 20.0),
            ...     spatial_service=spatial_service,
            ...     behavior=LearningAgentBehavior(...),
            ...     components=[
            ...         MovementComponent(config.movement),
            ...         ResourceComponent(100, config.resource),
            ...         CombatComponent(config.combat),
            ...     ]
            ... )
        """
        # Essential identity
        self.agent_id = agent_id
        self.alive = True

        # Core services (injected dependencies)
        self._spatial_service = spatial_service
        self._behavior = behavior
        self._time_service = time_service
        self._lifecycle_service = lifecycle_service

        # Component registry (composition)
        self._components: Dict[str, IAgentComponent] = {}
        if components:
            for component in components:
                self.add_component(component)

        # State management (single responsibility)
        self.state_manager = StateManager(self)
        self.state_manager.set_position(position)

        # Set birth time if time service available
        if self._time_service:
            self.state_manager.set_birth_time(self._time_service.current_time())

    def add_component(self, component: IAgentComponent) -> None:
        """
        Add a component to this agent (Open-Closed Principle).

        Components are attached and can access the agent through their
        _agent reference.

        Args:
            component: Component to add

        Example:
            >>> agent.add_component(PerceptionComponent(spatial_service, config))
        """
        component.attach(self)
        self._components[component.name] = component

    def get_component(self, name: str) -> Optional[IAgentComponent]:
        """
        Get a component by name.

        Args:
            name: Component name (e.g., "movement", "resource", "combat")

        Returns:
            Component instance or None if not found

        Example:
            >>> movement = agent.get_component("movement")
            >>> if movement:
            ...     movement.move_to((100, 100))
        """
        return self._components.get(name)

    def has_component(self, name: str) -> bool:
        """
        Check if agent has a specific component.

        Args:
            name: Component name to check

        Returns:
            bool: True if component exists

        Example:
            >>> if agent.has_component("combat"):
            ...     combat = agent.get_component("combat")
            ...     combat.attack(target)
        """
        return name in self._components

    def remove_component(self, name: str) -> Optional[IAgentComponent]:
        """
        Remove a component from the agent.

        Args:
            name: Component name to remove

        Returns:
            Removed component or None if not found

        Example:
            >>> old_combat = agent.remove_component("combat")
        """
        return self._components.pop(name, None)

    def act(self) -> None:
        """
        Execute one simulation step using the behavior strategy.

        This method coordinates the agent's turn:
        1. Pre-step: Notify all components (on_step_start)
        2. Execute: Run behavior strategy
        3. Post-step: Notify all components (on_step_end)

        Components can use on_step_start/on_step_end for per-turn logic
        (e.g., resource consumption, defense timer countdown).

        The behavior strategy determines what actions the agent takes.

        Example:
            >>> agent.act()  # Agent decides and acts for one turn
        """
        if not self.alive:
            return

        # Pre-step: notify all components
        for component in self._components.values():
            component.on_step_start()

        # Execute behavior strategy
        try:
            self._behavior.execute_turn(self)
        except Exception as e:
            logger.error(f"Agent {self.agent_id} behavior execution failed: {e}", exc_info=True)

        # Post-step: notify all components
        for component in self._components.values():
            component.on_step_end()

    def terminate(self) -> None:
        """
        Handle agent death and cleanup.

        This method:
        1. Marks agent as not alive
        2. Records death time
        3. Notifies all components (on_terminate)
        4. Removes agent from lifecycle service

        Example:
            >>> agent.terminate()  # Agent dies
        """
        if not self.alive:
            return

        self.alive = False

        # Record death time
        if self._time_service:
            self.state_manager.set_death_time(self._time_service.current_time())

        # Notify all components of termination
        for component in self._components.values():
            try:
                component.on_terminate()
            except Exception as e:
                logger.warning(f"Component {component.name} termination failed for agent {self.agent_id}: {e}")

        # Remove from lifecycle service
        if self._lifecycle_service:
            try:
                self._lifecycle_service.remove_agent(self)
            except Exception as e:
                logger.error(f"Failed to remove agent {self.agent_id} from lifecycle service: {e}")

        logger.info(f"Agent {self.agent_id} terminated at step {self.state_manager.death_time}")

    @property
    def position(self) -> tuple[float, float]:
        """
        Get current 2D position.

        Returns:
            tuple: (x, y) position

        Example:
            >>> x, y = agent.position
        """
        return self.state_manager.position

    @property
    def position_3d(self) -> tuple[float, float, float]:
        """
        Get current 3D position.

        Returns:
            tuple: (x, y, z) position

        Example:
            >>> x, y, z = agent.position_3d
        """
        return self.state_manager.position_3d

    def get_state_dict(self) -> dict:
        """
        Get complete agent state for serialization.

        Returns:
            dict: Complete state including:
                - agent_id
                - alive status
                - state_manager state
                - all component states
                - behavior state

        Example:
            >>> state = agent.get_state_dict()
            >>> # Save to file or database
        """
        return {
            "agent_id": self.agent_id,
            "alive": self.alive,
            "state_manager": self.state_manager.get_state_dict(),
            "components": {name: component.get_state() for name, component in self._components.items()},
            "behavior": self._behavior.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Load agent state from dictionary.

        Args:
            state: State dictionary from get_state_dict()

        Example:
            >>> agent.load_state_dict(saved_state)
        """
        self.agent_id = state.get("agent_id", self.agent_id)
        self.alive = state.get("alive", True)

        # Load state manager
        if "state_manager" in state:
            self.state_manager.load_state_dict(state["state_manager"])

        # Load component states
        if "components" in state:
            for name, component_state in state["components"].items():
                component = self._components.get(name)
                if component:
                    component.load_state(component_state)

        # Load behavior state
        if "behavior" in state:
            self._behavior.load_state(state["behavior"])

    def get_action_weights(self) -> dict:
        """
        Get action weights for this agent.

        Returns a dictionary mapping action names to their weights.
        This is used for logging and analysis purposes.

        Returns:
            dict: Action name to weight mapping

        Example:
            >>> weights = agent.get_action_weights()
            >>> print(f"Move weight: {weights.get('move', 0)}")
        """
        # For now, return empty dict as action weights are typically
        # managed by the behavior or decision modules
        # This can be extended to query the behavior for actual weights
        # TODO: Implement this
        return {}

    @property
    def resource_level(self) -> float:
        """
        Get current resource level from resource component.

        Returns:
            float: Current resource level, or 0.0 if no resource component
        """
        resource_component = self.get_component("resource")
        if resource_component:
            return float(resource_component.level)
        return 0.0

    @property
    def current_health(self) -> float:
        """
        Get current health from combat component.

        Returns:
            float: Current health, or 0.0 if no combat component
        """
        combat_component = self.get_component("combat")
        if combat_component:
            return combat_component.health
        return 0.0

    @property
    def starting_health(self) -> float:
        """
        Get starting/max health from combat component.

        Returns:
            float: Starting health, or 0.0 if no combat component
        """
        combat_component = self.get_component("combat")
        if combat_component:
            return combat_component.max_health
        return 0.0

    @property
    def generation(self) -> int:
        """
        Get generation from state manager.

        Returns:
            int: Generation number
        """
        return self.state_manager.generation

    @property
    def birth_time(self) -> int:
        """
        Get birth time from state manager.

        Returns:
            int: Birth time step
        """
        return self.state_manager.birth_time

    @property
    def total_reward(self) -> float:
        """
        Get total reward accumulated by this agent.

        Returns:
            float: Total reward, defaults to 0.0 if not tracked
        """
        # For now, return 0.0 as reward tracking is not implemented in the component system
        # This can be extended to track rewards through a reward component or behavior
        # TODO: Implement this
        return 0.0

    @property
    def genome_id(self) -> str:
        """
        Get genome ID from state manager.

        Returns:
            str: Genome ID
        """
        return self.state_manager.genome_id

    @property
    def starvation_counter(self) -> int:
        """
        Get starvation counter from resource component.

        Returns:
            int: Number of consecutive steps with zero resources, or 0 if no resource component
        """
        resource_component = self.get_component("resource")
        if resource_component:
            return resource_component.starvation_steps
        return 0

    @property
    def starvation_threshold(self) -> int:
        """
        Get starvation threshold from resource component config.

        Returns:
            int: Number of steps without resources before death, or 100 if no resource component
        """
        resource_component = self.get_component("resource")
        if resource_component:
            return resource_component._config.starvation_threshold
        return 100  # Default value

    @property
    def is_defending(self) -> bool:
        """
        Get defending status from combat component.

        Returns:
            bool: Whether agent is currently defending, or False if no combat component
        """
        combat_component = self.get_component("combat")
        if combat_component:
            return combat_component.is_defending
        return False

    def __repr__(self) -> str:
        """String representation for debugging."""
        components_list = ", ".join(self._components.keys())
        return (
            f"AgentCore(id='{self.agent_id}', "
            f"alive={self.alive}, "
            f"position={self.position}, "
            f"components=[{components_list}])"
        )

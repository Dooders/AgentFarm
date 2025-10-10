"""AgentFactory - Factory for creating agents with proper dependency injection.

Overview
--------
The AgentFactory module provides a comprehensive factory system for creating agents
in AgentFarm multi-agent simulations. It implements the Factory Pattern with proper
dependency injection, ensuring agents are created with all required services and
components configured correctly.

The factory follows SOLID design principles and provides a clean, type-safe API for
agent construction. It handles the complex process of wiring together components,
behaviors, and services while maintaining separation of concerns and enabling
flexible agent composition.

Key Responsibilities
--------------------
- Agent construction: Creates properly configured AgentCore instances
- Dependency injection: Injects required services (spatial, time, lifecycle, etc.)
- Component assembly: Creates and configures agent components based on configuration
- Behavior selection: Assigns appropriate behavior strategies to agents
- Configuration management: Applies default and custom configurations
- Type-specific parameters: Supports different agent types with specialized parameters

Factory Methods
---------------
The factory provides several specialized methods for different agent creation scenarios:

- **create_agent()**: Custom agent with specified behavior and components
- **create_default_agent()**: Standard agent with default (random) behavior
- **create_learning_agent()**: Agent with reinforcement learning behavior
- **create_minimal_agent()**: Minimal agent for testing specific components

Design Principles
-----------------
- **Single Responsibility Principle (SRP)**: Factory only handles agent construction
- **Dependency Inversion Principle (DIP)**: Depends on interfaces, not concrete classes
- **Builder Pattern**: Provides fluent API for configuration
- **Open-Closed Principle (OCP)**: Extensible for new agent types and components

Service Integration
-------------------
The factory injects essential services into created agents:
- **ISpatialQueryService**: Required for spatial queries and movement
- **ITimeService**: Optional service for simulation time access
- **IAgentLifecycleService**: Optional service for agent lifecycle management
- **IMetricsService**: Optional service for metrics recording
- **IValidationService**: Optional service for position validation
- **ILoggingService**: Optional service for event logging

Component System
----------------
The factory creates and configures standard agent components:
- **MovementComponent**: Agent locomotion and position validation
- **ResourceComponent**: Resource tracking and consumption
- **CombatComponent**: Health, attacks, and combat mechanics
- **PerceptionComponent**: Spatial awareness and observation
- **ReproductionComponent**: Agent reproduction and offspring creation

Behavior System
---------------
The factory supports different behavior strategies:
- **DefaultAgentBehavior**: Random action selection for baseline behavior
- **LearningAgentBehavior**: Reinforcement learning for intelligent decision-making
- **Custom Behaviors**: User-defined behaviors implementing IAgentBehavior

Agent Types
-----------
The factory supports different agent types through the agent_type parameter:
- **SystemAgent**: Cooperative agents with system-level behavior
- **IndependentAgent**: Self-interested agents with individual behavior
- **ControlAgent**: Agents with control or leadership capabilities

Note: Currently, agent_type is accepted for compatibility but doesn't yet modify
behavior. Future versions will apply type-specific parameters from the
agent_parameters configuration.

Configuration Management
------------------------
The factory handles configuration through:
- **Default Configuration**: Applied when no custom config is provided
- **Agent-Specific Configuration**: Custom configuration per agent
- **Type-Specific Parameters**: Parameters that vary by agent type
- **Component Configuration**: Individual component settings

Usage Examples
--------------
```python
# Create factory with services
factory = AgentFactory(
    spatial_service=spatial_service,
    default_config=AgentConfig(),
    time_service=time_service,
    lifecycle_service=lifecycle_service
)

# Create default agent
agent = factory.create_default_agent(
    agent_id="agent_001",
    position=(50.0, 50.0),
    initial_resources=100
)

# Create learning agent
learning_agent = factory.create_learning_agent(
    agent_id="learner_001",
    position=(25.0, 75.0),
    decision_module=decision_module
)

# Create custom agent
custom_agent = factory.create_agent(
    agent_id="custom_001",
    position=(10.0, 20.0),
    behavior=MyCustomBehavior(),
    components=[MovementComponent(config.movement)]
)
```

Notes
-----
- All agents are created with proper dependency injection
- Component lifecycle is managed automatically
- Services are injected based on availability and requirements
- Configuration inheritance follows a clear hierarchy
- Agent types enable future specialization and parameterization
"""

from typing import TYPE_CHECKING, Dict, List, Optional

from farm.core.agent.behaviors import (
    DefaultAgentBehavior,
    IAgentBehavior,
    LearningAgentBehavior,
)
from farm.core.agent.components import (
    CombatComponent,
    IAgentComponent,
    MovementComponent,
    PerceptionComponent,
    ReproductionComponent,
    ResourceComponent,
)
from farm.core.agent.config.agent_config import AgentConfig
from farm.core.agent.core import AgentCore

if TYPE_CHECKING:
    from farm.core.decision.decision import DecisionModule
    from farm.core.services.interfaces import (
        IAgentLifecycleService,
        ISpatialQueryService,
        ITimeService,
    )


from farm.utils.logging import get_logger

logger = get_logger(__name__)


class AgentFactory:
    """
    Factory for creating agents with proper dependency injection.

    This factory follows:
    - SRP: Only responsible for agent construction
    - DIP: Depends on interfaces, not concrete classes
    - Builder Pattern: Provides fluent API for configuration

    The factory handles:
    - Component creation and configuration
    - Behavior strategy selection
    - Service injection
    - Default configuration
    - Agent type-specific parameter application (future enhancement)

    Agent Types:
    The factory supports different agent types (SystemAgent, IndependentAgent,
    ControlAgent) through the agent_type parameter. While currently the agent_type
    is accepted for compatibility, future versions will apply type-specific
    parameters from the agent_parameters configuration.
    """

    def __init__(
        self,
        spatial_service: "ISpatialQueryService",
        default_config: Optional[AgentConfig] = None,
        time_service: Optional["ITimeService"] = None,
        lifecycle_service: Optional["IAgentLifecycleService"] = None,
        agent_parameters: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Initialize agent factory with required services.

        Args:
            spatial_service: Service for spatial queries (required)
            default_config: Default agent configuration
            time_service: Service for accessing simulation time (optional)
            lifecycle_service: Service for agent lifecycle management (optional)
            agent_parameters: Agent type-specific parameters (optional)
                Dictionary mapping agent types to their specific parameters.
                Format: {"SystemAgent": {...}, "IndependentAgent": {...}, ...}

        Example:
            >>> factory = AgentFactory(
            ...     spatial_service=spatial_service,
            ...     default_config=AgentConfig(),
            ...     time_service=time_service,
            ...     lifecycle_service=lifecycle_service,
            ...     agent_parameters={
            ...         "SystemAgent": {"gather_efficiency_multiplier": 0.4},
            ...         "IndependentAgent": {"gather_efficiency_multiplier": 0.7}
            ...     }
            ... )
        """
        self._spatial_service = spatial_service
        self._default_config = default_config or AgentConfig()
        self._time_service = time_service
        self._lifecycle_service = lifecycle_service
        self._agent_parameters = agent_parameters or {}

    def create_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        behavior: IAgentBehavior,
        components: Optional[List[IAgentComponent]] = None,
        config: Optional[AgentConfig] = None,
    ) -> AgentCore:
        """
        Create a custom agent with specified behavior and components.

        Args:
            agent_id: Unique identifier
            position: Starting (x, y) position
            behavior: Behavior strategy to use
            components: List of components to attach (uses defaults if None)
            config: Agent configuration (uses default if None)

        Returns:
            AgentCore: Configured agent instance

        Example:
            >>> agent = factory.create_agent(
            ...     agent_id="custom_001",
            ...     position=(10.0, 20.0),
            ...     behavior=MyCustomBehavior(),
            ...     components=[
            ...         MovementComponent(config.movement),
            ...         CombatComponent(config.combat),
            ...     ]
            ... )
        """
        config = config or self._default_config

        # Use provided components or create defaults
        if components is None:
            components = self._create_default_components(config)

        # Create agent core
        agent = AgentCore(
            agent_id=agent_id,
            position=position,
            spatial_service=self._spatial_service,
            behavior=behavior,
            components=components,
            time_service=self._time_service,
            lifecycle_service=self._lifecycle_service,
        )

        return agent

    def create_default_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        initial_resources: int = 100,
        config: Optional[AgentConfig] = None,
        agent_type: Optional[str] = None,
    ) -> AgentCore:
        """
        Create an agent with default (random) behavior.

        Useful for:
        - Testing
        - Baseline comparisons
        - Simple simulations

        Args:
            agent_id: Unique identifier
            position: Starting (x, y) position
            initial_resources: Starting resource level
            config: Agent configuration (uses default if None)
            agent_type: Type of agent (SystemAgent, IndependentAgent, ControlAgent)
                Currently accepted for compatibility but not yet used to modify
                behavior. Future versions will apply type-specific parameters.

        Returns:
            AgentCore: Agent with DefaultAgentBehavior and standard components

        Example:
            >>> agent = factory.create_default_agent(
            ...     agent_id="agent_001",
            ...     position=(50.0, 50.0),
            ...     initial_resources=100,
            ...     agent_type="SystemAgent"
            ... )

        Note:
            The agent_type parameter is currently accepted for backward compatibility
            but does not yet modify agent behavior. Future versions will apply
            type-specific parameters from the agent_parameters configuration.
        """
        config = config or self._default_config

        # TODO: Apply agent_type specific parameters from self._agent_parameters
        # For now, we accept the agent_type parameter but don't use it to modify behavior
        # This maintains backward compatibility while allowing future enhancements

        # Create components
        components = [
            MovementComponent(config.movement),
            ResourceComponent(initial_resources, config.resource),
            CombatComponent(config.combat),
            PerceptionComponent(self._spatial_service, config.perception),
            ReproductionComponent(
                config.reproduction,
                lifecycle_service=self._lifecycle_service,
                offspring_factory=self._create_offspring_factory(config),
            ),
        ]

        # Create behavior
        behavior = DefaultAgentBehavior()

        # Create agent
        agent = AgentCore(
            agent_id=agent_id,
            position=position,
            spatial_service=self._spatial_service,
            behavior=behavior,
            components=components,
            time_service=self._time_service,
            lifecycle_service=self._lifecycle_service,
        )

        return agent

    def create_learning_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        initial_resources: int = 100,
        config: Optional[AgentConfig] = None,
        decision_module: Optional["DecisionModule"] = None,
    ) -> AgentCore:
        """
        Create an agent with learning behavior.

        Uses reinforcement learning for intelligent decision-making.

        Args:
            agent_id: Unique identifier
            position: Starting (x, y) position
            initial_resources: Starting resource level
            config: Agent configuration (uses default if None)
            decision_module: DecisionModule for RL (optional, can be set later)

        Returns:
            AgentCore: Agent with LearningAgentBehavior

        Example:
            >>> agent = factory.create_learning_agent(
            ...     agent_id="learner_001",
            ...     position=(25.0, 75.0),
            ...     initial_resources=100,
            ...     decision_module=decision_module
            ... )
        """
        config = config or self._default_config

        # Create components
        components = [
            MovementComponent(config.movement),
            ResourceComponent(initial_resources, config.resource),
            CombatComponent(config.combat),
            PerceptionComponent(self._spatial_service, config.perception),
            ReproductionComponent(
                config.reproduction,
                lifecycle_service=self._lifecycle_service,
                offspring_factory=self._create_offspring_factory(config),
            ),
        ]

        # Create behavior
        behavior = LearningAgentBehavior(decision_module=decision_module)

        # Create agent
        agent = AgentCore(
            agent_id=agent_id,
            position=position,
            spatial_service=self._spatial_service,
            behavior=behavior,
            components=components,
            time_service=self._time_service,
            lifecycle_service=self._lifecycle_service,
        )

        return agent

    def create_minimal_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        components: Optional[List[IAgentComponent]] = None,
    ) -> AgentCore:
        """
        Create a minimal agent with only specified components.

        Useful for testing specific component interactions.

        Args:
            agent_id: Unique identifier
            position: Starting (x, y) position
            components: Components to attach (empty list creates bare agent)

        Returns:
            AgentCore: Minimal agent

        Example:
            >>> # Agent with only movement
            >>> agent = factory.create_minimal_agent(
            ...     agent_id="minimal_001",
            ...     position=(0.0, 0.0),
            ...     components=[MovementComponent(MovementConfig())]
            ... )
        """
        components = components if components is not None else []
        behavior = DefaultAgentBehavior()

        agent = AgentCore(
            agent_id=agent_id,
            position=position,
            spatial_service=self._spatial_service,
            behavior=behavior,
            components=components,
            time_service=self._time_service,
            lifecycle_service=self._lifecycle_service,
        )

        return agent

    def _create_default_components(self, config: AgentConfig) -> List[IAgentComponent]:
        """
        Create default component set.

        Args:
            config: Agent configuration

        Returns:
            List of default components
        """
        return [
            MovementComponent(config.movement),
            ResourceComponent(config.resource.initial_resources, config.resource),
            CombatComponent(config.combat),
            PerceptionComponent(self._spatial_service, config.perception),
            ReproductionComponent(
                config.reproduction,
                lifecycle_service=self._lifecycle_service,
            ),
        ]

    def _create_offspring_factory(self, config: AgentConfig):
        """
        Create factory function for offspring creation.

        Args:
            config: Configuration to use for offspring

        Returns:
            Callable that creates offspring agents
        """

        def offspring_factory(
            agent_id: str,
            position: tuple[float, float],
            initial_resources: int,
            parent_ids: List[str],
            generation: int,
        ) -> AgentCore:
            """Create offspring agent."""
            # Create offspring with same behavior type as parent
            agent = self.create_default_agent(
                agent_id=agent_id,
                position=position,
                initial_resources=initial_resources,
                config=config,
            )

            # Set genealogy
            agent.state_manager.set_generation(generation)
            agent.state_manager.set_parent_ids(parent_ids)

            return agent

        return offspring_factory

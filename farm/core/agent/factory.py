"""
AgentFactory for creating agents with proper dependency injection.

Provides clean, type-safe construction of agents with all dependencies.
"""

from typing import Optional, List, TYPE_CHECKING
from farm.core.agent.core import AgentCore
from farm.core.agent.components import (
    IAgentComponent,
    MovementComponent,
    ResourceComponent,
    CombatComponent,
    PerceptionComponent,
    ReproductionComponent,
)
from farm.core.agent.behaviors import (
    IAgentBehavior,
    DefaultAgentBehavior,
    LearningAgentBehavior,
)
from farm.core.agent.config.agent_config import AgentConfig

if TYPE_CHECKING:
    from farm.core.services.interfaces import (
        ISpatialQueryService,
        IAgentLifecycleService,
        ITimeService,
    )
    from farm.core.decision.decision import DecisionModule

try:
    from farm.utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        spatial_service: "ISpatialQueryService",
        default_config: Optional[AgentConfig] = None,
        time_service: Optional["ITimeService"] = None,
        lifecycle_service: Optional["IAgentLifecycleService"] = None,
    ):
        """
        Initialize agent factory with required services.

        Args:
            spatial_service: Service for spatial queries (required)
            default_config: Default agent configuration
            time_service: Service for accessing simulation time (optional)
            lifecycle_service: Service for agent lifecycle management (optional)

        Example:
            >>> factory = AgentFactory(
            ...     spatial_service=spatial_service,
            ...     default_config=AgentConfig(),
            ...     time_service=time_service,
            ...     lifecycle_service=lifecycle_service
            ... )
        """
        self._spatial_service = spatial_service
        self._default_config = default_config or AgentConfig()
        self._time_service = time_service
        self._lifecycle_service = lifecycle_service

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

        Returns:
            AgentCore: Agent with DefaultAgentBehavior

        Example:
            >>> agent = factory.create_default_agent(
            ...     agent_id="agent_001",
            ...     position=(50.0, 50.0),
            ...     initial_resources=100
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
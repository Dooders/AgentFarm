"""
Agent factory - creates agents with proper dependency injection.

The factory handles the complexity of assembling agents from components,
behaviors, and configurations, following the Factory and Builder patterns.
"""

from typing import Optional

from farm.core.agent.behaviors.default import DefaultAgentBehavior
from farm.core.agent.behaviors.learning import LearningAgentBehavior
from farm.core.agent.components.combat import CombatComponent
from farm.core.agent.components.movement import MovementComponent
from farm.core.agent.components.perception import PerceptionComponent
from farm.core.agent.components.reproduction import ReproductionComponent
from farm.core.agent.components.resource import ResourceComponent
from farm.core.agent.components.reward import RewardComponent
from farm.core.agent.config.component_configs import AgentComponentConfig
from farm.core.agent.core import AgentCore
from farm.core.agent.services import AgentServices
from farm.core.decision.decision import DecisionModule


class AgentFactory:
    """
    Factory for creating agents with proper component assembly and dependency injection.

    Handles:
    - Component instantiation and assembly
    - Behavior creation (default, learning)
    - Configuration application
    - Service injection
    """

    def __init__(self, services: AgentServices):
        """
        Initialize factory.

        Args:
            services: AgentServices container with all required services
        """
        self.services = services

    def create_default_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        initial_resources: float = 5.0,
        config: Optional[AgentComponentConfig] = None,
        environment=None,
        agent_type: str = "AgentCore",
    ) -> AgentCore:
        """
        Create a default agent with random action selection.

        Used for:
        - Testing and baselines
        - Non-learning agents
        - Simple simulations

        Args:
            agent_id: Unique agent ID
            position: Starting position
            initial_resources: Starting resources
            config: Agent configuration (uses default if None)
            environment: Optional environment reference
            agent_type: Type of agent (e.g., 'system', 'independent', 'control')

        Returns:
            AgentCore instance with default behavior
        """
        if config is None:
            config = AgentComponentConfig.default()

        # Create components
        components = [
            MovementComponent(self.services, config.movement),
            ResourceComponent(self.services, config.resource),
            CombatComponent(self.services, config.combat),
            PerceptionComponent(self.services, config.perception),
            ReproductionComponent(self.services, config.reproduction),
            RewardComponent(self.services, config.reward),
        ]

        # Create behavior
        behavior = DefaultAgentBehavior()

        # Create agent
        agent = AgentCore(
            agent_id=agent_id,
            position=position,
            services=self.services,
            behavior=behavior,
            components=components,
            config=config,
            environment=environment,
            initial_resources=initial_resources,
            agent_type=agent_type,
        )

        return agent

    def create_learning_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        initial_resources: float = 5.0,
        config: Optional[AgentComponentConfig] = None,
        environment=None,
        decision_module: Optional[DecisionModule] = None,
    ) -> AgentCore:
        """
        Create a learning agent with RL-based action selection.

        Used for:
        - RL training
        - Agents that learn from experience
        - Advanced simulations

        Args:
            agent_id: Unique agent ID
            position: Starting position
            initial_resources: Starting resources
            config: Agent configuration (uses default if None)
            environment: Optional environment reference
            decision_module: Optional DecisionModule (creates default if None)

        Returns:
            AgentCore instance with learning behavior
        """
        if config is None:
            config = AgentComponentConfig.default()

        # Create components
        components = [
            MovementComponent(self.services, config.movement),
            ResourceComponent(self.services, config.resource),
            CombatComponent(self.services, config.combat),
            PerceptionComponent(self.services, config.perception),
            ReproductionComponent(self.services, config.reproduction),
            RewardComponent(self.services, config.reward),
        ]

        # Create temporary default behavior to avoid None state
        from farm.core.agent.behaviors.default import DefaultAgentBehavior

        temp_behavior = DefaultAgentBehavior()

        # Create agent with temporary behavior first
        agent = AgentCore(
            agent_id=agent_id,
            position=position,
            services=self.services,
            behavior=temp_behavior,  # Temporary behavior to avoid None state
            components=components,
            config=config,
            environment=environment,
            initial_resources=initial_resources,
        )

        # Create decision module with the agent if not provided
        if decision_module is None:
            decision_config = config.decision if config else AgentComponentConfig.default().decision

            # Get action and observation spaces with fallbacks
            action_space = None
            observation_space = None

            if environment is not None:
                try:
                    action_space = environment.action_space()
                except (AttributeError, TypeError):
                    action_space = None

                try:
                    observation_space = environment.observation_space()
                except (AttributeError, TypeError):
                    observation_space = None

            # Create fallback spaces if environment doesn't provide them
            if action_space is None or observation_space is None:
                import numpy as np
                from gymnasium import spaces

                if action_space is None:
                    # Default to discrete action space with standard action count
                    action_space = spaces.Discrete(7)  # Standard action count

                if observation_space is None:
                    # Default to box observation space with standard dimension
                    observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

            decision_module = DecisionModule(
                agent=agent,  # Now we have the agent
                config=decision_config,
                action_space=action_space,
                observation_space=observation_space,
            )

        # Create and attach the learning behavior
        behavior = LearningAgentBehavior(decision_module)
        agent.behavior = behavior

        return agent

    def create_minimal_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        initial_resources: float = 5.0,
        config: Optional[AgentComponentConfig] = None,
        environment=None,
    ) -> AgentCore:
        """
        Create a minimal agent with only essential components.

        Used for:
        - Lightweight simulations
        - Memory-constrained scenarios
        - Testing core functionality

        Args:
            agent_id: Unique agent ID
            position: Starting position
            initial_resources: Starting resources
            config: Agent configuration
            environment: Optional environment reference

        Returns:
            AgentCore instance with minimal components
        """
        if config is None:
            config = AgentComponentConfig.default()

        # Only essential components
        components = [
            MovementComponent(self.services, config.movement),
            ResourceComponent(self.services, config.resource),
        ]

        # Random behavior
        behavior = DefaultAgentBehavior()

        # Create agent
        agent = AgentCore(
            agent_id=agent_id,
            position=position,
            services=self.services,
            behavior=behavior,
            components=components,
            config=config,
            environment=environment,
            initial_resources=initial_resources,
        )

        return agent

    def create_aggressive_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        environment=None,
    ) -> AgentCore:
        """
        Create an aggressive agent with high combat stats and learning.

        Args:
            agent_id: Unique agent ID
            position: Starting position
            environment: Optional environment reference

        Returns:
            Aggressive learning agent
        """
        return self.create_learning_agent(
            agent_id=agent_id,
            position=position,
            config=AgentComponentConfig.aggressive(),
            environment=environment,
        )

    def create_defensive_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        environment=None,
    ) -> AgentCore:
        """
        Create a defensive agent with high health and defense.

        Args:
            agent_id: Unique agent ID
            position: Starting position
            environment: Optional environment reference

        Returns:
            Defensive learning agent
        """
        return self.create_learning_agent(
            agent_id=agent_id,
            position=position,
            config=AgentComponentConfig.defensive(),
            environment=environment,
        )

    def create_efficient_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        environment=None,
    ) -> AgentCore:
        """
        Create an efficient agent with low resource consumption.

        Args:
            agent_id: Unique agent ID
            position: Starting position
            environment: Optional environment reference

        Returns:
            Efficient learning agent
        """
        return self.create_learning_agent(
            agent_id=agent_id,
            position=position,
            config=AgentComponentConfig.efficient(),
            environment=environment,
        )

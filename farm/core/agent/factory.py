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
        initial_resources: float = 100.0,
        config: Optional[AgentComponentConfig] = None,
        environment=None,
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
        )
        
        return agent
    
    def create_learning_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        initial_resources: float = 100.0,
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
        ]
        
        # Create behavior first (with placeholder decision module if needed)
        if decision_module is None:
            # Create a temporary placeholder decision module
            decision_config = config.decision if config else AgentComponentConfig.default().decision
            
            # We'll update this after agent creation
            decision_module = None
        
        # Create agent first (behavior will be set after decision_module is ready)
        agent = AgentCore(
            agent_id=agent_id,
            position=position,
            services=self.services,
            behavior=None,  # Will be set after decision module
            components=components,
            config=config,
            environment=environment,
            initial_resources=initial_resources,
        )
        
        # Now create decision module with the agent
        if decision_module is None:
            decision_config = config.decision if config else AgentComponentConfig.default().decision
            decision_module = DecisionModule(
                agent=agent,
                config=decision_config,
                action_space=environment.action_space() if environment else None,
                observation_space=environment.observation_space() if environment else None,
            )
        
        # Now create and attach behavior
        behavior = LearningAgentBehavior(decision_module)
        agent.behavior = behavior
        
        return agent
    
    def create_minimal_agent(
        self,
        agent_id: str,
        position: tuple[float, float],
        initial_resources: float = 100.0,
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

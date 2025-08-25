from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from farm.core.environment import Environment

from farm.actions.attack import attack_action, AttackConfig
from farm.actions.gather import gather_action
from farm.actions.move import move_action
from farm.actions.reproduce import reproduce_action
from farm.actions.share import share_action
from farm.agents.base_agent import BaseAgent
from farm.core.action import Action


class AlienAttackConfig(AttackConfig):
    """Enhanced attack configuration for alien agents."""
    
    # Alien-specific attack bonuses
    attack_base_cost: float = -0.15  # Lower cost than humans
    attack_success_reward: float = 1.3  # Higher reward for successful attacks
    attack_failure_penalty: float = -0.2  # Lower penalty than humans
    attack_defense_threshold: float = 0.4  # More aggressive threshold
    attack_defense_boost: float = 1.5  # Lower defense boost (prefer offense)
    
    # Surrounding mechanics
    surrounding_bonus: float = 1.5  # Damage bonus when multiple aliens attack same target
    swarm_coordination_range: float = 25.0  # Range for coordinated attacks


class AlienAgent(BaseAgent):
    """Alien agent with enhanced combat capabilities and aggressive behavior.
    
    Alien agents have advantages in:
    - Higher base attack damage
    - Lower attack costs
    - Coordination bonuses when swarming
    - More aggressive behavior patterns
    """

    def __init__(
        self,
        agent_id: str,
        position: tuple[float, float],
        resource_level: int,
        environment: "Environment",
        generation: int = 0,
        action_set: Optional[list[Action]] = None,
    ):
        """Initialize an AlienAgent with enhanced combat capabilities.

        Parameters
        ----------
        agent_id : str
            Unique identifier for this agent
        position : tuple[float, float]
            Initial (x,y) coordinates
        resource_level : int
            Starting resource amount
        environment : Environment
            Reference to simulation environment
        generation : int
            Generation number in evolutionary lineage
        action_set : list[Action], optional
            Custom action set for this agent
        """
        # Create aggressive action set if none provided
        if action_set is None:
            action_set = [
                Action("move", 0.20, move_action),  # Moderate movement for positioning
                Action("gather", 0.15, gather_action),  # Lower gathering (aggressive focus)
                Action("share", 0.05, share_action),  # Minimal sharing (selfish)
                Action("attack", 0.45, attack_action),  # High aggression
                Action("reproduce", 0.15, reproduce_action),  # Moderate reproduction
            ]

        # Initialize base agent
        super().__init__(
            agent_id=agent_id,
            position=position,
            resource_level=resource_level,
            environment=environment,
            action_set=action_set,
            generation=generation,
        )
        
        # Override attack module with alien-specific configuration
        self.attack_module.config = AlienAttackConfig()
        
        # Enhanced alien attributes
        self.alien_attack_multiplier = getattr(environment.config, 'alien_attack_multiplier', 1.3)
        self.swarm_coordination_range = getattr(environment.config, 'swarm_coordination_range', 25.0)
        
        # Configure modules for aggressive behavior
        self._configure_alien_modules()

    def _configure_alien_modules(self):
        """Configure modules for alien-specific aggressive behavior."""
        # Gathering: Less efficient, more aggressive
        self.gather_module.config.gather_efficiency_multiplier = 0.3
        self.gather_module.config.gather_cost_multiplier = 0.1
        self.gather_module.config.min_resource_threshold = 0.05
        
        # Sharing: Minimal sharing behavior
        self.share_module.config.share_success_reward = 0.1
        self.share_module.config.min_share_amount = 1
        
        # Movement: More aggressive positioning
        self.move_module.config.move_resource_approach_reward = 0.2
        self.move_module.config.move_base_cost = -0.05

    @property
    def attack_strength(self) -> float:
        """Calculate alien attack strength with alien-specific bonuses."""
        base_strength = super().attack_strength
        alien_bonus = base_strength * (self.alien_attack_multiplier - 1.0)
        
        # Check for swarm coordination bonus
        swarm_bonus = self._calculate_swarm_bonus()
        
        return base_strength + alien_bonus + swarm_bonus

    def _calculate_swarm_bonus(self) -> float:
        """Calculate bonus damage from coordinated alien attacks."""
        nearby_aliens = self._get_nearby_aliens()
        if len(nearby_aliens) >= 2:  # At least 2 other aliens nearby
            # Scale bonus with number of nearby aliens, max 50% bonus
            bonus_multiplier = min(0.5, len(nearby_aliens) * 0.15)
            return self.config.attack_base_damage * bonus_multiplier
        return 0.0

    def _get_nearby_aliens(self) -> list["AlienAgent"]:
        """Get nearby alien agents for coordination."""
        nearby_agents = self.environment.get_nearby_agents(
            self.position, self.swarm_coordination_range
        )
        return [
            agent for agent in nearby_agents 
            if isinstance(agent, AlienAgent) and agent != self and agent.alive
        ]

    def calculate_attack_reward(
        self, target: "BaseAgent", damage_dealt: float, action: int
    ) -> float:
        """Calculate alien-specific attack rewards with coordination bonuses."""
        base_reward = super().calculate_attack_reward(target, damage_dealt, action)
        
        # Bonus for attacking humans (primary objective)
        if hasattr(target, '__class__') and 'Human' in target.__class__.__name__:
            base_reward *= 1.2
            
        # Additional bonus for coordinated attacks
        nearby_aliens = self._get_nearby_aliens()
        if len(nearby_aliens) >= 1:
            coordination_bonus = len(nearby_aliens) * 0.1
            base_reward += coordination_bonus
            
        return base_reward

    def decide_action(self):
        """Enhanced decision making with alien-specific priorities."""
        # Aliens prefer aggressive actions when humans are nearby
        nearby_agents = self.environment.get_nearby_agents(self.position, 40.0)
        humans_nearby = [
            agent for agent in nearby_agents 
            if hasattr(agent, '__class__') and 'Human' in agent.__class__.__name__
        ]
        
        if humans_nearby:
            # Increase attack weight when humans are nearby
            for action in self.actions:
                if action.name == "attack":
                    action.weight *= 1.5
                elif action.name == "move":
                    action.weight *= 1.2  # Also increase movement for positioning
        
        return super().decide_action()
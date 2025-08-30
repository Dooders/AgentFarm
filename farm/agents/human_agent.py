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


class HumanAttackConfig(AttackConfig):
    """Enhanced defense configuration for human agents."""
    
    # Human-specific defensive bonuses
    attack_base_cost: float = -0.25  # Higher cost (prefer defense)
    attack_success_reward: float = 0.8  # Lower reward (less aggressive)
    attack_failure_penalty: float = -0.4  # Higher penalty (risk averse)
    attack_defense_threshold: float = 0.2  # More defensive threshold
    attack_defense_boost: float = 2.5  # Higher defense boost
    
    # Group defense mechanics
    group_defense_bonus: float = 1.3  # Defense bonus when grouped with other humans
    group_coordination_range: float = 20.0  # Range for group coordination


class HumanAgent(BaseAgent):
    """Human agent with enhanced defensive capabilities and cooperative behavior.
    
    Human agents excel in:
    - Group defense coordination
    - Resource sharing and cooperation
    - Defensive positioning
    - Survival-focused strategies
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
        """Initialize a HumanAgent with enhanced defensive capabilities.

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
        # Create defensive/cooperative action set if none provided
        if action_set is None:
            action_set = [
                Action("move", 0.25, move_action),  # Moderate movement for positioning
                Action("gather", 0.30, gather_action),  # High gathering (survival focus)
                Action("share", 0.25, share_action),  # High sharing (cooperative)
                Action("attack", 0.10, attack_action),  # Low aggression (defensive)
                Action("reproduce", 0.10, reproduce_action),  # Lower reproduction (survival focus)
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
        
        # Override attack module with human-specific configuration
        self.attack_module.config = HumanAttackConfig()
        
        # Enhanced human attributes
        self.group_defense_bonus = getattr(environment.config, 'human_defense_bonus', 1.2)
        self.group_coordination_range = getattr(environment.config, 'group_coordination_range', 20.0)
        
        # Configure modules for defensive/cooperative behavior
        self._configure_human_modules()

    def _configure_human_modules(self):
        """Configure modules for human-specific defensive behavior."""
        # Gathering: More efficient, cooperative
        self.gather_module.config.gather_efficiency_multiplier = 0.6
        self.gather_module.config.gather_cost_multiplier = 0.4
        self.gather_module.config.min_resource_threshold = 0.3
        
        # Sharing: High cooperation
        self.share_module.config.share_success_reward = 0.4
        self.share_module.config.altruism_bonus = 0.3
        self.share_module.config.min_share_amount = 2
        
        # Movement: Defensive positioning
        self.move_module.config.move_resource_approach_reward = 0.4
        self.move_module.config.move_base_cost = -0.15

    @property
    def defense_strength(self) -> float:
        """Calculate human defense strength with group coordination bonuses."""
        base_defense = super().defense_strength
        
        # Check for group defense bonus
        group_bonus = self._calculate_group_defense_bonus()
        
        return base_defense + group_bonus

    def _calculate_group_defense_bonus(self) -> float:
        """Calculate bonus defense from nearby human allies."""
        nearby_humans = self._get_nearby_humans()
        if len(nearby_humans) >= 1:  # At least 1 other human nearby
            # Scale bonus with number of nearby humans, max 40% bonus
            bonus_multiplier = min(0.4, len(nearby_humans) * 0.15)
            return self.config.base_defense_strength * bonus_multiplier
        return 0.0

    def _get_nearby_humans(self) -> list["HumanAgent"]:
        """Get nearby human agents for group coordination."""
        nearby_agents = self.environment.get_nearby_agents(
            self.position, self.group_coordination_range
        )
        return [
            agent for agent in nearby_agents 
            if isinstance(agent, HumanAgent) and agent != self and agent.alive
        ]

    def handle_combat(self, attacker: "BaseAgent", damage: float) -> float:
        """Handle incoming combat with group defense bonuses."""
        # Apply group defense bonus
        effective_defense = self.defense_strength
        
        # Reduce damage based on defense strength
        damage_reduction = min(damage * 0.3, effective_defense * 0.1)
        actual_damage = max(0, damage - damage_reduction)
        
        # Apply damage
        self.current_health -= actual_damage
        
        # Log the combat event
        if hasattr(self.environment, 'db') and self.environment.db:
            self.environment.db.log_health_incident(
                step_number=self.environment.time,
                agent_id=self.agent_id,
                health_before=self.current_health + actual_damage,
                health_after=self.current_health,
                cause="alien_attack",
                details={
                    "attacker_id": attacker.agent_id,
                    "original_damage": damage,
                    "actual_damage": actual_damage,
                    "defense_strength": effective_defense,
                    "nearby_humans": len(self._get_nearby_humans())
                }
            )
        
        return actual_damage

    def decide_action(self):
        """Enhanced decision making with survival and group coordination priorities."""
        # Humans prefer defensive actions when aliens are nearby
        nearby_agents = self.environment.get_nearby_agents(self.position, 50.0)
        aliens_nearby = [
            agent for agent in nearby_agents 
            if hasattr(agent, '__class__') and 'Alien' in agent.__class__.__name__
        ]
        
        if aliens_nearby:
            # Increase defensive behaviors when aliens are nearby
            for action in self.actions:
                if action.name == "share":
                    action.weight *= 1.4  # Increase cooperation
                elif action.name == "gather":
                    action.weight *= 1.3  # Focus on survival resources
                elif action.name == "move":
                    action.weight *= 1.2  # Move to better defensive positions
        
        # Check if we're isolated and need to group up
        nearby_humans = self._get_nearby_humans()
        if len(nearby_humans) == 0 and aliens_nearby:
            # Increase movement weight to find other humans
            for action in self.actions:
                if action.name == "move":
                    action.weight *= 1.5
        
        return super().decide_action()

    def calculate_move_reward(self, old_pos, new_pos):
        """Calculate movement reward with group coordination considerations."""
        base_reward = super().calculate_move_reward(old_pos, new_pos)
        
        # Bonus for moving closer to other humans
        nearby_humans_old = len(self.environment.get_nearby_agents(old_pos, 30.0))
        nearby_humans_new = len(self.environment.get_nearby_agents(new_pos, 30.0))
        
        if nearby_humans_new > nearby_humans_old:
            base_reward += 0.2  # Bonus for grouping up
        elif nearby_humans_new < nearby_humans_old:
            base_reward -= 0.1  # Penalty for isolating
            
        return base_reward
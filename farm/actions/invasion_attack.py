"""Enhanced attack actions for alien invasion combat simulation.

This module provides specialized attack mechanics for the alien invasion scenario,
including swarm coordination, group defense, and territorial combat bonuses.
"""

import random
import logging
from typing import TYPE_CHECKING, List

from farm.actions.attack import AttackActionSpace, attack_action
from farm.loggers.attack_logger import AttackLogger

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent
    from farm.agents.alien_agent import AlienAgent
    from farm.agents.human_agent import HumanAgent

logger = logging.getLogger(__name__)


def invasion_attack_action(agent: "BaseAgent") -> None:
    """Enhanced attack action with invasion-specific mechanics.
    
    This function extends the base attack action with:
    - Alien swarm coordination bonuses
    - Human group defense mechanics
    - Surrounding attack bonuses
    - Territorial control considerations
    
    Args:
        agent: The agent performing the attack action
    """
    # Get current state and health ratio for decision making
    state = agent.get_state()
    health_ratio = agent.current_health / agent.starting_health
    initial_resources = agent.resource_level

    # Initialize attack logger
    attack_logger = AttackLogger(agent.environment.db)

    # Select attack action using DQN with health-based defense boost
    action = agent.attack_module.select_action(
        state.to_tensor(agent.attack_module.device), health_ratio
    )

    # Handle defensive action
    if action == AttackActionSpace.DEFEND:
        agent.is_defending = True
        attack_logger.log_defense(
            step_number=agent.environment.time,
            agent=agent,
            resources_before=initial_resources,
            resources_after=initial_resources,
        )
        return

    # Calculate attack target position based on selected action direction
    target_pos = agent.calculate_attack_position(action)

    # Safety check for config
    if agent.config is None:
        logger.error(f"Agent {agent.agent_id} has no config, skipping attack action")
        return

    # Find potential targets within attack range
    targets = agent.environment.get_nearby_agents(target_pos, agent.config.attack_range)

    if not targets:
        attack_logger.log_attack_attempt(
            step_number=agent.environment.time,
            agent=agent,
            action_target_id=None,
            target_position=target_pos,
            resources_before=initial_resources,
            resources_after=initial_resources,
            success=False,
            targets_found=0,
            reason="no_targets",
        )
        return

    # Calculate attack cost (potentially reduced for aliens)
    attack_cost = agent.config.attack_base_cost * agent.resource_level
    agent.resource_level += attack_cost

    # Initialize combat statistics
    total_damage_dealt = 0.0
    successful_hits = 0

    # Update global combat counters
    agent.environment.combat_encounters += 1
    agent.environment.combat_encounters_this_step += 1

    # Filter valid targets (no self-targeting)
    valid_targets = [target for target in targets if target.agent_id != agent.agent_id]

    if not valid_targets:
        attack_logger.log_attack_attempt(
            step_number=agent.environment.time,
            agent=agent,
            action_target_id=None,
            target_position=target_pos,
            resources_before=initial_resources,
            resources_after=agent.resource_level,
            success=False,
            targets_found=0,
            reason="no_valid_targets",
        )
        return

    # Select target (prefer enemy faction if available)
    target = _select_priority_target(agent, valid_targets)

    # Calculate enhanced damage with invasion-specific bonuses
    base_damage = _calculate_invasion_damage(agent, target, target_pos)

    # Apply defensive damage reduction
    if target.is_defending:
        # Enhanced defense calculation for humans in groups
        defense_strength = target.defense_strength
        damage_reduction = min(base_damage * 0.4, defense_strength * 0.15)
        base_damage = max(0, base_damage - damage_reduction)

    # Apply damage to target
    if target.take_damage(base_damage):
        total_damage_dealt += base_damage
        successful_hits += 1

    # Update successful attack counters
    if successful_hits > 0:
        agent.environment.successful_attacks += 1
        agent.environment.successful_attacks_this_step += 1

    # Calculate enhanced reward with faction-specific bonuses
    reward = _calculate_invasion_reward(agent, target, total_damage_dealt, action)
    agent.total_reward += reward

    # Log enhanced attack outcome
    attack_logger.log_attack_attempt(
        step_number=agent.environment.time,
        agent=agent,
        action_target_id=target.agent_id,
        target_position=target_pos,
        resources_before=initial_resources,
        resources_after=agent.resource_level,
        success=successful_hits > 0,
        targets_found=len(valid_targets),
        damage_dealt=total_damage_dealt,
        reason="invasion_combat",
    )


def _select_priority_target(agent: "BaseAgent", targets: List["BaseAgent"]) -> "BaseAgent":
    """Select target with faction-based priorities."""
    # Import here to avoid circular imports
    from farm.agents.alien_agent import AlienAgent
    from farm.agents.human_agent import HumanAgent
    
    # Aliens prefer to target humans
    if isinstance(agent, AlienAgent):
        human_targets = [t for t in targets if isinstance(t, HumanAgent)]
        if human_targets:
            return random.choice(human_targets)
    
    # Humans prefer to target aliens
    elif isinstance(agent, HumanAgent):
        alien_targets = [t for t in targets if isinstance(t, AlienAgent)]
        if alien_targets:
            return random.choice(alien_targets)
    
    # Default: random target
    return random.choice(targets)


def _calculate_invasion_damage(agent: "BaseAgent", target: "BaseAgent", target_pos) -> float:
    """Calculate damage with invasion-specific bonuses."""
    # Import here to avoid circular imports
    from farm.agents.alien_agent import AlienAgent
    from farm.agents.human_agent import HumanAgent
    
    # Base damage calculation
    base_damage = agent.attack_strength * (agent.resource_level / agent.starting_health)
    
    # Alien swarm coordination bonus
    if isinstance(agent, AlienAgent):
        # Check for nearby aliens for swarm bonus
        nearby_aliens = [
            a for a in agent.environment.get_nearby_agents(agent.position, 25.0)
            if isinstance(a, AlienAgent) and a != agent and a.alive
        ]
        
        if len(nearby_aliens) >= 1:
            swarm_bonus = min(0.5, len(nearby_aliens) * 0.15)  # Max 50% bonus
            base_damage *= (1.0 + swarm_bonus)
        
        # Bonus damage against humans (primary objective)
        if isinstance(target, HumanAgent):
            base_damage *= 1.2
    
    # Human group defense considerations are handled in the target's defense_strength
    # and take_damage methods
    
    # Surrounding bonus - check if multiple attackers are targeting the same agent
    attackers_nearby = [
        a for a in agent.environment.get_nearby_agents(target_pos, 30.0)
        if a != agent and a.alive and hasattr(a, 'attack_module')
    ]
    
    # Surrounding bonus applies if 2+ attackers are near the target
    if len(attackers_nearby) >= 1:
        surrounding_bonus = getattr(agent.config, 'surrounding_bonus', 1.2)
        base_damage *= surrounding_bonus
    
    return base_damage


def _calculate_invasion_reward(agent: "BaseAgent", target: "BaseAgent", damage_dealt: float, action: int) -> float:
    """Calculate reward with invasion-specific bonuses."""
    # Import here to avoid circular imports
    from farm.agents.alien_agent import AlienAgent
    from farm.agents.human_agent import HumanAgent
    
    # Start with base attack reward calculation
    base_reward = agent.calculate_attack_reward(target, damage_dealt, action)
    
    # Faction-specific reward bonuses
    if isinstance(agent, AlienAgent) and isinstance(target, HumanAgent):
        # Aliens get extra reward for attacking humans
        base_reward *= 1.3
        
        # Extra bonus if target is eliminated
        if not target.alive:
            base_reward += 2.0
    
    elif isinstance(agent, HumanAgent) and isinstance(target, AlienAgent):
        # Humans get extra reward for successful defense
        base_reward *= 1.1
        
        # Bonus for defensive eliminations
        if not target.alive:
            base_reward += 1.5
    
    # Territorial control bonus
    if hasattr(agent.environment, 'territorial_control_ratio'):
        if isinstance(agent, AlienAgent):
            # Aliens get bonus for expanding territory
            if agent.environment.territorial_control_ratio < 0.5:
                base_reward += 0.3
        elif isinstance(agent, HumanAgent):
            # Humans get bonus for holding territory
            if agent.environment.territorial_control_ratio > 0.5:
                base_reward += 0.2
    
    return base_reward


def get_invasion_attack_stats(environment) -> dict:
    """Get comprehensive attack statistics for the invasion scenario."""
    from farm.agents.alien_agent import AlienAgent
    from farm.agents.human_agent import HumanAgent
    
    humans = [agent for agent in environment.agents if isinstance(agent, HumanAgent) and agent.alive]
    aliens = [agent for agent in environment.agents if isinstance(agent, AlienAgent) and agent.alive]
    
    return {
        'combat_encounters': environment.combat_encounters,
        'successful_attacks': environment.successful_attacks,
        'humans_alive': len(humans),
        'aliens_alive': len(aliens),
        'humans_eliminated': getattr(environment, 'humans_eliminated', 0),
        'aliens_eliminated': getattr(environment, 'aliens_eliminated', 0),
        'territorial_control': getattr(environment, 'territorial_control_ratio', 0.5),
        'total_agents': len(environment.agents),
        'combat_intensity': environment.combat_encounters / max(1, environment.time),
    }
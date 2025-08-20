"""
Attack action module for AgentFarm.

This module handles combat actions including directional attacks and defensive behavior.
Updated to use the new profile-based configuration system.
"""

from typing import TYPE_CHECKING, Optional, Tuple
import logging
import random

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNModule, BaseQNetwork, SharedEncoder, DEVICE
from farm.core.profiles import DQNProfile

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class AttackActionSpace:
    """Defines the available attack actions and their corresponding indices."""
    
    ATTACK_RIGHT: int = 0
    ATTACK_LEFT: int = 1
    ATTACK_UP: int = 2
    ATTACK_DOWN: int = 3
    DEFEND: int = 4


class AttackQNetwork(BaseQNetwork):
    """Neural network architecture for attack Q-value approximation."""
    
    def __init__(self, input_dim: int, hidden_size: int = 64, 
                 shared_encoder: Optional[SharedEncoder] = None) -> None:
        super().__init__(input_dim, 5, hidden_size, shared_encoder)  # 5 attack actions


class AttackModule(BaseDQNModule):
    """
    Deep Q-Learning module for combat actions.
    
    This module learns optimal attack and defense strategies through experience.
    It handles directional attacks and defensive stances, with rewards/costs
    configured through the profile system.
    """
    
    def __init__(
        self,
        dqn_profile: DQNProfile,
        rewards: dict = None,
        costs: dict = None, 
        thresholds: dict = None,
        device: torch.device = DEVICE,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        """
        Initialize the attack module.
        
        Args:
            dqn_profile: DQN learning configuration profile
            rewards: Attack-specific reward values (success, kill, failure_penalty)
            costs: Attack-specific costs (base)
            thresholds: Attack-specific thresholds (defense, defense_boost)
            device: Computation device
            shared_encoder: Optional shared encoder for efficiency
        """
        # Set default reward/cost values
        self.rewards = {
            "success": 1.0,
            "kill": 5.0, 
            "failure_penalty": -0.3,
            **(rewards or {})
        }
        
        self.costs = {
            "base": -0.2,
            **(costs or {})
        }
        
        self.thresholds = {
            "defense": 0.3,
            "defense_boost": 2.0,
            **(thresholds or {})
        }
        
        super().__init__(
            input_dim=8,
            output_dim=5, 
            dqn_profile=dqn_profile,
            device=device,
            shared_encoder=shared_encoder
        )
        
        self._setup_action_space()
    
    def _setup_action_space(self) -> None:
        """Initialize action space mapping."""
        self.action_space = AttackActionSpace()
        self.action_names = {
            0: "ATTACK_RIGHT",
            1: "ATTACK_LEFT", 
            2: "ATTACK_UP",
            3: "ATTACK_DOWN",
            4: "DEFEND"
        }
    
    def select_action(self, state: torch.Tensor, health_ratio: float) -> int:
        """
        Select attack action based on current state and health.
        
        Args:
            state: Current environment state tensor
            health_ratio: Agent's current health as ratio (0-1)
            
        Returns:
            Selected action index
        """
        # Adjust exploration based on health (more defensive when low health)
        epsilon = self.epsilon
        if health_ratio < self.thresholds["defense"]:
            epsilon = self.epsilon * 0.5  # Less exploration when vulnerable
            
        # Bias towards defense when health is low
        if health_ratio < self.thresholds["defense"] and random.random() < 0.3:
            return self.action_space.DEFEND
            
        return super().select_action(state, epsilon)


def attack_action(agent: "BaseAgent") -> None:
    """
    Execute attack action for an agent.
    
    This function handles the complete attack sequence including target selection,
    damage calculation, and reward/penalty application.
    
    Args:
        agent: The agent performing the attack action
    """
    from farm.loggers.attack_logger import AttackLogger
    
    # Get attack configuration from agent's config
    action_config = agent.environment.config.get_action_config("attack")
    attack_module = agent.attack_module
    
    step_number = agent.environment.time
    initial_position = agent.position
    resources_before = agent.resource_level
    
    # Get current state and select action
    state = agent.get_state().to_tensor(agent.device)
    health_ratio = agent.current_health / agent.starting_health
    action = attack_module.select_action(state, health_ratio)
    
    # Determine attack position based on selected action  
    attack_position = agent.calculate_attack_position(action)
    
    # Apply base cost
    base_cost = action_config.costs.get("base", -0.2)
    agent.resource_level = max(0, agent.resource_level + base_cost)
    
    success = False
    damage_dealt = 0.0
    targets_found = 0
    target_agent = None
    reason = None
    
    if action == AttackActionSpace.DEFEND:
        # Defensive action
        agent.is_defending = True
        reason = "defensive_stance"
        reward = 0.1  # Small positive reward for defense
        
        # Log defense action
        logger = AttackLogger(agent.environment.db)
        logger.log_defense(step_number, agent, resources_before, agent.resource_level)
        
    else:
        # Attack action - find targets in attack range
        nearby_agents = agent.environment.get_nearby_agents(
            attack_position, agent.environment.config.attack_range
        )
        
        targets_found = len(nearby_agents)
        
        if nearby_agents:
            # Select target (closest or random)
            target_agent = min(nearby_agents, key=lambda a: 
                np.linalg.norm(np.array(a.position) - np.array(attack_position))
            )
            
            # Calculate damage
            damage_dealt = agent.attack_strength
            target_killed = target_agent.take_damage(damage_dealt)
            
            if target_killed:
                success = True
                reward = action_config.rewards.get("kill", 5.0)
                reason = "target_killed"
            else:
                success = True  
                reward = action_config.rewards.get("success", 1.0)
                reason = "damage_dealt"
        else:
            # No targets found
            reward = action_config.rewards.get("failure_penalty", -0.3)
            reason = "no_targets"
    
    # Store experience for learning
    next_state = agent.get_state().to_tensor(agent.device)
    attack_module.store_experience(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=False,
        step_number=step_number,
        agent_id=agent.agent_id,
        module_type="attack",
        action_taken_mapped=attack_module.action_names.get(action, str(action))
    )
    
    # Log attack attempt
    attack_logger = AttackLogger(agent.environment.db)
    attack_logger.log_attack_attempt(
        step_number=step_number,
        agent=agent,
        action_target_id=target_agent.agent_id if target_agent else None,
        target_position=attack_position,
        resources_before=resources_before,
        resources_after=agent.resource_level,
        success=success,
        targets_found=targets_found,
        damage_dealt=damage_dealt,
        reason=reason
    )
    
    # Reset defending state if not defending
    if action != AttackActionSpace.DEFEND:
        agent.is_defending = False
    
    # Train the module periodically
    if len(attack_module.memory) > attack_module.profile.batch_size:
        if step_number % 4 == 0:  # Train every 4 steps
            attack_module.train(
                step_number=step_number,
                agent_id=agent.agent_id,
                module_type="attack"
            )
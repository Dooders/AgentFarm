"""
Reward management component.

Handles reward calculation, tracking, and distribution for agents.
Provides a pluggable reward system that can be customized for different scenarios.
"""

from typing import Any, Dict, Optional

from farm.core.agent.config.component_configs import RewardConfig
from farm.core.agent.services import AgentServices

from .base import AgentComponent


class RewardComponent(AgentComponent):
    """
    Manages agent reward calculation and tracking.

    Responsibilities:
    - Calculate rewards based on agent state and actions
    - Track cumulative and step rewards
    - Provide reward history and statistics
    - Support different reward calculation strategies
    """

    def __init__(self, services: AgentServices, config: RewardConfig):
        """
        Initialize reward component.

        Args:
            services: Agent services
            config: Reward configuration
        """
        super().__init__(services, "RewardComponent")
        self.config = config
        self.cumulative_reward = 0.0
        self.step_reward = 0.0
        self.reward_history: list[float] = []
        self.last_action_reward = 0.0
        self.pre_action_state: Optional[Dict[str, Any]] = None

    def attach(self, core) -> None:
        """Attach to core and initialize reward tracking."""
        super().attach(core)
        # Initialize reward tracking
        self.cumulative_reward = 0.0
        self.step_reward = 0.0
        self.reward_history = []

    def on_step_start(self) -> None:
        """Called at start of step - capture pre-action state for delta rewards."""
        if self.core:
            self.pre_action_state = self._capture_state()

    def on_step_end(self) -> None:
        """Called at end of step - calculate and apply rewards."""
        if self.core:
            self.step_reward = self._calculate_reward()
            self._apply_reward(self.step_reward)

    def on_terminate(self) -> None:
        """Called when agent dies - apply death penalty."""
        if self.core and not self.core.alive:
            # Apply death penalty
            death_penalty = self.config.death_penalty
            self._apply_reward(death_penalty)
            self._log_debug(f"Applied death penalty: {death_penalty}")

    def _capture_state(self) -> Dict[str, Any]:
        """Capture current agent state for delta reward calculation using existing state system."""
        if not self.core:
            return {}

        # Use the existing state system if available
        if hasattr(self.core, "state_manager") and self.core.state_manager:
            state = self.core.state_manager.get_state()
            return {
                "resource_level": float(state.resource_level),
                "health": float(state.health),
                "alive": bool(state.alive),
                "position": (float(state.position_x), float(state.position_y)),
                "age": int(state.age),
            }
        else:
            # Fallback to direct attribute access for backward compatibility
            return {
                "resource_level": float(getattr(self.core, "resource_level", 0.0)),
                "health": float(getattr(self.core, "current_health", 0.0)),
                "alive": bool(getattr(self.core, "alive", True)),
                "position": tuple(getattr(self.core, "position", (0.0, 0.0))),
                "age": int(getattr(self.core, "age", 0)),
            }

    def _calculate_reward(self) -> float:
        """
        Calculate reward for the current step.

        Uses delta-based rewards when pre-action state is available,
        otherwise falls back to state-based rewards.

        Returns:
            Calculated reward value
        """
        if not self.core:
            return 0.0

        # Use delta-based rewards if pre-action state is available
        if self.pre_action_state is not None:
            return self._calculate_delta_reward()
        else:
            return self._calculate_state_reward()

    def _calculate_delta_reward(self) -> float:
        """
        Calculate reward based on state changes (better for RL).
        
        Uses EXACT logic from main branch:
        - reward = resource_delta + health_delta * 0.5
        - reward += 0.1 if alive else -10.0
        """
        if not self.core or not self.pre_action_state:
            return 0.0

        current_state = self._capture_state()

        # Resource delta (direct, no scaling)
        resource_delta = current_state["resource_level"] - self.pre_action_state["resource_level"]
        
        # Health delta (scaled by 0.5)
        health_delta = current_state["health"] - self.pre_action_state["health"]
        
        # Base reward from deltas
        reward = resource_delta + health_delta * 0.5

        # Survival handling (matches main branch exactly)
        if current_state["alive"]:
            reward += 0.1  # Survival bonus for staying alive
        else:
            reward -= 10.0  # Death penalty
            self.last_action_reward = reward
            return reward  # Early return for dead agents

        self.last_action_reward = reward
        return reward

    def _calculate_state_reward(self) -> float:
        """
        Calculate reward based on current state (fallback method).
        
        Uses EXACT logic from main branch:
        - resource_reward = resource_level * 0.1
        - survival_reward = 0.1
        - health_reward = current_health / starting_health
        """
        if not self.core:
            return 0.0

        current_state = self._capture_state()

        # Early return for dead agents
        if not current_state["alive"]:
            self.last_action_reward = -10.0
            return -10.0

        # Get starting health for ratio calculation
        starting_health = 100.0  # Default
        combat_comp = self.core.get_component("combat")
        if combat_comp and hasattr(combat_comp, "config"):
            starting_health = combat_comp.config.starting_health

        # State-based rewards (matches main branch exactly)
        resource_reward = current_state["resource_level"] * 0.1
        survival_reward = 0.1
        health_reward = current_state["health"] / starting_health

        reward = resource_reward + survival_reward + health_reward
        self.last_action_reward = reward
        return reward

    def _calculate_action_reward(self) -> float:
        """Calculate action-specific rewards based on last action."""
        if not self.core:
            return 0.0

        # This would need to be integrated with the action system
        # For now, return 0 - can be extended when action tracking is available
        return 0.0

    def _apply_reward(self, reward: float) -> None:
        """Apply reward to agent's cumulative total using existing state system."""
        self.cumulative_reward += reward
        self.reward_history.append(reward)

        # Update agent's total reward using the existing state system
        if self.core:
            if hasattr(self.core, "state_manager") and self.core.state_manager:
                # Use the state manager to update total reward
                self.core.state_manager.add_reward(reward)
            elif hasattr(self.core, "total_reward"):
                # Fallback to direct attribute access for backward compatibility
                self.core.total_reward = self.cumulative_reward

        # Keep history within limits
        if len(self.reward_history) > self.config.max_history_length:
            self.reward_history = self.reward_history[-self.config.max_history_length :]

        self._log_debug(f"Applied reward: {reward:.3f}, cumulative: {self.cumulative_reward:.3f}")

    def add_reward(self, amount: float, reason: str = "") -> None:
        """
        Manually add reward to the agent.

        Args:
            amount: Reward amount to add
            reason: Optional reason for the reward
        """
        self._apply_reward(amount)
        if reason:
            self._log_debug(f"Manual reward added: {amount:.3f} ({reason})")

    def get_reward_stats(self) -> Dict[str, float]:
        """Get reward statistics for the agent."""
        if not self.reward_history:
            return {
                "cumulative": 0.0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "recent_average": 0.0,
            }

        recent_rewards = self.reward_history[-self.config.recent_window :]

        return {
            "cumulative": self.cumulative_reward,
            "average": sum(self.reward_history) / len(self.reward_history),
            "min": min(self.reward_history),
            "max": max(self.reward_history),
            "recent_average": sum(recent_rewards) / len(recent_rewards),
            "last_action": self.last_action_reward,
        }

    def reset_rewards(self) -> None:
        """Reset all reward tracking (useful for new episodes) using existing state system."""
        self.cumulative_reward = 0.0
        self.step_reward = 0.0
        self.reward_history = []
        self.last_action_reward = 0.0
        self.pre_action_state = None

        # Reset agent's total reward using the existing state system
        if self.core:
            if hasattr(self.core, "state_manager") and self.core.state_manager:
                # Use the dedicated reset_reward method in AgentStateManager
                self.core.state_manager.reset_reward()
            elif hasattr(self.core, "total_reward"):
                # Fallback to direct attribute access for backward compatibility
                self.core.total_reward = 0.0

        self._log_debug("Reward tracking reset")

    @property
    def current_reward(self) -> float:
        """Get current step reward."""
        return self.step_reward

    @property
    def total_reward(self) -> float:
        """Get cumulative reward."""
        return self.cumulative_reward

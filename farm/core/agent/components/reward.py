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
        if self.core:
            # Apply death penalty
            death_penalty = self.config.death_penalty
            self._apply_reward(death_penalty)
            self._log_debug(f"Applied death penalty: {death_penalty}")

    def _capture_state(self) -> Dict[str, Any]:
        """Capture current agent state for delta reward calculation."""
        if not self.core:
            return {}

        return {
            "resource_level": getattr(self.core, "resource_level", 0.0),
            "health": getattr(self.core, "current_health", 0.0),
            "alive": getattr(self.core, "alive", True),
            "position": getattr(self.core, "position", (0.0, 0.0)),
            "age": getattr(self.core, "age", 0),
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
        """Calculate reward based on state changes (better for RL)."""
        if not self.core or not self.pre_action_state:
            return 0.0

        current_state = self._capture_state()

        # Resource delta reward
        resource_delta = current_state["resource_level"] - self.pre_action_state["resource_level"]
        resource_reward = resource_delta * self.config.resource_reward_scale

        # Health delta reward
        health_delta = current_state["health"] - self.pre_action_state["health"]
        health_reward = health_delta * self.config.health_reward_scale

        # Survival reward
        survival_reward = 0.0
        if current_state["alive"] and self.pre_action_state["alive"]:
            survival_reward = self.config.survival_bonus
        elif not current_state["alive"] and self.pre_action_state["alive"]:
            survival_reward = self.config.death_penalty

        # Age reward (encourage longevity)
        age_reward = self.config.age_bonus if current_state["age"] > self.pre_action_state["age"] else 0.0

        # Action-specific rewards (if available)
        action_reward = self._calculate_action_reward()

        total_reward = resource_reward + health_reward + survival_reward + age_reward + action_reward

        self.last_action_reward = total_reward
        return total_reward

    def _calculate_state_reward(self) -> float:
        """Calculate reward based on current state (fallback method)."""
        if not self.core:
            return 0.0

        current_state = self._capture_state()

        # Normalize state values to [0, 1] to prevent unbounded rewards
        norm_resource = min(max(current_state["resource_level"] / self.config.max_resource_level, 0.0), 1.0)
        norm_health = min(max(current_state["health"] / self.config.max_health, 0.0), 1.0)
        norm_age = min(max(current_state["age"] / self.config.max_age, 0.0), 1.0)

        # Resource reward (normalized)
        resource_reward = norm_resource * self.config.resource_reward_scale

        # Health reward (normalized)
        health_reward = norm_health * self.config.health_reward_scale

        # Survival reward
        survival_reward = self.config.survival_bonus if current_state["alive"] else self.config.death_penalty

        # Age reward (normalized)
        age_reward = norm_age * self.config.age_bonus

        total_reward = resource_reward + health_reward + survival_reward + age_reward
        self.last_action_reward = total_reward
        return total_reward

    def _calculate_action_reward(self) -> float:
        """Calculate action-specific rewards based on last action."""
        if not self.core:
            return 0.0

        # This would need to be integrated with the action system
        # For now, return 0 - can be extended when action tracking is available
        return 0.0

    def _apply_reward(self, reward: float) -> None:
        """Apply reward to agent's cumulative total."""
        self.cumulative_reward += reward
        self.reward_history.append(reward)

        # Update agent's total reward if it has that attribute
        if self.core and hasattr(self.core, "total_reward"):
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
        """Reset all reward tracking (useful for new episodes)."""
        self.cumulative_reward = 0.0
        self.step_reward = 0.0
        self.reward_history = []
        self.last_action_reward = 0.0
        self.pre_action_state = None

        if self.core and hasattr(self.core, "total_reward"):
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

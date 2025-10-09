"""
Learning agent behavior using reinforcement learning.

Integrates with DecisionModule for intelligent action selection.
"""

from typing import TYPE_CHECKING, Optional, Any
from farm.core.agent.behaviors.base_behavior import IAgentBehavior

if TYPE_CHECKING:
    from farm.core.agent.core import AgentCore
    from farm.core.decision.decision import DecisionModule

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from farm.utils.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class LearningAgentBehavior(IAgentBehavior):
    """
    Reinforcement learning based agent behavior.

    This behavior integrates with DecisionModule to use learned policies
    for action selection. It supports various RL algorithms (DQN, PPO, SAC, etc.)
    and maintains learning state across episodes.

    The behavior:
    1. Observes current state from perception
    2. Selects action using DecisionModule
    3. Executes action via appropriate component
    4. Calculates reward based on state changes
    5. Updates DecisionModule with experience
    """

    def __init__(
        self,
        decision_module: Optional["DecisionModule"] = None,
        action_map: Optional[dict] = None,
    ):
        """
        Initialize learning behavior.

        Args:
            decision_module: DecisionModule for action selection (optional, can be set later)
            action_map: Map of action indices to (component_name, method_name) tuples

        Example:
            >>> behavior = LearningAgentBehavior(
            ...     decision_module=decision_module,
            ...     action_map={
            ...         0: ("movement", "random_move"),
            ...         1: ("combat", "attack"),
            ...         2: ("combat", "start_defense"),
            ...     }
            ... )
        """
        self._decision_module = decision_module
        self._action_map = action_map or self._default_action_map()

        # Learning state
        self._previous_state = None
        self._previous_action = None
        self._previous_reward = 0.0
        self._episode_reward = 0.0
        self._episode_count = 0
        self._turn_count = 0

    def _default_action_map(self) -> dict:
        """
        Get default action map.

        Maps action indices to component methods.

        Returns:
            dict: Map of index -> (component_name, method_name, args)
        """
        return {
            0: ("movement", "random_move", {}),
            1: ("movement", "move_by", {"delta_x": 5.0, "delta_y": 0.0}),
            2: ("movement", "move_by", {"delta_x": -5.0, "delta_y": 0.0}),
            3: ("movement", "move_by", {"delta_x": 0.0, "delta_y": 5.0}),
            4: ("movement", "move_by", {"delta_x": 0.0, "delta_y": -5.0}),
            5: ("combat", "attack", {}),
            6: ("combat", "start_defense", {}),
            7: ("reproduction", "reproduce", {}),
        }

    def set_decision_module(self, decision_module: "DecisionModule") -> None:
        """
        Set decision module after initialization.

        Args:
            decision_module: DecisionModule to use for action selection
        """
        self._decision_module = decision_module

    def execute_turn(self, agent: "AgentCore") -> None:
        """
        Execute one turn using reinforcement learning.

        Args:
            agent: The agent to control
        """
        if not self._decision_module:
            logger.warning(f"Agent {agent.agent_id} has no decision module, skipping turn")
            return

        self._turn_count += 1

        # Get current state observation
        current_state = self._create_state_observation(agent)

        # Select action using DecisionModule
        action_index = self._decision_module.decide_action(current_state)

        # Execute action
        action_success, action_result = self._execute_action(agent, action_index)

        # Calculate reward
        reward = self._calculate_reward(agent, action_success, action_result)
        self._episode_reward += reward

        # Update learning if we have previous state
        if self._previous_state is not None and HAS_TORCH:
            next_state = self._create_state_observation(agent)
            done = not agent.alive

            try:
                self._decision_module.update(
                    state=self._previous_state,
                    action=self._previous_action,
                    reward=self._previous_reward,
                    next_state=next_state,
                    done=done,
                )
            except Exception as e:
                logger.warning(f"Decision module update failed: {e}")

        # Store for next iteration
        self._previous_state = current_state
        self._previous_action = action_index
        self._previous_reward = reward

    def _create_state_observation(self, agent: "AgentCore") -> Any:
        """
        Create state observation for decision module using multi-channel perception.

        Combines information from various components into a single observation.

        Args:
            agent: Agent to observe

        Returns:
            State tensor or array for decision module
        """
        if not HAS_TORCH:
            # Return simple state without torch
            return [0.0] * 10

        # Get perception component
        perception = agent.get_component("perception")
        if perception:
            # Try to use multi-channel observation first
            observation = perception.get_observation()
            if observation is not None:
                try:
                    # Get multi-channel observation tensor
                    state = observation.tensor()  # Shape: (num_channels, 2R+1, 2R+1)
                    
                    # For now, maintain compatibility by flattening all channels
                    # Future: pass full multi-channel observation to decision module
                    if state.ndim == 3:
                        state = state.flatten()
                    elif state.ndim == 2:
                        state = state.unsqueeze(0).flatten()  # Add channel dimension then flatten
                    
                    return state
                except Exception as e:
                    # Fall back to old method if multi-channel fails
                    logger.warning(f"Multi-channel observation failed, falling back to grid: {e}")
            
            # Fallback to old perception grid method
            grid = perception.create_perception_grid()

            # Convert to tensor
            import torch
            if isinstance(grid, list):
                # Convert list to tensor
                import numpy as np
                grid = np.array(grid, dtype=np.float32)

            if hasattr(grid, 'shape'):
                # NumPy array
                state = torch.from_numpy(grid).float()
            else:
                # Fallback
                state = torch.zeros((11, 11), dtype=torch.float32)

            # Add batch dimension if needed
            if state.ndim == 2:
                state = state.unsqueeze(0)  # Add channel dimension

            return state

        # Fallback: simple state from components
        resource = agent.get_component("resource")
        combat = agent.get_component("combat")

        state_values = [
            agent.position[0] / 100.0,  # Normalized x
            agent.position[1] / 100.0,  # Normalized y
            resource.level / 100.0 if resource else 0.0,
            combat.health / combat.max_health if combat else 1.0,
        ]

        import torch
        return torch.tensor(state_values, dtype=torch.float32)

    def _execute_action(self, agent: "AgentCore", action_index: int) -> tuple[bool, dict]:
        """
        Execute selected action.

        Args:
            agent: Agent to execute action for
            action_index: Index of action to execute

        Returns:
            tuple: (success: bool, result: dict)
        """
        if action_index not in self._action_map:
            return False, {"error": f"Unknown action index {action_index}"}

        component_name, method_name, kwargs = self._action_map[action_index]

        # Get component
        component = agent.get_component(component_name)
        if not component:
            return False, {"error": f"Agent missing {component_name} component"}

        # Get method
        if not hasattr(component, method_name):
            return False, {"error": f"Component {component_name} has no method {method_name}"}

        method = getattr(component, method_name)

        # Execute method
        try:
            # Handle different return types
            result = method(**kwargs) if kwargs else method()

            # Normalize result
            if isinstance(result, dict):
                success = result.get("success", True)
            elif isinstance(result, bool):
                success = result
                result = {"success": success}
            else:
                success = True
                result = {"success": True, "value": result}

            return success, result

        except Exception as e:
            logger.warning(f"Action execution failed: {e}")
            return False, {"error": str(e)}

    def _calculate_reward(
        self,
        agent: "AgentCore",
        action_success: bool,
        action_result: dict
    ) -> float:
        """
        Calculate reward for the current turn.

        Reward components:
        - Resource level (positive)
        - Health level (positive)
        - Survival (positive)
        - Action success (small bonus)

        Args:
            agent: Agent that acted
            action_success: Whether action succeeded
            action_result: Result of action execution

        Returns:
            float: Calculated reward
        """
        reward = 0.0

        # Get components
        resource = agent.get_component("resource")
        combat = agent.get_component("combat")

        # Resource reward
        if resource:
            reward += resource.level * 0.01  # Small reward for having resources

        # Health reward
        if combat:
            reward += combat.health_ratio * 0.1  # Reward for being healthy

        # Survival reward
        if agent.alive:
            reward += 0.1
        else:
            reward -= 10.0  # Large penalty for death

        # Action success bonus
        if action_success:
            reward += 0.05

        return reward

    def reset(self) -> None:
        """
        Reset behavior for new episode.

        Clears episode-specific state but keeps learned policy.
        """
        self._previous_state = None
        self._previous_action = None
        self._previous_reward = 0.0
        self._episode_reward = 0.0
        self._episode_count += 1
        self._turn_count = 0

        # Reset decision module if it has reset method
        if self._decision_module and hasattr(self._decision_module, "reset"):
            self._decision_module.reset()

    def get_state(self) -> dict:
        """Get behavior state for serialization."""
        return {
            "episode_count": self._episode_count,
            "episode_reward": self._episode_reward,
            "turn_count": self._turn_count,
        }

    def load_state(self, state: dict) -> None:
        """Load behavior state from dictionary."""
        self._episode_count = state.get("episode_count", 0)
        self._episode_reward = state.get("episode_reward", 0.0)
        self._turn_count = state.get("turn_count", 0)
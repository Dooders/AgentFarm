"""Resource sharing module using Deep Q-Learning for intelligent cooperation.

This module implements a learning-based approach for agents to develop optimal
sharing strategies in a multi-agent environment. It considers factors like
resource levels, agent relationships, and environmental conditions to make
intelligent sharing decisions.

The sharing system uses Deep Q-Learning to learn optimal sharing policies based on:
- Current resource levels of the agent and nearby agents
- Historical cooperation patterns
- Environmental conditions and agent needs
- Reward signals from successful/failed sharing attempts

Key Components:
    - ShareConfig: Configuration parameters for sharing behavior
    - ShareActionSpace: Defines available sharing actions (NO_SHARE, SHARE_LOW, etc.)
    - ShareQNetwork: Neural network architecture for Q-value estimation
    - ShareModule: Main class handling sharing logic, learning, and decision making
    - share_action: Main function that executes sharing behavior
    - Experience Replay: Stores sharing interactions for learning (inherited from BaseDQNModule)
    - Reward System: Encourages beneficial sharing behavior with altruism bonuses

The module integrates with the broader AgentFarm system through:
- Database logging of sharing actions and outcomes
- Environment resource tracking
- Agent cooperation history maintenance
- Integration with other action modules (attack, gather, etc.)
"""

import logging
import random
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch

from farm.actions.base_dqn import BaseDQNModule, BaseQNetwork, SharedEncoder
from farm.actions.config import DEFAULT_SHARE_CONFIG, ShareConfig
from farm.core.action import action_registry

if TYPE_CHECKING:
    from farm.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ShareActionSpace:
    """Defines the available sharing actions and their corresponding amounts.

    This class provides a centralized definition of all possible sharing actions
    that agents can take. Each action corresponds to a different sharing amount,
    allowing for granular control over sharing behavior.

    Attributes:
        NO_SHARE: Action to not share any resources
        SHARE_LOW: Action to share minimum amount (1 resource)
        SHARE_MEDIUM: Action to share moderate amount (2 resources)
        SHARE_HIGH: Action to share larger amount (3 resources)
    """

    NO_SHARE: int = 0
    SHARE_LOW: int = 1  # Share minimum amount
    SHARE_MEDIUM: int = 2  # Share moderate amount
    SHARE_HIGH: int = 3  # Share larger amount


class ShareQNetwork(BaseQNetwork):
    """Neural network architecture for Q-value estimation in sharing decisions.

    This network takes the current state representation and outputs Q-values
    for each possible sharing action. The network architecture is optimized
    for the sharing domain with appropriate input and output dimensions.

    The input features represent the sharing context:
    - agent_resources: Normalized current resource level
    - nearby_agents: Normalized count of nearby agents
    - avg_neighbor_resources: Average resource level of nearby agents
    - min_neighbor_resources: Minimum resource level among neighbors
    - max_neighbor_resources: Maximum resource level among neighbors
    - cooperation_score: Historical cooperation score

    Args:
        input_dim: Number of input features (default: 6)
        hidden_size: Number of neurons in hidden layers (default: 64)
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_size: int = 64,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        # Input features: [agent_resources, nearby_agents, avg_neighbor_resources,
        #                 min_neighbor_resources, max_neighbor_resources, cooperation_score,
        #                 health_ratio, defensive_status]
        super().__init__(
            input_dim=input_dim,
            output_dim=4,  # NO_SHARE, SHARE_LOW, SHARE_MEDIUM, SHARE_HIGH
            hidden_size=hidden_size,
            shared_encoder=shared_encoder,
        )


class ShareModule(BaseDQNModule):
    """Main module for learning and executing intelligent sharing behavior.

    This class implements the core sharing logic using Deep Q-Learning. It manages
    the learning process, maintains cooperation history, and makes sharing decisions
    based on the current state and learned policies.

    The module integrates with the broader AgentFarm system by:
    - Logging sharing actions to the database
    - Updating environment resource counters
    - Maintaining cooperation history for future decisions
    - Providing reward signals for learning

    Attributes:
        cooperation_history: Dictionary mapping agent IDs to lists of cooperation scores
        q_network: Primary Q-network for action selection
        target_network: Target network for stable learning updates
    """

    def __init__(
        self,
        config: ShareConfig = DEFAULT_SHARE_CONFIG,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        """Initialize the ShareModule with configuration and device settings.

        Args:
            config: Configuration object containing sharing parameters
            device: PyTorch device for network computations (CPU/GPU)
            shared_encoder: Optional shared encoder for feature extraction
        """
        # Initialize parent class with share-specific network
        super().__init__(
            input_dim=8,  # State dimensions for sharing (matches SharedEncoder)
            output_dim=4,  # Number of sharing actions
            config=config,
            device=device,
        )
        self.cooperation_history = {}  # Track sharing interactions
        self._setup_action_space()

        # Initialize Q-network specific to sharing with shared encoder if provided
        self.q_network = ShareQNetwork(
            input_dim=8,
            hidden_size=config.dqn_hidden_size,
            shared_encoder=shared_encoder,
        ).to(device)
        self.target_network = ShareQNetwork(
            input_dim=8,
            hidden_size=config.dqn_hidden_size,
            shared_encoder=shared_encoder,
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

    def select_action(
        self, state: torch.Tensor, epsilon: Optional[float] = None
    ) -> int:
        """Select sharing action using epsilon-greedy strategy.

        This method implements the exploration-exploitation trade-off in action
        selection. With probability epsilon, a random action is chosen for
        exploration. Otherwise, the action with the highest Q-value is selected.

        Args:
            state: Current state tensor representing the sharing context
            epsilon: Optional override for exploration rate. If None, uses
                   the module's current epsilon value.

        Returns:
            int: Selected action index corresponding to ShareActionSpace constants
        """
        # Use the parent's select_action method which includes state caching
        return super().select_action(state, epsilon)

    def _setup_action_space(self) -> None:
        """Initialize the action space mapping for sharing decisions.

        This method creates a dictionary mapping action indices to their
        corresponding sharing amounts. The action space is used throughout
        the module for consistent action handling.
        """
        self.action_space = {
            ShareActionSpace.NO_SHARE: 0,
            ShareActionSpace.SHARE_LOW: 1,
            ShareActionSpace.SHARE_MEDIUM: 2,
            ShareActionSpace.SHARE_HIGH: 3,
        }

    def get_share_decision(
        self, agent: "BaseAgent", state: torch.Tensor
    ) -> Tuple[int, Optional["BaseAgent"], int]:
        """Determine the complete sharing decision including action, target, and amount.

        This method orchestrates the sharing decision process by:
        1. Selecting an action using the learned policy
        2. Finding suitable nearby agents if sharing is chosen
        3. Selecting the best target based on need and cooperation history
        4. Calculating the appropriate sharing amount

        Args:
            agent: The agent making the sharing decision
            state: Current state tensor representing the sharing context

        Returns:
            Tuple containing:
            - int: Selected action index from ShareActionSpace
            - Optional[BaseAgent]: Target agent for sharing (None if NO_SHARE)
            - int: Amount of resources to share (0 if NO_SHARE)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)

        action = self.select_action(state)

        if action == ShareActionSpace.NO_SHARE:
            return action, None, 0

        # Find potential sharing targets
        nearby_agents = self._get_nearby_agents(agent)
        if not nearby_agents:
            return ShareActionSpace.NO_SHARE, None, 0

        # Select target based on need and cooperation history
        target = self._select_target(agent, nearby_agents)
        share_amount = self._calculate_share_amount(agent, action)

        return action, target, share_amount

    def _get_nearby_agents(self, agent: "BaseAgent") -> List["BaseAgent"]:
        """Find all agents within sharing range, excluding the agent itself.

        This method queries the environment for nearby agents and filters out
        the agent itself to prevent self-sharing, which would be meaningless.

        Args:
            agent: The agent looking for nearby sharing targets

        Returns:
            List of BaseAgent objects within sharing range, excluding the agent itself
        """
        from typing import cast

        share_config = cast(ShareConfig, self.config)
        nearby_agents = agent.environment.get_nearby_agents(
            agent.position, share_config.range
        )
        # Filter out self to prevent agents from sharing with themselves
        return [a for a in nearby_agents if a.agent_id != agent.agent_id]

    def _select_target(
        self, agent: "BaseAgent", nearby_agents: List["BaseAgent"]
    ) -> "BaseAgent":
        """Select the best target agent based on need and cooperation history.

        This method implements a weighted selection algorithm that prioritizes:
        1. Agents with low resource levels (in need)
        2. Agents with positive cooperation history
        3. Deterministic selection among suitable candidates

        The selection uses probability weights to balance immediate need with
        long-term cooperation patterns.

        Args:
            agent: The agent making the selection
            nearby_agents: List of potential target agents

        Returns:
            BaseAgent: The selected target agent for sharing
        """
        # Calculate selection weights based on need and past cooperation
        weights = []
        for target in nearby_agents:
            weight = 1.0
            # Increase weight for agents with low resources
            if (
                target.config
                and target.resource_level < target.config.starvation_threshold
            ):
                weight *= 2.0
            # Consider past cooperation
            coop_score = self._get_cooperation_score(target.agent_id)
            weight *= 1.0 + coop_score
            weights.append(weight)

        # For deterministic behavior, select the agent with the highest weight
        # In case of ties, select the first one
        max_weight = max(weights)
        chosen_index = weights.index(max_weight)
        return nearby_agents[chosen_index]

    def _calculate_share_amount(self, agent: "BaseAgent", action: int) -> int:
        """Calculate the amount of resources to share based on action and availability.

        This method determines the actual sharing amount considering:
        - The selected action (SHARE_LOW, SHARE_MEDIUM, SHARE_HIGH)
        - The agent's current resource level
        - Minimum sharing requirements

        Args:
            agent: The agent performing the sharing action
            action: The selected action index from ShareActionSpace

        Returns:
            int: Amount of resources to share (0 if action is NO_SHARE or insufficient resources)
        """
        if action == ShareActionSpace.NO_SHARE:
            return 0

        from typing import cast

        share_config = cast(ShareConfig, self.config)
        available = max(0, agent.resource_level - share_config.min_amount)
        share_amounts = {
            ShareActionSpace.SHARE_LOW: min(1, available),
            ShareActionSpace.SHARE_MEDIUM: min(2, available),
            ShareActionSpace.SHARE_HIGH: min(3, available),
        }
        return share_amounts.get(action, 0)

    def _get_cooperation_score(self, agent_id: str) -> float:
        """Calculate the cooperation score for an agent based on interaction history.

        This method computes a running average of cooperation scores for a
        specific agent. Positive scores indicate cooperative behavior, while
        negative scores indicate uncooperative behavior.

        Args:
            agent_id: The ID of the agent to get cooperation score for

        Returns:
            float: Cooperation score between -1.0 and 1.0, or 0.0 if no history
        """
        from typing import cast

        share_config = cast(ShareConfig, self.config)
        if agent_id not in self.cooperation_history:
            return 0.0
        return sum(
            self.cooperation_history[agent_id][-share_config.cooperation_memory :]
        ) / len(self.cooperation_history[agent_id][-share_config.cooperation_memory :])

    def update_cooperation(self, agent_id: str, cooperative: bool) -> None:
        """Update the cooperation history for a specific agent.

        This method records the outcome of a sharing interaction to maintain
        the cooperation history used for future target selection decisions.

        Args:
            agent_id: The ID of the agent whose cooperation history to update
            cooperative: Whether the interaction was cooperative (True) or not (False)
        """
        if agent_id not in self.cooperation_history:
            self.cooperation_history[agent_id] = []
        self.cooperation_history[agent_id].append(1.0 if cooperative else -1.0)


def share_action(agent: "BaseAgent") -> None:
    """Execute a sharing action using the agent's learned sharing policy.

    This is the main entry point for sharing behavior. The function:
    1. Gathers the current state representation
    2. Uses the agent's ShareModule to make a sharing decision
    3. Executes the sharing if conditions are met
    4. Updates the environment and logs the action
    5. Calculates and applies rewards for learning

    The function handles both successful and failed sharing attempts,
    logging appropriate information to the database for analysis.

    Args:
        agent: The agent performing the sharing action
    """
    # Get state information
    state = _get_share_state(agent)
    initial_resources = agent.resource_level

    # Get sharing decision
    state_tensor = torch.FloatTensor(state).to(agent.share_module.device)
    action, target, share_amount = agent.share_module.get_share_decision(
        agent, state_tensor
    )

    if not target or share_amount <= 0 or agent.resource_level < share_amount:
        # Log failed share action
        if agent.environment.db is not None:
            agent.environment.db.logger.log_agent_action(
                step_number=agent.environment.time,
                agent_id=agent.agent_id,
                action_type="share",
                resources_before=initial_resources,
                resources_after=initial_resources,
                reward=DEFAULT_SHARE_CONFIG.failure_penalty,
                details={
                    "success": False,
                    "reason": "invalid_share_conditions",
                    "attempted_amount": share_amount,
                },
            )
        return

    # Execute sharing
    target_initial_resources = target.resource_level
    agent.resource_level -= share_amount
    target.resource_level += share_amount

    # Update environment's resources_shared counter
    agent.environment.resources_shared += share_amount
    agent.environment.resources_shared_this_step += share_amount

    # Calculate reward
    reward = _calculate_share_reward(agent, target, share_amount)
    agent.total_reward += reward

    # Update cooperation history
    agent.share_module.update_cooperation(target.agent_id, True)

    # Log successful share action
    if agent.environment.db is not None:
        agent.environment.db.logger.log_agent_action(
            step_number=agent.environment.time,
            agent_id=agent.agent_id,
            action_type="share",
            action_target_id=target.agent_id,  # Agent IDs are strings, not integers
            resources_before=initial_resources,
            resources_after=agent.resource_level,
            reward=reward,
            details={
                "success": True,
                "amount_shared": share_amount,
                "target_resources_before": target_initial_resources,
                "target_resources_after": target.resource_level,
                "target_was_starving": target.config
                and target_initial_resources < target.config.starvation_threshold,
            },
        )


def _get_share_state(agent: "BaseAgent") -> List[float]:
    """Create a normalized state representation for sharing decisions.

    This function constructs an 8-dimensional state vector that captures
    the current sharing context. All values are normalized to [0, 1] range
    for consistent neural network input.

    The state includes:
    - Agent's current resource level (normalized)
    - Density of nearby agents (normalized)
    - Average resource level of nearby agents (normalized)
    - Minimum resource level among nearby agents (normalized)
    - Maximum resource level among nearby agents (normalized)
    - Historical cooperation score
    - Agent's current health ratio
    - Agent's defensive status

    Args:
        agent: The agent for whom to create the state representation

    Returns:
        List[float]: 8-dimensional state vector for sharing decisions
    """
    nearby_agents = agent.environment.get_nearby_agents(
        agent.position, DEFAULT_SHARE_CONFIG.range
    )

    neighbor_resources = (
        [a.resource_level for a in nearby_agents] if nearby_agents else [0]
    )

    # Use ShareConfig's max_resources for normalization
    max_resources = DEFAULT_SHARE_CONFIG.max_resources

    return [
        agent.resource_level / max_resources,  # Normalized agent resources
        len(nearby_agents) / len(agent.environment.agents),  # Normalized nearby agents
        float(np.mean(neighbor_resources)) / max_resources,  # Avg neighbor resources
        min(neighbor_resources) / max_resources,  # Min neighbor resources
        max(neighbor_resources) / max_resources,  # Max neighbor resources
        agent.share_module._get_cooperation_score(agent.agent_id),  # Cooperation score
        agent.current_health / agent.starting_health,  # Health ratio
        float(agent.is_defending),  # Defensive status
    ]


def _calculate_share_reward(
    agent: "BaseAgent", target: "BaseAgent", amount: int
) -> float:
    """Calculate the reward for a sharing action based on context and outcome.

    This function implements the reward system for sharing behavior, considering:
    - Base reward for successful sharing
    - Altruism bonus for helping agents in need
    - Scaling based on amount shared

    The reward encourages beneficial sharing behavior while discouraging
    wasteful or harmful sharing actions.

    Args:
        agent: The agent performing the sharing action
        target: The agent receiving the shared resources
        amount: The amount of resources being shared

    Returns:
        float: Calculated reward value for the sharing action
    """
    reward = DEFAULT_SHARE_CONFIG.success_reward

    # Add altruism bonus if target was in need
    if target.config and target.resource_level < target.config.starvation_threshold:
        reward += DEFAULT_SHARE_CONFIG.altruism_bonus

    # Scale reward based on amount shared
    reward *= amount / DEFAULT_SHARE_CONFIG.min_amount

    return reward


# Register the action at the end of the file after the function is defined
action_registry.register("share", 0.2, share_action)

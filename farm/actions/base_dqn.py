"""Base DQN module providing common functionality for learning-based actions.

This module implements a Deep Q-Network (DQN) architecture with experience replay,
target networks, and epsilon-greedy exploration. It serves as a foundation for
agent learning modules that require Q-learning with neural network function approximation.

Key Features:
- Experience replay buffer for stable learning
- Target network for reducing overestimation bias
- Soft target network updates
- Epsilon-greedy exploration strategy
- Gradient clipping for training stability
- State caching for performance optimization
- Comprehensive logging and database integration

Example:
    ```python
    config = BaseDQNConfig(
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    dqn_module = BaseDQNModule(
        input_dim=10,
        output_dim=4,
        config=config
    )

    # Store experience
    dqn_module.store_experience(state, action, reward, next_state, done)

    # Train on batch
    batch = sample_from_memory()
    loss = dqn_module.train(batch)

    # Select action
    action = dqn_module.select_action(state_tensor)
    ```
"""

import logging
import random
from collections import deque
from typing import TYPE_CHECKING, Any, Deque, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if TYPE_CHECKING:
    from farm.database.database import SimulationDatabase


class BaseDQNConfig:
    """Configuration class for DQN modules.

    This class defines all hyperparameters and settings for DQN-based learning
    modules. It provides sensible defaults while allowing customization for
    specific use cases.

    Attributes:
        target_update_freq (int): Frequency of hard target network updates (unused with soft updates)
        memory_size (int): Maximum size of experience replay buffer
        learning_rate (float): Learning rate for the Adam optimizer
        gamma (float): Discount factor for future rewards
        epsilon_start (float): Initial exploration rate for epsilon-greedy strategy
        epsilon_min (float): Minimum exploration rate
        epsilon_decay (float): Decay rate for exploration (multiplied each step)
        dqn_hidden_size (int): Number of neurons in hidden layers
        batch_size (int): Number of experiences to sample for training
        tau (float): Soft update parameter for target network (0 < tau < 1)
        seed (Optional[int]): Random seed for reproducibility
    """

    target_update_freq: int = 100
    memory_size: int = 10000
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    dqn_hidden_size: int = 64
    batch_size: int = 32
    tau: float = 0.005
    seed: Optional[int] = None  # Seed for reproducibility


class SharedEncoder(nn.Module):
    """
    Shared encoder for all action modules.

    This class is used to extract common features across all action modules.
    It is a simple feedforward neural network with one hidden layer.

    Attributes:
        fc (nn.Linear): Fully connected layer for common features
    """
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_size)  # Shared layer for common features like position/health

    def forward(self, x):
        return F.relu(self.fc(x))  # Shared features


class BaseQNetwork(nn.Module):
    """Neural network architecture for Q-value approximation.

    A feedforward neural network with three fully connected layers, layer
    normalization, ReLU activation, and dropout for regularization. Designed
    to approximate Q-values for state-action pairs.

    Architecture:
        Input -> Linear -> LayerNorm -> ReLU -> Dropout ->
        Linear -> LayerNorm -> ReLU -> Dropout ->
        Linear -> Output

    Attributes:
        network (nn.Sequential): The complete neural network architecture
        shared_encoder (Optional[SharedEncoder]): Shared encoder for common features
        effective_input (int): Effective input dimension based on shared encoder
        network (nn.Sequential): The complete neural network architecture

    Usage:
        shared_encoder = SharedEncoder(input_dim=8, hidden_size=64)
        q_network = BaseQNetwork(input_dim=8, output_dim=4, hidden_size=64, shared_encoder=shared_encoder)
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, shared_encoder: Optional[SharedEncoder] = None) -> None:
        """Initialize the Q-network.
        
        If a shared encoder is provided, the input dimension is reduced to the hidden size.
        Otherwise, the input dimension is used directly.

        Parameters:
            input_dim (int): Dimension of the input state vector
            output_dim (int): Number of possible actions (output dimension)
            hidden_size (int): Number of neurons in hidden layers
        """
        super().__init__()
        self.shared_encoder = shared_encoder
        effective_input = hidden_size if shared_encoder else input_dim
        self.network = nn.Sequential(
            nn.Linear(effective_input, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, output_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier/Glorot initialization.

        This method ensures proper weight initialization for stable training.
        Xavier initialization is particularly effective for networks with
        ReLU activation functions.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Handles both single samples and batches. For single samples,
        adds a batch dimension, processes through the network, and
        removes the batch dimension from the output.

        Parameters:
            x (torch.Tensor): Input tensor of shape (input_dim,) or (batch_size, input_dim)

        Returns:
            torch.Tensor: Q-values for all actions, shape (output_dim,) or (batch_size, output_dim)
        """
        if self.shared_encoder:
            x = self.shared_encoder(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
            result = self.network(x)
            return result.squeeze(0)
        return self.network(x)


class BaseDQNModule:
    """Base class for DQN-based learning modules.

    This class provides a complete implementation of Deep Q-Learning with
    experience replay, target networks, and epsilon-greedy exploration.
    It serves as a foundation for specific learning modules like movement,
    combat, or other agent behaviors.

    Key Features:
        - Experience replay buffer for stable learning
        - Target network with soft updates to reduce overestimation bias
        - Epsilon-greedy exploration with decay
        - Gradient clipping for training stability
        - State caching for performance optimization
        - Comprehensive logging and database integration
        - Reproducible training with seed setting

    Attributes:
        device (torch.device): Device to run computations on (CPU/GPU)
        config (BaseDQNConfig): Configuration object with hyperparameters
        db (Optional[SimulationDatabase]): Database for logging experiences
        module_id (int): Unique identifier for this module instance
        logger: Logger instance for debugging and monitoring
        output_dim (int): Number of possible actions
        q_network (BaseQNetwork): Main Q-network for action selection
        target_network (BaseQNetwork): Target network for stable learning
        optimizer (torch.optim.Adam): Optimizer for network training
        memory (Deque): Experience replay buffer
        epsilon (float): Current exploration rate
        losses (List[float]): History of training losses
        episode_rewards (List[float]): History of episode rewards
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: BaseDQNConfig,
        device: torch.device = DEVICE,
        db: Optional["SimulationDatabase"] = None,
    ) -> None:
        """Initialize the DQN module.

        Parameters:
            input_dim (int): Dimension of the input state vector
            output_dim (int): Number of possible actions
            config (BaseDQNConfig): Configuration object with hyperparameters
            device (torch.device): Device to run computations on (default: auto-detect)
            db (Optional[SimulationDatabase]): Database for logging experiences
        """
        self.device = device
        self.config = config
        self.db = db
        self.module_id = id(self.__class__)
        self.logger = db.logger if db is not None else None
        self.output_dim = output_dim

        # Set seed if provided for reproducibility
        if config.seed is not None:
            self._set_seed(config.seed)

        self._setup_networks(input_dim, output_dim, config)
        self._setup_training(config)
        self.losses = []
        self.episode_rewards = []
        self.pending_experiences = []

        # Add caching for state tensors
        self._state_cache = {}
        self._max_cache_size = 100

    def _set_seed(self, seed: int) -> None:
        """Set seeds for all random number generators to ensure reproducibility.

        This method sets seeds for Python's random module, NumPy, and PyTorch
        to ensure that training runs are reproducible when the same seed is used.

        Parameters:
            seed (int): The seed value to use for all random number generators
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For completely deterministic results, uncomment the following:
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def _setup_networks(
        self, input_dim: int, output_dim: int, config: BaseDQNConfig
    ) -> None:
        """Initialize Q-networks and optimizer.

        Creates the main Q-network, target network, and Adam optimizer.
        The target network is initialized with the same weights as the main network.

        Parameters:
            input_dim (int): Dimension of the input state vector
            output_dim (int): Number of possible actions
            config (BaseDQNConfig): Configuration object with hyperparameters
        """
        self.q_network = BaseQNetwork(input_dim, output_dim, config.dqn_hidden_size).to(
            self.device
        )
        self.target_network = BaseQNetwork(
            input_dim, output_dim, config.dqn_hidden_size
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config.learning_rate
        )
        self.criterion = nn.SmoothL1Loss()

    def _setup_training(self, config: BaseDQNConfig) -> None:
        """Initialize training parameters and experience replay buffer.

        Sets up the experience replay buffer, exploration parameters,
        and training hyperparameters.

        Parameters:
            config (BaseDQNConfig): Configuration object with hyperparameters
        """
        self.memory: Deque = deque(maxlen=config.memory_size)
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.tau = config.tau
        self.steps = 0

    def _log_experience(
        self,
        step_number: int,
        agent_id: str,
        module_type: str,
        action_taken: int,
        action_taken_mapped: str,
        reward: float,
    ) -> None:
        """Log a learning experience to the database if available.

        This method logs detailed information about each learning experience
        to the database for analysis and debugging purposes.

        Parameters:
            step_number (int): Current simulation step
            agent_id (str): ID of the agent that took the action
            module_type (str): Type of DQN module (e.g., 'movement', 'combat')
            action_taken (int): Numeric action taken
            action_taken_mapped (str): Human-readable action description
            reward (float): Reward received for this experience
        """
        if self.db is not None:
            self.db.logger.log_learning_experience(
                step_number=step_number,
                agent_id=agent_id,
                module_type=module_type,
                module_id=self.module_id,
                action_taken=action_taken,
                action_taken_mapped=action_taken_mapped,
                reward=reward,
            )

    def store_experience(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        step_number: Optional[int] = None,
        agent_id: Optional[str] = None,
        module_type: Optional[str] = None,
        module_id: Optional[int] = None,
        action_taken_mapped: Optional[int] = None,
    ) -> None:
        """Store experience in replay memory and optionally log to database.

        This method stores a complete experience tuple (state, action, reward,
        next_state, done) in the replay buffer for later training. It also
        logs the experience to the database if logging is enabled.

        Parameters:
            state (torch.Tensor): Current state observation
            action (int): Action taken in the current state
            reward (float): Reward received for the action
            next_state (torch.Tensor): State observed after taking the action
            done (bool): Whether the episode ended after this action
            step_number (Optional[int]): Current simulation step for logging
            agent_id (Optional[str]): ID of the agent for logging
            module_type (Optional[str]): Type of DQN module for logging
            module_id (Optional[int]): Module ID for logging
            action_taken_mapped (Optional[int]): Human-readable action for logging
        """
        self.memory.append((state, action, reward, next_state, done))
        self.episode_rewards.append(reward)

        if (
            self.logger is not None
            and step_number is not None
            and agent_id is not None
            and module_type is not None
            and module_id is not None
            and action_taken_mapped is not None
        ):
            self.logger.log_learning_experience(
                step_number=step_number,
                agent_id=agent_id,
                module_type=module_type,
                module_id=module_id,
                action_taken=action,
                action_taken_mapped=str(action_taken_mapped),
                reward=reward,
            )

    def train(
        self,
        batch: list,
        step_number: Optional[int] = None,
        agent_id: Optional[str] = None,
        module_type: Optional[str] = None,
    ) -> Optional[float]:
        """Train the network using a batch of experiences.

        This method implements the core DQN training algorithm with the following
        features:
        - Double Q-Learning to reduce overestimation bias
        - Gradient clipping for training stability
        - Soft target network updates
        - Epsilon decay for exploration
        - Loss tracking and history

        Parameters:
            batch (list): List of experience tuples (state, action, reward, next_state, done)
            step_number (Optional[int]): Current simulation step for logging
            agent_id (Optional[str]): ID of the agent for logging
            module_type (Optional[str]): Type of DQN module for logging

        Returns:
            Optional[float]: Loss value if training occurred, None if batch too small

        Note:
            Training only occurs if the batch size is at least config.batch_size.
            The method uses Double Q-Learning where the main network selects actions
            and the target network evaluates them.
        """
        if len(batch) < self.config.batch_size:
            return None

        # Unpack batch
        states = torch.stack([x[0] for x in batch])
        actions = torch.tensor([x[1] for x in batch], device=self.device).unsqueeze(1)
        rewards = torch.tensor([x[2] for x in batch], device=self.device)
        next_states = torch.stack([x[3] for x in batch])
        dones = torch.tensor(
            [x[4] for x in batch], device=self.device, dtype=torch.float
        )

        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions)

        # Compute target Q values using Double Q-Learning
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = (
                rewards.unsqueeze(1)
                + (1 - dones.unsqueeze(1)) * self.gamma * next_q_values
            )

        # Compute loss and update network
        loss = self.criterion(current_q_values, target_q_values)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self._soft_update_target_network()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        loss_value = loss.item()
        self.losses.append(loss_value)

        #! Need to fix this to work with all modules
        # last_experience = batch[-1]
        # self._log_experience(
        #     step_number=step_number,
        #     agent_id=agent_id,
        #     module_type=module_type,
        #     action_taken=self.previous_action,
        #     action_taken_mapped=self.previous_action_mapped,
        #     reward=reward,
        # )

        return loss_value

    def _soft_update_target_network(self) -> None:
        """Soft update target network weights using tau parameter.

        This method performs a soft update of the target network weights using
        the formula: target_param = tau * local_param + (1 - tau) * target_param.
        This approach provides more stable learning compared to hard updates.
        """
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def get_state_dict(self) -> dict[str, Any]:
        """Get the current state of the DQN module for saving/loading.

        This method returns a complete state dictionary containing all necessary
        information to restore the module's state, including network weights,
        optimizer state, training parameters, and learning history.

        Returns:
            dict[str, Any]: State dictionary containing:
                - q_network_state: Main network weights
                - target_network_state: Target network weights
                - optimizer_state: Optimizer state and momentum
                - epsilon: Current exploration rate
                - steps: Training step counter
                - losses: History of training losses
                - episode_rewards: History of episode rewards
                - seed: Random seed for reproducibility
                - action_weights: Action weights (if applicable)
        """
        state_dict = {
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
            "losses": self.losses,
            "episode_rewards": self.episode_rewards,
            "seed": self.config.seed,
        }

        # Add action weights if this is a module related to action selection
        if hasattr(self, "action_weights"):
            state_dict["action_weights"] = self.action_weights

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a state dictionary to restore the DQN module's state.

        This method restores the complete state of the module from a previously
        saved state dictionary, allowing for checkpoint loading and model transfer.

        Parameters:
            state_dict (dict[str, Any]): State dictionary containing network weights
                and training parameters. Must contain the same keys as returned by
                get_state_dict().

        Note:
            The state_dict must contain all the keys returned by get_state_dict().
            Missing keys will cause the method to fail.
        """
        self.q_network.load_state_dict(state_dict["q_network_state"])
        self.target_network.load_state_dict(state_dict["target_network_state"])
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        self.epsilon = state_dict["epsilon"]
        self.steps = state_dict["steps"]
        self.losses = state_dict["losses"]
        self.episode_rewards = state_dict["episode_rewards"]

        # Load action weights if present
        if "action_weights" in state_dict and hasattr(self, "action_weights"):
            self.action_weights = state_dict["action_weights"]

    def cleanup(self):
        """Clean up pending experiences and flush to database.

        This method ensures that any pending experiences are properly logged
        to the database before the module is destroyed. It's called automatically
        during object destruction but can also be called manually.
        """
        if self.db is not None and self.pending_experiences:
            try:
                # Note: batch_log_learning_experiences method doesn't exist on SimulationDatabase
                # This would need to be implemented if batch logging is required
                logger.warning(
                    "batch_log_learning_experiences not implemented - skipping cleanup"
                )
                self.pending_experiences = []
            except Exception as e:
                logger.error(f"Error cleaning up DQN module experiences: {e}")

    def __del__(self):
        """Ensure cleanup on deletion.

        This destructor ensures that pending experiences are properly logged
        before the object is garbage collected.
        """
        self.cleanup()

    def select_action(
        self, state_tensor: torch.Tensor, epsilon: Optional[float] = None
    ) -> int:
        """Select an action using epsilon-greedy strategy with state caching.

        This method implements epsilon-greedy action selection with performance
        optimizations including state caching to avoid redundant network
        evaluations for repeated states.

        Parameters:
            state_tensor (torch.Tensor): Current state observation
            epsilon (Optional[float]): Exploration rate override. If None, uses
                the module's current epsilon value

        Returns:
            int: Selected action index (0 to output_dim - 1)

        Note:
            The method uses a simple caching mechanism based on tensor hash
            to avoid redundant network evaluations for repeated states.
            The cache has a maximum size to prevent memory issues.
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Use epsilon-greedy strategy
        # Type assertion to help linter understand epsilon is not None
        assert epsilon is not None
        if random.random() < epsilon:
            return random.randint(0, self.output_dim - 1)

        # Cache tensor hash for repeated states
        state_hash = hash(state_tensor.cpu().numpy().tobytes())

        if state_hash in self._state_cache:
            return self._state_cache[state_hash]

        with torch.no_grad():
            action = self.q_network(state_tensor).argmax().item()

        # Update cache with LRU behavior
        if len(self._state_cache) >= self._max_cache_size:
            # Remove a random item if cache is full
            self._state_cache.pop(next(iter(self._state_cache)))

        self._state_cache[state_hash] = action
        return action

"""
Base Deep Q-Network module for AgentFarm action learning.

This module provides the foundational DQN implementation that all action modules
inherit from. Updated to use the new profile-based configuration system.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Any, Dict, List, Optional

from farm.core.profiles import DQNProfile

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Type hint for database import (avoid circular imports)
try:
    from farm.database.database import SimulationDatabase
except ImportError:
    SimulationDatabase = None


class SharedEncoder(nn.Module):
    """
    Shared encoder for all action modules.
    
    This encoder processes the input state and can be shared across multiple
    action modules to improve learning efficiency and reduce memory usage.
    """
    
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.encoder(x)


class BaseQNetwork(nn.Module):
    """
    Base Q-Network architecture for action value estimation.
    
    This network takes state representations and outputs Q-values for each
    possible action. Can optionally use a shared encoder for efficiency.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 64, 
                 shared_encoder: Optional[SharedEncoder] = None) -> None:
        super().__init__()
        
        self.shared_encoder = shared_encoder
        
        if shared_encoder is not None:
            # Use shared encoder + action-specific head
            self.network = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dim)
            )
        else:
            # Full independent network
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), 
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dim)
            )
            
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.shared_encoder is not None:
            x = self.shared_encoder(x)
        return self.network(x)


class BaseDQNModule:
    """
    Base class for DQN-based learning modules.
    
    This class provides a complete implementation of Deep Q-Learning with
    experience replay, target networks, and epsilon-greedy exploration.
    Now uses the new profile-based configuration system.
    
    Key Features:
        - Experience replay buffer for stable learning
        - Target network with soft updates to reduce overestimation bias  
        - Epsilon-greedy exploration with decay
        - Gradient clipping for training stability
        - State caching for performance optimization
        - Comprehensive logging and database integration
        - Profile-based configuration for easy customization
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int, 
        dqn_profile: DQNProfile,
        device: torch.device = DEVICE,
        db: Optional["SimulationDatabase"] = None,
        shared_encoder: Optional[SharedEncoder] = None,
    ) -> None:
        """Initialize the DQN module.
        
        Parameters:
            input_dim (int): Dimension of the input state vector
            output_dim (int): Number of possible actions
            dqn_profile (DQNProfile): Configuration profile with hyperparameters
            device (torch.device): Device to run computations on (default: auto-detect)
            db (Optional[SimulationDatabase]): Database for logging experiences
            shared_encoder (Optional[SharedEncoder]): Shared encoder for efficiency
        """
        self.device = device
        self.profile = dqn_profile
        self.db = db
        self.module_id = id(self.__class__)
        self.logger = db.logger if db is not None else None
        self.output_dim = output_dim
        
        # Set seed if provided for reproducibility
        if dqn_profile.seed is not None:
            self._set_seed(dqn_profile.seed)
        
        self._setup_networks(input_dim, output_dim, dqn_profile, shared_encoder)
        self._setup_training(dqn_profile)
        self.losses = []
        self.episode_rewards = []
        self.pending_experiences = []
        
        # Add caching for state tensors
        self._state_cache = {}
        self._max_cache_size = 100
    
    def _set_seed(self, seed: int) -> None:
        """Set seeds for all random number generators to ensure reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def _setup_networks(
        self, 
        input_dim: int, 
        output_dim: int, 
        profile: DQNProfile,
        shared_encoder: Optional[SharedEncoder] = None
    ) -> None:
        """Initialize Q-networks and optimizer."""
        # Create Q-networks
        self.q_network = BaseQNetwork(
            input_dim, output_dim, profile.hidden_size, shared_encoder
        ).to(self.device)
        
        self.target_network = BaseQNetwork(
            input_dim, output_dim, profile.hidden_size, shared_encoder  
        ).to(self.device)
        
        # Initialize target network with same weights
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=profile.learning_rate)
    
    def _setup_training(self, profile: DQNProfile) -> None:
        """Initialize training components."""
        self.memory = deque(maxlen=profile.memory_size)
        self.epsilon = profile.epsilon_start
        self.steps_done = 0
        self.training_step = 0
    
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
        action_taken_mapped: Optional[str] = None,
    ) -> None:
        """Store experience in replay buffer."""
        experience = (
            state.cpu(),
            action,
            reward,
            next_state.cpu() if next_state is not None else None,
            done
        )
        self.memory.append(experience)
        
        # Log to database if available
        if self.db and step_number and agent_id:
            self._log_experience(
                step_number, agent_id, module_type or "unknown",
                action, action_taken_mapped or str(action), reward
            )
    
    def _log_experience(
        self,
        step_number: int,
        agent_id: str, 
        module_type: str,
        action_taken: int,
        action_taken_mapped: str,
        reward: float,
    ) -> None:
        """Log experience to database."""
        if self.db is None:
            return
            
        try:
            self.db.log_learning_experience(
                step_number=step_number,
                agent_id=agent_id,
                module_type=module_type,
                module_id=self.module_id,
                action_taken=action_taken, 
                action_taken_mapped=action_taken_mapped,
                reward=reward
            )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to log experience: {e}")
    
    def train(
        self,
        batch: Optional[List] = None,
        step_number: Optional[int] = None,
        agent_id: Optional[str] = None,
        module_type: Optional[str] = None,
    ) -> Optional[float]:
        """Train the network using experience replay."""
        if len(self.memory) < self.profile.batch_size:
            return None
            
        # Sample batch from memory
        if batch is None:
            batch = random.sample(list(self.memory), self.profile.batch_size)
        
        # Separate batch components
        states = torch.stack([exp[0] for exp in batch]).to(self.device)
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float).to(self.device)
        next_states = torch.stack([exp[3] for exp in batch if exp[3] is not None]).to(self.device)
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = torch.zeros(self.profile.batch_size).to(self.device)
            if len(next_states) > 0:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).max(1)[1].detach()
                next_q_values[~dones] = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            
            target_q_values = rewards + (self.profile.gamma * next_q_values)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Soft update target network
        self._soft_update_target_network()
        
        # Update epsilon
        self.epsilon = max(
            self.profile.epsilon_min,
            self.epsilon * self.profile.epsilon_decay
        )
        
        self.training_step += 1
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def _soft_update_target_network(self) -> None:
        """Soft update target network weights."""
        for target_param, local_param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.profile.tau * local_param.data + (1.0 - self.profile.tau) * target_param.data
            )
    
    def select_action(
        self, state_tensor: torch.Tensor, epsilon: Optional[float] = None
    ) -> int:
        """Select action using epsilon-greedy policy."""
        if epsilon is None:
            epsilon = self.epsilon
            
        if random.random() < epsilon:
            return random.randrange(self.output_dim)
        
        with torch.no_grad():
            state_tensor = state_tensor.to(self.device)
            q_values = self.q_network(state_tensor.unsqueeze(0))
            return q_values.max(1)[1].item()
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get module state for serialization."""
        return {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(), 
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
            "training_step": self.training_step,
            "losses": self.losses,
            "episode_rewards": self.episode_rewards
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load module state from serialization."""
        self.q_network.load_state_dict(state_dict["q_network"])
        self.target_network.load_state_dict(state_dict["target_network"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.epsilon = state_dict["epsilon"]
        self.steps_done = state_dict["steps_done"] 
        self.training_step = state_dict["training_step"]
        self.losses = state_dict["losses"]
        self.episode_rewards = state_dict["episode_rewards"]
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'memory'):
            self.memory.clear()
        if hasattr(self, '_state_cache'):
            self._state_cache.clear()
        
        # Clear GPU memory if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.cleanup()
        except:
            pass  # Ignore cleanup errors during destruction
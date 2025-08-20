"""
Reusable configuration profiles for AgentFarm.

This module contains predefined profiles that can be mixed and matched to create
different simulation configurations without repetition.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import copy


@dataclass
class DQNProfile:
    """Deep Q-Learning hyperparameter profile.
    
    These profiles can be reused across all action modules to ensure
    consistent learning behavior and reduce configuration duplication.
    """
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    memory_size: int = 10000
    batch_size: int = 32
    hidden_size: int = 64
    tau: float = 0.005
    target_update_freq: int = 100
    seed: Optional[int] = None
    
    def with_overrides(self, **kwargs) -> "DQNProfile":
        """Create a copy with specific parameter overrides."""
        profile = copy.deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
            else:
                raise ValueError(f"Unknown DQN parameter: {key}")
        return profile


@dataclass  
class AgentBehaviorProfile:
    """Agent behavior profile defining action preferences and strategies."""
    
    # Action weights (relative probabilities)
    move_weight: float = 0.25
    gather_weight: float = 0.25
    share_weight: float = 0.2
    attack_weight: float = 0.15
    reproduce_weight: float = 0.15
    
    # Behavior modifiers
    cooperation_tendency: float = 0.5  # 0=selfish, 1=cooperative
    aggression_level: float = 0.3      # 0=peaceful, 1=aggressive  
    risk_tolerance: float = 0.4        # 0=cautious, 1=risky
    efficiency_focus: float = 0.6      # 0=random, 1=optimal
    
    def with_overrides(self, **kwargs) -> "AgentBehaviorProfile":
        """Create a copy with specific behavior overrides."""
        profile = copy.deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
            else:
                raise ValueError(f"Unknown behavior parameter: {key}")
        return profile


@dataclass
class EnvironmentProfile:
    """Environment configuration profile."""
    width: int = 100
    height: int = 100
    initial_resources: int = 20
    resource_regen_rate: float = 0.1
    resource_regen_amount: int = 2
    max_resource_amount: int = 30
    
    def with_overrides(self, **kwargs) -> "EnvironmentProfile":
        """Create a copy with specific environment overrides."""
        profile = copy.deepcopy(self)
        for key, value in kwargs.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
            else:
                raise ValueError(f"Unknown environment parameter: {key}")
        return profile


# ===== PREDEFINED PROFILES =====

# DQN Learning Profiles
DQN_PROFILES = {
    "default": DQNProfile(),
    
    "fast_learning": DQNProfile(
        learning_rate=0.005,
        epsilon_decay=0.99,
        batch_size=64
    ),
    
    "stable_learning": DQNProfile(
        learning_rate=0.0005, 
        epsilon_decay=0.998,
        memory_size=20000,
        batch_size=16
    ),
    
    "exploration_focused": DQNProfile(
        epsilon_min=0.05,
        epsilon_decay=0.999,
        learning_rate=0.002
    ),
    
    "memory_efficient": DQNProfile(
        memory_size=5000,
        batch_size=16,
        hidden_size=32
    ),
    
    "high_performance": DQNProfile(
        memory_size=50000,
        batch_size=128, 
        hidden_size=128,
        learning_rate=0.01
    )
}

# Agent Behavior Profiles  
AGENT_BEHAVIOR_PROFILES = {
    "balanced": AgentBehaviorProfile(),
    
    "cooperative": AgentBehaviorProfile(
        share_weight=0.4,
        attack_weight=0.05,
        cooperation_tendency=0.9,
        aggression_level=0.1
    ),
    
    "aggressive": AgentBehaviorProfile(
        attack_weight=0.4,
        share_weight=0.05,
        cooperation_tendency=0.1,
        aggression_level=0.9,
        risk_tolerance=0.8
    ),
    
    "gatherer": AgentBehaviorProfile(
        gather_weight=0.5,
        move_weight=0.3,
        attack_weight=0.1,
        efficiency_focus=0.9
    ),
    
    "explorer": AgentBehaviorProfile(
        move_weight=0.4,
        gather_weight=0.3,
        risk_tolerance=0.8,
        efficiency_focus=0.3
    ),
    
    "survivor": AgentBehaviorProfile(
        reproduce_weight=0.3,
        share_weight=0.3, 
        gather_weight=0.3,
        aggression_level=0.2,
        risk_tolerance=0.2,
        cooperation_tendency=0.7
    )
}

# Environment Profiles
ENVIRONMENT_PROFILES = {
    "default": EnvironmentProfile(),
    
    "resource_rich": EnvironmentProfile(
        initial_resources=50,
        resource_regen_rate=0.2,
        resource_regen_amount=3
    ),
    
    "resource_scarce": EnvironmentProfile(
        initial_resources=10,
        resource_regen_rate=0.05,
        resource_regen_amount=1
    ),
    
    "large_world": EnvironmentProfile(
        width=200,
        height=200,
        initial_resources=80
    ),
    
    "small_world": EnvironmentProfile(
        width=50,
        height=50,
        initial_resources=8
    ),
    
    "dynamic": EnvironmentProfile(
        resource_regen_rate=0.3,
        resource_regen_amount=4,
        max_resource_amount=50
    )
}


def get_dqn_profile(name: str) -> DQNProfile:
    """Get a DQN profile by name."""
    if name not in DQN_PROFILES:
        available = ", ".join(DQN_PROFILES.keys())
        raise ValueError(f"Unknown DQN profile: {name}. Available: {available}")
    return copy.deepcopy(DQN_PROFILES[name])


def get_behavior_profile(name: str) -> AgentBehaviorProfile:
    """Get an agent behavior profile by name."""
    if name not in AGENT_BEHAVIOR_PROFILES:
        available = ", ".join(AGENT_BEHAVIOR_PROFILES.keys())
        raise ValueError(f"Unknown behavior profile: {name}. Available: {available}")
    return copy.deepcopy(AGENT_BEHAVIOR_PROFILES[name])


def get_environment_profile(name: str) -> EnvironmentProfile:
    """Get an environment profile by name.""" 
    if name not in ENVIRONMENT_PROFILES:
        available = ", ".join(ENVIRONMENT_PROFILES.keys())
        raise ValueError(f"Unknown environment profile: {name}. Available: {available}")
    return copy.deepcopy(ENVIRONMENT_PROFILES[name])


def list_profiles() -> Dict[str, list]:
    """List all available profiles by category."""
    return {
        "dqn": list(DQN_PROFILES.keys()),
        "behavior": list(AGENT_BEHAVIOR_PROFILES.keys()), 
        "environment": list(ENVIRONMENT_PROFILES.keys())
    }
"""Action modules for agent behavior using Deep Q-Learning (DQN).

This module provides a comprehensive set of DQN-based action modules for agent
behaviors in the AgentFarm simulation. Each action module implements specialized
learning for specific behaviors like movement, gathering, combat, sharing, and
reproduction.

Key Components:
    - Base DQN infrastructure with shared feature extraction
    - Specialized action modules for different behaviors
    - Unified training system across all modules
    - Curriculum learning for progressive complexity
    - Comprehensive logging and database integration

Available Actions:
    - move: Navigation and pathfinding with resource proximity rewards
    - gather: Resource collection with efficiency-based learning
    - attack: Combat interactions with health-based defensive behavior
    - share: Cooperative resource sharing with altruism rewards
    - reproduce: Population dynamics with environmental constraints
    - select: High-level action selection meta-controller

All actions are registered dynamically and can be accessed through the
action_registry for easy integration with agent decision systems.
"""

from farm.core.action import action_registry

# Actions are now registered dynamically through individual modules

__all__ = ['action_registry']

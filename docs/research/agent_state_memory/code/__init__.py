"""Memory module for agent memory systems.

This package provides memory systems for agents to store and retrieve
their experiences, enabling learning and adaptation.
"""

from redis_memory import AgentMemory, AgentMemoryManager, RedisMemoryConfig
from agent_memory.core import AgentMemorySystem
from agent_memory.memory_agent import MemoryAgent
from agent_memory.config import MemoryConfig

__all__ = [
    "AgentMemory",  # Legacy support
    "AgentMemoryManager",  # Legacy support
    "RedisMemoryConfig",  # Legacy support
    "AgentMemorySystem",  # New unified system
    "MemoryAgent",  # New memory agent
    "MemoryConfig",  # New configuration
]

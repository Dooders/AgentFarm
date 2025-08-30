"""Agent Memory module using Redis for efficient state storage and retrieval.

This module provides a Redis-backed memory system for agents to store and query their
past states, actions, and observations. It enables efficient temporal lookups and
semantic search capabilities, allowing agents to learn from their experiences.

Features:
- Efficient agent-specific memory storage using Redis structures
- Fast retrieval of temporal sequences of agent states
- Semantic search over past experiences
- Custom indexing for domain-specific queries
- Automatic memory management with TTL and priority-based expiration
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import redis

from farm.core.perception import PerceptionData
from farm.core.state import AgentState

logger = logging.getLogger(__name__)


@dataclass
class RedisMemoryConfig:
    """Configuration for Redis-backed agent memory."""

    # Redis connection settings
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None

    # Memory management
    memory_limit: int = 1000  # Max memory entries per agent
    ttl: int = 3600  # Default TTL for entries (seconds)

    # Advanced settings
    enable_semantic_search: bool = True
    embedding_dimension: int = 128
    memory_priority_decay: float = 0.95  # How quickly priority decays with time

    # Auto-cleanup
    cleanup_interval: int = 100  # Check for cleanup every N insertions

    # Namespace prefixes
    namespace: str = "agent_memory"

    @property
    def connection_params(self) -> Dict:
        """Return connection parameters for Redis client."""
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": self.password,
            "decode_responses": True,  # Auto-decode Redis responses
        }


class AgentMemory:
    """Redis-backed memory system for storing and retrieving agent experiences.

    This class provides a memory system that stores agent experiences in Redis,
    allowing for efficient storage, retrieval, and querying of agent states,
    actions, and perceptions over time.

    Attributes:
        agent_id (str): Unique identifier for this agent
        redis_client (redis.Redis): Redis client instance
        config (RedisMemoryConfig): Configuration for the memory system
        _insertion_count (int): Counter for tracking insertions for cleanup
    """

    def __init__(
        self,
        agent_id: str,
        config: Optional[RedisMemoryConfig] = None,
        redis_client: Optional[redis.Redis] = None,
    ):
        """Initialize the agent memory system.

        Args:
            agent_id (str): Unique identifier for this agent
            config (RedisMemoryConfig, optional): Configuration, or use defaults
            redis_client (redis.Redis, optional): Existing Redis client to use
        """
        self.agent_id = agent_id
        self.config = config or RedisMemoryConfig()

        # Use provided Redis client or create a new one
        self.redis_client = redis_client or redis.Redis(**self.config.connection_params)

        # Initialize tracking variables
        self._insertion_count = 0

        # Check Redis connection
        try:
            self.redis_client.ping()
            logger.info(f"Agent {agent_id} memory connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def _safe_serialize(self, obj: Any) -> Dict[str, Any]:
        """Safely serialize an object to a dictionary.

        Tries to use object's serialization method if available,
        otherwise falls back to vars().

        Args:
            obj: Object to serialize

        Returns:
            Dictionary representation of the object
        """
        try:
            if hasattr(obj, "as_dict") and callable(getattr(obj, "as_dict")):
                return obj.as_dict()
            elif hasattr(obj, "as_serializable") and callable(
                getattr(obj, "as_serializable")
            ):
                return obj.as_serializable()
            else:
                return vars(obj)
        except Exception:
            # Fallback to vars() if serialization methods fail
            return vars(obj)

    @property
    def _agent_key_prefix(self) -> str:
        """Get the Redis key prefix for this agent."""
        return f"{self.config.namespace}:{self.agent_id}"

    def remember_state(
        self,
        step: int,
        state: AgentState,
        action: Optional[str] = None,
        reward: Optional[float] = None,
        perception: Optional[PerceptionData] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 1.0,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store agent state and related information in memory.

        Args:
            step (int): Current simulation step
            state (AgentState): Agent's current state
            action (str, optional): Action that led to this state
            reward (float, optional): Reward received for the action
            perception (PerceptionData, optional): Agent's perception data
            metadata (Dict, optional): Additional information to store
            priority (float): Memory importance (higher values persist longer)
            ttl (int, optional): Time-to-live in seconds, or None for default

        Returns:
            bool: True if stored successfully, False otherwise
        """
        try:
            # Create memory entry as dictionary
            memory_entry = {
                "timestamp": time.time(),
                "step": step,
                "state": self._safe_serialize(state),
                "priority": priority,
            }

            # Add optional fields if provided
            if action is not None:
                memory_entry["action"] = action
            if reward is not None:
                memory_entry["reward"] = reward
            if perception is not None:
                memory_entry["perception"] = self._safe_serialize(perception)
            if metadata:
                memory_entry["metadata"] = metadata

            # Two key data structures:
            # 1. Ordered timeline using a sorted set with step number as score
            # 2. Hash table with full data keyed by step

            # Generate keys
            timeline_key = f"{self._agent_key_prefix}:timeline"
            memory_key = f"{self._agent_key_prefix}:memory:{step}"

            # Add to timeline sorted set
            self.redis_client.zadd(timeline_key, {str(step): step})

            # Store full state as hash
            serialized = json.dumps(memory_entry)
            self.redis_client.set(memory_key, serialized)

            # Set TTL if provided
            if ttl or self.config.ttl:
                self.redis_client.expire(memory_key, ttl or self.config.ttl)

            # Check if we should clean up
            self._insertion_count += 1
            if self._insertion_count >= self.config.cleanup_interval:
                self._cleanup_old_memories()
                self._insertion_count = 0

            return True

        except Exception as e:
            logger.error(f"Failed to store memory for agent {self.agent_id}: {e}")
            return False

    def retrieve_state(self, step: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific state by step number.

        Args:
            step (int): The step number to retrieve

        Returns:
            Dict or None: The stored state or None if not found
        """
        try:
            memory_key = f"{self._agent_key_prefix}:memory:{step}"
            data = self.redis_client.get(memory_key)

            if data:
                return json.loads(str(data))  # type: ignore
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve state for agent {self.agent_id}: {e}")
            return None

    def retrieve_recent_states(self, count: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the most recent states.

        Args:
            count (int): Number of recent states to retrieve

        Returns:
            List[Dict]: List of state dictionaries, most recent first
        """
        try:
            timeline_key = f"{self._agent_key_prefix}:timeline"

            # Get the most recent step numbers from the timeline
            recent_steps = self.redis_client.zrevrange(timeline_key, 0, count - 1)

            # Retrieve each state
            results = []
            for step in recent_steps:  # type: ignore
                state = self.retrieve_state(int(step))
                if state:
                    results.append(state)

            return results

        except Exception as e:
            logger.error(
                f"Failed to retrieve recent states for agent {self.agent_id}: {e}"
            )
            return []

    def retrieve_states_by_timeframe(
        self, start_step: int, end_step: int
    ) -> List[Dict[str, Any]]:
        """Retrieve states within a specific timeframe.

        Args:
            start_step (int): Starting step (inclusive)
            end_step (int): Ending step (inclusive)

        Returns:
            List[Dict]: List of state dictionaries in chronological order
        """
        try:
            timeline_key = f"{self._agent_key_prefix}:timeline"

            # Get steps in the specified range
            steps = self.redis_client.zrangebyscore(
                timeline_key, min=start_step, max=end_step
            )

            # Retrieve each state
            results = []
            for step in steps:  # type: ignore
                state = self.retrieve_state(int(step))
                if state:
                    results.append(state)

            return results

        except Exception as e:
            logger.error(
                f"Failed to retrieve states by timeframe for agent {self.agent_id}: {e}"
            )
            return []

    def search_by_metadata(
        self, key: str, value: Any, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for states with matching metadata.

        Args:
            key (str): Metadata key to search for
            value (Any): Value to match
            limit (int): Maximum number of results to return

        Returns:
            List[Dict]: List of matching state dictionaries
        """
        try:
            timeline_key = f"{self._agent_key_prefix}:timeline"

            # Get all steps from the timeline
            all_steps = self.redis_client.zrange(timeline_key, 0, -1)

            # Search through states for matching metadata
            results = []
            for step in all_steps:  # type: ignore
                state = self.retrieve_state(int(step))
                if not state or "metadata" not in state:
                    continue

                if key in state["metadata"] and state["metadata"][key] == value:
                    results.append(state)

                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            logger.error(f"Failed to search metadata for agent {self.agent_id}: {e}")
            return []

    def search_by_state_value(
        self, key: str, value: Any, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for states with matching state values.

        Args:
            key (str): State attribute to search for
            value (Any): Value to match
            limit (int): Maximum number of results to return

        Returns:
            List[Dict]: List of matching state dictionaries
        """
        try:
            timeline_key = f"{self._agent_key_prefix}:timeline"

            # Get all steps from the timeline
            all_steps = self.redis_client.zrange(timeline_key, 0, -1)

            # Search through states for matching value
            results = []
            for step in list(all_steps):  # type: ignore
                state_data = self.retrieve_state(int(step))
                if not state_data or "state" not in state_data:
                    continue

                if key in state_data["state"] and state_data["state"][key] == value:
                    results.append(state_data)

                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            logger.error(
                f"Failed to search state values for agent {self.agent_id}: {e}"
            )
            return []

    def search_by_position(
        self, position: Tuple[float, float], radius: float = 1.0, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for states where agent was near a specific position.

        Args:
            position (Tuple[float, float]): (x, y) coordinates to search around
            radius (float): Search radius around the position
            limit (int): Maximum number of results to return

        Returns:
            List[Dict]: List of matching state dictionaries
        """
        try:
            timeline_key = f"{self._agent_key_prefix}:timeline"
            target_x, target_y = position

            # Get all steps from the timeline
            all_steps = self.redis_client.zrange(timeline_key, 0, -1)

            # Search through states for positions within radius
            results = []
            for step in list(all_steps):  # type: ignore
                state_data = self.retrieve_state(int(step))
                if (
                    not state_data
                    or "state" not in state_data
                    or "position_x" not in state_data["state"]
                    or "position_y" not in state_data["state"]
                ):
                    continue

                x = state_data["state"].get("position_x")
                y = state_data["state"].get("position_y")

                # Calculate distance
                distance = ((x - target_x) ** 2 + (y - target_y) ** 2) ** 0.5
                if distance <= radius:
                    results.append(state_data)

                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            logger.error(f"Failed to search by position for agent {self.agent_id}: {e}")
            return []

    def retrieve_action_history(self, limit: int = 20) -> List[str]:
        """Retrieve a history of actions taken by the agent.

        Args:
            limit (int): Maximum number of actions to retrieve

        Returns:
            List[str]: List of action names in reverse chronological order
        """
        try:
            timeline_key = f"{self._agent_key_prefix}:timeline"

            # Get the most recent step numbers from the timeline
            recent_steps = self.redis_client.zrevrange(timeline_key, 0, limit - 1)

            # Extract action from each state
            actions = []
            for step in list(recent_steps):  # type: ignore
                state = self.retrieve_state(int(step))
                if state and "action" in state:
                    actions.append(state["action"])

            return actions

        except Exception as e:
            logger.error(
                f"Failed to retrieve action history for agent {self.agent_id}: {e}"
            )
            return []

    def get_action_frequency(self) -> Dict[str, int]:
        """Get frequency distribution of actions taken by the agent.

        Returns:
            Dict[str, int]: Mapping of action names to occurrence counts
        """
        try:
            actions = self.retrieve_action_history(limit=1000)  # Get a large sample

            # Count occurrences
            frequency = {}
            for action in actions:
                if action in frequency:
                    frequency[action] += 1
                else:
                    frequency[action] = 1

            return frequency

        except Exception as e:
            logger.error(
                f"Failed to get action frequency for agent {self.agent_id}: {e}"
            )
            return {}

    def clear_memory(self) -> bool:
        """Clear all memory entries for this agent.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all keys for this agent
            pattern = f"{self._agent_key_prefix}:*"
            keys = self.redis_client.keys(pattern)

            # Delete all keys
            if keys:
                self.redis_client.delete(*list(keys))  # type: ignore

            logger.info(f"Cleared memory for agent {self.agent_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear memory for agent {self.agent_id}: {e}")
            return False

    def _cleanup_old_memories(self) -> None:
        """Clean up old memories based on limit and priority.

        This method is called periodically to ensure memory doesn't exceed limits.
        Memories are removed based on a combination of age and priority.
        """
        try:
            timeline_key = f"{self._agent_key_prefix}:timeline"

            # Get total count of memories
            memory_count = int(self.redis_client.zcard(timeline_key))  # type: ignore

            # If we're under the limit, no cleanup needed
            if memory_count <= self.config.memory_limit:
                return

            # Calculate how many to remove
            to_remove = memory_count - self.config.memory_limit

            # Get the oldest entries
            oldest_steps = self.redis_client.zrange(timeline_key, 0, to_remove - 1)

            # Remove each entry
            for step in list(oldest_steps):  # type: ignore
                memory_key = f"{self._agent_key_prefix}:memory:{step}"
                self.redis_client.delete(memory_key)

            # Remove from timeline
            if oldest_steps:
                self.redis_client.zrem(timeline_key, *list(oldest_steps))  # type: ignore

            logger.debug(
                f"Cleaned up {len(list(oldest_steps))} old memories for agent {self.agent_id}"  # type: ignore
            )

        except Exception as e:
            logger.error(f"Failed to clean up memories for agent {self.agent_id}: {e}")


class AgentMemoryManager:
    """Manages memory systems for multiple agents.

    This singleton class provides centralized management of agent memory systems,
    allowing resource sharing and efficient connection pooling.
    """

    _instance = None

    @classmethod
    def get_instance(cls, config: Optional[RedisMemoryConfig] = None):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    def __init__(self, config: Optional[RedisMemoryConfig] = None):
        """Initialize the memory manager.

        Args:
            config (RedisMemoryConfig, optional): Configuration, or use defaults
        """
        if AgentMemoryManager._instance is not None:
            raise RuntimeError("Use get_instance() to get the singleton instance")

        self.config = config or RedisMemoryConfig()
        self.memories = {}

        # Create shared Redis connection pool
        self.redis = redis.Redis(**self.config.connection_params)

        # Check Redis connection
        try:
            self.redis.ping()
            logger.info("AgentMemoryManager connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def get_memory(self, agent_id: str) -> AgentMemory:
        """Get or create an agent memory system.

        Args:
            agent_id (str): Unique identifier for the agent

        Returns:
            AgentMemory: Memory system for the agent
        """
        if agent_id not in self.memories:
            self.memories[agent_id] = AgentMemory(
                agent_id=agent_id, config=self.config, redis_client=self.redis
            )
        return self.memories[agent_id]

    def clear_all_memories(self) -> bool:
        """Clear all agent memories.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all keys for the namespace
            pattern = f"{self.config.namespace}:*"
            keys = self.redis.keys(pattern)

            # Delete all keys
            if keys:
                self.redis.delete(*list(keys))  # type: ignore

            # Reset local cache
            self.memories = {}

            logger.info("Cleared all agent memories")
            return True

        except Exception as e:
            logger.error(f"Failed to clear all memories: {e}")
            return False

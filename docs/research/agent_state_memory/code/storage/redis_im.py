"""Redis-based Intermediate Memory (IM) storage for agent memory system.

This module provides a Redis-based implementation of the Intermediate Memory
storage tier with TTL-based expiration and level 1 compression.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from ..config import RedisIMConfig
from ..utils.error_handling import Priority, RedisUnavailableError, RedisTimeoutError
from .redis_client import ResilientRedisClient

logger = logging.getLogger(__name__)


class RedisIMStore:
    """Redis-based storage for Intermediate Memory (IM).
    
    This class provides storage operations for the medium-resolution, 
    intermediate-lived agent memory tier using Redis as the backing store.
    
    Attributes:
        config: Configuration for IM Redis storage
        redis: Resilient Redis client instance
        _key_prefix: Prefix for Redis keys
    """
    
    def __init__(self, config: RedisIMConfig):
        """Initialize the Redis IM store.
        
        Args:
            config: Configuration for IM Redis storage
        """
        self.config = config
        self._key_prefix = config.namespace
        
        # Create resilient Redis client
        self.redis = ResilientRedisClient(
            client_name="im",
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            # Circuit breaker and retry settings
            circuit_threshold=3,
            circuit_reset_timeout=300
        )
        
        logger.info(
            "Initialized RedisIMStore with namespace %s", 
            self._key_prefix
        )
    
    def store(
        self, 
        agent_id: str, 
        memory_entry: Dict[str, Any],
        priority: Priority = Priority.NORMAL
    ) -> bool:
        """Store a memory entry in IM.
        
        Args:
            agent_id: ID of the agent
            memory_entry: Memory entry to store (should be level 1 compressed)
            priority: Priority level for this operation
            
        Returns:
            True if the operation succeeded, False otherwise
        """
        memory_id = memory_entry.get("memory_id")
        if not memory_id:
            logger.error("Cannot store memory entry without memory_id")
            return False
            
        # Verify compression level
        compression_level = memory_entry.get("metadata", {}).get("compression_level")
        if compression_level != 1:
            logger.error(
                "Invalid compression level for IM storage: %s. Expected level 1.",
                compression_level
            )
            return False
        
        # Use store_with_retry for resilient storage
        return self.redis.store_with_retry(
            agent_id=agent_id,
            state_data=memory_entry,
            store_func=self._store_memory_entry,
            priority=priority
        )
    
    def _store_memory_entry(self, agent_id: str, memory_entry: Dict[str, Any]) -> bool:
        """Internal method to store a memory entry.
        
        Args:
            agent_id: ID of the agent
            memory_entry: Memory entry to store
            
        Returns:
            True if the operation succeeded, False otherwise
            
        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        memory_id = memory_entry["memory_id"]
        timestamp = memory_entry.get("timestamp", time.time())
        
        try:
            # Store the full memory entry
            key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
            self.redis.set(
                key,
                json.dumps(memory_entry),
                ex=self.config.ttl
            )
            
            # Add to agent's memory list
            agent_memories_key = f"{self._key_prefix}:{agent_id}:memories"
            self.redis.zadd(
                agent_memories_key,
                {memory_id: timestamp}
            )
            
            # Set TTL on the sorted set
            self.redis.expire(agent_memories_key, self.config.ttl)
            
            # Add to timeline index for chronological retrieval
            timeline_key = f"{self._key_prefix}:{agent_id}:timeline"
            self.redis.zadd(
                timeline_key,
                {memory_id: timestamp}
            )
            self.redis.expire(timeline_key, self.config.ttl)
            
            # Add to importance index for importance-based retrieval
            importance = memory_entry.get("metadata", {}).get("importance_score", 0.0)
            importance_key = f"{self._key_prefix}:{agent_id}:importance"
            self.redis.zadd(
                importance_key,
                {memory_id: importance}
            )
            self.redis.expire(importance_key, self.config.ttl)
            
            return True
            
        except (RedisUnavailableError, RedisTimeoutError) as e:
            # Let these propagate up for retry handling
            raise
        except Exception as e:
            logger.error(
                "Failed to store memory entry %s: %s",
                memory_id, str(e)
            )
            return False
    
    def get(self, agent_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory entry by ID.
        
        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory entry if found, None otherwise
        """
        key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
        try:
            data = self.redis.get(key)
            if not data:
                return None
                
            memory_entry = json.loads(data)
            self._update_access_metadata(agent_id, memory_id, memory_entry)
            return memory_entry
            
        except Exception as e:
            logger.error(
                "Failed to retrieve memory entry %s: %s",
                memory_id, str(e)
            )
            return None
    
    def get_by_timerange(
        self,
        agent_id: str,
        start_time: float,
        end_time: float,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve memories within a time range.
        
        Args:
            agent_id: ID of the agent
            start_time: Start of time range (Unix timestamp)
            end_time: End of time range (Unix timestamp)
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries within the time range
        """
        timeline_key = f"{self._key_prefix}:{agent_id}:timeline"
        try:
            # Get memory IDs within time range
            memory_ids = self.redis.zrangebyscore(
                timeline_key,
                min=start_time,
                max=end_time,
                start=0,
                num=limit
            )
            
            # Retrieve each memory entry
            memories = []
            for memory_id in memory_ids:
                memory = self.get(agent_id, memory_id)
                if memory:
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(
                "Failed to retrieve memories by time range: %s",
                str(e)
            )
            return []
    
    def get_by_importance(
        self,
        agent_id: str,
        min_importance: float = 0.0,
        max_importance: float = 1.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve memories by importance score range.
        
        Args:
            agent_id: ID of the agent
            min_importance: Minimum importance score (0.0-1.0)
            max_importance: Maximum importance score (0.0-1.0)
            limit: Maximum number of memories to return
            
        Returns:
            List of memory entries within the importance range
        """
        importance_key = f"{self._key_prefix}:{agent_id}:importance"
        try:
            # Get memory IDs within importance range
            memory_ids = self.redis.zrangebyscore(
                importance_key,
                min=min_importance,
                max=max_importance,
                start=0,
                num=limit
            )
            
            # Retrieve each memory entry
            memories = []
            for memory_id in memory_ids:
                memory = self.get(agent_id, memory_id)
                if memory:
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(
                "Failed to retrieve memories by importance: %s",
                str(e)
            )
            return []
    
    def delete(self, agent_id: str, memory_id: str) -> bool:
        """Delete a memory entry.
        
        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            # Remove from all indices
            key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
            agent_memories_key = f"{self._key_prefix}:{agent_id}:memories"
            timeline_key = f"{self._key_prefix}:{agent_id}:timeline"
            importance_key = f"{self._key_prefix}:{agent_id}:importance"
            
            self.redis.delete(key)
            self.redis.zrem(agent_memories_key, memory_id)
            self.redis.zrem(timeline_key, memory_id)
            self.redis.zrem(importance_key, memory_id)
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to delete memory entry %s: %s",
                memory_id, str(e)
            )
            return False
    
    def count(self, agent_id: str) -> int:
        """Get the number of memories for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Number of memories stored for the agent
        """
        agent_memories_key = f"{self._key_prefix}:{agent_id}:memories"
        try:
            return self.redis.zcard(agent_memories_key)
        except Exception as e:
            logger.error(
                "Failed to get memory count for agent %s: %s",
                agent_id, str(e)
            )
            return 0
    
    def clear(self, agent_id: str) -> bool:
        """Clear all memories for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            True if clearing was successful
        """
        try:
            # Get all memory IDs
            agent_memories_key = f"{self._key_prefix}:{agent_id}:memories"
            memory_ids = self.redis.zrange(agent_memories_key, 0, -1)
            
            # Delete each memory entry
            for memory_id in memory_ids:
                self.delete(agent_id, memory_id)
            
            return True
            
        except Exception as e:
            logger.error(
                "Failed to clear memories for agent %s: %s",
                agent_id, str(e)
            )
            return False
    
    def _update_access_metadata(
        self, 
        agent_id: str, 
        memory_id: str, 
        memory_entry: Dict[str, Any]
    ) -> None:
        """Update access metadata for a memory entry.
        
        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory
            memory_entry: Memory entry to update
        """
        try:
            # Update access time and retrieval count
            metadata = memory_entry.get("metadata", {})
            retrieval_count = metadata.get("retrieval_count", 0) + 1
            access_time = time.time()
            
            metadata.update({
                "last_access_time": access_time,
                "retrieval_count": retrieval_count
            })
            memory_entry["metadata"] = metadata
            
            # Store updated entry
            key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
            self.redis.set(
                key,
                json.dumps(memory_entry),
                ex=self.config.ttl
            )
            
            # Update importance based on access patterns
            # Increase importance for frequently accessed memories
            if retrieval_count > 1:
                importance = metadata.get("importance_score", 0.0)
                access_factor = min(retrieval_count / 10.0, 1.0)  # Cap at 1.0
                new_importance = importance + (access_factor * 0.1)  # Slight boost
                
                importance_key = f"{self._key_prefix}:{agent_id}:importance"
                self.redis.zadd(
                    importance_key,
                    {memory_id: new_importance}
                )
                
                # Update in memory entry
                metadata["importance_score"] = new_importance
                
        except Exception as e:
            # Non-critical operation, just log
            logger.warning(
                "Failed to update access metadata for memory %s: %s",
                memory_id, str(e)
            ) 
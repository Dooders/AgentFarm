"""Redis-based Short-Term Memory (STM) storage for agent memory system.

This module provides a Redis-based implementation of the Short-Term Memory
storage tier with comprehensive error handling and recovery mechanisms.
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from ..config import RedisSTMConfig
from ..utils.error_handling import Priority, RedisUnavailableError, RedisTimeoutError
from .redis_client import ResilientRedisClient

logger = logging.getLogger(__name__)


class RedisSTMStore:
    """Redis-based storage for Short-Term Memory (STM).
    
    This class provides storage operations for the high-resolution, 
    short-lived agent memory tier using Redis as the backing store.
    
    Attributes:
        config: Configuration for STM Redis storage
        redis: Resilient Redis client instance
        _key_prefix: Prefix for Redis keys
    """
    
    def __init__(self, config: RedisSTMConfig):
        """Initialize the Redis STM store.
        
        Args:
            config: Configuration for STM Redis storage
        """
        self.config = config
        self._key_prefix = config.namespace
        
        # Create resilient Redis client
        self.redis = ResilientRedisClient(
            client_name="stm",
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            # Circuit breaker and retry settings
            circuit_threshold=3,
            circuit_reset_timeout=300
        )
        
        logger.info(
            "Initialized RedisSTMStore with namespace %s", 
            self._key_prefix
        )
    
    def store(
        self, 
        agent_id: str, 
        memory_entry: Dict[str, Any],
        priority: Priority = Priority.NORMAL
    ) -> bool:
        """Store a memory entry in STM.
        
        Args:
            agent_id: ID of the agent
            memory_entry: Memory entry to store
            priority: Priority level for this operation
            
        Returns:
            True if the operation succeeded, False otherwise
        """
        memory_id = memory_entry.get("memory_id")
        if not memory_id:
            logger.error("Cannot store memory entry without memory_id")
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
            
            # If it has embeddings, store for vector search
            embeddings = memory_entry.get("embeddings", {})
            if embeddings and "full_vector" in embeddings:
                vector_key = f"{self._key_prefix}:{agent_id}:vector:{memory_id}"
                self.redis.set(
                    vector_key,
                    json.dumps(embeddings["full_vector"]),
                    ex=self.config.ttl
                )
            
            logger.debug(
                "Stored memory %s for agent %s in STM", 
                memory_id, agent_id
            )
            return True
            
        except (RedisUnavailableError, RedisTimeoutError) as e:
            # These exceptions are caught by store_with_retry
            raise e
        except Exception as e:
            logger.error(
                "Unexpected error storing memory %s for agent %s: %s", 
                memory_id, agent_id, str(e)
            )
            return False
    
    def get(self, agent_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory entry by ID.
        
        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory to retrieve
            
        Returns:
            Memory entry or None if not found
        """
        try:
            key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
            data = self.redis.get(key)
            
            if not data:
                return None
                
            memory_entry = json.loads(data)
            
            # Update access time
            self._update_access_metadata(agent_id, memory_id, memory_entry)
            
            return memory_entry
            
        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to retrieve memory %s for agent %s: %s",
                memory_id, agent_id, str(e)
            )
            return None
        except json.JSONDecodeError as e:
            logger.error(
                "JSON decoding error for memory %s: %s",
                memory_id, str(e)
            )
            return None
        except Exception as e:
            logger.error(
                "Unexpected error retrieving memory %s: %s",
                memory_id, str(e)
            )
            return None
    
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
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            limit: Maximum number of results
            
        Returns:
            List of memory entries
        """
        try:
            timeline_key = f"{self._key_prefix}:{agent_id}:timeline"
            memory_ids = self.redis.zrangebyscore(
                timeline_key,
                start_time,
                end_time,
                start=0,
                num=limit
            )
            
            results = []
            for memory_id in memory_ids:
                memory = self.get(agent_id, memory_id)
                if memory:
                    results.append(memory)
            
            return results
            
        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to retrieve memories by timerange for agent %s: %s",
                agent_id, str(e)
            )
            return []
        except Exception as e:
            logger.error(
                "Unexpected error retrieving memories by timerange: %s",
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
        """Retrieve memories by importance score.
        
        Args:
            agent_id: ID of the agent
            min_importance: Minimum importance score (inclusive)
            max_importance: Maximum importance score (inclusive)
            limit: Maximum number of results
            
        Returns:
            List of memory entries
        """
        try:
            importance_key = f"{self._key_prefix}:{agent_id}:importance"
            memory_ids = self.redis.zrangebyscore(
                importance_key,
                min_importance,
                max_importance,
                start=0,
                num=limit,
                desc=True  # Higher importance first
            )
            
            results = []
            for memory_id in memory_ids:
                memory = self.get(agent_id, memory_id)
                if memory:
                    results.append(memory)
            
            return results
            
        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to retrieve memories by importance for agent %s: %s",
                agent_id, str(e)
            )
            return []
        except Exception as e:
            logger.error(
                "Unexpected error retrieving memories by importance: %s",
                str(e)
            )
            return []
    
    def delete(self, agent_id: str, memory_id: str) -> bool:
        """Delete a memory entry.
        
        Args:
            agent_id: ID of the agent
            memory_id: ID of the memory to delete
            
        Returns:
            True if the memory was deleted, False otherwise
        """
        try:
            # Delete the memory entry
            key = f"{self._key_prefix}:{agent_id}:memory:{memory_id}"
            deleted = self.redis.delete(key) > 0
            
            if deleted:
                # Remove from indexes
                self.redis.zrem(
                    f"{self._key_prefix}:{agent_id}:memories",
                    memory_id
                )
                self.redis.zrem(
                    f"{self._key_prefix}:{agent_id}:timeline",
                    memory_id
                )
                self.redis.zrem(
                    f"{self._key_prefix}:{agent_id}:importance",
                    memory_id
                )
                
                # Remove vector if it exists
                self.redis.delete(f"{self._key_prefix}:{agent_id}:vector:{memory_id}")
                
                logger.debug(
                    "Deleted memory %s for agent %s from STM",
                    memory_id, agent_id
                )
            
            return deleted
            
        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to delete memory %s for agent %s: %s",
                memory_id, agent_id, str(e)
            )
            return False
        except Exception as e:
            logger.error(
                "Unexpected error deleting memory %s: %s",
                memory_id, str(e)
            )
            return False
    
    def count(self, agent_id: str) -> int:
        """Get the number of memories for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Number of memories
        """
        try:
            key = f"{self._key_prefix}:{agent_id}:memories"
            return self.redis.zcard(key)
            
        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to count memories for agent %s: %s",
                agent_id, str(e)
            )
            return 0
        except Exception as e:
            logger.error(
                "Unexpected error counting memories: %s",
                str(e)
            )
            return 0
    
    def clear(self, agent_id: str) -> bool:
        """Clear all memories for an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all memory IDs
            key = f"{self._key_prefix}:{agent_id}:memories"
            memory_ids = self.redis.zrange(key, 0, -1)
            
            # Delete each memory
            for memory_id in memory_ids:
                self.delete(agent_id, memory_id)
            
            # Delete indexes
            self.redis.delete(
                f"{self._key_prefix}:{agent_id}:memories",
                f"{self._key_prefix}:{agent_id}:timeline",
                f"{self._key_prefix}:{agent_id}:importance"
            )
            
            logger.info(
                "Cleared all %d memories for agent %s from STM",
                len(memory_ids), agent_id
            )
            return True
            
        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Failed to clear memories for agent %s: %s",
                agent_id, str(e)
            )
            return False
        except Exception as e:
            logger.error(
                "Unexpected error clearing memories: %s",
                str(e)
            )
            return False
    
    def check_health(self) -> Dict[str, Any]:
        """Check the health of the Redis STM store.
        
        Returns:
            Dictionary with health status information
        """
        try:
            start_time = time.time()
            self.redis.ping()
            latency = (time.time() - start_time) * 1000  # ms
            
            return {
                "status": "healthy",
                "latency_ms": latency,
                "client": "redis-stm"
            }
        except Exception as e:
            logger.error("STM health check failed: %s", str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "client": "redis-stm"
            } 
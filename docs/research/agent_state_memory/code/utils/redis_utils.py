"""Redis utilities for the agent memory system.

This module provides utility functions and classes for working with Redis
in the context of the agent memory system.
"""

import json
import logging
import struct
import time
from typing import Dict, Any, List, Optional, Union, Tuple, Set

import redis
from redis.client import Pipeline

from .serialization import MemorySerializer, serialize_memory, deserialize_memory

logger = logging.getLogger(__name__)


# Memory entry serialization/deserialization for Redis
def serialize_memory_entry(memory_entry: Dict[str, Any]) -> str:
    """Serialize a memory entry for Redis storage.
    
    Consistently handles serialization of memory entries to JSON strings
    for storage in Redis.
    
    Args:
        memory_entry: Memory entry to serialize
        
    Returns:
        JSON string representation of the memory entry
    
    Raises:
        ValueError: If serialization fails
    """
    # Using the serialization module for consistency
    return serialize_memory(memory_entry, format="json")


def deserialize_memory_entry(data_str: str) -> Dict[str, Any]:
    """Deserialize a memory entry from Redis storage.
    
    Consistently handles deserialization of JSON strings from Redis
    back into memory entry dictionaries.
    
    Args:
        data_str: JSON string from Redis
        
    Returns:
        Memory entry dictionary
    
    Raises:
        ValueError: If deserialization fails
    """
    # Using the serialization module for consistency
    return deserialize_memory(data_str, format="json")


# Vector serialization/deserialization for Redis
def serialize_vector(vector: List[float]) -> str:
    """Serialize an embedding vector for Redis storage.
    
    Specialized handling for embedding vectors to ensure consistent
    serialization across the codebase.
    
    Args:
        vector: List of float values representing an embedding
        
    Returns:
        JSON string representation of the vector
    
    Raises:
        ValueError: If serialization fails
    """
    return MemorySerializer.serialize_vector(vector)


def deserialize_vector(vector_str: str) -> List[float]:
    """Deserialize an embedding vector from Redis storage.
    
    Specialized handling for embedding vectors to ensure consistent
    deserialization across the codebase.
    
    Args:
        vector_str: JSON string from Redis
        
    Returns:
        List of float values representing an embedding
    
    Raises:
        ValueError: If deserialization fails
    """
    return MemorySerializer.deserialize_vector(vector_str)


def vector_to_bytes(vector: List[float]) -> bytes:
    """Convert a vector to a binary representation for Redis.
    
    This is used for RediSearch vector storage which requires binary format.
    
    Args:
        vector: List of float values
        
    Returns:
        Binary representation of the vector
    """
    return b''.join([struct.pack('f', x) for x in vector])


def bytes_to_vector(binary_data: bytes) -> List[float]:
    """Convert binary data back to a vector.
    
    Args:
        binary_data: Binary representation of a vector
        
    Returns:
        List of float values
    """
    # Calculate number of floats (4 bytes each)
    float_count = len(binary_data) // 4
    return list(struct.unpack(f'{float_count}f', binary_data))


class RedisConnectionManager:
    """Manage Redis connections for the agent memory system.
    
    This class provides a centralized way to manage Redis connections
    and reuse them across different components of the system.
    
    Attributes:
        connections: Dictionary of Redis connections by key
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'RedisConnectionManager':
        """Get the singleton instance of the connection manager.
        
        Returns:
            RedisConnectionManager instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize the Redis connection manager."""
        self.connections = {}
    
    def get_connection(
        self, 
        host: str = "localhost", 
        port: int = 6379, 
        db: int = 0,
        password: Optional[str] = None,
        decode_responses: bool = True
    ) -> redis.Redis:
        """Get or create a Redis connection.
        
        This method reuses existing connections with the same parameters
        to avoid creating too many connections.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (if required)
            decode_responses: Whether to decode response bytes to strings
            
        Returns:
            Redis client instance
        """
        # Create a connection key
        conn_key = f"{host}:{port}:{db}:{decode_responses}"
        
        # Reuse existing connection if available
        if conn_key in self.connections:
            return self.connections[conn_key]
        
        # Create a new connection
        client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=decode_responses
        )
        
        # Store for reuse
        self.connections[conn_key] = client
        
        return client
    
    def close_all(self):
        """Close all Redis connections."""
        for client in self.connections.values():
            try:
                client.close()
            except:
                pass
        self.connections.clear()


class RedisBatchProcessor:
    """Process Redis operations in batches for improved performance.
    
    This class provides methods to batch Redis operations and execute
    them in a single pipeline for better performance.
    
    Attributes:
        redis_client: Redis client
        max_batch_size: Maximum batch size
        auto_execute: Whether to automatically execute when batch size is reached
    """
    
    def __init__(
        self, 
        redis_client: redis.Redis,
        max_batch_size: int = 100,
        auto_execute: bool = True
    ):
        """Initialize the batch processor.
        
        Args:
            redis_client: Redis client
            max_batch_size: Maximum batch size
            auto_execute: Whether to auto-execute when batch size is reached
        """
        self.redis = redis_client
        self.max_batch_size = max_batch_size
        self.auto_execute = auto_execute
        self.pipeline = self.redis.pipeline(transaction=False)
        self.command_count = 0
    
    def add_command(self, method_name: str, *args, **kwargs) -> 'RedisBatchProcessor':
        """Add a command to the batch.
        
        Args:
            method_name: Redis method name (e.g., "set", "hset")
            *args: Positional arguments for the method
            **kwargs: Keyword arguments for the method
            
        Returns:
            Self for chaining
        """
        # Get the method from the pipeline
        method = getattr(self.pipeline, method_name)
        
        # Add the command to the pipeline
        method(*args, **kwargs)
        self.command_count += 1
        
        # Auto-execute if needed
        if self.auto_execute and self.command_count >= self.max_batch_size:
            self.execute()
        
        return self
    
    def execute(self) -> List:
        """Execute the batch of commands.
        
        Returns:
            List of results from the commands
        """
        if self.command_count == 0:
            return []
        
        try:
            results = self.pipeline.execute()
            self.pipeline = self.redis.pipeline(transaction=False)
            self.command_count = 0
            return results
        except Exception as e:
            logger.error("Failed to execute Redis batch: %s", str(e))
            self.pipeline = self.redis.pipeline(transaction=False)
            self.command_count = 0
            raise


def redis_key_exists(redis_client: redis.Redis, key: str) -> bool:
    """Check if a key exists in Redis.
    
    Args:
        redis_client: Redis client
        key: Key to check
        
    Returns:
        True if the key exists
    """
    try:
        return bool(redis_client.exists(key))
    except Exception as e:
        logger.error("Failed to check if key exists: %s", str(e))
        return False


def redis_memory_scan(
    redis_client: redis.Redis,
    pattern: str,
    count: int = 100
) -> List[Dict[str, Any]]:
    """Scan Redis for memory entries matching a pattern.
    
    This is a generator function that yields memory entries as they are found.
    
    Args:
        redis_client: Redis client
        pattern: Key pattern to match
        count: Number of keys to scan at a time
        
    Yields:
        Memory entries
    """
    cursor = 0
    while True:
        cursor, keys = redis_client.scan(cursor, pattern, count)
        
        if keys:
            # Get memory entries in batches
            for key in keys:
                try:
                    data = redis_client.get(key)
                    if data:
                        yield deserialize_memory_entry(data)
                except Exception as e:
                    logger.warning("Failed to deserialize memory at key %s: %s", key, str(e))
        
        if cursor == 0:
            break


def redis_create_index(
    redis_client: redis.Redis,
    index_name: str,
    prefix: str,
    schema: Dict[str, str]
) -> bool:
    """Create a RediSearch index.
    
    Args:
        redis_client: Redis client
        index_name: Name of the index
        prefix: Key prefix to index
        schema: Index schema mapping field names to types
        
    Returns:
        True if the index was created successfully
    """
    try:
        # Check if index exists
        try:
            indices = redis_client.execute_command("FT._LIST")
            if index_name.encode() in indices:
                # Index already exists
                return True
        except:
            # Ignore errors when checking index list
            pass
        
        # Build the schema command arguments
        schema_args = []
        for field_name, field_type in schema.items():
            schema_args.extend([field_name, field_type])
        
        # Create the index
        redis_client.execute_command(
            "FT.CREATE", index_name, 
            "ON", "HASH", 
            "PREFIX", "1", f"{prefix}:",
            "SCHEMA", *schema_args
        )
        
        return True
    except Exception as e:
        logger.error("Failed to create Redis index %s: %s", index_name, str(e))
        return False


def redis_drop_index(redis_client: redis.Redis, index_name: str) -> bool:
    """Drop a RediSearch index.
    
    Args:
        redis_client: Redis client
        index_name: Name of the index
        
    Returns:
        True if the index was dropped successfully
    """
    try:
        redis_client.execute_command("FT.DROPINDEX", index_name)
        return True
    except Exception as e:
        logger.error("Failed to drop Redis index %s: %s", index_name, str(e))
        return False


def get_redis_info(redis_client: redis.Redis) -> Dict[str, Any]:
    """Get Redis server information.
    
    Args:
        redis_client: Redis client
        
    Returns:
        Dictionary of Redis server information
    """
    try:
        info = redis_client.info()
        
        # Add RediSearch module availability
        try:
            modules = redis_client.execute_command("MODULE LIST")
            info["modules"] = {m[1].decode(): m[3].decode() for m in modules}
            info["has_redisearch"] = any(
                m[1].decode().lower() == "search" for m in modules
            )
        except:
            info["has_redisearch"] = False
        
        return info
    except Exception as e:
        logger.error("Failed to get Redis info: %s", str(e))
        return {"error": str(e)}


# Get a Redis connection manager instance
def get_redis_connection_manager() -> RedisConnectionManager:
    """Get the Redis connection manager instance.
    
    Returns:
        RedisConnectionManager instance
    """
    return RedisConnectionManager.get_instance() 
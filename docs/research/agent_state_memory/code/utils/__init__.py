"""Utility functions for the agent memory system.

This package provides various utility functions and classes used throughout
the agent memory system for serialization, Redis operations, and error handling.
"""

from farm.memory.agent_memory.utils.serialization import (
    MemorySerializer,
    serialize_memory,
    deserialize_memory,
    to_json,
    from_json
)

from farm.memory.agent_memory.utils.redis_utils import (
    serialize_memory_entry,
    deserialize_memory_entry,
    serialize_vector,
    deserialize_vector,
    vector_to_bytes,
    bytes_to_vector,
    RedisConnectionManager,
    RedisBatchProcessor,
    redis_key_exists,
    redis_memory_scan,
    redis_create_index,
    redis_drop_index,
    get_redis_info,
    get_redis_connection_manager
)

from farm.memory.agent_memory.utils.error_handling import (
    MemoryError,
    MemorySerializationError,
    MemoryStorageError,
    MemoryRetrievalError,
    MemoryEmbeddingError,
    handle_memory_errors
)

__all__ = [
    # Serialization
    "MemorySerializer", 
    "serialize_memory", 
    "deserialize_memory",
    "to_json",
    "from_json",
    
    # Redis utilities
    "serialize_memory_entry",
    "deserialize_memory_entry",
    "serialize_vector",
    "deserialize_vector",
    "vector_to_bytes",
    "bytes_to_vector",
    "RedisConnectionManager",
    "RedisBatchProcessor",
    "redis_key_exists",
    "redis_memory_scan",
    "redis_create_index",
    "redis_drop_index",
    "get_redis_info",
    "get_redis_connection_manager",
    
    # Error handling
    "MemoryError",
    "MemorySerializationError",
    "MemoryStorageError",
    "MemoryRetrievalError",
    "MemoryEmbeddingError",
    "handle_memory_errors"
] 
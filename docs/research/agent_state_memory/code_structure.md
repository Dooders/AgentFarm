## 1. Module Structure

Create the following structure within `farm/memory/`:

```
farm/memory/
├── __init__.py (update)
├── redis_memory.py (existing)
├── agent_memory/
│   ├── __init__.py
│   ├── core.py
│   ├── config.py
│   ├── memory_agent.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── autoencoder.py
│   │   ├── vector_store.py
│   │   └── compression.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── redis_stm.py
│   │   ├── redis_im.py
│   │   └── sqlite_ltm.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── similarity.py
│   │   ├── temporal.py
│   │   └── attribute.py
│   └── api/
│       ├── __init__.py
│       ├── memory_api.py
│       └── hooks.py
├── utils/
│   ├── __init__.py
│   ├── serialization.py
│   └── redis_utils.py
└── tests/
    ├── __init__.py
    ├── test_memory_agent.py
    ├── test_storage.py
    ├── test_retrieval.py
    └── test_embeddings.py
```

## 2. Initial Code Scaffolding

### 2.1. Update Existing Files

Update `farm/memory/__init__.py`:

```python
"""Memory module for agent memory systems.

This package provides memory systems for agents to store and retrieve
their experiences, enabling learning and adaptation.
"""

from farm.memory.redis_memory import AgentMemory, AgentMemoryManager, RedisMemoryConfig
from farm.memory.agent_memory.core import AgentMemorySystem
from farm.memory.agent_memory.memory_agent import MemoryAgent
from farm.memory.agent_memory.config import MemoryConfig

__all__ = [
    "AgentMemory",  # Legacy support
    "AgentMemoryManager",  # Legacy support
    "RedisMemoryConfig",  # Legacy support
    "AgentMemorySystem",  # New unified system
    "MemoryAgent",  # New memory agent
    "MemoryConfig",  # New configuration
]
```

### 2.2. Core Configuration

Create `farm/memory/agent_memory/config.py`:

```python
"""Configuration for the agent memory system."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class RedisSTMConfig:
    """Configuration for Short-Term Memory (Redis)."""
    
    host: str = "localhost"
    port: int = 6379
    db: int = 0  # STM uses database 0
    password: Optional[str] = None
    
    # Memory settings
    ttl: int = 86400  # 24 hours
    memory_limit: int = 1000  # Max entries per agent
    
    # Redis key prefixes
    namespace: str = "agent_memory:stm"
    
    @property
    def connection_params(self):
        """Return connection parameters for Redis client."""
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": self.password if self.password else None,
        }


@dataclass
class RedisIMConfig:
    """Configuration for Intermediate Memory (Redis)."""
    
    host: str = "localhost"
    port: int = 6379
    db: int = 1  # IM uses database 1
    password: Optional[str] = None
    
    # Memory settings
    ttl: int = 604800  # 7 days
    memory_limit: int = 10000  # Max entries per agent
    compression_level: int = 1  # Level 1 compression
    
    # Redis key prefixes
    namespace: str = "agent_memory:im"
    
    @property
    def connection_params(self):
        """Return connection parameters for Redis client."""
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": self.password if self.password else None,
        }


@dataclass
class SQLiteLTMConfig:
    """Configuration for Long-Term Memory (SQLite)."""
    
    db_path: str = "agent_memory.db"
    
    # Memory settings
    compression_level: int = 2  # Level 2 compression
    batch_size: int = 100  # Number of entries to batch write
    
    # Table naming
    table_prefix: str = "agent_ltm"


@dataclass
class AutoencoderConfig:
    """Configuration for the autoencoder-based embeddings."""
    
    # Model dimensions
    input_dim: int = 64  # Raw input feature dimension
    stm_dim: int = 384  # STM embedding dimension
    im_dim: int = 128   # IM embedding dimension
    ltm_dim: int = 32   # LTM embedding dimension
    
    # Training settings
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    # Model paths
    model_path: Optional[str] = None  # Path to saved model
    use_neural_embeddings: bool = True  # Whether to use the neural embeddings
    
    # Advanced settings
    hidden_layers: List[int] = field(default_factory=lambda: [512, 256])


@dataclass
class MemoryConfig:
    """Configuration for the agent memory system."""
    
    # Memory tier configurations
    stm_config: RedisSTMConfig = field(default_factory=RedisSTMConfig)
    im_config: RedisIMConfig = field(default_factory=RedisIMConfig)
    ltm_config: SQLiteLTMConfig = field(default_factory=SQLiteLTMConfig)
    
    # Embedding and compression configuration
    autoencoder_config: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    
    # Memory management settings
    cleanup_interval: int = 100  # Check for cleanup every N insertions
    memory_priority_decay: float = 0.95  # How quickly priority decays
    
    # Advanced settings
    enable_memory_hooks: bool = True  # Whether to install memory hooks
    logging_level: str = "INFO"  # Logging level for memory operations
```

### 2.3. Core Memory Classes

Create `farm/memory/agent_memory/core.py`:

```python
"""Core classes for the agent memory system."""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union

from farm.memory.agent_memory.config import MemoryConfig
from farm.memory.agent_memory.memory_agent import MemoryAgent

logger = logging.getLogger(__name__)


class AgentMemorySystem:
    """Central manager for all agent memory components.
    
    This class serves as the main entry point for the agent memory system,
    managing memory agents for multiple agents and providing global configuration.
    
    Attributes:
        config: Configuration for the memory system
        agents: Dictionary of agent_id to MemoryAgent instances
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls, config: Optional[MemoryConfig] = None) -> 'AgentMemorySystem':
        """Get or create the singleton instance of the AgentMemorySystem."""
        if cls._instance is None:
            cls._instance = cls(config or MemoryConfig())
        return cls._instance
    
    def __init__(self, config: MemoryConfig):
        """Initialize the AgentMemorySystem.
        
        Args:
            config: Configuration for the memory system
        """
        self.config = config
        self.agents: Dict[str, MemoryAgent] = {}
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, config.logging_level))
        logger.info("AgentMemorySystem initialized with configuration: %s", config)
    
    def get_memory_agent(self, agent_id: str) -> MemoryAgent:
        """Get or create a MemoryAgent for the specified agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            MemoryAgent instance for the specified agent
        """
        if agent_id not in self.agents:
            self.agents[agent_id] = MemoryAgent(agent_id, self.config)
            logger.debug("Created new MemoryAgent for agent %s", agent_id)
        
        return self.agents[agent_id]
    
    def store_agent_state(
        self, 
        agent_id: str, 
        state_data: Dict[str, Any], 
        step_number: int,
        priority: float = 1.0
    ) -> bool:
        """Store an agent's state in memory.
        
        Args:
            agent_id: Unique identifier for the agent
            state_data: Dictionary of state attributes
            step_number: Current simulation step
            priority: Importance of this memory (0.0-1.0)
            
        Returns:
            True if storage was successful
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.store_state(state_data, step_number, priority)
    
    def store_agent_interaction(
        self,
        agent_id: str, 
        interaction_data: Dict[str, Any], 
        step_number: int,
        priority: float = 1.0
    ) -> bool:
        """Store information about an agent's interaction.
        
        Args:
            agent_id: Unique identifier for the agent
            interaction_data: Dictionary of interaction attributes
            step_number: Current simulation step
            priority: Importance of this memory (0.0-1.0)
            
        Returns:
            True if storage was successful
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.store_interaction(interaction_data, step_number, priority)
    
    def store_agent_action(
        self,
        agent_id: str, 
        action_data: Dict[str, Any], 
        step_number: int,
        priority: float = 1.0
    ) -> bool:
        """Store information about an agent's action.
        
        Args:
            agent_id: Unique identifier for the agent
            action_data: Dictionary of action attributes
            step_number: Current simulation step
            priority: Importance of this memory (0.0-1.0)
            
        Returns:
            True if storage was successful
        """
        memory_agent = self.get_memory_agent(agent_id)
        return memory_agent.store_action(action_data, step_number, priority)
    
    def clear_all_memories(self) -> bool:
        """Clear all memory data for all agents.
        
        Returns:
            True if clearing was successful
        """
        success = True
        for agent_id, memory_agent in self.agents.items():
            if not memory_agent.clear_memory():
                logger.error("Failed to clear memory for agent %s", agent_id)
                success = False
        
        # Reset agent dictionary
        self.agents = {}
        return success
```

### 2.4. Memory Agent Implementation

Create `farm/memory/agent_memory/memory_agent.py`:

```python
"""Memory Agent implementation for agent state management."""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from farm.memory.agent_memory.config import MemoryConfig
from farm.memory.agent_memory.storage.redis_stm import RedisSTMStore
from farm.memory.agent_memory.storage.redis_im import RedisIMStore
from farm.memory.agent_memory.storage.sqlite_ltm import SQLiteLTMStore
from farm.memory.agent_memory.embeddings.compression import CompressionEngine
from farm.memory.agent_memory.embeddings.autoencoder import AutoencoderEmbeddingEngine

logger = logging.getLogger(__name__)


class MemoryAgent:
    """Manages an agent's memory across hierarchical storage tiers.
    
    This class provides a unified interface for storing and retrieving
    agent memories across different storage tiers with varying levels
    of compression and resolution.
    
    Attributes:
        agent_id: Unique identifier for the agent
        config: Configuration for the memory agent
        stm_store: Short-Term Memory store (Redis)
        im_store: Intermediate Memory store (Redis with TTL)
        ltm_store: Long-Term Memory store (SQLite)
        compression_engine: Engine for compressing memory entries
        embedding_engine: Optional neural embedding engine
    """
    
    def __init__(self, agent_id: str, config: MemoryConfig):
        """Initialize the MemoryAgent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration for the memory agent
        """
        self.agent_id = agent_id
        self.config = config
        
        # Initialize memory stores
        self.stm_store = RedisSTMStore(agent_id, config.stm_config)
        self.im_store = RedisIMStore(agent_id, config.im_config)
        self.ltm_store = SQLiteLTMStore(agent_id, config.ltm_config)
        
        # Initialize compression engine
        self.compression_engine = CompressionEngine(config.autoencoder_config)
        
        # Optional: Initialize neural embedding engine for advanced vectorization
        if config.autoencoder_config.use_neural_embeddings:
            self.embedding_engine = AutoencoderEmbeddingEngine(
                model_path=config.autoencoder_config.model_path,
                input_dim=config.autoencoder_config.input_dim,
                stm_dim=config.autoencoder_config.stm_dim,
                im_dim=config.autoencoder_config.im_dim,
                ltm_dim=config.autoencoder_config.ltm_dim
            )
        else:
            self.embedding_engine = None
        
        # Internal state
        self._insert_count = 0
        
        logger.debug("MemoryAgent initialized for agent %s", agent_id)
    
    def store_state(
        self, 
        state_data: Dict[str, Any], 
        step_number: int,
        priority: float = 1.0
    ) -> bool:
        """Store an agent state in memory.
        
        Args:
            state_data: Dictionary containing agent state attributes
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)
            
        Returns:
            True if storage was successful
        """
        # Create a standardized memory entry
        memory_entry = self._create_memory_entry(
            state_data, 
            step_number, 
            "state", 
            priority
        )
        
        # Store in Short-Term Memory first
        success = self.stm_store.store(memory_entry)
        
        # Increment insert count and check if cleanup is needed
        self._insert_count += 1
        if self._insert_count % self.config.cleanup_interval == 0:
            self._check_memory_transition()
        
        return success
    
    def store_interaction(
        self, 
        interaction_data: Dict[str, Any], 
        step_number: int,
        priority: float = 1.0
    ) -> bool:
        """Store an agent interaction in memory.
        
        Args:
            interaction_data: Dictionary containing interaction details
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)
            
        Returns:
            True if storage was successful
        """
        # Create a standardized memory entry
        memory_entry = self._create_memory_entry(
            interaction_data, 
            step_number, 
            "interaction", 
            priority
        )
        
        # Store in Short-Term Memory first
        success = self.stm_store.store(memory_entry)
        
        # Increment insert count and check if cleanup is needed
        self._insert_count += 1
        if self._insert_count % self.config.cleanup_interval == 0:
            self._check_memory_transition()
        
        return success
    
    def store_action(
        self, 
        action_data: Dict[str, Any], 
        step_number: int,
        priority: float = 1.0
    ) -> bool:
        """Store an agent action in memory.
        
        Args:
            action_data: Dictionary containing action details
            step_number: Current simulation step number
            priority: Importance of this memory (0.0-1.0)
            
        Returns:
            True if storage was successful
        """
        # Create a standardized memory entry
        memory_entry = self._create_memory_entry(
            action_data, 
            step_number, 
            "action", 
            priority
        )
        
        # Store in Short-Term Memory first
        success = self.stm_store.store(memory_entry)
        
        # Increment insert count and check if cleanup is needed
        self._insert_count += 1
        if self._insert_count % self.config.cleanup_interval == 0:
            self._check_memory_transition()
        
        return success
    
    def _create_memory_entry(
        self, 
        data: Dict[str, Any], 
        step_number: int,
        memory_type: str,
        priority: float
    ) -> Dict[str, Any]:
        """Create a standardized memory entry.
        
        Args:
            data: Raw data to store
            step_number: Current simulation step
            memory_type: Type of memory ("state", "interaction", "action")
            priority: Importance of this memory (0.0-1.0)
            
        Returns:
            Formatted memory entry
        """
        # Generate unique memory ID
        timestamp = int(time.time())
        memory_id = f"{self.agent_id}-{step_number}-{timestamp}"
        
        # Generate embeddings if available
        embeddings = {}
        if self.embedding_engine:
            embeddings = {
                "full_vector": self.embedding_engine.encode_stm(data),
                "compressed_vector": self.embedding_engine.encode_im(data),
                "abstract_vector": self.embedding_engine.encode_ltm(data)
            }
        
        # Create standardized memory entry
        return {
            "memory_id": memory_id,
            "agent_id": self.agent_id,
            "step_number": step_number,
            "timestamp": timestamp,
            
            "contents": data,
            
            "metadata": {
                "creation_time": timestamp,
                "last_access_time": timestamp,
                "compression_level": 0,
                "importance_score": priority,
                "retrieval_count": 0,
                "memory_type": memory_type
            },
            
            "embeddings": embeddings
        }
    
    def _check_memory_transition(self) -> None:
        """Check if memories need to be transitioned between tiers.
        
        This method handles the movement of memories from STM to IM
        and from IM to LTM based on capacity and age.
        """
        # Check if STM is at capacity
        stm_count = self.stm_store.count()
        if stm_count > self.config.stm_config.memory_limit:
            # Get oldest entries to transition to IM
            overflow = stm_count - self.config.stm_config.memory_limit
            oldest_entries = self.stm_store.get_oldest(overflow)
            
            # Compress and store in IM
            for entry in oldest_entries:
                # Apply level 1 compression
                compressed_entry = self.compression_engine.compress(entry, level=1)
                compressed_entry["metadata"]["compression_level"] = 1
                
                # Store in IM
                self.im_store.store(compressed_entry)
                
                # Remove from STM
                self.stm_store.delete(entry["memory_id"])
            
            logger.debug("Transitioned %d memories from STM to IM for agent %s", 
                        overflow, self.agent_id)
        
        # Check if IM is at capacity
        im_count = self.im_store.count()
        if im_count > self.config.im_config.memory_limit:
            # Get oldest entries to transition to LTM
            overflow = im_count - self.config.im_config.memory_limit
            oldest_entries = self.im_store.get_oldest(overflow)
            
            # Compress and store in LTM
            batch = []
            for entry in oldest_entries:
                # Apply level 2 compression
                compressed_entry = self.compression_engine.compress(entry, level=2)
                compressed_entry["metadata"]["compression_level"] = 2
                
                # Add to batch
                batch.append(compressed_entry)
                
                # Remove from IM
                self.im_store.delete(entry["memory_id"])
                
                # Process in batches
                if len(batch) >= self.config.ltm_config.batch_size:
                    self.ltm_store.store_batch(batch)
                    batch = []
            
            # Store any remaining entries
            if batch:
                self.ltm_store.store_batch(batch)
            
            logger.debug("Transitioned %d memories from IM to LTM for agent %s", 
                        overflow, self.agent_id)
    
    def clear_memory(self) -> bool:
        """Clear all memory data for this agent.
        
        Returns:
            True if clearing was successful
        """
        stm_success = self.stm_store.clear()
        im_success = self.im_store.clear()
        ltm_success = self.ltm_store.clear()
        
        return stm_success and im_success and ltm_success
```

### 2.5. Storage Implementation Example (RedisSTM)

Create `farm/memory/agent_memory/storage/redis_stm.py`:

```python
"""Redis-based storage for Short-Term Memory (STM)."""

import json
import logging
import time
from typing import Dict, Any, List, Optional, Union

import redis

from farm.memory.agent_memory.config import RedisSTMConfig

logger = logging.getLogger(__name__)


class RedisSTMStore:
    """Redis-based storage for Short-Term Memory (STM).
    
    This class implements the storage interface for the Short-Term Memory tier,
    providing high-performance, full-resolution memory storage with Redis.
    
    Attributes:
        agent_id: Unique identifier for the agent
        config: Configuration for the STM store
        redis: Redis client instance
    """
    
    def __init__(self, agent_id: str, config: RedisSTMConfig):
        """Initialize the RedisSTMStore.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration for the STM store
        """
        self.agent_id = agent_id
        self.config = config
        
        # Initialize Redis client
        self.redis = redis.Redis(
            host=config.host,
            port=config.port,
            db=config.db,
            password=config.password,
            decode_responses=True
        )
        
        logger.debug("RedisSTMStore initialized for agent %s", agent_id)
    
    @property
    def _key_prefix(self) -> str:
        """Get the Redis key prefix for this agent's STM.
        
        Returns:
            Redis key prefix
        """
        return f"{self.config.namespace}:{self.agent_id}"
    
    def store(self, memory_entry: Dict[str, Any]) -> bool:
        """Store a memory entry in Redis.
        
        Args:
            memory_entry: Memory entry to store
            
        Returns:
            True if storage was successful
        """
        try:
            memory_id = memory_entry["memory_id"]
            
            # Store the main memory entry as a JSON string
            key = f"{self._key_prefix}:memory:{memory_id}"
            self.redis.set(
                key,
                json.dumps(memory_entry),
                ex=self.config.ttl
            )
            
            # Add to timeline sorted set for chronological retrieval
            self.redis.zadd(
                f"{self._key_prefix}:timeline",
                {memory_id: memory_entry["step_number"]}
            )
            
            # Add to type index for type-based retrieval
            memory_type = memory_entry["metadata"]["memory_type"]
            self.redis.sadd(
                f"{self._key_prefix}:type:{memory_type}",
                memory_id
            )
            
            # Add to importance index for importance-based retrieval
            importance = memory_entry["metadata"]["importance_score"]
            self.redis.zadd(
                f"{self._key_prefix}:importance",
                {memory_id: importance}
            )
            
            # If it has embeddings, store for vector search
            if memory_entry["embeddings"] and "full_vector" in memory_entry["embeddings"]:
                # Serialize the vector for storage
                vector_key = f"{self._key_prefix}:vector:{memory_id}"
                vector = memory_entry["embeddings"]["full_vector"]
                self.redis.set(
                    vector_key,
                    json.dumps(vector),
                    ex=self.config.ttl
                )
            
            return True
        except Exception as e:
            logger.error("Failed to store memory in Redis: %s", str(e))
            return False
    
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a memory entry by ID.
        
        Args:
            memory_id: Unique identifier for the memory
            
        Returns:
            Memory entry or None if not found
        """
        try:
            key = f"{self._key_prefix}:memory:{memory_id}"
            data = self.redis.get(key)
            
            if data:
                # Update access time
                memory_entry = json.loads(data)
                memory_entry["metadata"]["last_access_time"] = int(time.time())
                memory_entry["metadata"]["retrieval_count"] += 1
                
                # Update in Redis
                self.redis.set(
                    key,
                    json.dumps(memory_entry),
                    ex=self.config.ttl
                )
                
                return memory_entry
            
            return None
        except Exception as e:
            logger.error("Failed to retrieve memory from Redis: %s", str(e))
            return None
    
    def get_by_step(self, step_number: int) -> Optional[Dict[str, Any]]:
        """Retrieve a memory entry by step number.
        
        Args:
            step_number: Simulation step number
            
        Returns:
            Memory entry or None if not found
        """
        try:
            # Use sorted set to find memory by step
            memory_ids = self.redis.zrangebyscore(
                f"{self._key_prefix}:timeline",
                step_number,
                step_number
            )
            
            if memory_ids:
                return self.get(memory_ids[0])
            
            return None
        except Exception as e:
            logger.error("Failed to retrieve memory by step: %s", str(e))
            return None
    
    def get_recent(self, count: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the most recent memory entries.
        
        Args:
            count: Maximum number of entries to retrieve
            
        Returns:
            List of memory entries
        """
        try:
            # Get the most recent memory IDs from timeline
            memory_ids = self.redis.zrevrange(
                f"{self._key_prefix}:timeline",
                0,
                count - 1
            )
            
            # Retrieve full memory entries
            memories = []
            for memory_id in memory_ids:
                memory = self.get(memory_id)
                if memory:
                    memories.append(memory)
            
            return memories
        except Exception as e:
            logger.error("Failed to retrieve recent memories: %s", str(e))
            return []
    
    def get_oldest(self, count: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the oldest memory entries.
        
        Args:
            count: Maximum number of entries to retrieve
            
        Returns:
            List of memory entries
        """
        try:
            # Get the oldest memory IDs from timeline
            memory_ids = self.redis.zrange(
                f"{self._key_prefix}:timeline",
                0,
                count - 1
            )
            
            # Retrieve full memory entries
            memories = []
            for memory_id in memory_ids:
                memory = self.get(memory_id)
                if memory:
                    memories.append(memory)
            
            return memories
        except Exception as e:
            logger.error("Failed to retrieve oldest memories: %s", str(e))
            return []
    
    def count(self) -> int:
        """Get the total number of memories in storage.
        
        Returns:
            Number of memories
        """
        try:
            return self.redis.zcard(f"{self._key_prefix}:timeline")
        except Exception as e:
            logger.error("Failed to get memory count: %s", str(e))
            return 0
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry.
        
        Args:
            memory_id: Unique identifier for the memory
            
        Returns:
            True if deletion was successful
        """
        try:
            # Get the memory to determine its type
            memory = self.get(memory_id)
            if not memory:
                return False
            
            memory_type = memory["metadata"]["memory_type"]
            
            # Remove from all indexes
            self.redis.delete(f"{self._key_prefix}:memory:{memory_id}")
            self.redis.zrem(f"{self._key_prefix}:timeline", memory_id)
            self.redis.srem(f"{self._key_prefix}:type:{memory_type}", memory_id)
            self.redis.zrem(f"{self._key_prefix}:importance", memory_id)
            self.redis.delete(f"{self._key_prefix}:vector:{memory_id}")
            
            return True
        except Exception as e:
            logger.error("Failed to delete memory: %s", str(e))
            return False
    
    def clear(self) -> bool:
        """Clear all memories for this agent.
        
        Returns:
            True if clearing was successful
        """
        try:
            # Use Redis key pattern to find and delete all keys
            pattern = f"{self._key_prefix}:*"
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(cursor, pattern, 100)
                if keys:
                    self.redis.delete(*keys)
                if cursor == 0:
                    break
            
            return True
        except Exception as e:
            logger.error("Failed to clear memories: %s", str(e))
            return False
```

### 2.6. API Implementation

Create `farm/memory/agent_memory/api/memory_api.py`:

```python
"""API interface for the agent memory system."""

import logging
from typing import Dict, Any, List, Optional, Union

from farm.memory.agent_memory.config import MemoryConfig
from farm.memory.agent_memory.core import AgentMemorySystem

logger = logging.getLogger(__name__)


class AgentMemoryAPI:
    """Interface for storing and retrieving agent states in the hierarchical memory system.
    
    This class provides a clean, standardized API for interacting with the
    agent memory system, abstracting away the details of the underlying
    storage mechanisms.
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """Initialize the AgentMemoryAPI.
        
        Args:
            config: Configuration for the memory system
        """
        self.memory_system = AgentMemorySystem.get_instance(config)
    
    def store_agent_state(self, agent_id: str, state_data: Dict[str, Any], step_number: int) -> bool:
        """Store an agent's state in short-term memory.
        
        Args:
            agent_id: Unique identifier for the agent
            state_data: Dictionary containing agent state attributes
            step_number: Current simulation step number
            
        Returns:
            True if storage was successful
        """
        return self.memory_system.store_agent_state(
            agent_id, 
            state_data, 
            step_number
        )
        
    def store_agent_interaction(
        self, 
        agent_id: str, 
        interaction_data: Dict[str, Any], 
        step_number: int
    ) -> bool:
        """Store information about an agent's interaction with environment or other agents.
        
        Args:
            agent_id: Unique identifier for the agent
            interaction_data: Dictionary containing interaction details
            step_number: Current simulation step number
            
        Returns:
            True if storage was successful
        """
        return self.memory_system.store_agent_interaction(
            agent_id,
            interaction_data,
            step_number
        )
        
    def store_agent_action(
        self, 
        agent_id: str, 
        action_data: Dict[str, Any], 
        step_number: int
    ) -> bool:
        """Store information about an action taken by an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            action_data: Dictionary containing action details
            step_number: Current simulation step number
            
        Returns:
            True if storage was successful
        """
        return self.memory_system.store_agent_action(
            agent_id,
            action_data,
            step_number
        )
    
    def retrieve_state_by_id(self, agent_id: str, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID.
        
        Args:
            agent_id: Unique identifier for the agent
            memory_id: Unique identifier for the memory
            
        Returns:
            Memory entry or None if not found
        """
        memory_agent = self.memory_system.get_memory_agent(agent_id)
        # Try retrieving from each tier in order
        memory = memory_agent.stm_store.get(memory_id)
        if not memory:
            memory = memory_agent.im_store.get(memory_id)
        if not memory:
            memory = memory_agent.ltm_store.get(memory_id)
        return memory
    
    def retrieve_recent_states(self, agent_id: str, count: int = 10) -> List[Dict[str, Any]]:
        """Retrieve the most recent states for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            count: Maximum number of states to retrieve
            
        Returns:
            List of memory entries
        """
        memory_agent = self.memory_system.get_memory_agent(agent_id)
        return memory_agent.stm_store.get_recent(count)
    
    def retrieve_states_by_similarity(
        self, 
        agent_id: str, 
        query_state: Dict[str, Any],
        count: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve states similar to the query state.
        
        Args:
            agent_id: Unique identifier for the agent
            query_state: State to compare against
            count: Maximum number of states to retrieve
            
        Returns:
            List of memory entries
        """
        # This would be implemented using vector similarity search
        # and would need to use the embedding engine
        memory_agent = self.memory_system.get_memory_agent(agent_id)
        
        if not memory_agent.embedding_engine:
            logger.warning("Vector similarity search requires embedding engine to be enabled")
            return []
        
        # This is a placeholder - actual implementation would use Redis vector search
        # or another vector similarity approach
        # For simplicity, return recent states
        return memory_agent.stm_store.get_recent(count)
    
    def clear_agent_memory(self, agent_id: str) -> bool:
        """Clear all memory for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            True if clearing was successful
        """
        memory_agent = self.memory_system.get_memory_agent(agent_id)
        return memory_agent.clear_memory()
```

### Create `farm/memory/agent_memory/api/hooks.py`:

```python
"""Memory hooks for integrating with agent lifecycle events."""

import functools
import logging
from typing import Any, Callable, Dict, Optional, Type

from farm.agents.base_agent import BaseAgent
from farm.memory.agent_memory.config import MemoryConfig
from farm.memory.agent_memory.core import AgentMemorySystem

logger = logging.getLogger(__name__)


def install_memory_hooks(agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
    """Install memory hooks on an agent class.
    
    This is a class decorator that adds memory hooks to BaseAgent subclasses.
    
    Args:
        agent_class: The agent class to install hooks on
        
    Returns:
        The modified agent class
    """
    original_init = agent_class.__init__
    original_act = agent_class.act
    original_get_state = agent_class.get_state
    
    @functools.wraps(original_init)
    def init_with_memory(self, *args, **kwargs):
        """Initialize with memory system support."""
        original_init(self, *args, **kwargs)
        
        # Get memory system
        memory_config = getattr(self.config, "memory_config", None)
        if isinstance(memory_config, dict):
            memory_config = MemoryConfig(**memory_config)
        
        self.memory_system = AgentMemorySystem.get_instance(memory_config)
    
    @functools.wraps(original_act)
    def act_with_memory(self, *args, **kwargs):
        """Act with memory integration."""
        # Get state before action
        state_before = self.get_state()
        step_number = getattr(self, "step_number", 0)
        
        # Call original act method
        result = original_act(self, *args, **kwargs)
        
        # Get state after action
        state_after = self.get_state()
        
        # Create action record
        action_data = {
            "action_type": getattr(result, "action_type", "unknown"),
            "action_params": getattr(result, "params", {}),
            "state_before": state_before,
            "state_after": state_after,
            "reward": getattr(result, "reward", 0.0),
        }
        
        # Store in memory
        self.memory_system.store_agent_action(
            self.agent_id,
            action_data,
            step_number
        )
        
        return result
    
    @functools.wraps(original_get_state)
    def get_state_with_memory(self, *args, **kwargs):
        """Get state with memory integration."""
        # Call original get_state method
        state = original_get_state(self, *args, **kwargs)
        
        # Store in memory if not already storing in act method
        if not hasattr(self, "_memory_recording"):
            step_number = getattr(self, "step_number", 0)
            self.memory_system.store_agent_state(
                self.agent_id,
                state,
                step_number
            )
        
        return state
    
    # Replace methods
    agent_class.__init__ = init_with_memory
    agent_class.act = act_with_memory
    agent_class.get_state = get_state_with_memory
    
    return agent_class


def with_memory(agent_instance: BaseAgent) -> BaseAgent:
    """Add memory capabilities to an existing agent instance.
    
    Args:
        agent_instance: The agent instance to add memory to
        
    Returns:
        The agent with memory capabilities
    """
    # Create a dynamic subclass with memory hooks
    agent_class = type(agent_instance)
    memory_class = type(
        f"{agent_class.__name__}WithMemory",
        (agent_class,),
        {}
    )
    
    # Install hooks on the new class
    memory_class = install_memory_hooks(memory_class)
    
    # Update the instance's class
    agent_instance.__class__ = memory_class
    
    return agent_instance
```

### Create `farm/memory/agent_memory/embeddings/autoencoder.py`:

```python
"""Autoencoder-based embeddings for agent memory states."""

import logging
import os
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from farm.memory.agent_memory.config import AutoencoderConfig

logger = logging.getLogger(__name__)


class StateAutoencoder(nn.Module):
    """Neural network autoencoder for agent state vectorization and compression.
    
    The autoencoder consists of:
    1. An encoder that compresses input features to the embedding space
    2. A decoder that reconstructs original features from embeddings
    3. Multiple "bottlenecks" for different compression levels
    """
    
    def __init__(self, input_dim: int, stm_dim: int = 384, im_dim: int = 128, ltm_dim: int = 32):
        """Initialize the multi-resolution autoencoder.
        
        Args:
            input_dim: Dimension of the flattened input features
            stm_dim: Dimension for Short-Term Memory (STM) embeddings
            im_dim: Dimension for Intermediate Memory (IM) embeddings
            ltm_dim: Dimension for Long-Term Memory (LTM) embeddings
        """
        super(StateAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        # Multi-resolution bottlenecks
        self.stm_bottleneck = nn.Linear(256, stm_dim)
        self.im_bottleneck = nn.Linear(stm_dim, im_dim)
        self.ltm_bottleneck = nn.Linear(im_dim, ltm_dim)
        
        # Expansion layers (from LTM to IM to STM)
        self.ltm_to_im = nn.Linear(ltm_dim, im_dim)
        self.im_to_stm = nn.Linear(im_dim, stm_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(stm_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )
    
    def encode_stm(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to STM embedding space.
        
        Args:
            x: Input tensor
            
        Returns:
            STM embedding
        """
        x = self.encoder(x)
        return self.stm_bottleneck(x)
    
    def encode_im(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to IM embedding space.
        
        Args:
            x: Input tensor
            
        Returns:
            IM embedding
        """
        x = self.encoder(x)
        x = self.stm_bottleneck(x)
        return self.im_bottleneck(x)
    
    def encode_ltm(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to LTM embedding space.
        
        Args:
            x: Input tensor
            
        Returns:
            LTM embedding
        """
        x = self.encoder(x)
        x = self.stm_bottleneck(x)
        x = self.im_bottleneck(x)
        return self.ltm_bottleneck(x)
    
    def decode_stm(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from STM embedding space.
        
        Args:
            z: STM embedding
            
        Returns:
            Reconstructed input
        """
        return self.decoder(z)
    
    def decode_im(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from IM embedding space.
        
        Args:
            z: IM embedding
            
        Returns:
            Reconstructed input
        """
        z = self.im_to_stm(z)
        return self.decoder(z)
    
    def decode_ltm(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from LTM embedding space.
        
        Args:
            z: LTM embedding
            
        Returns:
            Reconstructed input
        """
        z = self.ltm_to_im(z)
        z = self.im_to_stm(z)
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor, level: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder.
        
        Args:
            x: Input tensor
            level: Compression level (0=STM, 1=IM, 2=LTM)
            
        Returns:
            Tuple of (reconstructed output, embedding)
        """
        # Encode
        encoded = self.encoder(x)
        
        if level == 0:  # STM
            embedding = self.stm_bottleneck(encoded)
            decoded = self.decoder(embedding)
        elif level == 1:  # IM
            stm_embedding = self.stm_bottleneck(encoded)
            embedding = self.im_bottleneck(stm_embedding)
            expanded = self.im_to_stm(embedding)
            decoded = self.decoder(expanded)
        else:  # LTM
            stm_embedding = self.stm_bottleneck(encoded)
            im_embedding = self.im_bottleneck(stm_embedding)
            embedding = self.ltm_bottleneck(im_embedding)
            expanded_im = self.ltm_to_im(embedding)
            expanded_stm = self.im_to_stm(expanded_im)
            decoded = self.decoder(expanded_stm)
        
        return decoded, embedding


class AgentStateDataset(Dataset):
    """Dataset for training the autoencoder on agent states."""
    
    def __init__(self, states: List[Dict[str, Any]]):
        """Initialize the dataset.
        
        Args:
            states: List of agent state dictionaries
        """
        self.states = states
        self.vectors = self._prepare_vectors()
    
    def _prepare_vectors(self) -> np.ndarray:
        """Convert agent states to input vectors.
        
        Returns:
            Numpy array of flattened state vectors
        """
        # Extract numeric values from states
        vectors = []
        for state in self.states:
            # Extract all numeric values from the state dictionary
            vector = []
            for key, value in self._flatten_dict(state).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    vector.append(float(value))
            vectors.append(vector)
        
        # Pad to ensure uniform length
        max_len = max(len(v) for v in vectors)
        padded_vectors = []
        for v in vectors:
            padded = v + [0.0] * (max_len - len(v))
            padded_vectors.append(padded)
        
        return np.array(padded_vectors, dtype=np.float32)
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            prefix: Key prefix for nested dictionaries
            
        Returns:
            Flattened dictionary
        """
        result = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(self._flatten_dict(v, key))
            else:
                result[key] = v
        return result
    
    def __len__(self) -> int:
        """Get the number of samples.
        
        Returns:
            Dataset size
        """
        return len(self.vectors)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a sample by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tensor representation of the state
        """
        return torch.tensor(self.vectors[idx], dtype=torch.float32)


class AutoencoderEmbeddingEngine:
    """Engine for generating embeddings using the autoencoder model.
    
    This class handles the training and inference of the autoencoder model,
    providing methods to encode and decode agent states at different
    compression levels.
    
    Attributes:
        model: The autoencoder model
        input_dim: Dimension of the input features
        stm_dim: Dimension of STM embeddings
        im_dim: Dimension of IM embeddings
        ltm_dim: Dimension of LTM embeddings
        device: Device to run the model on
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_dim: int = 64,
        stm_dim: int = 384,
        im_dim: int = 128,
        ltm_dim: int = 32,
    ):
        """Initialize the embedding engine.
        
        Args:
            model_path: Path to saved model (if available)
            input_dim: Dimension of the input features
            stm_dim: Dimension of STM embeddings
            im_dim: Dimension of IM embeddings
            ltm_dim: Dimension of LTM embeddings
        """
        self.input_dim = input_dim
        self.stm_dim = stm_dim
        self.im_dim = im_dim
        self.ltm_dim = ltm_dim
        
        # Use CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = StateAutoencoder(input_dim, stm_dim, im_dim, ltm_dim).to(self.device)
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            logger.info("Loaded autoencoder model from %s", model_path)
        else:
            logger.warning("No pre-trained model found. Using untrained model.")
    
    def encode_stm(self, state: Dict[str, Any]) -> List[float]:
        """Encode state to STM embedding space.
        
        Args:
            state: Agent state dictionary
            
        Returns:
            STM embedding as list of floats
        """
        vector = self._state_to_vector(state)
        with torch.no_grad():
            x = torch.tensor(vector, dtype=torch.float32).to(self.device)
            embedding = self.model.encode_stm(x)
        return embedding.cpu().numpy().tolist()
    
    def encode_im(self, state: Dict[str, Any]) -> List[float]:
        """Encode state to IM embedding space.
        
        Args:
            state: Agent state dictionary
            
        Returns:
            IM embedding as list of floats
        """
        vector = self._state_to_vector(state)
        with torch.no_grad():
            x = torch.tensor(vector, dtype=torch.float32).to(self.device)
            embedding = self.model.encode_im(x)
        return embedding.cpu().numpy().tolist()
    
    def encode_ltm(self, state: Dict[str, Any]) -> List[float]:
        """Encode state to LTM embedding space.
        
        Args:
            state: Agent state dictionary
            
        Returns:
            LTM embedding as list of floats
        """
        vector = self._state_to_vector(state)
        with torch.no_grad():
            x = torch.tensor(vector, dtype=torch.float32).to(self.device)
            embedding = self.model.encode_ltm(x)
        return embedding.cpu().numpy().tolist()
    
    def _state_to_vector(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert a state dictionary to a vector.
        
        Args:
            state: Agent state dictionary
            
        Returns:
            Numpy array of state features
        """
        # Create a dataset with just this state
        dataset = AgentStateDataset([state])
        
        # Get the vector
        vector = dataset.vectors[0]
        
        # Ensure correct dimension
        if len(vector) < self.input_dim:
            vector = np.pad(vector, (0, self.input_dim - len(vector)))
        elif len(vector) > self.input_dim:
            vector = vector[:self.input_dim]
        
        return vector
    
    def train(
        self,
        states: List[Dict[str, Any]],
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict[str, List[float]]:
        """Train the autoencoder on agent states.
        
        Args:
            states: List of agent state dictionaries
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dictionary of training metrics
        """
        # Create dataset and dataloader
        dataset = AgentStateDataset(states)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        metrics = {"loss": [], "stm_loss": [], "im_loss": [], "ltm_loss": []}
        
        for epoch in range(epochs):
            running_loss = 0.0
            running_stm_loss = 0.0
            running_im_loss = 0.0
            running_ltm_loss = 0.0
            
            for batch in dataloader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Train on all compression levels
                stm_output, _ = self.model(batch, level=0)
                im_output, _ = self.model(batch, level=1)
                ltm_output, _ = self.model(batch, level=2)
                
                # Calculate losses
                stm_loss = criterion(stm_output, batch)
                im_loss = criterion(im_output, batch)
                ltm_loss = criterion(ltm_output, batch)
                
                # Combined loss (weighted by importance)
                loss = stm_loss + 0.5 * im_loss + 0.25 * ltm_loss
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Update metrics
                running_loss += loss.item()
                running_stm_loss += stm_loss.item()
                running_im_loss += im_loss.item()
                running_ltm_loss += ltm_loss.item()
            
            # Record epoch metrics
            metrics["loss"].append(running_loss / len(dataloader))
            metrics["stm_loss"].append(running_stm_loss / len(dataloader))
            metrics["im_loss"].append(running_im_loss / len(dataloader))
            metrics["ltm_loss"].append(running_ltm_loss / len(dataloader))
            
            logger.info(
                f"Epoch {epoch+1}/{epochs}, Loss: {metrics['loss'][-1]:.4f}, "
                f"STM: {metrics['stm_loss'][-1]:.4f}, "
                f"IM: {metrics['im_loss'][-1]:.4f}, "
                f"LTM: {metrics['ltm_loss'][-1]:.4f}"
            )
        
        return metrics
    
    def save_model(self, path: str) -> None:
        """Save the model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "stm_dim": self.stm_dim,
            "im_dim": self.im_dim,
            "ltm_dim": self.ltm_dim,
        }, path)
        logger.info("Model saved to %s", path)
    
    def load_model(self, path: str) -> None:
        """Load the model from a file.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.stm_dim = checkpoint["stm_dim"]
        self.im_dim = checkpoint["im_dim"]
        self.ltm_dim = checkpoint["ltm_dim"]
        
        # Re-initialize model with loaded dimensions
        self.model = StateAutoencoder(
            self.input_dim, 
            self.stm_dim, 
            self.im_dim, 
            self.ltm_dim
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        logger.info("Model loaded from %s", path)
```

## 3. Implementation Notes

### 3.1 Implementation Sequence

I recommend the following implementation sequence:

1. Create the directory structure
2. Implement the configuration classes first (`config.py`)
3. Create the core system and memory agent frameworks (`core.py`, `memory_agent.py`)
4. Implement the storage backends in this order:
   - Redis STM store (`redis_stm.py`)
   - Redis IM store (`redis_im.py`) 
   - SQLite LTM store (`sqlite_ltm.py`)
5. Build the compression and embedding components
6. Implement the API interfaces
7. Create the agent integration hooks
8. Write tests for each component

### 3.2 Testing Approach

For each component, create corresponding test files in the `tests/` directory:

- `test_memory_agent.py`: Test the memory agent functionality
- `test_storage.py`: Test all storage backends
- `test_retrieval.py`: Test memory retrieval mechanisms
- `test_embeddings.py`: Test autoencoder and embedding functionality

Use fixtures to set up test data and mock Redis/SQLite backends for unit testing.

### 3.3 Dependency Management

The implementation requires these additional dependencies:

- `redis`: For Redis connection
- `torch`: For autoencoder implementation
- `numpy`: For vector manipulation
- `sqlalchemy`: For SQLite integration

Add these to your `requirements.txt` file if not already present.

## 4. Integration with Existing Code

The implementation uses the singleton pattern for the `AgentMemorySystem` to ensure there's only one instance of the memory system across the application. This allows for easy integration with existing code through:

1. `AgentMemoryAPI`: Primary interface for direct interaction with the memory system
2. `install_memory_hooks`: Decorator for agent classes to automatically add memory capabilities
3. `with_memory`: Function to add memory capabilities to existing agent instances

You can start with minimal integration by just using the API, and then gradually adopt the hooks for deeper integration.

This implementation plan follows the architecture in the documentation and provides a flexible, extensible foundation for the AgentMemory system.

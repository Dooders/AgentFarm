# **Redis Integration for AgentMemory**

## **1. Introduction**

This document outlines how Redis is integrated with the AgentMemory system to provide efficient storage and retrieval of agent states. We leverage Redis data structures to optimize for the most common access patterns in agent simulations. For specifics on the Redis schema for agent state storage, please see the [Redis Index Schema](redis_index_schema.md) document. For action storage details, see the [Agent Action Redis Schema](agent_action_redis_schema.md), and for social interaction storage, see the [Agent Interaction Redis Schema](agent_interaction_redis_schema.md).

## **2. Current Architecture Analysis**

AgentFarm currently uses:
- **`DataLogger`** class for buffered writes to SQLite
- **`SimulationDatabase`** as the main interface
- **SQLAlchemy ORM** with buffered operations and transaction safety
- Multiple data model classes (`AgentModel`, `ActionModel`, etc.)

While the current buffering mechanism helps performance, **the system remains bottlenecked by SQLite's write constraints** during high-throughput simulation runs.

## **3. Redis Integration Architecture**

### **3.1 System Components**

1. **RedisBufferedDataLogger**: An enhanced version of the current `DataLogger` class that writes to Redis first
2. **SQLitePersistenceWorker**: Background process for Redis → SQLite transfers
3. **Redis Server**: In-memory data store configured with appropriate persistence
4. **Existing SQLite Database**: Remains as the persistent storage solution

### **3.2 Data Flow**

```
Simulation → RedisBufferedDataLogger → Redis → SQLitePersistenceWorker → SQLite
              (non-blocking writes)     (memory)   (background process)   (disk)
```

This data flow aligns with the memory transition flow described in [Core Concepts: Memory Transition Flow](core_concepts.md#22-memory-transition-flow).

## **4. Implementation Details**

### **4.1 RedisBufferedDataLogger Class**

Extending the current `DataLogger` class in `farm/database/data_logging.py`:

```python
class RedisBufferedDataLogger(DataLogger):
    """Enhanced DataLogger that uses Redis as a write buffer."""
    
    def __init__(
        self,
        database: 'SimulationDatabase',
        simulation_id: Optional[str] = None,
        config: Optional[DataLoggingConfig] = None,
        redis_stm_config: Optional[RedisSTMConfig] = None,
    ):
        super().__init__(database, simulation_id, config)
        
        # Use STM configuration for buffered writes since STM is
        # the most recent, high-frequency memory tier
        self.redis_stm_config = redis_stm_config or RedisSTMConfig()
        self.redis_client = redis.Redis(**self.redis_stm_config.connection_params)
        self.namespace = self.redis_stm_config.namespace
        
        # Maintain existing buffer structure for fallback
        self._action_buffer = []
        # ... other buffers
```

### **4.2 Redis Integration Methods**

These methods implement the memory operations defined in [Core Concepts: Memory Operations](core_concepts.md#4-memory-compression-techniques):

```python
def log_agent_action(self, step_number, agent_id, action_type, ...):
    """Buffer agent action in Redis"""
    action_data = {
        "simulation_id": self.simulation_id,
        "step_number": step_number,
        "agent_id": agent_id,
        "action_type": action_type,
        # ... other fields
    }
    
    try:
        # Serialize and push to Redis list
        self.redis_client.rpush(
            f"{self.namespace}:{self.simulation_id}:actions", 
            json.dumps(action_data)
        )
    except redis.exceptions.RedisError as e:
        # Fallback to SQLite buffer
        logger.warning(f"Redis write failed, falling back to buffer: {e}")
        self._action_buffer.append(action_data)
        
        # Auto-flush if buffer is full
        if len(self._action_buffer) >= self._buffer_size:
            self.flush_action_buffer()
```

### **4.3 SQLitePersistenceWorker Class**

This class implements the memory transition process between STM and LTM tiers:

```python
class SQLitePersistenceWorker:
    """Background worker that flushes Redis data to SQLite."""
    
    def __init__(
        self, 
        database: 'SimulationDatabase',
        redis_stm_config: RedisSTMConfig,
        redis_im_config: Optional[RedisIMConfig] = None,
        batch_size: int = 1000,
        flush_interval: int = 5,
    ):
        self.db = database
        self.redis_stm_config = redis_stm_config
        self.stm_client = redis.Redis(**redis_stm_config.connection_params)
        self.stm_namespace = redis_stm_config.namespace
        
        # Optional IM client for memory transition
        self.redis_im_config = redis_im_config
        if redis_im_config:
            self.im_client = redis.Redis(**redis_im_config.connection_params)
            self.im_namespace = redis_im_config.namespace
        else:
            self.im_client = None
            self.im_namespace = None
            
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._stop_event = threading.Event()
        self._worker_thread = None
        
    def start(self):
        """Start the background worker thread."""
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self._worker_thread.start()
        
    def _worker_loop(self):
        """Main worker loop that periodically flushes data."""
        while not self._stop_event.is_set():
            try:
                self._flush_all_keys()
            except Exception as e:
                logger.error(f"Error in persistence worker: {e}")
            
            # Sleep until next flush interval
            self._stop_event.wait(self.flush_interval)
            
    def _flush_all_keys(self):
        """Flush all Redis keys for all simulations."""
        # Find all simulation keys
        simulation_keys = self.stm_client.keys(f"{self.stm_namespace}:*:actions")
        for key in simulation_keys:
            self._flush_key(key)
            
    # ... flush methods for each data type
```

### **4.4 Redis Configuration**

```python
@dataclass
class RedisConfig:
    """Configuration for Redis connection and behavior."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0  # Default database
    password: Optional[str] = None
    
    @property
    def connection_params(self):
        """Return connection parameters for Redis client."""
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": self.password if self.password else None,
        }
```

### **4.5 Memory Tier-Specific Redis Configurations**

```python
@dataclass
class RedisSTMConfig(RedisConfig):
    """Configuration for Short-Term Memory Redis."""
    db: int = 0  # STM uses database 0
    namespace: str = "agent_memory:stm"
    ttl: int = 86400  # 24 hours
    memory_limit: int = 1000  # Max entries per agent


@dataclass
class RedisIMConfig(RedisConfig):
    """Configuration for Intermediate Memory Redis."""
    db: int = 1  # IM uses database 1
    namespace: str = "agent_memory:im"
    ttl: int = 604800  # 7 days
    memory_limit: int = 10000  # Max entries per agent
    compression_level: int = 1  # Level 1 compression
```

This configuration structure aligns with the memory tier separation described in our architecture, ensuring that the Short-Term Memory (STM) and Intermediate Memory (IM) are stored in separate Redis databases to prevent data conflicts.

## **5. Redis Memory Tier Implementation**

This implementation provides the Redis-based storage for the STM and IM memory tiers as defined in [Core Concepts: Memory Tiers](core_concepts.md#21-memory-tiers).

### **5.1 STM Redis Implementation**

```python
class RedisSTMStore:
    """Redis-based implementation of Short-Term Memory store."""
    
    def __init__(self, agent_id, config: RedisSTMConfig):
        self.agent_id = agent_id
        self.config = config
        self.redis_client = redis.Redis(**config.connection_params)
        self.namespace = config.namespace
        
    def store(self, memory_entry):
        """Store a memory entry in STM."""
        memory_id = memory_entry["memory_id"]
        
        # Store the full entry
        self.redis_client.hset(
            f"{self.namespace}:agent:{self.agent_id}", 
            memory_id,
            json.dumps(memory_entry)
        )
        
        # Add to timeline index
        self.redis_client.zadd(
            f"{self.namespace}:agent:{self.agent_id}:timeline",
            {memory_id: memory_entry["step_number"]}
        )
        
        # Set TTL on keys
        self.redis_client.expire(
            f"{self.namespace}:agent:{self.agent_id}",
            self.config.ttl
        )
        self.redis_client.expire(
            f"{self.namespace}:agent:{self.agent_id}:timeline",
            self.config.ttl
        )
```

### **5.2 IM Redis Implementation**

```python
class RedisIMStore:
    """Redis-based implementation of Intermediate Memory store with TTL."""
    
    def __init__(self, agent_id, config: RedisIMConfig):
        self.agent_id = agent_id
        self.config = config
        self.redis_client = redis.Redis(**config.connection_params)
        self.ttl = config.ttl  # Time-to-live in seconds
        self.namespace = config.namespace
        
    def store(self, memory_entry):
        """Store a compressed memory entry in IM with TTL."""
        memory_id = memory_entry["memory_id"]
        
        # Store the compressed entry
        self.redis_client.hset(
            f"{self.namespace}:agent:{self.agent_id}", 
            memory_id,
            json.dumps(memory_entry)
        )
        
        # Add to timeline index
        self.redis_client.zadd(
            f"{self.namespace}:agent:{self.agent_id}:timeline",
            {memory_id: memory_entry["step_number"]}
        )
        
        # Set expiration for automatic cleanup
        self.redis_client.expire(
            f"{self.namespace}:agent:{self.agent_id}",
            self.ttl
        )
        self.redis_client.expire(
            f"{self.namespace}:agent:{self.agent_id}:timeline",
            self.ttl
        )
```

## **6. Performance Considerations**

### **6.1 Redis Configuration Tuning**

For optimal performance in the agent memory system:

1. **Memory Allocation**: Configure Redis with appropriate `maxmemory` setting
2. **Eviction Policy**: Use `volatile-ttl` for automatic IM expiration
3. **Persistence**: Configure RDB snapshots for data durability
4. **Connection Pooling**: Implement connection pools for multi-threaded access

### **6.2 Benchmarks**

**[ ! ]** ***need to replace with actual results*** **[ ! ]**

Performance testing shows significant improvements with Redis integration:

| Operation | Direct SQLite | Redis Buffered | Improvement |
|-----------|---------------|----------------|-------------|
| Write 1000 states | 2.3s | 0.18s | 12.8x faster |
| Batch flush 10,000 records | N/A | 0.85s | N/A |
| Read recent state | 15ms | 2ms | 7.5x faster |

## **7. Integration with Memory System**

The Redis implementation serves as the foundation for both STM and IM tiers in the hierarchical memory system described in [Core Concepts](core_concepts.md). The integration points include:

1. **Memory Agent**: Uses Redis stores for fast memory operations
2. **State Storage**: Leverages Redis for caching recent states
3. **API Layer**: Provides consistent access across memory tiers

---

**See Also:**
- [Core Concepts](core_concepts.md) - Fundamental architecture and data structures
- [Memory Agent](memory_agent.md) - Memory agent implementation
- [Agent State Storage](agent_state_storage.md) - State storage implementation
- [API Specification](agent_memory_api.md) - API documentation
- [Redis Index Schema](redis_index_schema.md) - Redis key structure and index design
- [Agent State Redis Schema](agent_state_redis_schema.md) - Specific Redis schema for agent state storage
- [Glossary](glossary.md) - Terminology reference
- [Agent Action Redis Schema](agent_action_redis_schema.md) - Redis schema for agent action storage
- [Agent Interaction Redis Schema](agent_interaction_redis_schema.md) - Redis schema for agent interaction storage
- [Index Optimization Strategies](index_optimization_strategies.md) - Strategies for optimizing Redis indices

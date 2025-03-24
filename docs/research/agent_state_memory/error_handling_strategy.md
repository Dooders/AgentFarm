# **AgentMemory: Error Handling and Recovery Strategy**

## **1. Introduction**

This document outlines a comprehensive approach to error handling and recovery in the AgentMemory system. The hierarchical memory architecture relies on multiple storage tiers (Redis-based STM/IM and SQLite-based LTM), introducing potential points of failure that must be addressed to ensure system resilience. This strategy provides a framework for graceful degradation, automatic recovery, and proper error handling throughout the memory system.

## **2. Error Categorization**

### **2.1 Tier-Specific Failures**

| Tier | Failure Type | Potential Causes | Impact |
|------|--------------|------------------|--------|
| **STM** | Redis unavailable | Service down, network partition, authentication failure | Loss of recent state storage and fast retrieval |
| **IM** | Redis data corruption | Improper shutdown, memory limits, TTL misconfigurations | Inconsistent intermediate memory, retrieval issues |
| **LTM** | SQLite locked/corrupted | Concurrent access, disk errors, incomplete writes | Loss of historical data, persistence failures |
| **Embedding** | Neural network errors | Model loading failure, incompatible shapes, GPU resources | Inability to generate or compare embeddings |

### **2.2 Cross-Tier Failures**

| Failure Type | Description | Impact |
|--------------|-------------|--------|
| **Memory Transition Interruption** | Process termination during transfer between tiers | Inconsistent state between memory tiers |
| **Embedding Consistency** | Different embedding versions or incompatible dimensions | Semantic search failures, false similarities |
| **Transaction Rollbacks** | Failure of atomic operations spanning multiple tiers | Partial state updates, data inconsistency |

### **2.3 System-Level Failures**

| Failure Type | Description | Impact |
|--------------|-------------|--------|
| **Resource Exhaustion** | Memory limits, connection pools, file handles | Degraded performance, operation failures |
| **Dependency Failures** | Libraries, runtime dependencies | Unpredictable behavior, component failures |
| **Configuration Errors** | Mismatched settings, invalid parameters | System-wide functional issues |

## **3. Graceful Degradation Strategies**

### **3.1 Tier Fallback Mechanisms**

```python
def get_agent_state(agent_id, memory_id):
    """Retrieve agent state with tier fallback."""
    try:
        # Try STM first (fastest)
        return stm_store.get_state(agent_id, memory_id)
    except RedisUnavailableError:
        try:
            # Fall back to IM if STM is unavailable
            logger.warning("STM unavailable, falling back to IM")
            return im_store.get_state(agent_id, memory_id)
        except RedisUnavailableError:
            # Last resort: fall back to LTM
            logger.warning("IM unavailable, falling back to LTM")
            return ltm_store.get_state(agent_id, memory_id)
```

### **3.2 Functional Degradation Modes**

| Mode | Description | When Activated |
|------|-------------|----------------|
| **Read-Only Mode** | Disable write operations, serve from existing data | When write operations to Redis or SQLite fail |
| **Local Cache Mode** | Maintain temporary in-memory cache | When all persistent storage is unavailable |
| **Embedding-Free Mode** | Fall back to exact matching instead of semantic search | When embedding generation fails |
| **Single-Tier Mode** | Operate using only available tier | When multiple tiers experience failures |

### **3.3 Priority-Based Operation Handling**

```python
def store_agent_state(agent_id, state_data, priority=Priority.NORMAL):
    """Store agent state with priority-based handling."""
    try:
        stm_store.store(agent_id, state_data)
        return True
    except RedisUnavailableError:
        if priority == Priority.CRITICAL:
            # For critical states, attempt direct persist to LTM
            logger.warning("STM unavailable for critical state, writing directly to LTM")
            return ltm_store.store(agent_id, state_data)
        elif priority == Priority.HIGH:
            # For high priority, queue for retry
            recovery_queue.enqueue(StoreOperation(agent_id, state_data))
            return False
        else:
            # For normal priority, log and continue
            logger.error("Failed to store state for agent %s", agent_id)
            return False
```

## **4. Error Detection and Monitoring**

### **4.1 Proactive Health Checks**

```python
class MemorySystemHealthMonitor:
    """Monitor health of all memory tiers."""
    
    def __init__(self, check_interval=60):
        self.check_interval = check_interval
        self.last_status = {}
        
    def start_monitoring(self):
        """Start periodic health checks."""
        threading.Timer(self.check_interval, self._check_health).start()
        
    def _check_health(self):
        """Check health of all components."""
        self.last_status = {
            "stm": self._check_redis_stm(),
            "im": self._check_redis_im(),
            "ltm": self._check_sqlite_ltm(),
            "embedding_engine": self._check_embedding_engine()
        }
        
        # Schedule next check
        threading.Timer(self.check_interval, self._check_health).start()
        
    def _check_redis_stm(self):
        """Check STM Redis health."""
        try:
            redis_client.ping()
            return {"status": "healthy", "latency_ms": measure_redis_latency()}
        except Exception as e:
            logger.error("STM health check failed: %s", str(e))
            return {"status": "unhealthy", "error": str(e)}
```

### **4.2 Error Thresholds and Circuit Breakers**

```python
class CircuitBreaker:
    """Prevent repeated attempts to access failed resources."""
    
    def __init__(self, failure_threshold=3, reset_timeout=300):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
        
    def execute(self, operation):
        """Execute operation with circuit breaker pattern."""
        if self.state == CircuitState.OPEN:
            # Check if reset timeout has elapsed
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError("Circuit breaker is open")
                
        try:
            result = operation()
            
            # Success - reset failure count
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
            self.failure_count = 0
            return result
            
        except Exception as e:
            # Failure - increment count and update state
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                
            raise e
```

### **4.3 Metrics and Alerting**

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| **Redis Connection Failures** | Count of failed Redis connection attempts | >3 in 5 minutes |
| **SQLite Query Errors** | Count of SQLite query failures | >5 in 15 minutes |
| **Memory Transition Failures** | Failed attempts to transition between tiers | >2 in 1 hour |
| **Embedding Generation Errors** | Neural network failures when generating embeddings | >10 in 1 hour |
| **Recovery Queue Length** | Number of operations waiting for retry | >100 items or >10 minutes old |

## **5. Recovery Mechanisms**

### **5.1 Automated Retry Strategies**

```python
class RetryPolicy:
    """Define retry behavior for failed operations."""
    
    def __init__(self, max_retries=3, base_delay=1.0, backoff_factor=2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        
    def get_retry_delay(self, attempt):
        """Calculate delay before next retry."""
        return self.base_delay * (self.backoff_factor ** attempt)
        
    def should_retry(self, attempt, exception):
        """Determine if another retry should be attempted."""
        if attempt >= self.max_retries:
            return False
            
        # Only retry on transient errors
        return isinstance(exception, (
            RedisUnavailableError,
            RedisTimeoutError,
            SQLiteTemporaryError,
            ConnectionError
        ))
```

### **5.2 Recovery Queue System**

```python
class RecoveryQueue:
    """Queue for retrying failed operations."""
    
    def __init__(self, worker_count=2, retry_policy=None):
        self.queue = queue.PriorityQueue()
        self.retry_policy = retry_policy or RetryPolicy()
        self.worker_count = worker_count
        self.workers = []
        
    def start(self):
        """Start recovery queue workers."""
        for i in range(self.worker_count):
            worker = threading.Thread(target=self._process_queue)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
    def enqueue(self, operation, priority=0):
        """Add operation to recovery queue."""
        self.queue.put((priority, operation))
        
    def _process_queue(self):
        """Worker process to handle recovery operations."""
        while True:
            priority, operation = self.queue.get()
            
            try:
                operation.execute()
                logger.info("Successfully recovered operation: %s", operation)
            except Exception as e:
                attempt = operation.attempt + 1
                if self.retry_policy.should_retry(attempt, e):
                    # Calculate delay and requeue
                    delay = self.retry_policy.get_retry_delay(attempt)
                    operation.attempt = attempt
                    
                    logger.warning("Retry %d for operation %s after %.2f seconds", 
                                  attempt, operation, delay)
                    
                    # Schedule retry after delay
                    threading.Timer(
                        delay, 
                        lambda: self.enqueue(operation, priority)
                    ).start()
                else:
                    logger.error("Operation %s failed permanently after %d attempts: %s", 
                                operation, attempt, str(e))
                    # Log to permanent failure store
                    self._record_permanent_failure(operation, e)
                    
            finally:
                self.queue.task_done()
```

### **5.3 Inconsistency Detection and Resolution**

```python
def verify_memory_consistency(agent_id, step_range=None):
    """Check for consistency across memory tiers."""
    # Get memory IDs from each tier
    stm_ids = set(stm_store.get_memory_ids(agent_id, step_range))
    im_ids = set(im_store.get_memory_ids(agent_id, step_range))
    ltm_ids = set(ltm_store.get_memory_ids(agent_id, step_range))
    
    # Check for missing entries
    missing_in_im = stm_ids - im_ids
    missing_in_ltm = im_ids - ltm_ids
    
    # Detected inconsistencies
    if missing_in_im or missing_in_ltm:
        logger.warning("Memory inconsistency detected for agent %s", agent_id)
        
        # Repair strategy
        for memory_id in missing_in_im:
            # Re-transition from STM to IM
            state = stm_store.get_state(agent_id, memory_id)
            if state:
                im_store.store(agent_id, state)
                logger.info("Repaired IM entry for memory %s", memory_id)
                
        for memory_id in missing_in_ltm:
            # Re-transition from IM to LTM
            state = im_store.get_state(agent_id, memory_id)
            if state:
                ltm_store.store(agent_id, state)
                logger.info("Repaired LTM entry for memory %s", memory_id)
                
        return len(missing_in_im) + len(missing_in_ltm)
    
    return 0  # No inconsistencies
```

### **5.4 Transaction and Checkpoint Management**

```python
class MemoryTransaction:
    """Manage atomic operations across memory tiers."""
    
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.operations = []
        self.checkpoints = {}
        
    def add_operation(self, operation):
        """Add operation to transaction."""
        self.operations.append(operation)
        
    def set_checkpoint(self, checkpoint_name):
        """Mark checkpoint in transaction."""
        self.checkpoints[checkpoint_name] = len(self.operations)
        
    def execute(self):
        """Execute all operations atomically."""
        results = []
        completed = 0
        
        try:
            # Execute all operations
            for op in self.operations:
                result = op.execute()
                results.append(result)
                completed += 1
                
            return results
        except Exception as e:
            # Find nearest checkpoint for rollback
            rollback_point = 0
            for checkpoint, index in sorted(
                self.checkpoints.items(), 
                key=lambda x: x[1], 
                reverse=True
            ):
                if index <= completed:
                    rollback_point = index
                    logger.info("Rolling back to checkpoint: %s", checkpoint)
                    break
                    
            # Execute rollback operations
            for i in range(completed - 1, rollback_point - 1, -1):
                try:
                    self.operations[i].rollback()
                except Exception as rollback_error:
                    logger.error(
                        "Error during rollback of operation %d: %s", 
                        i, str(rollback_error)
                    )
                    
            raise TransactionError(f"Transaction failed at operation {completed}: {str(e)}")
```

## **6. Interrupted Memory Transition Recovery**

### **6.1 Memory Transition Tracking**

```python
class MemoryTransitionTracker:
    """Track memory transitions between tiers."""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.transition_key = "memory:transitions:active"
        
    def start_transition(self, from_tier, to_tier, memory_id, agent_id):
        """Mark beginning of a memory transition."""
        transition_data = {
            "memory_id": memory_id,
            "agent_id": agent_id,
            "from_tier": from_tier,
            "to_tier": to_tier,
            "start_time": time.time(),
            "status": "in_progress"
        }
        
        # Store in Redis with transition ID
        transition_id = f"{agent_id}:{memory_id}:{from_tier}:{to_tier}"
        self.redis.hset(self.transition_key, transition_id, json.dumps(transition_data))
        
        return transition_id
        
    def complete_transition(self, transition_id):
        """Mark transition as complete."""
        self.redis.hdel(self.transition_key, transition_id)
        
    def fail_transition(self, transition_id, error):
        """Mark transition as failed."""
        transition_data = json.loads(self.redis.hget(self.transition_key, transition_id))
        transition_data["status"] = "failed"
        transition_data["error"] = str(error)
        transition_data["failure_time"] = time.time()
        
        self.redis.hset(self.transition_key, transition_id, json.dumps(transition_data))
        
    def get_incomplete_transitions(self):
        """Retrieve all incomplete transitions."""
        all_transitions = self.redis.hgetall(self.transition_key)
        
        return {
            tid: json.loads(data) 
            for tid, data in all_transitions.items()
        }
```

### **6.2 Recovery Process for Interrupted Transitions**

```python
def recover_interrupted_transitions():
    """Find and recover incomplete memory transitions."""
    tracker = MemoryTransitionTracker(redis_client)
    incomplete = tracker.get_incomplete_transitions()
    
    if not incomplete:
        logger.info("No incomplete transitions found")
        return 0
        
    logger.info("Found %d incomplete transitions to recover", len(incomplete))
    recovered = 0
    
    for transition_id, transition_data in incomplete.items():
        try:
            agent_id = transition_data["agent_id"]
            memory_id = transition_data["memory_id"]
            from_tier = transition_data["from_tier"]
            to_tier = transition_data["to_tier"]
            
            # Check if destination already has the data
            if memory_exists_in_tier(to_tier, agent_id, memory_id):
                logger.info("Transition already complete for %s, cleaning up", transition_id)
                tracker.complete_transition(transition_id)
                recovered += 1
                continue
                
            # Check if source still has the data
            if not memory_exists_in_tier(from_tier, agent_id, memory_id):
                logger.error("Source data missing for transition %s, cannot recover", transition_id)
                tracker.fail_transition(transition_id, "Source data missing")
                continue
                
            # Re-execute the transition
            logger.info("Recovering transition %s: %s -> %s", 
                       memory_id, from_tier, to_tier)
                       
            memory_data = get_memory_from_tier(from_tier, agent_id, memory_id)
            store_memory_to_tier(to_tier, agent_id, memory_id, memory_data)
            
            tracker.complete_transition(transition_id)
            recovered += 1
            
        except Exception as e:
            logger.error("Error recovering transition %s: %s", transition_id, str(e))
            tracker.fail_transition(transition_id, str(e))
            
    logger.info("Recovered %d of %d incomplete transitions", recovered, len(incomplete))
    return recovered
```

### **6.3 Memory System Initialization with Recovery**

```python
def initialize_memory_system():
    """Initialize memory system with recovery checks."""
    # Start basic connectivity checks
    redis_stm_available = check_redis_stm_availability()
    redis_im_available = check_redis_im_availability()
    sqlite_available = check_sqlite_availability()
    
    # Record system state
    system_state = {
        "stm_available": redis_stm_available,
        "im_available": redis_im_available,
        "ltm_available": sqlite_available,
        "recovery_needed": False,
        "degraded_mode": False
    }
    
    if not redis_stm_available or not redis_im_available or not sqlite_available:
        system_state["degraded_mode"] = True
        logger.warning("Initializing memory system in degraded mode")
        
    if redis_stm_available and redis_im_available:
        # Check for and recover interrupted transitions
        incomplete_count = recover_interrupted_transitions()
        system_state["recovery_needed"] = incomplete_count > 0
        
        if incomplete_count > 0:
            logger.info("Recovered %d interrupted transitions during initialization", 
                       incomplete_count)
            
    # Start health monitoring
    health_monitor = MemorySystemHealthMonitor()
    health_monitor.start_monitoring()
    
    # Initialize recovery queue
    recovery_queue = RecoveryQueue(worker_count=2)
    recovery_queue.start()
    
    logger.info("Memory system initialized: %s", 
               "Fully operational" if not system_state["degraded_mode"] else "Degraded mode")
               
    return system_state
```

## **7. Error Handling Best Practices**

### **7.1 Exception Hierarchy**

```python
# Base exception for all memory-related errors
class MemoryError(Exception):
    """Base class for all memory system exceptions."""
    pass

# Tier-specific exceptions
class STMError(MemoryError):
    """Error in Short-Term Memory operations."""
    pass
    
class IMError(MemoryError):
    """Error in Intermediate Memory operations."""
    pass
    
class LTMError(MemoryError):
    """Error in Long-Term Memory operations."""
    pass

# Storage-specific exceptions  
class RedisUnavailableError(STMError, IMError):
    """Redis connection unavailable."""
    pass
    
class RedisTimeoutError(STMError, IMError):
    """Redis operation timed out."""
    pass
    
class SQLiteTemporaryError(LTMError):
    """Temporary SQLite error (lock, timeout)."""
    pass
    
class SQLitePermanentError(LTMError):
    """Permanent SQLite error (corruption)."""
    pass

# Operational exceptions
class MemoryTransitionError(MemoryError):
    """Error during memory transition between tiers."""
    pass
    
class EmbeddingGenerationError(MemoryError):
    """Error generating memory embeddings."""
    pass
    
class TransactionError(MemoryError):
    """Error during a multi-operation transaction."""
    pass
    
class CircuitOpenError(MemoryError):
    """Operation blocked by open circuit breaker."""
    pass
```

### **7.2 Logging Standards**

| Level | Usage | Example |
|-------|-------|---------|
| **ERROR** | Unrecoverable errors, system degradation | `"Redis connection failed: %s"` |
| **WARNING** | Recoverable errors, degraded performance | `"STM storage failed, falling back to LTM: %s"` |
| **INFO** | Normal operation events, recovery success | `"Memory transition complete: %s → %s"` |
| **DEBUG** | Detailed operation information | `"Generated embedding for agent %s state %s: %s"` |

All error logs should include:
1. Clear description of what failed
2. Component information (tier, agent ID, etc.)
3. Error details (exception message, error code)
4. Recovery action taken or suggested

### **7.3 Configuration Parameters**

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|--------|
| `retry_max_attempts` | Maximum retry attempts for failed operations | 3 | Higher values improve recovery chances but may delay error reporting |
| `retry_base_delay` | Base delay between retry attempts (seconds) | 1.0 | Higher values reduce system load during recovery |
| `retry_backoff_factor` | Exponential backoff multiplier | 2.0 | Higher values increase delay between successive retries |
| `circuit_breaker_threshold` | Failures before circuit opens | 3 | Lower values prevent repeated failures but may trigger false positives |
| `circuit_breaker_reset` | Time before circuit resets (seconds) | 300 | Higher values provide more protection during extended outages |
| `health_check_interval` | Time between health checks (seconds) | 60 | Lower values detect problems faster but increase system overhead |
| `recovery_workers` | Number of recovery queue workers | 2 | Higher values process recovery faster but consume more resources |

## **8. Testing and Verification**

### **8.1 Error Injection Testing**

```python
def test_redis_unavailable():
    """Test system behavior when Redis is unavailable."""
    # Setup: mock Redis to raise connection errors
    with mock.patch('redis.Redis.ping', side_effect=ConnectionError):
        with mock.patch('redis.Redis.get', side_effect=ConnectionError):
            # Test storing a state
            result = memory_agent.store_agent_state('test-agent', {'position': [0, 0]}, 42)
            
            # Verify it was queued for recovery
            assert result is False
            assert recovery_queue.size() == 1
            
            # Test retrieving a state (should fall back to LTM)
            state = memory_agent.get_agent_state('test-agent', 'memory-1')
            
            # Verify it attempted to get from LTM
            ltm_store.get_state.assert_called_once_with('test-agent', 'memory-1')
```

### **8.2 Recovery Verification Tests**

```python
def test_interrupted_transition_recovery():
    """Test recovery of interrupted memory transitions."""
    # Create a mock incomplete transition
    tracker = MemoryTransitionTracker(redis_client)
    transition_id = tracker.start_transition(
        'stm', 'im', 'memory-test', 'agent-test'
    )
    
    # Ensure data exists in source tier only
    stm_store.store('agent-test', {'data': 'test'}, 'memory-test')
    
    # Run recovery process
    recovered = recover_interrupted_transitions()
    
    # Verify data was moved to destination tier
    assert recovered == 1
    assert im_store.exists('agent-test', 'memory-test')
    assert tracker.get_incomplete_transitions() == {}
```

### **8.3 Performance Degradation Tests**

```python
def test_degraded_performance_under_load():
    """Test system performance in degraded mode."""
    # Setup: configure system to operate without Redis STM
    config = MemoryConfig(use_stm=False)
    memory_agent = MemoryAgent('test-agent', config)
    
    # Generate test load
    start_time = time.time()
    for i in range(100):
        memory_agent.store_agent_state('test-agent', generate_test_state(i), i)
    
    # Measure store latency in degraded mode
    store_latency = (time.time() - start_time) / 100
    
    # Test retrieval performance
    start_time = time.time()
    for i in range(100):
        memory_agent.get_agent_state('test-agent', f'test-agent-{i}')
    
    # Measure retrieval latency in degraded mode
    retrieval_latency = (time.time() - start_time) / 100
    
    # Verify performance is within acceptable bounds for degraded mode
    assert store_latency < DEGRADED_STORE_THRESHOLD
    assert retrieval_latency < DEGRADED_RETRIEVAL_THRESHOLD
```

## **9. Summary**

This error handling and recovery strategy provides a comprehensive framework for ensuring the AgentMemory system remains operational and data remains consistent even when facing various failure scenarios. Key aspects include:

1. **Graceful Degradation**: System continues functioning with reduced capability when components fail
2. **Automatic Recovery**: Background processes detect and repair inconsistencies
3. **Prioritized Handling**: Critical operations receive special treatment during failures
4. **Comprehensive Monitoring**: Early detection of potential issues
5. **Robust Testing**: Verification of recovery mechanisms through fault injection

Implementing this strategy will significantly enhance the reliability of the AgentMemory system, ensuring agents maintain access to their memory even during partial system failures. 
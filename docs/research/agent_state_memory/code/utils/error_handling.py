"""Error handling utilities for agent memory system.

This module implements error recovery mechanisms including circuit breakers,
retry policies, and recovery queues for the agent memory system.
"""

import enum
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

# Type variable for generic function return
T = TypeVar("T")


class CircuitState(enum.Enum):
    """Circuit breaker states."""
    CLOSED = "closed"     # Normal operation, requests pass through
    OPEN = "open"         # Circuit is open, requests are blocked
    HALF_OPEN = "half_open"  # Testing if service has recovered


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


class Priority(enum.IntEnum):
    """Priority levels for memory operations."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class CircuitBreaker:
    """Prevent repeated attempts to access failed resources.
    
    The circuit breaker pattern prevents cascading failures by stopping
    attempts to execute operations that are likely to fail.
    
    Attributes:
        name: Identifier for this circuit breaker
        failure_threshold: Number of failures before circuit opens
        reset_timeout: Seconds to wait before trying again (half-open)
        failure_count: Current count of consecutive failures
        last_failure_time: Timestamp of last failure
        state: Current state of the circuit
    """
    
    def __init__(self, name: str, failure_threshold: int = 3, reset_timeout: int = 300):
        """Initialize the circuit breaker.
        
        Args:
            name: Identifier for this circuit breaker
            failure_threshold: Number of failures before circuit opens
            reset_timeout: Seconds to wait before trying again
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitState.CLOSED
        
    def execute(self, operation: Callable[[], T]) -> T:
        """Execute operation with circuit breaker pattern.
        
        Args:
            operation: Function to execute
            
        Returns:
            Result of the operation
            
        Raises:
            CircuitOpenError: If the circuit is open
            Any exception raised by the operation
        """
        if self.state == CircuitState.OPEN:
            # Check if reset timeout has elapsed
            if time.time() - self.last_failure_time > self.reset_timeout:
                logger.info("Circuit %s changed from OPEN to HALF_OPEN", self.name)
                self.state = CircuitState.HALF_OPEN
            else:
                logger.warning("Circuit %s is OPEN, blocking operation", self.name)
                raise CircuitOpenError(f"Circuit breaker {self.name} is open")
                
        try:
            result = operation()
            
            # Success - reset failure count
            if self.state == CircuitState.HALF_OPEN:
                logger.info("Circuit %s changed from HALF_OPEN to CLOSED", self.name)
                self.state = CircuitState.CLOSED
            self.failure_count = 0
            return result
            
        except Exception as e:
            # Failure - increment count and update state
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                if self.state != CircuitState.OPEN:
                    logger.warning("Circuit %s changed to OPEN after %d failures", 
                                 self.name, self.failure_count)
                    self.state = CircuitState.OPEN
                
            raise e


class RetryPolicy:
    """Define retry behavior for failed operations.
    
    This class determines when and how operations should be retried
    after failure.
    
    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        backoff_factor: Multiplier for exponential backoff
    """
    
    def __init__(
        self, 
        max_retries: int = 3, 
        base_delay: float = 1.0, 
        backoff_factor: float = 2.0
    ):
        """Initialize the retry policy.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            backoff_factor: Multiplier for exponential backoff
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        
    def get_retry_delay(self, attempt: int) -> float:
        """Calculate delay before next retry using exponential backoff.
        
        Args:
            attempt: The current attempt number (0-based)
            
        Returns:
            Delay in seconds before next retry
        """
        return self.base_delay * (self.backoff_factor ** attempt)
        
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """Determine if another retry should be attempted.
        
        Args:
            attempt: The current attempt number (0-based)
            exception: The exception that caused the failure
            
        Returns:
            True if the operation should be retried, False otherwise
        """
        if attempt >= self.max_retries:
            return False
            
        # Only retry on transient errors
        return isinstance(exception, (
            RedisUnavailableError,
            RedisTimeoutError,
            SQLiteTemporaryError,
            ConnectionError
        ))


class RetryableOperation:
    """An operation that can be retried.
    
    Attributes:
        operation_id: Unique identifier for this operation
        attempt: Current attempt number (0-based)
        created_at: Timestamp when operation was created
        last_attempt_at: Timestamp of last attempt
    """
    
    def __init__(self, operation_id: str):
        """Initialize the retryable operation.
        
        Args:
            operation_id: Unique identifier for this operation
        """
        self.operation_id = operation_id
        self.attempt = 0
        self.created_at = time.time()
        self.last_attempt_at = 0
        
    def execute(self) -> bool:
        """Execute the operation.
        
        This method should be overridden by subclasses.
        
        Returns:
            True if the operation succeeded, False otherwise
        """
        raise NotImplementedError("Subclasses must implement execute()")
        
    def __str__(self) -> str:
        """Return string representation of operation."""
        return f"Operation(id={self.operation_id}, attempt={self.attempt})"


class StoreOperation(RetryableOperation):
    """Operation to store data in memory.
    
    Attributes:
        agent_id: ID of the agent
        state_data: Data to store
        store_function: Function that performs the actual storage
    """
    
    def __init__(
        self, 
        operation_id: str,
        agent_id: str, 
        state_data: Dict[str, Any],
        store_function: Callable[[str, Dict[str, Any]], bool]
    ):
        """Initialize the store operation.
        
        Args:
            operation_id: Unique identifier for this operation
            agent_id: ID of the agent
            state_data: Data to store 
            store_function: Function that performs the actual storage
        """
        super().__init__(operation_id)
        self.agent_id = agent_id
        self.state_data = state_data
        self.store_function = store_function
        
    def execute(self) -> bool:
        """Execute the store operation.
        
        Returns:
            True if the operation succeeded, False otherwise
        """
        self.attempt += 1
        self.last_attempt_at = time.time()
        return self.store_function(self.agent_id, self.state_data)


class RecoveryQueue:
    """Queue for retrying failed operations.
    
    This class manages a priority queue of operations that need to be
    retried after failure.
    
    Attributes:
        queue: Priority queue of operations
        retry_policy: Policy for retrying operations
        worker_count: Number of worker threads
        workers: List of worker threads
        running: Whether the queue is currently processing
    """
    
    def __init__(self, worker_count: int = 2, retry_policy: Optional[RetryPolicy] = None):
        """Initialize the recovery queue.
        
        Args:
            worker_count: Number of worker threads
            retry_policy: Policy for retrying operations
        """
        self.queue = queue.PriorityQueue()
        self.retry_policy = retry_policy or RetryPolicy()
        self.worker_count = worker_count
        self.workers: List[threading.Thread] = []
        self.running = False
        
    def start(self) -> None:
        """Start recovery queue workers."""
        if self.running:
            return
            
        self.running = True
        for i in range(self.worker_count):
            worker = threading.Thread(
                target=self._process_queue,
                name=f"recovery-worker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
        
        logger.info("Started %d recovery queue workers", self.worker_count)
            
    def stop(self) -> None:
        """Stop recovery queue workers."""
        self.running = False
        # Workers will stop when running becomes False
        logger.info("Stopping recovery queue workers")
            
    def enqueue(self, operation: RetryableOperation, priority: int = 0) -> None:
        """Add operation to recovery queue.
        
        Args:
            operation: Operation to enqueue
            priority: Priority level (lower number = higher priority)
        """
        # Ensure queue is running
        if not self.running:
            self.start()
            
        self.queue.put((priority, operation))
        logger.debug("Enqueued operation %s with priority %d", operation, priority)
        
    def _process_queue(self) -> None:
        """Worker process to handle recovery operations."""
        while self.running:
            try:
                # Get with timeout to allow checking running status
                priority, operation = self.queue.get(timeout=1.0)
                
                try:
                    success = operation.execute()
                    if success:
                        logger.info("Successfully recovered operation: %s", operation)
                        self.queue.task_done()
                    else:
                        # Operation reported failure
                        self._handle_retry(priority, operation)
                except Exception as e:
                    logger.exception("Error executing recovery operation: %s", operation)
                    self._handle_retry(priority, operation, e)
            except queue.Empty:
                # Queue timeout, loop and check running status
                continue
                
    def _handle_retry(
        self, 
        priority: int, 
        operation: RetryableOperation, 
        exception: Optional[Exception] = None
    ) -> None:
        """Handle retry logic for a failed operation.
        
        Args:
            priority: Priority level
            operation: Operation that failed
            exception: Exception that caused the failure, if any
        """
        if self.retry_policy.should_retry(operation.attempt, exception or Exception()):
            # Calculate delay and requeue
            delay = self.retry_policy.get_retry_delay(operation.attempt)
            
            logger.warning(
                "Retry %d for operation %s after %.2f seconds", 
                operation.attempt + 1, operation, delay
            )
            
            # Sleep for delay seconds
            time.sleep(delay)
            
            # Requeue the operation with the same priority
            self.queue.put((priority, operation))
        else:
            logger.error(
                "Operation %s failed after %d attempts, giving up", 
                operation, operation.attempt
            )
            self.queue.task_done()


class MemorySystemHealthMonitor:
    """Monitor health of all memory tiers.
    
    This class periodically checks the health of all memory system
    components and updates their status.
    
    Attributes:
        check_interval: Seconds between health checks
        last_status: Dictionary of component health statuses
        timer: Timer for scheduling health checks
        stm_client: Redis client for STM
        im_client: Redis client for IM
        ltm_store: SQLite store for LTM
        embedding_engine: Neural embedding engine
    """
    
    def __init__(
        self, 
        check_interval: int = 60,
        stm_client = None,
        im_client = None,
        ltm_store = None,
        embedding_engine = None
    ):
        """Initialize the health monitor.
        
        Args:
            check_interval: Seconds between health checks
            stm_client: Redis client for STM
            im_client: Redis client for IM
            ltm_store: SQLite store for LTM
            embedding_engine: Neural embedding engine
        """
        self.check_interval = check_interval
        self.last_status = {}
        self.timer = None
        self.stm_client = stm_client
        self.im_client = im_client
        self.ltm_store = ltm_store
        self.embedding_engine = embedding_engine
        
    def start_monitoring(self) -> None:
        """Start periodic health checks."""
        if self.timer:
            # Timer already started
            return
            
        self._check_health()
        
    def stop_monitoring(self) -> None:
        """Stop periodic health checks."""
        if self.timer:
            self.timer.cancel()
            self.timer = None
        
    def _check_health(self) -> None:
        """Check health of all components."""
        self.last_status = {
            "stm": self._check_redis_stm(),
            "im": self._check_redis_im(),
            "ltm": self._check_sqlite_ltm(),
            "embedding_engine": self._check_embedding_engine()
        }
        
        # Schedule next check
        self.timer = threading.Timer(self.check_interval, self._check_health)
        self.timer.daemon = True
        self.timer.start()
        
    def _check_redis_stm(self) -> Dict[str, Any]:
        """Check STM Redis health."""
        if not self.stm_client:
            return {"status": "unknown", "error": "No STM client configured"}
            
        try:
            start_time = time.time()
            self.stm_client.ping()
            latency = (time.time() - start_time) * 1000  # ms
            return {"status": "healthy", "latency_ms": latency}
        except Exception as e:
            logger.error("STM health check failed: %s", str(e))
            return {"status": "unhealthy", "error": str(e)}
            
    def _check_redis_im(self) -> Dict[str, Any]:
        """Check IM Redis health."""
        if not self.im_client:
            return {"status": "unknown", "error": "No IM client configured"}
            
        try:
            start_time = time.time()
            self.im_client.ping()
            latency = (time.time() - start_time) * 1000  # ms
            return {"status": "healthy", "latency_ms": latency}
        except Exception as e:
            logger.error("IM health check failed: %s", str(e))
            return {"status": "unhealthy", "error": str(e)}
            
    def _check_sqlite_ltm(self) -> Dict[str, Any]:
        """Check LTM SQLite health."""
        if not self.ltm_store:
            return {"status": "unknown", "error": "No LTM store configured"}
            
        try:
            # Attempt a simple query to check health
            start_time = time.time()
            # Placeholder for actual health check based on specific implementation
            # self.ltm_store.check_health()
            latency = (time.time() - start_time) * 1000  # ms
            return {"status": "healthy", "latency_ms": latency}
        except Exception as e:
            logger.error("LTM health check failed: %s", str(e))
            return {"status": "unhealthy", "error": str(e)}
            
    def _check_embedding_engine(self) -> Dict[str, Any]:
        """Check embedding engine health."""
        if not self.embedding_engine:
            return {"status": "unknown", "error": "No embedding engine configured"}
            
        try:
            # Placeholder for actual embedding engine health check
            # self.embedding_engine.check_health()
            return {"status": "healthy"}
        except Exception as e:
            logger.error("Embedding engine health check failed: %s", str(e))
            return {"status": "unhealthy", "error": str(e)}
            
    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the current health status of all components.
        
        Returns:
            Dictionary mapping component names to their health status
        """
        return self.last_status 
"""Redis client with error handling for agent memory system.

This module provides a Redis client wrapper with circuit breaker and
retry functionality for resilient Redis operations.
"""

import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

import redis

from ..utils.error_handling import (
    CircuitBreaker, 
    RecoveryQueue, 
    RetryPolicy,
    RedisUnavailableError,
    RedisTimeoutError,
    Priority,
    StoreOperation
)

logger = logging.getLogger(__name__)

# Type variable for generic function return
T = TypeVar("T")


class ResilientRedisClient:
    """Redis client with circuit breaker and retry functionality.
    
    This class wraps Redis operations with circuit breaker pattern
    to prevent cascading failures and retry mechanisms for transient
    errors.
    
    Attributes:
        client_name: Name of this Redis client instance
        connection_params: Redis connection parameters
        client: Redis client instance
        circuit_breaker: Circuit breaker for Redis operations
        recovery_queue: Queue for retrying failed operations
        retry_policy: Policy for automatic retries
    """
    
    def __init__(
        self,
        client_name: str,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        socket_timeout: float = 2.0,
        socket_connect_timeout: float = 2.0,
        retry_policy: Optional[RetryPolicy] = None,
        circuit_threshold: int = 3,
        circuit_reset_timeout: int = 300
    ):
        """Initialize the Redis client.
        
        Args:
            client_name: Name of this Redis client instance
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            socket_timeout: Timeout for Redis operations
            socket_connect_timeout: Timeout for connection attempts
            retry_policy: Policy for automatic retries
            circuit_threshold: Failures before circuit breaker opens
            circuit_reset_timeout: Seconds before circuit breaker resets
        """
        self.client_name = client_name
        self.connection_params = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
            "socket_timeout": socket_timeout,
            "socket_connect_timeout": socket_connect_timeout,
            "decode_responses": True
        }
        
        # Create Redis client
        self.client = self._create_redis_client()
        
        # Create circuit breaker
        self.circuit_breaker = CircuitBreaker(
            name=f"redis-{client_name}",
            failure_threshold=circuit_threshold,
            reset_timeout=circuit_reset_timeout
        )
        
        # Create retry policy
        self.retry_policy = retry_policy or RetryPolicy()
        
        # Create recovery queue
        self.recovery_queue = RecoveryQueue(retry_policy=self.retry_policy)
        
        logger.info(
            "Initialized Redis client %s (host=%s, port=%d, db=%d)",
            client_name, host, port, db
        )
        
    def _create_redis_client(self) -> redis.Redis:
        """Create a new Redis client instance.
        
        Returns:
            Redis client instance
        """
        try:
            client = redis.Redis(**self.connection_params)
            return client
        except Exception as e:
            logger.error("Failed to create Redis client: %s", str(e))
            raise RedisUnavailableError(f"Failed to create Redis client: {str(e)}")
    
    def _execute_with_circuit_breaker(
        self, 
        operation_name: str, 
        operation: Callable[[], T]
    ) -> T:
        """Execute operation with circuit breaker pattern.
        
        Args:
            operation_name: Name of the operation
            operation: Function to execute
            
        Returns:
            Result of the operation
            
        Raises:
            RedisUnavailableError: If Redis is unavailable
            RedisTimeoutError: If operation times out
        """
        try:
            return self.circuit_breaker.execute(operation)
        except Exception as e:
            if isinstance(e, redis.exceptions.ConnectionError):
                logger.error(
                    "Redis connection error in %s: %s", 
                    operation_name, str(e)
                )
                raise RedisUnavailableError(f"Redis unavailable: {str(e)}")
            elif isinstance(e, redis.exceptions.TimeoutError):
                logger.error(
                    "Redis timeout in %s: %s", 
                    operation_name, str(e)
                )
                raise RedisTimeoutError(f"Redis operation timed out: {str(e)}")
            else:
                logger.error(
                    "Redis error in %s: %s", 
                    operation_name, str(e)
                )
                raise e
    
    def ping(self) -> bool:
        """Ping Redis server to check connection.
        
        Returns:
            True if successful, raises exception otherwise
        """
        return self._execute_with_circuit_breaker(
            "ping",
            lambda: self.client.ping()
        )
    
    def get(self, key: str) -> Optional[str]:
        """Get value from Redis.
        
        Args:
            key: Redis key
            
        Returns:
            Value or None if not found
        """
        return self._execute_with_circuit_breaker(
            "get",
            lambda: self.client.get(key)
        )
    
    def set(
        self, 
        key: str, 
        value: str, 
        ex: Optional[int] = None, 
        px: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """Set key-value pair in Redis.
        
        Args:
            key: Redis key
            value: Value to set
            ex: Expiry in seconds
            px: Expiry in milliseconds
            nx: Only set if key does not exist
            xx: Only set if key exists
            
        Returns:
            True if successful
        """
        return self._execute_with_circuit_breaker(
            "set",
            lambda: self.client.set(key, value, ex=ex, px=px, nx=nx, xx=xx)
        )
    
    def delete(self, *keys: str) -> int:
        """Delete keys from Redis.
        
        Args:
            keys: Keys to delete
            
        Returns:
            Number of keys deleted
        """
        return self._execute_with_circuit_breaker(
            "delete",
            lambda: self.client.delete(*keys)
        )
    
    def exists(self, *keys: str) -> int:
        """Check if keys exist in Redis.
        
        Args:
            keys: Keys to check
            
        Returns:
            Number of keys that exist
        """
        return self._execute_with_circuit_breaker(
            "exists",
            lambda: self.client.exists(*keys)
        )
    
    def expire(self, key: str, time: int) -> bool:
        """Set expiry on key.
        
        Args:
            key: Redis key
            time: Expiry in seconds
            
        Returns:
            True if successful
        """
        return self._execute_with_circuit_breaker(
            "expire",
            lambda: self.client.expire(key, time)
        )
    
    def hset(self, name: str, key: str, value: str) -> int:
        """Set field in hash.
        
        Args:
            name: Hash name
            key: Field name
            value: Field value
            
        Returns:
            1 if new field, 0 if field existed
        """
        return self._execute_with_circuit_breaker(
            "hset",
            lambda: self.client.hset(name, key, value)
        )
    
    def hget(self, name: str, key: str) -> Optional[str]:
        """Get field from hash.
        
        Args:
            name: Hash name
            key: Field name
            
        Returns:
            Field value or None if not found
        """
        return self._execute_with_circuit_breaker(
            "hget",
            lambda: self.client.hget(name, key)
        )
    
    def hgetall(self, name: str) -> Dict[str, str]:
        """Get all fields from hash.
        
        Args:
            name: Hash name
            
        Returns:
            Dictionary of field names to values
        """
        return self._execute_with_circuit_breaker(
            "hgetall",
            lambda: self.client.hgetall(name)
        )
    
    def hmset(self, name: str, mapping: Dict[str, str]) -> bool:
        """Set multiple fields in hash.
        
        Args:
            name: Hash name
            mapping: Dictionary of field names to values
            
        Returns:
            True if successful
        """
        return self._execute_with_circuit_breaker(
            "hmset",
            lambda: cast(bool, self.client.hset(name, mapping=mapping))
        )
    
    def hdel(self, name: str, *keys: str) -> int:
        """Delete fields from hash.
        
        Args:
            name: Hash name
            keys: Field names to delete
            
        Returns:
            Number of fields deleted
        """
        return self._execute_with_circuit_breaker(
            "hdel",
            lambda: self.client.hdel(name, *keys)
        )
    
    def zadd(
        self, 
        name: str, 
        mapping: Dict[str, float], 
        nx: bool = False,
        xx: bool = False,
        ch: bool = False,
        incr: bool = False
    ) -> int:
        """Add members to sorted set.
        
        Args:
            name: Sorted set name
            mapping: Dictionary of member names to scores
            nx: Only add new elements
            xx: Only update existing elements
            ch: Return number of changed elements
            incr: Increment score instead of replacing
            
        Returns:
            Number of elements added or updated
        """
        return self._execute_with_circuit_breaker(
            "zadd",
            lambda: self.client.zadd(name, mapping, nx=nx, xx=xx, ch=ch, incr=incr)
        )
    
    def zrange(
        self, 
        name: str, 
        start: int, 
        end: int, 
        desc: bool = False,
        withscores: bool = False,
        score_cast_func: Callable[[Any], Any] = float
    ) -> List[Any]:
        """Get range of members from sorted set.
        
        Args:
            name: Sorted set name
            start: Start index
            end: End index
            desc: Order by descending score
            withscores: Include scores in result
            score_cast_func: Function to cast scores
            
        Returns:
            List of members or (member, score) tuples
        """
        return self._execute_with_circuit_breaker(
            "zrange",
            lambda: self.client.zrange(
                name, start, end, desc=desc, 
                withscores=withscores, score_cast_func=score_cast_func
            )
        )
    
    def zrangebyscore(
        self, 
        name: str, 
        min: float, 
        max: float, 
        start: Optional[int] = None,
        num: Optional[int] = None,
        withscores: bool = False,
        score_cast_func: Callable[[Any], Any] = float
    ) -> List[Any]:
        """Get members from sorted set with scores in range.
        
        Args:
            name: Sorted set name
            min: Minimum score
            max: Maximum score
            start: Start offset
            num: Number of elements
            withscores: Include scores in result
            score_cast_func: Function to cast scores
            
        Returns:
            List of members or (member, score) tuples
        """
        return self._execute_with_circuit_breaker(
            "zrangebyscore",
            lambda: self.client.zrangebyscore(
                name, min, max, start=start, num=num,
                withscores=withscores, score_cast_func=score_cast_func
            )
        )
    
    def zrem(self, name: str, *values: str) -> int:
        """Remove members from sorted set.
        
        Args:
            name: Sorted set name
            values: Members to remove
            
        Returns:
            Number of members removed
        """
        return self._execute_with_circuit_breaker(
            "zrem",
            lambda: self.client.zrem(name, *values)
        )
    
    def zcard(self, name: str) -> int:
        """Get number of members in sorted set.
        
        Args:
            name: Sorted set name
            
        Returns:
            Number of members
        """
        return self._execute_with_circuit_breaker(
            "zcard",
            lambda: self.client.zcard(name)
        )
    
    def store_with_retry(
        self, 
        agent_id: str, 
        state_data: Dict[str, Any], 
        store_func: Callable[[str, Dict[str, Any]], bool],
        priority: Priority = Priority.NORMAL
    ) -> bool:
        """Store data with automatic retry on failure.
        
        This method attempts to store data and enqueues a retry
        operation if it fails.
        
        Args:
            agent_id: ID of the agent
            state_data: Data to store
            store_func: Function that performs the actual storage
            priority: Priority level for the operation
            
        Returns:
            True if the operation succeeded, False if enqueued for retry
        """
        operation_id = str(uuid.uuid4())
        
        try:
            # Try immediate store
            success = store_func(agent_id, state_data)
            return success
        except (RedisUnavailableError, RedisTimeoutError) as e:
            logger.warning(
                "Redis operation failed for agent %s: %s", 
                agent_id, str(e)
            )
            
            if priority == Priority.CRITICAL:
                # For critical states, we need to retry immediately
                # with exponential backoff, up to a reasonable limit
                for attempt in range(3):  # Try 3 times for critical data
                    try:
                        delay = 0.5 * (2 ** attempt)  # 0.5, 1, 2 seconds
                        logger.info(
                            "Immediate retry %d for critical data after %.1f seconds", 
                            attempt + 1, delay
                        )
                        time.sleep(delay)
                        return store_func(agent_id, state_data)
                    except (RedisUnavailableError, RedisTimeoutError) as e:
                        logger.warning(
                            "Immediate retry %d failed: %s", 
                            attempt + 1, str(e)
                        )
                
                logger.error(
                    "Critical data store failed after immediate retries for agent %s", 
                    agent_id
                )
                return False
            elif priority == Priority.HIGH or priority == Priority.NORMAL:
                # For high/normal priority, enqueue for background retry
                operation = StoreOperation(
                    operation_id=operation_id,
                    agent_id=agent_id,
                    state_data=state_data,
                    store_function=store_func
                )
                self.recovery_queue.enqueue(
                    operation, 
                    priority=4 - priority.value  # Lower value = higher priority
                )
                logger.info(
                    "Enqueued %s priority store operation for agent %s", 
                    priority.name, agent_id
                )
                return False
            else:
                # For low priority, just log and continue
                logger.info(
                    "Low priority store operation failed for agent %s: %s", 
                    agent_id, str(e)
                )
                return False 
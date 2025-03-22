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
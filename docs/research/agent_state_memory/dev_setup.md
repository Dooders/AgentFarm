# Development Environment Setup for AgentMemory System

This document provides instructions for setting up the development environment for the AgentMemory system, including Redis for short-term and intermediate memory, SQLite for long-term memory, and testing harnesses for all memory tiers.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Redis Setup](#redis-setup)
3. [SQLite Configuration](#sqlite-configuration)
4. [Testing Environment Configuration](#testing-environment-configuration)
5. [Running Tests](#running-tests)

## Prerequisites

Ensure you have the following installed:
- Python 3.8+
- pip (Python package manager)

## Redis Setup

Redis is used for the Short-Term Memory (STM) and Intermediate Memory (IM) tiers.

### Local Redis Installation

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### macOS
```bash
brew install redis
brew services start redis
```

#### Windows
Download and install Redis from the [Redis Windows releases](https://github.com/microsoftarchive/redis/releases) or use WSL2 to run Redis as in Ubuntu.

### Docker-based Redis Setup (Recommended for Development)

A Docker-based setup ensures consistent environments and easy cleanup:

1. Create a `docker-compose.yml` file in your project root:

```yaml
version: '3'

services:
  redis-stm:
    image: redis:latest
    ports:
      - "6379:6379"
    volumes:
      - redis-stm-data:/data
    command: redis-server --appendonly yes --save 60 1 --loglevel warning
    networks:
      - agent-memory-network

  redis-im:
    image: redis:latest
    ports:
      - "6380:6379"
    volumes:
      - redis-im-data:/data
    command: redis-server --appendonly yes --save 60 1 --loglevel warning
    networks:
      - agent-memory-network

networks:
  agent-memory-network:
    driver: bridge

volumes:
  redis-stm-data:
  redis-im-data:
```

2. Start the Redis containers:
```bash
docker-compose up -d
```

### Redis Configuration for Development

Create a `redis_dev_config.py` file in your project:

```python
# Configuration for Redis in development environment

# STM Redis Configuration
STM_REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "password": None,
    "namespace": "dev:agent_memory:stm",
    "ttl": 3600  # 1 hour TTL for development
}

# IM Redis Configuration
IM_REDIS_CONFIG = {
    "host": "localhost",
    "port": 6380,  # Different port for IM
    "db": 0,
    "password": None,
    "namespace": "dev:agent_memory:im",
    "ttl": 86400  # 24 hour TTL for development
}
```

## SQLite Configuration

SQLite is used for the Long-Term Memory (LTM) tier.

### Setup SQLite for Development

1. Create a directory for SQLite databases:
```bash
mkdir -p data/sqlite
```

2. Create a configuration file `sqlite_dev_config.py`:

```python
# Configuration for SQLite in development environment
import os

# Ensure data directory exists
os.makedirs('data/sqlite', exist_ok=True)

# LTM SQLite Configuration
LTM_SQLITE_CONFIG = {
    "db_path": "data/sqlite/agent_memory_dev.db",
    "table_prefix": "dev_agent_ltm",
    "echo_sql": True,  # Set to True for SQL query logging during development
    "compression_level": 2
}

# Test database path - separate from development
TEST_SQLITE_CONFIG = {
    "db_path": "data/sqlite/agent_memory_test.db",
    "table_prefix": "test_agent_ltm",
    "echo_sql": True,
    "compression_level": 2
}
```

3. Create a SQLite database initialization script `init_sqlite_db.py`:

```python
"""Initialize SQLite database schemas for AgentMemory LTM."""

import sqlite3
import os
from sqlite_dev_config import LTM_SQLITE_CONFIG

def init_sqlite_db(config=None):
    """Initialize the SQLite database for the LTM tier."""
    if config is None:
        config = LTM_SQLITE_CONFIG
    
    # Ensure directory exists
    db_dir = os.path.dirname(config["db_path"])
    os.makedirs(db_dir, exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(config["db_path"])
    cursor = conn.cursor()
    
    # Create tables
    table_prefix = config["table_prefix"]
    
    # Agent memory entries table
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_prefix}_memories (
        memory_id TEXT PRIMARY KEY,
        agent_id TEXT NOT NULL,
        step_number INTEGER NOT NULL,
        timestamp INTEGER NOT NULL,
        
        content_json TEXT NOT NULL,
        metadata_json TEXT NOT NULL,
        
        compression_level INTEGER DEFAULT 2,
        importance_score REAL DEFAULT 0.0,
        retrieval_count INTEGER DEFAULT 0,
        memory_type TEXT NOT NULL,
        
        created_at INTEGER NOT NULL,
        last_accessed INTEGER NOT NULL
    )
    ''')
    
    # Create indices for faster retrieval
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_prefix}_agent_id ON {table_prefix}_memories (agent_id)')
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_prefix}_step ON {table_prefix}_memories (step_number)')
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_prefix}_type ON {table_prefix}_memories (memory_type)')
    cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_prefix}_importance ON {table_prefix}_memories (importance_score)')
    
    # Vector embeddings table
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {table_prefix}_embeddings (
        memory_id TEXT PRIMARY KEY,
        vector_blob BLOB NOT NULL,
        vector_dim INTEGER NOT NULL,
        
        FOREIGN KEY (memory_id) REFERENCES {table_prefix}_memories (memory_id) ON DELETE CASCADE
    )
    ''')
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"SQLite database initialized at {config['db_path']}")

if __name__ == "__main__":
    init_sqlite_db()
```

4. Run the initialization script:
```bash
python init_sqlite_db.py
```

## Testing Environment Configuration

Create a testing configuration and harness to test all memory tiers.

### Create Testing Configuration

Create a file `test_config.py`:

```python
"""Test configuration for AgentMemory system."""

# Master test configuration
TEST_CONFIG = {
    # Redis STM config
    "stm_config": {
        "host": "localhost",
        "port": 6379,
        "db": 1,  # Use a different DB for testing
        "namespace": "test:agent_memory:stm",
        "ttl": 300  # 5 minutes TTL for tests
    },
    
    # Redis IM config
    "im_config": {
        "host": "localhost",
        "port": 6380,
        "db": 1,  # Use a different DB for testing
        "namespace": "test:agent_memory:im",
        "ttl": 600  # 10 minutes TTL for tests
    },
    
    # SQLite LTM config
    "ltm_config": {
        "db_path": "data/sqlite/agent_memory_test.db",
        "table_prefix": "test_agent_ltm",
        "echo_sql": True
    },
    
    # Autoencoder config
    "autoencoder_config": {
        "input_dim": 16,  # Smaller dimension for testing
        "stm_dim": 12,
        "im_dim": 8,
        "ltm_dim": 4,
        "use_neural_embeddings": True
    }
}

# Test agent settings
TEST_AGENTS = [
    {"agent_id": "test-agent-1", "position": (10, 10), "resources": 100, "health": 1.0},
    {"agent_id": "test-agent-2", "position": (20, 20), "resources": 150, "health": 0.8},
    {"agent_id": "test-agent-3", "position": (30, 30), "resources": 50, "health": 0.6}
]

# Sample test states
TEST_STATES = [
    {
        "agent_id": "test-agent-1",
        "position_x": 10,
        "position_y": 10,
        "resource_level": 100,
        "current_health": 1.0,
        "is_defending": False
    },
    {
        "agent_id": "test-agent-1",
        "position_x": 11,
        "position_y": 10,
        "resource_level": 95,
        "current_health": 0.9,
        "is_defending": True
    },
    # Add more test states as needed
]

# Sample test actions
TEST_ACTIONS = [
    {
        "action_type": "move",
        "action_params": {"direction": "north"},
        "state_before": TEST_STATES[0],
        "state_after": TEST_STATES[1],
        "reward": 0.5
    },
    # Add more test actions as needed
]
```

### Create Testing Harnesses

#### 1. Basic Redis Connection Test

Create `test_redis_connection.py`:

```python
"""Test Redis connections for STM and IM tiers."""

import redis
import time
from test_config import TEST_CONFIG

def test_redis_connection():
    """Test basic Redis connectivity for STM and IM."""
    # Test STM Redis
    stm_config = TEST_CONFIG["stm_config"]
    try:
        stm_redis = redis.Redis(
            host=stm_config["host"],
            port=stm_config["port"],
            db=stm_config["db"],
            decode_responses=True
        )
        # Test set/get
        test_key = f"{stm_config['namespace']}:test"
        stm_redis.set(test_key, "test_value", ex=10)
        value = stm_redis.get(test_key)
        assert value == "test_value"
        print(f"✅ STM Redis connection successful on {stm_config['host']}:{stm_config['port']}")
    except Exception as e:
        print(f"❌ STM Redis connection failed: {e}")
    
    # Test IM Redis
    im_config = TEST_CONFIG["im_config"]
    try:
        im_redis = redis.Redis(
            host=im_config["host"],
            port=im_config["port"],
            db=im_config["db"],
            decode_responses=True
        )
        # Test set/get
        test_key = f"{im_config['namespace']}:test"
        im_redis.set(test_key, "test_value", ex=10)
        value = im_redis.get(test_key)
        assert value == "test_value"
        print(f"✅ IM Redis connection successful on {im_config['host']}:{im_config['port']}")
    except Exception as e:
        print(f"❌ IM Redis connection failed: {e}")

if __name__ == "__main__":
    test_redis_connection()
```

#### 2. SQLite Connection Test

Create `test_sqlite_connection.py`:

```python
"""Test SQLite connections for LTM tier."""

import sqlite3
import os
from test_config import TEST_CONFIG

def test_sqlite_connection():
    """Test basic SQLite connectivity for LTM."""
    ltm_config = TEST_CONFIG["ltm_config"]
    
    try:
        # Ensure directory exists
        db_dir = os.path.dirname(ltm_config["db_path"])
        os.makedirs(db_dir, exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(ltm_config["db_path"])
        cursor = conn.cursor()
        
        # Test table creation and query
        test_table = f"{ltm_config['table_prefix']}_test"
        cursor.execute(f"CREATE TABLE IF NOT EXISTS {test_table} (id INTEGER PRIMARY KEY, value TEXT)")
        cursor.execute(f"INSERT INTO {test_table} (value) VALUES ('test_value')")
        cursor.execute(f"SELECT value FROM {test_table} WHERE value = 'test_value'")
        result = cursor.fetchone()
        
        assert result[0] == "test_value"
        
        # Clean up
        cursor.execute(f"DROP TABLE {test_table}")
        conn.commit()
        conn.close()
        
        print(f"✅ SQLite connection successful at {ltm_config['db_path']}")
    except Exception as e:
        print(f"❌ SQLite connection failed: {e}")

if __name__ == "__main__":
    test_sqlite_connection()
```

#### 3. Comprehensive Memory Tier Test

Create `test_memory_tiers.py`:

```python
"""Test all memory tiers of the AgentMemory system."""

import json
import time
import random
import uuid
from test_config import TEST_CONFIG, TEST_STATES, TEST_ACTIONS

# Import memory components (adjust imports based on your implementation)
from farm.memory.agent_memory.storage.redis_stm import RedisSTMStore
from farm.memory.agent_memory.storage.redis_im import RedisIMStore
from farm.memory.agent_memory.storage.sqlite_ltm import SQLiteLTMStore
from farm.memory.agent_memory.config import RedisSTMConfig, RedisIMConfig, SQLiteLTMConfig

def create_test_memory_entry(agent_id, step_number, memory_type="state"):
    """Create a test memory entry."""
    timestamp = int(time.time())
    memory_id = f"{agent_id}-{step_number}-{timestamp}"
    
    # Use sample data from test states or generate random
    if memory_type == "state" and TEST_STATES:
        contents = random.choice(TEST_STATES).copy()
    elif memory_type == "action" and TEST_ACTIONS:
        contents = random.choice(TEST_ACTIONS).copy()
    else:
        contents = {
            "position_x": random.randint(0, 100),
            "position_y": random.randint(0, 100),
            "resource_level": random.randint(0, 200),
            "current_health": random.random(),
            "is_defending": random.choice([True, False])
        }
    
    # Add agent_id if not present
    if "agent_id" not in contents:
        contents["agent_id"] = agent_id
    
    return {
        "memory_id": memory_id,
        "agent_id": agent_id,
        "step_number": step_number,
        "timestamp": timestamp,
        
        "contents": contents,
        
        "metadata": {
            "creation_time": timestamp,
            "last_access_time": timestamp,
            "compression_level": 0,
            "importance_score": random.random(),
            "retrieval_count": 0,
            "memory_type": memory_type
        },
        
        "embeddings": {
            "full_vector": [random.random() for _ in range(10)],
            "compressed_vector": [random.random() for _ in range(5)],
            "abstract_vector": [random.random() for _ in range(3)]
        }
    }

def test_stm_storage():
    """Test STM storage operations."""
    print("\n--- Testing STM Storage ---")
    
    # Create config and store
    stm_config = RedisSTMConfig(**TEST_CONFIG["stm_config"])
    agent_id = "test-agent-" + str(uuid.uuid4())[:8]
    stm_store = RedisSTMStore(agent_id, stm_config)
    
    # Test store operation
    memory_entry = create_test_memory_entry(agent_id, 1, "state")
    success = stm_store.store(memory_entry)
    print(f"Store operation: {'✅ Success' if success else '❌ Failed'}")
    
    # Test get operation
    retrieved = stm_store.get(memory_entry["memory_id"])
    retrieval_success = retrieved is not None
    print(f"Get operation: {'✅ Success' if retrieval_success else '❌ Failed'}")
    
    # Test get_recent operation
    recent_entries = stm_store.get_recent(5)
    recent_success = len(recent_entries) > 0
    print(f"Get recent operation: {'✅ Success' if recent_success else '❌ Failed'}")
    
    # Test count operation
    count = stm_store.count()
    count_success = count > 0
    print(f"Count operation: {'✅ Success' if count_success else '❌ Failed'}")
    
    # Test delete operation
    delete_success = stm_store.delete(memory_entry["memory_id"])
    print(f"Delete operation: {'✅ Success' if delete_success else '❌ Failed'}")
    
    # Test clear operation
    clear_success = stm_store.clear()
    print(f"Clear operation: {'✅ Success' if clear_success else '❌ Failed'}")
    
    return all([success, retrieval_success, recent_success, count_success, delete_success, clear_success])

def test_im_storage():
    """Test IM storage operations."""
    print("\n--- Testing IM Storage ---")
    
    # Create config and store
    im_config = RedisIMConfig(**TEST_CONFIG["im_config"])
    agent_id = "test-agent-" + str(uuid.uuid4())[:8]
    im_store = RedisIMStore(agent_id, im_config)
    
    # Test store operation
    memory_entry = create_test_memory_entry(agent_id, 1, "state")
    # Set compression level for IM
    memory_entry["metadata"]["compression_level"] = 1
    success = im_store.store(memory_entry)
    print(f"Store operation: {'✅ Success' if success else '❌ Failed'}")
    
    # Test get operation
    retrieved = im_store.get(memory_entry["memory_id"])
    retrieval_success = retrieved is not None
    print(f"Get operation: {'✅ Success' if retrieval_success else '❌ Failed'}")
    
    # Test additional IM-specific operations here
    
    # Test clear operation
    clear_success = im_store.clear()
    print(f"Clear operation: {'✅ Success' if clear_success else '❌ Failed'}")
    
    return all([success, retrieval_success, clear_success])

def test_ltm_storage():
    """Test LTM storage operations."""
    print("\n--- Testing LTM Storage ---")
    
    # Create config and store
    ltm_config = SQLiteLTMConfig(**TEST_CONFIG["ltm_config"])
    agent_id = "test-agent-" + str(uuid.uuid4())[:8]
    ltm_store = SQLiteLTMStore(agent_id, ltm_config)
    
    # Test batch storage
    batch = []
    for i in range(5):
        memory_entry = create_test_memory_entry(agent_id, i+1, "state")
        # Set compression level for LTM
        memory_entry["metadata"]["compression_level"] = 2
        batch.append(memory_entry)
    
    batch_success = ltm_store.store_batch(batch)
    print(f"Batch store operation: {'✅ Success' if batch_success else '❌ Failed'}")
    
    # Test get operation
    retrieved = ltm_store.get(batch[0]["memory_id"])
    retrieval_success = retrieved is not None
    print(f"Get operation: {'✅ Success' if retrieval_success else '❌ Failed'}")
    
    # Test additional LTM-specific operations here
    
    # Test clear operation
    clear_success = ltm_store.clear()
    print(f"Clear operation: {'✅ Success' if clear_success else '❌ Failed'}")
    
    return all([batch_success, retrieval_success, clear_success])

def test_memory_transition():
    """Test memory transition between tiers."""
    print("\n--- Testing Memory Transition ---")
    
    # Create stores
    agent_id = "test-agent-" + str(uuid.uuid4())[:8]
    stm_store = RedisSTMStore(agent_id, RedisSTMConfig(**TEST_CONFIG["stm_config"]))
    im_store = RedisIMStore(agent_id, RedisIMConfig(**TEST_CONFIG["im_config"]))
    ltm_store = SQLiteLTMStore(agent_id, SQLiteLTMConfig(**TEST_CONFIG["ltm_config"]))
    
    # Store in STM
    memory_entry = create_test_memory_entry(agent_id, 1, "state")
    stm_success = stm_store.store(memory_entry)
    print(f"STM store: {'✅ Success' if stm_success else '❌ Failed'}")
    
    # Transition to IM (simulated)
    memory_entry["metadata"]["compression_level"] = 1
    im_success = im_store.store(memory_entry)
    print(f"IM store: {'✅ Success' if im_success else '❌ Failed'}")
    
    # Transition to LTM (simulated)
    memory_entry["metadata"]["compression_level"] = 2
    ltm_success = ltm_store.store(memory_entry)
    print(f"LTM store: {'✅ Success' if ltm_success else '❌ Failed'}")
    
    # Verify retrieval from each tier
    stm_retrieval = stm_store.get(memory_entry["memory_id"]) is not None
    print(f"STM retrieval: {'✅ Success' if stm_retrieval else '❌ Failed'}")
    
    im_retrieval = im_store.get(memory_entry["memory_id"]) is not None
    print(f"IM retrieval: {'✅ Success' if im_retrieval else '❌ Failed'}")
    
    ltm_retrieval = ltm_store.get(memory_entry["memory_id"]) is not None
    print(f"LTM retrieval: {'✅ Success' if ltm_retrieval else '❌ Failed'}")
    
    # Clean up
    stm_store.clear()
    im_store.clear()
    ltm_store.clear()
    
    return all([stm_success, im_success, ltm_success, stm_retrieval, im_retrieval, ltm_retrieval])

def run_all_tests():
    """Run all memory tier tests."""
    print("=== AgentMemory System Tier Tests ===")
    
    stm_result = test_stm_storage()
    im_result = test_im_storage()
    ltm_result = test_ltm_storage()
    transition_result = test_memory_transition()
    
    print("\n=== Test Results Summary ===")
    print(f"STM Storage: {'✅ PASSED' if stm_result else '❌ FAILED'}")
    print(f"IM Storage: {'✅ PASSED' if im_result else '❌ FAILED'}")
    print(f"LTM Storage: {'✅ PASSED' if ltm_result else '❌ FAILED'}")
    print(f"Memory Transition: {'✅ PASSED' if transition_result else '❌ FAILED'}")
    
    all_passed = all([stm_result, im_result, ltm_result, transition_result])
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    run_all_tests()
```

## Running Tests

### 1. Set Up the Test Environment

Make sure your Redis and SQLite testing environments are ready:

```bash
# Start Redis (if using Docker)
docker-compose up -d

# Initialize SQLite test database
python init_sqlite_db.py
```

### 2. Run Basic Connection Tests

```bash
# Test Redis connections
python test_redis_connection.py

# Test SQLite connection
python test_sqlite_connection.py
```

### 3. Run Comprehensive Memory Tier Tests

```bash
# Test all memory tiers
python test_memory_tiers.py
```

### 4. Automated Test Script

Create a `run_all_tests.sh` script for convenience:

```bash
#!/bin/bash
echo "=== AgentMemory Development Environment Test Suite ==="
echo ""

echo "Testing Redis connections..."
python test_redis_connection.py

echo ""
echo "Testing SQLite connection..."
python test_sqlite_connection.py

echo ""
echo "Running comprehensive memory tier tests..."
python test_memory_tiers.py

echo ""
echo "=== Test Suite Complete ==="
```

Make it executable:
```bash
chmod +x run_all_tests.sh
```

Run all tests:
```bash
./run_all_tests.sh
```

## Next Steps

Once your development environment is set up and all tests are passing, you can:

1. Implement the remaining components of the AgentMemory system
2. Write more specific unit tests for each component
3. Develop integration tests to verify the system as a whole
4. Begin integrating with the main agent framework

This setup provides a solid foundation for developing and testing the AgentMemory system with all its memory tiers properly configured.

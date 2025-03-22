# Agent Memory System

A hierarchical memory system for intelligent agents that provides efficient storage, retrieval, and compression of agent states and experiences across multiple memory tiers.

## Overview

The Agent Memory System implements a biologically-inspired memory architecture with three distinct memory tiers:

1. **Short-Term Memory (STM)**: High-resolution, rapid-access storage using Redis
2. **Intermediate Memory (IM)**: Compressed mid-term storage with Redis + TTL
3. **Long-Term Memory (LTM)**: Highly compressed long-term storage using SQLite

## Features

- **Hierarchical Storage**: Automatic memory transition between STM, IM, and LTM tiers
- **Neural Compression**: Autoencoder-based embedding generation for efficient storage
- **Flexible Integration**: Easy integration with existing agent systems via API or hooks
- **Priority-Based Memory**: Importance scoring for intelligent memory retention
- **Vector Search**: Similarity-based memory retrieval using embeddings
- **Configurable**: Extensive configuration options for all components

## Directory Structure

```
agent_memory/
├── __init__.py
├── core.py              # Core memory system implementation
├── config.py            # Configuration classes
├── memory_agent.py      # Memory agent implementation
├── embeddings/          # Neural embedding components
│   ├── autoencoder.py  # Autoencoder for compression
│   ├── vector_store.py # Vector storage utilities
│   └── compression.py  # Compression algorithms
├── storage/            # Storage backend implementations
│   ├── redis_stm.py   # Redis STM storage
│   ├── redis_im.py    # Redis IM storage
│   └── sqlite_ltm.py  # SQLite LTM storage
├── retrieval/          # Memory retrieval components
│   ├── similarity.py  # Similarity search
│   ├── temporal.py    # Time-based retrieval
│   └── attribute.py   # Attribute-based retrieval
├── api/               # API interfaces
│   ├── memory_api.py # Main API interface
│   └── hooks.py      # Agent integration hooks
└── utils/            # Utility functions
    ├── serialization.py
    └── redis_utils.py
```

## Requirements

- Python 3.8+
- PyTorch
- Redis
- SQLite
- NumPy
- SQLAlchemy

## Installation

1. Install Redis server:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # macOS
   brew install redis
   
   # Windows
   # Download from https://github.com/microsoftarchive/redis/releases
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Usage

```python
from farm.memory import AgentMemoryAPI, MemoryConfig

# Initialize memory system
memory_api = AgentMemoryAPI(MemoryConfig())

# Store agent state
memory_api.store_agent_state(
    agent_id="agent1",
    state_data={"position": [0, 0], "health": 100},
    step_number=1
)

# Store agent action
memory_api.store_agent_action(
    agent_id="agent1",
    action_data={"action": "move", "direction": "north"},
    step_number=1
)
```

### Using Memory Hooks

```python
from farm.memory import install_memory_hooks
from farm.agents import BaseAgent

@install_memory_hooks
class MyAgent(BaseAgent):
    def act(self, observation):
        # Memory hooks will automatically store states and actions
        return super().act(observation)
```

### Custom Configuration

```python
from farm.memory import MemoryConfig, RedisSTMConfig

config = MemoryConfig(
    stm_config=RedisSTMConfig(
        host="localhost",
        port=6379,
        ttl=86400,  # 24 hours
        memory_limit=1000
    )
)

memory_api = AgentMemoryAPI(config)
```

## Memory Tiers

### Short-Term Memory (STM)
- High-resolution storage of recent experiences
- Fast access and retrieval
- Limited capacity with automatic cleanup
- Full feature vectors with minimal compression

### Intermediate Memory (IM)
- Medium-term storage with moderate compression
- TTL-based expiration
- Balanced between resolution and storage efficiency
- Compressed feature vectors

### Long-Term Memory (LTM)
- Long-term persistent storage
- Highly compressed representations
- Efficient storage of essential information
- Abstract feature vectors

## Advanced Features

### Neural Compression
The system uses an autoencoder architecture to generate compressed embeddings:
- STM: 384-dimensional embeddings
- IM: 128-dimensional embeddings
- LTM: 32-dimensional embeddings

### Memory Transitions
Memories automatically transition between tiers based on:
- Age
- Importance score
- Access frequency
- Storage capacity limits

### Vector Similarity Search
Find similar memories using embedding-based similarity search:
```python
similar_states = memory_api.retrieve_states_by_similarity(
    agent_id="agent1",
    query_state=current_state,
    count=10
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
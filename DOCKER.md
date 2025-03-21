# Redis Docker Setup

This document provides instructions for running Redis using Docker for development and experiments.

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop/) installed on your system
- [Docker Compose](https://docs.docker.com/compose/install/) (included with Docker Desktop)

## Quick Start

To start the Redis server using Docker:

```bash
# Navigate to the benchmarks/redis directory
cd benchmarks/redis

# Start Redis in the background
docker-compose up -d
```

This will start a Redis server accessible at `localhost:6379`.

## Using Redis with the Benchmarks

When using Docker for Redis, you can run the benchmarks with the `--docker` flag, which will automatically set the correct Redis connection parameters:

```bash
# Using the shell script
./run_redis_benchmark.sh --docker

# Or with batch script on Windows
run_redis_benchmark.bat --docker

# To start Redis and run benchmarks in one command
./run_redis_benchmark.sh --docker --start-redis
```

### Manual Configuration

You can also manually specify the Redis environment in the Python scripts:

```bash
# Use the Docker configuration
python simple_redis_benchmark.py --redis-env DOCKER --memory-entries 500 --batch-size 100
```

## Redis Connection Configuration

The Redis connection settings are stored in `config.py`. The following environments are defined:

```python
REDIS_CONFIG = {
    # Default connection settings
    "DEFAULT": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": None,
        "decode_responses": True,
    },
    
    # Docker container connection
    "DOCKER": {
        "host": "localhost",
        "port": 6379,
        "db": 0,
        "password": None,
        "decode_responses": True,
    },
    
    # Remote Redis server (example)
    "REMOTE": {
        "host": os.environ.get("REDIS_HOST", "your-redis-server.com"),
        "port": int(os.environ.get("REDIS_PORT", 6379)),
        "db": int(os.environ.get("REDIS_DB", 0)),
        "password": os.environ.get("REDIS_PASSWORD", None),
        "decode_responses": True,
    },
}
```

To connect to a remote Redis server, set environment variables:

```bash
# For Linux/macOS
export REDIS_HOST=my-redis-server.example.com
export REDIS_PORT=6379
export REDIS_PASSWORD=mypassword

# For Windows
set REDIS_HOST=my-redis-server.example.com
set REDIS_PORT=6379
set REDIS_PASSWORD=mypassword

# Then run with REMOTE config
python simple_redis_benchmark.py --redis-env REMOTE
```

## Docker Commands

### View Redis Logs

```bash
docker logs redis-benchmark
```

### Redis CLI

Connect to the Redis server using the redis-cli inside the container:

```bash
docker exec -it redis-benchmark redis-cli
```

### Stop Redis Server

```bash
docker-compose down
```

### Stop Redis and Remove Data Volume

```bash
docker-compose down -v
```

## Redis Configuration

The Redis server is configured with:
- Append-only file persistence enabled (data durability)
- RDB snapshots every 60 seconds if at least 1 key changed
- Warning-level logging to reduce noise

To modify the configuration, edit the `command` line in `docker-compose.yml`.

## Performance Considerations

When running benchmarks with Docker:
- Redis in Docker may perform slightly differently than native installation
- For consistent benchmark results, always use the same environment
- Docker volume I/O can impact Redis performance if persistence is enabled

## Troubleshooting

1. **Port Conflict**: If you see an error about port 6379 being in use:
   ```bash
   # Find what's using the port
   netstat -ano | findstr :6379  # Windows
   lsof -i :6379                 # Linux/macOS
   
   # Stop the existing Redis server or change the port in docker-compose.yml
   ```

2. **Connection Refused**: If benchmarks can't connect to Redis:
   ```bash
   # Verify Redis container is running
   docker ps
   
   # Check Redis logs
   docker logs redis-benchmark
   ```

3. **Container Fails to Start**: Check system resources and Docker logs:
   ```bash
   docker-compose logs
   ``` 
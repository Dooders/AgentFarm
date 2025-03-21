"""
Redis benchmark configuration file.
This file contains settings for the Redis connection and benchmark parameters.
"""

import os
import pathlib

# Path to the docker-compose.yml file
DOCKER_COMPOSE_PATH = os.path.join(pathlib.Path(__file__).parent.parent.parent, "docker-compose.yml")

# Redis Connection Settings
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

# Benchmark Default Settings
BENCHMARK_CONFIG = {
    "memory_entries": 500,
    "batch_size": 100,
    "iterations": 3,
    "output_dir": "benchmark_results",
}


# Function to get Redis connection parameters
def get_redis_config(environment="DEFAULT"):
    """
    Get Redis connection parameters for the specified environment.

    Args:
        environment (str): The environment to use (DEFAULT, DOCKER, REMOTE)

    Returns:
        dict: Redis connection parameters
    """
    if environment in REDIS_CONFIG:
        return REDIS_CONFIG[environment]
    else:
        print(f"Warning: Unknown environment '{environment}'. Using DEFAULT.")
        return REDIS_CONFIG["DEFAULT"]

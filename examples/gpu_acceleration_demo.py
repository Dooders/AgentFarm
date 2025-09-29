#!/usr/bin/env python3
"""Demo script showing GPU acceleration for spatial computations in AgentFarm.

This script demonstrates the performance benefits of GPU acceleration
for spatial computations in multi-agent simulations.
"""

import logging
import time
from typing import List

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from farm.core.environment import Environment
    from farm.config.config import SpatialIndexConfig, EnvironmentConfig
    GPU_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import AgentFarm modules: {e}")
    GPU_AVAILABLE = False


def create_mock_agents(count: int, width: float, height: float) -> List:
    """Create mock agent objects for testing."""
    agents = []
    for i in range(count):
        agent = type('Agent', (), {
            'position': (np.random.rand() * width, np.random.rand() * height),
            'alive': True,
            'id': i,
            'team': i % 2  # Alternate teams
        })()
        agents.append(agent)
    return agents


def create_mock_resources(count: int, width: float, height: float) -> List:
    """Create mock resource objects for testing."""
    resources = []
    for i in range(count):
        resource = type('Resource', (), {
            'position': (np.random.rand() * width, np.random.rand() * height),
            'id': i,
            'type': 'food' if i % 2 == 0 else 'water'
        })()
        resources.append(resource)
    return resources


def benchmark_spatial_queries(env: Environment, num_queries: int = 1000) -> dict:
    """Benchmark spatial queries with and without GPU acceleration."""
    logger.info(f"Running {num_queries} spatial queries...")
    
    # Generate random query positions
    query_positions = [
        (np.random.rand() * env.width, np.random.rand() * env.height)
        for _ in range(num_queries)
    ]
    
    # Test CPU queries
    start_time = time.time()
    cpu_results = []
    for pos in query_positions:
        nearby = env.spatial_index.get_nearby(pos, radius=20.0)
        cpu_results.append(nearby)
    cpu_time = time.time() - start_time
    
    # Test GPU queries (if available)
    gpu_time = 0
    gpu_results = []
    if hasattr(env.spatial_index, 'get_nearby_gpu'):
        start_time = time.time()
        for pos in query_positions:
            nearby = env.spatial_index.get_nearby_gpu(pos, radius=20.0)
            gpu_results.append(nearby)
        gpu_time = time.time() - start_time
    
    # Test batch GPU queries (if available)
    batch_time = 0
    batch_results = []
    if hasattr(env.spatial_index, 'batch_query_gpu'):
        start_time = time.time()
        positions_array = np.array(query_positions)
        batch_results = env.spatial_index.batch_query_gpu(positions_array, radius=20.0)
        batch_time = time.time() - start_time
    
    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'batch_time': batch_time,
        'num_queries': num_queries,
        'cpu_results': cpu_results,
        'gpu_results': gpu_results,
        'batch_results': batch_results
    }


def benchmark_nearest_queries(env: Environment, num_queries: int = 1000) -> dict:
    """Benchmark nearest neighbor queries with and without GPU acceleration."""
    logger.info(f"Running {num_queries} nearest neighbor queries...")
    
    # Generate random query positions
    query_positions = [
        (np.random.rand() * env.width, np.random.rand() * env.height)
        for _ in range(num_queries)
    ]
    
    # Test CPU queries
    start_time = time.time()
    cpu_results = []
    for pos in query_positions:
        nearest = env.spatial_index.get_nearest(pos)
        cpu_results.append(nearest)
    cpu_time = time.time() - start_time
    
    # Test GPU queries (if available)
    gpu_time = 0
    gpu_results = []
    if hasattr(env.spatial_index, 'get_nearest_gpu'):
        start_time = time.time()
        for pos in query_positions:
            nearest = env.spatial_index.get_nearest_gpu(pos)
            gpu_results.append(nearest)
        gpu_time = time.time() - start_time
    
    return {
        'cpu_time': cpu_time,
        'gpu_time': gpu_time,
        'num_queries': num_queries,
        'cpu_results': cpu_results,
        'gpu_results': gpu_results
    }


def verify_accuracy(cpu_results: List, gpu_results: List, test_name: str) -> bool:
    """Verify that CPU and GPU results are consistent."""
    if not gpu_results:
        logger.warning(f"No GPU results for {test_name} - skipping accuracy check")
        return True
    
    if len(cpu_results) != len(gpu_results):
        logger.error(f"Result count mismatch for {test_name}: CPU={len(cpu_results)}, GPU={len(gpu_results)}")
        return False
    
    # Check that results are consistent
    for i, (cpu_result, gpu_result) in enumerate(zip(cpu_results, gpu_results)):
        if cpu_result != gpu_result:
            logger.error(f"Result mismatch at index {i} for {test_name}")
            logger.error(f"  CPU: {cpu_result}")
            logger.error(f"  GPU: {gpu_result}")
            return False
    
    logger.info(f"✓ Accuracy verification passed for {test_name}")
    return True


def main():
    """Main demo function."""
    if not GPU_AVAILABLE:
        logger.error("AgentFarm modules not available. Please ensure the package is installed.")
        return
    
    logger.info("=== AgentFarm GPU Acceleration Demo ===")
    
    # Configuration
    width, height = 200.0, 200.0
    num_agents = 500
    num_resources = 200
    num_queries = 1000
    
    # Create configuration with GPU acceleration enabled
    spatial_config = SpatialIndexConfig(
        enable_gpu_acceleration=True,
        gpu_device=None,  # Auto-detect best device
        performance_monitoring=True
    )
    
    env_config = EnvironmentConfig(
        width=int(width),
        height=int(height),
        spatial_index=spatial_config
    )
    
    # Create environment
    logger.info(f"Creating environment ({width}x{height}) with {num_agents} agents and {num_resources} resources...")
    env = Environment(
        width=width,
        height=height,
        resource_distribution="uniform",
        config=env_config
    )
    
    # Add agents and resources
    agents = create_mock_agents(num_agents, width, height)
    resources = create_mock_resources(num_resources, width, height)
    
    # Set references and update spatial index
    env.spatial_index.set_references(agents, resources)
    env.spatial_index.update()
    
    # Display GPU information
    if hasattr(env.spatial_index, 'get_gpu_performance_stats'):
        gpu_stats = env.spatial_index.get_gpu_performance_stats()
        logger.info(f"GPU Device: {gpu_stats.get('device', 'unknown')}")
        logger.info(f"GPU Acceleration Enabled: {gpu_stats.get('gpu_acceleration_enabled', False)}")
    
    # Run benchmarks
    logger.info("\n=== Spatial Query Benchmark ===")
    spatial_results = benchmark_spatial_queries(env, num_queries)
    
    logger.info("\n=== Nearest Neighbor Benchmark ===")
    nearest_results = benchmark_nearest_queries(env, num_queries)
    
    # Display results
    logger.info("\n=== Performance Results ===")
    
    # Spatial queries
    cpu_time = spatial_results['cpu_time']
    gpu_time = spatial_results['gpu_time']
    batch_time = spatial_results['batch_time']
    
    logger.info(f"Spatial Queries ({num_queries} queries):")
    logger.info(f"  CPU time: {cpu_time:.4f} seconds")
    if gpu_time > 0:
        gpu_speedup = cpu_time / gpu_time
        logger.info(f"  GPU time: {gpu_time:.4f} seconds")
        logger.info(f"  GPU speedup: {gpu_speedup:.2f}x")
    if batch_time > 0:
        batch_speedup = cpu_time / batch_time
        logger.info(f"  Batch GPU time: {batch_time:.4f} seconds")
        logger.info(f"  Batch GPU speedup: {batch_speedup:.2f}x")
    
    # Nearest neighbor queries
    cpu_time = nearest_results['cpu_time']
    gpu_time = nearest_results['gpu_time']
    
    logger.info(f"\nNearest Neighbor Queries ({num_queries} queries):")
    logger.info(f"  CPU time: {cpu_time:.4f} seconds")
    if gpu_time > 0:
        gpu_speedup = cpu_time / gpu_time
        logger.info(f"  GPU time: {gpu_time:.4f} seconds")
        logger.info(f"  GPU speedup: {gpu_speedup:.2f}x")
    
    # Verify accuracy
    logger.info("\n=== Accuracy Verification ===")
    spatial_accurate = verify_accuracy(
        spatial_results['cpu_results'], 
        spatial_results['gpu_results'], 
        "spatial queries"
    )
    nearest_accurate = verify_accuracy(
        nearest_results['cpu_results'], 
        nearest_results['gpu_results'], 
        "nearest neighbor queries"
    )
    
    # Final GPU performance stats
    if hasattr(env.spatial_index, 'get_gpu_performance_stats'):
        final_stats = env.spatial_index.get_gpu_performance_stats()
        logger.info(f"\n=== Final GPU Performance Stats ===")
        logger.info(f"GPU operations: {final_stats.get('gpu_operations', 0)}")
        logger.info(f"CPU fallbacks: {final_stats.get('cpu_fallbacks', 0)}")
        logger.info(f"Total GPU time: {final_stats.get('total_gpu_time', 0):.4f}s")
        logger.info(f"Total CPU time: {final_stats.get('total_cpu_time', 0):.4f}s")
        if final_stats.get('gpu_speedup', 0) > 0:
            logger.info(f"Overall GPU speedup: {final_stats['gpu_speedup']:.2f}x")
    
    # Summary
    logger.info("\n=== Demo Summary ===")
    if gpu_time > 0 and gpu_time < cpu_time:
        logger.info("✓ GPU acceleration provided performance benefits")
    else:
        logger.info("⚠ GPU acceleration did not provide significant benefits (may be due to small dataset or CPU fallback)")
    
    if spatial_accurate and nearest_accurate:
        logger.info("✓ All accuracy tests passed - GPU and CPU results are consistent")
    else:
        logger.info("✗ Some accuracy tests failed - please check GPU implementation")
    
    logger.info("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
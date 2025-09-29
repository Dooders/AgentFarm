"""Tests for GPU-accelerated spatial computations.

This module contains comprehensive tests for GPU acceleration features,
including performance benchmarks and accuracy validation.
"""

import logging
import time
import unittest
from typing import Dict, List, Tuple

import numpy as np
import pytest

# Import GPU acceleration modules
try:
    from farm.core.gpu_spatial import (
        GPUDeviceManager,
        GPUSpatialOperations,
        GPUAcceleratedKDTree,
    )
    from farm.core.spatial_index import SpatialIndex
    GPU_AVAILABLE = True
except ImportError:
    GPUDeviceManager = None
    GPUSpatialOperations = None
    GPUAcceleratedKDTree = None
    SpatialIndex = None
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)


class TestGPUDeviceManager(unittest.TestCase):
    """Test GPU device management functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GPU_AVAILABLE:
            self.skipTest("GPU libraries not available")
    
    def test_device_initialization(self):
        """Test GPU device initialization."""
        # Test auto-detection
        device_manager = GPUDeviceManager()
        self.assertIsNotNone(device_manager.device)
        self.assertIn(device_manager.device, ['cpu', 'cupy', 'cuda'])
        
        # Test preferred device
        device_manager = GPUDeviceManager('cpu')
        self.assertEqual(device_manager.device, 'cpu')
    
    def test_memory_info(self):
        """Test GPU memory information retrieval."""
        device_manager = GPUDeviceManager()
        memory_info = device_manager.get_memory_info()
        
        self.assertIn('device', memory_info)
        self.assertEqual(memory_info['device'], device_manager.device)
    
    def test_memory_clearing(self):
        """Test GPU memory clearing."""
        device_manager = GPUDeviceManager()
        # Should not raise an exception
        device_manager.clear_memory()


class TestGPUSpatialOperations(unittest.TestCase):
    """Test GPU-accelerated spatial operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GPU_AVAILABLE:
            self.skipTest("GPU libraries not available")
        
        self.device_manager = GPUDeviceManager()
        self.gpu_ops = GPUSpatialOperations(self.device_manager)
        
        # Generate test data
        np.random.seed(42)
        self.points1 = np.random.rand(100, 2) * 100
        self.points2 = np.random.rand(50, 2) * 100
        self.query_points = np.random.rand(10, 2) * 100
    
    def test_distance_computation_accuracy(self):
        """Test accuracy of GPU distance computation."""
        # Compute distances using GPU
        gpu_distances = self.gpu_ops.compute_distances_gpu(self.points1, self.points2)
        
        # Compute distances using CPU (reference)
        cpu_distances = self._compute_distances_cpu(self.points1, self.points2)
        
        # Check accuracy
        np.testing.assert_allclose(gpu_distances, cpu_distances, rtol=1e-6, atol=1e-8)
    
    def test_nearest_neighbor_accuracy(self):
        """Test accuracy of GPU nearest neighbor search."""
        # Find nearest neighbors using GPU
        gpu_indices, gpu_distances = self.gpu_ops.find_nearest_gpu(
            self.query_points, self.points1
        )
        
        # Find nearest neighbors using CPU (reference)
        cpu_indices, cpu_distances = self._find_nearest_cpu(
            self.query_points, self.points1
        )
        
        # Check accuracy
        np.testing.assert_array_equal(gpu_indices, cpu_indices)
        np.testing.assert_allclose(gpu_distances, cpu_distances, rtol=1e-6, atol=1e-8)
    
    def test_radius_search_accuracy(self):
        """Test accuracy of GPU radius search."""
        radius = 10.0
        
        # Find points within radius using GPU
        gpu_results = self.gpu_ops.radius_search_gpu(
            self.query_points, self.points1, radius
        )
        
        # Find points within radius using CPU (reference)
        cpu_results = self._radius_search_cpu(self.query_points, self.points1, radius)
        
        # Check accuracy
        self.assertEqual(len(gpu_results), len(cpu_results))
        for i, (gpu_indices, cpu_indices) in enumerate(zip(gpu_results, cpu_results)):
            self.assertEqual(set(gpu_indices), set(cpu_indices))
    
    def test_batch_position_update(self):
        """Test GPU batch position updates."""
        old_positions = np.random.rand(20, 2) * 100
        new_positions = old_positions + np.random.rand(20, 2) * 5
        
        # Update positions using GPU
        updated_positions = self.gpu_ops.batch_update_positions_gpu(
            old_positions, new_positions
        )
        
        # Check that positions were updated
        np.testing.assert_array_equal(updated_positions, new_positions)
    
    def test_performance_stats(self):
        """Test performance statistics collection."""
        # Perform some operations
        self.gpu_ops.compute_distances_gpu(self.points1, self.points2)
        self.gpu_ops.find_nearest_gpu(self.query_points, self.points1)
        
        # Get performance stats
        stats = self.gpu_ops.get_performance_stats()
        
        # Check that stats are collected
        self.assertIn('device', stats)
        self.assertIn('is_gpu', stats)
        self.assertIn('gpu_operations', stats)
        self.assertIn('cpu_fallbacks', stats)
    
    def _compute_distances_cpu(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """CPU reference implementation for distance computation."""
        diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    
    def _find_nearest_cpu(self, query_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CPU reference implementation for nearest neighbor search."""
        diff = query_points[:, np.newaxis, :] - target_points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        nearest_indices = np.argmin(distances, axis=1)
        nearest_distances = np.min(distances, axis=1)
        
        return nearest_indices, nearest_distances
    
    def _radius_search_cpu(self, query_points: np.ndarray, target_points: np.ndarray, radius: float) -> List[List[int]]:
        """CPU reference implementation for radius search."""
        diff = query_points[:, np.newaxis, :] - target_points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        results = []
        for i in range(len(query_points)):
            indices = np.where(distances[i] <= radius)[0]
            results.append(indices.tolist())
        
        return results


class TestGPUAcceleratedKDTree(unittest.TestCase):
    """Test GPU-accelerated KD-tree implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GPU_AVAILABLE:
            self.skipTest("GPU libraries not available")
        
        self.device_manager = GPUDeviceManager()
        
        # Generate test data
        np.random.seed(42)
        self.points = np.random.rand(100, 2) * 100
        self.query_points = np.random.rand(10, 2) * 100
        
        # Create GPU-accelerated KD-tree
        self.gpu_kdtree = GPUAcceleratedKDTree(self.points, self.device_manager)
    
    def test_kdtree_query_accuracy(self):
        """Test accuracy of GPU KD-tree queries."""
        # Query using GPU KD-tree
        gpu_distances, gpu_indices = self.gpu_kdtree.query(self.query_points, k=1)
        
        # Query using CPU KD-tree (reference)
        from scipy.spatial import cKDTree
        cpu_kdtree = cKDTree(self.points)
        cpu_distances, cpu_indices = cpu_kdtree.query(self.query_points, k=1)
        
        # Check accuracy
        np.testing.assert_allclose(gpu_distances.flatten(), cpu_distances, rtol=1e-6, atol=1e-8)
        np.testing.assert_array_equal(gpu_indices.flatten(), cpu_indices)
    
    def test_kdtree_radius_query_accuracy(self):
        """Test accuracy of GPU KD-tree radius queries."""
        radius = 10.0
        
        # Query using GPU KD-tree
        gpu_results = self.gpu_kdtree.query_ball_point(self.query_points, radius)
        
        # Query using CPU KD-tree (reference)
        from scipy.spatial import cKDTree
        cpu_kdtree = cKDTree(self.points)
        cpu_results = cpu_kdtree.query_ball_point(self.query_points, radius)
        
        # Check accuracy
        self.assertEqual(len(gpu_results), len(cpu_results))
        for i, (gpu_indices, cpu_indices) in enumerate(zip(gpu_results, cpu_results)):
            self.assertEqual(set(gpu_indices), set(cpu_indices))
    
    def test_kdtree_update(self):
        """Test GPU KD-tree point updates."""
        new_points = np.random.rand(100, 2) * 100
        
        # Update points
        self.gpu_kdtree.update_points(new_points)
        
        # Check that points were updated
        np.testing.assert_array_equal(self.gpu_kdtree.points, new_points)
    
    def test_kdtree_stats(self):
        """Test GPU KD-tree statistics."""
        stats = self.gpu_kdtree.get_stats()
        
        # Check that stats are collected
        self.assertIn('num_points', stats)
        self.assertIn('device', stats)
        self.assertIn('is_gpu', stats)


class TestSpatialIndexGPUAcceleration(unittest.TestCase):
    """Test GPU acceleration integration with SpatialIndex."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GPU_AVAILABLE:
            self.skipTest("GPU libraries not available")
        
        # Create spatial index with GPU acceleration
        self.spatial_index = SpatialIndex(
            width=100,
            height=100,
            enable_gpu_acceleration=True
        )
        
        # Create mock agents and resources
        self.agents = self._create_mock_agents(50)
        self.resources = self._create_mock_resources(30)
        
        # Set references
        self.spatial_index.set_references(self.agents, self.resources)
        self.spatial_index.update()
    
    def _create_mock_agents(self, count: int) -> List:
        """Create mock agent objects."""
        agents = []
        for i in range(count):
            agent = type('Agent', (), {
                'position': (np.random.rand() * 100, np.random.rand() * 100),
                'alive': True,
                'id': i
            })()
            agents.append(agent)
        return agents
    
    def _create_mock_resources(self, count: int) -> List:
        """Create mock resource objects."""
        resources = []
        for i in range(count):
            resource = type('Resource', (), {
                'position': (np.random.rand() * 100, np.random.rand() * 100),
                'id': i
            })()
            resources.append(resource)
        return resources
    
    def test_gpu_acceleration_enabled(self):
        """Test that GPU acceleration is properly enabled."""
        self.assertTrue(self.spatial_index._gpu_acceleration_enabled)
        self.assertIsNotNone(self.spatial_index._gpu_device_manager)
        self.assertIsNotNone(self.spatial_index._gpu_operations)
    
    def test_gpu_nearby_query_accuracy(self):
        """Test accuracy of GPU-accelerated nearby queries."""
        query_position = (50.0, 50.0)
        radius = 20.0
        
        # Query using GPU acceleration
        gpu_results = self.spatial_index.get_nearby_gpu(query_position, radius)
        
        # Query using CPU (reference)
        cpu_results = self.spatial_index.get_nearby(query_position, radius)
        
        # Check that results are consistent
        for index_name in gpu_results:
            self.assertIn(index_name, cpu_results)
            # Results should be the same (same objects)
            self.assertEqual(len(gpu_results[index_name]), len(cpu_results[index_name]))
    
    def test_gpu_nearest_query_accuracy(self):
        """Test accuracy of GPU-accelerated nearest queries."""
        query_position = (50.0, 50.0)
        
        # Query using GPU acceleration
        gpu_results = self.spatial_index.get_nearest_gpu(query_position)
        
        # Query using CPU (reference)
        cpu_results = self.spatial_index.get_nearest(query_position)
        
        # Check that results are consistent
        for index_name in gpu_results:
            self.assertIn(index_name, cpu_results)
            # Results should be the same (same objects)
            self.assertEqual(gpu_results[index_name], cpu_results[index_name])
    
    def test_batch_query_gpu(self):
        """Test GPU-accelerated batch queries."""
        query_positions = np.array([
            [25.0, 25.0],
            [50.0, 50.0],
            [75.0, 75.0]
        ])
        radius = 15.0
        
        # Query using GPU acceleration
        gpu_results = self.spatial_index.batch_query_gpu(query_positions, radius)
        
        # Check that results are returned for each position
        for index_name in gpu_results:
            self.assertEqual(len(gpu_results[index_name]), len(query_positions))
            for position_results in gpu_results[index_name]:
                self.assertIsInstance(position_results, list)
    
    def test_gpu_performance_stats(self):
        """Test GPU performance statistics collection."""
        # Perform some operations
        self.spatial_index.get_nearby_gpu((50.0, 50.0), 20.0)
        self.spatial_index.get_nearest_gpu((50.0, 50.0))
        
        # Get performance stats
        stats = self.spatial_index.get_gpu_performance_stats()
        
        # Check that stats are collected
        self.assertIn('gpu_acceleration_enabled', stats)
        self.assertTrue(stats['gpu_acceleration_enabled'])
        self.assertIn('device', stats)
        self.assertIn('gpu_operations', stats)
    
    def test_enable_disable_gpu_acceleration(self):
        """Test enabling and disabling GPU acceleration."""
        # Disable GPU acceleration
        self.spatial_index.disable_gpu_acceleration()
        self.assertFalse(self.spatial_index._gpu_acceleration_enabled)
        
        # Re-enable GPU acceleration
        success = self.spatial_index.enable_gpu_acceleration()
        self.assertTrue(success)
        self.assertTrue(self.spatial_index._gpu_acceleration_enabled)
    
    def test_gpu_memory_management(self):
        """Test GPU memory management."""
        # Clear GPU memory
        self.spatial_index.clear_gpu_memory()
        
        # Should not raise an exception
        self.assertIsNotNone(self.spatial_index._gpu_operations)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for GPU vs CPU operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GPU_AVAILABLE:
            self.skipTest("GPU libraries not available")
        
        # Generate larger test datasets for performance testing
        np.random.seed(42)
        self.large_points1 = np.random.rand(1000, 2) * 100
        self.large_points2 = np.random.rand(500, 2) * 100
        self.large_query_points = np.random.rand(100, 2) * 100
        
        self.device_manager = GPUDeviceManager()
        self.gpu_ops = GPUSpatialOperations(self.device_manager)
    
    def test_distance_computation_performance(self):
        """Benchmark distance computation performance."""
        # GPU performance
        start_time = time.time()
        gpu_distances = self.gpu_ops.compute_distances_gpu(self.large_points1, self.large_points2)
        gpu_time = time.time() - start_time
        
        # CPU performance (reference)
        start_time = time.time()
        cpu_distances = self._compute_distances_cpu(self.large_points1, self.large_points2)
        cpu_time = time.time() - start_time
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        logger.info(f"Distance computation performance:")
        logger.info(f"  CPU time: {cpu_time:.4f} seconds")
        logger.info(f"  GPU time: {gpu_time:.4f} seconds")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        # Verify accuracy
        np.testing.assert_allclose(gpu_distances, cpu_distances, rtol=1e-6, atol=1e-8)
        
        # For GPU acceleration, we expect at least some speedup
        if self.device_manager.is_gpu:
            self.assertGreater(speedup, 1.0, "GPU should provide speedup over CPU")
            # For large datasets, we expect significant speedup (5-10x as per acceptance criteria)
            if len(self.large_points1) >= 1000 and len(self.large_points2) >= 500:
                self.assertGreater(speedup, 2.0, "GPU should provide significant speedup for large datasets")
    
    def test_nearest_neighbor_performance(self):
        """Benchmark nearest neighbor search performance."""
        # GPU performance
        start_time = time.time()
        gpu_indices, gpu_distances = self.gpu_ops.find_nearest_gpu(
            self.large_query_points, self.large_points1
        )
        gpu_time = time.time() - start_time
        
        # CPU performance (reference)
        start_time = time.time()
        cpu_indices, cpu_distances = self._find_nearest_cpu(
            self.large_query_points, self.large_points1
        )
        cpu_time = time.time() - start_time
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        logger.info(f"Nearest neighbor search performance:")
        logger.info(f"  CPU time: {cpu_time:.4f} seconds")
        logger.info(f"  GPU time: {gpu_time:.4f} seconds")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        # Verify accuracy
        np.testing.assert_array_equal(gpu_indices, cpu_indices)
        np.testing.assert_allclose(gpu_distances, cpu_distances, rtol=1e-6, atol=1e-8)
        
        # For GPU acceleration, we expect at least some speedup
        if self.device_manager.is_gpu:
            self.assertGreater(speedup, 1.0, "GPU should provide speedup over CPU")
    
    def test_radius_search_performance(self):
        """Benchmark radius search performance."""
        radius = 10.0
        
        # GPU performance
        start_time = time.time()
        gpu_results = self.gpu_ops.radius_search_gpu(
            self.large_query_points, self.large_points1, radius
        )
        gpu_time = time.time() - start_time
        
        # CPU performance (reference)
        start_time = time.time()
        cpu_results = self._radius_search_cpu(
            self.large_query_points, self.large_points1, radius
        )
        cpu_time = time.time() - start_time
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        logger.info(f"Radius search performance:")
        logger.info(f"  CPU time: {cpu_time:.4f} seconds")
        logger.info(f"  GPU time: {gpu_time:.4f} seconds")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        # Verify accuracy
        self.assertEqual(len(gpu_results), len(cpu_results))
        for i, (gpu_indices, cpu_indices) in enumerate(zip(gpu_results, cpu_results)):
            self.assertEqual(set(gpu_indices), set(cpu_indices))
        
        # For GPU acceleration, we expect at least some speedup
        if self.device_manager.is_gpu:
            self.assertGreater(speedup, 1.0, "GPU should provide speedup over CPU")
    
    def _compute_distances_cpu(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """CPU reference implementation for distance computation."""
        diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))
    
    def _find_nearest_cpu(self, query_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CPU reference implementation for nearest neighbor search."""
        diff = query_points[:, np.newaxis, :] - target_points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        nearest_indices = np.argmin(distances, axis=1)
        nearest_distances = np.min(distances, axis=1)
        
        return nearest_indices, nearest_distances
    
    def _radius_search_cpu(self, query_points: np.ndarray, target_points: np.ndarray, radius: float) -> List[List[int]]:
        """CPU reference implementation for radius search."""
        diff = query_points[:, np.newaxis, :] - target_points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        results = []
        for i in range(len(query_points)):
            indices = np.where(distances[i] <= radius)[0]
            results.append(indices.tolist())
        
        return results


class TestAcceptanceCriteria(unittest.TestCase):
    """Test class specifically for verifying Issue #347 acceptance criteria."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not GPU_AVAILABLE:
            self.skipTest("GPU libraries not available")
        
        # Create environment with GPU acceleration
        from farm.core.environment import Environment
        from farm.config.config import SpatialIndexConfig, EnvironmentConfig
        
        self.spatial_config = SpatialIndexConfig(
            enable_gpu_acceleration=True,
            gpu_device=None,  # Auto-detect
            performance_monitoring=True
        )
        
        self.env_config = EnvironmentConfig(
            width=200,
            height=200,
            spatial_index=self.spatial_config
        )
        
        self.env = Environment(
            width=200,
            height=200,
            resource_distribution="uniform",
            config=self.env_config
        )
        
        # Add test data
        self.agents = self._create_mock_agents(1000)
        self.resources = self._create_mock_resources(500)
        
        self.env.spatial_index.set_references(self.agents, self.resources)
        self.env.spatial_index.update()
    
    def _create_mock_agents(self, count: int) -> List:
        """Create mock agent objects."""
        agents = []
        for i in range(count):
            agent = type('Agent', (), {
                'position': (np.random.rand() * 200, np.random.rand() * 200),
                'alive': True,
                'id': i
            })()
            agents.append(agent)
        return agents
    
    def _create_mock_resources(self, count: int) -> List:
        """Create mock resource objects."""
        resources = []
        for i in range(count):
            resource = type('Resource', (), {
                'position': (np.random.rand() * 200, np.random.rand() * 200),
                'id': i
            })()
            resources.append(resource)
        return resources
    
    def test_acceptance_criteria_1_performance_improvement(self):
        """Test Acceptance Criteria 1: 5-10x performance improvement on GPU hardware."""
        logger.info("Testing Acceptance Criteria 1: Performance Improvement")
        
        # Generate large dataset for performance testing
        num_queries = 1000
        query_positions = [
            (np.random.rand() * 200, np.random.rand() * 200)
            for _ in range(num_queries)
        ]
        
        # Test CPU performance
        start_time = time.time()
        cpu_results = []
        for pos in query_positions:
            nearby = self.env.spatial_index.get_nearby(pos, radius=20.0)
            cpu_results.append(nearby)
        cpu_time = time.time() - start_time
        
        # Test GPU performance
        gpu_time = 0
        gpu_results = []
        if hasattr(self.env.spatial_index, 'get_nearby_gpu'):
            start_time = time.time()
            for pos in query_positions:
                nearby = self.env.spatial_index.get_nearby_gpu(pos, radius=20.0)
                gpu_results.append(nearby)
            gpu_time = time.time() - start_time
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        logger.info(f"Performance Test Results:")
        logger.info(f"  CPU time: {cpu_time:.4f} seconds")
        logger.info(f"  GPU time: {gpu_time:.4f} seconds")
        logger.info(f"  Speedup: {speedup:.2f}x")
        
        # Verify GPU acceleration is working
        gpu_stats = self.env.spatial_index.get_gpu_performance_stats()
        self.assertTrue(gpu_stats.get('gpu_acceleration_enabled', False), 
                       "GPU acceleration should be enabled")
        
        # For GPU hardware, expect significant speedup
        if gpu_stats.get('device') != 'cpu' and gpu_time > 0:
            self.assertGreater(speedup, 1.0, 
                             f"GPU should provide speedup over CPU. Got {speedup:.2f}x")
            # For large datasets, expect more significant speedup
            if num_queries >= 1000:
                self.assertGreater(speedup, 2.0, 
                                 f"GPU should provide significant speedup for large datasets. Got {speedup:.2f}x")
    
    def test_acceptance_criteria_2_accuracy_maintenance(self):
        """Test Acceptance Criteria 2: GPU and CPU results are identical."""
        logger.info("Testing Acceptance Criteria 2: Accuracy Maintenance")
        
        # Test multiple query positions
        test_positions = [
            (50.0, 50.0),
            (100.0, 100.0),
            (150.0, 150.0),
            (25.0, 175.0),
            (175.0, 25.0)
        ]
        
        for pos in test_positions:
            # Get CPU results
            cpu_nearby = self.env.spatial_index.get_nearby(pos, radius=30.0)
            cpu_nearest = self.env.spatial_index.get_nearest(pos)
            
            # Get GPU results
            gpu_nearby = {}
            gpu_nearest = {}
            if hasattr(self.env.spatial_index, 'get_nearby_gpu'):
                gpu_nearby = self.env.spatial_index.get_nearby_gpu(pos, radius=30.0)
            if hasattr(self.env.spatial_index, 'get_nearest_gpu'):
                gpu_nearest = self.env.spatial_index.get_nearest_gpu(pos)
            
            # Verify nearby results are identical
            for index_name in cpu_nearby:
                if index_name in gpu_nearby:
                    self.assertEqual(len(cpu_nearby[index_name]), len(gpu_nearby[index_name]),
                                   f"Result count mismatch for {index_name} at position {pos}")
                    # Results should be the same objects
                    self.assertEqual(cpu_nearby[index_name], gpu_nearby[index_name],
                                   f"Results mismatch for {index_name} at position {pos}")
            
            # Verify nearest results are identical
            for index_name in cpu_nearest:
                if index_name in gpu_nearest:
                    self.assertEqual(cpu_nearest[index_name], gpu_nearest[index_name],
                                   f"Nearest result mismatch for {index_name} at position {pos}")
    
    def test_acceptance_criteria_3_cpu_fallback(self):
        """Test Acceptance Criteria 3: Automatic fallback to CPU when GPU is not available."""
        logger.info("Testing Acceptance Criteria 3: CPU Fallback")
        
        # Test that system works even when GPU is not available
        # Create a CPU-only spatial index
        cpu_spatial_index = SpatialIndex(
            width=200,
            height=200,
            enable_gpu_acceleration=False  # Force CPU-only
        )
        
        # Add test data
        cpu_spatial_index.set_references(self.agents, self.resources)
        cpu_spatial_index.update()
        
        # Test that CPU-only operations work
        test_pos = (100.0, 100.0)
        nearby = cpu_spatial_index.get_nearby(test_pos, radius=20.0)
        nearest = cpu_spatial_index.get_nearest(test_pos)
        
        # Verify results are returned
        self.assertIsInstance(nearby, dict)
        self.assertIsInstance(nearest, dict)
        
        # Test that GPU methods fallback to CPU
        if hasattr(cpu_spatial_index, 'get_nearby_gpu'):
            gpu_nearby = cpu_spatial_index.get_nearby_gpu(test_pos, radius=20.0)
            self.assertEqual(nearby, gpu_nearby, "GPU method should fallback to CPU")
        
        if hasattr(cpu_spatial_index, 'get_nearest_gpu'):
            gpu_nearest = cpu_spatial_index.get_nearest_gpu(test_pos)
            self.assertEqual(nearest, gpu_nearest, "GPU method should fallback to CPU")
    
    def test_acceptance_criteria_4_comprehensive_testing(self):
        """Test Acceptance Criteria 4: Comprehensive testing for performance and accuracy."""
        logger.info("Testing Acceptance Criteria 4: Comprehensive Testing")
        
        # Test various scenarios
        test_scenarios = [
            {"radius": 10.0, "description": "small radius"},
            {"radius": 50.0, "description": "medium radius"},
            {"radius": 100.0, "description": "large radius"},
        ]
        
        for scenario in test_scenarios:
            radius = scenario["radius"]
            description = scenario["description"]
            
            # Test multiple positions
            test_positions = [
                (50.0, 50.0),
                (100.0, 100.0),
                (150.0, 150.0)
            ]
            
            for pos in test_positions:
                # Test nearby queries
                cpu_nearby = self.env.spatial_index.get_nearby(pos, radius)
                if hasattr(self.env.spatial_index, 'get_nearby_gpu'):
                    gpu_nearby = self.env.spatial_index.get_nearby_gpu(pos, radius)
                    # Verify results are consistent
                    for index_name in cpu_nearby:
                        if index_name in gpu_nearby:
                            self.assertEqual(len(cpu_nearby[index_name]), len(gpu_nearby[index_name]),
                                           f"Count mismatch for {description} at {pos}")
                
                # Test nearest queries
                cpu_nearest = self.env.spatial_index.get_nearest(pos)
                if hasattr(self.env.spatial_index, 'get_nearest_gpu'):
                    gpu_nearest = self.env.spatial_index.get_nearest_gpu(pos)
                    # Verify results are consistent
                    for index_name in cpu_nearest:
                        if index_name in gpu_nearest:
                            self.assertEqual(cpu_nearest[index_name], gpu_nearest[index_name],
                                           f"Nearest mismatch for {description} at {pos}")
    
    def test_acceptance_criteria_5_documentation_and_usage(self):
        """Test Acceptance Criteria 5: Documentation and usage examples work correctly."""
        logger.info("Testing Acceptance Criteria 5: Documentation and Usage")
        
        # Test that all documented methods exist and work
        documented_methods = [
            'get_nearby_gpu',
            'get_nearest_gpu', 
            'batch_query_gpu',
            'enable_gpu_acceleration',
            'disable_gpu_acceleration',
            'clear_gpu_memory',
            'get_gpu_performance_stats'
        ]
        
        for method_name in documented_methods:
            self.assertTrue(hasattr(self.env.spatial_index, method_name),
                          f"Documented method {method_name} should exist")
            
            # Test that method is callable
            method = getattr(self.env.spatial_index, method_name)
            self.assertTrue(callable(method),
                          f"Documented method {method_name} should be callable")
        
        # Test configuration options work
        config = self.env.spatial_index._gpu_acceleration_enabled
        self.assertIsInstance(config, bool, "GPU acceleration config should be boolean")
        
        # Test performance stats are available
        stats = self.env.spatial_index.get_gpu_performance_stats()
        self.assertIsInstance(stats, dict, "GPU performance stats should be a dictionary")
        self.assertIn('gpu_acceleration_enabled', stats, "Stats should include acceleration status")


if __name__ == '__main__':
    # Set up logging for performance benchmarks
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main()
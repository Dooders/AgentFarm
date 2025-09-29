"""GPU-accelerated spatial computations for AgentFarm.

This module provides GPU-accelerated spatial operations using CuPy and PyTorch,
with automatic fallback to CPU operations when GPU is not available or suitable.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# Try to import GPU libraries with fallback
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    logger.info("CuPy available for GPU acceleration")
except ImportError:
    cp = None
    CUPY_AVAILABLE = False
    logger.warning("CuPy not available, falling back to CPU operations")

try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
    if TORCH_CUDA_AVAILABLE:
        logger.info("PyTorch CUDA available for GPU acceleration")
    else:
        logger.info("PyTorch available but CUDA not available")
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False
    logger.warning("PyTorch not available, falling back to CPU operations")


class GPUDeviceManager:
    """Manages GPU device selection and memory management."""
    
    def __init__(self, preferred_device: Optional[str] = None):
        """Initialize GPU device manager.
        
        Parameters
        ----------
        preferred_device : str, optional
            Preferred device ('cuda', 'cupy', 'cpu'). If None, auto-detects best available.
        """
        self.preferred_device = preferred_device
        self._device = None
        self._memory_pool = None
        self._initialize_device()
    
    def _initialize_device(self) -> None:
        """Initialize the best available GPU device."""
        if self.preferred_device == 'cpu':
            self._device = 'cpu'
            logger.info("Using CPU for spatial computations")
            return
        
        # Try CuPy first (better for NumPy-like operations)
        if CUPY_AVAILABLE and self.preferred_device in (None, 'cupy'):
            try:
                # Test CuPy functionality
                test_array = cp.array([1, 2, 3])
                _ = cp.linalg.norm(test_array)
                self._device = 'cupy'
                self._memory_pool = cp.get_default_memory_pool()
                logger.info("Using CuPy for GPU-accelerated spatial computations")
                return
            except Exception as e:
                logger.warning(f"CuPy initialization failed: {e}")
        
        # Try PyTorch CUDA
        if TORCH_CUDA_AVAILABLE and self.preferred_device in (None, 'cuda'):
            try:
                # Test PyTorch CUDA functionality
                test_tensor = torch.tensor([1, 2, 3], device='cuda')
                _ = torch.norm(test_tensor)
                self._device = 'cuda'
                logger.info("Using PyTorch CUDA for GPU-accelerated spatial computations")
                return
            except Exception as e:
                logger.warning(f"PyTorch CUDA initialization failed: {e}")
        
        # Fallback to CPU
        self._device = 'cpu'
        logger.info("Falling back to CPU for spatial computations")
    
    @property
    def device(self) -> str:
        """Get the current device type."""
        return self._device
    
    @property
    def is_gpu(self) -> bool:
        """Check if using GPU acceleration."""
        return self._device in ('cupy', 'cuda')
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information for the current device."""
        if self._device == 'cupy' and self._memory_pool:
            return {
                'device': 'cupy',
                'used_bytes': self._memory_pool.used_bytes(),
                'total_bytes': self._memory_pool.total_bytes(),
                'free_bytes': self._memory_pool.free_bytes(),
            }
        elif self._device == 'cuda' and TORCH_CUDA_AVAILABLE:
            return {
                'device': 'cuda',
                'allocated_memory': torch.cuda.memory_allocated(),
                'cached_memory': torch.cuda.memory_reserved(),
                'max_memory': torch.cuda.max_memory_allocated(),
            }
        else:
            return {'device': 'cpu'}
    
    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if self._device == 'cupy' and self._memory_pool:
            self._memory_pool.free_all_blocks()
        elif self._device == 'cuda' and TORCH_CUDA_AVAILABLE:
            torch.cuda.empty_cache()


class GPUSpatialOperations:
    """GPU-accelerated spatial operations with CPU fallback."""
    
    def __init__(self, device_manager: Optional[GPUDeviceManager] = None):
        """Initialize GPU spatial operations.
        
        Parameters
        ----------
        device_manager : GPUDeviceManager, optional
            Device manager instance. If None, creates a new one.
        """
        self.device_manager = device_manager or GPUDeviceManager()
        self._performance_stats = {
            'gpu_operations': 0,
            'cpu_fallbacks': 0,
            'total_gpu_time': 0.0,
            'total_cpu_time': 0.0,
        }
    
    def to_gpu_array(self, array: np.ndarray) -> Union[np.ndarray, 'cp.ndarray', 'torch.Tensor']:
        """Convert NumPy array to GPU array based on current device."""
        if self.device_manager.device == 'cupy':
            return cp.asarray(array)
        elif self.device_manager.device == 'cuda':
            return torch.from_numpy(array).cuda()
        else:
            return array
    
    def to_cpu_array(self, array: Union[np.ndarray, 'cp.ndarray', 'torch.Tensor']) -> np.ndarray:
        """Convert GPU array back to NumPy array."""
        if hasattr(array, 'get'):  # CuPy array
            return array.get()
        elif hasattr(array, 'cpu'):  # PyTorch tensor
            return array.cpu().numpy()
        else:
            return array
    
    def compute_distances_gpu(self, points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between two sets of points using GPU.
        
        Parameters
        ----------
        points1 : np.ndarray
            First set of points, shape (n, 2)
        points2 : np.ndarray
            Second set of points, shape (m, 2)
            
        Returns
        -------
        np.ndarray
            Distance matrix, shape (n, m)
        """
        start_time = time.time()
        
        try:
            if self.device_manager.device == 'cupy':
                # CuPy implementation
                gpu_points1 = cp.asarray(points1)
                gpu_points2 = cp.asarray(points2)
                
                # Compute pairwise distances using broadcasting
                diff = gpu_points1[:, np.newaxis, :] - gpu_points2[np.newaxis, :, :]
                distances = cp.sqrt(cp.sum(diff ** 2, axis=2))
                
                result = cp.asnumpy(distances)
                self._performance_stats['gpu_operations'] += 1
                self._performance_stats['total_gpu_time'] += time.time() - start_time
                return result
                
            elif self.device_manager.device == 'cuda':
                # PyTorch implementation
                gpu_points1 = torch.from_numpy(points1).cuda()
                gpu_points2 = torch.from_numpy(points2).cuda()
                
                # Compute pairwise distances using broadcasting
                diff = gpu_points1.unsqueeze(1) - gpu_points2.unsqueeze(0)
                distances = torch.norm(diff, dim=2)
                
                result = distances.cpu().numpy()
                self._performance_stats['gpu_operations'] += 1
                self._performance_stats['total_gpu_time'] += time.time() - start_time
                return result
                
            else:
                # CPU fallback
                return self._compute_distances_cpu(points1, points2, start_time)
                
        except Exception as e:
            logger.warning(f"GPU distance computation failed: {e}, falling back to CPU")
            return self._compute_distances_cpu(points1, points2, start_time)
    
    def _compute_distances_cpu(self, points1: np.ndarray, points2: np.ndarray, start_time: float) -> np.ndarray:
        """CPU fallback for distance computation."""
        diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        self._performance_stats['cpu_fallbacks'] += 1
        self._performance_stats['total_cpu_time'] += time.time() - start_time
        return distances
    
    def find_nearest_gpu(self, query_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest neighbors using GPU acceleration.
        
        Parameters
        ----------
        query_points : np.ndarray
            Query points, shape (n, 2)
        target_points : np.ndarray
            Target points to search in, shape (m, 2)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (indices, distances) for nearest neighbors
        """
        start_time = time.time()
        
        try:
            if self.device_manager.device == 'cupy':
                # CuPy implementation
                gpu_query = cp.asarray(query_points)
                gpu_target = cp.asarray(target_points)
                
                # Compute all pairwise distances
                diff = gpu_query[:, np.newaxis, :] - gpu_target[np.newaxis, :, :]
                distances = cp.sqrt(cp.sum(diff ** 2, axis=2))
                
                # Find nearest neighbors
                nearest_indices = cp.argmin(distances, axis=1)
                nearest_distances = cp.min(distances, axis=1)
                
                indices = cp.asnumpy(nearest_indices)
                dists = cp.asnumpy(nearest_distances)
                
                self._performance_stats['gpu_operations'] += 1
                self._performance_stats['total_gpu_time'] += time.time() - start_time
                return indices, dists
                
            elif self.device_manager.device == 'cuda':
                # PyTorch implementation
                gpu_query = torch.from_numpy(query_points).cuda()
                gpu_target = torch.from_numpy(target_points).cuda()
                
                # Compute all pairwise distances
                diff = gpu_query.unsqueeze(1) - gpu_target.unsqueeze(0)
                distances = torch.norm(diff, dim=2)
                
                # Find nearest neighbors
                nearest_indices = torch.argmin(distances, dim=1)
                nearest_distances = torch.min(distances, dim=1)[0]
                
                indices = nearest_indices.cpu().numpy()
                dists = nearest_distances.cpu().numpy()
                
                self._performance_stats['gpu_operations'] += 1
                self._performance_stats['total_gpu_time'] += time.time() - start_time
                return indices, dists
                
            else:
                # CPU fallback
                return self._find_nearest_cpu(query_points, target_points, start_time)
                
        except Exception as e:
            logger.warning(f"GPU nearest neighbor search failed: {e}, falling back to CPU")
            return self._find_nearest_cpu(query_points, target_points, start_time)
    
    def _find_nearest_cpu(self, query_points: np.ndarray, target_points: np.ndarray, start_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback for nearest neighbor search."""
        diff = query_points[:, np.newaxis, :] - target_points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        nearest_indices = np.argmin(distances, axis=1)
        nearest_distances = np.min(distances, axis=1)
        
        self._performance_stats['cpu_fallbacks'] += 1
        self._performance_stats['total_cpu_time'] += time.time() - start_time
        return nearest_indices, nearest_distances
    
    def radius_search_gpu(self, query_points: np.ndarray, target_points: np.ndarray, radius: float) -> List[List[int]]:
        """Find all points within radius using GPU acceleration.
        
        Parameters
        ----------
        query_points : np.ndarray
            Query points, shape (n, 2)
        target_points : np.ndarray
            Target points to search in, shape (m, 2)
        radius : float
            Search radius
            
        Returns
        -------
        List[List[int]]
            List of lists, where each inner list contains indices of points within radius
        """
        start_time = time.time()
        
        try:
            if self.device_manager.device == 'cupy':
                # CuPy implementation
                gpu_query = cp.asarray(query_points)
                gpu_target = cp.asarray(target_points)
                
                # Compute all pairwise distances
                diff = gpu_query[:, np.newaxis, :] - gpu_target[np.newaxis, :, :]
                distances = cp.sqrt(cp.sum(diff ** 2, axis=2))
                
                # Find points within radius
                within_radius = distances <= radius
                results = []
                for i in range(len(query_points)):
                    indices = cp.where(within_radius[i])[0]
                    results.append(cp.asnumpy(indices).tolist())
                
                self._performance_stats['gpu_operations'] += 1
                self._performance_stats['total_gpu_time'] += time.time() - start_time
                return results
                
            elif self.device_manager.device == 'cuda':
                # PyTorch implementation
                gpu_query = torch.from_numpy(query_points).cuda()
                gpu_target = torch.from_numpy(target_points).cuda()
                
                # Compute all pairwise distances
                diff = gpu_query.unsqueeze(1) - gpu_target.unsqueeze(0)
                distances = torch.norm(diff, dim=2)
                
                # Find points within radius
                within_radius = distances <= radius
                results = []
                for i in range(len(query_points)):
                    indices = torch.where(within_radius[i])[0]
                    results.append(indices.cpu().numpy().tolist())
                
                self._performance_stats['gpu_operations'] += 1
                self._performance_stats['total_gpu_time'] += time.time() - start_time
                return results
                
            else:
                # CPU fallback
                return self._radius_search_cpu(query_points, target_points, radius, start_time)
                
        except Exception as e:
            logger.warning(f"GPU radius search failed: {e}, falling back to CPU")
            return self._radius_search_cpu(query_points, target_points, radius, start_time)
    
    def _radius_search_cpu(self, query_points: np.ndarray, target_points: np.ndarray, radius: float, start_time: float) -> List[List[int]]:
        """CPU fallback for radius search."""
        diff = query_points[:, np.newaxis, :] - target_points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        results = []
        for i in range(len(query_points)):
            indices = np.where(distances[i] <= radius)[0]
            results.append(indices.tolist())
        
        self._performance_stats['cpu_fallbacks'] += 1
        self._performance_stats['total_cpu_time'] += time.time() - start_time
        return results
    
    def batch_update_positions_gpu(self, old_positions: np.ndarray, new_positions: np.ndarray) -> np.ndarray:
        """Batch update positions using GPU acceleration.
        
        Parameters
        ----------
        old_positions : np.ndarray
            Old positions, shape (n, 2)
        new_positions : np.ndarray
            New positions, shape (n, 2)
            
        Returns
        -------
        np.ndarray
            Updated positions
        """
        start_time = time.time()
        
        try:
            if self.device_manager.device == 'cupy':
                # CuPy implementation
                gpu_old = cp.asarray(old_positions)
                gpu_new = cp.asarray(new_positions)
                
                # Simple position update (could be more complex in real scenarios)
                updated_positions = gpu_new  # For now, just return new positions
                
                result = cp.asnumpy(updated_positions)
                self._performance_stats['gpu_operations'] += 1
                self._performance_stats['total_gpu_time'] += time.time() - start_time
                return result
                
            elif self.device_manager.device == 'cuda':
                # PyTorch implementation
                gpu_old = torch.from_numpy(old_positions).cuda()
                gpu_new = torch.from_numpy(new_positions).cuda()
                
                # Simple position update
                updated_positions = gpu_new
                
                result = updated_positions.cpu().numpy()
                self._performance_stats['gpu_operations'] += 1
                self._performance_stats['total_gpu_time'] += time.time() - start_time
                return result
                
            else:
                # CPU fallback
                self._performance_stats['cpu_fallbacks'] += 1
                self._performance_stats['total_cpu_time'] += time.time() - start_time
                return new_positions
                
        except Exception as e:
            logger.warning(f"GPU batch position update failed: {e}, falling back to CPU")
            self._performance_stats['cpu_fallbacks'] += 1
            self._performance_stats['total_cpu_time'] += time.time() - start_time
            return new_positions
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for GPU operations."""
        stats = dict(self._performance_stats)
        stats['device'] = self.device_manager.device
        stats['is_gpu'] = self.device_manager.is_gpu
        stats['memory_info'] = self.device_manager.get_memory_info()
        
        # Calculate speedup ratio
        if stats['total_cpu_time'] > 0 and stats['total_gpu_time'] > 0:
            stats['gpu_speedup'] = stats['total_cpu_time'] / stats['total_gpu_time']
        else:
            stats['gpu_speedup'] = 0.0
        
        return stats
    
    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        self.device_manager.clear_memory()


class GPUAcceleratedKDTree:
    """GPU-accelerated KD-tree implementation with CPU fallback."""
    
    def __init__(self, points: np.ndarray, device_manager: Optional[GPUDeviceManager] = None):
        """Initialize GPU-accelerated KD-tree.
        
        Parameters
        ----------
        points : np.ndarray
            Points to build KD-tree from, shape (n, 2)
        device_manager : GPUDeviceManager, optional
            Device manager instance
        """
        self.points = points
        self.device_manager = device_manager or GPUDeviceManager()
        self.gpu_ops = GPUSpatialOperations(self.device_manager)
        
        # Build CPU KD-tree as fallback
        self.cpu_kdtree = cKDTree(points)
        
        # Store GPU arrays if using GPU
        if self.device_manager.is_gpu:
            self.gpu_points = self.gpu_ops.to_gpu_array(points)
        else:
            self.gpu_points = None
    
    def query(self, query_points: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Query k nearest neighbors.
        
        Parameters
        ----------
        query_points : np.ndarray
            Query points, shape (m, 2)
        k : int
            Number of nearest neighbors to find
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple of (distances, indices)
        """
        if k == 1 and self.device_manager.is_gpu:
            # Use GPU for single nearest neighbor
            indices, distances = self.gpu_ops.find_nearest_gpu(query_points, self.points)
            return distances.reshape(-1, 1), indices.reshape(-1, 1)
        else:
            # Use CPU KD-tree for k > 1 or when GPU not available
            return self.cpu_kdtree.query(query_points, k=k)
    
    def query_ball_point(self, query_points: np.ndarray, radius: float) -> List[List[int]]:
        """Query all points within radius.
        
        Parameters
        ----------
        query_points : np.ndarray
            Query points, shape (m, 2)
        radius : float
            Search radius
            
        Returns
        -------
        List[List[int]]
            List of lists containing indices of points within radius
        """
        if self.device_manager.is_gpu:
            # Use GPU radius search
            return self.gpu_ops.radius_search_gpu(query_points, self.points, radius)
        else:
            # Use CPU KD-tree
            return self.cpu_kdtree.query_ball_point(query_points, radius)
    
    def update_points(self, new_points: np.ndarray) -> None:
        """Update the points in the KD-tree.
        
        Parameters
        ----------
        new_points : np.ndarray
            New points, shape (n, 2)
        """
        self.points = new_points
        
        # Update CPU KD-tree
        self.cpu_kdtree = cKDTree(new_points)
        
        # Update GPU arrays if using GPU
        if self.device_manager.is_gpu:
            self.gpu_points = self.gpu_ops.to_gpu_array(new_points)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the GPU-accelerated KD-tree."""
        stats = {
            'num_points': len(self.points),
            'device': self.device_manager.device,
            'is_gpu': self.device_manager.is_gpu,
        }
        stats.update(self.gpu_ops.get_performance_stats())
        return stats
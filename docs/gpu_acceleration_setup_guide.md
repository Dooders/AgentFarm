# GPU Acceleration Setup Guide

This guide provides step-by-step instructions for setting up GPU acceleration for spatial computations in AgentFarm.

## Overview

AgentFarm supports GPU acceleration for spatial computations, providing 5-10x performance improvements for large-scale simulations. The system automatically detects the best available GPU device and falls back to CPU operations when GPU is not available.

## System Requirements

### Hardware Requirements
- **CUDA-capable GPU** (NVIDIA GPU with compute capability 3.5 or higher)
- **Minimum 4GB GPU memory** (8GB+ recommended for large simulations)
- **Sufficient system RAM** (16GB+ recommended)

### Software Requirements
- **CUDA Toolkit** (version 10.x, 11.x, or 12.x)
- **Python 3.8+**
- **Compatible GPU drivers**

## Installation Steps

### 1. Install CUDA Toolkit

#### Ubuntu/Debian
```bash
# Download and install CUDA Toolkit 12.x (recommended)
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Windows
1. Download CUDA Toolkit from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
2. Run the installer and follow the setup wizard
3. Restart your system after installation

#### macOS
CUDA is not officially supported on macOS. Use CPU fallback mode.

### 2. Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Check GPU availability
nvidia-smi
```

### 3. Install AgentFarm with GPU Dependencies

```bash
# Clone the repository
git clone https://github.com/Dooders/AgentFarm.git
cd AgentFarm

# Install with GPU acceleration dependencies
pip install -r requirements.txt

# For CUDA 12.x (recommended)
pip install cupy-cuda12x>=12.0.0

# For CUDA 11.x
# pip install cupy-cuda11x>=11.0.0

# For CUDA 10.x
# pip install cupy-cuda10x>=10.0.0
```

### 4. Verify GPU Acceleration

```python
# Test GPU acceleration
python examples/gpu_acceleration_demo.py
```

## Configuration

### Basic Configuration

```python
from farm.config.config import SpatialIndexConfig, EnvironmentConfig
from farm.core.environment import Environment

# Enable GPU acceleration
spatial_config = SpatialIndexConfig(
    enable_gpu_acceleration=True,
    gpu_device=None,  # Auto-detect best device
    gpu_memory_pool_size=1024**3,  # 1GB GPU memory pool
    gpu_fallback_to_cpu=True,  # Fallback to CPU if GPU fails
    gpu_performance_threshold=1.5  # Minimum speedup to use GPU
)

env_config = EnvironmentConfig(
    width=200,
    height=200,
    spatial_index=spatial_config
)

# Create environment with GPU acceleration
env = Environment(
    width=200,
    height=200,
    resource_distribution="uniform",
    config=env_config
)
```

### Advanced Configuration

```python
# Force specific GPU device
spatial_config = SpatialIndexConfig(
    enable_gpu_acceleration=True,
    gpu_device='cupy',  # Force CuPy
    # gpu_device='cuda',  # Force PyTorch CUDA
    # gpu_device='cpu',   # Force CPU
)

# Configure memory management
spatial_config = SpatialIndexConfig(
    enable_gpu_acceleration=True,
    gpu_memory_pool_size=2048**3,  # 2GB GPU memory pool
    gpu_performance_threshold=2.0,  # Require 2x speedup
)
```

## Usage Examples

### Basic GPU-Accelerated Queries

```python
# GPU-accelerated nearby queries
nearby_agents = env.spatial_index.get_nearby_gpu(
    agent.position, 
    radius=20.0, 
    ["agents"]
)

# GPU-accelerated nearest neighbor queries
nearest_resource = env.spatial_index.get_nearest_gpu(
    agent.position, 
    ["resources"]
)

# Batch GPU queries (most efficient for multiple queries)
query_positions = np.array([[x1, y1], [x2, y2], [x3, y3]])
batch_results = env.spatial_index.batch_query_gpu(
    query_positions, 
    radius=15.0
)
```

### Performance Monitoring

```python
# Get GPU performance statistics
gpu_stats = env.spatial_index.get_gpu_performance_stats()
print(f"GPU Device: {gpu_stats['device']}")
print(f"GPU Operations: {gpu_stats['gpu_operations']}")
print(f"CPU Fallbacks: {gpu_stats['cpu_fallbacks']}")
print(f"GPU Speedup: {gpu_stats['gpu_speedup']:.2f}x")

# Monitor memory usage
memory_info = gpu_stats['memory_info']
print(f"GPU Memory Used: {memory_info.get('used_bytes', 0) / 1024**2:.1f} MB")
```

### Dynamic GPU Management

```python
# Enable GPU acceleration at runtime
success = env.spatial_index.enable_gpu_acceleration('cupy')
if success:
    print("GPU acceleration enabled successfully")

# Disable GPU acceleration
env.spatial_index.disable_gpu_acceleration()

# Clear GPU memory
env.spatial_index.clear_gpu_memory()
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```python
# Reduce GPU memory pool size
spatial_config = SpatialIndexConfig(
    gpu_memory_pool_size=512**3,  # 512MB instead of 1GB
)

# Clear GPU memory periodically
env.spatial_index.clear_gpu_memory()
```

#### 2. GPU Not Detected
```python
# Check GPU availability
import cupy as cp
print(f"CuPy available: {cp.cuda.is_available()}")

# Force CPU mode
spatial_config = SpatialIndexConfig(
    enable_gpu_acceleration=False
)
```

#### 3. Performance Issues
```python
# Check if GPU is actually being used
gpu_stats = env.spatial_index.get_gpu_performance_stats()
if gpu_stats['device'] == 'cpu':
    print("GPU acceleration not working, using CPU fallback")

# Verify CUDA installation
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create environment with debug mode
spatial_config = SpatialIndexConfig(
    enable_gpu_acceleration=True,
    debug_queries=True  # Enable query debugging
)
```

## Performance Optimization

### Best Practices

1. **Use Batch Operations**: Batch queries are significantly faster than individual queries
2. **Optimize Memory Usage**: Clear GPU memory periodically for long-running simulations
3. **Choose Appropriate Device**: CuPy is generally faster for spatial operations than PyTorch CUDA
4. **Monitor Performance**: Use performance statistics to optimize your configuration

### Performance Tuning

```python
# Optimize for your specific use case
spatial_config = SpatialIndexConfig(
    enable_gpu_acceleration=True,
    gpu_device='cupy',  # Usually fastest for spatial operations
    gpu_memory_pool_size=2048**3,  # Increase for large simulations
    gpu_performance_threshold=1.2,  # Lower threshold for smaller datasets
)
```

## Testing

### Run GPU Tests

```bash
# Run all GPU acceleration tests
python -m pytest tests/test_gpu_spatial_acceleration.py -v

# Run specific acceptance criteria tests
python -m pytest tests/test_gpu_spatial_acceleration.py::TestAcceptanceCriteria -v

# Run performance benchmarks
python examples/gpu_acceleration_demo.py
```

### Verify Installation

```python
# Quick verification script
from farm.core.gpu_spatial import GPUDeviceManager, GPUSpatialOperations
import numpy as np

# Test GPU device detection
device_manager = GPUDeviceManager()
print(f"Detected device: {device_manager.device}")

# Test GPU operations
gpu_ops = GPUSpatialOperations(device_manager)
points1 = np.random.rand(100, 2)
points2 = np.random.rand(50, 2)

# Test distance computation
distances = gpu_ops.compute_distances_gpu(points1, points2)
print(f"GPU distance computation successful: {distances.shape}")

# Get performance stats
stats = gpu_ops.get_performance_stats()
print(f"GPU performance stats: {stats}")
```

## Support

### Getting Help

1. **Check the logs**: Enable debug logging to see detailed GPU initialization messages
2. **Verify CUDA installation**: Ensure CUDA toolkit and drivers are properly installed
3. **Test with CPU fallback**: Disable GPU acceleration to verify the system works
4. **Check GPU memory**: Ensure sufficient GPU memory is available

### Common Error Messages

- `"CuPy not available"`: Install CuPy with the correct CUDA version
- `"CUDA out of memory"`: Reduce GPU memory pool size or clear memory
- `"GPU initialization failed"`: Check CUDA installation and GPU drivers
- `"No GPU devices found"`: Verify GPU is properly installed and recognized

For additional support, please refer to the [main documentation](spatial_indexing.md) or create an issue on the GitHub repository.
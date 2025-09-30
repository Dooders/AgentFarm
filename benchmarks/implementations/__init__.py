"""
Benchmark implementations package.

Avoid importing specific benchmarks here to prevent importing heavy optional
dependencies when not needed. Benchmarks are imported lazily by the CLI.
"""

from ..utils.spatial_visualization import SpatialBenchmarkVisualizer
from .memory_db_benchmark import MemoryDBBenchmark
from .observation_flow_benchmark import ObservationFlowBenchmark
from .perception_metrics_benchmark import PerceptionMetricsBenchmark
from .pragma_profile_benchmark import PragmaProfileBenchmark
from .redis_memory_benchmark import RedisMemoryBenchmark
from .spatial.comprehensive_spatial_benchmark import SpatialBenchmark
from .spatial.spatial_memory_profiler import SpatialMemoryBenchmark
from .spatial.spatial_performance_analyzer import SpatialPerformanceAnalyzer

__all__ = [
    "MemoryDBBenchmark",
    "PragmaProfileBenchmark",
    "RedisMemoryBenchmark",
    "ObservationFlowBenchmark",
    "PerceptionMetricsBenchmark",
    "SpatialBenchmark",
    "SpatialMemoryBenchmark",
    "SpatialPerformanceAnalyzer",
    "SpatialBenchmarkVisualizer",
]

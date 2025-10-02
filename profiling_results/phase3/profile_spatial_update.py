#!/usr/bin/env python3
"""Line profile spatial index updates"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from farm.core.spatial import SpatialIndex

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

# Patch spatial index update
original_update = SpatialIndex.update

@profile
def profiled_update(self):
    return original_update(self)

SpatialIndex.update = profiled_update

def main():
    """Profile spatial index updates."""
    # Create test entities
    class MockAgent:
        def __init__(self, agent_id, position):
            self.agent_id = agent_id
            self.position = position
            self.alive = True
    
    agents = [MockAgent(f"agent_{i}", (i*10, i*10)) for i in range(100)]
    
    # Create spatial index
    spatial_index = SpatialIndex(1000, 1000, enable_batch_updates=False)
    spatial_index.set_references(agents, [])
    
    print("Profiling spatial index updates...")
    # Profile multiple updates with position changes
    for iteration in range(10):
        # Move some agents
        for i in range(len(agents) // 2):
            agents[i].position = (i*10 + iteration, i*10 + iteration)
        
        spatial_index.mark_positions_dirty()
        spatial_index.update()
    
    print("Profiling complete!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quadtree Spatial Indexing Demo

This script demonstrates the Quadtree spatial indexing functionality
added to AgentFarm. It shows how Quadtrees provide efficient range
queries and dynamic updates compared to KD-trees.

Usage:
    python scripts/quadtree_demo.py
"""

import time

import numpy as np

from farm.core.spatial import SpatialIndex


class MockAgent:
    """Simple mock agent for demonstration."""

    def __init__(self, agent_id, position):
        self.agent_id = agent_id
        self.position = position
        self.alive = True


def demo_basic_quadtree():
    """Demonstrate basic Quadtree operations."""
    print("Basic Quadtree Operations Demo")
    print("=" * 50)

    # Create spatial index
    spatial_index = SpatialIndex(width=100, height=100)

    # Create some test agents
    agents = []
    for i in range(20):
        position = (np.random.uniform(0, 100), np.random.uniform(0, 100))
        agent = MockAgent(f"agent_{i}", position)
        agents.append(agent)

    # Register Quadtree index
    spatial_index.register_index(
        name="demo_quadtree",
        data_reference=agents,
        position_getter=lambda a: a.position,
        index_type="quadtree",
    )

    # Build the index
    spatial_index.update()

    print(f"Created Quadtree with {len(agents)} agents")

    # Test range query
    query_bounds = (25, 25, 50, 50)  # 50x50 rectangle centered at (50,50)
    results = spatial_index.get_nearby_range(query_bounds, ["demo_quadtree"])
    entities_in_range = results["demo_quadtree"]

    print(f"Range query {query_bounds}: found {len(entities_in_range)} agents")

    # Show Quadtree stats
    stats = spatial_index.get_quadtree_stats("demo_quadtree")
    print(f"Quadtree stats: {stats}")

    print()


def demo_performance_comparison():
    """Compare KD-tree vs Quadtree performance."""
    print("Performance Comparison: KD-tree vs Quadtree")
    print("=" * 50)

    # Create test data
    num_agents = 1000
    agents = []
    for i in range(num_agents):
        position = (np.random.uniform(0, 100), np.random.uniform(0, 100))
        agent = MockAgent(f"perf_agent_{i}", position)
        agents.append(agent)

    # Create spatial index
    spatial_index = SpatialIndex(width=100, height=100)

    # Register both index types
    spatial_index.register_index(
        name="perf_kdtree",
        data_reference=agents,
        position_getter=lambda a: a.position,
        index_type="kdtree",
    )

    spatial_index.register_index(
        name="perf_quadtree",
        data_reference=agents,
        position_getter=lambda a: a.position,
        index_type="quadtree",
    )

    spatial_index.update()

    # Test radial queries (KD-tree advantage)
    print("Testing radial queries (KD-tree optimized)...")
    radial_positions = [
        (np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(50)
    ]

    # KD-tree radial queries
    start_time = time.time()
    for pos in radial_positions:
        spatial_index.get_nearby(pos, 10, ["perf_kdtree"])
    kdtree_radial_time = time.time() - start_time

    # Quadtree radial queries
    start_time = time.time()
    for pos in radial_positions:
        spatial_index.get_nearby(pos, 10, ["perf_quadtree"])
    quadtree_radial_time = time.time() - start_time

    print(f"KD-tree radial queries: {kdtree_radial_time:.3f} seconds")
    print(f"Quadtree radial queries: {quadtree_radial_time:.3f} seconds")

    # Test rectangular queries (Quadtree advantage)
    print("\nTesting rectangular range queries (Quadtree optimized)...")
    rect_queries = [
        (np.random.uniform(0, 80), np.random.uniform(0, 80), 20, 20) for _ in range(50)
    ]

    # KD-tree rectangular queries
    start_time = time.time()
    for bounds in rect_queries:
        spatial_index.get_nearby_range(bounds, ["perf_kdtree"])
    kdtree_rect_time = time.time() - start_time

    # Quadtree rectangular queries
    start_time = time.time()
    for bounds in rect_queries:
        spatial_index.get_nearby_range(bounds, ["perf_quadtree"])
    quadtree_rect_time = time.time() - start_time

    print(f"KD-tree rectangular query time: {kdtree_rect_time:.3f} s")
    print(f"Quadtree rectangular query time: {quadtree_rect_time:.3f} s")

    # Calculate speedups
    if quadtree_rect_time < kdtree_rect_time:
        rect_speedup = kdtree_rect_time / quadtree_rect_time
        print(f"Rectangular query speedup (KD-tree / Quadtree): {rect_speedup:.1f}x")

    print()


def demo_dynamic_updates():
    """Demonstrate dynamic position updates."""
    print("Dynamic Position Updates Demo")
    print("=" * 50)

    # Create spatial index with Quadtree
    spatial_index = SpatialIndex(width=100, height=100)

    # Create a moving agent
    agent = MockAgent("moving_agent", (50, 50))
    spatial_index.register_index(
        name="moving_quadtree",
        data_reference=[agent],
        position_getter=lambda a: a.position,
        index_type="quadtree",
    )
    spatial_index.update()

    print("Agent starting at position (50, 50)")

    # Simulate movement
    for i in range(5):
        old_pos = agent.position
        new_pos = (
            old_pos[0] + np.random.uniform(-5, 5),
            old_pos[1] + np.random.uniform(-5, 5),
        )
        new_pos = (max(0, min(100, new_pos[0])), max(0, min(100, new_pos[1])))

        # Update position efficiently
        agent.position = new_pos
        spatial_index.update_entity_position(agent, old_pos, new_pos)

        # Verify agent is still findable at new position
        nearby = spatial_index.get_nearby(new_pos, 1, ["moving_quadtree"])
        found = len(nearby["moving_quadtree"]) > 0

        print(f"Move {i+1}: {old_pos} -> {new_pos} | Found: {found}")

    print()


def demo_quadtree_statistics():
    """Demonstrate Quadtree statistics and structure."""
    print("Statistics and Structure Demo")
    print("=" * 50)

    # Create spatial index with many agents
    spatial_index = SpatialIndex(width=100, height=100)

    # Create agents in a grid pattern to show subdivision
    agents = []
    for x in range(0, 100, 10):
        for y in range(0, 100, 10):
            agent = MockAgent(f"grid_agent_{x}_{y}", (x, y))
            agents.append(agent)

    spatial_index.register_index(
        name="stats_quadtree",
        data_reference=agents,
        position_getter=lambda a: a.position,
        index_type="quadtree",
    )
    spatial_index.update()

    # Get detailed statistics
    stats = spatial_index.get_quadtree_stats("stats_quadtree")
    print("Quadtree Statistics:")
    print(f"  Total entities: {stats['total_entities']}")
    print(f"  Local entities (root): {stats['local_entities']}")
    print(f"  Is divided: {stats['is_divided']}")
    print(f"  Children count: {stats['children_count']}")
    print(f"  Bounds: {stats['bounds']}")

    print()


def main():
    """Run all demonstrations."""
    print("AgentFarm Quadtree Spatial Indexing Demo")
    print("=" * 60)
    print()

    try:
        demo_basic_quadtree()
        demo_performance_comparison()
        demo_dynamic_updates()
        demo_quadtree_statistics()

        print("All demonstrations completed successfully!")
        print("\nKey Takeaways:")
        print("- Quadtrees excel at rectangular range queries")
        print("- KD-trees excel at radial and nearest-neighbor queries")
        print("- Quadtrees provide hierarchical spatial subdivision")
        print("- Both indexing strategies work together for optimal performance")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

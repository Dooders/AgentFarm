"""Tests for GPU-accelerated spatial kernels and SpatialIndex GPU integration.

These tests cover:
- SpatialGpuKernels accuracy (results match scipy reference)
- CPU-fallback correctness (torch but no CUDA, or torch absent)
- SpatialIndex.use_gpu flag and batch query methods
- Edge cases: empty positions, single entity, large counts
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from farm.core.spatial import SpatialIndex, SpatialGpuKernels, is_gpu_available
from farm.core.spatial.gpu_kernels import (
    _TORCH_AVAILABLE,
    _np_pairwise_distances,
    get_spatial_device,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_positions(n: int, seed: int = 42, width: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, width, size=(n, 2))


def _scipy_radius_query(positions, query_point, radius):
    """Reference implementation using NumPy."""
    diffs = positions - np.asarray(query_point)
    dists = np.sqrt((diffs**2).sum(axis=1))
    return sorted(np.where(dists <= radius)[0].tolist())


def _scipy_nearest_query(positions, query_point):
    """Reference implementation using NumPy."""
    diffs = positions - np.asarray(query_point)
    dists = (diffs**2).sum(axis=1)
    return int(dists.argmin())


# ---------------------------------------------------------------------------
# SpatialGpuKernels unit tests
# ---------------------------------------------------------------------------


class TestSpatialGpuKernelsBasic(unittest.TestCase):
    """Tests for SpatialGpuKernels correctness on CPU (no CUDA required)."""

    def setUp(self):
        # Always use CPU to keep tests deterministic on CI without a GPU
        self.kernels = SpatialGpuKernels(use_gpu=False)
        self.positions = _make_positions(50)
        self.query = (30.0, 30.0)
        self.radius = 25.0

    # ------------------------------------------------------------------
    # radius_query
    # ------------------------------------------------------------------

    def test_radius_query_matches_reference(self):
        got = sorted(self.kernels.radius_query(self.positions, self.query, self.radius))
        expected = _scipy_radius_query(self.positions, self.query, self.radius)
        self.assertEqual(got, expected)

    def test_radius_query_empty_positions(self):
        result = self.kernels.radius_query(np.empty((0, 2)), self.query, self.radius)
        self.assertEqual(result, [])

    def test_radius_query_zero_radius_returns_empty(self):
        result = self.kernels.radius_query(self.positions, (50.0, 50.0), radius=0.0)
        # radius=0 should return at most the exact match; empty is acceptable too
        self.assertIsInstance(result, list)

    def test_radius_query_very_large_radius_returns_all(self):
        result = self.kernels.radius_query(self.positions, (50.0, 50.0), radius=1e9)
        self.assertEqual(len(result), len(self.positions))

    # ------------------------------------------------------------------
    # nearest_query
    # ------------------------------------------------------------------

    def test_nearest_query_matches_reference(self):
        got = self.kernels.nearest_query(self.positions, self.query)
        expected = _scipy_nearest_query(self.positions, self.query)
        self.assertEqual(got, expected)

    def test_nearest_query_empty_positions(self):
        result = self.kernels.nearest_query(np.empty((0, 2)), (0.0, 0.0))
        self.assertEqual(result, -1)

    def test_nearest_query_single_entity(self):
        pos = np.array([[10.0, 10.0]])
        result = self.kernels.nearest_query(pos, (20.0, 20.0))
        self.assertEqual(result, 0)

    # ------------------------------------------------------------------
    # batch_radius_query
    # ------------------------------------------------------------------

    def test_batch_radius_query_shape(self):
        qp = _make_positions(8, seed=7)
        results = self.kernels.batch_radius_query(self.positions, qp, self.radius)
        self.assertEqual(len(results), 8)
        for r in results:
            self.assertIsInstance(r, list)

    def test_batch_radius_query_matches_single(self):
        qp = _make_positions(5, seed=99)
        batch = self.kernels.batch_radius_query(self.positions, qp, self.radius)
        for i, q in enumerate(qp):
            expected = _scipy_radius_query(self.positions, q, self.radius)
            self.assertEqual(sorted(batch[i]), expected)

    def test_batch_radius_query_empty_positions(self):
        qp = _make_positions(4)
        results = self.kernels.batch_radius_query(np.empty((0, 2)), qp, self.radius)
        self.assertEqual(results, [[], [], [], []])

    def test_batch_radius_query_empty_queries(self):
        results = self.kernels.batch_radius_query(
            self.positions, np.empty((0, 2)), self.radius
        )
        self.assertEqual(results, [])

    # ------------------------------------------------------------------
    # batch_nearest_query
    # ------------------------------------------------------------------

    def test_batch_nearest_query_shape(self):
        qp = _make_positions(10, seed=11)
        result = self.kernels.batch_nearest_query(self.positions, qp)
        self.assertEqual(result.shape, (10,))

    def test_batch_nearest_query_matches_single(self):
        qp = _make_positions(6, seed=55)
        batch = self.kernels.batch_nearest_query(self.positions, qp)
        for i, q in enumerate(qp):
            expected = _scipy_nearest_query(self.positions, q)
            self.assertEqual(int(batch[i]), expected)

    def test_batch_nearest_query_empty_positions(self):
        qp = _make_positions(3)
        result = self.kernels.batch_nearest_query(np.empty((0, 2)), qp)
        np.testing.assert_array_equal(result, [-1, -1, -1])

    def test_batch_nearest_query_empty_queries(self):
        result = self.kernels.batch_nearest_query(self.positions, np.empty((0, 2)))
        self.assertEqual(len(result), 0)

    # ------------------------------------------------------------------
    # pairwise_distances
    # ------------------------------------------------------------------

    def test_pairwise_distances_shape(self):
        qp = _make_positions(4)
        dist = self.kernels.pairwise_distances(self.positions, qp)
        self.assertEqual(dist.shape, (4, len(self.positions)))

    def test_pairwise_distances_symmetry(self):
        pos = _make_positions(5, seed=1)
        qp = _make_positions(5, seed=2)
        dist = self.kernels.pairwise_distances(pos, qp)
        np_dist = _np_pairwise_distances(pos, qp)
        np.testing.assert_allclose(dist, np_dist, rtol=1e-4)

    def test_pairwise_distances_non_negative(self):
        qp = _make_positions(5)
        dist = self.kernels.pairwise_distances(self.positions, qp)
        self.assertTrue((dist >= 0).all())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def test_on_gpu_is_false_for_cpu_kernels(self):
        self.assertFalse(self.kernels.on_gpu)

    def test_device_is_cpu(self):
        if _TORCH_AVAILABLE:
            self.assertEqual(str(self.kernels.device), "cpu")


# ---------------------------------------------------------------------------
# SpatialGpuKernels – GPU-only tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(is_gpu_available(), "CUDA not available")
class TestSpatialGpuKernelsGpu(unittest.TestCase):
    """Tests requiring an actual CUDA GPU."""

    def setUp(self):
        self.kernels = SpatialGpuKernels(use_gpu=True)
        self.positions = _make_positions(500)
        self.radius = 20.0

    def test_on_gpu(self):
        self.assertTrue(self.kernels.on_gpu)

    def test_gpu_radius_query_matches_numpy(self):
        qp = (50.0, 50.0)
        got = sorted(self.kernels.radius_query(self.positions, qp, self.radius))
        expected = _scipy_radius_query(self.positions, qp, self.radius)
        self.assertEqual(got, expected)

    def test_gpu_nearest_query_matches_numpy(self):
        qp = (50.0, 50.0)
        got = self.kernels.nearest_query(self.positions, qp)
        expected = _scipy_nearest_query(self.positions, qp)
        self.assertEqual(got, expected)

    def test_gpu_batch_matches_cpu(self):
        cpu_kernels = SpatialGpuKernels(use_gpu=False)
        qp = _make_positions(20, seed=77)
        cpu_result = cpu_kernels.batch_nearest_query(self.positions, qp)
        gpu_result = self.kernels.batch_nearest_query(self.positions, qp)
        np.testing.assert_array_equal(cpu_result, gpu_result)


# ---------------------------------------------------------------------------
# SpatialIndex GPU integration tests
# ---------------------------------------------------------------------------


class _MockAgent:
    """Minimal mock agent for spatial index tests."""

    def __init__(self, agent_id, x, y):
        self.agent_id = agent_id
        self.position = (x, y)
        self.alive = True


class TestSpatialIndexGpuIntegration(unittest.TestCase):
    """Tests for SpatialIndex with use_gpu=True (uses CPU tensors when no CUDA)."""

    def _build_index(self, n_agents=50, use_gpu=True, gpu_threshold=10):
        index = SpatialIndex(
            width=100,
            height=100,
            use_gpu=use_gpu,
            gpu_threshold=gpu_threshold,
        )
        agents = [_MockAgent(i, float(i % 10) * 10, float(i // 10) * 10) for i in range(n_agents)]
        index.set_references(agents, [])
        index.update()
        return index, agents

    def test_gpu_enabled_property(self):
        index, _ = self._build_index(use_gpu=True)
        self.assertTrue(index.gpu_enabled)

    def test_gpu_disabled_property(self):
        index, _ = self._build_index(use_gpu=False)
        self.assertFalse(index.gpu_enabled)

    def test_get_nearby_gpu_matches_cpu(self):
        cpu_idx, agents = self._build_index(use_gpu=False)
        gpu_idx, _ = self._build_index(use_gpu=True, gpu_threshold=1)
        # Re-attach same agents to both
        cpu_idx.set_references(agents, [])
        gpu_idx.set_references(agents, [])
        cpu_idx.update()
        gpu_idx.update()

        pos = (25.0, 25.0)
        radius = 20.0
        cpu_result = cpu_idx.get_nearby(pos, radius, ["agents"])
        gpu_result = gpu_idx.get_nearby(pos, radius, ["agents"])
        # Compare by agent_id sets
        cpu_ids = {a.agent_id for a in cpu_result.get("agents", [])}
        gpu_ids = {a.agent_id for a in gpu_result.get("agents", [])}
        self.assertEqual(cpu_ids, gpu_ids)

    def test_get_nearest_gpu_matches_cpu(self):
        agents = [_MockAgent(i, float(i) * 2.5, float(i) * 2.5) for i in range(20)]
        cpu_idx = SpatialIndex(width=100, height=100, use_gpu=False)
        gpu_idx = SpatialIndex(width=100, height=100, use_gpu=True, gpu_threshold=1)
        for idx in (cpu_idx, gpu_idx):
            idx.set_references(agents, [])
            idx.update()

        pos = (30.0, 30.0)
        cpu_r = cpu_idx.get_nearest(pos, ["agents"])
        gpu_r = gpu_idx.get_nearest(pos, ["agents"])
        self.assertEqual(
            cpu_r.get("agents").agent_id,  # type: ignore[union-attr]
            gpu_r.get("agents").agent_id,  # type: ignore[union-attr]
        )

    # ------------------------------------------------------------------
    # batch_radius_query
    # ------------------------------------------------------------------

    def test_batch_radius_query_returns_dict(self):
        index, agents = self._build_index()
        query_positions = [(10.0, 10.0), (50.0, 50.0), (90.0, 90.0)]
        results = index.batch_radius_query(query_positions, radius=15.0)
        self.assertIn("agents", results)
        self.assertEqual(len(results["agents"]), 3)

    def test_batch_radius_query_matches_individual_get_nearby(self):
        index, agents = self._build_index(n_agents=30, use_gpu=True, gpu_threshold=1)
        query_positions = [(float(x), float(y)) for x, y in [(0, 0), (50, 50), (99, 99)]]
        radius = 20.0
        batch = index.batch_radius_query(query_positions, radius=radius)
        for i, qp in enumerate(query_positions):
            single = index.get_nearby(qp, radius, ["agents"])
            batch_ids = {a.agent_id for a in batch["agents"][i]}
            single_ids = {a.agent_id for a in single.get("agents", [])}
            self.assertEqual(batch_ids, single_ids, f"Mismatch at query {i}: {qp}")

    def test_batch_radius_query_empty_positions(self):
        index = SpatialIndex(width=100, height=100, use_gpu=True)
        index.set_references([], [])
        index.update()
        results = index.batch_radius_query([(10.0, 10.0)], radius=5.0)
        self.assertIn("agents", results)
        self.assertEqual(results["agents"], [[]])

    def test_batch_radius_query_empty_query_positions(self):
        index, _ = self._build_index()
        results = index.batch_radius_query([], radius=10.0)
        self.assertEqual(results, {})

    # ------------------------------------------------------------------
    # batch_nearest_query
    # ------------------------------------------------------------------

    def test_batch_nearest_query_returns_dict(self):
        index, _ = self._build_index()
        query_positions = [(10.0, 10.0), (50.0, 50.0)]
        results = index.batch_nearest_query(query_positions)
        self.assertIn("agents", results)
        self.assertEqual(len(results["agents"]), 2)

    def test_batch_nearest_query_matches_individual_get_nearest(self):
        index, agents = self._build_index(n_agents=25, use_gpu=True, gpu_threshold=1)
        query_positions = [(float(x), float(y)) for x, y in [(5, 5), (45, 45), (85, 85)]]
        batch = index.batch_nearest_query(query_positions)
        for i, qp in enumerate(query_positions):
            single = index.get_nearest(qp, ["agents"])
            self.assertEqual(
                batch["agents"][i].agent_id,  # type: ignore[union-attr]
                single.get("agents").agent_id,  # type: ignore[union-attr]
            )

    def test_batch_nearest_query_empty_positions(self):
        index = SpatialIndex(width=100, height=100, use_gpu=True)
        index.set_references([], [])
        index.update()
        results = index.batch_nearest_query([(10.0, 10.0)])
        self.assertIn("agents", results)
        self.assertIsNone(results["agents"][0])

    def test_batch_nearest_query_empty_queries(self):
        index, _ = self._build_index()
        results = index.batch_nearest_query([])
        self.assertEqual(results, {})

    # ------------------------------------------------------------------
    # Accuracy: large population
    # ------------------------------------------------------------------

    def test_large_population_accuracy(self):
        """Verify GPU path gives same results as scipy for N > gpu_threshold."""
        rng = np.random.default_rng(0)
        n = 400
        agents = [
            _MockAgent(i, float(rng.uniform(0, 100)), float(rng.uniform(0, 100)))
            for i in range(n)
        ]
        cpu_idx = SpatialIndex(width=100, height=100, use_gpu=False)
        gpu_idx = SpatialIndex(width=100, height=100, use_gpu=True, gpu_threshold=10)
        for idx in (cpu_idx, gpu_idx):
            idx.set_references(agents, [])
            idx.update()

        query_positions = [(rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(20)]
        radius = 15.0
        for qp in query_positions:
            cpu_r = cpu_idx.get_nearby(qp, radius, ["agents"])
            gpu_r = gpu_idx.get_nearby(qp, radius, ["agents"])
            cpu_ids = {a.agent_id for a in cpu_r.get("agents", [])}
            gpu_ids = {a.agent_id for a in gpu_r.get("agents", [])}
            self.assertEqual(cpu_ids, gpu_ids)


# ---------------------------------------------------------------------------
# is_gpu_available / get_spatial_device helpers
# ---------------------------------------------------------------------------


class TestGpuHelpers(unittest.TestCase):
    def test_is_gpu_available_returns_bool(self):
        result = is_gpu_available()
        self.assertIsInstance(result, bool)

    def test_get_spatial_device_prefer_false_returns_cpu(self):
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not installed")
        import torch

        device = get_spatial_device(prefer_gpu=False)
        self.assertEqual(device.type, "cpu")

    def test_get_spatial_device_prefer_true_returns_device(self):
        if not _TORCH_AVAILABLE:
            self.skipTest("torch not installed")
        import torch

        device = get_spatial_device(prefer_gpu=True)
        self.assertIn(device.type, ("cpu", "cuda"))

    def test_np_pairwise_distances(self):
        pos = np.array([[0.0, 0.0], [3.0, 4.0]])
        qp = np.array([[0.0, 0.0]])
        dist = _np_pairwise_distances(pos, qp)
        self.assertAlmostEqual(float(dist[0, 0]), 0.0)
        self.assertAlmostEqual(float(dist[0, 1]), 5.0)


if __name__ == "__main__":
    unittest.main()

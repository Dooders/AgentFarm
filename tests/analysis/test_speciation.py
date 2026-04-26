"""Tests for the speciation / niche-detection analysis module.

Covers:
- detect_clusters_gmm: trivial (empty), single cluster, known two-cluster split
- detect_clusters_dbscan: trivial, basic cluster detection
- match_clusters_greedy: stable IDs across snapshots, new cluster allocation
- compute_speciation_index: single cluster → 0, multi-cluster → silhouette
- compute_niche_correlation: per-cluster spatial/resource means
- GeneTrajectoryLogger: speciation_index in trajectory, cluster_lineage.jsonl output
- plot_chromosome_space_clusters: smoke test (no file I/O error)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from farm.analysis.speciation import (
    ClusterLineageRecord,
    ClusterResult,
    compute_niche_correlation,
    compute_speciation_index,
    detect_clusters_dbscan,
    detect_clusters_gmm,
    match_clusters_greedy,
    plot_chromosome_space_clusters,
)
from farm.analysis.common.context import AnalysisContext


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_chromosomes(values: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Return chromosome dicts as-is (convenience alias for readability)."""
    return list(values)


def _bimodal_chromosomes(n_per_cluster: int = 20) -> List[Dict[str, float]]:
    """Two well-separated clusters: low-LR exploiters + high-LR explorers."""
    import random
    rng = random.Random(42)
    low_cluster = [
        {"learning_rate": rng.gauss(0.01, 0.002), "gamma": rng.gauss(0.99, 0.005)}
        for _ in range(n_per_cluster)
    ]
    high_cluster = [
        {"learning_rate": rng.gauss(0.1, 0.002), "gamma": rng.gauss(0.5, 0.005)}
        for _ in range(n_per_cluster)
    ]
    return low_cluster + high_cluster


# ---------------------------------------------------------------------------
# detect_clusters_gmm
# ---------------------------------------------------------------------------


class TestDetectClustersGMM:
    def test_empty_input_returns_zero_clusters(self):
        result = detect_clusters_gmm([], max_k=3, seed=0)
        assert isinstance(result, ClusterResult)
        assert result.k == 0
        assert result.labels == []
        assert result.centroids == []
        assert result.algorithm == "gmm"

    def test_single_agent_single_cluster(self):
        chromosomes = [{"lr": 0.01, "gamma": 0.99}]
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        assert result.k == 1
        assert len(result.labels) == 1
        assert len(result.centroids) == 1
        assert result.silhouette_score == 0.0

    def test_all_identical_agents_single_cluster(self):
        chromosomes = [{"lr": 0.05, "gamma": 0.9}] * 10
        result = detect_clusters_gmm(chromosomes, max_k=5, seed=0)
        assert result.k == 1

    def test_two_well_separated_clusters_detected(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=30)
        result = detect_clusters_gmm(chromosomes, max_k=5, seed=42)
        assert result.k == 2
        assert len(result.labels) == 60
        assert result.silhouette_score > 0.5

    def test_labels_length_matches_input(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=15)
        result = detect_clusters_gmm(chromosomes, max_k=4, seed=0)
        assert len(result.labels) == len(chromosomes)

    def test_gene_names_populated(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        assert set(result.gene_names) == {"learning_rate", "gamma"}

    def test_bic_scores_populated_for_gmm(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        assert isinstance(result.bic_scores, dict)
        assert len(result.bic_scores) >= 1

    def test_deterministic_given_same_seed(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        r1 = detect_clusters_gmm(chromosomes, max_k=4, seed=7)
        r2 = detect_clusters_gmm(chromosomes, max_k=4, seed=7)
        assert r1.k == r2.k
        assert r1.labels == r2.labels

    def test_gene_names_override(self):
        chromosomes = [{"lr": 0.01, "gamma": 0.99, "noise": 999.0}] * 10
        result = detect_clusters_gmm(
            chromosomes, max_k=2, seed=0, gene_names=["lr", "gamma"]
        )
        assert result.gene_names == ["lr", "gamma"]


# ---------------------------------------------------------------------------
# detect_clusters_dbscan
# ---------------------------------------------------------------------------


class TestDetectClustersDBSCAN:
    def test_empty_input_returns_zero_clusters(self):
        result = detect_clusters_dbscan([], eps=0.1, min_samples=2)
        assert result.k == 0
        assert result.bic_scores is None

    def test_single_agent_returns_single_or_noise(self):
        chromosomes = [{"lr": 0.01}]
        result = detect_clusters_dbscan(chromosomes, eps=0.5, min_samples=1)
        # With min_samples=1 a single agent should form a cluster
        assert result.k >= 0

    def test_two_clusters_detected_with_appropriate_eps(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        # Learning_rate values: low cluster ~0.01, high cluster ~0.1  → gap ~0.09
        result = detect_clusters_dbscan(chromosomes, eps=0.03, min_samples=3)
        # Should find 2 clusters (noise-free for well-separated data)
        assert result.k >= 2

    def test_bic_scores_none_for_dbscan(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_dbscan(chromosomes)
        assert result.bic_scores is None

    def test_algorithm_field_is_dbscan(self):
        result = detect_clusters_dbscan([{"lr": 0.01}])
        assert result.algorithm == "dbscan"


# ---------------------------------------------------------------------------
# match_clusters_greedy
# ---------------------------------------------------------------------------


class TestMatchClustersGreedy:
    def _make_record(self, cluster_id: str, centroid: Dict[str, float], size: int = 10) -> ClusterLineageRecord:
        return ClusterLineageRecord(
            step=0,
            cluster_id=cluster_id,
            centroid=centroid,
            size=size,
            parent_cluster_id=None,
        )

    def test_empty_prev_allocates_new_ids(self):
        centroids = [{"lr": 0.01}, {"lr": 0.1}]
        sizes = [20, 20]
        records, next_ctr = match_clusters_greedy([], centroids, sizes, ["lr"], step=100)
        assert len(records) == 2
        assert all(r.parent_cluster_id is None for r in records)
        assert {r.cluster_id for r in records} == {"c0", "c1"}
        assert next_ctr == 2

    def test_stable_id_carries_over_when_close(self):
        prev = [self._make_record("c0", {"lr": 0.01})]
        centroids = [{"lr": 0.011}]  # very close to prev
        records, _ = match_clusters_greedy(prev, centroids, [18], ["lr"], step=200)
        assert records[0].cluster_id == "c0"
        assert records[0].parent_cluster_id == "c0"

    def test_new_id_when_no_close_predecessor(self):
        prev = [self._make_record("c0", {"lr": 0.01})]
        # New centroid far away, max_distance very small so no match
        centroids = [{"lr": 0.5}]
        records, ctr = match_clusters_greedy(
            prev, centroids, [15], ["lr"], step=200, max_distance=0.05
        )
        assert records[0].cluster_id != "c0"
        assert records[0].parent_cluster_id is None

    def test_split_cluster_produces_new_id(self):
        """One previous cluster splits into two → one inherits ID, one is new."""
        prev = [self._make_record("c0", {"lr": 0.05})]
        centroids = [{"lr": 0.01}, {"lr": 0.1}]  # two new clusters
        records, ctr = match_clusters_greedy(prev, centroids, [10, 10], ["lr"], step=200)
        cluster_ids = {r.cluster_id for r in records}
        parent_ids = {r.parent_cluster_id for r in records}
        # One should have inherited "c0", one should be fresh
        assert "c0" in cluster_ids
        assert None in parent_ids  # one is a new lineage

    def test_step_field_populated_correctly(self):
        centroids = [{"lr": 0.05}]
        records, _ = match_clusters_greedy([], centroids, [5], ["lr"], step=999)
        assert records[0].step == 999

    def test_known_fixture_split_across_snapshots(self):
        """Full persistence scenario: start with 1 cluster, split into 2."""
        # Snapshot 0: one cluster
        r0, ctr0 = match_clusters_greedy([], [{"lr": 0.05}], [30], ["lr"], step=0)
        assert r0[0].cluster_id == "c0"

        # Snapshot 100: split into two distinct clusters
        r1, _ctr1 = match_clusters_greedy(
            r0, [{"lr": 0.01}, {"lr": 0.1}], [15, 15], ["lr"], step=100,
            id_counter_start=ctr0,
        )
        ids_100 = {r.cluster_id for r in r1}
        # "c0" should match one of them (closest to old centroid 0.05);
        # the other should be a new "c1"
        assert "c0" in ids_100
        assert "c1" in ids_100


# ---------------------------------------------------------------------------
# compute_speciation_index
# ---------------------------------------------------------------------------


class TestComputeSpeciationIndex:
    def _make_result(self, k: int, sil: float) -> ClusterResult:
        return ClusterResult(
            algorithm="gmm",
            k=k,
            labels=[0] * 5,
            centroids=[{"lr": 0.01}],
            sizes=[5],
            gene_names=["lr"],
            silhouette_score=sil,
            bic_scores=None,
        )

    def test_single_cluster_returns_zero(self):
        result = self._make_result(k=1, sil=0.9)
        assert compute_speciation_index(result) == 0.0

    def test_zero_clusters_returns_zero(self):
        result = self._make_result(k=0, sil=0.0)
        assert compute_speciation_index(result) == 0.0

    def test_two_clusters_returns_silhouette(self):
        result = ClusterResult(
            algorithm="gmm", k=2, labels=[0, 1],
            centroids=[{"lr": 0.01}, {"lr": 0.1}],
            sizes=[1, 1], gene_names=["lr"],
            silhouette_score=0.75, bic_scores=None,
        )
        assert compute_speciation_index(result) == pytest.approx(0.75)

    def test_clamped_to_zero_when_negative_silhouette(self):
        result = ClusterResult(
            algorithm="dbscan", k=2, labels=[0, 1],
            centroids=[{"lr": 0.01}, {"lr": 0.02}],
            sizes=[1, 1], gene_names=["lr"],
            silhouette_score=-0.2, bic_scores=None,
        )
        assert compute_speciation_index(result) == 0.0

    def test_value_in_unit_interval_for_well_separated_bimodal(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=30)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        index = compute_speciation_index(result)
        assert 0.0 <= index <= 1.0


# ---------------------------------------------------------------------------
# compute_niche_correlation
# ---------------------------------------------------------------------------


class TestComputeNicheCorrelation:
    def _make_simple_result(self, labels: List[int]) -> ClusterResult:
        k = max(labels) + 1 if labels else 0
        return ClusterResult(
            algorithm="gmm", k=k, labels=labels,
            centroids=[{"lr": 0.01}] * k,
            sizes=[labels.count(i) for i in range(k)],
            gene_names=["lr"], silhouette_score=0.5, bic_scores=None,
        )

    def test_empty_agents_returns_empty(self):
        result = self._make_simple_result([])
        niche = compute_niche_correlation(result, [])
        assert niche == []

    def test_mean_x_y_computed_correctly(self):
        labels = [0, 0, 1, 1]
        agents = [
            {"x": 1.0, "y": 2.0, "energy": 10.0},
            {"x": 3.0, "y": 4.0, "energy": 20.0},
            {"x": 5.0, "y": 6.0, "energy": 30.0},
            {"x": 7.0, "y": 8.0, "energy": 40.0},
        ]
        result = self._make_simple_result(labels)
        niche = compute_niche_correlation(result, agents)
        assert len(niche) == 2

        by_id = {n["cluster_id"]: n for n in niche}
        assert by_id[0]["mean_x"] == pytest.approx(2.0)
        assert by_id[0]["mean_y"] == pytest.approx(3.0)
        assert by_id[0]["mean_energy"] == pytest.approx(15.0)
        assert by_id[1]["mean_x"] == pytest.approx(6.0)
        assert by_id[1]["mean_energy"] == pytest.approx(35.0)

    def test_missing_field_returns_none(self):
        labels = [0, 0]
        agents = [{"x": 1.0}, {"x": 2.0}]  # no 'y' field
        result = self._make_simple_result(labels)
        niche = compute_niche_correlation(result, agents)
        assert niche[0]["mean_y"] is None

    def test_noise_agents_excluded(self):
        """DBSCAN noise (label=-1) agents should not appear in any cluster."""
        labels = [-1, 0, 0, -1]
        agents = [
            {"x": 99.0},
            {"x": 1.0},
            {"x": 3.0},
            {"x": 99.0},
        ]
        result = ClusterResult(
            algorithm="dbscan", k=1, labels=labels,
            centroids=[{"lr": 0.01}],
            sizes=[2], gene_names=["lr"],
            silhouette_score=0.0, bic_scores=None,
        )
        niche = compute_niche_correlation(result, agents)
        # Only one cluster; mean_x should be average of agents at index 1,2
        assert len(niche) == 1
        assert niche[0]["mean_x"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# GeneTrajectoryLogger with speciation
# ---------------------------------------------------------------------------


def _make_fake_chromosome(lr: float = 0.01, gamma: float = 0.99):
    """Build a minimal HyperparameterChromosome-like object."""
    from farm.core.hyperparameter_chromosome import chromosome_from_values
    return chromosome_from_values({"learning_rate": lr, "gamma": gamma})


def _make_fake_agent(lr: float = 0.01, gamma: float = 0.99):
    chromosome = _make_fake_chromosome(lr, gamma)
    state_inner = SimpleNamespace(parent_ids=["seed"])
    return SimpleNamespace(
        agent_id=f"agent_{lr:.4f}",
        agent_type="system",
        generation=0,
        alive=True,
        hyperparameter_chromosome=chromosome,
        state=SimpleNamespace(_state=state_inner),
    )


class _FakeEnvironment:
    def __init__(self, agents):
        self._agents = list(agents)

    @property
    def alive_agent_objects(self):
        return list(self._agents)


class TestGeneTrajectoryLoggerSpeciation:
    def test_no_speciation_by_default(self, tmp_path):
        """Without enable_speciation, trajectory records have no speciation_index."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=1)
        agents = [_make_fake_agent(lr=0.01), _make_fake_agent(lr=0.1)]
        env = _FakeEnvironment(agents)
        logger.snapshot(env, step=0)
        logger.close()

        traj_path = tmp_path / "intrinsic_gene_trajectory.jsonl"
        records = [json.loads(line) for line in traj_path.read_text().splitlines()]
        assert len(records) == 1
        assert "speciation_index" not in records[0]

    def test_speciation_index_present_when_enabled(self, tmp_path):
        """With enable_speciation=True, trajectory records include speciation_index."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=1, enable_speciation=True)
        agents = [_make_fake_agent(lr=0.01), _make_fake_agent(lr=0.1)]
        env = _FakeEnvironment(agents)
        logger.snapshot(env, step=0)
        logger.close()

        traj_path = tmp_path / "intrinsic_gene_trajectory.jsonl"
        records = [json.loads(line) for line in traj_path.read_text().splitlines()]
        assert len(records) == 1
        assert "speciation_index" in records[0]
        assert 0.0 <= records[0]["speciation_index"] <= 1.0

    def test_cluster_lineage_written_when_speciation_enabled(self, tmp_path):
        """cluster_lineage.jsonl is created and populated with speciation enabled."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        # Create a clearly bimodal population
        agents = (
            [_make_fake_agent(lr=0.01, gamma=0.99)] * 10
            + [_make_fake_agent(lr=0.5, gamma=0.4)] * 10
        )
        env = _FakeEnvironment(agents)

        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=1, enable_speciation=True)
        logger.snapshot(env, step=0)
        logger.close()

        lineage_path = tmp_path / "cluster_lineage.jsonl"
        assert lineage_path.exists()
        rows = [json.loads(line) for line in lineage_path.read_text().splitlines()]
        assert len(rows) >= 1
        # Verify schema
        row = rows[0]
        assert "step" in row
        assert "cluster_id" in row
        assert "centroid" in row
        assert "size" in row
        assert "parent_cluster_id" in row

    def test_cluster_lineage_not_written_without_speciation(self, tmp_path):
        """cluster_lineage.jsonl should not be created when speciation is off."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        agents = [_make_fake_agent()]
        env = _FakeEnvironment(agents)
        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=1)
        logger.snapshot(env, step=0)
        logger.close()

        lineage_path = tmp_path / "cluster_lineage.jsonl"
        assert not lineage_path.exists()

    def test_speciation_index_stable_between_snapshot_steps(self, tmp_path):
        """Between snapshot steps the speciation_index should remain the same value."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        agents = (
            [_make_fake_agent(lr=0.01)] * 10
            + [_make_fake_agent(lr=0.5)] * 10
        )
        env = _FakeEnvironment(agents)

        # snapshot_interval=5 means cluster detection at step 0, 5, 10, ...
        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=5, enable_speciation=True)
        for step in range(6):  # steps 0..5
            logger.snapshot(env, step=step)
        logger.close()

        traj_path = tmp_path / "intrinsic_gene_trajectory.jsonl"
        records = [json.loads(line) for line in traj_path.read_text().splitlines()]
        assert len(records) == 6
        # steps 1..4 should have same speciation_index as step 0
        idx_0 = records[0]["speciation_index"]
        idx_5 = records[5]["speciation_index"]
        for rec in records[1:5]:
            assert rec["speciation_index"] == pytest.approx(idx_0)
        # step 5 is a new snapshot step → may have updated value (possibly same)
        assert 0.0 <= idx_5 <= 1.0

    def test_speciation_index_bimodal_population_nonzero(self, tmp_path):
        """A clearly bimodal population should produce a nonzero speciation index."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        # Build highly separated bimodal population
        agents = (
            [_make_fake_agent(lr=0.001, gamma=0.99)] * 20
            + [_make_fake_agent(lr=0.9, gamma=0.1)] * 20
        )
        env = _FakeEnvironment(agents)
        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=1, enable_speciation=True)
        logger.snapshot(env, step=0)
        logger.close()

        traj_path = tmp_path / "intrinsic_gene_trajectory.jsonl"
        rec = json.loads(traj_path.read_text().splitlines()[0])
        assert rec["speciation_index"] > 0.0

    def test_invalid_speciation_algorithm_raises(self):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        with pytest.raises(ValueError, match="speciation_algorithm"):
            GeneTrajectoryLogger(None, snapshot_interval=1, speciation_algorithm="bad")

    def test_speciation_index_reserved_key_in_extra_fields(self, tmp_path):
        """Passing speciation_index via extra_fields should raise ValueError when speciation enabled."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=1, enable_speciation=True)
        env = _FakeEnvironment([_make_fake_agent()])
        with pytest.raises(ValueError, match="speciation_index"):
            logger.snapshot(env, step=0, extra_fields={"speciation_index": 0.5})
        logger.close()

    def test_speciation_index_not_reserved_when_disabled(self, tmp_path):
        """When speciation is disabled, speciation_index can be used as a custom field."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=1, enable_speciation=False)
        env = _FakeEnvironment([_make_fake_agent()])
        # Should NOT raise – speciation_index is only reserved when enable_speciation=True
        logger.snapshot(env, step=0, extra_fields={"speciation_index": 0.42})
        logger.close()

        traj_path = tmp_path / "intrinsic_gene_trajectory.jsonl"
        rec = json.loads(traj_path.read_text().splitlines()[0])
        assert rec["speciation_index"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# plot_chromosome_space_clusters
# ---------------------------------------------------------------------------


class TestPlotChromosomeSpaceClusters:
    def _ctx(self, tmp_path) -> AnalysisContext:
        return AnalysisContext(output_path=Path(tmp_path))

    def test_smoke_two_gene_two_clusters(self, tmp_path):
        chromosomes = _bimodal_chromosomes(n_per_cluster=15)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        ctx = self._ctx(tmp_path)
        out = plot_chromosome_space_clusters(chromosomes, result, ctx, step=0)
        assert out is not None
        assert Path(out).exists()

    def test_empty_chromosomes_returns_none(self, tmp_path):
        result = ClusterResult(
            algorithm="gmm", k=0, labels=[], centroids=[],
            sizes=[], gene_names=[], silhouette_score=0.0, bic_scores=None,
        )
        ctx = self._ctx(tmp_path)
        out = plot_chromosome_space_clusters([], result, ctx)
        assert out is None

    def test_single_gene_strip_plot(self, tmp_path):
        chromosomes = [{"lr": v} for v in [0.01, 0.02, 0.1, 0.11]]
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        ctx = self._ctx(tmp_path)
        out = plot_chromosome_space_clusters(chromosomes, result, ctx, step=5)
        assert out is not None
        assert Path(out).exists()

    def test_three_genes_uses_pca(self, tmp_path):
        import random
        rng = random.Random(0)
        chromosomes = [
            {"lr": rng.random(), "gamma": rng.random(), "eps": rng.random()}
            for _ in range(20)
        ]
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        ctx = self._ctx(tmp_path)
        out = plot_chromosome_space_clusters(chromosomes, result, ctx)
        assert out is not None

    def test_custom_output_filename(self, tmp_path):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        ctx = self._ctx(tmp_path)
        out = plot_chromosome_space_clusters(
            chromosomes, result, ctx, output_filename="my_custom_plot.png"
        )
        assert out is not None
        assert "my_custom_plot.png" in str(out)

    def test_dbscan_result_plot(self, tmp_path):
        chromosomes = _bimodal_chromosomes(n_per_cluster=15)
        result = detect_clusters_dbscan(chromosomes, eps=0.04, min_samples=3)
        ctx = self._ctx(tmp_path)
        out = plot_chromosome_space_clusters(chromosomes, result, ctx)
        assert out is not None

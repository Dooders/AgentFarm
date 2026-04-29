"""Tests for the speciation / niche-detection analysis module.

Covers:
- detect_clusters_gmm: trivial (empty), single cluster, known two-cluster split
- detect_clusters_dbscan: trivial, basic cluster detection
- match_clusters_greedy: stable IDs across snapshots, new cluster allocation
- compute_speciation_index: single cluster → 0, multi-cluster → silhouette
- compute_speciation_quality_bundle: full quality bundle validation
- compute_niche_correlation: per-cluster spatial/resource means
- GeneTrajectoryLogger: speciation_index in trajectory, cluster_lineage.jsonl output
- plot_chromosome_space_clusters: smoke test (no file I/O error)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from farm.analysis.speciation import (
    ClusterLineageRecord,
    ClusterResult,
    SpeciationQualityBundle,
    VALID_TRANSITION_TYPES,
    compute_niche_correlation,
    compute_speciation_index,
    compute_speciation_quality_bundle,
    detect_clusters_dbscan,
    detect_clusters_gmm,
    match_clusters_greedy,
    match_clusters_hungarian,
    plot_chromosome_space_clusters,
    suggest_dbscan_params,
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
# match_clusters_hungarian
# ---------------------------------------------------------------------------


class TestMatchClustersHungarian:
    """Tests for the Hungarian-algorithm-based cluster matcher."""

    def _make_record(
        self, cluster_id: str, centroid: Dict[str, float], size: int = 10
    ) -> ClusterLineageRecord:
        return ClusterLineageRecord(
            step=0,
            cluster_id=cluster_id,
            centroid=centroid,
            size=size,
            parent_cluster_id=None,
        )

    # --- basic correctness ---------------------------------------------------

    def test_empty_prev_all_founding(self):
        """With no previous clusters every new cluster is a founding."""
        centroids = [{"lr": 0.01}, {"lr": 0.1}]
        records, next_ctr = match_clusters_hungarian([], centroids, [20, 20], ["lr"], step=0)
        assert len(records) == 2
        assert all(r.transition_type == "founding" for r in records)
        assert all(r.parent_cluster_id is None for r in records)
        assert all(r.parent_cluster_ids == [] for r in records)
        assert next_ctr == 2

    def test_stable_id_continuation(self):
        """A closely-matching 1:1 pair gets transition_type 'continuation'."""
        prev = [self._make_record("c0", {"lr": 0.01})]
        records, _ = match_clusters_hungarian(prev, [{"lr": 0.011}], [18], ["lr"], step=1)
        assert records[0].cluster_id == "c0"
        assert records[0].parent_cluster_id == "c0"
        assert records[0].transition_type == "continuation"
        assert records[0].parent_cluster_ids == ["c0"]

    def test_founding_when_beyond_max_distance(self):
        """New cluster exceeding max_distance gets a fresh founding ID."""
        prev = [self._make_record("c0", {"lr": 0.01})]
        records, _ = match_clusters_hungarian(
            prev, [{"lr": 0.5}], [15], ["lr"], step=1, max_distance=0.05
        )
        assert records[0].cluster_id != "c0"
        assert records[0].parent_cluster_id is None
        assert records[0].transition_type == "founding"
        assert records[0].parent_cluster_ids == []

    def test_step_field_populated_correctly(self):
        records, _ = match_clusters_hungarian([], [{"lr": 0.05}], [5], ["lr"], step=999)
        assert records[0].step == 999

    def test_id_counter_auto_advances(self):
        """Counter skips over IDs already used in prev_records."""
        prev = [
            self._make_record("c0", {"lr": 0.01}),
            self._make_record("c3", {"lr": 0.9}),  # non-contiguous
        ]
        records, ctr = match_clusters_hungarian(
            prev, [{"lr": 0.5}], [10], ["lr"], step=1, max_distance=0.05
        )
        assert records[0].cluster_id == "c4"  # skips c0..c3
        assert ctr == 5

    # --- split detection -----------------------------------------------------

    def test_split_transition_type_detected(self):
        """One prev cluster splits into two new ones → both get 'split'."""
        prev = [self._make_record("c0", {"lr": 0.05})]
        records, _ = match_clusters_hungarian(
            prev, [{"lr": 0.01}, {"lr": 0.09}], [10, 10], ["lr"], step=1
        )
        assert "split" in {r.transition_type for r in records}

    def test_split_inheritor_keeps_prev_id(self):
        """The Hungarian-matched split child inherits the predecessor ID."""
        prev = [self._make_record("c0", {"lr": 0.05})]
        records, _ = match_clusters_hungarian(
            prev, [{"lr": 0.04}, {"lr": 0.08}], [10, 10], ["lr"], step=1, max_distance=0.5
        )
        assert "c0" in {r.cluster_id for r in records}

    def test_split_fixture_across_snapshots(self):
        """Full scenario: single cluster at step 0, splits at step 1."""
        r0, ctr0 = match_clusters_hungarian([], [{"lr": 0.05}], [30], ["lr"], step=0)
        assert r0[0].cluster_id == "c0"
        assert r0[0].transition_type == "founding"

        r1, _ = match_clusters_hungarian(
            r0, [{"lr": 0.01}, {"lr": 0.09}], [15, 15], ["lr"], step=1,
            id_counter_start=ctr0,
        )
        ids = {rec.cluster_id for rec in r1}
        types = {rec.transition_type for rec in r1}
        assert "c0" in ids
        assert "split" in types

    # --- merge detection -----------------------------------------------------

    def test_merge_transition_type_detected(self):
        """Two prev clusters converging into one → transition 'merge'."""
        prev = [
            self._make_record("c0", {"lr": 0.01}),
            self._make_record("c1", {"lr": 0.02}),
        ]
        records, _ = match_clusters_hungarian(
            prev, [{"lr": 0.015}], [20], ["lr"], step=1, max_distance=1.0
        )
        assert len(records) == 1
        assert records[0].transition_type == "merge"

    def test_merge_parent_cluster_ids_contains_all_parents(self):
        """Merged cluster lists all contributing predecessors."""
        prev = [
            self._make_record("c0", {"lr": 0.01}),
            self._make_record("c1", {"lr": 0.02}),
        ]
        records, _ = match_clusters_hungarian(
            prev, [{"lr": 0.015}], [20], ["lr"], step=1, max_distance=1.0
        )
        assert set(records[0].parent_cluster_ids) == {"c0", "c1"}

    def test_merge_fixture_across_snapshots(self):
        """Full scenario: two clusters at step 0, merge at step 1."""
        r0, ctr0 = match_clusters_hungarian(
            [], [{"lr": 0.01}, {"lr": 0.09}], [10, 10], ["lr"], step=0
        )
        assert {r.cluster_id for r in r0} == {"c0", "c1"}

        r1, _ = match_clusters_hungarian(
            r0, [{"lr": 0.05}], [20], ["lr"], step=1,
            id_counter_start=ctr0, max_distance=1.0,
        )
        assert len(r1) == 1
        assert r1[0].transition_type == "merge"
        assert set(r1[0].parent_cluster_ids) == {"c0", "c1"}

    # --- global optimality ---------------------------------------------------

    def test_hungarian_global_optimum(self):
        """Skewed layout where Hungarian gives lower total cost than greedy.

        prev: c0 at 0.0, c1 at 5.0
        new:  A at 4.0,  B at 6.0

        Greedy (A first): A→c1 (dist 1), B→c0 (dist 6), total 7
        Hungarian:        A→c0 (dist 4), B→c1 (dist 1), total 5 ← better
        """
        prev = [
            self._make_record("c0", {"lr": 0.0}),
            self._make_record("c1", {"lr": 5.0}),
        ]
        records, _ = match_clusters_hungarian(
            prev, [{"lr": 4.0}, {"lr": 6.0}], [10, 10], ["lr"], step=1, max_distance=10.0
        )
        id_map = {r.centroid["lr"]: r.cluster_id for r in records}
        assert id_map[4.0] == "c0"
        assert id_map[6.0] == "c1"

    def test_invalid_max_distance_raises(self):
        prev = [self._make_record("c0", {"lr": 0.01})]
        with pytest.raises(ValueError, match="finite and > 0"):
            match_clusters_hungarian(
                prev, [{"lr": 0.011}], [18], ["lr"], step=1, max_distance=float("inf")
            )

    # --- determinism ---------------------------------------------------------

    def test_deterministic_on_repeated_calls(self):
        """Repeated calls with identical input give identical output."""
        prev = [
            self._make_record("c0", {"lr": 0.01, "gamma": 0.9}),
            self._make_record("c1", {"lr": 0.1, "gamma": 0.5}),
        ]
        centroids = [{"lr": 0.015, "gamma": 0.88}, {"lr": 0.09, "gamma": 0.52}]
        r1, ctr1 = match_clusters_hungarian(prev, centroids, [10, 10], ["lr", "gamma"], step=5)
        r2, ctr2 = match_clusters_hungarian(prev, centroids, [10, 10], ["lr", "gamma"], step=5)
        assert ctr1 == ctr2
        assert [(r.cluster_id, r.transition_type) for r in r1] == [
            (r.cluster_id, r.transition_type) for r in r2
        ]

    # --- transition type validity --------------------------------------------

    def test_all_transition_types_are_valid(self):
        """All emitted transition_type values are in VALID_TRANSITION_TYPES."""
        prev = [
            self._make_record("c0", {"lr": 0.05}),
            self._make_record("c1", {"lr": 0.5}),
        ]
        centroids = [{"lr": 0.01}, {"lr": 0.1}, {"lr": 0.49}]
        records, _ = match_clusters_hungarian(
            prev, centroids, [10, 10, 10], ["lr"], step=1, max_distance=1.0
        )
        for r in records:
            assert r.transition_type in VALID_TRANSITION_TYPES


# ---------------------------------------------------------------------------
# ClusterLineageRecord new fields (backward compatibility)
# ---------------------------------------------------------------------------


class TestClusterLineageRecordNewFields:
    """Backward-compatible new fields on ClusterLineageRecord."""

    def test_transition_type_defaults_to_none(self):
        rec = ClusterLineageRecord(
            step=0, cluster_id="c0", centroid={"lr": 0.01}, size=5,
            parent_cluster_id=None,
        )
        assert rec.transition_type is None

    def test_parent_cluster_ids_defaults_to_empty_list(self):
        rec = ClusterLineageRecord(
            step=0, cluster_id="c0", centroid={"lr": 0.01}, size=5,
            parent_cluster_id=None,
        )
        assert rec.parent_cluster_ids == []

    def test_greedy_records_have_none_transition_type(self):
        """match_clusters_greedy leaves transition_type as None."""
        records, _ = match_clusters_greedy([], [{"lr": 0.01}], [10], ["lr"], step=0)
        assert records[0].transition_type is None
        assert records[0].parent_cluster_ids == []

    def test_hungarian_records_have_valid_transition_type(self):
        records, _ = match_clusters_hungarian([], [{"lr": 0.01}], [10], ["lr"], step=0)
        assert records[0].transition_type == "founding"
        assert records[0].parent_cluster_ids == []


# ---------------------------------------------------------------------------
# cluster_lineage.jsonl – new transition fields
# ---------------------------------------------------------------------------


class TestClusterLineageJsonlTransitionFields:
    """Verify transition_type and parent_cluster_ids appear in cluster_lineage.jsonl."""

    def test_cluster_lineage_contains_transition_fields(self, tmp_path):
        """cluster_lineage.jsonl rows include transition_type and parent_cluster_ids."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        agents = (
            [_make_fake_agent(lr=0.001, gamma=0.99)] * 10
            + [_make_fake_agent(lr=0.9, gamma=0.1)] * 10
        )
        env = _FakeEnvironment(agents)

        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=1, enable_speciation=True)
        logger.snapshot(env, step=0)
        logger.close()

        lineage_path = tmp_path / "cluster_lineage.jsonl"
        rows = [json.loads(line) for line in lineage_path.read_text().splitlines()]
        assert len(rows) >= 1
        for row in rows:
            assert "transition_type" in row
            assert "parent_cluster_ids" in row
            assert isinstance(row["parent_cluster_ids"], list)
            assert row["transition_type"] in VALID_TRANSITION_TYPES
            assert row["lineage_matcher"] == "hungarian"
            assert row["lineage_max_distance"] == pytest.approx(1.0)
            # Backward-compatible fields still present
            assert "parent_cluster_id" in row
            assert "step" in row
            assert "cluster_id" in row


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
# compute_speciation_quality_bundle
# ---------------------------------------------------------------------------


class TestComputeSpeciationQualityBundle:
    """Tests for the richer quality-metrics bundle."""

    def _make_result(
        self,
        k: int,
        sil: float,
        labels: Optional[List[int]] = None,
        sizes: Optional[List[int]] = None,
        algorithm: str = "gmm",
    ) -> ClusterResult:
        if labels is None:
            labels = [0] * 5
        if sizes is None:
            sizes = [5] if k >= 1 else []
        return ClusterResult(
            algorithm=algorithm,
            k=k,
            labels=labels,
            centroids=[{"lr": 0.01}] * k,
            sizes=sizes,
            gene_names=["lr"],
            silhouette_score=sil,
            bic_scores=None,
        )

    # --- return type ---

    def test_returns_speciation_quality_bundle_instance(self):
        result = self._make_result(k=1, sil=0.5)
        bundle = compute_speciation_quality_bundle(result)
        assert isinstance(bundle, SpeciationQualityBundle)

    def test_bundle_has_expected_fields(self):
        result = self._make_result(k=1, sil=0.5)
        bundle = compute_speciation_quality_bundle(result)
        for attr in ("speciation_index", "raw_silhouette", "noise_fraction",
                     "cluster_size_entropy", "n_clusters"):
            assert hasattr(bundle, attr)

    # --- speciation_index consistency ---

    def test_speciation_index_matches_scalar_function(self):
        result = self._make_result(k=2, sil=0.75, labels=[0, 1], sizes=[1, 1])
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.speciation_index == pytest.approx(compute_speciation_index(result))

    def test_speciation_index_zero_for_single_cluster(self):
        result = self._make_result(k=1, sil=0.9)
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.speciation_index == 0.0

    def test_speciation_index_in_unit_interval(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=30)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        bundle = compute_speciation_quality_bundle(result)
        assert 0.0 <= bundle.speciation_index <= 1.0

    # --- raw_silhouette ---

    def test_raw_silhouette_zero_for_single_cluster(self):
        result = self._make_result(k=1, sil=0.9)
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.raw_silhouette == 0.0

    def test_raw_silhouette_preserves_negative_value(self):
        """Negative silhouette must NOT be clipped to 0."""
        result = self._make_result(k=2, sil=-0.3, labels=[0, 1], sizes=[1, 1])
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.raw_silhouette == pytest.approx(-0.3)

    def test_raw_silhouette_in_minus1_to_1_range_for_bimodal(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=30)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        bundle = compute_speciation_quality_bundle(result)
        assert -1.0 <= bundle.raw_silhouette <= 1.0

    def test_raw_silhouette_positive_for_well_separated_clusters(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=30)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.raw_silhouette > 0.0

    # --- noise_fraction ---

    def test_noise_fraction_zero_for_gmm(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.noise_fraction == pytest.approx(0.0)

    def test_noise_fraction_computed_correctly(self):
        """2 noise, 2 labelled → fraction = 0.5."""
        result = ClusterResult(
            algorithm="dbscan",
            k=1,
            labels=[-1, 0, 0, -1],
            centroids=[{"lr": 0.01}],
            sizes=[2],
            gene_names=["lr"],
            silhouette_score=0.0,
            bic_scores=None,
        )
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.noise_fraction == pytest.approx(0.5)

    def test_noise_fraction_zero_when_no_noise(self):
        result = ClusterResult(
            algorithm="dbscan", k=2, labels=[0, 0, 1, 1],
            centroids=[{"lr": 0.01}, {"lr": 0.1}],
            sizes=[2, 2], gene_names=["lr"],
            silhouette_score=0.8, bic_scores=None,
        )
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.noise_fraction == pytest.approx(0.0)

    def test_noise_fraction_in_unit_interval(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        result = detect_clusters_dbscan(chromosomes, eps=0.04, min_samples=3)
        bundle = compute_speciation_quality_bundle(result)
        assert 0.0 <= bundle.noise_fraction <= 1.0

    # --- cluster_size_entropy ---

    def test_entropy_zero_for_single_cluster(self):
        result = self._make_result(k=1, sil=0.5)
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.cluster_size_entropy == pytest.approx(0.0)

    def test_entropy_zero_for_no_clusters(self):
        result = self._make_result(k=0, sil=0.0, labels=[], sizes=[])
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.cluster_size_entropy == pytest.approx(0.0)

    def test_entropy_maximised_for_equal_sizes(self):
        """Two equal clusters → entropy = ln(2)."""
        import math as _math
        result = ClusterResult(
            algorithm="gmm", k=2, labels=[0, 0, 1, 1],
            centroids=[{"lr": 0.01}, {"lr": 0.1}],
            sizes=[2, 2], gene_names=["lr"],
            silhouette_score=0.8, bic_scores=None,
        )
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.cluster_size_entropy == pytest.approx(_math.log(2))

    def test_entropy_lower_for_imbalanced_split(self):
        balanced = ClusterResult(
            algorithm="gmm", k=2, labels=[0] * 10 + [1] * 10,
            centroids=[{"lr": 0.01}, {"lr": 0.1}],
            sizes=[10, 10], gene_names=["lr"],
            silhouette_score=0.8, bic_scores=None,
        )
        imbalanced = ClusterResult(
            algorithm="gmm", k=2, labels=[0] * 18 + [1] * 2,
            centroids=[{"lr": 0.01}, {"lr": 0.1}],
            sizes=[18, 2], gene_names=["lr"],
            silhouette_score=0.5, bic_scores=None,
        )
        assert (compute_speciation_quality_bundle(balanced).cluster_size_entropy
                > compute_speciation_quality_bundle(imbalanced).cluster_size_entropy)

    def test_entropy_non_negative(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=30)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        assert compute_speciation_quality_bundle(result).cluster_size_entropy >= 0.0

    # --- n_clusters ---

    def test_n_clusters_matches_result_k(self):
        for k in (0, 1, 2, 3):
            sizes = [5] * k
            labels: List[int] = []
            for i in range(k):
                labels.extend([i] * 5)
            result = ClusterResult(
                algorithm="gmm", k=k, labels=labels,
                centroids=[{"lr": 0.01}] * k,
                sizes=sizes, gene_names=["lr"],
                silhouette_score=0.5 if k >= 2 else 0.0,
                bic_scores=None,
            )
            assert compute_speciation_quality_bundle(result).n_clusters == k

    # --- end-to-end with real clustering ---

    def test_bimodal_gmm_bundle_sensible_values(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=30)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        bundle = compute_speciation_quality_bundle(result)
        assert bundle.speciation_index > 0.5
        assert bundle.raw_silhouette > 0.5
        assert bundle.noise_fraction == pytest.approx(0.0)
        assert bundle.cluster_size_entropy > 0.0
        assert bundle.n_clusters == 2


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

    def test_invalid_lineage_matcher_raises(self):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        with pytest.raises(ValueError, match="speciation_lineage_matcher"):
            GeneTrajectoryLogger(None, snapshot_interval=1, speciation_lineage_matcher="bad")

    def test_invalid_lineage_max_distance_raises_for_hungarian(self):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        with pytest.raises(ValueError, match="speciation_lineage_max_distance"):
            GeneTrajectoryLogger(
                None,
                snapshot_interval=1,
                speciation_lineage_matcher="hungarian",
                speciation_lineage_max_distance=float("inf"),
            )

    def test_greedy_lineage_max_distance_inf_allowed(self):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        GeneTrajectoryLogger(
            None,
            snapshot_interval=1,
            speciation_lineage_matcher="greedy",
            speciation_lineage_max_distance=float("inf"),
        )

    def test_greedy_lineage_matcher_keeps_legacy_transition_fields(self, tmp_path):
        """Greedy matcher keeps transition_type unset for backward compatibility."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        agents = [_make_fake_agent(lr=0.01), _make_fake_agent(lr=0.1)]
        env = _FakeEnvironment(agents)
        logger = GeneTrajectoryLogger(
            str(tmp_path),
            snapshot_interval=1,
            enable_speciation=True,
            speciation_lineage_matcher="greedy",
        )
        logger.snapshot(env, step=0)
        logger.close()

        lineage_path = tmp_path / "cluster_lineage.jsonl"
        rows = [json.loads(line) for line in lineage_path.read_text().splitlines()]
        assert len(rows) >= 1
        for row in rows:
            assert row["lineage_matcher"] == "greedy"
            assert row["transition_type"] is None
            assert row["parent_cluster_ids"] == []

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

    def test_dbscan_all_noise_snapshot_resets_lineage(self, tmp_path):
        """All-noise DBSCAN snapshots should reset lineage state."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        logger = GeneTrajectoryLogger(
            str(tmp_path),
            snapshot_interval=1,
            enable_speciation=True,
            speciation_algorithm="dbscan",
        )

        # Step 0: one dense cluster near lr ~= 0.01
        env_cluster_1 = _FakeEnvironment(
            [_make_fake_agent(lr=0.01), _make_fake_agent(lr=0.011), _make_fake_agent(lr=0.012)]
        )
        logger.snapshot(env_cluster_1, step=0)

        # Step 1: spread-out points -> DBSCAN labels all as noise.
        env_all_noise = _FakeEnvironment(
            [_make_fake_agent(lr=0.001), _make_fake_agent(lr=0.3), _make_fake_agent(lr=0.9)]
        )
        logger.snapshot(env_all_noise, step=1)

        # Step 2: cluster reappears; should be treated as a new founding lineage.
        env_cluster_2 = _FakeEnvironment(
            [_make_fake_agent(lr=0.01), _make_fake_agent(lr=0.011), _make_fake_agent(lr=0.012)]
        )
        logger.snapshot(env_cluster_2, step=2)
        logger.close()

        lineage_path = tmp_path / "cluster_lineage.jsonl"
        rows = [json.loads(line) for line in lineage_path.read_text().splitlines()]

        step_0_rows = [row for row in rows if row["step"] == 0]
        step_2_rows = [row for row in rows if row["step"] == 2]
        assert len(step_0_rows) == 1
        assert len(step_2_rows) == 1
        assert step_2_rows[0]["parent_cluster_id"] is None

    def test_quality_bundle_emitted_in_trajectory(self, tmp_path):
        """With enable_speciation=True, trajectory includes speciation_quality dict."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

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
        assert "speciation_quality" in rec
        q = rec["speciation_quality"]
        assert isinstance(q, dict)
        assert set(q.keys()) >= {
            "speciation_index", "raw_silhouette", "noise_fraction",
            "cluster_size_entropy", "n_clusters",
        }

    def test_quality_bundle_field_bounds(self, tmp_path):
        """All quality bundle fields are within their documented bounds."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        agents = (
            [_make_fake_agent(lr=0.001, gamma=0.99)] * 20
            + [_make_fake_agent(lr=0.9, gamma=0.1)] * 20
        )
        env = _FakeEnvironment(agents)
        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=1, enable_speciation=True)
        logger.snapshot(env, step=0)
        logger.close()

        traj_path = tmp_path / "intrinsic_gene_trajectory.jsonl"
        q = json.loads(traj_path.read_text().splitlines()[0])["speciation_quality"]
        assert 0.0 <= q["speciation_index"] <= 1.0
        assert -1.0 <= q["raw_silhouette"] <= 1.0
        assert 0.0 <= q["noise_fraction"] <= 1.0
        assert q["cluster_size_entropy"] >= 0.0
        assert isinstance(q["n_clusters"], int) and q["n_clusters"] >= 0

    def test_quality_bundle_absent_when_speciation_disabled(self, tmp_path):
        """Without enable_speciation, trajectory records have no speciation_quality."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=1)
        env = _FakeEnvironment([_make_fake_agent(lr=0.01), _make_fake_agent(lr=0.1)])
        logger.snapshot(env, step=0)
        logger.close()

        traj_path = tmp_path / "intrinsic_gene_trajectory.jsonl"
        rec = json.loads(traj_path.read_text().splitlines()[0])
        assert "speciation_quality" not in rec

    def test_quality_bundle_speciation_index_matches_top_level(self, tmp_path):
        """quality bundle's speciation_index must equal the top-level field."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

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
        assert rec["speciation_quality"]["speciation_index"] == pytest.approx(
            rec["speciation_index"]
        )

    # ------------------------------------------------------------------
    # clustering_interval tests
    # ------------------------------------------------------------------

    def test_invalid_clustering_interval_raises(self):
        """clustering_interval=0 should raise ValueError."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        with pytest.raises(ValueError, match="clustering_interval"):
            GeneTrajectoryLogger(None, snapshot_interval=5, clustering_interval=0)

    def test_clustering_interval_none_preserves_snapshot_cadence(self, tmp_path):
        """Default (clustering_interval=None) matches snapshot cadence – backward compatible."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        agents = (
            [_make_fake_agent(lr=0.01)] * 10
            + [_make_fake_agent(lr=0.5)] * 10
        )
        env = _FakeEnvironment(agents)

        # snapshot_interval=5, no explicit clustering_interval → clustering at 0, 5
        logger = GeneTrajectoryLogger(
            str(tmp_path), snapshot_interval=5, enable_speciation=True
        )
        for step in range(6):
            logger.snapshot(env, step=step)
        logger.close()

        traj_path = tmp_path / "intrinsic_gene_trajectory.jsonl"
        records = [json.loads(line) for line in traj_path.read_text().splitlines()]
        assert len(records) == 6

        idx_0 = records[0]["speciation_index"]
        # Steps 1–4 should still carry the cached value from step 0
        for rec in records[1:5]:
            assert rec["speciation_index"] == pytest.approx(idx_0)

    def test_clustering_interval_finer_than_snapshot_updates_every_step(self, tmp_path):
        """clustering_interval=1 re-runs clustering every step even between snapshots."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        # Two populations that change between steps to produce different indices
        pop_a = [_make_fake_agent(lr=0.01)] * 10 + [_make_fake_agent(lr=0.5)] * 10
        pop_b = [_make_fake_agent(lr=0.01)] * 20  # single cluster

        envs = [
            _FakeEnvironment(pop_a),  # step 0 – bimodal
            _FakeEnvironment(pop_b),  # step 1 – unimodal
            _FakeEnvironment(pop_a),  # step 2 – bimodal again
        ]

        # snapshot_interval=10 means no full snapshot after step 0.
        # clustering_interval=1 means speciation runs at every step.
        logger = GeneTrajectoryLogger(
            str(tmp_path),
            snapshot_interval=10,
            enable_speciation=True,
            clustering_interval=1,
        )
        for step, env in enumerate(envs):
            logger.snapshot(env, step=step)
        logger.close()

        traj_path = tmp_path / "intrinsic_gene_trajectory.jsonl"
        records = [json.loads(line) for line in traj_path.read_text().splitlines()]
        assert len(records) == 3

        # All records must have a speciation_index
        for rec in records:
            assert "speciation_index" in rec
            assert 0.0 <= rec["speciation_index"] <= 1.0

        # Bimodal steps (0 and 2) should have a higher index than the unimodal step (1).
        bimodal_index = records[0]["speciation_index"]
        unimodal_index = records[1]["speciation_index"]
        assert bimodal_index > unimodal_index

    def test_clustering_interval_independent_of_snapshot_interval(self, tmp_path):
        """clustering_interval=2, snapshot_interval=10 → clustering at 0,2,4,…."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        agents = [_make_fake_agent(lr=0.01)] * 10 + [_make_fake_agent(lr=0.5)] * 10
        env = _FakeEnvironment(agents)

        logger = GeneTrajectoryLogger(
            str(tmp_path),
            snapshot_interval=10,
            enable_speciation=True,
            clustering_interval=2,
        )
        for step in range(7):  # steps 0..6
            logger.snapshot(env, step=step)
        logger.close()

        lineage_path = tmp_path / "cluster_lineage.jsonl"
        rows = [json.loads(line) for line in lineage_path.read_text().splitlines()]

        # Clustering should have happened at steps 0, 2, 4, 6
        clustering_steps = sorted({row["step"] for row in rows})
        assert clustering_steps == [0, 2, 4, 6]

    def test_clustering_interval_cluster_lineage_written_at_finer_cadence(self, tmp_path):
        """cluster_lineage.jsonl entries appear at clustering steps, not only snapshot steps."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        agents = [_make_fake_agent(lr=0.01)] * 10 + [_make_fake_agent(lr=0.5)] * 10
        env = _FakeEnvironment(agents)

        # snapshot_interval=100, clustering_interval=3 → snapshots only at 0;
        # lineage rows at 0, 3, 6, 9
        logger = GeneTrajectoryLogger(
            str(tmp_path),
            snapshot_interval=100,
            enable_speciation=True,
            clustering_interval=3,
        )
        for step in range(10):  # steps 0..9
            logger.snapshot(env, step=step)
        logger.close()

        lineage_path = tmp_path / "cluster_lineage.jsonl"
        assert lineage_path.exists()
        rows = [json.loads(line) for line in lineage_path.read_text().splitlines()]
        clustering_steps = sorted({row["step"] for row in rows})
        assert clustering_steps == [0, 3, 6, 9]




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

    def test_zero_gene_dimensions_returns_none(self, tmp_path):
        result = ClusterResult(
            algorithm="gmm",
            k=1,
            labels=[0],
            centroids=[{}],
            sizes=[1],
            gene_names=[],
            silhouette_score=0.0,
            bic_scores={1: 0.0},
        )
        ctx = self._ctx(tmp_path)
        out = plot_chromosome_space_clusters([{}], result, ctx)
        assert out is None

    def test_agent_id_annotation_length_mismatch_does_not_fail(self, tmp_path):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0)
        ctx = self._ctx(tmp_path)
        out = plot_chromosome_space_clusters(chromosomes, result, ctx, agent_ids=["a0"])
        assert out is not None


# ---------------------------------------------------------------------------
# Feature scaling helpers
# ---------------------------------------------------------------------------


def _mixed_scale_chromosomes(n_per_cluster: int = 30, seed: int = 42) -> List[Dict[str, float]]:
    """Two clusters where genes have very different numeric ranges.

    Cluster A: lr ~0.005 (small range), budget ~1500 (large range).
    Cluster B: lr ~0.95 (small range), budget ~8500 (large range).

    Without scaling the ``budget`` dimension dominates Euclidean distances.
    With standard/robust scaling both features contribute equally.
    """
    rng = random.Random(seed)
    cluster_a = [
        {"lr": rng.gauss(0.005, 0.001), "budget": rng.gauss(1500.0, 80.0)}
        for _ in range(n_per_cluster)
    ]
    cluster_b = [
        {"lr": rng.gauss(0.95, 0.001), "budget": rng.gauss(8500.0, 80.0)}
        for _ in range(n_per_cluster)
    ]
    return cluster_a + cluster_b


class TestScalerParameter:
    """Tests for the optional ``scaler`` parameter in cluster-detection functions."""

    # ---- VALID_SCALERS constant ----

    def test_valid_scalers_exported(self):
        from farm.analysis.speciation import VALID_SCALERS
        assert "none" in VALID_SCALERS
        assert "standard" in VALID_SCALERS
        assert "robust" in VALID_SCALERS

    # ---- GMM: scaler stored in result ----

    def test_gmm_scaler_none_stored_in_result(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0, scaler="none")
        assert result.scaler == "none"

    def test_gmm_scaler_standard_stored_in_result(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0, scaler="standard")
        assert result.scaler == "standard"

    def test_gmm_scaler_robust_stored_in_result(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_gmm(chromosomes, max_k=3, seed=0, scaler="robust")
        assert result.scaler == "robust"

    def test_gmm_invalid_scaler_raises(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=5)
        with pytest.raises(ValueError, match="scaler"):
            detect_clusters_gmm(chromosomes, max_k=2, seed=0, scaler="minmax")

    def test_gmm_invalid_scaler_raises_on_empty_input(self):
        with pytest.raises(ValueError, match="scaler"):
            detect_clusters_gmm([], max_k=2, seed=0, scaler="minmax")

    def test_gmm_empty_input_scaler_stored(self):
        result = detect_clusters_gmm([], max_k=3, seed=0, scaler="standard")
        assert result.scaler == "standard"
        assert result.k == 0

    # ---- DBSCAN: scaler stored in result ----

    def test_dbscan_scaler_none_stored_in_result(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_dbscan(chromosomes, scaler="none")
        assert result.scaler == "none"

    def test_dbscan_scaler_standard_stored_in_result(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_dbscan(chromosomes, scaler="standard")
        assert result.scaler == "standard"

    def test_dbscan_scaler_robust_stored_in_result(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_dbscan(chromosomes, scaler="robust")
        assert result.scaler == "robust"

    def test_dbscan_invalid_scaler_raises(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=5)
        with pytest.raises(ValueError, match="scaler"):
            detect_clusters_dbscan(chromosomes, scaler="minmax")

    def test_dbscan_invalid_scaler_raises_on_empty_input(self):
        with pytest.raises(ValueError, match="scaler"):
            detect_clusters_dbscan([], scaler="minmax")

    def test_dbscan_empty_input_scaler_stored(self):
        result = detect_clusters_dbscan([], scaler="robust")
        assert result.scaler == "robust"
        assert result.k == 0

    # ---- Determinism with scaling ----

    def test_gmm_standard_scaler_deterministic(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        r1 = detect_clusters_gmm(chromosomes, max_k=4, seed=7, scaler="standard")
        r2 = detect_clusters_gmm(chromosomes, max_k=4, seed=7, scaler="standard")
        assert r1.k == r2.k
        assert r1.labels == r2.labels

    def test_gmm_robust_scaler_deterministic(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        r1 = detect_clusters_gmm(chromosomes, max_k=4, seed=7, scaler="robust")
        r2 = detect_clusters_gmm(chromosomes, max_k=4, seed=7, scaler="robust")
        assert r1.k == r2.k
        assert r1.labels == r2.labels

    # ---- Mixed-scale fixture: scaling improves cluster recovery ----

    def test_gmm_mixed_scale_standard_detects_two_clusters(self):
        """Standard scaling allows GMM to find the correct 2 clusters even when
        the ``budget`` gene has a much larger numeric range than ``lr``."""
        chromosomes = _mixed_scale_chromosomes(n_per_cluster=30)
        result = detect_clusters_gmm(chromosomes, max_k=4, seed=0, scaler="standard")
        assert result.k == 2
        assert result.silhouette_score > 0.5

    def test_gmm_mixed_scale_robust_detects_two_clusters(self):
        """Robust scaling also allows correct cluster detection on mixed-scale genes."""
        chromosomes = _mixed_scale_chromosomes(n_per_cluster=30)
        result = detect_clusters_gmm(chromosomes, max_k=4, seed=0, scaler="robust")
        assert result.k == 2
        assert result.silhouette_score > 0.5

    def test_gmm_scaled_centroids_are_in_raw_gene_units(self):
        chromosomes = _mixed_scale_chromosomes(n_per_cluster=30)
        result = detect_clusters_gmm(chromosomes, max_k=4, seed=0, scaler="standard")
        budgets = sorted(c["budget"] for c in result.centroids)
        # Verify centroids are in original budget units (roughly 1500 and 8500),
        # not in z-score space near 0.
        assert budgets[0] < 3000.0
        assert budgets[-1] > 7000.0

    def test_dbscan_mixed_scale_standard_detects_two_clusters(self):
        """With standard scaling, DBSCAN's eps is applied in scaled space and
        finds both clusters correctly."""
        chromosomes = _mixed_scale_chromosomes(n_per_cluster=30)
        # eps=0.3 in standardised space catches the two well-separated clusters
        result = detect_clusters_dbscan(chromosomes, eps=0.3, min_samples=3, scaler="standard")
        assert result.k >= 2

    def test_dbscan_mixed_scale_robust_detects_two_clusters(self):
        chromosomes = _mixed_scale_chromosomes(n_per_cluster=30)
        result = detect_clusters_dbscan(chromosomes, eps=0.3, min_samples=3, scaler="robust")
        assert result.k >= 2

    def test_dbscan_scaled_centroids_are_in_raw_gene_units(self):
        chromosomes = _mixed_scale_chromosomes(n_per_cluster=30)
        result = detect_clusters_dbscan(
            chromosomes, eps=0.3, min_samples=3, scaler="standard"
        )
        budgets = sorted(c["budget"] for c in result.centroids)
        assert budgets[0] < 3000.0
        assert budgets[-1] > 7000.0

    # ---- ClusterResult default scaler ----

    def test_cluster_result_default_scaler_is_none(self):
        """ClusterResult.scaler defaults to 'none' for backward compatibility."""
        result = ClusterResult(
            algorithm="gmm", k=1, labels=[0],
            centroids=[{"lr": 0.01}], sizes=[1],
            gene_names=["lr"], silhouette_score=0.0, bic_scores=None,
        )
        assert result.scaler == "none"


class TestGeneTrajectoryLoggerScaler:
    """Tests for speciation_scaler parameter in GeneTrajectoryLogger."""

    def test_default_scaler_is_none(self, tmp_path):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger
        logger = GeneTrajectoryLogger(str(tmp_path), snapshot_interval=1, enable_speciation=True)
        assert logger._speciation_scaler == "none"
        logger.close()

    def test_standard_scaler_accepted(self, tmp_path):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger
        logger = GeneTrajectoryLogger(
            str(tmp_path), snapshot_interval=1,
            enable_speciation=True, speciation_scaler="standard",
        )
        assert logger._speciation_scaler == "standard"
        logger.close()

    def test_robust_scaler_accepted(self, tmp_path):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger
        logger = GeneTrajectoryLogger(
            str(tmp_path), snapshot_interval=1,
            enable_speciation=True, speciation_scaler="robust",
        )
        assert logger._speciation_scaler == "robust"
        logger.close()

    def test_invalid_scaler_raises(self):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger
        with pytest.raises(ValueError, match="speciation_scaler"):
            GeneTrajectoryLogger(None, snapshot_interval=1, speciation_scaler="minmax")

    def test_scaler_persisted_in_cluster_lineage(self, tmp_path):
        """When speciation is enabled, cluster_lineage.jsonl rows include 'scaler' field."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        agents = (
            [_make_fake_agent(lr=0.001, gamma=0.99)] * 15
            + [_make_fake_agent(lr=0.9, gamma=0.1)] * 15
        )
        env = _FakeEnvironment(agents)
        logger = GeneTrajectoryLogger(
            str(tmp_path), snapshot_interval=1,
            enable_speciation=True, speciation_scaler="standard",
        )
        logger.snapshot(env, step=0)
        logger.close()

        lineage_path = tmp_path / "cluster_lineage.jsonl"
        assert lineage_path.exists()
        rows = [json.loads(line) for line in lineage_path.read_text().splitlines()]
        assert len(rows) >= 1
        for row in rows:
            assert row.get("scaler") == "standard"

    def test_standard_scaler_mixed_scale_produces_speciation_index(self, tmp_path):
        """Standard scaling should still produce a valid speciation_index on mixed-scale genes."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        # Build mixed-scale chromosomes via the fake-agent helper using large gamma range
        agents = (
            [_make_fake_agent(lr=0.001, gamma=0.99)] * 20
            + [_make_fake_agent(lr=0.9, gamma=0.1)] * 20
        )
        env = _FakeEnvironment(agents)
        logger = GeneTrajectoryLogger(
            str(tmp_path), snapshot_interval=1,
            enable_speciation=True, speciation_scaler="standard",
        )
        logger.snapshot(env, step=0)
        logger.close()

        traj_path = tmp_path / "intrinsic_gene_trajectory.jsonl"
        rec = json.loads(traj_path.read_text().splitlines()[0])
        assert "speciation_index" in rec
        assert 0.0 <= rec["speciation_index"] <= 1.0


# ---------------------------------------------------------------------------
# Wide-range fixture: fixed defaults fail but auto_tune succeeds
# ---------------------------------------------------------------------------


def _wide_range_bimodal_chromosomes(n_per_cluster: int = 25) -> List[Dict[str, float]]:
    """Two well-separated clusters on a regular grid with spacing > 0.1.

    Points within each cluster are placed on a 5x5 grid with 0.5-unit spacing,
    so the minimum inter-point distance is 0.5.  With fixed ``eps=0.1`` DBSCAN
    treats every agent as noise (k=0) because no two agents are within 0.1 of
    each other.  Auto-tuning estimates ``eps`` from the actual k-NN distances
    (~0.5) and correctly recovers both clusters.

    The ``n_per_cluster`` argument is accepted for API compatibility but the
    fixture always returns 50 points (two 5×5 grids).
    """
    import itertools

    cluster1 = [
        {"lr": float(i) * 0.5, "gamma": float(j) * 0.5}
        for i, j in itertools.product(range(5), range(5))
    ]
    cluster2 = [
        {"lr": 8.0 + float(i) * 0.5, "gamma": 8.0 + float(j) * 0.5}
        for i, j in itertools.product(range(5), range(5))
    ]
    return cluster1 + cluster2


# ---------------------------------------------------------------------------
# suggest_dbscan_params
# ---------------------------------------------------------------------------


class TestSuggestDBSCANParams:
    """Tests for the parameter-estimation helper."""

    def test_returns_dict_with_required_keys(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        params = suggest_dbscan_params(chromosomes)
        assert set(params.keys()) >= {"eps", "min_samples", "method", "k_percentile"}

    def test_eps_is_positive(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        params = suggest_dbscan_params(chromosomes)
        assert params["eps"] > 0.0

    def test_min_samples_at_least_two(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        params = suggest_dbscan_params(chromosomes)
        assert params["min_samples"] >= 2

    def test_min_samples_scales_with_dimensionality(self):
        """min_samples should be >= n_dims + 1 for n_dims >= 1."""
        chromosomes = [{"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}] * 30
        params = suggest_dbscan_params(chromosomes)
        # 4 dims → min_samples >= 5
        assert params["min_samples"] >= 5

    def test_method_field_is_k_distance_percentile(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        params = suggest_dbscan_params(chromosomes)
        assert params["method"] == "k_distance_percentile"

    def test_k_percentile_stored_in_result(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        params = suggest_dbscan_params(chromosomes, k_percentile=75.0)
        assert params["k_percentile"] == 75.0

    def test_invalid_k_percentile_raises(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        with pytest.raises(ValueError, match="k_percentile"):
            suggest_dbscan_params(chromosomes, k_percentile=110.0)

    def test_invalid_scaler_raises(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        with pytest.raises(ValueError, match="scaler"):
            suggest_dbscan_params(chromosomes, scaler="minmax")

    def test_empty_chromosomes_returns_fallback(self):
        """Empty input should not raise; returns a safe fallback dict."""
        params = suggest_dbscan_params([])
        assert params["eps"] > 0.0
        assert params["min_samples"] >= 2

    def test_single_chromosome_returns_fallback(self):
        """Single point → not enough for k-NN; safe fallback returned."""
        params = suggest_dbscan_params([{"lr": 0.5}])
        assert params["eps"] > 0.0
        assert params["min_samples"] >= 2

    def test_wide_range_eps_larger_than_default(self):
        """For wide-range data (grid spacing 0.5), suggested eps should exceed the fixed default 0.1."""
        chromosomes = _wide_range_bimodal_chromosomes()
        params = suggest_dbscan_params(chromosomes)
        assert params["eps"] > 0.1

    def test_gene_names_override_respected(self):
        """Supplying gene_names restricts the genes used for estimation."""
        chromosomes = [{"lr": 0.01, "gamma": 0.99, "noise": 999.0}] * 20
        params = suggest_dbscan_params(chromosomes, gene_names=["lr", "gamma"])
        assert params["min_samples"] >= 3  # 2 dims + 1

    def test_can_unpack_into_detect_clusters_dbscan(self):
        """Returned dict can be passed directly as kwargs to detect_clusters_dbscan."""
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        params = suggest_dbscan_params(chromosomes)
        # Should not raise; eps and min_samples are valid positives.
        result = detect_clusters_dbscan(chromosomes, **{
            k: v for k, v in params.items() if k in ("eps", "min_samples")
        })
        assert isinstance(result, ClusterResult)

    def test_scaler_parameter_used_in_distance_computation(self):
        """With standard scaler, suggested eps differs from 'none'."""
        chromosomes = _wide_range_bimodal_chromosomes()
        params_none = suggest_dbscan_params(chromosomes, scaler="none")
        params_std = suggest_dbscan_params(chromosomes, scaler="standard")
        # eps in standardised space should be much smaller than in raw space
        assert params_std["eps"] < params_none["eps"]


# ---------------------------------------------------------------------------
# DBSCAN auto_tune parameter
# ---------------------------------------------------------------------------


class TestDBSCANAutoTune:
    """Tests for the ``auto_tune`` parameter of detect_clusters_dbscan."""

    # ---- Acceptance: auto_tune=False (default) leaves dbscan_params None ----

    def test_default_dbscan_params_is_none(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_dbscan(chromosomes)
        assert result.dbscan_params is None

    def test_explicit_auto_tune_false_dbscan_params_is_none(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_dbscan(chromosomes, auto_tune=False)
        assert result.dbscan_params is None

    # ---- auto_tune=True populates dbscan_params ----

    def test_auto_tune_populates_dbscan_params(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        result = detect_clusters_dbscan(chromosomes, auto_tune=True)
        assert result.dbscan_params is not None
        assert "eps" in result.dbscan_params
        assert "min_samples" in result.dbscan_params
        assert "method" in result.dbscan_params
        assert "k_percentile" in result.dbscan_params

    def test_auto_tune_eps_is_positive(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        result = detect_clusters_dbscan(chromosomes, auto_tune=True)
        assert result.dbscan_params["eps"] > 0.0

    def test_auto_tune_min_samples_at_least_two(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        result = detect_clusters_dbscan(chromosomes, auto_tune=True)
        assert result.dbscan_params["min_samples"] >= 2

    def test_auto_tune_percentile_stored(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        result = detect_clusters_dbscan(chromosomes, auto_tune=True, auto_tune_percentile=80.0)
        assert result.dbscan_params["k_percentile"] == 80.0

    # ---- Key acceptance criterion: fixed defaults fail, auto_tune succeeds ----

    def test_fixed_defaults_fail_on_wide_range_data(self):
        """Verify the premise: default eps=0.1 labels all wide-range (grid) agents as noise."""
        chromosomes = _wide_range_bimodal_chromosomes()
        result_default = detect_clusters_dbscan(chromosomes, eps=0.1, min_samples=2)
        # With eps=0.1 and grid spacing 0.5, every agent is isolated → all noise
        assert result_default.k == 0

    def test_auto_tune_finds_clusters_on_wide_range_data(self):
        """auto_tune=True should find at least 2 clusters where defaults find none."""
        chromosomes = _wide_range_bimodal_chromosomes()
        result_auto = detect_clusters_dbscan(chromosomes, auto_tune=True)
        assert result_auto.k >= 2

    def test_auto_tune_empty_input_still_returns_zero_clusters(self):
        result = detect_clusters_dbscan([], auto_tune=True)
        assert result.k == 0
        # dbscan_params is None for empty input (no data to estimate from)
        assert result.dbscan_params is None

    def test_auto_tune_invalid_percentile_raises_on_empty_input(self):
        with pytest.raises(ValueError, match="auto_tune_percentile"):
            detect_clusters_dbscan([], auto_tune=True, auto_tune_percentile=120.0)

    def test_auto_tune_bimodal_finds_clusters(self):
        """auto_tune should also work on normal-range bimodal data."""
        chromosomes = _bimodal_chromosomes(n_per_cluster=20)
        result = detect_clusters_dbscan(chromosomes, auto_tune=True)
        assert result.k >= 2

    def test_auto_tune_with_scaler_wide_range(self):
        """auto_tune + standard scaler on wide-range data should find clusters."""
        chromosomes = _wide_range_bimodal_chromosomes()
        result = detect_clusters_dbscan(
            chromosomes, auto_tune=True, scaler="standard"
        )
        assert result.k >= 2
        assert result.dbscan_params is not None

    def test_auto_tune_labels_length_matches_input(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=15)
        result = detect_clusters_dbscan(chromosomes, auto_tune=True)
        assert len(result.labels) == len(chromosomes)

    def test_auto_tune_algorithm_field_is_dbscan(self):
        chromosomes = _bimodal_chromosomes(n_per_cluster=10)
        result = detect_clusters_dbscan(chromosomes, auto_tune=True)
        assert result.algorithm == "dbscan"


# ---------------------------------------------------------------------------
# GeneTrajectoryLogger: auto_tune integration
# ---------------------------------------------------------------------------


class TestGeneTrajectoryLoggerAutoTune:
    """Tests for speciation_dbscan_auto_tune in GeneTrajectoryLogger."""

    def _make_wide_range_agents(self) -> List[Any]:
        """Agents with two well-separated clusters where auto_tune finds structure."""
        import itertools

        # Two 5×5 grids of agents separated by a large gap in lr/gamma space.
        # Spacing within each cluster is 0.02 (between adjacent grid points).
        # auto_tune estimates eps from the actual k-NN distances and recovers
        # at least one cluster; dbscan_params is then persisted in lineage rows.
        low = [
            _make_fake_agent(lr=0.1 + i * 0.02, gamma=0.1 + j * 0.02)
            for i, j in itertools.product(range(5), range(5))
        ]
        high = [
            _make_fake_agent(lr=0.7 + i * 0.02, gamma=0.7 + j * 0.02)
            for i, j in itertools.product(range(5), range(5))
        ]
        return low + high

    def test_default_auto_tune_is_false(self, tmp_path):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger
        log = GeneTrajectoryLogger(
            str(tmp_path), snapshot_interval=1,
            enable_speciation=True, speciation_algorithm="dbscan",
        )
        assert log._speciation_dbscan_auto_tune is False
        log.close()

    def test_auto_tune_flag_stored(self, tmp_path):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger
        log = GeneTrajectoryLogger(
            str(tmp_path), snapshot_interval=1,
            enable_speciation=True, speciation_algorithm="dbscan",
            speciation_dbscan_auto_tune=True,
        )
        assert log._speciation_dbscan_auto_tune is True
        log.close()

    def test_auto_tune_percentile_stored(self, tmp_path):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger
        log = GeneTrajectoryLogger(
            str(tmp_path), snapshot_interval=1,
            enable_speciation=True, speciation_algorithm="dbscan",
            speciation_dbscan_auto_tune=True,
            speciation_dbscan_auto_tune_percentile=75.0,
        )
        assert log._speciation_dbscan_auto_tune_percentile == 75.0
        log.close()

    def test_invalid_percentile_raises(self):
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger
        with pytest.raises(ValueError, match="speciation_dbscan_auto_tune_percentile"):
            GeneTrajectoryLogger(
                None, snapshot_interval=1,
                speciation_dbscan_auto_tune_percentile=150.0,
            )

    def test_auto_tune_persists_dbscan_params_in_lineage(self, tmp_path):
        """cluster_lineage.jsonl rows include 'dbscan_params' when auto_tune=True."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        agents = self._make_wide_range_agents()
        env = _FakeEnvironment(agents)
        log = GeneTrajectoryLogger(
            str(tmp_path), snapshot_interval=1,
            enable_speciation=True, speciation_algorithm="dbscan",
            speciation_dbscan_auto_tune=True,
        )
        log.snapshot(env, step=0)
        log.close()

        lineage_path = tmp_path / "cluster_lineage.jsonl"
        assert lineage_path.exists(), "cluster_lineage.jsonl should be written"
        rows = [json.loads(line) for line in lineage_path.read_text().splitlines()]
        assert len(rows) >= 1, "Should have detected at least one cluster"
        for row in rows:
            assert "dbscan_params" in row
            params = row["dbscan_params"]
            assert params is not None
            assert "eps" in params
            assert "min_samples" in params

    def test_no_auto_tune_dbscan_params_is_none_in_lineage(self, tmp_path):
        """Without auto_tune, dbscan_params in lineage rows should be null."""
        from farm.runners.gene_trajectory_logger import GeneTrajectoryLogger

        agents = (
            [_make_fake_agent(lr=0.001, gamma=0.99)] * 15
            + [_make_fake_agent(lr=0.9, gamma=0.1)] * 15
        )
        env = _FakeEnvironment(agents)
        log = GeneTrajectoryLogger(
            str(tmp_path), snapshot_interval=1,
            enable_speciation=True, speciation_algorithm="dbscan",
            speciation_dbscan_auto_tune=False,
            speciation_scaler="standard",
        )
        # Use eps wide enough to find clusters so we actually get lineage rows
        # We rely on the standard scaler to find clusters
        log.snapshot(env, step=0)
        log.close()

        lineage_path = tmp_path / "cluster_lineage.jsonl"
        assert lineage_path.exists(), "cluster_lineage.jsonl should be written"
        rows = [json.loads(line) for line in lineage_path.read_text().splitlines()]
        for row in rows:
            # dbscan_params key should exist but be null (no auto_tune)
            assert "dbscan_params" in row
            assert row["dbscan_params"] is None


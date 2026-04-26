"""Speciation and niche-detection analysis for intrinsic-evolution runs.

Provides algorithms to detect population sub-structure (speciation) from
per-snapshot chromosome vectors, to track cluster identity across time steps,
and to compute a scalar *speciation index* suitable for time-series plots.

Algorithms
----------
- **GMM-BIC** (primary): Gaussian Mixture Model with k selected by minimising
  the Bayesian Information Criterion over ``k = 1 … max_k``.  Fully
  deterministic given ``seed``.
- **DBSCAN** (baseline): Density-Based Spatial Clustering; does not require
  a pre-specified ``k``; noisy agents are assigned label ``-1``.

Both functions consume a sequence of chromosome dicts (``{gene: value}``) and
return a :class:`ClusterResult` describing the detected structure.

Cluster persistence
-------------------
:func:`match_clusters_greedy` greedily assigns stable string IDs to the
clusters found at each consecutive snapshot by matching new centroids to
previous ones at minimum Euclidean distance.  New clusters that have no
close predecessor receive a fresh monotonically-increasing ID (``"c0"``,
``"c1"``, …).

Speciation index
----------------
:func:`compute_speciation_index` collapses a :class:`ClusterResult` into a
single scalar in ``[0.0, 1.0]``:

- When ``k == 1`` (or all agents are noise): ``0.0``.
- Otherwise: the *silhouette score* of the detected cluster assignment
  (bounded to ``[0.0, 1.0]``).

Niche correlation
-----------------
:func:`compute_niche_correlation` accepts the per-snapshot agent list (as
emitted by :class:`~farm.runners.gene_trajectory_logger.GeneTrajectoryLogger`)
together with a :class:`ClusterResult` and returns per-cluster summary dicts
containing mean spatial position, mean energy, and mean reproduction cost when
those fields are present in the agent records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from farm.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ClusterResult:
    """Output of a single cluster-detection run.

    Attributes
    ----------
    algorithm:
        Either ``"gmm"`` or ``"dbscan"``.
    k:
        Number of detected clusters.  For DBSCAN this excludes the noise
        pseudo-cluster (label ``-1``).
    labels:
        Per-agent integer cluster label.  DBSCAN noise agents have label
        ``-1``.  Length equals the number of input chromosomes.
    centroids:
        Per-cluster centroid as a ``{gene_name: value}`` dict.  For DBSCAN
        the centroid is the mean of member agents (noise cluster excluded).
    sizes:
        Number of agents in each cluster (noise excluded for DBSCAN).
    gene_names:
        Ordered list of gene names used for clustering (the columns of the
        feature matrix).
    silhouette_score:
        Silhouette score over all labelled agents in ``[-1, 1]``.
        ``0.0`` when ``k == 1`` or when fewer than 2 labelled agents exist.
    bic_scores:
        Mapping ``{k: bic_value}`` for each ``k`` tried during GMM search.
        ``None`` for DBSCAN.
    """

    algorithm: str
    k: int
    labels: List[int]
    centroids: List[Dict[str, float]]
    sizes: List[int]
    gene_names: List[str]
    silhouette_score: float
    bic_scores: Optional[Dict[int, float]]


@dataclass
class ClusterLineageRecord:
    """One entry in the cluster-lineage history for a single snapshot step.

    Attributes
    ----------
    step:
        Simulation step at which this cluster was detected.
    cluster_id:
        Stable string identifier for this cluster across snapshots (e.g.
        ``"c0"``).
    centroid:
        Gene-name → centroid-value mapping for this cluster at this step.
    size:
        Number of agents assigned to this cluster.
    parent_cluster_id:
        The ``cluster_id`` of the closest-matching cluster at the previous
        snapshot, or ``None`` for founding clusters at the first snapshot.
    gene_stats:
        Optional per-gene statistics dict (``{gene: {mean, std, …}}``) for
        agents in this cluster, populated by callers that request it.
    """

    step: int
    cluster_id: str
    centroid: Dict[str, float]
    size: int
    parent_cluster_id: Optional[str]
    gene_stats: Optional[Dict[str, Dict[str, float]]] = field(default=None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chromosomes_to_matrix(
    chromosomes: Sequence[Dict[str, float]],
    gene_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Convert a sequence of chromosome dicts to a (N, D) float matrix.

    Parameters
    ----------
    chromosomes:
        Sequence of ``{gene_name: value}`` dicts; all dicts must share the
        same set of keys.
    gene_names:
        Ordered list of gene names to use as columns.  When ``None`` the
        union of all keys across all dicts is used (sorted deterministically).

    Returns
    -------
    matrix : np.ndarray, shape (N, D)
    gene_names : list[str]
        Column labels corresponding to matrix columns.
    """
    if not chromosomes:
        return np.empty((0, 0), dtype=float), gene_names or []

    if gene_names is None:
        # Use sorted union for determinism
        all_keys: set = set()
        for c in chromosomes:
            all_keys.update(c.keys())
        gene_names = sorted(all_keys)

    matrix = np.array(
        [[c.get(g, 0.0) for g in gene_names] for c in chromosomes],
        dtype=float,
    )
    return matrix, gene_names


def _centroid_to_dict(centroid_vec: np.ndarray, gene_names: List[str]) -> Dict[str, float]:
    return {g: float(centroid_vec[i]) for i, g in enumerate(gene_names)}


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score, returning 0.0 on any failure."""
    unique = np.unique(labels[labels >= 0])
    if len(unique) < 2 or X.shape[0] < 2:
        return 0.0
    # Only consider labelled agents (DBSCAN may have noise = -1)
    mask = labels >= 0
    if mask.sum() < 2:
        return 0.0
    try:
        from sklearn.metrics import silhouette_score as _sk_silhouette
        score = float(_sk_silhouette(X[mask], labels[mask]))
        return max(0.0, score)
    except Exception as exc:
        logger.debug("_silhouette: failed to compute silhouette score: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Cluster detection – GMM-BIC
# ---------------------------------------------------------------------------


def detect_clusters_gmm(
    chromosomes: Sequence[Dict[str, float]],
    *,
    max_k: int = 5,
    seed: int = 0,
    gene_names: Optional[List[str]] = None,
) -> ClusterResult:
    """Detect population clusters using Gaussian Mixture Models with BIC selection.

    Tries ``k = 1 … max_k`` components and picks the ``k`` that minimises the
    Bayesian Information Criterion.  The result is fully deterministic given
    the same ``seed``.

    When fewer than ``max_k`` distinct chromosomes are present, ``max_k`` is
    capped automatically.  When zero chromosomes are provided an empty
    (``k=0``) trivial result with no labels or centroids is returned.

    Parameters
    ----------
    chromosomes:
        Sequence of chromosome dicts (``{gene_name: value}``).
    max_k:
        Maximum number of Gaussian components to try.  Default 5.
    seed:
        Random seed for reproducibility.  Default 0.
    gene_names:
        Ordered subset of genes to use.  When ``None`` all genes present in
        the chromosomes are used (sorted).

    Returns
    -------
    ClusterResult
    """
    from sklearn.mixture import GaussianMixture

    X, gnames = _chromosomes_to_matrix(chromosomes, gene_names)
    n_agents = X.shape[0]

    if n_agents == 0:
        return ClusterResult(
            algorithm="gmm",
            k=0,
            labels=[],
            centroids=[],
            sizes=[],
            gene_names=gnames,
            silhouette_score=0.0,
            bic_scores={},
        )

    # Cap max_k to the total number of agents to avoid degenerate GMM fits
    effective_max_k = min(max_k, n_agents)

    bic_scores: Dict[int, float] = {}
    best_k = 1
    best_bic = math.inf
    best_gmm = None

    for k in range(1, effective_max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=seed,
                max_iter=200,
                n_init=3,
            )
            gmm.fit(X)
            bic = float(gmm.bic(X))
            bic_scores[k] = bic
            if bic < best_bic:
                best_bic = bic
                best_k = k
                best_gmm = gmm
        except Exception as exc:
            logger.debug("detect_clusters_gmm: GMM fit failed for k=%d: %s", k, exc)

    if best_gmm is None:
        # All fits failed; return trivial single-cluster result
        centroid = _centroid_to_dict(X.mean(axis=0), gnames)
        return ClusterResult(
            algorithm="gmm",
            k=1,
            labels=[0] * n_agents,
            centroids=[centroid],
            sizes=[n_agents],
            gene_names=gnames,
            silhouette_score=0.0,
            bic_scores=bic_scores,
        )

    labels: np.ndarray = best_gmm.predict(X).astype(int)

    centroids = [
        _centroid_to_dict(best_gmm.means_[i], gnames)
        for i in range(best_k)
    ]
    sizes = [int((labels == i).sum()) for i in range(best_k)]
    sil = _silhouette(X, labels)

    return ClusterResult(
        algorithm="gmm",
        k=best_k,
        labels=labels.tolist(),
        centroids=centroids,
        sizes=sizes,
        gene_names=gnames,
        silhouette_score=sil,
        bic_scores=bic_scores,
    )


# ---------------------------------------------------------------------------
# Cluster detection – DBSCAN
# ---------------------------------------------------------------------------


def detect_clusters_dbscan(
    chromosomes: Sequence[Dict[str, float]],
    *,
    eps: float = 0.1,
    min_samples: int = 2,
    gene_names: Optional[List[str]] = None,
) -> ClusterResult:
    """Detect population clusters using DBSCAN.

    DBSCAN does not require a pre-specified number of clusters.  Agents that
    do not belong to any dense region receive label ``-1`` (noise).

    Parameters
    ----------
    chromosomes:
        Sequence of chromosome dicts (``{gene_name: value}``).
    eps:
        Maximum distance between two samples to be considered in the same
        neighbourhood.  Default ``0.1`` (gene values are typically in
        ``[0, 1]`` after normalisation).
    min_samples:
        Minimum number of samples in a neighbourhood to form a core point.
        Default 2.
    gene_names:
        Ordered subset of genes to use.  When ``None`` all present genes are
        used (sorted).

    Returns
    -------
    ClusterResult
    """
    from sklearn.cluster import DBSCAN

    X, gnames = _chromosomes_to_matrix(chromosomes, gene_names)
    n_agents = X.shape[0]

    if n_agents == 0:
        return ClusterResult(
            algorithm="dbscan",
            k=0,
            labels=[],
            centroids=[],
            sizes=[],
            gene_names=gnames,
            silhouette_score=0.0,
            bic_scores=None,
        )

    db = DBSCAN(eps=eps, min_samples=min_samples)
    raw_labels: np.ndarray = db.fit_predict(X).astype(int)

    unique_labels = [lbl for lbl in sorted(set(raw_labels.tolist())) if lbl >= 0]
    k = len(unique_labels)

    centroids: List[Dict[str, float]] = []
    sizes: List[int] = []
    for lbl in unique_labels:
        mask = raw_labels == lbl
        centroids.append(_centroid_to_dict(X[mask].mean(axis=0), gnames))
        sizes.append(int(mask.sum()))

    # Re-index labels so they run 0..k-1 for labelled agents; noise stays -1
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    remapped = np.array(
        [label_map.get(lbl, -1) for lbl in raw_labels.tolist()],
        dtype=int,
    )
    sil = _silhouette(X, remapped)

    return ClusterResult(
        algorithm="dbscan",
        k=k,
        labels=remapped.tolist(),
        centroids=centroids,
        sizes=sizes,
        gene_names=gnames,
        silhouette_score=sil,
        bic_scores=None,
    )


# ---------------------------------------------------------------------------
# Cluster persistence
# ---------------------------------------------------------------------------


def match_clusters_greedy(
    prev_records: List[ClusterLineageRecord],
    new_centroids: List[Dict[str, float]],
    new_sizes: List[int],
    gene_names: List[str],
    step: int,
    *,
    max_distance: float = float("inf"),
    id_counter_start: int = 0,
) -> Tuple[List[ClusterLineageRecord], int]:
    """Greedily assign stable cluster IDs by matching centroids across snapshots.

    Each new cluster centroid is matched to the closest unmatched previous
    cluster centroid.  When the minimum distance exceeds ``max_distance`` (or
    there are no previous clusters), a fresh ID is allocated instead.

    Parameters
    ----------
    prev_records:
        :class:`ClusterLineageRecord` list from the previous snapshot
        (may be empty for the first snapshot).
    new_centroids:
        List of per-cluster centroid dicts for the current snapshot.
    new_sizes:
        Corresponding per-cluster sizes.
    gene_names:
        Ordered list of gene names used to build centroid vectors.
    step:
        Current snapshot step (used to populate the new records).
    max_distance:
        Maximum Euclidean centroid distance for a match to be accepted.
        Default: unlimited.
    id_counter_start:
        Counter used to generate new IDs (``"c{counter}"``).  The returned
        second element is the next counter value after allocating all new IDs
        in this call.

    Returns
    -------
    records : list[ClusterLineageRecord]
        One record per new cluster.
    next_id_counter : int
        Updated counter for subsequent calls.
    """

    import re as _re

    def _centroid_vec(d: Dict[str, float]) -> np.ndarray:
        return np.array([d.get(g, 0.0) for g in gene_names], dtype=float)

    # Build vectors for new and previous centroids
    new_vecs = [_centroid_vec(c) for c in new_centroids]
    prev_vecs = [_centroid_vec(r.centroid) for r in prev_records]
    prev_ids = [r.cluster_id for r in prev_records]

    # Auto-advance id_counter past any IDs already used in prev_records so new
    # IDs never collide with existing lineage IDs regardless of what the caller
    # passes for id_counter_start.
    id_counter = id_counter_start
    for r in prev_records:
        m = _re.match(r"^c(\d+)$", r.cluster_id)
        if m:
            id_counter = max(id_counter, int(m.group(1)) + 1)

    used_prev: set = set()
    records: List[ClusterLineageRecord] = []

    for i, (centroid, size) in enumerate(zip(new_centroids, new_sizes)):
        best_j: Optional[int] = None
        best_dist = math.inf

        for j, pv in enumerate(prev_vecs):
            if j in used_prev:
                continue
            dist = float(np.linalg.norm(new_vecs[i] - pv))
            if dist < best_dist and dist <= max_distance:
                best_dist = dist
                best_j = j

        if best_j is not None:
            cluster_id = prev_ids[best_j]
            parent_cluster_id = prev_ids[best_j]
            used_prev.add(best_j)
        else:
            cluster_id = f"c{id_counter}"
            id_counter += 1
            parent_cluster_id = None

        records.append(
            ClusterLineageRecord(
                step=step,
                cluster_id=cluster_id,
                centroid=centroid,
                size=size,
                parent_cluster_id=parent_cluster_id,
            )
        )

    return records, id_counter


# ---------------------------------------------------------------------------
# Speciation index
# ---------------------------------------------------------------------------


def compute_speciation_index(cluster_result: ClusterResult) -> float:
    """Collapse a :class:`ClusterResult` into a single scalar speciation metric.

    Returns a value in ``[0.0, 1.0]`` where:

    - ``0.0`` means the population is a single undivided cluster (or too small
      to cluster).
    - Values approaching ``1.0`` indicate well-separated, evenly-sized
      sub-populations.

    The metric is the **silhouette score** of the cluster assignment when
    ``k ≥ 2``; ``0.0`` when ``k ≤ 1``.

    Parameters
    ----------
    cluster_result:
        A :class:`ClusterResult` from :func:`detect_clusters_gmm` or
        :func:`detect_clusters_dbscan`.

    Returns
    -------
    float
        Speciation index in ``[0.0, 1.0]``.
    """
    if cluster_result.k <= 1:
        return 0.0
    return max(0.0, min(1.0, cluster_result.silhouette_score))


# ---------------------------------------------------------------------------
# Niche correlation
# ---------------------------------------------------------------------------


def compute_niche_correlation(
    cluster_result: ClusterResult,
    agents: Sequence[Any],
    *,
    agent_id_key: str = "agent_id",
    x_key: str = "x",
    y_key: str = "y",
    energy_key: str = "energy",
    reproduction_cost_key: str = "reproduction_cost",
) -> List[Dict[str, Any]]:
    """Compute per-cluster spatial and behavioural niche statistics.

    For each cluster identified in ``cluster_result``, this function computes
    the mean spatial position (``mean_x``, ``mean_y``), mean energy, and mean
    reproduction cost for the agents assigned to that cluster.

    Agents must appear in the same order as the ``chromosomes`` that were
    passed to the cluster-detection function.  Fields that are absent from an
    agent record are silently skipped; the mean is computed only over agents
    where the field is present.

    Parameters
    ----------
    cluster_result:
        A :class:`ClusterResult` from :func:`detect_clusters_gmm` or
        :func:`detect_clusters_dbscan`.
    agents:
        Sequence of agent dicts (or objects with attribute access) in the
        same order as the chromosomes used for clustering.
    agent_id_key:
        Key / attribute name for the agent identifier field.
    x_key, y_key:
        Spatial coordinate field names.
    energy_key:
        Agent energy field name.
    reproduction_cost_key:
        Reproduction cost field name.

    Returns
    -------
    list of dict
        One dict per cluster with keys: ``cluster_id`` (int index),
        ``size``, ``mean_x``, ``mean_y``, ``mean_energy``,
        ``mean_reproduction_cost``, ``agent_ids``.  Fields that cannot be
        computed are set to ``None``.
    """

    def _get(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    labels = cluster_result.labels
    k = cluster_result.k
    if k == 0 or len(labels) == 0:
        return []

    # Bucket agents by cluster index
    buckets: Dict[int, List[Any]] = {i: [] for i in range(k)}
    for idx, lbl in enumerate(labels):
        if lbl >= 0 and idx < len(agents):
            buckets[lbl].append(agents[idx])

    niche_records: List[Dict[str, Any]] = []
    for i in range(k):
        members = buckets[i]
        n = len(members)

        def _mean_field(fkey: str) -> Optional[float]:
            vals = []
            for a in members:
                v = _get(a, fkey)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        # Non-numeric values are intentionally ignored when
                        # computing per-field means for heterogeneous records.
                        continue
            return float(sum(vals) / len(vals)) if vals else None

        agent_ids = [_get(a, agent_id_key) for a in members]

        niche_records.append(
            {
                "cluster_id": i,
                "size": n,
                "mean_x": _mean_field(x_key),
                "mean_y": _mean_field(y_key),
                "mean_energy": _mean_field(energy_key),
                "mean_reproduction_cost": _mean_field(reproduction_cost_key),
                "agent_ids": agent_ids,
            }
        )

    return niche_records

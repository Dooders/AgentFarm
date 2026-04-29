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

:func:`match_clusters_hungarian` uses a globally optimal assignment
(Hungarian / Kuhn–Munkres algorithm via
``scipy.optimize.linear_sum_assignment``) under the same distance gate,
then classifies each record as ``"founding"``, ``"continuation"``,
``"split"`` (one-to-many), or ``"merge"`` (many-to-one).  The full set
of predecessor IDs is stored in
:attr:`ClusterLineageRecord.parent_cluster_ids`.

Speciation index
----------------
:func:`compute_speciation_index` collapses a :class:`ClusterResult` into a
single scalar in ``[0.0, 1.0]``:

- When ``k == 1`` (or all agents are noise): ``0.0``.
- Otherwise: the *silhouette score* of the detected cluster assignment
  (bounded to ``[0.0, 1.0]``).

Quality bundle
--------------
:func:`compute_speciation_quality_bundle` returns a
:class:`SpeciationQualityBundle` with richer diagnostics alongside
``speciation_index``:

- ``raw_silhouette``: unclipped silhouette in ``[-1, 1]`` (negative values
  signal overlapping clusters).
- ``noise_fraction``: fraction of DBSCAN noise agents (``0.0`` for GMM).
- ``cluster_size_entropy``: Shannon entropy (nats) of cluster-size
  distribution; ``0.0`` when ``k ≤ 1``; higher means more balanced clusters.
- ``n_clusters``: number of detected clusters.
- ``stability_score`` (optional): mean agreement under small perturbations of
  chromosome values, normalised to ``[0, 1]``.

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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score as _sk_adjusted_rand
    from sklearn.metrics import silhouette_score as _sk_silhouette
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import RobustScaler, StandardScaler
except ImportError:
    DBSCAN = None  # type: ignore[assignment]
    _sk_adjusted_rand = None  # type: ignore[assignment]
    _sk_silhouette = None  # type: ignore[assignment]
    GaussianMixture = None  # type: ignore[assignment]
    RobustScaler = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]

from farm.utils.logging import get_logger

logger = get_logger(__name__)

def _require_sklearn() -> None:
    if (
        DBSCAN is None
        or _sk_adjusted_rand is None
        or _sk_silhouette is None
        or GaussianMixture is None
        or RobustScaler is None
        or StandardScaler is None
    ):
        raise ImportError(
            "Speciation clustering requires scikit-learn. Install with: pip install scikit-learn"
        )


# ---------------------------------------------------------------------------
# Scaler constants
# ---------------------------------------------------------------------------

#: Valid choices for the ``scaler`` parameter in cluster-detection functions.
VALID_SCALERS: Tuple[str, ...] = ("none", "standard", "robust")

#: Valid ``transition_type`` values emitted by :func:`match_clusters_hungarian`.
VALID_TRANSITION_TYPES: Tuple[str, ...] = (
    "founding",
    "continuation",
    "split",
    "merge",
)

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
    scaler:
        Feature scaling applied before clustering.  One of ``"none"``,
        ``"standard"``, or ``"robust"``.
    dbscan_params:
        When DBSCAN was run with ``auto_tune=True``, a dict containing the
        chosen ``eps``, ``min_samples``, ``method``, and ``k_percentile`` for
        reproducibility.  ``None`` when parameters were supplied explicitly or
        the algorithm is GMM.
    """

    algorithm: str
    k: int
    labels: List[int]
    centroids: List[Dict[str, float]]
    sizes: List[int]
    gene_names: List[str]
    silhouette_score: float
    bic_scores: Optional[Dict[int, float]]
    scaler: str = "none"
    dbscan_params: Optional[Dict[str, Any]] = field(default=None)


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
        Backward-compatible single-parent field; use ``parent_cluster_ids``
        for the full list when a merge has occurred.
    transition_type:
        How this cluster relates to the previous snapshot.  One of:

        - ``"founding"`` – no predecessor within ``max_distance``; brand-new
          lineage.
        - ``"continuation"`` – one-to-one match with no split or merge.
        - ``"split"`` – the predecessor also spawned at least one other new
          cluster (one-to-many).
        - ``"merge"`` – multiple predecessors are within ``max_distance`` of
          this cluster (many-to-one).

        ``None`` when populated by :func:`match_clusters_greedy` (legacy
        matcher that does not classify transitions).
    parent_cluster_ids:
        All predecessor ``cluster_id`` values within ``max_distance``.  For a
        continuation or split this is a single-element list; for a merge it
        contains all contributing predecessors.  Empty list for founding
        clusters or when populated by :func:`match_clusters_greedy`.
    gene_stats:
        Optional per-gene statistics dict (``{gene: {mean, std, …}}``) for
        agents in this cluster, populated by callers that request it.
    """

    step: int
    cluster_id: str
    centroid: Dict[str, float]
    size: int
    parent_cluster_id: Optional[str]
    transition_type: Optional[str] = field(default=None)
    parent_cluster_ids: List[str] = field(default_factory=list)
    gene_stats: Optional[Dict[str, Dict[str, float]]] = field(default=None)


@dataclass
class SpeciationQualityBundle:
    """Rich quality metrics for a single cluster-detection result.

    This bundle supplements the scalar :func:`compute_speciation_index` with
    additional diagnostics that expose signal which the clipped scalar loses
    (e.g. negative silhouette) and that characterise noise prevalence and
    cluster balance.

    Attributes
    ----------
    speciation_index:
        Normalised speciation index in ``[0.0, 1.0]``, identical to the
        value returned by :func:`compute_speciation_index`.
    raw_silhouette:
        Unclipped silhouette score in ``[-1.0, 1.0]``.  ``0.0`` when
        ``k ≤ 1`` or when fewer than two labelled agents exist.  Negative
        values indicate overlapping or misassigned clusters.
    noise_fraction:
        Fraction of all input agents labelled as noise (label ``-1``) by
        DBSCAN.  Always ``0.0`` for GMM results.
    cluster_size_entropy:
        Shannon entropy (nats) of the cluster-size distribution, computed
        over labelled agents only.  ``0.0`` when ``k ≤ 1``.  Higher values
        indicate more balanced clusters; maximum is ``ln(k)``.
    n_clusters:
        Number of detected clusters (``ClusterResult.k``).
    stability_score:
        Optional robustness metric in ``[0.0, 1.0]`` measuring assignment
        agreement under small feature-space perturbations. ``None`` when not
        computed by the caller.
    """

    speciation_index: float
    raw_silhouette: float
    noise_fraction: float
    cluster_size_entropy: float
    n_clusters: int
    stability_score: Optional[float] = None


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


def _matrix_to_chromosomes(X: np.ndarray, gene_names: List[str]) -> List[Dict[str, float]]:
    """Convert a (N, D) matrix back to chromosome dicts."""
    if X.shape[0] == 0:
        return []
    return [
        {g: float(X[i, j]) for j, g in enumerate(gene_names)}
        for i in range(X.shape[0])
    ]


def _centroid_to_dict(centroid_vec: np.ndarray, gene_names: List[str]) -> Dict[str, float]:
    return {g: float(centroid_vec[i]) for i, g in enumerate(gene_names)}


def _silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute raw silhouette score in ``[-1, 1]``, returning ``0.0`` on failure.

    Unlike the previous behaviour this function no longer clips negative
    scores; callers that need a non-negative value apply ``max(0.0, …)``
    themselves (see :func:`compute_speciation_index`).
    """
    unique = np.unique(labels[labels >= 0])
    if len(unique) < 2 or X.shape[0] < 2:
        return 0.0
    # Only consider labelled agents (DBSCAN may have noise = -1)
    mask = labels >= 0
    if mask.sum() < 2:
        return 0.0
    try:
        return float(_sk_silhouette(X[mask], labels[mask]))
    except Exception as exc:
        logger.debug("_silhouette: failed to compute silhouette score: %s", exc)
        return 0.0


def _validate_scaler(scaler: str) -> None:
    if scaler not in VALID_SCALERS:
        raise ValueError(
            f"scaler must be one of {VALID_SCALERS!r}; got {scaler!r}."
        )


def _identity_inverse(X: np.ndarray) -> np.ndarray:
    return X


def _apply_scaler(
    X: np.ndarray, scaler: str
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Apply optional feature scaling to a (N, D) matrix before clustering.

    Parameters
    ----------
    X:
        Feature matrix, shape ``(N, D)``.
    scaler:
        One of ``"none"`` (identity), ``"standard"`` (z-score via
        :class:`sklearn.preprocessing.StandardScaler`), or ``"robust"``
        (median/IQR via :class:`sklearn.preprocessing.RobustScaler`).

    Returns
    -------
    tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]
        ``(scaled_matrix, inverse_transform_fn)`` where the inverse-transform
        converts vectors back to raw gene units.

    Raises
    ------
    ValueError
        When ``scaler`` is not one of the valid choices.
    """
    _validate_scaler(scaler)
    _require_sklearn()
    if scaler == "none" or X.shape[0] == 0:
        return X, _identity_inverse
    if scaler == "standard":
        fitted_scaler = StandardScaler().fit(X)
        return fitted_scaler.transform(X), fitted_scaler.inverse_transform
    # "robust"
    fitted_scaler = RobustScaler().fit(X)
    return fitted_scaler.transform(X), fitted_scaler.inverse_transform


# ---------------------------------------------------------------------------
# Cluster detection – GMM-BIC
# ---------------------------------------------------------------------------


def detect_clusters_gmm(
    chromosomes: Sequence[Dict[str, float]],
    *,
    max_k: int = 5,
    seed: int = 0,
    gene_names: Optional[List[str]] = None,
    scaler: str = "none",
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
    scaler:
        Optional feature scaling applied to the raw gene matrix before
        clustering.  One of ``"none"`` (default, identity), ``"standard"``
        (z-score), or ``"robust"`` (median/IQR).  Using ``"standard"`` or
        ``"robust"`` is recommended when genes have very different numeric
        ranges.

    Returns
    -------
    ClusterResult
    """
    _validate_scaler(scaler)
    _require_sklearn()
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
            scaler=scaler,
        )

    X_scaled, inverse_transform = _apply_scaler(X, scaler)

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
            gmm.fit(X_scaled)
            bic = float(gmm.bic(X_scaled))
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
            scaler=scaler,
        )

    labels: np.ndarray = best_gmm.predict(X_scaled).astype(int)
    centroids_raw = inverse_transform(best_gmm.means_)

    centroids = [
        _centroid_to_dict(centroids_raw[i], gnames)
        for i in range(best_k)
    ]
    sizes = [int((labels == i).sum()) for i in range(best_k)]
    sil = _silhouette(X_scaled, labels)

    return ClusterResult(
        algorithm="gmm",
        k=best_k,
        labels=labels.tolist(),
        centroids=centroids,
        sizes=sizes,
        gene_names=gnames,
        silhouette_score=sil,
        bic_scores=bic_scores,
        scaler=scaler,
    )


# ---------------------------------------------------------------------------
# Cluster detection – DBSCAN
# ---------------------------------------------------------------------------


def suggest_dbscan_params(
    chromosomes: Sequence[Dict[str, float]],
    *,
    gene_names: Optional[List[str]] = None,
    k_percentile: float = 90.0,
    scaler: str = "none",
) -> Dict[str, Any]:
    """Suggest DBSCAN ``eps`` and ``min_samples`` from data statistics.

    Uses the *k-NN distance percentile* heuristic:

    1. ``min_samples`` is set to ``max(2, n_dims + 1)`` — a common rule of
       thumb that scales with chromosome dimensionality.
    2. For each point the distance to its ``min_samples``-th nearest neighbour
       is computed.  ``eps`` is the ``k_percentile``-th percentile of those
       sorted distances.

    The returned dict can be unpacked directly into :func:`detect_clusters_dbscan`::

        params = suggest_dbscan_params(chromosomes, k_percentile=90)
        result = detect_clusters_dbscan(chromosomes, **params)

    Parameters
    ----------
    chromosomes:
        Sequence of chromosome dicts used to estimate parameters.
    gene_names:
        Ordered subset of genes to use.  When ``None`` all present genes are
        used (sorted).
    k_percentile:
        Percentile (0–100) of the sorted k-NN distance distribution used to
        estimate ``eps``.  Higher values produce a more inclusive neighbourhood
        and tend to merge smaller clusters.  Default 90.
    scaler:
        Feature scaling applied before computing distances.  Same options as
        :func:`detect_clusters_dbscan`.  Use the same value you intend to pass
        to :func:`detect_clusters_dbscan` so the suggested ``eps`` is valid in
        the same scaled space.

    Returns
    -------
    dict
        ``{"eps": float, "min_samples": int, "method": str, "k_percentile": float}``
        where ``method`` is always ``"k_distance_percentile"``.

    Raises
    ------
    ValueError
        When ``k_percentile`` is outside ``[0, 100]`` or ``scaler`` is invalid.
    ImportError
        When scikit-learn is not installed.
    """
    if not 0.0 <= k_percentile <= 100.0:
        raise ValueError(f"k_percentile must be in [0, 100]; got {k_percentile!r}.")
    _validate_scaler(scaler)
    _require_sklearn()

    from sklearn.neighbors import NearestNeighbors

    X, gnames = _chromosomes_to_matrix(chromosomes, gene_names)
    n_agents, n_dims = X.shape

    min_samples: int = max(2, n_dims + 1)

    fallback: Dict[str, Any] = {
        "eps": 0.5,
        "min_samples": min_samples,
        "method": "k_distance_percentile",
        "k_percentile": k_percentile,
    }

    if n_agents <= min_samples:
        # Too few points to fit NearestNeighbors reliably; return safe defaults.
        logger.info(
            "suggest_dbscan_params: insufficient data for k-NN estimation; using fallback",
            n_agents=n_agents,
            n_dims=n_dims,
            min_samples=min_samples,
        )
        return fallback

    X_scaled, _ = _apply_scaler(X, scaler)

    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    # distances[:, -1] is the distance to the min_samples-th nearest neighbour
    kth_distances = distances[:, -1]
    eps = float(np.percentile(kth_distances, k_percentile))

    # Guard against degenerate case where all points are identical (eps == 0).
    if eps <= 0.0:
        eps = float(np.percentile(kth_distances, min(k_percentile + 5.0, 100.0)))
    if eps <= 0.0:
        eps = 1e-6  # absolute fallback to avoid DBSCAN running with eps=0

    logger.info(
        "suggest_dbscan_params",
        n_agents=n_agents,
        n_dims=n_dims,
        suggested_eps=round(eps, 6),
        suggested_min_samples=min_samples,
        k_percentile=k_percentile,
    )
    return {
        "eps": eps,
        "min_samples": min_samples,
        "method": "k_distance_percentile",
        "k_percentile": k_percentile,
    }


def detect_clusters_dbscan(
    chromosomes: Sequence[Dict[str, float]],
    *,
    eps: float = 0.1,
    min_samples: int = 2,
    gene_names: Optional[List[str]] = None,
    scaler: str = "none",
    auto_tune: bool = False,
    auto_tune_percentile: float = 90.0,
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
        ``[0, 1]`` after normalisation).  Ignored when ``auto_tune=True``.
    min_samples:
        Minimum number of samples in a neighbourhood to form a core point.
        Default 2.  Ignored when ``auto_tune=True``.
    gene_names:
        Ordered subset of genes to use.  When ``None`` all present genes are
        used (sorted).
    scaler:
        Optional feature scaling applied to the raw gene matrix before
        clustering.  One of ``"none"`` (default, identity), ``"standard"``
        (z-score), or ``"robust"`` (median/IQR).  When genes have widely
        different numeric ranges scaling is strongly recommended, as DBSCAN's
        ``eps`` threshold is applied in the scaled space.
    auto_tune:
        When ``True``, ``eps`` and ``min_samples`` are estimated automatically
        from the chromosome distribution using :func:`suggest_dbscan_params`
        (k-NN distance percentile heuristic).  The chosen parameters are
        recorded in :attr:`ClusterResult.dbscan_params` for reproducibility.
        Default ``False``.
    auto_tune_percentile:
        Percentile (0–100) of the k-NN distance distribution used to estimate
        ``eps`` when ``auto_tune=True``.  Higher values produce a larger, more
        inclusive neighbourhood.  Default 90.

    Returns
    -------
    ClusterResult
        ``dbscan_params`` is populated when ``auto_tune=True`` and contains
        the chosen ``eps``, ``min_samples``, ``method``, and ``k_percentile``.
    """
    _validate_scaler(scaler)
    if auto_tune and not 0.0 <= auto_tune_percentile <= 100.0:
        raise ValueError(
            "auto_tune_percentile must be in [0, 100] when auto_tune=True; "
            f"got {auto_tune_percentile!r}."
        )
    _require_sklearn()
    X, gnames = _chromosomes_to_matrix(chromosomes, gene_names)
    n_agents = X.shape[0]

    chosen_params: Optional[Dict[str, Any]] = None
    if auto_tune and n_agents > 0:
        chosen_params = suggest_dbscan_params(
            chromosomes,
            gene_names=gnames,
            k_percentile=auto_tune_percentile,
            scaler=scaler,
        )
        eps = chosen_params["eps"]
        min_samples = chosen_params["min_samples"]

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
            scaler=scaler,
            dbscan_params=chosen_params,
        )

    X_scaled, inverse_transform = _apply_scaler(X, scaler)

    db = DBSCAN(eps=eps, min_samples=min_samples)
    raw_labels: np.ndarray = db.fit_predict(X_scaled).astype(int)

    unique_labels = [lbl for lbl in sorted(set(raw_labels.tolist())) if lbl >= 0]
    k = len(unique_labels)

    centroids: List[Dict[str, float]] = []
    sizes: List[int] = []
    for lbl in unique_labels:
        mask = raw_labels == lbl
        centroid_scaled = X_scaled[mask].mean(axis=0, keepdims=True)
        centroid_raw = inverse_transform(centroid_scaled)[0]
        centroids.append(_centroid_to_dict(centroid_raw, gnames))
        sizes.append(int(mask.sum()))

    # Re-index labels so they run 0..k-1 for labelled agents; noise stays -1
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    remapped = np.array(
        [label_map.get(lbl, -1) for lbl in raw_labels.tolist()],
        dtype=int,
    )
    sil = _silhouette(X_scaled, remapped)

    return ClusterResult(
        algorithm="dbscan",
        k=k,
        labels=remapped.tolist(),
        centroids=centroids,
        sizes=sizes,
        gene_names=gnames,
        silhouette_score=sil,
        bic_scores=None,
        scaler=scaler,
        dbscan_params=chosen_params,
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


def match_clusters_hungarian(
    prev_records: List[ClusterLineageRecord],
    new_centroids: List[Dict[str, float]],
    new_sizes: List[int],
    gene_names: List[str],
    step: int,
    *,
    max_distance: float = 1.0,
    id_counter_start: int = 0,
) -> Tuple[List[ClusterLineageRecord], int]:
    """Assign stable cluster IDs using a globally optimal (Hungarian) matcher.

    Unlike :func:`match_clusters_greedy`, which processes new clusters in
    iteration order and makes locally-optimal nearest-centroid assignments,
    this function solves the assignment problem globally via the Hungarian
    algorithm (``scipy.optimize.linear_sum_assignment``).  Global
    optimisation avoids order-dependence and gives better ID stability
    during rapid topology shifts.

    After the one-to-one global assignment, unmatched new clusters are
    linked to their nearest predecessor within ``max_distance`` (split
    children).  Every resulting record carries a
    :attr:`ClusterLineageRecord.transition_type` (one of ``"founding"``,
    ``"continuation"``, ``"split"``, or ``"merge"``) and a
    :attr:`ClusterLineageRecord.parent_cluster_ids` list of all predecessor
    IDs within ``max_distance``.

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
        Pairs whose distance exceeds this threshold are never matched.
        Must be finite and > 0 to avoid pathological over-linking of
        unrelated clusters.
    id_counter_start:
        Counter used to generate new IDs (``"c{counter}"``).  The returned
        second element is the next counter value after allocating all new IDs
        in this call.

    Returns
    -------
    records : list[ClusterLineageRecord]
        One record per new cluster.  ``transition_type`` is one of
        ``"founding"``, ``"continuation"``, ``"split"``, or ``"merge"``;
        ``parent_cluster_ids`` lists every predecessor within
        ``max_distance`` (empty for founding clusters).
    next_id_counter : int
        Updated counter for subsequent calls.

    Raises
    ------
    ImportError
        When ``scipy`` is not installed.
    ValueError
        If ``max_distance`` is not finite and positive.

    Notes
    -----
    **Determinism**: the Hungarian algorithm is deterministic for a fixed
    cost matrix.  Tie-breaking follows
    ``scipy.optimize.linear_sum_assignment``.

    **Transition-type semantics**:

    - ``"founding"`` – no predecessor within ``max_distance``; new lineage.
    - ``"continuation"`` – one predecessor matched via global assignment,
      no split or merge detected.
    - ``"split"`` – the predecessor also produced at least one other new
      cluster (one-to-many).  The Hungarian-matched cluster inherits the
      predecessor ID; additional split children receive fresh IDs.
    - ``"merge"`` – multiple predecessors are within ``max_distance``
      (many-to-one).  The Hungarian-matched predecessor donates its ID;
      all contributors are listed in ``parent_cluster_ids``.

    ``max_distance`` is intentionally required to be finite so merge/split
    classification remains local rather than linking arbitrarily distant
    centroids.
    """
    if not math.isfinite(max_distance) or max_distance <= 0.0:
        raise ValueError(
            "match_clusters_hungarian requires max_distance to be finite and > 0; "
            f"got {max_distance!r}."
        )

    try:
        from scipy.optimize import linear_sum_assignment as _lsa
    except ImportError as exc:
        raise ImportError(
            "match_clusters_hungarian requires scipy. "
            "Install with: pip install scipy"
        ) from exc

    import re as _re

    def _centroid_vec(d: Dict[str, float]) -> np.ndarray:
        return np.array([d.get(g, 0.0) for g in gene_names], dtype=float)

    n_new = len(new_centroids)
    n_prev = len(prev_records)

    new_vecs = [_centroid_vec(c) for c in new_centroids]
    prev_vecs = [_centroid_vec(r.centroid) for r in prev_records]
    prev_ids = [r.cluster_id for r in prev_records]

    # Auto-advance id_counter past any IDs already used in prev_records
    id_counter = id_counter_start
    for r in prev_records:
        m = _re.match(r"^c(\d+)$", r.cluster_id)
        if m:
            id_counter = max(id_counter, int(m.group(1)) + 1)

    records: List[ClusterLineageRecord] = []

    # -----------------------------------------------------------------------
    # Edge cases: no previous clusters
    # -----------------------------------------------------------------------
    if n_prev == 0 or n_new == 0:
        for centroid, size in zip(new_centroids, new_sizes):
            records.append(
                ClusterLineageRecord(
                    step=step,
                    cluster_id=f"c{id_counter}",
                    centroid=centroid,
                    size=size,
                    parent_cluster_id=None,
                    transition_type="founding",
                    parent_cluster_ids=[],
                )
            )
            id_counter += 1
        return records, id_counter

    # -----------------------------------------------------------------------
    # Pairwise distance matrix  D[i, j] = dist(new_i, prev_j)
    # -----------------------------------------------------------------------
    D = np.zeros((n_new, n_prev), dtype=float)
    for i, nv in enumerate(new_vecs):
        for j, pv in enumerate(prev_vecs):
            D[i, j] = float(np.linalg.norm(nv - pv))

    # -----------------------------------------------------------------------
    # Hungarian assignment on gated cost matrix.
    # Pairs beyond max_distance get a finite penalty that is strictly larger
    # than any valid in-gate distance, so the solver will always prefer a
    # within-gate assignment when one exists.
    # -----------------------------------------------------------------------
    # n_new > 0 and n_prev > 0 are guaranteed by the early-return above,
    # so D is non-empty here and np.max(D) is safe.
    gate_ceiling = max(float(max_distance), float(np.max(D)))
    if not np.isfinite(gate_ceiling) or gate_ceiling >= np.finfo(float).max:
        raise ValueError(
            "max_distance and lineage-assignment distances must be finite "
            "and smaller than the largest representable float"
        )
    # np.nextafter gives the smallest representable float strictly greater than
    # gate_ceiling, ensuring out-of-gate pairs always cost more than any valid
    # in-gate pair regardless of the magnitude of max_distance.
    hungarian_gate_penalty = np.nextafter(gate_ceiling, math.inf)
    cost = np.where(D <= max_distance, D, hungarian_gate_penalty)
    row_ind, col_ind = _lsa(cost)

    # Accept only within-gate pairs
    primary_matches: Dict[int, int] = {}  # new_i → prev_j
    for ri, ci in zip(row_ind.tolist(), col_ind.tolist()):
        if ri < n_new and ci < n_prev and D[ri, ci] <= max_distance:
            primary_matches[ri] = ci

    # -----------------------------------------------------------------------
    # For unmatched new clusters find the nearest predecessor (split child)
    # -----------------------------------------------------------------------
    all_new_to_prev: Dict[int, Optional[int]] = {}
    for i in range(n_new):
        if i in primary_matches:
            all_new_to_prev[i] = primary_matches[i]
        else:
            best_j: Optional[int] = None
            best_d = math.inf
            for j in range(n_prev):
                if D[i, j] < best_d and D[i, j] <= max_distance:
                    best_d = D[i, j]
                    best_j = j
            all_new_to_prev[i] = best_j

    # Reverse mapping: prev_j → all new_i that point to it (primary Hungarian
    # matches plus nearest-predecessor links for unmatched new clusters)
    prev_to_all_new: Dict[int, List[int]] = {}
    for ni, pj in all_new_to_prev.items():
        if pj is not None:
            prev_to_all_new.setdefault(pj, []).append(ni)

    # Per new cluster: all prev clusters within max_distance (for merge check)
    new_to_all_prev: Dict[int, List[int]] = {
        i: [j for j in range(n_prev) if D[i, j] <= max_distance]
        for i in range(n_new)
    }

    # -----------------------------------------------------------------------
    # Build ClusterLineageRecord for each new cluster
    # -----------------------------------------------------------------------
    for i, (centroid, size) in enumerate(zip(new_centroids, new_sizes)):
        primary_j: Optional[int] = all_new_to_prev[i]
        all_prev_js: List[int] = new_to_all_prev[i]

        if primary_j is None:
            records.append(
                ClusterLineageRecord(
                    step=step,
                    cluster_id=f"c{id_counter}",
                    centroid=centroid,
                    size=size,
                    parent_cluster_id=None,
                    transition_type="founding",
                    parent_cluster_ids=[],
                )
            )
            id_counter += 1
            continue

        is_merge = len(all_prev_js) > 1
        is_split = len(prev_to_all_new.get(primary_j, [])) > 1

        if is_merge:
            transition_type = "merge"
        elif is_split:
            transition_type = "split"
        else:
            transition_type = "continuation"

        # Hungarian-matched clusters inherit the predecessor ID;
        # fallback-matched split children receive fresh IDs.
        if i in primary_matches:
            cluster_id = prev_ids[primary_j]
        else:
            cluster_id = f"c{id_counter}"
            id_counter += 1

        records.append(
            ClusterLineageRecord(
                step=step,
                cluster_id=cluster_id,
                centroid=centroid,
                size=size,
                parent_cluster_id=prev_ids[primary_j],
                transition_type=transition_type,
                parent_cluster_ids=[prev_ids[j] for j in all_prev_js],
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


def compute_speciation_stability_score(
    chromosomes: Sequence[Dict[str, float]],
    *,
    algorithm: str = "gmm",
    gene_names: Optional[List[str]] = None,
    scaler: str = "none",
    gmm_max_k: int = 5,
    gmm_seed: int = 0,
    dbscan_eps: float = 0.1,
    dbscan_min_samples: int = 2,
    dbscan_auto_tune: bool = False,
    dbscan_auto_tune_percentile: float = 90.0,
    n_perturbations: int = 5,
    noise_std: float = 0.005,
    random_seed: int = 0,
) -> float:
    """Estimate clustering robustness under small feature perturbations.

    The baseline cluster assignment is compared against assignments from
    ``n_perturbations`` noisy replicas using adjusted Rand index (ARI).  ARI is
    mapped from ``[-1, 1]`` to ``[0, 1]`` and averaged.
    """
    if algorithm not in ("gmm", "dbscan"):
        raise ValueError(f"algorithm must be 'gmm' or 'dbscan'; got {algorithm!r}.")
    if n_perturbations < 1:
        raise ValueError(
            f"n_perturbations must be at least 1; got {n_perturbations!r}."
        )
    if noise_std < 0.0:
        raise ValueError(f"noise_std must be non-negative; got {noise_std!r}.")
    _validate_scaler(scaler)
    _require_sklearn()

    X, gnames = _chromosomes_to_matrix(chromosomes, gene_names)
    if X.shape[0] < 2:
        return 0.0

    if algorithm == "gmm":
        baseline = detect_clusters_gmm(
            chromosomes,
            max_k=gmm_max_k,
            seed=gmm_seed,
            gene_names=gnames,
            scaler=scaler,
        )
    else:
        baseline = detect_clusters_dbscan(
            chromosomes,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            gene_names=gnames,
            scaler=scaler,
            auto_tune=dbscan_auto_tune,
            auto_tune_percentile=dbscan_auto_tune_percentile,
        )

    baseline_labels = np.array(baseline.labels, dtype=int)
    rng = np.random.default_rng(random_seed)
    scores: List[float] = []

    for _ in range(n_perturbations):
        X_perturbed = X + rng.normal(loc=0.0, scale=noise_std, size=X.shape)
        perturbed_chromosomes = _matrix_to_chromosomes(X_perturbed, gnames)
        if algorithm == "gmm":
            perturbed = detect_clusters_gmm(
                perturbed_chromosomes,
                max_k=gmm_max_k,
                seed=gmm_seed,
                gene_names=gnames,
                scaler=scaler,
            )
        else:
            perturbed = detect_clusters_dbscan(
                perturbed_chromosomes,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                gene_names=gnames,
                scaler=scaler,
                auto_tune=dbscan_auto_tune,
                auto_tune_percentile=dbscan_auto_tune_percentile,
            )
        ari = float(_sk_adjusted_rand(baseline_labels, np.array(perturbed.labels, dtype=int)))
        scores.append((ari + 1.0) / 2.0)

    if not scores:
        return 0.0
    return max(0.0, min(1.0, float(sum(scores) / len(scores))))


def compute_speciation_quality_bundle(
    cluster_result: ClusterResult,
    *,
    stability_score: Optional[float] = None,
) -> SpeciationQualityBundle:
    """Compute a rich set of quality metrics for a cluster-detection result.

    Returns a :class:`SpeciationQualityBundle` that supplements the scalar
    :func:`compute_speciation_index` with diagnostics for noise prevalence
    and cluster balance.

    Parameters
    ----------
    cluster_result:
        A :class:`ClusterResult` from :func:`detect_clusters_gmm` or
        :func:`detect_clusters_dbscan`.

    Returns
    -------
    SpeciationQualityBundle
        Bundle with the following fields:

        - ``speciation_index`` – normalised index in ``[0.0, 1.0]``
          (same as :func:`compute_speciation_index`).
        - ``raw_silhouette`` – unclipped silhouette in ``[-1.0, 1.0]``.
        - ``noise_fraction`` – fraction of DBSCAN noise agents; ``0.0``
          for GMM.
        - ``cluster_size_entropy`` – Shannon entropy (nats) of the
          cluster-size distribution; ``0.0`` when ``k ≤ 1``.
        - ``n_clusters`` – number of detected clusters.
        - ``stability_score`` – optional robustness estimate in ``[0.0, 1.0]``
          from :func:`compute_speciation_stability_score`.
    """
    n_total = len(cluster_result.labels)

    # Noise fraction (DBSCAN label -1; always 0 for GMM)
    if cluster_result.algorithm == "dbscan" and n_total > 0:
        n_noise = sum(1 for lbl in cluster_result.labels if lbl < 0)
        noise_fraction = float(n_noise) / float(n_total)
    else:
        noise_fraction = 0.0

    # Cluster-size Shannon entropy (nats)
    if cluster_result.k >= 2 and cluster_result.sizes:
        total_labelled = float(sum(cluster_result.sizes))
        if total_labelled > 0.0:
            entropy = 0.0
            for sz in cluster_result.sizes:
                p = sz / total_labelled
                if p > 0.0:
                    entropy -= p * math.log(p)
        else:
            entropy = 0.0
    else:
        entropy = 0.0

    return SpeciationQualityBundle(
        speciation_index=compute_speciation_index(cluster_result),
        raw_silhouette=cluster_result.silhouette_score if cluster_result.k >= 2 else 0.0,
        noise_fraction=noise_fraction,
        cluster_size_entropy=entropy,
        n_clusters=cluster_result.k,
        stability_score=stability_score,
    )


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

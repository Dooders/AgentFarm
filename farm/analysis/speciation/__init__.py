"""Speciation and niche-detection analysis for intrinsic-evolution runs.

Detects population sub-structure (speciation) from per-snapshot chromosome
vectors, tracks cluster identity across time steps, and computes a scalar
*speciation index* for time-series plots.

Features
--------
- **GMM-BIC** cluster detection: Gaussian Mixture Model with ``k`` selected
  by minimising BIC.  Fully deterministic given a ``seed``.
- **DBSCAN** cluster detection (baseline): density-based; does not require a
  pre-specified ``k``.
- **Cluster persistence**: greedy centroid-distance matcher that assigns
  stable string IDs across consecutive snapshots.
- **Speciation index**: silhouette-based scalar in ``[0, 1]`` collapsed from
  a :class:`~farm.analysis.speciation.compute.ClusterResult`.
- **Quality bundle**: richer diagnostics via
  :func:`compute_speciation_quality_bundle` including raw (unclipped)
  silhouette, noise fraction, cluster-size entropy, and cluster count.
- **Niche correlation**: per-cluster mean spatial position, energy, and
  reproduction cost.
- **Plot helper**: PCA/scatter of agents in chromosome space coloured by
  detected cluster.

Quick start::

    >>> from farm.analysis.speciation import (
    ...     detect_clusters_gmm,
    ...     detect_clusters_dbscan,
    ...     match_clusters_greedy,
    ...     compute_speciation_index,
    ...     compute_speciation_quality_bundle,
    ...     compute_niche_correlation,
    ...     ClusterResult,
    ...     ClusterLineageRecord,
    ...     SpeciationQualityBundle,
    ...     plot_chromosome_space_clusters,
    ... )
    >>> chromosomes = [{"lr": 0.01, "gamma": 0.99}, {"lr": 0.1, "gamma": 0.5}]
    >>> result = detect_clusters_gmm(chromosomes, max_k=3, seed=42)
    >>> index = compute_speciation_index(result)
    >>> bundle = compute_speciation_quality_bundle(result)

See ``tests/analysis/test_speciation.py`` for examples with known cluster
fixtures and persistence across snapshots.
"""

from farm.analysis.speciation.compute import (
    ClusterResult,
    ClusterLineageRecord,
    SpeciationQualityBundle,
    VALID_SCALERS,
    VALID_TRANSITION_TYPES,
    detect_clusters_gmm,
    detect_clusters_dbscan,
    suggest_dbscan_params,
    match_clusters_greedy,
    match_clusters_hungarian,
    compute_speciation_index,
    compute_speciation_quality_bundle,
    compute_niche_correlation,
)
from farm.analysis.speciation.plot import plot_chromosome_space_clusters

__all__ = [
    # Data structures
    "ClusterResult",
    "ClusterLineageRecord",
    "SpeciationQualityBundle",
    # Configuration constants
    "VALID_SCALERS",
    "VALID_TRANSITION_TYPES",
    # Cluster detection
    "detect_clusters_gmm",
    "detect_clusters_dbscan",
    "suggest_dbscan_params",
    # Cluster persistence
    "match_clusters_greedy",
    "match_clusters_hungarian",
    # Scalar metrics
    "compute_speciation_index",
    "compute_speciation_quality_bundle",
    # Niche analysis
    "compute_niche_correlation",
    # Visualisation
    "plot_chromosome_space_clusters",
]

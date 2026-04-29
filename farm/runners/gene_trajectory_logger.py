"""Per-step chromosome telemetry for intrinsic-evolution simulations.

Writes append-only JSONL files alongside other run artifacts:

- ``intrinsic_gene_trajectory.jsonl`` -- one record per step containing
  aggregate per-gene statistics over alive agents.  Compact and cheap to
  append every step.  When speciation tracking is enabled, each record also
  contains a ``speciation_index`` scalar.
- ``intrinsic_gene_snapshots.jsonl`` -- one record every
  ``snapshot_interval`` steps containing each alive agent's full chromosome
  values.  Supports lineage analysis offline; bounded cost via the
  configurable cadence.
- ``cluster_lineage.jsonl`` -- written only when speciation tracking is
  enabled (``enable_speciation=True``).  One record per detected cluster per
  clustering step with fields ``step``, ``cluster_id``, ``centroid``, ``size``,
  ``parent_cluster_id``, ``transition_type``, ``parent_cluster_ids``,
  ``lineage_matcher``, and ``lineage_max_distance``.

Snapshot cadence (``snapshot_interval``) and clustering cadence
(``clustering_interval``) are independent.  Use a finer
``clustering_interval`` to capture short-lived speciation events without
paying the cost of a full per-agent snapshot payload on every step.

When ``output_dir`` is ``None``, JSONL files are not written; speciation
state is still updated in memory when ``enable_speciation=True``, allowing
the runner to use the logger unconditionally regardless of persisted artifacts.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, TextIO

from farm.analysis.speciation.compute import (
    VALID_SCALERS,
    compute_speciation_quality_bundle,
    compute_speciation_stability_score,
    detect_clusters_dbscan,
    detect_clusters_gmm,
    match_clusters_greedy,
    match_clusters_hungarian,
)
from farm.core.hyperparameter_chromosome import (
    HyperparameterChromosome,
    compute_gene_statistics,
)


class GeneTrajectoryLogger:
    """Buffered JSONL logger for chromosome distributions and snapshots.

    Parameters
    ----------
    output_dir:
        Directory in which to write the JSONL files.  When ``None``, file
        I/O is suppressed; ``snapshot`` still updates in-memory speciation
        when ``enable_speciation=True``.
    snapshot_interval:
        Write a full per-agent snapshot every this many steps.  Step 0 is
        always captured.  Must be at least 1.
    enable_speciation:
        When ``True``, run cluster detection according to the clustering
        cadence and include a ``speciation_index`` field in every trajectory
        record.  Also writes ``cluster_lineage.jsonl``.  Default ``False``.
    clustering_interval:
        Run cluster detection (and update ``speciation_index``) every this
        many steps.  Step 0 is always a clustering step.  When ``None``
        (default), the clustering cadence matches ``snapshot_interval`` so
        that existing behaviour is preserved.  Set to a smaller value than
        ``snapshot_interval`` to capture short-lived speciation events
        without writing full per-agent snapshots at each clustering step.
        Must be at least 1 when provided.
    speciation_algorithm:
        Which cluster-detection algorithm to use.  ``"gmm"`` (default) runs
        Gaussian Mixture Models with BIC-selected ``k``; ``"dbscan"`` runs
        density-based clustering.
    speciation_max_k:
        Maximum number of clusters to try when ``speciation_algorithm="gmm"``.
        Default 5.
    speciation_seed:
        Integer random seed for reproducible cluster detection.  Default 0.
    speciation_scaler:
        Optional feature scaling applied to chromosome vectors before
        clustering.  One of ``"none"`` (default, no scaling), ``"standard"``
        (z-score), or ``"robust"`` (median/IQR).  Scaling is recommended when
        genes have widely different numeric ranges.
    speciation_dbscan_auto_tune:
        When ``True`` and ``speciation_algorithm="dbscan"``, ``eps`` and
        ``min_samples`` are estimated automatically from each snapshot's
        chromosome distribution using the k-NN distance percentile heuristic.
        The chosen parameters are recorded in ``cluster_lineage.jsonl`` rows
        under the ``"dbscan_params"`` key for reproducibility.  Default ``False``.
    speciation_dbscan_auto_tune_percentile:
        Percentile (0–100) of the k-NN distance distribution used to estimate
        ``eps`` when ``speciation_dbscan_auto_tune=True``.  Default 90.
    speciation_lineage_matcher:
        Cluster-lineage matcher used to persist stable IDs across snapshots.
        ``"hungarian"`` (default) uses a globally optimal assignment and
        populates transition metadata; ``"greedy"`` preserves legacy behavior.
    speciation_lineage_max_distance:
        Maximum centroid distance accepted by the lineage matcher. For
        ``"hungarian"``, must be finite and > 0 (default ``1.0`` when omitted).
        For ``"greedy"``, may be ``float("inf")`` for unlimited matching (the
        historical default); when omitted, greedy uses unlimited distance.
        Lower finite values are more conservative and classify distant clusters
        as founding.
    speciation_include_stability:
        When ``True``, compute and include an optional ``stability_score`` in
        ``speciation_quality`` using perturbation-based robustness checks.
        Default ``False``.
    speciation_stability_perturbations:
        Number of perturbation replicas used for stability estimation.
        Must be at least 1. Default 5.
    speciation_stability_noise_std:
        Standard deviation of additive Gaussian noise in chromosome space for
        each perturbation replica. Must be non-negative. Default 0.005.
    speciation_stability_seed:
        Integer random seed used for deterministic perturbation sampling.
        Default 0.
    """

    TRAJECTORY_FILENAME = "intrinsic_gene_trajectory.jsonl"
    SNAPSHOT_FILENAME = "intrinsic_gene_snapshots.jsonl"
    CLUSTER_LINEAGE_FILENAME = "cluster_lineage.jsonl"
    VALID_LINEAGE_MATCHERS = ("hungarian", "greedy")

    def __init__(
        self,
        output_dir: Optional[str],
        snapshot_interval: int,
        *,
        enable_speciation: bool = False,
        clustering_interval: Optional[int] = None,
        speciation_algorithm: str = "gmm",
        speciation_max_k: int = 5,
        speciation_seed: int = 0,
        speciation_scaler: str = "none",
        speciation_dbscan_auto_tune: bool = False,
        speciation_dbscan_auto_tune_percentile: float = 90.0,
        speciation_lineage_matcher: str = "hungarian",
        speciation_lineage_max_distance: Optional[float] = None,
        speciation_include_stability: bool = False,
        speciation_stability_perturbations: int = 5,
        speciation_stability_noise_std: float = 0.005,
        speciation_stability_seed: int = 0,
    ) -> None:
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be at least 1.")
        if clustering_interval is not None and clustering_interval < 1:
            raise ValueError("clustering_interval must be at least 1 when provided.")
        if speciation_algorithm not in ("gmm", "dbscan"):
            raise ValueError("speciation_algorithm must be 'gmm' or 'dbscan'.")
        if speciation_lineage_matcher not in self.VALID_LINEAGE_MATCHERS:
            raise ValueError(
                "speciation_lineage_matcher must be one of "
                f"{self.VALID_LINEAGE_MATCHERS!r}; got {speciation_lineage_matcher!r}."
            )
        if speciation_lineage_max_distance is None:
            resolved_max_distance = (
                float("inf") if speciation_lineage_matcher == "greedy" else 1.0
            )
        elif speciation_lineage_matcher == "hungarian":
            if not math.isfinite(speciation_lineage_max_distance) or speciation_lineage_max_distance <= 0.0:
                raise ValueError(
                    "speciation_lineage_max_distance must be finite and > 0 for "
                    f"speciation_lineage_matcher='hungarian'; got {speciation_lineage_max_distance!r}."
                )
            resolved_max_distance = speciation_lineage_max_distance
        else:
            # greedy: unlimited (inf) or any positive finite distance
            d = speciation_lineage_max_distance
            if d != float("inf") and (not math.isfinite(d) or d <= 0.0):
                raise ValueError(
                    "speciation_lineage_max_distance must be float('inf') or finite and > 0 for "
                    f"speciation_lineage_matcher='greedy'; got {d!r}."
                )
            resolved_max_distance = d
        if speciation_scaler not in VALID_SCALERS:
            raise ValueError(
                f"speciation_scaler must be one of {VALID_SCALERS!r}; "
                f"got {speciation_scaler!r}."
            )
        if not 0.0 <= speciation_dbscan_auto_tune_percentile <= 100.0:
            raise ValueError(
                "speciation_dbscan_auto_tune_percentile must be in [0, 100]; "
                f"got {speciation_dbscan_auto_tune_percentile!r}."
            )
        if speciation_stability_perturbations < 1:
            raise ValueError(
                "speciation_stability_perturbations must be at least 1; "
                f"got {speciation_stability_perturbations!r}."
            )
        if speciation_stability_noise_std < 0.0:
            raise ValueError(
                "speciation_stability_noise_std must be non-negative; "
                f"got {speciation_stability_noise_std!r}."
            )

        if enable_speciation:
            try:
                import sklearn  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "GeneTrajectoryLogger: enable_speciation=True requires scikit-learn. "
                    "Install it with: pip install scikit-learn"
                ) from exc
            if speciation_lineage_matcher == "hungarian":
                try:
                    import scipy  # noqa: F401
                except ImportError as exc:
                    raise ImportError(
                        "GeneTrajectoryLogger: speciation_lineage_matcher='hungarian' requires scipy. "
                        "Install it with: pip install scipy"
                    ) from exc
        self._output_dir = output_dir
        self._snapshot_interval = snapshot_interval
        self._clustering_interval = clustering_interval
        self._enable_speciation = enable_speciation
        self._speciation_algorithm = speciation_algorithm
        self._speciation_max_k = speciation_max_k
        self._speciation_seed = speciation_seed
        self._speciation_scaler = speciation_scaler
        self._speciation_dbscan_auto_tune = speciation_dbscan_auto_tune
        self._speciation_dbscan_auto_tune_percentile = speciation_dbscan_auto_tune_percentile
        self._speciation_lineage_matcher = speciation_lineage_matcher
        self._speciation_lineage_max_distance = resolved_max_distance
        self._speciation_include_stability = speciation_include_stability
        self._speciation_stability_perturbations = speciation_stability_perturbations
        self._speciation_stability_noise_std = speciation_stability_noise_std
        self._speciation_stability_seed = speciation_stability_seed

        self._trajectory_handle: Optional[TextIO] = None
        self._snapshot_handle: Optional[TextIO] = None
        self._cluster_lineage_handle: Optional[TextIO] = None

        # Cached speciation state between snapshot steps
        self._cached_speciation_index: float = 0.0
        self._cached_speciation_quality: Optional[Dict[str, Any]] = None
        self._prev_cluster_records: List[Any] = []  # List[ClusterLineageRecord]
        self._cluster_id_counter: int = 0

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self._trajectory_handle = open(
                os.path.join(output_dir, self.TRAJECTORY_FILENAME),
                "w",
                encoding="utf-8",
            )
            self._snapshot_handle = open(
                os.path.join(output_dir, self.SNAPSHOT_FILENAME),
                "w",
                encoding="utf-8",
            )
            if enable_speciation:
                self._cluster_lineage_handle = open(
                    os.path.join(output_dir, self.CLUSTER_LINEAGE_FILENAME),
                    "w",
                    encoding="utf-8",
                )

    def snapshot(self, environment: Any, step: int, extra_fields: Optional[Dict[str, Any]] = None) -> None:
        """Record the chromosome distribution at ``step``.

        Always appends a one-line aggregate record to the trajectory file.
        When speciation tracking is enabled the record includes a
        ``speciation_index`` field updated at every clustering step and,
        when at least one chromosome is present at a snapshot step, a
        ``speciation_quality`` dict containing ``speciation_index``,
        ``raw_silhouette``, ``noise_fraction``, ``cluster_size_entropy``,
        and ``n_clusters`` plus ``computed_at_step`` (the clustering step
        when the quality bundle was last recomputed).  When
        ``speciation_include_stability=True``, ``speciation_quality`` also
        includes ``stability_score`` in ``[0, 1]``.  ``speciation_quality``
        is absent from non-snapshot steps and from snapshot steps where there
        are no chromosomes.
        Additionally appends a full per-agent snapshot when
        ``step % snapshot_interval == 0`` (so step 0 is always captured).

        The clustering cadence is controlled by ``clustering_interval``
        (defaults to ``snapshot_interval`` when not set).  A finer
        ``clustering_interval`` lets speciation track rapid population
        changes without writing full per-agent snapshot payloads.

        Args:
            environment: The live environment object.
            step: The current simulation step number.
            extra_fields: Optional dict of additional key/value pairs to merge
                into the trajectory record (e.g. telemetry computed by the
                runner such as ``mean_reproduction_cost`` or
                ``realized_birth_rate``).
        """
        alive_agents = list(environment.alive_agent_objects)
        chromosomes: List[HyperparameterChromosome] = [
            agent.hyperparameter_chromosome
            for agent in alive_agents
            if getattr(agent, "hyperparameter_chromosome", None) is not None
        ]
        gene_stats = compute_gene_statistics(chromosomes, evolvable_only=True)

        is_snapshot_step = step % self._snapshot_interval == 0

        # Determine effective clustering cadence: explicit clustering_interval
        # takes priority; when absent fall back to snapshot_interval so that
        # the default behaviour is unchanged.
        effective_clustering_interval = (
            self._clustering_interval
            if self._clustering_interval is not None
            else self._snapshot_interval
        )
        is_clustering_step = step % effective_clustering_interval == 0

        # Run speciation at clustering steps when enabled (even if trajectory file I/O is off).
        if self._enable_speciation and is_clustering_step:
            self._update_speciation(chromosomes, step)

        if self._trajectory_handle is None:
            return

        trajectory_record: Dict[str, Any] = {
            "step": step,
            "n_alive": len(alive_agents),
            "n_with_chromosome": len(chromosomes),
            "gene_stats": gene_stats,
        }
        if self._enable_speciation:
            trajectory_record["speciation_index"] = self._cached_speciation_index
            if is_snapshot_step and chromosomes and self._cached_speciation_quality is not None:
                trajectory_record["speciation_quality"] = self._cached_speciation_quality

        if extra_fields:
            _RESERVED_TRAJECTORY_KEYS = {"step", "n_alive", "n_with_chromosome", "gene_stats"}
            if self._enable_speciation:
                _RESERVED_TRAJECTORY_KEYS = _RESERVED_TRAJECTORY_KEYS | {
                    "speciation_index", "speciation_quality"
                }
            collisions = _RESERVED_TRAJECTORY_KEYS & extra_fields.keys()
            if collisions:
                raise ValueError(
                    f"extra_fields contains reserved trajectory key(s): {sorted(collisions)}. "
                    "Use a different name or nest telemetry under a sub-dict."
                )
            trajectory_record.update(extra_fields)
        self._trajectory_handle.write(json.dumps(trajectory_record) + "\n")

        if self._snapshot_handle is not None and is_snapshot_step:
            agents_payload: List[Dict[str, Any]] = []
            for agent in alive_agents:
                chromosome = getattr(agent, "hyperparameter_chromosome", None)
                if chromosome is None:
                    continue
                agents_payload.append(
                    {
                        "agent_id": getattr(agent, "agent_id", None),
                        "agent_type": getattr(agent, "agent_type", None),
                        "generation": getattr(agent, "generation", None),
                        "parent_ids": _read_parent_ids(agent),
                        "chromosome": {
                            gene.name: gene.value for gene in chromosome.genes
                        },
                    }
                )
            snapshot_record = {"step": step, "agents": agents_payload}
            self._snapshot_handle.write(json.dumps(snapshot_record) + "\n")

    # ------------------------------------------------------------------
    # Speciation helpers
    # ------------------------------------------------------------------

    def _update_speciation(
        self,
        chromosomes: List[HyperparameterChromosome],
        step: int,
    ) -> None:
        """Run cluster detection and update cached speciation state.

        Called at every clustering step when ``enable_speciation=True``.
        Writes cluster lineage records to ``cluster_lineage.jsonl`` when an
        output directory is configured.
        """
        if not chromosomes:
            self._cached_speciation_index = 0.0
            self._cached_speciation_quality = None
            self._prev_cluster_records = []
            self._cluster_id_counter = 0
            return

        # Extract evolvable gene dicts
        chrom_dicts = [
            {gene.name: gene.value for gene in c.genes if gene.evolvable}
            for c in chromosomes
        ]

        try:
            if self._speciation_algorithm == "gmm":
                result = detect_clusters_gmm(
                    chrom_dicts,
                    max_k=self._speciation_max_k,
                    seed=self._speciation_seed,
                    scaler=self._speciation_scaler,
                )
            else:
                result = detect_clusters_dbscan(
                    chrom_dicts,
                    scaler=self._speciation_scaler,
                    auto_tune=self._speciation_dbscan_auto_tune,
                    auto_tune_percentile=self._speciation_dbscan_auto_tune_percentile,
                )

            stability_score = None
            if self._speciation_include_stability:
                stability_score = compute_speciation_stability_score(
                    chrom_dicts,
                    algorithm=self._speciation_algorithm,
                    scaler=self._speciation_scaler,
                    gmm_max_k=self._speciation_max_k,
                    gmm_seed=self._speciation_seed,
                    dbscan_auto_tune=self._speciation_dbscan_auto_tune,
                    dbscan_auto_tune_percentile=self._speciation_dbscan_auto_tune_percentile,
                    n_perturbations=self._speciation_stability_perturbations,
                    noise_std=self._speciation_stability_noise_std,
                    random_seed=self._speciation_stability_seed,
                )

            bundle = compute_speciation_quality_bundle(
                result,
                stability_score=stability_score,
            )
            self._cached_speciation_index = bundle.speciation_index
            self._cached_speciation_quality = {
                "speciation_index": bundle.speciation_index,
                "raw_silhouette": bundle.raw_silhouette,
                "noise_fraction": bundle.noise_fraction,
                "cluster_size_entropy": bundle.cluster_size_entropy,
                "n_clusters": bundle.n_clusters,
                "computed_at_step": step,
            }
            if bundle.stability_score is not None:
                self._cached_speciation_quality["stability_score"] = bundle.stability_score

            # Persist cluster lineage
            if result.k == 0:
                # No clusters (e.g. DBSCAN all-noise snapshot): reset lineage
                # state so later clusters are treated as new founding lineages.
                self._prev_cluster_records = []
                self._cluster_id_counter = 0
                return

            if self._speciation_lineage_matcher == "hungarian":
                new_records, self._cluster_id_counter = match_clusters_hungarian(
                    self._prev_cluster_records,
                    result.centroids,
                    result.sizes,
                    result.gene_names,
                    step,
                    max_distance=self._speciation_lineage_max_distance,
                    id_counter_start=self._cluster_id_counter,
                )
            else:
                new_records, self._cluster_id_counter = match_clusters_greedy(
                    self._prev_cluster_records,
                    result.centroids,
                    result.sizes,
                    result.gene_names,
                    step,
                    max_distance=self._speciation_lineage_max_distance,
                    id_counter_start=self._cluster_id_counter,
                )
            self._prev_cluster_records = new_records

            if self._cluster_lineage_handle is not None:
                for rec in new_records:
                    row: Dict[str, Any] = {
                        "step": rec.step,
                        "cluster_id": rec.cluster_id,
                        "centroid": rec.centroid,
                        "size": rec.size,
                        "parent_cluster_id": rec.parent_cluster_id,
                        "transition_type": rec.transition_type,
                        "parent_cluster_ids": rec.parent_cluster_ids,
                        "lineage_matcher": self._speciation_lineage_matcher,
                        "lineage_max_distance": self._speciation_lineage_max_distance,
                        "scaler": result.scaler,
                        "dbscan_params": result.dbscan_params,
                    }
                    self._cluster_lineage_handle.write(json.dumps(row) + "\n")
        except Exception as exc:
            # Non-fatal: log and keep previous cached value
            warnings.warn(
                f"GeneTrajectoryLogger: speciation update failed at step {step}: {exc!r}",
                RuntimeWarning,
                stacklevel=4,
            )

    def close(self) -> None:
        """Flush and close all file handles; safe to call multiple times."""
        for attr in ("_trajectory_handle", "_snapshot_handle", "_cluster_lineage_handle"):
            handle = getattr(self, attr)
            if handle is not None:
                try:
                    handle.flush()
                    handle.close()
                except Exception:
                    # Best-effort teardown: ignore flush/close failures to keep shutdown non-fatal.
                    pass
                setattr(self, attr, None)


def _read_parent_ids(agent: Any) -> List[str]:
    """Best-effort extraction of an agent's parent_ids from its state manager."""
    state = getattr(agent, "state", None)
    if state is None:
        return []
    inner = getattr(state, "_state", None)
    parent_ids = getattr(inner, "parent_ids", None) if inner is not None else None
    if not parent_ids:
        return []
    return list(parent_ids)

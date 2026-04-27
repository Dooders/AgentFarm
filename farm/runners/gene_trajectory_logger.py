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
  snapshot step with fields ``step``, ``cluster_id``, ``centroid``, ``size``,
  and ``parent_cluster_id``.

When ``output_dir`` is ``None``, JSONL files are not written; speciation
state is still updated in memory when ``enable_speciation=True``, allowing
the runner to use the logger unconditionally regardless of persisted artifacts.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Optional, TextIO

from farm.analysis.speciation.compute import (
    VALID_SCALERS,
    compute_speciation_index,
    detect_clusters_dbscan,
    detect_clusters_gmm,
    match_clusters_greedy,
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
        When ``True``, run cluster detection at every snapshot step and
        include a ``speciation_index`` field in every trajectory record.
        Also writes ``cluster_lineage.jsonl``.  Default ``False``.
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
    """

    TRAJECTORY_FILENAME = "intrinsic_gene_trajectory.jsonl"
    SNAPSHOT_FILENAME = "intrinsic_gene_snapshots.jsonl"
    CLUSTER_LINEAGE_FILENAME = "cluster_lineage.jsonl"

    def __init__(
        self,
        output_dir: Optional[str],
        snapshot_interval: int,
        *,
        enable_speciation: bool = False,
        speciation_algorithm: str = "gmm",
        speciation_max_k: int = 5,
        speciation_seed: int = 0,
        speciation_scaler: str = "none",
    ) -> None:
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be at least 1.")
        if speciation_algorithm not in ("gmm", "dbscan"):
            raise ValueError("speciation_algorithm must be 'gmm' or 'dbscan'.")
        if speciation_scaler not in VALID_SCALERS:
            raise ValueError(
                f"speciation_scaler must be one of {VALID_SCALERS!r}; "
                f"got {speciation_scaler!r}."
            )

        if enable_speciation:
            try:
                import sklearn  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "GeneTrajectoryLogger: enable_speciation=True requires scikit-learn. "
                    "Install it with: pip install scikit-learn"
                ) from exc
        self._output_dir = output_dir
        self._snapshot_interval = snapshot_interval
        self._enable_speciation = enable_speciation
        self._speciation_algorithm = speciation_algorithm
        self._speciation_max_k = speciation_max_k
        self._speciation_seed = speciation_seed
        self._speciation_scaler = speciation_scaler

        self._trajectory_handle: Optional[TextIO] = None
        self._snapshot_handle: Optional[TextIO] = None
        self._cluster_lineage_handle: Optional[TextIO] = None

        # Cached speciation state between snapshot steps
        self._cached_speciation_index: float = 0.0
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
        ``speciation_index`` field updated at every snapshot step.
        Additionally appends a full per-agent snapshot when
        ``step % snapshot_interval == 0`` (so step 0 is always captured).

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

        # Run speciation at snapshot steps when enabled (even if trajectory file I/O is off).
        if self._enable_speciation and is_snapshot_step:
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

        if extra_fields:
            _RESERVED_TRAJECTORY_KEYS = {"step", "n_alive", "n_with_chromosome", "gene_stats"}
            if self._enable_speciation:
                _RESERVED_TRAJECTORY_KEYS = _RESERVED_TRAJECTORY_KEYS | {"speciation_index"}
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

        Called at every snapshot step when ``enable_speciation=True``.
        Writes cluster lineage records to ``cluster_lineage.jsonl`` when an
        output directory is configured.
        """
        if not chromosomes:
            self._cached_speciation_index = 0.0
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
                )

            self._cached_speciation_index = compute_speciation_index(result)

            # Persist cluster lineage
            if result.k == 0:
                # No clusters (e.g. DBSCAN all-noise snapshot): reset lineage
                # state so later clusters are treated as new founding lineages.
                self._prev_cluster_records = []
                self._cluster_id_counter = 0
                return

            new_records, self._cluster_id_counter = match_clusters_greedy(
                self._prev_cluster_records,
                result.centroids,
                result.sizes,
                result.gene_names,
                step,
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
                        "scaler": result.scaler,
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

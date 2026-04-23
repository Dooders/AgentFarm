"""Per-step chromosome telemetry for intrinsic-evolution simulations.

Writes two append-only JSONL files alongside other run artifacts:

- ``intrinsic_gene_trajectory.jsonl`` -- one record per step containing
  aggregate per-gene statistics over alive agents.  Compact and cheap to
  append every step.
- ``intrinsic_gene_snapshots.jsonl`` -- one record every
  ``snapshot_interval`` steps containing each alive agent's full chromosome
  values.  Supports lineage analysis offline; bounded cost via the
  configurable cadence.

Both files are no-ops when ``output_dir`` is ``None``, allowing the logger to
be used unconditionally inside the runner regardless of whether the user asked
for persisted artifacts.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TextIO

from farm.core.hyperparameter_chromosome import (
    HyperparameterChromosome,
    compute_gene_statistics,
)


class GeneTrajectoryLogger:
    """Buffered JSONL logger for chromosome distributions and snapshots."""

    TRAJECTORY_FILENAME = "intrinsic_gene_trajectory.jsonl"
    SNAPSHOT_FILENAME = "intrinsic_gene_snapshots.jsonl"

    def __init__(self, output_dir: Optional[str], snapshot_interval: int) -> None:
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be at least 1.")
        self._output_dir = output_dir
        self._snapshot_interval = snapshot_interval
        self._trajectory_handle: Optional[TextIO] = None
        self._snapshot_handle: Optional[TextIO] = None
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

    def snapshot(self, environment: Any, step: int) -> None:
        """Record the chromosome distribution at ``step``.

        Always appends a one-line aggregate record to the trajectory file.
        Additionally appends a full per-agent snapshot when
        ``step % snapshot_interval == 0`` (so step 0 is always captured).
        """
        if self._trajectory_handle is None:
            return

        alive_agents = list(environment.alive_agent_objects)
        chromosomes: List[HyperparameterChromosome] = [
            agent.hyperparameter_chromosome
            for agent in alive_agents
            if getattr(agent, "hyperparameter_chromosome", None) is not None
        ]
        gene_stats = compute_gene_statistics(chromosomes, evolvable_only=True)
        trajectory_record: Dict[str, Any] = {
            "step": step,
            "n_alive": len(alive_agents),
            "n_with_chromosome": len(chromosomes),
            "gene_stats": gene_stats,
        }
        self._trajectory_handle.write(json.dumps(trajectory_record) + "\n")

        if self._snapshot_handle is not None and step % self._snapshot_interval == 0:
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

    def close(self) -> None:
        """Flush and close both file handles; safe to call multiple times."""
        for attr in ("_trajectory_handle", "_snapshot_handle"):
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

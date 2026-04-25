"""Loader and summary helpers for ``intrinsic_gene_snapshots.jsonl``.

Parses the JSONL artifact written by
:class:`~farm.runners.gene_trajectory_logger.GeneTrajectoryLogger` and
constructs a :class:`~farm.analysis.phylogenetics.compute.PhylogeneticTree`
(really a DAG when two-parent reproduction is active) suitable for
lineage-tree visualisation.

Each line of ``intrinsic_gene_snapshots.jsonl`` has the shape::

    {
        "step": <int>,
        "agents": [
            {
                "agent_id": "abc",
                "agent_type": "system",
                "generation": 5,
                "parent_ids": ["parent1"],
                "chromosome": {"learning_rate": 0.012, "gamma": 0.99, ...}
            },
            ...
        ]
    }

The loader stitches together all snapshot steps into per-agent records where
``birth_time`` is the **first snapshot step** the agent appeared in and
``death_time`` is **None** when the agent was still alive at the final
snapshot step, or the **next snapshot step** after the last one it appeared
in (giving an approximate death window).

Public API
----------
- :func:`load_intrinsic_snapshots` – parse JSONL; return list of raw dicts.
- :func:`flatten_snapshots_to_agent_records` – flatten to per-agent records
  with inferred timing.
- :func:`build_intrinsic_lineage_dag` – load + flatten + build tree; the
  one-stop convenience function for notebooks.
- :func:`compute_surviving_lineage_count_over_time` – per-step lineage count.
- :func:`compute_lineage_depth_over_time` – per-step depth statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from farm.analysis.phylogenetics.compute import (
    PhylogeneticTree,
    build_phylogenetic_tree_from_records,
)
from farm.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Raw file loading
# ---------------------------------------------------------------------------


def load_intrinsic_snapshots(path: str | Path) -> List[Dict[str, Any]]:
    """Parse ``intrinsic_gene_snapshots.jsonl`` into a list of step dicts.

    Each element of the returned list corresponds to one line (snapshot) in
    the file and has the form ``{"step": <int>, "agents": [...]}``.  Lines
    that cannot be parsed as JSON are logged as warnings and skipped.

    Parameters
    ----------
    path:
        Path to the JSONL file (or the directory that contains it; in the
        latter case the function looks for
        ``intrinsic_gene_snapshots.jsonl`` inside that directory).

    Returns
    -------
    list of dict
        Snapshot dicts in file order (i.e. ascending step order for a
        well-formed run).
    """
    p = Path(path)
    if p.is_dir():
        p = p / "intrinsic_gene_snapshots.jsonl"

    if not p.exists():
        logger.warning("load_intrinsic_snapshots: file not found: %s", p)
        return []

    snapshots: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "load_intrinsic_snapshots: skipping malformed line %d in %s: %s",
                    lineno,
                    p,
                    exc,
                )
                continue
            if not isinstance(obj, dict):
                logger.warning(
                    "load_intrinsic_snapshots: skipping non-dict line %d in %s",
                    lineno,
                    p,
                )
                continue
            snapshots.append(obj)

    logger.info(
        "load_intrinsic_snapshots: loaded %d snapshot(s) from %s", len(snapshots), p
    )
    return snapshots


# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


def flatten_snapshots_to_agent_records(
    snapshots: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Flatten per-step snapshot dicts into deduplicated per-agent records.

    Each record in the returned list has the keys::

        agent_id, agent_type, generation, parent_ids,
        birth_time, death_time, chromosome

    ``birth_time`` is the first snapshot step the agent appeared in.

    ``death_time`` is:

    * ``None`` when the agent was present in the **final** snapshot step
      (still alive at the end of the run), or
    * the **next** snapshot step after its last appearance when it was absent
      from the final snapshot (bounding the death window from above).

    When only one snapshot step exists every agent has ``death_time = None``.

    Parameters
    ----------
    snapshots:
        List of raw snapshot dicts as returned by
        :func:`load_intrinsic_snapshots`.

    Returns
    -------
    list of dict
        One dict per unique ``agent_id``, suitable for passing to
        :func:`~farm.analysis.phylogenetics.compute.build_phylogenetic_tree_from_records`
        with ``id_key="agent_id"``.
    """
    if not snapshots:
        return []

    # Collect steps in sorted order
    steps: List[int] = sorted(
        {int(s.get("step", 0)) for s in snapshots if isinstance(s.get("step"), (int, float))}
    )
    if not steps:
        return []

    final_step = steps[-1]

    # Build step -> next_step map (for death_time approximation)
    next_step: Dict[int, int] = {}
    for i, step in enumerate(steps[:-1]):
        next_step[step] = steps[i + 1]

    # Index snapshots by step
    snap_by_step: Dict[int, Dict[str, Any]] = {}
    for snap in snapshots:
        s = int(snap.get("step", 0))
        snap_by_step[s] = snap

    # Per-agent tracking: first_step, last_step, last_record
    first_seen: Dict[str, int] = {}
    last_seen: Dict[str, int] = {}
    last_record: Dict[str, Dict[str, Any]] = {}

    for step in steps:
        snap = snap_by_step.get(step, {})
        agents = snap.get("agents", [])
        if not isinstance(agents, list):
            continue
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            agent_id = agent.get("agent_id")
            if agent_id is None:
                continue
            agent_id = str(agent_id)
            if agent_id not in first_seen:
                first_seen[agent_id] = step
            last_seen[agent_id] = step
            last_record[agent_id] = agent

    records: List[Dict[str, Any]] = []
    for agent_id, agent_rec in last_record.items():
        birth_time = first_seen[agent_id]
        ls = last_seen[agent_id]
        if ls == final_step:
            death_time: Optional[int] = None
        else:
            # Approximate: agent died before the next snapshot after its last appearance
            death_time = next_step.get(ls, ls + 1)

        # Normalise parent_ids: filter out None
        raw_parents = agent_rec.get("parent_ids", [])
        if isinstance(raw_parents, (list, tuple)):
            parent_ids = [str(p) for p in raw_parents if p is not None]
        else:
            parent_ids = []

        records.append(
            {
                "agent_id": agent_id,
                "agent_type": agent_rec.get("agent_type"),
                "generation": agent_rec.get("generation"),
                "parent_ids": parent_ids,
                "birth_time": birth_time,
                "death_time": death_time,
                "chromosome": agent_rec.get("chromosome") or {},
            }
        )

    return records


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------


def build_intrinsic_lineage_dag(
    path: str | Path,
    *,
    max_depth: Optional[int] = None,
) -> PhylogeneticTree:
    """Load ``intrinsic_gene_snapshots.jsonl`` and return a lineage DAG.

    This is the one-stop convenience function for notebooks and scripts:

    1. Loads the JSONL file (or finds it in a directory).
    2. Flattens per-step snapshots into per-agent records with inferred
       ``birth_time`` / ``death_time``.
    3. Calls
       :func:`~farm.analysis.phylogenetics.compute.build_phylogenetic_tree_from_records`
       with ``id_key="agent_id"``.

    Parameters
    ----------
    path:
        Path to the JSONL file **or** the run output directory that contains
        it.
    max_depth:
        Optional depth cap forwarded to
        :func:`~farm.analysis.phylogenetics.compute.build_phylogenetic_tree_from_records`.

    Returns
    -------
    PhylogeneticTree
        The built DAG.  :attr:`~farm.analysis.phylogenetics.compute.PhylogeneticTree.is_dag`
        will be ``True`` when any agent has two parents (crossover active).
    """
    snapshots = load_intrinsic_snapshots(path)
    if not snapshots:
        logger.warning("build_intrinsic_lineage_dag: no snapshots found at %s", path)
        return PhylogeneticTree(nodes={}, roots=[], is_dag=False)

    records = flatten_snapshots_to_agent_records(snapshots)
    if not records:
        logger.warning(
            "build_intrinsic_lineage_dag: no agent records after flattening at %s", path
        )
        return PhylogeneticTree(nodes={}, roots=[], is_dag=False)

    return build_phylogenetic_tree_from_records(
        records, id_key="agent_id", max_depth=max_depth
    )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def compute_surviving_lineage_count_over_time(
    tree: PhylogeneticTree,
    snapshots: Sequence[Dict[str, Any]],
) -> List[Tuple[int, int]]:
    """Return the number of distinct founder lineages alive at each snapshot.

    For each snapshot step *S* the function looks at the set of agents alive
    at that step (the ``"agents"`` list in the snapshot dict), traces each
    agent up to its founder via the lineage DAG, and counts distinct founders.

    Parameters
    ----------
    tree:
        A :class:`~farm.analysis.phylogenetics.compute.PhylogeneticTree`
        built from the same snapshots (e.g. via
        :func:`build_intrinsic_lineage_dag`).
    snapshots:
        Raw snapshot dicts as returned by :func:`load_intrinsic_snapshots`.

    Returns
    -------
    list of (step, count) tuples
        Sorted by step.  ``count`` is the number of distinct founder lineages
        represented at that step.  Returns an empty list when there are no
        snapshots or the tree is empty.
    """
    if not snapshots or not tree.nodes:
        return []

    result: List[Tuple[int, int]] = []
    for snap in sorted(snapshots, key=lambda s: int(s.get("step", 0))):
        step = int(snap.get("step", 0))
        agents_at_step = snap.get("agents", [])
        if not isinstance(agents_at_step, list):
            result.append((step, 0))
            continue

        founders_seen: set[str] = set()
        for agent in agents_at_step:
            if not isinstance(agent, dict):
                continue
            agent_id = agent.get("agent_id")
            if agent_id is None:
                continue
            agent_id = str(agent_id)
            founder = _trace_to_founder(tree, agent_id)
            if founder is not None:
                founders_seen.add(founder)

        result.append((step, len(founders_seen)))

    return result


def compute_lineage_depth_over_time(
    tree: PhylogeneticTree,
    snapshots: Sequence[Dict[str, Any]],
) -> List[Tuple[int, int, float]]:
    """Return lineage-depth statistics at each snapshot step.

    At each snapshot step, only considers agents alive at that step (present
    in the snapshot's ``"agents"`` list) and reports their depths in the full
    lineage tree.

    Parameters
    ----------
    tree:
        A :class:`~farm.analysis.phylogenetics.compute.PhylogeneticTree`
        built from the same snapshots.
    snapshots:
        Raw snapshot dicts as returned by :func:`load_intrinsic_snapshots`.

    Returns
    -------
    list of (step, max_depth, mean_depth) tuples
        Sorted by step.  ``max_depth`` and ``mean_depth`` are 0 when there
        are no agents with known depth at that step.
    """
    if not snapshots or not tree.nodes:
        return []

    result: List[Tuple[int, int, float]] = []
    for snap in sorted(snapshots, key=lambda s: int(s.get("step", 0))):
        step = int(snap.get("step", 0))
        agents_at_step = snap.get("agents", [])
        if not isinstance(agents_at_step, list):
            result.append((step, 0, 0.0))
            continue

        depths: List[int] = []
        for agent in agents_at_step:
            if not isinstance(agent, dict):
                continue
            agent_id = str(agent.get("agent_id", ""))
            node = tree.nodes.get(agent_id)
            if node is not None and node.depth >= 0:
                depths.append(node.depth)

        if depths:
            result.append((step, max(depths), sum(depths) / len(depths)))
        else:
            result.append((step, 0, 0.0))

    return result


def extract_chromosomes_from_snapshots(
    snapshots: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Return a mapping from ``agent_id`` to chromosome dict (last-seen values).

    Parameters
    ----------
    snapshots:
        Raw snapshot dicts as returned by :func:`load_intrinsic_snapshots`.

    Returns
    -------
    dict
        ``{agent_id: {"learning_rate": 0.012, ...}}``  Values come from the
        **last** snapshot in which the agent appeared (most recent chromosome).
    """
    chromosomes: Dict[str, Dict[str, float]] = {}
    for snap in sorted(snapshots, key=lambda s: int(s.get("step", 0))):
        for agent in snap.get("agents", []):
            if not isinstance(agent, dict):
                continue
            agent_id = agent.get("agent_id")
            if agent_id is None:
                continue
            chrom = agent.get("chromosome")
            if isinstance(chrom, dict):
                chromosomes[str(agent_id)] = chrom
    return chromosomes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trace_to_founder(tree: PhylogeneticTree, node_id: str) -> Optional[str]:
    """Walk up single-parent projection to the root; return its id or None."""
    visited: set[str] = set()
    current = node_id
    while current in tree.nodes:
        if current in visited:
            break  # cycle guard
        visited.add(current)
        node = tree.nodes[current]
        if node.is_root:
            return current
        parents_in_tree = [p for p in node.parent_ids if p in tree.nodes]
        if not parents_in_tree:
            return current
        current = parents_in_tree[0]
    return None

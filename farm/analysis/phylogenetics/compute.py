"""Phylogenetic tree construction and analysis from GenomeId lineage.

Builds an in-memory tree (or DAG for two-parent reproduction) from agents in a
simulation database or from ``evolution_lineage.json`` records.  Supports:

* JSON export of the full tree/DAG structure.
* Newick export (single-parent spanning-tree projection) for tree-shaped
  structures; the DAG case is clearly documented below.
* Summary statistics: depth, branching factor, lineage survival.
* Deterministic node ordering and explicit handling of orphan and
  dual-parent cases.

Two-parent (DAG) note
---------------------
When agents reproduce with two parents the resulting structure is a DAG, not a
strict tree, because each node can have two in-edges.  This module preserves the
full DAG internally (``PhylogeneticTree.is_dag == True``) and offers a
*single-parent spanning-tree projection* for formats that require a tree
(e.g. Newick): the first element of each node's ``parent_ids`` list is
selected as the canonical parent.  The second parent edge is **dropped** in
the projection but is always available in ``PhylogeneticNode.parent_ids``.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from farm.analysis.genetics.utils import parse_parent_ids
from farm.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass
class PhylogeneticNode:
    """A single node in the phylogenetic tree/DAG.

    Attributes
    ----------
    node_id:
        Unique agent identifier.
    parent_ids:
        IDs of parent agents (empty for founders).  Two parents indicates
        sexual/dual-parent reproduction; the resulting graph is a DAG.
    children_ids:
        IDs of child agents whose ``parent_ids`` include this node.
        Populated by the builder; sorted for determinism.
    depth:
        BFS distance from the nearest root (founder) node.  ``-1`` for nodes
        unreachable from any root (should not occur in well-formed data).
    generation:
        Agent generation number when available; ``-1`` when unknown.
    birth_time:
        Simulation step when the agent was born; ``-1`` when unknown.
    death_time:
        Simulation step when the agent died; ``None`` for still-alive agents.
    is_root:
        ``True`` when the node has no parents within the dataset (i.e. it is a
        founder or an orphan whose parents are absent from the dataset).
    is_orphan:
        ``True`` when the node encodes parent IDs in its genome but none of
        those parents appear in the dataset.
    """

    node_id: str
    parent_ids: List[str]
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    generation: int = -1
    birth_time: int = -1
    death_time: Optional[int] = None
    is_root: bool = False
    is_orphan: bool = False


@dataclass
class PhylogeneticTreeSummary:
    """Summary statistics for a phylogenetic tree/DAG.

    Attributes
    ----------
    num_nodes:
        Total number of nodes.
    num_founders:
        Number of root/founder nodes (no parents, or orphans).
    num_orphans:
        Nodes with encoded parent IDs that are absent from the dataset.
    max_depth:
        Maximum BFS depth from any root.
    mean_depth:
        Mean BFS depth over all nodes.
    mean_branching_factor:
        Mean number of children per internal node (nodes with at least one
        child).  ``0.0`` for fully leaf-only graphs.
    is_dag:
        ``True`` when any node has two or more parents.
    num_surviving_lineages:
        Number of founder lineages with at least one node surviving to the
        final observed simulation step.  ``-1`` if no timing data is available.
    lineage_survival_rate:
        ``num_surviving_lineages / num_founders``, or ``0.0`` when there are
        no founders.
    num_lineages_at_final_step:
        Number of distinct founder lineages represented by agents alive at
        the final simulation step.  ``-1`` if no timing data is available.
    """

    num_nodes: int
    num_founders: int
    num_orphans: int
    max_depth: int
    mean_depth: float
    mean_branching_factor: float
    is_dag: bool
    num_surviving_lineages: int
    lineage_survival_rate: float
    num_lineages_at_final_step: int


class PhylogeneticTree:
    """In-memory phylogenetic tree/DAG built from agent lineage data.

    When agents reproduce with two parents the structure becomes a DAG rather
    than a strict tree.  This class preserves the full DAG internally but
    offers convenience methods for tree-shaped projections (e.g. Newick
    export) that reduce dual-parent nodes to a single-parent representation
    by always choosing the *first* listed parent.

    Parameters
    ----------
    nodes:
        Mapping from ``node_id`` to :class:`PhylogeneticNode`.
    roots:
        Ordered list of root node IDs (founders), deterministically sorted.
    is_dag:
        ``True`` when the structure contains any node with two or more parents.
    max_step:
        Optional maximum simulation step; used for lineage survival statistics.
    """

    def __init__(
        self,
        nodes: Dict[str, PhylogeneticNode],
        roots: List[str],
        *,
        is_dag: bool = False,
        max_step: Optional[int] = None,
    ) -> None:
        self.nodes = nodes
        self.roots = roots
        self.is_dag = is_dag
        self.max_step = max_step

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def summary(self) -> PhylogeneticTreeSummary:
        """Compute and return summary statistics for this tree/DAG."""
        if not self.nodes:
            return PhylogeneticTreeSummary(
                num_nodes=0,
                num_founders=len(self.roots),
                num_orphans=0,
                max_depth=0,
                mean_depth=0.0,
                mean_branching_factor=0.0,
                is_dag=self.is_dag,
                num_surviving_lineages=-1,
                lineage_survival_rate=0.0,
                num_lineages_at_final_step=-1,
            )

        depths = [node.depth for node in self.nodes.values() if node.depth >= 0]
        max_depth = max(depths) if depths else 0
        mean_depth = sum(depths) / len(depths) if depths else 0.0
        num_orphans = sum(1 for n in self.nodes.values() if n.is_orphan)

        internal = [n for n in self.nodes.values() if n.children_ids]
        mean_branching_factor = (
            sum(len(n.children_ids) for n in internal) / len(internal)
            if internal
            else 0.0
        )

        # Lineage survival (requires birth_time / death_time data)
        has_timing = any(
            n.birth_time >= 0 or n.death_time is not None
            for n in self.nodes.values()
        )
        if has_timing and self.max_step is not None:
            surviving_founders = self._compute_surviving_founders()
            num_surviving = len(surviving_founders)
            num_founders = len(self.roots)
            survival_rate = num_surviving / num_founders if num_founders else 0.0
            alive_at_final = {
                n.node_id
                for n in self.nodes.values()
                if n.birth_time >= 0
                and n.birth_time <= self.max_step
                and (n.death_time is None or n.death_time >= self.max_step)
            }
            lineage_ids_at_final: Set[str] = set()
            for nid in alive_at_final:
                founder = self._get_founder(nid)
                if founder:
                    lineage_ids_at_final.add(founder)
            num_at_final = len(lineage_ids_at_final)
        else:
            num_surviving = -1
            survival_rate = 0.0
            num_at_final = -1

        return PhylogeneticTreeSummary(
            num_nodes=len(self.nodes),
            num_founders=len(self.roots),
            num_orphans=num_orphans,
            max_depth=max_depth,
            mean_depth=mean_depth,
            mean_branching_factor=mean_branching_factor,
            is_dag=self.is_dag,
            num_surviving_lineages=num_surviving,
            lineage_survival_rate=survival_rate,
            num_lineages_at_final_step=num_at_final,
        )

    def _compute_surviving_founders(self) -> Set[str]:
        """Return the set of founders that have at least one surviving descendant."""
        if self.max_step is None:
            return set()
        surviving: Set[str] = set()
        for node in self.nodes.values():
            alive = (
                node.birth_time >= 0
                and node.birth_time <= self.max_step
                and (node.death_time is None or node.death_time >= self.max_step)
            )
            if alive:
                founder = self._get_founder(node.node_id)
                if founder:
                    surviving.add(founder)
        return surviving

    def _get_founder(self, node_id: str) -> Optional[str]:
        """Walk up to the root via single-parent projection and return its ID."""
        visited: Set[str] = set()
        current = node_id
        while current in self.nodes:
            if current in visited:
                break  # cycle guard
            visited.add(current)
            node = self.nodes[current]
            if node.is_root:
                return current
            parents_in_tree = [p for p in node.parent_ids if p in self.nodes]
            if not parents_in_tree:
                return current
            current = parents_in_tree[0]  # single-parent projection
        return None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full tree/DAG to a JSON-compatible dictionary.

        The ``is_dag`` flag signals consumers whether the structure contains
        dual-parent relationships.  Node order and ``children_ids`` lists are
        sorted for determinism.
        """
        return {
            "is_dag": self.is_dag,
            "max_step": self.max_step,
            "roots": self.roots,
            "nodes": {
                nid: {
                    "node_id": node.node_id,
                    "parent_ids": node.parent_ids,
                    "children_ids": sorted(node.children_ids),
                    "depth": node.depth,
                    "generation": node.generation,
                    "birth_time": node.birth_time,
                    "death_time": node.death_time,
                    "is_root": node.is_root,
                    "is_orphan": node.is_orphan,
                }
                for nid, node in sorted(self.nodes.items())
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return the tree/DAG serialized as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_newick(self) -> str:
        """Export the tree as a Newick string using single-parent projection.

        When the tree is a DAG (dual-parent reproduction) each node's second
        parent edge is dropped; only the first parent is retained.  This
        projects the DAG onto a spanning tree that is compatible with the
        Newick format.

        Orphan nodes (roots whose parents were absent from the dataset) are
        included as top-level taxa within a synthetic root group.  When there
        is a single real root, no wrapping is added.

        Returns
        -------
        str
            Newick string terminated with ``;``.
        """
        if not self.nodes:
            return "();"

        # Build spanning-tree child map: first parent *present in this tree*
        spanning_children: Dict[str, List[str]] = defaultdict(list)
        for nid, node in sorted(self.nodes.items()):  # sorted for determinism
            if not node.is_root and node.parent_ids:
                parents_in_tree = [p for p in node.parent_ids if p in self.nodes]
                if parents_in_tree:
                    spanning_children[parents_in_tree[0]].append(nid)

        def _subtree(nid: str, visited: Set[str]) -> str:
            if nid in visited:
                return nid  # cycle guard
            visited = visited | {nid}
            kids = sorted(spanning_children.get(nid, []))
            if not kids:
                return nid
            inner = ",".join(_subtree(k, visited) for k in kids)
            return f"({inner}){nid}"

        if len(self.roots) == 1:
            result = _subtree(self.roots[0], set())
        else:
            parts = [_subtree(r, set()) for r in sorted(self.roots)]
            result = "(" + ",".join(parts) + ")"

        return result + ";"


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _build_tree_from_nodes(
    node_map: Dict[str, PhylogeneticNode],
    *,
    max_depth: Optional[int] = None,
) -> PhylogeneticTree:
    """Shared builder: wire parent→child edges, assign depths, detect DAG."""
    # Pass 1: identify roots / orphans and wire children
    for nid, node in sorted(node_map.items()):  # sorted for determinism
        parents_in_tree = [p for p in node.parent_ids if p in node_map]
        if not node.parent_ids:
            node.is_root = True
        elif not parents_in_tree:
            node.is_root = True
            node.is_orphan = True
        else:
            for pid in parents_in_tree:
                if nid not in node_map[pid].children_ids:
                    node_map[pid].children_ids.append(nid)

    roots = sorted(nid for nid, n in node_map.items() if n.is_root)

    # Pass 2: BFS from roots to assign depths
    for node in node_map.values():
        node.depth = -1

    queue: deque[Tuple[str, int]] = deque()
    for r in roots:
        queue.append((r, 0))
        node_map[r].depth = 0

    visited: Set[str] = set(roots)
    pruned_reachable: Set[str] = set()

    def _mark_pruned_descendants(start_node_id: str) -> None:
        """Mark descendants skipped due to max_depth as intentionally pruned."""
        stack = [start_node_id]
        local_seen: Set[str] = set()
        while stack:
            current_id = stack.pop()
            if current_id in local_seen or current_id in visited:
                continue
            local_seen.add(current_id)
            pruned_reachable.add(current_id)
            stack.extend(node_map[current_id].children_ids)

    while queue:
        nid, depth = queue.popleft()
        node_map[nid].depth = depth
        if max_depth is not None and depth >= max_depth:
            # These descendants are reachable but deliberately pruned from depth
            # assignment for upper-tree analysis views.
            for child_id in node_map[nid].children_ids:
                _mark_pruned_descendants(child_id)
            continue
        for child_id in sorted(node_map[nid].children_ids):
            if child_id not in visited:
                visited.add(child_id)
                queue.append((child_id, depth + 1))

    # Nodes not reachable from any root (data integrity issue)
    for nid, node in node_map.items():
        if nid not in visited and nid not in pruned_reachable:
            logger.warning("phylogenetic_tree: unreachable node node_id=%s", nid)

    # Sort children for determinism
    for node in node_map.values():
        node.children_ids.sort()

    # max_step from timing data
    max_step: Optional[int] = None
    all_times: List[int] = (
        [n.death_time for n in node_map.values() if n.death_time is not None]
        + [n.birth_time for n in node_map.values() if n.birth_time >= 0]
    )
    if all_times:
        max_step = max(all_times)

    is_dag = any(len(n.parent_ids) > 1 for n in node_map.values())

    return PhylogeneticTree(
        nodes=node_map,
        roots=roots,
        is_dag=is_dag,
        max_step=max_step,
    )


# ---------------------------------------------------------------------------
# Public builder functions
# ---------------------------------------------------------------------------


def build_phylogenetic_tree(
    agents: Sequence[Any],
    *,
    max_depth: Optional[int] = None,
) -> PhylogeneticTree:
    """Build a phylogenetic tree/DAG from a collection of agent objects.

    Each agent must have at minimum an ``agent_id`` and a ``genome_id``
    attribute.  The optional attributes ``generation``, ``birth_time``, and
    ``death_time`` are used when present for richer summary statistics.

    Orphan handling
    ~~~~~~~~~~~~~~~
    Agents whose ``parent_ids`` (decoded from ``genome_id``) contain IDs
    absent from the dataset are treated as roots with ``is_orphan=True``.

    Parameters
    ----------
    agents:
        Iterable of agent-like objects.
    max_depth:
        Optional depth cap for BFS traversal.  Nodes beyond ``max_depth`` are
        included in the tree but BFS stops there; useful for large simulations
        where you only want the upper portion of the tree.

    Returns
    -------
    PhylogeneticTree
    """
    def _coerce_int(value: Any, *, default: int, field_name: str, node_id: str) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(
                "build_phylogenetic_tree: invalid %s=%r for node_id=%s; using default=%s",
                field_name,
                value,
                node_id,
                default,
            )
            return default

    def _coerce_optional_int(
        value: Any, *, field_name: str, node_id: str
    ) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(
                "build_phylogenetic_tree: invalid %s=%r for node_id=%s; using None",
                field_name,
                value,
                node_id,
            )
            return None

    node_map: Dict[str, PhylogeneticNode] = {}
    for agent in agents:
        agent_id = str(agent.agent_id)
        genome_id = getattr(agent, "genome_id", None)
        if genome_id is not None:
            try:
                parent_ids: List[str] = parse_parent_ids(str(genome_id))
            except Exception:
                parent_ids = []
        else:
            parent_ids = []

        generation = getattr(agent, "generation", None)
        birth_time = getattr(agent, "birth_time", None)
        death_time = getattr(agent, "death_time", None)

        node_map[agent_id] = PhylogeneticNode(
            node_id=agent_id,
            parent_ids=parent_ids,
            generation=_coerce_int(
                generation, default=-1, field_name="generation", node_id=agent_id
            ),
            birth_time=_coerce_int(
                birth_time, default=-1, field_name="birth_time", node_id=agent_id
            ),
            death_time=_coerce_optional_int(
                death_time, field_name="death_time", node_id=agent_id
            ),
        )

    return _build_tree_from_nodes(node_map, max_depth=max_depth)


def build_phylogenetic_tree_from_records(
    records: Sequence[Dict[str, Any]],
    *,
    id_key: str = "candidate_id",
    parent_ids_key: str = "parent_ids",
    generation_key: str = "generation",
) -> PhylogeneticTree:
    """Build a phylogenetic tree/DAG from evolution-lineage JSON records.

    Accepts the list produced by
    :class:`~farm.runners.evolution_experiment.EvolutionExperiment` and
    written to ``evolution_lineage.json``.

    The function also accepts per-agent snapshot records from
    ``intrinsic_gene_snapshots.jsonl`` when the records have an ``"agent_id"``
    key; the ``id_key`` parameter can be adjusted accordingly.

    Parameters
    ----------
    records:
        List of dicts, each with at least ``candidate_id`` (or the value of
        ``id_key``) and ``parent_ids`` (list of strings).
    id_key:
        Key for the node identifier.  Falls back to ``"agent_id"`` then
        ``"node_id"`` when the primary key is absent.
    parent_ids_key:
        Key for the parent ID list.  Defaults to ``"parent_ids"``.
    generation_key:
        Key for the generation number.  Defaults to ``"generation"``.

    Returns
    -------
    PhylogeneticTree
    """
    def _coerce_int(value: Any, *, default: int, field_name: str, node_id: str) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(
                "build_phylogenetic_tree_from_records: invalid %s=%r for node_id=%s; using default=%s",
                field_name,
                value,
                node_id,
                default,
            )
            return default

    def _coerce_optional_int(
        value: Any, *, field_name: str, node_id: str
    ) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(
                "build_phylogenetic_tree_from_records: invalid %s=%r for node_id=%s; using None",
                field_name,
                value,
                node_id,
            )
            return None

    node_map: Dict[str, PhylogeneticNode] = {}

    for record in records:
        raw_id = (
            record.get(id_key)
            or record.get("agent_id")
            or record.get("node_id")
        )
        if raw_id is None:
            logger.warning(
                "build_phylogenetic_tree_from_records: record missing id_key=%r; skipping",
                id_key,
            )
            continue
        node_id = str(raw_id)

        raw_parents = record.get(parent_ids_key, [])
        if isinstance(raw_parents, (list, tuple)):
            parent_ids = [str(p) for p in raw_parents if p is not None]
        else:
            parent_ids = []

        generation = _coerce_int(
            record.get(generation_key, None),
            default=-1,
            field_name=generation_key,
            node_id=node_id,
        )
        birth_time = _coerce_int(
            record.get("birth_time", None),
            default=-1,
            field_name="birth_time",
            node_id=node_id,
        )
        death_time = _coerce_optional_int(
            record.get("death_time", None),
            field_name="death_time",
            node_id=node_id,
        )

        node_map[node_id] = PhylogeneticNode(
            node_id=node_id,
            parent_ids=parent_ids,
            generation=generation,
            birth_time=birth_time,
            death_time=death_time,
        )

    return _build_tree_from_nodes(node_map)

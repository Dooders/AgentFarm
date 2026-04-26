"""Phylogenetics Analysis Package

Provides tools for building, serialising, and visualising phylogenetic trees
(and DAGs for dual-parent reproduction) from agent lineage data stored in a
simulation database or in ``evolution_lineage.json`` artifacts.

Features
--------
- In-memory tree/DAG builder with deterministic node ordering
- Explicit handling of orphan nodes (missing parents) and dual-parent cases
- JSON export (full tree/DAG) and Newick export (single-parent projection)
- Summary statistics: depth, branching factor, lineage survival
- Matplotlib visualisation integrated with the ``farm/charts/``-style API
- Intrinsic-evolution loader: parse ``intrinsic_gene_snapshots.jsonl`` into a
  lineage DAG; per-gene coloured tree plot; surviving-lineage and depth
  statistics over time.

Quick Start::

    >>> from farm.analysis.phylogenetics import (
    ...     build_phylogenetic_tree,
    ...     build_phylogenetic_tree_from_records,
    ...     PhylogeneticTree,
    ...     PhylogeneticNode,
    ...     PhylogeneticTreeSummary,
    ...     analyze_phylogenetics,
    ...     plot_phylogenetic_tree,
    ... )

Intrinsic-evolution quick start::

    >>> from farm.analysis.phylogenetics import (
    ...     build_intrinsic_lineage_dag,
    ...     plot_intrinsic_lineage_tree,
    ...     compute_surviving_lineage_count_over_time,
    ...     compute_lineage_depth_over_time,
    ...     load_intrinsic_snapshots,
    ...     extract_chromosomes_from_snapshots,
    ... )

Two-parent (DAG) note
---------------------
When the simulation uses sexual reproduction each agent can have two parents,
making the structure a DAG rather than a strict tree.  The ``is_dag`` flag on
:class:`PhylogeneticTree` signals this.  Newick export uses a single-parent
spanning-tree projection (first parent wins); the JSON export preserves the
full DAG.
"""

from farm.analysis.phylogenetics.compute import (
    PhylogeneticNode,
    PhylogeneticTree,
    PhylogeneticTreeSummary,
    build_phylogenetic_tree,
    build_phylogenetic_tree_from_records,
)
from farm.analysis.phylogenetics.analyze import analyze_phylogenetics
from farm.analysis.phylogenetics.plot import plot_phylogenetic_tree, plot_intrinsic_lineage_tree
from farm.analysis.phylogenetics.module import PhylogeneticsModule, phylogenetics_module
from farm.analysis.phylogenetics.intrinsic_loader import (
    load_intrinsic_snapshots,
    flatten_snapshots_to_agent_records,
    build_intrinsic_lineage_dag,
    compute_surviving_lineage_count_over_time,
    compute_lineage_depth_over_time,
    extract_chromosomes_from_snapshots,
    trace_to_founder,
)

__all__ = [
    # Data structures
    "PhylogeneticNode",
    "PhylogeneticTree",
    "PhylogeneticTreeSummary",
    # Builder functions
    "build_phylogenetic_tree",
    "build_phylogenetic_tree_from_records",
    # Analysis
    "analyze_phylogenetics",
    # Visualisation
    "plot_phylogenetic_tree",
    "plot_intrinsic_lineage_tree",
    # Intrinsic-evolution loader & summaries
    "load_intrinsic_snapshots",
    "flatten_snapshots_to_agent_records",
    "build_intrinsic_lineage_dag",
    "compute_surviving_lineage_count_over_time",
    "compute_lineage_depth_over_time",
    "extract_chromosomes_from_snapshots",
    "trace_to_founder",
    # Module
    "PhylogeneticsModule",
    "phylogenetics_module",
]

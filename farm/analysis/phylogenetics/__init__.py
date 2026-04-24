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
from farm.analysis.phylogenetics.plot import plot_phylogenetic_tree
from farm.analysis.phylogenetics.module import PhylogeneticsModule, phylogenetics_module

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
    # Module
    "PhylogeneticsModule",
    "phylogenetics_module",
]

"""Phylogenetics analysis functions.

High-level analysis functions that operate on
:class:`~farm.analysis.phylogenetics.compute.PhylogeneticTree` objects.
"""

from __future__ import annotations

from typing import Any, Dict

from farm.analysis.phylogenetics.compute import PhylogeneticTree
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def analyze_phylogenetics(df: PhylogeneticTree) -> Dict[str, Any]:
    """Return a flat summary-statistics dictionary for a phylogenetic tree/DAG.

    Parameters
    ----------
    df:
        :class:`~farm.analysis.phylogenetics.compute.PhylogeneticTree` from
        the module data processor (parameter name ``df`` for the analysis
        framework wrapper).

    Returns
    -------
    dict
        Keys mirror :class:`~farm.analysis.phylogenetics.compute.PhylogeneticTreeSummary`
        fields, with additional ``"is_dag"`` and ``"roots"`` entries.
    """
    if not df.nodes:
        return {
            "num_nodes": 0,
            "num_founders": 0,
            "num_orphans": 0,
            "max_depth": 0,
            "mean_depth": 0.0,
            "mean_branching_factor": 0.0,
            "is_dag": False,
            "num_surviving_lineages": -1,
            "lineage_survival_rate": 0.0,
            "num_lineages_at_final_step": -1,
            "roots": [],
        }

    s = df.summary()
    return {
        "num_nodes": s.num_nodes,
        "num_founders": s.num_founders,
        "num_orphans": s.num_orphans,
        "max_depth": s.max_depth,
        "mean_depth": s.mean_depth,
        "mean_branching_factor": s.mean_branching_factor,
        "is_dag": s.is_dag,
        "num_surviving_lineages": s.num_surviving_lineages,
        "lineage_survival_rate": s.lineage_survival_rate,
        "num_lineages_at_final_step": s.num_lineages_at_final_step,
        "roots": list(df.roots),
    }

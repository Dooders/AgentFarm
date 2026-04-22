"""
Genetics Analysis Functions

High-level analysis functions that operate on the normalized DataFrames
produced by the genetics compute layer.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pandas as pd

from farm.utils.logging import get_logger

logger = get_logger(__name__)


def analyze_genetics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute summary statistics for a population-genetics DataFrame.

    Accepts either a DataFrame produced by
    :func:`~farm.analysis.genetics.compute.build_agent_genetics_dataframe`
    (DB-backed, columns include ``generation`` and ``action_weights``) or one
    produced by
    :func:`~farm.analysis.genetics.compute.build_evolution_experiment_dataframe`
    (evolution-experiment-backed, columns include ``fitness`` and
    ``chromosome_values``).

    Parameters
    ----------
    df:
        Input DataFrame.  Empty DataFrames are handled gracefully.

    Returns
    -------
    dict
        Summary statistics appropriate for the detected source type.
    """
    if df.empty:
        return {"total_agents": 0}

    result: Dict[str, Any] = {"total_agents": len(df)}

    # --- DB-backed frame ---
    if "generation" in df.columns:
        result["generation_counts"] = df["generation"].value_counts().to_dict()
        result["max_generation"] = int(df["generation"].max())
        result["mean_generation"] = float(df["generation"].mean())

    if "parent_ids" in df.columns:
        result["pct_with_parents"] = float(
            (df["parent_ids"].apply(lambda p: len(p) > 0)).mean() * 100
        )

    if "action_weights" in df.columns:
        non_empty = df["action_weights"].apply(bool)
        result["pct_with_action_weights"] = float(non_empty.mean() * 100)

    # --- Evolution-experiment frame ---
    if "fitness" in df.columns:
        result["best_fitness"] = float(df["fitness"].max())
        result["mean_fitness"] = float(df["fitness"].mean())
        result["min_fitness"] = float(df["fitness"].min())

    if "chromosome_values" in df.columns and not df["chromosome_values"].empty:
        values_by_gene: Dict[str, list] = {}
        skipped_rows = 0
        skipped_values = 0
        for row in df["chromosome_values"]:
            if not isinstance(row, dict):
                skipped_rows += 1
                continue
            for gene, raw_value in row.items():
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError):
                    skipped_values += 1
                    continue
                if not math.isfinite(numeric_value):
                    skipped_values += 1
                    continue
                values_by_gene.setdefault(gene, []).append(numeric_value)

        if skipped_rows or skipped_values:
            logger.warning(
                "analyze_genetics: skipped malformed chromosome data rows=%d values=%d",
                skipped_rows,
                skipped_values,
            )

        gene_stats: Dict[str, Any] = {}
        for gene, values in sorted(values_by_gene.items()):
            series = pd.Series(values, dtype=float)
            if not series.empty:
                gene_stats[gene] = {
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                    "min": float(series.min()),
                    "max": float(series.max()),
                }
        result["gene_statistics"] = gene_stats

    return result

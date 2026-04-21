"""
Genetics Analysis Visualization

Placeholder visualization functions for the genetics analysis module.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from farm.analysis.common.context import AnalysisContext
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def plot_generation_distribution(df: pd.DataFrame, ctx: AnalysisContext, **kwargs: Any) -> Optional[Any]:
    """Plot the distribution of agents across generations.

    Parameters
    ----------
    df:
        DataFrame produced by
        :func:`~farm.analysis.genetics.compute.build_agent_genetics_dataframe`.
    ctx:
        Analysis context supplying output paths and a logger.
    """
    if df.empty or "generation" not in df.columns:
        logger.warning("plot_generation_distribution: no generation data available")
        return None

    try:
        import matplotlib.pyplot as plt

        gen_counts = df["generation"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(gen_counts.index.astype(str), gen_counts.values)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Agent count")
        ax.set_title("Agent count by generation")

        output_file = ctx.get_output_file("genetics_generation_distribution.png")
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved generation distribution plot to %s", output_file)
        return output_file
    except Exception as exc:
        logger.warning("plot_generation_distribution failed: %s", exc)
        return None


def plot_fitness_over_generations(df: pd.DataFrame, ctx: AnalysisContext, **kwargs: Any) -> Optional[Any]:
    """Plot mean and best fitness per generation for evolution-experiment data.

    Parameters
    ----------
    df:
        DataFrame produced by
        :func:`~farm.analysis.genetics.compute.build_evolution_experiment_dataframe`.
    ctx:
        Analysis context supplying output paths and a logger.
    """
    if df.empty or "fitness" not in df.columns or "generation" not in df.columns:
        logger.warning("plot_fitness_over_generations: no fitness/generation data available")
        return None

    try:
        import matplotlib.pyplot as plt

        gen_grouped = df.groupby("generation")["fitness"]
        mean_fitness = gen_grouped.mean()
        best_fitness = gen_grouped.max()

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(mean_fitness.index, mean_fitness.values, marker="o", label="Mean fitness")
        ax.plot(best_fitness.index, best_fitness.values, marker="s", linestyle="--", label="Best fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness")
        ax.set_title("Fitness over generations")
        ax.legend()

        output_file = ctx.get_output_file("genetics_fitness_over_generations.png")
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved fitness-over-generations plot to %s", output_file)
        return output_file
    except Exception as exc:
        logger.warning("plot_fitness_over_generations failed: %s", exc)
        return None

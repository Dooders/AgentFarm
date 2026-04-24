"""
Genetics Analysis Visualization

Placeholder visualization functions for the genetics analysis module.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd

from farm.analysis.common.context import AnalysisContext
from farm.utils.logging import get_logger

logger = get_logger(__name__)

_SAFE_OUTPUT_TOKEN_RE = re.compile(r"[^0-9A-Za-z_.-]+")


def _sanitize_output_token(value: str) -> str:
    """Normalize arbitrary labels into filesystem-safe filename fragments."""
    sanitized = _SAFE_OUTPUT_TOKEN_RE.sub("_", value).strip("_")
    return sanitized or "unknown"


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


def plot_marginal_fitness_effect(
    df: pd.DataFrame,
    gene: str,
    ctx: AnalysisContext,
    **kwargs: Any,
) -> Optional[Any]:
    """Scatter plot of gene value vs fitness with an OLS regression line.

    Visualises the marginal effect of a single gene on fitness across all
    candidates in the evolution DataFrame.

    Parameters
    ----------
    df:
        DataFrame with at least ``fitness`` (float) and ``chromosome_values``
        (dict) columns.  Typically produced by
        :func:`~farm.analysis.genetics.compute.build_evolution_experiment_dataframe`.
    gene:
        Name of the gene to plot on the x-axis.
    ctx:
        Analysis context supplying output paths and a logger.

    Returns
    -------
    str or None
        Path to the saved PNG file, or ``None`` on error / missing data.
    """
    if df.empty or "fitness" not in df.columns or "chromosome_values" not in df.columns:
        logger.warning("plot_marginal_fitness_effect: missing required columns")
        return None

    try:
        import matplotlib.pyplot as plt
        from scipy import stats as _stats

        gene_vals = []
        fit_vals = []
        for _, row in df.iterrows():
            chrom = row.get("chromosome_values")
            if not isinstance(chrom, dict) or gene not in chrom:
                continue
            try:
                gv = float(chrom[gene])
                fv = float(row["fitness"])
            except (TypeError, ValueError):
                continue
            gene_vals.append(gv)
            fit_vals.append(fv)

        if len(gene_vals) < 3:
            logger.warning("plot_marginal_fitness_effect: not enough data for gene %r", gene)
            return None

        import numpy as np

        gene_arr = np.array(gene_vals)
        fit_arr = np.array(fit_vals)
        slope, intercept, r_value, p_value, _ = _stats.linregress(gene_arr, fit_arr)
        x_line = np.linspace(gene_arr.min(), gene_arr.max(), 200)
        y_line = slope * x_line + intercept

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(gene_arr, fit_arr, alpha=0.5, s=20, label="Candidates")
        ax.plot(x_line, y_line, color="red", linewidth=1.5,
                label=f"OLS (r={r_value:.2f}, p={p_value:.3g})")
        ax.set_xlabel(gene)
        ax.set_ylabel("Fitness")
        ax.set_title(f"Marginal fitness effect: {gene}")
        ax.legend()

        safe_gene = _sanitize_output_token(gene)
        output_file = ctx.get_output_file(f"genetics_marginal_{safe_gene}.png")
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved marginal fitness effect plot to %s", output_file)
        return output_file
    except Exception as exc:
        logger.warning("plot_marginal_fitness_effect failed: %s", exc)
        return None


def plot_fitness_landscape_2d(
    df: pd.DataFrame,
    gene_i: str,
    gene_j: str,
    ctx: AnalysisContext,
    plot_type: str = "scatter",
    **kwargs: Any,
) -> Optional[Any]:
    """2D scatter or heatmap of two genes coloured by fitness.

    Parameters
    ----------
    df:
        DataFrame with ``fitness`` and ``chromosome_values`` columns.
    gene_i:
        Name of the gene for the x-axis.
    gene_j:
        Name of the gene for the y-axis.
    ctx:
        Analysis context supplying output paths and a logger.
    plot_type:
        ``"scatter"`` (default) – scatter plot where colour encodes fitness;
        ``"heatmap"`` – bin (gene_i, gene_j) into a 2D grid, colour = mean
        fitness per bin.

    Returns
    -------
    str or None
        Path to the saved PNG file, or ``None`` on error / missing data.
    """
    if df.empty or "fitness" not in df.columns or "chromosome_values" not in df.columns:
        logger.warning("plot_fitness_landscape_2d: missing required columns")
        return None

    try:
        import matplotlib.pyplot as plt
        import numpy as np

        xi_vals, xj_vals, fit_vals = [], [], []
        for _, row in df.iterrows():
            chrom = row.get("chromosome_values")
            if not isinstance(chrom, dict):
                continue
            if gene_i not in chrom or gene_j not in chrom:
                continue
            try:
                xi = float(chrom[gene_i])
                xj = float(chrom[gene_j])
                fv = float(row["fitness"])
            except (TypeError, ValueError):
                continue
            xi_vals.append(xi)
            xj_vals.append(xj)
            fit_vals.append(fv)

        if len(xi_vals) < 3:
            logger.warning(
                "plot_fitness_landscape_2d: not enough data for genes %r / %r",
                gene_i, gene_j,
            )
            return None

        xi_arr = np.array(xi_vals)
        xj_arr = np.array(xj_vals)
        fit_arr = np.array(fit_vals)

        fig, ax = plt.subplots(figsize=(8, 6))

        if plot_type not in {"scatter", "heatmap"}:
            logger.warning(
                "plot_fitness_landscape_2d: invalid plot_type %r; expected 'scatter' or 'heatmap'",
                plot_type,
            )
            return None

        if plot_type == "heatmap":
            bins = kwargs.get("bins", 20)
            h, xedges, yedges = np.histogram2d(xi_arr, xj_arr, bins=bins)
            # Mean fitness per bin
            fit_sum, _, _ = np.histogram2d(xi_arr, xj_arr, bins=bins, weights=fit_arr)
            with np.errstate(invalid="ignore"):
                mean_fit = np.where(h > 0, fit_sum / h, np.nan)
            im = ax.imshow(
                mean_fit.T,
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                aspect="auto",
                cmap="viridis",
            )
            fig.colorbar(im, ax=ax, label="Mean fitness")
        else:
            sc = ax.scatter(xi_arr, xj_arr, c=fit_arr, cmap="viridis", alpha=0.6, s=20)
            fig.colorbar(sc, ax=ax, label="Fitness")

        ax.set_xlabel(gene_i)
        ax.set_ylabel(gene_j)
        ax.set_title(f"Fitness landscape: {gene_i} × {gene_j}")

        safe_i = _sanitize_output_token(gene_i)
        safe_j = _sanitize_output_token(gene_j)
        output_file = ctx.get_output_file(f"genetics_landscape_{safe_i}_x_{safe_j}.png")
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved 2D fitness landscape plot to %s", output_file)
        return output_file
    except Exception as exc:
        logger.warning("plot_fitness_landscape_2d failed: %s", exc)
        return None

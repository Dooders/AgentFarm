"""
Genetics Analysis Visualization

Visualization functions for the genetics analysis module covering:
- Generation distribution and fitness trajectories
- Allele-frequency trajectories per locus
- Per-generation diversity (heterozygosity, Shannon entropy)
- Wright-Fisher neutral drift overlay vs. observed frequencies
- 2D fitness-landscape heatmap / scatter
- Conserved-run timeline per locus
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Set non-interactive backend before any pyplot import
import matplotlib.pyplot as plt  # noqa: E402 – must follow backend selection

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
    ctx: AnalysisContext,
    gene: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Path]:
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
        Name of the gene to plot on the x-axis. If omitted, the function logs
        and returns ``None`` (for example when invoked via a group without
        per-function kwargs).
    ctx:
        Analysis context supplying output paths and a logger.

    Returns
    -------
    Path or None
        Path to the saved PNG file, or ``None`` on error / missing data.
    """
    if df.empty or "fitness" not in df.columns or "chromosome_values" not in df.columns:
        logger.warning("plot_marginal_fitness_effect: missing required columns")
        return None

    if not gene:
        logger.warning("plot_marginal_fitness_effect: missing required gene name")
        return None

    try:
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
    ctx: AnalysisContext,
    gene_i: Optional[str] = None,
    gene_j: Optional[str] = None,
    plot_type: str = "scatter",
    **kwargs: Any,
) -> Optional[Path]:
    """2D scatter or heatmap of two genes coloured by fitness.

    Parameters
    ----------
    df:
        DataFrame with ``fitness`` and ``chromosome_values`` columns.
    gene_i:
        Name of the gene for the x-axis. If omitted (with ``gene_j``), the
        function logs and returns ``None``.
    gene_j:
        Name of the gene for the y-axis. If omitted (with ``gene_i``), the
        function logs and returns ``None``.
    ctx:
        Analysis context supplying output paths and a logger.
    plot_type:
        ``"scatter"`` (default) – scatter plot where colour encodes fitness;
        ``"heatmap"`` – bin (gene_i, gene_j) into a 2D grid, colour = mean
        fitness per bin.

    Returns
    -------
    Path or None
        Path to the saved PNG file, or ``None`` on error / missing data.
    """
    if df.empty or "fitness" not in df.columns or "chromosome_values" not in df.columns:
        logger.warning("plot_fitness_landscape_2d: missing required columns")
        return None

    if plot_type not in {"scatter", "heatmap"}:
        logger.warning(
            "plot_fitness_landscape_2d: invalid plot_type %r; expected 'scatter' or 'heatmap'",
            plot_type,
        )
        return None

    if not gene_i or not gene_j:
        logger.warning(
            "plot_fitness_landscape_2d: missing required gene_i / gene_j names",
        )
        return None

    try:
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


def plot_allele_frequency_trajectories(
    df: pd.DataFrame,
    ctx: AnalysisContext,
    loci: Optional[Sequence[str]] = None,
    max_loci: int = 6,
    **kwargs: Any,
) -> Optional[Path]:
    """Plot per-locus allele-frequency trajectories across generations.

    Accepts the tidy DataFrame produced by
    :func:`~farm.analysis.genetics.compute.compute_allele_frequency_timeseries`
    **or** a raw genetics DataFrame (with ``generation`` plus
    ``chromosome_values`` / ``action_weights`` columns), computing the
    allele-frequency timeseries internally when needed.

    One subplot is drawn per locus.  Continuous loci show the population
    ``__mean__`` trajectory; categorical loci show one line per allele.

    Parameters
    ----------
    df:
        Allele-frequency tidy frame or raw genetics DataFrame.
    ctx:
        Analysis context supplying output paths and a logger.
    loci:
        Optional subset of locus names to plot.  When omitted, all loci up
        to *max_loci* are shown.
    max_loci:
        Maximum number of subplots when *loci* is not supplied.

    Returns
    -------
    Path or None
        Path to the saved PNG file, or ``None`` on error / missing data.
    """
    if df.empty:
        logger.warning("plot_allele_frequency_trajectories: empty DataFrame")
        return None

    # If the frame already has allele-frequency columns use it directly;
    # otherwise derive from the raw genetics DataFrame.
    freq_df: pd.DataFrame
    if "locus" in df.columns and "allele" in df.columns and "frequency" in df.columns:
        freq_df = df
    elif "generation" in df.columns and (
        "chromosome_values" in df.columns or "action_weights" in df.columns
    ):
        from farm.analysis.genetics.compute import compute_allele_frequency_timeseries

        freq_df = compute_allele_frequency_timeseries(df)
    else:
        logger.warning(
            "plot_allele_frequency_trajectories: DataFrame lacks required columns; "
            "expected either (locus, allele, frequency) columns (allele-frequency tidy frame) "
            "or (generation, chromosome_values/action_weights) columns (raw genetics frame)"
        )
        return None

    if freq_df.empty:
        logger.warning("plot_allele_frequency_trajectories: no allele-frequency data")
        return None

    try:
        from farm.analysis.genetics.compute import ALLELE_MEAN

        # Determine loci to plot
        available_loci: List[str] = list(freq_df["locus"].unique())
        if loci:
            plot_loci = [loc for loc in loci if loc in available_loci]
        else:
            plot_loci = available_loci[:max_loci]

        if not plot_loci:
            logger.warning("plot_allele_frequency_trajectories: no loci to plot")
            return None

        n_cols = min(3, len(plot_loci))
        n_rows = (len(plot_loci) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            squeeze=False,
        )
        axes_flat = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]

        for ax_idx, locus in enumerate(plot_loci):
            ax = axes_flat[ax_idx]
            locus_df = freq_df[freq_df["locus"] == locus].copy()
            locus_type = locus_df["locus_type"].iloc[0] if not locus_df.empty else "unknown"

            if locus_type == "continuous":
                # Show __mean__ trajectory; skip __variance__ for clarity
                mean_df = locus_df[locus_df["allele"] == ALLELE_MEAN].sort_values("generation")
                if not mean_df.empty:
                    ax.plot(mean_df["generation"], mean_df["frequency"], marker="o", markersize=3)
                ax.set_ylabel("Population mean")
            else:
                # Categorical: one line per allele
                for allele, allele_df in locus_df.groupby("allele"):
                    allele_df = allele_df.sort_values("generation")
                    ax.plot(allele_df["generation"], allele_df["frequency"], marker=".", markersize=3, label=str(allele))
                if len(locus_df["allele"].unique()) <= 8:
                    ax.legend(fontsize=7, loc="best")
                ax.set_ylabel("Allele frequency")

            ax.set_xlabel("Generation")
            ax.set_title(f"{locus}\n({locus_type})", fontsize=9)
            ax.set_ylim(bottom=0)

        # Hide unused axes
        for ax in axes_flat[len(plot_loci):]:
            ax.set_visible(False)

        fig.suptitle("Allele-frequency trajectories", fontsize=11, y=1.02)
        fig.tight_layout()

        output_file = ctx.get_output_file("genetics_allele_frequency_trajectories.png")
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved allele-frequency trajectory plot to %s", output_file)
        return output_file
    except Exception as exc:
        logger.warning("plot_allele_frequency_trajectories failed: %s", exc)
        return None


def plot_diversity_over_time(
    df: pd.DataFrame,
    ctx: AnalysisContext,
    **kwargs: Any,
) -> Optional[Path]:
    """Plot per-generation genetic diversity: heterozygosity and Shannon entropy.

    Accepts either:
    * A raw genetics DataFrame (with ``generation`` and ``chromosome_values`` /
      ``action_weights`` columns) – diversity metrics are computed per
      generation internally, or
    * A tidy allele-frequency DataFrame (from
      :func:`~farm.analysis.genetics.compute.compute_allele_frequency_timeseries`)
      – Shannon entropy is derived from allele-frequency distributions.

    Parameters
    ----------
    df:
        Genetics DataFrame or allele-frequency tidy frame.
    ctx:
        Analysis context supplying output paths and a logger.

    Returns
    -------
    Path or None
        Path to the saved PNG file, or ``None`` on error / missing data.
    """
    if df.empty:
        logger.warning("plot_diversity_over_time: empty DataFrame")
        return None

    try:
        import numpy as np

        generations: List[int] = []
        heterozygosity_vals: List[float] = []
        entropy_vals: List[float] = []

        is_freq_df = (
            "locus" in df.columns
            and "allele" in df.columns
            and "frequency" in df.columns
            and "generation" in df.columns
        )
        has_raw_columns = "generation" in df.columns and (
            "chromosome_values" in df.columns or "action_weights" in df.columns
        )

        if is_freq_df:
            # Derive diversity metrics from allele-frequency tidy frame
            for gen, gen_df in df.groupby("generation"):
                cats = gen_df[gen_df["locus_type"] == "categorical"]
                het_vals_gen: List[float] = []
                ent_vals_gen: List[float] = []
                for locus, locus_df in cats.groupby("locus"):
                    freqs = locus_df["frequency"].values
                    freqs = freqs[np.isfinite(freqs)]
                    if freqs.size == 0 or freqs.sum() <= 0:
                        het_vals_gen.append(float("nan"))
                        ent_vals_gen.append(float("nan"))
                    else:
                        freqs = freqs / freqs.sum()
                        het_vals_gen.append(1.0 - float(np.sum(freqs ** 2)))
                        nonzero = freqs[freqs > 0]
                        ent_vals_gen.append(float(-np.sum(nonzero * np.log(nonzero))))
                generations.append(int(gen))
                heterozygosity_vals.append(float(np.mean(het_vals_gen)) if het_vals_gen else float("nan"))
                entropy_vals.append(float(np.mean(ent_vals_gen)) if ent_vals_gen else float("nan"))

        elif has_raw_columns:
            from farm.analysis.genetics.compute import (
                compute_population_diversity,
            )
            for gen, gen_df in df.groupby("generation"):
                try:
                    gen_int = int(float(gen))
                except (TypeError, ValueError):
                    continue
                diversity = compute_population_diversity(
                    gen_df.reset_index(drop=True),
                    compute_continuous_entropy=True,
                )
                generations.append(gen_int)
                heterozygosity_vals.append(diversity.mean_heterozygosity)
                entropy_vals.append(diversity.mean_shannon_entropy)
        else:
            logger.warning("plot_diversity_over_time: DataFrame lacks required columns")
            return None

        if not generations:
            logger.warning("plot_diversity_over_time: no generation data extracted")
            return None

        sort_idx = sorted(range(len(generations)), key=lambda i: generations[i])
        generations = [generations[i] for i in sort_idx]
        heterozygosity_vals = [heterozygosity_vals[i] for i in sort_idx]
        entropy_vals = [entropy_vals[i] for i in sort_idx]

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax_het, ax_ent = axes

        het_arr = np.array(heterozygosity_vals, dtype=float)
        ent_arr = np.array(entropy_vals, dtype=float)

        valid_het = np.isfinite(het_arr)
        if valid_het.any():
            ax_het.plot(
                [generations[i] for i in range(len(generations)) if valid_het[i]],
                het_arr[valid_het],
                marker="o", markersize=4, color="steelblue",
            )
        ax_het.set_ylabel("Expected heterozygosity")
        ax_het.set_title("Per-generation genetic diversity")
        ax_het.set_ylim(bottom=0)

        valid_ent = np.isfinite(ent_arr)
        if valid_ent.any():
            ax_ent.plot(
                [generations[i] for i in range(len(generations)) if valid_ent[i]],
                ent_arr[valid_ent],
                marker="s", markersize=4, color="darkorange",
            )
        ax_ent.set_ylabel("Shannon entropy (nats)")
        ax_ent.set_xlabel("Generation")
        ax_ent.set_ylim(bottom=0)

        fig.tight_layout()
        output_file = ctx.get_output_file("genetics_diversity_over_time.png")
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved diversity-over-time plot to %s", output_file)
        return output_file
    except Exception as exc:
        logger.warning("plot_diversity_over_time failed: %s", exc)
        return None


def plot_wright_fisher_overlay(
    df: pd.DataFrame,
    ctx: AnalysisContext,
    n_effective: Optional[int] = None,
    n_generations: Optional[int] = None,
    seed: Optional[int] = 42,
    loci: Optional[Sequence[str]] = None,
    max_loci: int = 4,
    **kwargs: Any,
) -> Optional[Path]:
    """Overlay observed allele-frequency trajectories against Wright-Fisher neutral drift.

    For each selected locus, computes the observed allele-frequency trajectory
    from *df* and draws it against a simulated Wright-Fisher neutral baseline
    so that deviations from neutral expectation are immediately visible.

    Parameters
    ----------
    df:
        Raw genetics DataFrame (``generation`` + ``chromosome_values`` /
        ``action_weights``) or allele-frequency tidy frame.
    ctx:
        Analysis context supplying output paths and a logger.
    n_effective:
        Effective population size for the Wright-Fisher simulation.  Derived
        from the first-generation population count when omitted.
    n_generations:
        Number of generations to simulate.  Derived from *df* when omitted.
    seed:
        RNG seed for reproducibility.
    loci:
        Subset of loci to plot.  The first *max_loci* are shown when omitted.
    max_loci:
        Maximum loci plotted when *loci* is not supplied.

    Returns
    -------
    Path or None
        Path to the saved PNG file, or ``None`` on error / missing data.
    """
    if df.empty:
        logger.warning("plot_wright_fisher_overlay: empty DataFrame")
        return None

    try:
        import numpy as np
        from farm.analysis.genetics.compute import (
            ALLELE_MEAN,
            compute_allele_frequency_timeseries,
            simulate_wright_fisher,
        )

        # Derive observed allele-frequency frame
        is_freq_df = (
            "locus" in df.columns
            and "allele" in df.columns
            and "frequency" in df.columns
        )
        if is_freq_df:
            freq_df = df
        elif "generation" in df.columns and (
            "chromosome_values" in df.columns or "action_weights" in df.columns
        ):
            freq_df = compute_allele_frequency_timeseries(df)
        else:
            logger.warning("plot_wright_fisher_overlay: DataFrame lacks required columns")
            return None

        if freq_df.empty:
            logger.warning("plot_wright_fisher_overlay: no allele-frequency data")
            return None

        gens_sorted = sorted(freq_df["generation"].unique())
        observed_generation_count = len(gens_sorted)
        if observed_generation_count < 2:
            logger.warning("plot_wright_fisher_overlay: need at least 2 distinct generations for WF overlay")
            return None

        # `simulate_wright_fisher(..., n_generations=...)` expects the number of
        # generations beyond generation 0, so observed generations [0, 1, 2]
        # correspond to n_gens=2.
        n_gens = n_generations if n_generations is not None else (max(gens_sorted) - min(gens_sorted))

        # Infer N_e from first generation count if not provided
        if n_effective is None:
            first_gen = min(gens_sorted)
            n_effective = int(freq_df[freq_df["generation"] == first_gen]["n_individuals"].max())
            n_effective = max(1, n_effective)

        # Select loci
        available_loci = list(freq_df["locus"].unique())
        if loci:
            plot_loci = [loc for loc in loci if loc in available_loci]
        else:
            # Prefer continuous loci for the overlay (mean trajectory)
            cont_loci = [
                loc for loc in available_loci
                if freq_df[freq_df["locus"] == loc]["locus_type"].iloc[0] == "continuous"
            ]
            plot_loci = (cont_loci or available_loci)[:max_loci]

        if not plot_loci:
            logger.warning("plot_wright_fisher_overlay: no loci to plot")
            return None

        n_cols = min(2, len(plot_loci))
        n_rows = (len(plot_loci) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4 * n_rows),
            squeeze=False,
        )
        axes_flat = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]

        for ax_idx, locus in enumerate(plot_loci):
            ax = axes_flat[ax_idx]
            locus_df = freq_df[freq_df["locus"] == locus].copy()
            locus_type = locus_df["locus_type"].iloc[0] if not locus_df.empty else "unknown"

            # Observed trajectory
            if locus_type == "continuous":
                obs_df = locus_df[locus_df["allele"] == ALLELE_MEAN].sort_values("generation")
                obs_gens = obs_df["generation"].values
                obs_vals = obs_df["frequency"].values
                ax.plot(obs_gens, obs_vals, color="steelblue", linewidth=2, label="Observed (mean)", zorder=3)
                obs_label = "Population mean"

                # WF neutral drift: use initial mean as the "allele frequency".
                # The Wright-Fisher simulator requires frequency values in [0, 1]
                # that sum to 1.  For continuous loci we create a two-allele proxy:
                # "allele" = initial population mean (clipped to [0, 1]) and
                # "_complement" = 1 - initial_mean.  This provides a neutral-drift
                # baseline that shows the expected variance under random drift alone,
                # making directional shifts in the observed mean visually apparent.
                if len(obs_vals) > 0 and np.isfinite(obs_vals[0]):
                    # Normalize to [0, 1] range for WF simulator using a two-allele proxy
                    init_val = float(np.clip(obs_vals[0], 0.0, 1.0))
                    init_freqs = {"allele": init_val, "_complement": 1.0 - init_val}
                    try:
                        wf_df = simulate_wright_fisher(
                            initial_frequencies=init_freqs,
                            n_effective=n_effective,
                            n_generations=n_gens,
                            seed=seed,
                        )
                        wf_allele = wf_df[wf_df["allele"] == "allele"].sort_values("generation")
                        gen_offset = min(gens_sorted)
                        ax.plot(
                            wf_allele["generation"] + gen_offset,
                            wf_allele["frequency"],
                            color="gray", linewidth=1.2, linestyle="--",
                            alpha=0.8, label=f"WF neutral (N_e={n_effective})", zorder=2,
                        )
                    except (ValueError, Exception) as wf_exc:
                        logger.debug("WF simulation skipped for locus %r: %s", locus, wf_exc)
            else:
                # Categorical: one allele per line; WF run separately per allele
                alleles = list(locus_df["allele"].unique())
                first_gen_df = locus_df[locus_df["generation"] == min(gens_sorted)]
                init_freqs: Dict[str, float] = {}
                for allele in alleles:
                    allele_row = first_gen_df[first_gen_df["allele"] == allele]
                    if not allele_row.empty:
                        init_freqs[allele] = float(allele_row["frequency"].iloc[0])
                if init_freqs and abs(sum(init_freqs.values()) - 1.0) < 0.05:
                    try:
                        wf_df = simulate_wright_fisher(
                            initial_frequencies=init_freqs,
                            n_effective=n_effective,
                            n_generations=n_gens,
                            seed=seed,
                        )
                        for allele in alleles[:6]:
                            wf_allele = wf_df[wf_df["allele"] == allele].sort_values("generation")
                            gen_offset = min(gens_sorted)
                            ax.plot(
                                wf_allele["generation"] + gen_offset,
                                wf_allele["frequency"],
                                color="gray", linewidth=1.0, linestyle="--", alpha=0.6, zorder=2,
                            )
                    except (ValueError, Exception) as wf_exc:
                        logger.debug("WF simulation skipped for locus %r: %s", locus, wf_exc)

                for allele_name, allele_df in locus_df.groupby("allele"):
                    allele_df = allele_df.sort_values("generation")
                    ax.plot(
                        allele_df["generation"], allele_df["frequency"],
                        marker=".", markersize=3, label=str(allele_name), zorder=3,
                    )
                if len(alleles) <= 8:
                    ax.legend(fontsize=7)
                obs_label = "Allele frequency"

            ax.set_xlabel("Generation")
            ax.set_ylabel(obs_label)
            ax.set_title(f"{locus} – WF overlay", fontsize=9)
            ax.set_ylim(bottom=0)

            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, loc="best")

        for ax in axes_flat[len(plot_loci):]:
            ax.set_visible(False)

        fig.suptitle(
            f"Observed vs. Wright-Fisher neutral drift (N_e={n_effective})",
            fontsize=11, y=1.02,
        )
        fig.tight_layout()
        output_file = ctx.get_output_file("genetics_wright_fisher_overlay.png")
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved Wright-Fisher overlay plot to %s", output_file)
        return output_file
    except Exception as exc:
        logger.warning("plot_wright_fisher_overlay failed: %s", exc)
        return None


def _extract_lineage_graph(
    df: pd.DataFrame,
) -> Optional[Tuple[Dict[str, int], Dict[str, List[str]]]]:
    """Extract node generations and parent relationships from a genetics frame."""
    if df.empty or "generation" not in df.columns:
        return None

    id_col: Optional[str] = None
    if "candidate_id" in df.columns:
        id_col = "candidate_id"
    elif "agent_id" in df.columns:
        id_col = "agent_id"

    if id_col is None:
        return None

    node_generation: Dict[str, int] = {}
    node_parents: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        raw_id = row.get(id_col)
        if raw_id is None:
            continue
        try:
            gen_int = int(float(row["generation"]))
        except (TypeError, ValueError):
            continue

        node_id = str(raw_id)
        node_generation[node_id] = gen_int

        parent_ids_raw = row.get("parent_ids")
        if isinstance(parent_ids_raw, str):
            try:
                parsed = ast.literal_eval(parent_ids_raw)
            except (ValueError, SyntaxError):
                parsed = None
            parent_ids_raw = parsed if isinstance(parsed, (list, tuple)) else []
        if not isinstance(parent_ids_raw, (list, tuple)):
            node_parents[node_id] = []
            continue
        parent_ids: List[str] = []
        for parent_id in parent_ids_raw:
            if parent_id in (None, "", "seed"):
                continue
            parent_ids.append(str(parent_id))
        node_parents[node_id] = list(dict.fromkeys(parent_ids))

    if not node_generation:
        return None
    return node_generation, node_parents


def plot_phylogenetic_tree_basic(
    df: pd.DataFrame,
    ctx: AnalysisContext,
    annotate_ids: bool = False,
    **kwargs: Any,
) -> Optional[Path]:
    """Plot a full lineage tree using generation and parent-child links."""
    graph = _extract_lineage_graph(df)
    if graph is None:
        logger.warning("plot_phylogenetic_tree_basic: DataFrame lacks lineage columns")
        return None

    try:
        node_generation, node_parents = graph
        nodes_sorted = sorted(node_generation.keys(), key=lambda nid: (node_generation[nid], nid))
        node_y = {nid: idx for idx, nid in enumerate(nodes_sorted)}

        edges: List[Tuple[str, str]] = []
        for child_id, parents in node_parents.items():
            for parent_id in parents:
                if parent_id in node_generation:
                    edges.append((parent_id, child_id))

        fig_height = max(4.0, min(18.0, 2.0 + len(nodes_sorted) * 0.12))
        fig, ax = plt.subplots(figsize=(12, fig_height))

        for parent_id, child_id in edges:
            ax.plot(
                [node_generation[parent_id], node_generation[child_id]],
                [node_y[parent_id], node_y[child_id]],
                color="#c7c7c7",
                linewidth=0.8,
                alpha=0.9,
                zorder=1,
            )

        x_vals = [node_generation[node_id] for node_id in nodes_sorted]
        y_vals = [node_y[node_id] for node_id in nodes_sorted]
        ax.scatter(
            x_vals,
            y_vals,
            c=x_vals,
            cmap="viridis",
            s=22,
            edgecolors="black",
            linewidths=0.2,
            zorder=2,
        )

        if annotate_ids and len(nodes_sorted) <= 60:
            for node_id in nodes_sorted:
                ax.text(
                    node_generation[node_id] + 0.03,
                    node_y[node_id],
                    node_id,
                    fontsize=6,
                    va="center",
                )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Lineage index")
        ax.set_title(f"Phylogenetic tree (full) — {len(nodes_sorted)} nodes, {len(edges)} edges")
        ax.grid(axis="x", alpha=0.2)
        ax.set_ylim(-1, len(nodes_sorted))
        fig.tight_layout()

        output_file = ctx.get_output_file("genetics_phylogenetic_tree_basic.png")
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved basic phylogenetic tree plot to %s", output_file)
        return output_file
    except Exception as exc:
        logger.warning("plot_phylogenetic_tree_basic failed: %s", exc)
        return None


def plot_phylogenetic_tree_sampled(
    df: pd.DataFrame,
    ctx: AnalysisContext,
    max_nodes: int = 120,
    annotate_ids: bool = False,
    **kwargs: Any,
) -> Optional[Path]:
    """Plot a sampled lineage tree for large populations."""
    if max_nodes < 2:
        logger.warning("plot_phylogenetic_tree_sampled: max_nodes must be >= 2")
        return None

    graph = _extract_lineage_graph(df)
    if graph is None:
        logger.warning("plot_phylogenetic_tree_sampled: DataFrame lacks lineage columns")
        return None

    try:
        node_generation, node_parents = graph
        nodes_sorted = sorted(node_generation.keys(), key=lambda nid: (node_generation[nid], nid))
        if not nodes_sorted:
            return None

        if len(nodes_sorted) <= max_nodes:
            sampled_nodes = set(nodes_sorted)
        else:
            step = len(nodes_sorted) / float(max_nodes)
            sampled_nodes = {
                nodes_sorted[min(len(nodes_sorted) - 1, int(idx * step))]
                for idx in range(max_nodes)
            }
            # Add direct parents of sampled nodes when present for context.
            for node_id in list(sampled_nodes):
                for parent_id in node_parents.get(node_id, []):
                    if parent_id in node_generation:
                        sampled_nodes.add(parent_id)

        sampled_sorted = sorted(sampled_nodes, key=lambda nid: (node_generation[nid], nid))
        node_y = {nid: idx for idx, nid in enumerate(sampled_sorted)}

        edges: List[Tuple[str, str]] = []
        for child_id in sampled_sorted:
            for parent_id in node_parents.get(child_id, []):
                if parent_id in sampled_nodes:
                    edges.append((parent_id, child_id))

        fig_height = max(4.0, min(16.0, 2.0 + len(sampled_sorted) * 0.11))
        fig, ax = plt.subplots(figsize=(12, fig_height))

        for parent_id, child_id in edges:
            ax.plot(
                [node_generation[parent_id], node_generation[child_id]],
                [node_y[parent_id], node_y[child_id]],
                color="#c7c7c7",
                linewidth=0.8,
                alpha=0.9,
                zorder=1,
            )

        x_vals = [node_generation[node_id] for node_id in sampled_sorted]
        y_vals = [node_y[node_id] for node_id in sampled_sorted]
        ax.scatter(
            x_vals,
            y_vals,
            c=x_vals,
            cmap="plasma",
            s=26,
            edgecolors="black",
            linewidths=0.2,
            zorder=2,
        )

        if annotate_ids and len(sampled_sorted) <= 70:
            for node_id in sampled_sorted:
                ax.text(
                    node_generation[node_id] + 0.03,
                    node_y[node_id],
                    node_id,
                    fontsize=6,
                    va="center",
                )

        ax.set_xlabel("Generation")
        ax.set_ylabel("Sample lineage index")
        ax.set_title(
            f"Phylogenetic tree (sampled) — {len(sampled_sorted)} nodes, {len(edges)} edges"
        )
        ax.grid(axis="x", alpha=0.2)
        ax.set_ylim(-1, len(sampled_sorted))
        fig.tight_layout()

        output_file = ctx.get_output_file("genetics_phylogenetic_tree_sampled.png")
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved sampled phylogenetic tree plot to %s", output_file)
        return output_file
    except Exception as exc:
        logger.warning("plot_phylogenetic_tree_sampled failed: %s", exc)
        return None


def plot_conserved_run_timeline(
    df: pd.DataFrame,
    ctx: AnalysisContext,
    epsilon: float = 1e-4,
    min_run_length: int = 2,
    **kwargs: Any,
) -> Optional[Path]:
    """Plot a conserved-run timeline showing per-locus conservation across generations.

    Accepts the conserved-runs tidy DataFrame produced by
    :func:`~farm.analysis.genetics.compute.compute_conserved_runs` **or** a
    raw genetics DataFrame (with ``generation`` plus ``chromosome_values``
    and/or ``action_weights`` columns), deriving conserved runs internally
    when needed.

    For each locus a horizontal lane is drawn; a coloured block indicates a
    generation where the locus is conserved (variance below epsilon).

    Parameters
    ----------
    df:
        Conserved-runs DataFrame or raw genetics DataFrame.
    ctx:
        Analysis context supplying output paths and a logger.
    epsilon:
        Variance threshold used when deriving conserved runs from raw data.
    min_run_length:
        Minimum consecutive conserved generations to qualify as a run.

    Returns
    -------
    Path or None
        Path to the saved PNG file, or ``None`` on error / missing data.
    """
    if df.empty:
        logger.warning("plot_conserved_run_timeline: empty DataFrame")
        return None

    try:
        import matplotlib.patches as mpatches
        import numpy as np

        # Determine if this is already a conserved-runs frame
        conserved_cols = {"generation", "locus", "is_conserved"}
        if conserved_cols.issubset(df.columns):
            conserved_df = df
        elif "generation" in df.columns and (
            "chromosome_values" in df.columns or "action_weights" in df.columns
        ):
            from farm.analysis.genetics.compute import compute_conserved_runs
            conserved_df = compute_conserved_runs(df, epsilon=epsilon, min_run_length=min_run_length)
        else:
            logger.warning(
                "plot_conserved_run_timeline: DataFrame lacks required columns"
            )
            return None

        if conserved_df.empty:
            logger.warning("plot_conserved_run_timeline: no conserved-run data")
            return None

        loci = list(conserved_df["locus"].unique())
        if not loci:
            logger.warning("plot_conserved_run_timeline: no loci found")
            return None

        all_gens = sorted(conserved_df["generation"].unique())
        n_loci = len(loci)
        fig_height = max(3.0, n_loci * 0.6 + 1.5)
        fig, ax = plt.subplots(figsize=(max(8, len(all_gens) * 0.4), fig_height))

        cmap_conserved = "steelblue"
        cmap_not_conserved = "#e0e0e0"

        locus_to_y = {locus: i for i, locus in enumerate(loci)}
        has_run_id = "run_id" in conserved_df.columns

        def _row_in_qualifying_run(row: pd.Series) -> bool:
            if has_run_id:
                if pd.isna(row.get("run_id")):
                    return False
                if "run_length" in conserved_df.columns:
                    rl = row.get("run_length")
                    if pd.isna(rl) or int(rl) < int(min_run_length):
                        return False
                return True
            return bool(row["is_conserved"])

        for locus in loci:
            sub = conserved_df[conserved_df["locus"] == locus].sort_values("generation")
            y = locus_to_y[locus]
            run_start: Optional[int] = None
            run_len = 0
            run_conserved: Optional[bool] = None
            prev_gen: Optional[int] = None

            def _flush_run() -> None:
                nonlocal run_start, run_len, run_conserved
                if run_start is None or run_len <= 0 or run_conserved is None:
                    return
                c = cmap_conserved if run_conserved else cmap_not_conserved
                ax.barh(y, run_len, left=run_start, height=0.7, color=c, edgecolor="none", align="center")

            for _, row in sub.iterrows():
                try:
                    gen = int(float(row["generation"]))
                except (TypeError, ValueError):
                    continue
                conserved_block = _row_in_qualifying_run(row)
                extend = (
                    prev_gen is not None
                    and gen == prev_gen + 1
                    and conserved_block == run_conserved
                    and run_start is not None
                )
                if extend:
                    run_len += 1
                else:
                    _flush_run()
                    run_start = gen
                    run_len = 1
                    run_conserved = conserved_block
                prev_gen = gen
            _flush_run()

        ax.set_yticks(list(locus_to_y.values()))
        ax.set_yticklabels(loci, fontsize=8)
        ax.set_xlabel("Generation")
        ax.set_title("Conserved-region timeline per locus")

        legend_patches = [
            mpatches.Patch(color=cmap_conserved, label="Conserved"),
            mpatches.Patch(color=cmap_not_conserved, label="Not conserved", linewidth=0.5, edgecolor="gray"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

        fig.tight_layout()
        output_file = ctx.get_output_file("genetics_conserved_run_timeline.png")
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved conserved-run timeline plot to %s", output_file)
        return output_file
    except Exception as exc:
        logger.warning("plot_conserved_run_timeline failed: %s", exc)
        return None

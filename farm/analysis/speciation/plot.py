"""Speciation visualisation helpers.

Provides a scatter / projection plot of agents in chromosome space coloured
by their detected cluster assignment.  Dimensionality reduction (PCA) is
applied automatically when the number of evolvable genes exceeds 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend before pyplot import
import matplotlib.pyplot as plt  # noqa: E402 – must follow backend selection

import numpy as np

from farm.analysis.common.context import AnalysisContext
from farm.analysis.speciation.compute import ClusterResult
from farm.utils.logging import get_logger

logger = get_logger(__name__)

# Palette for up to 12 distinct clusters; noise (label -1) gets a neutral grey
_CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]
_NOISE_COLOR = "#cccccc"


def plot_chromosome_space_clusters(
    chromosomes: Sequence[Dict[str, float]],
    cluster_result: ClusterResult,
    ctx: AnalysisContext,
    *,
    step: Optional[int] = None,
    projection: str = "pca",
    output_filename: Optional[str] = None,
    agent_ids: Optional[Sequence[Any]] = None,
    title: Optional[str] = None,
) -> Optional[Path]:
    """Scatter agents in 2-D chromosome space coloured by detected cluster.

    When the chromosome has more than 2 genes, PCA is used to project to 2
    dimensions first.  When there are exactly 2 genes the raw gene values are
    used directly.  With a single gene a 1-D strip plot is drawn instead.

    Parameters
    ----------
    chromosomes:
        Sequence of chromosome dicts in the same order as
        ``cluster_result.labels``.  Must not be empty.
    cluster_result:
        A :class:`~farm.analysis.speciation.compute.ClusterResult` from
        :func:`~farm.analysis.speciation.compute.detect_clusters_gmm` or
        :func:`~farm.analysis.speciation.compute.detect_clusters_dbscan`.
    ctx:
        :class:`~farm.analysis.common.context.AnalysisContext` supplying the
        output directory.
    step:
        Simulation step number used in the default output filename and plot
        title.
    projection:
        Dimensionality reduction method.  Currently only ``"pca"`` is
        supported; passing any other value raises :exc:`ValueError`.
        Ignored when the number of genes is ≤ 2.
    output_filename:
        Custom output filename (without directory).  Defaults to
        ``"speciation_clusters_step{step}.png"`` or
        ``"speciation_clusters.png"`` when ``step`` is ``None``.
    agent_ids:
        Optional sequence of agent-ID labels to annotate points.  When
        provided and the population is small (≤ 30 agents) IDs are drawn
        next to each point.
    title:
        Custom title string.  Auto-generated when ``None``.

    Returns
    -------
    Path or None
        Absolute path to the saved PNG file, or ``None`` on failure.
    """
    if not chromosomes:
        logger.warning("plot_chromosome_space_clusters: no chromosomes provided; skipping plot")
        return None

    if projection != "pca":
        raise ValueError(
            f"plot_chromosome_space_clusters: unsupported projection {projection!r}. "
            "Only 'pca' is currently supported."
        )

    # Build feature matrix
    gene_names = cluster_result.gene_names
    X = np.array(
        [[c.get(g, 0.0) for g in gene_names] for c in chromosomes],
        dtype=float,
    )
    labels = np.array(cluster_result.labels, dtype=int)

    if len(labels) != X.shape[0]:
        logger.warning(
            "plot_chromosome_space_clusters: label count (%d) != chromosome count (%d); skipping",
            len(labels),
            X.shape[0],
        )
        return None

    n_genes = X.shape[1]
    # Track whether a fitted PCA object is available for centroid projection
    _pca = None

    # Dimensionality reduction when needed
    if n_genes > 2:
        try:
            from sklearn.decomposition import PCA

            n_components = 2
            _pca = PCA(n_components=n_components, random_state=0)
            X_2d: np.ndarray = _pca.fit_transform(X)
            explained = _pca.explained_variance_ratio_
            xlabel = f"PC1 ({explained[0] * 100:.1f}% var)"
            ylabel = f"PC2 ({explained[1] * 100:.1f}% var)"
        except Exception as exc:
            logger.warning("plot_chromosome_space_clusters: PCA failed (%s); using first 2 genes", exc)
            _pca = None
            X_2d = X[:, :2]
            xlabel = gene_names[0] if len(gene_names) > 0 else "gene_0"
            ylabel = gene_names[1] if len(gene_names) > 1 else "gene_1"
    elif n_genes == 2:
        X_2d = X
        xlabel = gene_names[0]
        ylabel = gene_names[1]
    else:
        # Single-gene strip plot: x = gene value, y = jittered 0
        X_2d = np.column_stack([X[:, 0], np.zeros(X.shape[0])])
        xlabel = gene_names[0] if gene_names else "gene"
        ylabel = ""

    try:
        fig, ax = plt.subplots(figsize=(7, 5))

        unique_labels = sorted(set(labels.tolist()))
        for lbl in unique_labels:
            mask = labels == lbl
            if lbl == -1:
                color = _NOISE_COLOR
                label_str = "noise"
                zorder = 1
            else:
                color = _CLUSTER_COLORS[lbl % len(_CLUSTER_COLORS)]
                size = cluster_result.sizes[lbl] if lbl < len(cluster_result.sizes) else 0
                label_str = f"cluster {lbl} (n={size})"
                zorder = 2

            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=color,
                label=label_str,
                s=40,
                alpha=0.75,
                edgecolors="none",
                zorder=zorder,
            )

            # Mark centroid with a larger marker
            if lbl >= 0 and lbl < len(cluster_result.centroids):
                cx = cluster_result.centroids[lbl]
                cx_vec = np.array([cx.get(g, 0.0) for g in gene_names], dtype=float)
                if n_genes > 2 and _pca is not None:
                    cx_2d = _pca.transform(cx_vec.reshape(1, -1))[0]
                elif n_genes > 2:
                    # PCA failed during setup; fall back to first two genes
                    cx_2d = cx_vec[:2]
                elif n_genes == 2:
                    cx_2d = cx_vec
                else:
                    cx_2d = np.array([cx_vec[0], 0.0])
                ax.scatter(
                    [cx_2d[0]], [cx_2d[1]],
                    c=color, s=150, marker="*", edgecolors="black",
                    linewidths=0.8, zorder=3,
                )

        # Optional agent-ID annotations for small populations
        if agent_ids is not None and len(agent_ids) <= 30:
            for idx, aid in enumerate(agent_ids):
                ax.annotate(str(aid), (X_2d[idx, 0], X_2d[idx, 1]),
                            fontsize=6, alpha=0.6, ha="left", va="bottom")

        ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        if title is None:
            algo = cluster_result.algorithm.upper()
            k = cluster_result.k
            sil = cluster_result.silhouette_score
            step_str = f" @ step {step}" if step is not None else ""
            title = f"Chromosome Space – {algo} k={k}, sil={sil:.3f}{step_str}"
        ax.set_title(title)

        if cluster_result.k > 1 or any(lbl == -1 for lbl in labels.tolist()):
            ax.legend(fontsize=8, loc="best")

        # Resolve output path
        if output_filename is None:
            step_tag = f"_step{step}" if step is not None else ""
            output_filename = f"speciation_clusters{step_tag}.png"

        output_file = ctx.get_output_file(output_filename)
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("plot_chromosome_space_clusters: saved to %s", output_file)
        return Path(output_file)

    except Exception as exc:
        logger.warning("plot_chromosome_space_clusters: plot failed: %s", exc)
        plt.close("all")
        return None

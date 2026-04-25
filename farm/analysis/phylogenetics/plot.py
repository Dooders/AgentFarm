"""Phylogenetic tree visualization.

Provides a matplotlib-based phylogeny plot integrated with the
``farm/charts/``-style API.  The layout is a layered dendrogram:

* **Y-axis** – depth from root (founders at the top, deepest leaves at the
  bottom).
* **X-axis** – horizontal spread within each depth level.
* **Edges** – lines from each parent node to each of its children.
* **Colours** – nodes are coloured by their founder lineage when
  ``color_by_lineage=True``.

Large populations
~~~~~~~~~~~~~~~~~
When the tree has more nodes than ``max_nodes``, the function prunes to the
first ``max_depth`` levels (defaulting to 6 when not specified) and, if still
too large, samples at most ``max_nodes`` nodes per depth level before drawing.

Intrinsic lineage variant
~~~~~~~~~~~~~~~~~~~~~~~~~
:func:`plot_intrinsic_lineage_tree` is a specialised variant for the
``intrinsic_gene_snapshots.jsonl`` data source.  It accepts an optional
``gene`` parameter to colour nodes by a specific chromosome gene value
(continuous colormap) rather than by founder lineage.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from farm.analysis.common.context import AnalysisContext
from farm.analysis.phylogenetics.compute import PhylogeneticNode, PhylogeneticTree
from farm.utils.logging import get_logger

logger = get_logger(__name__)

# Default caps for large-population pruning
_DEFAULT_MAX_NODES = 300
_DEFAULT_MAX_DEPTH_LARGE = 6


def _get_cmap(name: str) -> Any:
    """Return a matplotlib colormap by name, compatible with matplotlib ≥3.5 and older."""
    import matplotlib
    import matplotlib.pyplot as plt  # noqa: F401 (needed for plt.cm fallback)

    try:
        return matplotlib.colormaps.get_cmap(name)
    except AttributeError:
        return plt.cm.get_cmap(name)  # type: ignore[attr-defined]


def _prune_and_layout_layered_dendrogram(
    df: PhylogeneticTree,
    max_depth: Optional[int],
    max_nodes: int,
) -> Tuple[Dict[str, PhylogeneticNode], Dict[str, Tuple[float, float]], List[int]]:
    """Prune to render set and assign layered dendrogram coordinates (shared by tree plotters)."""
    effective_max_depth = max_depth
    if effective_max_depth is None and len(df.nodes) > max_nodes:
        effective_max_depth = _DEFAULT_MAX_DEPTH_LARGE

    if effective_max_depth is not None:
        render_ids: Set[str] = {
            nid for nid, n in df.nodes.items() if 0 <= n.depth <= effective_max_depth
        }
    else:
        render_ids = {nid for nid, n in df.nodes.items() if n.depth >= 0}

    if len(render_ids) > max_nodes:
        by_depth: Dict[int, List[str]] = defaultdict(list)
        for nid in sorted(render_ids):
            by_depth[df.nodes[nid].depth].append(nid)
        per_level = max(1, max_nodes // max(1, len(by_depth)))
        render_ids = set()
        for depth_level in sorted(by_depth):
            render_ids.update(sorted(by_depth[depth_level])[:per_level])

    nodes_to_render = {nid: df.nodes[nid] for nid in render_ids}

    by_depth_render: Dict[int, List[str]] = defaultdict(list)
    for nid, node in nodes_to_render.items():
        by_depth_render[node.depth].append(nid)
    for depth_level in by_depth_render:
        by_depth_render[depth_level].sort()

    positions: Dict[str, Tuple[float, float]] = {}
    depths_present = sorted(by_depth_render.keys())
    for depth_level in depths_present:
        ids_at_depth = by_depth_render[depth_level]
        count = len(ids_at_depth)
        for i, nid in enumerate(ids_at_depth):
            x = (i + 1) / (count + 1)
            positions[nid] = (x, -depth_level)

    return nodes_to_render, positions, depths_present


def _render_layered_dendrogram_figure(
    df: PhylogeneticTree,
    ctx: AnalysisContext,
    *,
    nodes_to_render: Dict[str, PhylogeneticNode],
    positions: Dict[str, Tuple[float, float]],
    depths_present: List[int],
    colour_map: Dict[str, Any],
    title: str,
    output_basename: str,
    log_prefix: str,
    show_lineage_legend: bool = False,
    lineage_legend_cmap: Any = None,
    gene_colorbar: Optional[Tuple[Any, Any, str]] = None,
) -> Any:
    """Draw edges, nodes, axes, optional lineage legend or gene colorbar, save PNG."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig_width = max(8.0, len(nodes_to_render) / 5)
    fig_height = max(5.0, (len(depths_present) + 1) * 0.8)
    fig, ax = plt.subplots(figsize=(min(fig_width, 24.0), min(fig_height, 16.0)))

    for nid, node in nodes_to_render.items():
        if nid not in positions:
            continue
        x_child, y_child = positions[nid]
        for pid in node.parent_ids:
            if pid in positions:
                x_parent, y_parent = positions[pid]
                ax.plot(
                    [x_parent, x_child],
                    [y_parent, y_child],
                    color=(0.6, 0.6, 0.6),
                    linewidth=0.8,
                    zorder=1,
                )

    node_size = max(20, min(60, 400 // max(len(nodes_to_render), 1)))
    for nid, (x, y) in positions.items():
        colour = colour_map.get(nid, (0.5, 0.5, 0.5, 1.0))
        marker = "*" if nodes_to_render[nid].is_root else "o"
        ax.scatter(
            x,
            y,
            s=node_size,
            c=[colour],
            marker=marker,
            zorder=2,
            linewidths=0.3,
            edgecolors="black",
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-max(depths_present) - 0.5, 0.5)
    ax.set_xlabel("Relative position within depth level")
    ax.set_ylabel("Depth from founder")
    ax.set_title(title)
    ax.set_yticks([-d for d in depths_present])
    ax.set_yticklabels([str(d) for d in depths_present])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    if gene_colorbar is not None:
        gcmap, gnorm, gene_label = gene_colorbar
        sm = plt.cm.ScalarMappable(cmap=gcmap, norm=gnorm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=gene_label, fraction=0.03, pad=0.04)
    elif show_lineage_legend and df.roots and lineage_legend_cmap is not None:
        handles = []
        for idx, root_id in enumerate(df.roots[:10]):
            patch = mpatches.Patch(
                color=lineage_legend_cmap(idx % 10), label=f"Lineage: {root_id}"
            )
            handles.append(patch)
        if len(df.roots) > 10:
            handles.append(
                mpatches.Patch(color="white", label=f"… +{len(df.roots) - 10} more")
            )
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)

    if df.is_dag:
        ax.text(
            0.01,
            0.01,
            "DAG (dual-parent reproduction; all edges shown)",
            transform=ax.transAxes,
            fontsize=7,
            color="gray",
            va="bottom",
        )

    if len(nodes_to_render) < len(df.nodes):
        ax.text(
            0.99,
            0.01,
            f"Showing {len(nodes_to_render)} of {len(df.nodes)} nodes",
            transform=ax.transAxes,
            fontsize=7,
            color="gray",
            ha="right",
            va="bottom",
        )

    output_file = ctx.get_output_file(output_basename)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("%s: saved to %s", log_prefix, output_file)
    return output_file


def plot_phylogenetic_tree(
    df: PhylogeneticTree,
    ctx: AnalysisContext,
    *,
    max_depth: Optional[int] = None,
    max_nodes: int = _DEFAULT_MAX_NODES,
    title: str = "Phylogenetic Tree",
    color_by_lineage: bool = True,
    **kwargs: Any,
) -> Optional[Any]:
    """Plot the phylogenetic tree/DAG as a layered dendrogram.

    Parameters
    ----------
    df:
        :class:`~farm.analysis.phylogenetics.compute.PhylogeneticTree` to plot
        (parameter name ``df`` for the analysis framework wrapper).
    ctx:
        :class:`~farm.analysis.common.context.AnalysisContext` supplying the
        output directory and a logger.
    max_depth:
        Maximum depth level to render.  When ``None`` and the tree is large
        (more nodes than ``max_nodes``), the plot is automatically pruned to
        ``_DEFAULT_MAX_DEPTH_LARGE`` levels.
    max_nodes:
        Maximum total nodes to render.  When the pruned tree still exceeds
        this limit, each depth level is sampled (deterministically, sorted
        by node ID).
    title:
        Plot title.
    color_by_lineage:
        When ``True`` nodes belonging to different founder lineages are drawn
        in different colours.

    Returns
    -------
    pathlib.Path or None
        Path to the saved PNG file on success, ``None`` on failure.
    """
    if not df.nodes:
        logger.warning("plot_phylogenetic_tree: tree is empty; skipping")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend for server/test contexts
    except ImportError as exc:
        logger.warning("plot_phylogenetic_tree: matplotlib not available: %s", exc)
        return None

    try:
        nodes_to_render, positions, depths_present = _prune_and_layout_layered_dendrogram(
            df, max_depth, max_nodes
        )

        colour_map: Dict[str, Any] = {}
        lineage_legend_cmap = None
        if color_by_lineage and df.roots:
            lineage_legend_cmap = _get_cmap("tab10")
            lineage_colour: Dict[str, Any] = {}
            for idx, root_id in enumerate(df.roots):
                lineage_colour[root_id] = lineage_legend_cmap(idx % 10)

            def _get_lineage_colour(nid: str) -> Any:
                visited_set: Set[str] = set()
                current = nid
                while current in nodes_to_render:
                    if current in visited_set:
                        break
                    visited_set.add(current)
                    if current in lineage_colour:
                        return lineage_colour[current]
                    node = nodes_to_render[current]
                    parents = [p for p in node.parent_ids if p in nodes_to_render]
                    if not parents:
                        break
                    current = parents[0]
                return lineage_colour.get(
                    df.roots[0] if df.roots else list(nodes_to_render)[0],
                    (0.5, 0.5, 0.5, 1.0),
                )

            for nid in nodes_to_render:
                colour_map[nid] = _get_lineage_colour(nid)
        else:
            default_colour = (0.2, 0.4, 0.8, 0.8)
            for nid in nodes_to_render:
                colour_map[nid] = default_colour  # type: ignore[assignment]

        return _render_layered_dendrogram_figure(
            df,
            ctx,
            nodes_to_render=nodes_to_render,
            positions=positions,
            depths_present=depths_present,
            colour_map=colour_map,
            title=title,
            output_basename="phylogenetics_tree.png",
            log_prefix="plot_phylogenetic_tree",
            show_lineage_legend=bool(color_by_lineage and df.roots),
            lineage_legend_cmap=lineage_legend_cmap,
        )

    except Exception as exc:
        logger.warning("plot_phylogenetic_tree: failed: %s", exc, exc_info=True)
        return None


def plot_intrinsic_lineage_tree(
    df: PhylogeneticTree,
    ctx: AnalysisContext,
    *,
    gene: Optional[str] = None,
    chromosomes: Optional[Dict[str, Dict[str, float]]] = None,
    max_depth: Optional[int] = None,
    max_nodes: int = _DEFAULT_MAX_NODES,
    title: str = "Intrinsic Lineage Tree",
    color_by_lineage: bool = True,
    **kwargs: Any,
) -> Optional[Any]:
    """Plot the intrinsic-evolution lineage tree with optional gene-value colouring.

    Extends :func:`plot_phylogenetic_tree` with per-gene chromosome colouring
    sourced from ``intrinsic_gene_snapshots.jsonl`` data.

    Parameters
    ----------
    df:
        :class:`~farm.analysis.phylogenetics.compute.PhylogeneticTree` built
        from intrinsic snapshot data (e.g. via
        :func:`~farm.analysis.phylogenetics.intrinsic_loader.build_intrinsic_lineage_dag`).
    ctx:
        :class:`~farm.analysis.common.context.AnalysisContext` supplying the
        output directory.
    gene:
        Name of the chromosome gene to use for colouring nodes (e.g.
        ``"learning_rate"``).  When provided and ``chromosomes`` contains the
        gene values, nodes are coloured on a continuous ``viridis`` scale.
        When ``None`` or ``chromosomes`` does not contain the gene, falls back
        to founder-lineage colouring (same as :func:`plot_phylogenetic_tree`).
    chromosomes:
        Mapping of ``agent_id`` → ``{gene_name: value, ...}`` produced by
        :func:`~farm.analysis.phylogenetics.intrinsic_loader.extract_chromosomes_from_snapshots`.
        Used for gene-based colouring when ``gene`` is set; if omitted or if
        the requested gene value is unavailable for a node, the plot falls
        back to founder-lineage colouring.
    max_depth:
        Maximum depth level to render.
    max_nodes:
        Maximum total nodes to render.
    title:
        Plot title.
    color_by_lineage:
        When ``True`` and ``gene`` is not set (or gene values are unavailable),
        colour nodes by founder lineage.

    Returns
    -------
    pathlib.Path or None
        Path to the saved PNG file on success, ``None`` on failure.
    """
    if not df.nodes:
        logger.warning("plot_intrinsic_lineage_tree: tree is empty; skipping")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
    except ImportError as exc:
        logger.warning("plot_intrinsic_lineage_tree: matplotlib not available: %s", exc)
        return None

    try:
        nodes_to_render, positions, depths_present = _prune_and_layout_layered_dendrogram(
            df, max_depth, max_nodes
        )

        colour_map: Dict[str, Any] = {}
        use_gene_colouring = False
        gene_colorbar: Optional[Tuple[Any, Any, str]] = None
        lineage_legend_cmap = None

        if gene and chromosomes:
            gene_values: Dict[str, float] = {}
            for nid in nodes_to_render:
                chrom = chromosomes.get(nid, {})
                val = chrom.get(gene)
                if val is not None:
                    try:
                        gene_values[nid] = float(val)
                    except (TypeError, ValueError):
                        pass

            if gene_values:
                use_gene_colouring = True
                v_min = min(gene_values.values())
                v_max = max(gene_values.values())
                gcmap = _get_cmap("viridis")
                if v_max > v_min:
                    gnorm = mcolors.Normalize(vmin=v_min, vmax=v_max)
                else:
                    gnorm = mcolors.Normalize(vmin=v_min - 1e-9, vmax=v_max + 1e-9)
                default_gene_colour = gcmap(0.5)
                for nid in nodes_to_render:
                    if nid in gene_values:
                        colour_map[nid] = gcmap(gnorm(gene_values[nid]))
                    else:
                        colour_map[nid] = default_gene_colour
                gene_colorbar = (gcmap, gnorm, gene)
            else:
                logger.warning(
                    "plot_intrinsic_lineage_tree: gene %r not found in chromosomes;"
                    " falling back to founder-lineage colouring",
                    gene,
                )
        elif gene and chromosomes is None:
            logger.warning(
                "plot_intrinsic_lineage_tree: gene %r provided but chromosomes is None;"
                " falling back to founder-lineage colouring",
                gene,
            )

        if not use_gene_colouring:
            if color_by_lineage and df.roots:
                lineage_legend_cmap = _get_cmap("tab10")
                lineage_colour: Dict[str, Any] = {}
                for idx, root_id in enumerate(df.roots):
                    lineage_colour[root_id] = lineage_legend_cmap(idx % 10)

                def _get_lineage_colour(nid: str) -> Any:
                    visited_set: Set[str] = set()
                    current = nid
                    while current in nodes_to_render:
                        if current in visited_set:
                            break
                        visited_set.add(current)
                        if current in lineage_colour:
                            return lineage_colour[current]
                        node = nodes_to_render[current]
                        parents = [p for p in node.parent_ids if p in nodes_to_render]
                        if not parents:
                            break
                        current = parents[0]
                    return lineage_colour.get(
                        df.roots[0] if df.roots else list(nodes_to_render)[0],
                        (0.5, 0.5, 0.5, 1.0),
                    )

                for nid in nodes_to_render:
                    colour_map[nid] = _get_lineage_colour(nid)
            else:
                default_colour = (0.2, 0.4, 0.8, 0.8)
                for nid in nodes_to_render:
                    colour_map[nid] = default_colour  # type: ignore[assignment]

        return _render_layered_dendrogram_figure(
            df,
            ctx,
            nodes_to_render=nodes_to_render,
            positions=positions,
            depths_present=depths_present,
            colour_map=colour_map,
            title=title,
            output_basename="intrinsic_lineage_tree.png",
            log_prefix="plot_intrinsic_lineage_tree",
            show_lineage_legend=bool(
                color_by_lineage and not use_gene_colouring and df.roots
            ),
            lineage_legend_cmap=lineage_legend_cmap,
            gene_colorbar=gene_colorbar,
        )

    except Exception as exc:
        logger.warning(
            "plot_intrinsic_lineage_tree: failed: %s", exc, exc_info=True
        )
        return None

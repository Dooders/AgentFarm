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


def _build_lineage_palette(roots: List[str]) -> Dict[str, Any]:
    """Return a colour for each root id, drawing from a wide qualitative palette.

    Cycles through tab20 + tab20b + tab20c (60 distinct colours) so adjacent
    founders rarely share a hue.
    """
    palette_cmaps = [_get_cmap("tab20"), _get_cmap("tab20b"), _get_cmap("tab20c")]
    palette: List[Any] = []
    for cmap in palette_cmaps:
        for i in range(20):
            palette.append(cmap(i / 19.0 if i else 0.0))
    return {root: palette[idx % len(palette)] for idx, root in enumerate(roots)}


def _prune_and_layout_layered_dendrogram(
    df: PhylogeneticTree,
    max_depth: Optional[int],
    max_nodes: int,
) -> Tuple[
    Dict[str, PhylogeneticNode],
    Dict[str, Tuple[float, float]],
    List[int],
    Dict[str, str],
    List[str],
]:
    """Prune to render set and assign tidy hierarchical coordinates.

    Children are placed centered above their leaf descendants, so each
    subtree forms a contiguous horizontal band and parent → child edges
    flow downward without crossing other lineages. For DAG inputs (multi-
    parent), a single primary parent is used for layout while every parent
    edge is still drawn at render time.
    """
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
        by_depth_ids: Dict[int, List[str]] = defaultdict(list)
        for nid in sorted(render_ids):
            by_depth_ids[df.nodes[nid].depth].append(nid)
        per_level = max(1, max_nodes // max(1, len(by_depth_ids)))
        render_ids = set()
        for depth_level in sorted(by_depth_ids):
            render_ids.update(sorted(by_depth_ids[depth_level])[:per_level])

    nodes_to_render = {nid: df.nodes[nid] for nid in render_ids}

    children: Dict[str, List[str]] = defaultdict(list)
    primary_parent: Dict[str, Optional[str]] = {}
    for nid, node in nodes_to_render.items():
        rendered_parents = [p for p in node.parent_ids if p in nodes_to_render]
        if rendered_parents:
            parent = rendered_parents[0]
            primary_parent[nid] = parent
            children[parent].append(nid)
        else:
            primary_parent[nid] = None
    for cs in children.values():
        cs.sort()

    rendered_roots: List[str] = sorted(
        nid for nid, parent in primary_parent.items() if parent is None
    )

    leaf_count: Dict[str, int] = {}

    def _compute_leaves(nid: str) -> int:
        if nid in leaf_count:
            return leaf_count[nid]
        kids = children.get(nid, [])
        leaf_count[nid] = 1 if not kids else sum(_compute_leaves(c) for c in kids)
        return leaf_count[nid]

    total_leaves = max(1, sum(_compute_leaves(r) for r in rendered_roots))

    positions: Dict[str, Tuple[float, float]] = {}
    cursor = [0.0]
    lineage_of: Dict[str, str] = {}

    def _assign(nid: str, root_id: str) -> float:
        lineage_of[nid] = root_id
        depth = nodes_to_render[nid].depth
        kids = children.get(nid, [])
        if not kids:
            x = (cursor[0] + 0.5) / total_leaves
            cursor[0] += 1.0
        else:
            child_xs = [_assign(c, root_id) for c in kids]
            x = sum(child_xs) / len(child_xs)
        positions[nid] = (x, -depth)
        return x

    for root_id in rendered_roots:
        _assign(root_id, root_id)

    depths_present = sorted({n.depth for n in nodes_to_render.values()})
    return nodes_to_render, positions, depths_present, lineage_of, rendered_roots


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
    lineage_of: Optional[Dict[str, str]] = None,
    rendered_roots: Optional[List[str]] = None,
    lineage_palette: Optional[Dict[str, Any]] = None,
    show_lineage_bands: bool = False,
    gene_colorbar: Optional[Tuple[Any, Any, str]] = None,
) -> Any:
    """Draw edges, nodes, axes, optional gene colorbar or lineage bands, save PNG."""
    import matplotlib.pyplot as plt

    n_render = max(len(nodes_to_render), 1)
    fig_width = min(20.0, max(9.0, n_render / 9.0))
    fig_height = min(12.0, max(4.0, (len(depths_present) + 1) * 1.1))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if (
        show_lineage_bands
        and lineage_of
        and rendered_roots
        and lineage_palette
    ):
        extents: Dict[str, Tuple[float, float]] = {}
        for nid, root in lineage_of.items():
            x = positions[nid][0]
            lo, hi = extents.get(root, (x, x))
            extents[root] = (min(lo, x), max(hi, x))
        ordered = sorted(rendered_roots, key=lambda r: extents.get(r, (0, 0))[0])
        y_top = 0.5
        y_bot = -max(depths_present) - 0.6
        for i, root in enumerate(ordered):
            lo, hi = extents.get(root, (0.0, 0.0))
            pad = 0.004
            colour = lineage_palette.get(root, (0.5, 0.5, 0.5, 1.0))
            ax.axvspan(
                max(lo - pad, -0.02),
                min(hi + pad, 1.02),
                ymin=0.0,
                ymax=1.0,
                color=colour,
                alpha=0.06 if i % 2 == 0 else 0.10,
                zorder=0,
                linewidth=0,
            )
        ax.set_ylim(y_bot, y_top)

    for nid, node in nodes_to_render.items():
        if nid not in positions:
            continue
        x_child, y_child = positions[nid]
        for pid in node.parent_ids:
            if pid in positions:
                x_parent, y_parent = positions[pid]
                # Elbow edge: down from parent, across, then into child anchor.
                y_mid = y_parent - 0.55
                ax.plot(
                    [x_parent, x_parent, x_child, x_child],
                    [y_parent, y_mid, y_mid, y_child],
                    color=(0.55, 0.55, 0.6),
                    linewidth=0.7,
                    alpha=0.6,
                    zorder=1,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                )

    node_size = max(28, min(110, 1400 // n_render))
    root_size = int(node_size * 1.6)
    for nid, (x, y) in positions.items():
        colour = colour_map.get(nid, (0.5, 0.5, 0.5, 1.0))
        is_root = nodes_to_render[nid].is_root
        ax.scatter(
            x,
            y,
            s=root_size if is_root else node_size,
            c=[colour],
            marker="*" if is_root else "o",
            zorder=3,
            linewidths=0.4,
            edgecolors="white",
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-max(depths_present) - 0.6, 0.6)
    ax.set_xlabel("")
    ax.set_ylabel("Depth from founder")
    ax.set_title(title, fontsize=13)
    ax.set_yticks([-d for d in depths_present])
    ax.set_yticklabels([str(d) for d in depths_present])
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for spine_name in ("top", "right", "bottom"):
        ax.spines[spine_name].set_visible(False)
    ax.grid(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.45)

    if gene_colorbar is not None:
        gcmap, gnorm, gene_label = gene_colorbar
        sm = plt.cm.ScalarMappable(cmap=gcmap, norm=gnorm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=gene_label, fraction=0.03, pad=0.04)
    elif show_lineage_bands and rendered_roots:
        ax.text(
            0.99,
            1.02,
            f"{len(rendered_roots)} founder lineages (each colored band = one lineage)",
            transform=ax.transAxes,
            fontsize=8,
            color="#555",
            ha="right",
            va="bottom",
        )

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
        (
            nodes_to_render,
            positions,
            depths_present,
            lineage_of,
            rendered_roots,
        ) = _prune_and_layout_layered_dendrogram(df, max_depth, max_nodes)

        colour_map: Dict[str, Any] = {}
        lineage_palette: Dict[str, Any] = {}
        if color_by_lineage and rendered_roots:
            lineage_palette = _build_lineage_palette(rendered_roots)
            for nid, root in lineage_of.items():
                colour_map[nid] = lineage_palette.get(root, (0.5, 0.5, 0.5, 1.0))
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
            lineage_of=lineage_of,
            rendered_roots=rendered_roots,
            lineage_palette=lineage_palette,
            show_lineage_bands=False,
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
        (
            nodes_to_render,
            positions,
            depths_present,
            lineage_of,
            rendered_roots,
        ) = _prune_and_layout_layered_dendrogram(df, max_depth, max_nodes)

        colour_map: Dict[str, Any] = {}
        use_gene_colouring = False
        gene_colorbar: Optional[Tuple[Any, Any, str]] = None
        lineage_palette: Dict[str, Any] = {}

        if gene and chromosomes:
            gene_values: Dict[str, float] = {}
            for nid in nodes_to_render:
                chrom = chromosomes.get(nid, {})
                val = chrom.get(gene)
                if val is not None:
                    try:
                        gene_values[nid] = float(val)
                    except (TypeError, ValueError):
                        # Non-numeric/malformed gene values are expected in some records;
                        # skip them so colouring uses only valid numeric values.
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
            if color_by_lineage and rendered_roots:
                lineage_palette = _build_lineage_palette(rendered_roots)
                for nid, root in lineage_of.items():
                    colour_map[nid] = lineage_palette.get(root, (0.5, 0.5, 0.5, 1.0))
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
            lineage_of=lineage_of,
            rendered_roots=rendered_roots,
            lineage_palette=lineage_palette,
            show_lineage_bands=False,
            gene_colorbar=gene_colorbar,
        )

    except Exception as exc:
        logger.warning(
            "plot_intrinsic_lineage_tree: failed: %s", exc, exc_info=True
        )
        return None

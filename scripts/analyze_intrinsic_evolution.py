#!/usr/bin/env python3
"""Evaluate and visualise an ``IntrinsicEvolutionExperiment`` run.

Reads the artifacts written by
:class:`~farm.runners.intrinsic_evolution_experiment.IntrinsicEvolutionExperiment`
(``intrinsic_gene_trajectory.jsonl``, ``intrinsic_gene_snapshots.jsonl``,
``cluster_lineage.jsonl``, ``intrinsic_evolution_metadata.json``) and emits:

- ``gene_trajectories.png`` -- per-step mean ± std for every evolvable gene.
- ``gene_distribution_history.png`` -- per-snapshot violin distributions.
- ``population_dynamics.png`` -- alive-count, birth/death rates, mean
  reproduction cost, effective selection strength over time.
- ``speciation_index.png`` -- per-step speciation index (when present).
- ``cluster_lineage_sizes.png`` -- per-cluster size over snapshot steps.
- ``speciation_clusters_step{first,mid,last}.png`` -- chromosome-space
  scatter at three representative snapshots.
- ``lineage_tree.png`` -- intrinsic lineage DAG, coloured by ``learning_rate``.
- ``lineage_summary.png`` -- surviving-lineage count and depth over time.
- ``analysis_summary.json`` -- machine-readable rollup of headline metrics.
- ``analysis_summary.md`` -- human-readable rollup of headline metrics.

The script is idempotent: re-running it overwrites the visual artifacts in
the same directory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Allow running directly from repo root without installing the package.
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.analysis.common.context import AnalysisContext  # noqa: E402
from farm.analysis.phylogenetics import (  # noqa: E402
    build_intrinsic_lineage_dag,
    compute_lineage_depth_over_time,
    compute_surviving_lineage_count_over_time,
    extract_chromosomes_from_snapshots,
    load_intrinsic_snapshots,
    plot_intrinsic_lineage_tree,
)
from farm.analysis.speciation import (  # noqa: E402
    compute_niche_correlation,
    compute_speciation_index,
    detect_clusters_gmm,
    plot_chromosome_space_clusters,
)


# ─── I/O helpers ───────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ─── Plot helpers ──────────────────────────────────────────────────────────

def _plot_gene_trajectories(
    trajectory: List[Dict[str, Any]], output: Path
) -> Optional[Path]:
    """Per-gene mean ± std band over time."""
    if not trajectory:
        return None
    steps = np.array([row["step"] for row in trajectory], dtype=float)
    gene_names: List[str] = []
    for row in trajectory:
        gene_names = list(row.get("gene_stats", {}).keys())
        if gene_names:
            break
    if not gene_names:
        return None

    fig, axes = plt.subplots(len(gene_names), 1, figsize=(9, 2.5 * len(gene_names)), sharex=True)
    if len(gene_names) == 1:
        axes = [axes]

    for ax, gene in zip(axes, gene_names):
        means = np.array([row["gene_stats"].get(gene, {}).get("mean", np.nan) for row in trajectory])
        stds = np.array([row["gene_stats"].get(gene, {}).get("std", 0.0) for row in trajectory])
        mins = np.array([row["gene_stats"].get(gene, {}).get("min", np.nan) for row in trajectory])
        maxs = np.array([row["gene_stats"].get(gene, {}).get("max", np.nan) for row in trajectory])

        ax.plot(steps, means, color="#1f77b4", lw=1.6, label="mean")
        ax.fill_between(steps, means - stds, means + stds, color="#1f77b4", alpha=0.20, label="± std")
        ax.plot(steps, mins, color="#888", lw=0.6, ls=":", label="min/max")
        ax.plot(steps, maxs, color="#888", lw=0.6, ls=":")
        ax.set_ylabel(gene)
        ax.grid(alpha=0.25)

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle("Intrinsic evolution: gene trajectories")
    fig.tight_layout()
    out = output / "gene_trajectories.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_distribution_history(
    snapshots: List[Dict[str, Any]], output: Path
) -> Optional[Path]:
    """Violin plot of per-gene distribution at each snapshot step."""
    if not snapshots:
        return None
    gene_names: List[str] = []
    for snap in snapshots:
        if snap.get("agents"):
            gene_names = list(snap["agents"][0].get("chromosome", {}).keys())
            break
    if not gene_names:
        return None

    steps = [snap["step"] for snap in snapshots]
    fig, axes = plt.subplots(len(gene_names), 1, figsize=(10, 2.5 * len(gene_names)), sharex=True)
    if len(gene_names) == 1:
        axes = [axes]

    for ax, gene in zip(axes, gene_names):
        per_step_values: List[List[float]] = []
        for snap in snapshots:
            vals: List[float] = []
            for agent in snap.get("agents", []):
                v = agent.get("chromosome", {}).get(gene)
                if v is None:
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            per_step_values.append(vals if vals else [float("nan")])

        positions = list(range(len(steps)))
        # Build violin only for non-empty slots; matplotlib violinplot tolerates lists of >=2 unique values.
        try:
            parts = ax.violinplot(per_step_values, positions=positions, widths=0.85, showmeans=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("#1f77b4")
                pc.set_alpha(0.35)
        except Exception:
            # Fallback: scatter when violin can't be drawn
            for x, vals in zip(positions, per_step_values):
                ax.scatter([x] * len(vals), vals, s=6, alpha=0.5, color="#1f77b4")

        ax.set_ylabel(gene)
        ax.grid(alpha=0.25)

    # X-axis labelled with snapshot step values, thinned out for legibility
    n = len(steps)
    tick_idx = list(range(0, n, max(1, n // 10)))
    axes[-1].set_xticks(tick_idx)
    axes[-1].set_xticklabels([str(steps[i]) for i in tick_idx])
    axes[-1].set_xlabel("snapshot step")
    fig.suptitle("Per-snapshot gene-value distributions (violins)")
    fig.tight_layout()
    out = output / "gene_distribution_history.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_population_dynamics(
    trajectory: List[Dict[str, Any]], output: Path
) -> Optional[Path]:
    if not trajectory:
        return None
    steps = np.array([row["step"] for row in trajectory], dtype=float)
    n_alive = np.array([row.get("n_alive", 0) for row in trajectory], dtype=float)
    birth = np.array([row.get("realized_birth_rate", 0.0) for row in trajectory], dtype=float)
    death = np.array([row.get("realized_death_rate", 0.0) for row in trajectory], dtype=float)
    cost = np.array([row.get("mean_reproduction_cost", 0.0) for row in trajectory], dtype=float)
    sel = np.array([row.get("effective_selection_strength", 0.0) for row in trajectory], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(steps, n_alive, color="#2ca02c")
    axes[0].set_ylabel("alive")
    axes[0].grid(alpha=0.25)
    axes[0].set_title("Population size")

    axes[1].plot(steps, birth, color="#1f77b4", label="birth rate")
    axes[1].plot(steps, death, color="#d62728", label="death rate")
    axes[1].set_ylabel("rate")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Per-step birth / death rate")

    axes[2].plot(steps, cost, color="#9467bd")
    axes[2].set_ylabel("mean cost")
    axes[2].grid(alpha=0.25)
    axes[2].set_title("Mean effective reproduction cost")

    axes[3].plot(steps, sel, color="#ff7f0e")
    axes[3].set_ylabel("CV(cost)")
    axes[3].set_xlabel("step")
    axes[3].grid(alpha=0.25)
    axes[3].set_title("Effective selection strength (cost coefficient of variation)")

    fig.tight_layout()
    out = output / "population_dynamics.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_speciation_index(
    trajectory: List[Dict[str, Any]], output: Path
) -> Optional[Path]:
    if not trajectory or "speciation_index" not in trajectory[0]:
        return None
    steps = np.array([row["step"] for row in trajectory])
    idx = np.array([row.get("speciation_index", 0.0) for row in trajectory])
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(steps, idx, color="#8c564b")
    ax.set_xlabel("step")
    ax.set_ylabel("speciation index")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.set_title("Speciation index over time (silhouette of GMM clusters)")
    fig.tight_layout()
    out = output / "speciation_index.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_cluster_lineage(cluster_rows: List[Dict[str, Any]], output: Path) -> Optional[Path]:
    if not cluster_rows:
        return None
    by_cluster: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for row in cluster_rows:
        cid = str(row.get("cluster_id"))
        step = int(row.get("step", 0))
        size = int(row.get("size", 0))
        by_cluster[cid].append((step, size))

    fig, ax = plt.subplots(figsize=(9, 4))
    cmap = plt.get_cmap("tab10")
    for i, (cid, hist) in enumerate(sorted(by_cluster.items())):
        hist.sort()
        steps = [h[0] for h in hist]
        sizes = [h[1] for h in hist]
        ax.plot(steps, sizes, marker="o", lw=1.4, ms=4, color=cmap(i % 10), label=cid)
    ax.set_xlabel("snapshot step")
    ax.set_ylabel("cluster size")
    ax.set_title("Cluster persistence: size over snapshot steps")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncols=2, loc="best")
    fig.tight_layout()
    out = output / "cluster_lineage_sizes.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_chromosome_space_at_steps(
    snapshots: List[Dict[str, Any]],
    ctx: AnalysisContext,
    seed: int,
    max_k: int,
) -> List[Path]:
    """Cluster + scatter at first / middle / last non-empty snapshot steps."""
    if not snapshots:
        return []
    non_empty = [s for s in snapshots if s.get("agents")]
    if not non_empty:
        return []
    indices = sorted({0, len(non_empty) // 2, len(non_empty) - 1})
    out_paths: List[Path] = []
    for idx in indices:
        snap = non_empty[idx]
        chrom_dicts = [a.get("chromosome", {}) for a in snap["agents"]]
        if len(chrom_dicts) < 2:
            continue
        try:
            result = detect_clusters_gmm(chrom_dicts, max_k=min(max_k, len(chrom_dicts)), seed=seed)
        except Exception:
            continue
        path = plot_chromosome_space_clusters(chrom_dicts, result, ctx, step=int(snap["step"]))
        if path is not None:
            out_paths.append(Path(path))
    return out_paths


def _plot_lineage_summary(
    surviving: List[Tuple[int, int]],
    depth: List[Tuple[int, int, float]],
    output: Path,
) -> Optional[Path]:
    if not surviving and not depth:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    if surviving:
        s_steps, s_count = zip(*surviving)
        axes[0].plot(s_steps, s_count, color="#1f77b4", marker="o", ms=4)
        axes[0].set_ylabel("# surviving founder lineages")
        axes[0].grid(alpha=0.25)
        axes[0].set_title("Surviving lineages over time")
    if depth:
        d_steps, d_max, d_mean = zip(*depth)
        axes[1].plot(d_steps, d_max, color="#d62728", marker="o", ms=4, label="max depth")
        axes[1].plot(d_steps, d_mean, color="#ff7f0e", marker="o", ms=4, label="mean depth")
        axes[1].set_xlabel("snapshot step")
        axes[1].set_ylabel("lineage depth")
        axes[1].grid(alpha=0.25)
        axes[1].legend(fontsize=8)
        axes[1].set_title("Lineage depth over time")
    fig.tight_layout()
    out = output / "lineage_summary.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


# ─── Summary metrics ───────────────────────────────────────────────────────


def _safe_mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _build_summary(
    metadata: Dict[str, Any],
    trajectory: List[Dict[str, Any]],
    snapshots: List[Dict[str, Any]],
    cluster_rows: List[Dict[str, Any]],
    surviving: List[Tuple[int, int]],
    depth: List[Tuple[int, int, float]],
    niches: Dict[int, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_steps_completed": metadata.get("num_steps_completed"),
        "final_population": metadata.get("final_population"),
        "snapshot_interval": metadata.get("snapshot_interval"),
        "policy": metadata.get("policy"),
        "seed": metadata.get("seed"),
    }

    if trajectory:
        n_alive_series = [row.get("n_alive", 0) for row in trajectory]
        summary["initial_population"] = n_alive_series[0]
        summary["peak_population"] = max(n_alive_series)
        summary["mean_population"] = _safe_mean(n_alive_series)
        summary["final_population_observed"] = n_alive_series[-1]
        if "speciation_index" in trajectory[0]:
            spec_series = [row.get("speciation_index", 0.0) for row in trajectory]
            summary["speciation_index_final"] = spec_series[-1]
            summary["speciation_index_max"] = max(spec_series)
            summary["speciation_index_mean"] = _safe_mean(spec_series)
        summary["birth_rate_mean"] = _safe_mean([row.get("realized_birth_rate", 0.0) for row in trajectory])
        summary["death_rate_mean"] = _safe_mean([row.get("realized_death_rate", 0.0) for row in trajectory])

    if snapshots and snapshots[-1].get("agents"):
        last = snapshots[-1]
        gene_names = list(last["agents"][0].get("chromosome", {}).keys())
        per_gene_initial: Dict[str, float] = {}
        per_gene_final: Dict[str, float] = {}
        if snapshots[0].get("agents"):
            for g in gene_names:
                vals = [
                    float(a["chromosome"][g])
                    for a in snapshots[0]["agents"]
                    if g in a.get("chromosome", {})
                ]
                per_gene_initial[g] = _safe_mean(vals)
        for g in gene_names:
            vals = [
                float(a["chromosome"][g])
                for a in last["agents"]
                if g in a.get("chromosome", {})
            ]
            per_gene_final[g] = _safe_mean(vals)
        summary["per_gene_initial_mean"] = per_gene_initial
        summary["per_gene_final_mean"] = per_gene_final
        summary["per_gene_shift"] = {
            g: per_gene_final[g] - per_gene_initial.get(g, float("nan"))
            for g in per_gene_final
        }

    summary["unique_clusters_observed"] = len({row.get("cluster_id") for row in cluster_rows})
    if surviving:
        summary["surviving_lineages_final"] = surviving[-1][1]
        summary["founder_lineage_count_initial"] = surviving[0][1]
    if depth:
        summary["max_lineage_depth_final"] = depth[-1][1]
        summary["mean_lineage_depth_final"] = depth[-1][2]
    if niches:
        summary["niche_correlation_last_snapshot"] = niches

    return summary


def _format_markdown_summary(summary: Dict[str, Any]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return "n/a"
            return f"{v:.4g}"
        if isinstance(v, dict):
            return ", ".join(f"{k}={fmt(val)}" for k, val in v.items())
        return str(v)

    lines: List[str] = ["# Intrinsic Evolution — Analysis Summary", ""]
    lines.append("## Run")
    for key in ("num_steps_completed", "snapshot_interval", "seed"):
        lines.append(f"- **{key}**: {fmt(summary.get(key))}")
    lines.append("")
    lines.append("## Population")
    for key in ("initial_population", "peak_population", "mean_population", "final_population_observed"):
        if key in summary:
            lines.append(f"- **{key}**: {fmt(summary.get(key))}")
    if "birth_rate_mean" in summary:
        lines.append(f"- **birth_rate_mean**: {fmt(summary['birth_rate_mean'])}")
        lines.append(f"- **death_rate_mean**: {fmt(summary['death_rate_mean'])}")
    lines.append("")
    if "per_gene_initial_mean" in summary:
        lines.append("## Gene means (initial → final)")
        initial = summary["per_gene_initial_mean"]
        final = summary["per_gene_final_mean"]
        shift = summary["per_gene_shift"]
        lines.append("| gene | initial mean | final mean | shift |")
        lines.append("| --- | --- | --- | --- |")
        for gene in final:
            lines.append(f"| `{gene}` | {fmt(initial.get(gene))} | {fmt(final[gene])} | {fmt(shift[gene])} |")
        lines.append("")
    if "speciation_index_final" in summary:
        lines.append("## Speciation")
        lines.append(f"- **final**: {fmt(summary['speciation_index_final'])}")
        lines.append(f"- **max**: {fmt(summary['speciation_index_max'])}")
        lines.append(f"- **mean**: {fmt(summary['speciation_index_mean'])}")
        lines.append(f"- **unique clusters tracked**: {fmt(summary.get('unique_clusters_observed'))}")
        lines.append("")
    if "surviving_lineages_final" in summary:
        lines.append("## Lineages")
        lines.append(f"- **founders at start**: {fmt(summary.get('founder_lineage_count_initial'))}")
        lines.append(f"- **surviving founders at end**: {fmt(summary['surviving_lineages_final'])}")
        if "max_lineage_depth_final" in summary:
            lines.append(f"- **max lineage depth (final snapshot)**: {fmt(summary['max_lineage_depth_final'])}")
            lines.append(f"- **mean lineage depth (final snapshot)**: {fmt(summary['mean_lineage_depth_final'])}")
        lines.append("")
    if "niche_correlation_last_snapshot" in summary and summary["niche_correlation_last_snapshot"]:
        lines.append("## Niche correlation (final snapshot)")
        for cluster_id, rows in summary["niche_correlation_last_snapshot"].items():
            lines.append(f"### Cluster {cluster_id}")
            for row in rows:
                lines.append(f"- size={row.get('size')}, mean_x={fmt(row.get('mean_x'))}, "
                             f"mean_y={fmt(row.get('mean_y'))}, mean_energy={fmt(row.get('mean_energy'))}")
        lines.append("")
    return "\n".join(lines)


# ─── Driver ────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyse an intrinsic-evolution run.")
    parser.add_argument("run_dir", type=str, help="Directory containing the run artifacts.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to write visualisations (defaults to <run_dir>/analysis).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-k", type=int, default=4)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = _read_json(run_dir / "intrinsic_evolution_metadata.json")
    trajectory = _read_jsonl(run_dir / "intrinsic_gene_trajectory.jsonl")
    snapshots = _read_jsonl(run_dir / "intrinsic_gene_snapshots.jsonl")
    cluster_rows = _read_jsonl(run_dir / "cluster_lineage.jsonl")

    print(f"Loaded {len(trajectory)} trajectory rows, {len(snapshots)} snapshots, "
          f"{len(cluster_rows)} cluster lineage rows from {run_dir}")

    artifacts: Dict[str, Optional[str]] = {}

    # Trajectories / population dynamics / speciation index
    p = _plot_gene_trajectories(trajectory, output_dir)
    artifacts["gene_trajectories"] = str(p) if p else None
    p = _plot_distribution_history(snapshots, output_dir)
    artifacts["gene_distribution_history"] = str(p) if p else None
    p = _plot_population_dynamics(trajectory, output_dir)
    artifacts["population_dynamics"] = str(p) if p else None
    p = _plot_speciation_index(trajectory, output_dir)
    artifacts["speciation_index"] = str(p) if p else None
    p = _plot_cluster_lineage(cluster_rows, output_dir)
    artifacts["cluster_lineage_sizes"] = str(p) if p else None

    # Chromosome-space scatter at sampled snapshots
    ctx = AnalysisContext(output_path=output_dir)
    cluster_snapshots = _plot_chromosome_space_at_steps(
        snapshots, ctx, seed=args.seed, max_k=args.max_k
    )
    artifacts["chromosome_space_scatters"] = [str(p) for p in cluster_snapshots]

    # Lineage tree + lineage summary
    surviving_counts: List[Tuple[int, int]] = []
    depth_history: List[Tuple[int, int, float]] = []
    niches: Dict[int, List[Dict[str, Any]]] = {}
    if snapshots:
        try:
            tree = build_intrinsic_lineage_dag(run_dir)
            chromosomes = extract_chromosomes_from_snapshots(snapshots)
            ts = tree.summary()
            print(f"Tree: nodes={ts.num_nodes}, max_depth={ts.max_depth}, is_dag={tree.is_dag}")
            artifacts["lineage_tree_nodes"] = ts.num_nodes  # type: ignore[assignment]
            artifacts["lineage_tree_max_depth"] = ts.max_depth  # type: ignore[assignment]

            tree_path = plot_intrinsic_lineage_tree(
                tree,
                ctx,
                gene="learning_rate",
                chromosomes=chromosomes,
                title="Intrinsic Lineage Tree (coloured by learning_rate)",
            )
            artifacts["lineage_tree"] = str(tree_path) if tree_path else None

            surviving_counts = compute_surviving_lineage_count_over_time(tree, snapshots)
            depth_history = compute_lineage_depth_over_time(tree, snapshots)
            p = _plot_lineage_summary(surviving_counts, depth_history, output_dir)
            artifacts["lineage_summary"] = str(p) if p else None
        except Exception as exc:
            print(f"Lineage analysis failed: {exc}", file=sys.stderr)

        # Niche correlation at the final snapshot (with agent x/y if present)
        last_snap = snapshots[-1]
        chrom_dicts = [a.get("chromosome", {}) for a in last_snap.get("agents", [])]
        if len(chrom_dicts) >= 2:
            try:
                result = detect_clusters_gmm(chrom_dicts, max_k=args.max_k, seed=args.seed)
                niche_rows = compute_niche_correlation(result, last_snap.get("agents", []))
                niches = {result.k: niche_rows}
            except Exception as exc:
                print(f"Niche correlation failed: {exc}", file=sys.stderr)

    summary = _build_summary(metadata, trajectory, snapshots, cluster_rows, surviving_counts, depth_history, niches)
    summary["artifacts"] = artifacts
    summary_path = output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    md_path = output_dir / "analysis_summary.md"
    md_path.write_text(_format_markdown_summary(summary))

    print(f"Analysis written to: {output_dir}")
    for k, v in artifacts.items():
        print(f"  {k}: {v}")
    print(f"Summary: {summary_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

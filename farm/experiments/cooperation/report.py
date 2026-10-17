"""``REPORT.md`` and figures for the cooperation experiment."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from farm.experiments.cooperation.analysis import AnalysisResult
from farm.experiments.cooperation.config import PRIMARY_EXIT_COST, PRIMARY_RECOGNITION, PRIMARY_SCOPE
from farm.experiments.cooperation.experiment import MatrixOutputs


def _pyplot():
    """Import pyplot on the non-interactive backend (lazily, so importing this module is cheap)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


FIGURES_DIRNAME = "figures"

GRID_PANELS: tuple[tuple[str, str, str], ...] = (
    ("bond_rmst", "Bond life (RMST, ticks)", "viridis"),
    ("welfare_d_lifespan", "Founder lifespan gain vs solo twin (ticks)", "magma"),
    ("trap_rate", "Share of member time trapped", "cividis"),
)

MODE_ORDER: tuple[str, ...] = ("healthy", "fragility", "override", "capture", "trap", "drift", "hollow", "unclassified")


def _fmt(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def plot_grid(conditions: pd.DataFrame, metric: str, title: str, cmap: str, path: Path) -> None:
    plt = _pyplot()
    grid = conditions[conditions["family"] == "grid"]
    fig, axes = plt.subplots(1, len(PRIMARY_RECOGNITION), figsize=(4.2 * len(PRIMARY_RECOGNITION), 3.6), sharey=True)
    values = grid[metric].to_numpy(dtype=float)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    image = None
    for ax, rec in zip(np.atleast_1d(axes), PRIMARY_RECOGNITION):
        sub = grid[np.isclose(grid["recognition"], rec)]
        mat = np.full((len(PRIMARY_EXIT_COST), len(PRIMARY_SCOPE)), np.nan)
        for i, e in enumerate(PRIMARY_EXIT_COST):
            for j, s in enumerate(PRIMARY_SCOPE):
                hit = sub[np.isclose(sub["exit_cost"], e) & np.isclose(sub["scope"], s)]
                if len(hit):
                    mat[i, j] = float(hit[metric].iloc[0])
        image = ax.imshow(mat, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}" if abs(mat[i, j]) < 10 else f"{mat[i, j]:.0f}",
                        ha="center",
                        va="center",
                        color="white" if mat[i, j] < (vmin + vmax) / 2 else "black",
                        fontsize=9,
                    )
        ax.set_xticks(range(len(PRIMARY_SCOPE)))
        ax.set_xticklabels([str(s) for s in PRIMARY_SCOPE])
        ax.set_yticks(range(len(PRIMARY_EXIT_COST)))
        ax.set_yticklabels([str(e) for e in PRIMARY_EXIT_COST])
        ax.set_xlabel("scope")
        ax.set_title(f"recognition {rec}")
    np.atleast_1d(axes)[0].set_ylabel("exit cost")
    fig.suptitle(title)
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.85)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_modes(modes: pd.DataFrame, names: list[str], path: Path) -> None:
    plt = _pyplot()
    present = [n for n in names if n in modes.index]
    if not present:
        return
    data = modes.loc[present].reindex(columns=[m for m in MODE_ORDER if m in modes.columns], fill_value=0.0)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(present) + 1.5))
    left = np.zeros(len(present))
    colors = plt.get_cmap("tab10")
    for k, mode in enumerate(data.columns):
        vals = data[mode].to_numpy()
        ax.barh(present, vals, left=left, label=mode, color=colors(k))
        left += vals
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of classified bonds (primary mode)")
    ax.invert_yaxis()
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_who_benefits(table: pd.DataFrame, names: list[str], path: Path) -> None:
    plt = _pyplot()
    sub = table[table["condition"].isin(names)]
    if sub.empty:
        return
    terciles = ["low", "middle", "high"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    width = 0.8 / len(terciles)
    x = np.arange(len(names))
    for k, t in enumerate(terciles):
        vals = [float(sub[(sub["condition"] == n) & (sub["skill_tercile"] == t)]["d_lifespan"].mean()) for n in names]
        ax.bar(x + (k - 1) * width, vals, width, label=f"{t} skill")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("lifespan gain vs solo twin (ticks)")
    ax.legend()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _test_table(tests) -> list[str]:
    lines = [
        "| Test | Statement | Mean diff [95% CI] | Seeds > 0 | p | p (Holm) | Result |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in tests:
        result = "significant" if t.significant else ("n/a" if not np.isfinite(t.p_holm) else "not significant")
        lines.append(
            f"| {t.key} | {t.statement} | {_fmt(t.mean_diff, 3)} [{_fmt(t.ci_low, 3)}, {_fmt(t.ci_high, 3)}] | "
            f"{t.positive_seeds}/{t.n} | {_fmt_p(t.p_value)} | {_fmt_p(t.p_holm)} | {result} |"
        )
    return lines


def write_report(outputs: MatrixOutputs, result: AnalysisResult, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    figs = out / FIGURES_DIRNAME
    figs.mkdir(parents=True, exist_ok=True)
    for metric, title, cmap in GRID_PANELS:
        if metric in result.conditions.columns:
            plot_grid(result.conditions, metric, title, cmap, figs / f"grid_{metric}.png")
    structured = [n for n in result.conditions.index if result.conditions.loc[n, "family"] != "grid"]
    anchors = [n for n in ("S0.8_E0.0_R0.0", "S0.8_E0.6_R1.0") if n in result.conditions.index]
    plot_modes(result.modes, anchors + structured, figs / "failure_modes.png")
    plot_who_benefits(result.who_benefits, anchors + structured, figs / "who_benefits.png")

    m = outputs.manifest
    lines = [
        "# Cooperation experiment: report",
        "",
        (
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from "
            f"{len(outputs.runs)} cooperative runs and {len(outputs.solo_runs)} solo twins "
            f"(seeds {m.get('seeds', ['?'])[0]}-{m.get('seeds', ['?'])[-1]}; "
            f"commit {m.get('git_commit') or 'unknown'})."
        ),
        "",
        "## Confirmatory tests",
        "",
        *_test_table(result.confirmatory),
        "",
        "Verdicts: " + ", ".join(f"{k} {v}" for k, v in result.verdicts.items()) if result.verdicts else "No verdicts.",
        "",
        "## Manipulation checks",
        "",
        *_test_table(result.manipulation_checks),
        "",
        "## Primary grid",
        "",
        "![bond life](figures/grid_bond_rmst.png)",
        "",
        "![welfare](figures/grid_welfare_d_lifespan.png)",
        "",
        "![trap](figures/grid_trap_rate.png)",
        "",
        "| Cell | RMST | Coverage | Welfare gain | Better | Worse | Trap rate | Healthy | Fragility | Trap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, row in result.grid.iterrows():
        lines.append(
            f"| {name} | {_fmt(row['bond_rmst'], 1)} | {_fmt(row['coverage_mean'])} | {_fmt(row['welfare_d_lifespan'], 1)} | "
            f"{_fmt(row['welfare_better_share'])} | {_fmt(row['welfare_worse_share'])} | {_fmt(row['trap_rate'], 3)} | "
            f"{_fmt(row.get('bond_primary_healthy', float('nan')))} | {_fmt(row.get('bond_primary_fragility', float('nan')))} | "
            f"{_fmt(row.get('bond_primary_trap', float('nan')))} |"
        )
    lines += [
        "",
        "## Presets, structure and motives",
        "",
        "| Condition | RMST | Coverage | Welfare gain | Starved (coop / solo) | Trap rate | Healthy | Override | Capture | Drift |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in structured:
        row = result.conditions.loc[name]
        lines.append(
            f"| {name} | {_fmt(row['bond_rmst'], 1)} | {_fmt(row['coverage_mean'])} | {_fmt(row['welfare_d_lifespan'], 1)} | "
            f"{_fmt(row['welfare_starved'])} / {_fmt(row['welfare_starved_solo'])} | {_fmt(row['trap_rate'], 3)} | "
            f"{_fmt(row.get('bond_primary_healthy', float('nan')))} | {_fmt(row.get('bond_primary_override', float('nan')))} | "
            f"{_fmt(row.get('bond_primary_capture', float('nan')))} | {_fmt(row.get('bond_primary_drift', float('nan')))} |"
        )
    lines += [
        "",
        "![failure modes](figures/failure_modes.png)",
        "",
        "## Who benefits",
        "",
        "Lifespan gain against the solo twin by founder skill tercile.",
        "",
        "![who benefits](figures/who_benefits.png)",
        "",
    ]
    if result.notes:
        lines += ["## Notes", "", *[f"- {n}" for n in result.notes], ""]
    path = out / "REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

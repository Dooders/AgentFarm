"""Figures for the consensus experiment (matplotlib, Agg backend)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from farm.experiments.consensus.paradigms import BASELINE_PARADIGMS, CONSTRAINED_PARADIGM, SELECTION_PARADIGMS

_PARADIGM_COLORS = {
    "party": "#c0392b",
    "individual": "#2980b9",
    "score": "#27ae60",
    "latent_match": "#8e44ad",
    "constrained_individual": "#f39c12",
    "random_winner": "#7f8c8d",
    "utilitarian": "#16a085",
    "egalitarian": "#2c3e50",
}


def _color(paradigm: str) -> str:
    return _PARADIGM_COLORS.get(paradigm, "#7f8c8d")


def _selection(trials: pd.DataFrame) -> pd.DataFrame:
    return trials[trials["paradigm"].isin(SELECTION_PARADIGMS)]


def _population_axes(trials: pd.DataFrame, title: str):
    populations = list(trials["population"].unique())
    fig, axes = plt.subplots(
        1, len(populations), figsize=(6.0 * len(populations), 4.6), squeeze=False, sharey=True
    )
    fig.suptitle(title)
    return fig, list(zip(populations, axes[0]))


def _welfare_by_paradigm(trials: pd.DataFrame, path: Path) -> None:
    fig, panels = _population_axes(trials, "Welfare by paradigm (mean ± std over trials)")
    metrics = [m for m in ("total_welfare", "majority_welfare", "minority_welfare") if m in trials.columns]
    if not metrics:
        metrics = ["total_welfare", "supporter_welfare", "loser_welfare"]
    width = 0.26
    for population, ax in panels:
        subset = trials[trials["population"] == population]
        paradigms = list(subset["paradigm"].unique())
        x = np.arange(len(paradigms))
        stats = subset.groupby("paradigm", sort=False)[metrics].agg(["mean", "std"])
        for k, metric in enumerate(metrics):
            ax.bar(
                x + (k - 1) * width,
                stats[(metric, "mean")],
                width,
                yerr=stats[(metric, "std")],
                capsize=3,
                label=metric.replace("_welfare", ""),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(paradigms, rotation=20, ha="right")
        ax.set_title(population)
        ax.set_ylabel("mean utility")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _gap_vs_loser_share(trials: pd.DataFrame, path: Path) -> None:
    fig, panels = _population_axes(trials, "Election-endogenous gap vs. loser share (per trial)")
    for population, ax in panels:
        subset = trials[trials["population"] == population]
        for paradigm, group in subset.groupby("paradigm", sort=False):
            if group["loser_share"].isna().all():
                continue
            ax.scatter(group["loser_share"], group["gap"], s=8, alpha=0.35, color=_color(paradigm), label=paradigm)
            ax.scatter(
                [group["loser_share"].mean()],
                [group["gap"].mean()],
                s=140,
                marker="X",
                edgecolor="black",
                color=_color(paradigm),
                zorder=5,
            )
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.set_xlabel("loser share (election-endogenous)")
        ax.set_ylabel("gap (supporter − loser welfare)")
        ax.set_title(population)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _lambda_by_paradigm(trials: pd.DataFrame, path: Path) -> None:
    fig, panels = _population_axes(trials, "Loyalty trait λ of the selected winner")
    for population, ax in panels:
        subset = trials[trials["population"] == population]
        paradigms = [p for p in subset["paradigm"].unique() if not subset.loc[subset["paradigm"] == p, "lambda_winner"].isna().all()]
        data = [subset.loc[subset["paradigm"] == p, "lambda_winner"].dropna().to_numpy() for p in paradigms]
        if not data:
            ax.set_title(population)
            continue
        boxes = ax.boxplot(data, tick_labels=paradigms, patch_artist=True, showmeans=True)
        for patch, paradigm in zip(boxes["boxes"], paradigms):
            patch.set_facecolor(_color(paradigm))
            patch.set_alpha(0.55)
        ax.axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
        ax.set_ylabel("λ of winner")
        ax.set_title(population)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _minority_welfare(trials: pd.DataFrame, path: Path) -> None:
    if "minority_welfare" not in trials.columns:
        return
    fig, panels = _population_axes(trials, "Minority-cluster welfare (fixed partition)")
    for population, ax in panels:
        subset = trials[trials["population"] == population]
        paradigms = list(subset["paradigm"].unique())
        data = [subset.loc[subset["paradigm"] == p, "minority_welfare"].dropna().to_numpy() for p in paradigms]
        boxes = ax.boxplot(data, tick_labels=paradigms, patch_artist=True, showmeans=True)
        for patch, paradigm in zip(boxes["boxes"], paradigms):
            patch.set_facecolor(_color(paradigm))
            patch.set_alpha(0.55)
        ax.set_ylabel("minority-cluster mean utility")
        ax.set_title(population)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_figures(trials: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    selection = _selection(trials)
    plotted = selection if not selection.empty else trials
    extra = trials[trials["paradigm"].isin((*BASELINE_PARADIGMS, CONSTRAINED_PARADIGM))]
    welfare_src = pd.concat([plotted, extra], ignore_index=True) if not extra.empty else plotted
    _welfare_by_paradigm(welfare_src, figures_dir / "welfare_by_paradigm.png")
    _gap_vs_loser_share(plotted, figures_dir / "gap_vs_loser_share.png")
    _lambda_by_paradigm(plotted, figures_dir / "lambda_by_paradigm.png")
    _minority_welfare(welfare_src, figures_dir / "minority_welfare.png")

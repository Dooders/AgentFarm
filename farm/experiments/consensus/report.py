"""Auto-generated markdown report for the consensus experiment.

Every number in the report is computed from the trials DataFrame of the run
that produced it; nothing is hard-coded.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from farm.experiments.consensus.metrics import ALLOCATION_COLUMNS
from farm.experiments.consensus.paradigms import PARADIGMS
from farm.experiments.consensus.population import PROJECTS

BASELINE = "party"
INDIVIDUAL_CENTERED = ("individual", "score", "latent_match")

# Effect-size thresholds used to phrase the hypothesis verdicts.
LAMBDA_UNCHANGED_EPS = 0.02
WELFARE_FLAT_REL_EPS = 0.005
LOSER_SHARE_LARGE_RISE = 0.15


def _fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return "nan"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[col], digits) for col in headers) + " |")
    return "\n".join(lines)


def _summary_display(summary: pd.DataFrame) -> pd.DataFrame:
    """Compact mean ± std table for the report."""
    metrics = ["total_welfare", "supporter_welfare", "loser_welfare", "gap", "lambda_winner", "loser_share"]
    out = summary[["population", "n_candidates", "paradigm"]].copy()
    for metric in metrics:
        out[metric] = [
            f"{mean:.4f} ± {std:.4f}"
            for mean, std in zip(summary[f"{metric}_mean"], summary[f"{metric}_std"])
        ]
    return out


def _cell_means(trials: pd.DataFrame) -> pd.DataFrame:
    return (
        trials.groupby(["population", "n_candidates", "paradigm"], sort=False)
        .agg(
            total_welfare=("total_welfare", "mean"),
            loser_welfare=("loser_welfare", "mean"),
            gap=("gap", "mean"),
            lambda_winner=("lambda_winner", "mean"),
            lambda_sem=("lambda_winner", "sem"),
            loser_share=("loser_share", "mean"),
        )
        .reset_index()
    )


def _hypothesis_lines(cell: pd.DataFrame) -> list[str]:
    """Evaluate the hypothesis for one (population, n_candidates) cell."""
    base = cell[cell["paradigm"] == BASELINE]
    if base.empty:
        return ["- Party baseline missing from this cell; hypothesis not evaluable."]
    base = base.iloc[0]

    lines: list[str] = []
    lambda_deltas = []
    for rule in INDIVIDUAL_CENTERED:
        sel = cell[cell["paradigm"] == rule]
        if sel.empty:
            continue
        sel = sel.iloc[0]
        d_lambda = sel["lambda_winner"] - base["lambda_winner"]
        d_loser = sel["loser_welfare"] - base["loser_welfare"]
        d_share = sel["loser_share"] - base["loser_share"]
        d_gap = sel["gap"] - base["gap"]
        d_total = sel["total_welfare"] - base["total_welfare"]
        lambda_deltas.append(abs(d_lambda))

        lower_lambda = d_lambda < -LAMBDA_UNCHANGED_EPS
        raises_losers = d_loser > 0
        big_share_rise = d_share > LOSER_SHARE_LARGE_RISE
        if lower_lambda and raises_losers and not big_share_rise:
            verdict = "supports the hypothesis"
        elif raises_losers and big_share_rise:
            verdict = "raises loser welfare but mostly by inflating loser share (partial support)"
        elif lower_lambda or raises_losers:
            verdict = "partial support"
        else:
            verdict = "does not support the hypothesis"
        lines.append(
            f"- `{rule}` vs `{BASELINE}`: Δλ_winner = {d_lambda:+.4f}, Δloser_welfare = {d_loser:+.5f}, "
            f"Δloser_share = {d_share:+.4f}, Δgap = {d_gap:+.5f}, Δtotal_welfare = {d_total:+.5f} "
            f"→ {verdict}."
        )

    lines.append("")
    lines.append("Falsifier checks:")
    max_dl = max(lambda_deltas) if lambda_deltas else float("nan")
    lines.append(
        f"- λ_winner unchanged across rules (max |Δλ| < {LAMBDA_UNCHANGED_EPS}): "
        f"max |Δλ| = {max_dl:.4f} → {'**triggered**' if max_dl < LAMBDA_UNCHANGED_EPS else 'not triggered'}."
    )
    others = cell[cell["paradigm"].isin(INDIVIDUAL_CENTERED)]
    rel_welfare = (
        (others["total_welfare"] - base["total_welfare"]).abs().max() / base["total_welfare"]
        if not others.empty
        else float("nan")
    )
    lines.append(
        f"- total welfare flat (max relative change < {WELFARE_FLAT_REL_EPS:.1%}): "
        f"max relative change = {rel_welfare:.4%} → "
        f"{'**triggered**' if rel_welfare < WELFARE_FLAT_REL_EPS else 'not triggered'}."
    )
    max_share_rise = (others["loser_share"] - base["loser_share"]).max() if not others.empty else float("nan")
    lines.append(
        f"- loser_share rises a lot (> +{LOSER_SHARE_LARGE_RISE}): max rise = {max_share_rise:+.4f} → "
        f"{'**triggered**' if max_share_rise > LOSER_SHARE_LARGE_RISE else 'not triggered'}."
    )
    mushy = others[(others["gap"] < base["gap"]) & (others["loser_welfare"] <= base["loser_welfare"])]
    lines.append(
        "- gap shrinks only because the winning bloc is mushier (gap down without loser welfare up): "
        + (
            f"**triggered** for {', '.join(f'`{p}`' for p in mushy['paradigm'])}."
            if not mushy.empty
            else "not triggered — every gap reduction coincides with higher loser welfare."
        )
    )
    return lines


def write_report(
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    allocations: pd.DataFrame,
    manifest: dict,
    path: Path,
) -> None:
    config = manifest["config"]
    populations = list(trials["population"].unique())
    cells = _cell_means(trials)
    paradigms = list(trials["paradigm"].unique())

    parts: list[str] = []
    parts.append("# Political consensus experiment — auto-generated report\n")
    parts.append(
        "Comparison of selection paradigms on a post-election budget-allocation task: "
        "does individual-centered selection (no parties) produce stewards who treat "
        "non-supporters better than party selection does?\n"
    )

    parts.append("## Methods\n")
    parts.append(
        f"- **Population**: {config['voters']} voters per trial, latent preference vectors in R^5 drawn "
        f"from cluster centers plus Gaussian noise (σ = 0.35); benefit vectors are a softmax "
        f"(temperature 0.55) of preferences, so rows are nonnegative and sum to 1. "
        f"Population types run here: {', '.join(f'`{p}`' for p in populations)}.\n"
        f"- **Projects**: {', '.join(f'`{p}`' for p in PROJECTS)}.\n"
        f"- **Candidates**: {config['candidates']} per trial (sweeps may vary this), platforms drawn from the "
        f"same cluster structure; loyalty trait λ ~ Beta(2.2, 2.2), "
        f"{'rank-coupled to platform extremity (`--lambda-correlated`)' if config['lambda_correlated'] else 'independent of the platform'}.\n"
        f"- **Allocation**: `raw = λ·mean(benefits[supporters]) + (1−λ)·mean(benefits[all])`, then "
        f"`raw = 0.72·raw + 0.28·clip(platform, 0, ∞)`, normalized to sum to 1. Zero supporters ⇒ all-voter direction only.\n"
        f"- **Paradigms**: {', '.join(f'`{p}`' for p in paradigms)}."
        + (
            f" `constrained_individual` caps the winner's effective λ at {config['lambda_cap']}.\n"
            if "constrained_individual" in paradigms
            else "\n"
        )
        + f"- **Trials**: {config['trials']} per cell, base seed {config['seed']}; every paradigm sees the "
        f"identical population and candidate slate within a trial.\n"
        f"- **Privacy**: only aggregates and winner allocations are persisted; no individual ballots.\n"
    )

    parts.append("## Results\n")
    parts.append("Mean ± std over trials, by population, candidate count, and paradigm:\n")
    parts.append(_markdown_table(_summary_display(summary)) + "\n")

    parts.append("### Mean winner allocations\n")
    alloc_display = allocations.copy()
    alloc_display.columns = [c.replace("alloc_", "") for c in alloc_display.columns]
    parts.append(_markdown_table(alloc_display) + "\n")

    parts.append("## Hypothesis evaluation\n")
    parts.append(
        "Hypothesis: individual-centered rules (`individual`, `score`, `latent_match`) select "
        "lower-λ winners than `party` and raise loser welfare without merely inflating loser share.\n"
    )
    for (population, n_candidates), _ in cells.groupby(["population", "n_candidates"], sort=False):
        cell = cells[(cells["population"] == population) & (cells["n_candidates"] == n_candidates)]
        parts.append(f"### {population}, {n_candidates} candidates\n")
        parts.append("\n".join(_hypothesis_lines(cell)) + "\n")

    parts.append("## Limitations\n")
    parts.append(
        "- Voters and candidates live in a stylized 5-dimensional project space; real preference "
        "structures are higher-dimensional and partly unobservable.\n"
        "- λ is exogenous by default; strategic candidate behavior, campaigning, and repeated "
        "elections are out of scope.\n"
        "- Party structure is idealized as two brands at cluster means with loyal nomination; real "
        "parties select through noisy primaries.\n"
        "- Supporter definitions differ across paradigms by design (that is part of the treatment), "
        "so loser-share differences should be read together with loser welfare, not alone.\n"
    )

    parts.append("## Reproduce\n")
    parts.append("```\n" + manifest["command"] + "\n```\n")
    parts.append(
        "Deterministic given identical parameters and seed (per-trial streams are derived from "
        "`numpy.random.default_rng([seed, population, candidates, trial])`).\n"
    )

    path.write_text("\n".join(parts))

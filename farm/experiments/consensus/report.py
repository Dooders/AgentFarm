"""Auto-generated markdown report for the consensus experiment.

Every number in the report is computed from the trials DataFrame of the run
that produced it; nothing is hard-coded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from farm.experiments.consensus.contrasts import CONTRAST_RULES, paired_contrasts
from farm.experiments.consensus.paradigms import BASELINE_PARADIGMS, CONSTRAINED_PARADIGM, SELECTION_PARADIGMS
from farm.experiments.consensus.population import PROJECTS

BASELINE = "party"


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


def _summary_display(summary: pd.DataFrame, paradigms: list[str]) -> pd.DataFrame:
    """Compact mean ± std table for the report."""
    metrics = [
        "total_welfare",
        "minority_welfare",
        "majority_welfare",
        "min_utility",
        "p10_utility",
        "gini_utility",
        "lambda_winner",
        "loser_share",
        "loser_welfare",
        "gap",
    ]
    subset = summary[summary["paradigm"].isin(paradigms)].copy()
    out = subset[["population", "n_candidates", "paradigm"]].copy()
    for metric in metrics:
        mean_col, std_col = f"{metric}_mean", f"{metric}_std"
        if mean_col not in subset.columns:
            continue
        out[metric] = [
            f"{mean:.4f} ± {std:.4f}" if np.isfinite(mean) else "nan"
            for mean, std in zip(subset[mean_col], subset[std_col])
        ]
    return out


def _question_text(config: dict) -> str:
    mechanism = config.get("mechanism", "oneshot")
    correlated = bool(config.get("lambda_correlated"))
    if mechanism == "reelection":
        return (
            "When winners *choose* λ to maximize the mean utility of a random "
            "observation sample plus a weight on loyal targeting, do party and "
            "individual-centered winners choose different λ because of coalition "
            "size — and does that change minority-cluster (fixed-partition) welfare?"
        )
    if correlated:
        return (
            "Robustness appendix (not the primary cell): when high λ is rank-coupled "
            "to platform extremity, do centrist-favoring rules select lower-λ winners, "
            "and does that raise minority-cluster welfare?"
        )
    return (
        "Holding λ's marginal distribution fixed and holding the set of people "
        "fixed, do individual-centered ballot formats change the winner's allocation "
        "in a way that raises minority-cluster welfare and/or total welfare relative "
        "to party? Election-endogenous loser welfare is reported but is not the "
        "primary contrast: 'supporters' is a different estimand under every rule."
    )


def _methods_lambda_line(config: dict) -> str:
    if config.get("mechanism") == "reelection":
        return (
            "loyalty λ is *chosen* by the winner on a 21-point grid to maximize "
            "re-election rate (observers retain if utility ≥ the λ=0 allocation) "
            "plus 0.15·λ; the drawn Beta type is unused for allocation."
        )
    if config.get("lambda_correlated"):
        return (
            "loyalty trait λ ~ Beta(2.2, 2.2), rank-coupled to platform extremity "
            "(`--lambda-correlated`). This is a robustness appendix, not the primary cell."
        )
    return (
        "loyalty trait λ ~ Beta(2.2, 2.2), independent of the platform and of "
        "anything voters see. No rule reads λ, so E[λ_winner] is the Beta mean "
        "under every paradigm by construction — not an empirical finding."
    )


def _contrast_lines(cell: pd.DataFrame) -> list[str]:
    if cell.empty:
        return ["- No paired contrasts for this cell."]
    lines: list[str] = []
    primary = cell[cell["family"] == "primary"]
    for _, row in primary.iterrows():
        adj = row["pvalue_adjusted"]
        adj_s = f"{adj:.4g}" if np.isfinite(adj) else "nan"
        lines.append(
            f"- `{row['paradigm']}` vs `{BASELINE}` on `{row['endpoint']}`: "
            f"Δ = {row['delta_mean']:+.5f} (95% CI [{row['ci_low']:+.5f}, {row['ci_high']:+.5f}]), "
            f"paired Cohen's d = {row['effect_size']:+.3f}, Wilcoxon p = {row['pvalue']:.4g}, "
            f"Holm-adjusted p = {adj_s}."
        )
    mushy = []
    for rule in CONTRAST_RULES:
        gap = cell[(cell["paradigm"] == rule) & (cell["endpoint"] == "gap")]
        minority = cell[(cell["paradigm"] == rule) & (cell["endpoint"] == "minority_welfare")]
        if gap.empty or minority.empty:
            continue
        if gap.iloc[0]["delta_mean"] < 0 and minority.iloc[0]["delta_mean"] <= 0:
            mushy.append(rule)
    lines.append("")
    lines.append("Mushy-bloc check (fixed partition):")
    if mushy:
        lines.append(
            "- election-endogenous gap shrank while minority-cluster welfare did not rise "
            f"for {', '.join(f'`{p}`' for p in mushy)}."
        )
    else:
        lines.append(
            "- not triggered — no rule shrank the election-endogenous gap without a "
            "rise in minority-cluster welfare."
        )
    return lines


def render_report(
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    allocations: pd.DataFrame,
    run_config: dict,
    contrasts: pd.DataFrame | None = None,
) -> str:
    config = run_config["config"]
    populations = list(trials["population"].unique())
    all_paradigms = list(trials["paradigm"].unique())
    selection = [p for p in all_paradigms if p in SELECTION_PARADIGMS]
    baselines = [p for p in all_paradigms if p in BASELINE_PARADIGMS]
    constrained = CONSTRAINED_PARADIGM in all_paradigms
    include_lambda = bool(config.get("lambda_correlated") or config.get("mechanism") == "reelection")
    if contrasts is None:
        contrasts = paired_contrasts(trials, include_lambda_primary=include_lambda)

    parts: list[str] = []
    parts.append("# Political consensus experiment — auto-generated report\n")
    parts.append(_question_text(config) + "\n")

    parts.append("## Methods\n")
    parts.append(
        f"- **Population**: {config['voters']} voters per trial, latent preference vectors in R^5 drawn "
        f"from cluster centers plus Gaussian noise (σ = 0.35); benefit vectors are a softmax "
        f"(temperature 0.55) of preferences, so rows are nonnegative and sum to 1. "
        f"Population types run here: {', '.join(f'`{p}`' for p in populations)}.\n"
        f"- **Projects** (generator tags, not winner-relative): {', '.join(f'`{p}`' for p in PROJECTS)}.\n"
        f"- **Candidates**: {config['candidates']} per trial (sweeps may vary this), platforms drawn from the "
        f"same cluster structure; {_methods_lambda_line(config)}\n"
        f"- **Allocation**: `directed = λ·mean(benefits[supporters]) + (1−λ)·mean(benefits[all])`, then "
        f"`raw = 0.72·directed + 0.28·simplex(clip(platform, 0, ∞))`. The platform is renormalized "
        f"before mixing so the weights are a convex combination of matched units. "
        f"Zero supporters ⇒ all-voter direction only. A platform with no positive mass is dropped.\n"
        f"- **Paradigms (selection treatments)**: {', '.join(f'`{p}`' for p in selection)}. "
        f"`constrained_individual` is a constitutional λ cap, not a voting rule"
        + (
            f", and is reported separately below (cap {config['lambda_cap']}).\n"
            if constrained
            else "; it is not in this run.\n"
        )
        + f"- **Voting**: `{config.get('voting', 'sincere')}`. Sincere plurality is the default baseline; "
        f"`abandon_trailing` is a Duverger-style heuristic on individual (and the cap overlay).\n"
        f"- **Mechanism**: `{config.get('mechanism', 'oneshot')}`. The default one-shot cell is a "
        f"selection-rule comparison with exogenous types, not a test of loyalty formation.\n"
        f"- **Trials**: {config['trials']} per cell, base seed {config['seed']}; every paradigm sees the "
        f"identical population and candidate slate within a trial.\n"
        f"- **Primary endpoints**: Δ minority-cluster welfare and Δ total welfare vs party"
        + (", plus Δλ_winner" if include_lambda else "")
        + ". Holm correction across that family. Wilcoxon signed-rank on paired trial-level "
        "differences; 95% CIs are Student-t; effect size is paired Cohen's d.\n"
        "- **Fixed partition**: welfare is reported on generator clusters (PCA split for "
        "`one_cluster`). Election-endogenous supporter/loser numbers are kept and labeled as "
        "such; they are not the primary contrast. `rural_town` party `loser_share` equals the "
        "minority bloc size by construction (~0.30) and is not a treatment effect.\n"
        "- **Baselines** (same population + candidate draw): `random_winner` (uniform candidate, "
        "nearest-pref supporters), `utilitarian` (simplex vertex maximizing mean utility), "
        "`egalitarian` (maximin LP). Normalized welfare is `(metric − random) / (utilitarian − random)` "
        "when the denominator is nonzero.\n"
        "- **Audit**: synthetic ballots, supporter masks, and cluster ids are written under "
        "`private/` when `--persist-ballots` is on (the default). They are not a privacy "
        "claim and they are not notarized. Official record: `summary.csv`, `trials.csv` aggregates, "
        "`contrasts.csv`.\n"
    )

    parts.append("## Results\n")
    parts.append(
        "Mean ± std over trials for **selection rules**. `loser_share` / `loser_welfare` / `gap` "
        "are election-endogenous (the loser *set* changes with the rule) and are not the "
        "primary contrast.\n"
    )
    parts.append(_markdown_table(_summary_display(summary, selection)) + "\n")

    if baselines:
        parts.append("### Allocation baselines (not selection treatments)\n")
        parts.append(
            "Random-winner floor and utilitarian / egalitarian brackets on the same draws. "
            "Normalized columns live on every row of `trials.csv`.\n"
        )
        parts.append(_markdown_table(_summary_display(summary, baselines)) + "\n")

    if constrained:
        parts.append("### Constitutional cap (not a selection treatment)\n")
        parts.append(
            "`constrained_individual` reuses the `individual` election and caps `λ_effective`. "
            "It manipulates the outcome function, not the selection rule, and is excluded from "
            "hypothesis contrasts and from 'which rule wins' language.\n"
        )
        parts.append(_markdown_table(_summary_display(summary, [CONSTRAINED_PARADIGM])) + "\n")

    parts.append("### Mean winner allocations\n")
    alloc_display = allocations[allocations["paradigm"].isin(selection + baselines)].copy()
    alloc_display.columns = [c.replace("alloc_", "") for c in alloc_display.columns]
    parts.append(_markdown_table(alloc_display) + "\n")

    parts.append("## Paired contrasts vs party\n")
    parts.append(
        "Same-trial differences. Primary endpoints carry Holm-adjusted p-values. "
        "Threshold constants are not used as a verdict machine.\n"
    )
    if not include_lambda and config.get("mechanism", "oneshot") == "oneshot":
        parts.append(
            "λ_winner is **not** a primary endpoint in this cell. Voters never see λ, so a "
            "flat λ profile is implied by the generator.\n"
        )
    contrast_keys = ["population", "n_candidates"]
    present_keys = [k for k in contrast_keys if k in contrasts.columns]
    if contrasts.empty:
        parts.append("No contrasts computed.\n")
    else:
        grouped = contrasts.groupby(present_keys, sort=False) if present_keys else [((), contrasts)]
        for key, cell in grouped:
            key = (key,) if not isinstance(key, tuple) else key
            label = ", ".join(str(v) for v in key)
            parts.append(f"### {label}\n")
            display = cell[cell["family"] == "primary"][
                ["paradigm", "endpoint", "delta_mean", "ci_low", "ci_high", "pvalue", "pvalue_adjusted", "effect_size"]
            ].copy()
            parts.append(_markdown_table(display) + "\n")
            parts.append("\n".join(_contrast_lines(cell)) + "\n")

    parts.append("## Limitations\n")
    parts.append(
        "- Voters and candidates live in a stylized 5-dimensional project space; real preference "
        "structures are higher-dimensional and partly unobservable.\n"
        "- The default one-shot cell has exogenous λ. It compares selection rules with drawn "
        "types; it does not test citizen-candidate entry or core- vs swing-voter targeting. "
        "Use `--mechanism reelection` for an incentive-based λ cell (still not an entry model).\n"
        "- Party structure is idealized as two brands at cluster means with loyal nomination; real "
        "parties select through noisy primaries.\n"
        "- Supporter definitions differ across paradigms by design. Primary welfare contrasts "
        "therefore use the fixed cluster partition, not the election-endogenous loser set.\n"
        "- `--lambda-correlated` is a researcher degree of freedom that can flip λ rankings. "
        "It is labeled a robustness appendix whenever it is the condition being reported.\n"
    )

    parts.append("## Reproduce\n")
    parts.append("```\n" + run_config["command"] + "\n```\n")
    parts.append(
        "Deterministic given identical parameters and seed (per-trial streams are derived from "
        "`numpy.random.default_rng([seed, population, candidates, trial])`).\n"
    )

    return "\n".join(parts)


def write_report(
    trials: pd.DataFrame,
    summary: pd.DataFrame,
    allocations: pd.DataFrame,
    run_config: dict,
    path: Path,
    contrasts: pd.DataFrame | None = None,
) -> None:
    path.write_text(render_report(trials, summary, allocations, run_config, contrasts=contrasts))

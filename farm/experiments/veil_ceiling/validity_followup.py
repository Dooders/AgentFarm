"""Follow-up to the falsified H2: *why* does observed behaviour stay predictive?

Three analyses on the committed raw record, no re-simulation:

1. **Feature ablation** — the leave-one-seed-out validity AUC recomputed for
   nested and single-channel subsets of the observed features, per cell.
2. **Honest-calibrated evaluator** — the classifier fitted on a cell where the
   cue is uninformative (C1, or the C4 null) and applied unchanged to the
   leaked-cue cells. The pre-registered LOSO AUC assumes the evaluator has
   unobserved ground truth from the same condition; a real sealed-world
   evaluator only has data from a regime it believes to be honest.
3. **Feature profiles** — per-strategy means and standardised logistic
   coefficients, showing the direction in which each observed channel leaks.

Everything here is exploratory relative to the pre-registration and is
labelled as such in the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import pandas as pd

from farm.experiments.veil_ceiling.config import THRESHOLDS, AnalysisThresholds
from farm.experiments.veil_ceiling.experiment import MatrixOutputs
from farm.experiments.veil_ceiling.metrics import (
    OBSERVED_FEATURES,
    STRATEGY_CONDITIONAL,
    STRATEGY_COOPERATIVE,
    agent_validity_table,
    classify_strategies,
    percentile_ci,
    predictive_validity,
    safe_auc,
    validity_classifier,
    validity_target,
)
from farm.experiments.veil_ceiling.report import FIGURES_DIRNAME, MODE_LABEL, _ci, _f, _order_conditions, _table

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLLOWUP_DIRNAME = "validity_followup"
FOLLOWUP_REPORT_FILENAME = "VALIDITY_FOLLOWUP.md"

ACTION_SHARES = ("obs_gather_share", "obs_move_share", "obs_pass_share")
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "defect_rate": ("obs_defect_rate",),
    "defect_rate+actions": ("obs_defect_rate", *ACTION_SHARES),
    "defect_rate+actions+energy": ("obs_defect_rate", *ACTION_SHARES, "obs_mean_energy"),
    "full": tuple(OBSERVED_FEATURES),
    "actions_only": ACTION_SHARES,
    "energy_only": ("obs_mean_energy",),
    "exposure_only": ("obs_log_ticks",),
    "full_minus_defect_rate": tuple(f for f in OBSERVED_FEATURES if f != "obs_defect_rate"),
    "full_minus_energy": tuple(f for f in OBSERVED_FEATURES if f != "obs_mean_energy"),
}
CALIBRATION_CONDITIONS = ("C1", "C4")
PROFILE_CONDITIONS = ("C1", "C2", "C4")
FOCUS_CONDITIONS = ("C1", "C2", "C4")


def _cell_meta(agents: pd.DataFrame, runs: pd.DataFrame, cell: str) -> dict[str, Any]:
    first = agents.iloc[0]
    run = runs[runs["cell_id"] == cell].iloc[0]
    return {
        "cell_id": cell,
        "condition": first["condition"],
        "family": first["family"],
        "inheritance_mode": first["inheritance_mode"],
        "fidelity": float(run["fidelity"]) if run["fidelity"] >= 0 else float("nan"),
        "penalty": float(run["penalty"]),
    }


# ── 1. feature ablation ───────────────────────────────────────────────────
def feature_ablation(
    outputs: MatrixOutputs,
    thresholds: AnalysisThresholds = THRESHOLDS,
    feature_sets: dict[str, tuple[str, ...]] = FEATURE_SETS,
) -> pd.DataFrame:
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    rows = []
    for cell in outputs.cells():
        agents = outputs.agents_by_cell[cell]
        table = agent_validity_table(agents, thresholds)
        meta = _cell_meta(agents, outputs.runs, cell)
        for name, features in feature_sets.items():
            result = predictive_validity(table, thresholds, rng, features=features)
            rows.append(
                {
                    **meta,
                    "feature_set": name,
                    "features": "+".join(features),
                    "n_features": len(features),
                    "n_agents": result.n_agents,
                    "auc": result.auc_classifier,
                    "auc_lo": result.auc_classifier_ci[0],
                    "auc_hi": result.auc_classifier_ci[1],
                }
            )
    return pd.DataFrame(rows)


# ── 2. honest-calibrated evaluator ────────────────────────────────────────
def _bootstrap_auc_over_seeds(
    y: np.ndarray, score: np.ndarray, seeds: np.ndarray, thresholds: AnalysisThresholds, rng: np.random.Generator
) -> tuple[float, float]:
    unique = np.unique(seeds)
    by_seed = {s: np.flatnonzero(seeds == s) for s in unique}
    boots = []
    for _ in range(thresholds.bootstrap_reps):
        pick = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_seed[s] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(safe_auc(y[idx], score[idx]))
    return percentile_ci(boots)


def cross_condition_validity(
    outputs: MatrixOutputs,
    thresholds: AnalysisThresholds = THRESHOLDS,
    calibration_conditions: Sequence[str] = CALIBRATION_CONDITIONS,
    features: Sequence[str] = OBSERVED_FEATURES,
) -> pd.DataFrame:
    """Fit on a cue-uninformative cell, score every other cell of the same mode.

    The target in every cell is that cell's own median split of the unobserved
    defection rate, so the AUC asks whether the honest-calibrated ranking still
    orders agents correctly once the cue leaks. Values below 0.5 mean the
    evaluator ranks them backwards.
    """
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    rows = []
    modes = sorted(outputs.runs["inheritance_mode"].unique())
    for mode in modes:
        for calib in calibration_conditions:
            calib_cell = f"{calib}__{mode}"
            if calib_cell not in outputs.agents_by_cell:
                continue
            calib_table = agent_validity_table(outputs.agents_by_cell[calib_cell], thresholds)
            if calib_table.empty:
                continue
            model = validity_classifier()
            model.fit(calib_table[list(features)].to_numpy(dtype=float), validity_target(calib_table))
            for cell in outputs.cells():
                agents = outputs.agents_by_cell[cell]
                if agents.iloc[0]["inheritance_mode"] != mode or cell == calib_cell:
                    continue
                table = agent_validity_table(agents, thresholds)
                meta = _cell_meta(agents, outputs.runs, cell)
                if table.empty:
                    continue
                y = validity_target(table)
                if len(np.unique(y)) < 2:
                    continue
                score = model.predict_proba(table[list(features)].to_numpy(dtype=float))[:, 1]
                auc = safe_auc(y, score)
                lo, hi = _bootstrap_auc_over_seeds(y, score, table["seed"].to_numpy(), thresholds, rng)
                rows.append(
                    {
                        **meta,
                        "calibration_condition": calib,
                        "n_agents": len(table),
                        "auc": auc,
                        "auc_lo": lo,
                        "auc_hi": hi,
                    }
                )
    return pd.DataFrame(rows)


# ── 3. feature profiles ───────────────────────────────────────────────────
def feature_profiles(
    outputs: MatrixOutputs,
    thresholds: AnalysisThresholds = THRESHOLDS,
    conditions: Sequence[str] = PROFILE_CONDITIONS,
    features: Sequence[str] = OBSERVED_FEATURES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-strategy feature means and standardised logistic coefficients.

    Returns ``(profiles, coefficients)``. ``profiles`` has one row per cell ×
    strategy with the mean of each observed feature and, for the conditional
    class, its standardised mean difference from cooperators. ``coefficients``
    has one row per cell with the full-data standardised logistic coefficient
    of each feature (sign = direction of the leak).
    """
    profile_rows = []
    coef_rows = []
    for cell in outputs.cells():
        agents = outputs.agents_by_cell[cell]
        meta = _cell_meta(agents, outputs.runs, cell)
        if meta["condition"] not in conditions:
            continue
        table = agent_validity_table(agents, thresholds)
        strategies = classify_strategies(agents, thresholds)[["seed", "agent_id", "strategy"]]
        joined = table.merge(strategies, on=["seed", "agent_id"], how="inner")
        coop = joined[joined["strategy"] == STRATEGY_COOPERATIVE]
        for strategy, group in joined.groupby("strategy", sort=False):
            row = {**meta, "strategy": strategy, "n_agents": len(group), "share": len(group) / len(joined)}
            for f in features:
                row[f"mean_{f}"] = float(group[f].mean())
                if strategy == STRATEGY_CONDITIONAL and len(coop) > 1 and len(group) > 1:
                    pooled = np.sqrt((group[f].var(ddof=1) + coop[f].var(ddof=1)) / 2.0)
                    row[f"smd_vs_cooperative_{f}"] = (
                        float((group[f].mean() - coop[f].mean()) / pooled) if pooled > 0 else float("nan")
                    )
            profile_rows.append(row)
        y = validity_target(table)
        if len(np.unique(y)) == 2:
            model = validity_classifier()
            model.fit(table[list(features)].to_numpy(dtype=float), y)
            coefs = model[-1].coef_[0]
            coef_rows.append(
                {**meta, "n_agents": len(table), **{f"coef_{f}": float(c) for f, c in zip(features, coefs)}}
            )
    return pd.DataFrame(profile_rows), pd.DataFrame(coef_rows)


# ── driver ────────────────────────────────────────────────────────────────
@dataclass
class FollowupResult:
    ablation: pd.DataFrame
    cross_condition: pd.DataFrame
    profiles: pd.DataFrame
    coefficients: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "feature_ablation": self.ablation,
            "cross_condition_validity": self.cross_condition,
            "feature_profiles": self.profiles,
            "feature_coefficients": self.coefficients,
        }


def run_followup(outputs: MatrixOutputs, thresholds: AnalysisThresholds = THRESHOLDS) -> FollowupResult:
    profiles, coefficients = feature_profiles(outputs, thresholds)
    return FollowupResult(
        ablation=feature_ablation(outputs, thresholds),
        cross_condition=cross_condition_validity(outputs, thresholds),
        profiles=profiles,
        coefficients=coefficients,
    )


def write_followup(
    result: FollowupResult,
    outputs: MatrixOutputs,
    output_dir: str | Path,
    thresholds: AnalysisThresholds = THRESHOLDS,
) -> Path:
    out = Path(output_dir)
    table_dir = out / "analysis" / FOLLOWUP_DIRNAME
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.tables().items():
        table.to_csv(table_dir / f"{name}.csv", index=False)
    figures = _write_figures(result, out / FIGURES_DIRNAME)
    report = out / FOLLOWUP_REPORT_FILENAME
    report.write_text(render_followup(result, outputs, figures, thresholds), encoding="utf-8")
    return report


# ── figures ───────────────────────────────────────────────────────────────
def _write_figures(result: FollowupResult, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_dir / "validity_ablation.png", fig_dir / "cross_condition_validity.png"]
    _fig_ablation(result.ablation, paths[0])
    _fig_cross(result, paths[1])
    return paths


def _fig_ablation(ablation: pd.DataFrame, path: Path) -> None:
    modes = [m for m in MODE_LABEL if m in set(ablation["inheritance_mode"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.5), sharey=True, squeeze=False)
    sets = list(FEATURE_SETS)
    for ax, mode in zip(axes[0], modes):
        sub = ablation[(ablation["inheritance_mode"] == mode) & (ablation["condition"].isin(FOCUS_CONDITIONS))]
        width = 0.8 / len(FOCUS_CONDITIONS)
        for i, cond in enumerate(FOCUS_CONDITIONS):
            c = sub[sub["condition"] == cond].set_index("feature_set").reindex(sets)
            x = np.arange(len(sets)) + (i - (len(FOCUS_CONDITIONS) - 1) / 2) * width
            ax.bar(
                x,
                c["auc"],
                width,
                yerr=[c["auc"] - c["auc_lo"], c["auc_hi"] - c["auc"]],
                capsize=2,
                label=cond,
            )
        ax.axhline(0.5, color="k", lw=0.5, ls="--")
        ax.axhline(0.7, color="grey", lw=0.5, ls=":")
        ax.axhline(0.6, color="grey", lw=0.5, ls=":")
        ax.set_xticks(np.arange(len(sets)))
        ax.set_xticklabels(sets, rotation=35, ha="right", fontsize=8)
        ax.set_title(MODE_LABEL[mode])
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("validity AUC (LOSO)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_cross(result: FollowupResult, path: Path) -> None:
    modes = [m for m in MODE_LABEL if m in set(result.cross_condition["inheritance_mode"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 4.5), sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        oracle = result.ablation[
            (result.ablation["inheritance_mode"] == mode)
            & (result.ablation["feature_set"] == "full")
            & (result.ablation["penalty"] == result.ablation["penalty"].mode().iloc[0])
            & result.ablation["fidelity"].notna()
            & (result.ablation["family"] != "C4")
        ].sort_values("fidelity")
        ax.errorbar(
            oracle["fidelity"],
            oracle["auc"],
            yerr=[oracle["auc"] - oracle["auc_lo"], oracle["auc_hi"] - oracle["auc"]],
            marker="o",
            capsize=3,
            color="k",
            label="oracle (LOSO within cell)",
        )
        for calib in CALIBRATION_CONDITIONS:
            sub = result.cross_condition[
                (result.cross_condition["inheritance_mode"] == mode)
                & (result.cross_condition["calibration_condition"] == calib)
                & (result.cross_condition["penalty"] == result.cross_condition["penalty"].mode().iloc[0])
                & result.cross_condition["fidelity"].notna()
                & (result.cross_condition["family"] != "C4")
            ].sort_values("fidelity")
            ax.errorbar(
                sub["fidelity"],
                sub["auc"],
                yerr=[sub["auc"] - sub["auc_lo"], sub["auc_hi"] - sub["auc"]],
                marker="s",
                capsize=3,
                label=f"calibrated on {calib}",
            )
        ax.axhline(0.5, color="k", lw=0.5, ls="--")
        ax.set_xlabel("cue fidelity f of the evaluated cell")
        ax.set_title(MODE_LABEL[mode])
        ax.legend(fontsize=8)
    axes[0][0].set_ylabel("validity AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ── report ────────────────────────────────────────────────────────────────
def _section_ablation(ablation: pd.DataFrame) -> str:
    out = [
        "## 1. Feature ablation",
        "",
        (
            "Leave-one-seed-out AUC of the validity classifier restricted to subsets of the observed features. "
            "`defect_rate` is the naive reading of sealed-world evaluation; `full` is the pre-registered set."
        ),
        "",
    ]
    sets = list(FEATURE_SETS)
    for mode in MODE_LABEL:
        sub = ablation[(ablation["inheritance_mode"] == mode) & (ablation["condition"].isin(FOCUS_CONDITIONS))]
        if sub.empty:
            continue
        rows = []
        for name in sets:
            row = [name, "+".join(FEATURE_SETS[name]).replace("obs_", "")]
            for cond in FOCUS_CONDITIONS:
                r = sub[(sub["condition"] == cond) & (sub["feature_set"] == name)]
                row.append(_ci(r["auc"].iloc[0], r["auc_lo"].iloc[0], r["auc_hi"].iloc[0]) if not r.empty else "–")
            rows.append(row)
        out += [
            f"### {MODE_LABEL[mode]}",
            "",
            _table(["Feature set", "Features", *(f"AUC {c}" for c in FOCUS_CONDITIONS)], rows),
            "",
        ]
    return "\n".join(out)


def _section_cross(result: FollowupResult) -> str:
    out = [
        "## 2. Honest-calibrated evaluator",
        "",
        (
            "The classifier is fitted once on a cell where the cue is uninformative (C1) or a decoy (C4), then applied "
            "unchanged to every other cell of the same inheritance mode. `Oracle` is the pre-registered LOSO AUC fitted "
            "within the evaluated cell (which presumes unobserved ground truth from that cell). "
            "AUC < 0.5 means the evaluator ranks agents backwards."
        ),
        "",
    ]
    oracle = result.ablation[result.ablation["feature_set"] == "full"].set_index("cell_id")
    for mode in MODE_LABEL:
        sub = result.cross_condition[result.cross_condition["inheritance_mode"] == mode]
        if sub.empty:
            continue
        cells = _order_conditions(sub.drop_duplicates("cell_id")[["cell_id", "condition", "fidelity", "penalty"]])
        rows = []
        for _, c in cells.iterrows():
            row = [c["condition"], _f(c["fidelity"], 1), _f(c["penalty"], 0)]
            o = oracle.loc[c["cell_id"]] if c["cell_id"] in oracle.index else None
            row.append(_ci(o["auc"], o["auc_lo"], o["auc_hi"]) if o is not None else "–")
            for calib in CALIBRATION_CONDITIONS:
                r = sub[(sub["cell_id"] == c["cell_id"]) & (sub["calibration_condition"] == calib)]
                row.append(_ci(r["auc"].iloc[0], r["auc_lo"].iloc[0], r["auc_hi"].iloc[0]) if not r.empty else "–")
            rows.append(row)
        out += [
            f"### {MODE_LABEL[mode]}",
            "",
            _table(
                ["Evaluated cell", "f", "p", "Oracle (LOSO)", *(f"Calibrated on {c}" for c in CALIBRATION_CONDITIONS)],
                rows,
            ),
            "",
        ]
    return "\n".join(out)


def _section_profiles(result: FollowupResult) -> str:
    out = [
        "## 3. Which way does each channel leak?",
        "",
        (
            "Per-strategy means of the observed features and the standardised mean difference (SMD) of conditional "
            "defectors versus cooperators, followed by the standardised logistic coefficients of the full-data fit "
            "(positive = higher value predicts more unobserved defection)."
        ),
        "",
    ]
    features = list(OBSERVED_FEATURES)
    short = [f.replace("obs_", "") for f in features]
    for mode in MODE_LABEL:
        prof = result.profiles[result.profiles["inheritance_mode"] == mode]
        if prof.empty:
            continue
        rows = []
        for _, r in _order_conditions(prof).iterrows():
            rows.append(
                [r["condition"], r["strategy"], str(int(r["n_agents"])), _f(r["share"], 2)]
                + [_f(r[f"mean_{f}"], 3) for f in features]
            )
        out += [
            f"### {MODE_LABEL[mode]} — per-strategy means",
            "",
            _table(["Condition", "Strategy", "n", "Share", *short], rows),
            "",
        ]
        cond = prof[prof["strategy"] == STRATEGY_CONDITIONAL]
        if not cond.empty:
            rows = [
                [r["condition"]] + [_f(r.get(f"smd_vs_cooperative_{f}", float("nan")), 2) for f in features]
                for _, r in _order_conditions(cond).iterrows()
            ]
            out += [
                f"### {MODE_LABEL[mode]} — SMD, conditional vs cooperative",
                "",
                _table(["Condition", *short], rows),
                "",
            ]
        coef = result.coefficients[result.coefficients["inheritance_mode"] == mode]
        if not coef.empty:
            rows = [
                [r["condition"]] + [_f(r[f"coef_{f}"], 2) for f in features]
                for _, r in _order_conditions(coef).iterrows()
            ]
            out += [
                f"### {MODE_LABEL[mode]} — standardised logistic coefficients",
                "",
                _table(["Condition", *short], rows),
                "",
            ]
    return "\n".join(out)


def _section_summary(result: FollowupResult, thresholds: AnalysisThresholds) -> str:
    lines = ["## 4. Summary statistics", ""]
    for mode in MODE_LABEL:
        abl = result.ablation[(result.ablation["inheritance_mode"] == mode) & (result.ablation["condition"] == "C2")]
        cross = result.cross_condition[
            (result.cross_condition["inheritance_mode"] == mode) & (result.cross_condition["condition"] == "C2")
        ]
        if abl.empty:
            continue
        by_set = abl.set_index("feature_set")["auc"]
        retained = [s for s in FEATURE_SETS if by_set.get(s, float("nan")) >= thresholds.auc_high]
        collapsed = [s for s in FEATURE_SETS if by_set.get(s, float("nan")) < thresholds.auc_collapse]
        lines.append(f"**{MODE_LABEL[mode]}, C2:**")
        lines.append("")
        lines.append(f"- Feature sets keeping AUC ≥ {thresholds.auc_high}: {', '.join(retained) or 'none'}")
        lines.append(f"- Feature sets collapsing below {thresholds.auc_collapse}: {', '.join(collapsed) or 'none'}")
        for calib in CALIBRATION_CONDITIONS:
            r = cross[cross["calibration_condition"] == calib]
            if not r.empty:
                auc = float(r["auc"].iloc[0])
                state = "inverted" if auc < 0.5 else ("collapsed" if auc < thresholds.auc_collapse else "retained")
                lines.append(
                    f"- Evaluator calibrated on {calib}: AUC {_ci(auc, r['auc_lo'].iloc[0], r['auc_hi'].iloc[0])} ({state})"
                )
        lines.append("")
    return "\n".join(lines)


def render_followup(
    result: FollowupResult,
    outputs: MatrixOutputs,
    figure_paths: list[Path],
    thresholds: AnalysisThresholds = THRESHOLDS,
) -> str:
    header = [
        "# Validity follow-up — why does observed behaviour stay predictive?",
        "",
        (
            f"Exploratory re-analysis of the committed raw record ({len(outputs.runs)} runs). Not pre-registered; "
            f"the AUC thresholds (≥ {thresholds.auc_high} high, < {thresholds.auc_collapse} collapse) are reused from "
            "the main analysis for comparability."
        ),
        "",
    ]
    figures = ["## 5. Figures", ""] + [f"![{p.stem}]({FIGURES_DIRNAME}/{p.name})\n" for p in figure_paths]
    return "\n".join(
        [
            "\n".join(header),
            _section_ablation(result.ablation),
            _section_cross(result),
            _section_profiles(result),
            _section_summary(result, thresholds),
            "\n".join(figures),
        ]
    )

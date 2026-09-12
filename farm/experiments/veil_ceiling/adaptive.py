"""Adaptive-monitor follow-up: can a fixed coverage budget catch concealment?

The monitor keeps ``c`` and ``p`` at the pre-registered values and reallocates
the next epoch's true map from last epoch's observed residual. Three policies:

* ``adaptive_cells`` — overweight cells with recent *observed defections*.
* ``adaptive_move`` — overweight cells with recent *observed movement* (the
  side channel that survived the H2 ablation).
* ``adaptive_blind`` — overweight cells with recent *observed occupancy*
  (agent-ticks on the current true map). Control: is the monitor just chasing
  busy cells rather than caught concealment?

Every comparison is seed-matched against the static cell of the same fidelity
(C2 / C3_f0.9 / C3_f0.7). Coverage is exact in every epoch; expected penalty
per defection at training start is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from farm.experiments.veil_ceiling.analysis import (
    AnalysisResult,
    analyze,
    paired_contrast,
    write_analysis,
)
from farm.experiments.veil_ceiling.config import (
    ADAPTIVE_FIDELITY_SWEEP,
    MONITOR_POLICY_ADAPTIVE_BLIND,
    MONITOR_POLICY_ADAPTIVE_CELLS,
    MONITOR_POLICY_ADAPTIVE_MOVE,
    THRESHOLDS,
    AnalysisThresholds,
)
from farm.experiments.veil_ceiling.experiment import MatrixOutputs
from farm.experiments.veil_ceiling.metrics import OBSERVED_FEATURES
from farm.experiments.veil_ceiling.report import FIGURES_DIRNAME, MODE_LABEL, _ci, _f, _table
from farm.experiments.veil_ceiling.validity_followup import cross_condition_validity

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ADAPTIVE_REPORT_FILENAME = "ADAPTIVE_REPORT.md"
ADAPTIVE_DIRNAME = "adaptive"

STATIC_MATCH = {1.0: "C2", 0.9: "C3_f0.9", 0.7: "C3_f0.7"}
POLICY_LABEL = {
    MONITOR_POLICY_ADAPTIVE_CELLS: "cells (observed defections)",
    MONITOR_POLICY_ADAPTIVE_MOVE: "move (observed movement)",
    MONITOR_POLICY_ADAPTIVE_BLIND: "blind (spatial density)",
    "static": "static",
}
ADAPTIVE_POLICIES = (
    MONITOR_POLICY_ADAPTIVE_CELLS,
    MONITOR_POLICY_ADAPTIVE_MOVE,
    MONITOR_POLICY_ADAPTIVE_BLIND,
)
CONTRAST_METRICS = (
    "rate",
    "delta_cue",
    "delta_true",
    "realised_enforcement",
    "mean_mask_overlap",
    "mean_weight_kl",
    "eval_heldout_delta_cue",
)


def _cell(run_metrics: pd.DataFrame, condition: str, mode: str) -> pd.DataFrame:
    sel = run_metrics[(run_metrics["condition"] == condition) & (run_metrics["inheritance_mode"] == mode)]
    return sel.set_index("seed")


def compute_adaptive_contrasts(run_metrics: pd.DataFrame, thresholds: AnalysisThresholds = THRESHOLDS) -> pd.DataFrame:
    """Seed-matched adaptive − static contrasts at matched fidelity, plus policy vs policy."""
    rows = []
    adaptive = run_metrics[run_metrics["family"] == "A"]
    if adaptive.empty:
        return pd.DataFrame()
    for mode in sorted(run_metrics["inheritance_mode"].unique()):
        for fidelity in ADAPTIVE_FIDELITY_SWEEP:
            static_name = STATIC_MATCH[fidelity]
            static = _cell(run_metrics, static_name, mode)
            if static.empty:
                continue
            for policy in ADAPTIVE_POLICIES:
                treat = adaptive[
                    (adaptive["inheritance_mode"] == mode)
                    & (adaptive["fidelity"] == fidelity)
                    & (adaptive["monitor_policy"] == policy)
                ]
                if treat.empty:
                    continue
                t = treat.set_index("seed")
                for metric in CONTRAST_METRICS:
                    if metric not in t or metric not in static:
                        continue
                    row = {
                        "inheritance_mode": mode,
                        "fidelity": fidelity,
                        "treatment": treat["condition"].iloc[0],
                        "baseline": static_name,
                        "policy": policy,
                        "contrast": "adaptive-static",
                        "metric": metric,
                    }
                    row.update(paired_contrast(t[metric], static[metric], thresholds))
                    rows.append(row)
            cells = adaptive[
                (adaptive["inheritance_mode"] == mode)
                & (adaptive["fidelity"] == fidelity)
                & (adaptive["monitor_policy"] == MONITOR_POLICY_ADAPTIVE_CELLS)
            ]
            move = adaptive[
                (adaptive["inheritance_mode"] == mode)
                & (adaptive["fidelity"] == fidelity)
                & (adaptive["monitor_policy"] == MONITOR_POLICY_ADAPTIVE_MOVE)
            ]
            blind = adaptive[
                (adaptive["inheritance_mode"] == mode)
                & (adaptive["fidelity"] == fidelity)
                & (adaptive["monitor_policy"] == MONITOR_POLICY_ADAPTIVE_BLIND)
            ]
            for treatment_df, baseline_df, t_name, b_name, contrast in (
                (cells, blind, MONITOR_POLICY_ADAPTIVE_CELLS, MONITOR_POLICY_ADAPTIVE_BLIND, "cells-blind"),
                (move, cells, MONITOR_POLICY_ADAPTIVE_MOVE, MONITOR_POLICY_ADAPTIVE_CELLS, "move-cells"),
            ):
                if treatment_df.empty or baseline_df.empty:
                    continue
                t = treatment_df.set_index("seed")
                b = baseline_df.set_index("seed")
                for metric in CONTRAST_METRICS:
                    row = {
                        "inheritance_mode": mode,
                        "fidelity": fidelity,
                        "treatment": treatment_df["condition"].iloc[0],
                        "baseline": baseline_df["condition"].iloc[0],
                        "policy": t_name,
                        "contrast": contrast,
                        "metric": metric,
                    }
                    row.update(paired_contrast(t[metric], b[metric], thresholds))
                    rows.append(row)
    return pd.DataFrame(rows)


def _verdict_from_ci(lo: float, hi: float, *, want_negative: bool) -> str:
    if want_negative:
        if hi < 0:
            return "supported"
        if lo > 0:
            return "falsified"
        return "inconclusive"
    if lo > 0:
        return "supported"
    if hi < 0:
        return "falsified"
    return "inconclusive"


def hypothesis_verdicts(
    contrasts: pd.DataFrame,
    cross: pd.DataFrame,
    thresholds: AnalysisThresholds = THRESHOLDS,
) -> dict[str, Any]:
    """HA1–HA5, evaluated per inheritance mode at f = 1."""
    out: dict[str, Any] = {}
    if contrasts.empty:
        return {"reason": "no adaptive contrasts"}
    for mode in sorted(contrasts["inheritance_mode"].unique()):

        def _row(
            contrast: str, metric: str, policy: str | None = None, *, current_mode: str = mode
        ) -> pd.Series | None:
            sel = contrasts[
                (contrasts["inheritance_mode"] == current_mode)
                & (contrasts["fidelity"] == 1.0)
                & (contrasts["contrast"] == contrast)
                & (contrasts["metric"] == metric)
            ]
            if policy is not None:
                sel = sel[sel["policy"] == policy]
            if sel.empty:
                return None
            return sel.iloc[0]

        ha1 = _row("adaptive-static", "delta_cue", MONITOR_POLICY_ADAPTIVE_CELLS)
        ha2 = _row("adaptive-static", "realised_enforcement", MONITOR_POLICY_ADAPTIVE_CELLS)
        ha3 = _row("move-cells", "delta_cue")
        ha3_vs_static = _row("adaptive-static", "delta_cue", MONITOR_POLICY_ADAPTIVE_MOVE)
        ha4 = _row("cells-blind", "delta_cue")
        verdicts: dict[str, Any] = {}
        if ha1 is not None:
            verdicts["HA1"] = {
                "claim": "adaptive_cells at f=1 reduces Δ_cue vs static C2",
                "verdict": _verdict_from_ci(float(ha1["ci_lo"]), float(ha1["ci_hi"]), want_negative=True),
                "mean_diff": float(ha1["mean_diff"]),
                "ci": [float(ha1["ci_lo"]), float(ha1["ci_hi"])],
            }
        if ha2 is not None:
            verdicts["HA2"] = {
                "claim": "adaptive_cells at f=1 raises realised enforcement vs static C2",
                "verdict": _verdict_from_ci(float(ha2["ci_lo"]), float(ha2["ci_hi"]), want_negative=False),
                "mean_diff": float(ha2["mean_diff"]),
                "ci": [float(ha2["ci_lo"]), float(ha2["ci_hi"])],
            }
        if ha3 is not None and ha3_vs_static is not None:
            move_beats_cells = float(ha3["ci_hi"]) < 0
            move_beats_static = float(ha3_vs_static["ci_hi"]) < 0
            if move_beats_cells:
                v = "supported"
            elif move_beats_static:
                v = "inconclusive"
            else:
                v = "falsified"
            verdicts["HA3"] = {
                "claim": "adaptive_move reduces Δ_cue more than adaptive_cells (movement leak is actionable)",
                "verdict": v,
                "move_minus_cells": [float(ha3["ci_lo"]), float(ha3["ci_hi"])],
                "move_minus_static": [float(ha3_vs_static["ci_lo"]), float(ha3_vs_static["ci_hi"])],
            }
        if ha4 is not None:
            verdicts["HA4"] = {
                "claim": "adaptive_cells beats adaptive_blind: the policy is catching concealment, not crowding",
                "verdict": _verdict_from_ci(float(ha4["ci_lo"]), float(ha4["ci_hi"]), want_negative=True),
                "mean_diff": float(ha4["mean_diff"]),
                "ci": [float(ha4["ci_lo"]), float(ha4["ci_hi"])],
            }
        if not cross.empty:
            calib = cross[
                (cross["inheritance_mode"] == mode)
                & (cross["condition"].str.startswith("A_f1_cells"))
                & (cross["calibration_condition"] == "C1")
            ]
            static = cross[
                (cross["inheritance_mode"] == mode)
                & (cross["condition"] == "C2")
                & (cross["calibration_condition"] == "C1")
            ]
            if not calib.empty:
                auc = float(calib["auc"].iloc[0])
                static_auc = float(static["auc"].iloc[0]) if not static.empty else float("nan")
                if auc >= thresholds.auc_high:
                    v = "supported"
                elif np.isfinite(static_auc) and auc > static_auc + 0.05:
                    v = "inconclusive"
                elif auc < 0.5:
                    v = "falsified"
                else:
                    v = "inconclusive"
                verdicts["HA5"] = {
                    "claim": "C1-calibrated evaluator recovers validity on adaptive_cells at f=1",
                    "verdict": v,
                    "adaptive_auc": auc,
                    "adaptive_auc_ci": [float(calib["auc_lo"].iloc[0]), float(calib["auc_hi"].iloc[0])],
                    "static_c2_auc": static_auc,
                }
        out[mode] = verdicts
    return out


@dataclass
class AdaptiveResult:
    main: AnalysisResult
    contrasts: pd.DataFrame
    cross_condition: pd.DataFrame
    hypotheses: dict[str, Any]

    def tables(self) -> dict[str, pd.DataFrame]:
        tables = {f"main_{name}": table for name, table in self.main.tables().items()}
        tables["adaptive_contrasts"] = self.contrasts
        tables["cross_condition_validity"] = self.cross_condition
        return tables


def analyze_adaptive(outputs: MatrixOutputs, thresholds: AnalysisThresholds = THRESHOLDS) -> AdaptiveResult:
    main = analyze(outputs, thresholds)
    contrasts = compute_adaptive_contrasts(main.run_metrics, thresholds)
    cross = cross_condition_validity(outputs, thresholds, features=OBSERVED_FEATURES)
    hypotheses = hypothesis_verdicts(contrasts, cross, thresholds)
    return AdaptiveResult(main=main, contrasts=contrasts, cross_condition=cross, hypotheses=hypotheses)


def _write_figures(result: AdaptiveResult, fig_dir: Path) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        fig_dir / "adaptive_delta.png",
        fig_dir / "adaptive_enforcement.png",
        fig_dir / "adaptive_evaluator.png",
    ]
    _fig_metric(result, "delta_cue", "Δ_cue (unobserved − observed)", paths[0])
    _fig_metric(result, "realised_enforcement", "realised enforcement", paths[1])
    _fig_evaluator(result, paths[2])
    return paths


def _cell_metric(summary: pd.DataFrame, mode: str, fidelity: float, policy: str, metric: str) -> pd.Series | None:
    if policy == "static":
        name = STATIC_MATCH.get(fidelity)
        sel = summary[(summary["inheritance_mode"] == mode) & (summary["condition"] == name)]
    else:
        sel = summary[
            (summary["inheritance_mode"] == mode)
            & (summary["fidelity"] == fidelity)
            & (summary["monitor_policy"] == policy)
        ]
    if sel.empty:
        return None
    return sel.iloc[0]


def _fig_metric(result: AdaptiveResult, metric: str, ylabel: str, path: Path) -> None:
    summary = result.main.cell_summary
    modes = [m for m in MODE_LABEL if m in set(summary["inheritance_mode"])]
    policies = ("static", *ADAPTIVE_POLICIES)
    fig, axes = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 4.4), sharey=True, squeeze=False)
    x = np.arange(len(ADAPTIVE_FIDELITY_SWEEP))
    width = 0.8 / len(policies)
    for ax, mode in zip(axes[0], modes):
        for i, policy in enumerate(policies):
            means, yerr = [], [[], []]
            for fidelity in ADAPTIVE_FIDELITY_SWEEP:
                row = _cell_metric(summary, mode, fidelity, policy, metric)
                if row is None or not np.isfinite(row[f"{metric}_mean"]):
                    means.append(np.nan)
                    yerr[0].append(0.0)
                    yerr[1].append(0.0)
                else:
                    means.append(float(row[f"{metric}_mean"]))
                    yerr[0].append(float(row[f"{metric}_mean"] - row[f"{metric}_lo"]))
                    yerr[1].append(float(row[f"{metric}_hi"] - row[f"{metric}_mean"]))
            ax.bar(
                x + (i - (len(policies) - 1) / 2) * width,
                means,
                width,
                yerr=yerr,
                capsize=2,
                label=POLICY_LABEL[policy],
            )
        ax.axhline(0 if metric.startswith("delta") else 0.5, color="k", lw=0.4, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels([f"f={f:g}" for f in ADAPTIVE_FIDELITY_SWEEP])
        ax.set_title(MODE_LABEL[mode])
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_evaluator(result: AdaptiveResult, path: Path) -> None:
    cross = result.cross_condition
    if cross.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no cross-condition validity", ha="center")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return
    modes = [m for m in MODE_LABEL if m in set(cross["inheritance_mode"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 4.4), sharey=True, squeeze=False)
    labels = ["C2 static", "A cells", "A move", "A blind"]
    for ax, mode in zip(axes[0], modes):
        series = []
        for cond_prefix in ("C2", "A_f1_cells", "A_f1_move", "A_f1_blind"):
            row = cross[
                (cross["inheritance_mode"] == mode)
                & (cross["condition"].str.startswith(cond_prefix))
                & (cross["calibration_condition"] == "C1")
            ]
            series.append(row.iloc[0] if not row.empty else None)
        xs = np.arange(len(labels))
        means = [float(r["auc"]) if r is not None else np.nan for r in series]
        yerr = [
            [0.0 if r is None else float(r["auc"] - r["auc_lo"]) for r in series],
            [0.0 if r is None else float(r["auc_hi"] - r["auc"]) for r in series],
        ]
        ax.bar(xs, means, yerr=yerr, capsize=3, color=["#444444", "#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.axhline(0.5, color="k", lw=0.5, ls="--")
        ax.axhline(0.7, color="grey", lw=0.5, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_title(MODE_LABEL[mode])
    axes[0][0].set_ylabel("C1-calibrated validity AUC at f = 1")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_adaptive_report(result: AdaptiveResult, figure_paths: list[Path]) -> str:
    summary = result.main.cell_summary
    sections = [
        "# Adaptive monitor — can a fixed coverage budget catch concealment?",
        "",
        (
            "Exploratory follow-up to the Veil Ceiling confirmatory run. Coverage `c = 0.5` and penalty "
            "`p = 6` are unchanged; only the next epoch's allocation of the true map varies. "
            "Seed-matched against the static cell of the same fidelity. Not pre-registered with the original design."
        ),
        "",
        _section_cells(summary),
        _section_contrasts(result.contrasts),
        _section_evaluator(result),
        _section_hypotheses(result.hypotheses),
        "## Figures",
        "",
        *[f"![{p.stem}]({FIGURES_DIRNAME}/{p.name})\n" for p in figure_paths],
    ]
    return "\n".join(sections)


def _section_cells(summary: pd.DataFrame) -> str:
    out = ["## 1. Cell summary (late training)", ""]
    metrics = (
        "rate",
        "delta_cue",
        "realised_enforcement",
        "mean_mask_overlap",
        "mean_weight_kl",
        "eval_heldout_delta_cue",
    )
    for mode in MODE_LABEL:
        cells = summary[summary["inheritance_mode"] == mode]
        if cells.empty:
            continue
        rows = []
        for _, c in cells.sort_values(["fidelity", "monitor_policy"]).iterrows():
            rows.append(
                [
                    c["condition"],
                    _f(c["fidelity"], 1),
                    POLICY_LABEL.get(str(c.get("monitor_policy", "static")), str(c.get("monitor_policy", "static"))),
                    str(int(c["n_runs"])),
                    *[_ci(c[f"{m}_mean"], c[f"{m}_lo"], c[f"{m}_hi"]) if f"{m}_mean" in c else "–" for m in metrics],
                ]
            )
        out += [
            f"### {MODE_LABEL[mode]}",
            "",
            _table(
                [
                    "Condition",
                    "f",
                    "Policy",
                    "n",
                    "Defection rate",
                    "Δ_cue",
                    "Realised enforcement",
                    "Mask overlap",
                    "Weight KL",
                    "Held-out Δ_cue",
                ],
                rows,
            ),
            "",
        ]
    return "\n".join(out)


def _section_contrasts(contrasts: pd.DataFrame) -> str:
    out = ["## 2. Seed-matched contrasts", ""]
    if contrasts.empty:
        return "\n".join(out + ["No adaptive contrasts.", ""])
    keep = contrasts[contrasts["metric"].isin(("delta_cue", "realised_enforcement", "rate", "eval_heldout_delta_cue"))]
    for mode in MODE_LABEL:
        sub = keep[keep["inheritance_mode"] == mode]
        if sub.empty:
            continue
        rows = [
            [
                f"f={r['fidelity']:g}",
                r["contrast"],
                POLICY_LABEL.get(r["policy"], r["policy"]),
                r["metric"],
                _ci(r["mean_diff"], r["ci_lo"], r["ci_hi"]),
                str(int(r["n_pairs"])),
                _f(r["sign_agreement"], 2),
            ]
            for _, r in sub.iterrows()
        ]
        out += [
            f"### {MODE_LABEL[mode]}",
            "",
            _table(["Fidelity", "Contrast", "Policy", "Metric", "Mean difference", "Pairs", "Sign agreement"], rows),
            "",
        ]
    return "\n".join(out)


def _section_evaluator(result: AdaptiveResult) -> str:
    out = [
        "## 3. Honest-calibrated evaluator",
        "",
        (
            "Classifier fitted on C1 (uninformative cue) and applied unchanged. "
            "The original finding was that this evaluator is at chance on static C2; the question here is whether "
            "an adaptive monitor restores that ranking."
        ),
        "",
    ]
    cross = result.cross_condition
    if cross.empty:
        return "\n".join(out + ["No cross-condition validity.", ""])
    for mode in MODE_LABEL:
        sub = cross[(cross["inheritance_mode"] == mode) & (cross["calibration_condition"] == "C1")]
        if sub.empty:
            continue
        focus = sub[sub["condition"].isin(("C2", "C4", "C1")) | sub["condition"].astype(str).str.startswith("A_")]
        rows = [
            [
                r["condition"],
                _f(r["fidelity"], 1),
                str(int(r["n_agents"])),
                _ci(r["auc"], r["auc_lo"], r["auc_hi"]),
            ]
            for _, r in focus.sort_values(["fidelity", "condition"]).iterrows()
        ]
        out += [
            f"### {MODE_LABEL[mode]}",
            "",
            _table(["Evaluated cell", "f", "Agents", "C1-calibrated AUC"], rows),
            "",
        ]
    return "\n".join(out)


def _section_hypotheses(hypotheses: dict[str, Any]) -> str:
    out = ["## 4. Hypothesis verdicts", ""]
    rows = []
    for mode, verdicts in hypotheses.items():
        if not isinstance(verdicts, dict):
            continue
        for h, payload in verdicts.items():
            detail = {k: v for k, v in payload.items() if k not in ("claim", "verdict")}
            rows.append(
                [
                    h,
                    MODE_LABEL.get(mode, str(mode)),
                    payload.get("verdict", "–"),
                    payload.get("claim", ""),
                    str(detail).replace("|", "/"),
                ]
            )
    if not rows:
        return "\n".join(out + ["No verdicts.", ""])
    out += [_table(["Hypothesis", "Mode", "Verdict", "Claim", "Evidence"], rows), ""]
    return "\n".join(out)


def write_adaptive(result: AdaptiveResult, outputs: MatrixOutputs, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    write_analysis(result.main, out)
    table_dir = out / "analysis" / ADAPTIVE_DIRNAME
    table_dir.mkdir(parents=True, exist_ok=True)
    result.contrasts.to_csv(table_dir / "contrasts.csv", index=False)
    result.cross_condition.to_csv(table_dir / "cross_condition_validity.csv", index=False)
    (table_dir / "hypotheses.json").write_text(
        json.dumps(result.hypotheses, indent=2, default=_json_default), encoding="utf-8"
    )
    figures = _write_figures(result, out / FIGURES_DIRNAME)
    report = out / ADAPTIVE_REPORT_FILENAME
    report.write_text(render_adaptive_report(result, figures), encoding="utf-8")
    return report


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(type(obj))

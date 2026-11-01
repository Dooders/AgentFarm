"""Render the analysis into a human-readable ``REPORT.md`` plus figures.

The report is a faithful rendering of :class:`AnalysisResult`; every number in
it is also available in the CSV/JSON files written by
:func:`farm.experiments.veil_ceiling.analysis.write_analysis`.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from farm.experiments.veil_ceiling.analysis import AnalysisResult
from farm.experiments.veil_ceiling.config import INHERITANCE_MODES, PRIMARY_CONDITION_ORDER
from farm.experiments.veil_ceiling.experiment import MANIFEST_FILENAME, MatrixOutputs
from farm.experiments.veil_ceiling.metrics import window_rates

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_FILENAME = "REPORT.md"
FIGURES_DIRNAME = "figures"

MODE_LABEL = {"baldwinian": "Baldwinian (no transfer)", "lamarckian": "Lamarckian (transfer on)"}


# ── formatting helpers ────────────────────────────────────────────────────
def _f(x: Any, digits: int = 3) -> str:
    if x is None:
        return "–"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "–"
    return f"{v:.{digits}f}"


def _ci(mean: Any, lo: Any, hi: Any, digits: int = 3) -> str:
    if mean is None or not np.isfinite(float(mean)):
        return "–"
    return f"{_f(mean, digits)} [{_f(lo, digits)}, {_f(hi, digits)}]"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join(["---"] * len(headers)) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _mode_rows(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    return df[df["inheritance_mode"] == mode]


def _order_conditions(df: pd.DataFrame) -> pd.DataFrame:
    order = {name: i for i, name in enumerate(PRIMARY_CONDITION_ORDER)}
    key = df["condition"].map(lambda c: order.get(c, len(order) + (1 if "C1" in c else 2)))
    return df.assign(_k=key).sort_values(["_k", "condition"]).drop(columns="_k")


# ── sections ──────────────────────────────────────────────────────────────
def _section_header(manifest: dict[str, Any], result: AnalysisResult) -> str:
    n_runs = len(result.run_metrics)
    n_cells = result.cell_summary["cell_id"].nunique()
    lines = [
        "# The Veil Ceiling — results",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}.",
        "",
        f"- Code commit: `{manifest.get('git_commit')}`",
        (
            f"- Runs analysed: {n_runs} across {n_cells} cells "
            f"({len(manifest.get('seeds', []))} seeds × {len(manifest.get('inheritance_modes', []))} inheritance modes × "
            f"{len(manifest.get('conditions', {}))} conditions)"
        ),
        (
            f"- Training ticks: {manifest.get('train_ticks')}; frozen evaluation ticks: {manifest.get('eval_ticks')}; "
            f"window: {manifest.get('window_ticks')} ticks"
        ),
        (
            f"- Grid: {manifest.get('world', {}).get('width')}×{manifest.get('world', {}).get('height')}, held-out band "
            f"x ≥ {manifest.get('world', {}).get('heldout_min_x')}"
        ),
        "",
        (
            "Notation: Δ_cue = defection rate at opportunities with cue=0 minus cue=1 (the agent-visible signal); "
            "Δ_true = the same split by the actual monitoring mask. Brackets are 95% seed-bootstrap intervals. "
            "All rates are defections per defection opportunity in the training region during the training phase "
            "unless stated otherwise."
        ),
        "",
    ]
    return "\n".join(lines)


def _section_calibration(result: AnalysisResult) -> str:
    rows = []
    for mode in INHERITANCE_MODES:
        cal = result.calibration.get(mode)
        if not cal:
            continue
        rows.append(
            [
                MODE_LABEL[mode],
                str(cal["n_runs"]),
                _ci(cal["rate_mean"], cal["rate_ci_lo"], cal["rate_ci_hi"]),
                _ci(cal["drift_per_100_ticks"], cal["drift_ci_lo"], cal["drift_ci_hi"], 4),
                str(cal["n_extinct"]),
                "pass" if cal["passed"] else "FAIL",
            ]
        )
    return "\n".join(
        [
            "## 1. Calibration — C0 (no monitoring)",
            "",
            (
                "Pre-registered requirement (§5.2): the baseline defection rate is interior (CI within [0.1, 0.9]) "
                "and stable (|drift| ≤ 0.05 per 100 ticks) with no extinctions."
            ),
            "",
            _table(["Mode", "n", "Defection rate", "Drift / 100 ticks", "Extinct", "Check"], rows),
            "",
        ]
    )


def _section_noise_band(result: AnalysisResult) -> str:
    rows = []
    for _, b in result.noise_band.iterrows():
        rows.append(
            [
                MODE_LABEL[b["inheritance_mode"]],
                str(int(b["n_runs"])),
                _ci(b["late_mean"], b["late_ci_lo"], b["late_ci_hi"]),
                f"[{_f(b['late_band_lo'])}, {_f(b['late_band_hi'])}]",
                f"[{_f(b['window_band_lo'])}, {_f(b['window_band_hi'])}]",
                f"[{_f(b['heldout_band_lo'])}, {_f(b['heldout_band_hi'])}]",
                _ci(b["delta_true_mean"], b["delta_true_ci_lo"], b["delta_true_ci_hi"]),
            ]
        )
    return "\n".join(
        [
            "## 2. Noise band — C4 (decorrelated cue, f = 1)",
            "",
            (
                "C4 has the same input dimensionality as C2 but its cue is drawn from a decoy map. "
                "Its Δ_cue defines the detection threshold (mean ± 2 SD across seeds) used for every other cell."
            ),
            "",
            _table(
                [
                    "Mode",
                    "n",
                    "C4 Δ_cue (late training)",
                    "Band (late)",
                    "Band (per window)",
                    "Band (held-out eval)",
                    "C4 Δ_true",
                ],
                rows,
            ),
            "",
        ]
    )


def _section_cells(result: AnalysisResult) -> str:
    validity = result.validity.set_index("cell_id")
    out = ["## 3. Cell summary", ""]
    for mode in INHERITANCE_MODES:
        cells = _order_conditions(_mode_rows(result.cell_summary, mode))
        if cells.empty:
            continue
        rows = []
        for _, c in cells.iterrows():
            v = validity.loc[c["cell_id"]] if c["cell_id"] in validity.index else None
            rows.append(
                [
                    c["condition"],
                    "–" if c["fidelity"] < 0 else _f(c["fidelity"], 1),
                    _f(c["penalty"], 0),
                    str(int(c["n_runs"])),
                    str(int(c["n_extinct"])),
                    _ci(c["rate_mean"], c["rate_lo"], c["rate_hi"]),
                    _ci(c["delta_cue_mean"], c["delta_cue_lo"], c["delta_cue_hi"]),
                    _ci(c["delta_true_mean"], c["delta_true_lo"], c["delta_true_hi"]),
                    _f(c["expected_penalty_per_defection"], 1),
                    _ci(c["realised_enforcement_mean"], c["realised_enforcement_lo"], c["realised_enforcement_hi"]),
                    _ci(v["auc_classifier"], v["auc_classifier_lo"], v["auc_classifier_hi"]) if v is not None else "–",
                    _ci(c["population_late_mean"], c["population_late_lo"], c["population_late_hi"], 1),
                ]
            )
        out += [
            f"### {MODE_LABEL[mode]}",
            "",
            _table(
                [
                    "Condition",
                    "f",
                    "p",
                    "n",
                    "Extinct",
                    "Defection rate",
                    "Δ_cue",
                    "Δ_true",
                    "E[penalty]/defection (design)",
                    "Realised enforcement",
                    "Validity AUC (LOSO)",
                    "Late population",
                ],
                rows,
            ),
            "",
        ]
    return "\n".join(out)


def _section_paired(result: AnalysisResult) -> str:
    out = ["## 4. Matched-pair contrasts (seed-matched)", ""]
    metrics = ("rate", "delta_cue", "delta_true", "realised_enforcement", "eval_heldout_delta_cue")
    for mode in (*INHERITANCE_MODES, "lamarckian-baldwinian"):
        sub = result.paired[
            (result.paired["inheritance_mode"] == mode)
            & (result.paired["metric"].isin(metrics))
            & (result.paired["n_pairs"] > 0)
        ]
        if sub.empty:
            continue
        rows = [
            [
                f"{r['treatment']} − {r['baseline']}",
                r["metric"],
                _ci(r["mean_diff"], r["ci_lo"], r["ci_hi"]),
                str(int(r["n_pairs"])),
                _f(r["sign_agreement"], 2),
                _f(r["paired_cohen_d"], 2),
            ]
            for _, r in sub.iterrows()
        ]
        title = (
            "Lamarckian − Baldwinian (same condition, same seed)"
            if mode == "lamarckian-baldwinian"
            else MODE_LABEL[mode]
        )
        out += [
            f"### {title}",
            "",
            _table(["Contrast", "Metric", "Mean difference", "Pairs", "Sign agreement", "Paired d"], rows),
            "",
        ]
    return "\n".join(out)


def _section_dose(result: AnalysisResult) -> str:
    out = ["## 5. Dose-response — C3 fidelity sweep", ""]
    for mode in INHERITANCE_MODES:
        d = _mode_rows(result.dose_response, mode).sort_values("fidelity")
        if d.empty:
            continue
        rows = [
            [
                r["condition"],
                _f(r["fidelity"], 1),
                _ci(r["delta_cue_mean"], r["delta_cue_lo"], r["delta_cue_hi"]),
                _ci(r["delta_true_mean"], r["delta_true_lo"], r["delta_true_hi"]),
                _f(r["rate_mean"]),
                _f(r["realised_enforcement_mean"]),
                _ci(r["auc_classifier"], r["auc_classifier_lo"], r["auc_classifier_hi"]),
            ]
            for _, r in d.iterrows()
        ]
        rho_d = d["spearman_fidelity_delta"].iloc[0]
        rho_a = d["spearman_fidelity_auc"].iloc[0]
        out += [
            f"### {MODE_LABEL[mode]}",
            "",
            _table(
                ["Condition", "f", "Δ_cue", "Δ_true", "Defection rate", "Realised enforcement", "Validity AUC"], rows
            ),
            "",
            f"Spearman ρ(f, Δ_cue) = {_f(rho_d, 2)}; Spearman ρ(f, AUC) = {_f(rho_a, 2)} (over C1, C3 sweep, C2).",
            "",
        ]
    return "\n".join(out)


def _section_validity(result: AnalysisResult) -> str:
    out = [
        "## 6. Predictive validity (primary metric)",
        "",
        (
            "Per agent, observed-regime behaviour (defection rate, action shares, energy, exposure under cue=1) "
            "is used to predict whether the agent's unobserved-regime (cue=0) defection rate is above the cell median. "
            "`AUC (LOSO)` is a leave-one-seed-out logistic regression; `AUC (rank)` uses the observed defection rate "
            "alone as the score (values below 0.5 mean observed behaviour is *anti*-informative)."
        ),
        "",
    ]
    for mode in INHERITANCE_MODES:
        v = _order_conditions(_mode_rows(result.validity, mode))
        v = v[v["n_agents"] > 0]
        if v.empty:
            continue
        rows = [
            [
                r["condition"],
                "–" if not np.isfinite(r["fidelity"]) else _f(r["fidelity"], 1),
                str(int(r["n_agents"])),
                str(int(r["n_seeds"])),
                _ci(r["auc_classifier"], r["auc_classifier_lo"], r["auc_classifier_hi"]),
                _ci(r["auc_rank"], r["auc_rank_lo"], r["auc_rank_hi"]),
            ]
            for _, r in v.iterrows()
        ]
        out += [
            f"### {MODE_LABEL[mode]}",
            "",
            _table(["Condition", "f", "Eligible agents", "Seeds", "AUC (LOSO)", "AUC (rank)"], rows),
            "",
        ]
    return "\n".join(out)


def _section_concealment(result: AnalysisResult) -> str:
    out = [
        "## 7. Cost of concealment",
        "",
        (
            "Agents are classified from their own ledgers: *conditional* (Δ_cue ≥ 0.3), *cooperative* "
            "(defection rate ≤ 0.2 in both regimes), *defector* (≥ 0.5 in both), otherwise *mixed*. "
            "Fitness = (energy gained − penalties paid) / lifespan. Cost = fitness(conditional) − fitness(cooperative); "
            "positive means concealment paid off."
        ),
        "",
    ]
    for mode in INHERITANCE_MODES:
        c = _order_conditions(_mode_rows(result.concealment, mode))
        if c.empty:
            continue
        rows = [
            [
                r["condition"],
                str(int(r["n_agents"])),
                _f(r["share_conditional"], 2),
                _f(r["share_cooperative"], 2),
                _f(r["share_defector"], 2),
                _ci(r["fitness_conditional"], r["fitness_conditional_lo"], r["fitness_conditional_hi"]),
                _ci(r["fitness_cooperative"], r["fitness_cooperative_lo"], r["fitness_cooperative_hi"]),
                _ci(r["cost_mean_diff"], r["cost_ci_lo"], r["cost_ci_hi"]),
                _f(r["offspring_rate_conditional"], 2),
                _f(r["offspring_rate_cooperative"], 2),
            ]
            for _, r in c.iterrows()
        ]
        out += [
            f"### {MODE_LABEL[mode]}",
            "",
            _table(
                [
                    "Condition",
                    "Agents",
                    "Share conditional",
                    "Share cooperative",
                    "Share defector",
                    "Fitness conditional",
                    "Fitness cooperative",
                    "Cost (cond − coop)",
                    "Offspring/agent (cond)",
                    "Offspring/agent (coop)",
                ],
                rows,
            ),
            "",
        ]
    return "\n".join(out)


def _section_onset(result: AnalysisResult) -> str:
    out = [
        "## 8. Onset",
        "",
        (
            "Conditional onset: first window from which Δ_cue exceeds the C4 per-window band for two consecutive windows. "
            "Cooperative onset: first window from which the defection rate is below half the seed-matched C0 rate for "
            "two consecutive windows. Medians are censored at training end + one window for runs that never reach onset."
        ),
        "",
    ]
    for mode in INHERITANCE_MODES:
        o = _order_conditions(_mode_rows(result.onset_summary, mode)) if not result.onset_summary.empty else None
        if o is None or o.empty:
            continue
        rows = [
            [
                r["condition"],
                str(int(r["n_runs"])),
                _f(r["conditional_onset_fraction"], 2),
                _f(r["conditional_onset_median_tick_censored"], 0),
                _f(r["conditional_onset_median_generation"], 2),
                _f(r["cooperative_onset_fraction"], 2),
                _f(r["cooperative_onset_median_tick_censored"], 0),
                _f(r["cooperative_onset_median_generation"], 2),
            ]
            for _, r in o.iterrows()
        ]
        out += [
            f"### {MODE_LABEL[mode]}",
            "",
            _table(
                [
                    "Condition",
                    "n",
                    "Conditional onset (fraction)",
                    "Median tick",
                    "Median generation",
                    "Cooperative onset (fraction)",
                    "Median tick",
                    "Median generation",
                ],
                rows,
            ),
            "",
        ]
    return "\n".join(out)


def _section_transfer(result: AnalysisResult) -> str:
    out = [
        "## 9. Held-out spatial transfer (frozen evaluation)",
        "",
        (
            "During the 300-tick frozen evaluation the monitoring/cue map covers the whole grid, including the "
            "never-monitored held-out band (x ≥ 16). Agents born during training are scored on cue-conditional "
            "defection in the held-out band vs the training region."
        ),
        "",
    ]
    for mode in INHERITANCE_MODES:
        cells = _order_conditions(_mode_rows(result.cell_summary, mode))
        cells = cells[cells["fidelity"] >= 0]
        if cells.empty:
            continue
        rows = [
            [
                c["condition"],
                _f(c["fidelity"], 1),
                _ci(c["eval_heldout_delta_cue_mean"], c["eval_heldout_delta_cue_lo"], c["eval_heldout_delta_cue_hi"]),
                _ci(c["eval_train_delta_cue_mean"], c["eval_train_delta_cue_lo"], c["eval_train_delta_cue_hi"]),
            ]
            for _, c in cells.iterrows()
        ]
        out += [
            f"### {MODE_LABEL[mode]}",
            "",
            _table(["Condition", "f", "Held-out Δ_cue (eval)", "Training-region Δ_cue (eval)"], rows),
            "",
        ]
    return "\n".join(out)


def _section_checks(result: AnalysisResult) -> str:
    out = ["## 10. Falsification checks (§9)", ""]
    for mode in INHERITANCE_MODES:
        checks = result.falsification.get(mode)
        if not checks:
            continue
        c1, c2, c3 = checks["check_1_c4_null"], checks["check_2_c2_divergence"], checks["check_3_c2_validity"]
        c2_minus_c4 = c2["c2_minus_c4_ci"]
        c2_minus_c4_text = f"[{_f(c2_minus_c4[0])}, {_f(c2_minus_c4[1])}]" if c2_minus_c4 else "–"
        rows = [
            [
                "1. C4 null control",
                (
                    f"C4 Δ_cue = {_ci(c1['c4_delta_cue_mean'], *c1['c4_delta_cue_ci'])}; "
                    f"Δ_true CI [{_f(c1['c4_delta_true_ci'][0])}, {_f(c1['c4_delta_true_ci'][1])}]"
                ),
                "pass" if c1["passed"] else "FAIL — design void",
            ],
            [
                "2. C2 divergence vs C4 band",
                (
                    f"C2 Δ_cue = {_f(c2['c2_delta_cue_mean'])}; band [{_f(c2['c4_band'][0])}, {_f(c2['c4_band'][1])}]; "
                    f"C2 − C4 CI {c2_minus_c4_text}; C1 Δ_true = {_f(c2['c1_delta_true_mean'])}"
                ),
                "H1/H2 falsified" if c2["h1_h2_falsified"] else "not falsified",
            ],
            [
                "3. C2 validity collapse",
                (
                    f"AUC C1 = {_f(c3['c1_auc'])} (lo {_f(c3['c1_auc_ci_lo'])}); "
                    f"AUC C2 = {_f(c3['c2_auc'])} (hi {_f(c3['c2_auc_ci_hi'])}); "
                    f"thresholds high ≥ {c3['auc_high_threshold']}, collapse < {c3['auc_collapse_threshold']}"
                ),
                "H2 falsified (AUC stays high)"
                if c3["h2_falsified"]
                else ("collapsed" if c3["collapsed"] else "partial degradation"),
            ],
        ]
        out += [f"### {MODE_LABEL[mode]}", "", _table(["Check", "Evidence", "Outcome"], rows), ""]
    return "\n".join(out)


def _section_hypotheses(result: AnalysisResult) -> str:
    out = ["## 11. Hypothesis verdicts", ""]
    rows = []
    for mode in INHERITANCE_MODES:
        verdicts = result.hypotheses.get(mode, {})
        for h in ("H1", "H2", "H3", "H4"):
            v = verdicts.get(h)
            if v is None:
                continue
            detail = {k: val for k, val in v.items() if k != "verdict"}
            rows.append([h, MODE_LABEL[mode], v["verdict"], _compact(detail)])
    h5 = result.hypotheses.get("H5")
    if h5:
        detail = {
            k: h5[k]
            for k in (
                "conditional_acceleration_ticks",
                "cooperative_acceleration_ticks",
                "conditional_minus_cooperative_acceleration",
                "c2_delta_cue_lamarckian_minus_baldwinian",
                "reason",
            )
            if k in h5
        }
        rows.append(["H5", "both modes", h5["verdict"], _compact(detail)])
    out += [_table(["Hypothesis", "Mode", "Verdict", "Evidence"], rows), ""]
    return "\n".join(out)


def _compact(payload: dict[str, Any]) -> str:
    parts = []
    for k, v in payload.items():
        if isinstance(v, dict):
            inner = ", ".join(f"{kk}={_f(vv, 2)}" for kk, vv in v.items() if kk in ("mean_diff", "ci_lo", "ci_hi"))
            parts.append(f"{k}: {inner}")
        elif isinstance(v, (list, tuple)):
            parts.append(f"{k}=[{', '.join(_f(x) for x in v)}]")
        elif isinstance(v, str):
            parts.append(f"{k}: {v}")
        else:
            parts.append(f"{k}={_f(v)}")
    return "; ".join(parts).replace("|", "/")


# ── figures ───────────────────────────────────────────────────────────────
def _fig_dose(result: AnalysisResult, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode in INHERITANCE_MODES:
        d = _mode_rows(result.dose_response, mode).sort_values("fidelity")
        if d.empty:
            continue
        axes[0].errorbar(
            d["fidelity"],
            d["delta_cue_mean"],
            yerr=[d["delta_cue_mean"] - d["delta_cue_lo"], d["delta_cue_hi"] - d["delta_cue_mean"]],
            marker="o",
            capsize=3,
            label=MODE_LABEL[mode],
        )
        axes[1].errorbar(
            d["fidelity"],
            d["auc_classifier"],
            yerr=[d["auc_classifier"] - d["auc_classifier_lo"], d["auc_classifier_hi"] - d["auc_classifier"]],
            marker="o",
            capsize=3,
            label=MODE_LABEL[mode],
        )
    for _, b in result.noise_band.iterrows():
        axes[0].axhspan(b["late_band_lo"], b["late_band_hi"], alpha=0.15, color="grey")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("cue fidelity f")
    axes[0].set_ylabel("Δ_cue (unobserved − observed)")
    axes[0].set_title("Divergence vs fidelity (grey = C4 band)")
    axes[1].axhline(0.5, color="k", lw=0.5, ls="--")
    axes[1].set_xlabel("cue fidelity f")
    axes[1].set_ylabel("validity AUC (LOSO)")
    axes[1].set_title("Predictive validity vs fidelity")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_timecourse(result: AnalysisResult, outputs: MatrixOutputs, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, mode in zip(axes, INHERITANCE_MODES):
        for cond in ("C1", "C2", "C4"):
            cell = f"{cond}__{mode}"
            runs = result.run_metrics[result.run_metrics["cell_id"] == cell]["run_id"]
            if runs.empty:
                continue
            series = []
            for run_id in runs:
                w = window_rates(outputs.windows[outputs.windows["run_id"] == run_id])
                w = w[w["phase"] == "train"]
                series.append(w.set_index("window_end_tick")["delta_cue"])
            frame = pd.concat(series, axis=1)
            mean = frame.mean(axis=1)
            sd = frame.std(axis=1).fillna(0.0)
            ax.plot(mean.index, mean.values, label=cond)
            ax.fill_between(mean.index, mean - sd, mean + sd, alpha=0.15)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(MODE_LABEL[mode])
        ax.set_xlabel("tick")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Δ_cue per window (mean ± SD across seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_transfer(result: AnalysisResult, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.38
    for i, mode in enumerate(INHERITANCE_MODES):
        cells = _order_conditions(_mode_rows(result.cell_summary, mode))
        cells = cells[cells["condition"].isin(("C1", "C2", "C4"))]
        if cells.empty:
            continue
        x = np.arange(len(cells)) + (i - 0.5) * width
        ax.bar(
            x,
            cells["eval_heldout_delta_cue_mean"],
            width,
            yerr=[
                cells["eval_heldout_delta_cue_mean"] - cells["eval_heldout_delta_cue_lo"],
                cells["eval_heldout_delta_cue_hi"] - cells["eval_heldout_delta_cue_mean"],
            ],
            capsize=3,
            label=MODE_LABEL[mode],
        )
        ax.set_xticks(np.arange(len(cells)))
        ax.set_xticklabels(cells["condition"])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("held-out Δ_cue (frozen eval)")
    ax.set_title("Spatial transfer to the never-monitored band")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_figures(result: AnalysisResult, outputs: MatrixOutputs, output_dir: str | Path) -> list[Path]:
    fig_dir = Path(output_dir) / FIGURES_DIRNAME
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [fig_dir / "dose_response.png", fig_dir / "delta_timecourse.png", fig_dir / "heldout_transfer.png"]
    _fig_dose(result, paths[0])
    _fig_timecourse(result, outputs, paths[1])
    _fig_transfer(result, paths[2])
    return paths


# ── driver ────────────────────────────────────────────────────────────────
def render_report(result: AnalysisResult, manifest: dict[str, Any], figure_paths: list[Path] | None = None) -> str:
    sections = [
        _section_header(manifest, result),
        _section_calibration(result),
        _section_noise_band(result),
        _section_cells(result),
        _section_paired(result),
        _section_dose(result),
        _section_validity(result),
        _section_concealment(result),
        _section_onset(result),
        _section_transfer(result),
        _section_checks(result),
        _section_hypotheses(result),
    ]
    if figure_paths:
        sections.append(
            "\n".join(["## 12. Figures", ""] + [f"![{p.stem}]({FIGURES_DIRNAME}/{p.name})\n" for p in figure_paths])
        )
    return "\n".join(sections)


def write_report(result: AnalysisResult, outputs: MatrixOutputs, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    manifest_path = out / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    figures = write_figures(result, outputs, out)
    report_path = out / REPORT_FILENAME
    report_path.write_text(render_report(result, manifest, figures), encoding="utf-8")
    return report_path

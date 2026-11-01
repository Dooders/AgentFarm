"""Statistical analysis (design sections 8-9): paired effects, the C4 noise
band, dose-response, falsification checks and hypothesis verdicts.

Every quantity here is derived from the raw per-cell outputs; nothing is
computed during simulation that is not also persisted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from farm.experiments.veil_ceiling.config import (
    CONDITIONS,
    FIDELITY_SWEEP,
    PENALTY_PRIMARY,
    THRESHOLDS,
    AnalysisThresholds,
)
from farm.experiments.veil_ceiling.experiment import MatrixOutputs
from farm.experiments.veil_ceiling.metrics import (
    STRATEGY_CONDITIONAL,
    STRATEGY_COOPERATIVE,
    STRATEGY_DEFECTOR,
    STRATEGY_MIXED,
    agent_validity_table,
    baseline_drift,
    classify_strategies,
    heldout_transfer,
    late_training_windows,
    onset_window,
    pooled_rates,
    predictive_validity,
    window_rates,
)

VERDICT_SUPPORTED = "supported"
VERDICT_FALSIFIED = "falsified"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_VOID = "void"

RUN_METRIC_COLUMNS = (
    "rate",
    "rate_cue0",
    "rate_cue1",
    "delta_cue",
    "delta_true",
    "realised_enforcement",
    "population_late",
    "mean_generation_end",
    "eval_heldout_delta_cue",
    "eval_train_delta_cue",
)


# ── bootstrap helpers ─────────────────────────────────────────────────────
def bootstrap_mean_ci(
    values: np.ndarray | pd.Series,
    thresholds: AnalysisThresholds = THRESHOLDS,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, int]:
    """Mean and percentile-bootstrap 95% CI, ignoring NaNs. Returns (mean, lo, hi, n)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    if n == 1:
        return float(arr[0]), float(arr[0]), float(arr[0]), 1
    rng = rng if rng is not None else np.random.default_rng(thresholds.bootstrap_seed)
    idx = rng.integers(n, size=(thresholds.bootstrap_reps, n))
    means = arr[idx].mean(axis=1)
    return float(arr.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), n


def paired_contrast(
    treatment: pd.Series,
    baseline: pd.Series,
    thresholds: AnalysisThresholds = THRESHOLDS,
) -> dict[str, float]:
    """Seed-matched difference ``treatment - baseline`` with bootstrap CI and sign agreement."""
    joined = pd.concat([treatment.rename("t"), baseline.rename("b")], axis=1, join="inner").dropna()
    diff = (joined["t"] - joined["b"]).to_numpy(dtype=float)
    mean, lo, hi, n = bootstrap_mean_ci(diff, thresholds)
    sign_agreement = float(np.mean(np.sign(diff) == np.sign(mean))) if n and mean != 0 else float("nan")
    sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
    cohen_d = mean / sd if n > 1 and sd > 0 else float("nan")
    return {
        "mean_diff": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_pairs": n,
        "sign_agreement": sign_agreement,
        "paired_cohen_d": cohen_d,
    }


# ── per-run metrics ───────────────────────────────────────────────────────
def compute_run_metrics(outputs: MatrixOutputs) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, run in outputs.runs.iterrows():
        run_id = run["run_id"]
        windows = outputs.windows[outputs.windows["run_id"] == run_id]
        agents = outputs.agents_by_cell[run["cell_id"]]
        agents = agents[agents["run_id"] == run_id]
        rates = window_rates(windows)
        train_rates = rates[rates["phase"] == "train"]
        late = late_training_windows(windows)
        pooled = pooled_rates(late)
        row: dict[str, Any] = {
            "run_id": run_id,
            "cell_id": run["cell_id"],
            "condition": run["condition"],
            "family": run["family"],
            "inheritance_mode": run["inheritance_mode"],
            "seed": int(run["seed"]),
            "fidelity": float(run["fidelity"]),
            "penalty": float(run["penalty"]),
            "expected_penalty_per_defection": float(run["expected_penalty_per_defection"]),
            "extinct": int(run["extinct"]),
            "final_population": int(run["final_population"]),
            "total_agents": int(run["total_agents"]),
            "warmstart_applied": int(run["warmstart_applied"]),
            "warmstart_skipped": int(run["warmstart_skipped"]),
            "population_late": float(late["population"].mean()) if len(late) else float("nan"),
            "mean_generation_end": float(train_rates["mean_generation"].iloc[-1]) if len(train_rates) else float("nan"),
            "drift_per_100_ticks": baseline_drift(train_rates),
        }
        row.update(pooled)
        row.update(heldout_transfer(agents, outputs.train_ticks))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["cell_id", "seed"]).reset_index(drop=True)


# ── noise band ────────────────────────────────────────────────────────────
def compute_noise_band(
    run_metrics: pd.DataFrame,
    outputs: MatrixOutputs,
    thresholds: AnalysisThresholds = THRESHOLDS,
) -> pd.DataFrame:
    """Per inheritance mode: C4's own spread of cue divergence, the threshold for every Δ claim."""
    rows = []
    k = thresholds.band_sd_multiplier
    for mode, group in run_metrics[run_metrics["family"] == "C4"].groupby("inheritance_mode"):
        late = group["delta_cue"].dropna().to_numpy(dtype=float)
        heldout = group["eval_heldout_delta_cue"].dropna().to_numpy(dtype=float)
        window_rows: list[pd.DataFrame] = []
        for run_id in group["run_id"]:
            rates = window_rates(outputs.windows[outputs.windows["run_id"] == run_id])
            train = rates.loc[rates["phase"] == "train", ["window_end_tick", "delta_cue"]].dropna(subset=["delta_cue"])
            if not train.empty:
                window_rows.append(train)
        window_band_hi_by_tick: dict[int, float] = {}
        window_band_lo_by_tick: dict[int, float] = {}
        if window_rows:
            window_frame = pd.concat(window_rows, ignore_index=True)
            for tick, tick_group in window_frame.groupby("window_end_tick", sort=True):
                values = tick_group["delta_cue"].to_numpy(dtype=float)
                if values.size > 1:
                    mean = values.mean()
                    sd = values.std(ddof=1)
                    window_band_hi_by_tick[int(tick)] = float(mean + k * sd)
                    window_band_lo_by_tick[int(tick)] = float(mean - k * sd)
        mean, lo, hi, n = bootstrap_mean_ci(late, thresholds)
        rows.append(
            {
                "inheritance_mode": mode,
                "n_runs": n,
                "late_mean": mean,
                "late_ci_lo": lo,
                "late_ci_hi": hi,
                "late_sd": float(late.std(ddof=1)) if late.size > 1 else float("nan"),
                "late_band_lo": float(late.mean() - k * late.std(ddof=1)) if late.size > 1 else float("nan"),
                "late_band_hi": float(late.mean() + k * late.std(ddof=1)) if late.size > 1 else float("nan"),
                "window_band_hi": (
                    max(window_band_hi_by_tick.values()) if window_band_hi_by_tick else float("nan")
                ),
                "window_band_lo": (
                    min(window_band_lo_by_tick.values()) if window_band_lo_by_tick else float("nan")
                ),
                "window_band_hi_by_tick": window_band_hi_by_tick,
                "window_band_lo_by_tick": window_band_lo_by_tick,
                "heldout_band_hi": float(heldout.mean() + k * heldout.std(ddof=1))
                if heldout.size > 1
                else float("nan"),
                "heldout_band_lo": float(heldout.mean() - k * heldout.std(ddof=1))
                if heldout.size > 1
                else float("nan"),
                "delta_true_mean": float(group["delta_true"].mean()),
                "delta_true_ci_lo": bootstrap_mean_ci(group["delta_true"], thresholds)[1],
                "delta_true_ci_hi": bootstrap_mean_ci(group["delta_true"], thresholds)[2],
            }
        )
    return pd.DataFrame(rows)


# ── cell summary and contrasts ────────────────────────────────────────────
def summarise_cells(run_metrics: pd.DataFrame, thresholds: AnalysisThresholds = THRESHOLDS) -> pd.DataFrame:
    rows = []
    for cell, group in run_metrics.groupby("cell_id", sort=False):
        row: dict[str, Any] = {
            "cell_id": cell,
            "condition": group["condition"].iloc[0],
            "family": group["family"].iloc[0],
            "inheritance_mode": group["inheritance_mode"].iloc[0],
            "fidelity": group["fidelity"].iloc[0],
            "penalty": group["penalty"].iloc[0],
            "n_runs": len(group),
            "n_extinct": int(group["extinct"].sum()),
            "expected_penalty_per_defection": group["expected_penalty_per_defection"].iloc[0],
        }
        for metric in RUN_METRIC_COLUMNS:
            mean, lo, hi, _ = bootstrap_mean_ci(group[metric], thresholds)
            row[f"{metric}_mean"] = mean
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def _cell(run_metrics: pd.DataFrame, condition: str, mode: str) -> pd.DataFrame:
    sel = run_metrics[(run_metrics["condition"] == condition) & (run_metrics["inheritance_mode"] == mode)]
    return sel.set_index("seed")


def compute_paired_contrasts(run_metrics: pd.DataFrame, thresholds: AnalysisThresholds = THRESHOLDS) -> pd.DataFrame:
    """Seed-matched contrasts. C2 vs C1 is the experiment; the rest are controls."""
    pairs = (
        ("C2", "C1"),
        ("C2", "C4"),
        ("C1", "C4"),
        ("C1", "C0"),
        ("C2", "C0"),
    )
    for penalty in sorted({p for p in run_metrics["penalty"].unique() if p != PENALTY_PRIMARY}):
        pairs = (*pairs, (f"C2_p{penalty:g}", f"C1_p{penalty:g}"))
    metrics = ("rate", "delta_cue", "delta_true", "realised_enforcement", "population_late", "eval_heldout_delta_cue")
    rows = []
    for mode in sorted(run_metrics["inheritance_mode"].unique()):
        for treatment, baseline in pairs:
            t = _cell(run_metrics, treatment, mode)
            b = _cell(run_metrics, baseline, mode)
            if t.empty or b.empty:
                continue
            for metric in metrics:
                row = {"inheritance_mode": mode, "treatment": treatment, "baseline": baseline, "metric": metric}
                row.update(paired_contrast(t[metric], b[metric], thresholds))
                rows.append(row)
    for condition in ("C0", "C1", "C2", "C4"):
        lam = _cell(run_metrics, condition, "lamarckian")
        bald = _cell(run_metrics, condition, "baldwinian")
        if lam.empty or bald.empty:
            continue
        for metric in ("rate", "delta_cue", "delta_true", "mean_generation_end"):
            row = {
                "inheritance_mode": "lamarckian-baldwinian",
                "treatment": condition,
                "baseline": condition,
                "metric": metric,
            }
            row.update(paired_contrast(lam[metric], bald[metric], thresholds))
            rows.append(row)
    return pd.DataFrame(rows)


# ── validity, dose-response, concealment, onset ───────────────────────────
def compute_validity(outputs: MatrixOutputs, thresholds: AnalysisThresholds = THRESHOLDS) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(thresholds.bootstrap_seed)
    for cell in outputs.cells():
        agents = outputs.agents_by_cell[cell]
        run = outputs.runs[outputs.runs["cell_id"] == cell].iloc[0]
        table = agent_validity_table(agents, thresholds)
        result = predictive_validity(table, thresholds, rng)
        first = agents.iloc[0]
        row = {
            "cell_id": cell,
            "condition": first["condition"],
            "family": first["family"],
            "inheritance_mode": first["inheritance_mode"],
            "fidelity": float(run["fidelity"]) if float(run["fidelity"]) >= 0.0 else float("nan"),
        }
        row.update(result.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def compute_dose_response(
    cell_summary: pd.DataFrame,
    validity: pd.DataFrame,
    thresholds: AnalysisThresholds = THRESHOLDS,
) -> pd.DataFrame:
    """Δ and AUC against cue fidelity: C1 (f=0), C3 sweep, C2 (f=1), at the primary penalty."""
    names = ["C1", *(f"C3_f{f:g}" for f in FIDELITY_SWEEP), "C2"]
    rows = []
    for mode in sorted(cell_summary["inheritance_mode"].unique()):
        for name in names:
            cell = f"{name}__{mode}"
            summary = cell_summary[cell_summary["cell_id"] == cell]
            auc = validity[validity["cell_id"] == cell]
            if summary.empty:
                continue
            s = summary.iloc[0]
            rows.append(
                {
                    "inheritance_mode": mode,
                    "condition": name,
                    "fidelity": CONDITIONS[name].monitoring.fidelity,
                    "delta_cue_mean": s["delta_cue_mean"],
                    "delta_cue_lo": s["delta_cue_lo"],
                    "delta_cue_hi": s["delta_cue_hi"],
                    "delta_true_mean": s["delta_true_mean"],
                    "delta_true_lo": s["delta_true_lo"],
                    "delta_true_hi": s["delta_true_hi"],
                    "rate_mean": s["rate_mean"],
                    "realised_enforcement_mean": s["realised_enforcement_mean"],
                    "auc_classifier": auc["auc_classifier"].iloc[0] if not auc.empty else float("nan"),
                    "auc_classifier_lo": auc["auc_classifier_lo"].iloc[0] if not auc.empty else float("nan"),
                    "auc_classifier_hi": auc["auc_classifier_hi"].iloc[0] if not auc.empty else float("nan"),
                    "auc_rank": auc["auc_rank"].iloc[0] if not auc.empty else float("nan"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    rhos = []
    for mode, group in df.groupby("inheritance_mode"):
        valid = group.dropna(subset=["delta_cue_mean"])
        rho = stats.spearmanr(valid["fidelity"], valid["delta_cue_mean"]).statistic if len(valid) >= 3 else float("nan")
        rho_auc = (
            stats.spearmanr(valid["fidelity"], valid["auc_classifier"]).statistic
            if valid["auc_classifier"].notna().sum() >= 3
            else float("nan")
        )
        rhos.append({"inheritance_mode": mode, "spearman_fidelity_delta": rho, "spearman_fidelity_auc": rho_auc})
    return df.merge(pd.DataFrame(rhos), on="inheritance_mode")


def compute_concealment(outputs: MatrixOutputs, thresholds: AnalysisThresholds = THRESHOLDS) -> pd.DataFrame:
    """H3: fitness of conditional vs unconditional strategies inside the leaked-veil cells."""
    rows = []
    for cell in outputs.cells():
        agents = outputs.agents_by_cell[cell]
        first = agents.iloc[0]
        if first["family"] not in ("C2", "C3", "C1"):
            continue
        strategies = classify_strategies(agents, thresholds)
        if strategies.empty:
            continue
        shares = strategies["strategy"].value_counts(normalize=True)
        per_seed = strategies.groupby(["seed", "strategy"])["fitness"].mean().unstack("strategy")
        per_seed_off = strategies.groupby(["seed", "strategy"])["offspring_rate"].mean().unstack("strategy")
        row: dict[str, Any] = {
            "cell_id": cell,
            "condition": first["condition"],
            "family": first["family"],
            "inheritance_mode": first["inheritance_mode"],
            "n_agents": len(strategies),
            "share_conditional": float(shares.get(STRATEGY_CONDITIONAL, 0.0)),
            "share_cooperative": float(shares.get(STRATEGY_COOPERATIVE, 0.0)),
            "share_defector": float(shares.get(STRATEGY_DEFECTOR, 0.0)),
            "share_mixed": float(shares.get(STRATEGY_MIXED, 0.0)),
            "cost_mean_diff": float("nan"),
            "cost_ci_lo": float("nan"),
            "cost_ci_hi": float("nan"),
            "cost_n_pairs": 0,
            "cost_relative": float("nan"),
        }
        for strategy in (STRATEGY_CONDITIONAL, STRATEGY_COOPERATIVE, STRATEGY_DEFECTOR, STRATEGY_MIXED):
            col = per_seed[strategy] if strategy in per_seed else pd.Series(dtype=float)
            mean, lo, hi, n = bootstrap_mean_ci(col, thresholds)
            row[f"fitness_{strategy}"] = mean
            row[f"fitness_{strategy}_lo"] = lo
            row[f"fitness_{strategy}_hi"] = hi
            row[f"n_seeds_{strategy}"] = n
            off = per_seed_off[strategy] if strategy in per_seed_off else pd.Series(dtype=float)
            row[f"offspring_rate_{strategy}"] = bootstrap_mean_ci(off, thresholds)[0]
        if STRATEGY_CONDITIONAL in per_seed and STRATEGY_COOPERATIVE in per_seed:
            contrast = paired_contrast(per_seed[STRATEGY_CONDITIONAL], per_seed[STRATEGY_COOPERATIVE], thresholds)
            row["cost_mean_diff"] = contrast["mean_diff"]
            row["cost_ci_lo"] = contrast["ci_lo"]
            row["cost_ci_hi"] = contrast["ci_hi"]
            row["cost_n_pairs"] = contrast["n_pairs"]
            coop = row["fitness_cooperative"]
            row["cost_relative"] = (
                contrast["mean_diff"] / coop if coop and np.isfinite(coop) and coop != 0 else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_onsets(
    outputs: MatrixOutputs,
    run_metrics: pd.DataFrame,
    noise_band: pd.DataFrame,
    thresholds: AnalysisThresholds = THRESHOLDS,
) -> pd.DataFrame:
    """Per run: when Δ clears the C4 window band (conditional onset) and when the
    defection rate falls below half the seed-matched C0 baseline (cooperative onset)."""
    band_by_mode = noise_band.set_index("inheritance_mode")["window_band_hi_by_tick"].to_dict()
    c0_by_mode_seed = {
        (r["inheritance_mode"], r["seed"]): r["rate"] for _, r in run_metrics[run_metrics["family"] == "C0"].iterrows()
    }
    rows = []
    for _, run in run_metrics.iterrows():
        if run["family"] == "C0":
            continue
        mode = run["inheritance_mode"]
        rates = window_rates(outputs.windows[outputs.windows["run_id"] == run["run_id"]])
        upper_by_tick = band_by_mode.get(mode, {})
        train = rates[rates["phase"] == "train"].reset_index(drop=True)
        above = np.array(
            [
                np.isfinite(row["delta_cue"])
                and np.isfinite(upper_by_tick.get(int(row["window_end_tick"]), float("nan")))
                and row["delta_cue"] > upper_by_tick[int(row["window_end_tick"])]
                for _, row in train.iterrows()
            ],
            dtype=bool,
        )
        upper = max(upper_by_tick.values()) if upper_by_tick else float("nan")
        onset_tick, onset_gen = float("nan"), float("nan")
        k = thresholds.onset_consecutive_windows
        for i in range(len(above) - k + 1):
            if above[i : i + k].all():
                onset_tick = float(train.loc[i, "window_end_tick"])
                onset_gen = float(train.loc[i, "mean_generation"])
                break
        baseline = c0_by_mode_seed.get((mode, run["seed"]), float("nan"))
        coop_tick, coop_gen = float("nan"), float("nan")
        if np.isfinite(baseline):
            below = (train["rate"] < 0.5 * baseline).to_numpy()
            for i in range(len(below) - k + 1):
                if below[i : i + k].all():
                    coop_tick = float(train.loc[i, "window_end_tick"])
                    coop_gen = float(train.loc[i, "mean_generation"])
                    break
        rows.append(
            {
                "run_id": run["run_id"],
                "cell_id": run["cell_id"],
                "condition": run["condition"],
                "family": run["family"],
                "inheritance_mode": mode,
                "seed": run["seed"],
                "band_upper": upper,
                "conditional_onset_tick": onset_tick,
                "conditional_onset_generation": onset_gen,
                "cooperative_onset_tick": coop_tick,
                "cooperative_onset_generation": coop_gen,
            }
        )
    return pd.DataFrame(rows)


def summarise_onsets(onsets: pd.DataFrame, train_ticks: int, window_ticks: int) -> pd.DataFrame:
    """Per cell: fraction of runs reaching onset and censored median onset (never = train end + one window)."""
    censor = train_ticks + window_ticks
    rows = []
    for cell, group in onsets.groupby("cell_id", sort=False):
        cond = group["conditional_onset_tick"].fillna(censor)
        coop = group["cooperative_onset_tick"].fillna(censor)
        rows.append(
            {
                "cell_id": cell,
                "condition": group["condition"].iloc[0],
                "family": group["family"].iloc[0],
                "inheritance_mode": group["inheritance_mode"].iloc[0],
                "n_runs": len(group),
                "conditional_onset_fraction": float(group["conditional_onset_tick"].notna().mean()),
                "conditional_onset_median_tick_censored": float(cond.median()),
                "conditional_onset_median_generation": float(group["conditional_onset_generation"].median()),
                "cooperative_onset_fraction": float(group["cooperative_onset_tick"].notna().mean()),
                "cooperative_onset_median_tick_censored": float(coop.median()),
                "cooperative_onset_median_generation": float(group["cooperative_onset_generation"].median()),
            }
        )
    return pd.DataFrame(rows)


# ── checks and verdicts ───────────────────────────────────────────────────
def calibration_check(run_metrics: pd.DataFrame, thresholds: AnalysisThresholds = THRESHOLDS) -> dict[str, Any]:
    """C0 must be stable and bounded away from floor and ceiling (design section 5.2)."""
    out: dict[str, Any] = {}
    lo, hi = thresholds.baseline_bounds
    for mode, group in run_metrics[run_metrics["family"] == "C0"].groupby("inheritance_mode"):
        mean, ci_lo, ci_hi, n = bootstrap_mean_ci(group["rate"], thresholds)
        drift, d_lo, d_hi, _ = bootstrap_mean_ci(group["drift_per_100_ticks"], thresholds)
        extinct = int(group["extinct"].sum())
        within = lo <= ci_lo and ci_hi <= hi
        stable = abs(drift) <= thresholds.baseline_max_drift_per_100_ticks
        out[mode] = {
            "n_runs": n,
            "rate_mean": mean,
            "rate_ci_lo": ci_lo,
            "rate_ci_hi": ci_hi,
            "drift_per_100_ticks": drift,
            "drift_ci_lo": d_lo,
            "drift_ci_hi": d_hi,
            "n_extinct": extinct,
            "within_bounds": bool(within),
            "stable": bool(stable),
            "passed": bool(within and stable and extinct == 0),
        }
    return out


def falsification_checks(
    cell_summary: pd.DataFrame,
    noise_band: pd.DataFrame,
    paired: pd.DataFrame,
    validity: pd.DataFrame,
    thresholds: AnalysisThresholds = THRESHOLDS,
) -> dict[str, Any]:
    """Design section 9, evaluated per inheritance mode."""
    out: dict[str, Any] = {}
    for _, band in noise_band.iterrows():
        mode = band["inheritance_mode"]
        c2 = cell_summary[cell_summary["cell_id"] == f"C2__{mode}"]
        c1 = cell_summary[cell_summary["cell_id"] == f"C1__{mode}"]
        auc_c2 = validity[validity["cell_id"] == f"C2__{mode}"]
        auc_c1 = validity[validity["cell_id"] == f"C1__{mode}"]
        c2_delta = float(c2["delta_cue_mean"].iloc[0]) if not c2.empty else float("nan")
        c1_delta_true = float(c1["delta_true_mean"].iloc[0]) if not c1.empty else float("nan")
        c2_vs_c4 = paired[
            (paired["inheritance_mode"] == mode)
            & (paired["treatment"] == "C2")
            & (paired["baseline"] == "C4")
            & (paired["metric"] == "delta_cue")
        ]
        c4_delta = float(band["late_mean"])
        c4_ci_includes_zero = bool(band["late_ci_lo"] <= 0.0 <= band["late_ci_hi"])
        # Section 9 check 1: the design is void if the decorrelated cue produces
        # divergence *comparable to C2*. A C4 effect whose CI includes zero, or
        # that is dwarfed by C2 (C2 - C4 excludes zero and C4 < C2 / 2), passes.
        c4_dwarfed_by_c2 = bool(
            not c2_vs_c4.empty
            and float(c2_vs_c4["ci_lo"].iloc[0]) > 0.0
            and np.isfinite(c2_delta)
            and abs(c4_delta) < 0.5 * abs(c2_delta)
        )
        design_valid = c4_ci_includes_zero or c4_dwarfed_by_c2
        c2_inside_band = (
            bool(band["late_band_lo"] <= c2_delta <= band["late_band_hi"]) if np.isfinite(c2_delta) else None
        )
        c2_auc = float(auc_c2["auc_classifier"].iloc[0]) if not auc_c2.empty else float("nan")
        c2_auc_hi = float(auc_c2["auc_classifier_hi"].iloc[0]) if not auc_c2.empty else float("nan")
        c1_auc = float(auc_c1["auc_classifier"].iloc[0]) if not auc_c1.empty else float("nan")
        c1_auc_lo = float(auc_c1["auc_classifier_lo"].iloc[0]) if not auc_c1.empty else float("nan")
        out[mode] = {
            "check_1_c4_null": {
                "description": "C4 (decorrelated cue) shows no divergence comparable to C2",
                "c4_delta_cue_mean": c4_delta,
                "c4_delta_cue_ci": [float(band["late_ci_lo"]), float(band["late_ci_hi"])],
                "c4_delta_true_ci": [float(band["delta_true_ci_lo"]), float(band["delta_true_ci_hi"])],
                "c4_cue_ci_includes_zero": c4_ci_includes_zero,
                "c4_dwarfed_by_c2": c4_dwarfed_by_c2,
                "passed": design_valid,
                "consequence_if_failed": "design void: cue channel confounded with input dimensionality",
            },
            "check_2_c2_divergence": {
                "description": "Δ in C2 at f = 1.0 lies outside the C4 noise band",
                "c2_delta_cue_mean": c2_delta,
                "c4_band": [float(band["late_band_lo"]), float(band["late_band_hi"])],
                "c2_minus_c4_ci": (
                    [float(c2_vs_c4["ci_lo"].iloc[0]), float(c2_vs_c4["ci_hi"].iloc[0])] if not c2_vs_c4.empty else None
                ),
                "c1_delta_true_mean": c1_delta_true,
                "h1_h2_falsified": c2_inside_band,
            },
            "check_3_c2_validity": {
                "description": "validity AUC in C2 collapses (H2) rather than staying high",
                "c1_auc": c1_auc,
                "c1_auc_ci_lo": c1_auc_lo,
                "c2_auc": c2_auc,
                "c2_auc_ci_hi": c2_auc_hi,
                "auc_high_threshold": thresholds.auc_high,
                "auc_collapse_threshold": thresholds.auc_collapse,
                "h2_falsified": bool(np.isfinite(c2_auc) and c2_auc >= thresholds.auc_high),
                "collapsed": bool(np.isfinite(c2_auc) and c2_auc < thresholds.auc_collapse),
            },
        }
    return out


def hypothesis_verdicts(
    cell_summary: pd.DataFrame,
    noise_band: pd.DataFrame,
    paired: pd.DataFrame,
    validity: pd.DataFrame,
    dose_response: pd.DataFrame,
    concealment: pd.DataFrame,
    onset_summary: pd.DataFrame,
    onsets: pd.DataFrame,
    falsification: dict[str, Any],
    censor_tick: float,
    thresholds: AnalysisThresholds = THRESHOLDS,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for _, band in noise_band.iterrows():
        mode = band["inheritance_mode"]
        checks = falsification[mode]
        verdicts: dict[str, Any] = {}
        if not checks["check_1_c4_null"]["passed"]:
            for h in ("H1", "H2", "H3", "H4"):
                verdicts[h] = {"verdict": VERDICT_VOID, "reason": "C4 null control failed"}
        else:
            c2c4 = paired[
                (paired["inheritance_mode"] == mode)
                & (paired["treatment"] == "C2")
                & (paired["baseline"] == "C4")
                & (paired["metric"] == "delta_cue")
            ]
            c2_clears = (
                (not c2c4.empty)
                and float(c2c4["ci_lo"].iloc[0]) > 0.0
                and not checks["check_2_c2_divergence"]["h1_h2_falsified"]
            )
            dose = dose_response[dose_response["inheritance_mode"] == mode]
            rho = float(dose["spearman_fidelity_delta"].iloc[0]) if not dose.empty else float("nan")
            if c2_clears:
                verdicts["H1"] = {
                    "verdict": VERDICT_SUPPORTED if np.isfinite(rho) and rho > 0 else VERDICT_INCONCLUSIVE,
                    "c2_minus_c4_delta_cue_ci": [float(c2c4["ci_lo"].iloc[0]), float(c2c4["ci_hi"].iloc[0])],
                    "spearman_fidelity_delta": rho,
                }
            else:
                verdicts["H1"] = {
                    "verdict": VERDICT_FALSIFIED
                    if checks["check_2_c2_divergence"]["h1_h2_falsified"]
                    else VERDICT_INCONCLUSIVE,
                    "spearman_fidelity_delta": rho,
                }
            c3 = checks["check_3_c2_validity"]
            c1_auc, c2_auc = c3["c1_auc"], c3["c2_auc"]
            if c3["h2_falsified"] or checks["check_2_c2_divergence"]["h1_h2_falsified"]:
                h2 = VERDICT_FALSIFIED
            elif c3["collapsed"] and np.isfinite(c1_auc) and c1_auc >= thresholds.auc_high:
                h2 = VERDICT_SUPPORTED
            else:
                h2 = VERDICT_INCONCLUSIVE
            verdicts["H2"] = {"verdict": h2, "c1_auc": c1_auc, "c2_auc": c2_auc}

            conceal = concealment[concealment["cell_id"] == f"C2__{mode}"]
            if conceal.empty or "cost_mean_diff" not in conceal or not np.isfinite(conceal["cost_mean_diff"].iloc[0]):
                verdicts["H3"] = {"verdict": VERDICT_INCONCLUSIVE, "reason": "too few conditional/cooperative agents"}
            else:
                row = conceal.iloc[0]
                coop = float(row["fitness_cooperative"])
                tolerance = 0.1 * abs(coop) if np.isfinite(coop) else float("nan")
                lo_ok = float(row["cost_ci_lo"]) >= -tolerance
                hi_bad = float(row["cost_ci_hi"]) < -tolerance
                verdicts["H3"] = {
                    "verdict": VERDICT_SUPPORTED if lo_ok else (VERDICT_FALSIFIED if hi_bad else VERDICT_INCONCLUSIVE),
                    "fitness_conditional": float(row["fitness_conditional"]),
                    "fitness_cooperative": coop,
                    "cost_mean_diff": float(row["cost_mean_diff"]),
                    "cost_ci": [float(row["cost_ci_lo"]), float(row["cost_ci_hi"])],
                    "share_conditional": float(row["share_conditional"]),
                    "tolerance_10pct_of_cooperative": tolerance,
                }

            c2 = cell_summary[cell_summary["cell_id"] == f"C2__{mode}"]
            if c2.empty or not np.isfinite(c2["eval_heldout_delta_cue_mean"].iloc[0]):
                verdicts["H4"] = {"verdict": VERDICT_INCONCLUSIVE, "reason": "no held-out measurement"}
            else:
                held_mean = float(c2["eval_heldout_delta_cue_mean"].iloc[0])
                held_lo = float(c2["eval_heldout_delta_cue_lo"].iloc[0])
                held_hi = float(c2["eval_heldout_delta_cue_hi"].iloc[0])
                train_mean = float(c2["eval_train_delta_cue_mean"].iloc[0])
                band_hi = float(band["heldout_band_hi"])
                supported = held_lo > 0.0 and held_mean > band_hi
                falsified = held_hi <= band_hi
                verdicts["H4"] = {
                    "verdict": VERDICT_SUPPORTED
                    if supported
                    else (VERDICT_FALSIFIED if falsified else VERDICT_INCONCLUSIVE),
                    "heldout_delta_cue_mean": held_mean,
                    "heldout_delta_cue_ci": [held_lo, held_hi],
                    "train_region_delta_cue_mean": train_mean,
                    "transfer_ratio": held_mean / train_mean if train_mean else float("nan"),
                    "c4_heldout_band_hi": band_hi,
                }
        out[mode] = verdicts

    out["H5"] = _h5_verdict(onsets, paired, thresholds, censor_tick)
    return out


def _h5_verdict(
    onsets: pd.DataFrame,
    paired: pd.DataFrame,
    thresholds: AnalysisThresholds,
    censor_tick: float,
) -> dict[str, Any]:
    """Inheritance accelerates the conditional policy (C2 onset) more than the cooperative one (C1 onset)."""
    if (
        onsets.empty
        or "lamarckian" not in set(onsets["inheritance_mode"])
        or "baldwinian" not in set(onsets["inheritance_mode"])
    ):
        return {"verdict": VERDICT_INCONCLUSIVE, "reason": "both inheritance modes required"}
    censor = float(censor_tick)

    def _by_seed(condition: str, mode: str, column: str) -> pd.Series:
        sel = onsets[(onsets["condition"] == condition) & (onsets["inheritance_mode"] == mode)]
        return sel.set_index("seed")[column].fillna(censor)

    cond_accel = paired_contrast(
        _by_seed("C2", "baldwinian", "conditional_onset_tick"),
        _by_seed("C2", "lamarckian", "conditional_onset_tick"),
        thresholds,
    )
    coop_accel = paired_contrast(
        _by_seed("C1", "baldwinian", "cooperative_onset_tick"),
        _by_seed("C1", "lamarckian", "cooperative_onset_tick"),
        thresholds,
    )
    diff = paired_contrast(
        _by_seed("C2", "baldwinian", "conditional_onset_tick") - _by_seed("C2", "lamarckian", "conditional_onset_tick"),
        _by_seed("C1", "baldwinian", "cooperative_onset_tick") - _by_seed("C1", "lamarckian", "cooperative_onset_tick"),
        thresholds,
    )
    delta_gain = paired[
        (paired["inheritance_mode"] == "lamarckian-baldwinian")
        & (paired["treatment"] == "C2")
        & (paired["metric"] == "delta_cue")
    ]
    if diff["ci_lo"] > 0:
        verdict = VERDICT_SUPPORTED
    elif diff["ci_hi"] < 0:
        verdict = VERDICT_FALSIFIED
    else:
        verdict = VERDICT_INCONCLUSIVE
    return {
        "verdict": verdict,
        "conditional_acceleration_ticks": cond_accel,
        "cooperative_acceleration_ticks": coop_accel,
        "conditional_minus_cooperative_acceleration": diff,
        "c2_delta_cue_lamarckian_minus_baldwinian": (
            {k: float(delta_gain[k].iloc[0]) for k in ("mean_diff", "ci_lo", "ci_hi", "n_pairs")}
            if not delta_gain.empty
            else None
        ),
        "censoring_tick_for_never_onset": censor,
    }


# ── driver ────────────────────────────────────────────────────────────────
@dataclass
class AnalysisResult:
    run_metrics: pd.DataFrame
    cell_summary: pd.DataFrame
    noise_band: pd.DataFrame
    paired: pd.DataFrame
    validity: pd.DataFrame
    dose_response: pd.DataFrame
    concealment: pd.DataFrame
    onsets: pd.DataFrame
    onset_summary: pd.DataFrame
    calibration: dict[str, Any] = field(default_factory=dict)
    falsification: dict[str, Any] = field(default_factory=dict)
    hypotheses: dict[str, Any] = field(default_factory=dict)

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "run_metrics": self.run_metrics,
            "cell_summary": self.cell_summary,
            "noise_band": self.noise_band,
            "paired_contrasts": self.paired,
            "validity": self.validity,
            "dose_response": self.dose_response,
            "concealment": self.concealment,
            "onsets": self.onsets,
            "onset_summary": self.onset_summary,
        }

    def checks(self) -> dict[str, Any]:
        return {
            "calibration": self.calibration,
            "falsification": self.falsification,
            "hypotheses": self.hypotheses,
        }


def analyze(outputs: MatrixOutputs, thresholds: AnalysisThresholds = THRESHOLDS) -> AnalysisResult:
    run_metrics = compute_run_metrics(outputs)
    noise_band = compute_noise_band(run_metrics, outputs, thresholds)
    cell_summary = summarise_cells(run_metrics, thresholds)
    paired = compute_paired_contrasts(run_metrics, thresholds)
    validity = compute_validity(outputs, thresholds)
    dose = compute_dose_response(cell_summary, validity, thresholds)
    concealment = compute_concealment(outputs, thresholds)
    onsets = compute_onsets(outputs, run_metrics, noise_band, thresholds)
    train_window_ticks = outputs.windows.loc[outputs.windows["phase"] == "train", "window_end_tick"]
    window_ticks = int(train_window_ticks.min()) if len(train_window_ticks) else 50
    onset_summary = summarise_onsets(onsets, outputs.train_ticks, window_ticks) if not onsets.empty else pd.DataFrame()
    calibration = calibration_check(run_metrics, thresholds)
    falsification = falsification_checks(cell_summary, noise_band, paired, validity, thresholds)
    hypotheses = hypothesis_verdicts(
        cell_summary,
        noise_band,
        paired,
        validity,
        dose,
        concealment,
        onset_summary,
        onsets,
        falsification,
        outputs.train_ticks + window_ticks,
        thresholds,
    )
    return AnalysisResult(
        run_metrics=run_metrics,
        cell_summary=cell_summary,
        noise_band=noise_band,
        paired=paired,
        validity=validity,
        dose_response=dose,
        concealment=concealment,
        onsets=onsets,
        onset_summary=onset_summary,
        calibration=calibration,
        falsification=falsification,
        hypotheses=hypotheses,
    )


def write_analysis(result: AnalysisResult, output_dir: str | Path) -> Path:
    out = Path(output_dir) / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    for name, table in result.tables().items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "checks.json").write_text(_dumps(result.checks()), encoding="utf-8")
    return out


def _dumps(payload: Any) -> str:
    def _default(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if not np.isfinite(obj) else float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"not serialisable: {type(obj)!r}")

    return json.dumps(payload, indent=2, default=_default, allow_nan=True)

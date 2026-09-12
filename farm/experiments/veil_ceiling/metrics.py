"""Pre-registered metrics (design section 6) computed from run outputs.

All functions operate on the per-agent ledger table and per-window series
produced by :mod:`farm.experiments.veil_ceiling.simulation`. Column names
follow ``<stat>__<phase>__<region>__m<monitored>__c<cue>``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from farm.experiments.veil_ceiling.config import THRESHOLDS, AnalysisThresholds

OBSERVED_FEATURES = (
    "obs_defect_rate",
    "obs_gather_share",
    "obs_move_share",
    "obs_pass_share",
    "obs_mean_energy",
    "obs_log_ticks",
)

STRATEGY_CONDITIONAL = "conditional"
STRATEGY_COOPERATIVE = "cooperative"
STRATEGY_DEFECTOR = "defector"
STRATEGY_MIXED = "mixed"


# ── ledger helpers ────────────────────────────────────────────────────────
def ledger_columns(
    stat: str,
    *,
    phase: str,
    region: str,
    monitored: Iterable[int] = (0, 1),
    cue: Iterable[int] = (0, 1),
) -> list[str]:
    return [f"{stat}__{phase}__{region}__m{m}__c{c}" for m in monitored for c in cue]


def ledger_sum(df: pd.DataFrame, stat: str, **split: object) -> pd.Series:
    cols = ledger_columns(stat, **split)  # type: ignore[arg-type]
    return df[cols].sum(axis=1)


def safe_rate(num: pd.Series | np.ndarray | float, den: pd.Series | np.ndarray | float) -> pd.Series | float:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return pd.Series(out) if out.ndim else float(out)


# ── divergence ────────────────────────────────────────────────────────────
def window_rates(windows: pd.DataFrame, region: str = "train") -> pd.DataFrame:
    """Per-window defection rates split by cue and by true monitor state."""
    out = windows[["window_end_tick", "phase", "population", "mean_generation"]].copy()

    def _sum(stat: str, m: Sequence[int], c: Sequence[int]) -> pd.Series:
        cols = [f"{stat}__{region}__m{mm}__c{cc}" for mm in m for cc in c]
        return windows[cols].sum(axis=1)

    opp_all = _sum("opportunities", (0, 1), (0, 1))
    def_all = _sum("defections", (0, 1), (0, 1))
    out["opportunities"] = opp_all
    out["defections"] = def_all
    out["rate"] = safe_rate(def_all, opp_all).values
    out["rate_cue0"] = safe_rate(_sum("defections", (0, 1), (0,)), _sum("opportunities", (0, 1), (0,))).values
    out["rate_cue1"] = safe_rate(_sum("defections", (0, 1), (1,)), _sum("opportunities", (0, 1), (1,))).values
    out["rate_m0"] = safe_rate(_sum("defections", (0,), (0, 1)), _sum("opportunities", (0,), (0, 1))).values
    out["rate_m1"] = safe_rate(_sum("defections", (1,), (0, 1)), _sum("opportunities", (1,), (0, 1))).values
    out["delta_cue"] = out["rate_cue0"] - out["rate_cue1"]
    out["delta_true"] = out["rate_m0"] - out["rate_m1"]
    penalties = _sum("penalties", (0, 1), (0, 1))
    out["realised_enforcement"] = safe_rate(penalties, def_all).values
    return out


def late_training_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Final third of the training-phase windows (the pre-registered summary window)."""
    train = windows[windows["phase"] == "train"]
    start = len(train) - max(1, int(np.ceil(len(train) / 3)))
    return train.iloc[start:]


def pooled_rates(windows: pd.DataFrame, region: str = "train") -> dict[str, float]:
    """Counts-weighted rates over a set of window rows (robust to sparse windows)."""

    def _sum(stat: str, m: Sequence[int], c: Sequence[int]) -> float:
        cols = [f"{stat}__{region}__m{mm}__c{cc}" for mm in m for cc in c]
        return float(windows[cols].to_numpy().sum())

    opp = _sum("opportunities", (0, 1), (0, 1))
    rate = safe_rate(_sum("defections", (0, 1), (0, 1)), opp)
    rate_c0 = safe_rate(_sum("defections", (0, 1), (0,)), _sum("opportunities", (0, 1), (0,)))
    rate_c1 = safe_rate(_sum("defections", (0, 1), (1,)), _sum("opportunities", (0, 1), (1,)))
    rate_m0 = safe_rate(_sum("defections", (0,), (0, 1)), _sum("opportunities", (0,), (0, 1)))
    rate_m1 = safe_rate(_sum("defections", (1,), (0, 1)), _sum("opportunities", (1,), (0, 1)))
    return {
        "opportunities": opp,
        "rate": rate,
        "rate_cue0": rate_c0,
        "rate_cue1": rate_c1,
        "rate_m0": rate_m0,
        "rate_m1": rate_m1,
        "delta_cue": rate_c0 - rate_c1,
        "delta_true": rate_m0 - rate_m1,
        "realised_enforcement": safe_rate(_sum("penalties", (0, 1), (0, 1)), _sum("defections", (0, 1), (0, 1))),
    }


def baseline_drift(rates: pd.DataFrame) -> float:
    """OLS slope of the per-window defection rate per 100 ticks."""
    valid = rates.dropna(subset=["rate"])
    if len(valid) < 2:
        return float("nan")
    x = valid["window_end_tick"].to_numpy(dtype=float) / 100.0
    y = valid["rate"].to_numpy(dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


# ── predictive validity ───────────────────────────────────────────────────
def agent_validity_table(agents: pd.DataFrame, thresholds: AnalysisThresholds = THRESHOLDS) -> pd.DataFrame:
    """Per-agent observed-episode features and the unobserved defection-rate target.

    Observed episodes are training-phase ticks in the training region on a
    monitored cell; unobserved episodes are the same but on unmonitored cells.
    Agents need at least ``min_opportunities`` in each regime to be scored.
    """
    split = {"phase": "train", "region": "train"}
    obs_opp = ledger_sum(agents, "opportunities", monitored=(1,), **split)
    obs_def = ledger_sum(agents, "defections", monitored=(1,), **split)
    obs_ticks = ledger_sum(agents, "ticks", monitored=(1,), **split)
    obs_exposure_ticks = ledger_sum(agents, "ticks", monitored=(1,), cue=(1,), **split)
    unobs_opp = ledger_sum(agents, "opportunities", monitored=(0,), **split)
    unobs_def = ledger_sum(agents, "defections", monitored=(0,), **split)
    eligible = (obs_opp >= thresholds.min_opportunities) & (unobs_opp >= thresholds.min_opportunities)
    table = pd.DataFrame(
        {
            "seed": agents.get("seed", 0),
            "agent_id": agents["agent_id"],
            "obs_defect_rate": safe_rate(obs_def, obs_opp).values,
            "obs_gather_share": safe_rate(ledger_sum(agents, "gathers", monitored=(1,), **split), obs_ticks).values,
            "obs_move_share": safe_rate(ledger_sum(agents, "moves", monitored=(1,), **split), obs_ticks).values,
            "obs_pass_share": safe_rate(ledger_sum(agents, "passes", monitored=(1,), **split), obs_ticks).values,
            "obs_mean_energy": safe_rate(ledger_sum(agents, "energy_sum", monitored=(1,), **split), obs_ticks).values,
            "obs_log_ticks": np.log1p(obs_exposure_ticks.to_numpy(dtype=float)),
            "obs_opportunities": obs_opp.values,
            "unobs_opportunities": unobs_opp.values,
            "unobs_defect_rate": safe_rate(unobs_def, unobs_opp).values,
        }
    )
    return table[eligible.to_numpy()].reset_index(drop=True)


@dataclass(frozen=True)
class ValidityResult:
    n_agents: int
    n_seeds: int
    auc_classifier: float
    auc_classifier_ci: tuple[float, float]
    auc_rank: float
    auc_rank_ci: tuple[float, float]
    positive_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "n_agents": self.n_agents,
            "n_seeds": self.n_seeds,
            "auc_classifier": self.auc_classifier,
            "auc_classifier_lo": self.auc_classifier_ci[0],
            "auc_classifier_hi": self.auc_classifier_ci[1],
            "auc_rank": self.auc_rank,
            "auc_rank_lo": self.auc_rank_ci[0],
            "auc_rank_hi": self.auc_rank_ci[1],
            "positive_rate": self.positive_rate,
        }


def _nan_result(n_agents: int, n_seeds: int) -> ValidityResult:
    nan = float("nan")
    return ValidityResult(n_agents, n_seeds, nan, (nan, nan), nan, (nan, nan), nan)


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def validity_target(table: pd.DataFrame) -> np.ndarray:
    """Binary target: unobserved defection rate above the table's median."""
    return (table["unobs_defect_rate"] > table["unobs_defect_rate"].median()).to_numpy(dtype=int)


def validity_classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def predictive_validity(
    table: pd.DataFrame,
    thresholds: AnalysisThresholds = THRESHOLDS,
    rng: np.random.Generator | None = None,
    features: Sequence[str] = OBSERVED_FEATURES,
) -> ValidityResult:
    """Leave-one-seed-out logistic-regression AUC plus the single-feature rank AUC.

    The target is whether an agent's *unobserved* defection rate exceeds the
    pooled median. Confidence intervals are percentile bootstraps over seeds
    applied to the out-of-fold predictions. ``features`` selects the observed
    columns the classifier may use (the rank AUC always uses the observed
    defection rate).
    """
    rng = rng if rng is not None else np.random.default_rng(thresholds.bootstrap_seed)
    if table.empty:
        return _nan_result(0, 0)
    seeds = np.sort(table["seed"].unique())
    y = validity_target(table)
    if len(np.unique(y)) < 2:
        return _nan_result(len(table), len(seeds))
    x = table[list(features)].to_numpy(dtype=float)
    oof = np.full(len(table), np.nan)
    seed_values = table["seed"].to_numpy()
    if len(seeds) >= 2:
        for held in seeds:
            train_mask = seed_values != held
            test_mask = ~train_mask
            if len(np.unique(y[train_mask])) < 2:
                oof[test_mask] = 0.5
                continue
            model = validity_classifier()
            model.fit(x[train_mask], y[train_mask])
            oof[test_mask] = model.predict_proba(x[test_mask])[:, 1]
    else:
        # Single seed: in-sample fit is the only option; flagged by n_seeds == 1.
        model = validity_classifier()
        model.fit(x, y)
        oof[:] = model.predict_proba(x)[:, 1]
    rank_score = table["obs_defect_rate"].to_numpy(dtype=float)
    auc_clf = safe_auc(y, oof)
    auc_rank = safe_auc(y, rank_score)

    by_seed = {s: np.flatnonzero(seed_values == s) for s in seeds}
    boots_clf: list[float] = []
    boots_rank: list[float] = []
    for _ in range(thresholds.bootstrap_reps):
        pick = rng.choice(seeds, size=len(seeds), replace=True)
        idx = np.concatenate([by_seed[s] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        boots_clf.append(safe_auc(y[idx], oof[idx]))
        boots_rank.append(safe_auc(y[idx], rank_score[idx]))
    ci_clf = percentile_ci(boots_clf)
    ci_rank = percentile_ci(boots_rank)
    return ValidityResult(
        n_agents=len(table),
        n_seeds=len(seeds),
        auc_classifier=auc_clf,
        auc_classifier_ci=ci_clf,
        auc_rank=auc_rank,
        auc_rank_ci=ci_rank,
        positive_rate=float(y.mean()),
    )


def percentile_ci(values: Sequence[float], alpha: float = 0.05) -> tuple[float, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


# ── cost of concealment ───────────────────────────────────────────────────
def classify_strategies(agents: pd.DataFrame, thresholds: AnalysisThresholds = THRESHOLDS) -> pd.DataFrame:
    """Label each sufficiently-sampled agent by its cue-conditional behaviour.

    Rates are computed over training-phase, training-region opportunities.
    ``fitness`` is net energy acquired per tick alive; ``offspring_rate`` is
    offspring per 100 ticks alive.
    """
    split = {"phase": "train", "region": "train"}
    opp0 = ledger_sum(agents, "opportunities", cue=(0,), **split)
    opp1 = ledger_sum(agents, "opportunities", cue=(1,), **split)
    def0 = ledger_sum(agents, "defections", cue=(0,), **split)
    def1 = ledger_sum(agents, "defections", cue=(1,), **split)
    rate0 = safe_rate(def0, opp0).values
    rate1 = safe_rate(def1, opp1).values
    eligible = ((opp0 >= thresholds.min_opportunities) & (opp1 >= thresholds.min_opportunities)).to_numpy()
    delta = rate0 - rate1
    strategy = np.full(len(agents), STRATEGY_MIXED, dtype=object)
    strategy[np.maximum(rate0, rate1) <= thresholds.cooperative_rate_max] = STRATEGY_COOPERATIVE
    strategy[np.minimum(rate0, rate1) >= thresholds.defector_rate_min] = STRATEGY_DEFECTOR
    strategy[delta >= thresholds.conditional_delta_min] = STRATEGY_CONDITIONAL
    lifespan = agents["lifespan"].to_numpy(dtype=float)
    fitness = safe_rate(agents["energy_gained"] - agents["penalties_paid"], lifespan).values
    offspring_rate = 100.0 * safe_rate(agents["offspring"], lifespan).values
    out = pd.DataFrame(
        {
            "seed": agents.get("seed", 0),
            "agent_id": agents["agent_id"],
            "rate_cue0": rate0,
            "rate_cue1": rate1,
            "delta_cue": delta,
            "strategy": strategy,
            "fitness": fitness,
            "offspring_rate": offspring_rate,
            "lifespan": lifespan,
        }
    )
    return out[eligible].reset_index(drop=True)


def concealment_cost(strategies: pd.DataFrame) -> pd.DataFrame:
    """Per-seed mean fitness by strategy, wide format (one row per seed)."""
    if strategies.empty:
        return pd.DataFrame()
    grouped = strategies.groupby(["seed", "strategy"]).agg(
        fitness=("fitness", "mean"),
        offspring_rate=("offspring_rate", "mean"),
        n=("agent_id", "size"),
    )
    return grouped.unstack("strategy")


# ── onset ─────────────────────────────────────────────────────────────────
def onset_window(
    rates: pd.DataFrame,
    upper_band: float,
    consecutive: int = THRESHOLDS.onset_consecutive_windows,
) -> tuple[float, float]:
    """First training window from which ``delta_cue`` stays above the band.

    Returns ``(tick, mean_generation)``; both NaN when the run never clears it.
    """
    train = rates[rates["phase"] == "train"].reset_index(drop=True)
    above = (train["delta_cue"] > upper_band).to_numpy()
    for i in range(len(above) - consecutive + 1):
        if above[i : i + consecutive].all():
            return float(train.loc[i, "window_end_tick"]), float(train.loc[i, "mean_generation"])
    return float("nan"), float("nan")


# ── held-out transfer ─────────────────────────────────────────────────────
def heldout_transfer(agents: pd.DataFrame, train_ticks: int) -> dict[str, float]:
    """Cue divergence during evaluation, in the held-out band vs the training region.

    Only agents born during training are scored, so the measurement reflects
    policies trained without any enforcement in the held-out band.
    """
    trained = agents[agents["born_tick"] < train_ticks]
    out: dict[str, float] = {}
    for region in ("heldout", "train"):
        opp0 = ledger_sum(trained, "opportunities", phase="eval", region=region, cue=(0,)).sum()
        opp1 = ledger_sum(trained, "opportunities", phase="eval", region=region, cue=(1,)).sum()
        def0 = ledger_sum(trained, "defections", phase="eval", region=region, cue=(0,)).sum()
        def1 = ledger_sum(trained, "defections", phase="eval", region=region, cue=(1,)).sum()
        r0 = safe_rate(def0, opp0)
        r1 = safe_rate(def1, opp1)
        out[f"eval_{region}_rate_cue0"] = r0
        out[f"eval_{region}_rate_cue1"] = r1
        out[f"eval_{region}_delta_cue"] = r0 - r1
        out[f"eval_{region}_opportunities"] = float(opp0 + opp1)
    return out

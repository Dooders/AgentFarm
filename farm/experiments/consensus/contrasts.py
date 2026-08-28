"""Paired contrasts for the consensus experiment.

Every paradigm already shares the population and candidate slate within a
trial. This module uses that pairing: for each non-party *selection* rule
versus party, on matching ``trial`` ids, it reports the mean difference,
SD, 95% t-interval, Wilcoxon signed-rank p-value, Holm-adjusted p, and
paired Cohen's d.

Multiple-comparison policy
--------------------------
Pre-registered primary endpoints (default / one-shot cell):

1. Δ minority-cluster welfare (fixed partition)
2. Δ total welfare

each for ``individual``, ``score``, and ``latent_match`` versus ``party``
(six tests per cell). Holm's step-down FWER correction is applied to those
six p-values. ``constrained_individual`` and the allocation baselines are
not in this family.

``λ_winner`` is *not* a primary endpoint in the default cell: voters cannot
condition on λ, so a null there is arithmetic. When ``lambda_correlated``
or ``mechanism == reelection``, ``lambda_winner`` is added as a primary
endpoint and Holm is applied to the larger family.

Everything else (election-endogenous loser welfare, gap, Gini, allocations)
is exploratory and is reported unadjusted.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from farm.experiments.consensus.paradigms import SELECTION_PARADIGMS

BASELINE = "party"
PRIMARY_ENDPOINTS_DEFAULT = ("minority_welfare", "total_welfare")
PRIMARY_ENDPOINTS_LAMBDA = ("minority_welfare", "total_welfare", "lambda_winner")
EXPLORATORY_ENDPOINTS = (
    "loser_welfare",
    "gap",
    "loser_share",
    "majority_welfare",
    "min_utility",
    "p10_utility",
    "gini_utility",
    "supporter_welfare",
    "lambda_winner",
)
CONTRAST_RULES = tuple(p for p in SELECTION_PARADIGMS if p != BASELINE)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the original order."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = [1.0] * n
    running = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx] * (n - rank)
        running = max(running, min(1.0, raw))
        adjusted[idx] = running
    return adjusted


def _paired_stats(treatment: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    delta = np.asarray(treatment, dtype=float) - np.asarray(baseline, dtype=float)
    finite = np.isfinite(delta)
    delta = delta[finite]
    n = int(delta.size)
    if n == 0:
        return {
            "n_pairs": 0,
            "delta_mean": float("nan"),
            "delta_sd": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "pvalue": float("nan"),
            "effect_size": float("nan"),
            "wilcoxon_stat": float("nan"),
        }
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and sd > 0:
        se = sd / np.sqrt(n)
        tcrit = float(stats.t.ppf(0.975, n - 1))
        ci_low, ci_high = mean - tcrit * se, mean + tcrit * se
        effect = mean / sd
    else:
        ci_low = ci_high = mean
        effect = 0.0 if sd == 0.0 else float("nan")
    if n < 2 or np.allclose(delta, 0.0):
        pvalue, stat = 1.0, 0.0
    else:
        stat, pvalue = stats.wilcoxon(delta, zero_method="wilcox", alternative="two-sided")
        stat, pvalue = float(stat), float(pvalue)
    return {
        "n_pairs": n,
        "delta_mean": mean,
        "delta_sd": sd,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "pvalue": pvalue,
        "effect_size": float(effect),
        "wilcoxon_stat": stat,
    }


def _aligned_pairs(trials: pd.DataFrame, treatment: str, endpoint: str) -> tuple[np.ndarray, np.ndarray]:
    base = trials.loc[trials["paradigm"] == BASELINE, ["trial", endpoint]].rename(columns={endpoint: "base"})
    treat = trials.loc[trials["paradigm"] == treatment, ["trial", endpoint]].rename(columns={endpoint: "treat"})
    merged = base.merge(treat, on="trial", how="inner")
    return merged["treat"].to_numpy(), merged["base"].to_numpy()


def paired_contrasts(
    trials: pd.DataFrame,
    *,
    primary_endpoints: Iterable[str] | None = None,
    include_lambda_primary: bool = False,
) -> pd.DataFrame:
    """Build the contrasts table for every (population, n_candidates) cell."""
    endpoints = tuple(primary_endpoints or (PRIMARY_ENDPOINTS_LAMBDA if include_lambda_primary else PRIMARY_ENDPOINTS_DEFAULT))
    rows: list[dict] = []
    group_keys = ["population", "n_candidates"]
    present = [k for k in group_keys if k in trials.columns]
    grouped = trials.groupby(present, sort=False) if present else [((), trials)]

    for key, cell in grouped:
        key = (key,) if not isinstance(key, tuple) else key
        meta = dict(zip(present, key))
        primary_index: list[int] = []
        rules = [r for r in CONTRAST_RULES if r in set(cell["paradigm"])]
        for rule in rules:
            for endpoint in endpoints:
                treat, base = _aligned_pairs(cell, rule, endpoint)
                stats_row = _paired_stats(treat, base)
                rows.append(
                    {
                        **meta,
                        "paradigm": rule,
                        "endpoint": endpoint,
                        "family": "primary",
                        **stats_row,
                    }
                )
                primary_index.append(len(rows) - 1)
            for endpoint in EXPLORATORY_ENDPOINTS:
                if endpoint in endpoints:
                    continue
                if endpoint not in cell.columns:
                    continue
                treat, base = _aligned_pairs(cell, rule, endpoint)
                stats_row = _paired_stats(treat, base)
                rows.append(
                    {
                        **meta,
                        "paradigm": rule,
                        "endpoint": endpoint,
                        "family": "exploratory",
                        **stats_row,
                    }
                )
        if primary_index:
            pvalues = [rows[i]["pvalue"] for i in primary_index]
            adjusted = holm_adjust(pvalues)
            for i, adj in zip(primary_index, adjusted):
                rows[i]["pvalue_adjusted"] = adj
        for row in rows:
            row.setdefault("pvalue_adjusted", float("nan"))

    if not rows:
        return pd.DataFrame(
            columns=[
                "population",
                "n_candidates",
                "paradigm",
                "endpoint",
                "family",
                "n_pairs",
                "delta_mean",
                "delta_sd",
                "ci_low",
                "ci_high",
                "pvalue",
                "pvalue_adjusted",
                "effect_size",
                "wilcoxon_stat",
            ]
        )
    return pd.DataFrame(rows)


def selection_trials(trials: pd.DataFrame) -> pd.DataFrame:
    """Rows that are real selection rules (not baselines or the λ cap)."""
    return trials[trials["paradigm"].isin(SELECTION_PARADIGMS)].copy()

"""
Cross-domain metrics for system dynamics (correlations, lag structure, heuristics).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from farm.utils.logging import get_logger

logger = get_logger(__name__)


def _pearson_clean(a: pd.Series, b: pd.Series) -> Dict[str, Any]:
    """Pearson correlation on pairwise-complete observations."""
    df = pd.DataFrame({"a": a, "b": b}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    if n < 3:
        return {"available": False, "reason": "insufficient_joint_samples", "n": int(n)}
    r, p = stats.pearsonr(df["a"].astype(float), df["b"].astype(float))
    return {
        "available": True,
        "pearson_r": float(r),
        "p_value": float(p),
        "n": int(n),
    }


def resource_population_coupling(df: pd.DataFrame) -> Dict[str, Any]:
    """Pearson and first-difference correlation between resources and population."""
    out: Dict[str, Any] = {"levels": None, "first_differences": None}

    if "total_resources" not in df.columns or "total_agents" not in df.columns:
        out["error"] = "missing_total_resources_or_total_agents"
        return out

    levels = _pearson_clean(df["total_resources"], df["total_agents"])
    out["levels"] = levels

    d = df.sort_values("step").copy()
    d_res = d["total_resources"].diff()
    d_pop = d["total_agents"].diff()
    out["first_differences"] = _pearson_clean(d_res, d_pop)

    return out


def lag_correlation_series(
    x: pd.Series, y: pd.Series, max_lag: int = 5
) -> Dict[str, Any]:
    """Correlate ``x`` with ``y`` shifted forward by lag (x_t vs y_{t+lag})."""
    clean = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < max_lag + 5:
        return {"available": False, "reason": "insufficient_length", "lags": []}

    x_arr = clean["x"].to_numpy(dtype=float)
    y_arr = clean["y"].to_numpy(dtype=float)
    lags_out: List[Dict[str, Any]] = []
    for lag in range(0, max_lag + 1):
        if lag == 0:
            xs, ys = x_arr, y_arr
        else:
            xs, ys = x_arr[:-lag], y_arr[lag:]
        if len(xs) < 3:
            continue
        r, p = stats.pearsonr(xs, ys)
        lags_out.append({"lag": lag, "pearson_r": float(r), "p_value": float(p), "n": int(len(xs))})

    if not lags_out:
        return {"available": False, "reason": "no_valid_lags", "lags": []}

    best = max(lags_out, key=lambda d: abs(d["pearson_r"]))
    return {"available": True, "lags": lags_out, "strongest": best}


def action_reward_lag_coupling(df: pd.DataFrame, max_lag: int = 5) -> Dict[str, Any]:
    """Lag correlation between per-step action volume and mean reward."""
    if "actions_per_step" not in df.columns or "mean_reward_per_step" not in df.columns:
        return {"available": False, "reason": "missing_temporal_aggregates"}

    d = df.sort_values("step").replace([np.inf, -np.inf], np.nan).dropna(
        subset=["actions_per_step", "mean_reward_per_step"]
    )
    if d.empty:
        return {"available": False, "reason": "empty_after_dropna"}

    # Actions may predict later mean reward (or vice versa): report both directions.
    forward = lag_correlation_series(d["actions_per_step"], d["mean_reward_per_step"], max_lag=max_lag)
    backward = lag_correlation_series(d["mean_reward_per_step"], d["actions_per_step"], max_lag=max_lag)
    return {
        "available": bool(forward.get("available") or backward.get("available")),
        "actions_lead_reward": forward,
        "reward_leads_actions": backward,
    }


def scarcity_population_volatility(df: pd.DataFrame, window: int = 5) -> Dict[str, Any]:
    """Couple resource scarcity (inverted, smoothed) with rolling population volatility."""
    if "total_resources" not in df.columns or "total_agents" not in df.columns:
        return {"available": False, "reason": "missing_columns"}

    d = df.sort_values("step").copy()
    d = d.dropna(subset=["total_resources", "total_agents"])
    if len(d) < window + 3:
        return {"available": False, "reason": "insufficient_rows"}

    scarcity = 1.0 / (d["total_resources"].astype(float).clip(lower=1e-9))
    vol = d["total_agents"].astype(float).rolling(window=window, min_periods=window).std()
    aligned = pd.DataFrame({"scarcity": scarcity, "pop_volatility": vol}).dropna()
    if len(aligned) < 3:
        return {"available": False, "reason": "insufficient_after_rolling"}

    corr = _pearson_clean(aligned["scarcity"], aligned["pop_volatility"])
    return {
        "available": corr.get("available", False),
        "window": window,
        "correlation": corr,
    }


def granger_resource_population_changes(
    df: pd.DataFrame, maxlag: int = 3
) -> Dict[str, Any]:
    """Granger causality on first differences (resources -> population and reverse)."""
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return {"available": False, "reason": "statsmodels_not_available"}

    if "total_resources" not in df.columns or "total_agents" not in df.columns:
        return {"available": False, "reason": "missing_columns"}

    d = df.sort_values("step").copy()
    d_res = d["total_resources"].diff()
    d_pop = d["total_agents"].diff()
    g = pd.DataFrame({"d_res": d_res, "d_pop": d_pop}).dropna()
    if len(g) < maxlag + 8:
        return {"available": False, "reason": "insufficient_observations", "n": len(g)}

    def _min_pvalue(res: Dict[int, Any]) -> float:
        pvals: List[float] = []
        for lag in range(1, maxlag + 1):
            if lag not in res:
                continue
            entry = res[lag]
            tests = entry[0] if isinstance(entry, (list, tuple)) else entry
            if isinstance(tests, dict):
                for t in tests.values():
                    if hasattr(t, "pvalue"):
                        pvals.append(float(t.pvalue))
                    elif isinstance(t, (list, tuple)) and len(t) >= 2:
                        pvals.append(float(t[1]))
        return float(min(pvals)) if pvals else float("nan")

    out: Dict[str, Any] = {"available": True, "maxlag": maxlag, "resource_to_population": {}, "population_to_resource": {}}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            r1 = grangercausalitytests(g[["d_pop", "d_res"]], maxlag=maxlag, verbose=False)
            r2 = grangercausalitytests(g[["d_res", "d_pop"]], maxlag=maxlag, verbose=False)
        out["resource_to_population"] = {"min_p_value_across_lags": _min_pvalue(r1)}
        out["population_to_resource"] = {"min_p_value_across_lags": _min_pvalue(r2)}
    except Exception as err:
        logger.debug("Granger test failed: %s", err)
        return {"available": False, "reason": str(err)}

    return out


def feedback_loop_candidates(
    df: pd.DataFrame,
    scarcity_quantile: float = 0.25,
    recovery_window: int = 5,
    min_resource_recovery_frac: float = 0.02,
) -> Dict[str, Any]:
    """Heuristic episodes: stressed resources + falling population + partial resource rebound."""
    need = {"step", "total_resources", "total_agents"}
    if not need.issubset(df.columns):
        return {"available": False, "reason": "missing_columns", "periods": []}

    d = df.sort_values("step").copy()
    d = d.dropna(subset=["total_resources", "total_agents"])
    if len(d) < recovery_window + 3:
        return {"available": False, "reason": "insufficient_rows", "periods": []}

    thresh = d["total_resources"].quantile(scarcity_quantile)
    d["d_agents"] = d["total_agents"].diff()
    d["d_res"] = d["total_resources"].diff()

    periods: List[Dict[str, Any]] = []
    steps = d["step"].to_numpy()
    res = d["total_resources"].to_numpy(dtype=float)
    d_ag = d["d_agents"].to_numpy(dtype=float)

    for i in range(1, len(d) - recovery_window - 1):
        if res[i] > thresh:
            continue
        if not np.isfinite(d_ag[i]) or d_ag[i] >= 0:
            continue
        base = res[i]
        future = res[i + 1 : i + 1 + recovery_window]
        if len(future) == 0:
            continue
        peak_rec = float(np.max(future))
        if base > 0 and (peak_rec - base) / base < min_resource_recovery_frac:
            continue
        periods.append(
            {
                "stress_step": int(steps[i]),
                "resource_at_stress": float(base),
                "population_change_at_stress": float(d_ag[i]),
                "max_resource_within_window": peak_rec,
                "recovery_window": recovery_window,
            }
        )

    return {
        "available": True,
        "scarcity_quantile": scarcity_quantile,
        "periods": periods,
        "count": len(periods),
    }


def synthesize_system_dynamics(df: pd.DataFrame) -> Dict[str, Any]:
    """Run all cross-domain computations for the unified report."""
    return {
        "resource_population": resource_population_coupling(df),
        "action_reward_lags": action_reward_lag_coupling(df),
        "scarcity_population_volatility": scarcity_population_volatility(df),
        "granger_changes": granger_resource_population_changes(df),
        "feedback_loop_candidates": feedback_loop_candidates(df),
    }


def json_safe(obj: Any) -> Any:
    """Recursively convert values for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    return str(obj)

"""Run-level metrics: unit persistence, failure modes, and founder welfare.

Persistence uses survival analysis because units still alive at the end of
a run are censored: the primary metric is the *restricted mean survival
time* (RMST) of bond units up to :data:`RMST_TAU` ticks -- the expected
lifetime of a unit, truncated at ``tau`` -- from a Kaplan-Meier estimate.

Welfare compares each founder with itself in the solo twin (same traits,
same shock sequence): lifespan difference, starvation, mean energy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from farm.core.cooperation import FailureMode
from farm.experiments.cooperation.config import BOND_KIND, FAMILY_SETTINGS, SURVIVAL_HORIZON
from farm.experiments.cooperation.simulation import RunResult

RMST_TAU: int = 200

MODE_COLUMNS: tuple[str, ...] = tuple(m.value for m in FailureMode if m is not FailureMode.INDETERMINATE)


def kaplan_meier(durations: Sequence[float], observed: Sequence[bool]) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival curve.

    Returns ``(times, survival)`` where ``survival[i]`` is the estimated
    probability of surviving beyond ``times[i]``; the curve starts at
    ``(0, 1)``.
    """
    d = np.asarray(durations, dtype=float)
    e = np.asarray(observed, dtype=bool)
    if d.size == 0:
        return np.array([0.0]), np.array([1.0])
    times = np.unique(d[e])
    surv = []
    s = 1.0
    for t in times:
        at_risk = np.sum(d >= t)
        events = np.sum((d == t) & e)
        if at_risk > 0:
            s *= 1.0 - events / at_risk
        surv.append(s)
    return np.concatenate([[0.0], times]), np.concatenate([[1.0], surv])


def survival_at(durations: Sequence[float], observed: Sequence[bool], t: float) -> float:
    """Kaplan-Meier probability of surviving beyond ``t``."""
    if len(durations) == 0:
        return float("nan")
    times, surv = kaplan_meier(durations, observed)
    idx = np.searchsorted(times, t, side="right") - 1
    return float(surv[max(idx, 0)])


def restricted_mean_survival(durations: Sequence[float], observed: Sequence[bool], tau: float) -> float:
    """Area under the Kaplan-Meier curve on ``[0, tau]``.

    If the longest follow-up ends before ``tau`` with the curve above zero,
    the last value is carried forward (standard RMST convention).
    """
    if len(durations) == 0:
        return float("nan")
    times, surv = kaplan_meier(durations, observed)
    area = 0.0
    for i in range(len(times)):
        start = times[i]
        if start >= tau:
            break
        end = times[i + 1] if i + 1 < len(times) else tau
        area += surv[i] * (min(end, tau) - start)
    return float(area)


def bond_persistence(units: pd.DataFrame) -> dict[str, float]:
    bonds = units[units["kind"] == BOND_KIND] if len(units) else units
    if bonds is None or len(bonds) == 0:
        return {"bonds_formed": 0, "bond_rmst": float("nan"), "bond_survival_100": float("nan")}
    durations = bonds["lifetime"].to_numpy(dtype=float)
    observed = ~bonds["censored"].to_numpy(dtype=bool)
    return {
        "bonds_formed": len(bonds),
        "bond_rmst": restricted_mean_survival(durations, observed, RMST_TAU),
        "bond_survival_100": survival_at(durations, observed, SURVIVAL_HORIZON),
        "bond_mean_observed_lifetime": float(np.mean(durations)),
    }


def mode_shares(units: pd.DataFrame, kind: str = BOND_KIND) -> dict[str, float]:
    """Primary-mode shares and any-mode prevalence among classified units."""
    out: dict[str, float] = {}
    pool = (
        units[(units["kind"] == kind) & (units["primary_mode"] != FailureMode.INDETERMINATE.value)]
        if len(units)
        else units
    )
    n = len(pool) if pool is not None else 0
    out[f"{kind}_classified"] = float(n)
    for mode in MODE_COLUMNS:
        if n:
            out[f"{kind}_primary_{mode}"] = float((pool["primary_mode"] == mode).mean())
            out[f"{kind}_any_{mode}"] = float(
                pool["modes"].fillna("").str.split("|").apply(lambda ms, m=mode: m in ms).mean()
            )
        else:
            out[f"{kind}_primary_{mode}"] = float("nan")
            out[f"{kind}_any_{mode}"] = float("nan")
    return out


def summarize_run(result: RunResult) -> dict:
    """Scalar outcomes of one run (cooperative or solo)."""
    cfg = result.config
    ts = result.timeseries
    agents = result.agents
    units = result.units
    counters = result.counters
    founders = agents[agents["founder"]]
    row: dict = {
        "run_id": cfg.run_id,
        "condition": cfg.condition.name if cfg.cooperative else "solo",
        "family": cfg.condition.family if cfg.cooperative else "solo",
        "seed": cfg.seed,
        "cooperative": cfg.cooperative,
        "scope": cfg.condition.bond.scope,
        "exit_cost": cfg.condition.bond.exit_cost,
        "recognition": cfg.condition.bond.recognition,
        "symmetry": cfg.condition.bond.symmetry,
        "nesting": cfg.condition.nesting,
        "ticks_run": len(ts),
        "population_mean": float(ts["population"].mean()) if len(ts) else float("nan"),
        "population_final": int(ts["population"].iloc[-1]) if len(ts) else 0,
        "coverage_mean": float(ts["coverage"].mean()) if len(ts) else 0.0,
        "effort_mean": float(ts["mean_effort"].mean()) if len(ts) else float("nan"),
        "energy_mean": float(ts["mean_energy"].mean()) if len(ts) else float("nan"),
        "births": int(counters.get("births", 0)),
        "births_blocked_cap": int(counters.get("births_blocked_cap", 0)),
        "deaths_starvation": int(counters.get("deaths_starvation", 0)),
        "deaths_age": int(counters.get("deaths_age", 0)),
        "energy_spilled": float(counters.get("energy_spilled", 0.0)),
        "founder_starvation": int((founders["death_cause"] == "starvation").sum()),
        "founder_lifespan_mean": float(founders["lifespan"].mean()) if len(founders) else float("nan"),
        "matches_chosen": int(counters.get("matches_chosen", 0)),
        "matches_arranged": int(counters.get("matches_arranged", 0)),
        "matches_arranged_unconsented": int(counters.get("matches_arranged_unconsented", 0)),
        "joins": int(counters.get("joins", 0)),
        "switches": int(counters.get("switches", 0)),
        "offers_blocked": int(counters.get("offers_blocked", 0)),
        "exits_bond": int(counters.get(f"exits_{BOND_KIND}", 0)),
        "exits_enforced_bond": int(counters.get(f"exits_enforced_{BOND_KIND}", 0)),
        "exits_vetoed_bond": int(counters.get(f"exits_vetoed_{BOND_KIND}", 0)),
        "families_formed": int(counters.get("families_formed", 0)),
        "exits_family": int(counters.get(f"exits_{FAMILY_SETTINGS.kind}", 0)),
    }
    if cfg.cooperative:
        row.update(bond_persistence(units))
        row.update(mode_shares(units, BOND_KIND))
        bonds = units[units["kind"] == BOND_KIND] if len(units) else units
        member_steps = float(bonds["member_steps"].sum()) if len(bonds) else 0.0
        row["trap_rate"] = float(bonds["trapped_steps"].sum()) / member_steps if member_steps else 0.0
        gross_all = float(agents["gross_income"].sum())
        row["surplus_share"] = float(bonds["surplus"].sum()) / gross_all if gross_all else 0.0
        row["pooled_share"] = float(agents["pooled"].sum()) / gross_all if gross_all else 0.0
    return row


def founder_pairs(coop: RunResult, solo: RunResult) -> pd.DataFrame:
    """One row per founder: its outcomes in the cooperative run and its solo twin."""
    if coop.config.seed != solo.config.seed or solo.config.cooperative:
        raise ValueError("founder_pairs needs a cooperative run and the solo twin of the same seed")
    cols = [
        "slot",
        "skill",
        "diligence",
        "motive",
        "lifespan",
        "death_cause",
        "mean_energy",
        "offspring",
        "member_share",
    ]
    c = coop.agents.loc[coop.agents["founder"], cols].set_index("slot")
    solo_cols = ["slot", "skill", "lifespan", "death_cause", "mean_energy", "offspring"]
    s = solo.agents.loc[solo.agents["founder"], solo_cols].set_index("slot")
    if not c.index.equals(s.index) or not np.allclose(c["skill"].to_numpy(), s["skill"].to_numpy()):
        raise ValueError("founder sets differ between the cooperative run and its solo twin")
    out = c.join(s.drop(columns=["skill"]), rsuffix="_solo")
    out["d_lifespan"] = out["lifespan"] - out["lifespan_solo"]
    out["starved"] = out["death_cause"] == "starvation"
    out["starved_solo"] = out["death_cause_solo"] == "starvation"
    out["d_mean_energy"] = out["mean_energy"] - out["mean_energy_solo"]
    out["d_offspring"] = out["offspring"] - out["offspring_solo"]
    out = out.reset_index()
    out.insert(0, "seed", coop.config.seed)
    out.insert(0, "condition", coop.config.condition.name)
    return out


def welfare_summary(pairs: pd.DataFrame) -> dict[str, float]:
    """Run-level welfare from :func:`founder_pairs`."""
    if pairs.empty:
        return {}
    return {
        "welfare_d_lifespan": float(pairs["d_lifespan"].mean()),
        "welfare_d_lifespan_median": float(pairs["d_lifespan"].median()),
        "welfare_better_share": float((pairs["d_lifespan"] > 0).mean()),
        "welfare_worse_share": float((pairs["d_lifespan"] < 0).mean()),
        "welfare_starved": float(pairs["starved"].mean()),
        "welfare_starved_solo": float(pairs["starved_solo"].mean()),
        "welfare_d_mean_energy": float(pairs["d_mean_energy"].mean()),
        "welfare_d_offspring": float(pairs["d_offspring"].mean()),
    }

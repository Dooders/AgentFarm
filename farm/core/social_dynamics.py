"""Per-step cooperation and competition metrics from agent action data."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from farm.database.models import ActionModel

# Targeted social actions (agent-to-agent or agent-to-target), aligned with social_behavior.compute
COOPERATION_ACTION_TYPES: Tuple[str, ...] = ("share", "assist", "defend")
COMPETITION_ACTION_TYPES: Tuple[str, ...] = ("attack", "steal")
SOCIAL_TARGETED_TYPES: Tuple[str, ...] = COOPERATION_ACTION_TYPES + COMPETITION_ACTION_TYPES


def social_dynamics_per_step(session: Session, simulation_id: Optional[str] = None) -> pd.DataFrame:
    """Aggregate cooperation and competition counts per simulation step.

    Denominator ``total_social_interactions`` counts only targeted actions whose type is
    in cooperation or competition sets (not moves, gathers, etc.).

    ``cooperation_rate`` is (share + assist + defend) / total_social_interactions.
    ``competition_intensity`` is attack_events / total_social_interactions (attacks only),
    matching the common combat proxy. ``competition_intensity_including_steal`` adds steals
    to the numerator.

    Parameters
    ----------
    session
        Open SQLAlchemy session.
    simulation_id
        When set, restrict to actions for this simulation.

    Returns
    -------
    pd.DataFrame
        Columns include step, counts, and derived rates (NaN when total is 0).
    """
    coop_cond = ActionModel.action_type.in_(COOPERATION_ACTION_TYPES)
    share_cond = ActionModel.action_type == "share"
    attack_cond = ActionModel.action_type == "attack"
    steal_cond = ActionModel.action_type == "steal"

    q = (
        session.query(
            ActionModel.step_number.label("step"),
            func.sum(case((coop_cond, 1), else_=0)).label("cooperation_actions"),
            func.sum(case((share_cond, 1), else_=0)).label("share_actions"),
            func.sum(case((attack_cond, 1), else_=0)).label("attack_events"),
            func.sum(case((steal_cond, 1), else_=0)).label("steal_events"),
            func.count(ActionModel.action_id).label("total_social_interactions"),
        )
        .filter(
            ActionModel.action_target_id.isnot(None),
            ActionModel.action_type.in_(SOCIAL_TARGETED_TYPES),
        )
    )
    if simulation_id is not None:
        q = q.filter(ActionModel.simulation_id == simulation_id)

    q = q.group_by(ActionModel.step_number).order_by(ActionModel.step_number)
    rows = q.all()
    columns = [
        "step",
        "cooperation_actions",
        "share_actions",
        "attack_events",
        "steal_events",
        "total_social_interactions",
    ]
    if not rows:
        empty = pd.DataFrame(columns=columns)
        for extra in (
            "cooperation_rate",
            "share_rate",
            "competition_intensity",
            "competition_intensity_including_steal",
        ):
            empty[extra] = pd.Series(dtype=float)
        return empty

    df = pd.DataFrame(rows, columns=columns)
    total = df["total_social_interactions"].replace(0, np.nan)
    df["cooperation_rate"] = df["cooperation_actions"] / total
    df["share_rate"] = df["share_actions"] / total
    df["competition_intensity"] = df["attack_events"] / total
    df["competition_intensity_including_steal"] = (df["attack_events"] + df["steal_events"]) / total
    return df


def compute_social_dynamics_trends(per_step: pd.DataFrame) -> Dict[str, Any]:
    """Summarize time-series behaviour of cooperation and competition rates.

    Includes linear trend (least squares slope over step order) and mean step-to-step
    change in ``competition_intensity`` ("combat escalation" proxy).
    """
    if per_step.empty or "cooperation_rate" not in per_step.columns:
        return {}

    out: Dict[str, Any] = {}
    ordered = per_step.sort_values("step").reset_index(drop=True)

    for col, prefix in (
        ("cooperation_rate", "cooperation_rate"),
        ("competition_intensity", "competition_intensity"),
    ):
        if col not in ordered.columns:
            continue
        series = ordered[col].dropna()
        if len(series) < 2:
            continue
        y = series.to_numpy(dtype=float)
        x = np.arange(len(y), dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        out[f"{prefix}_trend_slope"] = float(slope)
        out[f"{prefix}_trend_intercept"] = float(intercept)

    if "competition_intensity" in ordered.columns and len(ordered) >= 2:
        ci = ordered["competition_intensity"].astype(float)
        deltas = ci.diff().dropna()
        if len(deltas):
            out["combat_escalation_mean_delta"] = float(deltas.mean())

    return out


def per_step_records_for_json(df: pd.DataFrame) -> list:
    """Convert per-step frame to JSON-serializable records (NaN/inf -> None)."""
    if df.empty:
        return []
    safe = df.replace([np.inf, -np.inf], np.nan)
    records = safe.to_dict(orient="records")
    for row in records:
        for k, v in list(row.items()):
            if v is not None and isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                row[k] = None
    return records

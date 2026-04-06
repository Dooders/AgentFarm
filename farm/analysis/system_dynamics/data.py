"""
Merge population, resource, and temporal streams for system-dynamics analysis.
"""

from pathlib import Path
from typing import List

import pandas as pd

from farm.analysis.population.data import process_population_data
from farm.analysis.resources.data import process_resource_data
from farm.analysis.temporal.data import process_temporal_data
from farm.utils.logging import get_logger

logger = get_logger(__name__)


def _merge_outer_on_step(parts: List[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge frames on ``step``; skips empty inputs."""
    usable = [p for p in parts if p is not None and not p.empty and "step" in p.columns]
    if not usable:
        return pd.DataFrame(columns=["step"])

    out = usable[0]
    for nxt in usable[1:]:
        out = out.merge(nxt, on="step", how="outer")

    return out.sort_values("step").reset_index(drop=True)


def _aggregate_temporal_by_step(temp_df: pd.DataFrame) -> pd.DataFrame:
    """Per-step action / reward aggregates from raw temporal rows."""
    if temp_df.empty or "step" not in temp_df.columns:
        return pd.DataFrame(
            columns=[
                "step",
                "actions_per_step",
                "mean_reward_per_step",
                "total_reward_per_step",
            ]
        )

    work = temp_df.copy()
    if "reward" not in work.columns:
        work["reward"] = 0.0
    work["reward"] = pd.to_numeric(work["reward"], errors="coerce").fillna(0.0)

    return (
        work.groupby("step", as_index=False)
        .agg(
            actions_per_step=("reward", "count"),
            mean_reward_per_step=("reward", "mean"),
            total_reward_per_step=("reward", "sum"),
        )
        .sort_values("step")
        .reset_index(drop=True)
    )


def process_system_dynamics_data(
    experiment_path: Path, use_database: bool = True, **kwargs
) -> pd.DataFrame:
    """Load population, resources, and temporal data and align on ``step``.

    Args:
        experiment_path: Experiment directory (DB or CSV layout as other modules).
        use_database: Passed through to population/resources loaders.
        **kwargs: Extra args forwarded to loaders.

    Returns:
        Single DataFrame outer-joined on ``step`` with per-step metrics.
    """
    experiment_path = Path(experiment_path)

    pop_df = process_population_data(
        experiment_path, use_database=use_database, **kwargs
    )
    res_df = process_resource_data(
        experiment_path, use_database=use_database, **kwargs
    )
    temp_df = process_temporal_data(
        experiment_path, use_database=use_database, **kwargs
    )
    temporal_agg = _aggregate_temporal_by_step(temp_df)

    merged = _merge_outer_on_step([pop_df, res_df, temporal_agg])
    if merged.empty:
        logger.warning("System dynamics merge produced no rows for %s", experiment_path)
    else:
        logger.info("System dynamics merged frame: %s rows, columns=%s", len(merged), list(merged.columns))
    return merged

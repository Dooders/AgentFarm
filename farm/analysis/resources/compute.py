"""
Resource statistical computations.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from farm.analysis.common.utils import calculate_statistics, calculate_trend


def compute_resource_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute comprehensive resource statistics.

    Args:
        df: Resource data with columns: step, total_resources, etc.

    Returns:
        Dictionary of computed statistics
    """
    total_resources = df['total_resources'].values

    stats = {
        'total': calculate_statistics(total_resources),
        'peak_step': int(np.argmax(total_resources)),
        'peak_value': float(np.max(total_resources)),
        'final_value': float(total_resources[-1]),
        'trend': calculate_trend(total_resources),
        'resource_stability': float(1.0 / (1.0 + np.std(total_resources) / (np.mean(total_resources) + 1e-6))),
    }

    # Add consumption statistics if available
    if 'consumed_resources' in df.columns:
        consumed = df['consumed_resources'].values
        stats['consumption'] = {
            'total': float(np.sum(consumed)),
            'mean': float(np.mean(consumed)),
            'min': float(np.min(consumed)),
            'max': float(np.max(consumed)),
            'std': float(np.std(consumed)),
        }
        stats['depletion_rate'] = calculate_trend(total_resources)

    # Add efficiency statistics if available
    if 'resource_efficiency' in df.columns:
        efficiency = df['resource_efficiency'].values
        stats['efficiency'] = calculate_statistics(efficiency)

    # Distribution entropy statistics
    if 'distribution_entropy' in df.columns:
        entropy = df['distribution_entropy'].values
        stats['entropy'] = calculate_statistics(entropy)
        stats['avg_distribution_uniformity'] = float(np.mean(entropy))

    # Efficiency metrics
    efficiency_cols = ['utilization_rate', 'distribution_efficiency', 'consumption_efficiency', 'resource_efficiency']
    for col in efficiency_cols:
        if col in df.columns:
            stats[col] = calculate_statistics(df[col].values)

    return stats


def compute_consumption_patterns(df: pd.DataFrame) -> Dict[str, float]:
    """Compute resource consumption patterns.

    Args:
        df: Resource data with consumption metrics

    Returns:
        Dictionary of consumption metrics
    """
    if 'consumed_resources' not in df.columns:
        return {}

    consumed = df['consumed_resources'].values

    return {
        'trend': calculate_trend(consumed),
        'volatility': float(np.std(consumed) / (np.mean(consumed) + 1e-6)),
        'peak_consumption': float(np.max(consumed)),
        'increasing_periods': int(np.sum(np.diff(consumed) > 0)),
    }


def compute_resource_efficiency(df: pd.DataFrame) -> Dict[str, float]:
    """Compute resource utilization efficiency metrics.

    Args:
        df: Resource data with efficiency metrics

    Returns:
        Dictionary of efficiency metrics
    """
    efficiency_metrics = {}

    if 'utilization_rate' in df.columns:
        efficiency_metrics['avg_utilization_rate'] = float(df['utilization_rate'].mean())

    if 'distribution_efficiency' in df.columns:
        efficiency_metrics['avg_distribution_efficiency'] = float(df['distribution_efficiency'].mean())

    if 'consumption_efficiency' in df.columns:
        efficiency_metrics['avg_consumption_efficiency'] = float(df['consumption_efficiency'].mean())

    if 'regeneration_rate' in df.columns:
        efficiency_metrics['avg_regeneration_rate'] = float(df['regeneration_rate'].mean())

    # Overall efficiency score (weighted average)
    if efficiency_metrics:
        weights = {
            'avg_utilization_rate': 0.4,
            'avg_distribution_efficiency': 0.3,
            'avg_consumption_efficiency': 0.3,
        }
        overall_score = 0.0
        total_weight = 0.0
        for metric, weight in weights.items():
            if metric in efficiency_metrics:
                overall_score += efficiency_metrics[metric] * weight
                total_weight += weight

        if total_weight > 0:
            efficiency_metrics['overall_efficiency_score'] = overall_score / total_weight

    return efficiency_metrics


def compute_efficiency_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute efficiency metrics from resource data.

    Args:
        df: Resource data with efficiency columns

    Returns:
        Dictionary of efficiency metrics
    """
    if 'resource_efficiency' not in df.columns:
        return {}

    efficiency = df['resource_efficiency'].values

    return {
        'mean_efficiency': float(np.mean(efficiency)),
        'efficiency_trend': calculate_trend(efficiency),
        'efficiency_volatility': float(np.std(efficiency) / (np.mean(efficiency) + 1e-6)),
        'peak_efficiency': float(np.max(efficiency)),
    }


def _timeseries_hotspot_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Scalar concentration metrics from global total_resources over time."""
    series = df["total_resources"].dropna()
    if series.empty:
        return {
            "max_concentration": 0.0,
            "avg_concentration": 0.0,
            "concentration_ratio": 0.0,
            "hotspot_intensity": 0.0,
        }
    total_resources = series.values
    mean_resources = float(np.mean(total_resources))
    max_resources = float(np.max(total_resources))
    concentration_ratio = max_resources / (mean_resources + 1e-6)
    return {
        "max_concentration": max_resources,
        "avg_concentration": mean_resources,
        "concentration_ratio": concentration_ratio,
        "hotspot_intensity": float(concentration_ratio - 1.0),
    }


def _cell_key(x: Any, y: Any) -> Tuple[Any, Any]:
    return (x, y)


def _hotspots_for_step(
    step_df: pd.DataFrame, sigma_multiplier: float
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Per-step grid cells above mean + sigma * std on aggregated cell amounts."""
    if step_df.empty:
        return [], {"mean": 0.0, "std": 0.0, "threshold": 0.0}

    grouped = (
        step_df.groupby(["position_x", "position_y"], as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "cell_amount"})
    )
    amounts = grouped["cell_amount"].values.astype(float)
    mean_amt = float(np.mean(amounts))
    # Sample std across cells (ddof=1) so threshold matches "mean + k·σ" for small grids
    std_amt = float(np.std(amounts, ddof=1)) if len(amounts) > 1 else 0.0
    if not np.isfinite(std_amt) or std_amt <= 0:
        threshold = mean_amt
    else:
        threshold = mean_amt + sigma_multiplier * std_amt

    above = grouped[grouped["cell_amount"] > threshold].copy()
    hotspots: List[Dict[str, Any]] = []
    for _, row in above.iterrows():
        z = (
            (float(row["cell_amount"]) - mean_amt) / std_amt
            if std_amt > 0 and np.isfinite(std_amt)
            else 0.0
        )
        hotspots.append(
            {
                "position_x": float(row["position_x"]),
                "position_y": float(row["position_y"]),
                "cell_amount": float(row["cell_amount"]),
                "threshold": float(threshold),
                "z_score": float(z),
            }
        )
    meta = {"mean": mean_amt, "std": std_amt, "threshold": float(threshold)}
    return hotspots, meta


# Limit JSON payload size; centroids, mass, and persistence use the full hotspot set in-memory.
MAX_SERIALIZED_HOTSPOT_CELLS_PER_STEP = 500


def _hotspot_cells_for_export(
    hotspots: List[Dict[str, Any]], max_cells: int
) -> Tuple[List[Dict[str, Any]], bool]:
    """Subset of cells for serialization (highest |z_score| first)."""
    if len(hotspots) <= max_cells:
        return list(hotspots), False
    sorted_h = sorted(hotspots, key=lambda h: abs(h["z_score"]), reverse=True)
    return sorted_h[:max_cells], True


def _weighted_centroid(cells: List[Dict[str, Any]]) -> Optional[List[float]]:
    if not cells:
        return None
    total_w = sum(c["cell_amount"] for c in cells)
    if total_w <= 0:
        cx = float(np.mean([c["position_x"] for c in cells]))
        cy = float(np.mean([c["position_y"] for c in cells]))
        return [cx, cy]
    cx = sum(c["position_x"] * c["cell_amount"] for c in cells) / total_w
    cy = sum(c["position_y"] * c["cell_amount"] for c in cells) / total_w
    return [float(cx), float(cy)]


def _compute_spatial_resource_hotspots(
    resource_positions: pd.DataFrame,
    sigma_multiplier: float = 2.0,
) -> Dict[str, Any]:
    """Threshold-based hotspot detection on per-cell resource amounts per simulation step.

    Each step's ``hotspot_cells`` in the returned structure may be truncated to
    :data:`MAX_SERIALIZED_HOTSPOT_CELLS_PER_STEP` (highest |z_score|); centroids,
    mass totals, and persistence use the full hotspot set.
    """
    required = {"step", "position_x", "position_y", "amount"}
    if resource_positions.empty or not required.issubset(resource_positions.columns):
        return {}

    df = resource_positions.dropna(subset=["step", "position_x", "position_y", "amount"]).copy()
    if df.empty:
        return {}

    step_entries: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    n_hotspots_per_step: List[int] = []
    mass_per_step: List[float] = []
    centroids: List[Optional[List[float]]] = []
    for step, step_df in df.groupby("step", sort=True):
        hotspots, grid_meta = _hotspots_for_step(step_df, sigma_multiplier)
        cells_json, truncated = _hotspot_cells_for_export(
            hotspots, MAX_SERIALIZED_HOTSPOT_CELLS_PER_STEP
        )
        centroid = _weighted_centroid(hotspots)
        total_mass = sum(h["cell_amount"] for h in hotspots)
        row = {
            "step": int(step),
            "n_hotspot_cells": len(hotspots),
            "grid_mean_amount": grid_meta["mean"],
            "grid_std_amount": grid_meta["std"],
            "threshold": grid_meta["threshold"],
            "hotspot_centroid": centroid,
            "total_hotspot_amount": float(total_mass),
            "hotspot_cells": cells_json,
            "hotspot_cells_truncated": truncated,
        }
        step_entries.append((row, hotspots))
        n_hotspots_per_step.append(len(hotspots))
        mass_per_step.append(float(total_mass))
        centroids.append(centroid)

    # Movement: centroid displacement between consecutive steps when both have hotspots
    displacements: List[float] = []
    for i in range(1, len(centroids)):
        prev_c, c = centroids[i - 1], centroids[i]
        if prev_c is not None and c is not None:
            displacements.append(
                float(np.sqrt((c[0] - prev_c[0]) ** 2 + (c[1] - prev_c[1]) ** 2))
            )

    n_arr = np.array(n_hotspots_per_step, dtype=float)
    mass_arr = np.array(mass_per_step, dtype=float)

    # Per-cell persistence uses full hotspot sets, not the capped JSON lists.
    cell_counts: Dict[Tuple[Any, Any], int] = {}
    for _, hotspots in step_entries:
        for h in hotspots:
            key = _cell_key(h["position_x"], h["position_y"])
            cell_counts[key] = cell_counts.get(key, 0) + 1
    per_step = [row for row, _ in step_entries]
    n_steps = len(per_step)
    persistent_cells = [
        {
            "position_x": float(k[0]),
            "position_y": float(k[1]),
            "steps_as_hotspot": v,
            "persistence_ratio": float(v) / n_steps if n_steps else 0.0,
        }
        for k, v in sorted(cell_counts.items(), key=lambda x: -x[1])
    ]

    return {
        "threshold_sigma": float(sigma_multiplier),
        "n_steps": n_steps,
        "per_step": per_step,
        "centroid_displacements": displacements,
        "avg_centroid_displacement": float(np.mean(displacements)) if displacements else 0.0,
        "n_hotspot_cells_trend": calculate_trend(n_arr) if len(n_arr) > 1 else 0.0,
        "hotspot_mass_trend": calculate_trend(mass_arr) if len(mass_arr) > 1 else 0.0,
        "growth_decay_summary": {
            "n_hotspot_cells_first": int(n_hotspots_per_step[0]) if n_hotspots_per_step else 0,
            "n_hotspot_cells_last": int(n_hotspots_per_step[-1]) if n_hotspots_per_step else 0,
            "hotspot_mass_first": float(mass_per_step[0]) if mass_per_step else 0.0,
            "hotspot_mass_last": float(mass_per_step[-1]) if mass_per_step else 0.0,
        },
        "persistent_hotspot_cells": persistent_cells[:50],
    }


def compute_resource_hotspots(
    df: pd.DataFrame,
    spatial_resource_positions: Optional[pd.DataFrame] = None,
    *,
    hotspot_sigma: float = 2.0,
) -> Dict[str, Any]:
    """Analyze resource hotspot patterns using spatial grid data when available.

    When ``spatial_resource_positions`` is provided (e.g. from
    ``farm.analysis.spatial.data.process_spatial_data``), identifies cells whose
    per-step aggregated amount exceeds mean + ``hotspot_sigma`` * std across cells,
    tracks centroid motion and hotspot counts/mass over time, and returns coordinates.

    Falls back to global time-series concentration metrics from ``total_resources``
    when spatial columns are missing or the spatial frame is empty.

    Args:
        df: Resource time series (expects ``total_resources`` for fallback metrics).
        spatial_resource_positions: Optional DataFrame with columns step, position_x,
            position_y, amount (and optional resource_type).
        hotspot_sigma: Number of standard deviations above the per-step cell mean for
            the hotspot threshold (default 2).

    Returns:
        Empty dict when there is neither usable spatial data nor a ``total_resources``
        column. Otherwise includes ``mode`` (``spatial`` or ``timeseries_fallback``),
        optional ``spatial`` output, and legacy scalar keys when ``total_resources``
        is present.
    """
    if not math.isfinite(hotspot_sigma) or hotspot_sigma < 0:
        raise ValueError(
            f"hotspot_sigma must be a finite non-negative number, got {hotspot_sigma!r}"
        )

    spatial_block: Dict[str, Any] = {}
    if spatial_resource_positions is not None and not spatial_resource_positions.empty:
        spatial_block = _compute_spatial_resource_hotspots(
            spatial_resource_positions, sigma_multiplier=hotspot_sigma
        )

    has_timeseries = "total_resources" in df.columns

    if spatial_block:
        out: Dict[str, Any] = {
            "mode": "spatial",
            "spatial": spatial_block,
        }
        if has_timeseries:
            out.update(_timeseries_hotspot_metrics(df))
        return out

    if not has_timeseries:
        return {}

    return {
        "mode": "timeseries_fallback",
        "spatial": None,
        **_timeseries_hotspot_metrics(df),
    }

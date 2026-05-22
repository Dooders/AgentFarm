"""State-aware action re-weighter consuming Chromosome B multiplier/threshold genes.

The decision module combines policy probabilities with optional per-action
weights via multiplicative composition in
:meth:`farm.core.decision.decision.DecisionModule.decide_action`.
Until now, the only signal it received was the static
``core.actions[i].weight`` — meaning the multiplier and threshold genes that
live on :class:`farm.core.decision.config.DecisionConfig` had no consumer.

This module bridges that gap: given the current agent state, it scales the
base per-action weights using the eight state-aware multiplier genes plus
their three companion threshold genes:

- ``move_mult_no_resources``       (active when no resources are nearby)
- ``gather_mult_low_resources``    (active when the agent is below half resources)
- ``share_mult_wealthy``           (active when the agent has plenty of resources)
- ``share_mult_poor``              (active when the agent is below the wealth threshold)
- ``attack_mult_desperate``        (active when starvation risk exceeds
  ``attack_starvation_threshold``)
- ``attack_mult_stable``           (active when combat health is above
  ``attack_defense_threshold``)
- ``reproduce_mult_wealthy``       (active when resources are above
  ``reproduce_resource_threshold``)
- ``reproduce_mult_poor``          (active when below that same threshold)

The function is intentionally a pure helper so it can be unit-tested without
constructing a full agent / decision pipeline.  Heuristics are best-effort:
when an attribute is missing the corresponding multiplier is skipped so the
helper degrades gracefully on unusual agent shapes.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "ActionStateSignals",
    "compute_action_weights",
    "extract_signals",
]


class ActionStateSignals:
    """Plain-data container for the state signals consumed by the re-weighter.

    Values are normalized to the unit interval where applicable so the
    multiplier-gating thresholds (which all live in ``[0, 1]``) can be
    interpreted consistently.

    Attributes:
        resource_ratio: ``resource_level / max_resource_level`` clipped to
            ``[0, 1]``.  ``None`` when the max is unknown.
        health_ratio: ``current_health / starting_health`` clipped to
            ``[0, 1]``.  ``None`` when the agent has no combat component.
        starvation_risk: ``starvation_counter / starvation_threshold`` clipped
            to ``[0, 1]``.  Equals ``0.0`` for agents without a resource
            component.
        nearby_resources: ``True`` when the agent perceives at least one
            resource within its perception radius.  ``None`` when the
            component is unavailable.
    """

    __slots__ = (
        "resource_ratio",
        "health_ratio",
        "starvation_risk",
        "nearby_resources",
    )

    def __init__(
        self,
        *,
        resource_ratio: Optional[float] = None,
        health_ratio: Optional[float] = None,
        starvation_risk: float = 0.0,
        nearby_resources: Optional[bool] = None,
    ) -> None:
        self.resource_ratio = resource_ratio
        self.health_ratio = health_ratio
        self.starvation_risk = float(np.clip(starvation_risk, 0.0, 1.0))
        self.nearby_resources = nearby_resources


def _clip_ratio(numerator: float, denominator: float) -> Optional[float]:
    """Return ``numerator / denominator`` clipped to ``[0, 1]`` (``None`` on bad input)."""
    if denominator is None or denominator <= 0.0:
        return None
    try:
        return float(np.clip(float(numerator) / float(denominator), 0.0, 1.0))
    except (TypeError, ValueError):
        return None


def extract_signals(agent: Any) -> ActionStateSignals:
    """Best-effort extraction of state signals from an :class:`AgentCore`-shaped object.

    Designed to be tolerant of partial agents in tests.  Missing attributes
    map to ``None`` (or, for ``starvation_risk``, ``0.0``).
    """
    resource_level = getattr(agent, "resource_level", None)
    max_resource_level = None

    config = getattr(agent, "config", None)
    if config is not None:
        reward_cfg = getattr(config, "reward", None)
        if reward_cfg is not None:
            max_resource_level = getattr(reward_cfg, "max_resource_level", None)
    if max_resource_level is None:
        # Fall back to environment-level config when reward config is absent.
        env = getattr(agent, "environment", None)
        env_config = getattr(env, "config", None) if env is not None else None
        max_resource_level = getattr(env_config, "max_resource_amount", None) if env_config else None

    resource_ratio = (
        _clip_ratio(float(resource_level), float(max_resource_level))
        if resource_level is not None and max_resource_level is not None
        else None
    )

    starting_health = getattr(agent, "starting_health", None)
    current_health = getattr(agent, "current_health", None)
    health_ratio = (
        _clip_ratio(float(current_health), float(starting_health))
        if current_health is not None and starting_health is not None
        else None
    )

    starvation_risk = 0.0
    get_component = getattr(agent, "get_component", None)
    resource_comp = get_component("resource") if callable(get_component) else None
    if resource_comp is not None:
        threshold = getattr(getattr(resource_comp, "config", None), "starvation_threshold", None)
        counter = getattr(resource_comp, "starvation_counter", 0)
        if threshold is not None and threshold > 0:
            starvation_risk = float(np.clip(float(counter) / float(threshold), 0.0, 1.0))

    nearby_resources: Optional[bool] = None
    perception_comp = get_component("perception") if callable(get_component) else None
    if perception_comp is not None and hasattr(perception_comp, "_get_cached_spatial_query"):
        try:
            radius = getattr(getattr(perception_comp, "config", None), "perception_radius", None)
            if radius is not None:
                nearby = perception_comp._get_cached_spatial_query(int(radius), ["resources"])
                nearby_resources = bool(nearby.get("resources") if isinstance(nearby, dict) else nearby)
        except Exception:
            nearby_resources = None

    return ActionStateSignals(
        resource_ratio=resource_ratio,
        health_ratio=health_ratio,
        starvation_risk=starvation_risk,
        nearby_resources=nearby_resources,
    )


def _multiplier(value: Optional[float], default: float = 1.0) -> float:
    """Coerce a config value into a non-negative multiplier (``default`` on bad input)."""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(f) or f < 0.0:
        return default
    return f


def compute_action_weights(
    base_weights: Sequence[float],
    action_names: Sequence[str],
    decision_config: Any,
    signals: ActionStateSignals,
    enabled_actions: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Return a per-action weight vector with state-aware multipliers applied.

    Args:
        base_weights: Per-action base weights (already populated from
            :class:`farm.core.action.Action` ``weight`` fields, which
            themselves derive from Chromosome B action-weight genes via
            :meth:`AgentCore.refresh_action_weights_from_decision_config`).
        action_names: ``len(base_weights)``-aligned names (lowercase action
            identifiers).
        decision_config: The agent's :class:`DecisionConfig` (carries the
            multiplier and threshold genes).
        signals: Pre-computed :class:`ActionStateSignals`.
        enabled_actions: Optional iterable of indices that are currently
            allowed.  Disabled actions get a weight of ``0.0``.

    Returns:
        Float-64 numpy array of length ``len(base_weights)`` with normalized
        weights summing to ``1.0`` (or uniform across enabled actions when the
        scaled weights collapse to zero).
    """
    if len(base_weights) != len(action_names):
        raise ValueError(
            "base_weights and action_names must be the same length; "
            f"got {len(base_weights)} vs {len(action_names)}."
        )

    weights = np.asarray(base_weights, dtype=np.float64).copy()
    if weights.size == 0:
        return weights

    name_to_idx = {name: idx for idx, name in enumerate(action_names)}

    def _scale(action_name: str, factor: float) -> None:
        idx = name_to_idx.get(action_name)
        if idx is None:
            return
        weights[idx] = weights[idx] * factor

    # ── move ────────────────────────────────────────────────────────────────
    if signals.nearby_resources is False:
        _scale(
            "move",
            _multiplier(getattr(decision_config, "move_mult_no_resources", None)),
        )

    # ── gather ──────────────────────────────────────────────────────────────
    if signals.resource_ratio is not None and signals.resource_ratio < 0.5:
        _scale(
            "gather",
            _multiplier(getattr(decision_config, "gather_mult_low_resources", None)),
        )

    # ── share ───────────────────────────────────────────────────────────────
    if signals.resource_ratio is not None:
        if signals.resource_ratio >= 0.7:
            _scale("share", _multiplier(getattr(decision_config, "share_mult_wealthy", None)))
        elif signals.resource_ratio < 0.3:
            _scale("share", _multiplier(getattr(decision_config, "share_mult_poor", None)))

    # ── attack ──────────────────────────────────────────────────────────────
    starv_threshold = float(
        np.clip(
            float(getattr(decision_config, "attack_starvation_threshold", 0.5)),
            0.0,
            1.0,
        )
    )
    defense_threshold = float(
        np.clip(
            float(getattr(decision_config, "attack_defense_threshold", 0.3)),
            0.0,
            1.0,
        )
    )
    if signals.starvation_risk >= starv_threshold:
        _scale(
            "attack",
            _multiplier(getattr(decision_config, "attack_mult_desperate", None)),
        )
    if signals.health_ratio is not None and signals.health_ratio >= (1.0 - defense_threshold):
        _scale(
            "attack",
            _multiplier(getattr(decision_config, "attack_mult_stable", None)),
        )

    # ── reproduce ───────────────────────────────────────────────────────────
    repro_threshold = float(
        np.clip(
            float(getattr(decision_config, "reproduce_resource_threshold", 0.7)),
            0.0,
            1.0,
        )
    )
    if signals.resource_ratio is not None:
        if signals.resource_ratio >= repro_threshold:
            _scale(
                "reproduce",
                _multiplier(getattr(decision_config, "reproduce_mult_wealthy", None)),
            )
        else:
            _scale(
                "reproduce",
                _multiplier(getattr(decision_config, "reproduce_mult_poor", None)),
            )

    # ── enabled mask ────────────────────────────────────────────────────────
    enabled_list: Optional[List[int]] = None
    if enabled_actions is not None:
        enabled_list = [int(i) for i in enabled_actions]
        enabled_set = set(enabled_list)
        for idx in range(weights.size):
            if idx not in enabled_set:
                weights[idx] = 0.0

    # Guard against negative or non-finite values that may sneak in via
    # malformed configs; mutate_chromosome already keeps genes in range,
    # but defensive normalization keeps the helper robust to future edits.
    weights = np.where(np.isfinite(weights) & (weights >= 0.0), weights, 0.0)

    total = weights.sum()
    if total > 0.0:
        return weights / total

    # All-zero fallback: uniform across enabled (or all) actions.
    if enabled_list is not None:
        if enabled_list:
            uniform = np.zeros_like(weights)
            uniform[enabled_list] = 1.0 / len(enabled_list)
            return uniform
    return np.full_like(weights, 1.0 / weights.size)

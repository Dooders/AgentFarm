"""Helpers for policy-state inheritance across generations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from farm.utils.logging import get_logger

logger = get_logger(__name__)


def _policy_state_is_compatible(
    parent_policy_state: Dict[str, Any], child_policy: Any
) -> bool:
    """Return whether parent policy weights can be loaded into the child policy."""
    if child_policy is None or not hasattr(child_policy, "state_dict"):
        return False

    try:
        child_state = child_policy.state_dict()
    except Exception:
        return False

    parent_keys = set(parent_policy_state.keys())
    child_keys = set(child_state.keys())
    if parent_keys != child_keys:
        return False

    for key, parent_value in parent_policy_state.items():
        child_value = child_state[key]
        parent_shape = getattr(parent_value, "shape", None)
        child_shape = getattr(child_value, "shape", None)
        if parent_shape is not None and child_shape is not None and parent_shape != child_shape:
            return False
    return True


def apply_lamarckian_policy_warmstart(parent: Any, offspring: Any) -> bool:
    """Warm-start ``offspring`` policy weights from ``parent``.

    Returns ``True`` when a compatible policy payload is applied, otherwise
    ``False`` and the child continues with a cold start.
    """
    parent_behavior = getattr(parent, "behavior", None)
    child_behavior = getattr(offspring, "behavior", None)
    parent_module = getattr(parent_behavior, "decision_module", None)
    child_module = getattr(child_behavior, "decision_module", None)
    if parent_module is None or child_module is None:
        return False

    parent_algorithm = getattr(parent_module, "algorithm", None)
    child_algorithm = getattr(child_module, "algorithm", None)
    if parent_algorithm is None or child_algorithm is None:
        return False

    get_model_state = getattr(parent_algorithm, "get_model_state", None)
    load_model_state = getattr(child_algorithm, "load_model_state", None)
    if not callable(get_model_state) or not callable(load_model_state):
        return False

    try:
        parent_state = get_model_state()
    except Exception:
        logger.exception(
            "lamarckian_warmstart_parent_state_failed",
            parent_id=getattr(parent, "agent_id", None),
            offspring_id=getattr(offspring, "agent_id", None),
        )
        return False

    if not isinstance(parent_state, dict):
        return False
    policy_state = parent_state.get("policy_state_dict")
    if not isinstance(policy_state, dict) or not policy_state:
        return False

    child_policy = getattr(child_algorithm, "policy", None)
    if not _policy_state_is_compatible(policy_state, child_policy):
        return False

    payload = {"policy_state_dict": deepcopy(policy_state)}
    try:
        load_model_state(payload)
    except Exception:
        logger.exception(
            "lamarckian_warmstart_load_failed",
            parent_id=getattr(parent, "agent_id", None),
            offspring_id=getattr(offspring, "agent_id", None),
        )
        return False
    return True

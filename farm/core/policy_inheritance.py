"""Helpers for policy-state inheritance across generations.

This module provides infrastructure for transferring learned policy state from
parent agents to their offspring. The primary entry point is
:func:`apply_lamarckian_policy_warmstart`, which copies policy weights into a
child agent.

For richer inheritance payloads (P3 variant from docs/design/inherited_payload_design.md),
the replay buffer transfer API is available in
:class:`farm.core.decision.algorithms.rl_base.PrioritizedReplayBuffer`:

* :meth:`get_transfer_slice(max_size)` extracts a bounded, deterministic
  slice of the parent's replay buffer.
* :meth:`load_transfer_slice(slice_data)` loads the slice into the child's buffer.

These methods are automatically used by
:class:`farm.core.decision.algorithms.tianshou.TianshouWrapper` when
``get_model_state(include_replay_buffer=True, replay_buffer_limit=N)`` is called,
and are integrated with the existing ``load_model_state`` machinery.

P2–P4 variant hooks are also provided here (see the variant ladder in
``docs/design/inherited_payload_design.md``):

* :func:`apply_p2_policy_warmstart` — weights + plasticity damping.
* :func:`apply_p3_policy_warmstart` — weights + optimizer state + bounded replay slice.
* :func:`apply_p4_policy_warmstart` — gated/blended transfer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from farm.utils.logging import get_logger

logger = get_logger(__name__)


# Skip-reason constants emitted by warm-start helpers.
# Keeping them as module-level strings (instead of an Enum) lets downstream
# JSON metadata round-trip them without a custom encoder; the comparator
# script and tests rely on the literal values.
WARMSTART_REASON_NO_DECISION_MODULE = "no_decision_module"
WARMSTART_REASON_NO_ALGORITHM = "no_algorithm"
WARMSTART_REASON_NO_WARMSTART_API = "no_warmstart_api"
WARMSTART_REASON_PARENT_STATE_FAILED = "parent_state_failed"
WARMSTART_REASON_NO_POLICY_STATE = "no_policy_state"
WARMSTART_REASON_INCOMPATIBLE_STATE = "incompatible_state"
WARMSTART_REASON_LOAD_FAILED = "load_failed"

# P4-specific: parent did not clear the fitness gate.
WARMSTART_REASON_GATE_NOT_CLEARED = "gate_not_cleared"

# Default plasticity damping factor applied by the P2 variant.
# The child's inherited LR and ε are scaled by this factor so the
# copied weights are not immediately overwritten by noisy early updates.
P2_PLASTICITY_DAMPING = 0.5

# Replay-buffer slice size cap used by the P3 variant.
P3_REPLAY_BUFFER_LIMIT = 256

# Blend coefficient used by the P4 variant:
#   θ_child = P4_BLEND_ALPHA * θ_parent + (1 - P4_BLEND_ALPHA) * θ_init
P4_BLEND_ALPHA = 0.5

# P4 gate: minimum resource level (as a fraction of initial resources, or
# absolute level depending on the environment) the parent must hold to pass
# the fitness gate. Uses resource_level attribute when available.
P4_FITNESS_GATE_MIN_RESOURCES = 1.0


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


def apply_lamarckian_policy_warmstart(parent: Any, offspring: Any) -> Optional[str]:
    """Warm-start ``offspring`` policy weights from ``parent``.

    Returns ``None`` when a compatible policy payload was applied; otherwise
    returns a stable reason string drawn from ``WARMSTART_REASON_*`` so callers
    can attribute skips. The child continues with a cold start in any
    skip case.

    The parent's ``policy.state_dict()`` is passed through directly because
    ``load_state_dict`` performs its own copy into the child's parameters.
    Avoiding a defensive deepcopy keeps the per-reproduction overhead small
    enough to run on long simulations without a noticeable wall-clock
    penalty.
    """
    parent_behavior = getattr(parent, "behavior", None)
    child_behavior = getattr(offspring, "behavior", None)
    parent_module = getattr(parent_behavior, "decision_module", None)
    child_module = getattr(child_behavior, "decision_module", None)
    if parent_module is None or child_module is None:
        return WARMSTART_REASON_NO_DECISION_MODULE

    parent_algorithm = getattr(parent_module, "algorithm", None)
    child_algorithm = getattr(child_module, "algorithm", None)
    if parent_algorithm is None or child_algorithm is None:
        return WARMSTART_REASON_NO_ALGORITHM

    get_model_state = getattr(parent_algorithm, "get_model_state", None)
    load_model_state = getattr(child_algorithm, "load_model_state", None)
    if not callable(get_model_state) or not callable(load_model_state):
        return WARMSTART_REASON_NO_WARMSTART_API

    try:
        parent_state = get_model_state()
    except Exception:
        logger.exception(
            "lamarckian_warmstart_parent_state_failed",
            parent_id=getattr(parent, "agent_id", None),
            offspring_id=getattr(offspring, "agent_id", None),
        )
        return WARMSTART_REASON_PARENT_STATE_FAILED

    if not isinstance(parent_state, dict):
        return WARMSTART_REASON_NO_POLICY_STATE
    policy_state = parent_state.get("policy_state_dict")
    if not isinstance(policy_state, dict) or not policy_state:
        return WARMSTART_REASON_NO_POLICY_STATE

    child_policy = getattr(child_algorithm, "policy", None)
    if not _policy_state_is_compatible(policy_state, child_policy):
        return WARMSTART_REASON_INCOMPATIBLE_STATE

    try:
        load_model_state({"policy_state_dict": policy_state})
    except Exception:
        logger.exception(
            "lamarckian_warmstart_load_failed",
            parent_id=getattr(parent, "agent_id", None),
            offspring_id=getattr(offspring, "agent_id", None),
        )
        return WARMSTART_REASON_LOAD_FAILED
    return None


def _get_parent_algorithm_state(
    parent: Any,
    offspring: Any,
    *,
    include_optimizer_state: bool = False,
    include_replay_buffer: bool = False,
    replay_buffer_limit: Optional[int] = None,
    include_plasticity_state: bool = False,
) -> tuple:
    """Extract the parent's algorithm objects and model state.

    Returns a 4-tuple ``(parent_algorithm, child_algorithm, policy_state, skip_reason)``.
    If ``skip_reason`` is not ``None`` the caller should propagate it immediately;
    the other three elements will be ``None``.
    """
    parent_behavior = getattr(parent, "behavior", None)
    child_behavior = getattr(offspring, "behavior", None)
    parent_module = getattr(parent_behavior, "decision_module", None)
    child_module = getattr(child_behavior, "decision_module", None)
    if parent_module is None or child_module is None:
        return None, None, None, WARMSTART_REASON_NO_DECISION_MODULE

    parent_algorithm = getattr(parent_module, "algorithm", None)
    child_algorithm = getattr(child_module, "algorithm", None)
    if parent_algorithm is None or child_algorithm is None:
        return None, None, None, WARMSTART_REASON_NO_ALGORITHM

    get_model_state = getattr(parent_algorithm, "get_model_state", None)
    load_model_state = getattr(child_algorithm, "load_model_state", None)
    if not callable(get_model_state) or not callable(load_model_state):
        return None, None, None, WARMSTART_REASON_NO_WARMSTART_API

    try:
        parent_state = get_model_state(
            include_optimizer_state=include_optimizer_state,
            include_replay_buffer=include_replay_buffer,
            replay_buffer_limit=replay_buffer_limit,
            include_plasticity_state=include_plasticity_state,
        )
    except TypeError:
        # Fallback: the implementation does not accept extended kwargs (e.g. in
        # tests or older algorithm versions).  Call without arguments and accept
        # the core policy state without the optional payloads.
        try:
            parent_state = get_model_state()
        except Exception:
            logger.exception(
                "warmstart_parent_state_failed",
                parent_id=getattr(parent, "agent_id", None),
                offspring_id=getattr(offspring, "agent_id", None),
            )
            return None, None, None, WARMSTART_REASON_PARENT_STATE_FAILED
    except Exception:
        logger.exception(
            "warmstart_parent_state_failed",
            parent_id=getattr(parent, "agent_id", None),
            offspring_id=getattr(offspring, "agent_id", None),
        )
        return None, None, None, WARMSTART_REASON_PARENT_STATE_FAILED

    if not isinstance(parent_state, dict):
        return None, None, None, WARMSTART_REASON_NO_POLICY_STATE

    policy_state = parent_state.get("policy_state_dict")
    if not isinstance(policy_state, dict) or not policy_state:
        return None, None, None, WARMSTART_REASON_NO_POLICY_STATE

    child_policy = getattr(child_algorithm, "policy", None)
    if not _policy_state_is_compatible(policy_state, child_policy):
        return None, None, None, WARMSTART_REASON_INCOMPATIBLE_STATE

    return parent_algorithm, child_algorithm, parent_state, None


def apply_p2_policy_warmstart(
    parent: Any,
    offspring: Any,
    *,
    plasticity_damping: float = P2_PLASTICITY_DAMPING,
) -> Optional[str]:
    """P2: Warm-start offspring with parent weights plus plasticity damping.

    Copies the parent policy weights (same as Lamarckian / P1), and
    additionally reduces the child's initial learning rate and exploration
    rate (ε) by ``plasticity_damping`` so the inherited weights are not
    immediately overwritten by noisy early updates.  The parent's
    ``step_count`` is also carried over so the child's exploration schedule
    continues from the parent's position rather than resetting to zero.

    Returns ``None`` on success; returns a ``WARMSTART_REASON_*`` string on
    any skip so callers can attribute the outcome through telemetry.
    """
    parent_algorithm, child_algorithm, parent_state, skip_reason = (
        _get_parent_algorithm_state(parent, offspring, include_plasticity_state=True)
    )
    if skip_reason is not None:
        return skip_reason

    policy_state = parent_state["policy_state_dict"]
    plasticity_state = parent_state.get("plasticity_state", {})
    step_count = parent_state.get("step_count")

    # Build the child payload: weights + damped plasticity parameters.
    child_payload: Dict[str, Any] = {"policy_state_dict": policy_state}

    if step_count is not None:
        child_payload["step_count"] = step_count

    if plasticity_state:
        damped: Dict[str, Any] = {}
        if "learning_rate" in plasticity_state:
            damped["learning_rate"] = float(plasticity_state["learning_rate"]) * plasticity_damping
        if "learning_rates" in plasticity_state:
            damped["learning_rates"] = {
                k: float(v) * plasticity_damping
                for k, v in plasticity_state["learning_rates"].items()
            }
        if "epsilon" in plasticity_state:
            damped["epsilon"] = float(plasticity_state["epsilon"]) * plasticity_damping
        if "eps_test" in plasticity_state:
            damped["eps_test"] = float(plasticity_state["eps_test"]) * plasticity_damping
        if "train_mode" in plasticity_state:
            damped["train_mode"] = plasticity_state["train_mode"]
        if damped:
            child_payload["plasticity_state"] = damped

    load_model_state = getattr(child_algorithm, "load_model_state", None)
    try:
        load_model_state(child_payload)
    except Exception:
        logger.exception(
            "p2_warmstart_load_failed",
            parent_id=getattr(parent, "agent_id", None),
            offspring_id=getattr(offspring, "agent_id", None),
        )
        return WARMSTART_REASON_LOAD_FAILED

    logger.debug(
        "p2_warmstart_applied",
        parent_id=getattr(parent, "agent_id", None),
        offspring_id=getattr(offspring, "agent_id", None),
        plasticity_damping=plasticity_damping,
    )
    return None


def apply_p3_policy_warmstart(
    parent: Any,
    offspring: Any,
    *,
    replay_buffer_limit: int = P3_REPLAY_BUFFER_LIMIT,
) -> Optional[str]:
    """P3: Warm-start offspring with weights + optimizer state + replay slice.

    Copies the parent's policy weights, optimizer state (Adam moments), and
    a bounded slice of the parent's replay buffer so the offspring continues
    the parent's learning trajectory instead of restarting from scratch.

    Returns ``None`` on success; returns a ``WARMSTART_REASON_*`` string on
    any skip.
    """
    parent_algorithm, child_algorithm, parent_state, skip_reason = (
        _get_parent_algorithm_state(
            parent,
            offspring,
            include_optimizer_state=True,
            include_replay_buffer=True,
            replay_buffer_limit=replay_buffer_limit,
            include_plasticity_state=False,
        )
    )
    if skip_reason is not None:
        return skip_reason

    policy_state = parent_state["policy_state_dict"]

    # Build the child payload: weights + optional continuation machinery.
    child_payload: Dict[str, Any] = {"policy_state_dict": policy_state}

    step_count = parent_state.get("step_count")
    if step_count is not None:
        child_payload["step_count"] = step_count

    optimizer_state = parent_state.get("optimizer_state")
    if isinstance(optimizer_state, dict) and optimizer_state:
        child_payload["optimizer_state"] = optimizer_state

    replay_buffer_state = parent_state.get("replay_buffer_state")
    if isinstance(replay_buffer_state, dict) and replay_buffer_state:
        child_payload["replay_buffer_state"] = replay_buffer_state

    load_model_state = getattr(child_algorithm, "load_model_state", None)
    try:
        load_model_state(child_payload)
    except Exception:
        logger.exception(
            "p3_warmstart_load_failed",
            parent_id=getattr(parent, "agent_id", None),
            offspring_id=getattr(offspring, "agent_id", None),
        )
        return WARMSTART_REASON_LOAD_FAILED

    logger.debug(
        "p3_warmstart_applied",
        parent_id=getattr(parent, "agent_id", None),
        offspring_id=getattr(offspring, "agent_id", None),
        replay_buffer_limit=replay_buffer_limit,
        has_optimizer_state=bool(optimizer_state),
        has_replay_buffer=bool(replay_buffer_state),
    )
    return None


def apply_p4_policy_warmstart(
    parent: Any,
    offspring: Any,
    *,
    blend_alpha: float = P4_BLEND_ALPHA,
    fitness_gate_min_resources: float = P4_FITNESS_GATE_MIN_RESOURCES,
) -> Optional[str]:
    """P4: Gated and blended policy transfer.

    Applies transfer only when the parent clears a resource-based fitness gate.
    When the gate is cleared, weights are blended as::

        θ_child = blend_alpha * θ_parent + (1 - blend_alpha) * θ_init

    where ``θ_init`` is the offspring's current (freshly initialised) weights.
    This soft warm-start bounds local-niche lock-in by not fully committing to
    the parent's learned representation.

    Returns ``None`` on success; returns a ``WARMSTART_REASON_*`` string on any
    skip, including ``WARMSTART_REASON_GATE_NOT_CLEARED`` when the parent does
    not meet the fitness threshold.
    """
    # Fitness gate: check parent resource level.
    parent_resource_level = getattr(parent, "resource_level", None)
    if parent_resource_level is not None:
        if float(parent_resource_level) < fitness_gate_min_resources:
            logger.debug(
                "p4_warmstart_gate_not_cleared",
                parent_id=getattr(parent, "agent_id", None),
                resource_level=parent_resource_level,
                threshold=fitness_gate_min_resources,
            )
            return WARMSTART_REASON_GATE_NOT_CLEARED

    parent_algorithm, child_algorithm, parent_state, skip_reason = (
        _get_parent_algorithm_state(parent, offspring)
    )
    if skip_reason is not None:
        return skip_reason

    parent_policy_state = parent_state["policy_state_dict"]

    # Capture the child's current (init) weights before blending.
    child_policy = getattr(child_algorithm, "policy", None)
    try:
        child_init_state = child_policy.state_dict() if child_policy is not None else {}
    except Exception:
        child_init_state = {}

    # Blend: θ_child = α*θ_parent + (1-α)*θ_init
    blended_state: Dict[str, Any] = {}
    for key, parent_value in parent_policy_state.items():
        child_value = child_init_state.get(key)
        if child_value is None:
            blended_state[key] = parent_value
            continue
        try:
            import torch  # noqa: PLC0415

            if isinstance(parent_value, torch.Tensor) and isinstance(child_value, torch.Tensor):
                blended_state[key] = (
                    blend_alpha * parent_value + (1.0 - blend_alpha) * child_value
                ).detach()
            else:
                blended_state[key] = parent_value
        except Exception:
            blended_state[key] = parent_value

    step_count = parent_state.get("step_count")
    child_payload: Dict[str, Any] = {"policy_state_dict": blended_state}
    if step_count is not None:
        child_payload["step_count"] = step_count

    load_model_state = getattr(child_algorithm, "load_model_state", None)
    try:
        load_model_state(child_payload)
    except Exception:
        logger.exception(
            "p4_warmstart_load_failed",
            parent_id=getattr(parent, "agent_id", None),
            offspring_id=getattr(offspring, "agent_id", None),
        )
        return WARMSTART_REASON_LOAD_FAILED

    logger.debug(
        "p4_warmstart_applied",
        parent_id=getattr(parent, "agent_id", None),
        offspring_id=getattr(offspring, "agent_id", None),
        blend_alpha=blend_alpha,
    )
    return None

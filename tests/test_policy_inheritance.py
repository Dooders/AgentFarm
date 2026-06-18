"""Direct unit tests for ``farm.core.policy_inheritance`` helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from farm.core.inheritance_telemetry import InheritanceTelemetry
from farm.core.policy_inheritance import (
    WARMSTART_REASON_EXTENDED_STATE_UNSUPPORTED,
    WARMSTART_REASON_GATE_NOT_CLEARED,
    WARMSTART_REASON_INCOMPATIBLE_STATE,
    WARMSTART_REASON_LOAD_FAILED,
    WARMSTART_REASON_NO_ALGORITHM,
    WARMSTART_REASON_NO_DECISION_MODULE,
    WARMSTART_REASON_NO_POLICY_STATE,
    WARMSTART_REASON_NO_WARMSTART_API,
    WARMSTART_REASON_PARENT_STATE_FAILED,
    _policy_state_is_compatible,
    apply_lamarckian_policy_warmstart,
    apply_p2_policy_warmstart,
    apply_p3_policy_warmstart,
    apply_p4_policy_warmstart,
)


def _make_pair(parent_state, child_state, *, child_load=None, child_state_fn=None):
    """Build minimal parent/offspring objects exposing the warm-start API.

    ``parent_state`` is the dict ``parent.algorithm.get_model_state`` returns;
    ``child_state`` is the dict ``child.algorithm.policy.state_dict`` returns.
    Pass ``child_load`` to control ``load_model_state`` (defaults to a Mock).
    """
    parent = SimpleNamespace(
        agent_id="parent",
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(
                algorithm=SimpleNamespace(
                    get_model_state=lambda: parent_state,
                ),
            ),
        ),
    )
    child_policy = SimpleNamespace(
        state_dict=child_state_fn or (lambda: child_state),
    )
    offspring = SimpleNamespace(
        agent_id="child",
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(
                algorithm=SimpleNamespace(
                    policy=child_policy,
                    load_model_state=child_load if child_load is not None else Mock(),
                ),
            ),
        ),
    )
    return parent, offspring


def test_compatible_state_passes_keys_and_shapes():
    parent_state = {"w": SimpleNamespace(shape=(2, 2)), "b": SimpleNamespace(shape=(2,))}
    child_state = {"w": SimpleNamespace(shape=(2, 2)), "b": SimpleNamespace(shape=(2,))}
    child_policy = SimpleNamespace(state_dict=lambda: child_state)
    assert _policy_state_is_compatible(parent_state, child_policy) is True


def test_incompatible_keys_fail_compatibility():
    parent_state = {"w_p": SimpleNamespace(shape=(2, 2))}
    child_state = {"w_c": SimpleNamespace(shape=(2, 2))}
    child_policy = SimpleNamespace(state_dict=lambda: child_state)
    assert _policy_state_is_compatible(parent_state, child_policy) is False


def test_incompatible_shapes_fail_compatibility():
    parent_state = {"w": SimpleNamespace(shape=(4, 4))}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    child_policy = SimpleNamespace(state_dict=lambda: child_state)
    assert _policy_state_is_compatible(parent_state, child_policy) is False


def test_compatibility_handles_state_dict_failure():
    def _raises():
        raise RuntimeError("boom")

    child_policy = SimpleNamespace(state_dict=_raises)
    assert _policy_state_is_compatible({}, child_policy) is False


def test_apply_returns_none_on_success_and_calls_load():
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_pair(parent_state, child_state, child_load=load)

    reason = apply_lamarckian_policy_warmstart(parent, offspring)

    assert reason is None
    load.assert_called_once()
    payload = load.call_args.args[0]
    assert payload["policy_state_dict"] is parent_state["policy_state_dict"], (
        "apply_lamarckian_policy_warmstart must pass the parent policy state "
        "through without an intermediate copy"
    )


def test_apply_skips_when_decision_module_missing():
    parent = SimpleNamespace(behavior=SimpleNamespace(decision_module=None))
    offspring = SimpleNamespace(behavior=SimpleNamespace(decision_module=None))
    assert (
        apply_lamarckian_policy_warmstart(parent, offspring)
        == WARMSTART_REASON_NO_DECISION_MODULE
    )


def test_apply_skips_when_algorithm_missing():
    parent = SimpleNamespace(
        behavior=SimpleNamespace(decision_module=SimpleNamespace(algorithm=None))
    )
    offspring = SimpleNamespace(
        behavior=SimpleNamespace(decision_module=SimpleNamespace(algorithm=None))
    )
    assert (
        apply_lamarckian_policy_warmstart(parent, offspring)
        == WARMSTART_REASON_NO_ALGORITHM
    )


def test_apply_skips_when_warmstart_api_missing():
    parent = SimpleNamespace(
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(algorithm=SimpleNamespace())
        )
    )
    offspring = SimpleNamespace(
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(algorithm=SimpleNamespace())
        )
    )
    assert (
        apply_lamarckian_policy_warmstart(parent, offspring)
        == WARMSTART_REASON_NO_WARMSTART_API
    )


def test_apply_skips_when_parent_state_returns_non_dict():
    parent_state = "not-a-dict"
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    parent, offspring = _make_pair(parent_state, child_state)
    assert (
        apply_lamarckian_policy_warmstart(parent, offspring)
        == WARMSTART_REASON_NO_POLICY_STATE
    )


def test_apply_skips_when_policy_state_dict_missing():
    parent_state = {"step_count": 10}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    parent, offspring = _make_pair(parent_state, child_state)
    assert (
        apply_lamarckian_policy_warmstart(parent, offspring)
        == WARMSTART_REASON_NO_POLICY_STATE
    )


def test_apply_skips_on_shape_mismatch():
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(4, 4))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    parent, offspring = _make_pair(parent_state, child_state)
    assert (
        apply_lamarckian_policy_warmstart(parent, offspring)
        == WARMSTART_REASON_INCOMPATIBLE_STATE
    )


def test_apply_skips_when_get_model_state_raises():
    def _raises():
        raise RuntimeError("network gone")

    parent = SimpleNamespace(
        agent_id="parent",
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(
                algorithm=SimpleNamespace(get_model_state=_raises),
            )
        ),
    )
    offspring = SimpleNamespace(
        agent_id="child",
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(
                algorithm=SimpleNamespace(
                    policy=SimpleNamespace(state_dict=lambda: {}),
                    load_model_state=Mock(),
                ),
            )
        ),
    )
    assert (
        apply_lamarckian_policy_warmstart(parent, offspring)
        == WARMSTART_REASON_PARENT_STATE_FAILED
    )


def test_apply_returns_load_failed_when_load_raises():
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}

    def _load(_payload):
        raise RuntimeError("load failed")

    parent, offspring = _make_pair(parent_state, child_state, child_load=_load)
    assert (
        apply_lamarckian_policy_warmstart(parent, offspring)
        == WARMSTART_REASON_LOAD_FAILED
    )


def test_inheritance_telemetry_records_applied_and_skipped():
    telemetry = InheritanceTelemetry()
    telemetry.record_applied()
    telemetry.record_applied()
    telemetry.record_skipped(WARMSTART_REASON_INCOMPATIBLE_STATE)
    telemetry.record_skipped(WARMSTART_REASON_INCOMPATIBLE_STATE)
    telemetry.record_skipped(WARMSTART_REASON_NO_POLICY_STATE)

    payload = telemetry.to_dict()
    assert payload["warmstart_applied"] == 2
    assert payload["warmstart_skipped"] == 3
    assert payload["warmstart_skipped_reasons"] == {
        WARMSTART_REASON_INCOMPATIBLE_STATE: 2,
        WARMSTART_REASON_NO_POLICY_STATE: 1,
    }
    # Coverage = applied / (applied + skipped); no gate skips => gate hit-rate 1.0.
    assert payload["warmstart_coverage"] == 2 / 5
    assert payload["gate_not_cleared"] == 0
    assert payload["gate_hit_rate"] == 1.0
    assert payload["blend_alpha"] is None


def test_inheritance_telemetry_empty_metrics_are_none():
    payload = InheritanceTelemetry().to_dict()
    assert payload["warmstart_coverage"] is None
    assert payload["gate_hit_rate"] is None
    assert payload["blend_alpha"] is None


def test_inheritance_telemetry_gate_hit_rate_and_blend_alpha():
    telemetry = InheritanceTelemetry()
    telemetry.record_blend_alpha(0.25)
    telemetry.record_applied()
    telemetry.record_applied()
    telemetry.record_applied()
    telemetry.record_skipped(WARMSTART_REASON_GATE_NOT_CLEARED)

    payload = telemetry.to_dict()
    # 4 attempts, 1 gated => coverage 3/4, gate hit-rate (4-1)/4.
    assert payload["warmstart_coverage"] == 3 / 4
    assert payload["gate_not_cleared"] == 1
    assert payload["gate_hit_rate"] == 3 / 4
    assert payload["blend_alpha"] == 0.25


def test_inheritance_telemetry_records_decide_action_failures():
    telemetry = InheritanceTelemetry()
    telemetry.record_decide_action_failure("ValueError")
    telemetry.record_decide_action_failure("ValueError")
    telemetry.record_decide_action_failure("RuntimeError")

    payload = telemetry.to_dict()
    assert payload["decide_action_failures"] == 3
    assert payload["decide_action_failure_reasons"] == {
        "ValueError": 2,
        "RuntimeError": 1,
    }


def test_apply_skips_when_parent_behavior_is_none():
    """Missing ``parent.behavior`` must surface as NO_DECISION_MODULE, not raise."""
    parent = SimpleNamespace(behavior=None)
    offspring = SimpleNamespace(
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(algorithm=SimpleNamespace())
        )
    )
    assert (
        apply_lamarckian_policy_warmstart(parent, offspring)
        == WARMSTART_REASON_NO_DECISION_MODULE
    )


def test_apply_skips_when_offspring_behavior_is_none():
    """Symmetric guard: missing ``offspring.behavior`` is also NO_DECISION_MODULE."""
    parent = SimpleNamespace(
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(algorithm=SimpleNamespace())
        )
    )
    offspring = SimpleNamespace(behavior=None)
    assert (
        apply_lamarckian_policy_warmstart(parent, offspring)
        == WARMSTART_REASON_NO_DECISION_MODULE
    )


# ---------------------------------------------------------------------------
# Helpers for P2/P3/P4 tests
# ---------------------------------------------------------------------------


def _make_extended_pair(
    parent_state,
    child_state,
    *,
    child_load=None,
    parent_resource_level=None,
):
    """Build parent/offspring objects whose ``get_model_state`` accepts kwargs.

    Unlike :func:`_make_pair`, the ``get_model_state`` callable here accepts
    ``include_optimizer_state``, ``include_replay_buffer``, etc., mirroring the
    real :class:`TianshouWrapper` API used by P2/P3/P4.
    """

    def _get_model_state(
        include_optimizer_state=False,
        include_replay_buffer=False,
        replay_buffer_limit=None,
        include_plasticity_state=False,
    ):
        return parent_state

    parent_kwargs = dict(
        agent_id="parent",
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(
                algorithm=SimpleNamespace(
                    get_model_state=_get_model_state,
                ),
            ),
        ),
    )
    if parent_resource_level is not None:
        parent_kwargs["resource_level"] = parent_resource_level
    parent = SimpleNamespace(**parent_kwargs)

    child_policy = SimpleNamespace(
        state_dict=lambda: child_state,
    )
    offspring = SimpleNamespace(
        agent_id="child",
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(
                algorithm=SimpleNamespace(
                    policy=child_policy,
                    load_model_state=child_load if child_load is not None else Mock(),
                ),
            ),
        ),
    )
    return parent, offspring


# ---------------------------------------------------------------------------
# P2 tests
# ---------------------------------------------------------------------------


def test_p2_applies_weights_and_damped_plasticity():
    """P2 must load policy weights with dampened LR and ε."""
    parent_state = {
        "policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))},
        "step_count": 100,
        "plasticity_state": {
            "learning_rate": 0.01,
            "epsilon": 0.4,
            "train_mode": True,
        },
    }
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_extended_pair(parent_state, child_state, child_load=load)

    reason = apply_p2_policy_warmstart(parent, offspring, plasticity_damping=0.5)

    assert reason is None
    load.assert_called_once()
    payload = load.call_args.args[0]
    assert payload["policy_state_dict"] is parent_state["policy_state_dict"]
    assert payload["step_count"] == 100
    assert "plasticity_state" in payload
    damped = payload["plasticity_state"]
    assert abs(damped["learning_rate"] - 0.005) < 1e-9
    assert abs(damped["epsilon"] - 0.2) < 1e-9
    assert damped["train_mode"] is True


def test_p2_works_without_plasticity_state():
    """P2 must succeed even when the parent has no plasticity_state key."""
    parent_state = {
        "policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))},
    }
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_extended_pair(parent_state, child_state, child_load=load)

    reason = apply_p2_policy_warmstart(parent, offspring)

    assert reason is None
    load.assert_called_once()
    payload = load.call_args.args[0]
    assert payload["policy_state_dict"] is parent_state["policy_state_dict"]
    assert "plasticity_state" not in payload


def test_p2_skips_on_incompatible_state():
    """P2 must propagate INCOMPATIBLE_STATE when shapes mismatch."""
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(4, 4))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    parent, offspring = _make_extended_pair(parent_state, child_state)

    assert apply_p2_policy_warmstart(parent, offspring) == WARMSTART_REASON_INCOMPATIBLE_STATE


def test_p2_returns_load_failed_when_load_raises():
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}

    def _load(_payload):
        raise RuntimeError("load failed")

    parent, offspring = _make_extended_pair(parent_state, child_state, child_load=_load)
    assert apply_p2_policy_warmstart(parent, offspring) == WARMSTART_REASON_LOAD_FAILED


# ---------------------------------------------------------------------------
# P3 tests
# ---------------------------------------------------------------------------


def test_p3_applies_weights_optimizer_and_replay():
    """P3 must pass all three continuation-machinery payloads to load_model_state."""
    parent_state = {
        "policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))},
        "step_count": 50,
        "optimizer_state": {"optim": {"state": {}, "param_groups": []}},
        "replay_buffer_state": {"transitions": [], "size": 0},
    }
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_extended_pair(parent_state, child_state, child_load=load)

    reason = apply_p3_policy_warmstart(parent, offspring)

    assert reason is None
    load.assert_called_once()
    payload = load.call_args.args[0]
    assert payload["policy_state_dict"] is parent_state["policy_state_dict"]
    assert payload["step_count"] == 50
    assert payload["optimizer_state"] is parent_state["optimizer_state"]
    assert payload["replay_buffer_state"] is parent_state["replay_buffer_state"]


def test_p3_works_without_optional_payloads():
    """P3 must succeed even when optimizer/replay state is absent."""
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_extended_pair(parent_state, child_state, child_load=load)

    reason = apply_p3_policy_warmstart(parent, offspring)

    assert reason is None
    load.assert_called_once()
    payload = load.call_args.args[0]
    assert "optimizer_state" not in payload
    assert "replay_buffer_state" not in payload


def test_p3_skips_on_incompatible_state():
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(4, 4))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    parent, offspring = _make_extended_pair(parent_state, child_state)

    assert apply_p3_policy_warmstart(parent, offspring) == WARMSTART_REASON_INCOMPATIBLE_STATE


def test_p3_returns_load_failed_when_load_raises():
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}

    def _load(_payload):
        raise RuntimeError("load failed")

    parent, offspring = _make_extended_pair(parent_state, child_state, child_load=_load)
    assert apply_p3_policy_warmstart(parent, offspring) == WARMSTART_REASON_LOAD_FAILED


# ---------------------------------------------------------------------------
# P4 tests
# ---------------------------------------------------------------------------


def test_p4_gate_not_cleared_when_resources_too_low():
    """P4 must skip when the parent's resource level is below the fitness gate."""
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_extended_pair(
        parent_state, child_state, child_load=load, parent_resource_level=0.0
    )

    reason = apply_p4_policy_warmstart(
        parent, offspring, fitness_gate_min_resources=1.0
    )

    assert reason == WARMSTART_REASON_GATE_NOT_CLEARED
    load.assert_not_called()


def test_p4_gate_cleared_when_resources_sufficient():
    """P4 must proceed when the parent clears the fitness gate."""
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_extended_pair(
        parent_state, child_state, child_load=load, parent_resource_level=5.0
    )

    reason = apply_p4_policy_warmstart(
        parent, offspring, fitness_gate_min_resources=1.0
    )

    assert reason is None
    load.assert_called_once()


def test_p4_gate_skipped_when_no_resource_level():
    """P4 must proceed (not gate-fail) when the parent has no resource_level attr."""
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_extended_pair(parent_state, child_state, child_load=load)
    # parent has no resource_level attribute — gate check is skipped

    reason = apply_p4_policy_warmstart(parent, offspring, fitness_gate_min_resources=1.0)

    assert reason is None
    load.assert_called_once()


def test_p4_blends_tensor_weights():
    """P4 blending produces the expected linear combination when torch is available."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return  # skip on environments without torch

    parent_weights = torch.tensor([4.0, 4.0])
    child_weights = torch.tensor([0.0, 0.0])

    parent_state = {"policy_state_dict": {"w": parent_weights}}
    child_state = {"w": child_weights}
    load = Mock()
    parent, offspring = _make_extended_pair(
        parent_state, child_state, child_load=load, parent_resource_level=10.0
    )

    reason = apply_p4_policy_warmstart(
        parent, offspring, blend_alpha=0.5, fitness_gate_min_resources=1.0
    )

    assert reason is None
    load.assert_called_once()
    payload = load.call_args.args[0]
    blended = payload["policy_state_dict"]["w"]
    expected = torch.tensor([2.0, 2.0])
    assert torch.allclose(blended, expected), f"expected {expected}, got {blended}"


def test_p4_falls_back_to_parent_weights_for_non_tensor():
    """P4 must pass parent weights through unchanged for non-Tensor values."""
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_extended_pair(
        parent_state, child_state, child_load=load, parent_resource_level=10.0
    )

    reason = apply_p4_policy_warmstart(parent, offspring)

    assert reason is None
    payload = load.call_args.args[0]
    assert payload["policy_state_dict"]["w"] is parent_state["policy_state_dict"]["w"]


def test_p4_skips_on_incompatible_state():
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(4, 4))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    parent, offspring = _make_extended_pair(
        parent_state, child_state, parent_resource_level=10.0
    )

    assert apply_p4_policy_warmstart(parent, offspring) == WARMSTART_REASON_INCOMPATIBLE_STATE


def test_p4_returns_load_failed_when_load_raises():
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}

    def _load(_payload):
        raise RuntimeError("load failed")

    parent, offspring = _make_extended_pair(
        parent_state, child_state, child_load=_load, parent_resource_level=10.0
    )
    assert apply_p4_policy_warmstart(parent, offspring) == WARMSTART_REASON_LOAD_FAILED


# ---------------------------------------------------------------------------
# Extended-state capability detection (observable downgrade)
# ---------------------------------------------------------------------------


def test_p2_skips_when_extended_state_unsupported():
    """P2 must report EXTENDED_STATE_UNSUPPORTED rather than silently downgrade.

    ``_make_pair`` builds a ``get_model_state`` that takes no kwargs (like an
    older algorithm), so P2 cannot obtain plasticity state.
    """
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_pair(parent_state, child_state, child_load=load)

    reason = apply_p2_policy_warmstart(parent, offspring)

    assert reason == WARMSTART_REASON_EXTENDED_STATE_UNSUPPORTED
    load.assert_not_called()


def test_p3_skips_when_extended_state_unsupported():
    """P3 must report EXTENDED_STATE_UNSUPPORTED when kwargs aren't accepted."""
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_pair(parent_state, child_state, child_load=load)

    reason = apply_p3_policy_warmstart(parent, offspring)

    assert reason == WARMSTART_REASON_EXTENDED_STATE_UNSUPPORTED
    load.assert_not_called()


def test_p1_works_with_kwargless_get_model_state():
    """Lamarckian (P1) needs no extended state, so a kwargless API still works."""
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_pair(parent_state, child_state, child_load=load)

    assert apply_lamarckian_policy_warmstart(parent, offspring) is None
    load.assert_called_once()


def test_p4_works_with_kwargless_get_model_state():
    """P4 blends weights only, so it does not require extended-state support."""
    parent_state = {"policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))}}
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_pair(parent_state, child_state, child_load=load)

    assert apply_p4_policy_warmstart(parent, offspring) is None
    load.assert_called_once()


def test_extended_state_supported_via_var_keyword():
    """A ``**kwargs`` signature is treated as supporting extended state."""
    parent_state = {
        "policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))},
        "plasticity_state": {"learning_rate": 0.01},
    }
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()

    def _get_model_state(**_kwargs):
        return parent_state

    parent = SimpleNamespace(
        agent_id="parent",
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(
                algorithm=SimpleNamespace(get_model_state=_get_model_state),
            ),
        ),
    )
    offspring = SimpleNamespace(
        agent_id="child",
        behavior=SimpleNamespace(
            decision_module=SimpleNamespace(
                algorithm=SimpleNamespace(
                    policy=SimpleNamespace(state_dict=lambda: child_state),
                    load_model_state=load,
                ),
            ),
        ),
    )

    assert apply_p2_policy_warmstart(parent, offspring) is None
    load.assert_called_once()


# ---------------------------------------------------------------------------
# Non-fatal payload construction
# ---------------------------------------------------------------------------


def test_p2_malformed_plasticity_is_non_fatal():
    """A non-numeric learning_rate must yield LOAD_FAILED, never raise."""
    parent_state = {
        "policy_state_dict": {"w": SimpleNamespace(shape=(2, 2))},
        "plasticity_state": {"learning_rate": "not-a-number"},
    }
    child_state = {"w": SimpleNamespace(shape=(2, 2))}
    load = Mock()
    parent, offspring = _make_extended_pair(parent_state, child_state, child_load=load)

    reason = apply_p2_policy_warmstart(parent, offspring)

    assert reason == WARMSTART_REASON_LOAD_FAILED
    load.assert_not_called()


def test_p4_does_not_blend_integer_tensors():
    """Integer (non-float) tensors must pass through verbatim, not blend."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return  # skip on environments without torch

    parent_buffer = torch.tensor([4, 4], dtype=torch.long)
    child_buffer = torch.tensor([0, 0], dtype=torch.long)

    parent_state = {"policy_state_dict": {"n": parent_buffer}}
    child_state = {"n": child_buffer}
    load = Mock()
    parent, offspring = _make_extended_pair(
        parent_state, child_state, child_load=load, parent_resource_level=10.0
    )

    reason = apply_p4_policy_warmstart(parent, offspring, blend_alpha=0.5)

    assert reason is None
    payload = load.call_args.args[0]
    assert payload["policy_state_dict"]["n"] is parent_buffer

"""Direct unit tests for ``farm.core.policy_inheritance`` helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from farm.core.inheritance_telemetry import InheritanceTelemetry
from farm.core.policy_inheritance import (
    WARMSTART_REASON_INCOMPATIBLE_STATE,
    WARMSTART_REASON_LOAD_FAILED,
    WARMSTART_REASON_NO_ALGORITHM,
    WARMSTART_REASON_NO_DECISION_MODULE,
    WARMSTART_REASON_NO_POLICY_STATE,
    WARMSTART_REASON_NO_WARMSTART_API,
    WARMSTART_REASON_PARENT_STATE_FAILED,
    _policy_state_is_compatible,
    apply_lamarckian_policy_warmstart,
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
    assert payload["lamarckian_warmstart_applied"] == 2
    assert payload["lamarckian_warmstart_skipped"] == 3
    assert payload["lamarckian_warmstart_skipped_reasons"] == {
        WARMSTART_REASON_INCOMPATIBLE_STATE: 2,
        WARMSTART_REASON_NO_POLICY_STATE: 1,
    }


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

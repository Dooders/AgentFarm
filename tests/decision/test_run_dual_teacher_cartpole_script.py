"""Unit tests for ``scripts/run_dual_teacher_cartpole.py`` helpers."""

from __future__ import annotations

import importlib.util
import os
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPT_PATH = os.path.join(_REPO_ROOT, "scripts", "run_dual_teacher_cartpole.py")


def _load_dual_teacher_module() -> Any:
    """Load the script as a module (same ``sys.path`` pattern as running from repo root)."""
    scripts_dir = os.path.join(_REPO_ROOT, "scripts")
    for p in (_REPO_ROOT, scripts_dir):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("run_dual_teacher_cartpole", _SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dual_teacher() -> Any:
    return _load_dual_teacher_module()


@pytest.mark.ml
def test_resolve_pipeline_states_merges_a_and_b(dual_teacher: Any, tmp_path: Any) -> None:
    out = str(tmp_path)
    a = np.zeros((5, 4), dtype=np.float32)
    a[:, 0] = 1.0
    b = np.zeros((5, 4), dtype=np.float32)
    b[:, 1] = 2.0
    np.save(os.path.join(out, "replay_states_A.npy"), a)
    np.save(os.path.join(out, "replay_states_B.npy"), b)

    states, label = dual_teacher._resolve_pipeline_states(
        output_dir=out,
        states_file="",
        n_states=6,
        input_dim=4,
        seed=123,
    )
    assert states.shape == (6, 4)
    assert "merged" in label
    assert np.any(states[:, 0] == 1.0) and np.any(states[:, 1] == 2.0)


@pytest.mark.ml
def test_resolve_pipeline_states_subsamples_merged(dual_teacher: Any, tmp_path: Any) -> None:
    out = str(tmp_path)
    a = np.random.default_rng(1).standard_normal((100, 4)).astype(np.float32)
    b = np.random.default_rng(2).standard_normal((100, 4)).astype(np.float32)
    np.save(os.path.join(out, "replay_states_A.npy"), a)
    np.save(os.path.join(out, "replay_states_B.npy"), b)

    states, _label = dual_teacher._resolve_pipeline_states(
        output_dir=out,
        states_file="",
        n_states=50,
        input_dim=4,
        seed=7,
    )
    assert states.shape == (50, 4)


@pytest.mark.ml
def test_resolve_pipeline_states_explicit_file(dual_teacher: Any, tmp_path: Any) -> None:
    out = str(tmp_path)
    fpath = os.path.join(out, "custom.npy")
    data = np.ones((10, 4), dtype=np.float32)
    np.save(fpath, data)

    states, label = dual_teacher._resolve_pipeline_states(
        output_dir=out,
        states_file=fpath,
        n_states=2000,
        input_dim=4,
        seed=0,
    )
    assert label == fpath
    np.testing.assert_array_equal(states, data)


@pytest.mark.ml
def test_aggregate_pipeline_passed(dual_teacher: Any) -> None:
    agg = dual_teacher._aggregate_pipeline_passed
    ok_dist = {"passed": True}
    bad_dist = {"passed": False}
    ok_recomb = {"passed": True}
    bad_recomb = {"passed": False}

    overall, d_ok, r_ok = agg(
        report_only=False,
        distillation_reports=[ok_dist, ok_dist],
        recombination_report=ok_recomb,
    )
    assert overall is True and d_ok is True and r_ok is True

    overall, d_ok, r_ok = agg(
        report_only=False,
        distillation_reports=[ok_dist, bad_dist],
        recombination_report=ok_recomb,
    )
    assert overall is False and d_ok is False and r_ok is True

    overall, d_ok, r_ok = agg(
        report_only=False,
        distillation_reports=[ok_dist, ok_dist],
        recombination_report=bad_recomb,
    )
    assert overall is False and d_ok is True and r_ok is False

    overall, d_ok, r_ok = agg(
        report_only=True,
        distillation_reports=[ok_dist, bad_dist],
        recombination_report=bad_recomb,
    )
    assert overall is True and d_ok is False and r_ok is False


@pytest.mark.ml
def test_validate_cli_ranges_accepts_valid_values(dual_teacher: Any) -> None:
    dual_teacher._validate_cli_ranges(
        SimpleNamespace(
            finetune_teacher_weight_a=0.5,
            finetune_val_fraction=0.2,
        )
    )


@pytest.mark.ml
@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_validate_cli_ranges_rejects_invalid_teacher_weight(
    dual_teacher: Any, alpha: float
) -> None:
    with pytest.raises(ValueError, match="--finetune-teacher-weight-a"):
        dual_teacher._validate_cli_ranges(
            SimpleNamespace(
                finetune_teacher_weight_a=alpha,
                finetune_val_fraction=0.2,
            )
        )


@pytest.mark.ml
@pytest.mark.parametrize("val_fraction", [-0.1, 1.0, 1.5])
def test_validate_cli_ranges_rejects_invalid_val_fraction(
    dual_teacher: Any, val_fraction: float
) -> None:
    with pytest.raises(ValueError, match="--finetune-val-fraction"):
        dual_teacher._validate_cli_ranges(
            SimpleNamespace(
                finetune_teacher_weight_a=0.5,
                finetune_val_fraction=val_fraction,
            )
        )

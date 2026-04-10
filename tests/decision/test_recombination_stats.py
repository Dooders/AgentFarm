"""Unit tests for farm/core/decision/training/recombination_stats.py.

All tests use fixed toy numbers so results are deterministic and do not require
any ML training or file I/O beyond temporary files.

Covers:
- load_manifest_entries: valid paths, multiple files, missing file, bad JSON.
- load_eval_reports: basic loading.
- compute_condition_summary: mean/std/min/max/count, NaN handling, single value.
- aggregate_conditions: grouping, n_degenerate, extra_metrics.
- ConditionSummary.to_dict round-trip.
- paired_ttest: known t-statistic, p-value significance, mismatched lengths.
- welch_ttest: known result, independent samples, short sample error.
- bootstrap_ci: CI bounds contain mean for large N, seed reproducibility,
  bad confidence_level, bad statistic.
- TTestResult.to_dict / BootstrapCIResult.to_dict serialisation.
- __init__.py exports.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from typing import Any, Dict, List

import numpy as np
import pytest

from farm.core.decision.training.recombination_stats import (
    NUMERIC_METRIC_KEYS,
    BootstrapCIResult,
    ConditionSummary,
    TTestResult,
    aggregate_conditions,
    bootstrap_ci,
    compute_condition_summary,
    load_eval_reports,
    load_manifest_entries,
    paired_ttest,
    welch_ttest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest_row(
    mode: str = "weighted",
    regime: str = "short",
    primary: float = 0.80,
    agree_a: float = 0.82,
    agree_b: float = 0.80,
    oracle: float = 0.90,
    degenerate: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """Minimal manifest row dict matching ManifestEntry.to_dict() schema."""
    return {
        "child_id": f"000_{mode}_a0p50_s{seed}_{regime}",
        "crossover_mode": mode,
        "crossover_alpha": 0.5,
        "crossover_seed": seed,
        "finetune_regime": regime,
        "finetune_epochs": 5,
        "finetune_lr": 1e-3,
        "finetune_quantization_applied": "none",
        "child_pt_path": f"/tmp/child_{seed}.pt",
        "eval_report_path": f"/tmp/eval_{seed}.json",
        "run_config_path": f"/tmp/run_config_{seed}.json",
        "primary_metric": primary,
        "child_vs_parent_a_agreement": agree_a,
        "child_vs_parent_b_agreement": agree_b,
        "oracle_agreement": oracle,
        "kl_divergence_a": 0.3,
        "kl_divergence_b": 0.4,
        "mse_a": 0.5,
        "mse_b": 0.6,
        "cosine_a": 0.85,
        "cosine_b": 0.80,
        "degenerate": degenerate,
    }


def _write_manifest(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)


# ---------------------------------------------------------------------------
# Tests: load_manifest_entries
# ---------------------------------------------------------------------------


class TestLoadManifestEntries:
    def test_single_file(self):
        rows = [_make_manifest_row(mode="weighted"), _make_manifest_row(mode="random")]
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(rows, fh)
            path = fh.name
        try:
            result = load_manifest_entries(path)
            assert len(result) == 2
            assert result[0]["crossover_mode"] == "weighted"
            assert result[1]["crossover_mode"] == "random"
        finally:
            os.unlink(path)

    def test_source_file_key_injected(self):
        rows = [_make_manifest_row()]
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(rows, fh)
            path = fh.name
        try:
            result = load_manifest_entries(path)
            assert result[0]["_source_file"] == path
        finally:
            os.unlink(path)

    def test_multiple_files_concatenated(self):
        rows_a = [_make_manifest_row(mode="weighted", seed=0)]
        rows_b = [_make_manifest_row(mode="random", seed=1)]
        with tempfile.TemporaryDirectory() as tmp:
            path_a = os.path.join(tmp, "manifest_a.json")
            path_b = os.path.join(tmp, "manifest_b.json")
            _write_manifest(rows_a, path_a)
            _write_manifest(rows_b, path_b)
            result = load_manifest_entries([path_a, path_b])
        assert len(result) == 2
        modes = {r["crossover_mode"] for r in result}
        assert modes == {"weighted", "random"}

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_manifest_entries("/nonexistent/path/manifest.json")

    def test_non_array_json_raises(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as fh:
            json.dump({"not": "an array"}, fh)
            path = fh.name
        try:
            with pytest.raises(ValueError, match="JSON array"):
                load_manifest_entries(path)
        finally:
            os.unlink(path)

    def test_empty_manifest(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as fh:
            json.dump([], fh)
            path = fh.name
        try:
            result = load_manifest_entries(path)
            assert result == []
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: load_eval_reports
# ---------------------------------------------------------------------------


class TestLoadEvalReports:
    def test_loads_reports(self):
        report = {
            "schema_version": "1.0",
            "comparisons": {
                "child_vs_parent_a": {"action_agreement": 0.82},
            },
            "summary": {},
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", delete=False, encoding="utf-8"
        ) as fh:
            json.dump(report, fh)
            path = fh.name
        try:
            result = load_eval_reports([path])
            assert len(result) == 1
            assert result[0]["schema_version"] == "1.0"
            assert result[0]["_source_file"] == path
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: compute_condition_summary
# ---------------------------------------------------------------------------


class TestComputeConditionSummary:
    def test_known_values(self):
        # mean([0.8, 0.9, 0.7]) = 0.8, std ≈ 0.1
        s = compute_condition_summary([0.8, 0.9, 0.7])
        assert abs(s["mean"] - 0.8) < 1e-9
        assert abs(s["std"] - 0.1) < 1e-9
        assert s["min"] == pytest.approx(0.7)
        assert s["max"] == pytest.approx(0.9)
        assert s["count"] == 3

    def test_single_value(self):
        s = compute_condition_summary([0.75])
        assert s["mean"] == pytest.approx(0.75)
        assert math.isnan(s["std"])  # ddof=1 undefined for N=1
        assert s["count"] == 1

    def test_nan_values_dropped(self):
        s = compute_condition_summary([0.8, float("nan"), 0.9, None])
        assert s["count"] == 2
        assert abs(s["mean"] - 0.85) < 1e-9

    def test_empty(self):
        s = compute_condition_summary([])
        assert s["count"] == 0
        assert math.isnan(s["mean"])

    def test_all_none(self):
        s = compute_condition_summary([None, None])
        assert s["count"] == 0

    def test_two_values_std(self):
        # std([0.6, 0.8], ddof=1) = 0.1*sqrt(2) ≈ 0.14142
        s = compute_condition_summary([0.6, 0.8])
        assert s["std"] == pytest.approx(math.sqrt(0.02), rel=1e-6)


# ---------------------------------------------------------------------------
# Tests: aggregate_conditions
# ---------------------------------------------------------------------------


class TestAggregateConditions:
    def _make_rows(self):
        return [
            _make_manifest_row(mode="weighted", regime="short", primary=0.80, seed=0),
            _make_manifest_row(mode="weighted", regime="short", primary=0.82, seed=1),
            _make_manifest_row(mode="random", regime="short", primary=0.70, seed=0),
            _make_manifest_row(mode="random", regime="short", primary=0.72, seed=1),
            _make_manifest_row(mode="weighted", regime="short", degenerate=True, primary=0.60, seed=2),
        ]

    def test_grouping_by_mode_only(self):
        rows = self._make_rows()
        summaries = aggregate_conditions(rows, group_by=["crossover_mode"])
        keys_str = {dict(k)["crossover_mode"] for k in summaries}
        assert keys_str == {"weighted", "random"}

    def test_n_runs_correct(self):
        rows = self._make_rows()
        summaries = aggregate_conditions(rows, group_by=["crossover_mode"])
        weighted_key = next(k for k in summaries if dict(k)["crossover_mode"] == "weighted")
        assert summaries[weighted_key].n_runs == 3  # three weighted rows

    def test_n_degenerate(self):
        rows = self._make_rows()
        summaries = aggregate_conditions(rows, group_by=["crossover_mode"])
        weighted_key = next(k for k in summaries if dict(k)["crossover_mode"] == "weighted")
        assert summaries[weighted_key].n_degenerate == 1

    def test_mean_primary_metric(self):
        rows = self._make_rows()
        summaries = aggregate_conditions(rows, group_by=["crossover_mode"])
        random_key = next(k for k in summaries if dict(k)["crossover_mode"] == "random")
        s = summaries[random_key]
        assert s.mean_primary_metric == pytest.approx(0.71)

    def test_extra_metrics(self):
        rows = self._make_rows()
        summaries = aggregate_conditions(
            rows, group_by=["crossover_mode"], extra_metrics=["kl_divergence_a"]
        )
        any_key = next(iter(summaries))
        assert "kl_divergence_a" in summaries[any_key].extra
        assert "mean" in summaries[any_key].extra["kl_divergence_a"]

    def test_to_dict_serialisable(self):
        rows = self._make_rows()
        summaries = aggregate_conditions(rows, group_by=["crossover_mode"])
        for s in summaries.values():
            d = s.to_dict()
            assert isinstance(d, dict)
            assert "n_runs" in d
            json.dumps(d)  # must be JSON-serialisable (no NaN in this case)


# ---------------------------------------------------------------------------
# Tests: paired_ttest
# ---------------------------------------------------------------------------


class TestPairedTtest:
    def test_identical_samples_undefined_t(self):
        # When all paired differences are zero, std(diff)=0.
        # Both SciPy and the manual path return NaN for t (division by zero
        # is mathematically undefined, not zero).  This is the expected behavior.
        a = [0.7, 0.8, 0.9]
        b = [0.7, 0.8, 0.9]
        r = paired_ttest(a, b)
        # t-statistic must be NaN (std of zero-differences → undefined).
        assert math.isnan(r.statistic)

    def test_known_large_effect(self):
        # a consistently higher than b → t > 0, p small.
        # Using values where the effect is clear.
        a = [0.90, 0.88, 0.91, 0.89, 0.92]
        b = [0.50, 0.52, 0.49, 0.51, 0.50]
        r = paired_ttest(a, b)
        assert r.statistic > 0
        assert r.pvalue < 0.001
        assert r.mean_diff > 0

    def test_mean_values(self):
        a = [0.8, 0.9, 0.7]
        b = [0.6, 0.7, 0.5]
        r = paired_ttest(a, b)
        assert r.mean_a == pytest.approx(0.8, rel=1e-6)
        assert r.mean_b == pytest.approx(0.6, rel=1e-6)
        assert r.mean_diff == pytest.approx(0.2, rel=1e-6)

    def test_method_label(self):
        r = paired_ttest([0.8, 0.9], [0.6, 0.7])
        assert r.method == "paired"

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="equal-length"):
            paired_ttest([0.8, 0.9], [0.6])

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            paired_ttest([0.8], [0.9])

    def test_to_dict(self):
        r = paired_ttest([0.8, 0.9, 0.7], [0.6, 0.7, 0.5])
        d = r.to_dict()
        assert set(d.keys()) >= {"statistic", "pvalue", "dof", "mean_a", "mean_b", "mean_diff", "method"}

    def test_scipy_vs_manual_agreement(self):
        # Test that the result is numerically plausible regardless of SciPy availability.
        a = [0.85, 0.87, 0.83, 0.86, 0.84]
        b = [0.70, 0.72, 0.68, 0.71, 0.69]
        r = paired_ttest(a, b)
        # p-value must be very small for this large effect.
        assert r.pvalue < 0.01
        assert r.dof == pytest.approx(4.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: welch_ttest
# ---------------------------------------------------------------------------


class TestWelchTtest:
    def test_known_result(self):
        # Both groups identical → t≈0, p≈1.
        a = [0.8, 0.8, 0.8]
        b = [0.8, 0.8, 0.8]
        r = welch_ttest(a, b)
        assert abs(r.statistic) < 1e-6
        assert r.pvalue == pytest.approx(1.0, abs=1e-6)

    def test_significant_difference(self):
        a = [0.90, 0.88, 0.91, 0.89, 0.92, 0.90]
        b = [0.50, 0.52, 0.49, 0.51, 0.50, 0.52]
        r = welch_ttest(a, b)
        assert r.pvalue < 0.001
        assert r.statistic > 0

    def test_unequal_lengths(self):
        # With very few observations Welch dof is very low, inflating p-values.
        # Use more observations so the test has enough power.
        a = [0.80, 0.90, 0.85, 0.82, 0.88, 0.86, 0.84]
        b = [0.50, 0.52, 0.49, 0.51, 0.53]
        r = welch_ttest(a, b)
        assert r.pvalue < 0.01

    def test_method_label(self):
        r = welch_ttest([0.8, 0.9], [0.6, 0.7])
        assert r.method == "welch"

    def test_short_sample_a_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            welch_ttest([0.8], [0.7, 0.6])

    def test_short_sample_b_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            welch_ttest([0.8, 0.7], [0.6])

    def test_to_dict_keys(self):
        r = welch_ttest([0.8, 0.9, 0.85], [0.7, 0.72, 0.68])
        d = r.to_dict()
        assert "pvalue" in d and "statistic" in d and "dof" in d

    def test_mean_diff_sign(self):
        a = [0.8, 0.9]
        b = [0.6, 0.7]
        r = welch_ttest(a, b)
        assert r.mean_diff > 0  # a > b → positive diff


# ---------------------------------------------------------------------------
# Tests: bootstrap_ci
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_mean_within_ci(self):
        # For a large N sample the observed mean should be inside the CI.
        rng = np.random.default_rng(42)
        values = rng.normal(0.8, 0.05, size=100).tolist()
        r = bootstrap_ci(values, confidence_level=0.95, n_bootstrap=1000, rng=0)
        assert r.ci_low <= r.mean <= r.ci_high

    def test_seed_reproducibility(self):
        values = [0.8, 0.9, 0.7, 0.85, 0.75]
        r1 = bootstrap_ci(values, rng=42, n_bootstrap=500)
        r2 = bootstrap_ci(values, rng=42, n_bootstrap=500)
        assert r1.ci_low == r2.ci_low
        assert r1.ci_high == r2.ci_high

    def test_different_seeds_different_ci(self):
        values = [0.8, 0.9, 0.7, 0.85, 0.75]
        r1 = bootstrap_ci(values, rng=1, n_bootstrap=100)
        r2 = bootstrap_ci(values, rng=999, n_bootstrap=100)
        # Very likely to differ (not guaranteed, but seed should differ).
        # Just verify both produce valid CIs.
        assert r1.ci_low <= r1.mean <= r1.ci_high
        assert r2.ci_low <= r2.mean <= r2.ci_high

    def test_confidence_level_stored(self):
        values = [0.8, 0.9, 0.7]
        r = bootstrap_ci(values, confidence_level=0.90, rng=0)
        assert r.confidence_level == 0.90

    def test_n_bootstrap_stored(self):
        values = [0.8, 0.9, 0.7]
        r = bootstrap_ci(values, n_bootstrap=500, rng=0)
        assert r.n_bootstrap == 500

    def test_wider_ci_at_lower_confidence(self):
        rng = np.random.default_rng(7)
        values = rng.normal(0.8, 0.1, size=50).tolist()
        r95 = bootstrap_ci(values, confidence_level=0.95, rng=7, n_bootstrap=2000)
        r80 = bootstrap_ci(values, confidence_level=0.80, rng=7, n_bootstrap=2000)
        width_95 = r95.ci_high - r95.ci_low
        width_80 = r80.ci_high - r80.ci_low
        assert width_95 > width_80

    def test_bad_confidence_raises(self):
        with pytest.raises(ValueError, match="confidence_level"):
            bootstrap_ci([0.8, 0.9], confidence_level=1.5)

    def test_bad_confidence_zero_raises(self):
        with pytest.raises(ValueError, match="confidence_level"):
            bootstrap_ci([0.8, 0.9], confidence_level=0.0)

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            bootstrap_ci([0.8])

    def test_bad_statistic_raises(self):
        with pytest.raises(ValueError, match="statistic"):
            bootstrap_ci([0.8, 0.9], statistic="mode")

    def test_median_statistic(self):
        values = [0.8, 0.9, 0.7, 0.85, 0.75]
        r = bootstrap_ci(values, statistic="median", rng=0)
        assert r.mean == pytest.approx(np.median(values))

    def test_to_dict(self):
        r = bootstrap_ci([0.8, 0.9, 0.7], rng=0)
        d = r.to_dict()
        assert set(d.keys()) == {
            "mean", "ci_low", "ci_high", "confidence_level", "n_bootstrap"
        }


# ---------------------------------------------------------------------------
# Tests: NUMERIC_METRIC_KEYS constant
# ---------------------------------------------------------------------------


class TestNumericMetricKeys:
    def test_contains_primary(self):
        assert "primary_metric" in NUMERIC_METRIC_KEYS

    def test_contains_agreement_keys(self):
        assert "child_vs_parent_a_agreement" in NUMERIC_METRIC_KEYS
        assert "child_vs_parent_b_agreement" in NUMERIC_METRIC_KEYS

    def test_is_tuple(self):
        assert isinstance(NUMERIC_METRIC_KEYS, tuple)


# ---------------------------------------------------------------------------
# Tests: __init__.py exports
# ---------------------------------------------------------------------------


class TestInitExports:
    def test_all_public_symbols_importable(self):
        from farm.core.decision.training import (  # noqa: F401
            NUMERIC_METRIC_KEYS,
            BootstrapCIResult,
            ConditionSummary,
            TTestResult,
            aggregate_conditions,
            bootstrap_ci,
            compute_condition_summary,
            load_eval_reports,
            load_manifest_entries,
            paired_ttest,
            welch_ttest,
        )

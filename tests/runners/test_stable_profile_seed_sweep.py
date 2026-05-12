"""Tests for the stable-profile seed-sweep scripts.

Tests focus on the pure-logic helpers that don't require a full simulation:
- STABLE_SUB_PROFILES constants and InitialConditionsConfig construction
- Aggregation helpers (_mean, _variance, _t_ci, _pct_shift)
- Speciation-slope classifier
- Per-run metrics extraction from synthetic JSONL data
- Profile aggregation and robustness assessment
- Markdown report generation
- Run discovery from a synthetic directory tree
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ── Import helpers from both scripts ─────────────────────────────────────────

from scripts.run_stable_profile_seed_sweep import (  # noqa: E402
    DEFAULT_PROFILES,
    DEFAULT_SEEDS,
    STABLE_SUB_PROFILES,
)
from scripts.analyze_stable_profile_seed_sweep import (  # noqa: E402
    _aggregate_profile,
    _assess_robustness,
    _build_markdown,
    _classify_speciation_direction,
    _discover_runs,
    _extract_run_metrics,
    _mean,
    _pct_shift,
    _speciation_slope,
    _t_ci,
    _variance,
    PROFILE_ORDER,
)


# ── Helper builders ───────────────────────────────────────────────────────────

def _make_trajectory(
    steps: List[int],
    spec_values: List[float],
    n_alive: int = 100,
) -> List[Dict[str, Any]]:
    """Synthetic trajectory JSONL rows."""
    rows = []
    for step, sv in zip(steps, spec_values):
        rows.append({
            "step": step,
            "speciation_index": sv,
            "n_alive": n_alive,
            "realized_birth_rate": 0.01,
            "realized_death_rate": 0.01,
        })
    return rows


def _make_snapshot(
    step: int,
    n_agents: int = 10,
    learning_rate_vals: List[float] = None,
) -> Dict[str, Any]:
    """Synthetic snapshot row."""
    if learning_rate_vals is None:
        learning_rate_vals = [0.01] * n_agents
    agents = []
    for lr in learning_rate_vals:
        agents.append({
            "agent_id": f"a_{lr}",
            "chromosome": {
                "learning_rate": lr,
                "gamma": 0.99,
                "memory_size": 64,
            },
        })
    return {"step": step, "agents": agents}


def _write_run_dir(
    base: Path,
    profile: str,
    seed: int,
    trajectory: List[Dict[str, Any]],
    snapshots: List[Dict[str, Any]],
) -> Path:
    """Write a synthetic run directory under base/stable_{profile}/seed_{seed}/."""
    run_dir = base / f"stable_{profile}" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    traj_path = run_dir / "intrinsic_gene_trajectory.jsonl"
    snap_path = run_dir / "intrinsic_gene_snapshots.jsonl"
    with traj_path.open("w", encoding="utf-8") as fh:
        for row in trajectory:
            fh.write(json.dumps(row) + "\n")
    with snap_path.open("w", encoding="utf-8") as fh:
        for row in snapshots:
            fh.write(json.dumps(row) + "\n")
    return run_dir


# ── Tests: constants ──────────────────────────────────────────────────────────

class TestStableSubProfiles(unittest.TestCase):
    def test_all_three_profiles_defined(self):
        self.assertIn("conservative", STABLE_SUB_PROFILES)
        self.assertIn("balanced", STABLE_SUB_PROFILES)
        self.assertIn("buffered", STABLE_SUB_PROFILES)

    def test_resource_level_ordering(self):
        con = STABLE_SUB_PROFILES["conservative"]["initial_agent_resource_level"]
        bal = STABLE_SUB_PROFILES["balanced"]["initial_agent_resource_level"]
        buf = STABLE_SUB_PROFILES["buffered"]["initial_agent_resource_level"]
        self.assertLess(con, bal)
        self.assertLess(bal, buf)

    def test_resource_count_ordering(self):
        self.assertLess(
            STABLE_SUB_PROFILES["conservative"]["initial_resource_count"],
            STABLE_SUB_PROFILES["buffered"]["initial_resource_count"],
        )

    def test_regen_rate_ordering(self):
        self.assertLess(
            STABLE_SUB_PROFILES["conservative"]["resource_regen_rate"],
            STABLE_SUB_PROFILES["buffered"]["resource_regen_rate"],
        )

    def test_overrides_accepted_by_initial_conditions_config(self):
        from farm.runners.intrinsic_evolution_experiment import InitialConditionsConfig
        for profile, overrides in STABLE_SUB_PROFILES.items():
            with self.subTest(profile=profile):
                cfg = InitialConditionsConfig(profile="stable", **overrides)
                resolved = cfg.resolve()
                self.assertEqual(
                    resolved["initial_agent_resource_level"],
                    overrides["initial_agent_resource_level"],
                )
                self.assertEqual(
                    resolved["initial_resource_count"],
                    overrides["initial_resource_count"],
                )
                self.assertAlmostEqual(
                    resolved["resource_regen_rate"],
                    overrides["resource_regen_rate"],
                    places=6,
                )
                self.assertEqual(
                    resolved["resource_regen_amount"],
                    overrides["resource_regen_amount"],
                )

    def test_default_seeds_non_empty(self):
        self.assertGreater(len(DEFAULT_SEEDS), 0)

    def test_default_profiles_match_sub_profile_keys(self):
        for p in DEFAULT_PROFILES:
            self.assertIn(p, STABLE_SUB_PROFILES)


# ── Tests: statistical helpers ────────────────────────────────────────────────

class TestStatHelpers(unittest.TestCase):
    def test_mean_empty(self):
        self.assertTrue(math.isnan(_mean([])))

    def test_mean_single(self):
        self.assertAlmostEqual(_mean([5.0]), 5.0)

    def test_mean_multiple(self):
        self.assertAlmostEqual(_mean([1.0, 2.0, 3.0]), 2.0)

    def test_variance_insufficient(self):
        self.assertTrue(math.isnan(_variance([])))
        self.assertTrue(math.isnan(_variance([1.0])))

    def test_variance_known(self):
        # Variance of [2, 4] is ((2-3)^2 + (4-3)^2) / 1 = 2
        self.assertAlmostEqual(_variance([2.0, 4.0]), 2.0)

    def test_t_ci_single(self):
        lo, hi = _t_ci([5.0])
        self.assertAlmostEqual(lo, 5.0)
        self.assertAlmostEqual(hi, 5.0)

    def test_t_ci_symmetric(self):
        xs = [1.0, 2.0, 3.0]
        lo, hi = _t_ci(xs)
        m = _mean(xs)
        self.assertLess(lo, m)
        self.assertGreater(hi, m)
        # Should be symmetric around mean
        self.assertAlmostEqual(m - lo, hi - m, places=8)

    def test_pct_shift_zero_initial(self):
        self.assertTrue(math.isnan(_pct_shift(0.0, 5.0)))

    def test_pct_shift_known(self):
        self.assertAlmostEqual(_pct_shift(10.0, 12.0), 20.0)

    def test_pct_shift_negative(self):
        self.assertAlmostEqual(_pct_shift(10.0, 8.0), -20.0)

    def test_pct_shift_nan_inputs(self):
        self.assertTrue(math.isnan(_pct_shift(float("nan"), 1.0)))
        self.assertTrue(math.isnan(_pct_shift(1.0, float("nan"))))


# ── Tests: speciation direction classifier ────────────────────────────────────

class TestSpeciationClassifier(unittest.TestCase):
    def test_diverging_positive_slope(self):
        traj = _make_trajectory([0, 100, 200, 300], [0.5, 0.6, 0.7, 0.8])
        slope = _speciation_slope(traj)
        self.assertGreater(slope, 0)
        self.assertEqual(_classify_speciation_direction(slope), "diverging")

    def test_merging_negative_slope(self):
        traj = _make_trajectory([0, 100, 200, 300], [0.8, 0.7, 0.6, 0.5])
        slope = _speciation_slope(traj)
        self.assertLess(slope, 0)
        self.assertEqual(_classify_speciation_direction(slope), "merging")

    def test_stable_flat(self):
        traj = _make_trajectory([0, 100, 200, 300], [0.7, 0.7, 0.7, 0.7])
        slope = _speciation_slope(traj)
        self.assertEqual(_classify_speciation_direction(slope), "stable")

    def test_empty_trajectory_returns_nan_slope(self):
        self.assertTrue(math.isnan(_speciation_slope([])))

    def test_single_point_returns_unknown_direction(self):
        traj = _make_trajectory([0], [0.7])
        slope = _speciation_slope(traj)
        self.assertTrue(math.isnan(slope))
        self.assertEqual(_classify_speciation_direction(slope), "unknown")


# ── Tests: per-run metrics extraction ────────────────────────────────────────

class TestExtractRunMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_run(self, traj, snaps, profile="conservative", seed=42):
        return _write_run_dir(self.base, profile, seed, traj, snaps)

    def test_returns_none_for_missing_trajectory(self):
        run_dir = self.base / "stable_conservative" / "seed_42"
        run_dir.mkdir(parents=True)
        result = _extract_run_metrics(run_dir)
        self.assertIsNone(result)

    def test_basic_extraction(self):
        traj = _make_trajectory([0, 50, 100], [0.6, 0.65, 0.7])
        snaps = [
            _make_snapshot(0, learning_rate_vals=[0.01, 0.01, 0.01]),
            _make_snapshot(100, learning_rate_vals=[0.012, 0.012, 0.012]),
        ]
        run_dir = self._write_run(traj, snaps)
        metrics = _extract_run_metrics(run_dir)
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["speciation_final"], 0.7)
        self.assertGreater(metrics["speciation_slope"], 0)
        self.assertEqual(metrics["speciation_direction"], "diverging")

    def test_learning_rate_pct_shift_computed(self):
        traj = _make_trajectory([0, 100], [0.6, 0.7])
        snaps = [
            _make_snapshot(0, learning_rate_vals=[0.01, 0.01]),
            _make_snapshot(100, learning_rate_vals=[0.012, 0.012]),
        ]
        run_dir = self._write_run(traj, snaps)
        metrics = _extract_run_metrics(run_dir)
        self.assertIsNotNone(metrics)
        lr_shift = metrics["gene_pct_shift"].get("learning_rate")
        self.assertIsNotNone(lr_shift)
        self.assertAlmostEqual(lr_shift, 20.0, places=3)

    def test_population_mean_computed(self):
        traj = _make_trajectory([0, 100, 200], [0.6, 0.65, 0.7], n_alive=50)
        snaps = []
        run_dir = self._write_run(traj, snaps)
        metrics = _extract_run_metrics(run_dir)
        self.assertIsNotNone(metrics)
        self.assertAlmostEqual(metrics["population_mean"], 50.0)


# ── Tests: profile aggregation ────────────────────────────────────────────────

class TestAggregateProfile(unittest.TestCase):
    def _make_run(self, spec_final, spec_slope, lr_shift):
        direction = _classify_speciation_direction(spec_slope)
        return {
            "speciation_final": spec_final,
            "speciation_mean": spec_final,
            "speciation_slope": spec_slope,
            "speciation_direction": direction,
            "gene_pct_shift": {"learning_rate": lr_shift, "gamma": -2.0},
            "gene_initial": {"learning_rate": 0.01, "gamma": 0.99},
            "gene_final": {
                "learning_rate": 0.01 * (1 + lr_shift / 100),
                "gamma": 0.99 * (1 - 0.02),
            },
            "population_mean": 100.0,
            "population_final": 95.0,
            "trajectory": [],
        }

    def test_basic_aggregation(self):
        runs = [
            self._make_run(0.72, 0.001, 20.0),
            self._make_run(0.74, 0.0015, 22.0),
            self._make_run(0.70, 0.0008, 18.0),
        ]
        agg = _aggregate_profile(runs)
        self.assertEqual(agg["n_seeds"], 3)
        self.assertAlmostEqual(agg["speciation_final"]["mean"], (0.72 + 0.74 + 0.70) / 3, places=6)
        lr_agg = agg["per_gene"]["learning_rate"]
        self.assertAlmostEqual(lr_agg["mean_pct_shift"], 20.0, places=3)
        self.assertTrue(lr_agg["all_positive"])
        self.assertFalse(lr_agg["all_negative"])

    def test_sign_agreement_all_negative(self):
        runs = [
            self._make_run(0.68, -0.001, -8.0),
            self._make_run(0.66, -0.0012, -6.0),
        ]
        agg = _aggregate_profile(runs)
        lr_agg = agg["per_gene"]["learning_rate"]
        self.assertTrue(lr_agg["all_negative"])
        self.assertAlmostEqual(lr_agg["sign_agreement"], 1.0)

    def test_modal_direction_selection(self):
        runs = [
            self._make_run(0.7, 0.001, 20.0),   # diverging
            self._make_run(0.7, 0.0012, 21.0),   # diverging
            self._make_run(0.7, -0.0005, -5.0),  # merging
        ]
        agg = _aggregate_profile(runs)
        self.assertEqual(agg["speciation_direction_modal"], "diverging")
        self.assertAlmostEqual(agg["speciation_direction_agreement"], 2.0 / 3.0, places=6)

    def test_ci_wider_than_mean(self):
        runs = [
            self._make_run(0.68, -0.001, -8.0),
            self._make_run(0.72, 0.001, 20.0),
            self._make_run(0.70, 0.0005, 5.0),
        ]
        agg = _aggregate_profile(runs)
        sf = agg["speciation_final"]
        lo, hi = sf["ci95"]
        m = sf["mean"]
        self.assertLessEqual(lo, m)
        self.assertGreaterEqual(hi, m)


# ── Tests: robustness assessment ──────────────────────────────────────────────

class TestAssessRobustness(unittest.TestCase):
    def _agg_with_lr(self, profile, all_positive, all_negative, sign_agreement, modal_dir, dir_agreement):
        return {
            "n_seeds": 4,
            "speciation_direction_modal": modal_dir,
            "speciation_direction_agreement": dir_agreement,
            "speciation_final": {"mean": 0.7, "variance": 0.001, "ci95": [0.65, 0.75], "n": 4},
            "speciation_slope": {"mean": 0.001, "variance": 0.0, "ci95": [0.0, 0.002], "n": 4},
            "speciation_directions": [modal_dir] * 3 + ["stable"],
            "population_mean": {"mean": 100.0, "variance": 1.0, "ci95": [98.0, 102.0], "n": 4},
            "population_final": {"mean": 95.0, "variance": 2.0, "ci95": [92.0, 98.0], "n": 4},
            "per_gene": {
                "learning_rate": {
                    "mean_pct_shift": 20.0 if all_positive else -8.0,
                    "variance": 4.0,
                    "ci95": [15.0, 25.0] if all_positive else [-12.0, -4.0],
                    "n": 4,
                    "sign_agreement": sign_agreement,
                    "all_positive": all_positive,
                    "all_negative": all_negative,
                },
                "gamma": {
                    "mean_pct_shift": -2.0,
                    "variance": 0.5,
                    "ci95": [-3.0, -1.0],
                    "n": 4,
                    "sign_agreement": 1.0,
                    "all_positive": False,
                    "all_negative": True,
                },
            },
        }

    def test_lr_flip_detected_when_both_consistent(self):
        aggs = {
            "buffered": self._agg_with_lr("buffered", True, False, 1.0, "diverging", 1.0),
            "conservative": self._agg_with_lr("conservative", False, True, 1.0, "merging", 1.0),
        }
        robustness = _assess_robustness(aggs)
        self.assertTrue(robustness["learning_rate_flip_robust"])

    def test_lr_flip_not_detected_when_mixed(self):
        aggs = {
            "buffered": self._agg_with_lr("buffered", False, False, 0.5, "diverging", 1.0),
            "conservative": self._agg_with_lr("conservative", False, True, 1.0, "merging", 1.0),
        }
        robustness = _assess_robustness(aggs)
        self.assertFalse(robustness["learning_rate_flip_robust"])

    def test_speciation_direction_robust_threshold(self):
        aggs = {
            "buffered": self._agg_with_lr("buffered", True, False, 1.0, "diverging", 0.80),
            "conservative": self._agg_with_lr("conservative", False, True, 1.0, "merging", 0.60),
        }
        robustness = _assess_robustness(aggs)
        self.assertTrue(robustness["speciation_direction_robust"]["buffered"])
        self.assertFalse(robustness["speciation_direction_robust"]["conservative"])


# ── Tests: run discovery ──────────────────────────────────────────────────────

class TestDiscoverRuns(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _mkdir(self, profile, seed):
        d = self.base / f"stable_{profile}" / f"seed_{seed}"
        d.mkdir(parents=True)
        return d

    def test_discovers_expected_structure(self):
        self._mkdir("conservative", 42)
        self._mkdir("conservative", 7)
        self._mkdir("buffered", 42)

        found = _discover_runs(self.base, ["conservative", "balanced", "buffered"])
        self.assertIn("conservative", found)
        self.assertEqual(len(found["conservative"]), 2)
        self.assertIn("buffered", found)
        self.assertEqual(len(found["buffered"]), 1)
        self.assertNotIn("balanced", found)

    def test_ignores_non_seed_directories(self):
        self._mkdir("conservative", 42)
        (self.base / "stable_conservative" / "analysis").mkdir(parents=True)
        found = _discover_runs(self.base, ["conservative"])
        self.assertEqual(len(found["conservative"]), 1)

    def test_empty_sweep_dir(self):
        found = _discover_runs(self.base, ["conservative", "balanced", "buffered"])
        self.assertEqual(found, {})

    def test_profiles_filter_respected(self):
        self._mkdir("conservative", 42)
        self._mkdir("balanced", 42)
        self._mkdir("buffered", 42)
        found = _discover_runs(self.base, ["conservative"])
        self.assertIn("conservative", found)
        self.assertNotIn("balanced", found)
        self.assertNotIn("buffered", found)

    def test_seeds_returned_in_numeric_order(self):
        self._mkdir("conservative", 101)
        self._mkdir("conservative", 7)
        self._mkdir("conservative", 42)
        found = _discover_runs(self.base, ["conservative"])
        seeds = [s for s, _ in found["conservative"]]
        self.assertEqual(seeds, sorted(seeds))


# ── Tests: markdown generation ────────────────────────────────────────────────

class TestBuildMarkdown(unittest.TestCase):
    def _minimal_agg(self, profile):
        return {
            "n_seeds": 4,
            "speciation_final": {"mean": 0.70, "variance": 0.002, "ci95": [0.65, 0.75], "n": 4},
            "speciation_mean": {"mean": 0.68, "variance": 0.001, "ci95": [0.64, 0.72], "n": 4},
            "speciation_slope": {"mean": 0.001, "variance": 0.0, "ci95": [0.0, 0.002], "n": 4},
            "speciation_direction_modal": "diverging",
            "speciation_direction_agreement": 0.75,
            "speciation_directions": ["diverging"] * 3 + ["stable"],
            "population_mean": {"mean": 100.0, "variance": 1.0, "ci95": [98.0, 102.0], "n": 4},
            "population_final": {"mean": 95.0, "variance": 2.0, "ci95": [92.0, 98.0], "n": 4},
            "per_gene": {
                "learning_rate": {
                    "mean_pct_shift": 20.0,
                    "variance": 5.0,
                    "ci95": [14.0, 26.0],
                    "n": 4,
                    "sign_agreement": 1.0,
                    "all_positive": True,
                    "all_negative": False,
                },
            },
        }

    def test_markdown_contains_profile_names(self):
        aggs = {
            "conservative": self._minimal_agg("conservative"),
            "buffered": self._minimal_agg("buffered"),
        }
        robustness = {
            "speciation_direction_robust": {"conservative": True, "buffered": True},
            "learning_rate_flip_robust": False,
            "convergent_robust_genes": [],
            "direction_flip_robust_genes": ["learning_rate"],
            "seed_sensitive_genes": [],
        }
        md = _build_markdown(aggs, robustness, {"conservative": 4, "buffered": 4}, {})
        self.assertIn("conservative", md)
        self.assertIn("buffered", md)

    def test_markdown_contains_speciation_section(self):
        aggs = {"buffered": self._minimal_agg("buffered")}
        robustness = {
            "speciation_direction_robust": {"buffered": True},
            "learning_rate_flip_robust": False,
            "convergent_robust_genes": [],
            "direction_flip_robust_genes": [],
            "seed_sensitive_genes": [],
        }
        md = _build_markdown(aggs, robustness, {"buffered": 4}, {})
        self.assertIn("Speciation", md)
        self.assertIn("diverging", md)

    def test_markdown_contains_lr_section(self):
        aggs = {"buffered": self._minimal_agg("buffered")}
        robustness = {
            "speciation_direction_robust": {"buffered": True},
            "learning_rate_flip_robust": True,
            "convergent_robust_genes": [],
            "direction_flip_robust_genes": ["learning_rate"],
            "seed_sensitive_genes": [],
        }
        md = _build_markdown(aggs, robustness, {"buffered": 4}, {})
        self.assertIn("learning_rate", md)

    def test_markdown_is_valid_string(self):
        aggs = {}
        robustness = {
            "speciation_direction_robust": {},
            "learning_rate_flip_robust": False,
            "convergent_robust_genes": [],
            "direction_flip_robust_genes": [],
            "seed_sensitive_genes": [],
        }
        md = _build_markdown(aggs, robustness, {}, {})
        self.assertIsInstance(md, str)
        self.assertGreater(len(md), 0)


# ── Integration-style test: full extract → aggregate → robustness pipeline ───

class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_profile_runs(self, profile, n_seeds, spec_slope, lr_shift):
        for i, seed in enumerate(range(n_seeds)):
            traj = _make_trajectory(
                [0, 50, 100, 150, 200],
                [0.60 + j * spec_slope for j in range(5)],
            )
            snaps = [
                _make_snapshot(0, learning_rate_vals=[0.01] * 5),
                _make_snapshot(200, learning_rate_vals=[0.01 * (1 + lr_shift / 100)] * 5),
            ]
            _write_run_dir(self.base, profile, seed, traj, snaps)

    def test_pipeline_classifies_lr_flip_correctly(self):
        # buffered: rising spec, positive LR shift
        self._write_profile_runs("buffered", 4, 0.02, 23.0)
        # conservative: falling spec, negative LR shift
        self._write_profile_runs("conservative", 4, -0.02, -8.0)

        run_map = _discover_runs(self.base, ["conservative", "buffered"])

        profile_runs = {}
        for profile, seed_dirs in run_map.items():
            runs = []
            for seed, run_dir in seed_dirs:
                metrics = _extract_run_metrics(run_dir)
                if metrics:
                    metrics["seed"] = seed
                    runs.append(metrics)
            profile_runs[profile] = runs

        aggs = {p: _aggregate_profile(runs) for p, runs in profile_runs.items() if runs}
        robustness = _assess_robustness(aggs)

        self.assertTrue(robustness["learning_rate_flip_robust"])
        self.assertTrue(robustness["speciation_direction_robust"].get("buffered", False))
        self.assertTrue(robustness["speciation_direction_robust"].get("conservative", False))


if __name__ == "__main__":
    unittest.main()

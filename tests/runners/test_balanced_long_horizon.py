"""Tests for balanced long-horizon experiment scripts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts import analyze_balanced_long_horizon as lh_analyze  # noqa: E402
from scripts import run_balanced_long_horizon_experiment as lh_runner  # noqa: E402
from scripts import run_stable_profile_seed_sweep as sweep_mod  # noqa: E402


def _make_traj_step_spec(step: int, spec: float, n_alive: int = 50) -> Dict[str, Any]:
    return {"step": step, "speciation_index": spec, "n_alive": n_alive}


def _make_snap(step: int, lr: float = 0.01) -> Dict[str, Any]:
    return {
        "step": step,
        "agents": [{"agent_id": "a1", "chromosome": {"learning_rate": lr, "gamma": 0.99}}],
    }


class _NullLogger:
    def info(self, *_a, **_k):
        pass

    def error(self, *_a, **_k):
        pass


class TestLateWindowHelpers(unittest.TestCase):
    def test_late_window_bounds(self):
        lo, hi = lh_analyze._late_window_bounds(5000.0, 1000)
        self.assertAlmostEqual(hi, 5000.0)
        self.assertAlmostEqual(lo, 4001.0)

    def test_slope_full_vs_late_differ(self):
        # Early decline then rise in late window — full slope differs from late-only.
        traj = []
        for s in range(1, 501):
            traj.append(_make_traj_step_spec(s, 0.8 - 0.0006 * s))
        for s in range(501, 1001):
            traj.append(_make_traj_step_spec(s, 0.5 + 0.0005 * (s - 500)))
        full = lh_analyze._speciation_slope(traj)
        lo, hi = lh_analyze._late_window_bounds(1000.0, 200)
        late = lh_analyze._speciation_slope_window(traj, lo, hi)
        self.assertFalse(math.isnan(full))
        self.assertFalse(math.isnan(late))
        self.assertGreater(late, full)

    def test_empty_window_returns_nan_slope(self):
        traj = [_make_traj_step_spec(1, 0.5), _make_traj_step_spec(2, 0.55)]
        slope = lh_analyze._speciation_slope_window(traj, 100.0, 200.0)
        self.assertTrue(math.isnan(slope))


class TestFinalNClusters(unittest.TestCase):
    def test_from_trajectory_last_quality_row(self):
        traj = [
            _make_traj_step_spec(0, 0.5),
            {"step": 50, "speciation_index": 0.6, "speciation_quality": {"n_clusters": 2}},
            {"step": 100, "speciation_index": 0.7, "speciation_quality": {"n_clusters": 4}},
            {"step": 150, "speciation_index": 0.71},
        ]
        self.assertEqual(lh_analyze._final_n_clusters_from_trajectory(traj), 4)

    def test_lineage_fallback_counts_last_step_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            lineage = run_dir / "cluster_lineage.jsonl"
            rows = [
                {"step": 100, "cluster_id": 1},
                {"step": 100, "cluster_id": 2},
                {"step": 50, "cluster_id": 0},
            ]
            with lineage.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            self.assertEqual(lh_analyze._final_n_clusters_from_lineage(run_dir), 2)


class TestLongHorizonDryRun(unittest.TestCase):
    def test_dry_run_exits_zero(self):
        argv = [
            "run_balanced_long_horizon_experiment.py",
            "--dry-run",
            "--output-dir",
            "/tmp/should_not_create_long_horizon_test",
        ]
        with patch.object(sys, "argv", argv):
            rc = lh_runner.main()
        self.assertEqual(rc, 0)


class TestResumeIntegration(unittest.TestCase):
    def test_resume_skips_when_metadata_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_dir = out / "stable_balanced" / "seed_42"
            run_dir.mkdir(parents=True)
            meta = {
                "num_steps_completed": 5000,
                "num_steps_configured": 5000,
                "final_population": 99,
            }
            with (run_dir / "intrinsic_evolution_metadata.json").open("w", encoding="utf-8") as fh:
                json.dump(meta, fh)

            args = argparse.Namespace(
                resume=True,
                num_steps=5000,
                profiles=["balanced"],
                seeds=[42],
                fail_fast=False,
                environment="development",
                warmup_steps=200,
                snapshot_interval=50,
                mutation_rate=0.15,
                mutation_scale=0.10,
                selection_pressure="low",
                initial_diversity_mutation_rate=1.0,
                initial_diversity_mutation_scale=0.25,
            )

            ok, record = sweep_mod._run_one("balanced", 42, args, out, _NullLogger())
            self.assertTrue(ok)
            self.assertEqual(record["status"], "skipped_done")
            self.assertEqual(record["num_steps_completed"], 5000)


class TestExtractLongHorizonMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_combines_slope_and_clusters(self):
        run_dir = self.base / "stable_balanced" / "seed_7"
        run_dir.mkdir(parents=True)
        traj = []
        for i in range(21):
            step = i * 50
            row = _make_traj_step_spec(step, 0.5 + 0.001 * i)
            if step % 100 == 0:
                row = dict(row)
                row["speciation_quality"] = {"n_clusters": 3}
            traj.append(row)
        snaps = [_make_snap(0), _make_snap(1000)]
        with (run_dir / "intrinsic_gene_trajectory.jsonl").open("w", encoding="utf-8") as fh:
            for r in traj:
                fh.write(json.dumps(r) + "\n")
        with (run_dir / "intrinsic_gene_snapshots.jsonl").open("w", encoding="utf-8") as fh:
            for r in snaps:
                fh.write(json.dumps(r) + "\n")

        m = lh_analyze._extract_long_horizon_metrics(run_dir, window_steps=500)
        self.assertIsNotNone(m)
        self.assertFalse(math.isnan(m["speciation_slope_late"]))
        self.assertEqual(m["n_clusters_final"], 3.0)


if __name__ == "__main__":
    unittest.main()

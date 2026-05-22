"""Unit tests for the shared lineage-metrics helper.

The helper used to live as a private function in
``scripts/compare_crossover_arms.py`` and was reached across the script
boundary via name-mangled private import; promoting it required a stable
contract.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from farm.analysis.lineage_metrics import (
    CLUSTER_LINEAGE_FILENAME,
    lineage_metrics,
)


def _write_lineage(run_dir: Path, rows):
    path = run_dir / CLUSTER_LINEAGE_FILENAME
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


class TestLineageMetrics(unittest.TestCase):
    def test_missing_file_returns_empty_shape(self):
        with TemporaryDirectory() as tmp:
            metrics = lineage_metrics(Path(tmp))
        self.assertEqual(metrics["cluster_count_trace"], [])
        self.assertTrue(math.isnan(metrics["mean_k"]))
        self.assertTrue(math.isnan(metrics["churn_rate"]))

    def test_single_step_returns_nan_churn(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_lineage(
                run_dir,
                [
                    {"step": 0, "cluster_id": "a"},
                    {"step": 0, "cluster_id": "b"},
                ],
            )
            metrics = lineage_metrics(run_dir)
        self.assertEqual(metrics["cluster_count_trace"], [(0, 2)])
        self.assertEqual(metrics["mean_k"], 2.0)
        self.assertTrue(math.isnan(metrics["churn_rate"]))

    def test_churn_rate_counts_disappearing_clusters(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            _write_lineage(
                run_dir,
                [
                    # Step 0: clusters {a, b, c, d}.
                    {"step": 0, "cluster_id": "a"},
                    {"step": 0, "cluster_id": "b"},
                    {"step": 0, "cluster_id": "c"},
                    {"step": 0, "cluster_id": "d"},
                    # Step 1: clusters {b, c} survive; {a, d} died (50% churn).
                    {"step": 1, "cluster_id": "b"},
                    {"step": 1, "cluster_id": "c"},
                ],
            )
            metrics = lineage_metrics(run_dir)
        self.assertEqual(metrics["cluster_count_trace"], [(0, 4), (1, 2)])
        self.assertAlmostEqual(metrics["mean_k"], 3.0)
        self.assertAlmostEqual(metrics["churn_rate"], 0.5)

    def test_malformed_lines_are_skipped(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            path = run_dir / CLUSTER_LINEAGE_FILENAME
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"step": 0, "cluster_id": "a"}),
                        "{not valid json",
                        json.dumps({"step": 0, "cluster_id": "b"}),
                        "",
                        json.dumps({"step": 1}),  # missing cluster_id
                    ]
                ),
                encoding="utf-8",
            )
            metrics = lineage_metrics(run_dir)
        self.assertEqual(metrics["cluster_count_trace"], [(0, 2)])
        self.assertEqual(metrics["mean_k"], 2.0)


if __name__ == "__main__":
    unittest.main()

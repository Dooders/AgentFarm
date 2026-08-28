"""Tests for matrix guest-status watchdog publishing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import publish_matrix_guest_status as watchdog


class PublishMatrixGuestStatusTests(unittest.TestCase):
    def test_parse_resume_aware_launch_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            master = Path(tmp) / "matrix_master.log"
            master.write_text(
                "Launching 96 runs across 1 worker(s) into out ...\n"
                "  [1/96] ok    rc=0 10.0s out/pop-sim__pressure-none__geneflow-mutation seed=42\n"
                "Launching 63 remaining of 96 runs (33 already complete) across 1 worker(s) "
                "into experiments/intrinsic_matrix ...\n",
                encoding="utf-8",
            )
            n_ok, n_fail, n_killed, total, workers, skipped, recent = (
                watchdog._parse_latest_launch(master)
            )
            self.assertEqual(n_ok, 0)
            self.assertEqual(n_fail, 0)
            self.assertEqual(n_killed, 0)
            self.assertEqual(total, 96)
            self.assertEqual(workers, 1)
            self.assertEqual(skipped, 33)
            self.assertEqual(recent, [])

    def test_main_uses_disk_completed_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            master = root / "matrix_master.log"
            master.write_text(
                "Launching 63 remaining of 96 runs (33 already complete) across 1 worker(s) "
                "into out ...\n",
                encoding="utf-8",
            )
            seed_dir = (
                root
                / "pop-sim__pressure-high__geneflow-mutation"
                / "stable_balanced"
                / "seed_512"
            )
            seed_dir.mkdir(parents=True)
            (seed_dir / "intrinsic_evolution_metadata.json").write_text(
                json.dumps({"num_steps_completed": 10000}),
                encoding="utf-8",
            )

            with patch.object(watchdog, "publish_live_status") as publish:
                rc = watchdog.main(
                    [
                        "--output-dir",
                        str(root),
                        "--total-jobs",
                        "96",
                        "--num-steps",
                        "10000",
                        "--no-guest-attributes",
                    ]
                )
            self.assertEqual(rc, 0)
            kwargs = publish.call_args.kwargs
            self.assertEqual(kwargs["n_ok"], 33)  # max(disk=1, skipped+log=33)
            self.assertEqual(kwargs["total_jobs"], 96)
            self.assertIn("disk_ok=1", kwargs["note"])


if __name__ == "__main__":
    unittest.main()

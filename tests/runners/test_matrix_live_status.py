"""Tests for matrix live-status snapshots (SSH-independent monitoring)."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from farm.runners.matrix_live_status import (
    LIVE_STATUS_FILENAME,
    build_live_status,
    parse_seed_log_progress,
    publish_live_status,
    scan_active_runs,
    write_live_status,
)


class TestParseSeedLogProgress(unittest.TestCase):
    def test_parses_latest_tqdm_line(self):
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "seed_42.log"
            # Carriage returns mimic live tqdm rewrites.
            log.write_text(
                "noise\r"
                "Simulation progress:  10%|█         | 100/1000 [00:01<00:09]\r"
                "Simulation progress:  42%|████      | 420/1000 [00:04<00:05]\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_seed_log_progress(log), (42, 420, 1000))

    def test_missing_file_returns_none(self):
        self.assertIsNone(parse_seed_log_progress(Path("/no/such/seed.log")))


class TestBuildAndWriteLiveStatus(unittest.TestCase):
    def test_write_atomic_snapshot_and_scan_running(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "matrix"
            cell = out / "pop-sim__pressure-low__geneflow-mutation"
            cell.mkdir(parents=True)
            log = cell / "seed_7.log"
            log.write_text(
                "Simulation progress:  55%|█████     | 550/1000 [00:10<00:08]\n",
                encoding="utf-8",
            )
            status = build_live_status(
                output_dir=out,
                total_jobs=96,
                n_ok=3,
                n_fail=1,
                workers=7,
                recent=[{"status": "ok", "seed": 42}],
                note="unit-test",
            )
            self.assertEqual(status.n_done, 4)
            self.assertEqual(status.n_pending, 92)
            self.assertEqual(len(status.running), 1)
            self.assertEqual(status.running[0].seed, 7)
            self.assertEqual(status.running[0].percent, 55)

            path = write_live_status(out, status)
            self.assertEqual(path.name, LIVE_STATUS_FILENAME)
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["n_ok"], 3)
            self.assertEqual(loaded["running"][0]["current_step"], 550)

    def test_publish_skips_guest_attrs_when_disabled(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch(
                "farm.runners.matrix_live_status.publish_guest_attribute"
            ) as publish:
                status = publish_live_status(
                    out,
                    total_jobs=2,
                    n_ok=0,
                    n_fail=0,
                    workers=1,
                    guest_attributes=False,
                )
                publish.assert_not_called()
                self.assertTrue((out / LIVE_STATUS_FILENAME).is_file())
                self.assertEqual(status.total_jobs, 2)

    def test_guest_attribute_payload_includes_recent_errors(self):
        from farm.runners.matrix_live_status import (
            MatrixLiveStatus,
            publish_guest_attribute,
        )

        status = MatrixLiveStatus(
            updated_at="2026-01-01T00:00:00+00:00",
            output_dir="/tmp",
            total_jobs=2,
            n_ok=0,
            n_fail=1,
            n_done=1,
            n_pending=1,
            workers=1,
            recent=[
                {
                    "status": "error",
                    "returncode": 1,
                    "seed": 42,
                    "population": "sim",
                    "pressure": "none",
                    "gene_flow": "mutation",
                    "elapsed_seconds": 12.3,
                    "log_tail": "Traceback\nRuntimeError: boom",
                }
            ],
            note="job_complete",
        )
        with patch("farm.runners.matrix_live_status._on_gce", return_value=True):
            with patch("urllib.request.urlopen") as urlopen:
                urlopen.return_value.__enter__.return_value.status = 200
                self.assertTrue(publish_guest_attribute(status))
                request = urlopen.call_args[0][0]
                body = json.loads(request.data.decode("utf-8"))
                self.assertEqual(body["note"], "job_complete")
                self.assertEqual(body["recent"][0]["rc"], 1)
                self.assertIn("RuntimeError", body["recent"][0]["err"])

    def test_scan_skips_stale_completed_logs(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            cell = out / "pop-dev__pressure-none__geneflow-mutation"
            cell.mkdir(parents=True)
            log = cell / "seed_1.log"
            log.write_text(
                "Simulation progress: 100%|██████████| 1000/1000 [00:01<00:00]\n",
                encoding="utf-8",
            )
            # Make mtime look old so 100% logs are ignored.
            os.utime(log, (0, 0))
            self.assertEqual(scan_active_runs(out), [])


if __name__ == "__main__":
    unittest.main()

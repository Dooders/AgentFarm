"""Resume skip helpers for the intrinsic-evolution matrix orchestrator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_intrinsic_evolution_matrix import (
    Job,
    is_job_complete,
    partition_resume_jobs,
)


class IntrinsicEvolutionMatrixResumeTests(unittest.TestCase):
    def test_is_job_complete_requires_metadata_at_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = root / "pop-sim__pressure-none__geneflow-mutation"
            seed_dir = cell / "stable_balanced" / "seed_42"
            seed_dir.mkdir(parents=True)
            self.assertFalse(is_job_complete(cell, 42, 10000))

            (seed_dir / "intrinsic_evolution_metadata.json").write_text(
                json.dumps({"num_steps_completed": 5000}),
                encoding="utf-8",
            )
            self.assertFalse(is_job_complete(cell, 42, 10000))

            (seed_dir / "intrinsic_evolution_metadata.json").write_text(
                json.dumps({"num_steps_completed": 10000}),
                encoding="utf-8",
            )
            self.assertTrue(is_job_complete(cell, 42, 10000))

    def test_partition_resume_jobs_skips_complete_only_when_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job_done = Job("sim", "high", "mutation", 512)
            job_todo = Job("sim", "none", "mutation", 42)
            done_cell = root / job_done.cell_name
            seed_dir = done_cell / "stable_balanced" / f"seed_{job_done.seed}"
            seed_dir.mkdir(parents=True)
            (seed_dir / "intrinsic_evolution_metadata.json").write_text(
                json.dumps({"num_steps_completed": 10000, "final_population": 10}),
                encoding="utf-8",
            )

            skipped, pending = partition_resume_jobs(
                [job_done, job_todo],
                root,
                10000,
                resume=False,
            )
            self.assertEqual(skipped, [])
            self.assertEqual(pending, [job_done, job_todo])

            skipped, pending = partition_resume_jobs(
                [job_done, job_todo],
                root,
                10000,
                resume=True,
            )
            self.assertEqual(len(skipped), 1)
            self.assertEqual(skipped[0].status, "ok")
            self.assertEqual(skipped[0].job["seed"], 512)
            self.assertEqual(pending, [job_todo])


if __name__ == "__main__":
    unittest.main()

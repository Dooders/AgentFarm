"""Tests for intrinsic-evolution mid-run checkpointing."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from farm.runners.intrinsic_evolution_checkpoint import (
    CHECKPOINT_VERSION,
    CheckpointMeta,
    clear_checkpoint,
    has_resumable_checkpoint,
    read_checkpoint_meta,
    save_checkpoint,
)
from scripts.run_stable_profile_seed_sweep import _prepare_run_dir_for_resume


class TestCheckpointMeta(unittest.TestCase):
    def test_save_and_detect_resumable(self):
        with TemporaryDirectory() as tmp:
            payload = {
                "version": CHECKPOINT_VERSION,
                "logical_step": 250,
                "num_steps_configured": 1000,
                "total_sim_steps": 1200,
                "effective_warmup": 200,
                "agents": [],
                "resources": [],
            }
            meta = save_checkpoint(tmp, payload)
            self.assertEqual(meta.logical_step, 250)
            self.assertTrue(has_resumable_checkpoint(tmp, num_steps=1000))
            loaded = read_checkpoint_meta(tmp)
            assert loaded is not None
            self.assertEqual(loaded.logical_step, 250)

    def test_complete_relative_to_warmup_not_resumable(self):
        with TemporaryDirectory() as tmp:
            payload = {
                "version": CHECKPOINT_VERSION,
                "logical_step": 1200,
                "num_steps_configured": 1000,
                "total_sim_steps": 1200,
                "effective_warmup": 200,
                "agents": [],
                "resources": [],
            }
            save_checkpoint(tmp, payload)
            # post-warmup completed == 1000 → not resumable as partial
            self.assertFalse(has_resumable_checkpoint(tmp, num_steps=1000))

    def test_clear_checkpoint(self):
        with TemporaryDirectory() as tmp:
            save_checkpoint(
                tmp,
                {
                    "version": CHECKPOINT_VERSION,
                    "logical_step": 10,
                    "num_steps_configured": 100,
                    "total_sim_steps": 100,
                    "effective_warmup": 0,
                    "agents": [],
                    "resources": [],
                },
            )
            clear_checkpoint(tmp)
            self.assertIsNone(read_checkpoint_meta(tmp))
            self.assertFalse(has_resumable_checkpoint(tmp, num_steps=100))


class TestPrepareRunDirKeepsCheckpoint(unittest.TestCase):
    def test_keeps_incomplete_with_checkpoint(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_7"
            run_dir.mkdir()
            (run_dir / "partial.db").write_text("x", encoding="utf-8")
            save_checkpoint(
                str(run_dir),
                {
                    "version": CHECKPOINT_VERSION,
                    "logical_step": 40,
                    "num_steps_configured": 100,
                    "total_sim_steps": 100,
                    "effective_warmup": 0,
                    "agents": [],
                    "resources": [],
                },
            )
            _prepare_run_dir_for_resume(run_dir, num_steps=100, resume=True)
            self.assertTrue(run_dir.exists())
            self.assertTrue((run_dir / "partial.db").exists())

    def test_still_removes_incomplete_without_checkpoint(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_9"
            run_dir.mkdir()
            (run_dir / "partial.db").write_text("x", encoding="utf-8")
            _prepare_run_dir_for_resume(run_dir, num_steps=100, resume=True)
            self.assertFalse(run_dir.exists())

    def test_keeps_complete_metadata(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "seed_42"
            run_dir.mkdir()
            meta = {"num_steps_completed": 100, "final_population": 12}
            (run_dir / "intrinsic_evolution_metadata.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            _prepare_run_dir_for_resume(run_dir, num_steps=100, resume=True)
            self.assertTrue(run_dir.exists())


class TestCheckpointMetaRoundTrip(unittest.TestCase):
    def test_from_dict(self):
        meta = CheckpointMeta.from_dict(
            {
                "version": 1,
                "logical_step": 5,
                "num_steps_configured": 50,
                "total_sim_steps": 50,
                "effective_warmup": 0,
            }
        )
        self.assertEqual(meta.to_dict()["logical_step"], 5)


if __name__ == "__main__":
    unittest.main()

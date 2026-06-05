"""Tests for scripts/analyze_early_life_fitness.py helper logic.

Covers the parts most likely to break silently: parent-id parsing, the
robustness verdict, the JSON portability sanitizer, and the per-run
extraction (offspring selection, right-censoring, survivor-conditioned reward
vs. action cohorts, and parent-anchored reward gap).
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.analyze_early_life_fitness import (  # noqa: E402
    _extract_run_early_life,
    _json_safe,
    _parent_of,
    _verdict,
)


class TestParentOf(unittest.TestCase):
    def test_normal(self):
        self.assertEqual(_parent_of("P:7"), "P")

    def test_founder(self):
        self.assertIsNone(_parent_of("::1"))

    def test_uuid_parent(self):
        self.assertEqual(_parent_of("abc-123-def:42"), "abc-123-def")

    def test_none_and_empty(self):
        self.assertIsNone(_parent_of(None))
        self.assertIsNone(_parent_of(""))


class TestVerdict(unittest.TestCase):
    def test_too_few_samples(self):
        out = _verdict([0.5])
        self.assertEqual(out["n"], 1)
        self.assertFalse(out["robust"])
        # Single sample cannot exclude zero.
        self.assertFalse(out["ci_excludes_zero"])

    def test_robust_effect(self):
        # Tightly clustered, same-sign deltas -> CI excludes zero, full agreement.
        out = _verdict([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
        self.assertTrue(out["ci_excludes_zero"])
        self.assertEqual(out["sign_agreement"], 1.0)
        self.assertTrue(out["robust"])

    def test_non_robust_when_straddling_zero(self):
        out = _verdict([1.0, -1.0, 0.5, -0.5, 0.2, -0.2])
        self.assertFalse(out["ci_excludes_zero"])
        self.assertFalse(out["robust"])

    def test_nans_dropped(self):
        out = _verdict([float("nan"), 1.0])
        self.assertEqual(out["n"], 1)


class TestJsonSafe(unittest.TestCase):
    def test_replaces_non_finite(self):
        cleaned = _json_safe(
            {"a": float("nan"), "b": [1.0, float("inf"), -float("inf")], "c": "x"}
        )
        self.assertIsNone(cleaned["a"])
        self.assertEqual(cleaned["b"], [1.0, None, None])
        self.assertEqual(cleaned["c"], "x")

    def test_round_trips_in_strict_json(self):
        payload = _json_safe({"ci": [float("nan"), 0.5]})
        # allow_nan=False raises on NaN/Infinity tokens; this must not raise.
        text = json.dumps(payload, allow_nan=False)
        self.assertEqual(json.loads(text), {"ci": [None, 0.5]})


def _build_db(path: Path) -> None:
    """Create a tiny simulation DB exercising the extraction edge cases.

    Layout (warmup=2, last_step=20):
    - ``P`` founder (genome ``::1``), parent of all offspring.
    - ``A`` born at 3, survives the whole run.
    - ``B`` born at 3, dies at 6 (reaches age 3 but not age 5).
    - ``C`` born at 18, too late to reach age 3 before the run ends (censored).
    """
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE agents (agent_id TEXT, birth_time INT, death_time INT, "
        "genome_id TEXT)"
    )
    cur.execute(
        "CREATE TABLE agent_states (agent_id TEXT, step_number INT, "
        "total_reward REAL, resource_level REAL, current_health REAL, age INT)"
    )
    cur.execute(
        "CREATE TABLE agent_actions (agent_id TEXT, step_number INT, reward REAL)"
    )

    cur.executemany(
        "INSERT INTO agents VALUES (?, ?, ?, ?)",
        [
            ("P", 0, None, "::1"),
            ("A", 3, None, "P:1"),
            ("B", 3, 6, "P:2"),
            ("C", 18, None, "P:3"),
        ],
    )

    states = []
    # Parent P: cumulative reward = 2 * step, for steps 0..20.
    for step in range(0, 21):
        states.append(("P", step, 2.0 * step, 10.0, 100.0, step))
    # A: cumulative reward = 1 * (step - birth), steps 3..20.
    for step in range(3, 21):
        states.append(("A", step, 1.0 * (step - 3), 10.0, 100.0, step - 3))
    # B: cumulative reward = 0.5 * (step - birth), steps 3..6 (dies at 6).
    for step in range(3, 7):
        states.append(("B", step, 0.5 * (step - 3), 10.0, 100.0, step - 3))
    # C: present but never reaches a scored age.
    for step in range(18, 21):
        states.append(("C", step, 1.0 * (step - 18), 10.0, 100.0, step - 18))
    cur.executemany("INSERT INTO agent_states VALUES (?, ?, ?, ?, ?, ?)", states)

    actions = [
        ("A", 3, 0.135),
        ("A", 4, -0.1),
        ("A", 5, 0.135),
        ("A", 6, 0.135),
        ("A", 7, 0.135),
        ("B", 3, 0.135),
        ("B", 4, 0.135),
        ("B", 5, 0.135),
    ]
    cur.executemany("INSERT INTO agent_actions VALUES (?, ?, ?)", actions)
    con.commit()
    con.close()


class TestExtractRunEarlyLife(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "simulation_sim_test.db"
        _build_db(self.db_path)
        self.data = _extract_run_early_life(self.db_path, warmup=2, ages=[3, 5])

    def tearDown(self):
        self._tmp.cleanup()

    def test_offspring_selection_excludes_founder(self):
        # P (born 0 <= warmup) is excluded; A, B, C are offspring.
        self.assertEqual(self.data["n_offspring"], 3)
        self.assertEqual(self.data["last_step"], 20)

    def test_age3_censors_late_offspring(self):
        age3 = self.data["per_age"][3]
        # C (born 18) cannot reach age 3 before step 20, so only A and B count.
        self.assertEqual(age3["n_uncensored"], 2.0)
        self.assertEqual(age3["survival_rate"], 1.0)
        self.assertEqual(age3["n_reached"], 2.0)
        # A snap=3.0, B snap=1.5 -> mean 2.25.
        self.assertAlmostEqual(age3["rl_reward_at_age"], 2.25)
        self.assertAlmostEqual(age3["resource_at_age"], 10.0)

    def test_age3_parent_gap(self):
        # Parent window [3, 6] = 12 - 6 = 6. |3-6|=3, |1.5-6|=4.5 -> mean 3.75.
        self.assertAlmostEqual(self.data["per_age"][3]["parent_reward_gap"], 3.75)

    def test_age5_cohorts_differ(self):
        age5 = self.data["per_age"][5]
        # B dies at 6: reaches age 3 but not age 5.
        self.assertEqual(age5["survival_rate"], 0.5)
        # Reward cohort is survivors only (A); action cohort includes B too.
        self.assertEqual(age5["n_reached"], 1.0)
        self.assertEqual(age5["n_acted"], 2.0)
        self.assertAlmostEqual(age5["rl_reward_at_age"], 5.0)

    def test_curves_present(self):
        self.assertTrue(self.data["survival_curve"])
        self.assertTrue(self.data["rl_reward_curve"])

    def test_no_offspring_returns_none(self):
        # With a warmup past every birth, there are no offspring.
        self.assertIsNone(_extract_run_early_life(self.db_path, warmup=100, ages=[3]))


if __name__ == "__main__":
    unittest.main()

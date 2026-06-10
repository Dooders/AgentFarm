"""
Cross-process determinism tests.

In-process A/B comparisons cannot detect nondeterminism that depends on
per-process interpreter state: both runs share the same hash randomization seed
(PYTHONHASHSEED), import order, and warmed caches. These tests run each
simulation in a fresh subprocess - deliberately with *different* PYTHONHASHSEED
values - so that any dependence on dict/set hash ordering or other per-process
state shows up as a state mismatch.

The subprocesses do not pre-seed RNGs; reproducibility must come entirely from
``run_simulation(seed=...)``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.determinism]

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER_SCRIPT = REPO_ROOT / "tests" / "helpers" / "run_sim_snapshot.py"
SUBPROCESS_TIMEOUT_SECONDS = 300


def _run_simulation_in_subprocess(output_path: Path, seed: int, steps: int, hash_seed: str) -> dict:
    """Run one simulation in a fresh interpreter and return its final state snapshot."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hash_seed
    env["PYTHONPATH"] = str(REPO_ROOT)

    result = subprocess.run(
        [
            sys.executable,
            str(HELPER_SCRIPT),
            "--environment", "testing",
            "--steps", str(steps),
            "--seed", str(seed),
            "--output", str(output_path),
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"Simulation subprocess failed (PYTHONHASHSEED={hash_seed}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return json.loads(output_path.read_text())


def test_cross_process_repeatability_under_hash_randomization(tmp_path):
    """Same seed in two fresh processes with different PYTHONHASHSEED must match."""
    seed = 42
    steps = 25

    state1 = _run_simulation_in_subprocess(tmp_path / "run1.json", seed=seed, steps=steps, hash_seed="1")
    state2 = _run_simulation_in_subprocess(tmp_path / "run2.json", seed=seed, steps=steps, hash_seed="2")

    assert state1 == state2, (
        "Simulation results differ across processes with different PYTHONHASHSEED values - "
        "behavior likely depends on dict/set hash ordering or other per-process state"
    )


def test_cross_process_seed_sensitivity(tmp_path):
    """Different seeds in fresh processes must produce different results."""
    steps = 25

    state1 = _run_simulation_in_subprocess(tmp_path / "seed_a.json", seed=42, steps=steps, hash_seed="0")
    state2 = _run_simulation_in_subprocess(tmp_path / "seed_b.json", seed=1042, steps=steps, hash_seed="0")

    assert state1 != state2, (
        "Two different seeds produced identical final states - "
        "the seed is likely not plumbed through to the simulation"
    )

#!/usr/bin/env python3
"""
Subprocess helper for cross-process determinism tests.

Runs one simulation with the given seed (relying solely on production seeding via
``run_simulation(seed=...)``) and writes a JSON snapshot of the final state to the
given output path. Invoked in a fresh interpreter so that per-process state
(PYTHONHASHSEED, import order, module caches) cannot be shared between runs.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one simulation and snapshot its final state")
    parser.add_argument("--environment", type=str, default="testing")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=str, required=True, help="Path to write the JSON state snapshot")
    args = parser.parse_args()

    from farm.config import SimulationConfig
    from farm.core.simulation import run_simulation
    from tests.test_deterministic import capture_simulation_state

    config = SimulationConfig.from_centralized_config(environment=args.environment)
    config.seed = args.seed
    config.database.use_in_memory_db = True
    config.database.persist_db_on_completion = False

    with tempfile.TemporaryDirectory() as temp_dir:
        environment = run_simulation(
            num_steps=args.steps,
            config=config,
            path=temp_dir,
            save_config=False,
            seed=args.seed,
        )
        try:
            state = capture_simulation_state(environment)
        finally:
            environment.cleanup()

    Path(args.output).write_text(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

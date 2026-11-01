#!/usr/bin/env python3
"""Smoke-test an installed ``agentfarm`` wheel.

Run from a directory that is *not* the repository checkout so that ``import
farm`` resolves to the installed distribution rather than the source tree.
Verifies that:

1. the package and its main subsystems import,
2. the bundled YAML configuration ships inside the wheel, and
3. a short simulation runs end to end using only installed files.

Exit code is non-zero on any failure.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from importlib.resources import files
from pathlib import Path

SUBSYSTEMS = (
    "farm.core.simulation",
    "farm.core.environment",
    "farm.config",
    "farm.database.database",
    "farm.api.server",
    "farm.runners.experiment_runner",
)

STEPS = 5
SEED = 0


def _assert_installed_distribution(package_path: Path) -> None:
    if "site-packages" not in package_path.parts and "dist-packages" not in package_path.parts:
        sys.exit(
            f"farm imported from {package_path}, which is not an installed distribution. "
            "Run this script from outside the repository so the installed wheel is tested."
        )


def main() -> None:
    import farm

    package_path = Path(farm.__file__).resolve().parent
    _assert_installed_distribution(package_path)
    print(f"farm {farm.__version__} imported from {package_path}")

    for name in SUBSYSTEMS:
        importlib.import_module(name)
        print(f"  imported {name}")

    config_root = files("farm.config")
    for rel in ("default.yaml", "environments/testing.yaml", "schema.json"):
        if not (config_root / rel).is_file():
            sys.exit(f"packaged config file missing from wheel: farm/config/{rel}")
    print("  packaged configuration files present")

    from farm.config import SimulationConfig
    from farm.core.simulation import run_simulation

    config = SimulationConfig.from_centralized_config(
        environment="testing",
        config_dir=str(config_root),
        use_cache=False,
    )

    with tempfile.TemporaryDirectory() as tmp:
        env = run_simulation(
            num_steps=STEPS,
            config=config,
            path=tmp,
            seed=SEED,
            disable_console_logging=True,
        )
        produced = sorted(os.listdir(tmp))
        if not produced:
            sys.exit("simulation produced no output files")
        print(f"  ran {STEPS} steps; time={env.time}; outputs={produced}")

    print("package smoke test passed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a tiny grid simulation and write the intro-page GIF.

The committed animation lives at ``docs/assets/intro-grid-example.gif``.
Regenerate it with:

    PYTHONHASHSEED=0 python scripts/render_intro_grid_example.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Match run_simulation.py: pin hash seed before the heavy imports.
if os.environ.get("PYTHONHASHSEED") != "0":
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from farm.config import SimulationConfig
from farm.config.config import DatabaseConfig
from farm.core.animation import write_grid_gif
from farm.core.simulation import run_simulation
from farm.utils.logging import get_logger

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "assets" / "intro-grid-example.gif"

INTRO_WIDTH = 16
INTRO_HEIGHT = 16
INTRO_STEPS = 60
INTRO_SEED = 0
INTRO_SIM_ID = "intro_grid_example"


def build_intro_config() -> SimulationConfig:
    """Small, readable world: short walks, nearby food, few agents."""
    config = SimulationConfig.from_centralized_config(environment="testing")
    config.environment.width = INTRO_WIDTH
    config.environment.height = INTRO_HEIGHT
    config.environment.use_bilinear_interpolation = False
    config.population.system_agents = 3
    config.population.independent_agents = 3
    config.population.control_agents = 2
    config.population.max_population = 16
    config.resources.initial_resources = 18
    config.resources.resource_regen_rate = 0.2
    config.resources.resource_regen_amount = 2
    config.agent_behavior.max_movement = 2
    config.agent_behavior.gathering_range = 2
    config.agent_behavior.perception_radius = 3
    config.agent_behavior.social_range = 3
    config.combat.attack_range = 2.0
    config.database = DatabaseConfig(
        use_in_memory_db=False,
        persist_db_on_completion=True,
    )
    config.seed = INTRO_SEED
    config.simulation_steps = INTRO_STEPS
    config.max_steps = INTRO_STEPS
    return config


def render_intro_gif(output_path: Path, steps: int = INTRO_STEPS) -> Path:
    """Run the intro simulation and write ``output_path``."""
    config = build_intro_config()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        env = run_simulation(
            num_steps=steps,
            config=config,
            path=temp_dir,
            save_config=True,
            seed=INTRO_SEED,
            simulation_id=INTRO_SIM_ID,
            disable_console_logging=True,
        )
        try:
            db_path = Path(temp_dir) / f"simulation_{INTRO_SIM_ID}.db"
            if not db_path.exists():
                raise FileNotFoundError(f"Expected simulation database at {db_path}")
            write_grid_gif(
                db_path,
                output_path,
                INTRO_WIDTH,
                INTRO_HEIGHT,
                fps=6,
                skip_frames=2,
                title_prefix="A small grid world",
            )
        finally:
            env.cleanup()

    logger.info("intro_grid_gif_ready", path=str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the gentle-introduction grid GIF.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"GIF destination (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--steps", type=int, default=INTRO_STEPS, help="Simulation steps to run")
    args = parser.parse_args()
    path = render_intro_gif(args.output, steps=args.steps)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

"""Shared CLI plumbing for the P2–P4 warm-start tuning knobs.

The intrinsic-evolution runner scripts (`run_intrinsic_evolution_experiment`,
`run_stable_profile_seed_sweep`, and the `run_inheritance_mode_ab` orchestrator)
all expose the same four warm-start tuning flags and forward them identically to
:class:`farm.runners.intrinsic_evolution_experiment.IntrinsicEvolutionPolicy`.
Centralising the argparse definitions and the args→kwargs extraction here keeps
the flag names, defaults, and help text in one place so a new knob is added once.

Range validation intentionally lives on ``IntrinsicEvolutionPolicy`` (a single
source of truth) rather than being duplicated into each parser.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict

from farm.core.policy_inheritance import (
    P2_PLASTICITY_DAMPING,
    P3_REPLAY_BUFFER_LIMIT,
    P4_BLEND_ALPHA,
    P4_FITNESS_GATE_MIN_RESOURCES,
)


def add_warmstart_tuning_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the four P2–P4 warm-start tuning flags on ``parser``."""
    parser.add_argument(
        "--warmstart-plasticity-damping",
        type=float,
        default=P2_PLASTICITY_DAMPING,
        help="P2 damping factor in (0, 1] for child LR/epsilon.",
    )
    parser.add_argument(
        "--warmstart-replay-buffer-limit",
        type=int,
        default=P3_REPLAY_BUFFER_LIMIT,
        help="P3 cap (>= 1) on replay transitions transferred to the child.",
    )
    parser.add_argument(
        "--warmstart-blend-alpha",
        type=float,
        default=P4_BLEND_ALPHA,
        help="P4 blend coefficient in [0, 1] weighting the parent's policy.",
    )
    parser.add_argument(
        "--warmstart-fitness-gate-min-resources",
        type=float,
        default=P4_FITNESS_GATE_MIN_RESOURCES,
        help="P4 minimum parent resource level (>= 0) to clear the fitness gate.",
    )


def warmstart_tuning_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Extract the warm-start tuning values from ``args`` as policy kwargs.

    Uses ``getattr`` with the module defaults so callers that build a partial
    ``Namespace`` (or omit the flags entirely) still get the documented values.
    """
    return {
        "warmstart_plasticity_damping": float(
            getattr(args, "warmstart_plasticity_damping", P2_PLASTICITY_DAMPING)
        ),
        "warmstart_replay_buffer_limit": int(
            getattr(args, "warmstart_replay_buffer_limit", P3_REPLAY_BUFFER_LIMIT)
        ),
        "warmstart_blend_alpha": float(
            getattr(args, "warmstart_blend_alpha", P4_BLEND_ALPHA)
        ),
        "warmstart_fitness_gate_min_resources": float(
            getattr(
                args,
                "warmstart_fitness_gate_min_resources",
                P4_FITNESS_GATE_MIN_RESOURCES,
            )
        ),
    }

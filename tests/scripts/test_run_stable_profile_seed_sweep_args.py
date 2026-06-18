"""CLI parser tests for ``scripts/run_stable_profile_seed_sweep.py``.

These cover the smaller surface that the inheritance-mode A/B harness adds
without requiring the full intrinsic-evolution simulation to spin up.
"""

from __future__ import annotations

import os
import sys
import unittest

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pytest  # noqa: E402

from farm.core.policy_inheritance import (  # noqa: E402
    P2_PLASTICITY_DAMPING,
    P3_REPLAY_BUFFER_LIMIT,
    P4_BLEND_ALPHA,
    P4_FITNESS_GATE_MIN_RESOURCES,
)
from scripts.run_stable_profile_seed_sweep import (  # noqa: E402
    INHERITANCE_MODES,
    _build_parser,
    _inheritance_settings_dict,
)


def _expected(mode: str, **overrides) -> dict:
    base = {
        "inheritance_mode": mode,
        "warmstart_plasticity_damping": P2_PLASTICITY_DAMPING,
        "warmstart_replay_buffer_limit": P3_REPLAY_BUFFER_LIMIT,
        "warmstart_blend_alpha": P4_BLEND_ALPHA,
        "warmstart_fitness_gate_min_resources": P4_FITNESS_GATE_MIN_RESOURCES,
    }
    base.update(overrides)
    return base


class TestInheritanceModeFlag(unittest.TestCase):
    def test_default_is_baldwinian(self):
        args = _build_parser().parse_args([])
        self.assertEqual(args.inheritance_mode, "baldwinian")
        self.assertEqual(_inheritance_settings_dict(args), _expected("baldwinian"))

    def test_lamarckian_accepted(self):
        args = _build_parser().parse_args(["--inheritance-mode", "lamarckian"])
        self.assertEqual(args.inheritance_mode, "lamarckian")
        self.assertEqual(_inheritance_settings_dict(args), _expected("lamarckian"))

    def test_invalid_value_rejected(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--inheritance-mode", "epigenetic"])

    def test_choices_match_module_constant(self):
        self.assertEqual(set(INHERITANCE_MODES), {"baldwinian", "lamarckian", "p2", "p3", "p4"})

    def test_p2_accepted(self):
        args = _build_parser().parse_args(["--inheritance-mode", "p2"])
        self.assertEqual(args.inheritance_mode, "p2")
        self.assertEqual(_inheritance_settings_dict(args), _expected("p2"))

    def test_p3_accepted(self):
        args = _build_parser().parse_args(["--inheritance-mode", "p3"])
        self.assertEqual(args.inheritance_mode, "p3")
        self.assertEqual(_inheritance_settings_dict(args), _expected("p3"))

    def test_p4_accepted(self):
        args = _build_parser().parse_args(["--inheritance-mode", "p4"])
        self.assertEqual(args.inheritance_mode, "p4")
        self.assertEqual(_inheritance_settings_dict(args), _expected("p4"))


class TestWarmstartTuningFlags(unittest.TestCase):
    def test_defaults_match_module_constants(self):
        args = _build_parser().parse_args([])
        self.assertEqual(args.warmstart_plasticity_damping, P2_PLASTICITY_DAMPING)
        self.assertEqual(args.warmstart_replay_buffer_limit, P3_REPLAY_BUFFER_LIMIT)
        self.assertEqual(args.warmstart_blend_alpha, P4_BLEND_ALPHA)
        self.assertEqual(
            args.warmstart_fitness_gate_min_resources,
            P4_FITNESS_GATE_MIN_RESOURCES,
        )

    def test_overrides_are_parsed_and_surfaced(self):
        args = _build_parser().parse_args(
            [
                "--inheritance-mode", "p4",
                "--warmstart-plasticity-damping", "0.25",
                "--warmstart-replay-buffer-limit", "64",
                "--warmstart-blend-alpha", "0.75",
                "--warmstart-fitness-gate-min-resources", "5.0",
            ]
        )
        self.assertEqual(
            _inheritance_settings_dict(args),
            _expected(
                "p4",
                warmstart_plasticity_damping=0.25,
                warmstart_replay_buffer_limit=64,
                warmstart_blend_alpha=0.75,
                warmstart_fitness_gate_min_resources=5.0,
            ),
        )


if __name__ == "__main__":
    unittest.main()

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

from scripts.run_stable_profile_seed_sweep import (  # noqa: E402
    INHERITANCE_MODES,
    _build_parser,
    _inheritance_settings_dict,
)


class TestInheritanceModeFlag(unittest.TestCase):
    def test_default_is_baldwinian(self):
        args = _build_parser().parse_args([])
        self.assertEqual(args.inheritance_mode, "baldwinian")
        self.assertEqual(
            _inheritance_settings_dict(args),
            {"inheritance_mode": "baldwinian"},
        )

    def test_lamarckian_accepted(self):
        args = _build_parser().parse_args(["--inheritance-mode", "lamarckian"])
        self.assertEqual(args.inheritance_mode, "lamarckian")
        self.assertEqual(
            _inheritance_settings_dict(args),
            {"inheritance_mode": "lamarckian"},
        )

    def test_invalid_value_rejected(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--inheritance-mode", "epigenetic"])

    def test_choices_match_module_constant(self):
        self.assertEqual(set(INHERITANCE_MODES), {"baldwinian", "lamarckian", "p2", "p3", "p4"})

    def test_p2_accepted(self):
        args = _build_parser().parse_args(["--inheritance-mode", "p2"])
        self.assertEqual(args.inheritance_mode, "p2")
        self.assertEqual(
            _inheritance_settings_dict(args),
            {"inheritance_mode": "p2"},
        )

    def test_p3_accepted(self):
        args = _build_parser().parse_args(["--inheritance-mode", "p3"])
        self.assertEqual(args.inheritance_mode, "p3")
        self.assertEqual(
            _inheritance_settings_dict(args),
            {"inheritance_mode": "p3"},
        )

    def test_p4_accepted(self):
        args = _build_parser().parse_args(["--inheritance-mode", "p4"])
        self.assertEqual(args.inheritance_mode, "p4")
        self.assertEqual(
            _inheritance_settings_dict(args),
            {"inheritance_mode": "p4"},
        )


if __name__ == "__main__":
    unittest.main()

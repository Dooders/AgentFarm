"""Tests for the ``scripts/run_inheritance_mode_ab.py`` CLI / preset table.

Covers the subset that can be exercised without a full simulation: parser
defaults, choice validation, and the ARM_PRESETS / DEFAULT_ARMS constants.
"""

from __future__ import annotations

import os
import sys
import unittest

_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pytest  # noqa: E402

from scripts.run_inheritance_mode_ab import (  # noqa: E402
    ARM_PRESETS,
    DEFAULT_ARMS,
    _build_parser,
)


class TestArmPresetsConstant(unittest.TestCase):
    def test_all_p_modes_present(self):
        for mode in ("p2", "p3", "p4"):
            self.assertIn(mode, ARM_PRESETS, f"ARM_PRESETS missing '{mode}'")

    def test_baseline_arms_present(self):
        self.assertIn("baldwinian", ARM_PRESETS)
        self.assertIn("lamarckian", ARM_PRESETS)

    def test_each_preset_maps_to_correct_mode(self):
        for arm, preset in ARM_PRESETS.items():
            self.assertEqual(
                preset["inheritance_mode"],
                arm,
                f"ARM_PRESETS[{arm!r}]['inheritance_mode'] != {arm!r}",
            )

    def test_default_arms_are_subset_of_presets(self):
        for arm in DEFAULT_ARMS:
            self.assertIn(arm, ARM_PRESETS)


class TestABParserArmChoices(unittest.TestCase):
    def test_default_arms_are_baseline_pair(self):
        args = _build_parser().parse_args([])
        self.assertEqual(args.arms, ["baldwinian", "lamarckian"])

    def test_p2_arm_accepted(self):
        args = _build_parser().parse_args(["--arms", "p2"])
        self.assertEqual(args.arms, ["p2"])

    def test_p3_arm_accepted(self):
        args = _build_parser().parse_args(["--arms", "p3"])
        self.assertEqual(args.arms, ["p3"])

    def test_p4_arm_accepted(self):
        args = _build_parser().parse_args(["--arms", "p4"])
        self.assertEqual(args.arms, ["p4"])

    def test_multiple_arms_including_new_modes(self):
        args = _build_parser().parse_args(["--arms", "baldwinian", "p2", "p3", "p4"])
        self.assertEqual(args.arms, ["baldwinian", "p2", "p3", "p4"])

    def test_invalid_arm_rejected(self):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(["--arms", "epigenetic"])

    def test_all_arms_accepted_together(self):
        args = _build_parser().parse_args(
            ["--arms", "baldwinian", "lamarckian", "p2", "p3", "p4"]
        )
        self.assertEqual(set(args.arms), set(ARM_PRESETS))


if __name__ == "__main__":
    unittest.main()

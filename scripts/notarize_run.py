#!/usr/bin/env python3
"""Write a FarmNotary manifest for an AgentFarm run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from farm.provenance.notary import farm_notary_available, notarize_run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--runner")
    parser.add_argument("--anchor", action="store_true", help="Submit via FarmNotary backend (dry-run unless configured)")
    args = parser.parse_args()

    if not farm_notary_available():
        print(
            "farm_notary is not installed. Clone Dooders/FarmNotary and pip install -e ../FarmNotary",
            file=sys.stderr,
        )
        return 1

    receipt = notarize_run_dir(Path(args.run_dir), runner=args.runner, anchor=args.anchor)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

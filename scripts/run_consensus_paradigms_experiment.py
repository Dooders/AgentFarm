#!/usr/bin/env python3
from __future__ import annotations

import argparse

from farm.runners.consensus_paradigms_experiment import ConsensusParadigmsExperiment


def main() -> int:
    p = argparse.ArgumentParser(description="Compare consensus paradigms; optional FarmNotary stamp")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--voters", type=int, default=200)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--out", default="experiments/consensus_paradigms")
    p.add_argument("--no-notarize", action="store_true")
    args = p.parse_args()
    path = ConsensusParadigmsExperiment(args.out).run(
        trials=args.trials,
        voters=args.voters,
        candidates=args.candidates,
        notarize=not args.no_notarize,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

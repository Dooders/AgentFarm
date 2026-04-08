#!/usr/bin/env python
"""Benchmark all three crossover strategies and print a results table.

This script is the canonical repro for the numbers documented in
``docs/design/crossover_strategies.md``.  Run it with::

    python scripts/benchmark_crossover.py

or (from within a virtual environment)::

    source venv/bin/activate
    python scripts/benchmark_crossover.py

Optional flags
--------------
--input-dim   INT    Input dimension of the Q-network (default: 8)
--hidden-size INT    Hidden layer width (default: 64)
--output-dim  INT    Number of actions / output dimension (default: 4)
--n-states    INT    Number of synthetic evaluation states (default: 256)
--n-repeats   INT    Crossover iterations for timing average (default: 20)
--seed-a      INT    Torch seed for parent A (default: 0)
--seed-b      INT    Torch seed for parent B (default: 1)
--state-seed  INT    NumPy seed for evaluation states (default: 42)
--alpha       FLOAT  Blend coefficient for weighted crossover (default: 0.5)
--output-csv  PATH   Optional path to write CSV results

Example
-------
::

    python scripts/benchmark_crossover.py --n-repeats 50 --output-csv reports/crossover_bench.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Allow the script to run from the repo root without installing the package
# ---------------------------------------------------------------------------
try:
    from farm.core.decision.base_dqn import BaseQNetwork
    from farm.core.decision.training.crossover import crossover_quantized_state_dict
except ImportError as exc:
    sys.exit(
        f"Import error: {exc}\n"
        "Run 'pip install -e .' from the repository root, or activate the venv first."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_model(seed: int, input_dim: int, output_dim: int, hidden_size: int) -> BaseQNetwork:
    torch.manual_seed(seed)
    return BaseQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=hidden_size,
    )


def make_states(n: int, input_dim: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((n, input_dim)).astype("float32"))


def q_values(model: nn.Module, states: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(states)


def quality_metrics(child_q: torch.Tensor, ref_q: torch.Tensor) -> Dict[str, float]:
    diff = (child_q - ref_q).abs()
    return {
        "mean_q_error": diff.mean().item(),
        "max_q_error": diff.max().item(),
        "action_agreement": (
            (child_q.argmax(dim=1) == ref_q.argmax(dim=1)).float().mean().item()
        ),
    }


def time_crossover(
    sd_a: dict,
    sd_b: dict,
    mode: str,
    n_repeats: int,
    input_dim: int,
    output_dim: int,
    hidden_size: int,
    **kwargs,
) -> float:
    """Return mean wall-clock seconds for crossover + load_state_dict."""
    model = BaseQNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_size=hidden_size,
    )
    times: List[float] = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        child_sd = crossover_quantized_state_dict(sd_a, sd_b, mode=mode, **kwargs)
        model.load_state_dict(child_sd)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark crossover strategies for Q-network state dicts."
    )
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--n-states", type=int, default=256)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--seed-a", type=int, default=0)
    parser.add_argument("--seed-b", type=int, default=1)
    parser.add_argument("--state-seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--output-csv", type=str, default=None)
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Build parents and evaluation states
    # ------------------------------------------------------------------
    print("Building parents and evaluation states…")
    pa = make_model(args.seed_a, args.input_dim, args.output_dim, args.hidden_size)
    pb = make_model(args.seed_b, args.input_dim, args.output_dim, args.hidden_size)
    states = make_states(args.n_states, args.input_dim, args.state_seed)
    ref_q = q_values(pa, states)   # parent A as float reference

    sd_a = pa.state_dict()
    sd_b = pb.state_dict()

    strategies = [
        ("random",   {"seed": 42}),
        ("layer",    {}),
        ("weighted", {"alpha": args.alpha}),
    ]

    rows = []
    for mode, kwargs in strategies:
        print(f"Benchmarking {mode!r} ({args.n_repeats} repeats)…")
        mean_t = time_crossover(
            sd_a, sd_b, mode, args.n_repeats,
            args.input_dim, args.output_dim, args.hidden_size,
            **kwargs,
        )
        child_sd = crossover_quantized_state_dict(sd_a, sd_b, mode=mode, **kwargs)
        child = BaseQNetwork(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_size=args.hidden_size,
        )
        child.load_state_dict(child_sd)
        child_q = q_values(child, states)
        m = quality_metrics(child_q, ref_q)
        rows.append({
            "strategy": mode,
            "alpha": kwargs.get("alpha", "N/A"),
            "mean_time_ms": f"{mean_t * 1000:.3f}",
            "mean_q_error": f"{m['mean_q_error']:.6f}",
            "max_q_error": f"{m['max_q_error']:.6f}",
            "action_agreement": f"{m['action_agreement']:.4f}",
        })

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    col_w = [12, 8, 14, 14, 12, 16]
    headers = ["Strategy", "Alpha", "Time (ms)", "Mean Q Err", "Max Q Err", "Act. Agree"]
    sep = "-" * sum(col_w)

    print()
    print("Crossover Strategy Benchmark")
    print("=" * sum(col_w))
    print("".join(h.ljust(w) for h, w in zip(headers, col_w)))
    print(sep)
    for row in rows:
        vals = [
            row["strategy"],
            str(row["alpha"]),
            row["mean_time_ms"],
            row["mean_q_error"],
            row["max_q_error"],
            row["action_agreement"],
        ]
        print("".join(v.ljust(w) for v, w in zip(vals, col_w)))
    print("=" * sum(col_w))

    print()
    print("Experimental setup")
    print(f"  input_dim   = {args.input_dim}")
    print(f"  hidden_size = {args.hidden_size}")
    print(f"  output_dim  = {args.output_dim}")
    print(f"  n_states    = {args.n_states}")
    print(f"  n_repeats   = {args.n_repeats}")
    print(f"  seed_a      = {args.seed_a}")
    print(f"  seed_b      = {args.seed_b}")
    print(f"  state_seed  = {args.state_seed}")
    print(f"  alpha       = {args.alpha}  (weighted crossover)")
    print()
    print("Quality metrics are vs parent A (float reference).")

    # ------------------------------------------------------------------
    # Optional CSV export
    # ------------------------------------------------------------------
    if args.output_csv:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()

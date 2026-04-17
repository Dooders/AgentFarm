#!/usr/bin/env python3
"""Plot per-generation hyperparameter evolution metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot hyperparameter evolution from generation summary JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Path to evolution_generation_summaries.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hyperparameter_evolution.png"),
        help="Destination image path.",
    )
    return parser.parse_args()


def _load_summaries(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as handle:
        summaries = json.load(handle)
    if not summaries:
        raise ValueError(f"No generation summaries found in {path}")
    return summaries


def main() -> int:
    args = _parse_args()
    summaries = _load_summaries(args.summary_json)

    generations = [entry["generation"] for entry in summaries]
    best_fitness = [entry["best_fitness"] for entry in summaries]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(generations, best_fitness, marker="o", label="best_fitness")
    axes[0].set_ylabel("Fitness")
    axes[0].set_title("Fitness Convergence")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    gene_names = sorted(summaries[0].get("gene_statistics", {}).keys())
    if gene_names:
        for gene_name in gene_names:
            means = [entry["gene_statistics"][gene_name]["mean"] for entry in summaries]
            stds = [entry["gene_statistics"][gene_name]["std"] for entry in summaries]
            lower = [mean - std for mean, std in zip(means, stds)]
            upper = [mean + std for mean, std in zip(means, stds)]
            axes[1].plot(generations, means, marker="o", label=f"{gene_name} mean")
            axes[1].fill_between(generations, lower, upper, alpha=0.2, label=f"{gene_name} ±1 std")
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            "No gene_statistics found in summary JSON",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Gene Value")
    axes[1].set_title("Gene Evolution (mean ± std)")
    axes[1].grid(alpha=0.3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote hyperparameter evolution plot to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

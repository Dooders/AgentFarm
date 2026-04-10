#!/usr/bin/env python
"""Multi-generation evolutionary crossover search.

Runs :func:`~farm.core.decision.training.run_multi_generation_search` for
*--num-generations* rounds.  In each generation a population of child
Q-networks is produced via crossover + fine-tune; the best child(ren) are
promoted as parents for the next generation.  Optional Gaussian weight-noise
mutation can be applied to the promoted parents before each new round.

Per-generation output is written to ``<run-dir>/gen_<N>/`` with the standard
:func:`~farm.core.decision.training.run_crossover_search` artefacts (manifest,
leaderboard, recommendation).  After all generations, a summary JSON and
lineage JSON are written to ``<run-dir>/``.

Quick start (synthetic parents, 2 generations, minimal search space)
--------------------------------------------------------------------
::

    python scripts/run_multi_gen_search.py \\
        --num-generations 2 \\
        --search-space minimal \\
        --run-dir runs/multi_gen

With real parent checkpoints::

    python scripts/run_multi_gen_search.py \\
        --parent-a-ckpt checkpoints/agent_a.pt \\
        --parent-b-ckpt checkpoints/agent_b.pt \\
        --states-file data/replay_states.npy \\
        --num-generations 3 \\
        --run-dir runs/multi_gen_real

Enable Gaussian weight mutation between generations::

    python scripts/run_multi_gen_search.py \\
        --num-generations 3 \\
        --mutation-std 0.01 \\
        --run-dir runs/multi_gen_mutated

Use the ``best_vs_original`` selection strategy (best child vs original parent B)::

    python scripts/run_multi_gen_search.py \\
        --num-generations 3 \\
        --selection-strategy best_vs_original \\
        --run-dir runs/multi_gen_bvo

Output files
------------
``<run-dir>/gen_<N>/``
    Standard crossover search outputs per generation.
``<run-dir>/multi_gen_summary.json``
    One summary record per generation (best/mean primary metric, promoted
    parents, run directory).
``<run-dir>/lineage.json``
    Full lineage registry: one record per child across all generations,
    with parent IDs, metric, and mutation flag.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

# Allow running directly from repo root without pip install -e .
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.decision.base_dqn import BaseQNetwork  # noqa: E402
from farm.core.decision.training import (  # noqa: E402
    CrossoverRecipe,
    FineTuneRegime,
    GenerationConfig,
    MutationConfig,
    SearchConfig,
    run_multi_generation_search,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-generation evolutionary crossover search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Parents
    p.add_argument(
        "--parent-a-ckpt",
        default=None,
        help="Path to parent A checkpoint (.pt state dict). "
        "Omit to use a random synthetic parent.",
    )
    p.add_argument(
        "--parent-b-ckpt",
        default=None,
        help="Path to parent B checkpoint (.pt state dict). "
        "Omit to use a random synthetic parent.",
    )

    # State buffer
    p.add_argument(
        "--states-file",
        default=None,
        help="Path to .npy file of shape (N, input_dim) with evaluation states. "
        "Omit to generate synthetic states.",
    )
    p.add_argument(
        "--n-states",
        type=int,
        default=200,
        help="Number of synthetic states to generate (ignored when --states-file is set).",
    )

    # Architecture (used when creating synthetic parents or loading checkpoints)
    p.add_argument("--input-dim", type=int, default=8)
    p.add_argument("--output-dim", type=int, default=4)
    p.add_argument("--hidden-size", type=int, default=64)

    # Multi-generation settings
    p.add_argument("--num-generations", type=int, default=2,
                   help="Number of generations to run (>= 1).")
    p.add_argument(
        "--selection-strategy",
        choices=["best", "best_vs_original"],
        default="best",
        help="How to select parents for the next generation.",
    )
    p.add_argument(
        "--keep-top-k",
        type=int,
        default=None,
        help="Number of top children to retain in lineage per generation. "
        "None keeps all.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed for reproducibility.",
    )

    # Mutation
    p.add_argument(
        "--mutation-std",
        type=float,
        default=0.0,
        help="Gaussian noise std for weight mutation between generations. "
        "0 disables mutation.",
    )
    p.add_argument(
        "--mutation-fraction",
        type=float,
        default=1.0,
        help="Fraction of weight elements to mutate (1.0 = all).",
    )
    p.add_argument(
        "--mutation-seed",
        type=int,
        default=None,
        help="RNG seed for mutation noise.",
    )

    # Search space
    p.add_argument(
        "--search-space",
        choices=["default", "minimal", "default-qat", "minimal-qat", "custom"],
        default="minimal",
        help="Predefined search space. Use 'custom' with --crossover-modes etc.",
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Cap on children per generation (truncates Cartesian product).",
    )
    p.add_argument(
        "--crossover-modes",
        nargs="+",
        choices=["random", "layer", "weighted"],
        default=["random", "layer", "weighted"],
        help="Crossover modes (used with --search-space custom).",
    )
    p.add_argument(
        "--alpha-values",
        nargs="+",
        type=float,
        default=[0.5],
        help="Alpha values for random/weighted crossover (custom search).",
    )
    p.add_argument(
        "--crossover-seeds",
        nargs="+",
        type=int,
        default=[0],
        help="RNG seeds for random crossover (custom search).",
    )
    p.add_argument(
        "--finetune-regimes",
        nargs="+",
        choices=["short", "medium", "long"],
        default=["short"],
        help="Fine-tune regime names (custom search).",
    )

    # Output
    p.add_argument(
        "--run-dir",
        default="runs/multi_gen_search",
        help="Root output directory.",
    )

    return p.parse_args()


def _build_search_config(args: argparse.Namespace) -> SearchConfig:
    if args.search_space == "custom":
        recipes: list = []
        for mode in args.crossover_modes:
            if mode == "layer":
                recipes.append(CrossoverRecipe("layer"))
            else:
                for alpha in args.alpha_values:
                    if mode == "random":
                        for seed in args.crossover_seeds:
                            recipes.append(CrossoverRecipe(mode, alpha=alpha, seed=seed))
                    else:
                        recipes.append(CrossoverRecipe(mode, alpha=alpha))

        _regime_map = {
            "short": FineTuneRegime("short", epochs=5, lr=1e-3, seed=42),
            "medium": FineTuneRegime("medium", epochs=10, lr=5e-4, seed=42),
            "long": FineTuneRegime("long", epochs=20, lr=1e-4, seed=42),
        }
        regimes = [_regime_map[r] for r in args.finetune_regimes]
        return SearchConfig(
            crossover_recipes=recipes,
            finetune_regimes=regimes,
            max_runs=args.max_runs,
        )

    cfg_map = {
        "default": SearchConfig.default,
        "minimal": SearchConfig.minimal,
        "default-qat": SearchConfig.default_with_qat,
        "minimal-qat": SearchConfig.minimal_with_qat,
    }
    cfg = cfg_map[args.search_space]()
    if args.max_runs is not None:
        cfg.max_runs = args.max_runs
    return cfg


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------
    # Parents
    # ------------------------------------------------------------------
    if args.parent_a_ckpt:
        parent_a = BaseQNetwork(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_size=args.hidden_size,
        )
        parent_a.load_state_dict(
            torch.load(args.parent_a_ckpt, map_location="cpu", weights_only=True)
        )
    else:
        seed_a = args.seed if args.seed is not None else 0
        torch.manual_seed(seed_a)
        parent_a = BaseQNetwork(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_size=args.hidden_size,
        )

    if args.parent_b_ckpt:
        parent_b = BaseQNetwork(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_size=args.hidden_size,
        )
        parent_b.load_state_dict(
            torch.load(args.parent_b_ckpt, map_location="cpu", weights_only=True)
        )
    else:
        seed_b = (args.seed + 1) if args.seed is not None else 1
        torch.manual_seed(seed_b)
        parent_b = BaseQNetwork(
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            hidden_size=args.hidden_size,
        )

    # ------------------------------------------------------------------
    # States
    # ------------------------------------------------------------------
    if args.states_file:
        states = np.load(args.states_file)
    else:
        rng = np.random.default_rng(args.seed)
        states = rng.standard_normal((args.n_states, args.input_dim)).astype("float32")

    # ------------------------------------------------------------------
    # Configs
    # ------------------------------------------------------------------
    search_config = _build_search_config(args)

    mutation_config: MutationConfig | None = None
    if args.mutation_std > 0.0:
        mutation_config = MutationConfig(
            noise_std=args.mutation_std,
            noise_fraction=args.mutation_fraction,
            seed=args.mutation_seed,
        )

    gen_config = GenerationConfig(
        num_generations=args.num_generations,
        search_config=search_config,
        keep_top_k=args.keep_top_k,
        mutation_config=mutation_config,
        selection_strategy=args.selection_strategy,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    print(f"Starting multi-generation search: {args.num_generations} generation(s)")
    print(f"  Search space  : {args.search_space}")
    print(f"  Selection     : {args.selection_strategy}")
    print(f"  Mutation std  : {args.mutation_std}")
    print(f"  Output        : {args.run_dir}")
    print()

    summaries, lineage = run_multi_generation_search(
        parent_a,
        parent_b,
        states,
        gen_config,
        args.run_dir,
    )

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("Multi-generation search complete")
    print("=" * 72)
    print(f"{'Gen':>4}  {'Children':>8}  {'Best':>8}  {'Mean':>8}  {'Best child ID'}")
    print("-" * 72)
    for s in summaries:
        print(
            f"{s.generation:>4}  {s.n_children:>8}  "
            f"{s.best_primary_metric:>8.4f}  {s.mean_primary_metric:>8.4f}  "
            f"{s.best_child_id}"
        )
    print("=" * 72)
    print(f"\nTotal children produced : {len(lineage)}")
    print(f"Summary JSON            : {os.path.join(args.run_dir, 'multi_gen_summary.json')}")
    print(f"Lineage JSON            : {os.path.join(args.run_dir, 'lineage.json')}")


if __name__ == "__main__":
    main()

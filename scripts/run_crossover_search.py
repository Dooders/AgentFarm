#!/usr/bin/env python
"""Systematic crossover + fine-tune search for optimal recombination strategy.

Generates *N* child Q-networks (configurable, ≥ 9 for a minimal 3×3 grid) by
sweeping over crossover strategies × fine-tune hyperparameter recipes.  Every
child is scored with the **same** state buffer / evaluation harness so results
are directly comparable.  A leaderboard (CSV + JSON) and a short text
recommendation are written to *--run-dir*.  Use ``--search-space default-qat``
or ``minimal-qat`` to include a **QAT** fine-tune regime (``ptq_dynamic``);
``--workers N`` runs children in parallel processes (float ``BaseQNetwork``
parents only).

Quick start (synthetic data, 9 children, fast regimes)
------------------------------------------------------
::

    python scripts/run_crossover_search.py \\
        --search-space minimal \\
        --run-dir runs/crossover_search

Full default search (14 children)::

    python scripts/run_crossover_search.py \\
        --run-dir runs/crossover_search_full

With real parent checkpoints and states::

    python scripts/run_crossover_search.py \\
        --parent-a-ckpt checkpoints/parent_a.pt \\
        --parent-b-ckpt checkpoints/parent_b.pt \\
        --states-file data/replay_states.npy \\
        --run-dir runs/crossover_search_real

Limit to 3 children (smoke test)::

    python scripts/run_crossover_search.py \\
        --max-runs 3 \\
        --run-dir /tmp/crossover_smoke

Custom search space (crossover modes + alpha values + fine-tune regimes)::

    python scripts/run_crossover_search.py \\
        --crossover-modes random weighted \\
        --alpha-values 0.3 0.5 0.7 \\
        --crossover-seeds 0 1 2 \\
        --finetune-regimes short long \\
        --run-dir runs/custom_search

Architecture flags (--input-dim, --output-dim, --hidden-size) must match the
checkpoints.  The defaults (8, 4, 64) are the standard AgentFarm dimensions.

Output files
------------
``<run-dir>/manifest.json``
    One entry per child with paths + hyperparameters + metrics.
``<run-dir>/leaderboard.csv``
    Sorted by primary metric (min(agree_a, agree_b)); parent baselines appended.
``<run-dir>/leaderboard.json``
    Same data in JSON format.
``<run-dir>/recommendation.txt``
    Human-readable strategy recommendation.
``<run-dir>/<child_id>/child.pt``
    Best-checkpoint child state dict.
``<run-dir>/<child_id>/eval_report.json``
    Full evaluation report (RecombinationReport JSON).
``<run-dir>/<child_id>/run_config.json``
    Full config JSON for exact reproducibility.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# Allow running directly from repo root without pip install -e .
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.decision.base_dqn import BaseQNetwork  # noqa: E402
from farm.core.decision.training.crossover import CROSSOVER_MODES  # noqa: E402
from farm.core.decision.training.crossover_search import (  # noqa: E402
    CrossoverRecipe,
    FineTuneRegime,
    SearchConfig,
    build_leaderboard,
    generate_recommendation,
    run_crossover_search,
)
from farm.core.decision.training.distillation_script_helpers import (  # noqa: E402
    load_base_qnetwork_checkpoint,
    load_distillation_states,
)
from farm.core.decision.training.recombination_eval import (  # noqa: E402
    RecombinationThresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_search_config_from_cli_args(args: argparse.Namespace) -> SearchConfig:
    """Construct a SearchConfig from parsed CLI arguments.

    Shared with ``run_multi_generation_search.py`` so both CLIs use the same
    search-space presets and custom-grid logic.
    """
    if args.search_space == "default":
        cfg = SearchConfig.default()
    elif args.search_space == "minimal":
        cfg = SearchConfig.minimal()
    elif args.search_space == "default-qat":
        cfg = SearchConfig.default_with_qat()
    elif args.search_space == "minimal-qat":
        cfg = SearchConfig.minimal_with_qat()
    else:
        # Custom search space from individual CLI flags
        recipes: list[CrossoverRecipe] = []

        # Determine which modes to sweep
        modes = args.crossover_modes or list(CROSSOVER_MODES)
        alphas = args.alpha_values or [0.3, 0.5, 0.7]
        seeds = args.crossover_seeds

        for mode in modes:
            if mode == "layer":
                # layer mode has no alpha/seed knob
                recipes.append(CrossoverRecipe("layer"))
            elif mode == "random":
                for sd in (seeds if seeds else [0, 1, 2]):
                    for alpha in alphas:
                        recipes.append(CrossoverRecipe("random", alpha=alpha, seed=sd))
            else:  # weighted
                for alpha in alphas:
                    recipes.append(CrossoverRecipe("weighted", alpha=alpha))

        # Fine-tune regimes
        _PRESET_REGIMES = {
            "short": FineTuneRegime("short", epochs=5, lr=1e-3, seed=42),
            "medium": FineTuneRegime("medium", epochs=10, lr=5e-4, seed=42),
            "long": FineTuneRegime("long", epochs=20, lr=1e-4, seed=42),
            "lr_high": FineTuneRegime("lr_high", epochs=5, lr=5e-3, seed=42),
            "short_qat": FineTuneRegime(
                "short_qat",
                epochs=5,
                lr=1e-4,
                seed=42,
                quantization_applied="ptq_dynamic",
                batch_size=16,
                val_fraction=0.1,
            ),
        }
        regime_names = args.finetune_regimes or ["short", "long"]
        regimes: list[FineTuneRegime] = []
        for name in regime_names:
            if name not in _PRESET_REGIMES:
                raise ValueError(
                    f"Unknown finetune regime {name!r}. "
                    f"Available: {sorted(_PRESET_REGIMES)}"
                )
            regimes.append(_PRESET_REGIMES[name])

        cfg = SearchConfig(
            crossover_recipes=recipes,
            finetune_regimes=regimes,
        )

    if args.max_runs is not None:
        cfg.max_runs = args.max_runs

    if args.degenerate_threshold is not None:
        cfg.degenerate_threshold = args.degenerate_threshold

    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_crossover_search_arg_parser(
    *,
    description: str | None = None,
) -> argparse.ArgumentParser:
    """Build the shared ArgumentParser for crossover search CLIs."""
    p = argparse.ArgumentParser(
        description=description
        or (
            "Systematic crossover + fine-tune search for optimal recombination strategy."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Search space
    p.add_argument(
        "--search-space",
        choices=["default", "minimal", "default-qat", "minimal-qat", "custom"],
        default="default",
        help=(
            "Pre-defined search: 'default' (14 = 7×2 float), 'minimal' (9 = 3×3 float), "
            "'default-qat' (21 = 7×3 adds short_qat), 'minimal-qat' (9 = 3×3 with short_qat), "
            "or 'custom'."
        ),
    )
    p.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Cap the total number of children generated (useful for smoke tests).",
    )
    p.add_argument(
        "--degenerate-threshold",
        type=float,
        default=None,
        help=(
            "Children whose primary metric falls below this threshold are flagged as "
            "degenerate.  0.0 (default) disables the flag."
        ),
    )

    # Custom search space knobs (used when --search-space=custom)
    p.add_argument(
        "--crossover-modes",
        nargs="+",
        choices=list(CROSSOVER_MODES),
        default=None,
        help="Crossover modes to sweep (custom mode only).",
    )
    p.add_argument(
        "--alpha-values",
        nargs="+",
        type=float,
        default=None,
        help="Alpha values to sweep for random/weighted modes (custom mode only).",
    )
    p.add_argument(
        "--crossover-seeds",
        nargs="+",
        type=int,
        default=None,
        help="RNG seeds to sweep for the random crossover mode (custom mode only).",
    )
    p.add_argument(
        "--finetune-regimes",
        nargs="+",
        choices=["short", "medium", "long", "lr_high", "short_qat"],
        default=None,
        help="Fine-tune regime names to sweep (custom mode only).",
    )

    # Architecture
    p.add_argument("--input-dim", type=int, default=8, help="Network input feature dimension.")
    p.add_argument("--output-dim", type=int, default=4, help="Number of output actions.")
    p.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size.")

    # Parent checkpoints (optional — random weights used when absent)
    p.add_argument(
        "--parent-a-ckpt",
        default="",
        help="Path to parent A state-dict checkpoint (.pt).",
    )
    p.add_argument(
        "--parent-b-ckpt",
        default="",
        help="Path to parent B state-dict checkpoint (.pt).",
    )

    # States
    p.add_argument(
        "--states-file",
        default="",
        help="Path to .npy states file of shape (N, input_dim).",
    )
    p.add_argument(
        "--n-states",
        type=int,
        default=1000,
        help="Number of synthetic states to generate when --states-file is absent.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for synthetic state generation.",
    )

    # Evaluation thresholds (recorded in reports for analysis; search always runs in report-only mode)
    p.add_argument("--min-action-agreement", type=float, default=0.70)
    p.add_argument("--max-kl-divergence", type=float, default=1.0)
    p.add_argument("--max-mse", type=float, default=5.0)
    p.add_argument("--min-cosine-similarity", type=float, default=0.70)

    # Misc
    p.set_defaults(include_parent_baseline=True)
    p.add_argument(
        "--include-parent-baseline",
        action="store_true",
        help="Include a parent A vs parent B comparison in each eval report (default: enabled).",
    )
    p.add_argument(
        "--no-parent-baseline",
        dest="include_parent_baseline",
        action="store_false",
        help="Disable the parent A vs parent B baseline comparison.",
    )
    p.add_argument(
        "--run-dir",
        default="runs/crossover_search",
        help="Root directory for all outputs.",
    )
    p.add_argument(
        "--eval-batch-size",
        type=int,
        default=2048,
        help=(
            "Max states per evaluation forward pass (lower = less GPU/RAM). "
            "Use 0 to run all states in a single batch."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel child runs via ProcessPoolExecutor when >1. "
            "Requires float BaseQNetwork parents (state dicts cached under run-dir). "
            "Quantized parents are not supported; use 1."
        ),
    )

    return p


def parse_crossover_search_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse argv for a single-generation crossover search (default CLI)."""
    return build_crossover_search_arg_parser().parse_args(argv)


def main() -> None:
    args = parse_crossover_search_cli_args()

    print("\n" + "=" * 72)
    print("Crossover + Fine-tune Search")
    print("=" * 72)

    # 1. Build search config
    search_cfg = build_search_config_from_cli_args(args)
    pairs = search_cfg.pairs()
    print(f"\nSearch space: {len(pairs)} children")
    print(f"  Crossover recipes : {len(search_cfg.crossover_recipes)}")
    print(f"  Fine-tune regimes : {len(search_cfg.finetune_regimes)}")
    if search_cfg.max_runs is not None:
        print(f"  (capped at {search_cfg.max_runs})")
    if args.workers > 1:
        print(f"  Parallel workers   : {args.workers} (process pool; BaseQNetwork parents only)")

    # 2. Load parents
    print("\n[1/4] Loading parent networks …")
    parent_a = load_base_qnetwork_checkpoint(
        args.parent_a_ckpt,
        args.input_dim,
        args.output_dim,
        args.hidden_size,
        random_weights_message="  No checkpoint path provided — using random weights.",
    )
    parent_b = load_base_qnetwork_checkpoint(
        args.parent_b_ckpt,
        args.input_dim,
        args.output_dim,
        args.hidden_size,
        random_weights_message="  No checkpoint path provided — using random weights.",
    )

    # 3. Load / generate states
    print("\n[2/4] Preparing evaluation states …")
    states = load_distillation_states(
        args.states_file, args.n_states, args.input_dim, args.seed
    )

    # 4. Build thresholds
    # The search runner always uses report-only mode: thresholds are recorded
    # in each eval_report.json for post-hoc analysis but never gate the run.
    thresholds = RecombinationThresholds(
        min_action_agreement=args.min_action_agreement,
        max_kl_divergence=args.max_kl_divergence,
        max_mse=args.max_mse,
        min_cosine_similarity=args.min_cosine_similarity,
        report_only=True,
    )

    # 5. Run search
    print(f"\n[3/4] Running {len(pairs)} children → {args.run_dir} …")
    eval_bs = (
        args.eval_batch_size
        if args.eval_batch_size and args.eval_batch_size > 0
        else None
    )
    manifest, leaderboard = run_crossover_search(
        parent_a=parent_a,
        parent_b=parent_b,
        states=states,
        search_config=search_cfg,
        run_dir=args.run_dir,
        thresholds=thresholds,
        include_parent_baseline=args.include_parent_baseline,
        eval_batch_size=eval_bs,
        num_workers=max(1, args.workers),
    )

    # 6. Print summary
    print("\n[4/4] Results")
    print("=" * 72)
    print(f"  Children evaluated : {len(manifest)}")
    print(f"  Degenerate         : {sum(1 for e in manifest if e.degenerate)}")
    print("\n  Leaderboard (top-5):")
    header = f"  {'#':>3}  {'child_id':<45}  {'primary':>8}  {'agree_a':>7}  {'agree_b':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in leaderboard[:5]:
        if row.get("is_parent_baseline"):
            continue
        print(
            f"  {row['rank']:>3}  {row['child_id']:<45}  "
            f"{row['primary_metric']:>8.4f}  "
            f"{row['child_vs_parent_a_agreement']:>7.4f}  "
            f"{row['child_vs_parent_b_agreement']:>7.4f}"
        )

    print(f"\n  Outputs written to: {args.run_dir}")
    print("    manifest.json")
    print("    leaderboard.csv / leaderboard.json")
    print("    recommendation.txt")

    # Print full recommendation
    parent_baseline: float | None = None
    for row in leaderboard:
        if row.get("is_parent_baseline"):
            parent_baseline = row.get("primary_metric")
            break

    print("\n" + generate_recommendation(manifest, parent_baseline))
    print("\nSearch complete.")


if __name__ == "__main__":
    main()

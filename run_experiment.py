#!/usr/bin/env python3
"""CLI for the political consensus experiment (farm.experiments.consensus).

Default run:

    python run_experiment.py --trials 250 --voters 400 --candidates 8 \
        --population two_cluster --seed 0 --out results/consensus

Sweep across populations and candidate counts:

    python run_experiment.py sweep --trials 100 \
        --populations two_cluster,three_cluster,rural_town --candidates 6,8,12

Render a social-media MP4 of one trial's dynamic (requires ffmpeg):

    python run_experiment.py animate --seed 0 --trial 0 \
        --out results/consensus_media/consensus_dynamics.mp4

Render the produced audience explainer from run outputs (requires manim):

    python run_experiment.py overview --results results/consensus \
        --correlated-results results/consensus_lambda_correlated \
        --out results/consensus_media/consensus_overview.mp4
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from farm.experiments.consensus.experiment import (
    DEFAULT_LAMBDA_CAP,
    ExperimentConfig,
    SweepConfig,
    config_manifest,
    run_sweep,
    run_trials,
    summarize,
    sweep_manifest,
    write_outputs,
)
from farm.experiments.consensus.population import POPULATION_TYPES


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trials", type=int, default=250, help="Trials per experiment cell")
    parser.add_argument("--voters", type=int, default=400, help="Voters per trial")
    parser.add_argument("--seed", type=int, default=0, help="Base seed for the deterministic RNG streams")
    parser.add_argument(
        "--include-constrained",
        action="store_true",
        help="Also run the constrained_individual paradigm (individual voting with a λ cap)",
    )
    parser.add_argument(
        "--lambda-cap",
        type=float,
        default=DEFAULT_LAMBDA_CAP,
        help="Effective λ cap used by constrained_individual",
    )
    parser.add_argument(
        "--lambda-correlated",
        action="store_true",
        help="Rank-couple candidate λ to platform extremity instead of drawing it independently",
    )


def _build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the consensus experiment for one population type.")
    _add_common_arguments(parser)
    parser.add_argument("--candidates", type=int, default=8, help="Candidates per trial")
    parser.add_argument("--population", choices=POPULATION_TYPES, default="two_cluster")
    parser.add_argument("--out", type=Path, default=Path("results/consensus"), help="Output directory")
    return parser


def _build_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} sweep",
        description="Sweep the consensus experiment across populations and candidate counts.",
    )
    _add_common_arguments(parser)
    parser.add_argument(
        "--populations",
        default=",".join(POPULATION_TYPES),
        help="Comma-separated population types",
    )
    parser.add_argument("--candidates", default="8", help="Comma-separated candidate counts")
    parser.add_argument("--out", type=Path, default=Path("results/consensus_sweep"), help="Output directory")
    return parser


def _build_animate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} animate",
        description="Render an MP4 animation of one trial under every paradigm (for social media).",
    )
    parser.add_argument("--voters", type=int, default=400, help="Voters in the animated trial")
    parser.add_argument("--candidates", type=int, default=8, help="Candidates in the animated trial")
    parser.add_argument("--population", choices=POPULATION_TYPES, default="two_cluster")
    parser.add_argument("--seed", type=int, default=0, help="Base seed (same stream as the experiment)")
    parser.add_argument("--trial", type=int, default=0, help="Trial index to animate")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/consensus_media/consensus_dynamics.mp4"),
        help="Output MP4 path",
    )
    return parser


def _build_overview_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{Path(sys.argv[0]).name} overview",
        description="Render the produced audience-explainer MP4 from experiment outputs (requires manim).",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/consensus"),
        help="Run directory whose summary.csv provides the headline numbers",
    )
    parser.add_argument(
        "--correlated-results",
        type=Path,
        default=Path("results/consensus_lambda_correlated"),
        help="Run directory of the --lambda-correlated condition (for the twist segment)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument(
        "--quality",
        choices=("high", "preview"),
        default="high",
        help="high = 1280x720, preview = 854x480 for fast iteration",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/consensus_media/consensus_overview.mp4"),
        help="Output MP4 path",
    )
    return parser


def _base_config(args: argparse.Namespace, candidates: int, population: str) -> ExperimentConfig:
    return ExperimentConfig(
        trials=args.trials,
        voters=args.voters,
        candidates=candidates,
        population=population,
        seed=args.seed,
        include_constrained=args.include_constrained,
        lambda_cap=args.lambda_cap,
        lambda_correlated=args.lambda_correlated,
    )


def _print_summary(trials: pd.DataFrame, out_dir: Path) -> None:
    summary = summarize(trials)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print("\n=== Summary (means and stds by population, candidates, paradigm) ===")
        print(summary.to_string(index=False))
    print(f"\nArtifacts written to {out_dir}/ (trials.csv, summary.csv, allocation_means.csv, figures/, REPORT.md)")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "python run_experiment.py " + " ".join(argv)

    if argv and argv[0] == "animate":
        from farm.experiments.consensus.animate import render_animation

        args = _build_animate_parser().parse_args(argv[1:])
        path = render_animation(
            out_path=args.out,
            voters=args.voters,
            n_candidates=args.candidates,
            population_type=args.population,
            seed=args.seed,
            trial=args.trial,
            fps=args.fps,
        )
        print(f"Animation written to {path}")
        return 0

    if argv and argv[0] == "overview":
        args = _build_overview_parser().parse_args(argv[1:])
        try:
            from farm.experiments.consensus.overview_video import render_overview
        except ImportError as exc:
            raise SystemExit(
                "The overview renderer needs the optional 'manim' dependency: pip install manim "
                f"(import failed: {exc})"
            ) from exc
        path = render_overview(
            out_path=args.out,
            default_results=args.results,
            correlated_results=args.correlated_results,
            fps=args.fps,
            quality=args.quality,
        )
        print(f"Overview video written to {path}")
        return 0

    if argv and argv[0] == "sweep":
        args = _build_sweep_parser().parse_args(argv[1:])
        populations = [p.strip() for p in args.populations.split(",") if p.strip()]
        unknown = [p for p in populations if p not in POPULATION_TYPES]
        if unknown:
            raise SystemExit(f"Unknown population type(s) {unknown}; expected subset of {POPULATION_TYPES}")
        candidate_counts = [int(c) for c in str(args.candidates).split(",") if c.strip()]
        sweep = SweepConfig(
            base=_base_config(args, candidates=candidate_counts[0], population=populations[0]),
            populations=populations,
            candidate_counts=candidate_counts,
        )
        trials = run_sweep(sweep)
        write_outputs(trials, args.out, sweep_manifest(sweep, command))
    else:
        args = _build_run_parser().parse_args(argv)
        config = _base_config(args, candidates=args.candidates, population=args.population)
        trials = run_trials(config)
        write_outputs(trials, args.out, config_manifest(config, command))

    _print_summary(trials, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

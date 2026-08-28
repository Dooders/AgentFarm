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

Verify that the derived artifacts (summary.csv, allocation_means.csv, REPORT.md)
recompute byte-identically from trials.csv:

    python run_experiment.py verify-report --results results/consensus

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
    allocation_means,
    config_manifest,
    run_cell,
    run_sweep,
    summarize,
    sweep_manifest,
    write_outputs,
)
from farm.experiments.consensus.mechanism import MECHANISMS
from farm.experiments.consensus.paradigms import VOTING_MODES
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
        help="Robustness appendix: rank-couple candidate λ to platform extremity (not the primary cell)",
    )
    parser.add_argument(
        "--mechanism",
        choices=MECHANISMS,
        default="oneshot",
        help="oneshot = exogenous λ; reelection = winner chooses λ to maximize observed-sample utility",
    )
    parser.add_argument(
        "--voting",
        choices=VOTING_MODES,
        default="sincere",
        help="sincere is the default baseline; abandon_trailing is a plurality heuristic",
    )
    parser.add_argument(
        "--no-persist-ballots",
        action="store_true",
        help="Do not write synthetic ballots / supporter masks under private/",
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
        persist_ballots=not args.no_persist_ballots,
        mechanism=args.mechanism,
        voting=args.voting,
    )


def _print_summary(trials: pd.DataFrame, out_dir: Path) -> None:
    summary = summarize(trials)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        print("\n=== Summary (means and stds by population, candidates, paradigm) ===")
        print(summary.to_string(index=False))
    print(f"\nArtifacts written to {out_dir}/ (trials.csv, summary.csv, allocation_means.csv, contrasts.csv, figures/, REPORT.md)")


def _portable_command(argv: list) -> str:
    """The recorded command with the output directory replaced by {run_dir}.

    Keeps REPORT.md byte-identical across re-runs into different directories
    and makes the record directly usable by `farm-notary reproduce`.
    """
    parts: list = []
    saw_out = False
    skip = False
    for tok in argv:
        if skip:
            parts.append("{run_dir}")
            skip = False
        elif tok == "--out":
            parts.append(tok)
            saw_out = skip = True
        elif tok.startswith("--out="):
            parts.append("--out={run_dir}")
            saw_out = True
        else:
            parts.append(tok)
    if not saw_out:
        parts += ["--out", "{run_dir}"]
    import shlex

    return "python run_experiment.py " + shlex.join(parts)


def _verify_report(results: Path) -> int:
    """Recompute summary.csv, allocation_means.csv, contrasts.csv, and REPORT.md from trials.csv.

    Byte-compares the recomputation against the files on disk, so the derived
    artifacts are verified to follow from the raw trial data.
    """
    import json

    from farm.experiments.consensus.contrasts import paired_contrasts
    from farm.experiments.consensus.report import render_report

    # round_trip parsing recovers the exact float64 values that were written;
    # the default fast parser can be off by one ulp, which breaks byte equality.
    trials = pd.read_csv(results / "trials.csv", float_precision="round_trip")
    run_config = json.loads((results / "run_config.json").read_text())
    config = run_config.get("config", {})
    include_lambda = bool(config.get("lambda_correlated") or config.get("mechanism") == "reelection")
    summary = summarize(trials)
    allocations = allocation_means(trials)
    contrasts = paired_contrasts(trials, include_lambda_primary=include_lambda)

    expected = {
        "summary.csv": summary.to_csv(index=False),
        "allocation_means.csv": allocations.to_csv(index=False),
        "contrasts.csv": contrasts.to_csv(index=False),
        "REPORT.md": render_report(trials, summary, allocations, run_config, contrasts=contrasts),
    }
    failures = []
    for name, text in expected.items():
        actual = (results / name).read_text()
        if actual == text:
            print(f"OK {name} recomputed from trials.csv is byte-identical")
        else:
            failures.append(name)
            print(f"FAIL {name} does not match its recomputation from trials.csv")
    return 1 if failures else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = _portable_command(argv)

    if argv and argv[0] == "verify-report":
        parser = argparse.ArgumentParser(
            prog="run_experiment.py verify-report",
            description="Verify that derived artifacts follow from trials.csv",
        )
        parser.add_argument("--results", type=Path, default=Path("results/consensus"))
        args = parser.parse_args(argv[1:])
        return _verify_report(args.results)

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
        write_outputs(trials, args.out, sweep_manifest(sweep, command), persist_ballots=sweep.base.persist_ballots)
    else:
        args = _build_run_parser().parse_args(argv)
        config = _base_config(args, candidates=args.candidates, population=args.population)
        run = run_cell(config)
        trials = run.trials
        write_outputs(
            trials,
            args.out,
            config_manifest(config, command),
            audit=run.audit,
            persist_ballots=config.persist_ballots,
        )

    _print_summary(trials, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

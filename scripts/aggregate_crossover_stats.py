#!/usr/bin/env python
"""Aggregate repeated-run crossover-search results and run significance tests.

This script loads one or more ``manifest.json`` files produced by
:func:`farm.core.decision.training.crossover_search.run_crossover_search` (one
file per seed / repeated run), computes per-condition summary statistics
(mean ± std), and optionally compares two named conditions with a Welch t-test
or bootstrap confidence interval.

Quick-start examples
--------------------
::

    # Summarise all conditions across three seed runs.
    python scripts/aggregate_crossover_stats.py \\
        runs/seed0/manifest.json \\
        runs/seed1/manifest.json \\
        runs/seed2/manifest.json

    # Group only by crossover_mode (ignore fine-tune regime).
    python scripts/aggregate_crossover_stats.py \\
        runs/seed*/manifest.json \\
        --group-by crossover_mode

    # Compare "weighted" vs "random" crossover on primary_metric.
    python scripts/aggregate_crossover_stats.py \\
        runs/seed*/manifest.json \\
        --compare weighted random \\
        --compare-field crossover_mode

    # Also write a JSON summary to a file.
    python scripts/aggregate_crossover_stats.py \\
        runs/seed*/manifest.json \\
        --output-json results/stats_summary.json

Output
------
Prints a table of conditions to *stdout* with columns:
  condition | n_runs | mean_primary | std_primary | mean_agree_a | mean_agree_b | n_degenerate

If ``--compare`` is given, additionally prints:
  - Welch t-test result
  - 95 % bootstrap CI for each group
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

# Allow running without installing the package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from farm.core.decision.training.recombination_stats import (  # noqa: E402
    NUMERIC_METRIC_KEYS,
    ConditionSummary,
    aggregate_conditions,
    bootstrap_ci,
    load_manifest_entries,
    welch_ttest,
)

_METRIC_CHOICES = NUMERIC_METRIC_KEYS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate repeated-run crossover-search manifests and run "
            "significance tests."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "manifests",
        nargs="+",
        metavar="MANIFEST_JSON",
        help="One or more manifest.json files (one per seed run).",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["crossover_mode", "finetune_regime"],
        metavar="FIELD",
        help=(
            "Manifest fields to group by when computing per-condition summaries. "
            "Default: crossover_mode finetune_regime."
        ),
    )
    parser.add_argument(
        "--metric",
        default="primary_metric",
        metavar="FIELD",
        choices=list(_METRIC_CHOICES),
        help=(
            "Numeric manifest field for the main Mean/Std column and comparisons. "
            f"Default: primary_metric. Choices: {', '.join(_METRIC_CHOICES)}."
        ),
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("VALUE_A", "VALUE_B"),
        default=None,
        help=(
            "Two condition values to compare with a Welch t-test and bootstrap CI "
            "(e.g. 'weighted random').  Use --compare-field to specify which column "
            "the values refer to."
        ),
    )
    parser.add_argument(
        "--compare-field",
        default="crossover_mode",
        metavar="FIELD",
        help="Manifest field used to select rows for --compare.  Default: crossover_mode.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        metavar="LEVEL",
        help="Confidence level for bootstrap CI (default: 0.95).",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        metavar="N",
        help="Number of bootstrap resamples (default: 2000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for bootstrap resampling (default: random).",
    )
    parser.add_argument(
        "--output-json",
        metavar="PATH",
        default=None,
        help="If given, write a JSON summary of all conditions to this file.",
    )
    return parser.parse_args(argv)


def _fmt_float(v: float) -> str:
    if v != v:  # NaN
        return "    N/A"
    return f"{v:7.4f}"


def _summary_metric_mean(summary: ConditionSummary, metric: str) -> float:
    """Mean of *metric* for one ConditionSummary (for sorting / display)."""
    if metric == "primary_metric":
        return float(summary.mean_primary_metric)
    extra = summary.extra.get(metric, {})
    return float(extra.get("mean", float("nan")))


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print(f"\nLoading {len(args.manifests)} manifest file(s) …")
    rows = load_manifest_entries(args.manifests)
    print(f"  Total rows loaded: {len(rows)}")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    extra_metrics = (args.metric,) if args.metric != "primary_metric" else ()
    summaries = aggregate_conditions(rows, group_by=args.group_by, extra_metrics=extra_metrics)

    if not summaries:
        print("\nNo rows found after grouping.  Check --group-by fields.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    metric = args.metric

    print(f"\nCondition Summary  (metric: {metric})")
    print("=" * 100)

    col_w = [40, 7, 10, 10, 12, 12, 12]
    header = ["Condition", "N runs", "Mean", "Std", "Agree-A", "Agree-B", "N degen"]
    print("".join(h.ljust(w) for h, w in zip(header, col_w)))
    print("-" * 100)

    for key, s in sorted(
        summaries.items(),
        key=lambda kv: -_summary_metric_mean(kv[1], metric),
    ):
        cond_str = ", ".join(f"{k}={v}" for k, v in key)
        # Get mean/std for requested metric.
        if metric == "primary_metric":
            mean_val = s.mean_primary_metric
            std_val = s.std_primary_metric
        else:
            extra = s.extra.get(metric, {})
            mean_val = float(extra.get("mean", float("nan")))
            std_val = float(extra.get("std", float("nan")))

        row = [
            cond_str[:39],
            str(s.n_runs),
            _fmt_float(mean_val),
            _fmt_float(std_val),
            _fmt_float(s.mean_agree_a),
            _fmt_float(s.mean_agree_b),
            str(s.n_degenerate),
        ]
        print("".join(v.ljust(w) for v, w in zip(row, col_w)))

    print("=" * 100)

    # ------------------------------------------------------------------
    # Significance tests
    # ------------------------------------------------------------------
    if args.compare:
        val_a, val_b = args.compare
        field = args.compare_field

        group_a = [r.get(metric) for r in rows if r.get(field) == val_a]
        group_b = [r.get(metric) for r in rows if r.get(field) == val_b]

        group_a = [v for v in group_a if v is not None]
        group_b = [v for v in group_b if v is not None]

        print(f"\nComparison: {field}={val_a!r} vs {field}={val_b!r}  (metric: {metric})")
        print(f"  Group A ({val_a}): N={len(group_a)}")
        print(f"  Group B ({val_b}): N={len(group_b)}")

        if len(group_a) < 2 or len(group_b) < 2:
            print("  [!] Need at least 2 observations per group for significance tests.")
        else:
            ttest = welch_ttest(group_a, group_b)
            print("\n  Welch t-test:")
            print(f"    t = {ttest.statistic:.4f}   p = {ttest.pvalue:.4f}   dof = {ttest.dof:.1f}")
            print(f"    mean_A = {ttest.mean_a:.4f}   mean_B = {ttest.mean_b:.4f}")
            print(f"    mean_diff (A − B) = {ttest.mean_diff:.4f}")
            sig = "< 0.05 → statistically significant" if ttest.pvalue < 0.05 else "≥ 0.05 → not significant"
            print(f"    p {sig}")

            ci_a = bootstrap_ci(
                group_a,
                confidence_level=args.confidence,
                n_bootstrap=args.n_bootstrap,
                rng=args.bootstrap_seed,
            )
            ci_b = bootstrap_ci(
                group_b,
                confidence_level=args.confidence,
                n_bootstrap=args.n_bootstrap,
                rng=args.bootstrap_seed,
            )
            pct = int(args.confidence * 100)
            print(f"\n  Bootstrap {pct}% CI (n_resamples={args.n_bootstrap}):")
            print(
                f"    {val_a}: {ci_a.point_estimate:.4f}  "
                f"[{ci_a.ci_low:.4f}, {ci_a.ci_high:.4f}]"
            )
            print(
                f"    {val_b}: {ci_b.point_estimate:.4f}  "
                f"[{ci_b.ci_low:.4f}, {ci_b.ci_high:.4f}]"
            )

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    if args.output_json:
        out = {
            "manifests": args.manifests,
            "group_by": args.group_by,
            "metric": metric,
            "conditions": [s.to_dict() for s in summaries.values()],
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, allow_nan=False)
        print(f"\nJSON summary written to: {args.output_json}")

    print()


if __name__ == "__main__":
    main()

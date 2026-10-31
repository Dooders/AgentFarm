"""Calibration report for an intrinsic cooperation pilot.

Reads one ``intrinsic_cooperation_summary.json`` (from a single run or from
``merge_intrinsic_cooperation_runs.py``) and reports what the pre-registration
needs before the confirmatory seeds are spent:

- learning positive controls (L1/L2) per replicate;
- the population regime (how close each arm sits to the cap, turnover,
  deaths by cause, lifespans, generations reached);
- the institution's activity (coverage, units, unit lifetime, forfeits per
  enforced exit);
- neutral drift of every cooperation locus in the placebo (the noise S4 is
  tested against) and the paired cooperation - placebo differences;
- a power estimate for every registered hypothesis at the confirmatory ``n``
  under the Holm family's strictest threshold, with the minimum detectable
  effect and the replicates needed for the pilot effect.

Usage::

    python scripts/calibrate_intrinsic_cooperation.py <summary.json> [--output-dir DIR] [--target-n 20]

Writes ``calibration.md`` and ``calibration.json`` to ``--output-dir``
(default: next to the summary) and prints the markdown.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats as scipy_stats

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.core.cooperation.genes import COOPERATION_GENE_NAMES  # noqa: E402
from farm.runners.intrinsic_cooperation_experiment import (  # noqa: E402
    ARM_CONTRASTS,
    HYPOTHESES,
    LEARNING_CHECKS,
    metric_arrays,
)

REPORT_FILE = "calibration.md"
REPORT_JSON = "calibration.json"

#: Wilcoxon signed-rank vs paired t asymptotic relative efficiency under
#: normal differences; the power model below is a paired t at this fraction
#: of the sample size.
WILCOXON_ARE = 0.955

#: Per-arm descriptives reported in the regime table.
REGIME_METRICS: tuple[str, ...] = (
    "mean_population",
    "peak_population",
    "final_population",
    "total_births",
    "total_deaths",
    "starvation_deaths",
    "combat_deaths",
    "mean_lifespan_at_death",
    "max_generation",
    "mean_coverage",
    "final_coverage",
    "units_formed",
    "mean_unit_lifetime",
    "forfeits_per_enforced_exit",
    "action_share[bond]",
    "action_share[leave]",
    "policy_bond_prob_late",
    "acting_bond_prob_late",
    "bond_rank_late",
    "policy_peak_late",
)

#: Pre-flight gates: ``name -> (statement, predicate over the report dict)``.
GATES: dict[str, str] = {
    "learning_L1": "Policies prefer gather over uniform late in the run (L1 > 0) in every baseline replicate.",
    "learning_L2": "Late gather share exceeds the innate prior (L2 > 0) in every baseline replicate.",
    "units_form": "Units form in the cooperation arm (mean coverage > 0.05 and units_formed > 0 in every replicate).",
    "generations": "Selection has material: max_generation >= 3 in every cooperation replicate.",
    "not_pinned": "The population is regulated below the cap: mean_population < 0.95 x max_population in every arm.",
    "exit_cost_bounded": "Enforced exits cost something finite and bounded (0 < forfeits per enforced exit < 10).",
}


# ------------------------------------------------------------------ power
def _finite(x: float) -> float:
    # scipy's non-central t underflows to nan far in a tail; that mass is zero.
    return float(x) if np.isfinite(x) else 0.0


def power_paired(dz: float, n: int, alpha: float, direction: str) -> float:
    """Power of a paired test at ``n`` pairs for a standardised effect ``dz``.

    ``dz`` is signed in the hypothesised direction for one-sided tests, so a
    pilot effect that points the wrong way gets the power it deserves.
    """
    n_eff = max(2.0, WILCOXON_ARE * n)
    df = n_eff - 1
    if direction == "two-sided":
        nc = abs(dz) * math.sqrt(n_eff)
        crit = scipy_stats.t.ppf(1 - alpha / 2, df)
        return _finite(scipy_stats.nct.sf(crit, df, nc)) + _finite(scipy_stats.nct.cdf(-crit, df, nc))
    nc = dz * math.sqrt(n_eff)
    crit = scipy_stats.t.ppf(1 - alpha, df)
    return _finite(scipy_stats.nct.sf(crit, df, nc))


def mde_paired(n: int, alpha: float, direction: str, power: float = 0.8) -> float:
    """Smallest ``dz`` with at least ``power`` at ``n`` pairs (bisection)."""
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_paired(mid, n, alpha, direction) < power:
            lo = mid
        else:
            hi = mid
    return hi


def n_for_power(dz: float, alpha: float, direction: str, power: float = 0.8, n_max: int = 1000) -> int | None:
    """Pairs needed for ``power``; ``None`` when no ``n <= n_max`` reaches it."""
    if not np.isfinite(dz) or dz == 0.0 or (direction != "two-sided" and dz < 0.0):
        return None
    for n in range(3, n_max + 1):
        if power_paired(dz, n, alpha, direction) >= power:
            return n
    return None


def _signed(diff: np.ndarray, direction: str) -> np.ndarray:
    # The power model wants the effect in the hypothesised direction.
    return -diff if direction == "less" else diff


# ----------------------------------------------------------------- report
def _describe(values: np.ndarray) -> dict[str, float]:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return {"mean": float("nan"), "sd": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
    return {"mean": float(v.mean()), "sd": sd, "min": float(v.min()), "max": float(v.max()), "n": int(v.size)}


def build_report(summary: Mapping[str, Any], target_n: int = 20, power: float = 0.8) -> dict[str, Any]:
    config = summary["config"]
    replicates: Sequence[Mapping[str, Any]] = summary["replicates"]
    arms = [a for a in config["arms"] if all(a in r for r in replicates)]
    arrays = metric_arrays(replicates, arms)
    alpha = float(config.get("alpha", 0.05))
    cap = config.get("max_population")

    regime = {arm: {m: _describe(arrays[arm][m]) for m in REGIME_METRICS if m in arrays[arm]} for arm in arms}

    learning: dict[str, dict[str, Any]] = {}
    for key, (metric, arm, _direction, statement) in LEARNING_CHECKS.items():
        if arm in arrays and metric in arrays[arm]:
            values = arrays[arm][metric]
            learning[key] = {
                "statement": statement,
                "metric": metric,
                "arm": arm,
                "per_replicate": [float(x) for x in values],
                "all_positive": bool(np.all(values[np.isfinite(values)] > 0)) and bool(np.isfinite(values).any()),
            }
    # The same two numbers in every arm, so a reader can see learning is not arm-specific.
    learning_by_arm = {
        arm: {m: _describe(arrays[arm][m]) for m in ("policy_gather_lift_late", "gather_share_over_prior_late")}
        for arm in arms
    }

    drift: dict[str, dict[str, Any]] = {}
    if "placebo" in arrays:
        loci = [f"change_{g}" for g in COOPERATION_GENE_NAMES] + ["change_expected_exit_payment"]
        for metric in loci:
            if metric not in arrays["placebo"]:
                continue
            entry: dict[str, Any] = {"placebo": _describe(arrays["placebo"][metric])}
            if "cooperation" in arrays:
                diff = arrays["cooperation"][metric] - arrays["placebo"][metric]
                entry["cooperation"] = _describe(arrays["cooperation"][metric])
                entry["paired_diff"] = _describe(diff)
            drift[metric] = entry

    m_family = len(HYPOTHESES)
    alpha_strict = alpha / m_family
    hypotheses: dict[str, dict[str, Any]] = {}
    for key, (metric, contrast, direction, _statement) in HYPOTHESES.items():
        b, t = ARM_CONTRASTS[contrast]
        if b not in arrays or t not in arrays or metric not in arrays[t]:
            continue
        diff = arrays[t][metric] - arrays[b][metric]
        diff = diff[np.isfinite(diff)]
        if diff.size < 2:
            continue
        signed = _signed(diff, direction)
        sd = float(np.std(signed, ddof=1))
        dz = float(signed.mean() / sd) if sd > 0 else (math.inf if signed.mean() != 0 else 0.0)
        hypotheses[key] = {
            "metric": metric,
            "contrast": contrast,
            "direction": direction,
            "pilot_n": int(diff.size),
            "mean_diff": float(diff.mean()),
            "sd_diff": sd,
            "dz": dz,
            "power_at_target_n": power_paired(dz, target_n, alpha_strict, direction) if np.isfinite(dz) else 1.0,
            "mde_dz_at_target_n": mde_paired(target_n, alpha_strict, direction, power),
            "n_for_power": n_for_power(dz, alpha_strict, direction, power) if np.isfinite(dz) else 3,
        }

    gates = _evaluate_gates(arrays, arms, learning, cap)
    return {
        "summary_config": {
            k: config.get(k)
            for k in (
                "num_steps",
                "seeds",
                "seed",
                "num_replicates",
                "arms",
                "max_population",
                "selection_pressure",
                "policy_temperature",
                "initial_diversity_scale",
                "freeze_platform_genes",
                "git_commit",
            )
        },
        "cooperation_config": config.get("cooperation"),
        "target_n": target_n,
        "alpha": alpha,
        "family_size": m_family,
        "alpha_strict": alpha_strict,
        "regime": regime,
        "learning": learning,
        "learning_by_arm": learning_by_arm,
        "drift": drift,
        "hypotheses": hypotheses,
        "gates": gates,
    }


def _evaluate_gates(
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    arms: Sequence[str],
    learning: Mapping[str, Mapping[str, Any]],
    cap: Any,
) -> dict[str, dict[str, Any]]:
    def gate(name: str, passed: bool | None, detail: str) -> dict[str, Any]:
        return {"statement": GATES[name], "passed": passed, "detail": detail}

    out: dict[str, dict[str, Any]] = {}
    for key, name in (("L1", "learning_L1"), ("L2", "learning_L2")):
        if key in learning:
            vals = learning[key]["per_replicate"]
            out[name] = gate(name, learning[key]["all_positive"], f"{key} per replicate: {[round(v, 3) for v in vals]}")
        else:
            out[name] = gate(name, None, "baseline arm not run")
    coop = arrays.get("cooperation")
    if coop is not None:
        cov, units = coop["mean_coverage"], coop["units_formed"]
        out["units_form"] = gate(
            "units_form",
            bool(np.all(cov > 0.05) and np.all(units > 0)),
            f"mean coverage {np.round(cov, 3).tolist()}, units formed {units.astype(int).tolist()}",
        )
        gen = coop["max_generation"]
        out["generations"] = gate("generations", bool(np.all(gen >= 3)), f"max generation {gen.astype(int).tolist()}")
        fpe = coop["forfeits_per_enforced_exit"]
        finite = fpe[np.isfinite(fpe)]
        out["exit_cost_bounded"] = gate(
            "exit_cost_bounded",
            bool(finite.size and np.all((finite > 0) & (finite < 10))),
            f"forfeits per enforced exit {np.round(fpe, 3).tolist()}",
        )
    else:
        for name in ("units_form", "generations", "exit_cost_bounded"):
            out[name] = gate(name, None, "cooperation arm not run")
    if cap:
        ratios = {arm: float(np.mean(arrays[arm]["mean_population"])) / float(cap) for arm in arms}
        out["not_pinned"] = gate(
            "not_pinned",
            bool(all(r < 0.95 for r in ratios.values())),
            "mean population / cap: " + ", ".join(f"{a} {r:.2f}" for a, r in ratios.items()),
        )
    else:
        out["not_pinned"] = gate("not_pinned", None, "no population cap in the config")
    return out


# --------------------------------------------------------------- markdown
def _fmt(x: Any, digits: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf"
        return f"{x:.{digits}g}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:.{digits}f}"
    return str(x)


def render_markdown(report: Mapping[str, Any]) -> str:
    cfg = report["summary_config"]
    lines = ["# Intrinsic cooperation calibration", ""]
    lines.append(
        f"Steps {cfg.get('num_steps')}, seeds {cfg.get('seeds') or cfg.get('seed')}, arms {cfg.get('arms')}, "
        f"cap {cfg.get('max_population')}, selection {cfg.get('selection_pressure')}, "
        f"temperature {cfg.get('policy_temperature')}, initial diversity {cfg.get('initial_diversity_scale')}, "
        f"frozen A/C {cfg.get('freeze_platform_genes')}, commit `{(cfg.get('git_commit') or 'unknown')[:10]}`."
    )
    lines += ["", "## Gates", "", "| Gate | Result | Detail |", "| --- | --- | --- |"]
    for name, g in report["gates"].items():
        verdict = "—" if g["passed"] is None else ("pass" if g["passed"] else "FAIL")
        lines.append(f"| {name} | {verdict} | {g['statement']} {g['detail']} |")

    lines += ["", "## Learning positive controls", ""]
    for key, entry in report["learning"].items():
        lines.append(
            f"- **{key}** (`{entry['metric']}`, {entry['arm']}): {[round(v, 3) for v in entry['per_replicate']]}"
        )
    lines += ["", "| Arm | gather lift (policy − uniform) | gather share − prior |", "| --- | --- | --- |"]
    for arm, entry in report["learning_by_arm"].items():
        a, b = entry["policy_gather_lift_late"], entry["gather_share_over_prior_late"]
        lines.append(f"| {arm} | {_fmt(a['mean'])} ± {_fmt(a['sd'])} | {_fmt(b['mean'])} ± {_fmt(b['sd'])} |")

    arms = list(report["regime"])
    lines += ["", "## Regime (mean ± sd over replicates)", "", "| Metric | " + " | ".join(arms) + " |"]
    lines.append("| --- |" + " --- |" * len(arms))
    for metric in REGIME_METRICS:
        row = [
            f"{_fmt(report['regime'][a][metric]['mean'])} ± {_fmt(report['regime'][a][metric]['sd'])}"
            if metric in report["regime"][a]
            else "—"
            for a in arms
        ]
        lines.append(f"| `{metric}` | " + " | ".join(row) + " |")

    if report["drift"]:
        lines += [
            "",
            "## Locus drift (run-long change in the population mean)",
            "",
            "| Locus | placebo mean ± sd | cooperation mean ± sd | paired diff mean ± sd |",
            "| --- | --- | --- | --- |",
        ]
        for metric, entry in report["drift"].items():
            p = entry["placebo"]
            c = entry.get("cooperation", {})
            d = entry.get("paired_diff", {})
            lines.append(
                f"| `{metric}` | {_fmt(p['mean'])} ± {_fmt(p['sd'])} | "
                f"{_fmt(c.get('mean', float('nan')))} ± {_fmt(c.get('sd', float('nan')))} | "
                f"{_fmt(d.get('mean', float('nan')))} ± {_fmt(d.get('sd', float('nan')))} |"
            )

    lines += [
        "",
        (
            f"## Power at n = {report['target_n']} (paired, Holm strictest alpha = {report['alpha_strict']:.4g}, "
            f"family of {report['family_size']})"
        ),
        "",
        "| Key | Metric | Direction | Pilot mean diff | Pilot sd | dz (toward hypothesis) | Power | MDE dz | n for 80% |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, h in report["hypotheses"].items():
        lines.append(
            f"| {key} | `{h['metric']}` | {h['direction']} | {_fmt(h['mean_diff'])} | {_fmt(h['sd_diff'])} | "
            f"{_fmt(h['dz'])} | {_fmt(h['power_at_target_n'])} | {_fmt(h['mde_dz_at_target_n'])} | "
            f"{_fmt(h['n_for_power'])} |"
        )
    lines += [
        "",
        (
            "Power uses a paired t at the Wilcoxon asymptotic relative efficiency (0.955 n); pilot effect sizes "
            "from three seeds are rough and the MDE column is the number to plan against."
        ),
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary", help="intrinsic_cooperation_summary.json (single run or merged).")
    parser.add_argument("--output-dir", default=None, help="Where to write calibration.md/json (default: beside it).")
    parser.add_argument("--target-n", type=int, default=20, help="Confirmatory replicates to size power for.")
    parser.add_argument("--power", type=float, default=0.8)
    args = parser.parse_args(argv)

    with open(args.summary, encoding="utf-8") as handle:
        summary = json.load(handle)
    if len(summary.get("replicates") or []) < 2:
        raise SystemExit("calibration needs at least two replicates (merge the seeds first).")
    report = build_report(summary, target_n=args.target_n, power=args.power)
    markdown = render_markdown(report)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.summary))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, REPORT_JSON), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    with open(os.path.join(out_dir, REPORT_FILE), "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

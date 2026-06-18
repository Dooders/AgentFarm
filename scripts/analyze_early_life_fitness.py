#!/usr/bin/env python3
"""Measure early-life offspring fitness in a Baldwinian-vs-Lamarckian A/B sweep.

The 05-21 inheritance A/B showed that Lamarckian policy warm-start fires for
~85% of reproduction events yet whole-population summaries (population,
speciation) do not move robustly. This script tests the natural follow-up: if
the mechanism does anything, it should show up *most strongly right after
birth*, before ecology washes it out.

For each paired (profile, seed) run across both arms it reads the per-agent
SQLite database, identifies offspring (born after the warmup window), and
scores their first ``N`` steps of life. It then computes paired deltas
(Lamarckian - Baldwinian) per profile and applies the same robustness gate used
elsewhere in the project: 95% CI excludes zero AND within-profile sign
agreement >= 75%.

Cohort note
-----------
Per-offspring "warm-start applied vs skipped" is not recoverable from the
stored artifacts (only aggregate counts live in
``intrinsic_evolution_metadata.json``), so the comparison is at the *arm* level:
Lamarckian offspring (~85% warm-started) vs Baldwinian offspring (cold start by
design). The per-run warm-start rate is reported alongside for context.

Reward signals (two distinct channels)
--------------------------------------
This script touches two unrelated reward columns; do not conflate them.

- ``agent_states.total_reward`` is the **cumulative RL reward at that step**
  (sum of per-step ``resource_delta + 0.5*health_delta + survival_bonus``;
  see ``farm/core/agent/components/reward.py``). It is reported here as
  ``rl_reward_at_age`` and as the reward-vs-age curve.
- ``agent_actions.reward`` is a separate **per-action module reward** (nearly
  constant at ~0.135 for most actions). It only feeds ``decision_success_rate``
  (the fraction of early actions that returned a positive value), which is a
  coarse "how often did an action avoid a negative outcome" readout.

Cohort caveat
-------------
The RL-reward readouts (``rl_reward_at_age``, ``resource_at_age``,
``parent_reward_gap``) are necessarily conditioned on offspring that *survived*
to age N. ``decision_success_rate`` instead spans every offspring that took an
action in the first N steps, including those that died before N. These cohorts
differ (sizes are reported as ``n_reached`` and ``n_acted``), so the reward and
action-fraction deltas should not be read as two measurements of the same
population.

Lineage
-------
``agents.genome_id`` encodes 0, 1, or 2 parents plus an optional counter
(``"::n"``, ``"<parent>:n"``, or ``"<parent1>:<parent2>:n"``). This script uses
parent-anchored reward gaps only when exactly one parent is encoded; founder and
two-parent offspring are excluded from that metric.

Usage
-----
::

    python scripts/analyze_early_life_fitness.py \\
        --ab-dir experiments/inheritance_ab \\
        --baseline-arm baldwinian \\
        --treatment-arm lamarckian
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.analyze_stable_profile_seed_sweep import (  # noqa: E402
    PROFILE_ORDER,
    SIGN_AGREEMENT_THRESHOLD,
    _mean,
    _sign_agreement,
    _t_ci,
    _variance,
)
from farm.database.data_types import GenomeId  # noqa: E402

# Ages (in steps lived) at which offspring are scored.
DEFAULT_AGES: Tuple[int, ...] = (10, 25, 50)

# Metrics whose paired delta we report and gate. Each maps to a human label and
# the direction in which "Lamarckian helps" points (+1 means higher is better).
METRIC_LABELS: Dict[str, str] = {
    "survival_rate": "survival to age N",
    "rl_reward_at_age": "net RL reward at age N (resource+health+survival delta)",
    "decision_success_rate": "positive-reward action fraction (first N steps)",
    "resource_at_age": "resource level at age N",
    "parent_reward_gap": "|offspring - parent| RL-reward gap (lower = closer)",
}


# ── SQLite extraction ──────────────────────────────────────────────────────────

def _find_db(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("simulation_sim_*.db"))
    if not candidates:
        candidates = sorted(run_dir.glob("*.db"))
    if len(candidates) > 1:
        print(
            f"  WARNING: {run_dir} has {len(candidates)} candidate DBs; "
            f"using {candidates[0].name}",
            file=sys.stderr,
        )
    return candidates[0] if candidates else None


DEFAULT_WARMUP = 200


def _read_warmup(run_dir: Path) -> int:
    meta_path = run_dir / "intrinsic_evolution_metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        for key in ("initial_conditions", "resolved_initial_conditions"):
            block = meta.get(key)
            if isinstance(block, dict) and block.get("warmup_steps") is not None:
                return int(block["warmup_steps"])
    print(
        f"  WARNING: {run_dir} missing warmup_steps metadata; "
        f"falling back to {DEFAULT_WARMUP}",
        file=sys.stderr,
    )
    return DEFAULT_WARMUP


def _read_warmstart_rate(run_dir: Path) -> Optional[float]:
    meta_path = run_dir / "intrinsic_evolution_metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    pim = meta.get("policy_inheritance_metrics")
    if not isinstance(pim, dict):
        return None
    applied = pim.get("warmstart_applied")
    skipped = pim.get("warmstart_skipped")
    if applied is None or skipped is None:
        return None
    total = float(applied) + float(skipped)
    return float(applied) / total if total > 0 else float("nan")


def _parent_of(genome_id: Optional[str]) -> Optional[str]:
    """Return parent id only for single-parent genome IDs.

    Genome IDs can encode zero, one, or two parents. Parent-anchored reward
    gap is only well-defined for single-parent lineage, so founders and
    two-parent offspring resolve to ``None``.
    """
    if not genome_id:
        return None
    try:
        parsed = GenomeId.from_string(genome_id)
    except Exception:
        return None
    if len(parsed.parent_ids) != 1:
        return None
    return parsed.parent_ids[0]


def _extract_run_early_life(
    db_path: Path, warmup: int, ages: Sequence[int]
) -> Optional[Dict[str, Any]]:
    """Compute per-run early-life offspring fitness from one simulation DB."""
    con = sqlite3.connect(str(db_path))
    try:
        cur = con.cursor()
        last_step = cur.execute("SELECT MAX(step_number) FROM agent_states").fetchone()[0]
        if last_step is None:
            return None
        last_step = int(last_step)

        # Agents: id -> (birth, death, parent_id).
        agents: Dict[str, Tuple[int, Optional[int], Optional[str]]] = {}
        for agent_id, birth, death, genome_id in cur.execute(
            "SELECT agent_id, birth_time, death_time, genome_id FROM agents"
        ):
            agents[agent_id] = (
                int(birth),
                int(death) if death is not None else None,
                _parent_of(genome_id),
            )

        offspring = {
            aid: info for aid, info in agents.items() if info[0] > warmup
        }
        if not offspring:
            return None

        # Per-agent (cumulative RL reward, resource level) by absolute step:
        # needed for age-N snapshots (offspring) and parent-window reward lookups.
        states: Dict[str, Dict[int, Tuple[float, float]]] = defaultdict(dict)
        for agent_id, step, total_reward, resource_level in cur.execute(
            "SELECT agent_id, step_number, total_reward, resource_level "
            "FROM agent_states"
        ):
            states[agent_id][int(step)] = (
                float(total_reward),
                float(resource_level),
            )

        # Per-agent actions (step, reward) for decision-success over first N.
        actions: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for agent_id, step, reward in cur.execute(
            "SELECT agent_id, step_number, reward FROM agent_actions"
        ):
            if reward is not None:
                actions[agent_id].append((int(step), float(reward)))
    finally:
        con.close()

    per_age: Dict[int, Dict[str, float]] = {}
    reward_by_age: Dict[int, List[float]] = defaultdict(list)

    for age in ages:
        survived: List[float] = []
        reward_vals: List[float] = []
        resource_vals: List[float] = []
        success_vals: List[float] = []
        parent_gap_vals: List[float] = []

        for aid, (birth, death, parent_id) in offspring.items():
            # Right-censoring: only count offspring that *could* reach age N
            # before the simulation ended.
            if birth + age > last_step:
                continue
            lifespan = (death if death is not None else last_step) - birth
            reached = lifespan >= age
            survived.append(1.0 if reached else 0.0)

            agent_states = states.get(aid, {})
            snap = agent_states.get(birth + age)
            if reached and snap is not None:
                reward_vals.append(snap[0])
                resource_vals.append(snap[1])

                # Parent-anchored gap: RL reward the parent earned over the same
                # absolute window [birth, birth+age].
                if parent_id is not None:
                    p_states = states.get(parent_id, {})
                    p_start = p_states.get(birth)
                    p_end = p_states.get(birth + age)
                    if p_start is not None and p_end is not None:
                        parent_window = p_end[0] - p_start[0]
                        parent_gap_vals.append(abs(snap[0] - parent_window))

            # Positive-reward action fraction over all actions taken within the
            # first N *steps* of life (agents take multiple actions per step).
            acts = [r for (s, r) in actions.get(aid, []) if birth <= s < birth + age]
            if acts:
                success_vals.append(sum(1.0 for r in acts if r > 0) / len(acts))

        per_age[age] = {
            "n_uncensored": float(len(survived)),
            "n_reached": float(len(reward_vals)),
            # Cohort note: rl_reward_at_age / resource_at_age / parent_reward_gap
            # are conditioned on *surviving* to age N (n_reached), whereas
            # decision_success_rate is over every offspring that took an action
            # in the first N steps (n_acted), which includes early deaths. The
            # two cohorts differ, so don't read the reward and action-fraction
            # deltas as if measured on the same population.
            "n_acted": float(len(success_vals)),
            "survival_rate": _mean(survived),
            "rl_reward_at_age": _mean(reward_vals),
            "resource_at_age": _mean(resource_vals),
            "decision_success_rate": _mean(success_vals),
            "parent_reward_gap": _mean(parent_gap_vals),
        }

    # RL-reward-vs-age and survival-vs-age curves (pooled offspring) for the
    # decay and survival figures.
    max_age = max(ages)
    survival_reached: Dict[int, int] = defaultdict(int)
    survival_eligible: Dict[int, int] = defaultdict(int)
    for aid, (birth, death, _parent) in offspring.items():
        agent_states = states.get(aid, {})
        lifespan = (death if death is not None else last_step) - birth
        for a in range(1, max_age + 1):
            if birth + a > last_step:
                break
            survival_eligible[a] += 1
            if lifespan >= a:
                survival_reached[a] += 1
            snap = agent_states.get(birth + a)
            if snap is not None:
                reward_by_age[a].append(snap[0])

    rl_reward_curve = {
        a: _mean(vals) for a, vals in sorted(reward_by_age.items()) if vals
    }
    survival_curve = {
        a: survival_reached[a] / survival_eligible[a]
        for a in sorted(survival_eligible)
        if survival_eligible[a] > 0
    }

    return {
        "n_offspring": len(offspring),
        "last_step": last_step,
        "per_age": per_age,
        "rl_reward_curve": rl_reward_curve,
        "survival_curve": survival_curve,
    }


# ── Discovery ───────────────────────────────────────────────────────────────────

def _discover_arm_runs(
    arm_dir: Path, profiles: Sequence[str]
) -> Dict[str, List[Tuple[int, Path]]]:
    """Discover ``stable_{profile}/seed_{seed}`` run dirs under one arm."""
    found: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for profile in profiles:
        profile_dir = arm_dir / f"stable_{profile}"
        if not profile_dir.is_dir():
            continue
        for seed_dir in sorted(profile_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except ValueError:
                continue
            found[profile].append((seed, seed_dir))
        found[profile].sort(key=lambda t: t[0])
    return dict(found)


# ── Paired deltas ────────────────────────────────────────────────────────────────

def _verdict(deltas: Sequence[float]) -> Dict[str, Any]:
    """Robustness verdict for a list of paired deltas (one per seed)."""
    vals = [d for d in deltas if not math.isnan(d)]
    if len(vals) < 2:
        return {
            "n": len(vals),
            "mean_delta": _mean(vals) if vals else float("nan"),
            "ci95": [float("nan"), float("nan")],
            "sign_agreement": float("nan"),
            "ci_excludes_zero": False,
            "robust": False,
        }
    lo, hi = _t_ci(vals)
    sign_agreement = _sign_agreement(vals)
    ci_excludes_zero = (lo > 0 and hi > 0) or (lo < 0 and hi < 0)
    return {
        "n": len(vals),
        "mean_delta": _mean(vals),
        "variance": _variance(vals),
        "ci95": [lo, hi],
        "sign_agreement": sign_agreement,
        "ci_excludes_zero": ci_excludes_zero,
        "robust": ci_excludes_zero and sign_agreement >= SIGN_AGREEMENT_THRESHOLD,
    }


def _pair_profile(
    baseline: Dict[int, Dict[str, Any]],
    treatment: Dict[int, Dict[str, Any]],
    ages: Sequence[int],
) -> Dict[str, Any]:
    """Compute paired (treatment - baseline) deltas for one profile."""
    seeds = sorted(set(baseline) & set(treatment))
    out: Dict[str, Any] = {"seeds": seeds, "n_seeds": len(seeds), "ages": {}}

    for age in ages:
        metric_deltas: Dict[str, List[float]] = {m: [] for m in METRIC_LABELS}
        per_seed: Dict[int, Dict[str, float]] = {}
        for seed in seeds:
            b = baseline[seed]["per_age"].get(age, {})
            t = treatment[seed]["per_age"].get(age, {})
            seed_row: Dict[str, float] = {}
            for metric in METRIC_LABELS:
                bv = b.get(metric, float("nan"))
                tv = t.get(metric, float("nan"))
                delta = (
                    float(tv) - float(bv)
                    if not (math.isnan(bv) or math.isnan(tv))
                    else float("nan")
                )
                metric_deltas[metric].append(delta)
                seed_row[metric] = delta
            per_seed[seed] = seed_row

        out["ages"][age] = {
            "per_seed_delta": per_seed,
            "verdicts": {m: _verdict(metric_deltas[m]) for m in METRIC_LABELS},
            "baseline_means": {
                m: _mean([baseline[s]["per_age"].get(age, {}).get(m, float("nan"))
                          for s in seeds])
                for m in METRIC_LABELS
            },
            "treatment_means": {
                m: _mean([treatment[s]["per_age"].get(age, {}).get(m, float("nan"))
                          for s in seeds])
                for m in METRIC_LABELS
            },
        }
    return out


# ── Markdown ─────────────────────────────────────────────────────────────────────

def _fmt(v: Any, decimals: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if math.isnan(v):
            return "n/a"
        return f"{v:.{decimals}f}"
    if isinstance(v, (list, tuple)):
        return f"[{_fmt(v[0], decimals)}, {_fmt(v[1], decimals)}]"
    return str(v)


def _build_markdown(
    paired: Dict[str, Dict[str, Any]],
    warmstart_rates: Dict[str, float],
    ages: Sequence[int],
    baseline_arm: str,
    treatment_arm: str,
) -> str:
    profiles = [p for p in PROFILE_ORDER if p in paired]
    lines: List[str] = [
        f"# Early-life offspring fitness: {treatment_arm} vs {baseline_arm}",
        "",
        "Paired deltas are computed per (profile, seed) as "
        f"`{treatment_arm} - {baseline_arm}` and gated with the project rule: "
        "95% CI excludes zero AND within-profile sign agreement >= 0.75.",
        "",
    ]

    if warmstart_rates:
        lines += ["## Warm-start coverage (treatment arm)", ""]
        lines += ["| Profile | Mean warm-start rate |", "| --- | --- |"]
        for profile in profiles:
            lines.append(
                f"| {profile} | {_fmt(warmstart_rates.get(profile, float('nan')), 3)} |"
            )
        lines.append("")

    for age in ages:
        lines += [f"## First {age} steps of life", ""]
        lines += [
            "| Profile | Metric | "
            f"{baseline_arm} mean | {treatment_arm} mean | Mean Δ | 95% CI | "
            "Sign agr. | Verdict |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for profile in profiles:
            age_block = paired[profile]["ages"].get(age, {})
            verdicts = age_block.get("verdicts", {})
            bmeans = age_block.get("baseline_means", {})
            tmeans = age_block.get("treatment_means", {})
            for metric, label in METRIC_LABELS.items():
                v = verdicts.get(metric, {})
                verdict = "robust" if v.get("robust") else "no robust effect"
                lines.append(
                    f"| {profile} | {label} "
                    f"| {_fmt(bmeans.get(metric), 3)} "
                    f"| {_fmt(tmeans.get(metric), 3)} "
                    f"| {_fmt(v.get('mean_delta'), 3)} "
                    f"| {_fmt(v.get('ci95'), 3)} "
                    f"| {_fmt(v.get('sign_agreement'), 2)} "
                    f"| {verdict} |"
                )
        lines.append("")

    # Headline: any robust effects?
    robust_hits: List[str] = []
    for profile in profiles:
        for age in ages:
            verdicts = paired[profile]["ages"].get(age, {}).get("verdicts", {})
            for metric, v in verdicts.items():
                if v.get("robust"):
                    robust_hits.append(
                        f"{profile} / {METRIC_LABELS[metric]} / N={age} "
                        f"(Δ {_fmt(v.get('mean_delta'), 3)})"
                    )
    lines += ["## Headline", ""]
    if robust_hits:
        lines.append("Robust effects (CI excludes zero, sign agreement >= 0.75):")
        lines += [f"- {hit}" for hit in robust_hits]
    else:
        lines.append(
            "**No robust early-life effect** in any profile, metric, or horizon. "
            "Even at the mechanism-proximal level, Lamarckian warm-start does not "
            "clear the gate."
        )
    lines.append("")
    return "\n".join(lines)


# ── Driver ─────────────────────────────────────────────────────────────────────

def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with ``None`` for portable JSON.

    ``json.dumps`` emits bare ``NaN``/``Infinity`` tokens by default, which are
    invalid in strict JSON (and break JS / non-Python consumers). Mapping them
    to ``null`` keeps the file parseable everywhere.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure early-life offspring fitness across A/B inheritance arms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ab-dir", type=str, required=True,
                        help="Root dir containing the two arm sub-directories.")
    parser.add_argument("--baseline-arm", type=str, default="baldwinian")
    parser.add_argument("--treatment-arm", type=str, default="lamarckian")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Defaults to <ab-dir>/early_life.")
    parser.add_argument("--profiles", nargs="+", default=list(PROFILE_ORDER),
                        choices=list(PROFILE_ORDER), metavar="PROFILE")
    parser.add_argument("--ages", nargs="+", type=int, default=list(DEFAULT_AGES))
    args = parser.parse_args()

    ab_dir = Path(args.ab_dir)
    baseline_dir = ab_dir / args.baseline_arm
    treatment_dir = ab_dir / args.treatment_arm
    if not baseline_dir.is_dir() or not treatment_dir.is_dir():
        print(f"Expected arm dirs {baseline_dir} and {treatment_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else ab_dir / "early_life"
    output_dir.mkdir(parents=True, exist_ok=True)

    ages = sorted(set(int(a) for a in args.ages))

    def _load_arm(arm_dir: Path, label: str) -> Tuple[
        Dict[str, Dict[int, Dict[str, Any]]], Dict[str, List[float]]
    ]:
        runs = _discover_arm_runs(arm_dir, args.profiles)
        per_profile: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        warmstart: Dict[str, List[float]] = defaultdict(list)
        for profile, seed_dirs in runs.items():
            for seed, run_dir in seed_dirs:
                db = _find_db(run_dir)
                if db is None:
                    print(f"  [{label}] {run_dir}: no .db, skipping", file=sys.stderr)
                    continue
                warmup = _read_warmup(run_dir)
                data = _extract_run_early_life(db, warmup, ages)
                if data is None:
                    print(f"  [{label}] {run_dir}: no offspring, skipping",
                          file=sys.stderr)
                    continue
                per_profile[profile][seed] = data
                rate = _read_warmstart_rate(run_dir)
                if rate is not None and not math.isnan(rate):
                    warmstart[profile].append(rate)
                print(f"  [{label}] {profile} seed={seed}: "
                      f"{data['n_offspring']} offspring, warmup={warmup}")
        return dict(per_profile), dict(warmstart)

    print(f"Loading baseline arm ({args.baseline_arm})...")
    baseline, _ = _load_arm(baseline_dir, args.baseline_arm)
    print(f"Loading treatment arm ({args.treatment_arm})...")
    treatment, warmstart_raw = _load_arm(treatment_dir, args.treatment_arm)

    warmstart_rates = {p: _mean(v) for p, v in warmstart_raw.items() if v}

    paired: Dict[str, Dict[str, Any]] = {}
    for profile in args.profiles:
        b = baseline.get(profile, {})
        t = treatment.get(profile, {})
        if not b or not t:
            continue
        paired[profile] = _pair_profile(b, t, ages)

    if not paired:
        print("No paired profiles found; nothing to compare.", file=sys.stderr)
        return 1

    summary = {
        "baseline_arm": args.baseline_arm,
        "treatment_arm": args.treatment_arm,
        "ages": ages,
        "profiles_analyzed": [p for p in PROFILE_ORDER if p in paired],
        "warmstart_rates": warmstart_rates,
        "paired": paired,
        "rl_reward_curves": {
            "baseline": {
                p: {s: baseline[p][s]["rl_reward_curve"] for s in baseline.get(p, {})}
                for p in paired
            },
            "treatment": {
                p: {s: treatment[p][s]["rl_reward_curve"] for s in treatment.get(p, {})}
                for p in paired
            },
        },
        "survival_curves": {
            "baseline": {
                p: {s: baseline[p][s]["survival_curve"] for s in baseline.get(p, {})}
                for p in paired
            },
            "treatment": {
                p: {s: treatment[p][s]["survival_curve"] for s in treatment.get(p, {})}
                for p in paired
            },
        },
    }
    summary_path = output_dir / "early_life_summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, allow_nan=False, default=str),
        encoding="utf-8",
    )

    md = _build_markdown(paired, warmstart_rates, ages, args.baseline_arm,
                         args.treatment_arm)
    md_path = output_dir / "early_life_summary.md"
    md_path.write_text(md, encoding="utf-8")

    print(f"\nDone. Outputs in: {output_dir}")
    print(f"  summary JSON : {summary_path}")
    print(f"  summary MD   : {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

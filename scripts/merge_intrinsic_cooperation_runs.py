"""Merge intrinsic cooperation runs made in separate processes into one summary.

The runner tests its hypotheses across the replicates of *one* invocation.
When the registered seeds are run as separate processes (one seed per
machine, as in ``docs/research/experiments/cooperation/GcpRunbook.md``),
each writes its own ``intrinsic_cooperation_summary.json``. This script
concatenates their ``replicates`` blocks and recomputes the aggregate and
the Holm-corrected hypothesis family with the runner's own functions, so
the result is what a single ``--num-replicates N`` run would have produced.

Example::

    python scripts/merge_intrinsic_cooperation_runs.py \\
        results/seed_*/intrinsic_cooperation_summary.json \\
        --output-dir results/merged

Shards must share the run configuration (everything but ``seed``,
``num_replicates`` and ``output_dir``); pass ``--allow-config-mismatch`` to
merge anyway. Duplicate seeds are an error.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from farm.runners.intrinsic_cooperation_experiment import (
    aggregate_summaries,
    hypothesis_kind,
    test_hypotheses,
)

_KIND_LABELS = {"manipulation_check": " (check)", "learning_check": " (learning check)", "exploratory": " (exploratory)"}

#: Config keys that legitimately differ between shards of one run.
SHARD_SPECIFIC_CONFIG_KEYS: frozenset[str] = frozenset({"seed", "num_replicates"})

MERGED_SUMMARY_FILE = "intrinsic_cooperation_summary.json"
HYPOTHESES_TABLE_FILE = "hypotheses.md"


def _comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in config.items() if k not in SHARD_SPECIFIC_CONFIG_KEYS}
    cooperation = dict(out.get("cooperation") or {})
    cooperation.pop("output_dir", None)
    cooperation.pop("seed", None)
    out["cooperation"] = cooperation
    return out


def load_shards(paths: list[str]) -> list[dict[str, Any]]:
    shards = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        for key in ("config", "replicates"):
            if key not in payload:
                raise ValueError(f"{path}: not an intrinsic cooperation summary (missing {key!r})")
        payload["_path"] = path
        shards.append(payload)
    if not shards:
        raise ValueError("no summary files given")
    return shards


def merge_shards(shards: list[dict[str, Any]], *, allow_config_mismatch: bool = False) -> dict[str, Any]:
    """One summary payload from several, with the aggregate and tests recomputed."""
    reference = _comparable_config(shards[0]["config"])
    for shard in shards[1:]:
        if _comparable_config(shard["config"]) != reference:
            message = f"{shard['_path']}: run configuration differs from {shards[0]['_path']}"
            if not allow_config_mismatch:
                raise ValueError(message + " (pass --allow-config-mismatch to merge anyway)")
            print("warning:", message, file=sys.stderr)

    replicates = sorted((r for shard in shards for r in shard["replicates"]), key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in replicates]
    duplicates = sorted({s for s in seeds if seeds.count(s) > 1})
    if duplicates:
        raise ValueError(f"duplicate seeds across shards: {duplicates}")
    for index, replicate in enumerate(replicates):
        replicate["index"] = index

    arms = list(shards[0]["config"]["arms"])
    alpha = float(shards[0]["config"].get("alpha", 0.05))
    config = {**shards[0]["config"], "seed": seeds[0], "num_replicates": len(replicates), "seeds": seeds}
    many = len(replicates) > 1
    return {
        "config": config,
        "sources": [shard["_path"] for shard in shards],
        "replicates": replicates,
        "aggregate": aggregate_summaries(replicates, arms) if many else None,
        "hypotheses": test_hypotheses(replicates, arms, alpha) if many else None,
    }


def hypotheses_table(hypotheses: list[dict[str, Any]] | None) -> str:
    if not hypotheses:
        return "No hypothesis tests (fewer than two replicates).\n"
    lines = [
        "| Key | Metric | Contrast | n | Mean diff [95% CI] | Seeds > 0 | p | p (Holm) | Significant |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for h in hypotheses:
        p_holm = "—" if h.get("p_holm") is None else f"{h['p_holm']:.4f}"
        key = h["key"] + _KIND_LABELS.get(hypothesis_kind(h), "")
        lines.append(
            f"| {key} | `{h['metric']}` | {h['contrast']} | {h['n']} | "
            f"{h['mean_diff']:+.4g} [{h['ci_low']:.4g}, {h['ci_high']:.4g}] | {h['positive_seeds']} | "
            f"{h['p_value']:.4f} | {p_holm} | {'yes' if h.get('significant') else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def _expand(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", nargs="+", help="Per-shard intrinsic_cooperation_summary.json files (globs ok).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-config-mismatch", action="store_true")
    args = parser.parse_args(argv)

    merged = merge_shards(load_shards(_expand(args.summaries)), allow_config_mismatch=args.allow_config_mismatch)
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, MERGED_SUMMARY_FILE)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2, default=str)
    table = hypotheses_table(merged["hypotheses"])
    with open(os.path.join(args.output_dir, HYPOTHESES_TABLE_FILE), "w", encoding="utf-8") as handle:
        handle.write(table)
    print(f"merged {len(merged['replicates'])} replicates (seeds {merged['config']['seeds']}) -> {summary_path}")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Union-emergence Layer C experiment: exclusive bonds on the AgentFarm chromosome.

Four arms (solo / promiscuous / optional / forced) crossed with the
pre-registered baseline cell and one-at-a-time ablations of courtship,
exit tax, and social range. Learning genes are frozen; union, share, and
goal loci keep evolving under implicit selection.
"""

from __future__ import annotations

import json
import os
import random
import statistics
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from farm.config import SimulationConfig
from farm.core.hyperparameter_chromosome import (
    BoundaryMode,
    MutationMode,
)
from farm.core.initial_diversity import InitialDiversityConfig, SeedingMode
from farm.core.simulation import run_simulation
from farm.core.union_bonds import (
    UNION_GENE_NAMES,
    UnionPolicy,
    freeze_learning_genes,
    policy_for_arm,
    snapshot_union_metrics,
    tick_union_bonds,
)
from farm.runners.intrinsic_evolution_experiment import IntrinsicEvolutionPolicy
from farm.utils.logging import get_logger

logger = get_logger(__name__)

ARM_NAMES: Tuple[str, ...] = (
    "solo_only",
    "promiscuous",
    "optional_union",
    "forced_union",
)
TRACKED_ACTIONS: Tuple[str, ...] = (
    "move",
    "gather",
    "share",
    "bond",
    "leave",
    "reproduce",
    "attack",
    "pass",
)
TRACKED_GENES: Tuple[str, ...] = UNION_GENE_NAMES + ("share_weight", "reward_share_bonus")


@dataclass(frozen=True)
class UnionCell:
    """One world cell of the Layer C grid."""

    name: str
    courtship_steps: int = 12
    exit_tax: float = 1.1
    social_range: float = 3.2
    bonding_cost: float = 0.8


def default_cells() -> Tuple[UnionCell, ...]:
    baseline = UnionCell(name="baseline")
    return (
        baseline,
        replace(baseline, name="no_courtship", courtship_steps=0),
        replace(baseline, name="cheap_exit", exit_tax=0.2),
        replace(baseline, name="costly_exit", exit_tax=2.2),
        replace(baseline, name="tight_range", social_range=2.0),
        replace(baseline, name="wide_range", social_range=8.0),
    )


@dataclass
class UnionEmergenceExperimentConfig:
    """Configuration for :class:`UnionEmergenceExperiment`."""

    num_steps: int = 150
    seed: int = 42
    num_replicates: int = 2
    output_dir: str = "experiments/union_emergence/layer_c"
    record_interval: int = 5
    selection_pressure: Any = "low"
    mutation_rate: float = 0.1
    mutation_scale: float = 0.1
    mutation_mode: MutationMode = MutationMode.GAUSSIAN
    boundary_mode: BoundaryMode = BoundaryMode.REFLECT
    courtship_steps: int = 12
    exit_tax: float = 1.1
    social_range: float = 3.2
    bonding_cost: float = 0.8
    max_population: int = 80
    world_size: int = 24
    initial_agents: int = 24
    initial_resources: int = 40
    initial_agent_resource_level: float = 12.0
    arms: Tuple[str, ...] = ARM_NAMES
    cells: Tuple[UnionCell, ...] = field(default_factory=default_cells)
    in_memory_db: bool = True

    def __post_init__(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if self.num_replicates <= 0:
            raise ValueError("num_replicates must be positive.")
        unknown = [arm for arm in self.arms if arm not in ARM_NAMES]
        if unknown:
            raise ValueError(f"unknown arms: {unknown}")


@dataclass
class CellArmResult:
    """Telemetry for one (cell, arm, seed) run."""

    cell: str
    arm: str
    seed: int
    steps: List[int] = field(default_factory=list)
    paired_frac: List[float] = field(default_factory=list)
    synergy: List[Optional[float]] = field(default_factory=list)
    mean_energy: List[float] = field(default_factory=list)
    extraction: List[Optional[float]] = field(default_factory=list)
    leave_events: List[int] = field(default_factory=list)
    gene_means: Dict[str, List[float]] = field(default_factory=dict)
    action_mix: Dict[str, List[float]] = field(default_factory=dict)
    final_population: int = 0
    start_genes: Dict[str, float] = field(default_factory=dict)
    end_genes: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
            present = [value for value in values if value is not None]
            return statistics.fmean(present) if present else None

        return {
            "cell": self.cell,
            "arm": self.arm,
            "seed": self.seed,
            "final_population": self.final_population,
            "synergy_index": _mean(self.synergy[-max(1, len(self.synergy) // 4) :]),
            "paired_frac": _mean(self.paired_frac[-max(1, len(self.paired_frac) // 4) :]),
            "mean_energy": _mean(self.mean_energy[-max(1, len(self.mean_energy) // 4) :]),
            "mean_extraction": _mean(self.extraction[-max(1, len(self.extraction) // 4) :]),
            "leave_rate": (
                (self.leave_events[-1] - self.leave_events[0]) / max(1, self.steps[-1] - self.steps[0])
                if len(self.leave_events) >= 2
                else None
            ),
            "delta_pair_commitment": self.end_genes.get("pair_commitment", 0.0)
            - self.start_genes.get("pair_commitment", 0.0),
            "delta_fidelity": self.end_genes.get("fidelity", 0.0) - self.start_genes.get("fidelity", 0.0),
            "delta_specialize": self.end_genes.get("specialize", 0.0) - self.start_genes.get("specialize", 0.0),
            "end_genes": self.end_genes,
            "start_genes": self.start_genes,
        }


class UnionEmergenceExperiment:
    """Run the Layer C union-emergence grid and write artifacts."""

    def __init__(
        self,
        base_config: SimulationConfig,
        config: Optional[UnionEmergenceExperimentConfig] = None,
    ) -> None:
        self.base_config = base_config
        self.config = config or UnionEmergenceExperimentConfig()

    def run(self) -> Dict[str, Any]:
        os.makedirs(self.config.output_dir, exist_ok=True)
        runs: List[CellArmResult] = []
        for cell in self.config.cells:
            for arm in self.config.arms:
                for offset in range(self.config.num_replicates):
                    seed = self.config.seed + offset
                    logger.info(
                        "union_emergence_run_start",
                        cell=cell.name,
                        arm=arm,
                        seed=seed,
                    )
                    runs.append(self._run_cell_arm(cell, arm, seed))

        cells = _aggregate(runs)
        payload = {
            "config": {
                "num_steps": self.config.num_steps,
                "seed": self.config.seed,
                "num_replicates": self.config.num_replicates,
                "arms": list(self.config.arms),
                "cells": [cell.name for cell in self.config.cells],
                "courtship_steps": self.config.courtship_steps,
                "exit_tax": self.config.exit_tax,
                "social_range": self.config.social_range,
                "bonding_cost": self.config.bonding_cost,
                "selection_pressure": self.config.selection_pressure,
            },
            "cells": cells,
            "runs": [run.summary() for run in runs],
            "win_conditions": _evaluate_win_conditions(cells),
        }
        summary_path = os.path.join(self.config.output_dir, "union_emergence_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        payload["summary_path"] = summary_path
        payload["figure_paths"] = _write_figures(self.config.output_dir, payload, runs)
        _write_markdown(self.config.output_dir, payload)
        logger.info("union_emergence_experiment_complete", summary_path=summary_path)
        return payload

    def _build_run_config(self) -> SimulationConfig:
        run_config = self.base_config.copy()
        run_config.initial_diversity = InitialDiversityConfig(mode=SeedingMode.NONE)
        run_config.environment.width = int(self.config.world_size)
        run_config.environment.height = int(self.config.world_size)
        run_config.population.max_population = int(self.config.max_population)
        run_config.population.system_agents = int(self.config.initial_agents)
        run_config.population.independent_agents = 0
        run_config.population.control_agents = 0
        run_config.resources.initial_resources = int(self.config.initial_resources)
        run_config.agent_behavior.initial_resource_level = float(
            self.config.initial_agent_resource_level
        )
        run_config.union_enabled = True
        run_config.agent_behavior.base_consumption_rate = 0.08
        run_config.agent_behavior.max_movement = 2
        run_config.agent_behavior.gathering_range = 5
        run_config.resources.resource_regen_rate = 0.20
        run_config.resources.resource_regen_amount = 3
        if self.config.in_memory_db:
            database = getattr(run_config, "database", None)
            if database is not None and hasattr(database, "use_in_memory_db"):
                database.use_in_memory_db = True
                database.persist_db_on_completion = False
        return run_config

    def _run_cell_arm(self, cell: UnionCell, arm: str, seed: int) -> CellArmResult:
        run_config = self._build_run_config()
        evolution = IntrinsicEvolutionPolicy(
            enabled=True,
            mutation_rate=self.config.mutation_rate,
            mutation_scale=self.config.mutation_scale,
            mutation_mode=self.config.mutation_mode,
            boundary_mode=self.config.boundary_mode,
            crossover_enabled=True,
            selection_pressure=self.config.selection_pressure,
            seed=seed,
        )
        union_policy = policy_for_arm(
            arm,
            UnionPolicy(
                enabled=True,
                bonding_cost=cell.bonding_cost,
                courtship_steps=cell.courtship_steps,
                exit_tax=cell.exit_tax,
                social_range=cell.social_range,
            ),
        )
        result = CellArmResult(cell=cell.name, arm=arm, seed=seed)
        for gene in TRACKED_GENES:
            result.gene_means[gene] = []
        for action in TRACKED_ACTIONS:
            result.action_mix[action] = []

        def _on_ready(environment: Any) -> None:
            environment.union_policy = union_policy
            environment.union_rng = random.Random(seed + 17)
            environment.intrinsic_evolution_policy = evolution
            environment.intrinsic_evolution_rng = random.Random(seed)
            for agent in environment.alive_agent_objects:
                chromosome = getattr(agent, "hyperparameter_chromosome", None)
                if chromosome is not None:
                    agent.hyperparameter_chromosome = freeze_learning_genes(chromosome)
            tick_union_bonds(environment)
            snap = snapshot_union_metrics(environment)
            result.start_genes = dict(snap["gene_means"])

        def _on_step_end(environment: Any, step: int) -> None:
            if step % self.config.record_interval != 0:
                return
            snap = snapshot_union_metrics(environment)
            result.steps.append(step)
            result.paired_frac.append(float(snap["paired_frac"]))
            result.synergy.append(snap["synergy_index"])
            result.mean_energy.append(float(snap["mean_energy"]))
            result.extraction.append(snap["mean_extraction"])
            result.leave_events.append(int(snap["leave_events"]))
            for gene, value in snap["gene_means"].items():
                if gene in result.gene_means:
                    result.gene_means[gene].append(float(value))
            alive = list(environment.alive_agent_objects)
            n = max(1, len(alive))
            for action in TRACKED_ACTIONS:
                count = sum(1 for agent in alive if getattr(agent, "last_action_name", None) == action)
                result.action_mix[action].append(count / n)
            result.final_population = len(alive)
            result.end_genes = dict(snap["gene_means"])

        run_dir = os.path.join(self.config.output_dir, f"{cell.name}_{arm}_s{seed}")
        os.makedirs(run_dir, exist_ok=True)
        run_simulation(
            num_steps=self.config.num_steps,
            config=run_config,
            path=None if self.config.in_memory_db else run_dir,
            save_config=False,
            seed=seed,
            disable_console_logging=True,
            on_environment_ready=_on_ready,
            on_step_end=_on_step_end,
        )
        return result


def _aggregate(runs: Sequence[CellArmResult]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[CellArmResult]] = {}
    for run in runs:
        grouped.setdefault((run.cell, run.arm), []).append(run)
    cells: List[Dict[str, Any]] = []
    for (cell, arm), group in grouped.items():
        summaries = [run.summary() for run in group]
        keys = (
            "synergy_index",
            "paired_frac",
            "mean_energy",
            "mean_extraction",
            "leave_rate",
            "delta_pair_commitment",
            "delta_fidelity",
            "delta_specialize",
            "final_population",
        )
        row: Dict[str, Any] = {
            "cell": cell,
            "arm": arm,
            "n_seeds": len(group),
            "seeds": [run.seed for run in group],
        }
        for key in keys:
            values = [summary.get(key) for summary in summaries]
            present = [value for value in values if isinstance(value, (int, float))]
            row[key] = statistics.fmean(present) if present else None
            row[f"{key}_std"] = statistics.pstdev(present) if len(present) > 1 else (0.0 if present else None)
        cells.append(row)
    return cells


def _evaluate_win_conditions(cells: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_key = {(row["cell"], row["arm"]): row for row in cells}

    def get(cell: str, arm: str, metric: str) -> Optional[float]:
        row = by_key.get((cell, arm))
        if row is None:
            return None
        value = row.get(metric)
        return float(value) if isinstance(value, (int, float)) else None

    optional_syn = get("baseline", "optional_union", "synergy_index")
    forced_syn = get("baseline", "forced_union", "synergy_index")
    cheap_syn = get("cheap_exit", "optional_union", "synergy_index")
    noco_syn = get("no_courtship", "optional_union", "synergy_index")
    opt_energy = get("baseline", "optional_union", "mean_energy")
    promiscuous_energy = get("baseline", "promiscuous", "mean_energy")
    opt_extract = get("baseline", "optional_union", "mean_extraction")
    forced_extract = get("baseline", "forced_union", "mean_extraction")
    opt_commit = get("baseline", "optional_union", "delta_pair_commitment")
    opt_fid = get("baseline", "optional_union", "delta_fidelity")
    checks = {
        "baseline_optional_synergy_gt_1": optional_syn is not None and optional_syn > 1.0,
        "baseline_forced_synergy_gt_1": forced_syn is not None and forced_syn > 1.0,
        "optional_energy_beats_promiscuous": (
            opt_energy is not None
            and promiscuous_energy is not None
            and opt_energy > promiscuous_energy
        ),
        "optional_commitment_does_not_climb": opt_commit is not None and opt_commit <= 0.03,
        "optional_fidelity_rises_on_baseline": opt_fid is not None and opt_fid > 0.0,
        "cheap_exit_drops_optional_synergy": (
            optional_syn is not None and cheap_syn is not None and cheap_syn <= optional_syn + 0.02
        ),
        "no_courtship_inflates_optional_synergy": (
            optional_syn is not None and noco_syn is not None and noco_syn >= optional_syn
        ),
        "forced_extraction_gt_optional": (
            opt_extract is not None and forced_extract is not None and forced_extract > opt_extract
        ),
    }
    return {
        "checks": checks,
        "passed": sum(1 for ok in checks.values() if ok),
        "total": len(checks),
        "headline": {
            "baseline_optional_synergy": optional_syn,
            "baseline_forced_synergy": forced_syn,
            "baseline_optional_energy": opt_energy,
            "baseline_promiscuous_energy": promiscuous_energy,
            "baseline_optional_delta_commitment": opt_commit,
            "baseline_optional_delta_fidelity": opt_fid,
        },
    }


def _fmt_metric(row: Dict[str, Any], key: str, digits: int = 3) -> str:
    value = row.get(key)
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _write_markdown(output_dir: str, payload: Dict[str, Any]) -> str:
    lines = [
        "# Union emergence Layer C",
        "",
        "| cell | arm | synergy | energy | paired | Δ commit | Δ fidelity | extraction | pop |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["cells"]:
        lines.append(
            f"| {row['cell']} | {row['arm']} | {_fmt_metric(row, 'synergy_index')} | "
            f"{_fmt_metric(row, 'mean_energy', 1)} | {_fmt_metric(row, 'paired_frac')} | "
            f"{_fmt_metric(row, 'delta_pair_commitment')} | {_fmt_metric(row, 'delta_fidelity')} | "
            f"{_fmt_metric(row, 'mean_extraction')} | {_fmt_metric(row, 'final_population', 1)} |"
        )
    wins = payload.get("win_conditions", {})
    lines.append("")
    lines.append(f"Win-condition checks: {wins.get('passed', 0)}/{wins.get('total', 0)}.")
    for name, ok in wins.get("checks", {}).items():
        lines.append(f"- {'PASS' if ok else 'FAIL'}: {name}")
    path = os.path.join(output_dir, "LAYER_C.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def _write_figures(
    output_dir: str,
    payload: Dict[str, Any],
    runs: Sequence[CellArmResult],
) -> Dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    paths: Dict[str, str] = {}
    baseline = [row for row in payload["cells"] if row["cell"] == "baseline"]
    if baseline:
        order = [arm for arm in ARM_NAMES if any(row["arm"] == arm for row in baseline)]
        by_arm = {row["arm"]: row for row in baseline}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
        energies = [by_arm[arm].get("mean_energy") or 0.0 for arm in order]
        axes[0].bar(
            [arm.replace("_", "\n") for arm in order],
            energies,
            color=["#6b7280", "#9ca3af", "#2563eb", "#dc2626"],
        )
        axes[0].set_title("Baseline — population energy")
        axes[0].set_ylabel("Mean energy")
        syn_arms = [arm for arm in ("optional_union", "forced_union") if arm in by_arm]
        axes[1].bar(
            ["optional", "forced"][: len(syn_arms)],
            [by_arm[arm].get("synergy_index") or 0.0 for arm in syn_arms],
            color=["#2563eb", "#dc2626"],
        )
        axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
        axes[1].set_title("Baseline — synergy")
        axes[1].set_ylabel("Paired / unpaired energy")
        fig.tight_layout()
        path = os.path.join(output_dir, "synergy_bar.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths["synergy_bar"] = path

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for run in runs:
        if run.cell != "baseline" or not run.steps:
            continue
        ax.plot(run.steps, run.paired_frac, label=f"{run.arm} s{run.seed}", linewidth=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("paired fraction")
    ax.set_title("Baseline paired-frac trajectory")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = os.path.join(output_dir, "paired_frac_trajectory.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths["paired_frac_trajectory"] = path

    fig, ax = plt.subplots(figsize=(8, 4.2))
    plotted = False
    for run in runs:
        if run.cell != "baseline" or run.arm not in ("optional_union", "forced_union"):
            continue
        if "pair_commitment" in run.gene_means:
            ax.plot(run.steps, run.gene_means["pair_commitment"], label=f"{run.arm} commit s{run.seed}", linestyle="--")
            plotted = True
        if "fidelity" in run.gene_means:
            ax.plot(run.steps, run.gene_means["fidelity"], label=f"{run.arm} fidelity s{run.seed}")
            plotted = True
    if plotted:
        ax.set_xlabel("step")
        ax.set_ylabel("gene mean")
        ax.set_title("Baseline gene drift (commitment vs fidelity)")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        path = os.path.join(output_dir, "gene_drift.png")
        fig.savefig(path, dpi=140)
        paths["gene_drift"] = path
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    xs: List[float] = []
    ys: List[float] = []
    labels: List[str] = []
    for row in payload["cells"]:
        leave = row.get("leave_rate")
        extract = row.get("mean_extraction")
        if leave is None or extract is None:
            continue
        xs.append(float(leave))
        ys.append(float(extract))
        labels.append(f"{row['cell'][:4]}:{row['arm'][:3]}")
    if xs:
        ax.scatter(xs, ys)
        for x_val, y_val, label in zip(xs, ys, labels):
            ax.annotate(label, (x_val, y_val), fontsize=7)
        ax.set_xlabel("leave rate")
        ax.set_ylabel("extraction")
        ax.set_title("Extraction vs leave-rate")
        fig.tight_layout()
        path = os.path.join(output_dir, "extraction_vs_leave.png")
        fig.savefig(path, dpi=140)
        paths["extraction_vs_leave"] = path
    plt.close(fig)
    return paths

"""Intrinsic evolution adapter for the dashboard experiment interface."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from farm.config import SimulationConfig
from farm.experiments.interfaces import ExperimentAdapterProtocol, ViewDescriptor
from farm.experiments.manifest import (
    ExperimentManifest,
    INTRINSIC_EVOLUTION_EXPERIMENT_TYPE,
    ManifestValidationResult,
)
from farm.runners.intrinsic_evolution_experiment import (
    InitialConditionsConfig,
    IntrinsicEvolutionExperiment,
    IntrinsicEvolutionExperimentConfig,
    IntrinsicEvolutionPolicy,
    SpeciationConfig,
)


def _read_json(path: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not os.path.exists(path):
        return default or {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def _to_series_payload(
    x_axis: List[int],
    series: Iterable[Dict[str, Any]],
    title: str,
) -> Dict[str, Any]:
    return {
        "view_type": "timeseries",
        "title": title,
        "x": x_axis,
        "series": list(series),
    }


class IntrinsicEvolutionAdapter(ExperimentAdapterProtocol):
    """Adapter between intrinsic-evolution artifacts and dashboard payloads."""

    experiment_type = INTRINSIC_EVOLUTION_EXPERIMENT_TYPE

    def validate_manifest(self, manifest: ExperimentManifest) -> ManifestValidationResult:
        try:
            self._build_config(manifest, output_dir="results/validation_only")
        except Exception as exc:  # noqa: BLE001
            return ManifestValidationResult(is_valid=False, errors=[str(exc)])
        return ManifestValidationResult(is_valid=True, normalized_manifest=manifest.dict())

    def run_experiment(
        self,
        run_id: str,
        manifest: ExperimentManifest,
        runtime_options: Dict[str, Any],
    ) -> Dict[str, Any]:
        output_dir = runtime_options.get("output_dir") or os.path.join(
            "results", "intrinsic_dashboard_runs", run_id
        )
        os.makedirs(output_dir, exist_ok=True)
        base_config, experiment_config = self._build_config(manifest, output_dir=output_dir)

        experiment = IntrinsicEvolutionExperiment(
            base_config=base_config,
            config=experiment_config,
        )
        result = experiment.run()
        return {
            "run_id": run_id,
            "experiment_type": self.experiment_type,
            "output_dir": output_dir,
            "num_steps_completed": result.num_steps_completed,
            "final_population": result.final_population,
            "startup_transient_metrics": result.startup_transient_metrics,
        }

    def list_views(self, run_context: Dict[str, Any]) -> List[ViewDescriptor]:
        output_dir = run_context["output_dir"]
        metadata = _read_json(os.path.join(output_dir, "intrinsic_evolution_metadata.json"))
        views = [
            ViewDescriptor(
                view_id="summary_cards",
                view_type="summary_cards",
                title="Run Summary",
                description="Resolved policy, completion stats, and final outcomes.",
            ),
            ViewDescriptor(
                view_id="gene_trajectories",
                view_type="timeseries",
                title="Gene Trajectories",
                description="Per-gene mean and variability over simulation steps.",
            ),
            ViewDescriptor(
                view_id="population_dynamics",
                view_type="timeseries",
                title="Population Dynamics",
                description="Population plus realized birth/death and selection pressure.",
            ),
            ViewDescriptor(
                view_id="gene_distributions",
                view_type="distribution_over_time",
                title="Gene Distributions",
                description="Snapshot-based per-gene distribution values.",
            ),
        ]
        if metadata.get("speciation", {}).get("enabled"):
            views.append(
                ViewDescriptor(
                    view_id="speciation_index",
                    view_type="lineage_or_clusters",
                    title="Speciation Index",
                    description="Cluster separation and lineage-oriented speciation tracking.",
                )
            )
        return views

    def get_view_data(
        self,
        run_context: Dict[str, Any],
        view_id: str,
        filters: Dict[str, Any],
    ) -> Dict[str, Any]:
        output_dir = run_context["output_dir"]
        trajectory = _read_jsonl(os.path.join(output_dir, "intrinsic_gene_trajectory.jsonl"))
        snapshots = _read_jsonl(os.path.join(output_dir, "intrinsic_gene_snapshots.jsonl"))
        metadata = _read_json(os.path.join(output_dir, "intrinsic_evolution_metadata.json"))
        clusters = _read_jsonl(os.path.join(output_dir, "cluster_lineage.jsonl"))

        if view_id == "summary_cards":
            return self._summary_cards(metadata, trajectory)
        if view_id == "gene_trajectories":
            return self._gene_trajectories(trajectory, filters)
        if view_id == "population_dynamics":
            return self._population_dynamics(trajectory)
        if view_id == "gene_distributions":
            return self._gene_distributions(snapshots, filters)
        if view_id == "speciation_index":
            return self._speciation_index(trajectory, clusters)
        raise KeyError(f"Unknown view_id={view_id!r}")

    def _build_config(
        self,
        manifest: ExperimentManifest,
        *,
        output_dir: str,
    ) -> tuple[SimulationConfig, IntrinsicEvolutionExperimentConfig]:
        base_config = SimulationConfig.from_dict(manifest.base_simulation_config)
        config_payload = dict(manifest.experiment_config)

        policy_payload = config_payload.pop("policy", {})
        initial_conditions_payload = config_payload.pop("initial_conditions", {})
        speciation_payload = config_payload.pop("speciation", {})
        config_payload["output_dir"] = config_payload.get("output_dir") or output_dir

        experiment_config = IntrinsicEvolutionExperimentConfig(
            policy=IntrinsicEvolutionPolicy(**policy_payload),
            initial_conditions=InitialConditionsConfig(**initial_conditions_payload),
            speciation=SpeciationConfig(**speciation_payload),
            **config_payload,
        )
        return base_config, experiment_config

    def _summary_cards(
        self,
        metadata: Dict[str, Any],
        trajectory_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cards = [
            {"id": "steps", "label": "Steps Completed", "value": metadata.get("num_steps_completed")},
            {"id": "population", "label": "Final Population", "value": metadata.get("final_population")},
            {
                "id": "snapshot_interval",
                "label": "Snapshot Interval",
                "value": metadata.get("snapshot_interval"),
            },
            {
                "id": "selection_pressure",
                "label": "Selection Pressure",
                "value": metadata.get("policy", {}).get("selection_pressure"),
            },
            {
                "id": "speciation_enabled",
                "label": "Speciation Enabled",
                "value": metadata.get("speciation", {}).get("enabled", False),
            },
        ]
        if trajectory_rows:
            cards.append(
                {
                    "id": "samples",
                    "label": "Trajectory Samples",
                    "value": len(trajectory_rows),
                }
            )
        return {"view_type": "summary_cards", "cards": cards, "metadata": metadata}

    def _gene_trajectories(
        self,
        trajectory_rows: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> Dict[str, Any]:
        selected_genes = set(filters.get("genes", []))
        x_axis = [int(row.get("step", index + 1)) for index, row in enumerate(trajectory_rows)]
        by_gene: Dict[str, Dict[str, List[float]]] = {}

        for row in trajectory_rows:
            stats = row.get("gene_stats", {})
            for gene_name, gene_stats in stats.items():
                if selected_genes and gene_name not in selected_genes:
                    continue
                slot = by_gene.setdefault(
                    gene_name,
                    {"mean": [], "std": [], "min": [], "max": []},
                )
                slot["mean"].append(float(gene_stats.get("mean", 0.0)))
                slot["std"].append(float(gene_stats.get("std", 0.0)))
                slot["min"].append(float(gene_stats.get("min", 0.0)))
                slot["max"].append(float(gene_stats.get("max", 0.0)))

        series: List[Dict[str, Any]] = []
        for gene_name, values in sorted(by_gene.items()):
            series.append(
                {
                    "id": f"{gene_name}_mean",
                    "label": f"{gene_name} mean",
                    "values": values["mean"],
                    "bands": {
                        "std": values["std"],
                        "min": values["min"],
                        "max": values["max"],
                    },
                }
            )
        return _to_series_payload(x_axis, series, "Gene trajectories")

    def _population_dynamics(self, trajectory_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        x_axis = [int(row.get("step", index + 1)) for index, row in enumerate(trajectory_rows)]
        population = [
            int(row.get("n_alive", row.get("population", row.get("n_with_chromosome", 0))))
            for row in trajectory_rows
        ]
        births = [float(row.get("realized_birth_rate", 0.0)) for row in trajectory_rows]
        deaths = [float(row.get("realized_death_rate", 0.0)) for row in trajectory_rows]
        selection = [
            float(row.get("effective_selection_strength", 0.0)) for row in trajectory_rows
        ]
        series = [
            {"id": "population", "label": "Population", "values": population},
            {"id": "birth_rate", "label": "Birth Rate", "values": births},
            {"id": "death_rate", "label": "Death Rate", "values": deaths},
            {
                "id": "selection_strength",
                "label": "Selection Strength",
                "values": selection,
            },
        ]
        return _to_series_payload(x_axis, series, "Population dynamics")

    def _gene_distributions(
        self,
        snapshot_rows: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> Dict[str, Any]:
        max_snapshots = int(filters.get("max_snapshots", 25))
        reduced_rows = snapshot_rows[-max_snapshots:] if max_snapshots > 0 else snapshot_rows
        payload: List[Dict[str, Any]] = []
        for row in reduced_rows:
            step = int(row.get("step", 0))
            by_gene: Dict[str, List[float]] = {}
            for agent in row.get("agents", []):
                chromosome = agent.get("chromosome", {})
                for gene_name, value in chromosome.items():
                    by_gene.setdefault(gene_name, []).append(float(value))
            payload.append({"step": step, "by_gene": by_gene})
        return {
            "view_type": "distribution_over_time",
            "title": "Gene distribution history",
            "snapshots": payload,
        }

    def _speciation_index(
        self,
        trajectory_rows: List[Dict[str, Any]],
        cluster_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        x_axis = [int(row.get("step", index + 1)) for index, row in enumerate(trajectory_rows)]
        speciation_values = [float(row.get("speciation_index", 0.0)) for row in trajectory_rows]
        cluster_sizes = [
            {
                "step": int(row.get("step", 0)),
                "cluster_id": row.get("cluster_id"),
                "size": int(row.get("size", 0)),
            }
            for row in cluster_rows
        ]
        return {
            "view_type": "lineage_or_clusters",
            "title": "Speciation index",
            "timeseries": _to_series_payload(
                x_axis,
                [{"id": "speciation_index", "label": "Speciation Index", "values": speciation_values}],
                "Speciation index",
            ),
            "clusters": cluster_sizes,
        }

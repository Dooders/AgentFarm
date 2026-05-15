import json

from farm.experiments.intrinsic_dashboard import IntrinsicEvolutionAdapter


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_adapter_views_and_payloads(tmp_path):
    output_dir = tmp_path / "intrinsic-run"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "num_steps_completed": 2,
        "final_population": 7,
        "snapshot_interval": 1,
        "policy": {"selection_pressure": "medium"},
        "speciation": {"enabled": True},
    }
    with open(output_dir / "intrinsic_evolution_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)

    _write_jsonl(
        output_dir / "intrinsic_gene_trajectory.jsonl",
        [
            {
                "step": 1,
                "n_alive": 6,
                "gene_stats": {"learning_rate": {"mean": 0.1, "std": 0.01, "min": 0.08, "max": 0.12}},
                "realized_birth_rate": 0.2,
                "realized_death_rate": 0.1,
                "effective_selection_strength": 0.05,
                "speciation_index": 0.1,
            },
            {
                "step": 2,
                "n_alive": 7,
                "gene_stats": {"learning_rate": {"mean": 0.11, "std": 0.02, "min": 0.07, "max": 0.13}},
                "realized_birth_rate": 0.3,
                "realized_death_rate": 0.05,
                "effective_selection_strength": 0.06,
                "speciation_index": 0.12,
            },
        ],
    )

    _write_jsonl(
        output_dir / "intrinsic_gene_snapshots.jsonl",
        [
            {"step": 1, "agents": [{"chromosome": {"learning_rate": 0.1}}]},
            {"step": 2, "agents": [{"chromosome": {"learning_rate": 0.12}}]},
        ],
    )
    _write_jsonl(
        output_dir / "cluster_lineage.jsonl",
        [{"step": 2, "cluster_id": 1, "size": 3}],
    )

    adapter = IntrinsicEvolutionAdapter()
    run_context = {"output_dir": str(output_dir)}
    views = adapter.list_views(run_context)
    view_ids = [item.view_id for item in views]
    assert "summary_cards" in view_ids
    assert "gene_trajectories" in view_ids
    assert "speciation_index" in view_ids

    summary = adapter.get_view_data(run_context, "summary_cards", {})
    assert summary["view_type"] == "summary_cards"
    assert summary["cards"]

    trajectories = adapter.get_view_data(
        run_context,
        "gene_trajectories",
        {"genes": ["learning_rate"]},
    )
    assert trajectories["view_type"] == "timeseries"
    assert trajectories["series"][0]["id"] == "learning_rate_mean"

    distributions = adapter.get_view_data(run_context, "gene_distributions", {})
    assert distributions["view_type"] == "distribution_over_time"
    assert len(distributions["snapshots"]) == 2

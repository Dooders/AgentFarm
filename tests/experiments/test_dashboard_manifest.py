from farm.experiments.manifest import (
    INTRINSIC_EVOLUTION_EXPERIMENT_TYPE,
    SUPPORTED_SCHEMA_VERSION,
    validate_experiment_manifest,
)


def test_validate_manifest_accepts_intrinsic_payload():
    payload = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "experiment_type": INTRINSIC_EVOLUTION_EXPERIMENT_TYPE,
        "experiment_name": "intrinsic-smoke",
        "base_simulation_config": {"environment.width": 100},
        "experiment_config": {"num_steps": 10, "snapshot_interval": 2},
        "dashboard_preset": {"default_views": ["summary_cards"]},
    }

    result = validate_experiment_manifest(payload)
    assert result.is_valid is True
    assert result.normalized_manifest is not None
    assert result.normalized_manifest["experiment_type"] == INTRINSIC_EVOLUTION_EXPERIMENT_TYPE


def test_validate_manifest_rejects_unsupported_type():
    payload = {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "experiment_type": "unsupported_experiment",
        "experiment_name": "bad-run",
        "base_simulation_config": {},
        "experiment_config": {},
    }

    result = validate_experiment_manifest(payload)
    assert result.is_valid is False
    assert result.errors

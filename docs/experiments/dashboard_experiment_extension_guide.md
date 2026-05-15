# Dashboard Experiment Extension Guide

This guide explains how to add a new dashboard-backed experiment type
without redesigning the API or editor flow.

Use `intrinsic_evolution` as the reference implementation:

- Adapter: `farm/experiments/intrinsic_dashboard.py`
- Contract: `farm/experiments/interfaces.py`
- Manifest schema: `farm/experiments/manifest.py`
- API wiring: `farm/api/server.py`
- Editor consumer: `farm/editor/components/sidebar.js` and
  `farm/editor/components/experiment-dashboard.js`

## What Exists Today

The current dashboard experiment pipeline is:

1. Editor builds a manifest and posts it to
   `/api/experiments/manifests/validate`.
2. API validates/normalizes via `validate_experiment_manifest()`.
3. API resolves an adapter from `ExperimentRegistry`.
4. API runs the adapter in background via `/api/experiments/run`.
5. Editor loads available views from `/api/experiments/{run_id}/views`.
6. Editor loads a specific payload from
   `/api/experiments/{run_id}/views/{view_id}`.

Any new experiment type should plug into this flow.

## Manifest Schema and Versioning Rules

The dashboard manifest shape is defined by `ExperimentManifest` in
`farm/experiments/manifest.py`:

```json
{
  "schema_version": 1,
  "experiment_type": "intrinsic_evolution",
  "experiment_name": "my-run",
  "base_simulation_config": {},
  "experiment_config": {},
  "dashboard_preset": {
    "default_views": [],
    "default_gene_ids": [],
    "step_window": null
  }
}
```

Rules:

- `schema_version` is required and currently must equal
  `SUPPORTED_SCHEMA_VERSION` (currently `1`).
- `experiment_type` is required and must map to a registered adapter.
- `experiment_name` is required and must be non-empty.
- `base_simulation_config` and `experiment_config` are experiment-specific
  payloads passed to the adapter.
- `dashboard_preset` configures editor defaults, not simulation semantics.

Versioning guidance:

- Keep `schema_version` unchanged for additive, backward-compatible fields.
- Bump `schema_version` for breaking contract changes (field rename/removal,
  meaning changes, required-field changes).
- When bumping, keep validation errors explicit about expected versions.
- Add tests covering old/new schema behavior before changing API consumers.

## Adapter Interface Contract and Registry Wiring

Every experiment type must provide an adapter implementing
`ExperimentAdapterProtocol` (`farm/experiments/interfaces.py`):

- `experiment_type: str`
- `validate_manifest(manifest) -> ManifestValidationResult`
- `run_experiment(run_id, manifest, runtime_options) -> dict`
- `list_views(run_context) -> List[ViewDescriptor]`
- `get_view_data(run_context, view_id, filters) -> dict`

`ViewDescriptor` contract:

- `view_id` (stable identifier used by view-data endpoint)
- `view_type` (renderer-level payload type)
- `title`
- `description`

Registry wiring:

1. Implement adapter, e.g. `farm/experiments/<your_type>_dashboard.py`.
2. Export it from `farm/experiments/__init__.py` if desired for clean imports.
3. Register it at API startup in `farm/api/server.py`:
   `experiment_registry.register(YourAdapter())`.
4. Ensure manifest validation accepts the new `experiment_type`.

## API Expectations: View Descriptors and View Payloads

The API endpoints used by the editor are in `farm/api/server.py`.

### Validate manifest

- **Endpoint:** `POST /api/experiments/manifests/validate`
- **Body:** `{ "manifest": { ... } }`
- **Response data shape:** `ManifestValidationResult`
  - `is_valid: bool`
  - `errors: list[str]`
  - `warnings: list[str]`
  - `normalized_manifest: dict | null`

### Run experiment

- **Endpoint:** `POST /api/experiments/run`
- **Body:** `{ "manifest": { ... }, "runtime_options": { ... } }`
- **Response:** accepted run with `run_id`

### List views

- **Endpoint:** `GET /api/experiments/{run_id}/views`
- **Response data shape:** list of `ViewDescriptor.to_dict()` values

Example item:

```json
{
  "view_id": "summary_cards",
  "view_type": "summary_cards",
  "title": "Run Summary",
  "description": "Resolved policy, completion stats, and final outcomes."
}
```

### Fetch view payload

- **Endpoint:** `POST /api/experiments/{run_id}/views/{view_id}`
- **Body:** `{ "filters": { ... } }`
- **Response data shape:** adapter-specific payload keyed by `view_type`

Current editor renderers (`farm/editor/components/experiment-dashboard.js`)
expect:

- `summary_cards` with `cards`
- `timeseries` with `series`
- `distribution_over_time` with `snapshots`
- `lineage_or_clusters` with `timeseries` and optional `clusters`

If a view is listed, its payload must match its declared `view_type`.

## End-to-End Checklist for Adding a New Experiment Type

Use this as your implementation checklist:

- [ ] Define the new type string (for example, `my_new_experiment`).
- [ ] Implement adapter class following `ExperimentAdapterProtocol`.
- [ ] Validate adapter-specific manifest content in `validate_manifest()`.
- [ ] Build runnable config in `run_experiment()` and write artifacts to
      stable output paths.
- [ ] Implement `list_views()` returning only views that are truly available.
- [ ] Implement `get_view_data()` for each `view_id` from `list_views()`.
- [ ] Register adapter in `experiment_registry` in `farm/api/server.py`.
- [ ] Update manifest validation to allow the new `experiment_type`.
- [ ] Add editor-side defaults (`dashboard_preset.default_views`) if needed.
- [ ] Add tests (manifest validation, registry lookup, adapter view list,
      adapter payload shapes, endpoint-level smoke).
- [ ] Add docs links in `docs/experiments.md` and `docs/README.md`.

## Required Tests (Minimum)

When adding a new type, include at least:

- Manifest acceptance/rejection tests:
  `tests/experiments/test_dashboard_manifest.py` pattern.
- Adapter behavior tests:
  follow `tests/experiments/test_intrinsic_dashboard_adapter.py`.
- Registry error-path test for unsupported types (clear message with supported
  set).
- API smoke tests for:
  - `/api/experiments/manifests/validate`
  - `/api/experiments/{run_id}/views`
  - `/api/experiments/{run_id}/views/{view_id}`
- Editor test updates if you add a new `view_type` renderer in
  `farm/editor/components/experiment-dashboard.js`.

## Common Pitfalls

- Artifact shape mismatch: `get_view_data()` payload does not match
  `view_type` expected by the dashboard renderer.
- Optional views listed unconditionally: `list_views()` should only include
  views that can actually produce data (for example, speciation view only
  when enabled/artifacts exist).
- Incomplete manifest validation: generic validation passes but adapter
  runtime fails due to missing experiment-specific fields.
- Unregistered adapter: manifest validates but run fails because registry
  cannot resolve `experiment_type`.
- Non-stable `view_id`: changing IDs breaks bookmarked or preset view
  selections.

## Intrinsic Evolution as Reference

`IntrinsicEvolutionAdapter` demonstrates all required parts:

- manifest validation delegated plus adapter-specific config build
- run execution and run summary output
- conditional optional views (`speciation_index`) based on metadata
- normalized payloads for summary, time series, distributions, and
  cluster-oriented data

Before introducing a new type, do a quick side-by-side comparison against
`farm/experiments/intrinsic_dashboard.py` and confirm each protocol method
is represented with equivalent test coverage.

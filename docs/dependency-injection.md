## Dependency Injection Improvements

This document explains recent refactors that introduce dependency injection (DI) for configuration and improve service injection patterns across the project. The primary goal is to reduce direct environment and concrete dependency coupling, improving testability, modularity, and maintainability.

### What Changed

- Added `IConfigService` interface and a default `EnvConfigService` implementation.
  - Location: `farm/core/services/interfaces.py`, `farm/core/services/implementations.py`
  - Exported via `farm/core/services/__init__.py`
- Refactored modules that previously read environment variables directly to accept a `config_service` (or use `EnvConfigService` by default):
  - `farm/charts/llm_client.py` now accepts `config_service` and no longer reads `OPENAI_API_KEY` directly.
  - `farm/analysis/registry.register_modules` accepts `config_service` and no longer reads `FARM_ANALYSIS_MODULES` directly.
  - `farm/analysis/service.AnalysisService` injects `EnvConfigService` into `register_modules`.
  - `scripts/analysis_config.py` injects `EnvConfigService` into `register_modules`.
  - `farm/utils/run_analysis.setup_environment` uses `IConfigService` to validate existence of `OPENAI_API_KEY`.
- A lightweight singleton `farm.analysis.null_module.null_module` was exposed to make testing the registry easier.

These changes align with SRP, DIP, and KISS principles and prefactor the codebase for broader DI adoption.

### New Interfaces and Implementations

- `IConfigService`
  - Methods:
    - `get(key: str, default: Optional[str] = None) -> Optional[str]`
    - `get_analysis_module_paths(env_var: str = "FARM_ANALYSIS_MODULES") -> List[str]`
    - `get_openai_api_key() -> Optional[str]`
- `EnvConfigService` (default)
  - Reads from environment variables via `os.getenv` and implements `IConfigService`.

### Updated APIs

- `LLMClient`
  - Before: `LLMClient(api_key: Optional[str] = None)` reads env directly when `api_key` not provided.
  - Now: `LLMClient(api_key: Optional[str] = None, config_service: Optional[IConfigService] = None)`
    - If `api_key` is `None`, the client uses `config_service.get_openai_api_key()`.
    - Defaults to `EnvConfigService()` when not provided.

- `farm.analysis.registry.register_modules`
  - Before: `register_modules(config_env_var: str = "FARM_ANALYSIS_MODULES")` reads env directly.
  - Now: `register_modules(config_env_var: str = "FARM_ANALYSIS_MODULES", *, config_service: Optional[IConfigService] = None)`
    - Uses `config_service.get_analysis_module_paths(config_env_var)`.
    - Defaults to `EnvConfigService()` when not provided.
    - Will attempt to register all configured modules; if none successfully register, it falls back to the built-in dominance module.

- `AnalysisService` and `scripts/analysis_config.py`
  - Both now call `register_modules(config_service=EnvConfigService())`.

- `utils/run_analysis.setup_environment`
  - Before: validated `OPENAI_API_KEY` via `os.getenv`.
  - Now: validates via `config_service.get_openai_api_key()` and defaults to `EnvConfigService()`.

### Example Usage

Injecting a custom config service for testing or alternative configuration sources:

```python
from farm.core.services import IConfigService
from farm.charts.llm_client import LLMClient
from farm.analysis.registry import register_modules


class StaticConfig(IConfigService):
    def __init__(self, api_key: str, modules: list[str] = None):
        self._api_key = api_key
        self._modules = modules or []

    def get(self, key: str, default: str | None = None) -> str | None:
        return default

    def get_analysis_module_paths(self, env_var: str = "FARM_ANALYSIS_MODULES") -> list[str]:
        return list(self._modules)

    def get_openai_api_key(self) -> str | None:
        return self._api_key


cfg = StaticConfig(api_key="sk-test", modules=["farm.analysis.null_module.null_module"])
register_modules(config_service=cfg)
client = LLMClient(config_service=cfg)
```

### Backward Compatibility

- Existing code that did not pass a `config_service` continues to work due to defaulting to `EnvConfigService`.
- `register_modules` still loads the built-in dominance module if no configured modules register successfully.

### Testing

New tests validate the DI behavior:
- `tests/test_config_service.py` covers `EnvConfigService` behaviors.
- `tests/test_analysis_registry_di.py` ensures the registry uses an injected module list provider.

To run only the new tests:

```bash
python3 -m pytest -q tests/test_config_service.py tests/test_analysis_registry_di.py
```

Note: The full test suite depends on additional packages (e.g., `pydantic`, `psutil`, `stable_baselines3`). Those are unrelated to these DI changes.

### Migration Notes

- If you previously relied on direct environment variable reads for:
  - `OPENAI_API_KEY` in `LLMClient`: prefer passing `api_key` directly or provide a custom `IConfigService`.
  - `FARM_ANALYSIS_MODULES` in analysis registry: pass module paths via a custom `IConfigService` or set the env var as before (default service will use it).
- For other modules, prefer adding an optional `config_service: IConfigService | None = None` parameter and default to `EnvConfigService()` internally when needed.

### Related Files

- `farm/core/services/interfaces.py`
- `farm/core/services/implementations.py`
- `farm/core/services/__init__.py`
- `farm/charts/llm_client.py`
- `farm/charts/chart_analyzer.py`
- `farm/analysis/registry.py`
- `farm/analysis/service.py`
- `farm/utils/run_analysis.py`
- `scripts/analysis_config.py`


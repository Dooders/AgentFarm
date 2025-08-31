from types import SimpleNamespace

from farm.analysis.registry import registry, register_modules


class DummyConfigService:
    def __init__(self, paths):
        self._paths = paths

    def get(self, key, default=None):
        return default

    def get_analysis_module_paths(self, env_var: str = "FARM_ANALYSIS_MODULES"):
        return list(self._paths)

    def get_openai_api_key(self):
        return None


def test_register_modules_with_injected_paths(monkeypatch):
    # Clear existing registry state for this test by reinitializing its dict
    registry._modules.clear()

    # Provide a known good lightweight module path
    paths = ["farm.analysis.null_module.null_module"]
    cfg = DummyConfigService(paths)

    register_modules(config_service=cfg)

    # The registry should now contain the null module by its name
    assert "null" in registry.get_module_names()


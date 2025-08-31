import os

from farm.core.services import EnvConfigService


def test_env_config_service_basic_get(monkeypatch):
    monkeypatch.setenv("TEST_KEY_X", "value123")
    cfg = EnvConfigService()
    assert cfg.get("TEST_KEY_X") == "value123"
    assert cfg.get("MISSING_KEY", "default") == "default"


def test_env_config_service_analysis_paths(monkeypatch):
    monkeypatch.setenv(
        "FARM_ANALYSIS_MODULES",
        " farm.analysis.dominance.module.dominance_module ,farm.analysis.template.module.template_module ",
    )
    cfg = EnvConfigService()
    paths = cfg.get_analysis_module_paths()
    assert paths[0] == "farm.analysis.dominance.module.dominance_module"
    assert paths[1] == "farm.analysis.template.module.template_module"


def test_env_config_service_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = EnvConfigService()
    assert cfg.get_openai_api_key() is None
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert cfg.get_openai_api_key() == "sk-test"


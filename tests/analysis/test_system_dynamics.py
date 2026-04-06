"""Tests for system dynamics analysis module."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from farm.analysis.common.context import AnalysisContext
from farm.analysis.registry import registry
from farm.analysis.service import AnalysisRequest, AnalysisService
from farm.analysis.system_dynamics.compute import (
    resource_population_coupling,
    synthesize_system_dynamics,
    feedback_loop_candidates,
)
from farm.analysis.system_dynamics.data import process_system_dynamics_data
from farm.analysis.system_dynamics.analyze import (
    analyze_system_dynamics_synthesis,
    write_unified_system_dynamics_report,
)
from farm.analysis.system_dynamics.module import system_dynamics_module


@pytest.fixture
def coupled_system_df():
    """Per-step population and resources moving together (+ noise)."""
    rng = np.random.default_rng(42)
    steps = np.arange(30)
    resources = 1000 - steps * 15 + rng.normal(0, 5, size=len(steps))
    agents = 80 - steps + rng.normal(0, 2, size=len(steps))
    return pd.DataFrame(
        {
            "step": steps.astype(int),
            "total_resources": np.clip(resources, 50, None),
            "total_agents": np.clip(agents, 5, None).astype(int),
            "actions_per_step": rng.integers(5, 50, size=len(steps)),
            "mean_reward_per_step": rng.normal(0.1, 0.05, size=len(steps)),
            "total_reward_per_step": rng.normal(2.0, 0.5, size=len(steps)),
        }
    )


class TestResourcePopulationCoupling:
    def test_positive_correlation_levels(self, coupled_system_df):
        out = resource_population_coupling(coupled_system_df)
        assert out["levels"]["available"] is True
        assert out["levels"]["n"] >= 3
        assert -1.0 <= out["levels"]["pearson_r"] <= 1.0

    def test_missing_columns(self):
        out = resource_population_coupling(pd.DataFrame({"step": [0, 1]}))
        assert "error" in out


class TestFeedbackLoopCandidates:
    def test_detects_stress_and_recovery_pattern(self):
        # Sharp dip in resources with population falling then resources rebound
        df = pd.DataFrame(
            {
                "step": range(12),
                "total_resources": [500, 400, 200, 180, 190, 220, 260, 300, 320, 330, 340, 350],
                "total_agents": [50, 49, 48, 45, 44, 44, 45, 46, 47, 48, 49, 50],
            }
        )
        out = feedback_loop_candidates(df, scarcity_quantile=0.3, recovery_window=4)
        assert out["available"] is True
        assert out["count"] >= 1


class TestProcessSystemDynamicsData:
    def test_merge_via_mocks(self, tmp_path):
        pop = pd.DataFrame({"step": [0, 1, 2], "total_agents": [10, 9, 8]})
        res = pd.DataFrame({"step": [0, 1, 2], "total_resources": [100, 90, 80]})
        tmp = pd.DataFrame(
            {
                "step": [0, 0, 1, 2],
                "agent_id": [1, 2, 1, 1],
                "action_type": ["a", "b", "a", "a"],
                "reward": [1.0, 2.0, 0.5, -0.2],
            }
        )

        with (
            patch(
                "farm.analysis.system_dynamics.data.process_population_data",
                return_value=pop,
            ),
            patch(
                "farm.analysis.system_dynamics.data.process_resource_data",
                return_value=res,
            ),
            patch(
                "farm.analysis.system_dynamics.data.process_temporal_data",
                return_value=tmp,
            ),
        ):
            merged = process_system_dynamics_data(tmp_path)

        assert list(merged["step"]) == [0, 1, 2]
        assert "actions_per_step" in merged.columns
        assert merged.loc[merged["step"] == 0, "actions_per_step"].iloc[0] == 2


class TestAnalyzeAndReport:
    def test_synthesis_writes_json(self, coupled_system_df, tmp_path):
        ctx = AnalysisContext(
            output_path=tmp_path,
            metadata={"experiment_path": str(tmp_path / "exp")},
        )
        analyze_system_dynamics_synthesis(coupled_system_df, ctx)
        p = tmp_path / "system_dynamics_synthesis.json"
        assert p.is_file()
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "resource_population" in data

    def test_unified_report(self, coupled_system_df, tmp_path):
        ctx = AnalysisContext(
            output_path=tmp_path,
            metadata={"experiment_path": str(tmp_path / "exp")},
        )
        analyze_system_dynamics_synthesis(coupled_system_df, ctx)
        write_unified_system_dynamics_report(coupled_system_df, ctx)
        rep = tmp_path / "system_dynamics_report.json"
        assert rep.is_file()
        html = tmp_path / "system_dynamics_report.html"
        assert html.is_file()
        bundle = json.loads(rep.read_text(encoding="utf-8"))
        assert bundle["module"] == "system_dynamics"
        assert "synthesis" in bundle


class TestRegistryAndService:
    def test_module_protocol_and_name(self):
        assert system_dynamics_module.name == "system_dynamics"
        assert system_dynamics_module.get_data_processor() is not None
        assert "synthesis" in system_dynamics_module.get_function_groups()

    def test_get_module_after_register(self, config_service_mock, tmp_path):
        registry.register(system_dynamics_module)
        try:
            service = AnalysisService(
                config_service=config_service_mock,
                auto_register=False,
            )
            exp = tmp_path / "experiment"
            exp.mkdir()
            out = tmp_path / "out"
            out.mkdir()

            df = pd.DataFrame(
                {
                    "step": [0, 1, 2],
                    "total_agents": [10, 9, 8],
                    "total_resources": [100.0, 90.0, 80.0],
                }
            )
            with patch.object(
                system_dynamics_module,
                "get_data_processor",
                return_value=type(
                    "P",
                    (),
                    {
                        "process": lambda self, path, **kw: df,
                    },
                )(),
            ):
                req = AnalysisRequest(
                    module_name="system_dynamics",
                    experiment_path=exp,
                    output_path=out,
                    group="synthesis",
                    enable_caching=False,
                )
                result = service.run(req)
            assert result.success
            assert (out / "system_dynamics_report.json").is_file()
        finally:
            registry.unregister("system_dynamics")


class TestSynthesize:
    def test_synthesize_keys(self, coupled_system_df):
        syn = synthesize_system_dynamics(coupled_system_df)
        for key in (
            "resource_population",
            "action_reward_lags",
            "scarcity_population_volatility",
            "granger_changes",
            "feedback_loop_candidates",
        ):
            assert key in syn

"""
Comprehensive tests for advantage analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.advantage import (
    advantage_module,
    compute_advantages,
    analyze_advantage_patterns,
    plot_advantage_results,
    plot_advantage_correlation_matrix,
    plot_advantage_distribution,
    plot_advantage_timeline,
)
from farm.analysis.common.context import AnalysisContext


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = MagicMock()
    return session


@pytest.fixture
def sample_advantage_data():
    """Create sample advantage data."""
    return pd.DataFrame(
        {
            "iteration": range(25),
            "agent_id": [f"agent_{i}" for i in range(25)],
            "advantage_score": np.random.uniform(0, 1, 25),
            "relative_advantage": np.random.uniform(-0.5, 0.5, 25),
            "evolutionary_fitness": np.random.uniform(0, 1, 25),
            "resource_advantage": np.random.uniform(0, 1, 25),
            "survival_advantage": np.random.uniform(0, 1, 25),
            "reproduction_advantage": np.random.uniform(0, 1, 25),
            "dominant_type": np.random.choice(["system", "independent", "control"], 25),
            "system_dominance_score": np.random.uniform(0, 1, 25),
            "independent_dominance_score": np.random.uniform(0, 1, 25),
            "control_dominance_score": np.random.uniform(0, 1, 25),
            "system_vs_independent_resource_acquisition_early_phase_advantage": np.random.uniform(-0.5, 0.5, 25),
        }
    )


@pytest.fixture
def sample_advantage_analysis():
    """Create sample advantage analysis results."""
    return {
        "advantage_significance": {
            "system_vs_independent_advantage": {
                "mean": 0.3,
                "t_statistic": 2.5,
                "p_value": 0.02,
                "significant": True,
            }
        },
        "dominance_correlations": {
            "system": {"system_vs_independent_advantage": 0.8},
            "independent": {"independent_vs_control_advantage": 0.6},
            "control": {},
        },
        "advantage_category_importance": {
            "by_category": {
                "resource_acquisition": {
                    "system": {"average_relevance": 0.7, "max_relevance": 0.9},
                }
            },
            "overall_ranking": {
                "resource_acquisition": 0.7,
                "reproduction": 0.5,
                "survival": 0.4,
            },
        },
        "advantage_timing_analysis": {
            "system": {
                "early": {"adv1": {"average_value": 0.5, "favors_agent": True}},
                "mid": {},
                "late": {},
            }
        },
        "agent_type_specific_analysis": {
            "system": {
                "top_predictors": {
                    "system_vs_independent_resource_acquisition_early_phase_advantage": {
                        "dominant_avg": 0.5,
                        "non_dominant_avg": 0.2,
                        "effect_size": 1.2,
                        "p_value": 0.01,
                        "significant": True,
                    }
                },
                "significant_predictors": {},
            }
        },
        "advantage_threshold_analysis": {
            "system": {
                "system_vs_independent_advantage": {
                    "optimal_threshold": 0.3,
                    "dominance_ratio": 2.5,
                }
            }
        },
        "avg_advantage_score": 0.45,
        "advantage_distribution": {"mean": 0.4, "std": 0.2},
        "correlation_matrix": {},
    }


class TestAdvantageComputations:
    """Test advantage statistical computations."""

    def test_compute_advantages(self, mock_session):
        """Test advantage computation."""
        # Set up mock to return consistent values for all queries
        mock_session.query.return_value.scalar.return_value = 0
        mock_session.query.return_value.first.return_value = (0, 0, 0, 0)

        advantages = compute_advantages(mock_session)

        assert isinstance(advantages, dict)
        # Should have advantage categories
        expected_categories = ["resource_acquisition", "reproduction", "survival", "population_growth", "combat"]
        # At least some categories should be present
        assert any(cat in advantages for cat in expected_categories)

    def test_compute_advantages_with_focus_type(self, mock_session):
        """Test advantage computation with focus agent type."""
        # Set up mock to return consistent values for all queries
        mock_session.query.return_value.scalar.return_value = 0
        mock_session.query.return_value.first.return_value = (0, 0, 0, 0)

        advantages = compute_advantages(mock_session, focus_agent_type="system")

        assert isinstance(advantages, dict)

    def test_compute_advantages_no_data(self, mock_session):
        """Test advantage computation with no data."""
        # Set up mock to return 0 for all queries (no data scenario)
        mock_session.query.return_value.scalar.return_value = 0
        mock_session.query.return_value.first.return_value = (0, 0, 0, 0)

        advantages = compute_advantages(mock_session)

        # Should return empty or minimal structure
        assert isinstance(advantages, dict)


class TestAdvantageAnalysis:
    """Test advantage analysis functions."""

    def test_analyze_advantage_patterns(self, sample_advantage_data):
        """Test advantage pattern analysis."""
        result = analyze_advantage_patterns(sample_advantage_data)

        assert isinstance(result, dict)
        assert "avg_advantage_score" in result
        assert "advantage_distribution" in result
        assert "correlation_matrix" in result
        assert "advantage_significance" in result

    def test_analyze_advantage_patterns_empty(self):
        """Test advantage patterns with empty DataFrame."""
        result = analyze_advantage_patterns(pd.DataFrame())

        assert isinstance(result, dict)
        assert "avg_advantage_score" in result
        assert result["avg_advantage_score"] == 0.0

    def test_analyze_advantage_patterns_no_dominance(self):
        """Test advantage patterns without dominant_type column."""
        df = pd.DataFrame(
            {
                "iteration": range(10),
                "system_vs_independent_resource_acquisition_early_phase_advantage": np.random.uniform(-0.5, 0.5, 10),
            }
        )

        result = analyze_advantage_patterns(df)

        # Should still compute some results
        assert "advantage_significance" in result

    def test_analyze_advantage_patterns_correlations(self, sample_advantage_data):
        """Test correlation analysis in advantage patterns."""
        result = analyze_advantage_patterns(sample_advantage_data)

        assert "dominance_correlations" in result
        # Should have correlations for each agent type
        for agent_type in ["system", "independent", "control"]:
            assert agent_type in result["dominance_correlations"]

    def test_analyze_advantage_category_importance(self, sample_advantage_data):
        """Test advantage category importance analysis."""
        result = analyze_advantage_patterns(sample_advantage_data)

        assert "advantage_category_importance" in result

    def test_get_advantage_recommendations(self, sample_advantage_analysis):
        """Test generating recommendations from analysis."""
        from farm.analysis.advantage.analyze import get_advantage_recommendations

        recommendations = get_advantage_recommendations(sample_advantage_analysis)

        assert isinstance(recommendations, dict)
        assert "system" in recommendations
        assert "independent" in recommendations
        assert "control" in recommendations

        # Check structure
        for agent_type in ["system", "independent", "control"]:
            assert "key_advantages" in recommendations[agent_type]
            assert "critical_thresholds" in recommendations[agent_type]
            assert "phase_importance" in recommendations[agent_type]

    def test_get_advantage_recommendations_empty(self):
        """Test recommendations with empty analysis."""
        from farm.analysis.advantage.analyze import get_advantage_recommendations

        recommendations = get_advantage_recommendations({})

        # Should return empty recommendations structure
        assert isinstance(recommendations, dict)
        for agent_type in ["system", "independent", "control"]:
            assert agent_type in recommendations


class TestAdvantageVisualization:
    """Test advantage visualization functions."""

    @patch("farm.analysis.advantage.plot.plt")
    def test_plot_advantage_distribution(self, mock_plt, sample_advantage_data, tmp_path):
        """Test advantage distribution plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        result = plot_advantage_distribution(sample_advantage_data, ctx)

        assert result is None
        assert mock_plt.savefig.called

    @patch("farm.analysis.advantage.plot.plt")
    def test_plot_advantage_distribution_empty(self, mock_plt, tmp_path):
        """Test distribution plotting with empty data."""
        ctx = AnalysisContext(output_path=tmp_path)

        result = plot_advantage_distribution(pd.DataFrame(), ctx)

        # Should handle gracefully
        assert result is None

    @patch("farm.analysis.advantage.plot.plt")
    def test_plot_advantage_timeline(self, mock_plt, sample_advantage_data, tmp_path):
        """Test advantage timeline plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        result = plot_advantage_timeline(sample_advantage_data, ctx)

        assert result is None
        assert mock_plt.savefig.called

    @patch("farm.analysis.advantage.plot.plt")
    def test_plot_advantage_correlation_matrix(self, mock_plt, sample_advantage_data, tmp_path):
        """Test advantage correlation matrix plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_advantage_correlation_matrix(sample_advantage_data, str(tmp_path), ctx=ctx)

        # Should create plot if data is suitable
        assert mock_plt.savefig.called or mock_plt.figure.called

    @patch("farm.analysis.advantage.plot.plt")
    def test_plot_advantage_results(self, mock_plt, sample_advantage_data, sample_advantage_analysis, tmp_path):
        """Test comprehensive advantage results plotting."""
        ctx = AnalysisContext(output_path=tmp_path)

        plot_advantage_results(sample_advantage_data, sample_advantage_analysis, str(tmp_path), ctx=ctx)

        # Should create multiple plots
        assert mock_plt.savefig.called


class TestAdvantageModule:
    """Test advantage module integration."""

    def test_advantage_module_registration(self):
        """Test module registration."""
        assert advantage_module.name == "advantage"
        assert (
            advantage_module.description
            == "Analysis of relative advantages between agent types and their impact on dominance patterns"
        )

    def test_advantage_module_function_names(self):
        """Test module function names."""
        functions = advantage_module.get_function_names()
        expected_functions = [
            "plot_advantage_distribution",
            "plot_advantage_timeline",
            "plot_advantage_correlations",
            "plot_advantage_evolution",
            "plot_advantage_comparison",
            "plot_advantage_optimization",
            "analyze_advantage_patterns",
            "analyze_advantage_evolution",
        ]

        for func_name in expected_functions:
            assert func_name in functions

    def test_advantage_module_function_groups(self):
        """Test module function groups."""
        groups = advantage_module.get_function_groups()
        assert "all" in groups
        assert "plots" in groups
        assert "analysis" in groups

    def test_advantage_module_data_processor(self):
        """Test module data processor."""
        processor = advantage_module.get_data_processor()
        assert processor is not None

    def test_module_validator(self):
        """Test module validator."""
        validator = advantage_module.get_validator()
        assert validator is not None

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        functions = advantage_module.get_functions()
        assert len(functions) >= 8


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_analyze_patterns_with_nan_values(self):
        """Test pattern analysis with NaN values."""
        df = pd.DataFrame(
            {
                "iteration": range(10),
                "dominant_type": ["system"] * 10,
                "system_vs_independent_advantage": [0.5, np.nan, 0.3, np.nan, 0.4, 0.6, np.nan, 0.5, 0.7, 0.4],
                "system_dominance_score": [0.8, 0.7, np.nan, 0.9, 0.8, 0.85, 0.9, np.nan, 0.95, 0.9],
            }
        )

        result = analyze_advantage_patterns(df)

        # Should handle NaN gracefully
        assert "advantage_significance" in result

    def test_analyze_patterns_single_agent_type(self):
        """Test pattern analysis with single agent type."""
        df = pd.DataFrame(
            {
                "iteration": range(20),
                "dominant_type": ["system"] * 20,
                "system_dominance_score": np.random.uniform(0, 1, 20),
            }
        )

        result = analyze_advantage_patterns(df)

        # Should handle single type
        assert isinstance(result, dict)

    def test_analyze_patterns_insufficient_samples(self):
        """Test pattern analysis with insufficient samples."""
        df = pd.DataFrame(
            {
                "iteration": [0, 1, 2],
                "dominant_type": ["system", "independent", "control"],
                "system_dominance_score": [0.8, 0.3, 0.2],
            }
        )

        result = analyze_advantage_patterns(df)

        # Should handle small sample size
        assert isinstance(result, dict)

    def test_compute_advantages_database_error(self, mock_session):
        """Test advantage computation with database error."""
        mock_session.query.side_effect = Exception("Database error")

        # Should raise or handle error
        with pytest.raises(Exception):
            compute_advantages(mock_session)

    def test_plot_correlation_matrix_empty(self, tmp_path):
        """Test correlation matrix plotting with empty DataFrame."""
        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.advantage.plot.plt"):
            # Should handle empty data gracefully
            plot_advantage_correlation_matrix(pd.DataFrame(), str(tmp_path), ctx=ctx)

    def test_plot_correlation_matrix_missing_columns(self, tmp_path):
        """Test correlation matrix without required columns."""
        df = pd.DataFrame(
            {
                "value1": [1, 2, 3],
                "value2": [4, 5, 6],
            }
        )

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.advantage.plot.plt"):
            # Should skip plotting if no advantage columns
            plot_advantage_correlation_matrix(df, str(tmp_path), ctx=ctx)

    def test_analyze_with_infinite_values(self):
        """Test analysis with infinite values."""
        df = pd.DataFrame(
            {
                "iteration": range(10),
                "system_vs_independent_advantage": [np.inf, 0.5, 0.3, -np.inf, 0.4, 0.6, 0.5, 0.5, 0.7, 0.4],
                "system_dominance_score": np.random.uniform(0, 1, 10),
            }
        )

        result = analyze_advantage_patterns(df)

        # Should handle infinity values
        assert isinstance(result, dict)

    def test_recommendations_minimal_data(self):
        """Test recommendations with minimal analysis data."""
        from farm.analysis.advantage.analyze import get_advantage_recommendations

        minimal_analysis = {
            "agent_type_specific_analysis": {
                "system": {"top_predictors": {}, "significant_predictors": {}},
            }
        }

        recommendations = get_advantage_recommendations(minimal_analysis)

        assert "system" in recommendations
        assert recommendations["system"]["key_advantages"] == []

    def test_correlation_analysis_single_column(self):
        """Test correlation analysis with single advantage column."""
        df = pd.DataFrame(
            {
                "iteration": range(20),
                "dominant_type": ["system"] * 20,
                "system_vs_independent_advantage": np.random.uniform(-0.5, 0.5, 20),
                "system_dominance_score": np.random.uniform(0, 1, 20),
            }
        )

        result = analyze_advantage_patterns(df)

        assert "dominance_correlations" in result

    def test_plot_with_data_cleaned_flag(self, sample_advantage_data, sample_advantage_analysis, tmp_path):
        """Test plotting with data_cleaned flag."""
        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.advantage.plot.plt"):
            plot_advantage_results(
                sample_advantage_data, sample_advantage_analysis, str(tmp_path), ctx=ctx, data_cleaned=True
            )

    @patch("farm.analysis.advantage.compute.logger")
    def test_compute_with_logging(self, mock_logger, mock_session):
        """Test that computation logs appropriately."""
        # Set up mock to return consistent values for all queries
        mock_session.query.return_value.scalar.return_value = 0
        mock_session.query.return_value.first.return_value = (0, 0, 0, 0)

        # Should log debug messages
        compute_advantages(mock_session)

        # Verify logging was called
        assert mock_logger.debug.called


class TestAdvantageHelperFunctions:
    """Test helper functions in advantage module."""

    def test_process_single_simulation(self, mock_session):
        """Test processing single simulation."""
        from farm.analysis.advantage.analyze import process_single_simulation

        with patch("farm.analysis.advantage.analyze.compute_comprehensive_dominance") as mock_dom, patch(
            "farm.analysis.advantage.analyze.compute_advantages"
        ) as mock_adv, patch("farm.analysis.advantage.analyze.compute_advantage_dominance_correlation") as mock_corr:
            mock_dom.return_value = {
                "dominant_type": "system",
                "scores": {"system": 0.8, "independent": 0.3, "control": 0.2},
            }
            mock_adv.return_value = {
                "resource_acquisition": {"system_vs_independent": {"advantage": 0.5}},
                "composite_advantage": {
                    "system_vs_independent": {"score": 0.6, "components": {"resource": 0.3, "survival": 0.3}}
                },
            }
            mock_corr.return_value = {
                "summary": {
                    "advantage_ratio": 1.5,
                    "advantages_favoring_dominant": 5,
                    "total_advantages": 8,
                }
            }

            result = process_single_simulation(mock_session, iteration=1, config={})

            assert result is not None
            assert "iteration" in result
            assert "dominant_type" in result

    def test_process_single_simulation_error(self, mock_session):
        """Test processing simulation with error."""
        from farm.analysis.advantage.analyze import process_single_simulation

        with patch("farm.analysis.advantage.analyze.compute_comprehensive_dominance") as mock_dom:
            mock_dom.side_effect = Exception("Database error")

            result = process_single_simulation(mock_session, iteration=1, config={})

            # Should return None on error
            assert result is None

    def test_analyze_advantages_function(self, tmp_path):
        """Test analyze_advantages main function."""
        from farm.analysis.advantage.analyze import analyze_advantages

        exp_path = tmp_path / "experiment"
        exp_path.mkdir()

        with patch("farm.analysis.advantage.analyze.setup_and_process_simulations") as mock_setup:
            mock_setup.return_value = [
                {"iteration": 0, "dominant_type": "system"},
                {"iteration": 1, "dominant_type": "independent"},
            ]

            result = analyze_advantages(str(exp_path))

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2

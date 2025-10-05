"""
Comprehensive tests for population analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from farm.analysis.population import (
    population_module,
    compute_population_statistics,
    compute_birth_death_rates,
    compute_population_stability,
    compute_growth_rate_analysis,
    compute_demographic_metrics,
    analyze_population_dynamics,
    analyze_agent_composition,
    analyze_comprehensive_population,
    plot_population_over_time,
    plot_birth_death_rates,
    plot_agent_composition,
    plot_population_dashboard,
)
from farm.analysis.common.context import AnalysisContext


@pytest.fixture
def sample_population_data():
    """Create sample population data."""
    return pd.DataFrame({
        'step': range(100),
        'total_agents': [100 + i + (i % 10) * 2 for i in range(100)],
        'system_agents': [50 + i // 2 for i in range(100)],
        'independent_agents': [30 + i // 3 for i in range(100)],
        'control_agents': [20 + i // 5 for i in range(100)],
        'births': [5 + i % 7 for i in range(100)],
        'deaths': [2 + i % 5 for i in range(100)],
    })


@pytest.fixture
def sample_birth_death_data():
    """Create sample birth/death data."""
    return pd.DataFrame({
        'step': range(50),
        'births': [5 + i % 10 for i in range(50)],
        'deaths': [2 + i % 8 for i in range(50)],
        'total_agents': [100 + i * 2 for i in range(50)],
    })


@pytest.fixture
def sample_composition_data():
    """Create sample agent composition data."""
    return pd.DataFrame({
        'step': range(30),
        'total_agents': [100 + i * 2 for i in range(30)],
        'system_agents': [50 + i for i in range(30)],
        'independent_agents': [30 + i // 2 for i in range(30)],
        'control_agents': [20 + i // 3 for i in range(30)],
    })


class TestPopulationComputations:
    """Test population statistical computations."""

    def test_compute_population_statistics(self, sample_population_data):
        """Test population statistics computation."""
        stats = compute_population_statistics(sample_population_data)
        
        assert isinstance(stats, dict)
        assert 'total' in stats
        assert 'peak_step' in stats
        assert 'peak_value' in stats
        assert 'final_value' in stats
        assert 'trend' in stats
        assert 'survival_rate' in stats
        
        # Check agent type statistics
        assert 'system_agents' in stats
        assert 'independent_agents' in stats
        assert 'control_agents' in stats

    def test_compute_population_statistics_minimal(self):
        """Test population statistics with minimal data."""
        df = pd.DataFrame({
            'step': range(10),
            'total_agents': [100 + i * 5 for i in range(10)],
        })
        
        stats = compute_population_statistics(df)
        
        assert stats['peak_value'] == 145
        assert stats['final_value'] == 145

    def test_compute_birth_death_rates(self, sample_birth_death_data):
        """Test birth and death rate computation."""
        rates = compute_birth_death_rates(sample_birth_death_data)
        
        assert isinstance(rates, dict)
        assert 'total_births' in rates
        assert 'total_deaths' in rates
        assert 'birth_rate' in rates
        assert 'death_rate' in rates
        assert 'net_growth' in rates
        assert 'growth_rate' in rates
        
        assert rates['net_growth'] == rates['total_births'] - rates['total_deaths']

    def test_compute_birth_death_rates_no_data(self):
        """Test birth/death rates with missing columns."""
        df = pd.DataFrame({
            'step': range(10),
            'total_agents': [100 + i for i in range(10)],
        })
        
        rates = compute_birth_death_rates(df)
        
        assert rates == {}

    def test_compute_population_stability(self, sample_population_data):
        """Test population stability computation."""
        stability = compute_population_stability(sample_population_data)
        
        assert isinstance(stability, dict)
        assert 'mean_cv' in stability
        assert 'stability_score' in stability
        assert 'volatility' in stability
        assert 'max_fluctuation' in stability
        assert 'mean_relative_change' in stability
        assert 'max_relative_change' in stability
        
        assert 0 <= stability['stability_score'] <= 1

    def test_compute_population_stability_short_data(self):
        """Test stability with short time series."""
        df = pd.DataFrame({
            'step': range(5),
            'total_agents': [100, 105, 110, 108, 112],
        })
        
        stability = compute_population_stability(df, window=10)
        
        # Should handle short data by adjusting window
        assert isinstance(stability, dict)
        assert 'mean_cv' in stability

    def test_compute_population_stability_constant(self):
        """Test stability with constant population."""
        df = pd.DataFrame({
            'step': range(20),
            'total_agents': [100] * 20,
        })
        
        stability = compute_population_stability(df)
        
        # Constant population should be very stable
        assert stability['volatility'] == 0.0
        assert stability['max_fluctuation'] == 0.0

    def test_compute_growth_rate_analysis(self, sample_population_data):
        """Test growth rate analysis computation."""
        analysis = compute_growth_rate_analysis(sample_population_data)
        
        assert isinstance(analysis, dict)
        assert 'average_growth_rate' in analysis
        assert 'max_growth_rate' in analysis
        assert 'min_growth_rate' in analysis
        assert 'time_in_growth' in analysis
        assert 'time_in_decline' in analysis
        assert 'time_stable' in analysis

    def test_compute_growth_rate_analysis_exponential(self):
        """Test growth rate analysis with exponential growth."""
        df = pd.DataFrame({
            'step': range(50),
            'total_agents': [int(100 * (1.05 ** i)) for i in range(50)],
        })
        
        analysis = compute_growth_rate_analysis(df)
        
        # Should detect growth phase and doubling time
        assert analysis['time_in_growth'] > 0
        if analysis['doubling_time'] is not None:
            assert analysis['doubling_time'] > 0

    def test_compute_demographic_metrics(self, sample_composition_data):
        """Test demographic metrics computation."""
        metrics = compute_demographic_metrics(sample_composition_data)
        
        assert isinstance(metrics, dict)
        assert 'diversity_index' in metrics
        assert 'dominance_index' in metrics
        assert 'type_proportions' in metrics

    def test_compute_demographic_metrics_missing_types(self):
        """Test demographic metrics without agent type columns."""
        df = pd.DataFrame({
            'step': range(10),
            'total_agents': [100 + i for i in range(10)],
        })
        
        metrics = compute_demographic_metrics(df)
        
        # Should return empty or minimal metrics
        assert isinstance(metrics, dict)


class TestPopulationAnalysis:
    """Test population analysis functions."""

    def test_analyze_population_dynamics(self, tmp_path, sample_population_data):
        """Test population dynamics analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_population_dynamics(sample_population_data, ctx)
        
        output_file = tmp_path / "population_statistics.json"
        assert output_file.exists()
        
        with open(output_file) as f:
            data = json.load(f)
        
        assert 'statistics' in data
        assert 'rates' in data
        assert 'stability' in data

    def test_analyze_agent_composition(self, tmp_path, sample_composition_data):
        """Test agent composition analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_agent_composition(sample_composition_data, ctx)
        
        output_file = tmp_path / "agent_composition.csv"
        assert output_file.exists()
        
        df = pd.read_csv(output_file)
        
        # Should have percentage columns
        assert 'system_agents_pct' in df.columns
        assert 'independent_agents_pct' in df.columns
        assert 'control_agents_pct' in df.columns

    def test_analyze_comprehensive_population(self, tmp_path, sample_population_data):
        """Test comprehensive population analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        analyze_comprehensive_population(sample_population_data, ctx)
        
        # Check JSON output
        json_file = tmp_path / "comprehensive_population_analysis.json"
        assert json_file.exists()
        
        with open(json_file) as f:
            data = json.load(f)
        
        assert 'statistics' in data
        assert 'stability' in data
        assert 'summary' in data
        
        # Check text report
        report_file = tmp_path / "population_report.txt"
        assert report_file.exists()

    def test_analyze_comprehensive_population_minimal(self, tmp_path):
        """Test comprehensive analysis with minimal data."""
        df = pd.DataFrame({
            'step': range(20),
            'total_agents': [100 + i * 2 for i in range(20)],
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        analyze_comprehensive_population(df, ctx)
        
        json_file = tmp_path / "comprehensive_population_analysis.json"
        assert json_file.exists()

    def test_analyze_with_progress_callback(self, tmp_path, sample_population_data):
        """Test analysis with progress reporting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        progress_calls = []
        ctx.report_progress = lambda msg, prog: progress_calls.append((msg, prog))
        
        analyze_population_dynamics(sample_population_data, ctx)
        
        # Should have called progress
        assert len(progress_calls) > 0


class TestPopulationVisualization:
    """Test population visualization functions."""

    def test_plot_population_over_time(self, tmp_path, sample_population_data):
        """Test population over time plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_population_over_time(sample_population_data, ctx)
        
        plot_file = tmp_path / "population_over_time.png"
        assert plot_file.exists()

    def test_plot_population_over_time_minimal(self, tmp_path):
        """Test population plotting with minimal data."""
        df = pd.DataFrame({
            'step': range(10),
            'total_agents': [100 + i * 5 for i in range(10)],
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_population_over_time(df, ctx)
        
        plot_file = tmp_path / "population_over_time.png"
        assert plot_file.exists()

    def test_plot_birth_death_rates(self, tmp_path, sample_birth_death_data):
        """Test birth/death rates plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_birth_death_rates(sample_birth_death_data, ctx)
        
        plot_file = tmp_path / "birth_death_rates.png"
        assert plot_file.exists()

    def test_plot_birth_death_rates_no_data(self, tmp_path):
        """Test birth/death plotting with missing data."""
        df = pd.DataFrame({
            'step': range(10),
            'total_agents': [100 + i for i in range(10)],
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_birth_death_rates(df, ctx)
        
        # Should not create plot file
        plot_file = tmp_path / "birth_death_rates.png"
        assert not plot_file.exists()

    def test_plot_agent_composition(self, tmp_path, sample_composition_data):
        """Test agent composition plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_agent_composition(sample_composition_data, ctx)
        
        plot_file = tmp_path / "agent_composition.png"
        assert plot_file.exists()

    def test_plot_agent_composition_no_types(self, tmp_path):
        """Test composition plotting without agent types."""
        df = pd.DataFrame({
            'step': range(10),
            'total_agents': [100 + i for i in range(10)],
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_agent_composition(df, ctx)
        
        # Should not create plot
        plot_file = tmp_path / "agent_composition.png"
        assert not plot_file.exists()

    def test_plot_population_dashboard(self, tmp_path, sample_population_data):
        """Test population dashboard plotting."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_population_dashboard(sample_population_data, ctx)
        
        # Should create dashboard
        plot_file = tmp_path / "population_dashboard.png"
        assert plot_file.exists()

    def test_plot_with_custom_options(self, tmp_path, sample_population_data):
        """Test plotting with custom options."""
        ctx = AnalysisContext(output_path=tmp_path)
        
        plot_population_over_time(sample_population_data, ctx, figsize=(10, 8), dpi=150)
        
        plot_file = tmp_path / "population_over_time.png"
        assert plot_file.exists()


class TestPopulationModule:
    """Test population module integration."""

    def test_population_module_registration(self):
        """Test module registration."""
        assert population_module.name == "population"
        assert population_module.description == "Analysis of population dynamics, births, deaths, and agent composition"

    def test_population_module_function_names(self):
        """Test module function names."""
        functions = population_module.get_function_names()
        
        assert "analyze_dynamics" in functions
        assert "analyze_composition" in functions
        assert "analyze_comprehensive" in functions
        assert "plot_population" in functions
        assert "plot_births_deaths" in functions
        assert "plot_composition" in functions

    def test_population_module_function_groups(self):
        """Test module function groups."""
        groups = population_module.get_function_groups()
        
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups
        assert "basic" in groups
        assert "comprehensive" in groups

    def test_population_module_data_processor(self):
        """Test module data processor."""
        processor = population_module.get_data_processor()
        assert processor is not None

    def test_module_validator(self):
        """Test module validator."""
        validator = population_module.get_validator()
        assert validator is not None

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        functions = population_module.get_functions()
        assert len(functions) >= 7


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_statistics_single_step(self):
        """Test statistics with single time step."""
        df = pd.DataFrame({
            'step': [0],
            'total_agents': [100],
        })
        
        stats = compute_population_statistics(df)
        
        assert stats['peak_value'] == 100
        assert stats['final_value'] == 100

    def test_compute_statistics_declining_population(self):
        """Test statistics with declining population."""
        df = pd.DataFrame({
            'step': range(20),
            'total_agents': [100 - i * 2 for i in range(20)],
        })
        
        stats = compute_population_statistics(df)
        
        # Should detect negative trend
        assert stats['trend'] < 0

    def test_compute_stability_with_nan(self):
        """Test stability with NaN values."""
        df = pd.DataFrame({
            'step': range(20),
            'total_agents': [100 if i % 3 != 0 else np.nan for i in range(20)],
        })
        
        # Should handle NaN gracefully or raise appropriate error
        try:
            stability = compute_population_stability(df.fillna(method='ffill'))
            assert isinstance(stability, dict)
        except Exception:
            pass  # Expected for problematic data

    def test_compute_birth_death_with_zeros(self):
        """Test birth/death rates with zero values."""
        df = pd.DataFrame({
            'step': range(10),
            'births': [0] * 10,
            'deaths': [0] * 10,
        })
        
        rates = compute_birth_death_rates(df)
        
        assert rates['total_births'] == 0
        assert rates['total_deaths'] == 0
        assert rates['net_growth'] == 0

    def test_compute_growth_analysis_negative_growth(self):
        """Test growth analysis with population decline."""
        df = pd.DataFrame({
            'step': range(50),
            'total_agents': [100 - i for i in range(50)],
        })
        
        analysis = compute_growth_rate_analysis(df)
        
        # Should detect declining trend
        assert analysis['time_in_decline'] > 0
        assert analysis['average_growth_rate'] < 0

    def test_compute_demographic_single_type(self):
        """Test demographic metrics with single agent type."""
        df = pd.DataFrame({
            'step': range(20),
            'total_agents': [100 + i for i in range(20)],
            'system_agents': [100 + i for i in range(20)],
            'independent_agents': [0] * 20,
            'control_agents': [0] * 20,
        })
        
        metrics = compute_demographic_metrics(df)
        
        # Single type should have low diversity
        if 'diversity_index' in metrics:
            assert metrics['diversity_index']['mean'] < 0.5

    def test_analyze_composition_division_by_zero(self, tmp_path):
        """Test composition analysis with zero total agents."""
        df = pd.DataFrame({
            'step': range(10),
            'total_agents': [0] * 5 + [100 + i for i in range(5)],
            'system_agents': [0] * 5 + [50 + i for i in range(5)],
            'independent_agents': [0] * 5 + [30 + i // 2 for i in range(5)],
            'control_agents': [0] * 5 + [20 + i // 3 for i in range(5)],
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        
        # Should handle zero division
        with patch('pandas.DataFrame.to_csv'):  # Mock to avoid actual file operations
            analyze_agent_composition(df, ctx)

    def test_plot_population_extreme_values(self, tmp_path):
        """Test plotting with extreme population values."""
        df = pd.DataFrame({
            'step': range(20),
            'total_agents': [10000 * (1 + i) for i in range(20)],
        })
        
        ctx = AnalysisContext(output_path=tmp_path)
        plot_population_over_time(df, ctx)
        
        plot_file = tmp_path / "population_over_time.png"
        assert plot_file.exists()

    def test_stability_oscillating_population(self):
        """Test stability with oscillating population."""
        df = pd.DataFrame({
            'step': range(50),
            'total_agents': [100 + 50 * np.sin(i * 0.5) for i in range(50)],
        })
        
        stability = compute_population_stability(df)
        
        # Oscillating population should have lower stability
        assert stability['volatility'] > 0
        assert stability['stability_score'] < 1.0

    def test_growth_analysis_very_short_data(self):
        """Test growth analysis with very short time series."""
        df = pd.DataFrame({
            'step': range(3),
            'total_agents': [100, 105, 110],
        })
        
        analysis = compute_growth_rate_analysis(df)
        
        # Should handle short data
        assert isinstance(analysis, dict)

    def test_demographic_all_equal_types(self):
        """Test demographic metrics with equal distribution."""
        df = pd.DataFrame({
            'step': range(30),
            'total_agents': [300] * 30,
            'system_agents': [100] * 30,
            'independent_agents': [100] * 30,
            'control_agents': [100] * 30,
        })
        
        metrics = compute_demographic_metrics(df)
        
        # Equal distribution should have high diversity
        if 'diversity_index' in metrics:
            assert metrics['diversity_index']['mean'] > 0.8


class TestPopulationHelperFunctions:
    """Test helper functions in population module."""

    def test_process_population_data(self, tmp_path):
        """Test processing population data from experiment."""
        from farm.analysis.population.data import process_population_data
        
        exp_path = tmp_path / "experiment"
        exp_path.mkdir()
        
        # Create mock database
        db_path = exp_path / "simulation.db"
        db_path.touch()
        
        with patch('farm.analysis.population.data.SessionManager') as mock_sm:
            mock_session = MagicMock()
            mock_sm.return_value.__enter__.return_value = mock_session
            
            # Mock repository
            mock_repo = MagicMock()
            mock_repo.get_population_over_time.return_value = []
            
            with patch('farm.analysis.population.data.PopulationRepository', return_value=mock_repo):
                result = process_population_data(exp_path)
                
                assert isinstance(result, pd.DataFrame)

    def test_calculate_trend_increasing(self):
        """Test trend calculation with increasing population."""
        from farm.analysis.common.utils import calculate_trend
        
        pop = np.array([100, 110, 120, 130, 140, 150])
        trend = calculate_trend(pop)
        
        assert trend > 0

    def test_calculate_trend_decreasing(self):
        """Test trend calculation with decreasing population."""
        from farm.analysis.common.utils import calculate_trend
        
        pop = np.array([150, 140, 130, 120, 110, 100])
        trend = calculate_trend(pop)
        
        assert trend < 0

    def test_calculate_statistics_helper(self):
        """Test statistics calculation helper."""
        from farm.analysis.common.utils import calculate_statistics
        
        data = np.array([100, 110, 120, 130, 140])
        stats = calculate_statistics(data)
        
        assert 'mean' in stats
        assert 'median' in stats
        assert stats['mean'] == 120.0

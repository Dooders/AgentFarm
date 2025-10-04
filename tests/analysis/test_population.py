"""
Tests for population analysis module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json

from farm.analysis.population import (
    population_module,
    compute_population_statistics,
    compute_birth_death_rates,
    compute_population_stability,
    analyze_population_dynamics,
    analyze_agent_composition,
    plot_population_over_time,
    plot_birth_death_rates,
    plot_agent_composition,
)
from farm.analysis.common.context import AnalysisContext


class TestPopulationComputations:
    """Test population statistical computations."""

    def test_compute_population_statistics(self):
        """Test population statistics computation."""
        df = pd.DataFrame({
            'step': range(10),
            'total_agents': [100, 105, 110, 108, 115, 120, 118, 125, 130, 135],
            'system_agents': [50, 52, 55, 53, 58, 60, 59, 62, 65, 68],
            'independent_agents': [30, 32, 35, 33, 38, 40, 39, 42, 45, 48],
            'control_agents': [20, 21, 20, 22, 19, 20, 20, 21, 20, 19],
        })

        stats = compute_population_statistics(df)

        assert 'total' in stats
        assert 'peak_step' in stats
        assert 'peak_value' in stats
        assert stats['peak_value'] == 135
        assert stats['peak_step'] == 9
        assert 'trend' in stats
        assert isinstance(stats['trend'], float)

        # Check per-type statistics
        assert 'system_agents' in stats
        assert 'independent_agents' in stats
        assert 'control_agents' in stats

    def test_compute_birth_death_rates(self):
        """Test birth and death rate computation."""
        df = pd.DataFrame({
            'step': range(10),
            'births': [5, 3, 7, 2, 8, 4, 6, 1, 9, 3],
            'deaths': [2, 1, 3, 1, 4, 2, 2, 0, 5, 1],
        })

        rates = compute_birth_death_rates(df)

        assert 'total_births' in rates
        assert 'total_deaths' in rates
        assert 'birth_rate' in rates
        assert 'death_rate' in rates
        assert 'net_growth' in rates
        assert 'growth_rate' in rates

        assert rates['total_births'] == 48  # sum of births
        assert rates['total_deaths'] == 21  # sum of deaths
        assert rates['net_growth'] == 27   # births - deaths
        assert rates['birth_rate'] == 4.8   # births per step
        assert rates['death_rate'] == 2.1   # deaths per step

    def test_compute_birth_death_rates_no_data(self):
        """Test birth/death rates with missing data."""
        df = pd.DataFrame({
            'step': range(5),
            'total_agents': [100, 105, 110, 108, 112],
        })

        rates = compute_birth_death_rates(df)
        assert rates == {}

    def test_compute_population_stability(self):
        """Test population stability computation."""
        # Create data with some variation
        df = pd.DataFrame({
            'step': range(50),
            'total_agents': [100 + 5 * np.sin(i/5) + np.random.normal(0, 2) for i in range(50)],
        })

        stability = compute_population_stability(df)

        assert 'mean_cv' in stability
        assert 'stability_score' in stability
        assert 0 <= stability['stability_score'] <= 1
        assert stability['mean_cv'] >= 0

    def test_compute_population_stability_short_window(self):
        """Test stability with very short data."""
        df = pd.DataFrame({
            'step': range(3),
            'total_agents': [100, 105, 110],
        })

        stability = compute_population_stability(df, window=10)  # Window larger than data

        # Should handle gracefully
        assert 'mean_cv' in stability
        assert 'stability_score' in stability


class TestPopulationAnalysis:
    """Test population analysis functions."""

    def test_analyze_population_dynamics(self, tmp_path):
        """Test population dynamics analysis."""
        df = pd.DataFrame({
            'step': range(10),
            'total_agents': range(100, 110),
            'system_agents': range(50, 60),
            'independent_agents': range(30, 40),
            'control_agents': range(20, 30),
            'births': [5] * 10,
            'deaths': [2] * 10,
        })

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_population_dynamics(df, ctx)

        # Check output files
        stats_file = tmp_path / "population_statistics.json"
        assert stats_file.exists()

        with open(stats_file) as f:
            data = json.load(f)

        assert 'statistics' in data
        assert 'rates' in data
        assert 'stability' in data

        # Verify rates data
        assert data['rates']['total_births'] == 50  # 5 * 10
        assert data['rates']['total_deaths'] == 20  # 2 * 10

    def test_analyze_agent_composition(self, tmp_path):
        """Test agent composition analysis."""
        df = pd.DataFrame({
            'step': range(5),
            'total_agents': [100, 105, 110, 108, 112],
            'system_agents': [50, 52, 55, 53, 56],
            'independent_agents': [30, 32, 35, 33, 36],
            'control_agents': [20, 21, 20, 22, 20],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        analyze_agent_composition(df, ctx)

        # Check output file
        comp_file = tmp_path / "agent_composition.csv"
        assert comp_file.exists()

        comp_df = pd.read_csv(comp_file)

        # Should have original columns plus percentage columns
        expected_cols = [
            'step', 'total_agents', 'system_agents', 'independent_agents', 'control_agents',
            'system_agents_pct', 'independent_agents_pct', 'control_agents_pct'
        ]

        for col in expected_cols:
            assert col in comp_df.columns

        # Check percentage calculations (should sum to 100% within tolerance)
        pct_cols = [col for col in comp_df.columns if col.endswith('_pct')]
        pct_sums = comp_df[pct_cols].sum(axis=1)
        assert all(abs(s - 100.0) < 0.1 for s in pct_sums)


class TestPopulationVisualization:
    """Test population visualization functions."""

    def test_plot_population_over_time(self, tmp_path):
        """Test population over time plotting."""
        df = pd.DataFrame({
            'step': range(10),
            'total_agents': range(100, 110),
            'system_agents': range(50, 60),
            'independent_agents': range(30, 40),
            'control_agents': range(20, 30),
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_population_over_time(df, ctx)

        # Check output file
        plot_file = tmp_path / "population_over_time.png"
        assert plot_file.exists()

    def test_plot_birth_death_rates(self, tmp_path):
        """Test birth/death rates plotting."""
        df = pd.DataFrame({
            'step': range(10),
            'births': [5, 3, 7, 2, 8, 4, 6, 1, 9, 3],
            'deaths': [2, 1, 3, 1, 4, 2, 2, 0, 5, 1],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_birth_death_rates(df, ctx)

        plot_file = tmp_path / "birth_death_rates.png"
        assert plot_file.exists()

    def test_plot_birth_death_rates_no_data(self, tmp_path):
        """Test birth/death plotting with missing data."""
        df = pd.DataFrame({
            'step': range(5),
            'total_agents': range(100, 105),
        })

        ctx = AnalysisContext(output_path=tmp_path)
        # Should not raise error, just skip plotting
        plot_birth_death_rates(df, ctx)

        # No plot file should be created
        plot_file = tmp_path / "birth_death_rates.png"
        assert not plot_file.exists()

    def test_plot_agent_composition(self, tmp_path):
        """Test agent composition plotting."""
        df = pd.DataFrame({
            'step': range(5),
            'system_agents': [50, 52, 55, 53, 56],
            'independent_agents': [30, 32, 35, 33, 36],
            'control_agents': [20, 21, 20, 22, 20],
        })

        ctx = AnalysisContext(output_path=tmp_path)
        plot_agent_composition(df, ctx)

        plot_file = tmp_path / "agent_composition.png"
        assert plot_file.exists()

    def test_plot_agent_composition_no_data(self, tmp_path):
        """Test composition plotting with no agent type data."""
        df = pd.DataFrame({
            'step': range(5),
            'total_agents': range(100, 105),
        })

        ctx = AnalysisContext(output_path=tmp_path)
        # Should handle gracefully
        plot_agent_composition(df, ctx)

        plot_file = tmp_path / "agent_composition.png"
        assert not plot_file.exists()


class TestPopulationModule:
    """Test population module integration."""

    def test_population_module_registration(self):
        """Test module is properly registered."""
        assert population_module.name == "population"
        assert len(population_module.get_function_names()) > 0
        assert "analyze_dynamics" in population_module.get_function_names()
        assert "plot_population" in population_module.get_function_names()

    def test_population_module_function_groups(self):
        """Test module function groups."""
        groups = population_module.get_function_groups()
        assert "all" in groups
        assert "analysis" in groups
        assert "plots" in groups
        assert "basic" in groups

    def test_population_module_data_processor(self):
        """Test module data processor."""
        processor = population_module.get_data_processor()
        assert processor is not None

        # Test with mock path and database - use patch to mock database access
        import tempfile
        from unittest.mock import patch, MagicMock

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "simulation.db"
            temp_path.touch()  # Create empty file

            # Mock the database and repository
            mock_db = MagicMock()
            mock_session_manager = MagicMock()
            mock_db.session_manager = mock_session_manager

            mock_repository = MagicMock()
            mock_population_data = [
                MagicMock(step=i, total_agents=100+i, system_agents=50+i//2,
                         independent_agents=30+i//3, control_agents=20+i//4,
                         avg_resources=50.0) for i in range(5)
            ]
            mock_repository.get_population_over_time.return_value = mock_population_data

            with patch('farm.analysis.population.data.SimulationDatabase', return_value=mock_db), \
                 patch('farm.analysis.population.data.PopulationRepository', return_value=mock_repository):

                result = processor.process(Path(temp_dir))
                assert isinstance(result, pd.DataFrame)
                assert len(result) == 5
                assert 'step' in result.columns
                assert 'total_agents' in result.columns

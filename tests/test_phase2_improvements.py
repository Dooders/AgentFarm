#!/usr/bin/env python3
"""
Comprehensive tests for Phase 2 improvements.

Tests all new statistical methods, time series analysis, ML validation,
effect size calculations, and reproducibility features.
"""

import unittest
import tempfile
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to the path to import the analysis module
import sys
sys.path.append(str(Path(__file__).parent.parent))

from analysis.simulation_analysis import SimulationAnalyzer


class TestPhase2Improvements(unittest.TestCase):
    """Test cases for Phase 2 improvements."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Create test database with comprehensive data
        self._create_comprehensive_test_database()
        
        # Initialize analyzer with reproducibility features
        self.analyzer = SimulationAnalyzer(self.temp_db.name, random_seed=42)
    
    def tearDown(self):
        """Clean up test fixtures."""
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def _create_comprehensive_test_database(self):
        """Create a comprehensive test database for Phase 2 testing."""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_step_models (
                id INTEGER PRIMARY KEY,
                simulation_id INTEGER,
                step_number INTEGER,
                system_agents INTEGER,
                independent_agents INTEGER,
                control_agents INTEGER,
                total_agents INTEGER,
                resource_efficiency REAL,
                average_agent_health REAL,
                average_reward REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS action_models (
                id INTEGER PRIMARY KEY,
                simulation_id INTEGER,
                agent_id INTEGER,
                action_target_id INTEGER,
                action_type TEXT,
                step_number INTEGER,
                reward REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_models (
                id INTEGER PRIMARY KEY,
                agent_id INTEGER,
                agent_type TEXT,
                simulation_id INTEGER,
                birth_time INTEGER,
                death_time INTEGER,
                generation INTEGER
            )
        ''')
        
        # Insert comprehensive test data
        test_simulation_id = 1
        
        # Create time series with trends, seasonality, and noise
        np.random.seed(42)  # For reproducible test data
        
        for step in range(1, 201):  # 200 steps for time series analysis
            # Create trending data with seasonality
            trend = 0.1 * step
            seasonal = 10 * np.sin(2 * np.pi * step / 20)  # 20-step cycle
            noise = np.random.normal(0, 2)
            
            system_agents = max(0, int(50 + trend + seasonal + noise))
            independent_agents = max(0, int(30 + 0.05 * step + 5 * np.cos(2 * np.pi * step / 25) + np.random.normal(0, 1.5)))
            control_agents = max(0, int(20 + 0.02 * step + 3 * np.sin(2 * np.pi * step / 30) + np.random.normal(0, 1)))
            
            cursor.execute('''
                INSERT INTO simulation_step_models 
                (simulation_id, step_number, system_agents, independent_agents, 
                 control_agents, total_agents, resource_efficiency, average_agent_health, average_reward)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (test_simulation_id, step, system_agents, independent_agents, control_agents,
                  system_agents + independent_agents + control_agents,
                  0.8 + 0.1 * np.sin(step * 0.1) + 0.05 * np.random.random(),
                  0.7 + 0.2 * np.cos(step * 0.15) + 0.1 * np.random.random(),
                  0.5 + 0.3 * np.sin(step * 0.2) + 0.15 * np.random.random()))
        
        # Insert test agents
        agent_types = ['system', 'independent', 'control']
        for i, agent_type in enumerate(agent_types):
            for j in range(20):  # More agents for better ML analysis
                agent_id = i * 20 + j + 1
                cursor.execute('''
                    INSERT INTO agent_models 
                    (agent_id, agent_type, simulation_id, birth_time, death_time, generation)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (agent_id, agent_type, test_simulation_id, 1, 
                      None if np.random.random() > 0.3 else np.random.randint(50, 200),
                      np.random.randint(1, 5)))
        
        # Insert test actions with patterns
        for step in range(1, 201, 3):  # Every 3rd step
            for _ in range(np.random.randint(1, 8)):
                attacker_id = np.random.randint(1, 61)
                target_id = np.random.randint(1, 61)
                action_type = 'attack' if np.random.random() > 0.4 else 'move'
                
                cursor.execute('''
                    INSERT INTO action_models 
                    (simulation_id, agent_id, action_target_id, action_type, step_number, reward)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (test_simulation_id, attacker_id, target_id, action_type, step,
                      0.1 + 0.4 * np.random.random()))
        
        conn.commit()
        conn.close()
    
    def test_time_series_analysis(self):
        """Test comprehensive time series analysis."""
        result = self.analyzer.analyze_temporal_patterns(1)
        
        # Check return structure
        self.assertIn('time_series_analysis', result)
        self.assertIn('cross_correlations', result)
        self.assertIn('metadata', result)
        
        # Check time series analysis
        ts_analysis = result['time_series_analysis']
        self.assertIsInstance(ts_analysis, dict)
        self.assertGreater(len(ts_analysis), 0)
        
        # Check that each time series has required components
        for series_name, analysis in ts_analysis.items():
            self.assertIn('stationarity', analysis)
            self.assertIn('trend', analysis)
            self.assertIn('seasonality', analysis)
            self.assertIn('change_points', analysis)
            self.assertIn('autocorrelation', analysis)
            self.assertIn('summary', analysis)
            
            # Check stationarity tests
            stationarity = analysis['stationarity']
            if 'adf_test' in stationarity and 'error' not in stationarity['adf_test']:
                self.assertIn('statistic', stationarity['adf_test'])
                self.assertIn('p_value', stationarity['adf_test'])
                self.assertIn('is_stationary', stationarity['adf_test'])
            
            # Check trend analysis
            trend = analysis['trend']
            if 'linear_trend' in trend:
                self.assertIn('slope', trend['linear_trend'])
                self.assertIn('r_squared', trend['linear_trend'])
                self.assertIn('p_value', trend['linear_trend'])
                self.assertIn('significant_trend', trend['linear_trend'])
    
    def test_advanced_ml_analysis(self):
        """Test advanced machine learning analysis."""
        result = self.analyzer.analyze_with_advanced_ml(1)
        
        # Check return structure
        self.assertIn('feature_selection', result)
        self.assertIn('individual_models', result)
        self.assertIn('ensemble_models', result)
        self.assertIn('hyperparameter_tuning', result)
        self.assertIn('performance_comparison', result)
        self.assertIn('best_model', result)
        
        # Check feature selection
        feature_selection = result['feature_selection']
        self.assertIn('univariate', feature_selection)
        self.assertIn('rfe', feature_selection)
        self.assertIn('model_based', feature_selection)
        
        # Check individual models
        individual_models = result['individual_models']
        expected_models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'SVM', 'Decision Tree']
        for model_name in expected_models:
            if model_name in individual_models and 'error' not in individual_models[model_name]:
                model_result = individual_models[model_name]
                self.assertIn('test_accuracy', model_result)
                self.assertIn('cv_mean', model_result)
                self.assertIn('cv_std', model_result)
                self.assertIn('cv_scores', model_result)
        
        # Check ensemble models
        ensemble_models = result['ensemble_models']
        self.assertIn('voting', ensemble_models)
        self.assertIn('bagging', ensemble_models)
        
        # Check performance comparison
        performance_comparison = result['performance_comparison']
        self.assertIsInstance(performance_comparison, dict)
        self.assertGreater(len(performance_comparison), 0)
    
    def test_effect_size_calculations(self):
        """Test effect size calculations in population dynamics."""
        result = self.analyzer.analyze_population_dynamics(1)
        
        # Check that effect sizes are calculated
        statistical_analysis = result['statistical_analysis']
        pairwise_comparisons = statistical_analysis['pairwise_comparisons']
        
        for comparison, comparison_result in pairwise_comparisons.items():
            if 'error' not in comparison_result:
                self.assertIn('effect_sizes', comparison_result)
                self.assertIn('power_analysis', comparison_result)
                self.assertIn('sample_sizes', comparison_result)
                self.assertIn('descriptive_stats', comparison_result)
                
                # Check effect sizes
                effect_sizes = comparison_result['effect_sizes']
                if 'error' not in effect_sizes:
                    self.assertIn('cohens_d', effect_sizes)
                    self.assertIn('cohens_d_interpretation', effect_sizes)
                    self.assertIn('hedges_g', effect_sizes)
                    self.assertIn('eta_squared', effect_sizes)
                    self.assertIn('eta_squared_interpretation', effect_sizes)
                
                # Check power analysis
                power_analysis = comparison_result['power_analysis']
                if 'error' not in power_analysis:
                    self.assertIn('observed_power', power_analysis)
                    self.assertIn('power_interpretation', power_analysis)
                    self.assertIn('effect_size', power_analysis)
                    self.assertIn('effect_size_interpretation', power_analysis)
    
    def test_reproducibility_features(self):
        """Test reproducibility features."""
        # Test that analyzer has reproducibility features
        self.assertIsNotNone(self.analyzer.random_seed)
        self.assertEqual(self.analyzer.random_seed, 42)
        
        # Test that reproducibility manager is available
        if hasattr(self.analyzer, 'repro_manager') and self.analyzer.repro_manager is not None:
            # Test environment info capture
            env_info = self.analyzer.repro_manager.get_environment_info()
            self.assertIn('timestamp', env_info)
            self.assertIn('python_version', env_info)
            self.assertIn('platform', env_info)
            
            # Test analysis hash creation
            analysis_params = {"simulation_id": 1, "significance_level": 0.05}
            analysis_hash = self.analyzer.repro_manager.create_analysis_hash(analysis_params)
            self.assertIsInstance(analysis_hash, str)
            self.assertEqual(len(analysis_hash), 32)  # MD5 hash length
        
        # Test that validator is available
        if hasattr(self.analyzer, 'validator') and self.analyzer.validator is not None:
            # Test validation of a simple result
            test_result = {
                "dataframe": pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]}),
                "statistical_analysis": {"confidence_intervals": {}},
                "summary": {"total_steps": 3}
            }
            
            validation_result = self.analyzer.validator.validate_analysis_result(
                "population_dynamics", test_result
            )
            
            self.assertIn('analysis_type', validation_result)
            self.assertIn('valid', validation_result)
            self.assertIn('errors', validation_result)
            self.assertIn('warnings', validation_result)
    
    def test_complete_analysis_with_phase2(self):
        """Test complete analysis with all Phase 2 improvements."""
        result = self.analyzer.run_complete_analysis(1, significance_level=0.05)
        
        # Check that all Phase 2 analyses are included
        self.assertIn('temporal_patterns', result)
        self.assertIn('advanced_ml', result)
        
        # Check that validation report is included
        if hasattr(self.analyzer, 'validator') and self.analyzer.validator is not None:
            self.assertIn('validation_report', result)
            
            validation_report = result['validation_report']
            self.assertIn('overall_valid', validation_report)
            self.assertIn('analysis_validations', validation_report)
            self.assertIn('summary', validation_report)
        
        # Check metadata includes Phase 2 methods
        metadata = result['metadata']
        self.assertIn('analysis_version', metadata)
        self.assertEqual(metadata['analysis_version'], 'Phase 2 - Statistical Enhancement')
        
        methods_used = metadata['statistical_methods_used']
        phase2_methods = [
            "Effect size calculations (Cohen's d, Hedges' g, eta-squared)",
            "Statistical power analysis",
            "Time series analysis (ADF, KPSS, seasonal decomposition)",
            "Advanced ML (ensemble methods, feature selection)",
            "Cross-validation and hyperparameter tuning"
        ]
        
        for method in phase2_methods:
            self.assertIn(method, methods_used)
    
    def test_statistical_validation_enhancements(self):
        """Test enhanced statistical validation."""
        # Test population dynamics with enhanced statistics
        result = self.analyzer.analyze_population_dynamics(1)
        
        statistical_analysis = result['statistical_analysis']
        
        # Check that enhanced statistics are present
        self.assertIn('kruskal_wallis', statistical_analysis)
        self.assertIn('pairwise_comparisons', statistical_analysis)
        self.assertIn('confidence_intervals', statistical_analysis)
        
        # Check Kruskal-Wallis test
        kruskal_test = statistical_analysis['kruskal_wallis']
        if 'error' not in kruskal_test:
            self.assertIn('h_statistic', kruskal_test)
            self.assertIn('p_value', kruskal_test)
            self.assertIn('significant_difference', kruskal_test)
        
        # Check pairwise comparisons have effect sizes
        pairwise = statistical_analysis['pairwise_comparisons']
        for comparison, comparison_result in pairwise.items():
            if 'error' not in comparison_result:
                self.assertIn('effect_sizes', comparison_result)
                self.assertIn('power_analysis', comparison_result)
    
    def test_visualization_enhancements(self):
        """Test that enhanced visualizations are created."""
        # Mock matplotlib to avoid actual file creation
        with patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.close') as mock_close:
            
            # Run analyses that create visualizations
            self.analyzer.analyze_population_dynamics(1)
            self.analyzer.analyze_temporal_patterns(1)
            self.analyzer.analyze_with_advanced_ml(1)
            
            # Check that savefig was called (indicating plots were created)
            self.assertGreater(mock_savefig.call_count, 0)
    
    def test_error_handling_robustness(self):
        """Test that Phase 2 improvements handle errors gracefully."""
        # Test with insufficient data for time series analysis
        small_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        small_db.close()
        
        conn = sqlite3.connect(small_db.name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_step_models (
                id INTEGER PRIMARY KEY,
                simulation_id INTEGER,
                step_number INTEGER,
                system_agents INTEGER
            )
        ''')
        
        # Insert only 5 steps (insufficient for time series analysis)
        for step in range(1, 6):
            cursor.execute('''
                INSERT INTO simulation_step_models 
                (simulation_id, step_number, system_agents)
                VALUES (?, ?, ?)
            ''', (1, step, 10))
        
        conn.commit()
        conn.close()
        
        analyzer = SimulationAnalyzer(small_db.name, random_seed=42)
        
        # Test time series analysis with insufficient data
        result = analyzer.analyze_temporal_patterns(1)
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Insufficient data for time series analysis')
        
        # Test ML analysis with insufficient data
        result = analyzer.analyze_with_advanced_ml(1)
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Insufficient data for ML analysis')
        
        Path(small_db.name).unlink()
    
    def test_reproducibility_consistency(self):
        """Test that results are reproducible with same seed."""
        if hasattr(self.analyzer, 'repro_manager') and self.analyzer.repro_manager is not None:
            # Run analysis twice with same parameters
            result1 = self.analyzer.analyze_population_dynamics(1)
            result2 = self.analyzer.analyze_population_dynamics(1)
            
            # Results should be identical (or very close for floating point)
            self._compare_analysis_results(result1, result2)
    
    def _compare_analysis_results(self, result1, result2, tolerance=1e-10):
        """Compare two analysis results for consistency."""
        def compare_values(val1, val2, path=""):
            if type(val1) != type(val2):
                self.fail(f"Type mismatch at {path}: {type(val1)} vs {type(val2)}")
            
            if isinstance(val1, dict):
                keys1, keys2 = set(val1.keys()), set(val2.keys())
                self.assertEqual(keys1, keys2, f"Key mismatch at {path}")
                
                for key in keys1:
                    compare_values(val1[key], val2[key], f"{path}.{key}" if path else key)
            
            elif isinstance(val1, (list, tuple)):
                self.assertEqual(len(val1), len(val2), f"Length mismatch at {path}")
                for i, (v1, v2) in enumerate(zip(val1, val2)):
                    compare_values(v1, v2, f"{path}[{i}]")
            
            elif isinstance(val1, (int, float)):
                if not np.isclose(val1, val2, rtol=tolerance, atol=tolerance):
                    self.fail(f"Value mismatch at {path}: {val1} vs {val2}")
            
            elif isinstance(val1, str):
                self.assertEqual(val1, val2, f"String mismatch at {path}")
            
            else:
                self.assertEqual(val1, val2, f"Value mismatch at {path}")
        
        compare_values(result1, result2)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
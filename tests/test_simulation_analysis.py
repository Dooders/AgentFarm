#!/usr/bin/env python3
"""
Unit tests for simulation_analysis.py

Tests the core analysis functions with proper statistical validation.
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


class TestSimulationAnalyzer(unittest.TestCase):
    """Test cases for SimulationAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        
        # Create test database with sample data
        self._create_test_database()
        
        # Initialize analyzer
        self.analyzer = SimulationAnalyzer(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def _create_test_database(self):
        """Create a test database with sample simulation data."""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        # Create tables (simplified schema for testing)
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
        
        # Insert test data
        test_simulation_id = 1
        
        # Insert simulation steps with varying populations
        for step in range(1, 101):
            system_agents = max(0, 50 + int(10 * np.sin(step * 0.1)) + np.random.randint(-5, 5))
            independent_agents = max(0, 30 + int(5 * np.cos(step * 0.15)) + np.random.randint(-3, 3))
            control_agents = max(0, 20 + int(3 * np.sin(step * 0.2)) + np.random.randint(-2, 2))
            
            cursor.execute('''
                INSERT INTO simulation_step_models 
                (simulation_id, step_number, system_agents, independent_agents, 
                 control_agents, total_agents, resource_efficiency, average_agent_health, average_reward)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (test_simulation_id, step, system_agents, independent_agents, control_agents,
                  system_agents + independent_agents + control_agents,
                  0.8 + 0.1 * np.random.random(), 0.7 + 0.2 * np.random.random(),
                  0.5 + 0.3 * np.random.random()))
        
        # Insert test agents
        agent_types = ['system', 'independent', 'control']
        for i, agent_type in enumerate(agent_types):
            for j in range(10):
                agent_id = i * 10 + j + 1
                cursor.execute('''
                    INSERT INTO agent_models 
                    (agent_id, agent_type, simulation_id, birth_time, death_time, generation)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (agent_id, agent_type, test_simulation_id, 1, 
                      None if np.random.random() > 0.3 else np.random.randint(50, 100),
                      np.random.randint(1, 5)))
        
        # Insert test actions
        for step in range(1, 101, 5):  # Every 5th step
            for _ in range(np.random.randint(1, 5)):
                attacker_id = np.random.randint(1, 31)
                target_id = np.random.randint(1, 31)
                action_type = 'attack' if np.random.random() > 0.5 else 'move'
                
                cursor.execute('''
                    INSERT INTO action_models 
                    (simulation_id, agent_id, action_target_id, action_type, step_number, reward)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (test_simulation_id, attacker_id, target_id, action_type, step,
                      0.1 + 0.4 * np.random.random()))
        
        conn.commit()
        conn.close()
    
    def test_analyzer_initialization(self):
        """Test that analyzer initializes correctly."""
        self.assertIsNotNone(self.analyzer.engine)
        self.assertIsNotNone(self.analyzer.session)
    
    def test_analyze_population_dynamics_basic(self):
        """Test basic population dynamics analysis."""
        result = self.analyzer.analyze_population_dynamics(1)
        
        # Check return structure
        self.assertIn('dataframe', result)
        self.assertIn('statistical_analysis', result)
        self.assertIn('summary', result)
        
        # Check dataframe
        df = result['dataframe']
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn('step', df.columns)
        self.assertIn('system_agents', df.columns)
        
        # Check statistical analysis
        stats = result['statistical_analysis']
        self.assertIn('confidence_intervals', stats)
        
        # Check summary
        summary = result['summary']
        self.assertIn('total_steps', summary)
        self.assertIn('significant_differences', summary)
    
    def test_analyze_population_dynamics_insufficient_data(self):
        """Test population dynamics with insufficient data."""
        # Create empty database
        empty_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        empty_db.close()
        
        conn = sqlite3.connect(empty_db.name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_step_models (
                id INTEGER PRIMARY KEY,
                simulation_id INTEGER,
                step_number INTEGER,
                system_agents INTEGER
            )
        ''')
        conn.commit()
        conn.close()
        
        analyzer = SimulationAnalyzer(empty_db.name)
        result = analyzer.analyze_population_dynamics(1)
        
        self.assertIn('error', result)
        self.assertEqual(result['error'], 'Insufficient data')
        
        Path(empty_db.name).unlink()
    
    def test_identify_critical_events_statistical(self):
        """Test critical event identification with statistical methods."""
        result = self.analyzer.identify_critical_events(1, significance_level=0.05)
        
        # Check return structure
        self.assertIsInstance(result, list)
        
        # If events are found, check their structure
        if result:
            event = result[0]
            self.assertIn('step', event)
            self.assertIn('agent_type', event)
            self.assertIn('change_rate', event)
            self.assertIn('z_score', event)
            self.assertIn('p_value', event)
            self.assertIn('is_significant', event)
    
    def test_identify_critical_events_insufficient_data(self):
        """Test critical event identification with insufficient data."""
        # Create database with only 5 steps
        small_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        small_db.close()
        
        conn = sqlite3.connect(small_db.name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS simulation_step_models (
                id INTEGER PRIMARY KEY,
                simulation_id INTEGER,
                step_number INTEGER,
                system_agents INTEGER,
                independent_agents INTEGER,
                control_agents INTEGER
            )
        ''')
        
        for step in range(1, 6):
            cursor.execute('''
                INSERT INTO simulation_step_models 
                (simulation_id, step_number, system_agents, independent_agents, control_agents)
                VALUES (?, ?, ?, ?, ?)
            ''', (1, step, 10, 5, 3))
        
        conn.commit()
        conn.close()
        
        analyzer = SimulationAnalyzer(small_db.name)
        result = analyzer.identify_critical_events(1)
        
        self.assertEqual(result, [])
        
        Path(small_db.name).unlink()
    
    def test_analyze_agent_interactions_basic(self):
        """Test basic agent interactions analysis."""
        result = self.analyzer.analyze_agent_interactions(1)
        
        # Check return structure
        self.assertIn('interaction_patterns', result)
        self.assertIn('interaction_matrix', result)
        self.assertIn('statistical_analysis', result)
        self.assertIn('summary', result)
        
        # Check interaction matrix
        matrix = result['interaction_matrix']
        self.assertIsInstance(matrix, pd.DataFrame)
        
        # Check statistical analysis
        stats = result['statistical_analysis']
        if 'chi_square_test' in stats:
            chi2_test = stats['chi_square_test']
            self.assertIn('chi2_statistic', chi2_test)
            self.assertIn('p_value', chi2_test)
            self.assertIn('significant_association', chi2_test)
    
    def test_analyze_agent_interactions_no_data(self):
        """Test agent interactions with no attack data."""
        # Create database with no attack actions
        no_attacks_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        no_attacks_db.close()
        
        conn = sqlite3.connect(no_attacks_db.name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS action_models (
                id INTEGER PRIMARY KEY,
                simulation_id INTEGER,
                agent_id INTEGER,
                action_target_id INTEGER,
                action_type TEXT,
                step_number INTEGER
            )
        ''')
        
        # Insert only non-attack actions
        cursor.execute('''
            INSERT INTO action_models 
            (simulation_id, agent_id, action_target_id, action_type, step_number)
            VALUES (?, ?, ?, ?, ?)
        ''', (1, 1, 2, 'move', 1))
        
        conn.commit()
        conn.close()
        
        analyzer = SimulationAnalyzer(no_attacks_db.name)
        result = analyzer.analyze_agent_interactions(1)
        
        self.assertEqual(result['interaction_patterns'], {})
        self.assertIn('error', result['statistical_analysis'])
        
        Path(no_attacks_db.name).unlink()
    
    def test_run_complete_analysis(self):
        """Test complete analysis workflow."""
        result = self.analyzer.run_complete_analysis(1, significance_level=0.05)
        
        # Check return structure
        self.assertIn('simulation_id', result)
        self.assertIn('significance_level', result)
        self.assertIn('population_dynamics', result)
        self.assertIn('agent_interactions', result)
        self.assertIn('critical_events', result)
        self.assertIn('metadata', result)
        
        # Check metadata
        metadata = result['metadata']
        self.assertIn('analysis_timestamp', metadata)
        self.assertIn('statistical_methods_used', metadata)
        self.assertIn('significance_level', metadata)
    
    def test_run_complete_analysis_error_handling(self):
        """Test complete analysis with error handling."""
        # Test with non-existent simulation
        result = self.analyzer.run_complete_analysis(999, significance_level=0.05)
        
        # Should handle errors gracefully
        self.assertIn('simulation_id', result)
        self.assertEqual(result['simulation_id'], 999)
    
    def test_statistical_methods_validation(self):
        """Test that statistical methods are properly applied."""
        result = self.analyzer.analyze_population_dynamics(1)
        
        # Check that confidence intervals are calculated
        ci = result['statistical_analysis']['confidence_intervals']
        for agent_type in ['system_agents', 'independent_agents', 'control_agents']:
            if agent_type in ci:
                self.assertIn('mean', ci[agent_type])
                self.assertIn('ci_lower', ci[agent_type])
                self.assertIn('ci_upper', ci[agent_type])
                self.assertIn('sample_size', ci[agent_type])
                
                # Check that CI bounds are reasonable
                mean_val = ci[agent_type]['mean']
                ci_lower = ci[agent_type]['ci_lower']
                ci_upper = ci[agent_type]['ci_upper']
                
                self.assertLessEqual(ci_lower, mean_val)
                self.assertGreaterEqual(ci_upper, mean_val)
    
    def test_significance_level_parameter(self):
        """Test that significance level parameter is properly used."""
        # Test with different significance levels
        result_05 = self.analyzer.identify_critical_events(1, significance_level=0.05)
        result_01 = self.analyzer.identify_critical_events(1, significance_level=0.01)
        
        # More stringent significance level should generally find fewer events
        # (though this isn't guaranteed due to randomness)
        self.assertIsInstance(result_05, list)
        self.assertIsInstance(result_01, list)
        
        # Check that significance flags are set correctly
        for event in result_05:
            if 'is_significant' in event:
                if event['p_value'] < 0.05:
                    self.assertTrue(event['is_significant'])
                else:
                    self.assertFalse(event['is_significant'])
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plotting_functions_dont_crash(self, mock_close, mock_savefig):
        """Test that plotting functions don't crash the analysis."""
        # This test ensures that matplotlib issues don't break the analysis
        result = self.analyzer.analyze_population_dynamics(1)
        
        # Analysis should complete successfully even if plotting fails
        self.assertIn('dataframe', result)
        self.assertIn('statistical_analysis', result)
        
        # Verify that savefig was called (indicating plot was attempted)
        mock_savefig.assert_called()


class TestStatisticalValidation(unittest.TestCase):
    """Test statistical validation methods."""
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculation."""
        # Create test data
        data = np.random.normal(100, 15, 100)
        
        # Calculate confidence interval manually
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        n = len(data)
        
        from scipy import stats
        ci_lower, ci_upper = stats.t.interval(0.95, n-1, loc=mean_val, scale=std_val/np.sqrt(n))
        
        # Check that CI is reasonable
        self.assertLess(ci_lower, mean_val)
        self.assertGreater(ci_upper, mean_val)
        self.assertLess(ci_upper - ci_lower, 2 * std_val)  # CI should be narrower than 2*std
    
    def test_z_score_calculation(self):
        """Test z-score calculation for change detection."""
        # Create test time series with a clear change point
        data = np.concatenate([
            np.random.normal(100, 5, 50),  # Stable period
            np.random.normal(150, 5, 50)   # Change period
        ])
        
        # Calculate rolling statistics
        window_size = 10
        rolling_mean = pd.Series(data).rolling(window=window_size, min_periods=1).mean()
        rolling_std = pd.Series(data).rolling(window=window_size, min_periods=1).std()
        
        # Calculate z-scores
        z_scores = (data - rolling_mean) / rolling_std.replace(0, 1)
        
        # Check that z-scores are calculated correctly
        self.assertEqual(len(z_scores), len(data))
        
        # The change point should have high absolute z-score
        change_point_z_scores = z_scores[45:55]  # Around the change point
        self.assertTrue(np.any(np.abs(change_point_z_scores) > 2.0))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
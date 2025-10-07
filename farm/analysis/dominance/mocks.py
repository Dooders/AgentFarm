"""
Mock implementations for dominance analysis testing.

This module provides mock implementations of all dominance analysis protocols,
enabling easy unit testing without database dependencies.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from farm.analysis.dominance.interfaces import (
    DominanceAnalyzerProtocol,
    DominanceComputerProtocol,
    DominanceDataProviderProtocol,
)


class MockDominanceComputer:
    """
    Mock implementation of DominanceComputerProtocol for testing.
    
    Returns predictable values for all methods, making tests deterministic.
    """

    def __init__(self, analyzer: Optional[DominanceAnalyzerProtocol] = None):
        """Initialize with optional analyzer dependency."""
        self.analyzer = analyzer
        self.call_count = {}  # Track method calls for verification

    def _track_call(self, method_name: str):
        """Track method calls for testing."""
        self.call_count[method_name] = self.call_count.get(method_name, 0) + 1

    def compute_population_dominance(self, sim_session) -> Optional[str]:
        """Return mock population dominance."""
        self._track_call("compute_population_dominance")
        return "system"

    def compute_survival_dominance(self, sim_session) -> Optional[str]:
        """Return mock survival dominance."""
        self._track_call("compute_survival_dominance")
        return "independent"

    def compute_comprehensive_dominance(self, sim_session) -> Optional[Dict[str, Any]]:
        """Return mock comprehensive dominance."""
        self._track_call("compute_comprehensive_dominance")
        return {
            "dominant_type": "system",
            "scores": {
                "system": 0.6,
                "independent": 0.3,
                "control": 0.1
            },
            "metrics": {
                "auc": {"system": 1000, "independent": 500, "control": 200},
                "recency_weighted_auc": {"system": 1200, "independent": 600, "control": 250},
                "dominance_duration": {"system": 60, "independent": 30, "control": 10},
                "growth_trends": {"system": 0.1, "independent": -0.05, "control": -0.1},
                "final_ratios": {"system": 0.6, "independent": 0.3, "control": 0.1}
            },
            "normalized_metrics": {
                "auc": {"system": 0.59, "independent": 0.29, "control": 0.12},
                "recency_weighted_auc": {"system": 0.58, "independent": 0.29, "control": 0.12},
                "dominance_duration": {"system": 0.6, "independent": 0.3, "control": 0.1}
            }
        }

    def compute_dominance_switches(self, sim_session) -> Optional[Dict[str, Any]]:
        """Return mock dominance switches."""
        self._track_call("compute_dominance_switches")
        return {
            "total_switches": 5,
            "switches_per_step": 0.05,
            "switches_detail": [
                {"step": 10, "from": "system", "to": "independent", "phase": "early"},
                {"step": 25, "from": "independent", "to": "system", "phase": "early"},
                {"step": 40, "from": "system", "to": "control", "phase": "middle"},
                {"step": 60, "from": "control", "to": "system", "phase": "middle"},
                {"step": 85, "from": "system", "to": "independent", "phase": "late"},
            ],
            "avg_dominance_periods": {
                "system": 25.0,
                "independent": 15.0,
                "control": 10.0
            },
            "phase_switches": {
                "early": 2,
                "middle": 2,
                "late": 1
            },
            "transition_matrix": {
                "system": {"system": 0, "independent": 2, "control": 1},
                "independent": {"system": 2, "independent": 0, "control": 0},
                "control": {"system": 1, "independent": 0, "control": 0}
            },
            "transition_probabilities": {
                "system": {"system": 0.0, "independent": 0.67, "control": 0.33},
                "independent": {"system": 1.0, "independent": 0.0, "control": 0.0},
                "control": {"system": 1.0, "independent": 0.0, "control": 0.0}
            }
        }

    def compute_dominance_switch_factors(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Return mock switch factors."""
        self._track_call("compute_dominance_switch_factors")
        
        if df.empty or "total_switches" not in df.columns:
            return None
        
        return {
            "top_positive_correlations": {
                "initial_independent_count": 0.45,
                "resource_proximity": 0.32,
                "initial_resource_count": 0.28
            },
            "top_negative_correlations": {
                "initial_system_count": -0.38,
                "dominance_stability": -0.42,
                "system_dominance_score": -0.35
            },
            "switches_by_dominant_type": {
                "system": 3.5,
                "independent": 4.2,
                "control": 2.8
            },
            "reproduction_correlations": {
                "system_reproduction_success_rate": -0.25,
                "independent_reproduction_success_rate": 0.18,
                "control_reproduction_success_rate": 0.12
            }
        }

    def aggregate_reproduction_analysis_results(
        self, df: pd.DataFrame, numeric_repro_cols: List[str]
    ) -> Dict[str, Any]:
        """Return mock aggregated reproduction results."""
        self._track_call("aggregate_reproduction_analysis_results")
        
        if self.analyzer is None:
            return {}
        
        # Delegate to analyzer if available (real behavior)
        # For mock, return empty dict
        return {}


class MockDominanceAnalyzer:
    """
    Mock implementation of DominanceAnalyzerProtocol for testing.
    
    Returns DataFrames with mock analysis columns added.
    """

    def __init__(self, computer: Optional[DominanceComputerProtocol] = None):
        """Initialize with optional computer dependency."""
        self.computer = computer
        self.call_count = {}  # Track method calls for verification

    def _track_call(self, method_name: str):
        """Track method calls for testing."""
        self.call_count[method_name] = self.call_count.get(method_name, 0) + 1

    def analyze_by_agent_type(
        self, df: pd.DataFrame, numeric_repro_cols: List[str]
    ) -> pd.DataFrame:
        """Add mock agent type analysis columns."""
        self._track_call("analyze_by_agent_type")
        
        # Add mock columns
        df = df.copy()
        df["agent_type_analysis_done"] = True
        
        return df

    def analyze_high_vs_low_switching(
        self, df: pd.DataFrame, numeric_repro_cols: List[str]
    ) -> pd.DataFrame:
        """Add mock high vs low switching columns."""
        self._track_call("analyze_high_vs_low_switching")
        
        df = df.copy()
        for col in numeric_repro_cols:
            df[f"{col}_high_switching_mean"] = 0.75
            df[f"{col}_low_switching_mean"] = 0.45
            df[f"{col}_difference"] = 0.30
            df[f"{col}_percent_difference"] = 66.67
        
        return df

    def analyze_reproduction_advantage(
        self, df: pd.DataFrame, numeric_repro_cols: List[str]
    ) -> pd.DataFrame:
        """Add mock reproduction advantage columns."""
        self._track_call("analyze_reproduction_advantage")
        
        df = df.copy()
        if "switches_per_step" in df.columns:
            df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)
        
        return df

    def analyze_reproduction_efficiency(
        self, df: pd.DataFrame, numeric_repro_cols: List[str]
    ) -> pd.DataFrame:
        """Add mock reproduction efficiency columns."""
        self._track_call("analyze_reproduction_efficiency")
        
        df = df.copy()
        if "switches_per_step" in df.columns and "dominance_stability" not in df.columns:
            df["dominance_stability"] = 1 / (df["switches_per_step"] + 0.01)
        
        return df

    def analyze_reproduction_timing(
        self, df: pd.DataFrame, numeric_repro_cols: List[str]
    ) -> pd.DataFrame:
        """Add mock reproduction timing columns."""
        self._track_call("analyze_reproduction_timing")
        
        df = df.copy()
        df["reproduction_timing_analyzed"] = True
        
        return df

    def analyze_dominance_switch_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mock dominance switch factor columns."""
        self._track_call("analyze_dominance_switch_factors")
        
        df = df.copy()
        df["positive_corr_resource_proximity"] = 0.32
        df["negative_corr_dominance_stability"] = -0.42
        df["system_avg_switches"] = 3.5
        
        return df

    def analyze_reproduction_dominance_switching(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mock reproduction dominance switching columns."""
        self._track_call("analyze_reproduction_dominance_switching")
        
        df = df.copy()
        df["reproduction_switching_analyzed"] = True
        
        return df


class MockDominanceDataProvider:
    """
    Mock implementation of DominanceDataProviderProtocol for testing.
    
    Returns predictable test data without database access.
    """

    def __init__(self):
        """Initialize mock data provider."""
        self.call_count = {}  # Track method calls for verification

    def _track_call(self, method_name: str):
        """Track method calls for testing."""
        self.call_count[method_name] = self.call_count.get(method_name, 0) + 1

    def get_final_population_counts(self, sim_session) -> Optional[Dict[str, int]]:
        """Return mock final population counts."""
        self._track_call("get_final_population_counts")
        return {
            "system_agents": 15,
            "independent_agents": 8,
            "control_agents": 3,
            "total_agents": 26,
            "final_step": 100
        }

    def get_agent_survival_stats(self, sim_session) -> Optional[Dict[str, Any]]:
        """Return mock survival statistics."""
        self._track_call("get_agent_survival_stats")
        return {
            "system_count": 20,
            "system_alive": 15,
            "system_dead": 5,
            "system_avg_survival": 75.5,
            "system_dead_ratio": 0.25,
            "independent_count": 15,
            "independent_alive": 8,
            "independent_dead": 7,
            "independent_avg_survival": 60.2,
            "independent_dead_ratio": 0.47,
            "control_count": 10,
            "control_alive": 3,
            "control_dead": 7,
            "control_avg_survival": 45.8,
            "control_dead_ratio": 0.70
        }

    def get_reproduction_stats(self, sim_session) -> Optional[Dict[str, Any]]:
        """Return mock reproduction statistics."""
        self._track_call("get_reproduction_stats")
        return {
            "system_reproduction_attempts": 25,
            "system_reproduction_successes": 20,
            "system_reproduction_failures": 5,
            "system_reproduction_success_rate": 0.8,
            "system_first_reproduction_time": 15,
            "independent_reproduction_attempts": 18,
            "independent_reproduction_successes": 12,
            "independent_reproduction_failures": 6,
            "independent_reproduction_success_rate": 0.67,
            "independent_first_reproduction_time": 20,
            "control_reproduction_attempts": 10,
            "control_reproduction_successes": 5,
            "control_reproduction_failures": 5,
            "control_reproduction_success_rate": 0.5,
            "control_first_reproduction_time": 25
        }

    def get_initial_positions_and_resources(
        self, sim_session, config: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Return mock initial positions and resources."""
        self._track_call("get_initial_positions_and_resources")
        return {
            "initial_system_count": 5,
            "initial_independent_count": 5,
            "initial_control_count": 5,
            "initial_resource_count": 10,
            "initial_resource_amount": 500.0,
            "system_nearest_resource_dist": 15.5,
            "independent_nearest_resource_dist": 18.2,
            "control_nearest_resource_dist": 22.1
        }


# ============================================================================
# Convenience Functions for Creating Mocks
# ============================================================================

def create_mock_computer(
    population_dominance: str = "system",
    survival_dominance: str = "independent",
    comprehensive_dominance: Optional[Dict[str, Any]] = None
) -> MockDominanceComputer:
    """
    Create a mock computer with customizable return values.
    
    Parameters
    ----------
    population_dominance : str
        Value to return from compute_population_dominance
    survival_dominance : str
        Value to return from compute_survival_dominance
    comprehensive_dominance : Optional[Dict[str, Any]]
        Value to return from compute_comprehensive_dominance
        
    Returns
    -------
    MockDominanceComputer
        Configured mock computer instance
    """
    mock = MockDominanceComputer()
    
    # Override return values
    original_pop = mock.compute_population_dominance
    mock.compute_population_dominance = lambda session: (
        mock._track_call("compute_population_dominance") or population_dominance
    )
    
    original_surv = mock.compute_survival_dominance
    mock.compute_survival_dominance = lambda session: (
        mock._track_call("compute_survival_dominance") or survival_dominance
    )
    
    if comprehensive_dominance is not None:
        original_comp = mock.compute_comprehensive_dominance
        mock.compute_comprehensive_dominance = lambda session: (
            mock._track_call("compute_comprehensive_dominance") or comprehensive_dominance
        )
    
    return mock


def create_mock_analyzer() -> MockDominanceAnalyzer:
    """
    Create a mock analyzer instance.
    
    Returns
    -------
    MockDominanceAnalyzer
        Mock analyzer instance
    """
    return MockDominanceAnalyzer()


def create_mock_data_provider(
    custom_data: Optional[Dict[str, Any]] = None
) -> MockDominanceDataProvider:
    """
    Create a mock data provider with optional custom data.
    
    Parameters
    ----------
    custom_data : Optional[Dict[str, Any]]
        Custom data to override default mock values
        
    Returns
    -------
    MockDominanceDataProvider
        Configured mock data provider instance
    """
    return MockDominanceDataProvider()


# ============================================================================
# Sample Test Data Generators
# ============================================================================

def create_sample_simulation_data() -> pd.DataFrame:
    """
    Create sample simulation data for testing.
    
    Returns
    -------
    pd.DataFrame
        Sample DataFrame with dominance analysis data
    """
    return pd.DataFrame({
        "iteration": range(20),
        "population_dominance": ["system"] * 12 + ["independent"] * 5 + ["control"] * 3,
        "survival_dominance": ["system"] * 8 + ["independent"] * 10 + ["control"] * 2,
        "comprehensive_dominance": ["system"] * 10 + ["independent"] * 7 + ["control"] * 3,
        "system_dominance_score": [0.6, 0.65, 0.7, 0.62, 0.68] * 4,
        "independent_dominance_score": [0.3, 0.28, 0.32, 0.35, 0.25] * 4,
        "control_dominance_score": [0.1, 0.12, 0.08, 0.15, 0.11] * 4,
        "total_switches": [2, 3, 1, 4, 2, 3, 5, 2, 1, 3, 4, 2, 3, 1, 5, 2, 4, 3, 2, 1],
        "switches_per_step": [0.02, 0.03, 0.01, 0.04, 0.02] * 4,
        "system_reproduction_success_rate": [0.8, 0.75, 0.82, 0.78, 0.81] * 4,
        "independent_reproduction_success_rate": [0.65, 0.68, 0.62, 0.70, 0.66] * 4,
        "control_reproduction_success_rate": [0.50, 0.52, 0.48, 0.55, 0.51] * 4,
        "system_first_reproduction_time": [15, 18, 12, 20, 16] * 4,
        "independent_first_reproduction_time": [20, 22, 18, 25, 21] * 4,
        "control_first_reproduction_time": [25, 28, 23, 30, 26] * 4,
    })


def create_minimal_simulation_data() -> pd.DataFrame:
    """
    Create minimal simulation data for basic tests.
    
    Returns
    -------
    pd.DataFrame
        Minimal DataFrame for testing
    """
    return pd.DataFrame({
        "iteration": [1, 2, 3],
        "total_switches": [2, 3, 1],
        "comprehensive_dominance": ["system", "independent", "system"],
    })

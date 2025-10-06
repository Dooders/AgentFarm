"""
Comprehensive tests for significant events analysis module.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, mock_open

import numpy as np
import pandas as pd
import pytest

from farm.analysis.significant_events import (
    significant_events_module,
    compute_event_severity,
    compute_event_patterns,
    compute_event_impact,
    detect_significant_events,
    analyze_significant_events,
    analyze_event_patterns,
    analyze_event_impact,
    plot_event_timeline,
    plot_event_severity_distribution,
    plot_event_impact_analysis,
    plot_significant_events,
)
from farm.analysis.common.context import AnalysisContext


@pytest.fixture
def sample_events():
    """Create sample event data."""
    return [
        {"type": "agent_death", "step": 120, "impact_scale": 0.7, "details": {"agent_id": "agent_42", "agent_type": "system", "generation": 3}},
        {
            "type": "resource_depletion",
            "step": 135,
            "impact_scale": 0.9,
            "details": {"total_resources_before": 1000, "total_resources_after": 100, "average_per_agent": 2.0, "drop_rate": 0.9},
        },
        {
            "type": "population_crash",
            "step": 150,
            "impact_scale": 1.0,
            "details": {"population_before": 200, "population_after": 50, "change_rate": 0.75, "deaths": 150},
        },
        {
            "type": "mass_combat",
            "step": 160,
            "impact_scale": 0.8,
            "details": {"combat_encounters": 25, "successful_attacks": 15, "total_agents": 100, "combat_rate": 0.25},
        },
        {"type": "agent_birth", "step": 170, "impact_scale": 0.3, "details": {"offspring_id": "agent_99", "parent_id": "agent_42", "generation": 4}},
        {
            "type": "health_critical",
            "step": 180,
            "impact_scale": 0.75,
            "details": {"agent_id": "agent_50", "health_before": 100.0, "health_after": 15.0, "cause": "combat", "drop_rate": 0.85},
        },
        {
            "type": "population_boom",
            "step": 190,
            "impact_scale": 0.6,
            "details": {"population_before": 50, "population_after": 100, "change_rate": 1.0, "births": 50},
        },
    ]


@pytest.fixture
def sample_events_with_severity(sample_events):
    """Create sample events with severity scores."""
    return compute_event_severity(sample_events.copy())


@pytest.fixture
def mock_db_with_events(request):
    """Create a mock database with realistic event data."""
    
    # Use a list to track call order (mutable so it persists across calls in same test)
    call_order = []
    
    # Reset the call order at the end of each test
    def reset_order():
        call_order.clear()
    request.addfinalizer(reset_order)
    
    def create_mock_query(query_num):
        """Create a fresh mock query with data for specific query number."""
        mock_query = MagicMock()
        
        # Query 1: agent deaths
        if query_num == 1:
            mock_query.filter.return_value.filter.return_value.all.return_value = [
                ('agent_1', 120, 'system', 3),
                ('agent_2', 150, 'independent', 5),
            ]
        # Query 2: agent births (reproduction events)
        elif query_num == 2:
            mock_query.filter.return_value.filter.return_value.all.return_value = [
                (170, 'agent_1', 'agent_10', True, 4),
            ]
        # Query 3: population changes (uses filter().filter().order_by())
        elif query_num == 3:
            # Handle the chained filters
            mock_filter1 = MagicMock()
            mock_filter2 = MagicMock()
            mock_filter2.order_by.return_value.all.return_value = [
                (100, 100, 5, 2),
                (110, 150, 55, 5),  # population boom
                (120, 140, 10, 15),
                (150, 50, 5, 95),   # population crash
            ]
            mock_filter1.filter.return_value = mock_filter2
            mock_query.filter.return_value = mock_filter1
        # Query 4: health incidents (uses filter().filter())
        elif query_num == 4:
            # Handle the chained filters
            mock_filter1 = MagicMock()
            mock_filter1.filter.return_value.all.return_value = [
                (160, 'agent_3', 100.0, 15.0, 'combat'),
            ]
            mock_query.filter.return_value = mock_filter1
        # Query 5: mass combat
        elif query_num == 5:
            mock_query.filter.return_value.filter.return_value.all.return_value = [
                (180, 25, 15, 100),
            ]
        # Query 6: resource depletion (uses filter().filter().order_by())
        elif query_num == 6:
            # Handle the chained filters
            mock_filter1 = MagicMock()
            mock_filter2 = MagicMock()
            mock_filter2.order_by.return_value.all.return_value = [
                (100, 1000.0, 10.0, 100),
                (135, 100.0, 1.0, 100),  # resource depletion
            ]
            mock_filter1.filter.return_value = mock_filter2
            mock_query.filter.return_value = mock_filter1
        else:
            # Default empty results
            mock_query.all.return_value = []
            mock_query.filter.return_value.all.return_value = []
            mock_query.filter.return_value.filter.return_value.all.return_value = []
            mock_query.filter.return_value.order_by.return_value.all.return_value = []
        
        return mock_query
    
    def mock_query_side_effect(*args, **kwargs):
        """Track calls and return appropriate mock query."""
        call_order.append(1)
        query_num = len(call_order)
        return create_mock_query(query_num)
    
    mock_session = MagicMock()
    mock_session.query.side_effect = mock_query_side_effect
    
    mock_db = MagicMock()
    def mock_execute(func):
        # Don't clear call_order - it should persist across multiple execute_with_retry calls
        # within a single detect_significant_events call
        return func(mock_session)
    
    mock_db.execute_with_retry = mock_execute
    
    return mock_db


class TestSignificantEventsComputations:
    """Test significant events statistical computations."""

    def test_detect_significant_events(self, mock_db_with_events):
        """Test event detection with realistic mock data."""
        events = detect_significant_events(mock_db_with_events, start_step=0, end_step=200, min_severity=0.3)

        assert isinstance(events, list)
        # Should detect multiple types of events
        assert len(events) > 0
        
        # Check that we have the expected event fields
        for event in events:
            assert "type" in event
            assert "step" in event
            assert "impact_scale" in event
            assert "details" in event
        
        # Check that we detect different event types
        event_types = {e["type"] for e in events}
        # Should have at least some of: agent_death, agent_birth, population_crash, etc.
        assert len(event_types) > 0

    def test_detect_significant_events_with_filters(self):
        """Test event detection with different filters."""
        # Create a mock session with empty results
        mock_session = MagicMock()
        
        def create_empty_query():
            mock_q = MagicMock()
            mock_q.filter.return_value.filter.return_value.all.return_value = []
            mock_q.filter.return_value.all.return_value = []
            mock_q.filter.return_value.order_by.return_value.all.return_value = []
            return mock_q
        
        mock_session.query.side_effect = lambda *args: create_empty_query()
        
        mock_db = MagicMock()
        def mock_execute(func):
            return func(mock_session)
        
        mock_db.execute_with_retry = mock_execute

        # Test with high severity threshold
        events = detect_significant_events(mock_db, min_severity=0.8)
        assert isinstance(events, list)

        # Test with specific time range
        events = detect_significant_events(mock_db, start_step=100, end_step=150)
        assert isinstance(events, list)
    
    def test_detect_agent_deaths(self, mock_db_with_events):
        """Test that agent death events are detected."""
        events = detect_significant_events(mock_db_with_events, start_step=0, end_step=200)
        
        death_events = [e for e in events if e["type"] == "agent_death"]
        assert len(death_events) > 0
        
        # Check death event structure
        for event in death_events:
            assert "agent_id" in event["details"]
            assert "agent_type" in event["details"]
            assert 0 <= event["impact_scale"] <= 1.0
    
    def test_detect_agent_births(self, mock_db_with_events):
        """Test that agent birth events are detected."""
        events = detect_significant_events(mock_db_with_events, start_step=0, end_step=200)
        
        birth_events = [e for e in events if e["type"] == "agent_birth"]
        assert len(birth_events) > 0
        
        # Check birth event structure
        for event in birth_events:
            assert "offspring_id" in event["details"]
            assert "parent_id" in event["details"]
            assert 0 <= event["impact_scale"] <= 1.0
    
    def test_detect_population_changes(self, mock_db_with_events):
        """Test that population crashes and booms are detected."""
        events = detect_significant_events(mock_db_with_events, start_step=0, end_step=200)
        
        population_events = [e for e in events if e["type"] in ["population_crash", "population_boom"]]
        assert len(population_events) > 0
        
        # Check population event structure
        for event in population_events:
            assert "population_before" in event["details"]
            assert "population_after" in event["details"]
            assert "change_rate" in event["details"]
            assert 0 <= event["impact_scale"] <= 1.0
    
    def test_detect_health_critical(self, mock_db_with_events):
        """Test that critical health incidents are detected."""
        events = detect_significant_events(mock_db_with_events, start_step=0, end_step=200)
        
        health_events = [e for e in events if e["type"] == "health_critical"]
        assert len(health_events) > 0
        
        # Check health event structure
        for event in health_events:
            assert "agent_id" in event["details"]
            assert "health_before" in event["details"]
            assert "health_after" in event["details"]
            assert "cause" in event["details"]
            assert 0 <= event["impact_scale"] <= 1.0
    
    def test_detect_mass_combat(self, mock_db_with_events):
        """Test that mass combat events are detected."""
        events = detect_significant_events(mock_db_with_events, start_step=0, end_step=200)
        
        combat_events = [e for e in events if e["type"] == "mass_combat"]
        assert len(combat_events) > 0
        
        # Check combat event structure
        for event in combat_events:
            assert "combat_encounters" in event["details"]
            assert "total_agents" in event["details"]
            assert 0 <= event["impact_scale"] <= 1.0
    
    def test_detect_resource_depletion(self, mock_db_with_events):
        """Test that resource depletion events are detected."""
        events = detect_significant_events(mock_db_with_events, start_step=0, end_step=200)
        
        resource_events = [e for e in events if e["type"] == "resource_depletion"]
        assert len(resource_events) > 0
        
        # Check resource event structure
        for event in resource_events:
            assert "total_resources_after" in event["details"]
            assert 0 <= event["impact_scale"] <= 1.0

    def test_compute_event_severity(self, sample_events):
        """Test event severity computation."""
        events_with_severity = compute_event_severity(sample_events)

        assert len(events_with_severity) == len(sample_events)

        for event in events_with_severity:
            assert "severity" in event
            assert "severity_category" in event
            assert 0 <= event["severity"] <= 1
            assert event["severity_category"] in ["low", "medium", "high"]

    def test_compute_event_severity_categories(self, sample_events):
        """Test event severity categorization."""
        events_with_severity = compute_event_severity(sample_events)

        # Check specific severity levels
        population_crash = next(e for e in events_with_severity if e["type"] == "population_crash")
        assert population_crash["severity_category"] == "high"

        agent_birth = next(e for e in events_with_severity if e["type"] == "agent_birth")
        assert agent_birth["severity_category"] == "low"
        
        # Check new event types
        mass_combat = next(e for e in events_with_severity if e["type"] == "mass_combat")
        assert mass_combat["severity_category"] in ["medium", "high"]
        
        health_critical = next(e for e in events_with_severity if e["type"] == "health_critical")
        assert health_critical["severity_category"] in ["medium", "high"]

    def test_compute_event_severity_empty_events(self):
        """Test severity computation with empty event list."""
        events = compute_event_severity([])
        assert events == []

    def test_compute_event_patterns(self, sample_events_with_severity):
        """Test event pattern computation."""
        patterns = compute_event_patterns(sample_events_with_severity)

        assert "event_frequency" in patterns
        assert "inter_event_times" in patterns
        assert "event_types" in patterns
        assert "severity_distribution" in patterns

        # Check that event types are counted
        assert isinstance(patterns["event_types"], dict)
        assert len(patterns["event_types"]) > 0

    def test_compute_event_patterns_empty(self):
        """Test pattern computation with no events."""
        patterns = compute_event_patterns([])
        assert patterns == {}

    def test_compute_event_patterns_single_event(self):
        """Test pattern computation with single event."""
        single_event = [
            {
                "type": "agent_death",
                "step": 100,
                "severity": 0.7,
                "impact_scale": 0.7,
            }
        ]

        patterns = compute_event_patterns(single_event)

        assert "event_types" in patterns
        assert "severity_distribution" in patterns
        # Should not have inter_event_times with single event
        assert "inter_event_times" not in patterns

    def test_compute_event_impact(self, sample_events_with_severity):
        """Test event impact computation."""
        impact = compute_event_impact(sample_events_with_severity)

        assert "impact_by_type" in impact
        assert "overall_impact" in impact

        # Check impact by type structure
        assert isinstance(impact["impact_by_type"], dict)
        for event_type, metrics in impact["impact_by_type"].items():
            assert "mean" in metrics
            assert "std" in metrics
            assert "count" in metrics

    def test_compute_event_impact_empty(self):
        """Test impact computation with no events."""
        impact = compute_event_impact([])
        assert impact == {}

    def test_compute_event_impact_single_type(self):
        """Test impact computation with single event type."""
        events = [
            {"type": "agent_death", "impact_scale": 0.7},
            {"type": "agent_death", "impact_scale": 0.8},
            {"type": "agent_death", "impact_scale": 0.6},
        ]

        impact = compute_event_impact(events)

        assert "impact_by_type" in impact
        assert "agent_death" in impact["impact_by_type"]
        assert impact["impact_by_type"]["agent_death"]["count"] == 3


class TestSignificantEventsAnalysis:
    """Test significant events analysis functions."""

    def test_analyze_significant_events(self, tmp_path, sample_events):
        """Test significant events analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        mock_db = MagicMock()

        with patch("farm.analysis.significant_events.analyze.detect_significant_events") as mock_detect:
            mock_detect.return_value = sample_events

            analyze_significant_events(ctx, db_connection=mock_db, min_severity=0.3)

        # Check output file
        output_file = tmp_path / "significant_events.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert "total_events_detected" in data
        assert "significant_events" in data
        assert "events" in data

    def test_analyze_significant_events_with_filters(self, tmp_path, sample_events):
        """Test event analysis with different filters."""
        ctx = AnalysisContext(output_path=tmp_path)
        mock_db = MagicMock()

        with patch("farm.analysis.significant_events.analyze.detect_significant_events") as mock_detect:
            mock_detect.return_value = sample_events

            analyze_significant_events(ctx, db_connection=mock_db, min_severity=0.7)

        output_file = tmp_path / "significant_events.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        # Should have filtered out low-severity events
        assert data["significant_events"] < data["total_events_detected"]

    def test_analyze_event_patterns(self, tmp_path, sample_events):
        """Test event patterns analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        mock_db = MagicMock()

        with patch("farm.analysis.significant_events.analyze.detect_significant_events") as mock_detect:
            mock_detect.return_value = sample_events

            analyze_event_patterns(ctx, db_connection=mock_db)

        # Check output file
        output_file = tmp_path / "event_patterns.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert "event_types" in data or "event_frequency" in data

    def test_analyze_event_impact(self, tmp_path, sample_events):
        """Test event impact analysis."""
        ctx = AnalysisContext(output_path=tmp_path)
        mock_db = MagicMock()

        with patch("farm.analysis.significant_events.analyze.detect_significant_events") as mock_detect:
            mock_detect.return_value = sample_events

            analyze_event_impact(ctx, db_connection=mock_db)

        # Check output file
        output_file = tmp_path / "event_impact.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert "impact_by_type" in data or "overall_impact" in data

    def test_analyze_with_no_events(self, tmp_path):
        """Test analysis with no events detected."""
        ctx = AnalysisContext(output_path=tmp_path)
        mock_db = MagicMock()

        with patch("farm.analysis.significant_events.analyze.detect_significant_events") as mock_detect:
            mock_detect.return_value = []

            analyze_significant_events(ctx, db_connection=mock_db)

        output_file = tmp_path / "significant_events.json"
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert data["total_events_detected"] == 0
        assert data["significant_events"] == 0


class TestSignificantEventsVisualization:
    """Test significant events visualization functions."""

    def setup_events_file(self, tmp_path, events):
        """Helper to set up events file for plotting."""
        events_file = tmp_path / "significant_events.json"
        data = {
            "total_events_detected": len(events),
            "significant_events": len(events),
            "events": events,
        }
        with open(events_file, "w") as f:
            json.dump(data, f)

    @patch("farm.analysis.significant_events.plot.plt")
    def test_plot_event_timeline(self, mock_plt, tmp_path, sample_events_with_severity):
        """Test event timeline plotting."""
        # Mock subplots to return proper figure and axis objects
        mock_fig = mock_plt.figure.return_value
        mock_ax = mock_plt.subplot.return_value
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        self.setup_events_file(tmp_path, sample_events_with_severity)

        ctx = AnalysisContext(output_path=tmp_path)
        plot_event_timeline(ctx)

        # Check that plot was created
        assert mock_plt.savefig.called
        assert mock_plt.close.called

    @patch("farm.analysis.significant_events.plot.plt")
    def test_plot_event_timeline_no_file(self, mock_plt, tmp_path):
        """Test timeline plotting with no events file."""
        ctx = AnalysisContext(output_path=tmp_path)

        # Should handle missing file gracefully
        plot_event_timeline(ctx)

        # Should not try to plot
        assert not mock_plt.savefig.called

    @patch("farm.analysis.significant_events.plot.plt")
    def test_plot_event_timeline_empty_events(self, mock_plt, tmp_path):
        """Test timeline plotting with empty events."""
        self.setup_events_file(tmp_path, [])

        ctx = AnalysisContext(output_path=tmp_path)
        plot_event_timeline(ctx)

        # Should not create plot with no events
        assert not mock_plt.savefig.called

    @patch("farm.analysis.significant_events.plot.plt")
    def test_plot_event_severity_distribution(self, mock_plt, tmp_path, sample_events_with_severity):
        """Test event severity distribution plotting."""
        # Mock subplots to return proper figure and axis objects
        mock_fig = mock_plt.figure.return_value
        mock_ax1 = mock_plt.subplot.return_value
        mock_ax2 = mock_plt.subplot.return_value
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))

        self.setup_events_file(tmp_path, sample_events_with_severity)

        ctx = AnalysisContext(output_path=tmp_path)
        plot_event_severity_distribution(ctx)

        # Check that plot was created
        assert mock_plt.savefig.called

    @patch("farm.analysis.significant_events.plot.plt")
    @patch("farm.analysis.significant_events.plot.sns")
    def test_plot_event_impact_analysis(self, mock_sns, mock_plt, tmp_path, sample_events_with_severity):
        """Test event impact analysis plotting."""
        # Mock subplots to return proper figure and axis objects
        mock_fig = mock_plt.figure.return_value
        mock_ax = mock_plt.subplot.return_value
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        self.setup_events_file(tmp_path, sample_events_with_severity)

        ctx = AnalysisContext(output_path=tmp_path)
        plot_event_impact_analysis(ctx)

        # Check that plot was created
        assert mock_plt.savefig.called

    @patch("farm.analysis.significant_events.plot.plt")
    @patch("farm.analysis.significant_events.plot.sns")
    def test_plot_significant_events_comprehensive(self, mock_sns, mock_plt, tmp_path, sample_events_with_severity):
        """Test comprehensive significant events plotting."""
        # Mock subplots to return proper figure and axis objects
        mock_fig = mock_plt.figure.return_value
        mock_ax = mock_plt.subplot.return_value
        mock_ax1 = mock_plt.subplot.return_value
        mock_ax2 = mock_plt.subplot.return_value
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        # For the severity distribution plot that uses (1, 2) subplots
        mock_plt.subplots.side_effect = [
            (mock_fig, mock_ax),  # timeline
            (mock_fig, (mock_ax1, mock_ax2)),  # severity distribution
            (mock_fig, mock_ax),  # impact analysis
        ]

        self.setup_events_file(tmp_path, sample_events_with_severity)

        ctx = AnalysisContext(output_path=tmp_path)
        plot_significant_events(ctx)

        # Should create multiple plots
        assert mock_plt.savefig.call_count >= 3


class TestSignificantEventsModule:
    """Test significant events module integration."""

    def test_significant_events_module_registration(self):
        """Test module registration."""
        assert significant_events_module.name == "significant_events"
        assert (
            significant_events_module.description
            == "Analysis of significant events, their severity, patterns, and impact"
        )

    def test_significant_events_module_function_names(self):
        """Test module function names."""
        functions = significant_events_module.get_function_names()
        expected_functions = [
            "analyze_events",
            "analyze_patterns",
            "analyze_impact",
            "plot_timeline",
            "plot_severity",
            "plot_impact",
        ]

        for func_name in expected_functions:
            assert func_name in functions

    def test_significant_events_module_function_groups(self):
        """Test module function groups."""
        groups = significant_events_module.get_function_groups()
        assert "all" in groups
        assert "plots" in groups
        assert "analysis" in groups

    def test_significant_events_module_data_processor(self):
        """Test module data processor."""
        processor = significant_events_module.get_data_processor()
        # Significant events module uses database queries directly
        assert processor is None

    def test_module_validator(self):
        """Test module validator."""
        validator = significant_events_module.get_validator()
        assert validator is not None

    def test_module_all_functions_registered(self):
        """Test that all expected functions are registered."""
        functions = significant_events_module.get_functions()
        assert len(functions) >= 6


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_compute_severity_unknown_event_type(self):
        """Test severity computation with unknown event type."""
        events = [
            {
                "type": "unknown_event",
                "step": 100,
                "impact_scale": 0.5,
            }
        ]

        events_with_severity = compute_event_severity(events)

        # Should handle unknown type with default severity
        assert events_with_severity[0]["severity"] > 0

    def test_compute_patterns_missing_fields(self):
        """Test pattern computation with missing fields."""
        events = [
            {"type": "agent_death"},
            {"step": 100},
            {"severity": 0.7},
        ]

        patterns = compute_event_patterns(events)

        # Should handle missing fields gracefully
        assert isinstance(patterns, dict)

    def test_compute_impact_with_nan_values(self):
        """Test impact computation with NaN values."""
        events = [
            {"type": "agent_death", "impact_scale": 0.7},
            {"type": "agent_death", "impact_scale": np.nan},
            {"type": "agent_death", "impact_scale": 0.8},
        ]

        impact = compute_event_impact(events)

        # Should handle NaN gracefully
        assert "overall_impact" in impact

    def test_analyze_with_progress_callback(self, tmp_path, sample_events):
        """Test analysis with progress callback."""
        progress_calls = []

        def progress_callback(message, progress):
            progress_calls.append((message, progress))

        ctx = AnalysisContext(output_path=tmp_path, progress_callback=progress_callback)
        mock_db = MagicMock()

        with patch("farm.analysis.significant_events.analyze.detect_significant_events") as mock_detect:
            mock_detect.return_value = sample_events

            analyze_significant_events(ctx, db_connection=mock_db)

        # Should have called progress callback
        assert len(progress_calls) > 0
        assert any("complete" in msg.lower() for msg, _ in progress_calls)

    def test_plot_timeline_missing_step_column(self, tmp_path):
        """Test timeline plotting with missing step column."""
        events = [
            {"type": "agent_death", "severity": 0.7},
            {"type": "agent_birth", "severity": 0.3},
        ]

        events_file = tmp_path / "significant_events.json"
        with open(events_file, "w") as f:
            json.dump({"events": events}, f)

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.significant_events.plot.plt"):
            # Should handle missing step column gracefully
            plot_event_timeline(ctx)

    def test_plot_severity_missing_severity_column(self, tmp_path):
        """Test severity plotting with missing severity column."""
        events = [
            {"type": "agent_death", "step": 100},
            {"type": "agent_birth", "step": 120},
        ]

        events_file = tmp_path / "significant_events.json"
        with open(events_file, "w") as f:
            json.dump({"events": events}, f)

        ctx = AnalysisContext(output_path=tmp_path)

        with patch("farm.analysis.significant_events.plot.plt"):
            # Should handle missing severity column gracefully
            plot_event_severity_distribution(ctx)

    def test_event_detection_with_extreme_values(self):
        """Test event detection handles extreme values."""
        mock_db = MagicMock()

        # Should not crash with extreme values
        events = detect_significant_events(mock_db, start_step=-1000, end_step=999999)

        assert isinstance(events, list)

    def test_severity_computation_extreme_impact_scale(self):
        """Test severity with extreme impact scale values."""
        events = [
            {"type": "agent_death", "impact_scale": 10.0},  # Very high
            {"type": "agent_birth", "impact_scale": -1.0},  # Negative
            {"type": "environmental_change", "impact_scale": 0.0},  # Zero
        ]

        events_with_severity = compute_event_severity(events)

        # All severities should be capped at 1.0
        for event in events_with_severity:
            assert 0 <= event["severity"] <= 1.0

    def test_patterns_with_simultaneous_events(self):
        """Test pattern computation with events at same step."""
        events = [
            {"type": "agent_death", "step": 100, "severity": 0.7},
            {"type": "agent_birth", "step": 100, "severity": 0.3},
            {"type": "resource_depletion", "step": 100, "severity": 0.9},
        ]

        patterns = compute_event_patterns(events)

        assert "event_frequency" in patterns
        assert "event_types" in patterns

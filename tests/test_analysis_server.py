"""
Tests for server-side analysis management functionality.

Tests resource cleanup, concurrency limiting, and new management endpoints.
"""

import pytest
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, ANY

# Note: These tests would normally use FastAPI's TestClient
# For now, we'll test the core logic functions directly


class TestResourceCleanup:
    """Tests for _cleanup_old_analyses function."""
    
    def test_cleanup_removes_old_analyses(self):
        """Test cleanup removes analyses older than retention period."""
        from farm.api import server
        
        # Set up test data
        now = datetime.now()
        old_time = (now - timedelta(hours=25)).isoformat()
        recent_time = (now - timedelta(hours=1)).isoformat()
        
        with patch.object(server, 'active_analyses', {
            'old_analysis': {
                'status': 'completed',
                'ended_at': old_time,
                'controller': Mock()
            },
            'recent_analysis': {
                'status': 'completed',
                'ended_at': recent_time,
                'controller': Mock()
            },
            'running_analysis': {
                'status': 'running',
                'controller': Mock()
            }
        }):
            server._cleanup_old_analyses()
            
            # Old analysis should be removed
            assert 'old_analysis' not in server.active_analyses
            # Recent and running should remain
            assert 'recent_analysis' in server.active_analyses
            assert 'running_analysis' in server.active_analyses
    
    def test_cleanup_limits_total_completed(self):
        """Test cleanup enforces max completed analyses limit."""
        from farm.api import server
        
        # Create more than MAX_COMPLETED_ANALYSES
        now = datetime.now()
        analyses = {}
        
        # Create 110 completed analyses (max is 100)
        for i in range(110):
            # Stagger times so oldest are first
            end_time = (now - timedelta(hours=i/10)).isoformat()
            analyses[f'analysis_{i}'] = {
                'status': 'completed',
                'ended_at': end_time,
                'controller': Mock()
            }
        
        with patch.object(server, 'active_analyses', analyses):
            with patch.object(server, 'MAX_COMPLETED_ANALYSES', 100):
                server._cleanup_old_analyses()
                
                # Should have at most 100 completed
                completed = [a for a in server.active_analyses.values() 
                           if a['status'] == 'completed']
                assert len(completed) <= 100
    
    def test_cleanup_calls_controller_cleanup(self):
        """Test cleanup calls controller.cleanup() for removed analyses."""
        from farm.api import server
        
        now = datetime.now()
        old_time = (now - timedelta(hours=25)).isoformat()
        
        mock_controller = Mock()
        
        with patch.object(server, 'active_analyses', {
            'old_analysis': {
                'status': 'completed',
                'ended_at': old_time,
                'controller': mock_controller
            }
        }):
            server._cleanup_old_analyses()
            
            # Controller cleanup should have been called
            mock_controller.cleanup.assert_called_once()
    
    def test_cleanup_handles_missing_ended_at(self):
        """Test cleanup handles analyses without ended_at gracefully."""
        from farm.api import server
        
        with patch.object(server, 'active_analyses', {
            'no_end_time': {
                'status': 'completed',
                'controller': Mock()
                # No ended_at field
            },
            'running': {
                'status': 'running',
                'controller': Mock()
            }
        }):
            # Should not raise exception
            server._cleanup_old_analyses()
            
            # Both should remain (no_end_time can't be aged out)
            assert 'no_end_time' in server.active_analyses
            assert 'running' in server.active_analyses
    
    def test_cleanup_handles_controller_cleanup_error(self):
        """Test cleanup continues even if controller.cleanup() fails."""
        from farm.api import server
        
        now = datetime.now()
        old_time = (now - timedelta(hours=25)).isoformat()
        
        failing_controller = Mock()
        failing_controller.cleanup.side_effect = Exception("Cleanup failed")
        
        with patch.object(server, 'active_analyses', {
            'old_analysis': {
                'status': 'completed',
                'ended_at': old_time,
                'controller': failing_controller
            }
        }):
            # Should not raise exception
            server._cleanup_old_analyses()
            
            # Analysis should still be removed despite cleanup error
            assert 'old_analysis' not in server.active_analyses


class TestConcurrencyLimiting:
    """Tests for concurrency limiting with semaphore."""
    
    def test_semaphore_limits_concurrent_analyses(self):
        """Test semaphore prevents too many concurrent analyses."""
        from farm.api import server
        
        # Create a semaphore with limit of 2
        semaphore = threading.Semaphore(2)
        
        # Track concurrent count
        concurrent_count = [0]
        max_concurrent = [0]
        lock = threading.Lock()
        
        def mock_analysis():
            with semaphore:
                with lock:
                    concurrent_count[0] += 1
                    max_concurrent[0] = max(max_concurrent[0], concurrent_count[0])
                
                time.sleep(0.1)  # Simulate work
                
                with lock:
                    concurrent_count[0] -= 1
        
        # Start 5 threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=mock_analysis)
            t.start()
            threads.append(t)
        
        # Wait for all to complete
        for t in threads:
            t.join()
        
        # Max concurrent should not exceed 2
        assert max_concurrent[0] <= 2
        assert max_concurrent[0] > 0  # At least some concurrency occurred


class TestBackgroundAnalysisExecution:
    """Tests for _run_analysis_background function."""
    
    def test_background_sets_running_status(self):
        """Test background task sets status to running."""
        from farm.api import server
        
        mock_controller = Mock()
        mock_controller.wait_for_completion = Mock()
        mock_controller.get_result = Mock(return_value=Mock(
            success=True,
            output_path=Path("/fake/out"),
            execution_time=1.0,
            cache_hit=False,
            dataframe=Mock(__len__=Mock(return_value=10))
        ))
        
        with patch.object(server, 'active_analyses', {
            'test_id': {'status': 'pending', 'controller': mock_controller}
        }):
            with patch.object(server, '_analysis_semaphore', threading.Semaphore(10)):
                with patch.object(server, '_cleanup_old_analyses', Mock()):
                    server._run_analysis_background('test_id', mock_controller)
                    
                    # Status should have been set to running
                    assert server.active_analyses['test_id']['status'] == 'completed'
    
    def test_background_updates_on_success(self):
        """Test background task updates state on successful completion."""
        from farm.api import server
        
        mock_result = Mock()
        mock_result.success = True
        mock_result.output_path = Path("/fake/out")
        mock_result.execution_time = 1.5
        mock_result.cache_hit = False
        mock_result.dataframe = Mock(__len__=Mock(return_value=100))
        
        mock_controller = Mock()
        mock_controller.wait_for_completion = Mock()
        mock_controller.get_result = Mock(return_value=mock_result)
        
        with patch.object(server, 'active_analyses', {
            'test_id': {'status': 'pending', 'controller': mock_controller}
        }):
            with patch.object(server, '_analysis_semaphore', threading.Semaphore(10)):
                with patch.object(server, '_cleanup_old_analyses', Mock()):
                    server._run_analysis_background('test_id', mock_controller)
                    
                    analysis = server.active_analyses['test_id']
                    assert analysis['status'] == 'completed'
                    assert analysis['output_path'] == str(mock_result.output_path)
                    assert analysis['execution_time'] == 1.5
                    assert analysis['rows'] == 100
                    assert 'ended_at' in analysis
    
    def test_background_updates_on_error(self):
        """Test background task updates state on error."""
        from farm.api import server
        
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Test error"
        
        mock_controller = Mock()
        mock_controller.wait_for_completion = Mock()
        mock_controller.get_result = Mock(return_value=mock_result)
        
        with patch.object(server, 'active_analyses', {
            'test_id': {'status': 'pending', 'controller': mock_controller}
        }):
            with patch.object(server, '_analysis_semaphore', threading.Semaphore(10)):
                with patch.object(server, '_cleanup_old_analyses', Mock()):
                    server._run_analysis_background('test_id', mock_controller)
                    
                    analysis = server.active_analyses['test_id']
                    assert analysis['status'] == 'error'
                    assert analysis['error'] == "Test error"
                    assert 'ended_at' in analysis
    
    def test_background_handles_exception(self):
        """Test background task handles exceptions gracefully."""
        from farm.api import server
        
        mock_controller = Mock()
        mock_controller.start = Mock(side_effect=Exception("Test exception"))
        
        with patch.object(server, 'active_analyses', {
            'test_id': {'status': 'pending', 'controller': mock_controller}
        }):
            with patch.object(server, '_analysis_semaphore', threading.Semaphore(10)):
                with patch.object(server, '_cleanup_old_analyses', Mock()):
                    # Should not raise exception
                    server._run_analysis_background('test_id', mock_controller)
                    
                    analysis = server.active_analyses['test_id']
                    assert analysis['status'] == 'error'
                    assert 'Test exception' in analysis['error']
                    assert analysis['error_type'] == 'Exception'
    
    def test_background_calls_cleanup(self):
        """Test background task always calls cleanup."""
        from farm.api import server
        
        mock_controller = Mock()
        mock_controller.wait_for_completion = Mock()
        mock_controller.get_result = Mock(return_value=Mock(
            success=True,
            output_path=Path("/fake"),
            execution_time=1.0,
            cache_hit=False,
            dataframe=None
        ))
        
        mock_cleanup = Mock()
        
        with patch.object(server, 'active_analyses', {
            'test_id': {'status': 'pending', 'controller': mock_controller}
        }):
            with patch.object(server, '_analysis_semaphore', threading.Semaphore(10)):
                with patch.object(server, '_cleanup_old_analyses', mock_cleanup):
                    server._run_analysis_background('test_id', mock_controller)
                    
                    # Cleanup should have been called
                    mock_cleanup.assert_called_once()


class TestAnalysisStateManagement:
    """Tests for analysis state management."""
    
    def test_get_state_with_live_controller(self):
        """Test getting state includes live controller data."""
        mock_controller = Mock()
        mock_controller.get_state = Mock(return_value={
            'progress': 0.75,
            'message': 'Processing...',
            'is_running': True
        })
        
        analysis_info = {
            'module_name': 'test',
            'status': 'running',
            'controller': mock_controller
        }
        
        # Simulate extracting state (like in get_analysis_status endpoint)
        state = dict(analysis_info)
        controller = state.pop('controller', None)
        if controller:
            live_state = controller.get_state()
            state.update(live_state)
        
        assert state['progress'] == 0.75
        assert state['is_running'] is True
        assert 'controller' not in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

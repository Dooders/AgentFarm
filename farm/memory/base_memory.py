"""
Base memory classes and utilities for agent memory systems.

This module provides common base classes and mixins for agent memory implementations,
reducing code duplication across different memory backends.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable, Optional
import logging

logger = logging.getLogger(__name__)


class MemorySearchMixin:
    """
    Mixin class providing common search functionality for agent memory systems.

    This mixin contains the common search logic used by different memory implementations,
    reducing code duplication while allowing for backend-specific customization.
    """

    @abstractmethod
    def _get_timeline_key(self, agent_id: str) -> str:
        """Get the Redis key for the agent's timeline."""
        raise NotImplementedError

    @abstractmethod
    def _get_all_timeline_steps(self, timeline_key: str) -> List[str]:
        """Get all step IDs from the timeline."""
        raise NotImplementedError

    @abstractmethod
    def _retrieve_state_by_step(self, step: int) -> Optional[Dict[str, Any]]:
        """Retrieve a state dictionary by step number."""
        raise NotImplementedError

    def _search_states_by_criteria(
        self,
        agent_id: str,
        criteria_func: Callable[[Dict[str, Any]], bool],
        limit: int = 10,
        error_context: str = "search"
    ) -> List[Dict[str, Any]]:
        """
        Search for states matching given criteria.

        This is a common implementation used by search_by_metadata and search_by_state_value.

        Args:
            agent_id: ID of the agent whose states to search
            criteria_func: Function that takes a state dict and returns True if it matches
            limit: Maximum number of results to return
            error_context: Context string for error logging

        Returns:
            List of matching state dictionaries
        """
        try:
            timeline_key = self._get_timeline_key(agent_id)
            all_steps = self._get_all_timeline_steps(timeline_key)

            results = []
            for step in all_steps:
                state = self._retrieve_state_by_step(int(step))
                if state and criteria_func(state):
                    results.append(state)

                if len(results) >= limit:
                    break

            return results

        except Exception as e:
            logger.error(f"Failed to {error_context} for agent {agent_id}: {e}")
            return []

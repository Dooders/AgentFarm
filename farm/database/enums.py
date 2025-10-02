from enum import Enum


class AnalysisScope(str, Enum):
    """Scope levels for analysis queries.

    SIMULATION: All data (no filters)
    STEP: Single step
    STEP_RANGE: Range of steps
    AGENT: Single agent
    EPISODE: Episode level analysis
    """

    SIMULATION = "simulation"
    STEP = "step"
    STEP_RANGE = "step_range"
    AGENT = "agent"
    EPISODE = "episode"

    @classmethod
    def from_string(cls, scope_str: str) -> "AnalysisScope":
        """Convert string to AnalysisScope, case-insensitive."""
        try:
            return cls(scope_str.lower())
        except ValueError as exc:
            valid_scopes = [s.value for s in cls]
            raise ValueError(
                f"Invalid scope '{scope_str}'. Must be one of: {valid_scopes}"
            ) from exc

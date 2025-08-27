from typing import Optional, Tuple, Union

from sqlalchemy import func
from sqlalchemy.orm import Query

from farm.database.enums import AnalysisScope
from farm.database.models import AgentModel


def filter_scope(
    query: Query,
    scope: Union[str, AnalysisScope],
    agent_id: Optional[str] = None,
    step: Optional[int] = None,
    step_range: Optional[Tuple[int, int]] = None,
) -> Query:
    """Apply scope filters to the query based on the provided parameters.

    Parameters
    ----------
    query : Query
        SQLAlchemy query to be filtered
    scope : Union[str, AnalysisScope]
        Level at which to perform the analysis:
        - "simulation": Analyze all data without filters
        - "step": Analyze a specific step
        - "step_range": Analyze a range of steps
        - "agent": Analyze a specific agent
        Can be provided as string or AnalysisScope enum.
    agent_id : Optional[str], default=None
        ID of agent to analyze. Required when scope is "agent".
        Must be a valid agent ID in the database.
    step : Optional[int], default=None
        Specific step number to analyze. Required when scope is "step".
        Must be >= 0.
    step_range : Optional[Tuple[int, int]], default=None
        Range of steps to analyze as (start_step, end_step). Required when
        scope is "step_range". Both values must be >= 0 and start <= end.

    Returns
    -------
    Query
        SQLAlchemy query with appropriate scope filters applied

    Raises
    ------
    ValueError
        If required parameters are missing for the specified scope:
        - step missing when scope is "step"
        - step_range missing when scope is "step_range"
    TypeError
        If parameters are of incorrect type

    Examples
    --------
    >>> # Filter for specific agent
    >>> query = session.query(AgentAction)
    >>> query = filter_scope(query, "agent", agent_id=1)

    >>> # Filter for step range
    >>> query = session.query(AgentAction)
    >>> query = filter_scope(query, "step_range", step_range=(100, 200))

    Notes
    -----
    - The simulation scope returns the unmodified query without filters
    - String scopes are converted to AnalysisScope enum values
    - This method is typically used internally by analysis methods that
      support different scoping options
    """
    # Convert string scope to enum if needed
    if isinstance(scope, str):
        scope = AnalysisScope.from_string(scope)

    # For AGENT scope, randomly select an agent_id if none provided
    if scope == AnalysisScope.AGENT and agent_id is None:
        # Get a random agent_id from the database
        random_agent = (
            query.session.query(AgentModel.agent_id).order_by(func.random()).first()
        )
        if random_agent is None:
            raise ValueError("No agents found in database")
        agent_id = random_agent[0]

    # Validate remaining parameters based on scope
    if scope == AnalysisScope.STEP and step is None:
        raise ValueError("step is required when scope is STEP")
    if scope == AnalysisScope.STEP_RANGE and step_range is None:
        raise ValueError("step_range is required when scope is STEP_RANGE")

    # Apply filters based on scope
    if scope == AnalysisScope.AGENT:
        query = query.filter(AgentModel.agent_id == agent_id)
    elif scope == AnalysisScope.STEP:
        query = query.filter(AgentModel.step_number == step)
    elif scope == AnalysisScope.STEP_RANGE:
        if step_range is None:
            raise ValueError("step_range is required when scope is STEP_RANGE")
        start_step, end_step = step_range
        query = query.filter(
            AgentModel.step_number >= start_step,
            AgentModel.step_number <= end_step,
        )
    elif scope == AnalysisScope.EPISODE:
        # For episode scope, we don't apply additional filters
        # as it represents a complete episode of actions
        pass
    # SIMULATION scope requires no filters

    return query

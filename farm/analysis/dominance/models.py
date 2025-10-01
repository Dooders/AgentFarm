from typing import Dict, List, Optional, Union

from farm.utils.logging_config import get_logger

logger = get_logger(__name__)

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class DominanceDataModel(BaseModel):
    """
    Pydantic model representing the columns in the DataFrame returned by process_dominance_data.

    This model defines the expected structure and types for the dominance analysis data.
    """

    # Basic simulation metadata
    iteration: int = Field(..., description="Iteration number of the simulation")

    # Dominance type results
    population_dominance: str = Field(
        ..., description="Agent type with highest population"
    )
    survival_dominance: str = Field(
        ..., description="Agent type with best survival metrics"
    )
    comprehensive_dominance: str = Field(
        ..., description="Agent type with highest overall dominance"
    )

    # Dominance scores for each agent type
    system_dominance_score: float = Field(
        ..., description="Overall dominance score for system agents"
    )
    independent_dominance_score: float = Field(
        ..., description="Overall dominance score for independent agents"
    )
    control_dominance_score: float = Field(
        ..., description="Overall dominance score for control agents"
    )

    # Area under curve metrics
    system_auc: float = Field(
        ..., description="Area under curve for system agent population"
    )
    independent_auc: float = Field(
        ..., description="Area under curve for independent agent population"
    )
    control_auc: float = Field(
        ..., description="Area under curve for control agent population"
    )

    # Recency weighted area under curve
    system_recency_weighted_auc: float = Field(
        ..., description="Recency weighted AUC for system agents"
    )
    independent_recency_weighted_auc: float = Field(
        ..., description="Recency weighted AUC for independent agents"
    )
    control_recency_weighted_auc: float = Field(
        ..., description="Recency weighted AUC for control agents"
    )

    # Dominance duration metrics
    system_dominance_duration: float = Field(
        ..., description="Duration of system agent dominance"
    )
    independent_dominance_duration: float = Field(
        ..., description="Duration of independent agent dominance"
    )
    control_dominance_duration: float = Field(
        ..., description="Duration of control agent dominance"
    )

    # Growth trend metrics
    system_growth_trend: float = Field(
        ..., description="Growth trend for system agents"
    )
    independent_growth_trend: float = Field(
        ..., description="Growth trend for independent agents"
    )
    control_growth_trend: float = Field(
        ..., description="Growth trend for control agents"
    )

    # Final population ratio metrics
    system_final_ratio: float = Field(
        ..., description="Final population ratio for system agents"
    )
    independent_final_ratio: float = Field(
        ..., description="Final population ratio for independent agents"
    )
    control_final_ratio: float = Field(
        ..., description="Final population ratio for control agents"
    )

    # Dominance switching metrics
    total_switches: Optional[int] = Field(
        None, description="Total number of dominance switches"
    )
    switches_per_step: Optional[float] = Field(
        None, description="Average switches per simulation step"
    )

    # Average dominance periods
    system_avg_dominance_period: Optional[float] = Field(
        None, description="Average dominance period for system agents"
    )
    independent_avg_dominance_period: Optional[float] = Field(
        None, description="Average dominance period for independent agents"
    )
    control_avg_dominance_period: Optional[float] = Field(
        None, description="Average dominance period for control agents"
    )

    # Phase-specific switch counts
    early_phase_switches: Optional[int] = Field(
        None, description="Number of dominance switches in early phase"
    )
    middle_phase_switches: Optional[int] = Field(
        None, description="Number of dominance switches in middle phase"
    )
    late_phase_switches: Optional[int] = Field(
        None, description="Number of dominance switches in late phase"
    )

    # Transition matrix data
    system_to_system: Optional[float] = Field(
        None, description="Probability of system→system transition"
    )
    system_to_independent: Optional[float] = Field(
        None, description="Probability of system→independent transition"
    )
    system_to_control: Optional[float] = Field(
        None, description="Probability of system→control transition"
    )
    independent_to_system: Optional[float] = Field(
        None, description="Probability of independent→system transition"
    )
    independent_to_independent: Optional[float] = Field(
        None, description="Probability of independent→independent transition"
    )
    independent_to_control: Optional[float] = Field(
        None, description="Probability of independent→control transition"
    )
    control_to_system: Optional[float] = Field(
        None, description="Probability of control→system transition"
    )
    control_to_independent: Optional[float] = Field(
        None, description="Probability of control→independent transition"
    )
    control_to_control: Optional[float] = Field(
        None, description="Probability of control→control transition"
    )

    # Final population counts
    system_agents: Optional[int] = Field(
        None, description="Final count of system agents"
    )
    independent_agents: Optional[int] = Field(
        None, description="Final count of independent agents"
    )
    control_agents: Optional[int] = Field(
        None, description="Final count of control agents"
    )
    total_agents: Optional[int] = Field(None, description="Final count of all agents")
    final_step: Optional[int] = Field(None, description="Final simulation step number")

    # Agent survival statistics
    system_count: Optional[int] = Field(
        None, description="Total count of system agents"
    )
    system_alive: Optional[int] = Field(
        None, description="Count of alive system agents"
    )
    system_dead: Optional[int] = Field(None, description="Count of dead system agents")
    system_avg_survival: Optional[float] = Field(
        None, description="Average survival time for system agents"
    )
    system_dead_ratio: Optional[float] = Field(
        None, description="Ratio of dead system agents"
    )

    independent_count: Optional[int] = Field(
        None, description="Total count of independent agents"
    )
    independent_alive: Optional[int] = Field(
        None, description="Count of alive independent agents"
    )
    independent_dead: Optional[int] = Field(
        None, description="Count of dead independent agents"
    )
    independent_avg_survival: Optional[float] = Field(
        None, description="Average survival time for independent agents"
    )
    independent_dead_ratio: Optional[float] = Field(
        None, description="Ratio of dead independent agents"
    )

    control_count: Optional[int] = Field(
        None, description="Total count of control agents"
    )
    control_alive: Optional[int] = Field(
        None, description="Count of alive control agents"
    )
    control_dead: Optional[int] = Field(
        None, description="Count of dead control agents"
    )
    control_avg_survival: Optional[float] = Field(
        None, description="Average survival time for control agents"
    )
    control_dead_ratio: Optional[float] = Field(
        None, description="Ratio of dead control agents"
    )

    # Reproduction statistics - these fields are dynamically generated based on the data
    # Common reproduction metrics
    system_reproduction_attempts: Optional[int] = Field(
        None, description="Number of reproduction attempts by system agents"
    )
    system_reproduction_successes: Optional[int] = Field(
        None, description="Number of successful reproductions by system agents"
    )
    system_reproduction_failures: Optional[int] = Field(
        None, description="Number of failed reproductions by system agents"
    )
    system_reproduction_success_rate: Optional[float] = Field(
        None, description="Success rate of reproduction for system agents"
    )
    system_first_reproduction_time: Optional[float] = Field(
        None, description="Time of first reproduction for system agents"
    )
    system_reproduction_efficiency: Optional[float] = Field(
        None, description="Reproduction efficiency for system agents"
    )

    independent_reproduction_attempts: Optional[int] = Field(
        None, description="Number of reproduction attempts by independent agents"
    )
    independent_reproduction_successes: Optional[int] = Field(
        None, description="Number of successful reproductions by independent agents"
    )
    independent_reproduction_failures: Optional[int] = Field(
        None, description="Number of failed reproductions by independent agents"
    )
    independent_reproduction_success_rate: Optional[float] = Field(
        None, description="Success rate of reproduction for independent agents"
    )
    independent_first_reproduction_time: Optional[float] = Field(
        None, description="Time of first reproduction for independent agents"
    )
    independent_reproduction_efficiency: Optional[float] = Field(
        None, description="Reproduction efficiency for independent agents"
    )

    control_reproduction_attempts: Optional[int] = Field(
        None, description="Number of reproduction attempts by control agents"
    )
    control_reproduction_successes: Optional[int] = Field(
        None, description="Number of successful reproductions by control agents"
    )
    control_reproduction_failures: Optional[int] = Field(
        None, description="Number of failed reproductions by control agents"
    )
    control_reproduction_success_rate: Optional[float] = Field(
        None, description="Success rate of reproduction for control agents"
    )
    control_first_reproduction_time: Optional[float] = Field(
        None, description="Time of first reproduction for control agents"
    )
    control_reproduction_efficiency: Optional[float] = Field(
        None, description="Reproduction efficiency for control agents"
    )

    # Initial positions and resources data - these fields are dynamically generated
    # Common initial position metrics
    initial_system_count: Optional[int] = Field(
        None, description="Initial count of system agents"
    )
    initial_independent_count: Optional[int] = Field(
        None, description="Initial count of independent agents"
    )
    initial_control_count: Optional[int] = Field(
        None, description="Initial count of control agents"
    )
    initial_resource_count: Optional[int] = Field(
        None, description="Initial count of resources"
    )
    initial_resource_amount: Optional[float] = Field(
        None, description="Initial total amount of resources"
    )

    # Allow for additional fields not explicitly defined
    model_config = ConfigDict(extra="allow")


def validate_dataframe(df: pd.DataFrame) -> List[Union[DominanceDataModel, Dict]]:
    """
    Validate each row in a DataFrame against the DominanceDataModel schema.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing dominance analysis data

    Returns
    -------
    List[Union[DominanceDataModel, Dict]]
        List of validated DominanceDataModel instances or original dictionaries if validation failed
    """
    validated_data = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            # Validate the row data against the model
            model_instance = DominanceDataModel(**row_dict)
            validated_data.append(model_instance)
        except ValidationError as e:
            logger.warning("validation_error", error=str(e))
            # Include the original data if validation fails
            validated_data.append(row_dict)

    return validated_data


def dataframe_to_models(df: pd.DataFrame) -> List[DominanceDataModel]:
    """
    Convert a DataFrame to a list of DominanceDataModel instances.
    Only includes rows that pass validation.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing dominance analysis data

    Returns
    -------
    List[DominanceDataModel]
        List of validated DominanceDataModel instances
    """
    validated_models = []
    validation_errors = 0

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        try:
            # Validate the row data against the model
            model_instance = DominanceDataModel(**row_dict)
            validated_models.append(model_instance)
        except ValidationError as e:
            validation_errors += 1
            logger.warning("validation_error", error=str(e))

    if validation_errors > 0:
        logger.warning(
            "validation_errors_encountered",
            error_count=validation_errors,
            total_rows=len(df),
        )

    return validated_models

import json
from typing import List, Optional, Tuple, Union

from database.data_types import CausalAnalysis
from database.enums import AnalysisScope
from database.repositories.action_repository import ActionRepository


class CausalAnalyzer:
    """
    A class for analyzing causal relationships between agent actions and their outcomes.

    This analyzer examines sequences of actions to determine their impact and transition probabilities
    between different states.
    """

    def __init__(self, repository: ActionRepository):
        """
        Initialize the CausalAnalyzer.

        Args:
            repository (AgentActionRepository): Repository for accessing agent action data.
        """
        self.repository = repository

    def analyze(
        self,
        action_type: str,
        scope: Union[str, AnalysisScope] = AnalysisScope.SIMULATION,
        agent_id: Optional[int] = None,
        step_range: Optional[Tuple[int, int]] = None,
    ) -> CausalAnalysis:
        """
        Analyze the causal impact and state transitions for a specific action type.

        This method examines sequences of actions to:
        1. Calculate the average reward (causal impact) for the action type
        2. Determine transition probabilities between different states
        3. Consider context such as success/failure, resource changes, and action-specific details

        Args:
            action_type (str): The type of action to analyze
            scope (Union[str, AnalysisScope], optional): Scope of analysis. Defaults to AnalysisScope.SIMULATION.
            agent_id (Optional[int], optional): Specific agent to analyze. Defaults to None.
            step_range (Optional[Tuple[int, int]], optional): Range of steps to analyze. Defaults to None.

        Returns:
            CausalAnalysis: Analysis results containing:
                - action_type: The analyzed action type
                - causal_impact: Average reward for the action (e.g., 2.5 means on average,
                  this action type results in a positive reward of 2.5)
                - state_transition_probs: Dictionary mapping transition states to their probabilities
                  (e.g., {"gather|success_True": 0.7, "share|success_False": 0.3} means 70% chance
                  of transitioning to a successful gather and 30% chance of transitioning to a failed share)

        Note:
            State transitions are encoded as strings containing context information such as:
            - Success/failure status
            - Resource changes
            - Whether the action was targeted
            - Action-specific details (e.g., amounts gathered or shared)

        Example:
            A state transition string might look like:
            "gather|success_True,next_success_False,resource_change_+2.0,gathered_3,next_gathered_1"

            This indicates:
            - Next action is "gather"
            - Current action was successful
            - Next action failed
            - Resources increased by 2.0
            - Current action gathered 3 units
            - Next action gathered 1 unit

            For a complete "gather" action analysis, the result might look like:

            CausalAnalysis(
                action_type="gather",
                causal_impact=1.5,  # On average, gather actions yield +1.5 reward
                state_transition_probs={
                    "gather|success_True,resource_change_+2.0,gathered_3": 0.6,  # 60% chance
                    "share|success_True,resource_change_-1.0,shared_2": 0.3,    # 30% chance
                    "gather|success_False,resource_change_0.0": 0.1             # 10% chance
                }
            )

            This indicates:
            - Gather actions typically result in positive rewards (+1.5 on average)
            - After a gather action:
              * 60% chance of another successful gather that yields +2.0 resources
              * 30% chance of transitioning to a successful share action
              * 10% chance of a failed gather attempt
        """
        actions = self.repository.get_actions_by_scope(
            scope, agent_id, step_range=step_range
        )
        filtered_actions = [
            action for action in actions if action.action_type == action_type
        ]

        if not filtered_actions:
            return CausalAnalysis(
                action_type=action_type,
                causal_impact=0.0,
                state_transition_probs={},
            )

        causal_impact = sum(a.reward or 0 for a in filtered_actions) / len(
            filtered_actions
        )

        def parse_details(details):
            if not details:
                return {}
            if isinstance(details, str):
                try:
                    return json.loads(details)
                except json.JSONDecodeError:
                    return {}
            return details if isinstance(details, dict) else {}

        state_transitions = {}
        for i in range(len(filtered_actions) - 1):
            current = filtered_actions[i]
            next_action = filtered_actions[i + 1]

            context_parts = []

            current_details = parse_details(current.details)
            next_details = parse_details(next_action.details)

            success = current_details.get("success", False)
            next_success = next_details.get("success", False)
            context_parts.append(f"success_{success}")
            context_parts.append(f"next_success_{next_success}")

            if (
                current.resources_before is not None
                and current.resources_after is not None
            ):
                resource_change = current.resources_after - current.resources_before
                if abs(resource_change) > 0:
                    context_parts.append(f"resource_change_{resource_change:+.1f}")

            if current.action_target_id:
                context_parts.append("targeted")

            if current.action_type == "gather":
                amount = current_details.get("amount_gathered")
                if amount is not None:
                    context_parts.append(f"gathered_{amount}")
            elif current.action_type == "share":
                amount = current_details.get("amount_shared")
                if amount is not None:
                    context_parts.append(f"shared_{amount}")

            if next_action.action_type == "gather":
                amount = next_details.get("amount_gathered")
                if amount is not None:
                    context_parts.append(f"next_gathered_{amount}")
            elif next_action.action_type == "share":
                amount = next_details.get("amount_shared")
                if amount is not None:
                    context_parts.append(f"next_shared_{amount}")

            transition_key = f"{next_action.action_type}|{','.join(context_parts)}"

            if transition_key not in state_transitions:
                state_transitions[transition_key] = 0
            state_transitions[transition_key] += 1

        total_transitions = sum(state_transitions.values())
        state_transition_probs = (
            {k: v / total_transitions for k, v in state_transitions.items()}
            if total_transitions > 0
            else {}
        )

        return CausalAnalysis(
            action_type=action_type,
            causal_impact=causal_impact,
            state_transition_probs=state_transition_probs,
        )

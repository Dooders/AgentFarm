import os

import streamlit.components.v1 as components

# Declare custom component
_component_func = components.declare_component(
    "my_dashboard", path=os.path.join(os.path.dirname(__file__), "frontend/build")
)


def my_dashboard(
    step_numbers, resource_levels, health_levels, action_types, action_frequencies
):
    return _component_func(
        stepNumbers=step_numbers,
        resourceLevels=resource_levels,
        healthLevels=health_levels,
        actionTypes=action_types,
        actionFrequencies=action_frequencies,
    )

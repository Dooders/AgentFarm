import sqlite3

import pandas as pd
import streamlit as st

from dashboard import my_dashboard


# Load data from SQLite database
def load_data():
    conn = sqlite3.connect(
        "your_database_file.sqlite"
    )  # Replace with your SQLite file path
    states_query = "SELECT step_number, resource_level, current_health FROM AgentStates WHERE agent_id = ?"
    actions_query = "SELECT action_type, COUNT(action_type) AS frequency FROM AgentActions WHERE agent_id = ? GROUP BY action_type"

    agent_id = 1  # Example agent ID
    states = pd.read_sql_query(states_query, conn, params=(agent_id,))
    actions = pd.read_sql_query(actions_query, conn, params=(agent_id,))
    conn.close()

    return states, actions


# Fetch data
states, actions = load_data()

# Prepare data for the custom JavaScript component
step_numbers = states["step_number"].tolist()
resource_levels = states["resource_level"].tolist()
health_levels = states["current_health"].tolist()
action_types = actions["action_type"].tolist()
action_frequencies = actions["frequency"].tolist()

# Render the JavaScript-based dashboard
st.title("Agent Dashboard with JavaScript Charts")
my_dashboard(
    step_numbers=step_numbers,
    resource_levels=resource_levels,
    health_levels=health_levels,
    action_types=action_types,
    action_frequencies=action_frequencies,
)

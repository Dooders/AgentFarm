#!/usr/bin/env python3

"""
dominance_analysis.py

This script demonstrates how to:
1. Load simulation metadata from a master database.
2. For each simulation, open its simulation database (using its file path),
   and compute:
     - Population Dominance: Which agent type (system, independent, or control)
       has the highest count in the final simulation step.
     - Survival Dominance: Which agent type has the highest average survival time,
       where survival time is (death_time - birth_time) for dead agents.
3. Combine the simulation parameters (features) with the computed labels.
4. Train and evaluate a Random Forest classifier for each dominance definition.

Before running, ensure:
- Your master database (e.g., "master.db") contains a "simulations" table
  with simulation metadata, including a JSON field "parameters" and a field
  "simulation_db_path" (the path to the simulation's database file).
- Each simulation database contains tables for SimulationStepModel and AgentModel.
- The simulation parameters are stored as a JSON dictionary of numeric features.

Adjust connection strings and feature processing as needed.
"""

import json
import os

import pandas as pd
import sqlalchemy
from sqlalchemy.orm import sessionmaker

from farm.database.models import AgentModel, Simulation, SimulationStepModel


def compute_population_dominance(sim_session):
    """
    Compute the dominant agent type by final population.
    Query the final simulation step and choose the type with the highest count.
    """
    final_step = (
        sim_session.query(SimulationStepModel)
        .order_by(SimulationStepModel.step_number.desc())
        .first()
    )
    if final_step is None:
        return None
    # Create a dictionary of agent counts
    counts = {
        "system": final_step.system_agents,
        "independent": final_step.independent_agents,
        "control": final_step.control_agents,
    }
    # Return the key with the maximum count
    return max(counts, key=lambda k: counts.get(k, 0))


def compute_survival_dominance(sim_session):
    """
    Compute the dominant agent type by average survival time.
    For each agent, compute survival time as (death_time - birth_time) if the agent has died.
    (For agents still alive, you might want to use the final step as a proxy – here we use 0.)
    Then, for each agent type, compute the average survival time.
    Return the type with the highest average.
    """
    agents = sim_session.query(AgentModel).all()
    survival_by_type = {}
    count_by_type = {}
    for agent in agents:
        # For alive agents, we assume survival time is not defined (or could use a proxy)
        if agent.death_time is not None:
            survival = agent.death_time - agent.birth_time
        else:
            survival = 0
        survival_by_type.setdefault(agent.agent_type, 0)
        count_by_type.setdefault(agent.agent_type, 0)
        survival_by_type[agent.agent_type] += survival
        count_by_type[agent.agent_type] += 1

    avg_survival = {
        agent_type: (survival_by_type[agent_type] / count_by_type[agent_type])
        for agent_type in survival_by_type
        if count_by_type[agent_type] > 0
    }
    if not avg_survival:
        return None
    return max(avg_survival, key=lambda k: avg_survival.get(k, 0))


def load_simulation_features_and_labels(master_session):
    """
    Load simulations from the master database.
    For each simulation record:
      - Read simulation parameters (assumed to be stored as JSON)
      - Open the simulation database (using simulation_db_path)
      - Compute the dominance labels for population and survival
    Returns a Pandas DataFrame with one row per simulation.
    """
    sims = master_session.query(Simulation).all()
    data = []
    for sim in sims:
        # Get simulation parameters as a dict (if stored as JSON)
        if isinstance(sim.parameters, str):
            features = json.loads(sim.parameters)
        else:
            features = sim.parameters

        # Open the simulation database
        sim_db_path = sim.simulation_db_path  # adjust path if needed
        sim_engine = sqlalchemy.create_engine(f"sqlite:///{sim_db_path}")
        Session = sessionmaker(bind=sim_engine)
        sim_session = Session()

        # Compute dominance labels
        pop_dom = compute_population_dominance(sim_session)
        surv_dom = compute_survival_dominance(sim_session)
        sim_session.close()

        # Create a combined dictionary
        entry = {
            "simulation_id": sim.simulation_id,
            "population_dominance": pop_dom,
            "survival_dominance": surv_dom,
        }
        # Merge the feature dict into the entry
        entry.update(features)
        data.append(entry)

    return pd.DataFrame(data)


def train_classifier(X, y, label_name):
    """
    Train a Random Forest classifier and print a classification report and feature importances.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n=== Classification Report for {label_name} Dominance ===")
    print(classification_report(y_test, y_pred))

    # Print feature importances
    importances = clf.feature_importances_
    feature_names = X.columns
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    print("Feature Importances:")
    for feat, imp in feat_imp:
        print(f"{feat}: {imp:.3f}")
    return clf


def main():
    # Connect to the master database (adjust connection string as needed)
    db_path = os.path.join(os.path.dirname(__file__), "simulations/simulation.db")
    master_engine = sqlalchemy.create_engine(f"sqlite:///{db_path}")
    MasterSession = sessionmaker(bind=master_engine)
    master_session = MasterSession()

    # Load the combined features and labels from all simulations
    df = load_simulation_features_and_labels(master_session)
    master_session.close()

    if df.empty:
        print("No simulation data found.")
        return

    print("Data loaded from simulations:")
    print(df.head())

    # Determine feature columns.
    # Exclude non-feature columns (simulation_id and the dominance labels)
    feature_cols = [
        col
        for col in df.columns
        if col not in ["simulation_id", "population_dominance", "survival_dominance"]
    ]
    X = df[feature_cols]

    # For each dominance definition, train a classifier
    for label in ["population_dominance", "survival_dominance"]:
        y = df[label]
        print(f"\nTraining classifier for {label}...")
        train_classifier(X, y, label)


if __name__ == "__main__":
    main()

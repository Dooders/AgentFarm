import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def train_classifier(X, y, label_name):
    """
    Train a Random Forest classifier and print a classification report and feature importances.
    """
    # Handle categorical features with one-hot encoding
    categorical_cols = X.select_dtypes(exclude=["number"]).columns
    if not categorical_cols.empty:
        logging.info(
            f"One-hot encoding {len(categorical_cols)} categorical features..."
        )
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logging.info(f"\n=== Classification Report for {label_name} Dominance ===")
    logging.info(classification_report(y_test, y_pred))

    # Print confusion matrix
    logging.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logging.info(cm)

    # Print feature importances
    importances = clf.feature_importances_
    feature_names = X.columns
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    logging.info("\nTop 15 Feature Importances:")
    for feat, imp in feat_imp[:15]:
        logging.info(f"{feat}: {imp:.3f}")

    return clf, feat_imp


def prepare_features_for_classification(df):
    """
    Prepare features for classification by excluding non-feature columns and handling missing values.

    Args:
        df: DataFrame containing the data

    Returns:
        X: DataFrame containing only feature columns with missing values handled
        feature_cols: List of feature column names
        exclude_cols: List of excluded column names
    """
    # Exclude non-feature columns and outcome variables
    exclude_cols = [
        "iteration",
        "population_dominance",
        "survival_dominance",
        "system_agents",
        "independent_agents",
        "control_agents",
        "total_agents",
        "final_step",
    ]

    # Also exclude derived statistics columns that are outcomes, not predictors
    for prefix in ["system_", "independent_", "control_"]:
        for suffix in ["count", "alive", "dead", "avg_survival", "dead_ratio"]:
            exclude_cols.append(f"{prefix}{suffix}")

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Create an explicit copy to avoid SettingWithCopyWarning
    X = df[feature_cols].copy()

    # Handle missing values - separate numeric and non-numeric columns
    numeric_cols = X.select_dtypes(include=["number"]).columns
    categorical_cols = X.select_dtypes(exclude=["number"]).columns

    # Fill numeric columns with mean
    if not numeric_cols.empty:
        for col in numeric_cols:
            X.loc[:, col] = X[col].fillna(X[col].mean())

    # Fill categorical columns with mode (most frequent value)
    if not categorical_cols.empty:
        for col in categorical_cols:
            X.loc[:, col] = X[col].fillna(
                X[col].mode()[0] if not X[col].mode().empty else "unknown"
            )

    return X, feature_cols, exclude_cols


def run_dominance_classification(df, dominance_output_path):
    """
    Run classification analysis for dominance types.

    Args:
        df: DataFrame containing the data
        dominance_output_path: Path to save output files
    """
    # Check if we have enough data for classification
    if len(df) <= 10:
        logging.info("Not enough data for classification (need > 10 samples)")
        return

    X, feature_cols, _ = prepare_features_for_classification(df)

    if len(feature_cols) == 0:
        logging.info("No feature columns available for classification")
        return

    # Train classifiers for each dominance type
    for label in ["population_dominance", "survival_dominance"]:
        if df[label].nunique() > 1:  # Only if we have multiple classes
            logging.info(f"\nTraining classifier for {label}...")
            y = df[label]
            clf, feat_imp = train_classifier(X, y, label)

            # Plot feature importance
            from farm.analysis.dominance.plot import plot_feature_importance

            plot_feature_importance(feat_imp, label, dominance_output_path)

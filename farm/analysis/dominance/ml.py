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
    Prepare features for classification by handling missing values and selecting relevant columns.

    Args:
        df: DataFrame containing the data

    Returns:
        X: Feature matrix
        feature_cols: List of feature column names
        exclude_cols: List of excluded column names
    """
    # Check for duplicate columns
    if len(df.columns) != len(set(df.columns)):
        logging.warning("Duplicate column names detected in DataFrame")
        # Get list of duplicate columns
        duplicates = df.columns[df.columns.duplicated()].tolist()
        logging.warning(f"Duplicate columns: {duplicates}")
        # Drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        logging.info(f"Dropped duplicate columns, new shape: {df.shape}")

    # Columns to exclude from features
    exclude_cols = [
        "iteration",
        "population_dominance",
        "survival_dominance",
        "comprehensive_dominance",
    ]

    # Get feature columns (all except excluded ones)
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Create an explicit copy to avoid SettingWithCopyWarning
    X = df[feature_cols].copy()

    # Handle missing values - separate numeric and non-numeric columns
    numeric_cols = X.select_dtypes(include=["number"]).columns
    categorical_cols = X.select_dtypes(exclude=["number"]).columns

    # Fill numeric columns with mean
    if not numeric_cols.empty:
        for col in numeric_cols:
            if X[col].isna().any():  # Only fill if there are NaN values
                mean_val = X[col].mean()
                if pd.isna(mean_val):  # If mean is also NaN, use 0
                    X.loc[:, col] = X[col].fillna(0)
                else:
                    X.loc[:, col] = X[col].fillna(mean_val)

    # Fill categorical columns with mode (most frequent value)
    if not categorical_cols.empty:
        for col in categorical_cols:
            if X[col].isna().any():  # Only fill if there are NaN values
                mode_vals = X[col].mode()
                if not mode_vals.empty:
                    X.loc[:, col] = X[col].fillna(mode_vals[0])
                else:
                    X.loc[:, col] = X[col].fillna("unknown")

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

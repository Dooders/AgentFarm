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

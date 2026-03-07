"""
Model Training
Trains the malicious package detection model.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from typing import Optional, Tuple


def load_training_data(filepath: str) -> pd.DataFrame:
    """Load processed training data from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        A pandas DataFrame.
    """
    return pd.read_csv(filepath)


def prepare_features(df: pd.DataFrame, target_col: str = "is_malicious",
                     drop_cols: Optional[list] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector.

    Args:
        df: The input DataFrame.
        target_col: Name of the target column.
        drop_cols: Additional columns to drop.

    Returns:
        A tuple of (X, y) where X is the feature matrix and y is the target.
    """
    cols_to_drop = [target_col]
    if drop_cols:
        cols_to_drop.extend(drop_cols)

    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df[target_col] if target_col in df.columns else pd.Series()

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series,
                test_size: float = 0.2,
                random_state: int = 42) -> Tuple[XGBClassifier, StandardScaler, dict]:
    """Train an XGBoost classifier.

    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple of (model, scaler, split_info).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    model.fit(X_train_scaled, y_train)

    split_info = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
    }

    return model, scaler, split_info


def save_model(model: XGBClassifier, scaler: StandardScaler,
               output_dir: str = "models") -> dict:
    """Save the trained model and scaler to disk.

    Args:
        model: The trained XGBoost model.
        scaler: The fitted scaler.
        output_dir: Directory to save model artifacts.

    Returns:
        A dictionary with paths to saved files.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "sentinel_model.joblib")
    scaler_path = os.path.join(output_dir, "sentinel_scaler.joblib")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return {"model": model_path, "scaler": scaler_path}

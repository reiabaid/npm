
import sys
import os
import joblib
import pandas as pd
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Add src to path
sys.path.append(os.getcwd())

from src.model.train import load_training_data, prepare_features, train_model, save_model

def main():
    print("Loading data...")
    df = load_training_data("data/processed/dataset.csv")
    
    # In dataset.csv the column is 'label', while train.py defaults to 'is_malicious'
    # Also we should drop 'name'
    print("Preparing features...")
    X, y = prepare_features(df, target_col="label", drop_cols=["name"])
    
    print("Training model...")
    model, scaler, _ = train_model(X, y)
    
    print("Saving model to models/...")
    save_model(model, scaler, output_dir="models")
    print("Success!")

if __name__ == "__main__":
    main()

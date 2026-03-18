"""
Preprocessing Pipeline for SCOPE Model
======================================
This module defines the feature sets and builds the scikit-learn preprocessing pipeline.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Define feature lists
# Based on the output of feature_engineer.py
numerical_features = [
    'days_since_created',
    'days_since_last_update',
    'num_versions',
    'release_velocity',
    'num_maintainers',
    'description_length',
    'stargazers_count',
    'forks_count',
    'open_issues_count',
    'subscribers_count',
    'contributor_count',
    'days_since_last_commit'
]

binary_features = [
    'has_postinstall',
    'license_is_standard',
    'has_github_repo'
]

categorical_features = [] # Currently, all engineered features are numerical or binary.

def build_preprocessor() -> ColumnTransformer:
    """
    Builds and returns the scikit-learn ColumnTransformer for data preprocessing.
    
    Why this specific pipeline?
    
    1. Pipeline Order: SimpleImputer comes BEFORE StandardScaler because 
       StandardScaler cannot operate on data containing NaN values. 
       We must fill missing values first.
       
    2. Imputation Strategy (Median vs Mean): We use the median ('median') 
       instead of the mean. Many features in NPM packages (like stars, 
       forks, and downloads) are highly skewed with massive outliers 
       (e.g., React or Lodash have thousands of stars, while most have 0). 
       The mean would be heavily distorted by these outliers, whereas 
       the median is robust and provides a typical fallback value.
    """
    
    # 1. Pipeline for numerical features
    #   - Impute missing values (e.g., from pipeline bugs converted to NaN) with median
    #   - Scale features to zero mean and unit variance for models like Logistic Regression or SVM
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Pipeline for binary features
    #   - Binary features (0 or 1) are ready to go, so we just "passthrough"
    #   - Optional: If missing values exist in binary columns, an imputer with strategy='most_frequent' could be used.
    
    # Combine everything using a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('bin', 'passthrough', binary_features)
            # Categorical step would go here if we had any nominal non-binary features
        ],
        remainder='drop' # Drop any features not explicitly listed in the lists above (like 'name' or 'label')
    )
    
    return preprocessor

if __name__ == "__main__":
    preprocessor = build_preprocessor()
    print("Preprocess pipeline built successfully.")
    
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(base_dir, 'data', 'processed', 'dataset.csv')
    
    # Load dataset.csv
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the dataset has been generated before running this script.")
        exit(1)
        
    print(f"Dataset loaded with shape: {df.shape}")

    # Separate X (features) from y (label column)
    X = df.drop(columns=['label'])
    y = df['label']

    # Create test set (15%) - Set aside - do not touch until Day 8
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Split remaining data again for validation set
    # Using test_size=0.176 on the remaining 85% gives ~15% of total data for validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
    )

    print("\nVerifying class balance preserved in each split:")
    print("--- Original y ---")
    print(y.value_counts(normalize=True))
    print("\n--- Train set y ---")
    print(y_train.value_counts(normalize=True))
    print("\n--- Validation set y ---")
    print(y_val.value_counts(normalize=True))
    print("\n--- Test set y ---")
    print(y_test.value_counts(normalize=True))

    print("\nFitting preprocessor on X_train...")
    # Fit preprocessor ONLY on X_train to prevent data leakage
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    print("Transforming validation and test splits...")
    # Then transform all three splits. Never fit on val or test.
    X_val_transformed = preprocessor.transform(X_val)
    X_test_transformed = preprocessor.transform(X_test)
    
    print("\nSuccessfully preprocessed all splits.")
    print(f"X_train_transformed shape: {X_train_transformed.shape}")
    print(f"X_val_transformed shape: {X_val_transformed.shape}")
    print(f"X_test_transformed shape: {X_test_transformed.shape}")

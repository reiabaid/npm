"""
Preprocessing Pipeline for SCOPE Model
======================================
This module defines the feature sets and builds the scikit-learn preprocessing pipeline.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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
    ---------------------------
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
    print("Transformers:", preprocessor.transformers)

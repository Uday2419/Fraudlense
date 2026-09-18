"""Preprocessing pipeline for FraudLense.

Builds a reproducible ColumnTransformer that handles numeric scaling
and categorical encoding. Fit on training data only.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric and categorical columns.

    Parameters
    ----------
    X : pd.DataFrame
        Training features. Used only to detect column types.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor. Call .fit(X_train) before use.
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ],
        remainder='drop'
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> list:
    """Return human-readable names for all output features after preprocessing."""
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]

    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = ohe.get_feature_names_out(cat_cols)

    return list(num_cols) + list(cat_feature_names)
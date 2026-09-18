"""Training module for FraudLense.

Provides data preparation and model training helpers.
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)
from feature_engineering import add_features, drop_leaky_or_useless
from preprocessing import build_preprocessor

RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def prepare_data(raw_path=None):
    if raw_path is None:
        raw_path = DATA_DIR / 'credit_card_fraud_clean.csv'

    df = pd.read_csv(raw_path)
    df = add_features(df)
    df = drop_leaky_or_useless(df)

    X = df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train.to_csv(DATA_DIR / 'X_train.csv', index=False)
    X_test.to_csv(DATA_DIR / 'X_test.csv', index=False)
    y_train.to_csv(DATA_DIR / 'y_train.csv', index=False)
    y_test.to_csv(DATA_DIR / 'y_test.csv', index=False)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean()*100:.4f}%")
    print(f"Test  fraud rate: {y_test.mean()*100:.4f}%")

    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train):
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, solver='liblinear'
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_train)
    y_proba = pipeline.predict_proba(X_train)[:, 1]
    print(f"\nTraining metrics (LR):")
    print(f"  Precision: {precision_score(y_train, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_train, y_pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_train, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_train, y_proba):.4f}")
    print(f"  PR-AUC:    {average_precision_score(y_train, y_proba):.4f}")
    return pipeline


def save_model(model, name: str):
    path = MODELS_DIR / f'{name}.pkl'
    joblib.dump(model, path)
    print(f"\nSaved: {path}")
    return path


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_logistic_regression(X_train, y_train)
    save_model(model, 'lr_baseline')
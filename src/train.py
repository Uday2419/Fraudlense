"""Training module for FraudLense.

Provides data preparation and model training helpers.
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
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

def train_random_forest(X_train, y_train, X_test=None, y_test=None):
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            min_samples_split=10,
            max_features='sqrt',
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])
    pipeline.fit(X_train, y_train)

    # Train metrics
    y_pred_tr = pipeline.predict(X_train)
    y_proba_tr = pipeline.predict_proba(X_train)[:, 1]
    print(f"\nMetrics (RF) — TRAIN:")
    print(f"  Precision: {precision_score(y_train, y_pred_tr, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_train, y_pred_tr, zero_division=0):.4f}")
    print(f"  PR-AUC:    {average_precision_score(y_train, y_proba_tr):.4f}")

    # Test metrics (only if provided)
    if X_test is not None:
        y_pred_te = pipeline.predict(X_test)
        y_proba_te = pipeline.predict_proba(X_test)[:, 1]
        print(f"Metrics (RF) — TEST:")
        print(f"  Precision: {precision_score(y_test, y_pred_te, zero_division=0):.4f}")
        print(f"  Recall:    {recall_score(y_test, y_pred_te, zero_division=0):.4f}")
        print(f"  PR-AUC:    {average_precision_score(y_test, y_proba_te):.4f}")
    return pipeline

from xgboost import XGBClassifier


def train_xgboost(X_train, y_train):
    """Train an XGBoost classifier wrapped with the preprocessor."""
    preprocessor = build_preprocessor(X_train)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            eval_metric='aucpr',
            n_jobs=-1,
            random_state=RANDOM_STATE,
            tree_method='hist'
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_train)
    y_proba = pipeline.predict_proba(X_train)[:, 1]
    print(f"\nTraining metrics (XGBoost):")
    print(f"  Precision: {precision_score(y_train, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_train, y_pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_train, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_train, y_proba):.4f}")
    print(f"  PR-AUC:    {average_precision_score(y_train, y_proba):.4f}")
    return pipeline

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_data()
    model = train_logistic_regression(X_train, y_train)
    save_model(model, 'lr_baseline')
   
    
    rf_model = train_random_forest(X_train, y_train, X_test, y_test)
    save_model(rf_model, 'rf_baseline')

    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    save_model(xgb_model, 'xgb_baseline')
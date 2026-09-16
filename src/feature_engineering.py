"""Feature engineering module for FraudLense.

This module defines all engineered features used by the model pipeline.
It is imported by training and inference code to guarantee consistency.
"""

import pandas as pd
import numpy as np


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a raw transaction dataframe."""
    df = df.copy()

    # Ratio features
    df['amount_to_balance_ratio'] = df['amount_usd'] / (df['account_balance_usd'] + 1)
    df['amount_per_txn_24h'] = df['amount_usd'] / (df['txn_count_last_24h'] + 1)
    df['balance_after_txn'] = df['account_balance_usd'] - df['amount_usd']

    # Time features
    df['is_night'] = df['time_of_day_hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6, 'Saturday', 'Sunday']).astype(int)

    # Velocity
    df['txn_velocity'] = df['txn_count_last_24h'] / (df['hours_since_last_txn'] + 1)

    # Risk aggregation
    df['risk_flag_count'] = (
        df['is_foreign_transaction']
        + df['used_vpn']
        + df['ip_country_mismatch']
        + df['billing_shipping_mismatch']
        + df['is_new_merchant']
    )

    # Interactions
    df['vpn_x_foreign'] = df['used_vpn'] * df['is_foreign_transaction']
    df['vpn_x_country_mismatch'] = df['used_vpn'] * df['ip_country_mismatch']

    # Bins
    df['amount_bin'] = pd.qcut(df['amount_usd'], q=5,
                                labels=['very_low', 'low', 'medium', 'high', 'very_high'])
    df['balance_bin'] = pd.qcut(df['account_balance_usd'], q=5,
                                 labels=['very_low', 'low', 'medium', 'high', 'very_high'],
                                 duplicates='drop')

    return df


def drop_leaky_or_useless(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that should not reach the model."""
    drop_cols = ['transaction_id']
    return df.drop(columns=[c for c in drop_cols if c in df.columns])
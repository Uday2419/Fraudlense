# Dataset

**File:** `raw/credit_card_fraud_2026.csv`
**Rows:** 20,000
**Columns:** 26 (25 features + 1 target)
**Target:** `is_fraud` (binary)
**Class balance:** 19,661 non-fraud (98.31%) / 339 fraud (1.69%)
**Imbalance ratio:** ~58:1

## Columns
transaction_id, amount_usd, merchant_category, card_type, auth_method,
channel, device_type, is_foreign_transaction, hours_since_last_txn,
txn_count_last_24h, distance_from_home_km, card_age_months, customer_age,
account_balance_usd, is_new_merchant, used_vpn, ip_country_mismatch,
billing_shipping_mismatch, cvv_retry_count, velocity_score, time_of_day_hour,
day_of_week, is_ai_generated_scam_attempt, merchant_risk_score,
prior_disputes, is_fraud
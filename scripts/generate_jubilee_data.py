from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from random import choices, randint, uniform

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "jubilee_transactions.csv"
NUM_CUSTOMERS = 10_000
MIN_TRANSACTIONS_PER_CUSTOMER = 5
MAX_TRANSACTIONS_PER_CUSTOMER = 8

COUNTRY_DISTRIBUTION = {
    "Kenya": 0.7,
    "Uganda": 0.18,
    "Tanzania": 0.12,
}

REGION_MAP = {
    "Kenya": ["Nairobi Metro", "Coastal", "Rift Valley", "Western", "Mount Kenya", "North Eastern"],
    "Uganda": ["Kampala", "Central", "Western", "Northern"],
    "Tanzania": ["Dar es Salaam", "Dodoma", "Arusha", "Coastal", "Zanzibar"],
}

CHANNELS = ["branch", "mobile_app", "agency", "third_party"]
CHANNEL_PROBS = [0.28, 0.45, 0.18, 0.09]

TRANSACTION_TYPES = [
    "premium_payment",
    "claim_payout",
    "investment",
    "savings_deposit",
    "reinsurance",
    "refund",
]
TRANSACTION_WEIGHTS = [0.52, 0.15, 0.08, 0.12, 0.03, 0.07]

POLICY_TYPES = ["life", "health", "motor", "property", "savings"]
PRODUCT_LINES = ["Personal", "Commercial", "Group"]
DESTINATION_RISK_MAP = {"Kenya": "Low", "Uganda": "Medium", "Tanzania": "Medium", "Somalia": "High", "South Sudan": "High", "DRC": "High", "Rwanda": "Low"}
DESTINATION_COUNTRIES = list(DESTINATION_RISK_MAP.keys())
DESTINATION_PROBS = [0.55, 0.12, 0.08, 0.08, 0.05, 0.04, 0.08]

REFERENCE_DATE = datetime(2025, 1, 1)


def generate_transaction_amount(txn_type: str) -> float:
    if txn_type == "claim_payout":
        return round(uniform(100_000, 450_000), 2)
    if txn_type == "investment":
        return round(uniform(30_000, 250_000), 2)
    if txn_type == "reinsurance":
        return round(uniform(50_000, 400_000), 2)
    if txn_type == "savings_deposit":
        return round(uniform(5_000, 120_000), 2)
    if txn_type == "refund":
        return round(uniform(2_000, 25_000), 2)
    return round(uniform(4_000, 200_000), 2)  # premium_payment or other


def compute_rule_score(row: pd.Series) -> float:
    score = 0
    if row["amount"] > 25000:
        score += 30
    if row["amount"] > 50000:
        score += 10
    if row["destination_risk"] == "High":
        score += 25
    if row["transaction_type"] in {"claim_payout", "reinsurance"}:
        score += 25
    if row["channel"] == "third_party":
        score += 15
    score += min(row["past_flags"], 3) * 5
    if row["policy_age_days"] < 180:
        score += 5
    return min(score / 100, 1.0)


def generate_transactions() -> pd.DataFrame:
    records = []
    txn_counter = 1

    country_choices = list(COUNTRY_DISTRIBUTION.keys())
    country_weights = list(COUNTRY_DISTRIBUTION.values())

    for customer_id in range(1, NUM_CUSTOMERS + 1):
        country = choices(country_choices, weights=country_weights, k=1)[0]
        regions = REGION_MAP[country]
        policy_age_days = randint(60, 4000)
        customer_segment = choices(["Retail", "Corporate"], weights=[0.68, 0.32], k=1)[0]
        past_flags = randint(0, 4)
        num_transactions = randint(MIN_TRANSACTIONS_PER_CUSTOMER, MAX_TRANSACTIONS_PER_CUSTOMER)

        for _ in range(num_transactions):
            txn_type = choices(TRANSACTION_TYPES, weights=TRANSACTION_WEIGHTS, k=1)[0]
            channel = choices(CHANNELS, weights=CHANNEL_PROBS, k=1)[0]
            region = choices(regions, k=1)[0]
            policy_type = choices(POLICY_TYPES, k=1)[0]
            product_line = choices(PRODUCT_LINES, weights=[0.55, 0.25, 0.20], k=1)[0]
            destination_country = choices(DESTINATION_COUNTRIES, weights=DESTINATION_PROBS, k=1)[0]
            destination_risk = DESTINATION_RISK_MAP[destination_country]
            amount = generate_transaction_amount(txn_type)

            record = {
                "transaction_id": f"JUB-{txn_counter:07d}",
                "customer_id": f"C{customer_id:06d}",
                "customer_country": country,
                "channel": channel,
                "transaction_type": txn_type,
                "policy_type": policy_type,
                "product_line": product_line,
                "region": region,
                "account_age_days": policy_age_days,
                "policy_age_days": policy_age_days,
                "past_flags": past_flags,
                "transaction_date": (
                    REFERENCE_DATE + timedelta(days=randint(0, 700))
                ).strftime("%Y-%m-%d"),
                "amount": amount,
                "destination_country": destination_country,
                "destination_risk": destination_risk,
            }

            records.append(record)
            txn_counter += 1

    df = pd.DataFrame(records)
    df["rule_score"] = df.apply(compute_rule_score, axis=1)
    df["is_suspicious"] = (df["rule_score"] > 0.4).astype(int)
    df["transaction_month"] = pd.to_datetime(df["transaction_date"]).dt.to_period("M").astype(str)
    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_transactions()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {len(df):,} transactions for {NUM_CUSTOMERS:,} customers and saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

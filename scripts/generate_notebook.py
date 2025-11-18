import json
from pathlib import Path
from textwrap import dedent


def make_source(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines]


cells = []

cells.append({
    "cell_type": "markdown",
    "metadata": {"language": "markdown"},
    "source": make_source(dedent("""
# Hybrid AML Transaction Monitoring for Jubilee Insurance (Kenya)

This notebook implements the Jubilee Insurance research objectives by blending transparent rule-based checks with machine learning (XGBoost and Isolation Forest) to detect suspicious transactions denominated in Kenyan Shillings (KES).

We document scoring rationale so compliance teams can audit every flagged case while still surfacing evolving laundering patterns across Nairobi, Mombasa, and beyond.
"""))
})

cells.append({
    "cell_type": "code",
    "metadata": {"language": "python"},
    "source": make_source(dedent("""
# Core dependencies and display settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

pd.options.display.float_format = "{:,.2f}".format
sns.set_theme(style="whitegrid", palette="muted")
"""))
})

cells.append({
    "cell_type": "markdown",
    "metadata": {"language": "markdown"},
    "source": make_source(dedent("""
## Jubilee transaction ledger (Kenya, Uganda, Tanzania)

The dataset at `../data/jubilee_transactions.csv` contains 64k+ transactions sourced from the generated Jubilee Insurance ledger covering Kenya, Uganda, and Tanzania, with multiple interactions per customer.
"""))
})

cells.append({
    "cell_type": "code",
    "metadata": {"language": "python"},
    "source": make_source(dedent("""
data_path = Path("..") / "data" / "jubilee_transactions.csv"
friendly_df = pd.read_csv(data_path, parse_dates=["transaction_date"])
friendly_df["transaction_month"] = friendly_df["transaction_date"].dt.to_period("M").astype(str)
friendly_df["policy_age_days"] = friendly_df["policy_age_days"].fillna(friendly_df["account_age_days"])
friendly_df.head()
"""))
})

cells.append({
    "cell_type": "code",
    "metadata": {"language": "python"},
    "source": make_source(dedent("""
print(f"Total transactions: {len(friendly_df):,}")
print(f"Unique customers: {friendly_df['customer_id'].nunique():,}")
print("\nDestination risk mix (percent):")
print(friendly_df["destination_risk"].value_counts(normalize=True).mul(100).round(1))
print(f"\nLabelled suspicious share: {friendly_df['is_suspicious'].mean():.1%}")
print(f"Median transactions per customer: {friendly_df.groupby('customer_id').size().median():.0f}\n")
print("Top transaction types (percent):")
print(friendly_df["transaction_type"].value_counts(normalize=True).mul(100).round(1))
"""))
})

cells.append({
    "cell_type": "markdown",
    "metadata": {"language": "markdown"},
    "source": make_source(dedent("""
## Rule-based scoring for auditors

We assign explicit points for thresholds such as high-value claims, third-party channels, and destination countries flagged as high risk. This produces a rule score between 0 and 1 that compliance officers can inspect verbatim.
"""))
})

cells.append({
    "cell_type": "code",
    "metadata": {"language": "python"},
    "source": make_source(dedent("""
def compute_rule_score(row):
    score = 0
    if row["amount"] > 25000:
        score += 30
    if row["amount"] > 50000:
        score += 10
    if row["destination_risk"] == "High":
        score += 25
    if row["transaction_type"] in {"claim_payout", "reinsurance"}:
        score += 20
    if row["channel"] == "third_party":
        score += 15
    score += min(row["past_flags"], 3) * 5
    if row["account_age_days"] < 180:
        score += 5
    return min(score, 100)

friendly_df["rule_score"] = friendly_df.apply(compute_rule_score, axis=1) / 100
friendly_df[["transaction_id", "amount", "transaction_type", "channel", "destination_risk", "rule_score"]].sort_values("rule_score", ascending=False).head()
"""))
})

cells.append({
    "cell_type": "markdown",
    "metadata": {"language": "markdown"},
    "source": make_source(dedent("""
## Machine learning pipeline with XGBoost

We now train an XGBoost classifier on core transaction features (amount, transaction type, channel, destination, region) so the model captures complex, non-linear interactions that are harder to encode in rules.
"""))
})

cells.append({
    "cell_type": "code",
    "metadata": {"language": "python"},
    "source": make_source(dedent("""
feature_cols = ["amount", "transaction_type", "channel", "destination_risk", "region", "account_age_days", "past_flags"]
target = "is_suspicious"
numeric_cols = ["amount", "account_age_days", "past_flags"]
categorical_cols = ["transaction_type", "channel", "destination_risk", "region"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols),
    ],
    remainder="passthrough"
)

model_pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
            ),
        ),
    ]
)

X = friendly_df[feature_cols]
y = friendly_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
y_proba = model_pipeline.predict_proba(X_test)[:, 1]

eval_df = X_test.copy()
eval_df["rule_score"] = friendly_df.loc[eval_df.index, "rule_score"]
eval_df["ml_probability"] = y_proba
eval_df["suspected_by_model"] = y_pred

eval_df.sort_values("ml_probability", ascending=False).head()
"""))
})

cells.append({
    "cell_type": "code",
    "metadata": {"language": "python"},
    "source": make_source(dedent("""
print("Confusion matrix (threshold = 0.5):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))

importance_df = pd.Series(
    model_pipeline.named_steps["classifier"].feature_importances_,
    index=model_pipeline.named_steps["preprocessor"].get_feature_names_out(),
).sort_values(ascending=False)
importance_df.head(8)
"""))
})

cells.append({
    "cell_type": "markdown",
    "metadata": {"language": "markdown"},
    "source": make_source(dedent("""
## Isolation Forest and hybrid scoring

We overlay Isolation Forest anomaly detection to capture highly unusual combinations, then blend rule scores, model probabilities, and anomaly flags into a single hybrid risk tier for Jubilee's risk committee.
"""))
})

cells.append({
    "cell_type": "code",
    "metadata": {"language": "python"},
    "source": make_source(dedent("""
ml_probs_full = model_pipeline.predict_proba(X)[:, 1]
feature_matrix = model_pipeline.named_steps["preprocessor"].transform(X)
iso = IsolationForest(contamination=0.04, random_state=42)
iso.fit(feature_matrix)

friendly_df["ml_probability"] = ml_probs_full
friendly_df["anomaly_score"] = -iso.score_samples(feature_matrix)
friendly_df["anomaly_flag"] = iso.predict(feature_matrix) == -1
friendly_df["hybrid_risk"] = (
    0.5 * friendly_df["rule_score"]
    + 0.4 * friendly_df["ml_probability"]
    + 0.1 * friendly_df["anomaly_flag"].astype(float)
)
risk_bins = [0, 0.25, 0.5, 0.75, 1.0]
risk_labels = ["Very Low", "Moderate", "Elevated", "Critical"]
friendly_df["risk_tier"] = pd.cut(
    friendly_df["hybrid_risk"],
    bins=risk_bins,
    labels=risk_labels,
    include_lowest=True,
)

friendly_df[["transaction_id", "amount", "rule_score", "ml_probability", "anomaly_flag", "hybrid_risk", "risk_tier"]].sort_values("hybrid_risk", ascending=False).head()
"""))
})

cells.append({
    "cell_type": "markdown",
    "metadata": {"language": "markdown"},
    "source": make_source(dedent("""
## Summary of findings

This hybrid system preserves explainability through explicit rule scores while leveraging XGBoost to capture complex interactions and Isolation Forest to highlight unusual patterns. Jubilee’s compliance teams can focus remediation on the highest-risk transactions and provide transparent reasoning to regulators.
"""))
})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

Path("notebooks/hybrid_aml_jubilee.ipynb").write_text(json.dumps(notebook, indent=2))

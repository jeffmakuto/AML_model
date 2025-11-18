from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path("data") / "jubilee_transactions.csv"


@st.cache_data
def load_transactions() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["transaction_date"])
    df["transaction_month"] = df["transaction_date"].dt.to_period("M").astype(str)
    df["risk_tier"] = pd.cut(
        df["rule_score"],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        include_lowest=True,
        labels=["Very Low", "Moderate", "Elevated", "Critical"],
    )
    return df


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Refine view")
    country = st.sidebar.multiselect(
        "Customer country", options=df["customer_country"].unique(), default=df["customer_country"].unique()
    )
    region = st.sidebar.multiselect("Region", options=df["region"].unique(), default=df["region"].unique())
    txn_type = st.sidebar.multiselect(
        "Transaction type", options=df["transaction_type"].unique(), default=df["transaction_type"].unique()
    )
    product_line = st.sidebar.multiselect(
        "Product line", options=df["product_line"].unique(), default=df["product_line"].unique()
    )

    filtered = df[  # noqa: E203
        df["customer_country"].isin(country)
        & df["region"].isin(region)
        & df["transaction_type"].isin(txn_type)
        & df["product_line"].isin(product_line)
    ]
    return filtered


def render_header(df: pd.DataFrame) -> None:
    st.title("Jubilee Insurance Hybrid AML Dashboard")
    st.caption("Rule-based and machine learning insights derived from Jubilee's synthetic cross-border ledger.")
    cols = st.columns(3)
    cols[0].metric("Total transactions", f"{len(df):,}")
    cols[1].metric("Active customers", f"{df['customer_id'].nunique():,}")
    cols[2].metric("Avg. rule score", f"{df['rule_score'].mean():.2f}")


def render_charts(filtered: pd.DataFrame) -> None:
    st.subheader("Risk mix")
    risk_counts = (
        filtered["risk_tier"].value_counts(normalize=True)
        .reindex(["Very Low", "Moderate", "Elevated", "Critical"])
        .fillna(0)
        * 100
    )
    st.bar_chart(risk_counts)

    st.subheader("Monthly volume")
    monthly = filtered.groupby("transaction_month")["transaction_id"].count()
    st.line_chart(monthly)

    st.subheader("Top critical transactions")
    top = filtered.sort_values(["rule_score", "amount"], ascending=[False, False]).head(20)
    st.dataframe(
        top[
            [
                "transaction_id",
                "customer_id",
                "transaction_date",
                "transaction_type",
                "channel",
                "amount",
                "rule_score",
                "risk_tier",
                "destination_country",
                "destination_risk",
            ]
        ],
        use_container_width=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Hybrid AML", page_icon="🛡️", layout="wide"
    )
    df = load_transactions()
    render_header(df)
    filtered = filter_dataframe(df)
    if filtered.empty:
        st.warning("No rows match the selected filters; please expand the filter criteria.")
        return
    render_charts(filtered)
    st.markdown(
        "---\nStreamlit data powered by `data/jubilee_transactions.csv`. Update the ledger via `python scripts/generate_jubilee_data.py`."
    )


if __name__ == "__main__":
    main()

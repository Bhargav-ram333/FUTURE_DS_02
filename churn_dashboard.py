import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

st.set_page_config(page_title="Churn Analysis Dashboard", layout="wide")
st.title("Telco Customer Churn Analysis Dashboard")

# ==========================
# Load Data
# ==========================
@st.cache_data
def load_data():
    if not os.path.exists("cleaned_telco_churn.csv"):
        return pd.DataFrame()

    df = pd.read_csv("cleaned_telco_churn.csv")

    if df.empty:
        return df

    # Clean TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Clean Churn
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.strip().str.upper()
        df["Churn_num"] = df["Churn"].map({"YES": 1, "NO": 0})
        df = df.dropna(subset=["Churn_num"])
        df["Churn_num"] = df["Churn_num"].astype(int)

    # Tenure Group
    if "tenure" in df.columns:
        df["TenureGroup"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 1000],
            labels=["0-12", "12-24", "24-48", "48+"],
            include_lowest=True
        )

    return df


df = load_data()

# ==========================
# Stop if no data
# ==========================
if df.empty:
    st.error("Dataset not found or empty. Please check cleaned_telco_churn.csv.")
    st.stop()

# ==========================
# KPIs
# ==========================
st.header("Key Performance Indicators")

overall_churn_rate = df["Churn_num"].mean() * 100
avg_tenure = df["tenure"].mean()
avg_tenure_churned = df[df["Churn_num"] == 1]["tenure"].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Overall Churn Rate", f"{overall_churn_rate:.2f}%")
col2.metric("Average Tenure (Months)", f"{avg_tenure:.2f}")
col3.metric("Avg Tenure (Churned)", f"{avg_tenure_churned:.2f}")

st.divider()

# ==========================
# Row 1
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Contract Type")
    contract_churn = df.groupby("Contract")["Churn_num"].mean() * 100
    if not contract_churn.empty:
        fig, ax = plt.subplots()
        contract_churn.plot(kind="bar", ax=ax)
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)
    else:
        st.warning("No Contract data available.")

with col2:
    st.subheader("Churn Rate by Payment Method")
    payment_churn = df.groupby("PaymentMethod")["Churn_num"].mean() * 100
    if not payment_churn.empty:
        fig, ax = plt.subplots()
        payment_churn.plot(kind="bar", ax=ax)
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)
    else:
        st.warning("No Payment Method data available.")

st.divider()

# ==========================
# Row 2
# ==========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Tenure Group")
    tenure_churn = df.groupby("TenureGroup", observed=True)["Churn_num"].mean() * 100
    if not tenure_churn.empty:
        fig, ax = plt.subplots()
        tenure_churn.plot(kind="bar", ax=ax)
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)
    else:
        st.warning("No Tenure Group data available.")

with col2:
    st.subheader("Monthly Charges vs Churn")
    if df["Churn"].nunique() > 1:
        fig, ax = plt.subplots()
        sns.boxplot(
            x="Churn",
            y="MonthlyCharges",
            data=df,
            ax=ax
        )
        st.pyplot(fig)
    else:
        st.warning("Not enough categories for boxplot.")

st.divider()

# ==========================
# Internet Service
# ==========================
st.subheader("Churn Rate by Internet Service")
internet_churn = df.groupby("InternetService")["Churn_num"].mean() * 100
if not internet_churn.empty:
    fig, ax = plt.subplots()
    internet_churn.plot(kind="bar", ax=ax)
    ax.set_ylabel("Churn Rate (%)")
    st.pyplot(fig)
else:
    st.warning("No Internet Service data available.")

st.divider()

# ==========================
# Cohort Table
# ==========================
st.header("Cohort Retention Rates by Tenure Group")

cohort = df.groupby("TenureGroup", observed=True)["Churn_num"].agg(["count", "sum"])
if not cohort.empty:
    cohort["RetentionRate (%)"] = (1 - cohort["sum"] / cohort["count"]) * 100
    cohort.rename(columns={
        "count": "Total Customers",
        "sum": "Churned Customers"
    }, inplace=True)
    st.dataframe(cohort.style.format({"RetentionRate (%)": "{:.2f}%"}))
else:
    st.warning("No cohort data available.")

st.divider()

# ==========================
# High Risk Segment
# ==========================
st.header("⚠️ High-Risk Customer Segment")

high_risk = df[
    (df["Contract"] == "Month-to-month") &
    (df["tenure"] < 12) &
    (df["MonthlyCharges"] > df["MonthlyCharges"].median())
]

col1, col2 = st.columns(2)
col1.metric("High-Risk Customer Count", len(high_risk))

hr_rate = high_risk["Churn_num"].mean() * 100 if len(high_risk) > 0 else 0
col2.metric("High-Risk Churn Rate", f"{hr_rate:.2f}%")

display_cols = [c for c in [
    "customerID", "gender", "tenure",
    "Contract", "MonthlyCharges",
    "TotalCharges", "Churn"
] if c in high_risk.columns]

if not high_risk.empty:
    st.dataframe(high_risk[display_cols].head(10).reset_index(drop=True))
else:
    st.info("No high-risk customers found.")

st.divider()

# ==========================
# Insights
# ==========================
st.header("💡 Key Insights")
st.markdown("""
1. Month-to-month customers churn the most.
2. Customers in first 12 months show highest churn.
3. Higher monthly charges increase churn probability.
4. Fiber optic users churn more than DSL users.
5. Retention improves with long-term contracts.
""")

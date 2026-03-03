import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Analysis Dashboard", layout="wide")

st.title("Telco Customer Churn Analysis Dashboard")

# ==========================================
# Load Data
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_telco_churn.csv")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df["Churn"] = df["Churn"].astype(str).str.strip().str.upper()
    df["Churn_num"] = df["Churn"].map({"YES": 1, "NO": 0})
    df = df.dropna(subset=["Churn_num"])
    df["Churn_num"] = df["Churn_num"].astype(int)

    # 👇 REPLACED TENURE GROUP CODE
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 1000],
        labels=["0-12", "12-24", "24-48", "48+"],
        include_lowest=True
    )

    return df


df = load_data()

# ==========================================
# KPI Section
# ==========================================
overall_churn_rate = df["Churn_num"].mean() * 100
avg_tenure = df["tenure"].mean()
avg_tenure_churned = df[df["Churn_num"] == 1]["tenure"].mean()

st.header("Key Performance Indicators")

col1, col2, col3 = st.columns(3)

col1.metric("Overall Churn Rate", f"{overall_churn_rate:.2f}%")
col2.metric("Average Tenure (Months)", f"{avg_tenure:.2f}")
col3.metric("Avg Tenure (Churned)", f"{avg_tenure_churned:.2f}")

st.divider()

# ==========================================
# Row 1
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Contract Type")
    contract_churn = df.groupby("Contract")["Churn_num"].mean() * 100
    fig, ax = plt.subplots()
    contract_churn.plot(kind="bar", ax=ax, color="lightcoral")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Contract Type")
    st.pyplot(fig)

with col2:
    st.subheader("Churn Rate by Payment Method")
    payment_churn = df.groupby("PaymentMethod")["Churn_num"].mean() * 100
    fig, ax = plt.subplots()
    payment_churn.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Payment Method")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

st.divider()

# ==========================================
# Row 2
# ==========================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Tenure Group")
    tenure_churn = df.groupby("TenureGroup", observed=True)["Churn_num"].mean() * 100
    fig, ax = plt.subplots()
    tenure_churn.plot(kind="bar", ax=ax, color="mediumseagreen")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Tenure Group")
    st.pyplot(fig)

with col2:
    st.subheader("Monthly Charges vs Churn")
    fig, ax = plt.subplots()
    sns.boxplot(
        x="Churn",
        y="MonthlyCharges",
        data=df,
        palette="Set2",
        ax=ax
    )
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges ($)")
    st.pyplot(fig)

st.divider()

# ==========================================
# Internet Service
# ==========================================
st.subheader("Churn Rate by Internet Service")

internet_churn = df.groupby("InternetService")["Churn_num"].mean() * 100
fig, ax = plt.subplots()
internet_churn.plot(kind="bar", ax=ax, color="mediumpurple")
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Internet Service Type")
st.pyplot(fig)

st.divider()

# ==========================================
# Cohort Retention
# ==========================================
st.header("Cohort Retention Rates by Tenure Group")

cohort = df.groupby("TenureGroup", observed=True)["Churn_num"].agg(["count", "sum"])
cohort["RetentionRate (%)"] = (1 - (cohort["sum"] / cohort["count"])) * 100
cohort.rename(columns={
    "count": "Total Customers",
    "sum": "Churned Customers"
}, inplace=True)

st.dataframe(cohort.style.format({"RetentionRate (%)": "{:.2f}%"}))

st.divider()

# ==========================================
# High Risk Segment
# ==========================================
st.header("⚠️ High-Risk Customer Segment")

st.write("""
Customers on **Month-to-Month** contracts,
with **tenure < 12 months**
and **above-median monthly charges**
are flagged as high risk.
""")

high_risk = df[
    (df["Contract"] == "Month-to-month") &
    (df["tenure"] < 12) &
    (df["MonthlyCharges"] > df["MonthlyCharges"].median())
]

col1, col2 = st.columns(2)

col1.metric("High-Risk Customer Count", len(high_risk))

if len(high_risk) > 0:
    hr_rate = high_risk["Churn_num"].mean() * 100
else:
    hr_rate = 0

col2.metric("High-Risk Churn Rate", f"{hr_rate:.2f}%")

display_cols = [
    col for col in
    ["customerID", "gender", "tenure", "Contract",
     "MonthlyCharges", "TotalCharges", "Churn"]
    if col in high_risk.columns
]

st.dataframe(high_risk[display_cols].head(10).reset_index(drop=True))

st.divider()

# ==========================================
# Insights
# ==========================================
st.header("💡 Key Insights")

st.markdown("""
1. Month-to-month customers churn the most.
2. Customers in their first 12 months show highest churn.
3. Higher monthly charges increase churn probability.
4. Fiber optic users churn more than DSL users.
5. Retention improves with long-term contracts.
""")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Analysis Dashboard", layout="wide")

st.title("Telco Customer Churn Analysis Dashboard")

# Load and process dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_telco_churn.csv")
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    df.drop_duplicates(inplace=True)
    
    # Map Churn values for calculations while keeping original for display labels if needed
    df['Churn_Label'] = df['Churn']
    df['Churn_num'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Create Tenure Group binning
    bins = [0, 12, 24, 48, 72]
    labels = ['0-12', '12-24', '24-48', '48+']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    
    return df

df = load_data()

# Calculate KPIs
overall_churn_rate = df['Churn_num'].mean() * 100
avg_tenure = df['tenure'].mean()
avg_tenure_churned = df[df['Churn'] == 'Yes']['tenure'].mean()

st.header("Key Performance Indicators")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Overall Churn Rate", f"{overall_churn_rate:.2f}%")
with col2:
    st.metric("Average Tenure (Months)", f"{avg_tenure:.2f}")
with col3:
    st.metric("Avg Tenure (Churned)", f"{avg_tenure_churned:.2f}")

st.divider()

# Row 1 of Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Contract Type")
    contract_churn = df.groupby('Contract')['Churn_num'].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    contract_churn.plot(kind='bar', ax=ax, color='lightcoral')
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Contract Type")
    plt.xticks(rotation=0)
    st.pyplot(fig)

with col2:
    st.subheader("Churn Rate by Payment Method")
    payment_churn = df.groupby('PaymentMethod')['Churn_num'].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    payment_churn.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Payment Method")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.divider()

# Row 2 of Visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Tenure Group")
    tenure_churn = df.groupby('TenureGroup', observed=True)['Churn_num'].mean() * 100
    fig, ax = plt.subplots(figsize=(8, 5))
    tenure_churn.plot(kind='bar', ax=ax, color='mediumseagreen')
    ax.set_ylabel("Churn Rate (%)")
    ax.set_xlabel("Tenure Group (Months)")
    plt.xticks(rotation=0)
    st.pyplot(fig)

with col2:
    st.subheader("Monthly Charges vs Churn")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        x="Churn",
        y="MonthlyCharges",
        data=df
    )
    ax.set_xlabel("Churn")
    ax.set_ylabel("Monthly Charges ($)")
    st.pyplot(fig)

st.divider()

# Row 3: Churn by Internet Service
st.subheader("Churn Rate by Internet Service")
internet_churn = df.groupby('InternetService')['Churn_num'].mean() * 100
fig, ax = plt.subplots(figsize=(8, 5))
internet_churn.plot(kind='bar', ax=ax, color='mediumpurple')
ax.set_ylabel("Churn Rate (%)")
ax.set_xlabel("Internet Service Type")
plt.xticks(rotation=0)
st.pyplot(fig)

st.divider()

# Data Tables (Cohort Retention)
st.header("Cohort Retention Rates by Tenure Group")

cohort = df.groupby('TenureGroup', observed=True)['Churn_num'].agg(['count', 'sum'])
cohort['RetentionRate (%)'] = (1 - cohort['sum'] / cohort['count']) * 100
cohort = cohort.rename(columns={'count': 'Total Customers', 'sum': 'Churned Customers'})

st.dataframe(cohort.style.format({'RetentionRate (%)': "{:.2f}%"}), width='stretch')

st.divider()

# High-Risk Customer Segment
st.header("⚠️ High-Risk Customer Segment")
st.write("Customers on **Month-to-Month** contracts, with **tenure < 12 months** and **above-median monthly charges** are flagged as high risk.")

high_risk = df[
    (df['Contract'] == 'Month-to-month') &
    (df['tenure'] < 12) &
    (df['MonthlyCharges'] > df['MonthlyCharges'].median())
]

col1, col2 = st.columns(2)
with col1:
    st.metric("High-Risk Customer Count", len(high_risk))
with col2:
    high_risk_churn_rate = high_risk['Churn_num'].mean() * 100
    st.metric("High-Risk Churn Rate", f"{high_risk_churn_rate:.2f}%")

st.dataframe(
    high_risk[['customerID', 'gender', 'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges', 'Churn_Label']]
    .head(10)
    .reset_index(drop=True),
    width='stretch'
)

st.divider()

# Key Insights
st.header("💡 Key Insights")
st.markdown("""
Based on the churn analysis, the following patterns stand out:

1. **Month-to-month customers churn the most** — the churn rate for month-to-month contracts (~43%) is dramatically higher than one-year (~11%) and two-year (~3%) contracts.
2. **Customers in their first 12 months have the highest churn** — nearly half of new customers leave within the first year.
3. **High monthly charge users are more likely to leave** — churned customers pay significantly higher monthly charges on average.
4. **Customers without tech support churn more** — adding tech support services may improve retention.
5. **Fiber optic subscribers churn more** — despite faster speeds, fiber optic customers have notably higher churn rates than DSL users.
""")

import pickle
import pandas as pd
import streamlit as st

from detection import detect_churn

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📞", layout="centered")

# Load saved model and training column schema
@st.cache_resource
def load_artifacts():
    with open("churn_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    with open("model_columns.pkl", "rb") as f:
        loaded_columns = pickle.load(f)
    return loaded_model, loaded_columns


model, model_columns = load_artifacts()

st.title("📞 Customer Churn Prediction System")
st.caption("ML probability scoring combined with defined, data-backed churn risk criteria.")

# Sidebar: Detection Criteria & Threshold Controls
with st.sidebar:
    st.header("Detection Criteria")
    threshold = st.slider(
        "Churn Probability Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Customers at or above this probability are flagged as likely to churn.",
    )
    enable_rule_override = st.checkbox(
        "Flag Critical Rule Breaches",
        value=True,
        help="Flag customers who breach critical thresholds even if ML probability is below the slider.",
    )
    st.markdown(
        """
**Risk Tiers (Probability)**
- **Low Risk:** `< 30%`
- **Moderate Risk:** `30% – 59%`
- **High Risk:** `60% – 84%`
- **Critical Risk:** `>= 85%`

**Critical Rule Thresholds**
- Contract Length = `Monthly`
- Support Calls `> 5`
- Payment Delay `> 20 days`
- Total Spend `< $500`

**Warning Thresholds**
- Support Calls `4–5`
- Payment Delay `15–20 days`
- Total Spend `$500–$599`
- Age `> 50`
- Usage Frequency `<= 5`
- Last Interaction `>= 20 days`
"""
    )

st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    usage = st.number_input("Usage Frequency", min_value=0, max_value=100, value=15)
    last_interaction = st.number_input("Last Interaction (Days Ago)", min_value=0, max_value=60, value=10)

with col2:
    support_calls = st.number_input("Support Calls", min_value=0, max_value=20, value=1)
    payment_delay = st.number_input("Payment Delay (Days)", min_value=0, max_value=60, value=5)
    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
    total_spend = st.number_input("Total Spend ($)", min_value=0, max_value=10000, value=650)

# Build model input dataframe
input_data = {
    "Age": age,
    "Tenure": tenure,
    "Usage Frequency": usage,
    "Support Calls": support_calls,
    "Payment Delay": payment_delay,
    "Total Spend": total_spend,
    "Last Interaction": last_interaction,
    "Gender_Male": 1 if gender == "Male" else 0,
    "Subscription Type_Premium": 1 if subscription == "Premium" else 0,
    "Subscription Type_Standard": 1 if subscription == "Standard" else 0,
    "Contract Length_Monthly": 1 if contract == "Monthly" else 0,
    "Contract Length_Quarterly": 1 if contract == "Quarterly" else 0,
}

input_df = pd.DataFrame([input_data]).reindex(columns=model_columns, fill_value=0)

customer_profile = {
    "age": age,
    "gender": gender,
    "tenure": tenure,
    "usage": usage,
    "support_calls": support_calls,
    "payment_delay": payment_delay,
    "subscription": subscription,
    "contract": contract,
    "total_spend": total_spend,
    "last_interaction": last_interaction,
}

if st.button("Predict Churn", type="primary"):
    result = detect_churn(
        model=model,
        input_df=input_df,
        customer=customer_profile,
        threshold=threshold,
        enable_rule_override=enable_rule_override,
    )

    prob = result["probability"]
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Churn Probability", f"{prob:.1%}")
    m2.metric("Risk Tier", result["risk_tier"])
    m3.metric("Detection Threshold", f"{threshold:.0%}")

    st.progress(min(max(prob, 0.0), 1.0))

    if result["is_churn"]:
        reason = (
            "ML probability and critical risk rules triggered."
            if prob >= threshold and result["rule_triggered"]
            else "Critical risk rule threshold breached."
            if result["rule_triggered"]
            else f"Churn probability ({prob:.1%}) meets/exceeds threshold ({threshold:.0%})."
        )
        st.error(f"⚠️ Customer is likely to CHURN — {reason}")
    else:
        st.success(f"✅ Customer is likely to STAY (below {threshold:.0%} threshold, no critical breaches)")

    if result["critical_signals"]:
        st.markdown("#### 🚨 Critical Risk Criteria Triggered")
        for sig in result["critical_signals"]:
            st.error(f"**{sig.factor}:** {sig.detail}")

    if result["warning_signals"]:
        st.markdown("#### ⚠️ Warning Criteria Triggered")
        for sig in result["warning_signals"]:
            st.warning(f"**{sig.factor}:** {sig.detail}")

    if result["positive_signals"]:
        st.markdown("#### ✅ Protective Factors")
        for sig in result["positive_signals"]:
            st.info(f"**{sig.factor}:** {sig.detail}")

    if result["recommendations"]:
        st.markdown("#### 💡 Recommended Retention Actions")
        for rec in result["recommendations"]:
            st.write(f"- {rec}")

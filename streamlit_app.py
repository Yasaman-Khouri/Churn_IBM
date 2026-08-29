import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Start Page Styling
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title(" Telecom Customer Churn Risk Predictor")
st.markdown(
    """
Predict customer churn probability using an optimized ML classifier.
Adjust the **decision threshold** in the sidebar to prioritize recall.
"""
)


# Loading Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("artifacts/best_xgb.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    training_columns = joblib.load("artifacts/training_columns.pkl")
    numeric_cols = joblib.load("artifacts/numeric_cols.pkl")
    return model, scaler, training_columns, numeric_cols

try:
    model, scaler, training_columns, numeric_cols = load_artifacts()
except FileNotFoundError as e:
    st.error(
        "❌ Artifacts not found! Please run the training/export script first to generate files in `artifacts/`."
    )
    st.stop()


# Sidebar Design: Business & Threshold Configuration
st.sidebar.header("⚙️ Model Settings")
threshold = st.sidebar.slider(
    "Decision Threshold (Recall Tuning)",
    min_value=0.10,
    max_value=0.90,
    value=0.40,
    step=0.05,
    help="Default is 0.40. Lower thresholds catch more churning customers (higher recall) at the cost of some precision.",
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Telecom Retention Strategy:**\n"
    "Setting threshold to **0.40** triggers early retention offers "
    "before high-value customers complete their cancellation."
)


# User Input
with st.form("customer_input_form"):
    st.subheader("👤 Customer Demographic & Account Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12, step=1)

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with col3:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    st.subheader("💳 Financial Data")
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    with f_col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=780.0, step=10.0)

    submitted = st.form_submit_button("🔍 Calculate Churn Risk")


# Inference
if submitted:
    # 1. Assemble raw input into a 1-row DataFrame
    raw_input_dict = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    input_df = pd.DataFrame([raw_input_dict])

    # 2. Apply get_dummies
    input_encoded = pd.get_dummies(input_df, drop_first=True, dtype=int)

    # 3. CRITICAL STEP: Reindex to match the training feature schema exactly
    #    Missing dummy categories become 0; unexpected columns are ignored.
    input_aligned = input_encoded.reindex(columns=training_columns, fill_value=0)

    # 4. Scale numeric columns using the fitted scaler
    input_aligned[numeric_cols] = scaler.transform(input_aligned[numeric_cols])

    # 5. Predict probabilities
    churn_proba = model.predict_proba(input_aligned)[0, 1]
    is_churn = int(churn_proba >= threshold)


    # UI Presentation
    st.markdown("---")
    st.subheader("📊 Assessment Result")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(
            label="Churn Probability",
            value=f"{churn_proba:.1%}",
            delta=f"Threshold: {threshold:.2f}",
            delta_color="inverse" if churn_proba >= threshold else "normal",
        )
        st.progress(float(churn_proba))

    with res_col2:
        if is_churn == 1:
            st.error("⚠️ **HIGH CHURN RISK**")
            st.markdown(
                f"""
                - **Probability:** `{churn_proba:.3f}` exceeds decision threshold `{threshold:.2f}`.
                - **Recommended Action:** Escalate to customer retention team. Offer contract renewal discounts or service upgrades.
                """
            )
        else:
            st.success("✅ **LOW CHURN RISK (LOYAL CUSTOMER)**")
            st.markdown(
                f"""
                - **Probability:** `{churn_proba:.3f}` is below threshold `{threshold:.2f}`.
                - **Recommended Action:** Customer is stable. Standard support and upsell campaigns apply.
                """
            )
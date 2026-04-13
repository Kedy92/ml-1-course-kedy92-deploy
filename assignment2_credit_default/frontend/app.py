import os

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.getenv("API_URL", "https://credit-card-default-api.onrender.com").rstrip("/")

st.set_page_config(
    page_title="Credit Default Predictor",
    page_icon="💳",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(53, 94, 255, 0.10), transparent 28%),
            radial-gradient(circle at bottom right, rgba(33, 150, 243, 0.10), transparent 30%),
            linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
    }
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #101828 0%, #1d3557 60%, #3a86ff 100%);
        color: white;
        box-shadow: 0 18px 40px rgba(16, 24, 40, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.2rem;
    }
    .hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 1rem;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 16px;
        background: white;
        border: 1px solid rgba(15, 23, 42, 0.06);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
    }
    </style>
    <div class="hero">
      <h1>Credit Default Predictor</h1>
      <p>Compare the PyTorch MLP and CatBoost model on the same credit-card customer profile.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"API base URL: {API_BASE_URL}")

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Customer Profile")
    model_choice = st.radio(
        "Prediction model",
        options=["tree", "mlp"],
        horizontal=True,
        format_func=lambda value: "CatBoost Tree" if value == "tree" else "PyTorch MLP",
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        limit_bal = st.number_input("Credit limit", min_value=10000, max_value=1000000, value=120000, step=10000)
        sex = st.selectbox("Sex", ["female", "male"])
        education = st.selectbox("Education", ["graduate_school", "university", "high_school", "other", "unknown"])
    with col2:
        marriage = st.selectbox("Marriage", ["single", "married", "other", "unknown"])
        age = st.slider("Age", min_value=18, max_value=80, value=26)
        pay_0 = st.slider("Recent repayment status (PAY_0)", min_value=-2, max_value=8, value=-1)
    with col3:
        pay_2 = st.slider("PAY_2", min_value=-2, max_value=8, value=2)
        pay_3 = st.slider("PAY_3", min_value=-2, max_value=8, value=0)
        pay_4 = st.slider("PAY_4", min_value=-2, max_value=8, value=0)

    repayment_cols = st.columns(2)
    with repayment_cols[0]:
        pay_5 = st.slider("PAY_5", min_value=-2, max_value=8, value=0)
        pay_6 = st.slider("PAY_6", min_value=-2, max_value=8, value=2)
        bill_amt1 = st.number_input("Bill amount 1", value=2682.0, step=100.0)
        bill_amt2 = st.number_input("Bill amount 2", value=1725.0, step=100.0)
        bill_amt3 = st.number_input("Bill amount 3", value=2682.0, step=100.0)
        bill_amt4 = st.number_input("Bill amount 4", value=3272.0, step=100.0)
    with repayment_cols[1]:
        bill_amt5 = st.number_input("Bill amount 5", value=3455.0, step=100.0)
        bill_amt6 = st.number_input("Bill amount 6", value=3261.0, step=100.0)
        pay_amt1 = st.number_input("Payment amount 1", value=0.0, step=100.0)
        pay_amt2 = st.number_input("Payment amount 2", value=1000.0, step=100.0)
        pay_amt3 = st.number_input("Payment amount 3", value=1000.0, step=100.0)
        pay_amt4 = st.number_input("Payment amount 4", value=1000.0, step=100.0)

    payment_tail_cols = st.columns(2)
    with payment_tail_cols[0]:
        pay_amt5 = st.number_input("Payment amount 5", value=0.0, step=100.0)
    with payment_tail_cols[1]:
        pay_amt6 = st.number_input("Payment amount 6", value=2000.0, step=100.0)

    payload = {
        "limit_bal": int(limit_bal),
        "sex": sex,
        "education": education,
        "marriage": marriage,
        "age": int(age),
        "pay_0": int(pay_0),
        "pay_2": int(pay_2),
        "pay_3": int(pay_3),
        "pay_4": int(pay_4),
        "pay_5": int(pay_5),
        "pay_6": int(pay_6),
        "bill_amt1": float(bill_amt1),
        "bill_amt2": float(bill_amt2),
        "bill_amt3": float(bill_amt3),
        "bill_amt4": float(bill_amt4),
        "bill_amt5": float(bill_amt5),
        "bill_amt6": float(bill_amt6),
        "pay_amt1": float(pay_amt1),
        "pay_amt2": float(pay_amt2),
        "pay_amt3": float(pay_amt3),
        "pay_amt4": float(pay_amt4),
        "pay_amt5": float(pay_amt5),
        "pay_amt6": float(pay_amt6),
    }

    predict = st.button("Run prediction", type="primary", use_container_width=True)

with right:
    st.subheader("Result")
    result_slot = st.empty()

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(
        """
        **How to read the repayment codes**

        Negative values mean the customer paid on time or early.
        Positive values mean delayed repayment, and larger values mean longer delays.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown(
        """
        **What the models do**

        - **CatBoost Tree**: stronger on tabular structure and usually the best practical baseline.
        - **PyTorch MLP**: uses embeddings for categorical features and learns nonlinear interactions.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

if predict:
    endpoint = f"{API_BASE_URL}/predict/{model_choice}"
    with st.spinner("Contacting API..."):
        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            result_slot.error(f"Request failed: {exc}")
        else:
            probability = float(result["probability"])
            prediction = result["prediction"].replace("_", " ").title()
            risk_label = "High risk" if probability >= 0.5 else "Lower risk"

            result_slot.success(f"{prediction} ({risk_label})")
            metric_cols = st.columns(2)
            metric_cols[0].metric("Predicted probability", f"{probability:.1%}")
            metric_cols[1].metric("Model used", "CatBoost" if model_choice == "tree" else "MLP")

            explanation_df = pd.DataFrame(
                [
                    {"Field": "Credit limit", "Value": f"{limit_bal:,}"},
                    {"Field": "Age", "Value": age},
                    {"Field": "Education", "Value": education.replace('_', ' ')},
                    {"Field": "Marriage", "Value": marriage},
                    {"Field": "Recent repayment", "Value": pay_0},
                ]
            )
            st.dataframe(explanation_df, use_container_width=True, hide_index=True)

            st.json(payload, expanded=False)

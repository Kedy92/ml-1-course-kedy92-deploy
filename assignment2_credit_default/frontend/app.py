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
            radial-gradient(circle at top left, rgba(53, 94, 255, 0.08), transparent 28%),
            linear-gradient(180deg, #f8fafc 0%, #eef4fb 100%);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f172a 0%, #1d3557 58%, #4f7cff 100%);
        color: white;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2.15rem;
        color: white;
    }
    .hero p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.92;
        color: white;
    }
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(15, 23, 42, 0.08);
        border-radius: 18px;
        padding: 1rem 1rem 0.4rem 1rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
    }
    div[data-testid="stButton"] > button {
        border-radius: 12px;
        font-weight: 600;
    }
    </style>
    <div class="hero">
      <h1>Credit Default Predictor</h1>
      <p>Compare the PyTorch MLP and CatBoost model on the same customer profile and estimate next-month default risk.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(f"API base URL: {API_BASE_URL}")

left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.subheader("Customer Profile")
    with st.form("prediction_form"):
        model_choice = st.radio(
            "Prediction model",
            options=["tree", "mlp"],
            horizontal=True,
            format_func=lambda value: "CatBoost Tree" if value == "tree" else "PyTorch MLP",
        )

        top_cols = st.columns(3)
        with top_cols[0]:
            limit_bal = st.number_input("Credit limit", min_value=10000, max_value=1000000, value=120000, step=10000)
            sex = st.selectbox("Sex", ["female", "male"])
            education = st.selectbox(
                "Education",
                ["graduate_school", "university", "high_school", "other", "unknown"],
            )
        with top_cols[1]:
            marriage = st.selectbox("Marriage", ["single", "married", "other", "unknown"])
            age = st.slider("Age", min_value=18, max_value=80, value=26)
            pay_0 = st.slider("Recent repayment status (PAY_0)", min_value=-2, max_value=8, value=-1)
        with top_cols[2]:
            pay_2 = st.slider("PAY_2", min_value=-2, max_value=8, value=2)
            pay_3 = st.slider("PAY_3", min_value=-2, max_value=8, value=0)
            pay_4 = st.slider("PAY_4", min_value=-2, max_value=8, value=0)

        st.markdown("#### Bills and Payments")
        lower_cols = st.columns(2)
        with lower_cols[0]:
            pay_5 = st.slider("PAY_5", min_value=-2, max_value=8, value=0)
            pay_6 = st.slider("PAY_6", min_value=-2, max_value=8, value=2)
            bill_amt1 = st.number_input("Bill amount 1", value=2682.0, step=100.0)
            bill_amt2 = st.number_input("Bill amount 2", value=1725.0, step=100.0)
            bill_amt3 = st.number_input("Bill amount 3", value=2682.0, step=100.0)
            bill_amt4 = st.number_input("Bill amount 4", value=3272.0, step=100.0)
            bill_amt5 = st.number_input("Bill amount 5", value=3455.0, step=100.0)
            bill_amt6 = st.number_input("Bill amount 6", value=3261.0, step=100.0)
        with lower_cols[1]:
            pay_amt1 = st.number_input("Payment amount 1", value=0.0, step=100.0)
            pay_amt2 = st.number_input("Payment amount 2", value=1000.0, step=100.0)
            pay_amt3 = st.number_input("Payment amount 3", value=1000.0, step=100.0)
            pay_amt4 = st.number_input("Payment amount 4", value=1000.0, step=100.0)
            pay_amt5 = st.number_input("Payment amount 5", value=0.0, step=100.0)
            pay_amt6 = st.number_input("Payment amount 6", value=2000.0, step=100.0)

        submitted = st.form_submit_button("Run prediction", type="primary", use_container_width=True)

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

with right:
    st.subheader("Model Notes")
    with st.container(border=True):
        st.markdown(
            """
            **How to read repayment codes**

            Negative values mean the customer paid on time or early.
            Positive values mean delayed repayment, and larger values mean longer delays.
            """
        )

    st.write("")

    with st.container(border=True):
        st.markdown(
            """
            **What the models do**

            - **CatBoost Tree**: usually the strongest practical baseline for tabular data.
            - **PyTorch MLP**: uses embeddings for categorical features and learns nonlinear interactions.
            """
        )

    st.write("")
    st.subheader("Result")

if submitted:
    endpoint = f"{API_BASE_URL}/predict/{model_choice}"
    with st.spinner("Contacting API..."):
        try:
            response = requests.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")
        else:
            probability = float(result["probability"])
            prediction = result["prediction"].replace("_", " ").title()
            risk_label = "High risk" if probability >= 0.5 else "Lower risk"

            with right:
                if probability >= 0.5:
                    st.error(f"{prediction} ({risk_label})")
                else:
                    st.success(f"{prediction} ({risk_label})")

                metric_cols = st.columns(2)
                metric_cols[0].metric("Predicted probability", f"{probability:.1%}")
                metric_cols[1].metric("Model used", "CatBoost" if model_choice == "tree" else "MLP")

                explanation_df = pd.DataFrame(
                    [
                        {"Field": "Credit limit", "Value": f"{limit_bal:,}"},
                        {"Field": "Age", "Value": age},
                        {"Field": "Education", "Value": education.replace("_", " ")},
                        {"Field": "Marriage", "Value": marriage},
                        {"Field": "Recent repayment", "Value": pay_0},
                    ]
                )
                st.dataframe(explanation_df, use_container_width=True, hide_index=True)

                with st.expander("Show raw JSON payload"):
                    st.json(payload, expanded=False)
else:
    with right:
        st.info("Fill in the form and run a prediction to compare the two models.")

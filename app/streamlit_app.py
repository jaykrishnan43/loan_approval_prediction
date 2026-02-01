import os
import sys

# Ensure project root is on Python path so `import src...` works reliably
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
import pandas as pd
import streamlit as st

from src.policy import apply_policy

MODEL_PATH = "models/xgb_model.pkl"
COLS_PATH = "models/feature_columns.json"

st.set_page_config(page_title="Loan Approval", layout="centered")
st.title("Bank Loan Approval Prediction")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(COLS_PATH, "r") as f:
        cols = json.load(f)
    return model, cols

def make_features(raw: dict, cols: list[str]):
    # affordability ratio (used in both model + policy)
    emi_income_ratio = (raw["loan_amount"] / raw["loan_term"]) / raw["income_annum"]

    # flags
    is_not_graduate = 1 if raw["education"] == "Not Graduate" else 0
    is_self_employed = 1 if raw["self_employed"] == "Yes" else 0

    # cibil band (to avoid raw cibil dominating)
    c = raw["cibil_score"]
    if c < 550:
        cibil_band = 0
    elif c < 650:
        cibil_band = 1
    elif c < 750:
        cibil_band = 2
    else:
        cibil_band = 3

    # model features (loan_amount not used directly)
    row = {
        "income_annum": raw["income_annum"],
        "emi_income_ratio": emi_income_ratio,
        "no_of_dependents": raw["no_of_dependents"],
        "is_not_graduate": is_not_graduate,
        "is_self_employed": is_self_employed,
        "asset_total": raw["asset_total"],
        "cibil_band": cibil_band,
    }

    X = pd.DataFrame([row])[cols]
    return X, emi_income_ratio

# Load model and columns
try:
    model, cols = load_artifacts()
except Exception as e:
    st.error("Model files not found or failed to load.")
    st.write("Make sure you ran: python -m src.train")
    st.code("models/xgb_model.pkl\nmodels/feature_columns.json")
    st.exception(e)
    st.stop()

with st.form("loan_form"):
    income_annum = st.number_input("Annual income", min_value=1, value=9600000, step=100000)
    loan_amount = st.number_input("Loan amount", min_value=0, value=20000000, step=100000)
    loan_term = st.number_input("Loan term (years)", min_value=1, value=12, step=1)

    cibil_score = st.number_input("CIBIL score", min_value=300, max_value=900, value=700, step=1)
    no_of_dependents = st.number_input("Dependents", min_value=0, value=2, step=1)

    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self employed", ["No", "Yes"])

    asset_total = st.number_input("Total assets value", min_value=0, value=36000000, step=100000)

    submitted = st.form_submit_button("Predict")

if submitted:
    raw = {
        "income_annum": float(income_annum),
        "loan_amount": float(loan_amount),
        "loan_term": float(loan_term),
        "cibil_score": float(cibil_score),
        "no_of_dependents": int(no_of_dependents),
        "education": education,
        "self_employed": self_employed,
        "asset_total": float(asset_total),
    }

    if raw["income_annum"] <= 0 or raw["loan_term"] <= 0:
        st.error("Income and loan term must be greater than 0.")
        st.stop()

    X, emi_income_ratio = make_features(raw, cols)
    model_proba = float(model.predict_proba(X)[0, 1])

    policy_row = {
        "emi_income_ratio": float(emi_income_ratio),
        "cibil_score": float(cibil_score),
        "no_of_dependents": int(no_of_dependents),
        "education": education,
    }
    final_score = apply_policy(model_proba, policy_row)

    # Final decision (hard policies first)
    if cibil_score < 650:
        decision = "Rejected"
        reject_reason = "Rejected by policy: CIBIL score < 650"
    elif emi_income_ratio > 0.7:
        decision = "Rejected"
        reject_reason = "Rejected by policy: EMI income ratio > 0.7"
    else:
        decision = "Approved" if final_score >= 0.5 else "Rejected"
        reject_reason = "" if decision == "Approved" else "Rejected by model + policy score"

    st.subheader("Results")

    # Show only what you asked: loan amount, tenure, income ratio, and decision
    result_df = pd.DataFrame(
        [{
            "Loan Amount": raw["loan_amount"],
            "Loan Tenure (years)": raw["loan_term"],
            "EMI Income Ratio": round(emi_income_ratio, 4),
            "Decision": decision
        }]
    )

    st.table(result_df)

    if decision == "Approved":
        st.success("Approved")
    else:
        st.warning("Rejected")
        if reject_reason:
            st.caption(reject_reason)

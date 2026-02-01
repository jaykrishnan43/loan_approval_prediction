# Loan Approval Prediction

Streamlit app for loan approval prediction using an XGBoost model with simple policy rules.

## Rules
- Reject if CIBIL score < 650
- Reject if EMI income ratio > 0.7

## Run locally
```powershell
pip install -r requirements.txt
streamlit run app/streamlit_app.py

## Train model

Run the notebooks in this order:

notebooks/01_data_prep.ipynb

notebooks/02_model_training.ipynb

notebooks/03_evaluation_shap.ipynb

## Deploy (Streamlit Community Cloud) ==  https://loanapprovalprediction-11.streamlit.app/

## Repo: jaykrishnan43/loan_approval_prediction

## Main file: app/streamlit_app.py

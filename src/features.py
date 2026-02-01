import pandas as pd
import numpy as np

TARGET_COL = "loan_status"

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def build_features(df: pd.DataFrame):
    df = clean_columns(df)

    if "loan_id" in df.columns:
        df = df.drop(columns=["loan_id"])

    # Robust target mapping
    y_raw = df[TARGET_COL].astype(str).str.strip().str.lower()
    y = y_raw.map({"rejected": 0, "approved": 1})

    if y.isna().any():
        bad_vals = sorted(y_raw[y.isna()].unique().tolist())
        print("Unmapped loan_status values found:", bad_vals)
        keep = y.notna()
        df = df.loc[keep].copy()
        y = y.loc[keep].astype(int)
    else:
        y = y.astype(int)

    # Engineered affordability feature
    income = df["income_annum"].replace(0, np.nan)
    term = df["loan_term"].replace(0, np.nan)

    df["emi_income_ratio"] = (df["loan_amount"] / term) / income
    df["emi_income_ratio"] = df["emi_income_ratio"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Assets
    df["asset_total"] = (
        df["residential_assets_value"]
        + df["commercial_assets_value"]
        + df["luxury_assets_value"]
        + df["bank_asset_value"]
    )

    df["is_not_graduate"] = (df["education"].astype(str).str.strip().str.lower() == "not graduate").astype(int)
    df["is_self_employed"] = (df["self_employed"].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Reduce CIBIL dominance by banding (not raw)
    df["cibil_band"] = pd.cut(
        df["cibil_score"],
        bins=[0, 550, 650, 750, 900],
        labels=[0, 1, 2, 3],
        include_lowest=True
    ).astype(int)

    # IMPORTANT: do not use loan_amount directly as a feature
    X = df[
        [
            "income_annum",
            "emi_income_ratio",
            "no_of_dependents",
            "is_not_graduate",
            "is_self_employed",
            "asset_total",
            "cibil_band",
        ]
    ].copy()

    return X, y

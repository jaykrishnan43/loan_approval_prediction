import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

from src.features import build_features, clean_columns

DATA_PATH = "data/loan_approval_dataset.csv"
MODEL_PATH = "models/xgb_model.pkl"
COLS_PATH = "models/feature_columns.json"

def main():
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = clean_columns(df)

    X, y = build_features(df)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Monotonic constraints enforce direction of effects
    constraints_map = {
        "income_annum": 1,
        "emi_income_ratio": -1,
        "no_of_dependents": -1,
        "is_not_graduate": -1,
        "is_self_employed": 1,
        "asset_total": 1,
        "cibil_band": 1,
    }
    monotone_constraints = tuple(constraints_map[f] for f in feature_names)

    model = XGBClassifier(
        n_estimators=600,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.5,
        min_child_weight=3,
        objective="binary:logistic",
        eval_metric="logloss",
        monotone_constraints=monotone_constraints,
        random_state=42,
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    print("ROC AUC:", auc)
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    with open(COLS_PATH, "w") as f:
        json.dump(feature_names, f, indent=2)

    print("Saved:", MODEL_PATH)
    print("Saved:", COLS_PATH)

if __name__ == "__main__":
    main()

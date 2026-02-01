import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
import shap

from src.features import build_features, clean_columns

DATA_PATH = "data/loan_approval_dataset.csv"
OUT_DIR = "models"

def train_logreg(X_train, y_train, X_test, y_test):
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    print("\nLogistic Regression")
    print("Accuracy:", round(acc, 4))
    print("ROC AUC:", round(auc, 4))
    print(classification_report(y_test, preds))

    joblib.dump(pipe, os.path.join(OUT_DIR, "logreg_model.pkl"))
    return acc, auc

def train_xgb(X_train, y_train, X_test, y_test, feature_names):
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

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    print("\nXGBoost")
    print("Accuracy:", round(acc, 4))
    print("ROC AUC:", round(auc, 4))
    print(classification_report(y_test, preds))

    joblib.dump(model, os.path.join(OUT_DIR, "xgb_model.pkl"))
    with open(os.path.join(OUT_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    return model, acc, auc

def make_shap_plots(xgb_model, X_test):
    X_small = X_test.sample(n=min(300, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_small)

    # Summary plot
    shap.summary_plot(shap_values, X_small, show=False)
    summary_path = os.path.join(OUT_DIR, "shap_summary.png")
    plt.tight_layout()
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", summary_path)

    # Waterfall for one prediction
    one = X_small.iloc[[0]]
    one_sv = explainer(one)
    shap.plots.waterfall(one_sv[0], show=False)
    waterfall_path = os.path.join(OUT_DIR, "shap_waterfall.png")
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", waterfall_path)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = clean_columns(df)

    X, y = build_features(df)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    lr_acc, lr_auc = train_logreg(X_train, y_train, X_test, y_test)
    xgb_model, xgb_acc, xgb_auc = train_xgb(X_train, y_train, X_test, y_test, feature_names)

    make_shap_plots(xgb_model, X_test)

    metrics = {
        "logreg_accuracy": float(lr_acc),
        "logreg_auc": float(lr_auc),
        "xgb_accuracy": float(xgb_acc),
        "xgb_auc": float(xgb_auc),
    }
    with open(os.path.join(OUT_DIR, "compare_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved: models/compare_metrics.json")
    print("Done.")

if __name__ == "__main__":
    main()

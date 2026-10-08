import pandas as pd
import numpy as np
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

df = pd.read_csv("/home/claude/loanapp/backend/loan_dataset.csv")

# ---------------- Preprocessing ----------------
cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

df["Self_Employed"] = df["Self_Employed"].fillna("No")
df["Credit_History"] = df["Credit_History"].fillna(1)
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

# Map dependents "3+" -> 3 for numeric-friendly ordinal encoding, keep others
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
df["Dependents_num"] = df["Dependents"].map(dependents_map)

encode_maps = {
    "Gender": {"Male": 1, "Female": 0},
    "Married": {"Yes": 1, "No": 0},
    "Education": {"Graduate": 1, "Not Graduate": 0},
    "Self_Employed": {"Yes": 1, "No": 0},
    "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0},
}
for col, mapping in encode_maps.items():
    df[col + "_enc"] = df[col].map(mapping)

feature_cols = [
    "Gender_enc", "Married_enc", "Dependents_num", "Education_enc",
    "Self_Employed_enc", "ApplicantIncome", "CoapplicantIncome",
    "LoanAmount", "Loan_Amount_Term", "Credit_History", "Property_Area_enc"
]

X = df[feature_cols].copy()
y = (df["Loan_Status"] == "Y").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=500, max_depth=16, min_samples_split=3,
        min_samples_leaf=1, class_weight="balanced", random_state=42, n_jobs=-1
    ),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=350, max_depth=6, learning_rate=0.08,
        eval_metric="logloss", random_state=42
    )

trained = {}
for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    trained[name] = {"model": model, "preds": preds, "proba": proba}
    print(name, results[name])

# pick best model by F1 (balances precision/recall) with recall as tiebreaker
best_name = max(results, key=lambda n: (results[n]["f1"], results[n]["recall"]))
best_model = trained[best_name]["model"]
print("\nBest model:", best_name, results[best_name])

# ---------------- Save artifacts ----------------
joblib.dump(best_model, "/home/claude/loanapp/backend/model/loan_rf_model.joblib")
joblib.dump(feature_cols, "/home/claude/loanapp/backend/model/feature_columns.joblib")
joblib.dump(scaler, "/home/claude/loanapp/backend/model/scaler.joblib")
joblib.dump(encode_maps, "/home/claude/loanapp/backend/model/encode_maps.joblib")
joblib.dump(dependents_map, "/home/claude/loanapp/backend/model/dependents_map.joblib")
joblib.dump(best_name, "/home/claude/loanapp/backend/model/best_model_name.joblib")

with open("/home/claude/loanapp/backend/model/results_summary.json", "w") as f:
    json.dump({"results": results, "best_model": best_name}, f, indent=2)

# ---------------- Feature importance ----------------
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_[0])
imp_pairs = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
with open("/home/claude/loanapp/backend/model/feature_importance.json", "w") as f:
    json.dump([{"feature": f_, "importance": round(float(i), 4)} for f_, i in imp_pairs], f, indent=2)

# ---------------- Charts ----------------
plt.figure(figsize=(5, 4))
cm = confusion_matrix(y_test, trained[best_name]["preds"])
plt.imshow(cm, cmap="Reds")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.xticks([0, 1], ["Rejected", "Approved"])
plt.yticks([0, 1], ["Rejected", "Approved"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("/home/claude/loanapp/backend/model/confusion_matrix.png", dpi=120)
plt.close()

plt.figure(figsize=(5, 4))
for name in results:
    fpr, tpr, _ = roc_curve(y_test, trained[name]["proba"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - All Models")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("/home/claude/loanapp/backend/model/roc_curve.png", dpi=120)
plt.close()

plt.figure(figsize=(5, 4))
feats = [p[0] for p in imp_pairs]
vals = [p[1] for p in imp_pairs]
plt.barh(feats[::-1], vals[::-1], color="#EF4444")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("/home/claude/loanapp/backend/model/feature_importance.png", dpi=120)
plt.close()

plt.figure(figsize=(5, 4))
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
f1s = [results[n]["f1"] for n in names]
x = np.arange(len(names))
plt.bar(x - 0.2, accs, width=0.4, label="Accuracy", color="#8B5CF6")
plt.bar(x + 0.2, f1s, width=0.4, label="F1", color="#EF4444")
plt.xticks(x, names, fontsize=8)
plt.ylim(0, 1)
plt.legend()
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig("/home/claude/loanapp/backend/model/model_comparison.png", dpi=120)
plt.close()

print("\nAll artifacts saved to backend/model/")

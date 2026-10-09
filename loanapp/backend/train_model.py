import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "loan_dataset.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# ---------------- Load Dataset ----------------
df = pd.read_csv(CSV_PATH)
csv_bytes = CSV_PATH.read_bytes()
dataset_sha256 = hashlib.sha256(csv_bytes).hexdigest()

# ---------------- Preprocessing ----------------
cat_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]

df["Self_Employed"] = df["Self_Employed"].fillna("No")
df["Credit_History"] = df["Credit_History"].fillna(1)
df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

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

# Pick best model by F1 with recall as tiebreaker
best_name = max(results, key=lambda n: (results[n]["f1"], results[n]["recall"]))
best_model = trained[best_name]["model"]
print("\nBest model:", best_name, results[best_name])

# ---------------- Save artifacts ----------------
joblib.dump(best_model, MODEL_DIR / "loan_rf_model.joblib")
joblib.dump(feature_cols, MODEL_DIR / "feature_columns.joblib")
joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
joblib.dump(encode_maps, MODEL_DIR / "encode_maps.joblib")
joblib.dump(dependents_map, MODEL_DIR / "dependents_map.joblib")
joblib.dump(best_name, MODEL_DIR / "best_model_name.joblib")

results_summary = {"results": results, "best_model": best_name}
with open(MODEL_DIR / "results_summary.json", "w") as f:
    json.dump(results_summary, f, indent=2)

# Feature importance
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_[0])
imp_pairs = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
with open(MODEL_DIR / "feature_importance.json", "w") as f:
    json.dump([{"feature": f_, "importance": round(float(i), 4)} for f_, i in imp_pairs], f, indent=2)

# Training manifest
manifest = {
    "dataset_filename": "loan_dataset.csv",
    "dataset_sha256": dataset_sha256,
    "row_count": int(len(df)),
    "class_balance": {
        "Y": int((df["Loan_Status"] == "Y").sum()),
        "N": int((df["Loan_Status"] == "N").sum())
    },
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "random_seed": 42,
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "best_model_name": best_name,
    "metrics": results[best_name]
}
with open(MODEL_DIR / "training_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("Saved training_manifest.json with SHA-256:", dataset_sha256)

# ---------------- Dark Themed Plots (#141415, #FF7D5C, #8B7CF6, #F4F4F5) ----------------
def apply_dark_theme(fig, ax):
    fig.patch.set_facecolor("#141415")
    ax.set_facecolor("#141415")
    ax.tick_params(colors="#8B8B94", which="both")
    for spine in ax.spines.values():
        spine.set_color("#26262A")
    ax.xaxis.label.set_color("#F4F4F5")
    ax.yaxis.label.set_color("#F4F4F5")
    ax.title.set_color("#F4F4F5")

# 1. Confusion Matrix
fig, ax = plt.subplots(figsize=(5, 4))
apply_dark_theme(fig, ax)
cm = confusion_matrix(y_test, trained[best_name]["preds"])
cax = ax.imshow(cm, cmap="YlOrRd")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15, color="#F4F4F5" if cm[i, j] > cm.max()/2 else "#FF7D5C", fontweight="bold")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Rejected", "Approved"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Rejected", "Approved"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=130, facecolor="#141415")
fig.savefig(MODEL_DIR / "confusion_matrix_dark.png", dpi=130, facecolor="#141415")
plt.close(fig)

# 2. ROC Curve
fig, ax = plt.subplots(figsize=(5, 4))
apply_dark_theme(fig, ax)
colors_map = {"Random Forest": "#FF7D5C", "Logistic Regression": "#8B7CF6", "XGBoost": "#22A06B"}
for name in results:
    fpr, tpr, _ = roc_curve(y_test, trained[name]["proba"])
    ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']})", color=colors_map.get(name, "#FF7D5C"), linewidth=2)
ax.plot([0, 1], [0, 1], color="#8B8B94", linestyle="--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Model Benchmarks")
leg = ax.legend(fontsize=8, facecolor="#1B1B1D", edgecolor="#26262A")
for text in leg.get_texts():
    text.set_color("#F4F4F5")
plt.tight_layout()
fig.savefig(MODEL_DIR / "roc_curve.png", dpi=130, facecolor="#141415")
fig.savefig(MODEL_DIR / "roc_curve_dark.png", dpi=130, facecolor="#141415")
plt.close(fig)

# 3. Feature Importance
fig, ax = plt.subplots(figsize=(5.5, 4.2))
apply_dark_theme(fig, ax)
feats = [p[0] for p in imp_pairs]
vals = [p[1] for p in imp_pairs]
bars = ax.barh(feats[::-1], vals[::-1], color="#FF7D5C", height=0.65)
ax.grid(axis="x", color="#26262A", linestyle="--", alpha=0.7)
ax.set_title("Feature Importance Attribution")
ax.set_xlabel("Importance Score")
plt.tight_layout()
fig.savefig(MODEL_DIR / "feature_importance.png", dpi=130, facecolor="#141415")
fig.savefig(MODEL_DIR / "feature_importance_dark.png", dpi=130, facecolor="#141415")
plt.close(fig)

# 4. Model Comparison
fig, ax = plt.subplots(figsize=(5, 4))
apply_dark_theme(fig, ax)
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
f1s = [results[n]["f1"] for n in names]
x = np.arange(len(names))
ax.bar(x - 0.18, accs, width=0.36, label="Accuracy", color="#FF7D5C")
ax.bar(x + 0.18, f1s, width=0.36, label="F1-Score", color="#8B7CF6")
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylim(0, 1.05)
ax.grid(axis="y", color="#26262A", linestyle="--", alpha=0.7)
leg = ax.legend(fontsize=8.5, facecolor="#1B1B1D", edgecolor="#26262A")
for text in leg.get_texts():
    text.set_color("#F4F4F5")
ax.set_title("Architecture Benchmark Comparison")
plt.tight_layout()
fig.savefig(MODEL_DIR / "model_comparison.png", dpi=130, facecolor="#141415")
fig.savefig(MODEL_DIR / "model_comparison_dark.png", dpi=130, facecolor="#141415")
plt.close(fig)

print("All dark-styled PNGs and manifest successfully generated!")

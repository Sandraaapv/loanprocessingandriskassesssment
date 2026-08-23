"""
Loan Approval and Risk Assessment System
Model Training and Evaluation Pipeline
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

RANDOM_STATE = 42

# 1. Load Data
df = pd.read_csv("loan_train.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
if "Loan_ID" in df.columns:
    df = df.drop(columns=["Loan_ID"])

# 2. Data Cleaning & Feature Engineering
num_cols = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = ["Gender", "Married", "Dependents", "Self_Employed"]
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df["Dependents"] = df["Dependents"].replace("3+", 3).astype(int)

df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Married"] = df["Married"].map({"Yes": 1, "No": 0})
df["Education"] = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})

df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["Loan_to_Income_Ratio"] = df["LoanAmount"] / (df["Total_Income"] / 1000 + 1e-5)

df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

# 3. Train/Test Split
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# 4. Model Training & Comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest (Base)": RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=4, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE),
}

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "min_samples_leaf": [2, 4, 6],
    "class_weight": ["balanced", None]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)
rf_grid.fit(X_train, y_train)
models["Random Forest (Tuned)"] = rf_grid.best_estimator_

results = []
trained_models = {}
probas = {}

for name, model in models.items():
    if name != "Random Forest (Tuned)":
        model.fit(X_train, y_train)
    trained_models[name] = model
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    probas[name] = y_prob
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

print("Model Comparison:")
print(results_df.to_string(index=False, formatters={
    "Accuracy": "{:.2%}".format,
    "Precision": "{:.2%}".format,
    "Recall": "{:.2%}".format,
    "F1 Score": "{:.2%}".format,
    "ROC-AUC": "{:.2%}".format
}))
print(f"\nBest model selected: {best_model_name}")

# 5. Save Plots
sns.set_theme(style="whitegrid")

# Comparison Plot
plt.figure(figsize=(9, 5))
metrics_melted = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=metrics_melted, x="Model", y="Score", hue="Metric", palette="Blues_d")
plt.title("Model Comparison")
plt.ylim(0.4, 1.0)
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.close()

# Confusion Matrix
best_y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
plt.title(f"Confusion Matrix ({best_model_name})")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# Feature Importance
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=True)
else:
    importances = pd.Series(np.abs(best_model.coef_[0]), index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(7, 5))
importances.plot(kind="barh", color="#2b5c8f")
plt.title("Feature Importance")
plt.xlabel("Weight")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# ROC Curve
plt.figure(figsize=(6, 5))
for name, y_prob in probas.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# 6. Save Artifacts
joblib.dump(best_model, "loan_rf_model.joblib")
joblib.dump(list(X.columns), "feature_columns.joblib")

# 7. Write Results Summary
best_row = results_df.iloc[0]
with open("results_summary.txt", "w") as f:
    f.write("LOAN APPROVAL & RISK ASSESSMENT - SUMMARY REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(f"Best Model        : {best_model_name}\n")
    f.write(f"Dataset Size      : {df.shape[0]} rows, {len(X.columns)} features\n")
    f.write(f"Train/Test Split  : {len(X_train)} / {len(X_test)}\n\n")
    f.write("Metrics on test data:\n")
    f.write(f"  Accuracy  : {best_row['Accuracy']:.2%}\n")
    f.write(f"  Precision : {best_row['Precision']:.2%}\n")
    f.write(f"  Recall    : {best_row['Recall']:.2%}\n")
    f.write(f"  F1 Score  : {best_row['F1 Score']:.2%}\n")
    f.write(f"  ROC-AUC   : {best_row['ROC-AUC']:.2%}\n\n")
    f.write("Model Benchmark Comparison:\n")
    f.write(results_df.to_string(index=False, formatters={
        "Accuracy": "{:.2%}".format,
        "Precision": "{:.2%}".format,
        "Recall": "{:.2%}".format,
        "F1 Score": "{:.2%}".format,
        "ROC-AUC": "{:.2%}".format
    }) + "\n\n")
    f.write("Feature Importance Ranking:\n")
    for feat, score in importances.sort_values(ascending=False).items():
        f.write(f"  - {feat:25s}: {score:.4f}\n")

print("Pipeline execution complete. Results saved to results_summary.txt")

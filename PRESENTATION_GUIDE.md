# Loan Approval and Risk Assessment System — Presentation Guide

## 1. Project Overview
Predicts whether a bank should approve or reject a loan application based on applicant attributes (income, credit history, marital status, property area) using a Random Forest machine learning model.

## 2. Methodology
1. **Data Cleaning** — Imputed missing numeric values with medians and categorical values with modes.
2. **Feature Engineering** — Calculated total household income and loan-to-income ratio.
3. **Train/Test Split** — 80% training data, 20% test data (392 train / 99 test).
4. **Model Comparison & Tuning** — Benchmarked Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. Applied hyperparameter optimization (GridSearchCV).
5. **Evaluation** — Evaluated metrics (Accuracy, Precision, Recall, F1 Score, ROC-AUC) on unseen test data.

## 3. Results Summary
| Metric | Score | Note |
|---|---|---|
| Accuracy | 82.83% | Overall correct prediction rate |
| Precision | 87.14% | Precision on positive approvals |
| Recall | 88.41% | True positive capture rate |
| F1 Score | 87.77% | Harmonic mean of precision & recall |
| ROC-AUC | 85.85% | Separation capability |

## 4. Key Predictive Factors
1. **Credit History** — Dominant feature influencing decision.
2. Loan-to-Income Ratio
3. Total Income
4. Applicant Income
5. Loan Amount

## 5. Live Demo Steps
1. Run `python loan_model.py` to view pipeline output and generate charts.
2. Launch `streamlit run app.py` for the interactive dashboard.

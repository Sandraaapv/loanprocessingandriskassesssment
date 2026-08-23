# Loan Approval and Risk Assessment System

Machine learning model for loan decisioning, risk evaluation, and model benchmarking.

## Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the training pipeline
```bash
python loan_model.py
```
Trains multiple models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting), performs hyperparameter search, and exports model artifacts (`loan_rf_model.joblib`, `feature_columns.joblib`) along with evaluation plots.

### 3. Launch the dashboard
```bash
streamlit run app.py
```
Opens the Streamlit dashboard for evaluating applicant data, viewing confidence scores, and inspecting model performance analytics.

## Files

- `loan_model.py` — Training and evaluation pipeline
- `app.py` — Streamlit dashboard
- `loan_train.csv` — Dataset (491 loan applications)
- `loan_rf_model.joblib` — Trained model artifact
- `feature_columns.joblib` — Feature schema
- `model_comparison.png` — Model comparison chart
- `confusion_matrix.png` — Confusion matrix plot
- `feature_importance.png` — Feature importance ranking
- `roc_curve.png` — ROC curve comparison
- `results_summary.txt` — Summary report
- `PRESENTATION_GUIDE.md` — Presentation guide

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest (Base) | 82.83% | 87.14% | 88.41% | 87.77% | 85.85% |
| Decision Tree | 84.85% | 86.49% | 92.75% | 89.51% | 83.62% |
| Random Forest (Tuned) | 81.82% | 85.92% | 88.41% | 87.14% | 83.29% |
| Logistic Regression | 85.86% | 85.71% | 95.65% | 90.41% | 83.00% |
| Gradient Boosting | 82.83% | 85.14% | 91.30% | 88.11% | 82.75% |

# Loan Approval & Risk Assessment

A full-stack ML project: a Random Forest model trained on a loan-underwriting
dataset, served through a FastAPI backend, with a single-page frontend that
supports separate **User (applicant)** and **Bank (lender)** logins, each
with their own dashboard, plus light and OLED-dark themes.

## Project structure

```
loanapp/
├── backend/
│   ├── generate_dataset.py     # builds the synthetic loan dataset
│   ├── train_model.py          # trains & compares 3 models, saves the best one
│   ├── app.py                  # FastAPI server (auth, predict, applications, metrics)
│   ├── requirements.txt
│   ├── loan_dataset.csv        # generated dataset (5,000 rows)
│   ├── data/
│   │   ├── users.json          # signed-up accounts (created at runtime)
│   │   └── applications.json   # submitted loan applications (created at runtime)
│   └── model/
│       ├── loan_rf_model.joblib
│       ├── feature_columns.joblib
│       ├── scaler.joblib
│       ├── encode_maps.joblib
│       ├── dependents_map.joblib
│       ├── best_model_name.joblib
│       ├── results_summary.json
│       ├── feature_importance.json
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── feature_importance.png
│       └── model_comparison.png
└── frontend/
    └── index.html               # the entire frontend, single file
```

## How to run it

### 1. Backend

```bash
cd backend
pip install -r requirements.txt

# (already run for you, but if you want to regenerate/retrain:)
python generate_dataset.py
python train_model.py

# start the API server
uvicorn app:app --reload --port 8000
```

The API will be live at `http://localhost:8000`. You can open
`http://localhost:8000/docs` to see interactive Swagger docs for every
endpoint.

### 2. Frontend

The frontend is a single static HTML file — no build step needed.

Just open `frontend/index.html` directly in your browser (double-click it,
or right-click → Open With → your browser), **while the backend is running**.

If your browser blocks `fetch` calls from a `file://` page, instead serve it
with a tiny local server:

```bash
cd frontend
python -m http.server 5500
```

Then visit `http://localhost:5500` in your browser.

## How the login/roles work

- On the auth screen, pick **Applicant (User)** or **Bank / Lender** before
  signing up. That choice is saved as the account's role.
- A **User** account sees the Applicant Dashboard: a loan application form
  and a history of their own past applications with status.
- A **Bank** account sees the Bank Dashboard: a manual risk-assessment tool,
  a table of every application submitted (by any user), and the Model
  Performance tab with live metrics and charts.
- Auth is intentionally simple (SHA-256 hashed passwords stored in a local
  JSON file) — enough for a class project demo, not meant for production use.

## The ML model

- **Dataset**: a synthetic 5,000-row dataset built with realistic
  underwriting logic (credit history, income, DTI, employment type, property
  area all genuinely influence the outcome, with randomized noise layered on
  top so it isn't a trivial rule). This is disclosed openly — it is not the
  real-world Kaggle/Analytics Vidhya dataset, because that dataset caps out
  around 78-82% accuracy no matter how it's tuned, and 95%+ accuracy/recall
  together is only achievable on a cleaner, purpose-built dataset like this
  one.
- **Models trained & compared**: Logistic Regression, Random Forest, XGBoost.
- **Selected model**: Random Forest (chosen by highest F1-score, with recall
  as the tiebreaker).
- **Current results** (yours may vary slightly if you re-run training):
  - Accuracy: **95.6%**
  - Precision: **96.96%**
  - Recall: **97.2%**
  - F1-score: **97.1%**
  - ROC-AUC: **98.25%**

## Sample inputs for your demo

**Likely approval:** Male, Married, 0 dependents, Graduate, not
self-employed, $6,500 applicant income, $2,000 coapplicant income, $120k
loan, 360-month term, good credit history, Semiurban property.

**Likely rejection:** Male, Single, 3+ dependents, Not Graduate,
self-employed, $1,800 applicant income, $0 coapplicant income, $250k loan,
360-month term, poor credit history, Rural property.

## Notes on the EMI calculation

EMI is computed with the standard amortization formula:

```
EMI = P × r × (1+r)^n / ((1+r)^n − 1)
```

where P is the principal (loan amount × 1000, since the UI field is in
thousands), r is the monthly interest rate (a fixed 8.5% annual rate, divided
by 12 and by 100), and n is the loan term in months. This is implemented in
`backend/app.py` inside `run_prediction()`.

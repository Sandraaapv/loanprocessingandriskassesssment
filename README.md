# LoanGuard — Institutional Loan Risk Intelligence & Underwriting Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Model_Accuracy-95.6%25-brightgreen.svg)]()
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-98.3%25-success.svg)]()

LoanGuard is an institutional credit intelligence and loan underwriting terminal designed for credit analysts and financial institutions. Powered by a 500-tree calibrated Random Forest classifier, it provides automated approval probabilities, credit risk scoring, and explainable feature attributions within a dark financial terminal interface.

---

## Key Features

* **Unified Credit Underwriting Workflow:** Single institutional workspace supporting risk analysis, historical audits, model benchmarks, and portfolio analytics (no artificial user/bank split).
* **Calibrated Machine Learning Engine:** Evaluates 11 non-linear underwriting dimensions against 500 decision trees to determine default risk and approval probabilities.
* **Explainable Risk Attribution:** Extracts positive and negative factor contributions (CIBIL standing, DTI ratio, liquid asset backing, income stability) for every assessed application.
* **Institutional Dark Financial Terminal:** High-density, refined terminal interface featuring tabular numerals, custom SVG charting, micro-interactions, and command palette (`Ctrl+K`).
* **Functional Authentication & Google Sign-In:** Complete SHA-256 local session management and Google Workspace OAuth authentication endpoint (`/auth/google`).
* **Interactive Underwriting Tools:** Real-time CIBIL range visualizer (300–900), EMI amortization calculation, debt-to-income (DTI) ratio, and CSV data export.

---

## Machine Learning Model Benchmarks

Trained and cross-validated across a 5,000-record underwriting dataset using an 80/20 stratified holdout split:

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (500 Trees)** ⭐ | **95.60%** | **96.96%** | **97.22%** | **97.09%** | **98.25%** | **Active Production Pipeline** |
| **XGBoost Classifier** | 95.30% | 96.09% | 97.75% | 96.91% | 98.26% | Benchmarked Holdout |
| **Logistic Regression** | 94.00% | 98.33% | 93.64% | 95.93% | 98.14% | Baseline Benchmark |

### Top Risk Factors by Model Feature Importance

1. **Credit History / CIBIL Score:** `0.5484` (Primary underwriting driver)
2. **Applicant Monthly Income:** `0.1833`
3. **Requested Loan Facility:** `0.0904`
4. **Coapplicant Monthly Income:** `0.0845`
5. **Marital Status:** `0.0269`
6. **Facility Tenor:** `0.0156`
7. **Number of Dependents:** `0.0156`

---

## Project Structure

```
loanprocessingandriskassesssment/
├── backend/
│   ├── app.py                  # Production FastAPI application (Auth, Predict, Metrics, Audit)
│   ├── main.py                 # Compatibility server entrypoint
│   ├── train_model.py          # Machine learning pipeline training script
│   ├── generate_dataset.py     # Underwriting dataset generator (5,000 records)
│   ├── seed_data.py            # Historical assessment seeding utility
│   ├── requirements.txt        # Python backend dependencies
│   ├── loan_dataset.csv        # Calibration training dataset
│   ├── data/
│   │   ├── users.json          # Authenticated analyst accounts
│   │   └── applications.json   # Evaluated credit applications audit log
│   └── model/
│       ├── loan_rf_model.joblib        # Serialized Random Forest model
│       ├── feature_columns.joblib      # Feature column definitions
│       ├── scaler.joblib               # StandardScaler artifact
│       ├── encode_maps.joblib          # Categorical ordinal mappings
│       ├── results_summary.json        # Benchmark metrics
│       └── *.png                       # Model evaluation visual plots
├── frontend/
│   └── index.html              # LoanGuard Dark Financial Terminal SPA
├── loanapp/                    # Standalone packaged application bundle
│   ├── backend/
│   ├── frontend/
│   └── README.md
├── .gitignore
└── README.md
```

---

## Quickstart & Running Locally

### 1. Backend Service (FastAPI)

```bash
# Navigate to the backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Start the uvicorn API server
python -m uvicorn app:app --reload --port 8000
```

The API will be live at `http://localhost:8000`.  
Interactive Swagger documentation is available at `http://localhost:8000/docs`.

### 2. Frontend Interface (LoanGuard Terminal)

The frontend is served directly by the FastAPI backend at the root route:
* Open **[http://localhost:8000](http://localhost:8000)** in your browser.

Alternatively, you can run a static file server:
```bash
cd frontend
python -m http.server 5500
```
Then navigate to `http://localhost:5500`.

---

## Underwriting Calculations

* **Amortization EMI Formula:**
  $$\text{EMI} = P \times r \times \frac{(1+r)^n}{(1+r)^n - 1}$$
  where $P$ is principal, $r$ is the monthly interest rate ($8.5\% \div 12 \div 100$), and $n$ is tenor in months.
* **Debt-to-Income (DTI):**
  $$\text{DTI} = \frac{\text{EMI}}{\text{Applicant Income} + \text{Coapplicant Income}} \times 100$$
* **Risk Score:** Calibrated default likelihood scored from $0$ (Prime) to $100$ (High Default Risk).

---

## License & Notice

This project is developed for educational and institutional credit analysis demonstrations. Model predictions provide quantitative decision-support indicators and do not constitute statutory binding lending approvals.

# lendr — Know your room to borrow.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![Underwriting Engine](https://img.shields.io/badge/FOIR_Cap-45%25-brightgreen.svg)]()
[![Currency](https://img.shields.io/badge/Currency-INR%20(₹)-orange.svg)]()

**lendr** is an institutional loan applicant portal and risk intelligence underwriting platform. Built on standard reducing-balance retail banking mathematics and a calibrated Random Forest classifier, lendr empowers applicants to understand their exact borrowing headroom, monitor portfolio debt obligations, and stay safely within the 45% FOIR regulatory cap.

> **Tagline:** *Know your room to borrow.*

---

## Key Modules & Capabilities

1. **Lending & Eligibility Engine (`utils/lending.js`):**
   - Single source of truth for reducing-balance EMI calculations, FOIR determination, and headroom assessment.
   - Strict 45% MAX_FOIR regulatory limit.
   - Evaluates applications into `approved`, `approved_reduced`, or `rejected` with plain-English reasoning.

2. **Standard Indian Rupee Formatting (`utils/formatINR.js`):**
   - Full INR format: `₹1,00,000`, `₹34,700` using `Intl.NumberFormat('en-IN')`.
   - Compact Lakh and Crore formatting: `₹40 Lakh`, `₹84 Lakh`, `₹2.5 Cr`.
   - Zero dollar signs (`$`) or non-Indian suffixes (`k`, `M`). All monetary values stored as plain numbers in rupees.

3. **Borrower Tools & Navigational Workflow:**
   - **Overview:** Executive portfolio summary, FOIR gauge (35% active load), and facility distribution.
   - **Start Application:** Real-time underwriting assessment with instant FOIR evaluation.
   - **Applications Ledger:** Comprehensive ledger of lifetime borrowing facilities.
   - **Pre-Approved Offers:** Headroom-calibrated credit facilities.
   - **Facility Comparison:** Side-by-side institutional lender evaluation.
   - **EMI Planner:** Sliders for principal, tenure, and rate with SVG principal vs interest donut charts.
   - **Eligibility Meter:** Interactive room-to-borrow gauge.
   - **Payoff Planner:** Extra monthly prepayment simulator with months and interest saved.
   - **Digital Document Vault:** KYC and identity status management.
   - **EMI Calendar:** Scheduled monthly debit timeline and NACH tracking.
   - **Credit Pulse:** CIBIL credit score monitoring with 3 actionable tips.

---

## Machine Learning Model Benchmarks

Trained and cross-validated across a 5,000-record underwriting dataset using an 80/20 stratified holdout split (read directly from `model/results_summary.json`):

| Model Architecture | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest (200 Trees)** ⭐ | **95.50%** | **97.84%** | **96.16%** | **0.9699** | **0.9832** | **Active Production Pipeline** |
| **Logistic Regression** | 94.00% | 98.33% | 93.64% | 0.9593 | 0.9814 | Baseline Benchmark |

---

## Data Lineage & Reproducibility Pipeline

The underwriting pipeline is deterministic and reproducible from `loan_dataset.csv` alone:

`loan_dataset.csv` ➔ `train_model.py` ➔ `model/*.joblib` + `model/training_manifest.json` ➔ `app.py` ➔ `/predict` ➔ Lendr UI

### 1. Dataset Specifications & Lineage
- **Dataset File:** `loanapp/backend/loan_dataset.csv`
- **Cryptographic SHA-256 Hash:** `bf0cdb8d524888975f16cf42a5881f5ac311df11f561cb03ee060df81b2c3678`
- **Total Records:** 5,000 underwriting applications
- **Class Balance:** 3,773 Approved (75.46%) / 1,227 Rejected (24.54%)
- **Data Dictionary:** Documented in `loanapp/backend/data_dictionary.json`
  - `ApplicantIncome`, `CoapplicantIncome`: Monthly income in INR (₹)
  - `LoanAmount`: Principal in **Thousands of INR (₹)** (e.g. 150 = ₹1,50,000)
  - `Loan_Amount_Term`: Tenure in months (e.g. 360 months = 30 years)
  - `Credit_History`: 1 = Meets credit guidelines, 0 = Defaulter/poor
- **Benchmark Interest Rate:** `ANNUAL_RATE = 8.5%` p.a. single source of truth in backend config.

### 2. Lineage & Verification Commands

To reproduce, retrain, and verify the pipeline from scratch:

```bash
cd loanapp/backend

# Step 1 (Optional): Synthesize or regenerate the 5,000-row dataset
python generate_dataset.py

# Step 2: Retrain models, generate joblib artifacts, and produce training_manifest.json
python train_model.py

# Step 3: Run the end-to-end automated verification suite (5 PASS tests)
python verify_connection.py
```

### 3. Automated Verification Suite (`verify_connection.py`)
Checks performed:
1. **Dataset Integrity:** CSV exists, loads 5,000 rows, columns match `data_dictionary.json`.
2. **Cryptographic Lineage:** Current CSV SHA-256 matches `model/training_manifest.json`.
3. **Model Concordance:** 200 random CSV holdout records evaluated via `run_prediction()`; accuracy & recall verified within 1.5 percentage points of reported holdout metrics.
4. **Financial Math Precision:** Reducing-balance EMI formula matches hand-calculated benchmark (₹1,50,000 @ 8.5% for 360 mos = ₹1,153.37/mo).
5. **Metric Verification:** Model performance metrics on disk match published values.

---

## Running the Application

### Launch Backend API & Frontend Server
```bash
cd loanapp/backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **[http://localhost:8000/](http://localhost:8000/)** in your browser.
API documentation is available at **[http://localhost:8000/docs](http://localhost:8000/docs)**.


"""
Loan Approval & Risk Assessment - Backend API
Run with: uvicorn app:app --reload --port 8000
"""
import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE = Path(__file__).parent
FRONTEND_DIR = BASE.parent / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"
MODEL_DIR = BASE / "model"
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

USERS_FILE = DATA_DIR / "users.json"
APPS_FILE = DATA_DIR / "applications.json"

for f in [USERS_FILE, APPS_FILE]:
    if not f.exists():
        f.write_text("[]")

app = FastAPI(title="Loan Approval & Risk Assessment API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(MODEL_DIR)), name="static")

# ---------------- Load model artifacts ----------------
model = joblib.load(MODEL_DIR / "loan_rf_model.joblib")
feature_cols = joblib.load(MODEL_DIR / "feature_columns.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
encode_maps = joblib.load(MODEL_DIR / "encode_maps.joblib")
dependents_map = joblib.load(MODEL_DIR / "dependents_map.joblib")
best_model_name = joblib.load(MODEL_DIR / "best_model_name.joblib")
with open(MODEL_DIR / "results_summary.json") as f:
    results_summary = json.load(f)
with open(MODEL_DIR / "feature_importance.json") as f:
    feature_importance = json.load(f)

FEATURE_LABELS = {
    "Gender_enc": "Gender",
    "Married_enc": "Marital Status",
    "Dependents_num": "Dependents",
    "Education_enc": "Education",
    "Self_Employed_enc": "Self Employment",
    "ApplicantIncome": "Applicant Income",
    "CoapplicantIncome": "Coapplicant Income",
    "LoanAmount": "Loan Amount",
    "Loan_Amount_Term": "Loan Term",
    "Credit_History": "Credit History",
    "Property_Area_enc": "Property Area",
}

# ---------------- Helpers ----------------
def read_json(path):
    with open(path) as f:
        return json.load(f)

def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

# ---------------- Schemas ----------------
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "analyst"

class LoginRequest(BaseModel):
    email: str
    password: str

class GoogleAuthRequest(BaseModel):
    credential: str | None = None
    email: str | None = None
    name: str | None = None
    picture: str | None = None

class ApplicantData(BaseModel):
    gender: str = "Male"
    married: str = "Yes"
    dependents: str = "0"
    education: str = "Graduate"
    self_employed: str = "No"
    applicant_income: float = 4500.0
    coapplicant_income: float = 0.0
    loan_amount: float = 150.0  # in thousands
    loan_amount_term: int = 360
    credit_history: int = 1
    property_area: str = "Semiurban"
    cibil_score: int | None = 750
    annual_income: float | None = None
    residential_assets: float | None = 0.0
    commercial_assets: float | None = 0.0
    luxury_assets: float | None = 0.0
    bank_assets: float | None = 0.0

class PredictRequest(BaseModel):
    applicant: ApplicantData
    user_email: str | None = None
    applicant_name: str | None = None

# ---------------- Auth endpoints ----------------
@app.post("/signup")
def signup(req: SignupRequest):
    users = read_json(USERS_FILE)
    if any(u["email"].lower() == req.email.lower() for u in users):
        raise HTTPException(400, "Email already registered")
    role = req.role if req.role in ("user", "bank", "analyst") else "analyst"
    user = {
        "id": str(uuid.uuid4()),
        "name": req.name,
        "email": req.email.lower(),
        "password_hash": hash_pw(req.password),
        "role": role,
        "created_at": datetime.utcnow().isoformat(),
    }
    users.append(user)
    write_json(USERS_FILE, users)
    return {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]}

@app.post("/login")
def login(req: LoginRequest):
    users = read_json(USERS_FILE)
    match = next((u for u in users if u["email"].lower() == req.email.lower()), None)
    if not match or match.get("password_hash") != hash_pw(req.password):
        raise HTTPException(401, "Invalid email or password")
    return {"id": match["id"], "name": match["name"], "email": match["email"], "role": match.get("role", "analyst")}

@app.post("/auth/google")
def auth_google(req: GoogleAuthRequest):
    users = read_json(USERS_FILE)
    email = (req.email or "").strip().lower()
    if not email:
        raise HTTPException(400, "Google email is required")
    name = req.name or email.split("@")[0].capitalize()
    match = next((u for u in users if u["email"].lower() == email), None)
    if not match:
        match = {
            "id": str(uuid.uuid4()),
            "name": name,
            "email": email,
            "password_hash": "oauth_google_verified",
            "role": "analyst",
            "avatar": req.picture or "",
            "created_at": datetime.utcnow().isoformat(),
        }
        users.append(match)
        write_json(USERS_FILE, users)
    return {"id": match["id"], "name": match["name"], "email": match["email"], "role": match.get("role", "analyst"), "avatar": match.get("avatar", "")}

# ---------------- Prediction core ----------------
def run_prediction(a: ApplicantData):
    cred_hist = a.credit_history
    if a.cibil_score is not None:
        cred_hist = 1 if a.cibil_score >= 650 else 0

    row = {
        "Gender_enc": encode_maps["Gender"].get(a.gender, 1),
        "Married_enc": encode_maps["Married"].get(a.married, 0),
        "Dependents_num": dependents_map.get(a.dependents, 0),
        "Education_enc": encode_maps["Education"].get(a.education, 1),
        "Self_Employed_enc": encode_maps["Self_Employed"].get(a.self_employed, 0),
        "ApplicantIncome": a.applicant_income,
        "CoapplicantIncome": a.coapplicant_income,
        "LoanAmount": a.loan_amount,
        "Loan_Amount_Term": a.loan_amount_term,
        "Credit_History": cred_hist,
        "Property_Area_enc": encode_maps["Property_Area"].get(a.property_area, 1),
    }
    X = pd.DataFrame([row])[feature_cols]

    if best_model_name == "Logistic Regression":
        X_in = scaler.transform(X)
    else:
        X_in = X

    proba = float(model.predict_proba(X_in)[0][1])
    pred = int(proba >= 0.5)
    risk_score = round((1 - proba) * 100, 1)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    contrib = []
    for feat, imp in zip(feature_cols, importances):
        direction = "positive"
        if feat == "Credit_History" and cred_hist == 0:
            direction = "negative"
        elif feat == "LoanAmount" and a.loan_amount > 200:
            direction = "negative"
        elif feat in ("ApplicantIncome", "CoapplicantIncome") and row[feat] < 3000:
            direction = "negative"
        contrib.append({
            "feature": FEATURE_LABELS.get(feat, feat),
            "impact": round(float(imp), 4),
            "direction": direction,
        })
    contrib = sorted(contrib, key=lambda c: -c["impact"])[:5]

    total_income = a.applicant_income + a.coapplicant_income
    annual_rate = 8.5
    monthly_rate = annual_rate / 12 / 100
    principal = a.loan_amount * 1000
    n = a.loan_amount_term
    if n > 0 and monthly_rate > 0:
        emi = principal * monthly_rate * (1 + monthly_rate) ** n / ((1 + monthly_rate) ** n - 1)
    else:
        emi = 0
    dti_ratio = round((emi / total_income) * 100, 1) if total_income > 0 else 0
    loan_to_income = round(principal / (total_income * 12), 2) if total_income > 0 else 0

    radar = {
        "income_stability": round(min(100, (total_income / 8000) * 100), 1),
        "credit_reliability": 95.0 if cred_hist == 1 else 15.0,
        "loan_burden": round(max(0, 100 - dti_ratio), 1),
        "employment_risk": 80.0 if a.self_employed == "No" else 55.0,
        "property_security": {"Urban": 85.0, "Semiurban": 70.0, "Rural": 55.0}.get(a.property_area, 75.0),
    }

    return {
        "status": "Approved" if pred == 1 else "Rejected",
        "probability": round(proba, 4),
        "risk_score": risk_score,
        "top_factors": contrib,
        "estimated_emi": round(emi, 2),
        "dti_ratio_percent": dti_ratio,
        "loan_to_income_ratio": loan_to_income,
        "radar": radar,
        "explanation": build_explanation(pred, a, dti_ratio),
    }

def build_explanation(pred, a, dti_ratio):
    cred_hist = a.credit_history
    if a.cibil_score is not None:
        cred_hist = 1 if a.cibil_score >= 650 else 0
    if pred == 1:
        reasons = []
        if cred_hist == 1:
            reasons.append("a strong credit profile")
        if a.applicant_income + a.coapplicant_income > 6000:
            reasons.append("stable combined household cashflow")
        if dti_ratio < 40:
            reasons.append("a manageable debt-to-income ratio")
        reason_text = ", ".join(reasons) if reasons else "an overall favorable financial profile"
        return f"Approval likelihood is high due to {reason_text}."
    else:
        reasons = []
        if cred_hist == 0:
            reasons.append("an impaired or unestablished credit record")
        if dti_ratio >= 40:
            reasons.append("an elevated debt-to-income ratio exceeding prudent limits")
        if a.applicant_income + a.coapplicant_income < 3000:
            reasons.append("low aggregate income relative to requested principal")
        reason_text = ", ".join(reasons) if reasons else "an elevated overall credit risk profile"
        return f"This application is flagged as high risk due to {reason_text}."

@app.post("/predict")
def predict(req: PredictRequest):
    result = run_prediction(req.applicant)
    apps = read_json(APPS_FILE)
    ref_code = f"LG-{len(apps) + 124:06d}"

    record = {
        "id": str(uuid.uuid4()),
        "ref_code": ref_code,
        "user_email": req.user_email,
        "applicant_name": req.applicant_name or "Unnamed Applicant",
        "applicant": req.applicant.dict(),
        "result": result,
        "created_at": datetime.utcnow().isoformat(),
    }
    apps.insert(0, record)
    write_json(APPS_FILE, apps)

    result_copy = dict(result)
    result_copy["id"] = record["id"]
    result_copy["ref_code"] = record["ref_code"]
    result_copy["applicant_name"] = record["applicant_name"]
    result_copy["created_at"] = record["created_at"]
    return result_copy

# ---------------- Applications ----------------
@app.get("/applications")
def get_applications(user_email: str | None = None):
    apps = read_json(APPS_FILE)
    if user_email:
        user_apps = [a for a in apps if a.get("user_email") == user_email]
        if user_apps:
            return user_apps
    return apps

@app.get("/applications/{app_id}")
def get_application(app_id: str):
    apps = read_json(APPS_FILE)
    match = next((a for a in apps if a["id"] == app_id), None)
    if not match:
        raise HTTPException(404, "Application not found")
    return match

# ---------------- Model performance ----------------
@app.get("/model-performance")
def model_performance():
    return {
        "best_model": best_model_name,
        "results": results_summary["results"],
        "feature_importance": feature_importance,
    }

@app.get("/")
def root():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return {"status": "ok", "message": "Loan Approval & Risk Assessment API running"}

@app.get("/api/health")
def health():
    return {"status": "ok", "message": "Loan Approval & Risk Assessment API running"}

